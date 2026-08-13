"""Phase B: content encoder, title posteriors, cold-start MCV."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn
from tqdm import tqdm

from catalog_value.config import Config
from catalog_value.data.genome import content_features
from catalog_value.data.movielens import processed_dir, require_core
from catalog_value.device import pick_device
from catalog_value.models.audience.mixture import load_audience, subsample_audience
from catalog_value.models.catalog_value.mcv import marginal_content_value
from catalog_value.models.catalog_value.posterior_mcv import mcv_moments
from catalog_value.models.content.encoder import ContentEncoder
from catalog_value.models.content.posterior import residual_variance, shrink_posterior
from catalog_value.models.types import TitlePosterior, TitleReps
from catalog_value.paths import output_dir
from catalog_value.phase_a import _load_titles, _popular_rows, artifact_dir
from catalog_value.visualization.phase_b import (
    plot_mcv_transfer,
    plot_mcv_uncertainty,
    plot_neighbor_swap,
    plot_reconstruction,
)
from catalog_value.visualization.style import figures_dir

PROBES = [
    "Halloween (1978)",
    "Super Mario Bros. (1993)",
    "Paths of Glory (1957)",
    "Notebook, The (2004)",
]


def phase_b_dir() -> Path:
    path = output_dir() / "phase_b"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-8)
    bn = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-8)
    return (an * bn).sum(axis=1)


def _neighbor_table(movies: pl.DataFrame, z: np.ndarray, name: str, k: int = 6) -> pl.DataFrame | None:
    hit = movies.filter(pl.col("title") == name)
    if hit.is_empty():
        return None
    idx = int(hit["movie_row"][0])
    unit = z / np.linalg.norm(z, axis=1, keepdims=True).clip(min=1e-8)
    scores = unit @ unit[idx]
    scores[idx] = -np.inf
    popular = set(
        int(r) for r in movies.filter(pl.col("n_ratings") >= 1500)["movie_row"].to_list()
    )
    ranked = [int(i) for i in np.argsort(scores)[::-1] if int(i) in popular][:k]
    titles = dict(zip(movies["movie_row"].to_list(), movies["title"].to_list()))
    genres = dict(zip(movies["movie_row"].to_list(), movies["genres"].to_list()))
    return pl.DataFrame(
        {
            "title": [titles[i] for i in ranked],
            "genres": [genres[i] for i in ranked],
            "cosine": [float(scores[i]) for i in ranked],
        }
    )


def train_encoder(
    x: np.ndarray,
    z: np.ndarray,
    bias: np.ndarray,
    train_idx: np.ndarray,
    *,
    hidden: int,
    epochs: int,
    lr: float,
    seed: int,
) -> ContentEncoder:
    device = pick_device()
    torch.manual_seed(seed)
    model = ContentEncoder(in_dim=x.shape[1], out_dim=z.shape[1], hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.from_numpy(x[train_idx]).to(device)
    zt = torch.from_numpy(z[train_idx].astype(np.float32)).to(device)
    bt = torch.from_numpy(bias[train_idx].astype(np.float32)).to(device)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in tqdm(range(epochs), desc="content encoder"):
        pred_z, pred_b = model(xt)
        loss = loss_fn(pred_z, zt) + 0.25 * loss_fn(pred_b, bt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{epochs}  loss={float(loss.detach()):.4f}")
    return model


@torch.no_grad()
def encode_all(model: ContentEncoder, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    pred_z, pred_b = model(torch.from_numpy(x).to(device))
    return (
        pred_z.cpu().numpy().astype(np.float64),
        pred_b.cpu().numpy().astype(np.float64),
    )


def run_phase_b(config: Config) -> Path:
    if not (processed_dir() / "genome_scores.parquet").exists():
        raise FileNotFoundError("Genome tags missing. Run: python -m catalog_value ingest")
    movies = pl.read_parquet(require_core() / "movies.parquet")
    titles = _load_titles(artifact_dir(), config)
    audience_full, _ = load_audience(artifact_dir() / "audience.npz")
    audience = subsample_audience(audience_full, n=config.catalog_value.n_eval_users, seed=config.seed)
    cfg = config.phase_b
    out = phase_b_dir()

    print("Building genome + genre + year features")
    features = content_features(movies)
    n_ratings = movies.sort("movie_row")["n_ratings"].to_numpy()
    eligible = np.flatnonzero(n_ratings >= cfg.min_train_ratings)
    rng = np.random.default_rng(config.seed)
    rng.shuffle(eligible)
    n_hold = max(1, int(len(eligible) * cfg.holdout_frac))
    holdout_idx = np.sort(eligible[:n_hold])
    train_idx = np.sort(eligible[n_hold:])
    holdout_mask = np.zeros(len(n_ratings), dtype=bool)
    holdout_mask[holdout_idx] = True
    print(f"Encoder train titles={len(train_idx):,}  holdout={len(holdout_idx):,}")

    model = train_encoder(
        features,
        titles.z,
        titles.bias,
        train_idx,
        hidden=cfg.hidden,
        epochs=cfg.epochs,
        lr=cfg.lr,
        seed=config.seed,
    )
    z_content, b_content = encode_all(model, features)
    cosine = _cosine(z_content, titles.z)
    print(
        f"mean cosine train={cosine[train_idx].mean():.3f}  "
        f"holdout={cosine[holdout_idx].mean():.3f}"
    )

    content_var = residual_variance(titles.z, z_content, ~holdout_mask)
    collab_mask = ~holdout_mask
    posterior = shrink_posterior(
        titles.z,
        z_content,
        titles.bias,
        b_content,
        n_ratings,
        titles.movie_row,
        n0=cfg.n0,
        content_var=content_var,
        collab_mask=collab_mask,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": model.in_dim,
            "out_dim": model.out_dim,
            "hidden": model.hidden,
            "holdout_idx": holdout_idx,
        },
        out / "content_encoder.pt",
    )
    np.savez_compressed(
        out / "posterior.npz",
        mu=posterior.mu,
        var=posterior.var,
        bias=posterior.bias,
        movie_row=posterior.movie_row,
        z_content=z_content,
        b_content=b_content,
        holdout_idx=holdout_idx,
        content_var=np.array([content_var]),
    )
    print(f"Wrote {out / 'content_encoder.pt'} and {out / 'posterior.npz'}")
    print(f"content residual var={content_var:.4f}  mean posterior std={np.sqrt(posterior.var).mean():.3f}")

    content_titles = TitleReps(z=z_content, bias=b_content, movie_row=titles.movie_row)
    catalog = _popular_rows(movies, config.phase_a.catalog_size)
    candidate_n = config.phase_a.n_candidates
    ranked = _popular_rows(movies, config.phase_a.catalog_size + candidate_n)
    candidates = ranked[config.phase_a.catalog_size :]
    holdout_candidates = np.array([int(i) for i in candidates if holdout_mask[int(i)]], dtype=np.int64)
    print(f"held-out candidates in MCV pool: {len(holdout_candidates):,}")

    tau = config.catalog_value.tau
    mcv_collab = marginal_content_value(audience, titles, catalog, holdout_candidates, tau)
    mcv_content = marginal_content_value(audience, content_titles, catalog, holdout_candidates, tau)
    spearman = float(
        pl.DataFrame({"a": mcv_collab, "b": mcv_content})
        .select(pl.corr("a", "b", method="spearman"))
        .item()
    )
    print(f"Spearman MCV collab vs content (holdout candidates)={spearman:.3f}")

    # Mix held-out (high posterior variance) with well-observed titles.
    uncertainty_idx = np.unique(
        np.concatenate(
            [
                holdout_idx[:160],
                train_idx[-160:],
                candidates[:80],
            ]
        )
    )
    mean_mcv, std_mcv = mcv_moments(
        audience,
        posterior,
        catalog,
        uncertainty_idx,
        tau,
        n_samples=cfg.n_posterior_samples,
        seed=config.seed,
    )

    panels = []
    for name in PROBES:
        collab = _neighbor_table(movies, titles.z, name)
        content = _neighbor_table(movies, z_content, name)
        if collab is not None and content is not None:
            panels.append((name, collab, content))
            print(f"\n{name}")
            print("  collab ", collab["title"].to_list())
            print("  content", content["title"].to_list())

    published = figures_dir("phase_b")
    plot_idx = np.concatenate([train_idx, holdout_idx])
    recon = plot_reconstruction(
        n_ratings[plot_idx],
        cosine[plot_idx],
        holdout_mask[plot_idx],
        out / "reconstruction.png",
    )
    swap = plot_neighbor_swap(panels, out / "neighbor_swap.png")
    title_of = dict(zip(movies["movie_row"].to_list(), movies["title"].to_list()))
    transfer = plot_mcv_transfer(
        mcv_collab,
        mcv_content,
        [title_of[int(i)] for i in holdout_candidates],
        out / "mcv_transfer.png",
    )
    uncertainty = plot_mcv_uncertainty(
        mean_mcv, std_mcv, n_ratings[uncertainty_idx], out / "mcv_uncertainty.png"
    )
    for src in (recon, swap, transfer, uncertainty):
        shutil.copy2(src, published / src.name)
        print(f"Wrote {published / src.name}")

    pl.DataFrame(
        {
            "movie_row": holdout_candidates,
            "title": [title_of[int(i)] for i in holdout_candidates],
            "mcv_collab": mcv_collab,
            "mcv_content": mcv_content,
        }
    ).write_csv(out / "holdout_mcv.csv")
    return published


def load_posterior(path: Path | None = None) -> tuple[TitlePosterior, np.ndarray, np.ndarray]:
    payload = np.load(path or phase_b_dir() / "posterior.npz")
    posterior = TitlePosterior(
        mu=payload["mu"],
        var=payload["var"],
        bias=payload["bias"],
        movie_row=payload["movie_row"],
    )
    return posterior, payload["z_content"], payload["holdout_idx"]
