from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from catalog_value.config import Config
from catalog_value.device import pick_device
from catalog_value.models.audience.mixture import save_audience
from catalog_value.models.audience.taste_tokens import TasteTokenEncoder
from catalog_value.models.types import AudienceStates


class UserHistoryDataset(Dataset):
    def __init__(
        self,
        movie_lists: list[np.ndarray],
        rating_lists: list[np.ndarray],
        max_history: int,
        n_holdout: int,
        seed: int,
    ) -> None:
        self.movie_lists = movie_lists
        self.rating_lists = rating_lists
        self.max_history = max_history
        self.n_holdout = n_holdout
        self.seed = seed

    def __len__(self) -> int:
        return len(self.movie_lists)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        movies = self.movie_lists[idx]
        ratings = self.rating_lists[idx]
        rng = np.random.default_rng(self.seed + idx)
        n_holdout = min(self.n_holdout, max(1, len(movies) // 5))
        perm = rng.permutation(len(movies))
        hold = perm[:n_holdout]
        rest = perm[n_holdout:]
        if rest.size == 0:
            rest = hold
        rest = rest[: self.max_history]
        return {
            "history": movies[rest].astype(np.int64),
            "targets": movies[hold].astype(np.int64),
            "target_ratings": ratings[hold].astype(np.float32),
            "user_row": np.int64(idx),
        }


def _pad_1d(seqs: list[np.ndarray], pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    length = max(int(s.shape[0]) for s in seqs)
    batch = len(seqs)
    out = torch.full((batch, length), pad, dtype=torch.long)
    mask = torch.zeros((batch, length), dtype=torch.bool)
    for i, seq in enumerate(seqs):
        n = int(seq.shape[0])
        out[i, :n] = torch.from_numpy(seq)
        mask[i, :n] = True
    return out, mask


def collate_histories(batch: list[dict[str, np.ndarray]], pad_id: int) -> dict[str, torch.Tensor]:
    history, history_mask = _pad_1d([row["history"] for row in batch], pad_id)
    targets, target_mask = _pad_1d([row["targets"] for row in batch], pad_id)
    rating_mat = torch.zeros_like(targets, dtype=torch.float32)
    for i, row in enumerate(batch):
        n = row["target_ratings"].shape[0]
        rating_mat[i, :n] = torch.from_numpy(row["target_ratings"])
    return {
        "history": history,
        "history_mask": history_mask,
        "targets": targets,
        "target_mask": target_mask,
        "target_ratings": rating_mat,
        "user_row": torch.tensor([int(row["user_row"]) for row in batch], dtype=torch.long),
    }


def _user_lists(ratings: pl.DataFrame) -> tuple[list[np.ndarray], list[np.ndarray]]:
    grouped = (
        ratings.sort(["user_row", "timestamp"])
        .group_by("user_row", maintain_order=True)
        .agg(pl.col("movie_row"), pl.col("rating"))
        .sort("user_row")
    )
    movies = [np.asarray(row, dtype=np.int64) for row in grouped["movie_row"].to_list()]
    rating_vals = [np.asarray(row, dtype=np.float32) for row in grouped["rating"].to_list()]
    return movies, rating_vals


def train_taste_tokens(
    ratings: pl.DataFrame,
    n_users: int,
    n_movies: int,
    config: Config,
    artifact_dir: Path,
) -> TasteTokenEncoder:
    train_cfg = config.train
    device = pick_device()
    print(f"Training taste-token encoder on {device} ({n_users:,} users, {n_movies:,} movies)")
    movie_lists, rating_lists = _user_lists(ratings)
    if len(movie_lists) != n_users:
        raise ValueError(f"Expected {n_users} users, grouped {len(movie_lists)}")

    dataset = UserHistoryDataset(
        movie_lists,
        rating_lists,
        max_history=train_cfg.max_history,
        n_holdout=train_cfg.n_holdout,
        seed=config.seed,
    )
    model = TasteTokenEncoder(
        n_movies=n_movies,
        dim=config.representation.embedding_dim,
        n_interests=config.representation.n_interests,
        n_heads=train_cfg.n_heads,
    ).to(device)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate_histories(batch, model.pad_id),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    mse = nn.MSELoss(reduction="none")

    model.train()
    for epoch in range(train_cfg.epochs):
        running = 0.0
        n_batches = 0
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}/{train_cfg.epochs}"):
            history = batch["history"].to(device)
            history_mask = batch["history_mask"].to(device)
            targets = batch["targets"].to(device)
            target_mask = batch["target_mask"].to(device)
            y = batch["target_ratings"].to(device)

            pi, z = model.encode_history(history, history_mask)
            pred = model.predicted_affinity(pi, z, targets)
            rating_loss = (mse(pred, y) * target_mask).sum() / target_mask.sum().clamp_min(1)
            div = model.diversity_loss()
            ent = model.entropy(pi)
            loss = (
                rating_loss
                + train_cfg.diversity_weight * div
                - train_cfg.entropy_weight * ent
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.detach())
            n_batches += 1
        print(
            f"epoch {epoch + 1}: loss={running / max(n_batches, 1):.4f} "
            f"diversity={float(div.detach()):.4f} entropy={float(ent.detach()):.4f}"
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    ckpt = artifact_dir / "taste_tokens.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_movies": n_movies,
            "dim": config.representation.embedding_dim,
            "n_interests": config.representation.n_interests,
            "n_heads": train_cfg.n_heads,
        },
        ckpt,
    )
    print(f"Wrote {ckpt}")
    return model


@torch.no_grad()
def encode_all_users(
    model: TasteTokenEncoder,
    movie_lists: list[np.ndarray],
    max_history: int,
    batch_size: int,
) -> AudienceStates:
    device = next(model.parameters()).device
    model.eval()
    pis = []
    zs = []
    pad = model.pad_id
    for start in tqdm(range(0, len(movie_lists), batch_size), desc="encode users"):
        chunk = movie_lists[start : start + batch_size]
        truncated = [m[-max_history:] if m.size > max_history else m for m in chunk]
        history, mask = _pad_1d(truncated, pad)
        pi, z = model.encode_history(history.to(device), mask.to(device))
        pis.append(pi.cpu().numpy())
        zs.append(z.cpu().numpy())
    pi = np.concatenate(pis, axis=0).astype(np.float64)
    z = np.concatenate(zs, axis=0).astype(np.float64)
    return AudienceStates(
        pi=pi,
        z=z,
        user_row=np.arange(len(movie_lists), dtype=np.int64),
    )


def export_neural_fit(
    model: TasteTokenEncoder,
    ratings: pl.DataFrame,
    n_users: int,
    config: Config,
    artifact_dir: Path,
) -> None:
    movie_lists, _ = _user_lists(ratings)
    if len(movie_lists) != n_users:
        raise ValueError("user count mismatch while exporting")
    audience = encode_all_users(
        model,
        movie_lists,
        max_history=config.train.max_history,
        batch_size=config.train.batch_size,
    )
    queries = model.taste_queries.detach().float().cpu().numpy().astype(np.float64)
    save_audience(audience, queries, artifact_dir / "audience.npz")
    titles = model.title_reps()
    np.savez_compressed(
        artifact_dir / "titles.npz",
        z=titles.z,
        bias=titles.bias,
        movie_row=titles.movie_row,
    )
    print(f"Wrote {artifact_dir / 'audience.npz'} and {artifact_dir / 'titles.npz'}")
