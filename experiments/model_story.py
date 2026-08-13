"""Walk the trained model and write figures for the model story."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.decomposition import PCA

from catalog_value.data.movielens import require_core
from catalog_value.models.audience.mixture import load_audience
from catalog_value.models.catalog_value.analytical import coverage_g
from catalog_value.models.catalog_value.mcv import affinity, value_of_catalog
from catalog_value.models.types import AudienceStates, TitleReps
from catalog_value.phase_a import artifact_dir
from catalog_value.paths import project_root

STORY = project_root() / "docs" / "story"
PROBES = [
    "Toy Story (1995)",
    "Pulp Fiction (1994)",
    "Dark Knight, The (2008)",
    "Halloween (1978)",
    "Notebook, The (2004)",
    "Paths of Glory (1957)",
    "Super Mario Bros. (1993)",
    "Planet Earth (2006)",
]


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_artifacts() -> tuple[pl.DataFrame, TitleReps, AudienceStates, np.ndarray]:
    movies = pl.read_parquet(require_core() / "movies.parquet")
    titles_np = np.load(artifact_dir() / "titles.npz")
    titles = TitleReps(
        z=titles_np["z"],
        bias=titles_np["bias"],
        movie_row=titles_np["movie_row"],
    )
    audience, queries = load_audience(artifact_dir() / "audience.npz")
    return movies, titles, audience, queries


def primary_genre(genres: str) -> str:
    return genres.split("|")[0] if genres else "Unknown"


def cosine_neighbors(z: np.ndarray, idx: int, k: int = 8) -> np.ndarray:
    norms = np.linalg.norm(z, axis=1, keepdims=True).clip(min=1e-8)
    unit = z / norms
    scores = unit @ unit[idx]
    scores[idx] = -np.inf
    return np.argsort(scores)[::-1][:k]


def find_title(movies: pl.DataFrame, name: str) -> int | None:
    hit = movies.filter(pl.col("title") == name)
    if hit.is_empty():
        hit = movies.filter(pl.col("title").str.contains(name, literal=True))
    if hit.is_empty():
        return None
    return int(hit["movie_row"][0])


def plot_pca(movies: pl.DataFrame, z: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2, random_state=0)
    xy = pca.fit_transform(z)
    genre = np.array([primary_genre(g) for g in movies["genres"].to_list()])
    keep = movies["n_ratings"].to_numpy() >= 2000
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    top = ["Action", "Comedy", "Drama", "Horror", "Children", "Documentary", "Crime", "Adventure"]
    palette = {
        "Action": "#c44536",
        "Comedy": "#e09f3e",
        "Drama": "#2a6f97",
        "Horror": "#540b0e",
        "Children": "#90be6d",
        "Documentary": "#577590",
        "Crime": "#9b2226",
        "Adventure": "#4d908e",
    }
    ax.scatter(xy[keep, 0], xy[keep, 1], c="#d0d0d0", s=8, alpha=0.4, linewidths=0, label="other")
    for g in top:
        mask = keep & (genre == g)
        if mask.sum() == 0:
            continue
        ax.scatter(xy[mask, 0], xy[mask, 1], c=palette[g], s=12, alpha=0.7, linewidths=0, label=g)
    ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.0f}% var)")
    ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.0f}% var)")
    ax.set_title("Title embeddings: genre is visible, not the whole story")
    ax.legend(frameon=False, fontsize=8, loc="best")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return pca.explained_variance_ratio_


def plot_query_cosine(queries: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    q = queries / np.linalg.norm(queries, axis=1, keepdims=True).clip(min=1e-8)
    gram = q @ q.T
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(gram, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(queries)), labels=[f"q{k}" for k in range(len(queries))])
    ax.set_yticks(range(len(queries)), labels=[f"q{k}" for k in range(len(queries))])
    ax.set_title("Taste queries are nearly orthogonal")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    off = gram - np.eye(len(queries))
    return float(np.sqrt((off**2).mean()))


def plot_pi_entropy(pi: np.ndarray, path: Path) -> float:
    import matplotlib.pyplot as plt

    ent = -(pi * np.log(np.clip(pi, 1e-12, 1))).sum(axis=1)
    max_ent = np.log(pi.shape[1])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(ent, bins=40, color="#2a6f97", alpha=0.85)
    ax.axvline(max_ent, color="#c44536", linestyle="--", label=f"uniform (log K = {max_ent:.2f})")
    ax.set_xlabel("Entropy of user mixing weights π_u")
    ax.set_ylabel("Users")
    ax.set_title("Mixing weights stay close to uniform")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return float(ent.mean())


def plot_diminishing_returns(
    audience: AudienceStates,
    titles: TitleReps,
    movies: pl.DataFrame,
    path: Path,
) -> dict[str, list[float]]:
    import matplotlib.pyplot as plt

    def rows_for(*names: str) -> np.ndarray:
        ids = []
        for name in names:
            idx = find_title(movies, name)
            if idx is not None:
                ids.append(idx)
        return np.array(ids, dtype=np.int64)

    action = rows_for(
        "Die Hard (1988)",
        "Die Hard 2 (1990)",
        "Die Hard: With a Vengeance (1995)",
        "Lethal Weapon (1987)",
        "Terminator 2: Judgment Day (1991)",
    )
    mixed = rows_for(
        "Die Hard (1988)",
        "Toy Story (1995)",
        "Silence of the Lambs, The (1991)",
        "When Harry Met Sally... (1989)",
        "Planet Earth (2006)",
    )
    tau = 0.5
    rng = np.random.default_rng(0)
    users = rng.choice(audience.pi.shape[0], size=min(1500, audience.pi.shape[0]), replace=False)
    aud = AudienceStates(
        pi=audience.pi[users],
        z=audience.z[users],
        user_row=audience.user_row[users],
    )

    def curve(order: np.ndarray) -> list[float]:
        vals = []
        for n in range(1, len(order) + 1):
            vals.append(value_of_catalog(aud, titles, order[:n], tau).mean)
        return vals

    action_v = curve(action)
    mixed_v = curve(mixed)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(range(1, len(action_v) + 1), action_v, marker="o", color="#c44536", label="near-substitutes (action)")
    ax.plot(range(1, len(mixed_v) + 1), mixed_v, marker="o", color="#2d6a4f", label="diverse tastes")
    ax.set_xlabel("Catalog size")
    ax.set_ylabel("Audience coverage V(S)")
    ax.set_title("Diverse titles keep adding coverage; substitutes flatten")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return {"action": action_v, "mixed": mixed_v}


def top_movies_for_query(
    movies: pl.DataFrame,
    z: np.ndarray,
    query: np.ndarray,
    n: int = 8,
    min_ratings: int = 2000,
) -> pl.DataFrame:
    scores = z @ query
    table = movies.with_columns(pl.Series("score", scores))
    return (
        table.filter(pl.col("n_ratings") >= min_ratings)
        .sort("score", descending=True)
        .head(n)
        .select(["title", "genres", "n_ratings", "score"])
    )


def neighbor_table(movies: pl.DataFrame, z: np.ndarray, name: str) -> pl.DataFrame | None:
    idx = find_title(movies, name)
    if idx is None:
        return None
    popular = set(
        int(r) for r in movies.filter(pl.col("n_ratings") >= 2000)["movie_row"].to_list()
    )
    nbr = [int(i) for i in cosine_neighbors(z, idx, k=40) if int(i) in popular][:6]
    rows = movies.filter(pl.col("movie_row").is_in(nbr)).join(
        pl.DataFrame({"movie_row": nbr, "rank": np.arange(1, len(nbr) + 1)}),
        on="movie_row",
    ).sort("rank")
    return rows.select(["rank", "title", "genres", "n_ratings"])


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    STORY.mkdir(parents=True, exist_ok=True)
    movies, titles, audience, queries = load_artifacts()
    z = titles.z
    print(f"movies={movies.height:,} users={audience.pi.shape[0]:,} dim={z.shape[1]} K={queries.shape[0]}")

    var = plot_pca(movies, z, STORY / "title_pca.png")
    print(f"PCA var {var[0]:.3f} {var[1]:.3f}")

    rms = plot_query_cosine(queries, STORY / "taste_query_cosine.png")
    print(f"query off-diagonal RMS cosine={rms:.4f}")

    mean_ent = plot_pi_entropy(audience.pi, STORY / "pi_entropy.png")
    print(f"mean π entropy={mean_ent:.3f} (max {np.log(queries.shape[0]):.3f})")

    # Within-user taste diversity (attended z_uk, not the global queries)
    z_u = audience.z
    z_u = z_u / np.linalg.norm(z_u, axis=2, keepdims=True).clip(min=1e-8)
    grams = np.einsum("ukd,uld->ukl", z_u, z_u)
    k = z_u.shape[1]
    off = grams - np.eye(k)[None, :, :]
    mean_off = float(np.sqrt((off**2).mean()))
    print(f"mean within-user taste cosine RMS={mean_off:.4f}")

    curves = plot_diminishing_returns(
        audience, titles, movies, STORY / "diminishing_returns.png"
    )
    print("diminishing returns action", [round(v, 3) for v in curves["action"]])
    print("diminishing returns mixed", [round(v, 3) for v in curves["mixed"]])

    print("\n=== nearest neighbors ===")
    for name in PROBES:
        table = neighbor_table(movies, z, name)
        print(f"\n{name}")
        if table is None:
            print("  not found")
            continue
        print(table)

    print("\n=== top titles per taste query ===")
    for k_i, q in enumerate(queries):
        top = top_movies_for_query(movies, z, q)
        print(f"\nq{k_i}")
        print(top)

    # Within vs across genre cosine
    genre = np.array([primary_genre(g) for g in movies["genres"].to_list()])
    popular = movies["n_ratings"].to_numpy() >= 5000
    unit = z / np.linalg.norm(z, axis=1, keepdims=True).clip(min=1e-8)
    rng = np.random.default_rng(0)
    same, diff = [], []
    idx = np.flatnonzero(popular)
    for _ in range(2000):
        a, b = rng.choice(idx, size=2, replace=False)
        sim = float(unit[a] @ unit[b])
        if genre[a] == genre[b]:
            same.append(sim)
        else:
            diff.append(sim)
    print(f"\nmean cosine same-genre={np.mean(same):.3f} cross-genre={np.mean(diff):.3f}")

    # Bias vs popularity
    logn = np.log10(movies["n_ratings"].to_numpy().clip(min=1))
    corr = float(np.corrcoef(titles.bias, logn)[0, 1])
    print(f"corr(item_bias, log10 n_ratings)={corr:.3f}")


if __name__ == "__main__":
    main()
