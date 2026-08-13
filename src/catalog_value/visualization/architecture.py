"""Architecture diagrams: system DAG, taste-token encoder, valuation readout."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from catalog_value.visualization.style import INK, MUTED, PAPER, apply_style, figures_dir

def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return path


BOX = {
    "data": "#e7e5e4",
    "net": "#dbeafe",
    "state": "#d1fae5",
    "value": "#fde68a",
    "readout": "#ffedd5",
}


def _box(ax, x, y, w, h, text, fill, *, fontsize=8.5, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.04",
        linewidth=1.1,
        edgecolor=INK,
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        weight=weight,
        wrap=True,
    )
    return (x + w / 2, y + h / 2, x, y, w, h)


def _arrow(ax, a, b, *, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.15,
            color=INK,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=6,
            shrinkB=6,
        )
    )


def plot_system_dag(path: Path) -> Path:
    """End-to-end DAG of what is trained vs what is a readout."""
    apply_style()
    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("How the system is wired", loc="left", fontsize=14, pad=8)

    # Column 0: data
    ratings = _box(ax, 0.02, 0.70, 0.16, 0.16, "MovieLens\nratings  D", BOX["data"], weight="medium")
    tags = _box(ax, 0.02, 0.42, 0.16, 0.16, "Genome tags\ngenre, year  Xᵢ", BOX["data"])
    catalog = _box(ax, 0.02, 0.08, 0.16, 0.16, "Catalog  S\n(set of titles)", BOX["data"])

    # Column 1: networks
    enc = _box(ax, 0.26, 0.66, 0.20, 0.22, "Taste-token\nencoder\n(trained)", BOX["net"], weight="medium")
    mlp = _box(ax, 0.26, 0.38, 0.20, 0.20, "Content MLP\nf(Xᵢ)\n(trained)", BOX["net"], weight="medium")

    # Column 2: states
    user = _box(ax, 0.54, 0.72, 0.18, 0.14, "User tastes\nπᵤ, zᵤₖ", BOX["state"], weight="medium")
    collab = _box(ax, 0.54, 0.52, 0.18, 0.14, "Collab titles\nzᵢ, bᵢ", BOX["state"], weight="medium")
    post = _box(ax, 0.54, 0.30, 0.18, 0.16, "Title posterior\nμᵢ, σᵢ²\n(shrinkage)", BOX["state"], weight="medium")

    # Column 3: value
    aff = _box(ax, 0.80, 0.62, 0.17, 0.14, "Affinity\naᵤₖᵢ = zᵤₖᵀzᵢ + bᵢ", BOX["value"])
    util = _box(ax, 0.80, 0.38, 0.17, 0.14, "Coverage\nVᵤ(S),  V(S)", BOX["value"], weight="medium")
    mcv = _box(ax, 0.80, 0.10, 0.17, 0.16, "MCVᵢ(S)\nthen PACV, greedy", BOX["readout"], weight="medium")

    _arrow(ax, (ratings[0] + 0.08, ratings[1]), (enc[0] - 0.10, enc[1] + 0.02))
    _arrow(ax, (ratings[0] + 0.08, ratings[1] - 0.04), (collab[0] - 0.09, collab[1] + 0.04), rad=-0.12)
    _arrow(ax, (tags[0] + 0.08, tags[1]), (mlp[0] - 0.10, mlp[1]))
    _arrow(ax, (enc[0] + 0.10, enc[1] + 0.04), (user[0] - 0.09, user[1]))
    _arrow(ax, (enc[0] + 0.10, enc[1] - 0.06), (collab[0] - 0.09, collab[1] + 0.02))
    _arrow(ax, (mlp[0] + 0.10, mlp[1]), (post[0] - 0.09, post[1] + 0.02))
    _arrow(ax, (collab[0], collab[1] - 0.07), (post[0], post[1] + 0.08))
    _arrow(ax, (user[0] + 0.09, user[1] - 0.02), (aff[0] - 0.085, aff[1] + 0.02))
    _arrow(ax, (post[0] + 0.09, post[1] + 0.04), (aff[0] - 0.085, aff[1] - 0.04), rad=0.08)
    _arrow(ax, (catalog[0] + 0.08, catalog[1] + 0.04), (aff[0] - 0.04, aff[1] - 0.06), rad=-0.18)
    _arrow(ax, (aff[0], aff[1] - 0.07), (util[0], util[1] + 0.07))
    _arrow(ax, (util[0], util[1] - 0.07), (mcv[0], mcv[1] + 0.08))
    _arrow(ax, (catalog[0] + 0.08, catalog[1]), (mcv[0] - 0.085, mcv[1]), rad=-0.05)

    ax.text(0.10, 0.93, "Data", ha="center", fontsize=9, color=MUTED)
    ax.text(0.36, 0.93, "Networks (trained)", ha="center", fontsize=9, color=MUTED)
    ax.text(0.63, 0.93, "Frozen states", ha="center", fontsize=9, color=MUTED)
    ax.text(0.88, 0.93, "Readout (not trained)", ha="center", fontsize=9, color=MUTED)
    ax.text(
        0.02,
        0.01,
        "Only the blue boxes see gradients. V and MCV are formulas on top of the embeddings.",
        fontsize=8,
        color=MUTED,
        ha="left",
    )
    return _save(fig, path)


def plot_encoder(path: Path) -> Path:
    """Taste-token encoder internals."""
    apply_style()
    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Taste-token encoder (what is actually trained)", loc="left", fontsize=14, pad=8)

    ids = _box(ax, 0.03, 0.78, 0.18, 0.12, "User history\nmovie IDs\n(set, max 64)", BOX["data"])
    table = _box(ax, 0.28, 0.78, 0.20, 0.12, "Embedding table\nzⱼ ∈ ℝ⁶⁴,  bⱼ", BOX["net"], weight="medium")
    hist = _box(ax, 0.56, 0.78, 0.18, 0.12, "History set\n{zⱼ}  no positions", BOX["state"])
    q = _box(ax, 0.03, 0.50, 0.18, 0.16, "Learned queries\nq₁ … q₈\n(same for all users)", BOX["net"], weight="medium")
    mha = _box(ax, 0.30, 0.48, 0.22, 0.20, "Multi-head attention\nQ = queries\nK, V = history\n4 heads", BOX["net"], weight="medium")
    res = _box(ax, 0.60, 0.50, 0.18, 0.16, "Residual + LN\nzᵤₖ = LN(att + qₖ)", BOX["state"])
    z = _box(ax, 0.82, 0.58, 0.15, 0.12, "zᵤₖ\nK × 64", BOX["state"], weight="medium")
    pi = _box(ax, 0.82, 0.38, 0.15, 0.12, "linear → softmax\nπᵤ  (K weights)", BOX["state"], weight="medium")
    tgt = _box(ax, 0.03, 0.12, 0.18, 0.14, "Held-out titles\nIDs + ratings r", BOX["data"])
    zt = _box(ax, 0.28, 0.12, 0.18, 0.14, "Look up zᵢ, bᵢ\n(same table)", BOX["net"])
    mix = _box(
        ax,
        0.54,
        0.08,
        0.28,
        0.22,
        "Mixture affinity\nΣₖ πᵤₖ (zᵤₖᵀ zᵢ + bᵢ) + 3.5\n= predicted rating  r̂",
        BOX["value"],
        weight="medium",
        fontsize=8.2,
    )
    loss = _box(ax, 0.84, 0.10, 0.13, 0.16, "MSE(r̂, r)\n+ query\ndiversity\n− H(π)", BOX["readout"], fontsize=7.8)

    _arrow(ax, (ids[0] + 0.09, ids[1]), (table[0] - 0.10, table[1]))
    _arrow(ax, (table[0] + 0.10, table[1]), (hist[0] - 0.09, hist[1]))
    _arrow(ax, (q[0] + 0.09, q[1]), (mha[0] - 0.11, mha[1] - 0.02))
    _arrow(ax, (hist[0], hist[1] - 0.06), (mha[0] + 0.02, mha[1] + 0.10))
    _arrow(ax, (mha[0] + 0.11, mha[1]), (res[0] - 0.09, res[1]))
    _arrow(ax, (q[0] + 0.09, q[1] + 0.04), (res[0] - 0.09, res[1] + 0.04), rad=0.18)
    _arrow(ax, (res[0] + 0.09, res[1] + 0.04), (z[0] - 0.075, z[1]))
    _arrow(ax, (res[0] + 0.09, res[1] - 0.04), (pi[0] - 0.075, pi[1] + 0.02))
    _arrow(ax, (tgt[0] + 0.09, tgt[1]), (zt[0] - 0.09, zt[1]))
    _arrow(ax, (table[0], table[1] - 0.06), (zt[0], zt[1] + 0.07), rad=0.12)
    _arrow(ax, (z[0] - 0.02, z[1] - 0.06), (mix[0] + 0.08, mix[1] + 0.11), rad=0.05)
    _arrow(ax, (pi[0] - 0.02, pi[1] - 0.06), (mix[0] + 0.10, mix[1] + 0.11))
    _arrow(ax, (zt[0] + 0.09, zt[1]), (mix[0] - 0.14, mix[1]))
    _arrow(ax, (mix[0] + 0.14, mix[1]), (loss[0] - 0.065, loss[1] + 0.02))

    ax.text(
        0.03,
        0.02,
        "Queries are global. User-specific state is whatever attention writes into zᵤₖ and πᵤ.",
        fontsize=8,
        color=MUTED,
    )
    return _save(fig, path)


def plot_valuation(path: Path) -> Path:
    """Frozen embeddings → coverage → MCV. No extra network."""
    apply_style()
    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Valuation readout (no extra training)", loc="left", fontsize=14, pad=8)

    zu = _box(ax, 0.03, 0.62, 0.16, 0.18, "Every user\nπᵤ, zᵤₖ", BOX["state"])
    zi = _box(ax, 0.03, 0.28, 0.16, 0.18, "Every title\nzᵢ or μᵢ, σᵢ", BOX["state"])
    s = _box(ax, 0.03, 0.04, 0.16, 0.14, "Catalog S\nand candidate i", BOX["data"])
    dots = _box(ax, 0.28, 0.40, 0.20, 0.22, "Dot products\naᵤₖᵢ for i ∈ S\nand for candidate", BOX["value"], weight="medium")
    cov = _box(
        ax,
        0.56,
        0.48,
        0.22,
        0.28,
        "Per taste, soft OR\nτ log(1 + Σ exp(a/τ))\nτ = 0.5\nthen mix with πᵤ",
        BOX["value"],
        weight="medium",
        fontsize=8,
    )
    vs = _box(ax, 0.82, 0.58, 0.15, 0.16, "Vᵤ(S)\nV(S)=Eᵤ[Vᵤ]", BOX["readout"], weight="medium")
    mcv = _box(ax, 0.82, 0.22, 0.15, 0.20, "MCVᵢ(S)\n= V(S∪{i})−V(S)", BOX["readout"], weight="medium", fontsize=8)
    pacv = _box(ax, 0.56, 0.08, 0.22, 0.16, "Repeat over many S\n→ PACV   φᵢ\nGreedy: argmax MCV", BOX["readout"], fontsize=8)

    _arrow(ax, (zu[0] + 0.08, zu[1] - 0.04), (dots[0] - 0.10, dots[1] + 0.04))
    _arrow(ax, (zi[0] + 0.08, zi[1]), (dots[0] - 0.10, dots[1] - 0.02))
    _arrow(ax, (s[0] + 0.08, s[1] + 0.04), (dots[0] - 0.08, dots[1] - 0.08), rad=-0.12)
    _arrow(ax, (dots[0] + 0.10, dots[1] + 0.04), (cov[0] - 0.11, cov[1] - 0.02))
    _arrow(ax, (cov[0] + 0.11, cov[1] + 0.04), (vs[0] - 0.075, vs[1]))
    _arrow(ax, (cov[0] + 0.11, cov[1] - 0.08), (mcv[0] - 0.075, mcv[1] + 0.04))
    _arrow(ax, (vs[0], vs[1] - 0.08), (mcv[0], mcv[1] + 0.10))
    _arrow(ax, (mcv[0] - 0.02, mcv[1] - 0.04), (pacv[0] + 0.10, pacv[1] + 0.08), rad=0.12)
    ax.text(
        0.28,
        0.12,
        "Same formulas if zᵢ is a draw from N(μᵢ, σᵢ² I).\nMCV then has a mean and a standard deviation.",
        fontsize=8,
        color=MUTED,
        va="center",
    )
    return _save(fig, path)


def write_architecture_figures() -> list[Path]:
    out = figures_dir("architecture")
    paths = [
        plot_system_dag(out / "system_dag.png"),
        plot_encoder(out / "encoder.png"),
        plot_valuation(out / "valuation.png"),
    ]
    for p in paths:
        print(f"Wrote {p}")
    return paths


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    write_architecture_figures()
