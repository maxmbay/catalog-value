# Portfolio-Aware Content Valuation

The value of a title is contextual. It depends on the audiences it serves, the
needs already satisfied by the existing catalog, the titles with which it
substitutes or complements, and our uncertainty about its audience.

This repository estimates the distribution over **marginal audience coverage**
of adding or removing a title from a catalog, then uses those estimates to
construct counterfactual content portfolios.

The primary modeling target is

\[
p(\mathrm{MCV}_i(S) \mid D),\qquad \mathrm{MCV}_i(S) = V(S \cup \{i\}) - V(S)
\]

not \(P(\text{viewer watches } i)\) or \(E[\text{rating}]\) alone.

## What the model estimates

Audience **preference coverage**: how well a set of titles covers heterogeneous
latent viewing needs.

It does **not** estimate subscription retention, acquisition, ad revenue,
licensing cost, profit, causal engagement, or platform recommendation exposure.
MovieLens missingness is non-random; an unrated title is not a dislike.
Streaming availability is not homepage exposure.

Language such as "Netflix should license X" is out of scope. Prefer: *under the
model's audience-coverage objective, X has high estimated marginal value
relative to the observed Netflix-US catalog.*

## Four learned objects

1. **Audience** → mixture of latent demand states \(Z_u = \{(\pi_{uk}, z_{uk})\}\)
2. **Content** → probabilistic title representation \(z_i \sim p(z_i \mid D)\)
3. **Audience states + catalog** → catalog utility \(V(S)\)
4. **Catalog-utility posterior** → portfolio decisions

Phase A implements (1)–(3) with a collaborative SVD backbone, per-user taste
mixtures, and an analytical log-sum-exp coverage function. Later phases add
amortized content encoders, Bayesian title posteriors, complementarity,
real streaming catalogs, Shapley/PACV, and uncertainty-aware optimization.

See `docs/model.md` for the compact formalization.

## Setup

```bash
./setup/bootstrap.sh
```

Requires [uv](https://docs.astral.sh/uv/). Python 3.12.

## Phase A

Learn multi-interest user representations from MovieLens 25M, value a popular
catalog with a submodular coverage function, and plot **popularity vs MCV**.

```bash
uv run python -m catalog_value ingest
uv run python -m catalog_value fit
uv run python -m catalog_value figure1
```

Or all three:

```bash
uv run python -m catalog_value phase-a
```

Artifacts land in `outputs/phase_a/`. Tests (no MovieLens download required):

```bash
uv run pytest
```

## Repository layout

```text
configs/                experiment configs
data/{raw,intermediate,processed}/
docs/model.md           mathematical objects
src/catalog_value/
  data/                 MovieLens ingest
  models/audience/      multi-interest user states
  models/content/       collaborative (later: hybrid/content) title reps
  models/catalog_value/ coverage function, MCV
  optimization/         greedy / later: risk-aware portfolios
  visualization/
experiments/
tests/
```

## Phases

| Phase | Scientific question | Status |
| --- | --- | --- |
| A | Are popularity and MCV different? | in progress |
| B | Do probabilistic / content representations change MCV under cold start? | next |
| C | Do real U.S. streaming catalogs occupy different audience regions? | later |
| D | Ablations, PACV, fairness, risk frontiers, write-up | later |
