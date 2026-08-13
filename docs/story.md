# What the model found

A title is not worth the number of people who rated it. It is worth how
much **new audience-preference coverage** it adds given a catalog. That
sentence survives four phases of evidence: a neural embedding of who
likes what, a content encoder that knows what a movie *is*, maps of
real US catalogs, and a portfolio-adjusted value that averages over
many catalogs instead of one.

This is coverage, not a licensing recommendation. MovieLens missingness
is not dislike. A TMDB catalog is not a homepage.

```bash
uv run python -m catalog_value phase-a
uv run python -m catalog_value phase-b
uv run python -m catalog_value phase-c
uv run python -m catalog_value phase-d
```

## The map is not the model

Each title is a learned 64-dimensional vector $z_i$ inside a
taste-token encoder: an embedding table trained with eight queries that
attend over a user's title *set*. PCA below is a camera. Neighbors,
$V(S)$, and MCV all live in the 64-d space.

![Atlas of learned title embeddings](figures/phase_a/atlas.png)

PC1 (36% of variance) runs roughly **prestige / cinephile ← → flop /
multiplex leftover**. *The Godfather*, *Shawshank*, *Paths of Glory*,
and *Planet Earth* sit on the left. *Super Mario Bros.* is isolated on
the right. *The Notebook* drops on PC2. Genre is visible and is not the
map: same-genre cosine 0.13, cross-genre 0.06.

The orange rings are the 500 most-rated titles. They occupy the dense
middle. Being far from that blob is not automatically valuable — it
depends *which way* you are far.

## “Nearby” means “the same people liked these”

![Collaborative neighbors for four probes](figures/phase_a/neighbors.png)

- **Super Mario Bros.** lives with *Striptease*, *Anaconda*, *Police
  Academy 6*. Widely rated, widely disliked, substitutable with other
  widely rated dislikes.
- **Paths of Glory** lives with *Paris, Texas*, *The Third Man*, *The
  Lives of Others*. A cinephile neighborhood the popular catalog
  under-covers.
- **Halloween** lives with *Speed*, *American Pie*, *Jumanji*. Not
  other slashers. 1990s multiplex co-rating.
- **Toy Story** lives with *Up* and then *Saving Private Ryan* and
  *The Matrix*. Not “animation.” The cluster of titles a huge slice of
  MovieLens loved.

The embedding is a map of **shared raters**. When two titles are close,
covering one taste already covers a lot of the other.

## Popularity is not portfolio value

Against that top-500 catalog, MCV among the next 1,500 titles has
Spearman **0.07** with rating count.

![Popularity versus marginal catalog value](figures/phase_a/figure1_popularity_vs_mcv.png)

![MCV painted on the same embedding map](figures/phase_a/mcv_landscape.png)

High-MCV titles are the left-hand cinephile region: *Paths of Glory*
(0.0029), *Barry Lyndon*, *In the Mood for Love*. High-pop / low-MCV is
the flop neighborhood: *Super Mario Bros.* ($4\times 10^{-5}$),
*Showgirls*, *Batman & Robin*. Isolation on the right is the “people
rated this and did not like it” direction, so affinity stays low and
marginal coverage stays ~0.

The coverage function that produces those numbers is a soft-OR inside
each of eight tastes, $\tau = 0.5$:

```math
V_u(S)=\sum_k \pi_{uk}\,\tau\log\Bigl(1+\sum_{i\in S}e^{a_{uki}/\tau}\Bigr)
```

A Die Hard stack flattens. A mixed stack keeps climbing.

![Diverse titles keep adding coverage](figures/phase_a/diminishing_returns.png)

Five near-substitutes reach $V \approx 1.12$. Five distinct tastes
reach **1.51**. The architecture *can* represent eight interests; in
this checkpoint the mixing weights $\pi_u$ are still almost uniform
(entropy 2.077 vs $\log 8 = 2.079$). The coverage function is doing
the work more than the mixture.

## Content changes what “nearby” means

Collaborative $z_i$ cannot see a title with no ratings. Phase B trains
an MLP from MovieLens **genome tags + genre + year** to the same 64-d
space, then shrinks toward that prior:

```math
\mu_i = w_i\, z_i^{\mathrm{collab}} + (1-w_i)\, f(X_i),
\qquad w_i = n_i / (n_i + n_0)
```

Held-out titles get $w_i = 0$: a simulated cold start. Mean cosine with
collaborative $z$ is 0.24 on train, 0.23 on holdout — the vectors are
short, so angles are noisy — but **MCV rank is preserved** (Spearman
**0.79** on held-out candidates). The geometry that matters for
portfolio value survives the drop to content.

The qualitative jump is in the neighbors.

![Collaborative vs content neighbors](figures/phase_b/neighbor_swap.png)

| Probe | Collaborative “who rated this” | Content “what this is” |
| --- | --- | --- |
| *Halloween* | *Speed*, *American Pie*, *Jumanji* | *Texas Chainsaw Massacre*, *The Fog*, *Evil Dead*, *The Omen* |
| *The Notebook* | *The Italian Job*, *Furious 7*, *Pirates* | *Little Women*, *When Harry Met Sally*, *The Fault in Our Stars* |
| *Paths of Glory* | cinephile co-rating | *Das Boot*, *Full Metal Jacket*, *The Deer Hunter* |
| *Super Mario Bros.* | flop cluster (*Striptease*, *Anaconda*) | video-game / kids-flop cluster (*Street Fighter*, *Mortal Kombat: Annihilation*) |

That is Phase B’s result: **cold-start embeddings become semantic
without throwing away MCV rankings.** *Halloween* is finally a horror
film. *The Notebook* is finally a romance. The flop is still a flop,
just a more coherent one.

![Content MCV vs collaborative MCV on held-out titles](figures/phase_b/mcv_transfer.png)

Content systematically **compresses the high end**: *Double Indemnity*
and *12 Years a Slave* are more valuable collaboratively than their
tags admit. *Eraserhead* is the reverse — content thinks it covers a
gap that collaborative raters did not treat as high-affinity. Those
disagreements are the interesting cold-start errors, not a failure of
the encoder.

Sampling $z_i \sim \mathcal{N}(\mu_i, \sigma_i^2 I)$ turns MCV into a
mean and a standard deviation. Titles with no collaborative weight are
the uncertain ones.

![Posterior uncertainty of MCV](figures/phase_b/mcv_uncertainty.png)

## Catalogs barely share titles. They still sit on the same map.

US flatrate catalogs from TMDB, intersected with MovieLens (not full
2026 libraries):

| Service | $\|S \cap \mathrm{ML}\|$ | $V(S)$ | $V$ per title |
| --- | ---: | ---: | ---: |
| Prime Video | 1,807 | 4.01 | 0.0022 |
| Max | 617 | 3.59 | 0.0058 |
| Netflix | 486 | 3.32 | 0.0068 |
| Disney+ | 508 | 3.20 | 0.0063 |
| Hulu | 174 | 2.83 | 0.016 |

Prime is 10× Hulu in titles and ~1.4× in coverage. Hybrid posterior
means shave every $V(S)$ by ~5% and do not change the order.

![Where each catalog sits in title space](figures/phase_c/occupancy.png)

![Title Jaccard vs map occupancy](figures/phase_c/overlap.png)

Title Jaccard is **≤ 0.07**. These libraries almost do not overlap.
Occupancy cosine on the embedding map is 0.68–0.89: they still live in
the same popular-middle of $z$, with Disney+/Max pulling different
tails and Hulu the sparsest. Taste-fingerprint cosine is identically 1
— a limit of this checkpoint, not a discovery about streaming. $\pi_u$
never specialized, so every catalog “covers” the eight queries the
same way. The map occupancy is the honest geographic statement;
the collapsed fingerprints are a warning not to over-read $K$.

## Averaging over catalogs does not rescue popularity

PACV $\phi_i = \mathbb{E}_S[\mathrm{MCV}_i(S)]$ averages MCV over 80
random catalogs of size 60. The ranking does not become a popularity
list. *Paths of Glory* still leads. The flop cluster still sits on the
floor.

![PACV versus popularity](figures/phase_d/pacv.png)

Building a 40-title catalog by greedy MCV reaches $V = 2.55$; the 40
most-rated titles reach **2.47**. The lift is modest because both
pools are already well-observed hits. The composition change is not
modest: popularity stacks Action/Adventure; greedy MCV doubles Comedy,
Crime, and Drama and cuts Action nearly in half.

![Greedy MCV vs popularity growth](figures/phase_d/greedy_vs_popular.png)

![Genre mix of greedy vs popularity catalogs](figures/phase_d/genre_mix.png)

Swap the backbone and the *ranking* of MCV is stable: Spearman **0.89**
vs SVD (separate audience + titles), **0.84** vs the content encoder.
Absolute scale is not. SVD compresses MCV; content overvalues the low
end and undervalues the cinephile high end. The story “these titles
are the gap” is robust. The number on the axis is not.

![MCV under neural vs SVD vs content](figures/phase_d/ablation.png)

## What this still is not

- **Not eight named tastes.** Queries are orthogonal; $\pi_u$ is
  uniform; platform fingerprints collapsed. Longer training or a
  catalog-value loss would have to split them.
- **Not plot semantics in the collaborative $z$.** That required
  genome tags. Overviews, posters, and trailers are still unused.
- **Not a full streaming library.** TMDB US flatrate $\cap$ MovieLens
  core, mostly pre-2020 theatrical titles.
- **Not complementarity.** Log-sum-exp is substitution. Universes and
  “watch A then B” are out of scope.
- **Not retention, revenue, or a buy list.** Audience-preference
  coverage only.

The sentence that survives: **collaborative geometry already separates
popularity from marginal coverage; content makes that geometry
legible as genre; catalogs occupy overlapping regions of it even when
they share no titles; averaging over catalogs does not turn the flop
cluster into a gap.** The titles that keep showing up as high-value —
*Paths of Glory*, *Notorious*, *The Lives of Others* — sit in a region
of $z$ that a popularity stack leaves open. The titles that keep
showing up as worthless — *Super Mario Bros.*, *Batman & Robin*,
*Showgirls* — sit where many people already told the model they were
done.
