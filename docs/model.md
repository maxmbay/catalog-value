# Model objects

This file is the compact mapping from the research brief to code. Implementation
may approximate computation; it should not collapse these objects.

## Catalog value

For catalog $S$ and user $u$ with tastes $`Z_u = \{(\pi_{uk}, z_{uk})\}_{k=1}^{K}`$:

```math
a_{uki} = z_{uk}^\top z_i + b_i
```

```math
V_u(S) = \sum_k \pi_{uk}\, \tau \log\Bigl(1 + \sum_{i \in S} e^{a_{uki}/\tau}\Bigr)
```

```math
V(S) = \mathbb{E}_u[V_u(S)]
```

The log-sum-exp coverage function is monotone and submodular in $S$. Later
phases replace or augment it with a learned set function and a complementarity
term.

## Marginal content value

```math
\mathrm{MCV}_i(S) = V(S \cup \{i\}) - V(S)
```

The modeling target is $`p(\mathrm{MCV}_i(S) \mid D)`$, not $P(\text{watch})$
or $E[\text{rating}]$ alone. Phase A reports a point estimate. Phase B
puts an isotropic Gaussian around each $z_i$ (content prior, collaborative
likelihood) and reports $\mathbb{E}[\mathrm{MCV}]$ and
$\mathrm{Std}(\mathrm{MCV})$ from posterior draws.

## Portfolio-adjusted value

```math
\phi_i = \mathbb{E}_S[\mathrm{MCV}_i(S)]
```

Estimated by Monte Carlo over catalog contexts (Shapley-style). Phase D
reports $\phi_i$ for a probe set of high- and low-MCV titles.

## What this is not

Audience-preference coverage is not retention, licensing profit, causal
engagement, or a recommendation to license a title.
