# Model objects

This file is the compact mapping from the research brief to code. Implementation
may approximate computation; it should not collapse these objects.

## Catalog value

For catalog \(S\) and user \(u\) with tastes \(Z_u = \{(\pi_{uk}, z_{uk})\}_{k=1}^{K}\):

\[
a_{uki} = z_{uk}^\top z_i + b_i
\]

\[
V_u(S) = \sum_k \pi_{uk}\, \tau \log\Bigl(1 + \sum_{i \in S} e^{a_{uki}/\tau}\Bigr)
\]

\[
V(S) = \mathbb{E}_u[V_u(S)]
\]

The log-sum-exp coverage function is monotone and submodular in \(S\). Later
phases replace or augment it with a learned set function and a complementarity
term.

## Marginal content value

\[
\mathrm{MCV}_i(S) = V(S \cup \{i\}) - V(S)
\]

The modeling target is \(p(\mathrm{MCV}_i(S) \mid D)\), not \(P(\text{watch})\)
or \(E[\text{rating}]\) alone. Phase A reports a point estimate; later phases
expose \(\mathbb{E}[\mathrm{MCV}]\) and \(\mathrm{Var}(\mathrm{MCV})\).

## Portfolio-adjusted value

\[
\phi_i = \mathbb{E}_S[\mathrm{MCV}_i(S)]
\]

Estimated by Monte Carlo over catalog contexts (Shapley-style). Not required
for Phase A.

## What this is not

Audience-preference coverage is not retention, licensing profit, causal
engagement, or a recommendation to license a title.
