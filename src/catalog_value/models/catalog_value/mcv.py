from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from catalog_value.models.catalog_value.analytical import catalog_value, logsumexp_with_zero
from catalog_value.models.types import AudienceStates, CatalogValueEstimate, TitleReps

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def affinity(audience: AudienceStates, titles: TitleReps, item_index: IntArray) -> FloatArray:
    """a_uki = z_ukᵀ z_i + b_i, shape [n_users, n_interests, n_items]."""
    z_items = titles.z[item_index]
    bias = titles.bias[item_index]
    dots = np.einsum("ukd,nd->ukn", audience.z, z_items)
    return dots + bias[None, None, :]


def value_of_catalog(
    audience: AudienceStates,
    titles: TitleReps,
    catalog: IntArray,
    tau: float,
) -> CatalogValueEstimate:
    aff = affinity(audience, titles, np.asarray(catalog, dtype=np.int64))
    per_user = catalog_value(audience.pi, aff, tau)
    return CatalogValueEstimate(mean=float(per_user.mean()), per_user=per_user)


def marginal_content_value(
    audience: AudienceStates,
    titles: TitleReps,
    catalog: IntArray,
    candidates: IntArray,
    tau: float,
) -> FloatArray:
    """MCV_i(S) = E_u[V_u(S ∪ {i}) − V_u(S)] for each candidate.

    Candidates already in S contribute 0. Uses a cached log(1 + Σ exp(a/τ))
    on S so each candidate is a logaddexp increment per (user, taste).
    """
    catalog = np.asarray(catalog, dtype=np.int64)
    candidates = np.asarray(candidates, dtype=np.int64)
    catalog_set = set(int(i) for i in catalog)

    aff_s = affinity(audience, titles, catalog)
    log_one_plus_a = logsumexp_with_zero(aff_s / tau)
    v_s = (audience.pi * (tau * log_one_plus_a)).sum(axis=-1)

    aff_c = affinity(audience, titles, candidates)
    log_one_plus_a_new = np.logaddexp(log_one_plus_a[..., None], aff_c / tau)
    v_new = (audience.pi[..., None] * (tau * log_one_plus_a_new)).sum(axis=1)
    mcv = v_new.mean(axis=0) - float(v_s.mean())

    already = np.array([int(i) in catalog_set for i in candidates], dtype=bool)
    return np.where(already, 0.0, mcv).astype(np.float64)
