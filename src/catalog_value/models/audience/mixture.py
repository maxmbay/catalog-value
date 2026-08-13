from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans

from catalog_value.models.content.svd import CollaborativeFit
from catalog_value.models.types import AudienceStates

FloatArray = NDArray[np.float64]


def fit_item_prototypes(item_factors: FloatArray, n_interests: int, seed: int) -> FloatArray:
    """Global taste prototypes as k-means centers in item latent space."""
    n_items = item_factors.shape[0]
    n_interests = min(n_interests, n_items)
    km = MiniBatchKMeans(
        n_clusters=n_interests,
        random_state=seed,
        batch_size=min(4096, n_items),
        n_init=10,
    )
    km.fit(item_factors)
    return km.cluster_centers_.astype(np.float64)


def build_audience_states(
    ratings: csr_matrix,
    fit: CollaborativeFit,
    prototypes: FloatArray,
) -> AudienceStates:
    """Per-user mixture over K tastes.

    Each rated item is assigned to its nearest global prototype. Mixing weights
    are rating mass in that cluster; taste vectors are rating-weighted means of
    the user's items in the cluster (prototype used if the user has no mass).
    """
    item_factors = fit.item_factors
    n_users = ratings.shape[0]
    n_interests, dim = prototypes.shape

    distances = (
        np.sum(item_factors**2, axis=1, keepdims=True)
        - 2.0 * item_factors @ prototypes.T
        + np.sum(prototypes**2, axis=1)[None, :]
    )
    item_cluster = np.argmin(distances, axis=1).astype(np.int64)

    pi = np.zeros((n_users, n_interests), dtype=np.float64)
    z = np.zeros((n_users, n_interests, dim), dtype=np.float64)

    for k in range(n_interests):
        items_k = np.flatnonzero(item_cluster == k)
        if items_k.size == 0:
            z[:, k, :] = prototypes[k]
            continue
        r_k = ratings[:, items_k]
        mass = np.asarray(r_k.sum(axis=1)).ravel()
        pi[:, k] = mass
        weighted = r_k @ item_factors[items_k]
        nonempty = mass > 0
        z[nonempty, k, :] = weighted[nonempty] / mass[nonempty, None]
        z[~nonempty, k, :] = prototypes[k]

    row_mass = pi.sum(axis=1, keepdims=True)
    row_mass = np.maximum(row_mass, 1e-8)
    pi = pi / row_mass

    return AudienceStates(
        pi=pi,
        z=z,
        user_row=np.arange(n_users, dtype=np.int64),
    )


def subsample_audience(states: AudienceStates, n: int, seed: int) -> AudienceStates:
    n_users = states.pi.shape[0]
    n = min(n, n_users)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n_users, size=n, replace=False))
    return AudienceStates(pi=states.pi[idx], z=states.z[idx], user_row=states.user_row[idx])


def save_audience(states: AudienceStates, prototypes: FloatArray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pi=states.pi,
        z=states.z,
        user_row=states.user_row,
        prototypes=prototypes,
    )


def load_audience(path: Path) -> tuple[AudienceStates, FloatArray]:
    payload = np.load(path)
    states = AudienceStates(
        pi=payload["pi"],
        z=payload["z"],
        user_row=payload["user_row"],
    )
    return states, payload["prototypes"]
