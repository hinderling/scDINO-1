"""Single canonical UMAP embedding computation."""

from __future__ import annotations

import numpy as np

from ._compat import get_umap_class


def compute_embedding(
    feats: np.ndarray,
    *,
    use_gpu: bool = True,
    topometry: bool = False,
    n_neighbors: int = 50,
    n_components: int = 2,
    min_dist: float = 0.1,
    random_state: int = 42,
    umap_class=None,
) -> np.ndarray:
    """Compute a 2-D UMAP embedding, optionally via TopOMetry.

    This is the **single** implementation that replaces the three duplicates
    previously scattered across the notebook.

    Parameters
    ----------
    feats : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    use_gpu : bool
        Prefer GPU-accelerated UMAP (cuML) when available.
    topometry : bool
        Apply TopOMetry graph enhancement before UMAP.
    n_neighbors, n_components, min_dist, random_state
        Standard UMAP hyper-parameters.
    umap_class : class, optional
        Explicit UMAP class to use.  When ``None`` the best available class
        is auto-detected via :func:`get_umap_class`.

    Returns
    -------
    embedding : np.ndarray
        Array of shape ``(n_samples, n_components)``.
    """
    if topometry:
        import topo as tp

        print("Applying TopOMetry to enhance graph...")
        tg = tp.TopOGraph(
            n_eigs=50,
            base_knn=n_neighbors,
            graph_knn=n_neighbors,
            random_state=random_state,
            n_jobs=-1,
        )
        tg.fit(feats)
        tg.transform(feats)

        kernel_obj = list(tg.GraphKernelDict.values())[0]
        K = kernel_obj.K.astype(float)
        K = K / K.max()

        D = K.copy()
        D.data = 1.0 - D.data
        D.setdiag(0)
        D.eliminate_zeros()
        data_for_umap = D.toarray()
        metric = "precomputed"
        # TopOMetry forces CPU UMAP
        UMAP_Class = umap_class or get_umap_class(prefer_gpu=False)
    else:
        print("Running UMAP directly on features...")
        data_for_umap = feats
        metric = "cosine"
        UMAP_Class = umap_class or get_umap_class(prefer_gpu=use_gpu)

    reducer = UMAP_Class(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(data_for_umap)
    return embedding
