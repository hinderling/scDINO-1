"""UMAP hyper-parameter grid search and visualisation."""

from __future__ import annotations

import time
from itertools import product
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .embedding import compute_embedding


def run_umap_grid_search(
    feats: np.ndarray,
    param_grid: Dict[str, List],
    *,
    visualization: bool = True,
    umap_class=None,
) -> tuple[pd.DataFrame, list[np.ndarray | None]]:
    """Run UMAP for every combination in *param_grid*.

    Parameters
    ----------
    feats : np.ndarray
        Feature matrix ``(n_samples, n_features)``.
    param_grid : dict
        Must contain ``"n_neighbors"`` and ``"min_dist"`` lists.
    visualization : bool
        If ``True``, call :func:`plot_grid_embeddings` after the sweep.
    umap_class : class, optional
        Explicit UMAP class forwarded to :func:`compute_embedding`.

    Returns
    -------
    results_df : pd.DataFrame
        One row per combination with timing and status.
    embeddings : list[np.ndarray | None]
        Corresponding embedding arrays (``None`` on failure).
    """
    param_combinations = list(
        product(param_grid["n_neighbors"], param_grid["min_dist"])
    )
    print(f"Testing {len(param_combinations)} parameter combinations...\n")
    print("=" * 60)

    results: list[dict] = []
    embeddings: list[np.ndarray | None] = []

    for i, (n_neighbors, min_dist) in enumerate(param_combinations, 1):
        start = time.time()
        try:
            embedding = compute_embedding(
                feats,
                topometry=False,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=42,
                umap_class=umap_class,
            )
            elapsed = time.time() - start
            results.append(
                {
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "time_seconds": elapsed,
                    "status": "Success",
                }
            )
            embeddings.append(embedding)
        except Exception as e:
            elapsed = time.time() - start
            results.append(
                {
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "time_seconds": elapsed,
                    "status": f"Failed: {e}",
                }
            )
            embeddings.append(None)

    results_df = pd.DataFrame(results)

    if visualization:
        plot_grid_embeddings(results_df, embeddings, param_grid)

    return results_df, embeddings


def plot_grid_embeddings(
    results_df: pd.DataFrame,
    embeddings: list[np.ndarray | None],
    param_grid: Dict[str, List],
    *,
    ax_grid: np.ndarray | None = None,
) -> None:
    """Plot all grid-search embeddings in a matplotlib subplot grid.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of :func:`run_umap_grid_search`.
    embeddings : list
        Embedding arrays aligned with *results_df* rows.
    param_grid : dict
        The parameter grid used for the search.
    ax_grid : np.ndarray, optional
        Pre-allocated 2-D array of ``matplotlib.axes.Axes``.  If ``None`` a new
        figure is created.
    """
    n_neighbors_list = param_grid["n_neighbors"]
    min_dist_list = param_grid["min_dist"]
    n_rows = len(n_neighbors_list)
    n_cols = len(min_dist_list)

    if ax_grid is None:
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows), num="UMAP Grid-Search"
        )
    else:
        axes = ax_grid

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    param_map = {
        (row["n_neighbors"], row["min_dist"]): (i, row)
        for i, row in results_df.iterrows()
    }

    for row_idx, n_neighbors in enumerate(n_neighbors_list):
        for col_idx, min_dist in enumerate(min_dist_list):
            ax = axes[row_idx, col_idx]

            if (n_neighbors, min_dist) in param_map:
                idx, result = param_map[(n_neighbors, min_dist)]
                embedding = embeddings[idx]

                if embedding is not None and result["status"] == "Success":
                    ax.scatter(
                        embedding[:, 0],
                        embedding[:, 1],
                        s=0.1,
                        alpha=0.5,
                        rasterized=True,
                    )
                    ax.set_title(
                        f"k={n_neighbors}, dist={min_dist} "
                        f"(T: {result['time_seconds']:.2f}s)",
                        fontsize=8,
                    )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "FAILED",
                        ha="center",
                        va="center",
                        fontsize=20,
                        color="red",
                        transform=ax.transAxes,
                    )
                    ax.set_title(
                        f"k={n_neighbors}, dist={min_dist} (Failed)", fontsize=10
                    )
            else:
                ax.set_title(
                    f"k={n_neighbors}, dist={min_dist} (Missing)",
                    fontsize=10,
                    color="gray",
                )

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()
