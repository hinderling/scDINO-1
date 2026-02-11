"""Density contour helpers for plotly figure widgets."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter

from .colors import hex_to_rgb


def add_global_density_contour(
    fig: go.FigureWidget,
    data,
    x_range: list[float],
    y_range: list[float],
) -> None:
    """Add a single global density contour to *fig*.

    Parameters
    ----------
    fig : plotly FigureWidget
    data : pd.DataFrame
        Must have ``umap_1`` and ``umap_2`` columns.
    x_range, y_range : list
        ``[min, max]`` ranges for the histogram.
    """
    if len(data) < 100:
        return

    x = data["umap_1"].values
    y = data["umap_2"].values

    H, xedges, yedges = np.histogram2d(
        x, y, bins=100, range=[x_range, y_range], density=True
    )
    H = gaussian_filter(H, sigma=1.5)
    H = np.ma.masked_where(H < H.max() * 0.1, H)

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    fig.add_trace(
        go.Contour(
            x=xcenters,
            y=ycenters,
            z=H.T,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "black"]],
            line=dict(width=2, color="black"),
            contours=dict(
                coloring="lines",
                showlabels=False,
                start=0,
                end=float(H.max()),
                size=float(H.max()) / 10,
            ),
            opacity=0.7,
            showscale=False,
            name="Global density",
            hoverinfo="skip",
        )
    )


def add_per_phenotype_density(
    fig: go.FigureWidget,
    data,
    color_map: dict[str, str],
    x_range: list[float],
    y_range: list[float],
) -> None:
    """Add per-phenotype density contour traces to *fig*.

    Parameters
    ----------
    fig : plotly FigureWidget
    data : pd.DataFrame
        Must have ``umap_1``, ``umap_2``, and ``phenotype`` columns.
    color_map : dict
        Phenotype name -> hex colour.
    x_range, y_range : list
        ``[min, max]`` ranges for the histogram.
    """
    if len(data) < 100:
        return

    H_combined = []
    for pheno in color_map:
        sub = data[data["phenotype"] == pheno]
        if len(sub) < 30:
            continue
        H_sub, _, _ = np.histogram2d(
            sub["umap_1"].values,
            sub["umap_2"].values,
            bins=80,
            range=[x_range, y_range],
            density=True,
        )
        H_sub = gaussian_filter(H_sub, sigma=2.0)
        H_combined.append(H_sub)

    if not H_combined:
        return
    H_max_global = max(H.max() for H in H_combined)
    if H_max_global == 0:
        return

    for pheno, color in color_map.items():
        sub = data[data["phenotype"] == pheno]
        if len(sub) < 30:
            continue

        H, xedges, yedges = np.histogram2d(
            sub["umap_1"].values,
            sub["umap_2"].values,
            bins=80,
            range=[x_range, y_range],
            density=True,
        )
        H = gaussian_filter(H, sigma=2.0)
        H_normalized = H / H_max_global
        mask_threshold = (H.max() * 0.05) / H_max_global
        H_masked = np.ma.masked_where(H_normalized < mask_threshold, H_normalized)

        R, G, B = hex_to_rgb(color)
        min_alpha = 0.3
        colorscale = [
            [0.0, "rgba(0,0,0,0)"],
            [0.0001, f"rgba({R},{G},{B},{min_alpha})"],
            [1.0, color],
        ]

        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2

        fig.add_trace(
            go.Contour(
                x=xcenters,
                y=ycenters,
                z=H_masked.T,
                colorscale=colorscale,
                line=dict(color=color, width=2.5),
                contours=dict(coloring="lines", showlabels=False),
                opacity=1.0,
                showscale=False,
                name=f"{pheno} density",
                hoverinfo="skip",
            )
        )
