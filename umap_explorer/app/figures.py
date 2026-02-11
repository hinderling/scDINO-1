"""Build the main Scattergl figure from server state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from ..visualization.density import add_global_density_contour, add_per_phenotype_density
from . import theme

if TYPE_CHECKING:
    from .app import ServerState

# Threshold above which we skip customdata to keep the figure lightweight
_CUSTOMDATA_THRESHOLD = 200_000


def build_main_figure(
    state: ServerState,
    *,
    color_by: str = "phenotype",
    opacity: float = 0.7,
    point_size: int = 4,
    show_density: bool = False,
    per_phenotype_density: bool = False,
    show_trajectories: bool = False,
    trajectory_color_mode: str = "Track",
    drag_mode: str = "zoom",
) -> go.Figure:
    """Build the complete UMAP figure.

    Parameters
    ----------
    state : ServerState
        Current server state (provides df_filtered, color_map, selected_tracks).
    color_by : str
        Column to use for point colors.
    opacity : float
        Marker opacity.
    point_size : int
        Marker size in px.
    show_density, per_phenotype_density : bool
        Density contour flags.
    show_trajectories : bool
        Whether to draw trajectory lines for selected tracks.
    trajectory_color_mode : str
        One of "Track", "Time", "Phenotype", "Apo Status".
    drag_mode : str
        Plotly drag mode.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    scatter_data = state.df_filtered

    if scatter_data.empty:
        fig.update_layout(_base_layout(drag_mode))
        return fig

    # Ensure color column exists
    if color_by not in scatter_data.columns:
        color_by = "phenotype"

    color_values, color_mapping, unique_labels = state.color_map(scatter_data[color_by])

    # Build customdata + hovertemplate only for manageable sizes
    n = len(scatter_data)
    if n < _CUSTOMDATA_THRESHOLD:
        customdata = np.column_stack((
            scatter_data["track_id"].values,
            scatter_data["original_track_id"].values if "original_track_id" in scatter_data.columns
            else scatter_data["track_id"].values,
            scatter_data["t"].values if "t" in scatter_data.columns
            else np.full(n, ""),
            scatter_data[color_by].values,
        ))
        hovertemplate = (
            "<b>Track:</b> %{customdata[0]}<br>"
            "<b>Orig:</b> %{customdata[1]}<br>"
            "<b>UMAP:</b> (%{x:.3f}, %{y:.3f})<br>"
            + ("<b>t:</b> %{customdata[2]}<br>" if "t" in scatter_data.columns else "")
            + f"<b>{color_by}:</b> %{{customdata[3]}}<extra></extra>"
        )
    else:
        customdata = None
        hovertemplate = "<b>UMAP:</b> (%{x:.3f}, %{y:.3f})<extra></extra>"

    fig.add_trace(go.Scattergl(
        x=scatter_data["umap_1"].values,
        y=scatter_data["umap_2"].values,
        mode="markers",
        name="Points",
        showlegend=False,
        marker=dict(
            size=point_size,
            opacity=opacity,
            color=color_values,
        ),
        customdata=customdata,
        hovertemplate=hovertemplate,
    ))

    # Legend traces
    legend_labels = [l for l in sorted(set(scatter_data[color_by].dropna().astype(str))) if l != "unlabeled"]
    for label in legend_labels:
        fig.add_trace(go.Scattergl(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=10, color=color_mapping.get(label, "gray")),
            name=label,
            showlegend=True,
        ))

    # Density contours
    if show_density and not scatter_data.empty:
        x_min, x_max = scatter_data["umap_1"].min(), scatter_data["umap_1"].max()
        y_min, y_max = scatter_data["umap_2"].min(), scatter_data["umap_2"].max()
        x_range = [x_min - 0.5, x_max + 0.5]
        y_range = [y_min - 0.5, y_max + 0.5]
        if per_phenotype_density:
            _, pheno_map, _ = state.color_map(scatter_data["phenotype"])
            add_per_phenotype_density(fig, scatter_data, pheno_map, x_range, y_range)
        else:
            add_global_density_contour(fig, scatter_data, x_range, y_range)

    # Track overlays
    _add_track_overlays(
        fig, state, show_trajectories=show_trajectories,
        color_mode=trajectory_color_mode,
    )

    fig.update_layout(_base_layout(drag_mode))
    return fig


def _base_layout(drag_mode: str = "zoom") -> dict:
    """Return common layout kwargs."""
    return dict(
        dragmode=drag_mode,
        uirevision="constant",
        hovermode="closest",
        hoverdistance=10,
        paper_bgcolor=theme.BASE3,
        plot_bgcolor=theme.BASE3,
        font=dict(family=theme.FONT_STACK, size=11, color=theme.BASE00),
        margin=dict(l=40, r=10, t=10, b=40),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            title="UMAP 1",
            color=theme.BASE00,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            title="UMAP 2",
            color=theme.BASE00,
        ),
        legend=dict(
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
    )


def _add_track_overlays(
    fig: go.Figure,
    state: ServerState,
    *,
    show_trajectories: bool,
    color_mode: str,
) -> None:
    """Add selected-track marker/line overlays to fig."""
    if not state.selected_tracks:
        return

    _, pheno_map, _ = state.color_map(state.df["phenotype"])

    for track_id in state.selected_tracks:
        track_data = state.df_filtered[state.df_filtered["track_id"] == track_id]
        if track_data.empty:
            continue
        track_data = track_data.sort_values("t") if "t" in track_data.columns else track_data

        if color_mode == "Track":
            color = px.colors.qualitative.Plotly[
                state.selected_tracks.index(track_id) % 10
            ]
            marker_colors: str | list = color
            line_color = color
        elif color_mode == "Time" and "t" in track_data.columns:
            t_norm = (track_data["t"] - track_data["t"].min()) / (
                track_data["t"].max() - track_data["t"].min() + 1e-8
            )
            marker_colors = px.colors.sample_colorscale("viridis", t_norm.tolist())
            line_color = "gray"
        elif color_mode == "Phenotype":
            marker_colors = [
                pheno_map.get(str(p), "gray") for p in track_data["phenotype"]
            ]
            line_color = "gray"
        elif color_mode == "Apo Status":
            label = track_data["label_manual"].iloc[0] if "label_manual" in track_data.columns else ""
            has_apo = isinstance(label, str) and "apo" in label.lower()
            color = "red" if has_apo else "cyan"
            marker_colors = color
            line_color = color
        else:
            marker_colors = "gray"
            line_color = "gray"

        # Always show markers for selected tracks
        fig.add_trace(go.Scattergl(
            x=track_data["umap_1"].values,
            y=track_data["umap_2"].values,
            mode="markers",
            marker=dict(
                size=max(7, 4 + 2),
                color=marker_colors,
                line=dict(width=2, color="white"),
            ),
            showlegend=False,
            hoverinfo="none",
        ))

        # Connect with lines if requested
        if show_trajectories and len(track_data) >= 2:
            hover_text = (
                track_data["t"].astype(str).tolist()
                if "t" in track_data.columns
                else [str(i) for i in range(len(track_data))]
            )
            fig.add_trace(go.Scattergl(
                x=track_data["umap_1"].values,
                y=track_data["umap_2"].values,
                mode="lines+markers",
                line=dict(color=line_color, width=1.8),
                marker=dict(
                    size=7,
                    color=marker_colors,
                    line=dict(width=1.2, color="white"),
                ),
                name=f"Track {track_id}",
                text=hover_text,
                hovertemplate=f"<b>Track {track_id}</b><br>t=%{{text}}<extra></extra>",
            ))
