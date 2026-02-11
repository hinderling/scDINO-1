"""Trajectory visualisation functions for track-based UMAP data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def _extract_coords(track_data, umap_col):
    """Extract x, y arrays from *track_data* given *umap_col* spec."""
    if isinstance(umap_col, (list, tuple)):
        return track_data[umap_col[0]].values, track_data[umap_col[1]].values
    coords = np.array(track_data[umap_col].tolist())
    return coords[:, 0], coords[:, 1]


def plot_track_trajectories(
    df,
    umap_col=("umap_1", "umap_2"),
    track_id_col="track_id",
    time_col="t",
    figsize=(12, 10),
    max_tracks=None,
    show_arrows=True,
    show_points=True,
    ax=None,
):
    """Plot UMAP with trajectories showing temporal connections within tracks."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    unique_tracks = df[track_id_col].unique()
    if max_tracks is not None:
        unique_tracks = np.random.choice(unique_tracks, min(max_tracks, len(unique_tracks)), replace=False)

    cmap = plt.cm.magma
    colors = cmap(np.linspace(0.2, 0.8, len(unique_tracks)))

    for track_idx, track_id in enumerate(unique_tracks):
        track_data = df[df[track_id_col] == track_id].sort_values(time_col)
        x, y = _extract_coords(track_data, umap_col)

        if show_points:
            ax.scatter(x, y, c=[colors[track_idx]], s=20, alpha=0.6, zorder=2)

        if len(x) > 1:
            ax.plot(x, y, c=colors[track_idx], alpha=0.4, linewidth=1.5, zorder=1)
            if show_arrows:
                for i in range(0, len(x) - 1, max(1, len(x) // 5)):
                    dx = x[i + 1] - x[i]
                    dy = y[i + 1] - y[i]
                    ax.arrow(
                        x[i], y[i], dx * 0.8, dy * 0.8,
                        head_width=0.3, head_length=0.2,
                        fc=colors[track_idx], ec=colors[track_idx],
                        alpha=0.6, zorder=3,
                    )

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.grid(True, alpha=0.3)

    if ax.figure is not None:
        plt.tight_layout()
    return ax


def plot_movement_intensity(
    df,
    umap_col=("umap_1", "umap_2"),
    track_id_col="track_id",
    time_col="t",
    figsize=(12, 10),
    ax=None,
):
    """Visualise movement intensity (speed) along trajectories using colour gradients."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    unique_tracks = df[track_id_col].unique()
    all_disp: list[float] = []

    for track_id in unique_tracks:
        track_data = df[df[track_id_col] == track_id].sort_values(time_col)
        x, y = _extract_coords(track_data, umap_col)
        if len(x) < 2:
            continue
        all_disp.extend(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))

    vmin, vmax = (np.min(all_disp), np.max(all_disp)) if all_disp else (0, 1)
    norm = plt.Normalize(vmin, vmax)

    for track_id in unique_tracks:
        track_data = df[df[track_id_col] == track_id].sort_values(time_col)
        x, y = _extract_coords(track_data, umap_col)
        if len(x) < 2:
            continue
        displacements = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap="hot", norm=norm, alpha=0.7, linewidth=2)
        lc.set_array(displacements)
        ax.add_collection(lc)

    sm = plt.cm.ScalarMappable(cmap="hot", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Movement Intensity")

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title("Movement Intensity in UMAP Space", fontsize=14)
    ax.autoscale()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax


def plot_vector_field(
    df,
    umap_col=("umap_1", "umap_2"),
    track_id_col="track_id",
    time_col="t",
    grid_size=20,
    figsize=(12, 10),
    ax=None,
):
    """Create a vector field showing dominant movement directions."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    movements = []
    for track_id in df[track_id_col].unique():
        track_data = df[df[track_id_col] == track_id].sort_values(time_col)
        x, y = _extract_coords(track_data, umap_col)
        if len(x) < 2:
            continue
        for i in range(len(x) - 1):
            movements.append({"x": x[i], "y": y[i], "dx": x[i + 1] - x[i], "dy": y[i + 1] - y[i]})

    movements_df = pd.DataFrame(movements)
    if movements_df.empty:
        return ax

    x_bins = np.linspace(movements_df["x"].min(), movements_df["x"].max(), grid_size)
    y_bins = np.linspace(movements_df["y"].min(), movements_df["y"].max(), grid_size)

    grid_x, grid_y, grid_dx, grid_dy, grid_intensity = [], [], [], [], []

    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            mask = (
                (movements_df["x"] >= x_bins[i])
                & (movements_df["x"] < x_bins[i + 1])
                & (movements_df["y"] >= y_bins[j])
                & (movements_df["y"] < y_bins[j + 1])
            )
            if mask.sum() > 0:
                grid_x.append((x_bins[i] + x_bins[i + 1]) / 2)
                grid_y.append((y_bins[j] + y_bins[j + 1]) / 2)
                grid_dx.append(movements_df.loc[mask, "dx"].mean())
                grid_dy.append(movements_df.loc[mask, "dy"].mean())
                grid_intensity.append(np.sqrt(grid_dx[-1] ** 2 + grid_dy[-1] ** 2))

    ax.quiver(
        grid_x, grid_y, grid_dx, grid_dy, grid_intensity,
        cmap="viridis", scale=None, scale_units="xy", angles="xy", alpha=0.8,
    )
    ax.scatter(movements_df["x"].values, movements_df["y"].values, s=1, c="gray", alpha=0.2, zorder=1)
    ax.axis("off")
    plt.tight_layout()
    return ax


def plot_trajectories_by_time(
    df,
    umap_col=("umap_1", "umap_2"),
    track_id_col="track_id",
    time_col="t",
    figsize=(12, 10),
    max_tracks=None,
    show_points=False,
    ax=None,
):
    """Plot trajectories coloured by absolute time (blue=early, red=late)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    unique_tracks = df[track_id_col].unique()
    if max_tracks is not None:
        unique_tracks = unique_tracks[:max_tracks]

    t_min, t_max = df[time_col].min(), df[time_col].max()
    norm = plt.Normalize(t_min, t_max)

    for track_id in unique_tracks:
        track_data = df[df[track_id_col] == track_id].sort_values(time_col)
        x, y = _extract_coords(track_data, umap_col)
        t = track_data[time_col].values
        if len(x) < 2:
            continue
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        segment_times = (t[:-1] + t[1:]) / 2
        lc = LineCollection(segments, cmap="coolwarm", norm=norm, alpha=0.7, linewidth=2)
        lc.set_array(segment_times)
        ax.add_collection(lc)
        if show_points:
            ax.scatter(x, y, c=t, cmap="coolwarm", norm=norm, s=10, alpha=0.5, zorder=2)

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Time")

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title("Trajectories Colored by Time (Blue=Early, Red=Late)", fontsize=14)
    ax.autoscale()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax


def plot_trajectories_by_distance_to_end(
    df,
    umap_col=("umap_1", "umap_2"),
    track_id_col="track_id",
    time_col="t",
    figsize=(12, 10),
    max_tracks=None,
    show_points=False,
    ax=None,
):
    """Plot trajectories coloured by distance to termination."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    unique_tracks = df[track_id_col].unique()
    if max_tracks is not None:
        unique_tracks = unique_tracks[:max_tracks]

    max_steps = max(
        (len(df[df[track_id_col] == tid]) for tid in unique_tracks), default=1
    )
    norm = plt.Normalize(0, max_steps)

    for track_id in unique_tracks:
        track_data = df[df[track_id_col] == track_id].sort_values(time_col)
        x, y = _extract_coords(track_data, umap_col)
        if len(x) < 2:
            continue
        steps_to_term = np.arange(len(x), 0, -1)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        segment_steps = (steps_to_term[:-1] + steps_to_term[1:]) / 2
        lc = LineCollection(segments, cmap="coolwarm_r", norm=norm, alpha=0.7, linewidth=2)
        lc.set_array(segment_steps)
        ax.add_collection(lc)
        if show_points:
            ax.scatter(x, y, c=steps_to_term, cmap="coolwarm_r", norm=norm, s=10, alpha=0.5, zorder=2)

    sm = plt.cm.ScalarMappable(cmap="coolwarm_r", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Steps to Termination")

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title("Trajectories Colored by Distance to Termination (Blue=Far, Red=Near)", fontsize=14)
    ax.autoscale()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax


def plot_trajectories_normalized_time(
    df,
    umap_col=("umap_1", "umap_2"),
    track_id_col="track_id",
    time_col="t",
    figsize=(12, 10),
    max_tracks=None,
    show_points=False,
    ax=None,
):
    """Plot trajectories coloured by normalised time (0=start, 1=end)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    unique_tracks = df[track_id_col].unique()
    if max_tracks is not None:
        unique_tracks = unique_tracks[:max_tracks]

    norm = plt.Normalize(0, 1)

    for track_id in unique_tracks:
        track_data = df[df[track_id_col] == track_id].sort_values(time_col)
        x, y = _extract_coords(track_data, umap_col)
        if len(x) < 2:
            continue
        normalized_time = np.linspace(0, 1, len(x))
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        segment_times = (normalized_time[:-1] + normalized_time[1:]) / 2
        lc = LineCollection(segments, cmap="viridis", norm=norm, alpha=0.7, linewidth=2)
        lc.set_array(segment_times)
        ax.add_collection(lc)
        if show_points:
            ax.scatter(x, y, c=normalized_time, cmap="viridis", norm=norm, s=10, alpha=0.5, zorder=2)

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Normalized Track Progress")

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title("Trajectories Colored by Normalized Time (Purple=Start, Yellow=End)", fontsize=14)
    ax.autoscale()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax


def plot_all_temporal_views(
    df,
    umap_col=("umap_1", "umap_2"),
    track_id_col="track_id",
    time_col="t",
    max_tracks=50,
    figsize=(18, 5),
    axes=None,
):
    """Create a 3-panel figure showing all temporal colouring schemes.

    Parameters
    ----------
    axes : array-like of 3 Axes, optional
        Pre-allocated axes.  A new figure is created when ``None``.
    """
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)

    unique_tracks = df[track_id_col].unique()
    if max_tracks is not None:
        unique_tracks = unique_tracks[:max_tracks]

    # --- Panel 1: Absolute time ---
    t_min, t_max = df[time_col].min(), df[time_col].max()
    norm_time = plt.Normalize(t_min, t_max)

    for track_id in unique_tracks:
        td = df[df[track_id_col] == track_id].sort_values(time_col)
        x, y = _extract_coords(td, umap_col)
        t = td[time_col].values
        if len(x) < 2:
            continue
        pts = np.array([x, y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="coolwarm", norm=norm_time, alpha=0.3, linewidth=1.5)
        lc.set_array((t[:-1] + t[1:]) / 2)
        axes[0].add_collection(lc)

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm_time)
    sm.set_array([])
    plt.colorbar(sm, ax=axes[0], label="Time")
    axes[0].set_title("Absolute Time")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].autoscale()
    axes[0].grid(True, alpha=0.3)

    # --- Panel 2: Distance to termination ---
    max_steps = max(
        (len(df[df[track_id_col] == tid]) for tid in unique_tracks), default=1
    )
    norm_steps = plt.Normalize(0, max_steps)

    for track_id in unique_tracks:
        td = df[df[track_id_col] == track_id].sort_values(time_col)
        x, y = _extract_coords(td, umap_col)
        if len(x) < 2:
            continue
        steps = np.arange(len(x), 0, -1)
        pts = np.array([x, y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="coolwarm_r", norm=norm_steps, alpha=0.7, linewidth=1.5)
        lc.set_array((steps[:-1] + steps[1:]) / 2)
        axes[1].add_collection(lc)

    sm = plt.cm.ScalarMappable(cmap="coolwarm_r", norm=norm_steps)
    sm.set_array([])
    plt.colorbar(sm, ax=axes[1], label="Steps to End")
    axes[1].set_title("Distance to Termination")
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    axes[1].autoscale()
    axes[1].grid(True, alpha=0.3)

    # --- Panel 3: Normalised time ---
    norm_n = plt.Normalize(0, 1)

    for track_id in unique_tracks:
        td = df[df[track_id_col] == track_id].sort_values(time_col)
        x, y = _extract_coords(td, umap_col)
        if len(x) < 2:
            continue
        nt = np.linspace(0, 1, len(x))
        pts = np.array([x, y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", norm=norm_n, alpha=0.7, linewidth=1.5)
        lc.set_array((nt[:-1] + nt[1:]) / 2)
        axes[2].add_collection(lc)

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm_n)
    sm.set_array([])
    plt.colorbar(sm, ax=axes[2], label="Progress")
    axes[2].set_title("Normalized Track Progress")
    axes[2].set_xlabel("UMAP 1")
    axes[2].set_ylabel("UMAP 2")
    axes[2].autoscale()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return axes
