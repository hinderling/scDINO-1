"""Visualization utilities for UMAP exploration."""

from .colors import PersistentColorMap, hex_to_rgb
from .density import add_global_density_contour, add_per_phenotype_density
from .trajectories import (
    plot_all_temporal_views,
    plot_movement_intensity,
    plot_track_trajectories,
    plot_trajectories_by_distance_to_end,
    plot_trajectories_by_time,
    plot_trajectories_normalized_time,
    plot_vector_field,
)

__all__ = [
    "PersistentColorMap",
    "hex_to_rgb",
    "add_global_density_contour",
    "add_per_phenotype_density",
    "plot_track_trajectories",
    "plot_movement_intensity",
    "plot_vector_field",
    "plot_trajectories_by_time",
    "plot_trajectories_by_distance_to_end",
    "plot_trajectories_normalized_time",
    "plot_all_temporal_views",
]
