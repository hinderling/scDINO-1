"""Persistent colour mapping for categorical labels."""

from __future__ import annotations

from typing import Sequence, Tuple

import pandas as pd
import plotly.express as px
from matplotlib import colors as mcolors


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert a hex colour string (e.g. ``'#1f77b4'``) to ``(R, G, B)``."""
    try:
        rgb_float = mcolors.to_rgb(hex_color)
        return tuple(int(c * 255) for c in rgb_float)
    except ValueError:
        return (0, 0, 0)


class PersistentColorMap:
    """Assigns stable colours to categorical labels across repeated calls.

    New labels are assigned the next colour in the Plotly qualitative palette.
    ``'unlabeled'`` is always mapped to grey (``#AAAAAA``).
    """

    _PALETTE = px.colors.qualitative.Plotly + px.colors.qualitative.Safe

    def __init__(self) -> None:
        self._map: dict[str, str] = {"unlabeled": "#AAAAAA"}

    @property
    def mapping(self) -> dict[str, str]:
        """Return a copy of the current label -> colour mapping."""
        return dict(self._map)

    def __call__(
        self, series: pd.Series
    ) -> Tuple[Sequence[str], dict[str, str], list[str]]:
        """Map *series* values to colours.

        Returns
        -------
        color_values : list[str]
            One colour per element of *series*.
        mapping : dict[str, str]
            The full label -> hex colour mapping.
        unique_labels : list[str]
            Sorted unique labels.
        """
        series = series.fillna("unlabeled").astype(str)

        for label in series:
            if label not in self._map:
                idx = len(self._map) - 1  # offset for 'unlabeled'
                self._map[label] = self._PALETTE[idx % len(self._PALETTE)]

        color_values = series.map(self._map).tolist()
        return color_values, dict(self._map), sorted(self._map.keys())
