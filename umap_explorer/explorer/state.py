"""Mutable state separated from the BioExplorer UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


@dataclass
class ExplorerState:
    """Pure-data state for the UMAP explorer.

    Separating mutable state from the UI makes it easier to serialise,
    test, and eventually migrate to a different frontend framework.
    """

    df: pd.DataFrame
    feats: np.ndarray
    df_filtered: pd.DataFrame = field(init=False)
    all_track_ids: list = field(init=False)

    selected_tracks: List[int] = field(default_factory=list)
    k_neighbors: int = 5
    point_opacity: float = 0.7
    current_annotation_text: str = "new_label"
    current_highlighted_neighbor: int | None = None

    def __post_init__(self) -> None:
        self.df_filtered = self.df.copy()
        self.all_track_ids = self.df["track_id"].unique().tolist()
