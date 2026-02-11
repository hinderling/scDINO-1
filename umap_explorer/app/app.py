"""Dash app factory and server-side state."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ..classification import PhenotypeClassifier
from ..visualization.colors import PersistentColorMap


@dataclass
class ServerState:
    """Mutable server-side state for the single-user Dash app."""

    df: pd.DataFrame
    feats: np.ndarray
    df_filtered: pd.DataFrame = field(init=False)
    color_map: PersistentColorMap = field(default_factory=PersistentColorMap)
    classifier: PhenotypeClassifier = field(default_factory=PhenotypeClassifier)
    nbrs: NearestNeighbors | None = None
    selected_tracks: List[int] = field(default_factory=list)
    k_neighbors: int = 5
    annotation_mode: bool = False
    annotation_text: str = "new_label"

    def __post_init__(self) -> None:
        self.df_filtered = self.df.copy()

    def update_filter(
        self,
        phenotype_filter: list[str] | None = None,
        track_filter: bool = False,
    ) -> None:
        """Recompute df_filtered from df using current filters."""
        data = self.df
        if phenotype_filter:
            data = data[data["phenotype"].isin(phenotype_filter)]
        if track_filter and self.selected_tracks:
            data = data[data["track_id"].isin(self.selected_tracks)]
        self.df_filtered = data

    def update_neighbors_model(self) -> None:
        """Rebuild the KNN model on filtered UMAP coordinates."""
        if len(self.df_filtered) > self.k_neighbors:
            X = self.df_filtered[["umap_1", "umap_2"]].values
            self.nbrs = NearestNeighbors(
                n_neighbors=min(self.k_neighbors + 1, len(self.df_filtered)),
                algorithm="auto",
            ).fit(X)
        else:
            self.nbrs = None


# Module-level singleton — set by create_app()
state: ServerState | None = None


def create_app(df: pd.DataFrame, feats: np.ndarray) -> "dash.Dash":
    """Create and configure the Dash application.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared explorer DataFrame (must have umap_1, umap_2, track_id, etc.).
    feats : np.ndarray
        Feature matrix aligned with df rows.

    Returns
    -------
    dash.Dash
    """
    import dash

    from .layout import build_layout
    from . import callbacks

    global state
    state = ServerState(df=df, feats=feats)

    # Train initial classifier
    state.classifier.train(df, feats, label_col="phenotype")
    state.update_neighbors_model()

    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")

    app = dash.Dash(
        __name__,
        assets_folder=assets_dir,
        suppress_callback_exceptions=True,
    )
    app.layout = build_layout(state)
    callbacks.register(app)

    return app
