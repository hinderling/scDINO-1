"""umap_explorer — reusable library for interactive single-cell UMAP exploration."""

from .embedding import compute_embedding
from .io import (
    apply_apoptosis_annotation,
    apply_phenotype,
    load_features,
    load_tiff_image,
    prepare_explorer_dataframe,
    sample_dataframe,
)
from .grid_search import plot_grid_embeddings, run_umap_grid_search
from .classification import ClassifierReport, PhenotypeClassifier
from .visualization.colors import PersistentColorMap, hex_to_rgb
from .explorer import BioExplorer, ExplorerState

__all__ = [
    # embedding
    "compute_embedding",
    # io
    "load_features",
    "apply_phenotype",
    "apply_apoptosis_annotation",
    "prepare_explorer_dataframe",
    "load_tiff_image",
    "sample_dataframe",
    # grid search
    "run_umap_grid_search",
    "plot_grid_embeddings",
    # classification
    "PhenotypeClassifier",
    "ClassifierReport",
    # visualization
    "PersistentColorMap",
    "hex_to_rgb",
    # explorer
    "BioExplorer",
    "ExplorerState",
]
