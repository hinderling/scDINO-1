"""Data loading utilities for UMAP exploration.

Handles JSONL feature files, metadata CSVs, TIFF images, and DataFrame sampling.
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
import pandas as pd


def load_features(features_df_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load a JSONL features file and extract the embedding matrix.

    Parameters
    ----------
    features_df_path : str
        Path to a ``.jsonl`` file with columns including ``embedding``.

    Returns
    -------
    df : pd.DataFrame
        The full DataFrame read from the file.
    feats : np.ndarray
        2-D array of shape ``(n_samples, embedding_dim)``.
    """
    df = pd.read_json(features_df_path, orient="records", lines=True)
    feats = np.array(df["embedding"].tolist())
    return df, feats


def apply_phenotype(df: pd.DataFrame, metadata_path: str) -> pd.DataFrame:
    """Merge phenotype labels from a metadata CSV into *df*.

    The metadata CSV must contain columns ``track_id``, ``t_start`` (renamed to
    ``t``), and ``class`` (renamed to ``phenotype``).

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with ``track_id`` and ``t`` columns.
    metadata_path : str
        Path to the metadata CSV.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with an added ``phenotype`` column.
    """
    metadata = pd.read_csv(metadata_path)
    df_meta = metadata.rename(columns={"t_start": "t"})
    df_combined = pd.merge(
        df,
        df_meta[["track_id", "t", "class"]],
        on=["track_id", "t"],
        how="left",
    )
    df_combined = df_combined.rename(columns={"class": "phenotype"})
    return df_combined


def apply_apoptosis_annotation(
    df: pd.DataFrame,
    apoptosis_file: str,
    apoptosis_label: str = "apo",
    normal_label: str = "non-apo",
) -> pd.DataFrame:
    """Annotate apoptosis events in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``track_id`` and ``t`` columns.
    apoptosis_file : str
        CSV with ``matching_track`` and ``correct_t`` columns.
    apoptosis_label, normal_label : str
        Labels assigned to apoptotic / non-apoptotic timepoints.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``phenotype`` column set appropriately.
    """
    df = df.copy()
    df_apo = pd.read_csv(apoptosis_file)
    df_apo = (
        df_apo[["matching_track", "correct_t"]]
        .rename(columns={"matching_track": "track_id", "correct_t": "t_apoptosis"})
        .drop_duplicates(subset=["track_id"])
    )

    df["phenotype"] = normal_label
    df = pd.merge(df, df_apo, on="track_id", how="left")
    df["phenotype"] = np.where(
        df["t"] >= df["t_apoptosis"], apoptosis_label, df["phenotype"]
    )
    return df


def prepare_explorer_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a DataFrame for use with :class:`BioExplorer`.

    Ensures ``filename``, ``original_track_id``, globally-unique ``track_id``,
    and ``label_manual`` columns exist.

    Parameters
    ----------
    df : pd.DataFrame
        Raw feature DataFrame (must have ``embedding`` column).

    Returns
    -------
    pd.DataFrame
        A copy with the required columns added/normalised.
    """
    df = df.copy()

    if "filename" not in df.columns:
        df["filename"] = "unknown_experiment"

    if "original_track_id" not in df.columns:
        if "track_id" in df.columns:
            df = df.rename(columns={"track_id": "original_track_id"})
        else:
            df["original_track_id"] = range(len(df))

    df["track_id"] = df.groupby(["filename", "original_track_id"]).ngroup()

    if "label_manual" not in df.columns:
        df["label_manual"] = "unlabeled"
    else:
        df["label_manual"] = df["label_manual"].astype("object")

    return df


def load_tiff_image(path: str) -> np.ndarray | None:
    """Load a TIFF image and return it in ``(H, W, C)`` layout.

    Returns ``None`` when *path* is missing or unreadable.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        import tifffile

        img = tifffile.imread(path)
        if img.ndim == 3 and img.shape[0] <= 32 and img.shape[0] < img.shape[1]:
            img = img.transpose(1, 2, 0)
        if img.ndim == 2:
            img = img[..., np.newaxis]
        return img
    except Exception:
        return None


def sample_dataframe(
    df: pd.DataFrame,
    mode: Literal["sample_tracks", "sample_per_track"] = "sample_per_track",
    n_tracks: int = 200,
    n_per_track: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sub-sample a track-based DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a ``track_id`` column.
    mode : str
        ``"sample_tracks"`` keeps *n_tracks* whole tracks.
        ``"sample_per_track"`` keeps up to *n_per_track* rows per track.
    n_tracks : int
        Number of tracks to keep (``sample_tracks`` mode).
    n_per_track : int
        Max rows per track (``sample_per_track`` mode).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
    """
    unique_ids = df["track_id"].unique()

    if mode == "sample_tracks":
        n_actual = min(n_tracks, len(unique_ids))
        sampled = pd.Series(unique_ids).sample(n=n_actual, random_state=random_state)
        return df[df["track_id"].isin(sampled)].copy()

    if mode == "sample_per_track":
        indices = (
            df.groupby("track_id", group_keys=False)
            .apply(
                lambda g: g.sample(
                    n=min(n_per_track, len(g)), random_state=random_state
                )
            )
            .index
        )
        return df.loc[indices].reset_index(drop=True)

    raise ValueError(
        f"Invalid mode '{mode}'. Choose 'sample_tracks' or 'sample_per_track'."
    )
