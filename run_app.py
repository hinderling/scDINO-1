#!/usr/bin/env python
"""Entry point for the Dash UMAP Explorer.

Usage
-----
    python run_app.py --features path/to/features.jsonl [--metadata path/to/metadata.csv]

Or with pre-embedded data (JSONL must already contain umap_1, umap_2):
    python run_app.py --features path/to/features.jsonl
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from umap_explorer.io import load_features, apply_phenotype, prepare_explorer_dataframe
from umap_explorer.embedding import compute_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the UMAP Explorer web app")
    parser.add_argument(
        "--features", required=True,
        help="Path to JSONL features file (must have 'embedding' column)",
    )
    parser.add_argument(
        "--metadata", default=None,
        help="Optional path to metadata CSV with phenotype labels",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8050,
        help="Port to serve on (default: 8050)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run Dash in debug mode",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading features from {args.features}...")
    df, feats = load_features(args.features)
    print(f"  Loaded {len(df):,} rows, {feats.shape[1]} features")

    # Apply phenotype labels if metadata provided
    if args.metadata:
        print(f"Applying phenotype labels from {args.metadata}...")
        df = apply_phenotype(df, args.metadata)

    # Ensure phenotype column exists
    if "phenotype" not in df.columns:
        df["phenotype"] = "unlabeled"

    # Compute UMAP if not already present
    if "umap_1" not in df.columns or "umap_2" not in df.columns:
        print("Computing UMAP embedding...")
        embedding = compute_embedding(feats)
        df["umap_1"] = embedding[:, 0]
        df["umap_2"] = embedding[:, 1]

    # Prepare for explorer
    df = prepare_explorer_dataframe(df)

    print(f"Starting Dash app on http://{args.host}:{args.port}/")

    from umap_explorer.app import create_app
    app = create_app(df, feats)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
