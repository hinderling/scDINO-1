"""Phenotype classifier with track-level train/val splits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


UNLABELED_VALUES = frozenset({"unknown", "unlabeled", "nan", "<NA>", "", "None", "NaN"})


@dataclass
class ClassifierReport:
    """Stores results from a classifier training run."""

    source_column: str
    n_labeled_used: int
    status: str  # "success" or "failed"
    reason: str = ""
    n_train_tracks: int = 0
    n_val_tracks: int = 0
    n_val_cells: int = 0
    n_val_balanced: int = 0
    accuracy_balanced: float = 0.0
    per_class: Dict[str, dict] = field(default_factory=dict)
    class_order: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable report string."""
        if self.status == "failed":
            lines = [
                "Classifier training failed",
                "=" * 50,
                f"Reason: {self.reason}",
                f"Labeled points: {self.n_labeled_used}",
            ]
            return "\n".join(lines)

        lines = [
            "CLASSIFIER REPORT — Track-level split (no leakage)",
            "=" * 62,
            f"Trained on            : {self.source_column}",
            f"Labeled points used   : {self.n_labeled_used:,}",
            f"Train tracks          : {self.n_train_tracks:,}",
            f"Held-out val tracks   : {self.n_val_tracks:,}",
            f"Validation cells      : {self.n_val_cells:,}",
            f"Balanced val samples  : {self.n_val_balanced:,}",
            f"Balanced accuracy     : {self.accuracy_balanced:.1%}",
            "",
            "Per-class performance (balanced validation):",
            "-" * 62,
        ]
        if self.per_class:
            df = pd.DataFrame(self.per_class).T[
                ["precision", "recall", "f1-score", "support"]
            ]
            lines.append(df.round(4).to_string())
        else:
            lines.append("   (no validation samples)")
        lines.append(f"\nClasses: {', '.join(self.class_order)}")
        return "\n".join(lines)


class PhenotypeClassifier:
    """Logistic-regression classifier with track-level data splits.

    Prevents data leakage by splitting on ``track_id`` rather than individual
    timepoints.
    """

    def __init__(self) -> None:
        self.classifier: LogisticRegression | None = None
        self.label_encoder: LabelEncoder | None = None
        self.last_report: ClassifierReport | None = None
        self.predicted_col = "phenotype_predicted"

    def train(
        self,
        df: pd.DataFrame,
        feats: np.ndarray,
        label_col: str = "phenotype",
        split_ratio: float = 0.8,
        random_seed: int = 42,
    ) -> ClassifierReport:
        """Train on labelled data and predict on the full dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain *label_col* and ``track_id``.
        feats : np.ndarray
            Feature matrix aligned with *df* rows.
        label_col : str
            Column to use as ground truth.
        split_ratio : float
            Fraction of tracks used for training (rest is validation).
        random_seed : int
            Random seed.

        Returns
        -------
        ClassifierReport
        """
        raw_labels = df[label_col].astype(str)
        valid_mask = ~raw_labels.isin(UNLABELED_VALUES) & raw_labels.notna()
        n_labeled = int(valid_mask.sum())

        if n_labeled < 10:
            self.classifier = None
            self.label_encoder = None
            df[self.predicted_col] = "untrained"
            self.last_report = ClassifierReport(
                source_column=label_col,
                n_labeled_used=n_labeled,
                status="failed",
                reason="too_few_labeled_points",
            )
            return self.last_report

        df_labeled = df[valid_mask]
        X_labeled = feats[valid_mask]
        y_str = raw_labels[valid_mask]

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_str)
        real_classes = self.label_encoder.classes_

        # --- Track-level stratified split ---
        track_ids = df_labeled["track_id"].values
        temp = pd.DataFrame({"track_id": track_ids, "phenotype": y_str.values})
        phen_order = temp["phenotype"].value_counts().sort_values().index

        track_to_phens = temp.groupby("track_id")["phenotype"].apply(set)
        phen_to_tracks = {
            p: np.array([t for t, s in track_to_phens.items() if p in s])
            for p in phen_order
        }

        rng = np.random.default_rng(random_seed)
        train_set: set = set()
        val_set: set = set()

        for p in phen_order:
            tracks = phen_to_tracks[p]
            rng.shuffle(tracks)
            assigned = train_set | val_set
            unassigned = [t for t in tracks if t not in assigned]
            if not unassigned:
                continue
            split_idx = int(split_ratio * len(unassigned))
            train_set.update(unassigned[:split_idx])
            val_set.update(unassigned[split_idx:])

        train_tracks = np.array(sorted(train_set))
        val_tracks = np.array(sorted(val_set))

        train_mask = np.isin(track_ids, train_tracks)
        val_mask = np.isin(track_ids, val_tracks)

        X_train, y_train = X_labeled[train_mask], y_encoded[train_mask]
        X_val, y_val = X_labeled[val_mask], y_encoded[val_mask]
        y_val_str = y_str.values[val_mask]

        # --- Train ---
        self.classifier = LogisticRegression(
            solver="lbfgs",
            max_iter=3000,
            n_jobs=-1,
            class_weight="balanced",
            random_state=random_seed,
        )
        self.classifier.fit(X_train, y_train)

        # Predict full dataset
        preds_full = self.classifier.predict(feats)
        df[self.predicted_col] = self.label_encoder.inverse_transform(preds_full)

        # --- Balanced validation ---
        n_val_cells = len(y_val_str)
        balanced_accuracy = 0.0
        per_class_metrics: dict = {}
        n_balanced = 0

        if n_val_cells > 0:
            val_series = pd.Series(y_val_str)
            min_size = val_series.value_counts().min()
            rng2 = np.random.default_rng(123)
            balanced_idx = []
            for cls in val_series.unique():
                idx = np.where(y_val_str == cls)[0]
                keep = rng2.choice(idx, size=min(min_size, len(idx)), replace=False)
                balanced_idx.extend(keep)
            balanced_idx = np.array(balanced_idx)
            n_balanced = len(balanced_idx)

            y_val_bal = y_val[balanced_idx]
            y_pred_bal = self.classifier.predict(X_val[balanced_idx])
            y_true_lbl = self.label_encoder.inverse_transform(y_val_bal)
            y_pred_lbl = self.label_encoder.inverse_transform(y_pred_bal)

            balanced_accuracy = accuracy_score(y_true_lbl, y_pred_lbl)
            report = classification_report(y_true_lbl, y_pred_lbl, output_dict=True)
            per_class_metrics = {
                k: v
                for k, v in report.items()
                if isinstance(v, dict) and "f1-score" in v
            }

        self.last_report = ClassifierReport(
            source_column=label_col,
            n_labeled_used=n_labeled,
            status="success",
            n_train_tracks=len(train_tracks),
            n_val_tracks=len(val_tracks),
            n_val_cells=n_val_cells,
            n_val_balanced=n_balanced,
            accuracy_balanced=balanced_accuracy,
            per_class=per_class_metrics,
            class_order=sorted(real_classes),
        )
        return self.last_report

    def predict(self, feats: np.ndarray) -> np.ndarray:
        """Predict phenotype labels for *feats*.

        Returns an array of string labels, or ``"untrained"`` if no classifier
        has been trained yet.
        """
        if self.classifier is None or self.label_encoder is None:
            return np.full(len(feats), "untrained")
        return self.label_encoder.inverse_transform(self.classifier.predict(feats))
