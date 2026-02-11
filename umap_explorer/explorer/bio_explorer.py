"""BioExplorer — interactive ipywidgets/plotly dashboard for UMAP exploration.

This is the UI orchestrator.  All heavy logic is delegated to the extracted
library modules (``embedding``, ``classification``, ``visualization``).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as widgets
from ipywidgets import (
    Layout, VBox, HBox, Button, IntSlider, FloatSlider,
    SelectMultiple, Checkbox, Dropdown, Text, HTML, Output,
)
from IPython.display import display
from sklearn.neighbors import NearestNeighbors

from ..io import load_tiff_image, prepare_explorer_dataframe
from ..embedding import compute_embedding
from ..classification import PhenotypeClassifier
from ..visualization.colors import PersistentColorMap
from ..visualization.density import add_global_density_contour, add_per_phenotype_density
from .state import ExplorerState


class BioExplorer:
    """Interactive UMAP dashboard powered by ipywidgets and Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.  Must contain ``embedding``, ``umap_1``, ``umap_2``
        columns.  Will be normalised via :func:`prepare_explorer_dataframe`.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        df = prepare_explorer_dataframe(df)
        feats = np.array(df["embedding"].tolist())

        self.state = ExplorerState(df=df, feats=feats)
        self.color_map = PersistentColorMap()
        self.phenotype_classifier = PhenotypeClassifier()

        # Convenience aliases
        self.df = self.state.df
        self.feats = self.state.feats

        # Train initial classifier on original phenotypes
        self.phenotype_classifier.train(self.df, self.feats, label_col="phenotype")

        # KNN model for neighbour queries
        self.nbrs: NearestNeighbors | None = None

        # --- Plot ---
        self.fig = go.FigureWidget()
        self.fig.update_layout(
            dragmode="zoom", height=700, width=900,
            title="scDINO UMAP - Interactive Explorer",
        )
        with self.fig.batch_update():
            self.fig.add_trace(go.Scattergl(x=[], y=[], mode="markers", name="Points"))
            self.fig.data[0].on_click(self.on_click_point)

        # --- Widgets & callbacks ---
        self._create_widgets()
        self._attach_callbacks()

        # --- Initial draw ---
        self._update_neighbors_model()
        self.update_plot()
        self._build_layout()
        display(self.ui)

    # ------------------------------------------------------------------ #
    #  Widgets
    # ------------------------------------------------------------------ #

    def _create_widgets(self) -> None:
        s = self.state
        # Filters
        self.phenotype_filter = SelectMultiple(
            options=sorted(self.df["phenotype"].unique()),
            value=[], description="Phenotypes:",
            layout=Layout(width="200px", height="100px"),
        )
        # Track management
        self.track_id_dropdown = Dropdown(
            options=sorted(s.all_track_ids, key=lambda x: int(x) if str(x).isdigit() else x),
            description="Add Track:",
        )
        self.track_input = Text(value="", placeholder="e.g., 1,5,12", description="Track IDs:", layout=Layout(width="200px"))
        self.trajectory_color_mode = Dropdown(
            options=["Track", "Time", "Phenotype", "Apo Status"],
            value="Track", description="Color by:", layout=Layout(width="180px"),
        )
        self.track_add_button = Button(description="Add", button_style="success")
        self.track_clear_button = Button(description="Clear", button_style="danger")
        self.track_list_display = HTML(value="Selected: None")
        self.track_connect_checkbox = Checkbox(description="Show trajectories")
        self.track_filter_checkbox = Checkbox(value=False, description="Show only selected tracks", indent=False)

        # Visuals
        self.point_opacity_slider = FloatSlider(value=s.point_opacity, min=0.05, max=1.0, step=0.05, description="Opacity:")
        self.point_size_slider = IntSlider(value=6, min=1, max=20, step=1, description="Point size:", continuous_update=False)
        self.show_density_checkbox = Checkbox(value=False, description="Show density contours")
        self.density_per_phenotype = Checkbox(value=False, description="Per-phenotype density")
        self.k_slider = IntSlider(value=s.k_neighbors, min=1, max=20, step=1, description="Neighbors (k):", continuous_update=False)
        self.coords_display = Text(value="", placeholder="Click on a point", disabled=True, description="Coords:")

        # Annotation & ML
        self.annotation_mode_checkbox = Checkbox(description="Annotation mode")
        self.annotation_text_input = Text(value=s.current_annotation_text, placeholder="Label text...")
        self.color_by_dropdown = Dropdown(
            options=[
                ("Original phenotype", "phenotype"),
                ("Manual labels", "label_manual"),
                ("Predicted phenotype", self.phenotype_classifier.predicted_col),
            ],
            value="phenotype", description="Color by:",
        )
        self.clear_annotations_button = Button(description="Clear Labels", button_style="warning")
        self.output_neighbors = Output()
        self.output_selection_stats = Output()
        self.dragmode_dropdown = Dropdown(
            options=["zoom", "pan", "select", "lasso"],
            value="zoom", description="Drag Mode:", layout=Layout(width="150px"),
        )

        # UMAP recompute
        self.umap_n_neighbors = IntSlider(value=50, min=5, max=200, step=5, description="n_neighbors:", continuous_update=False)
        self.umap_min_dist = FloatSlider(value=0.1, min=0.0, max=1.0, step=0.01, description="min_dist:", continuous_update=False)
        self.umap_topometry = Checkbox(value=False, description="Use TopOMetry")
        self.umap_recompute_btn = Button(description="Recompute UMAP", button_style="info", icon="refresh")

        # ML panel
        self.label_source_dropdown = Dropdown(
            options=[("Original phenotype", "phenotype"), ("Manual annotations", "label_manual")],
            value="phenotype", description="Train on:", layout=Layout(width="220px"),
        )
        self.retrain_button = Button(description="Retrain Classifier", button_style="primary", icon="brain", layout=Layout(width="180px"))
        self.classifier_report_button = Button(description="Classifier Report", button_style="info", icon="chart-bar")

    def _attach_callbacks(self) -> None:
        self.k_slider.observe(self._on_k_change, names="value")
        self.phenotype_filter.observe(self._on_phenotype_filter_change, names="value")
        self.point_opacity_slider.observe(self._on_opacity_change, names="value")
        self.point_size_slider.observe(self._on_point_size_change, names="value")
        self.color_by_dropdown.observe(self.update_plot, names="value")
        self.show_density_checkbox.observe(self.update_plot, names="value")
        self.density_per_phenotype.observe(self.update_plot, names="value")
        self.track_connect_checkbox.observe(self.update_plot, names="value")
        self.track_filter_checkbox.observe(self.update_plot, names="value")
        self.trajectory_color_mode.observe(self.update_plot, names="value")
        self.dragmode_dropdown.observe(self._on_dragmode_change, names="value")
        self.umap_recompute_btn.on_click(self._on_recompute_umap)
        self.track_input.continuous_update = False
        self.track_input.observe(self._on_track_input_change, names="value")
        self.track_add_button.on_click(self._on_track_add_button)
        self.track_clear_button.on_click(self._on_track_clear_button)
        self.clear_annotations_button.on_click(self._on_clear_annotations)
        self.classifier_report_button.on_click(self._show_classifier_report)
        self.retrain_button.on_click(self._on_retrain_classifier)

    def _build_layout(self) -> None:
        top_row = HBox([self.coords_display, self.k_slider, self.dragmode_dropdown])
        track_panel = VBox([
            HTML(value="<b>Track Selection:</b>"),
            HBox([self.track_id_dropdown, self.track_add_button, self.track_input, self.track_clear_button]),
            self.track_list_display,
            HBox([self.track_filter_checkbox, self.track_connect_checkbox]),
            HBox([HTML(value="<b>Trajectory coloring:</b>", layout=Layout(width="120px")), self.trajectory_color_mode]),
        ], layout=Layout(border="1px solid #ddd", padding="10px", margin="5px"))
        controls_panel = HBox([
            VBox([HTML(value="<b>Filters:</b>"), self.phenotype_filter]),
            VBox([HTML(value="<b>Visuals:</b>"), self.point_opacity_slider, self.point_size_slider, self.show_density_checkbox, self.density_per_phenotype]),
            VBox([HTML(value="<b>Annotation/Color:</b>"), self.annotation_mode_checkbox, self.annotation_text_input, self.color_by_dropdown, self.clear_annotations_button]),
        ], layout=Layout(spacing="20px"))
        umap_panel = VBox([
            HTML("<b>UMAP Parameters</b>"),
            HBox([self.umap_n_neighbors, self.umap_min_dist]),
            HBox([self.umap_topometry, self.umap_recompute_btn]),
        ], layout=Layout(border="1px solid #ccc", padding="10px", margin="10px 0"))
        ml_panel = VBox([
            HTML("<b>Machine Learning Panel</b>"),
            self.label_source_dropdown,
            HBox([self.retrain_button, self.classifier_report_button]),
        ], layout=Layout(border="1px solid #ccc", padding="12px", margin="10px 0"))
        outputs_row = HBox([
            VBox([HTML(value="<b>Neighbors:</b>"), self.output_neighbors]),
            VBox([HTML(value="<b>Stats:</b>"), self.output_selection_stats]),
        ])
        self.ui = VBox([top_row, track_panel, controls_panel, umap_panel, ml_panel, self.fig, outputs_row])

    # ------------------------------------------------------------------ #
    #  Core logic
    # ------------------------------------------------------------------ #

    def _update_neighbors_model(self) -> None:
        s = self.state
        if len(s.df_filtered) > s.k_neighbors:
            X = s.df_filtered[["umap_1", "umap_2"]].values
            self.nbrs = NearestNeighbors(n_neighbors=s.k_neighbors, algorithm="auto").fit(X)
        else:
            self.nbrs = None

    def update_plot(self, _=None) -> None:
        s = self.state
        data = self.df.copy()

        if self.phenotype_filter.value:
            data = data[data["phenotype"].isin(self.phenotype_filter.value)]
        if self.track_filter_checkbox.value and s.selected_tracks:
            data = data[data["track_id"].isin(s.selected_tracks)]

        s.df_filtered = data
        self._update_neighbors_model()

        color_col = self.color_by_dropdown.value
        scatter_data = s.df_filtered

        with self.fig.batch_update():
            self.fig.data = []

            if not scatter_data.empty:
                color_values, color_mapping, unique_labels = self.color_map(scatter_data[color_col])

                self.fig.add_trace(go.Scattergl(
                    x=scatter_data["umap_1"], y=scatter_data["umap_2"],
                    mode="markers", name="Points", showlegend=False,
                    marker=dict(size=self.point_size_slider.value, opacity=self.point_opacity_slider.value, color=color_values),
                    customdata=np.column_stack((
                        scatter_data["track_id"],
                        scatter_data["original_track_id"],
                        scatter_data["filename"],
                        scatter_data["t"] if "t" in scatter_data.columns else np.full(len(scatter_data), ""),
                        scatter_data[color_col],
                    )),
                    hovertemplate=(
                        "<b>Global Track ID:</b> %{customdata[0]}<br>"
                        "<b>Original Track ID:</b> %{customdata[1]}<br>"
                        "<b>File:</b> %{customdata[2]}<br>"
                        "<b>UMAP:</b> (%{x:.3f}, %{y:.3f})<br>"
                        + ("<b>Time:</b> %{customdata[3]}<br>" if "t" in scatter_data.columns else "")
                        + f"<b>{color_col}:</b> %{{customdata[4]}}<extra></extra>"
                    ),
                    text=scatter_data["track_id"].astype(str)
                    + (" / t=" + scatter_data["t"].astype(str) if "t" in scatter_data.columns else ""),
                ))

                # Legend traces
                legend_labels: list[str] = []
                if color_col in ("phenotype", "label_manual"):
                    legend_labels = sorted(set(self.df[color_col].dropna().astype(str)))
                legend_labels = [l for l in legend_labels if l != "unlabeled"]
                for label in legend_labels:
                    self.fig.add_trace(go.Scattergl(
                        x=[None], y=[None], mode="markers",
                        marker=dict(size=10, color=color_mapping.get(label, "gray")),
                        name=label, showlegend=True,
                    ))

                # Density contours
                if self.show_density_checkbox.value and not scatter_data.empty:
                    x_min, x_max = scatter_data["umap_1"].min(), scatter_data["umap_1"].max()
                    y_min, y_max = scatter_data["umap_2"].min(), scatter_data["umap_2"].max()
                    x_range = [x_min - 0.5, x_max + 0.5]
                    y_range = [y_min - 0.5, y_max + 0.5]
                    if self.density_per_phenotype.value:
                        _, pheno_map, _ = self.color_map(scatter_data["phenotype"])
                        add_per_phenotype_density(self.fig, scatter_data, pheno_map, x_range, y_range)
                    else:
                        add_global_density_contour(self.fig, scatter_data, x_range, y_range)

                # Trajectories
                self._add_selected_tracks_overlay()

        # Re-attach handlers
        if self.fig.data and self.fig.data[0].name == "Points":
            self.fig.data[0].on_click(self.on_click_point)
            self.fig.data[0].on_selection(self.on_lasso_select)

        self._update_track_list_display()

    # ------------------------------------------------------------------ #
    #  Track overlay
    # ------------------------------------------------------------------ #

    def _add_selected_tracks_overlay(self) -> None:
        s = self.state
        if not s.selected_tracks:
            return

        mode = self.trajectory_color_mode.value
        _, pheno_map, _ = self.color_map(self.df["phenotype"])

        for track_id in s.selected_tracks:
            track_data = s.df_filtered[s.df_filtered["track_id"] == track_id]
            if len(track_data) == 0:
                continue
            track_data = track_data.sort_values("t")

            if mode == "Track":
                color = px.colors.qualitative.Plotly[s.selected_tracks.index(track_id) % 10]
                marker_colors: str | list = color
                line_color = color
            elif mode == "Time":
                t_norm = (track_data["t"] - track_data["t"].min()) / (track_data["t"].max() - track_data["t"].min() + 1e-8)
                marker_colors = px.colors.sample_colorscale("viridis", t_norm.tolist())
                line_color = "gray"
            elif mode == "Phenotype":
                marker_colors = [pheno_map.get(str(p), "gray") for p in track_data["phenotype"]]
                line_color = "gray"
            elif mode == "Apo Status":
                label = track_data["label_manual"].iloc[0]
                has_apo = pd.notna(label) and "apo" in str(label).lower()
                color = "red" if has_apo else "cyan"
                marker_colors = color
                line_color = color
            else:
                marker_colors = "gray"
                line_color = "gray"

            self.fig.add_trace(go.Scattergl(
                x=track_data["umap_1"], y=track_data["umap_2"],
                mode="markers",
                marker=dict(size=7, color=marker_colors, line=dict(width=2, color="white")),
                showlegend=False, hoverinfo="none",
            ))

            if self.track_connect_checkbox.value and len(track_data) >= 2:
                self.fig.add_trace(go.Scattergl(
                    x=track_data["umap_1"], y=track_data["umap_2"],
                    mode="lines+markers",
                    line=dict(color=line_color, width=1.8),
                    marker=dict(size=7, color=marker_colors, line=dict(width=1.2, color="white")),
                    name=f"Track {track_id}",
                    text=track_data["t"].astype(str),
                    hovertemplate=f"<b>Track {track_id}</b><br>t=%{{text}}<extra></extra>",
                ))

    # ------------------------------------------------------------------ #
    #  Callbacks
    # ------------------------------------------------------------------ #

    def _on_k_change(self, change) -> None:
        self.state.k_neighbors = change.new
        self._update_neighbors_model()

    def _on_opacity_change(self, change) -> None:
        self.state.point_opacity = change.new
        if self.fig.data and self.fig.data[0].name == "Points":
            self.fig.data[0].marker.opacity = self.state.point_opacity

    def _on_point_size_change(self, change) -> None:
        new_size = change.new
        if self.fig.data and self.fig.data[0].name == "Points":
            with self.fig.batch_update():
                self.fig.data[0].marker.size = new_size
        for trace in self.fig.data:
            if trace.mode in ("markers", "lines+markers") and trace.name != "Points":
                with self.fig.batch_update():
                    trace.marker.size = max(8, new_size + 2)

    def _on_phenotype_filter_change(self, _) -> None:
        self.update_plot()

    def _on_track_add_button(self, _) -> None:
        s = self.state
        track_id = self.track_id_dropdown.value
        if track_id is not None and track_id not in s.selected_tracks:
            s.selected_tracks.append(track_id)
            self.update_plot()
            self._update_track_list_display()

    def _on_track_clear_button(self, _) -> None:
        self.state.selected_tracks = []
        self.update_plot()

    def _on_track_input_change(self, change) -> None:
        s = self.state
        text = change.new.strip()
        if not text:
            return
        try:
            candidate_ids = [int(x) for x in text.split(",") if x.strip().isdigit()]
        except Exception:
            self.track_list_display.value = "<span style='color:red'>Invalid input</span>"
            return
        track_id_set = set(map(int, s.all_track_ids))
        valid_ids = [tid for tid in candidate_ids if tid in track_id_set]
        if not valid_ids:
            self.track_list_display.value = "<span style='color:orange'>No valid track IDs found</span>"
            return
        new_tracks = [tid for tid in valid_ids if tid not in s.selected_tracks]
        s.selected_tracks.extend(new_tracks)
        self.update_plot()
        self.track_input.value = ""
        self._update_track_list_display()

    def _on_clear_annotations(self, _) -> None:
        self.df["label_manual"] = "unlabeled"
        self.df["label_manual"] = self.df["label_manual"].astype("object")
        if "label_manual" in self.state.df_filtered.columns:
            self.state.df_filtered["label_manual"] = "unlabeled"
        self.update_plot()

    def _on_recompute_umap(self, btn) -> None:
        btn.disabled = True
        btn.description = "Computing..."
        try:
            n_neighbors = self.umap_n_neighbors.value
            min_dist = self.umap_min_dist.value
            topometry = self.umap_topometry.value
            print(f"Computing new UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, topometry={topometry})...")
            embedding = compute_embedding(
                self.feats,
                topometry=topometry,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=42,
            )
            self.df[["umap_1", "umap_2"]] = embedding
            self.state.df_filtered[["umap_1", "umap_2"]] = embedding[
                self.df.index.isin(self.state.df_filtered.index)
            ]
            self._update_neighbors_model()
            self.update_plot()
            print("New UMAP computed and applied!")
        except Exception as e:
            print(f"Error during UMAP computation: {e}")
        finally:
            btn.disabled = False
            btn.description = "Recompute UMAP"

    def _on_dragmode_change(self, change) -> None:
        with self.fig.batch_update():
            self.fig.layout.dragmode = change.new

    def _on_retrain_classifier(self, btn) -> None:
        btn.disabled = True
        btn.description = "Training..."
        try:
            label_col = self.label_source_dropdown.value
            self.phenotype_classifier.train(self.df, self.feats, label_col=label_col)
            self._show_classifier_report()
        finally:
            btn.disabled = False
            btn.description = "Retrain Classifier"

    def _update_track_list_display(self) -> None:
        s = self.state
        if s.selected_tracks:
            info = []
            for gid in s.selected_tracks:
                row = self.df[self.df["track_id"] == gid].iloc[0]
                orig = row["original_track_id"]
                file = row["filename"].split("/")[-1]
                info.append(f"{gid} (orig {orig}, {file})")
            self.track_list_display.value = f"<b>Selected:</b> {', '.join(info)}"
        else:
            self.track_list_display.value = "Selected: None"

    # ------------------------------------------------------------------ #
    #  Click / selection handlers
    # ------------------------------------------------------------------ #

    def on_click_point(self, trace, points, state) -> None:
        if not points.point_inds:
            return
        clicked_idx = points.point_inds[0]
        try:
            original_idx = self.state.df_filtered.index[clicked_idx]
        except IndexError:
            return
        row = self.df.loc[original_idx]
        self.coords_display.value = f"({row['umap_1']:.2f}, {row['umap_2']:.2f})"
        self._show_neighbors(original_idx)

    def on_lasso_select(self, trace, points, selector) -> None:
        if not points.point_inds:
            return
        sel_idx = [self.state.df_filtered.index[i] for i in points.point_inds]
        if self.annotation_mode_checkbox.value:
            label = self.annotation_text_input.value or "new_label"
            self.df.loc[sel_idx, "label_manual"] = label
            self.state.df_filtered.loc[sel_idx, "label_manual"] = label
            self.update_plot()
            self.output_selection_stats.clear_output(wait=True)
            with self.output_selection_stats:
                print(f"Labeled {len(sel_idx)} points as '{label}'")
        else:
            selected = self.state.df_filtered.iloc[points.point_inds]
            self.output_selection_stats.clear_output(wait=True)
            with self.output_selection_stats:
                print(f"Selected {len(selected)} points")
                print(selected["phenotype"].value_counts())

    # ------------------------------------------------------------------ #
    #  Neighbor display
    # ------------------------------------------------------------------ #

    def _show_neighbors(self, original_idx) -> None:
        s = self.state
        self.output_neighbors.clear_output(wait=True)
        self.output_selection_stats.clear_output(wait=True)

        if self.nbrs is None or len(s.df_filtered) < s.k_neighbors:
            with self.output_neighbors:
                print("KNN Model not available or data is too small.")
            return

        point_idx = s.df_filtered.index.get_loc(original_idx)
        X_filt = s.df_filtered[["umap_1", "umap_2"]].values
        distances, indices = self.nbrs.kneighbors(X_filt[point_idx].reshape(1, -1))
        neighbor_data = s.df_filtered.iloc[indices.flatten()].copy()
        neighbor_data["distance"] = distances.flatten()

        with self.output_neighbors:
            print(f"Nearest {s.k_neighbors - 1} Neighbors:")
            print(neighbor_data[["track_id", "t", "phenotype", "label_manual", "distance"]])

        self._show_neighbor_images(neighbor_data)

        with self.output_selection_stats:
            print("Neighbor Phenotype Distribution:")
            print(neighbor_data["phenotype"].value_counts(normalize=True).mul(100).round(1).astype(str) + "%")

    def _show_neighbor_images(self, neighbors_df) -> None:
        import matplotlib.pyplot as plt

        k = len(neighbors_df)
        if k == 0:
            return

        n_channels = 5
        for _, row in neighbors_df.iterrows():
            path = row.get("path") or row.get("frame_path")
            img = load_tiff_image(str(path) if pd.notna(path) else "")
            if img is not None:
                n_channels = img.shape[-1]
                break

        self.output_neighbors.clear_output(wait=True)
        with self.output_neighbors:
            fig, axes = plt.subplots(k, n_channels, figsize=(1.2 * n_channels, 1.0 * k), constrained_layout=True)
            if k == 1:
                axes = axes.reshape(1, -1)
            if n_channels == 1:
                axes = axes.reshape(-1, 1)

            row_highlights = []
            neighbor_indices = []

            for i, (idx, row) in enumerate(neighbors_df.iterrows()):
                neighbor_indices.append(idx)
                path = row.get("path") or row.get("frame_path")
                track_id = row["track_id"]
                t = row.get("t", "?")
                if not self.annotation_mode_checkbox.value:
                    label = row["phenotype"] if pd.notna(row["phenotype"]) else "unknown"
                else:
                    label = row.get("label_manual", "unlabeled")
                dist = row.get("distance", 0.0)

                img = load_tiff_image(str(path) if pd.notna(path) else "")
                if img is None:
                    for j in range(n_channels):
                        axes[i, j].imshow(np.zeros((128, 128)), cmap="gray")
                        axes[i, j].axis("off")
                    axes[i, 0].text(0.5, 0.5, "No Image", transform=axes[i, 0].transAxes, color="red", ha="center", va="center", fontsize=11)
                    row_highlights.append(None)
                    continue

                actual_ch = img.shape[-1]
                row_rect = plt.Rectangle(
                    (-0.35, -0.1), n_channels + 0.4, 1.1,
                    transform=axes[i, 0].transAxes, facecolor="yellow", alpha=0.0, zorder=-1, linewidth=3, edgecolor="none",
                )
                axes[i, 0].add_patch(row_rect)
                row_highlights.append(row_rect)

                for j in range(actual_ch):
                    axes[i, j].imshow(img[..., j], cmap="gray")
                    axes[i, j].axis("off")
                    if i == 0:
                        axes[i, j].set_title(f"Ch {j + 1}", fontsize=10, pad=8)

                info_text = f"Track {track_id}\nt={t}\n{label}\nd={dist:.3f}"
                axes[i, 0].text(
                    -0.28, 0.5, info_text, transform=axes[i, 0].transAxes,
                    fontsize=9.5, ha="right", va="center",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
                )

            # Click handler
            def on_click(event, _ni=neighbor_indices, _rh=row_highlights):
                if event.inaxes is None:
                    return
                clicked_row = None
                for ri in range(k):
                    for ci in range(n_channels):
                        if event.inaxes == axes[ri, ci]:
                            clicked_row = ri
                            break
                    if clicked_row is not None:
                        break
                if clicked_row is None:
                    return
                sel_idx = _ni[clicked_row]
                if self.annotation_mode_checkbox.value:
                    lbl = self.annotation_text_input.value.strip() or "new_label"
                    self.df.loc[sel_idx, "label_manual"] = lbl
                    if sel_idx in self.state.df_filtered.index:
                        self.state.df_filtered.loc[sel_idx, "label_manual"] = lbl
                    with self.output_selection_stats:
                        self.output_selection_stats.clear_output(wait=True)
                        print(f"Annotated Track {self.df.loc[sel_idx, 'track_id']} -> '{lbl}'")
                    self.update_plot()
                else:
                    r = self.df.loc[sel_idx]
                    with self.output_selection_stats:
                        self.output_selection_stats.clear_output(wait=True)
                        print(f"Clicked neighbor: Track {r['track_id']}, t={r.get('t', '?')}, Phenotype: {r['phenotype']}, Label: {r['label_manual']}")
                for ii, rect in enumerate(_rh):
                    if rect is not None:
                        if ii == clicked_row:
                            rect.set_alpha(0.0 if rect.get_alpha() > 0 else 0.2)
                            rect.set_edgecolor("none" if rect.get_alpha() == 0 else "yellow")
                        else:
                            rect.set_alpha(0.0)
                            rect.set_edgecolor("none")
                fig.canvas.draw_idle()
                self._highlight_neighbor_in_umap(_ni[clicked_row])

            fig.canvas.mpl_connect("button_press_event", on_click)
            fig.canvas.draw_idle()
            plt.show()

    def _highlight_neighbor_in_umap(self, df_index) -> None:
        s = self.state
        if s.current_highlighted_neighbor == df_index:
            with self.fig.batch_update():
                self.fig.data = tuple(t for t in self.fig.data if t.name != "Neighbor Highlight")
            s.current_highlighted_neighbor = None
            return

        s.current_highlighted_neighbor = df_index
        row = self.df.loc[df_index]

        with self.fig.batch_update():
            self.fig.data = tuple(t for t in self.fig.data if t.name != "Neighbor Highlight")
            self.fig.add_trace(go.Scattergl(
                x=[row["umap_1"]], y=[row["umap_2"]],
                mode="markers",
                marker=dict(size=16, color="orange", symbol="star-open", line=dict(width=2, color="orange")),
                name="Neighbor Highlight", showlegend=False,
                hovertemplate=f"<b>Selected Neighbor</b><br>Track: {row['track_id']}<br>UMAP: ({row['umap_1']:.3f}, {row['umap_2']:.3f})<extra></extra>",
            ))

    # ------------------------------------------------------------------ #
    #  Classifier report
    # ------------------------------------------------------------------ #

    def _show_classifier_report(self, _=None) -> None:
        report = self.phenotype_classifier.last_report
        with self.output_selection_stats:
            self.output_selection_stats.clear_output(wait=True)
            if report is None:
                print("No classifier has been trained yet.")
            else:
                print(report.summary())
                if report.status == "success":
                    print("\nTraining complete and predictions updated!")
