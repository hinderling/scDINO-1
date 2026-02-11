"""Full Dash layout: left sidebar (tabs), main UMAP area, right sidebar.

All tab panels are always present in the DOM so that callback inputs
are never missing.  Visibility is toggled via a callback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dash import dcc, html

from . import theme

if TYPE_CHECKING:
    from .app import ServerState


def build_layout(state: ServerState) -> html.Div:
    """Return the complete app layout."""
    all_phenotypes = sorted(state.df["phenotype"].dropna().unique().tolist())
    all_track_ids = sorted(
        state.df["track_id"].unique().tolist(),
        key=lambda x: int(x) if str(x).isdigit() else x,
    )
    track_options = [{"label": str(t), "value": t} for t in all_track_ids]
    n_points = len(state.df)

    return html.Div(
        className="app-container",
        children=[
            # ── Left sidebar ──
            html.Div(
                className="left-sidebar",
                children=[
                    html.Div("scDINO UMAP Explorer", className="sidebar-header"),
                    html.Div(
                        className="sidebar-tabs",
                        children=[
                            dcc.Tabs(
                                id="sidebar-tabs",
                                value="tab-view",
                                className="custom-tabs",
                                children=[
                                    dcc.Tab(label="View", value="tab-view", className="tab"),
                                    dcc.Tab(label="Tracks", value="tab-tracks", className="tab"),
                                    dcc.Tab(label="Labels", value="tab-labels", className="tab"),
                                    dcc.Tab(label="UMAP", value="tab-umap", className="tab"),
                                    dcc.Tab(label="ML", value="tab-ml", className="tab"),
                                ],
                            ),
                            # All panels always in DOM; visibility toggled
                            html.Div(id="panel-view", children=_view_tab(all_phenotypes)),
                            html.Div(id="panel-tracks", children=_tracks_tab(track_options),
                                     style={"display": "none"}),
                            html.Div(id="panel-labels", children=_labels_tab(),
                                     style={"display": "none"}),
                            html.Div(id="panel-umap", children=_umap_tab(),
                                     style={"display": "none"}),
                            html.Div(id="panel-ml", children=_ml_tab(),
                                     style={"display": "none"}),
                        ],
                    ),
                    html.Div(
                        id="status-bar",
                        className="sidebar-status",
                        children=f"{n_points:,} points loaded",
                    ),
                ],
            ),
            # ── Main area ──
            html.Div(
                className="main-area",
                children=[
                    dcc.Graph(
                        id="umap-graph",
                        config={
                            "scrollZoom": True,
                            "displayModeBar": True,
                            "modeBarButtonsToAdd": ["lasso2d", "select2d"],
                        },
                        style={"height": "100vh", "width": "100%"},
                    ),
                ],
            ),
            # ── Right sidebar ──
            html.Div(
                className="right-sidebar",
                children=[
                    html.Div(
                        className="right-sidebar-section",
                        children=[
                            html.H4("Coordinates"),
                            html.Div(
                                id="coords-display",
                                className="coords-display",
                                children="Click a point",
                            ),
                        ],
                    ),
                    html.Div(
                        className="neighbors-list",
                        children=[
                            html.H4(
                                "Nearest Neighbors",
                                style={
                                    "margin": "0 0 6px 0",
                                    "fontSize": "11px",
                                    "fontWeight": "700",
                                    "color": theme.BASE01,
                                    "textTransform": "uppercase",
                                    "letterSpacing": "0.5px",
                                },
                            ),
                            html.Div(id="neighbors-list"),
                        ],
                    ),
                    html.Div(
                        className="stats-panel",
                        children=[
                            html.H4("Selection Stats"),
                            html.Div(id="selection-stats"),
                        ],
                    ),
                ],
            ),
            # ── Hidden stores ──
            dcc.Store(id="umap-store", data=0),
            dcc.Store(id="figure-trigger", data=0),
        ],
    )


# ------------------------------------------------------------------ #
#  Tab panels — always in the DOM
# ------------------------------------------------------------------ #

def _view_tab(phenotypes: list[str]) -> html.Div:
    return html.Div(
        className="tab-content",
        children=[
            html.Label("Point Opacity"),
            html.Div(
                className="ctrl-row",
                children=[
                    dcc.Slider(
                        id="point-opacity",
                        min=0.05, max=1.0, step=0.05, value=0.7,
                        marks=None, tooltip={"always_visible": False},
                    ),
                ],
            ),
            html.Label("Point Size"),
            html.Div(
                className="ctrl-row",
                children=[
                    dcc.Slider(
                        id="point-size",
                        min=1, max=20, step=1, value=4,
                        marks=None, tooltip={"always_visible": False},
                    ),
                ],
            ),
            html.Label("Color By"),
            html.Div(
                className="ctrl-row",
                children=[
                    dcc.Dropdown(
                        id="color-by",
                        options=[
                            {"label": "Original phenotype", "value": "phenotype"},
                            {"label": "Manual labels", "value": "label_manual"},
                            {"label": "Predicted phenotype", "value": "phenotype_predicted"},
                        ],
                        value="phenotype",
                        clearable=False,
                    ),
                ],
            ),
            html.Label("Drag Mode"),
            html.Div(
                className="ctrl-row",
                children=[
                    dcc.RadioItems(
                        id="drag-mode",
                        options=[
                            {"label": "Zoom", "value": "zoom"},
                            {"label": "Pan", "value": "pan"},
                            {"label": "Box", "value": "select"},
                            {"label": "Lasso", "value": "lasso"},
                        ],
                        value="zoom",
                        inline=True,
                        className="flex-row-wrap",
                    ),
                ],
            ),
            html.Label("Density Contours"),
            html.Div(
                className="ctrl-row",
                children=[
                    dcc.Checklist(
                        id="density-toggles",
                        options=[
                            {"label": "Show density", "value": "density"},
                            {"label": "Per-phenotype", "value": "per_phenotype"},
                        ],
                        value=[],
                    ),
                ],
            ),
            html.Label("Filter Phenotypes"),
            html.Div(
                className="ctrl-row phenotype-checklist",
                children=[
                    dcc.Checklist(
                        id="phenotype-filter",
                        options=[{"label": p, "value": p} for p in phenotypes],
                        value=[],
                    ),
                ],
            ),
        ],
    )


def _tracks_tab(track_options: list[dict]) -> html.Div:
    return html.Div(
        className="tab-content",
        children=[
            html.Label("Select Track"),
            html.Div(
                className="ctrl-row flex-row",
                children=[
                    dcc.Dropdown(
                        id="track-dropdown",
                        options=track_options,
                        placeholder="Track ID...",
                        style={"flex": "1"},
                    ),
                    html.Button("Add", id="track-add-btn", className="btn-success"),
                ],
            ),
            html.Label("Comma-Separated IDs"),
            html.Div(
                className="ctrl-row",
                children=[
                    dcc.Input(
                        id="track-text-input",
                        type="text",
                        placeholder="e.g. 1,5,12",
                        debounce=True,
                        style={"width": "100%"},
                    ),
                ],
            ),
            html.Label("Selected Tracks"),
            html.Div(id="track-list", className="ctrl-row",
                      children="None"),
            html.Button("Clear All", id="track-clear-btn", className="btn-danger mt-4",
                        style={"width": "100%"}),
            html.Div(
                className="ctrl-row mt-8",
                children=[
                    dcc.Checklist(
                        id="track-options",
                        options=[
                            {"label": "Show only selected", "value": "filter"},
                            {"label": "Show trajectories", "value": "trajectories"},
                        ],
                        value=[],
                    ),
                ],
            ),
            html.Label("Trajectory Color"),
            html.Div(
                className="ctrl-row",
                children=[
                    dcc.Dropdown(
                        id="trajectory-color-mode",
                        options=[
                            {"label": "Track", "value": "Track"},
                            {"label": "Time", "value": "Time"},
                            {"label": "Phenotype", "value": "Phenotype"},
                            {"label": "Apo Status", "value": "Apo Status"},
                        ],
                        value="Track",
                        clearable=False,
                    ),
                ],
            ),
        ],
    )


def _labels_tab() -> html.Div:
    return html.Div(
        className="tab-content",
        children=[
            html.Label("Annotation Mode"),
            html.Div(
                className="ctrl-row",
                children=[
                    dcc.Checklist(
                        id="annotation-mode",
                        options=[{"label": "Enable annotation", "value": "on"}],
                        value=[],
                    ),
                ],
            ),
            html.Label("Label Text"),
            html.Div(
                className="ctrl-row",
                children=[
                    dcc.Input(
                        id="annotation-text",
                        type="text",
                        value="new_label",
                        debounce=True,
                        style={"width": "100%"},
                    ),
                ],
            ),
            html.Button(
                "Clear All Labels", id="clear-labels-btn",
                className="btn-warning mt-8",
                style={"width": "100%"},
            ),
        ],
    )


def _umap_tab() -> html.Div:
    return html.Div(
        className="tab-content",
        children=[
            html.Label("n_neighbors"),
            html.Div(
                className="ctrl-row",
                children=[
                    dcc.Slider(
                        id="umap-n-neighbors",
                        min=5, max=200, step=5, value=50,
                        marks=None, tooltip={"always_visible": False},
                    ),
                ],
            ),
            html.Label("min_dist"),
            html.Div(
                className="ctrl-row",
                children=[
                    dcc.Slider(
                        id="umap-min-dist",
                        min=0.0, max=1.0, step=0.01, value=0.1,
                        marks=None, tooltip={"always_visible": False},
                    ),
                ],
            ),
            html.Div(
                className="ctrl-row mt-4",
                children=[
                    dcc.Checklist(
                        id="umap-topometry",
                        options=[{"label": "Use TopOMetry", "value": "on"}],
                        value=[],
                    ),
                ],
            ),
            dcc.Loading(
                id="umap-loading",
                type="circle",
                children=[
                    html.Button(
                        "Recompute UMAP", id="umap-recompute-btn",
                        className="btn-primary mt-8",
                        style={"width": "100%"},
                    ),
                ],
            ),
        ],
    )


def _ml_tab() -> html.Div:
    return html.Div(
        className="tab-content",
        children=[
            html.Label("Train On"),
            html.Div(
                className="ctrl-row",
                children=[
                    dcc.Dropdown(
                        id="ml-train-source",
                        options=[
                            {"label": "Original phenotype", "value": "phenotype"},
                            {"label": "Manual annotations", "value": "label_manual"},
                        ],
                        value="phenotype",
                        clearable=False,
                    ),
                ],
            ),
            html.Button(
                "Retrain Classifier", id="ml-retrain-btn",
                className="btn-primary mt-8",
                style={"width": "100%"},
            ),
            html.Button(
                "Show Report", id="ml-report-btn",
                className="btn-info mt-4",
                style={"width": "100%"},
            ),
            html.Div(id="ml-report-output", className="report-output mt-8"),
        ],
    )
