"""
Robotic results viewer.

Purpose
-------
Diagnostic viewer for Robotic cycle-monitor outputs produced by:

    run_2_robotic_lnu_cycle_monitor.py
    run_3_train_mlp_from_robotic_lnu_weights.py

This viewer shows only the useful diagnostic outputs:
    - PCA 2-D visualization of LNU weights,
    - heatmap of LNU weights across files/windows,
    - trajectories of LNU weights,
    - saliency map,
    - MLP training results.

Important behavior
------------------
    - Scalar trajectories are blue.
    - PCA 2-D points are not connected by lines.
    - Left mouse drag uses rectangular zoom.
    - X-axes are not synchronized in this results viewer.
"""

# ============================================================
# USER SETUP
# ============================================================

from pathlib import Path
import json
import sys

RESULTS_ROOT = Path("robotic_lnu_cycle_monitor_results")
MLP_RESULTS_ROOT = Path("robotic_mlp_from_lnu_weights_results")

SHOW_GRID = True
PLOT_LINE_WIDTH = 2.5
WINDOW_TITLE = "Robotic Results Viewer"

BLUE_PEN = "b"
WEIGHT_TRACE_ALPHA = 90

# Weight heatmap scaling.
# Default requested mode:
#     "global_column_zscore"
#
# This z-scores each individual weight trajectory across all files/windows:
#     M[k,i] = (W[k,i] - mean_k(W[:,i])) / std_k(W[:,i]).
#
# This is useful for visualizing the relative activity pattern/code-word of the
# weight vector, because each weight receives its own contrast scale.
#
# Other options:
#     "raw"                  -> W
#     "delta_previous"       -> W[k] - W[k-1]
#     "abs_delta_previous"   -> abs(W[k] - W[k-1])
#     "reference_delta_zscore" -> delta from reference weights scaled by reference deviation
WEIGHT_HEATMAP_MODE = "global_column_zscore"
HEATMAP_ZSCORE = False
HEATMAP_STD_EPS = 1e-12
WEIGHT_HEATMAP_CMAP = "viridis"
SALIENCY_HEATMAP_CMAP = "plasma"
MAX_LEGEND_ITEMS = 12

# Candidate output file names.
WEIGHTS_FILE_CANDIDATES = [
    "weights_by_file.csv",
    "window_final_weights.csv",
    "final_weights.csv",
]

WEIGHT_PCA_FILE_CANDIDATES = [
    "weight_pca_2d.csv",
    "weights_pca_2d.csv",
]

SALIENCY_FILE_CANDIDATES = [
    "saliency_continual.csv",
    "saliency_vs_reference.csv",
    "saliency_map.csv",
]

MLP_PROB_FILE_CANDIDATES = [
    "mlp_probabilities_from_robotic_lnu_weights.csv",
    "mlp_probabilities.csv",
    "predicted_probabilities.csv",
]

MLP_CONFUSION_FILE_CANDIDATES = [
    "confusion_matrix.csv",
    "mlp_confusion_matrix.csv",
]

MLP_SUMMARY_FILE_CANDIDATES = [
    "training_summary.json",
    "mlp_training_summary.json",
]

MLP_REPORT_FILE_CANDIDATES = [
    "classification_report.txt",
    "mlp_classification_report.txt",
]

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets
import os
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
import pyqtgraph as pg


# ============================================================
# HELPERS
# ============================================================

def first_existing(root: Path, candidates):
    for name in candidates:
        path = root / name
        if path.exists():
            return path
    return None


def read_csv_if_exists(path):
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def read_text_if_exists(path):
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(errors="replace")
        except Exception:
            return None


def read_json_if_exists(path):
    text = read_text_if_exists(path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def numeric_columns(df: pd.DataFrame, exclude=None):
    exclude = set(exclude or [])
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def weight_columns(df: pd.DataFrame):
    cols = [c for c in df.columns if str(c).startswith("w_")]

    def key_fun(c):
        try:
            return int(str(c).split("w_", 1)[1])
        except Exception:
            return str(c)

    return sorted(cols, key=key_fun)


def load_weight_matrix(results_root: Path, caller_name: str = "weights"):
    """
    Load the LNU weight matrix exactly once in a consistent way.

    Returns
    -------
    path : Path
        Source CSV path.
    df : pandas.DataFrame
        Full CSV table.
    w_cols : list[str]
        Sorted columns w_0, w_1, ...
    W : ndarray, shape (n_files_or_windows, n_weights)
        Weight matrix. Rows are files/windows, columns are LNU weights.

    This function is used by BOTH the heatmap and the trajectory plot, so the
    number of displayed weights must be identical in both panels.
    """
    path = first_existing(results_root, WEIGHTS_FILE_CANDIDATES)
    df = read_csv_if_exists(path)
    if df is None or df.empty:
        raise ValueError(f"{caller_name}: missing or empty weights file.")
    w_cols = weight_columns(df)
    if not w_cols:
        raise ValueError(f"{caller_name}: no columns named w_* found in {path.name}.")
    W = df[w_cols].to_numpy(dtype=float)
    return path, df, w_cols, W


def probability_columns(df: pd.DataFrame):
    return [
        c for c in df.columns
        if str(c).startswith("P_") and pd.api.types.is_numeric_dtype(df[c])
    ]


def zscore_columns(A: np.ndarray):
    mean = np.nanmean(A, axis=0, keepdims=True)
    std = np.nanstd(A, axis=0, keepdims=True)
    std[std < HEATMAP_STD_EPS] = 1.0
    return (A - mean) / std


def reference_index_from_weights_df(df: pd.DataFrame) -> int:
    """Return the reference row index if metadata exist; otherwise use first row."""
    if "is_reference_file" in df.columns:
        try:
            flags = df["is_reference_file"].astype(bool).to_numpy()
            idx = np.where(flags)[0]
            if len(idx) > 0:
                return int(idx[0])
        except Exception:
            pass
    return 0


def make_weight_heatmap_matrix(W: np.ndarray, df: pd.DataFrame):
    """
    Build the matrix shown in the weight heatmap.

    Important: the recommended mode does NOT compute mean/std from all displayed
    runs. It uses the reference run as the baseline so that the visualization
    remains consistent with the continual-monitoring convention.
    """
    W = np.asarray(W, dtype=float)
    ref_idx = reference_index_from_weights_df(df)
    w_ref = W[ref_idx:ref_idx + 1, :]

    if WEIGHT_HEATMAP_MODE == "raw":
        return W, "raw weights"

    if WEIGHT_HEATMAP_MODE == "delta_previous":
        M = np.zeros_like(W)
        if W.shape[0] > 1:
            M[1:, :] = W[1:, :] - W[:-1, :]
        return M, "weight change vs previous file/window"

    if WEIGHT_HEATMAP_MODE == "abs_delta_previous":
        M = np.zeros_like(W)
        if W.shape[0] > 1:
            M[1:, :] = np.abs(W[1:, :] - W[:-1, :])
        return M, "absolute weight change vs previous file/window"

    if WEIGHT_HEATMAP_MODE == "global_column_zscore":
        return zscore_columns(W), "each weight z-scored individually"

    if WEIGHT_HEATMAP_MODE == "reference_delta_zscore":
        dW_ref = W - w_ref

        # Estimate a reference scale without recomputing a global z-score from
        # all runs. Prefer the first non-reference deviation; otherwise use a
        # robust absolute scale from the reference vector.
        non_ref = [i for i in range(W.shape[0]) if i != ref_idx]
        if non_ref:
            first = non_ref[0]
            scale = np.abs(W[first:first + 1, :] - w_ref)
        else:
            scale = np.abs(w_ref)

        # Avoid division by tiny numbers while preserving sensitivity.
        fallback = np.nanmedian(np.abs(dW_ref), axis=0, keepdims=True)
        fallback[fallback < HEATMAP_STD_EPS] = 1.0
        scale = np.where(scale < HEATMAP_STD_EPS, fallback, scale)
        scale = np.where(scale < HEATMAP_STD_EPS, 1.0, scale)

        return dW_ref / scale, "delta from reference weights scaled by reference deviation"

    raise ValueError(f"Unsupported WEIGHT_HEATMAP_MODE: {WEIGHT_HEATMAP_MODE}")


def dataframe_to_matrix_from_long_saliency(df: pd.DataFrame):
    """
    Convert long-form saliency table to a 2-D matrix.

    Expected possible columns:
        global_step, window_index, file_index, embedding_component, saliency...
    """
    if "embedding_component" not in df.columns:
        return None, None, None

    value_cols = numeric_columns(df, exclude=[
        "global_step",
        "window_index",
        "file_index",
    ])
    if not value_cols:
        return None, None, None

    value_col = value_cols[-1]

    if "global_step" in df.columns:
        step_col = "global_step"
    elif "window_index" in df.columns:
        step_col = "window_index"
    elif "file_index" in df.columns:
        step_col = "file_index"
    else:
        df = df.copy()
        df["_step"] = np.arange(len(df), dtype=int)
        step_col = "_step"

    pivot = df.pivot_table(
        index=step_col,
        columns="embedding_component",
        values=value_col,
        aggfunc="mean",
    )
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    matrix = pivot.to_numpy(dtype=float)
    x_labels = [str(x) for x in pivot.index]
    y_labels = [str(x) for x in pivot.columns]
    return matrix, x_labels, y_labels


class RectZoomPlotWidget(pg.PlotWidget):
    double_clicked = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.getViewBox().setMouseMode(pg.ViewBox.RectMode)

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit()
        event.accept()


class RoboticResultsViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1650, 980)

        self.results_root = Path(RESULTS_ROOT)
        self.mlp_results_root = Path(MLP_RESULTS_ROOT)

        self.plots = []
        self.curves = []

        self._build_ui()
        self.reload_results()

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        row1 = QtWidgets.QHBoxLayout()
        main_layout.addLayout(row1)

        row1.addWidget(QtWidgets.QLabel("Cycle-monitor results folder:"))
        self.root_edit = QtWidgets.QLineEdit(str(self.results_root))
        row1.addWidget(self.root_edit, stretch=1)

        browse_btn = QtWidgets.QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_root)
        row1.addWidget(browse_btn)

        row2 = QtWidgets.QHBoxLayout()
        main_layout.addLayout(row2)

        row2.addWidget(QtWidgets.QLabel("MLP results folder:"))
        self.mlp_root_edit = QtWidgets.QLineEdit(str(self.mlp_results_root))
        row2.addWidget(self.mlp_root_edit, stretch=1)

        browse_mlp_btn = QtWidgets.QPushButton("Browse")
        browse_mlp_btn.clicked.connect(self.browse_mlp_root)
        row2.addWidget(browse_mlp_btn)

        reload_btn = QtWidgets.QPushButton("Reload")
        reload_btn.clicked.connect(self.reload_results)
        row2.addWidget(reload_btn)

        row2.addWidget(QtWidgets.QLabel("Line width:"))
        self.line_width_spin = QtWidgets.QDoubleSpinBox()
        self.line_width_spin.setRange(0.5, 10.0)
        self.line_width_spin.setSingleStep(0.5)
        self.line_width_spin.setValue(PLOT_LINE_WIDTH)
        self.line_width_spin.valueChanged.connect(self.update_plots)
        row2.addWidget(self.line_width_spin)

        reset_btn = QtWidgets.QPushButton("Reset zoom")
        reset_btn.clicked.connect(self.reset_all_zoom)
        row2.addWidget(reset_btn)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        splitter.addWidget(left)

        self.info_label = QtWidgets.QLabel("No results loaded.")
        self.info_label.setWordWrap(True)
        left_layout.addWidget(self.info_label)

        # ----------------------------------------------------
        # Visualization controls (top of left panel)
        # ----------------------------------------------------

        self.show_weight_pca = QtWidgets.QCheckBox("PCA 2-D weights")
        self.show_weight_pca.setChecked(True)
        self.show_weight_pca.stateChanged.connect(self.update_plots)
        left_layout.addWidget(self.show_weight_pca)

        self.show_weight_heatmap = QtWidgets.QCheckBox("Weight heatmap")
        self.show_weight_heatmap.setChecked(True)
        self.show_weight_heatmap.stateChanged.connect(self.update_plots)
        left_layout.addWidget(self.show_weight_heatmap)

        self.show_weight_trajectories = QtWidgets.QCheckBox("Weight trajectories")
        self.show_weight_trajectories.setChecked(False)
        self.show_weight_trajectories.stateChanged.connect(self.update_plots)
        left_layout.addWidget(self.show_weight_trajectories)

        self.show_saliency_map = QtWidgets.QCheckBox("Saliency map")
        self.show_saliency_map.setChecked(True)
        self.show_saliency_map.stateChanged.connect(self.update_plots)
        left_layout.addWidget(self.show_saliency_map)

        self.show_mlp_probs = QtWidgets.QCheckBox("MLP output probabilities")
        self.show_mlp_probs.setChecked(False)
        self.show_mlp_probs.stateChanged.connect(self.update_plots)
        left_layout.addWidget(self.show_mlp_probs)

        self.show_mlp_confusion = QtWidgets.QCheckBox("MLP confusion matrix")
        self.show_mlp_confusion.setChecked(False)
        self.show_mlp_confusion.stateChanged.connect(self.update_plots)
        left_layout.addWidget(self.show_mlp_confusion)

        # ----------------------------------------------------
        # Text panels
        # ----------------------------------------------------

        left_layout.addWidget(QtWidgets.QLabel("Global index mapping:"))
        self.pca_mapping_box = QtWidgets.QPlainTextEdit()
        self.pca_mapping_box.setReadOnly(True)
        self.pca_mapping_box.setMinimumHeight(90)
        self.pca_mapping_box.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.pca_mapping_box.setPlaceholderText(
            "Global file/run index mapping used across PCA, MLP and diagnostics"
        )
        left_layout.addWidget(self.pca_mapping_box, stretch=2)

        left_layout.addWidget(QtWidgets.QLabel("MLP class labels:"))
        self.mlp_class_box = QtWidgets.QPlainTextEdit()
        self.mlp_class_box.setReadOnly(True)
        self.mlp_class_box.setMinimumHeight(70)
        self.mlp_class_box.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )
        self.mlp_class_box.setPlaceholderText("MLP class index mapping")
        left_layout.addWidget(self.mlp_class_box, stretch=1)

        left_layout.addWidget(QtWidgets.QLabel("MLP training summary:"))
        self.mlp_summary_box = QtWidgets.QPlainTextEdit()
        self.mlp_summary_box.setReadOnly(True)
        self.mlp_summary_box.setMinimumHeight(90)
        self.mlp_summary_box.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.mlp_summary_box.setPlaceholderText(
            "MLP training summary and classification report"
        )
        left_layout.addWidget(self.mlp_summary_box, stretch=2)

        left_layout.addStretch(1)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        splitter.addWidget(right)
        splitter.setSizes([360, 1290])

        self.plot_scroll = QtWidgets.QScrollArea()
        self.plot_scroll.setWidgetResizable(True)

        self.plot_container = QtWidgets.QWidget()
        self.plot_container.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.plot_layout = QtWidgets.QVBoxLayout(self.plot_container)
        self.plot_layout.setContentsMargins(2, 2, 2, 2)
        self.plot_layout.setSpacing(6)
        self.plot_scroll.setWidget(self.plot_container)

        right_layout.addWidget(self.plot_scroll)

    def browse_root(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select cycle-monitor results folder",
            str(self.results_root),
        )
        if folder:
            self.root_edit.setText(folder)
            self.reload_results()

    def browse_mlp_root(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select MLP results folder",
            str(self.mlp_results_root),
        )
        if folder:
            self.mlp_root_edit.setText(folder)
            self.reload_results()

    # --------------------------------------------------------
    # General plotting
    # --------------------------------------------------------

    def reload_results(self):
        self.results_root = Path(self.root_edit.text()).expanduser()
        self.mlp_results_root = Path(self.mlp_root_edit.text()).expanduser()

        if hasattr(self, "pca_mapping_box"):
            self.pca_mapping_box.clear()
        if hasattr(self, "mlp_class_box"):
            self.mlp_class_box.clear()
        if hasattr(self, "mlp_summary_box"):
            self.mlp_summary_box.clear()

        cycle_csv = []
        mlp_csv = []
        if self.results_root.exists():
            cycle_csv = sorted(p.name for p in self.results_root.glob("*.csv"))
        if self.mlp_results_root.exists():
            mlp_csv = sorted(p.name for p in self.mlp_results_root.glob("*.csv"))

        weight_info = "weights: not found"
        try:
            _, _, w_cols, W = load_weight_matrix(self.results_root, "Info")
            weight_info = f"weights_by_file: rows={W.shape[0]}, weights={W.shape[1]}"
        except Exception:
            pass

        self.info_label.setText(
            f"Cycle-monitor results:\n{self.results_root}\n"
            f"CSV files: {len(cycle_csv)}\n"
            f"{weight_info}\n\n"
            f"MLP results:\n{self.mlp_results_root}\n"
            f"CSV files: {len(mlp_csv)}"
        )

        self.update_plots()

    def clear_plots(self):
        while self.plot_layout.count():
            item = self.plot_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.plots = []
        self.curves = []

    def add_plot(self, title, x_label="index", y_label="value"):
        plot = RectZoomPlotWidget()
        plot.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        plot.setMinimumHeight(220)
        plot.setBackground("w")
        plot.showGrid(x=SHOW_GRID, y=SHOW_GRID)
        plot.setTitle(title)
        plot.setLabel("bottom", x_label)
        plot.setLabel("left", y_label)
        plot.double_clicked.connect(self.reset_all_zoom)
        self.plot_layout.addWidget(plot, stretch=1)
        self.plots.append(plot)
        return plot

    def add_heatmap(
        self,
        title,
        matrix,
        x_label="index",
        y_label="component",
        cmap_name="viridis",
        y_tick_labels=None,
    ):
        """
        Draw heatmap as explicit rectangular cells.

        Matrix convention is fixed:
            matrix.shape = (n_files_or_windows, n_weights_or_components)

        Display convention is fixed:
            x axis = file/window index
            y axis = weight/component index

        Therefore, if there are 18 files and 3 weights, the heatmap has:
            18 columns and 3 rows.
        """
        matrix = np.asarray(matrix, dtype=float)
        n_x, n_y = matrix.shape

        self._current_heatmap_y_labels = y_tick_labels

        graphics = pg.GraphicsLayoutWidget()
        graphics.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        graphics.setBackground("w")
        graphics.setMinimumHeight(max(220, 38 * max(n_y, 3)))

        plot = graphics.addPlot(title=title)
        plot.setLabel("bottom", x_label)
        plot.setLabel("left", y_label)
        plot.showGrid(x=SHOW_GRID, y=SHOW_GRID)
        plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)

        vmin = float(np.nanmin(matrix))
        vmax = float(np.nanmax(matrix))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0

        try:
            cmap = pg.colormap.get(cmap_name)
        except Exception:
            cmap = pg.colormap.get("viridis")

        # Explicit cell drawing avoids any row/column ambiguity from ImageItem
        # or PColorMeshItem. One matrix column is exactly one visible row.
        for ix in range(n_x):
            for iy in range(n_y):
                val = matrix[ix, iy]
                if not np.isfinite(val):
                    continue
                q = (float(val) - vmin) / (vmax - vmin + 1e-12)
                q = min(1.0, max(0.0, q))
                color = cmap.map(q, mode="qcolor")

                rect = QtWidgets.QGraphicsRectItem(float(ix), float(iy), 1.0, 1.0)
                rect.setPen(QtGui.QPen(QtCore.Qt.NoPen))
                rect.setBrush(QtGui.QBrush(color))
                plot.addItem(rect)

        plot.setXRange(0, n_x, padding=0.0)
        plot.setYRange(0, n_y, padding=0.0)

        # Show exact row labels for weights/components/embedding coordinates.
        if n_y <= 80:
            if hasattr(self, "_current_heatmap_y_labels") and self._current_heatmap_y_labels is not None:
                labels = self._current_heatmap_y_labels
                y_ticks = [(i + 0.5, str(labels[i])) for i in range(min(n_y, len(labels)))]
            else:
                y_ticks = [(i + 0.5, str(i)) for i in range(n_y)]

            plot.getAxis("left").setTicks([y_ticks])

        if n_x <= 80:
            x_ticks = [(i + 0.5, str(i)) for i in range(n_x)]
            plot.getAxis("bottom").setTicks([x_ticks])

        # Add a color bar using a tiny invisible image only for the bar scale.
        try:
            scale_img = pg.ImageItem(np.array([[vmin, vmax]], dtype=float))
            scale_img.setVisible(False)
            plot.addItem(scale_img)
            bar = pg.ColorBarItem(values=(vmin, vmax), colorMap=cmap)
            bar.setImageItem(scale_img, insert_in=plot)
        except Exception:
            pass

        self.plot_layout.addWidget(graphics, stretch=1)
        self.plots.append(plot)
        return plot


    def add_text_box(self, title, text):
        """
        Route diagnostic text to the left panel.

        The graph area is intentionally kept clean: only plots, heatmaps,
        axes, legends and colorbars are displayed there.
        """
        title_l = str(title).lower()

        if "class" in title_l and hasattr(self, "mlp_class_box"):
            self.mlp_class_box.setPlainText(str(text))
            return

        if "mlp" in title_l and hasattr(self, "mlp_summary_box"):
            self.mlp_summary_box.setPlainText(str(text))
            return

        if "mapping" in title_l and hasattr(self, "pca_mapping_box"):
            self.pca_mapping_box.setPlainText(str(text))
            return

        # Fallback: append to the summary box if available.
        if hasattr(self, "mlp_summary_box"):
            current = self.mlp_summary_box.toPlainText()
            prefix = "" if not current else "\n\n"
            self.mlp_summary_box.setPlainText(current + prefix + f"{title}\n{text}")
            return

    def update_plots(self):
        self.clear_plots()
        if not self.results_root.exists():
            self.info_label.setText(f"Cycle-monitor results folder does not exist:\n{self.results_root}")
            return

        width = self.line_width_spin.value()

        if self.show_weight_pca.isChecked():
            self.plot_weight_pca(width)

        if self.show_weight_heatmap.isChecked():
            self.plot_weight_heatmap()

        if self.show_weight_trajectories.isChecked():
            self.plot_weight_trajectories(width)

        if self.show_saliency_map.isChecked():
            self.plot_saliency_map()

        if self.show_mlp_probs.isChecked():
            self.plot_mlp_probabilities(width)

        if self.show_mlp_confusion.isChecked():
            self.plot_mlp_confusion()

        # MLP summary is text in the left panel, not a graph; always update it.
        self.plot_mlp_summary()

        self.plot_container.updateGeometry()
        # No trailing stretch: visible plots share the available height.

    # --------------------------------------------------------
    # Required result panels
    # --------------------------------------------------------

    def plot_weight_pca(self, width):
        path = first_existing(self.results_root, WEIGHT_PCA_FILE_CANDIDATES)
        df = read_csv_if_exists(path)
        if df is None or df.empty:
            self.add_text_box("PCA 2-D weights", "Missing weight PCA file.")
            return

        if {"c1", "c2"}.issubset(df.columns):
            c1 = df["c1"].to_numpy(dtype=float)
            c2 = df["c2"].to_numpy(dtype=float)
        else:
            cols = numeric_columns(df)
            if len(cols) < 2:
                self.add_text_box("PCA 2-D weights", f"Could not identify two numeric PCA columns in {path.name}.")
                return
            c1 = df[cols[0]].to_numpy(dtype=float)
            c2 = df[cols[1]].to_numpy(dtype=float)

        plot = self.add_plot("PCA 2-D visualization of LNU weights", "PC1", "PC2")

        # Important: no connecting line between PCA points.
        curve = plot.plot(
            c1,
            c2,
            pen=None,
            symbol="o",
            symbolSize=8,
            symbolBrush=BLUE_PEN,
            symbolPen=BLUE_PEN,
        )
        self.curves.append(curve)

        # Add numeric labels corresponding to file/run indices.
        for i in range(len(c1)):
            txt = pg.TextItem(str(i), color=(0, 0, 0), anchor=(0.5, -0.2))
            txt.setPos(float(c1[i]), float(c2[i]))
            plot.addItem(txt)

        # Highlight the reference run only.
        ref_idx = 0
        try:
            if "is_reference_file" in df.columns:
                flags = df["is_reference_file"].astype(bool).to_numpy()
                idx = np.where(flags)[0]
                if len(idx) > 0:
                    ref_idx = int(idx[0])
        except Exception:
            pass

        if len(c1) > ref_idx:
            plot.plot(
                [c1[ref_idx]],
                [c2[ref_idx]],
                pen=None,
                symbol="s",
                symbolSize=12,
                symbolBrush="r",
                symbolPen="r",
                name="reference run",
            )

        legend = plot.addLegend()

        # Update mapping list in the left-side panel.
        try:
            mapping_lines = []
            header = [
                "Global index mapping used across:",
                "  - PCA 2-D map",
                "  - MLP outputs",
                "  - confusion matrix",
                "  - LE plots",
                "  - saliency maps",
                "  - weight heatmaps",
                "",
            ]
            mapping_lines.extend(header)

            if "file_name" in df.columns:
                file_names = df["file_name"].astype(str).tolist()
            elif "run_name" in df.columns:
                file_names = df["run_name"].astype(str).tolist()
            else:
                file_names = []

            loads = df["load_g"].tolist() if "load_g" in df.columns else [None] * len(file_names)
            angles = df["angle_deg"].tolist() if "angle_deg" in df.columns else [None] * len(file_names)

            for i, fn in enumerate(file_names):
                lg = loads[i]
                ag = angles[i]

                extra = []
                if lg is not None and str(lg) != "nan":
                    extra.append(f"load={lg}g")
                if ag is not None and str(ag) != "nan":
                    extra.append(f"angle={ag}deg")

                suffix = ""
                if extra:
                    suffix = " | " + ", ".join(extra)

                mapping_lines.append(f"{i:2d} -> {fn}{suffix}")

            self.pca_mapping_box.setPlainText("\n".join(mapping_lines))
        except Exception:
            pass


    def plot_weight_heatmap(self):
        try:
            path, df, w_cols, W = load_weight_matrix(self.results_root, "Weight heatmap")
        except Exception as exc:
            self.add_text_box("Weight heatmap", str(exc))
            return

        M, mode_label = make_weight_heatmap_matrix(W, df)

        title = (
            "Heatmap of LNU weights across files/windows "
            f"[data files={W.shape[0]}, weights={W.shape[1]}] ({mode_label})"
        )
        self.add_heatmap(title, M, "file/window index", "weight index", WEIGHT_HEATMAP_CMAP)

    def plot_weight_trajectories(self, width):
        try:
            path, df, w_cols, W = load_weight_matrix(self.results_root, "Weight trajectories")
        except Exception as exc:
            self.add_text_box("Weight trajectories", str(exc))
            return

        x = np.arange(W.shape[0], dtype=float)
        plot = self.add_plot(
            f"LNU weight trajectories across files/windows [rows={W.shape[0]}, weights={W.shape[1]}]",
            "file/window index",
            "weight value",
        )

        for i, col in enumerate(w_cols):
            pen = pg.mkPen((0, 0, 255, WEIGHT_TRACE_ALPHA), width=max(1.0, 0.55 * width))
            name = col if i < MAX_LEGEND_ITEMS else None
            curve = plot.plot(x, W[:, i], pen=pen, name=name)
            self.curves.append(curve)

        if len(w_cols) <= MAX_LEGEND_ITEMS:
            plot.addLegend()

    def plot_saliency_map(self):
        path = first_existing(self.results_root, SALIENCY_FILE_CANDIDATES)
        df = read_csv_if_exists(path)
        if df is None or df.empty:
            self.add_text_box("Saliency map", "Missing saliency file.")
            return

        matrix = None
        labels = None

        # Prefer embedding labels stored in the LNU dataset. These labels
        # correspond to true saliency variables, while saliency CSV files may
        # also contain numeric metadata columns such as load_g or angle_deg.
        dataset_labels = None
        try:
            npz_path = self.results_root / "lnu_robotic_dataset.npz"
            if npz_path.exists():
                D = np.load(npz_path, allow_pickle=True)
                if "x_labels" in D:
                    dataset_labels = [str(v) for v in D["x_labels"]]
        except Exception:
            dataset_labels = None

        # Long-form saliency table.
        matrix, x_labels, y_labels = dataframe_to_matrix_from_long_saliency(df)
        if matrix is not None:
            labels = y_labels

        # Wide saliency table fallback.
        if matrix is None:
            # First try to use exactly the variables stored in the dataset.
            # This prevents numeric metadata columns from entering the saliency
            # heatmap and corrupting both color levels and y-axis labels.
            value_cols = []
            if dataset_labels:
                value_cols = [
                    col for col in dataset_labels
                    if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
                ]

            # Fallback for older files without an NPZ dataset.
            if not value_cols:
                exclude = [
                    "global_step",
                    "window_index",
                    "file_index",
                    "file",
                    "file_name",
                    "previous_file_name",
                    "run_name",
                    "load_g",
                    "angle_deg",
                    "label",
                    "class",
                    "target",
                ]
                value_cols = numeric_columns(df, exclude=exclude)

            if value_cols:
                matrix = df[value_cols].to_numpy(dtype=float)
                labels = [str(col) for col in value_cols]

        if matrix is None:
            self.add_text_box("Saliency map", f"Could not convert {path.name} to a saliency matrix.")
            return

        # If dataset labels match the final matrix width exactly, use them.
        # Otherwise keep the labels derived from the actual selected columns.
        if dataset_labels and len(dataset_labels) == matrix.shape[1]:
            labels = dataset_labels

        self.add_heatmap(
            "Saliency map",
            matrix,
            "file/window index",
            "input variable / embedding component",
            SALIENCY_HEATMAP_CMAP,
            y_tick_labels=labels,
        )

    def plot_mlp_probabilities(self, width):
        path = first_existing(self.mlp_results_root, MLP_PROB_FILE_CANDIDATES)
        df = read_csv_if_exists(path)
        if df is None or df.empty:
            self.add_text_box("MLP output probabilities", "Missing MLP probabilities file.")
            return

        p_cols = probability_columns(df)
        if not p_cols:
            # fallback: all numeric probability-like columns
            p_cols = numeric_columns(df, exclude=["file_index", "sample_index"])

        if not p_cols:
            self.add_text_box("MLP output probabilities", f"No probability columns found in {path.name}.")
            return

        x = np.arange(len(df), dtype=float)
        plot = self.add_plot("MLP output probabilities from LNU weights", "file/run index", "probability")

        # MLP probabilities need class-specific colors; blue-only would be unreadable.
        for i, col in enumerate(p_cols):
            color = pg.intColor(i, hues=max(3, len(p_cols)))
            curve = plot.plot(
                x,
                df[col].to_numpy(dtype=float),
                pen=pg.mkPen(color, width=width),
                symbol="o",
                symbolSize=5,
                name=col if i < MAX_LEGEND_ITEMS else None,
            )
            self.curves.append(curve)

        if len(p_cols) <= MAX_LEGEND_ITEMS:
            plot.addLegend()

    def plot_mlp_confusion(self):
        path = first_existing(self.mlp_results_root, MLP_CONFUSION_FILE_CANDIDATES)
        df = read_csv_if_exists(path)
        if df is None or df.empty:
            self.add_text_box("MLP confusion matrix", "Missing MLP confusion matrix file.")
            return

        if not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
            labels = df.iloc[:, 0].astype(str).tolist()
            cm = df.iloc[:, 1:].to_numpy(dtype=float)
        else:
            labels = [str(i) for i in range(df.shape[0])]
            cm = df.to_numpy(dtype=float)

        self.add_heatmap("MLP confusion matrix", cm, "predicted class", "true class", "viridis")

        if labels and len(labels) <= 80 and hasattr(self, "mlp_class_box"):
            self.mlp_class_box.setPlainText("\n".join(f"{i}: {lab}" for i, lab in enumerate(labels)))

    def plot_mlp_summary(self):
        """Always show MLP training summary/report in the left panel."""
        summary_path = first_existing(self.mlp_results_root, MLP_SUMMARY_FILE_CANDIDATES)
        report_path = first_existing(self.mlp_results_root, MLP_REPORT_FILE_CANDIDATES)

        summary = read_json_if_exists(summary_path)
        report = read_text_if_exists(report_path)

        parts = []

        if summary is not None:
            preferred_keys = [
                "n_samples",
                "n_features",
                "n_classes",
                "label_mode",
                "label_source",
                "training_accuracy",
                "test_accuracy",
                "hidden_layer_sizes",
                "activation",
                "solver",
                "standardize_weights",
                "exclude_reference_file",
                "note",
            ]
            for key in preferred_keys:
                if key in summary:
                    parts.append(f"{key}: {summary[key]}")

            if "classes" in summary:
                parts.append("\nclasses:")
                for i, cls in enumerate(summary["classes"]):
                    parts.append(f"  {i}: {cls}")

        if report:
            parts.append("\nclassification report:")
            parts.append(report)

        if not parts:
            parts.append("MLP training summary/report files were not found.")

        if hasattr(self, "mlp_summary_box"):
            self.mlp_summary_box.setPlainText("\n".join(parts))
        else:
            self.add_text_box("MLP training summary", "\n".join(parts))

    # --------------------------------------------------------
    # Global actions
    # --------------------------------------------------------

    def reset_all_zoom(self):
        for plot in self.plots:
            try:
                plot.enableAutoRange()
            except Exception:
                pass


def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    win = RoboticResultsViewer()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
