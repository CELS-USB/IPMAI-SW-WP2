# -*- coding: utf-8 -*-
"""
IPMAI variable-across-runs viewer.

Purpose
-------
Interactive pyqtgraph viewer for comparing the same measured variable across
all simulated runs/measurements.

This viewer intentionally keeps Y axes independent. Only the X axis can be
optionally synchronized. Left-mouse dragging is configured as rectangle zoom,
not panning, so it behaves like the requested zoom-by-drag viewers.

Windows usage
-------------
    python run_ipmai_variable_comparison_viewer.py
"""

# ============================================================
# USER SETUP
# ============================================================

from pathlib import Path
import sys

DEFAULT_DATA_ROOT = "simulated_runs"
MEASURED_FILENAME = "measured.csv"
CONFIG_FILENAME = "config.txt"

TIME_COLUMN_CANDIDATES = ["t", "time", "timestamp", "Time", "Timestamp"]
EXCLUDED_DEFAULT_COLUMNS = []

MAX_INITIAL_RUNS = 20
SHOW_GRID = True

DEFAULT_LINE_WIDTH = 2.0
DEFAULT_AXIS_FONT_SIZE = 12
DEFAULT_POINT_SIZE = 4
DEFAULT_SHOW_POINTS = False

WINDOW_TITLE = "IPMAI Variable-Across-Runs Viewer - same variable in all runs"

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
# DATA HELPERS
# ============================================================

def find_run_folders(data_root, measured_filename=MEASURED_FILENAME):
    data_root = Path(data_root)
    if not data_root.exists():
        return []
    return [
        p for p in sorted(data_root.iterdir())
        if p.is_dir() and (p / measured_filename).exists()
    ]


def read_measured_csv(run_folder, measured_filename=MEASURED_FILENAME):
    return pd.read_csv(Path(run_folder) / measured_filename)


def choose_time_column(df):
    for col in TIME_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


def numeric_signal_columns(df, time_col):
    cols = []
    for col in df.columns:
        if col == time_col:
            continue
        if col in EXCLUDED_DEFAULT_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def common_numeric_columns(run_folders):
    common = None
    for run_folder in run_folders:
        try:
            df = read_measured_csv(run_folder)
        except Exception:
            continue
        time_col = choose_time_column(df)
        cols = set(numeric_signal_columns(df, time_col))
        if common is None:
            common = cols
        else:
            common &= cols
    return sorted(common) if common else []


# ============================================================
# PLOT WIDGET
# ============================================================

class ResettablePlotWidget(pg.PlotWidget):
    double_clicked = QtCore.Signal()

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit()
        event.accept()


# ============================================================
# MAIN WINDOW
# ============================================================

class VariableAcrossRunsViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1550, 950)

        self.data_root = Path(DEFAULT_DATA_ROOT)
        self.run_folders = []
        self.run_checkboxes = []
        self.plots = []
        self.curves = []
        self.initial_ranges = []

        self._build_ui()
        self.reload_runs()

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        top = QtWidgets.QHBoxLayout()
        main_layout.addLayout(top)

        top.addWidget(QtWidgets.QLabel("Runs root:"))
        self.root_edit = QtWidgets.QLineEdit(str(self.data_root))
        top.addWidget(self.root_edit, stretch=1)

        browse_btn = QtWidgets.QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_root)
        top.addWidget(browse_btn)

        reload_btn = QtWidgets.QPushButton("Reload")
        reload_btn.clicked.connect(self.reload_runs)
        top.addWidget(reload_btn)

        top.addWidget(QtWidgets.QLabel("Variable:"))
        self.variable_combo = QtWidgets.QComboBox()
        self.variable_combo.currentIndexChanged.connect(self.update_plots)
        top.addWidget(self.variable_combo, stretch=1)

        top.addWidget(QtWidgets.QLabel("Axis font:"))
        self.axis_font_spin = QtWidgets.QSpinBox()
        self.axis_font_spin.setRange(7, 32)
        self.axis_font_spin.setValue(DEFAULT_AXIS_FONT_SIZE)
        self.axis_font_spin.valueChanged.connect(self.update_plot_fonts)
        top.addWidget(self.axis_font_spin)

        top.addWidget(QtWidgets.QLabel("Line width:"))
        self.line_width_spin = QtWidgets.QDoubleSpinBox()
        self.line_width_spin.setRange(0.2, 10.0)
        self.line_width_spin.setSingleStep(0.2)
        self.line_width_spin.setValue(DEFAULT_LINE_WIDTH)
        self.line_width_spin.valueChanged.connect(self.update_curve_styles)
        top.addWidget(self.line_width_spin)

        self.show_points_checkbox = QtWidgets.QCheckBox("Show data points")
        self.show_points_checkbox.setChecked(DEFAULT_SHOW_POINTS)
        self.show_points_checkbox.stateChanged.connect(self.update_plots)
        top.addWidget(self.show_points_checkbox)

        top.addWidget(QtWidgets.QLabel("Point size:"))
        self.point_size_spin = QtWidgets.QSpinBox()
        self.point_size_spin.setRange(1, 30)
        self.point_size_spin.setValue(DEFAULT_POINT_SIZE)
        self.point_size_spin.valueChanged.connect(self.update_plots)
        top.addWidget(self.point_size_spin)

        self.sync_x_checkbox = QtWidgets.QCheckBox("Sync X zoom only")
        self.sync_x_checkbox.setChecked(True)
        self.sync_x_checkbox.stateChanged.connect(self.update_plots)
        top.addWidget(self.sync_x_checkbox)

        reset_btn = QtWidgets.QPushButton("Reset zoom")
        reset_btn.clicked.connect(self.reset_all_zoom)
        top.addWidget(reset_btn)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        # Left panel
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        splitter.addWidget(left)

        self.info_label = QtWidgets.QLabel("No data loaded.")
        self.info_label.setWordWrap(True)
        left_layout.addWidget(self.info_label)

        left_layout.addWidget(QtWidgets.QLabel("Runs to display:"))
        self.run_scroll = QtWidgets.QScrollArea()
        self.run_scroll.setWidgetResizable(True)
        self.run_box = QtWidgets.QWidget()
        self.run_layout = QtWidgets.QVBoxLayout(self.run_box)
        self.run_scroll.setWidget(self.run_box)
        left_layout.addWidget(self.run_scroll, stretch=1)

        btns = QtWidgets.QHBoxLayout()
        left_layout.addLayout(btns)
        select_all_btn = QtWidgets.QPushButton("Select all")
        select_all_btn.clicked.connect(lambda: self.set_all_runs(True))
        btns.addWidget(select_all_btn)
        select_none_btn = QtWidgets.QPushButton("Select none")
        select_none_btn.clicked.connect(lambda: self.set_all_runs(False))
        btns.addWidget(select_none_btn)

        note = QtWidgets.QLabel(
            "Left mouse drag = rectangle zoom. "
            "Y axes are independent for every run; only X synchronization is optional."
        )
        note.setWordWrap(True)
        left_layout.addWidget(note)

        # Right panel
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        splitter.addWidget(right)

        self.plot_scroll = QtWidgets.QScrollArea()
        self.plot_scroll.setWidgetResizable(True)
        self.plot_container = QtWidgets.QWidget()
        self.plot_layout = QtWidgets.QVBoxLayout(self.plot_container)
        self.plot_scroll.setWidget(self.plot_container)
        right_layout.addWidget(self.plot_scroll)

        splitter.setSizes([390, 1160])

    # --------------------------------------------------------
    # Loading
    # --------------------------------------------------------

    def browse_root(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select runs root", str(self.data_root)
        )
        if folder:
            self.root_edit.setText(folder)
            self.reload_runs()

    def reload_runs(self):
        self.data_root = Path(self.root_edit.text()).expanduser()
        self.run_folders = find_run_folders(self.data_root)

        self.populate_run_checkboxes()

        cols = common_numeric_columns(self.run_folders)
        self.variable_combo.blockSignals(True)
        self.variable_combo.clear()
        self.variable_combo.addItems(cols)
        self.variable_combo.blockSignals(False)

        self.info_label.setText(
            f"Runs root: {self.data_root}\n"
            f"Runs found: {len(self.run_folders)}\n"
            f"Common numeric variables: {len(cols)}\n"
            "Y axes: independent per run. X axis can be synchronized."
        )

        self.update_plots()

    def populate_run_checkboxes(self):
        while self.run_layout.count() > 0:
            item = self.run_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        self.run_checkboxes = []

        for i, run_folder in enumerate(self.run_folders):
            cb = QtWidgets.QCheckBox(run_folder.name)
            cb.setChecked(i < MAX_INITIAL_RUNS)
            cb.stateChanged.connect(self.update_plots)
            self.run_layout.addWidget(cb)
            self.run_checkboxes.append(cb)

        self.run_layout.addStretch(1)

    def selected_run_folders(self):
        selected = []
        for cb, run_folder in zip(self.run_checkboxes, self.run_folders):
            if cb.isChecked():
                selected.append(run_folder)
        return selected

    def set_all_runs(self, checked):
        for cb in self.run_checkboxes:
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        self.update_plots()

    # --------------------------------------------------------
    # Plotting
    # --------------------------------------------------------

    def clear_plots(self):
        for plot in self.plots:
            plot.setParent(None)

        self.plots = []
        self.curves = []
        self.initial_ranges = []

        while self.plot_layout.count() > 0:
            item = self.plot_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    def update_plots(self):
        variable = self.variable_combo.currentText()
        if not variable:
            self.clear_plots()
            self.plot_layout.addWidget(QtWidgets.QLabel("No common numeric variable available."))
            return

        runs = self.selected_run_folders()
        self.clear_plots()

        if not runs:
            self.plot_layout.addWidget(QtWidgets.QLabel("No run selected."))
            return

        prev_plot = None
        for run_folder in runs:
            try:
                df = read_measured_csv(run_folder)
            except Exception as exc:
                self.plot_layout.addWidget(QtWidgets.QLabel(f"{run_folder.name}: cannot read CSV: {exc}"))
                continue

            time_col = choose_time_column(df)
            if time_col is None:
                x = np.arange(len(df), dtype=float)
                xlabel = "sample index"
            else:
                x = df[time_col].to_numpy(dtype=float)
                xlabel = time_col

            if variable not in df.columns:
                continue

            y = df[variable].to_numpy(dtype=float)

            plot = ResettablePlotWidget()
            plot.setMinimumHeight(190)
            plot.setBackground("w")
            plot.showGrid(x=SHOW_GRID, y=SHOW_GRID, alpha=0.35)

            # Critical: rectangle zoom by left-button drag.
            # pyqtgraph otherwise defaults to left-drag panning.
            plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)

            plot.setTitle(run_folder.name)
            plot.setLabel("left", variable)
            plot.setLabel("bottom", xlabel)

            plot.double_clicked.connect(self.reset_all_zoom)

            if prev_plot is not None and self.sync_x_checkbox.isChecked():
                plot.setXLink(prev_plot)

            # Do NOT call setYLink. Y axes must remain independent.
            # Do NOT reuse a global y range. Each run keeps its own amplitude scale.

            if self.show_points_checkbox.isChecked():
                symbol = "o"
                symbol_size = int(self.point_size_spin.value())
            else:
                symbol = None
                symbol_size = None

            curve = plot.plot(
                x,
                y,
                pen=pg.mkPen('blue', width=float(self.line_width_spin.value())),
                symbol=symbol,
                symbolSize=symbol_size,
                symbolPen=pg.mkPen('blue') if symbol else None,
                symbolBrush=pg.mkBrush('blue') if symbol else None,
            )

            self.plot_layout.addWidget(plot)
            self.plots.append(plot)
            self.curves.append(curve)
            prev_plot = plot

        self.plot_layout.addStretch(1)
        self.update_plot_fonts()
        QtCore.QTimer.singleShot(0, self.store_initial_ranges)

    def update_curve_styles(self):
        width = float(self.line_width_spin.value())
        for curve in self.curves:
            curve.setPen(pg.mkPen('blue', width=width))

    def update_plot_fonts(self):
        font_size = int(self.axis_font_spin.value())
        font = QtGui.QFont()
        font.setPointSize(font_size)

        title_font = QtGui.QFont()
        title_font.setPointSize(font_size + 1)
        title_font.setBold(True)

        for plot in self.plots:
            for axis_name in ("left", "bottom"):
                axis = plot.getAxis(axis_name)
                axis.setTickFont(font)
                axis.label.setFont(font)

            title_item = plot.getPlotItem().titleLabel
            title_item.item.setFont(title_font)

    def store_initial_ranges(self):
        self.initial_ranges = []
        for plot in self.plots:
            vb = plot.getViewBox()
            vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
            plot.autoRange()
            xr, yr = vb.viewRange()
            self.initial_ranges.append((tuple(xr), tuple(yr)))

    def reset_all_zoom(self):
        if not self.plots:
            return

        if not self.initial_ranges or len(self.initial_ranges) != len(self.plots):
            self.store_initial_ranges()
            return

        for plot, ranges in zip(self.plots, self.initial_ranges):
            xr, yr = ranges
            plot.setXRange(xr[0], xr[1], padding=0.0)
            plot.setYRange(yr[0], yr[1], padding=0.05)


# ============================================================
# MAIN
# ============================================================

def main():
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    viewer = VariableAcrossRunsViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
