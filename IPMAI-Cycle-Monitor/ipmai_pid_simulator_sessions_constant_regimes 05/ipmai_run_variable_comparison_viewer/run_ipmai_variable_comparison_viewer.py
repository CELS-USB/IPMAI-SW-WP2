# ============================================================
# IPMAI VARIABLE COMPARISON VIEWER ONLY
# Standalone PySide6 viewer for existing run folders.
# It compares one selected measured variable across behavior types.
# No simulation. No LNU/PCA/LE processing.
# ============================================================

# ============================================================
# USER SETUP
# ============================================================

DEFAULT_DATA_ROOT = "simulated_runs"
MEASURED_FILENAME = "measured.csv"
ANNOTATED_FILENAME = "annotated.csv"
EVENTS_FILENAME = "events.csv"
CONFIG_FILENAME = "config.txt"

TIME_COLUMN_CANDIDATES = ["t", "time", "timestamp", "Time", "Timestamp"]
BEHAVIOR_COLUMN_CANDIDATES = ["fault_name", "behavior", "regime", "regime_name", "class_name"]
FAULT_ID_COLUMN_CANDIDATES = ["fault_id", "class_id", "cid"]
EXCLUDED_VARIABLE_COLUMNS = []

PLOT_LINE_WIDTH = 2.5
PLOT_LINE_COLOR = "#0057D9"
SHOW_GRID = True
WINDOW_TITLE = "IPMAI Variable Comparison Viewer - one variable across behaviors"

# ============================================================
# IMPORTS
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from PySide6 import QtCore, QtWidgets
import os
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
import pyqtgraph as pg


# ============================================================
# DATA LOADING
# ============================================================


def find_run_folders(data_root, measured_filename=MEASURED_FILENAME):
    data_root = Path(data_root)
    if not data_root.exists():
        return []
    run_folders = []
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and (p / measured_filename).exists():
            run_folders.append(p)
    return run_folders


def read_csv_if_exists(path):
    path = Path(path)
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


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
        if col in EXCLUDED_VARIABLE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def first_existing_value(df, column_candidates):
    if df is None or df.empty:
        return None
    for col in column_candidates:
        if col in df.columns:
            values = df[col].dropna().unique()
            if len(values) > 0:
                return values[0]
    return None


def infer_behavior_name(run_folder):
    run_folder = Path(run_folder)

    annotated = read_csv_if_exists(run_folder / ANNOTATED_FILENAME)
    behavior = first_existing_value(annotated, BEHAVIOR_COLUMN_CANDIDATES)
    fault_id = first_existing_value(annotated, FAULT_ID_COLUMN_CANDIDATES)

    if behavior is None:
        events = read_csv_if_exists(run_folder / EVENTS_FILENAME)
        behavior = first_existing_value(events, BEHAVIOR_COLUMN_CANDIDATES)
        if fault_id is None:
            fault_id = first_existing_value(events, FAULT_ID_COLUMN_CANDIDATES)

    if behavior is not None:
        if fault_id is not None:
            return f"{behavior} (id={fault_id})"
        return str(behavior)

    # Fallback: derive a readable behavior key from folder name.
    name = run_folder.name
    if name.startswith("sim_"):
        name = name[4:]
    parts = name.split("_")
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return "_".join(parts) if parts else run_folder.name


class RunRecord:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.name = self.folder.name
        self.measured_path = self.folder / MEASURED_FILENAME
        self.df = pd.read_csv(self.measured_path)
        self.time_col = choose_time_column(self.df)
        if self.time_col is None:
            self.x = np.arange(len(self.df), dtype=float)
            self.x_label = "sample index"
        else:
            self.x = self.df[self.time_col].to_numpy(dtype=float)
            self.x_label = self.time_col
        self.variables = numeric_signal_columns(self.df, self.time_col)
        self.behavior_name = infer_behavior_name(self.folder)


# ============================================================
# PLOT WIDGET WITH DOUBLE-CLICK RESET SIGNAL
# ============================================================


class ResettablePlotWidget(pg.PlotWidget):
    double_clicked = QtCore.Signal()

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit()
        event.accept()


# ============================================================
# MAIN WINDOW
# ============================================================


class VariableComparisonViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1550, 950)

        self.data_root = Path(DEFAULT_DATA_ROOT)
        self.run_records = []
        self.groups = {}
        self.plots = []
        self.curves = []
        self.initial_ranges = []

        self._build_ui()
        self.reload_runs()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        top_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(top_layout)

        top_layout.addWidget(QtWidgets.QLabel("Runs root:"))
        self.root_edit = QtWidgets.QLineEdit(str(self.data_root))
        top_layout.addWidget(self.root_edit, stretch=1)

        browse_btn = QtWidgets.QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_root)
        top_layout.addWidget(browse_btn)

        reload_btn = QtWidgets.QPushButton("Reload runs")
        reload_btn.clicked.connect(self.reload_runs)
        top_layout.addWidget(reload_btn)

        top_layout.addWidget(QtWidgets.QLabel("Variable:"))
        self.variable_combo = QtWidgets.QComboBox()
        self.variable_combo.currentIndexChanged.connect(self.update_plots)
        top_layout.addWidget(self.variable_combo, stretch=1)

        top_layout.addWidget(QtWidgets.QLabel("Line width:"))
        self.line_width_spin = QtWidgets.QDoubleSpinBox()
        self.line_width_spin.setRange(0.5, 10.0)
        self.line_width_spin.setSingleStep(0.5)
        self.line_width_spin.setValue(PLOT_LINE_WIDTH)
        self.line_width_spin.valueChanged.connect(self.update_curve_pens)
        top_layout.addWidget(self.line_width_spin)

        self.show_legend_checkbox = QtWidgets.QCheckBox("Show legends")
        self.show_legend_checkbox.setChecked(True)
        self.show_legend_checkbox.stateChanged.connect(self.update_plots)
        top_layout.addWidget(self.show_legend_checkbox)

        reset_btn = QtWidgets.QPushButton("Reset zoom")
        reset_btn.clicked.connect(self.reset_all_zoom)
        top_layout.addWidget(reset_btn)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        splitter.addWidget(left_panel)

        self.info_label = QtWidgets.QLabel("No runs loaded.")
        self.info_label.setWordWrap(True)
        left_layout.addWidget(self.info_label)

        left_layout.addWidget(QtWidgets.QLabel("Behavior groups:"))
        self.group_list = QtWidgets.QListWidget()
        left_layout.addWidget(self.group_list, stretch=1)

        self.notes = QtWidgets.QPlainTextEdit()
        self.notes.setReadOnly(True)
        left_layout.addWidget(QtWidgets.QLabel("Notes:"))
        left_layout.addWidget(self.notes, stretch=1)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([360, 1190])

        self.plot_scroll = QtWidgets.QScrollArea()
        self.plot_scroll.setWidgetResizable(True)
        self.plot_container = QtWidgets.QWidget()
        self.plot_layout = QtWidgets.QGridLayout(self.plot_container)
        self.plot_scroll.setWidget(self.plot_container)
        right_layout.addWidget(self.plot_scroll)

    def browse_root(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select runs root", str(self.data_root))
        if folder:
            self.root_edit.setText(folder)
            self.reload_runs()

    def reload_runs(self):
        self.data_root = Path(self.root_edit.text()).expanduser()
        run_folders = find_run_folders(self.data_root)
        records = []
        load_errors = []
        for folder in run_folders:
            try:
                records.append(RunRecord(folder))
            except Exception as exc:
                load_errors.append(f"{folder.name}: {exc}")

        self.run_records = records
        self.groups = {}
        for rec in records:
            self.groups.setdefault(rec.behavior_name, []).append(rec)

        self.populate_variable_combo()
        self.populate_group_list()
        self.update_info(load_errors)
        self.update_plots()

    def populate_variable_combo(self):
        old_variable = self.variable_combo.currentText()
        common_vars = None
        for rec in self.run_records:
            vars_set = set(rec.variables)
            common_vars = vars_set if common_vars is None else common_vars.intersection(vars_set)
        variables = sorted(common_vars) if common_vars is not None else []

        self.variable_combo.blockSignals(True)
        self.variable_combo.clear()
        self.variable_combo.addItems(variables)
        if old_variable in variables:
            self.variable_combo.setCurrentText(old_variable)
        self.variable_combo.blockSignals(False)

    def populate_group_list(self):
        self.group_list.clear()
        for behavior, records in sorted(self.groups.items(), key=lambda item: item[0]):
            self.group_list.addItem(f"{behavior}: {len(records)} run(s)")

    def update_info(self, load_errors):
        n_runs = len(self.run_records)
        n_groups = len(self.groups)
        variables = [self.variable_combo.itemText(i) for i in range(self.variable_combo.count())]
        text = (
            f"Root: {self.data_root}\n"
            f"Loaded runs: {n_runs}\n"
            f"Behavior groups: {n_groups}\n"
            f"Common numeric variables: {len(variables)}"
        )
        self.info_label.setText(text)

        notes = [
            "This viewer compares one selected variable across behavior groups.",
            "Each subplot corresponds to one behavior group; all runs/trials in that group are overlaid.",
            "Both x and y axes are synchronized because all plots show the same variable.",
            "Double-click any plot or press Reset zoom to restore the full common range.",
        ]
        if load_errors:
            notes.append("")
            notes.append("Load errors:")
            notes.extend(load_errors)
        self.notes.setPlainText("\n".join(notes))

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
        if not hasattr(self, "plot_layout"):
            return
        self.clear_plots()

        variable = self.variable_combo.currentText()
        if not variable or not self.run_records:
            label = QtWidgets.QLabel("No common variable available. Check that run folders contain measured.csv files with shared numeric columns.")
            label.setWordWrap(True)
            self.plot_layout.addWidget(label, 0, 0)
            return

        behavior_items = sorted(self.groups.items(), key=lambda item: item[0])
        n_groups = len(behavior_items)
        n_cols = int(np.ceil(np.sqrt(max(n_groups, 1))))
        n_cols = max(1, min(n_cols, 3))

        first_plot = None
        global_x_min, global_x_max = np.inf, -np.inf
        global_y_min, global_y_max = np.inf, -np.inf

        # Precompute global ranges for robust synchronized reset.
        for _, records in behavior_items:
            for rec in records:
                if variable not in rec.df.columns:
                    continue
                x = rec.x
                y = rec.df[variable].to_numpy(dtype=float)
                finite = np.isfinite(x) & np.isfinite(y)
                if np.any(finite):
                    global_x_min = min(global_x_min, float(np.min(x[finite])))
                    global_x_max = max(global_x_max, float(np.max(x[finite])))
                    global_y_min = min(global_y_min, float(np.min(y[finite])))
                    global_y_max = max(global_y_max, float(np.max(y[finite])))

        if not np.isfinite(global_x_min) or not np.isfinite(global_y_min):
            label = QtWidgets.QLabel(f"Variable '{variable}' has no finite data.")
            self.plot_layout.addWidget(label, 0, 0)
            return

        if global_y_min == global_y_max:
            dy = 1.0 if global_y_min == 0 else abs(global_y_min) * 0.05
            global_y_min -= dy
            global_y_max += dy

        for i, (behavior, records) in enumerate(behavior_items):
            row = i // n_cols
            col = i % n_cols

            plot = ResettablePlotWidget()
            plot.setMinimumHeight(260)
            plot.setBackground("w")
            plot.showGrid(x=SHOW_GRID, y=SHOW_GRID, alpha=0.35)
            plot.setTitle(f"{behavior}   ({len(records)} run(s))")
            plot.setLabel("left", variable)
            plot.setLabel("bottom", records[0].x_label if records else "time")
            plot.double_clicked.connect(self.reset_all_zoom)

            if first_plot is not None:
                plot.setXLink(first_plot)
                plot.setYLink(first_plot)
            else:
                first_plot = plot

            legend = None
            if self.show_legend_checkbox.isChecked():
                legend = plot.addLegend(offset=(10, 10))

            for rec in records:
                if variable not in rec.df.columns:
                    continue
                y = rec.df[variable].to_numpy(dtype=float)
                curve = plot.plot(
                    rec.x,
                    y,
                    pen=pg.mkPen(color=PLOT_LINE_COLOR, width=self.line_width_spin.value()),
                    name=rec.name,
                )
                self.curves.append(curve)

            self.plot_layout.addWidget(plot, row, col)
            self.plots.append(plot)

        QtCore.QTimer.singleShot(0, lambda: self.set_common_ranges(global_x_min, global_x_max, global_y_min, global_y_max))

    def set_common_ranges(self, x_min, x_max, y_min, y_max):
        if not self.plots:
            return
        y_pad = 0.05 * (y_max - y_min)
        x_pad = 0.0
        x_range = (x_min - x_pad, x_max + x_pad)
        y_range = (y_min - y_pad, y_max + y_pad)
        self.initial_ranges = [(x_range, y_range) for _ in self.plots]
        for plot in self.plots:
            plot.setXRange(x_range[0], x_range[1], padding=0.0)
            plot.setYRange(y_range[0], y_range[1], padding=0.0)

    def update_curve_pens(self):
        width = self.line_width_spin.value()
        for curve in self.curves:
            curve.setPen(pg.mkPen(color=PLOT_LINE_COLOR, width=width))

    def reset_all_zoom(self):
        if not self.plots or not self.initial_ranges:
            return
        for plot, ranges in zip(self.plots, self.initial_ranges):
            x_range, y_range = ranges
            plot.setXRange(x_range[0], x_range[1], padding=0.0)
            plot.setYRange(y_range[0], y_range[1], padding=0.0)


# ============================================================
# MAIN
# ============================================================


def main():
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    viewer = VariableComparisonViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
