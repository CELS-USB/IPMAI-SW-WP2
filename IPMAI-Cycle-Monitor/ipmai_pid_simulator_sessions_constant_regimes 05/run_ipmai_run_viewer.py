# ============================================================
# IPMAI RUN VIEWER ONLY
# Standalone PySide6 viewer for existing run folders.
# No simulation. No LNU/PCA/LE processing.
# ============================================================

# ============================================================
# USER SETUP
# ============================================================

DEFAULT_DATA_ROOT = "simulated_runs"
MEASURED_FILENAME = "measured.csv"
CONFIG_FILENAME = "config.txt"

TIME_COLUMN_CANDIDATES = ["t", "time", "timestamp", "Time", "Timestamp"]
EXCLUDED_DEFAULT_COLUMNS = []
MAX_INITIAL_SIGNALS = 12
SHOW_GRID = True
PLOT_LINE_WIDTH = 2.5
PLOT_LINE_COLOR = "#0057D9"
WINDOW_TITLE = "IPMAI Single-Run Viewer - one measurement/run"


PLOT_BLUE_COLOR = "blue"

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


def read_measured_csv(run_folder, measured_filename=MEASURED_FILENAME):
    csv_path = Path(run_folder) / measured_filename
    return pd.read_csv(csv_path)


def read_config_text(run_folder, config_filename=CONFIG_FILENAME):
    config_path = Path(run_folder) / config_filename
    if config_path.exists():
        return config_path.read_text(encoding="utf-8", errors="replace")
    return "config.txt not found."


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


class RunViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1500, 950)

        self.data_root = Path(DEFAULT_DATA_ROOT)
        self.run_folders = []
        self.current_df = None
        self.current_time_col = None
        self.current_x = None
        self.signal_checkboxes = []
        self.plots = []
        self.curves = []
        self.initial_ranges = []
        self._syncing = False

        self._build_ui()
        self.reload_runs()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        top_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(top_layout)

        self.root_edit = QtWidgets.QLineEdit(str(self.data_root))
        top_layout.addWidget(QtWidgets.QLabel("Runs root:"))
        top_layout.addWidget(self.root_edit, stretch=1)

        browse_btn = QtWidgets.QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_root)
        top_layout.addWidget(browse_btn)

        reload_btn = QtWidgets.QPushButton("Reload runs")
        reload_btn.clicked.connect(self.reload_runs)
        top_layout.addWidget(reload_btn)

        self.run_combo = QtWidgets.QComboBox()
        self.run_combo.currentIndexChanged.connect(self.load_selected_run)
        top_layout.addWidget(QtWidgets.QLabel("Run:"))
        top_layout.addWidget(self.run_combo, stretch=1)

        top_layout.addWidget(QtWidgets.QLabel("Line width:"))
        self.line_width_spin = QtWidgets.QDoubleSpinBox()
        self.line_width_spin.setRange(0.5, 10.0)
        self.line_width_spin.setSingleStep(0.5)
        self.line_width_spin.setValue(PLOT_LINE_WIDTH)
        self.line_width_spin.valueChanged.connect(self.update_curve_pens)
        top_layout.addWidget(self.line_width_spin)

        self.sync_x_checkbox = QtWidgets.QCheckBox("Sync X zoom")
        self.sync_x_checkbox.setChecked(True)
        self.sync_x_checkbox.stateChanged.connect(self.update_plots)
        top_layout.addWidget(self.sync_x_checkbox)

        reset_btn = QtWidgets.QPushButton("Reset zoom")
        reset_btn.clicked.connect(self.reset_all_zoom)
        top_layout.addWidget(reset_btn)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        splitter.addWidget(left_panel)

        self.info_label = QtWidgets.QLabel("No run loaded.")
        self.info_label.setWordWrap(True)
        left_layout.addWidget(self.info_label)

        left_layout.addWidget(QtWidgets.QLabel("Signals:"))
        self.signal_scroll = QtWidgets.QScrollArea()
        self.signal_scroll.setWidgetResizable(True)
        self.signal_box = QtWidgets.QWidget()
        self.signal_layout = QtWidgets.QVBoxLayout(self.signal_box)
        self.signal_layout.addStretch(1)
        self.signal_scroll.setWidget(self.signal_box)
        left_layout.addWidget(self.signal_scroll, stretch=2)

        self.config_text = QtWidgets.QPlainTextEdit()
        self.config_text.setReadOnly(True)
        left_layout.addWidget(QtWidgets.QLabel("config.txt:"))
        left_layout.addWidget(self.config_text, stretch=1)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([360, 1140])

        self.plot_scroll = QtWidgets.QScrollArea()
        self.plot_scroll.setWidgetResizable(True)
        self.plot_container = QtWidgets.QWidget()
        self.plot_layout = QtWidgets.QVBoxLayout(self.plot_container)
        self.plot_scroll.setWidget(self.plot_container)
        right_layout.addWidget(self.plot_scroll)

    def browse_root(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select runs root", str(self.data_root))
        if folder:
            self.root_edit.setText(folder)
            self.reload_runs()

    def reload_runs(self):
        self.data_root = Path(self.root_edit.text()).expanduser()
        self.run_folders = find_run_folders(self.data_root)
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for p in self.run_folders:
            self.run_combo.addItem(p.name, str(p))
        self.run_combo.blockSignals(False)

        if self.run_folders:
            self.run_combo.setCurrentIndex(0)
            self.load_selected_run()
        else:
            self.current_df = None
            self.info_label.setText(f"No run folders with {MEASURED_FILENAME} found in: {self.data_root}")
            self.config_text.setPlainText("")
            self.clear_signal_checkboxes()
            self.clear_plots()

    def load_selected_run(self):
        idx = self.run_combo.currentIndex()
        if idx < 0:
            return
        run_folder = Path(self.run_combo.itemData(idx))
        try:
            df = read_measured_csv(run_folder)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load error", f"Could not read measured.csv:\n{exc}")
            return

        time_col = choose_time_column(df)
        if time_col is None:
            x = np.arange(len(df), dtype=float)
            x_label = "sample index"
        else:
            x = df[time_col].to_numpy(dtype=float)
            x_label = time_col

        self.current_df = df
        self.current_time_col = time_col
        self.current_x = x

        numeric_cols = numeric_signal_columns(df, time_col)
        self.info_label.setText(
            f"Run: {run_folder.name}\n"
            f"Rows: {len(df)}\n"
            f"Time axis: {x_label}\n"
            f"Numeric signals: {len(numeric_cols)}\n"
            f"Y axes: independent per signal; X zoom can be synchronized."
        )
        self.config_text.setPlainText(read_config_text(run_folder))
        self.populate_signal_checkboxes(numeric_cols)
        self.update_plots()

    def clear_signal_checkboxes(self):
        for cb in self.signal_checkboxes:
            cb.setParent(None)
        self.signal_checkboxes = []

    def populate_signal_checkboxes(self, columns):
        self.clear_signal_checkboxes()
        while self.signal_layout.count() > 0:
            item = self.signal_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        select_all_btn = QtWidgets.QPushButton("Select all")
        select_all_btn.clicked.connect(lambda: self.set_all_signals(True))
        self.signal_layout.addWidget(select_all_btn)

        select_none_btn = QtWidgets.QPushButton("Select none")
        select_none_btn.clicked.connect(lambda: self.set_all_signals(False))
        self.signal_layout.addWidget(select_none_btn)

        for i, col in enumerate(columns):
            cb = QtWidgets.QCheckBox(col)
            cb.setChecked(i < MAX_INITIAL_SIGNALS)
            cb.stateChanged.connect(self.update_plots)
            self.signal_layout.addWidget(cb)
            self.signal_checkboxes.append(cb)

        self.signal_layout.addStretch(1)

    def set_all_signals(self, checked):
        for cb in self.signal_checkboxes:
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        self.update_plots()

    def selected_signal_columns(self):
        return [cb.text() for cb in self.signal_checkboxes if cb.isChecked()]

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
        if self.current_df is None:
            return

        columns = self.selected_signal_columns()
        self.clear_plots()

        if not columns:
            label = QtWidgets.QLabel("No signal selected.")
            self.plot_layout.addWidget(label)
            return

        x = self.current_x
        prev_plot = None
        for col in columns:
            y = self.current_df[col].to_numpy(dtype=float)
            plot = ResettablePlotWidget()
            plot.setMinimumHeight(170)
            plot.setBackground("w")
            plot.showGrid(x=SHOW_GRID, y=SHOW_GRID, alpha=0.35)
            plot.setLabel("left", col)
            plot.setLabel("bottom", self.current_time_col if self.current_time_col is not None else "sample index")
            plot.double_clicked.connect(self.reset_all_zoom)

            if prev_plot is not None and self.sync_x_checkbox.isChecked():
                plot.setXLink(prev_plot)
            # Y axes are intentionally independent in the single-run viewer,
            # because each selected signal can have a different physical unit and scale.

            curve = plot.plot(
                x,
                y,
                pen=pg.mkPen(PLOT_BLUE_COLOR, width=self.line_width_spin.value()),
            )
            self.plot_layout.addWidget(plot)
            self.plots.append(plot)
            self.curves.append(curve)
            prev_plot = plot

        self.plot_layout.addStretch(1)
        QtCore.QTimer.singleShot(0, self.store_and_reset_ranges)

    def update_curve_pens(self):
        width = self.line_width_spin.value()
        for curve in self.curves:
            curve.setPen(pg.mkPen(PLOT_BLUE_COLOR, width=width))

    def store_and_reset_ranges(self):
        self.initial_ranges = []
        for plot in self.plots:
            vb = plot.getViewBox()
            vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
            plot.autoRange()
            view_range = vb.viewRange()
            self.initial_ranges.append((tuple(view_range[0]), tuple(view_range[1])))

    def reset_all_zoom(self):
        if not self.plots:
            return
        if not self.initial_ranges or len(self.initial_ranges) != len(self.plots):
            self.store_and_reset_ranges()
            return
        for plot, ranges in zip(self.plots, self.initial_ranges):
            x_range, y_range = ranges
            plot.setXRange(x_range[0], x_range[1], padding=0.0)
            plot.setYRange(y_range[0], y_range[1], padding=0.05)


# ============================================================
# MAIN
# ============================================================


def main():
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    viewer = RunViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
