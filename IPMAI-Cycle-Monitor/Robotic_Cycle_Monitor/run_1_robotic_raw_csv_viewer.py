"""
Robotic raw CSV viewer.

Purpose
-------
Interactive viewer for real robotic CSV files stored directly in DATA_ROOT.
This viewer is for inspecting source signals before running the LNU cycle monitor.
It does not assume simulated run folders and it does not require measured.csv.

Windows usage
-------------
    python run_robotic_raw_csv_viewer.py

Authoring convention: user setups are at the beginning of the script.
"""

# ============================================================
# USER SETUP
# ============================================================

from pathlib import Path
import sys

DATA_ROOT = Path("data_robotic_cycle_monitor")
FILE_GLOB = "*.csv"
CSV_SEPARATOR = r"\s+"
CSV_ENGINE = "python"
TIME_COLUMN_CANDIDATES = ["timestamp", "t", "time", "Time", "Timestamp"]
EXCLUDED_COLUMNS = []
MAX_INITIAL_SIGNALS = 8
SHOW_GRID = True
PLOT_LINE_WIDTH = 2.5
WINDOW_TITLE = "Robotic Raw CSV Viewer"

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
from PySide6 import QtCore, QtWidgets
import os
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
import pyqtgraph as pg


# ============================================================
# DATA HELPERS
# ============================================================


def find_csv_files(data_root: Path):
    data_root = Path(data_root)
    if not data_root.exists():
        return []
    return sorted(data_root.glob(FILE_GLOB), key=lambda p: p.name)


def read_robotic_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=CSV_SEPARATOR, engine=CSV_ENGINE)


def choose_time_column(df: pd.DataFrame):
    for col in TIME_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


def numeric_signal_columns(df: pd.DataFrame, time_col):
    cols = []
    for col in df.columns:
        if col == time_col or col in EXCLUDED_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


class ResettablePlotWidget(pg.PlotWidget):
    double_clicked = QtCore.Signal()

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit()
        event.accept()


class RoboticRawCSVViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1550, 950)
        self.data_root = Path(DATA_ROOT)
        self.files = []
        self.df = None
        self.time_col = None
        self.signal_checkboxes = []
        self.plots = []
        self.curves = []
        self.initial_ranges = []
        self._build_ui()
        self.reload_files()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        top = QtWidgets.QHBoxLayout()
        main_layout.addLayout(top)

        top.addWidget(QtWidgets.QLabel("Data folder:"))
        self.root_edit = QtWidgets.QLineEdit(str(self.data_root))
        top.addWidget(self.root_edit, stretch=1)

        browse_btn = QtWidgets.QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_root)
        top.addWidget(browse_btn)

        reload_btn = QtWidgets.QPushButton("Reload")
        reload_btn.clicked.connect(self.reload_files)
        top.addWidget(reload_btn)

        top.addWidget(QtWidgets.QLabel("File:"))
        self.file_combo = QtWidgets.QComboBox()
        self.file_combo.currentIndexChanged.connect(self.load_selected_file)
        top.addWidget(self.file_combo, stretch=2)

        top.addWidget(QtWidgets.QLabel("Line width:"))
        self.line_width_spin = QtWidgets.QDoubleSpinBox()
        self.line_width_spin.setRange(0.5, 10.0)
        self.line_width_spin.setSingleStep(0.5)
        self.line_width_spin.setValue(PLOT_LINE_WIDTH)
        self.line_width_spin.valueChanged.connect(self.update_curve_pens)
        top.addWidget(self.line_width_spin)

        reset_btn = QtWidgets.QPushButton("Reset zoom")
        reset_btn.clicked.connect(self.reset_all_zoom)
        top.addWidget(reset_btn)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        splitter.addWidget(left)

        self.info_label = QtWidgets.QLabel("No file loaded.")
        self.info_label.setWordWrap(True)
        left_layout.addWidget(self.info_label)

        left_layout.addWidget(QtWidgets.QLabel("Signals:"))
        self.signal_scroll = QtWidgets.QScrollArea()
        self.signal_scroll.setWidgetResizable(True)
        self.signal_box = QtWidgets.QWidget()
        self.signal_layout = QtWidgets.QVBoxLayout(self.signal_box)
        self.signal_layout.addStretch(1)
        self.signal_scroll.setWidget(self.signal_box)
        left_layout.addWidget(self.signal_scroll, stretch=10)

        select_all_btn = QtWidgets.QPushButton("Select all")
        select_all_btn.clicked.connect(lambda: self.set_all_signals(True))
        left_layout.addWidget(select_all_btn)

        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.clicked.connect(lambda: self.set_all_signals(False))
        left_layout.addWidget(clear_btn)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        splitter.addWidget(right)
        splitter.setSizes([360, 1190])

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
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select robotic CSV folder", str(self.data_root))
        if folder:
            self.root_edit.setText(folder)
            self.reload_files()

    def reload_files(self):
        self.data_root = Path(self.root_edit.text()).expanduser()
        self.files = find_csv_files(self.data_root)
        self.file_combo.blockSignals(True)
        self.file_combo.clear()
        for p in self.files:
            self.file_combo.addItem(p.name, str(p))
        self.file_combo.blockSignals(False)
        if self.files:
            self.file_combo.setCurrentIndex(0)
            self.load_selected_file()
        else:
            self.df = None
            self.info_label.setText(f"No CSV files found in: {self.data_root}")
            self.clear_signal_checkboxes()
            self.clear_plots()

    def load_selected_file(self):
        if not self.files or self.file_combo.currentIndex() < 0:
            return
        path = Path(self.file_combo.currentData())
        try:
            self.df = read_robotic_csv(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Read error", str(exc))
            return
        self.time_col = choose_time_column(self.df)
        variables = numeric_signal_columns(self.df, self.time_col)
        self.info_label.setText(
            f"File: {path.name}\nRows: {len(self.df)}\nColumns: {len(self.df.columns)}\nTime column: {self.time_col or 'sample index'}"
        )
        self.populate_signal_checkboxes(variables)
        self.update_plots()

    def clear_signal_checkboxes(self):
        for cb in self.signal_checkboxes:
            cb.setParent(None)
        self.signal_checkboxes = []

    def populate_signal_checkboxes(self, variables):
        self.clear_signal_checkboxes()
        for i, col in enumerate(variables):
            cb = QtWidgets.QCheckBox(col)
            cb.setChecked(i < MAX_INITIAL_SIGNALS)
            cb.stateChanged.connect(self.update_plots)
            self.signal_layout.insertWidget(self.signal_layout.count() - 1, cb)
            self.signal_checkboxes.append(cb)

    def selected_signals(self):
        return [cb.text() for cb in self.signal_checkboxes if cb.isChecked()]

    def set_all_signals(self, checked):
        for cb in self.signal_checkboxes:
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        self.update_plots()

    def clear_plots(self):
        while self.plot_layout.count():
            item = self.plot_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.plots = []
        self.curves = []
        self.initial_ranges = []

    def update_plots(self):
        self.clear_plots()
        if self.df is None:
            return
        if self.time_col is None:
            x = list(range(len(self.df)))
            x_label = "sample index"
        else:
            x = self.df[self.time_col].to_numpy(dtype=float)
            x_label = self.time_col
        for col in self.selected_signals():
            y = self.df[col].to_numpy(dtype=float)
            plot = ResettablePlotWidget()
            plot.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding,
            )
            plot.setMinimumHeight(180)
            plot.setBackground("w")
            plot.showGrid(x=SHOW_GRID, y=SHOW_GRID)
            plot.setLabel("bottom", x_label)
            plot.setLabel("left", col)
            plot.setTitle(col)
            self.configure_time_plot_interaction(plot)
            curve = plot.plot(x, y, pen=pg.mkPen('b', width=self.line_width_spin.value()))
            plot.double_clicked.connect(self.reset_all_zoom)
            self.plot_layout.addWidget(plot, stretch=1)
            self.plots.append(plot)
            self.curves.append(curve)
            self.initial_ranges.append((plot.viewRange()[0], plot.viewRange()[1]))
        self.link_time_axes()
        self.plot_container.updateGeometry()

    def update_curve_pens(self):
        width = self.line_width_spin.value()
        for curve in self.curves:
            curve.setPen(pg.mkPen('b', width=width))

    def configure_time_plot_interaction(self, plot):
        """Use left-button rectangular zoom for time plots."""
        view_box = plot.getViewBox()
        view_box.setMouseMode(pg.ViewBox.RectMode)

    def link_time_axes(self):
        """Synchronize x-axes across all displayed time plots."""
        if not self.plots:
            return
        master_plot = self.plots[0]
        for plot in self.plots[1:]:
            plot.setXLink(master_plot)

    def reset_all_zoom(self):
        for plot in self.plots:
            plot.enableAutoRange()


def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    win = RoboticRawCSVViewer()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
