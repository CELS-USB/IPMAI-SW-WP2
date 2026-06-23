# -*- coding: utf-8 -*-
"""
Shared utilities for PID/IPMAI vector result viewers.

"""

# ============================================================
# USER SETUP
# ============================================================

from pathlib import Path
import sys
import os
import subprocess
import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent

DEFAULT_AXIS_FONT_SIZE = 13
DEFAULT_POINT_FONT_SIZE = 10
DEFAULT_MARKER_SIZE = 90
DEFAULT_LINE_MARKER_SIZE = 7

# ============================================================
# IMPORTS
# ============================================================

from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


# ============================================================
# DATA HELPERS
# ============================================================

def read_csv_if_exists(path: Path):
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def numeric_weight_columns(df):
    return [c for c in df.columns if str(c).startswith('w_')]


def saliency_columns(df):
    skip = {
        'seq_index', 'segment_label', 'run_name',
        'fault_id', 'fault_name', 'fault_magnitude',
        'LE_vs_healthy', 'LE_vs_healthy_norm',
        'LE_from_previous_segment', 'LE_from_previous_segment_norm',
        'c1', 'c2', 'pc1', 'pc2', 'label', 'class',
    }
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]



def zscore_rows(data, eps=1e-12):
    """Z-score each row independently along columns/runs.

    Used for weight-codeword heatmaps: each weight gets its own scale across
    runs so small but systematic changes are visible even when absolute weight
    magnitudes differ strongly.
    """
    arr = np.asarray(data, dtype=float)
    mu = np.nanmean(arr, axis=1, keepdims=True)
    sigma = np.nanstd(arr, axis=1, keepdims=True)
    sigma = np.where(sigma < eps, 1.0, sigma)
    z = (arr - mu) / sigma
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def row_labels(df):
    if 'segment_label' in df.columns:
        return df['segment_label'].astype(str).to_numpy()
    if 'run_name' in df.columns:
        return df['run_name'].astype(str).to_numpy()
    if 'seq_index' in df.columns:
        return df['seq_index'].astype(str).to_numpy()
    return np.arange(len(df)).astype(str)


def get_xy_columns(df):
    if {'c1', 'c2'}.issubset(df.columns):
        return 'c1', 'c2'
    if {'pc1', 'pc2'}.issubset(df.columns):
        return 'pc1', 'pc2'
    raise KeyError('CSV must contain c1/c2 or pc1/pc2 columns.')


# ============================================================
# MATPLOTLIB FORMATTING
# ============================================================

def apply_plot_fonts(fig: Figure, axis_font_size: int, point_font_size: int):
    for ax in fig.axes:
        ax.title.set_fontsize(axis_font_size + 1)
        ax.xaxis.label.set_fontsize(axis_font_size)
        ax.yaxis.label.set_fontsize(axis_font_size)
        ax.tick_params(axis='both', labelsize=max(7, axis_font_size - 1))

        for txt in ax.texts:
            txt.set_fontsize(point_font_size)

        leg = ax.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_fontsize(max(7, axis_font_size - 1))
            if leg.get_title() is not None:
                leg.get_title().set_fontsize(axis_font_size)

    fig.tight_layout()


# ============================================================
# BASE VIEWER
# ============================================================

class BasePlotViewer(QtWidgets.QMainWindow):
    WINDOW_TITLE = 'PID result viewer'
    RESULT_DIR = Path('.')
    PLOTS = []

    def __init__(self):
        super().__init__()
        self.result_dir = PACKAGE_ROOT / self.RESULT_DIR
        self.axis_font_size = DEFAULT_AXIS_FONT_SIZE
        self.point_font_size = DEFAULT_POINT_FONT_SIZE
        self.marker_size = DEFAULT_MARKER_SIZE
        self.line_marker_size = DEFAULT_LINE_MARKER_SIZE

        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(1260, 850)

        self._build_ui()
        self.refresh_status()

        if self.PLOTS:
            self.plot_combo.setCurrentIndex(0)
            self.draw_selected_plot()

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main = QtWidgets.QVBoxLayout(central)

        top = QtWidgets.QHBoxLayout()
        main.addLayout(top)

        top.addWidget(QtWidgets.QLabel('Plot:'))
        self.plot_combo = QtWidgets.QComboBox()
        for label, key in self.PLOTS:
            self.plot_combo.addItem(label, key)
        self.plot_combo.currentIndexChanged.connect(self.draw_selected_plot)
        top.addWidget(self.plot_combo, stretch=1)

        top.addWidget(QtWidgets.QLabel('Axis font:'))
        self.axis_font_spin = QtWidgets.QSpinBox()
        self.axis_font_spin.setRange(7, 36)
        self.axis_font_spin.setValue(self.axis_font_size)
        self.axis_font_spin.valueChanged.connect(self.on_style_change)
        top.addWidget(self.axis_font_spin)

        top.addWidget(QtWidgets.QLabel('Point/text font:'))
        self.point_font_spin = QtWidgets.QSpinBox()
        self.point_font_spin.setRange(6, 32)
        self.point_font_spin.setValue(self.point_font_size)
        self.point_font_spin.valueChanged.connect(self.on_style_change)
        top.addWidget(self.point_font_spin)

        top.addWidget(QtWidgets.QLabel('Point size:'))
        self.marker_size_spin = QtWidgets.QSpinBox()
        self.marker_size_spin.setRange(5, 500)
        self.marker_size_spin.setSingleStep(5)
        self.marker_size_spin.setValue(self.marker_size)
        self.marker_size_spin.valueChanged.connect(self.on_style_change)
        top.addWidget(self.marker_size_spin)

        self.zoom_check = QtWidgets.QCheckBox('Zoom left mouse')
        self.zoom_check.setChecked(True)
        self.zoom_check.stateChanged.connect(self.apply_zoom_mode)
        top.addWidget(self.zoom_check)

        self.btn_refresh = QtWidgets.QPushButton('Refresh')
        self.btn_refresh.clicked.connect(self.draw_selected_plot)
        top.addWidget(self.btn_refresh)

        self.btn_open_folder = QtWidgets.QPushButton('Open result folder')
        self.btn_open_folder.clicked.connect(lambda: self.open_folder(self.result_dir))
        top.addWidget(self.btn_open_folder)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main.addWidget(splitter, stretch=1)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        splitter.addWidget(left)

        self.status_box = QtWidgets.QPlainTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setMinimumWidth(330)
        left_layout.addWidget(self.status_box, stretch=1)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        splitter.addWidget(right)

        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Custom left-button drag zoom. This is deliberately independent of
        # matplotlib toolbar state, because some Windows/backend combinations
        # do not keep toolbar zoom mode active after redraws.
        self._drag_zoom_start = None
        self._drag_zoom_cids = [
            self.canvas.mpl_connect('button_press_event', self._on_mouse_press),
            self.canvas.mpl_connect('button_release_event', self._on_mouse_release),
        ]

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas, stretch=1)

        self.info = QtWidgets.QLabel(
            'With Zoom left mouse enabled, drag with the left mouse button to zoom. '
            'Use Home/Back in the toolbar to reset the view.'
        )
        self.info.setWordWrap(True)
        right_layout.addWidget(self.info)

        splitter.setSizes([350, 910])

    def on_style_change(self):
        self.axis_font_size = int(self.axis_font_spin.value())
        self.point_font_size = int(self.point_font_spin.value())
        self.marker_size = int(self.marker_size_spin.value())
        self.line_marker_size = max(2, int(round((self.marker_size / 90.0) ** 0.5 * 7)))
        self.draw_selected_plot()

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    def refresh_status(self):
        rows = [
            f'Package root: {PACKAGE_ROOT}',
            f'Result folder: {self.result_dir}',
            '',
            'Expected data files:',
        ]
        for p in self.expected_files():
            path = self.result_dir / p
            rows.append(f'  {p}: {"OK" if path.exists() else "MISSING"}')
        rows.append('')
        self.status_box.setPlainText('\n'.join(rows))

    def expected_files(self):
        return []

    # --------------------------------------------------------
    # Drawing
    # --------------------------------------------------------

    def draw_selected_plot(self):
        key = self.plot_combo.currentData()
        self.fig.clear()
        try:
            self.draw_plot(key)
            apply_plot_fonts(self.fig, self.axis_font_size, self.point_font_size)
            self.canvas.draw_idle()
            self.apply_zoom_mode()
        except Exception as exc:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.axis('off')
            ax.text(
                0.02, 0.98,
                f'Cannot draw plot from data files:\n{exc}',
                va='top', ha='left', fontsize=self.axis_font_size,
                transform=ax.transAxes,
            )
            self.canvas.draw_idle()
        self.refresh_status()

    def draw_plot(self, key):
        raise NotImplementedError

    def apply_zoom_mode(self):
        """Keep the checkbox state for the custom left-button drag zoom.

        The actual zoom is handled by _on_mouse_press/_on_mouse_release so it
        remains active after redraws and works consistently in all viewers.
        The toolbar is still visible for Home/Back/Save and manual navigation.
        """
        if not hasattr(self, 'toolbar'):
            return
        mode_text = str(getattr(self.toolbar, 'mode', '')).lower()
        if 'zoom' in mode_text:
            # Do not leave toolbar zoom toggled on; custom drag zoom is more
            # reliable and avoids backend-specific toolbar state problems.
            try:
                self.toolbar.zoom()
            except Exception:
                pass


    def _on_mouse_press(self, event):
        """Remember the start of a left-button drag zoom operation."""
        if not hasattr(self, 'zoom_check') or not self.zoom_check.isChecked():
            return
        if event.button != 1 or event.inaxes is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        # If pan mode is active, do not fight with matplotlib navigation.
        mode_text = str(getattr(self.toolbar, 'mode', '')).lower() if hasattr(self, 'toolbar') else ''
        if 'pan' in mode_text:
            return

        self._drag_zoom_start = (event.inaxes, float(event.xdata), float(event.ydata))

    def _on_mouse_release(self, event):
        """Zoom to the rectangle selected by left-button drag."""
        if self._drag_zoom_start is None:
            return
        ax0, x0, y0 = self._drag_zoom_start
        self._drag_zoom_start = None

        if not hasattr(self, 'zoom_check') or not self.zoom_check.isChecked():
            return
        if event.button != 1 or event.inaxes is not ax0:
            return
        if event.xdata is None or event.ydata is None:
            return

        x1 = float(event.xdata)
        y1 = float(event.ydata)
        if abs(x1 - x0) < 1e-12 or abs(y1 - y0) < 1e-12:
            return

        ax0.set_xlim(min(x0, x1), max(x0, x1))
        ax0.set_ylim(min(y0, y1), max(y0, y1))
        self.canvas.draw_idle()

    # --------------------------------------------------------
    # Common plotting helpers
    # --------------------------------------------------------

    def scatter_with_labels(self, ax, x, y, labels, color_values=None, color_label='value', title=''):
        if color_values is None:
            sc = ax.scatter(x, y, s=self.marker_size)
        else:
            sc = ax.scatter(x, y, c=color_values, s=self.marker_size)
            cb = self.fig.colorbar(sc, ax=ax)
            cb.set_label(color_label)

        for xi, yi, label in zip(x, y, labels):
            ax.annotate(
                str(label),
                (xi, yi),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=self.point_font_size,
            )

        ax.set_title(title)
        ax.grid(True)
        return sc

    def line_with_markers(self, ax, x, y, **kwargs):
        kwargs.setdefault('marker', 'o')
        kwargs.setdefault('linewidth', 1.5)
        kwargs.setdefault('markersize', self.line_marker_size)
        return ax.plot(x, y, **kwargs)

    def heatmap_with_labels(self, ax, data, xlabels, ylabels, title, xlabel, ylabel, colorbar_label=None, vmin=None, vmax=None):
        im = ax.imshow(data, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        cb = self.fig.colorbar(im, ax=ax)
        if colorbar_label:
            cb.set_label(colorbar_label)
        return im

    # --------------------------------------------------------
    # Folders
    # --------------------------------------------------------

    def open_folder(self, folder: Path):
        folder = Path(folder)
        if not folder.exists():
            QtWidgets.QMessageBox.warning(self, 'Missing folder', f'Folder not found:\n{folder}')
            return
        if os.name == 'nt':
            os.startfile(str(folder))
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', str(folder)])
        else:
            subprocess.Popen(['xdg-open', str(folder)])
