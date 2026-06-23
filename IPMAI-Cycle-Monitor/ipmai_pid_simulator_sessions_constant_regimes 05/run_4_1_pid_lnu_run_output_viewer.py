# -*- coding: utf-8 -*-
"""
Interactive vector viewer for actual outputs of run_2.1_ipmai_lnu_run_analyzer.py.
"""

# ============================================================
# USER SETUP
# ============================================================

import sys
import numpy as np

# ============================================================
# IMPORTS
# ============================================================

from PySide6 import QtWidgets
from run_4_0_pid_viewer_common import (
    BasePlotViewer,
    read_csv_if_exists,
    saliency_columns,
    numeric_weight_columns,
    row_labels,
    zscore_rows,
    get_xy_columns,
)


class LnuRunOutputViewer(BasePlotViewer):
    WINDOW_TITLE = 'PID LNU 2.1 output viewer - redrawn from analyzer data'
    RESULT_DIR = 'lnu_run_analysis_results'
    PLOTS = [
        ('PCA of final LNU weights', 'pca'),
        ('Learning Entropy vs healthy', 'le'),
        ('Saliency heatmap: x=run, y=variable', 'saliency'),
        ('Weights heatmap: x=run, y=weight', 'weights'),
    ]

    def expected_files(self):
        return [
            'weight_pca_2d.csv',
            'learning_entropy_vs_healthy.csv',
            'saliency_vs_healthy.csv',
            'weights_by_run.csv',
        ]

    def draw_plot(self, key):
        if key == 'pca':
            df = read_csv_if_exists(self.result_dir / 'weight_pca_2d.csv')
            if df is None:
                raise FileNotFoundError('weight_pca_2d.csv')
            xcol, ycol = get_xy_columns(df)
            ccol = 'LE_vs_healthy_norm' if 'LE_vs_healthy_norm' in df.columns else None
            if ccol is None and 'fault_id' in df.columns:
                ccol = 'fault_id'
            ax = self.fig.add_subplot(111)
            labels = row_labels(df)
            colors = df[ccol].to_numpy(dtype=float) if ccol else None
            self.scatter_with_labels(
                ax,
                df[xcol].to_numpy(dtype=float),
                df[ycol].to_numpy(dtype=float),
                labels,
                color_values=colors,
                color_label=ccol or 'value',
                title='2.1 weight-space PCA, one point = one run',
            )
            ax.set_xlabel('PC1 of final LNU weights')
            ax.set_ylabel('PC2 of final LNU weights')
            return

        if key == 'le':
            df = read_csv_if_exists(self.result_dir / 'learning_entropy_vs_healthy.csv')
            if df is None:
                raise FileNotFoundError('learning_entropy_vs_healthy.csv')
            ycol = 'LE_vs_healthy_norm' if 'LE_vs_healthy_norm' in df.columns else 'LE_vs_healthy'
            x = np.arange(len(df))
            labels = row_labels(df)
            ax = self.fig.add_subplot(111)
            ax.bar(x, df[ycol].to_numpy(dtype=float))
            ax.plot(x, df[ycol].to_numpy(dtype=float), 'o', markersize=self.line_marker_size)
            for xi, yi, lab in zip(x, df[ycol].to_numpy(dtype=float), labels):
                ax.annotate(str(lab), (xi, yi), xytext=(0, 5), textcoords='offset points',
                            ha='center', fontsize=self.point_font_size)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_xlabel('run index')
            ax.set_ylabel(ycol)
            ax.set_title('Learning Entropy vs healthy reference')
            ax.grid(True, axis='y')
            return

        if key == 'saliency':
            df = read_csv_if_exists(self.result_dir / 'saliency_vs_healthy.csv')
            if df is None:
                raise FileNotFoundError('saliency_vs_healthy.csv')
            cols = saliency_columns(df)
            if not cols:
                raise ValueError('No numeric saliency columns found.')
            data_raw = df[cols].to_numpy(dtype=float).T
            data = zscore_rows(data_raw)
            ax = self.fig.add_subplot(111)
            self.heatmap_with_labels(
                ax, data, row_labels(df), cols,
                title='2.1 saliency vs healthy, x=run index, y=variable',
                xlabel='run index',
                ylabel='variable / delayed feature',
                colorbar_label='normalized saliency',
            )
            return

        if key == 'weights':
            df = read_csv_if_exists(self.result_dir / 'weights_by_run.csv')
            if df is None:
                raise FileNotFoundError('weights_by_run.csv')
            cols = numeric_weight_columns(df)
            if not cols:
                raise ValueError('No weight columns w_* found.')
            data_raw = df[cols].to_numpy(dtype=float).T
            data = zscore_rows(data_raw)
            ax = self.fig.add_subplot(111)
            self.heatmap_with_labels(
                ax, data, row_labels(df), cols,
                title='Final LNU weight codewords by run, row-wise z-score',
                xlabel='run index',
                ylabel='LNU weight',
                colorbar_label='z-score of each weight across runs',
            )
            return


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = LnuRunOutputViewer()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
