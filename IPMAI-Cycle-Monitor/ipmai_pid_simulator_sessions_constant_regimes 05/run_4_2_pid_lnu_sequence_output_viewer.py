# -*- coding: utf-8 -*-
"""
Interactive vector viewer for actual outputs of run2.2_ipmai_lnu_run_analyzer.py.

It redraws the analyzer CSV outputs in matplotlib, with run/sequence index on the
x-axis and variables/features on the y-axis for heatmaps.
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


class LnuSequenceOutputViewer(BasePlotViewer):
    WINDOW_TITLE = 'PID LNU 2.2 sequence viewer - redrawn from analyzer data'
    RESULT_DIR = 'lnu_run_sequence_analysis_results'
    PLOTS = [
        ('Sequence PCA of LNU weights', 'pca'),
        ('Learning Entropy sequence', 'le'),
        ('Saliency sequence: x=run index, y=variable', 'saliency'),
        ('Sequential weights heatmap', 'weights'),
    ]

    def expected_files(self):
        return [
            'weight_pca_2d_sequence.csv',
            'learning_entropy_sequence.csv',
            'saliency_sequence.csv',
            'sequence_weights.csv',
        ]

    def draw_plot(self, key):
        if key == 'pca':
            df = read_csv_if_exists(self.result_dir / 'weight_pca_2d_sequence.csv')
            if df is None:
                df = read_csv_if_exists(self.result_dir / 'weight_pca_2d.csv')
            if df is None:
                raise FileNotFoundError('weight_pca_2d_sequence.csv')
            xcol, ycol = get_xy_columns(df)
            ccol = 'LE_from_previous_segment_norm' if 'LE_from_previous_segment_norm' in df.columns else None
            if ccol is None and 'fault_id' in df.columns:
                ccol = 'fault_id'
            labels = row_labels(df)
            ax = self.fig.add_subplot(111)
            colors = df[ccol].to_numpy(dtype=float) if ccol else None
            self.scatter_with_labels(
                ax,
                df[xcol].to_numpy(dtype=float),
                df[ycol].to_numpy(dtype=float),
                labels,
                color_values=colors,
                color_label=ccol or 'value',
                title='2.2 sequence weight-space PCA',
            )
            ax.plot(df[xcol].to_numpy(dtype=float), df[ycol].to_numpy(dtype=float),
                    '-', linewidth=1.2, alpha=0.6)
            ax.set_xlabel('PC1 of sequential LNU weights')
            ax.set_ylabel('PC2 of sequential LNU weights')
            return

        if key == 'le':
            df = read_csv_if_exists(self.result_dir / 'learning_entropy_sequence.csv')
            if df is None:
                raise FileNotFoundError('learning_entropy_sequence.csv')
            x = df['seq_index'].to_numpy() if 'seq_index' in df.columns else np.arange(len(df))
            ycol = 'LE_from_previous_segment_norm' if 'LE_from_previous_segment_norm' in df.columns else 'LE_from_previous_segment'
            labels = row_labels(df)
            y = df[ycol].to_numpy(dtype=float)
            ax = self.fig.add_subplot(111)
            self.line_with_markers(ax, x, y)
            for xi, yi, lab in zip(x, y, labels):
                ax.annotate(str(lab), (xi, yi), xytext=(5, 5), textcoords='offset points',
                            fontsize=self.point_font_size)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_xlabel('sequence index / run order')
            ax.set_ylabel(ycol)
            ax.set_title('Learning Entropy from previous segment')
            ax.grid(True)
            return

        if key == 'saliency':
            df = read_csv_if_exists(self.result_dir / 'saliency_sequence.csv')
            if df is None:
                raise FileNotFoundError('saliency_sequence.csv')
            cols = saliency_columns(df)
            if not cols:
                raise ValueError('No numeric saliency columns found.')
            data_raw = df[cols].to_numpy(dtype=float).T
            data = zscore_rows(data_raw)
            ax = self.fig.add_subplot(111)
            self.heatmap_with_labels(
                ax, data, row_labels(df), cols,
                title='2.2 saliency sequence, x=run index, y=variable',
                xlabel='sequence index / run order',
                ylabel='variable / delayed feature',
                colorbar_label='normalized saliency',
            )
            return

        if key == 'weights':
            df = read_csv_if_exists(self.result_dir / 'sequence_weights.csv')
            if df is None:
                df = read_csv_if_exists(self.result_dir / 'weights_by_run.csv')
            if df is None:
                raise FileNotFoundError('sequence_weights.csv')
            cols = numeric_weight_columns(df)
            if not cols:
                raise ValueError('No weight columns w_* found.')
            data_raw = df[cols].to_numpy(dtype=float).T
            data = zscore_rows(data_raw)
            ax = self.fig.add_subplot(111)
            self.heatmap_with_labels(
                ax, data, row_labels(df), cols,
                title='Sequential LNU weight codewords, row-wise z-score',
                xlabel='sequence index / run order',
                ylabel='LNU weight',
                colorbar_label='z-score of each weight across runs',
            )
            return


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = LnuSequenceOutputViewer()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
