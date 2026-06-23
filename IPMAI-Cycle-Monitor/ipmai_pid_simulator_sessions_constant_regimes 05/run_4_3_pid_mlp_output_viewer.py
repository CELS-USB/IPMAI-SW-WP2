# -*- coding: utf-8 -*-
"""
Interactive vector viewer for actual outputs of run_3_train_mlp_from_saved_lnu_weights.py.

It redraws confusion matrix and probability heatmaps from CSV data instead of
showing exported PNG images.
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
from run_4_0_pid_viewer_common import BasePlotViewer, read_csv_if_exists, row_labels


class MlpOutputViewer(BasePlotViewer):
    WINDOW_TITLE = 'PID MLP output viewer - redrawn from classifier data'
    RESULT_DIR = 'mlp_from_saved_lnu_weights_results'
    PLOTS = [
        ('Confusion matrix', 'confusion'),
        ('Output probabilities by run', 'probabilities'),
    ]

    def expected_files(self):
        return [
            'confusion_matrix.csv',
            'mlp_probabilities_from_lnu_weights.csv',
            'classification_report.txt',
            'training_summary.json',
        ]

    def draw_plot(self, key):
        if key == 'confusion':
            df = read_csv_if_exists(self.result_dir / 'confusion_matrix.csv')
            if df is None:
                raise FileNotFoundError('confusion_matrix.csv')
            labels = df.iloc[:, 0].astype(str).to_list()
            mat = df.iloc[:, 1:].to_numpy(dtype=float)
            ax = self.fig.add_subplot(111)
            im = ax.imshow(mat, aspect='auto')
            ax.set_xticks(np.arange(mat.shape[1]))
            ax.set_xticklabels(df.columns[1:], rotation=45, ha='right')
            ax.set_yticks(np.arange(mat.shape[0]))
            ax.set_yticklabels(labels)
            ax.set_xlabel('predicted class')
            ax.set_ylabel('true class')
            ax.set_title('MLP confusion matrix from LNU-weight features')
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(
                        j, i, f'{mat[i, j]:.0f}',
                        ha='center', va='center',
                        fontsize=self.point_font_size,
                    )
            cb = self.fig.colorbar(im, ax=ax)
            cb.set_label('count')
            return

        if key == 'probabilities':
            df = read_csv_if_exists(self.result_dir / 'mlp_probabilities_from_lnu_weights.csv')
            if df is None:
                raise FileNotFoundError('mlp_probabilities_from_lnu_weights.csv')
            prob_cols = [c for c in df.columns if str(c).startswith('P_')]
            if not prob_cols:
                raise ValueError('No probability columns P_* found.')
            data = df[prob_cols].to_numpy(dtype=float).T
            labels = row_labels(df)
            class_labels = [c[2:] for c in prob_cols]
            ax = self.fig.add_subplot(111)
            self.heatmap_with_labels(
                ax, data, labels, class_labels,
                title='MLP output probabilities from saved LNU weights',
                xlabel='run index',
                ylabel='MLP class',
                colorbar_label='probability',
                vmin=0,
                vmax=1,
            )

            # Add probability values directly into heatmap cells; controlled by Point/text font.
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                            fontsize=self.point_font_size)
            return


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MlpOutputViewer()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
