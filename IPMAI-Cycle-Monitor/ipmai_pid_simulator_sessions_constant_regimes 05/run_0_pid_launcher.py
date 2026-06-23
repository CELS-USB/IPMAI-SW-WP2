# -*- coding: utf-8 -*-
"""
IPMAI PID simulator launcher.

This launcher starts the original PID pipeline scripts and vector result viewers.
The result viewers redraw analyzer CSV outputs; they do not merely show exported
PNG screenshots.
"""

# ============================================================
# USER SETUP
# ============================================================

from pathlib import Path
import sys
import os
import subprocess
import webbrowser

PACKAGE_ROOT = Path(__file__).resolve().parent

SCRIPT_SIMULATOR = PACKAGE_ROOT / 'run_1_controlled_system_simulator.py'
SCRIPT_LNU_RUN_ANALYZER = PACKAGE_ROOT / 'run_2.1_ipmai_lnu_run_analyzer.py'
SCRIPT_LNU_SEQUENCE_ANALYZER = PACKAGE_ROOT / 'run2.2_ipmai_lnu_run_analyzer.py'
SCRIPT_MLP_TRAINING = PACKAGE_ROOT / 'run_3_train_mlp_from_saved_lnu_weights.py'
SCRIPT_ORIGINAL_RUN_VIEWER = PACKAGE_ROOT / 'run_ipmai_run_viewer.py'
SCRIPT_ORIGINAL_VARIABLE_VIEWER = PACKAGE_ROOT / 'run_ipmai_variable_comparison_viewer.py'
SCRIPT_LNU_RUN_OUTPUT_VIEWER = PACKAGE_ROOT / 'run_4_1_pid_lnu_run_output_viewer.py'
SCRIPT_LNU_SEQUENCE_OUTPUT_VIEWER = PACKAGE_ROOT / 'run_4_2_pid_lnu_sequence_output_viewer.py'
SCRIPT_MLP_OUTPUT_VIEWER = PACKAGE_ROOT / 'run_4_3_pid_mlp_output_viewer.py'

DOCS_ROOT = PACKAGE_ROOT / 'ipmai_sphinx_docs_windows_detailed'
DOCS_INDEX_HTML = DOCS_ROOT / '_build' / 'html' / 'index.html'
PROCESS_OUTPUT_LIMIT_CHARS = 200000

# ============================================================
# IMPORTS
# ============================================================

from PySide6 import QtCore, QtGui, QtWidgets


class PIDLauncher(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('IPMAI PID Simulator Launcher')
        self.resize(1180, 760)
        self.process = None
        self._build_ui()
        self.refresh_status()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main = QtWidgets.QVBoxLayout(central)

        title = QtWidgets.QLabel('IPMAI PID Simulator')
        font = title.font()
        font.setPointSize(18)
        font.setBold(True)
        title.setFont(font)
        main.addWidget(title)

        subtitle = QtWidgets.QLabel(
            'Launcher for simulation, LNU/MLP processing, and vector viewers of saved analyzer data.'
        )
        subtitle.setWordWrap(True)
        main.addWidget(subtitle)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main.addWidget(splitter, stretch=1)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        splitter.addWidget(left)

        self.status_box = QtWidgets.QPlainTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setMinimumHeight(250)
        left_layout.addWidget(self.status_box, stretch=3)

        actions_group = QtWidgets.QGroupBox('Pipeline actions')
        actions_layout = QtWidgets.QVBoxLayout(actions_group)
        left_layout.addWidget(actions_group)

        self._add_button(actions_layout, '1. Run PID simulator', lambda: self.start_script(SCRIPT_SIMULATOR, False))
        self._add_button(actions_layout, '2.1 Run LNU run analyzer', lambda: self.start_script(SCRIPT_LNU_RUN_ANALYZER, False))
        self._add_button(actions_layout, '2.2 Run LNU sequence analyzer', lambda: self.start_script(SCRIPT_LNU_SEQUENCE_ANALYZER, False))
        self._add_button(actions_layout, '3. Train MLP from saved LNU weights', lambda: self.start_script(SCRIPT_MLP_TRAINING, False))

        viewers_group = QtWidgets.QGroupBox('Interactive vector viewers')
        viewers_layout = QtWidgets.QVBoxLayout(viewers_group)
        left_layout.addWidget(viewers_group)

        self._add_button(viewers_layout, 'Open single-run viewer (one measurement/run)', lambda: self.start_script(SCRIPT_ORIGINAL_RUN_VIEWER, True))
        self._add_button(viewers_layout, 'Open variable-across-runs viewer (same variable in all runs)', lambda: self.start_script(SCRIPT_ORIGINAL_VARIABLE_VIEWER, True))
        self._add_button(viewers_layout, 'Open 2.1 LNU result viewer (PCA, LE, saliency, weights)', lambda: self.start_script(SCRIPT_LNU_RUN_OUTPUT_VIEWER, True))
        self._add_button(viewers_layout, 'Open 2.2 sequence/codeword viewer across runs', lambda: self.start_script(SCRIPT_LNU_SEQUENCE_OUTPUT_VIEWER, True))
        self._add_button(viewers_layout, 'Open MLP classification result viewer', lambda: self.start_script(SCRIPT_MLP_OUTPUT_VIEWER, True))

        docs_group = QtWidgets.QGroupBox('Documentation and folders')
        docs_layout = QtWidgets.QVBoxLayout(docs_group)
        left_layout.addWidget(docs_group)
        self._add_button(docs_layout, 'Open Sphinx documentation', self.open_documentation)
        self._add_button(docs_layout, 'Build Sphinx HTML documentation', self.build_documentation)
        self._add_button(docs_layout, 'Open package folder', lambda: self.open_folder(PACKAGE_ROOT))

        bottom = QtWidgets.QHBoxLayout()
        left_layout.addLayout(bottom)
        self._add_button(bottom, 'Refresh status', self.refresh_status)
        self._add_button(bottom, 'Stop running process', self.stop_process)

        self.run_status_label = QtWidgets.QLabel('Idle.')
        left_layout.addWidget(self.run_status_label)

        self.busy_bar = QtWidgets.QProgressBar()
        self.busy_bar.setRange(0, 1)
        self.busy_bar.setValue(0)
        self.busy_bar.setFormat('Idle')
        left_layout.addWidget(self.busy_bar)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        splitter.addWidget(right)
        right_layout.addWidget(QtWidgets.QLabel('Process log:'))
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        right_layout.addWidget(self.log_box, stretch=1)

        splitter.setSizes([420, 760])

    def _add_button(self, layout, text, callback):
        btn = QtWidgets.QPushButton(text)
        btn.clicked.connect(callback)
        layout.addWidget(btn)
        return btn

    def refresh_status(self):
        files = [
            SCRIPT_SIMULATOR,
            SCRIPT_LNU_RUN_ANALYZER,
            SCRIPT_LNU_SEQUENCE_ANALYZER,
            SCRIPT_MLP_TRAINING,
            SCRIPT_ORIGINAL_RUN_VIEWER,
            SCRIPT_ORIGINAL_VARIABLE_VIEWER,
            SCRIPT_LNU_RUN_OUTPUT_VIEWER,
            SCRIPT_LNU_SEQUENCE_OUTPUT_VIEWER,
            SCRIPT_MLP_OUTPUT_VIEWER,
        ]
        rows = [f'Package root: {PACKAGE_ROOT}', '', 'Scripts:']
        for p in files:
            rows.append(f'  {p.name}: {"OK" if p.exists() else "MISSING"}')
        rows += ['', 'Result folders:']
        for name in [
            'simulated_runs',
            'lnu_run_analysis_results',
            'lnu_run_sequence_analysis_results',
            'mlp_from_saved_lnu_weights_results',
        ]:
            p = PACKAGE_ROOT / name
            rows.append(f'  {name}: {"OK" if p.exists() else "MISSING"}')
        self.status_box.setPlainText('\n'.join(rows))

    def start_script(self, script_path: Path, detached: bool):
        if not script_path.exists():
            QtWidgets.QMessageBox.critical(self, 'Missing script', f'Script not found:\n{script_path}')
            return
        if detached:
            try:
                subprocess.Popen(
                    [sys.executable, str(script_path)],
                    cwd=str(PACKAGE_ROOT),
                    creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0,
                )
                self.append_log(f'[STARTED DETACHED] {script_path.name}\n')
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, 'Launch error', str(exc))
            return

        if self.process is not None and self.process.state() != QtCore.QProcess.NotRunning:
            QtWidgets.QMessageBox.warning(self, 'Process already running', 'A processing task is already running.')
            return

        self.process = QtCore.QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments([str(script_path)])
        self.process.setWorkingDirectory(str(PACKAGE_ROOT))
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.handle_process_output)
        self.process.finished.connect(self.handle_process_finished)
        self.process.errorOccurred.connect(self.handle_process_error)
        self.append_log(f'\n[START] {script_path.name}\n')
        self.set_busy(True, f'Running: {script_path.name}')
        self.process.start()

    def set_busy(self, busy, message=''):
        self.run_status_label.setText(message or ('Running...' if busy else 'Idle.'))
        if busy:
            self.busy_bar.setRange(0, 0)
            self.busy_bar.setFormat('Running...')
        else:
            self.busy_bar.setRange(0, 1)
            self.busy_bar.setValue(0)
            self.busy_bar.setFormat('Idle')

    def handle_process_output(self):
        data = bytes(self.process.readAllStandardOutput()).decode(errors='replace')
        self.append_log(data)

    def handle_process_finished(self, exit_code, exit_status):
        self.append_log(f'\n[FINISHED] exit_code={exit_code}, exit_status={exit_status}\n')
        self.set_busy(False, f'Finished: exit_code={exit_code}')
        self.refresh_status()

    def handle_process_error(self, error):
        self.append_log(f'\n[PROCESS ERROR] {error}\n')
        self.set_busy(False, 'Process error.')

    def stop_process(self):
        if self.process is None or self.process.state() == QtCore.QProcess.NotRunning:
            self.append_log('[INFO] No active process to stop.\n')
            return
        self.process.terminate()
        if not self.process.waitForFinished(3000):
            self.process.kill()
        self.set_busy(False, 'Stopped.')
        self.append_log('[STOP REQUESTED]\n')

    def append_log(self, text):
        updated = self.log_box.toPlainText() + str(text)
        if len(updated) > PROCESS_OUTPUT_LIMIT_CHARS:
            updated = updated[-PROCESS_OUTPUT_LIMIT_CHARS:]
        self.log_box.setPlainText(updated)
        cursor = self.log_box.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.log_box.setTextCursor(cursor)

    def open_documentation(self):
        if DOCS_INDEX_HTML.exists():
            webbrowser.open(DOCS_INDEX_HTML.resolve().as_uri())
        else:
            QtWidgets.QMessageBox.warning(self, 'Documentation not built', 'HTML documentation was not found.')


    def build_documentation(self):
        if not DOCS_ROOT.exists():
            QtWidgets.QMessageBox.critical(self, 'Missing docs folder', f'Sphinx documentation folder not found:\n{DOCS_ROOT}')
            return
        if self.process is not None and self.process.state() != QtCore.QProcess.NotRunning:
            QtWidgets.QMessageBox.warning(self, 'Process already running', 'A process is already running. Stop it first if needed.')
            return
        self.process = QtCore.QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments(['-m', 'sphinx', '-E', '-b', 'html', '-d', str(DOCS_ROOT / '_build' / 'doctrees'), str(DOCS_ROOT), str(DOCS_ROOT / '_build' / 'html')])
        self.process.setWorkingDirectory(str(PACKAGE_ROOT))
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.handle_process_output)
        self.process.finished.connect(self.handle_process_finished)
        self.process.errorOccurred.connect(self.handle_process_error)
        self.append_log('\n[START] python -m sphinx -E -b html -d _build\\doctrees . _build\\html\n')
        self.set_busy(True, 'Building Sphinx documentation...')
        self.process.start()

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


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = PIDLauncher()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
