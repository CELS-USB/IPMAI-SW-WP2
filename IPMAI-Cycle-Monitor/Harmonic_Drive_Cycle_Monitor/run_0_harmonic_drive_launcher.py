# -*- coding: utf-8 -*-
"""
Harmonic Drive Cycle Monitor launcher.

Purpose
-------
Main PySide6 launcher window for the complete Harmonic Drive cycle-monitor package.

It provides one GUI entry point for:

    1. raw XLS/XLSX/CSV inspection,
    2. LNU/LE/saliency cycle-monitor processing,
    3. MLP training from LNU weights,
    4. results visualization,
    5. Sphinx documentation opening/building.

Windows usage
-------------
    python run_0_harmonic_drive_launcher.py

Authoring convention: all user setups are intentionally kept at the top.
"""

# ============================================================
# USER SETUP
# ============================================================

from pathlib import Path
import sys
import os
import webbrowser
import subprocess

PACKAGE_ROOT = Path(__file__).resolve().parent


def first_existing_script(*names):
    """Return the first existing script in PACKAGE_ROOT, otherwise the first candidate path."""
    for name in names:
        path = PACKAGE_ROOT / name
        if path.exists():
            return path
    return PACKAGE_ROOT / names[0]


SCRIPT_RAW_VIEWER = first_existing_script(
    "run_1_harmonic_drive_raw_xls_viewer.py",
    "run_1_harmonic_drive_raw_excel_viewer.py",
    "run_harmonic_drive_signal_viewer.py",
)

SCRIPT_LNU_MONITOR = first_existing_script(
    "run_2_harmonic_drive_lnu_cycle_monitor.py",
    "run_harmonic_drive_analysis.py",
)

SCRIPT_MLP_TRAINING = first_existing_script(
    "run_3_train_mlp_from_harmonic_drive_lnu_weights.py",
)

SCRIPT_RESULTS_VIEWER = first_existing_script(
    "run_4_harmonic_drive_results_viewer.py",
    "run_harmonic_drive_feature_viewer.py",
    "run_harmonic_drive_variable_comparison_viewer.py",
)

DATA_ROOT = PACKAGE_ROOT / "data_harmonic_drive_cycle_monitor"
CYCLE_RESULTS_ROOT = PACKAGE_ROOT / "harmonic_drive_lnu_cycle_monitor_results"
MLP_RESULTS_ROOT = PACKAGE_ROOT / "harmonic_drive_mlp_from_lnu_weights_results"

DOCS_ROOT = PACKAGE_ROOT / "harmonic_drive_cycle_monitor_sphinx_docs_windows_detailed"
DOCS_INDEX_HTML = DOCS_ROOT / "_build" / "html" / "index.html"
DOCS_INDEX_RST = DOCS_ROOT / "index.rst"

WINDOW_TITLE = "Harmonic Drive Cycle Monitor Launcher"
WINDOW_WIDTH = 1180
WINDOW_HEIGHT = 760

# For long-running processing scripts, keep output visible in the launcher.
PROCESS_OUTPUT_LIMIT_CHARS = 200000

# ============================================================
# IMPORTS
# ============================================================

from PySide6 import QtCore, QtGui, QtWidgets


# ============================================================
# IMPLEMENTATION
# ============================================================

class HarmonicDriveLauncher(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

        self.process = None

        self._build_ui()
        self.refresh_status()

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main = QtWidgets.QVBoxLayout(central)

        title = QtWidgets.QLabel("Harmonic Drive Cycle Monitor")
        font = title.font()
        font.setPointSize(18)
        font.setBold(True)
        title.setFont(font)
        main.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "Main launcher for raw data inspection, LNU cycle monitoring, MLP training, "
            "results visualization, and Sphinx documentation."
        )
        subtitle.setWordWrap(True)
        main.addWidget(subtitle)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main.addWidget(splitter, stretch=1)

        # Left panel -------------------------------------------------------
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        splitter.addWidget(left)

        self.status_box = QtWidgets.QPlainTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setMinimumHeight(240)
        self.status_box.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        left_layout.addWidget(self.status_box, stretch=3)

        actions_group = QtWidgets.QGroupBox("Pipeline actions")
        actions_layout = QtWidgets.QVBoxLayout(actions_group)
        left_layout.addWidget(actions_group)

        self.btn_raw = QtWidgets.QPushButton("1. Open raw XLS/XLSX/XLS/XLSX/CSV viewer")
        self.btn_raw.clicked.connect(lambda: self.start_script(SCRIPT_RAW_VIEWER, detached=True))
        actions_layout.addWidget(self.btn_raw)

        self.btn_lnu = QtWidgets.QPushButton("2. Run LNU cycle monitor")
        self.btn_lnu.clicked.connect(lambda: self.start_script(SCRIPT_LNU_MONITOR, detached=False))
        actions_layout.addWidget(self.btn_lnu)

        self.btn_mlp = QtWidgets.QPushButton("3. Train MLP from LNU weights")
        self.btn_mlp.clicked.connect(lambda: self.start_script(SCRIPT_MLP_TRAINING, detached=False))
        actions_layout.addWidget(self.btn_mlp)

        self.btn_results = QtWidgets.QPushButton("4. Open results viewer")
        self.btn_results.clicked.connect(lambda: self.start_script(SCRIPT_RESULTS_VIEWER, detached=True))
        actions_layout.addWidget(self.btn_results)

        docs_group = QtWidgets.QGroupBox("Documentation")
        docs_layout = QtWidgets.QVBoxLayout(docs_group)
        left_layout.addWidget(docs_group)

        self.btn_docs = QtWidgets.QPushButton("Open Sphinx documentation")
        self.btn_docs.clicked.connect(self.open_documentation)
        docs_layout.addWidget(self.btn_docs)

        self.btn_build_docs = QtWidgets.QPushButton("Build Sphinx HTML documentation")
        self.btn_build_docs.clicked.connect(self.build_documentation)
        docs_layout.addWidget(self.btn_build_docs)

        folders_group = QtWidgets.QGroupBox("Folders")
        folders_layout = QtWidgets.QVBoxLayout(folders_group)
        left_layout.addWidget(folders_group)

        self.btn_open_root = QtWidgets.QPushButton("Open package folder")
        self.btn_open_root.clicked.connect(lambda: self.open_folder(PACKAGE_ROOT))
        folders_layout.addWidget(self.btn_open_root)

        self.btn_open_data = QtWidgets.QPushButton("Open data_harmonic_drive_cycle_monitor folder")
        self.btn_open_data.clicked.connect(lambda: self.open_folder(DATA_ROOT))
        folders_layout.addWidget(self.btn_open_data)

        self.btn_open_results = QtWidgets.QPushButton("Open LNU results folder")
        self.btn_open_results.clicked.connect(lambda: self.open_folder(CYCLE_RESULTS_ROOT))
        folders_layout.addWidget(self.btn_open_results)

        self.btn_open_mlp = QtWidgets.QPushButton("Open MLP results folder")
        self.btn_open_mlp.clicked.connect(lambda: self.open_folder(MLP_RESULTS_ROOT))
        folders_layout.addWidget(self.btn_open_mlp)

        bottom_controls = QtWidgets.QHBoxLayout()
        left_layout.addLayout(bottom_controls)

        self.btn_refresh = QtWidgets.QPushButton("Refresh status")
        self.btn_refresh.clicked.connect(self.refresh_status)
        bottom_controls.addWidget(self.btn_refresh)

        self.btn_stop = QtWidgets.QPushButton("Stop running process")
        self.btn_stop.clicked.connect(self.stop_process)
        bottom_controls.addWidget(self.btn_stop)

        self.run_status_label = QtWidgets.QLabel("Idle.")
        left_layout.addWidget(self.run_status_label)

        self.busy_bar = QtWidgets.QProgressBar()
        self.busy_bar.setRange(0, 1)
        self.busy_bar.setValue(0)
        self.busy_bar.setTextVisible(True)
        self.busy_bar.setFormat("Idle")
        left_layout.addWidget(self.busy_bar)

        # No trailing stretch: status/log-related controls use the available height.

        # Right panel ------------------------------------------------------
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        splitter.addWidget(right)

        right_layout.addWidget(QtWidgets.QLabel("Process log:"))
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        right_layout.addWidget(self.log_box, stretch=1)

        splitter.setSizes([390, 790])

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    def refresh_status(self):
        rows = []
        rows.append(f"Package root: {PACKAGE_ROOT}")
        rows.append("")
        rows.append("Scripts:")
        for label, path in [
            ("Raw XLS/XLSX/XLS/XLSX/CSV viewer", SCRIPT_RAW_VIEWER),
            ("LNU cycle monitor", SCRIPT_LNU_MONITOR),
            ("MLP training", SCRIPT_MLP_TRAINING),
            ("Results viewer", SCRIPT_RESULTS_VIEWER),
        ]:
            rows.append(f"  {label}: {'OK' if path.exists() else 'MISSING'}  ({path.name})")

        rows.append("")
        rows.append("Folders:")
        for label, path in [
            ("data_harmonic_drive_cycle_monitor", DATA_ROOT),
            ("LNU results", CYCLE_RESULTS_ROOT),
            ("MLP results", MLP_RESULTS_ROOT),
            ("Sphinx docs", DOCS_ROOT),
        ]:
            rows.append(f"  {label}: {'OK' if path.exists() else 'MISSING'}")

        rows.append("")
        rows.append("Documentation:")
        rows.append(f"  HTML index: {'OK' if DOCS_INDEX_HTML.exists() else 'MISSING'}")
        rows.append(f"  RST index:  {'OK' if DOCS_INDEX_RST.exists() else 'MISSING'}")

        self.status_box.setPlainText("\n".join(rows))

    def set_busy(self, busy: bool, message: str = ""):
        """Update GUI busy indicator for long-running tasks."""
        if busy:
            self.run_status_label.setText(message or "Running...")
            self.busy_bar.setRange(0, 0)  # indeterminate/busy mode
            self.busy_bar.setFormat("Running...")
            self.btn_lnu.setEnabled(False)
            self.btn_mlp.setEnabled(False)
            self.btn_build_docs.setEnabled(False)
        else:
            self.run_status_label.setText(message or "Idle.")
            self.busy_bar.setRange(0, 1)
            self.busy_bar.setValue(0)
            self.busy_bar.setFormat("Idle")
            self.btn_lnu.setEnabled(True)
            self.btn_mlp.setEnabled(True)
            self.btn_build_docs.setEnabled(True)

    # --------------------------------------------------------
    # Process execution
    # --------------------------------------------------------

    def start_script(self, script_path: Path, detached: bool):
        script_path = Path(script_path)

        if not script_path.exists():
            QtWidgets.QMessageBox.critical(
                self,
                "Missing script",
                f"Script not found:\n{script_path}",
            )
            return

        if detached:
            try:
                subprocess.Popen(
                    [sys.executable, str(script_path)],
                    cwd=str(PACKAGE_ROOT),
                    creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0,
                )
                self.append_log(f"[STARTED DETACHED] {script_path.name}\n")
                self.run_status_label.setText(f"Opened: {script_path.name}")
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Launch error", str(exc))
            return

        if self.process is not None and self.process.state() != QtCore.QProcess.NotRunning:
            QtWidgets.QMessageBox.warning(
                self,
                "Process already running",
                "A processing task is already running. Stop it first if needed.",
            )
            return

        self.process = QtCore.QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments([str(script_path)])
        self.process.setWorkingDirectory(str(PACKAGE_ROOT))
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)

        self.process.readyReadStandardOutput.connect(self.handle_process_output)
        self.process.finished.connect(self.handle_process_finished)
        self.process.errorOccurred.connect(self.handle_process_error)

        self.append_log(f"\n[START] {script_path.name}\n")
        self.set_busy(True, f"Running: {script_path.name}")
        self.process.start()

    def handle_process_output(self):
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        self.append_log(data)

    def handle_process_finished(self, exit_code, exit_status):
        self.append_log(f"\n[FINISHED] exit_code={exit_code}, exit_status={exit_status}\n")
        self.set_busy(False, f"Finished: exit_code={exit_code}")
        self.refresh_status()

    def handle_process_error(self, error):
        self.append_log(f"\n[PROCESS ERROR] {error}\n")
        self.set_busy(False, "Process error.")

    def stop_process(self):
        if self.process is None or self.process.state() == QtCore.QProcess.NotRunning:
            self.append_log("[INFO] No active process to stop.\n")
            return
        self.process.terminate()
        if not self.process.waitForFinished(3000):
            self.process.kill()
        self.set_busy(False, "Stopped.")
        self.append_log("[STOP REQUESTED]\n")

    def append_log(self, text):
        current = self.log_box.toPlainText()
        updated = current + str(text)

        if len(updated) > PROCESS_OUTPUT_LIMIT_CHARS:
            updated = updated[-PROCESS_OUTPUT_LIMIT_CHARS:]

        self.log_box.setPlainText(updated)
        cursor = self.log_box.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.log_box.setTextCursor(cursor)

    # --------------------------------------------------------
    # Documentation and folders
    # --------------------------------------------------------

    def open_documentation(self):
        if DOCS_INDEX_HTML.exists():
            webbrowser.open(DOCS_INDEX_HTML.resolve().as_uri())
            self.append_log(f"[OPEN DOCS] {DOCS_INDEX_HTML}\n")
            return

        QtWidgets.QMessageBox.warning(
            self,
            "Documentation not built",
            "HTML documentation was not found. Use 'Build Sphinx HTML documentation' first.",
        )

    def build_documentation(self):
        if not DOCS_ROOT.exists():
            QtWidgets.QMessageBox.critical(
                self,
                "Missing docs folder",
                f"Sphinx documentation folder not found:\n{DOCS_ROOT}",
            )
            return

        if self.process is not None and self.process.state() != QtCore.QProcess.NotRunning:
            QtWidgets.QMessageBox.warning(
                self,
                "Process already running",
                "A process is already running. Stop it first if needed.",
            )
            return

        self.process = QtCore.QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments(["-m", "sphinx", "-E", "-b", "html", "-d", str(DOCS_ROOT / "_build" / "doctrees"), str(DOCS_ROOT), str(DOCS_ROOT / "_build" / "html")])
        self.process.setWorkingDirectory(str(PACKAGE_ROOT))
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)

        self.process.readyReadStandardOutput.connect(self.handle_process_output)
        self.process.finished.connect(self.handle_process_finished)
        self.process.errorOccurred.connect(self.handle_process_error)

        self.append_log("\n[START] python -m sphinx -E -b html -d _build\\doctrees . _build\\html\n")
        self.set_busy(True, "Building Sphinx documentation...")
        self.process.start()

    def open_folder(self, folder: Path):
        folder = Path(folder)
        if not folder.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "Missing folder",
                f"Folder not found:\n{folder}",
            )
            return

        if os.name == "nt":
            os.startfile(str(folder))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Harmonic Drive Cycle Monitor Launcher")
    win = HarmonicDriveLauncher()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
