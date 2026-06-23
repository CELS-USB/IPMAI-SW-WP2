# -*- coding: utf-8 -*-
"""
IPMAI Main Launcher for three modules.

Purpose
-------
Root-level PySide6 launcher for the complete Cycle Monitor Feature Extraction SW.
It starts the module-specific launchers for:

    1. PID simulator,
    2. Harmonic Drive Cycle Monitor,
    3. Robotic Cycle Monitor.

Windows usage
-------------
    python main_launcher.py

Authoring convention: all user setups are intentionally kept at the top.
"""

# ============================================================
# USER SETUP
# ============================================================

from pathlib import Path
import sys
import subprocess

BASE_DIR = Path(__file__).resolve().parent

PID_DIR = BASE_DIR / "ipmai_pid_simulator_sessions_constant_regimes 05"
HARMONIC_DRIVE_DIR = BASE_DIR / "Harmonic_Drive_Cycle_Monitor"
ROBOTIC_DIR = BASE_DIR / "Robotic_Cycle_Monitor"

PID_LAUNCHER = PID_DIR / "run_0_pid_launcher.py"
HARMONIC_DRIVE_LAUNCHER = HARMONIC_DRIVE_DIR / "run_0_harmonic_drive_launcher.py"
ROBOTIC_LAUNCHER = ROBOTIC_DIR / "run_0_robotic_launcher.py"

MAIN_IMAGE = BASE_DIR / "img" / "main_launcher_02.png"
#MAIN_IMAGE = BASE_DIR / "img" / "main_launcher_01.png"
FALLBACK_IMAGE = BASE_DIR / "img" / "main_launcher.png"

WINDOW_TITLE = "IPMAI Main Launcher"
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 850
WINDOW_MIN_WIDTH = 1000
WINDOW_MIN_HEIGHT = 620

BUTTON_PANEL_WIDTH_RATIO = 0.82
BUTTON_PANEL_HEIGHT_RATIO = 0.15
BUTTON_PANEL_Y_RATIO = 0.705

LOG_WIDTH_RATIO = 0.62
LOG_HEIGHT_RATIO = 0.075
LOG_Y_RATIO = 0.900

EXIT_WIDTH_RATIO = 0.12
EXIT_HEIGHT_RATIO = 0.055
EXIT_Y_RATIO = 0.025

BUTTON_SPACING = 28
BUTTON_MARGIN_LEFT_RIGHT = 24
BUTTON_MARGIN_TOP_BOTTOM = 10

# ============================================================
# IMPORTS
# ============================================================

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QTextEdit,
    QMessageBox,
    QHBoxLayout,
)


# ============================================================
# BACKGROUND WIDGET
# ============================================================

class BackgroundWidget(QWidget):
    def __init__(self, image_path):
        super().__init__()
        self.image_path = Path(image_path)
        if not self.image_path.exists() and FALLBACK_IMAGE.exists():
            self.image_path = FALLBACK_IMAGE
        self.pixmap = QPixmap(str(self.image_path))

    def paintEvent(self, event):
        painter = QPainter(self)

        if self.pixmap.isNull():
            painter.fillRect(self.rect(), Qt.black)
            return

        scaled = self.pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation,
        )

        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)


# ============================================================
# MAIN WINDOW
# ============================================================

class IPMAIMainLauncher(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        self.background = BackgroundWidget(MAIN_IMAGE)
        self.background.setParent(self)

        self.button_panel = QWidget(self)
        self.button_panel.setStyleSheet("background: transparent;")

        button_layout = QHBoxLayout(self.button_panel)
        button_layout.setContentsMargins(
            BUTTON_MARGIN_LEFT_RIGHT,
            BUTTON_MARGIN_TOP_BOTTOM,
            BUTTON_MARGIN_LEFT_RIGHT,
            BUTTON_MARGIN_TOP_BOTTOM,
        )
        button_layout.setSpacing(BUTTON_SPACING)

        self.btn_pid = QPushButton("PID SIMULATOR\nSimulation, detection and classification of \n dynamic system faults in PID control loop ")
        self.btn_pid.clicked.connect(self.start_pid)

        self.btn_harmonic_drive = QPushButton("HARMONIC DRIVE CYCLE MONITOR\nHarmonic gearbox cycle data")
        self.btn_harmonic_drive.clicked.connect(self.start_harmonic_drive)

        self.btn_robotic = QPushButton("ROBOTIC CYCLE MONITOR\nAnalysis of robot data")
        self.btn_robotic.clicked.connect(self.start_robotic)

        self.btn_pid.setStyleSheet(self.module_button_style(
            rgba="rgba(0, 78, 168, 205)",
            rgba_hover="rgba(0, 120, 255, 235)",
            rgba_pressed="rgba(0, 55, 130, 235)",
            border="#4da3ff",
            border_hover="#9fd0ff",
        ))

        self.btn_harmonic_drive.setStyleSheet(self.module_button_style(
            rgba="rgba(24, 132, 92, 210)",
            rgba_hover="rgba(34, 170, 120, 238)",
            rgba_pressed="rgba(16, 92, 64, 238)",
            border="#75e6b3",
            border_hover="#b8ffdd",
        ))

        self.btn_robotic.setStyleSheet(self.module_button_style(
            rgba="rgba(185, 88, 0, 205)",
            rgba_hover="rgba(255, 140, 0, 235)",
            rgba_pressed="rgba(135, 62, 0, 235)",
            border="#ffb347",
            border_hover="#ffd08a",
        ))

        button_layout.addWidget(self.btn_pid)
        button_layout.addWidget(self.btn_harmonic_drive)
        button_layout.addWidget(self.btn_robotic)

        self.log = QTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Launcher log...")
        self.log.setStyleSheet("""
            QTextEdit {
                background-color: rgba(0, 0, 0, 155);
                color: #dddddd;
                border: 1px solid rgba(180, 180, 180, 110);
                border-radius: 10px;
                font-size: 10pt;
                padding: 7px;
            }
        """)

        self.btn_exit = QPushButton("Exit", self)
        self.btn_exit.clicked.connect(self.close)
        self.btn_exit.setStyleSheet("""
            QPushButton {
                background-color: rgba(20, 20, 20, 155);
                color: #dddddd;
                border: 1px solid rgba(180, 180, 180, 90);
                border-radius: 10px;
                font-size: 12pt;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: rgba(80, 80, 80, 210);
                color: white;
            }
        """)

        self.update_layout_geometry()
        self.refresh_button_states()
        self.append_log("[READY] IPMAI Main Launcher")
        self.append_log(f"[IMAGE] {self.background.image_path.name}")

    @staticmethod
    def module_button_style(rgba, rgba_hover, rgba_pressed, border, border_hover):
        return f"""
            QPushButton {{
                background-color: {rgba};
                color: white;
                border: 2px solid {border};
                border-radius: 18px;
                font-size: 18px;
                font-weight: bold;
                padding: 16px;
                text-align: center;
            }}

            QPushButton:hover {{
                background-color: {rgba_hover};
                border: 2px solid {border_hover};
            }}

            QPushButton:pressed {{
                background-color: {rgba_pressed};
            }}

            QPushButton:disabled {{
                background-color: rgba(70, 70, 70, 155);
                color: rgba(210, 210, 210, 170);
                border: 2px solid rgba(150, 150, 150, 120);
            }}
        """

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_layout_geometry()

    def update_layout_geometry(self):
        w = self.width()
        h = self.height()

        self.background.setGeometry(0, 0, w, h)

        panel_w = int(w * BUTTON_PANEL_WIDTH_RATIO)
        panel_h = int(h * BUTTON_PANEL_HEIGHT_RATIO)
        panel_x = int((w - panel_w) / 2)
        panel_y = int(h * BUTTON_PANEL_Y_RATIO)
        self.button_panel.setGeometry(panel_x, panel_y, panel_w, panel_h)

        log_w = int(w * LOG_WIDTH_RATIO)
        log_h = int(h * LOG_HEIGHT_RATIO)
        log_x = int((w - log_w) / 2)
        log_y = int(h * LOG_Y_RATIO)
        self.log.setGeometry(log_x, log_y, log_w, log_h)

        exit_w = max(120, int(w * EXIT_WIDTH_RATIO))
        exit_h = max(36, int(h * EXIT_HEIGHT_RATIO))
        exit_x = int(w - exit_w - 24)
        exit_y = int(h * EXIT_Y_RATIO)
        self.btn_exit.setGeometry(exit_x, exit_y, exit_w, exit_h)

        self.background.lower()
        self.button_panel.raise_()
        self.log.raise_()
        self.btn_exit.raise_()

    def refresh_button_states(self):
        modules = [
            ("PID", self.btn_pid, PID_LAUNCHER),
            ("Harmonic Drive", self.btn_harmonic_drive, HARMONIC_DRIVE_LAUNCHER),
            ("Robotic", self.btn_robotic, ROBOTIC_LAUNCHER),
        ]
        for label, button, launcher in modules:
            exists = launcher.exists()
            button.setEnabled(exists)
            if exists:
                self.append_log(f"[OK] {label} launcher: {launcher.relative_to(BASE_DIR)}")
            else:
                self.append_log(f"[MISSING] {label} launcher: {launcher.relative_to(BASE_DIR)}")

    def append_log(self, text):
        self.log.append(str(text))

    def start_process(self, script_path):
        script_path = Path(script_path)
        if not script_path.exists():
            QMessageBox.critical(
                self,
                "Missing launcher",
                f"Cannot find launcher:\n{script_path}",
            )
            self.append_log(f"[ERROR] Missing launcher: {script_path}")
            self.refresh_button_states()
            return

        try:
            subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(script_path.parent),
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform.startswith("win") else 0,
            )
            self.append_log(f"[STARTED] {script_path.parent.name} / {script_path.name}")
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Launch error",
                f"Could not start:\n{script_path}\n\n{exc}",
            )
            self.append_log(f"[ERROR] {script_path.name}: {exc}")

    def start_pid(self):
        self.start_process(PID_LAUNCHER)

    def start_harmonic_drive(self):
        self.start_process(HARMONIC_DRIVE_LAUNCHER)

    def start_robotic(self):
        self.start_process(ROBOTIC_LAUNCHER)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IPMAIMainLauncher()
    window.show()
    sys.exit(app.exec())
