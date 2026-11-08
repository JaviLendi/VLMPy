# Author: Javier Lendínez
# SPDX-License-Identifier: AGPL-3.0-only

import importlib
import os
import sys
import traceback
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets


BASE_DIR = Path(sys._MEIPASS) / "vlm" if getattr(sys, "frozen", False) else Path(__file__).parent
IMAGE_DIR = BASE_DIR / "static" / "images"


class AppLoader(QtCore.QThread):
    loaded = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def run(self):
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            self.loaded.emit(importlib.import_module("app"))
        except BaseException:
            self.failed.emit(traceback.format_exc())


class LoadingScreen(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self._background = QtGui.QPixmap(str(IMAGE_DIR / "splash.png"))
        self.setWindowTitle("VLMPy")
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.WindowStaysOnTopHint)
        self.setFixedSize(self._background.size())

        self._progress = QtWidgets.QProgressBar(self)
        self._progress.setRange(0, 0)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet("""
            QProgressBar { background: #1e344e; border: 0; border-radius: 3px; }
            QProgressBar::chunk {
                width: 100px;
                border-radius: 3px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0547b9, stop:0.58 #0086fc, stop:1 #57bcf4);
            }
        """)

        self._status = QtWidgets.QLabel("Loading aerodynamic modules...", self)
        self._status.setStyleSheet("color: #82a0be; background: transparent; font: 12px 'Segoe UI';")

    def resizeEvent(self, event):
        width, height = self.width(), self.height()
        self._progress.setGeometry(round(width * 60 / 640), round(height * 298 / 360),
                                   round(width * 520 / 640), max(6, round(height * 6 / 360)))
        self._status.setGeometry(round(width * 60 / 640), round(height * 307 / 360),
                                 round(width * 400 / 640), round(height * 28 / 360))
        super().resizeEvent(event)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawPixmap(self.rect(), self._background)


class Launcher(QtCore.QObject):
    def __init__(self, application):
        super().__init__()
        self._application = application
        self._screen = LoadingScreen()
        self._screen.show()
        self._worker = AppLoader(self)
        self._worker.loaded.connect(self._start_app)
        self._worker.failed.connect(self._show_error)
        self._worker.start()
        self._application.aboutToQuit.connect(self._worker.quit)

    @QtCore.Slot(object)
    def _start_app(self, app_module):
        try:
            self._screen._status.setText("Starting the embedded browser...")
            self._application.processEvents()
            if os.name == "nt":
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("VLMPy.App")

            icon_path = IMAGE_DIR / "icon.ico"
            self._application.setWindowIcon(QtGui.QIcon(str(icon_path)))
            self._application.setApplicationName("VLMPy")
            self._ui = app_module.WebUI(
                app_module.app,
                port=5001,
                debug=False,
                icon_path=str(icon_path),
                app_name="VLMPy",
            )
            self._ui.start()
            self._screen.close()
            self._application.aboutToQuit.connect(
                lambda: app_module.thread_pool.shutdown(wait=False, cancel_futures=True)
            )
        except Exception:
            self._show_error(traceback.format_exc())

    @QtCore.Slot(str)
    def _show_error(self, details):
        self._screen.close()
        message = QtWidgets.QMessageBox()
        message.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        message.setWindowTitle("VLMPy startup failed")
        message.setText("VLMPy could not start.")
        message.setDetailedText(details)
        message.exec()
        self._application.quit()


def main():
    if os.name == "nt":
        import multiprocessing
        multiprocessing.freeze_support()

    application = QtWidgets.QApplication(sys.argv)
    application.setApplicationName("VLMPy")
    launcher = Launcher(application)
    return application.exec()

if __name__ == "__main__":
    sys.exit(main())