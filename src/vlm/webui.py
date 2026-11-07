# Author: Javier Lendínez
# SPDX-License-Identifier: AGPL-3.0-only

from threading import Thread
import json
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
import psutil
from PySide6 import QtCore, QtWidgets, QtWebEngineCore, QtWebEngineWidgets, QtGui
from waitress import create_server
from waitress.wasyncore import close_all


class _Notifier(QtCore.QObject):
    message = QtCore.Signal(str)


class LocalWSGIServer:
    def __init__(self, app, host, port, threads):
        self._socket_map = {}
        self._server = create_server(app, host=host, port=port, threads=threads,
                                     map=self._socket_map, asyncore_loop_timeout=1)
        self._thread = Thread(target=self._server.run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        close_all(self._socket_map)
        self._server.task_dispatcher.shutdown(cancel_pending=True, timeout=1)
        if self._thread.is_alive():
            self._thread.join(timeout=2)

class WebUI:
    DEFAULT_URL = "localhost"
    DEFAULT_PORT = 5000
    MAX_WORKERS = min(multiprocessing.cpu_count() * 4, 8)  # Limit to 2x cores or 8

    def __init__(self, app, url=DEFAULT_URL, port=DEFAULT_PORT, debug=False,
                 using_win64=False, icon_path=None, app_name=None):
        self._flask_app = app
        self._debug = debug
        self._url = f"http://{url}:{port}"
        self._using_win64 = using_win64
        self._host = url
        self._port = port
        self._server = None

        # Initialize thread pool for background tasks
        self._thread_pool = ThreadPoolExecutor(max_workers=self.MAX_WORKERS)

        # Initialize Qt application
        self._qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        if icon_path:
            self._qt_app.setWindowIcon(QtGui.QIcon(icon_path))
        if app_name:
            self._qt_app.setApplicationName(app_name)

        # Initialize main window and web view
        self._main_window = QtWidgets.QMainWindow()
        self._view = QtWebEngineWidgets.QWebEngineView()
        self._page = CustomWebEnginePage(self._view)
        self._view.setPage(self._page)

        self._qt_app.aboutToQuit.connect(self.stop)

        self._load_failures = 0
        self._notifier = _Notifier()
        self._notifier.message.connect(self.notify_error)
        self._page.renderProcessTerminated.connect(self._handle_render_crash)
        self._install_exception_hooks()

        # Optimize process priority
        self._optimize_resources()

        # Setup modern UI
        self._setup_ui()

    def run(self):
        """Start Flask and GUI components."""
        self.start()
        self._qt_app.exec()

    def start(self):
        """Start Flask and show the GUI within the existing Qt event loop."""
        if self._server is None:
            self._flask_app.debug = self._debug
            self._server = LocalWSGIServer(self._flask_app, self._host, self._port, self.MAX_WORKERS)
            self._server.start()
        self._run_gui()

    def stop(self):
        if self._server is not None:
            self._server.stop()
            self._server = None

    def _setup_ui(self):
        """Configure modern UI components."""
        # Set window properties
        self._main_window.setWindowTitle(self._qt_app.applicationName() or "Web Application")
        self._main_window.resize(1280, 720)

        # Create central widget and layout
        central_widget = QtWidgets.QWidget()
        self._main_window.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

    
        # Add web view to layout
        layout.addWidget(self._view)

        # Add status bar
        self._status_bar = QtWidgets.QStatusBar()
        self._status_bar.setStyleSheet("""
            QStatusBar {
                background: #2d2d2d;
                color: #ffffff;
                border-top: 1px solid #4a4a4a;
            }
        """)

    def _load_url(self):
        """Load URL from the URL bar."""
        url = self._url_bar.text()
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        self._view.load(QtCore.QUrl(url))


    def _handle_load_finished(self, ok):
        if ok:
            self._load_failures = 0
            return
        self._load_failures += 1
        if self._load_failures <= 10:
            QtCore.QTimer.singleShot(1000, self._view.reload)  # Wait 1 second, then reload.
            return
        self._view.setHtml(
            "<body style='background:#0a1017;color:#e7eef0;font-family:sans-serif;text-align:center;padding-top:20vh'>"
            "<h2>VLMPy could not load the interface</h2>"
            "<button onclick='location.href=\"" + self._url + "\"'>Retry</button></body>"
        )
        self._load_failures = 0

    def notify_error(self, message):
        """Show an error through the in-page notification center."""
        js = f"window.messageCenter && window.messageCenter.error({json.dumps(str(message))}, 10000);"
        self._page.runJavaScript(js)

    def _handle_render_crash(self, status, exit_code):
        self.notify_error(f"The renderer stopped unexpectedly (code {exit_code}). Reloading...")
        QtCore.QTimer.singleShot(500, self._view.reload)

    def _install_exception_hooks(self):
        def report(exc_type, exc, tb):
            traceback.print_exception(exc_type, exc, tb)
            self._notifier.message.emit(f"Unexpected error: {exc}")

        sys.excepthook = report
        threading.excepthook = lambda args: report(args.exc_type, args.exc_value, args.exc_traceback)


    def _run_gui(self):
        """Configure and start the GUI."""
        self._view.load(QtCore.QUrl(self._url))

        # Handle an initial page load failure.
        self._view.loadFinished.connect(self._handle_load_finished)

        settings = self._view.page().settings()
        settings.setAttribute(QtWebEngineCore.QWebEngineSettings.LocalStorageEnabled, True)
        settings.setAttribute(QtWebEngineCore.QWebEngineSettings.PluginsEnabled, True)

        self._main_window.showMaximized()

    def _optimize_resources(self):
        """Optimize CPU and memory resource allocation."""
        try:
            p = psutil.Process()
            if os.name == 'nt':
                p.nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                p.nice(-5)
            p.cpu_affinity(list(range(multiprocessing.cpu_count())))
            mem_info = p.memory_info()
            if mem_info.rss > 4 * 1024 * 1024 * 1024:
                print("Warning: High memory usage detected:", mem_info.rss / (1024 * 1024), "MB")
        except Exception as e:
            print(f"Resource optimization failed: {e}")

    def submit_background_task(self, task, *args, **kwargs):
        """Submit a background task to the thread pool."""
        return self._thread_pool.submit(task, *args, **kwargs)

    def __del__(self):
        """Cleanup resources."""
        self._thread_pool.shutdown(wait=True)

class CustomWebEnginePage(QtWebEngineCore.QWebEnginePage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.profile().downloadRequested.connect(self.handle_download)

    def createWindow(self, window_type):
        page = CustomWebEnginePage(self)
        page.urlChanged.connect(self._open_browser)
        return page

    def _open_browser(self, url):
        QtGui.QDesktopServices.openUrl(url)
        self.sender().deleteLater()

    def handle_download(self, download_item):
        """Handle file downloads with updated PySide6 API."""
        try:
            # Use downloadFileName() for Qt 6.4+
            suggested_filename = download_item.downloadFileName()
            default_dir = QtCore.QDir.homePath()
            if not suggested_filename:
                # Fallback if no suggested filename is provided
                suggested_filename = download_item.url().fileName() or "downloaded_file"

            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                None,
                "Save File",
                QtCore.QDir(default_dir).filePath(suggested_filename),
                "All Files (*)"
            )

            if save_path:
                # Preserve file extension if not provided in save path
                if not os.path.splitext(save_path)[1] and suggested_filename:
                    extension = os.path.splitext(suggested_filename)[1]
                    save_path += extension

                # Use setDownloadFileName() instead of setPath()
                download_item.setDownloadFileName(save_path)
                download_item.accept()
                # Handle download completion with isFinishedChanged
                download_item.isFinishedChanged.connect(lambda: self._on_download_finished(save_path))
                # Handle progress with receivedBytesChanged
                download_item.receivedBytesChanged.connect(
                    lambda: self._on_download_progress(download_item.receivedBytes(), download_item.totalBytes())
                )
                print(f"Download started: {save_path}")
            else:
                download_item.cancel()
                print("Download canceled by user.")
        except Exception as e:
            print(f"Download error: {e}")
            download_item.cancel()
            QtWidgets.QMessageBox.critical(
                None,
                "Download Error",
                f"Failed to initiate download: {str(e)}"
            )

    def _on_download_progress(self, bytes_received, bytes_total):
        """Display download progress in the status bar."""
        parent = self.view().parent()
        while parent and not isinstance(parent, QtWidgets.QMainWindow):
            parent = parent.parent()
        if parent:
            status_bar = parent.statusBar()
            progress_bar = status_bar.findChild(QtWidgets.QProgressBar)
            if bytes_total > 0:
                progress_bar.setVisible(True)
                progress_bar.setMaximum(100)
                progress_bar.setValue(int((bytes_received / bytes_total) * 100))
                status_bar.showMessage(f"Downloading: {bytes_received}/{bytes_total} bytes")
            else:
                progress_bar.setVisible(False)
                status_bar.clearMessage()

    def _on_download_finished(self, save_path):
        """Handle download completion."""
        parent = self.view().parent()
        while parent and not isinstance(parent, QtWidgets.QMainWindow):
            parent = parent.parent()
        if parent:
            status_bar = parent.statusBar()
            progress_bar = status_bar.findChild(QtWidgets.QProgressBar)
            progress_bar.setVisible(False)
            status_bar.showMessage(f"Download completed: {save_path}", 5000)
        print(f"Download completed: {save_path}")
        QtWidgets.QMessageBox.information(
            None,
            "Download Complete",
            f"File successfully downloaded to:\n{save_path}"
        )