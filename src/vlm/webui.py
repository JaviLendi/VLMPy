from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
import psutil
from PySide6 import QtCore, QtWidgets, QtWebEngineCore, QtWebEngineWidgets, QtGui

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

        # Initialize thread pool for background tasks
        self._thread_pool = ThreadPoolExecutor(max_workers=self.MAX_WORKERS)

        # Initialize Qt application
        self._qt_app = QtWidgets.QApplication([])
        if icon_path:
            self._qt_app.setWindowIcon(QtGui.QIcon(icon_path))
        if app_name:
            self._qt_app.setApplicationName(app_name)

        # Initialize main window and web view
        self._main_window = QtWidgets.QMainWindow()
        self._view = QtWebEngineWidgets.QWebEngineView()
        self._page = CustomWebEnginePage(self._view)
        self._view.setPage(self._page)

        # Initialize Flask thread
        self._flask_thread = Thread(
            target=self._run_flask,
            args=(url, port, debug, using_win64),
            daemon=True
        )

        # Optimize process priority
        self._optimize_resources()

        # Setup modern UI
        self._setup_ui()

    def run(self):
        """Start Flask and GUI components."""
        self._flask_thread.start()
        self._run_gui()

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
        if not ok:
            print("[INFO] Reloading...")
            QtCore.QTimer.singleShot(1000, self._view.reload)  # espera 1s y recarga


    def _run_gui(self):
        """Configure and start the GUI."""
        self._view.load(QtCore.QUrl(self._url))

        # Añadir gestión de fallo en carga inicial
        self._view.loadFinished.connect(self._handle_load_finished)

        settings = self._view.page().settings()
        settings.setAttribute(QtWebEngineCore.QWebEngineSettings.LocalStorageEnabled, True)
        settings.setAttribute(QtWebEngineCore.QWebEngineSettings.PluginsEnabled, True)

        self._main_window.showMaximized()
        self._qt_app.exec()

    def _run_flask(self, host, port, debug=False, using_win32=False):
        """Run the Flask application with multithreading."""
        if using_win32:
            import pythoncom
            pythoncom.CoInitialize()
        self._flask_app.run(host=host, port=port, debug=debug, use_reloader=False, threaded=True)

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