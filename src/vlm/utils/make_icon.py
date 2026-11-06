# Generates docs/img/icon.ico (multi-size) and icon_1024.png from icon.svg using Qt's SVG renderer.
import io
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageFilter
from PySide6.QtCore import QBuffer, QIODevice, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

IMG = Path(__file__).resolve().parents[3] / "docs" / "img"
SIZES = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]

app = QGuiApplication(sys.argv)
renderer = QSvgRenderer(str(IMG / "icon.svg"))


def render(size: int) -> Image.Image:
    image = QImage(size, size, QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)
    renderer.render(painter)
    painter.end()
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    image.save(buf, "PNG")
    return Image.open(io.BytesIO(bytes(buf.data()))).convert("RGBA")


master = render(1024)
master.save(IMG / "icon_1024.png")

# Downscale from the large master and sharpen small sizes so details survive.
frames = []
for size in SIZES:
    frame = master.resize((size, size), Image.LANCZOS)
    if size <= 48:
        frame = frame.filter(ImageFilter.UnsharpMask(radius=0.6, percent=90, threshold=2))
    frames.append(frame)

frames[-1].save(IMG / "icon.ico", format="ICO", sizes=[(s, s) for s in SIZES], append_images=frames[:-1])
# Copy used by the Qt window/taskbar icon at runtime.
shutil.copyfile(IMG / "icon.ico", IMG.parents[1] / "src" / "vlm" / "static" / "images" / "icon.ico")
print("ico sizes:", Image.open(IMG / "icon.ico").info.get("sizes"))
