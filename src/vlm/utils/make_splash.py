"""Generate the Qt loading screen background using the app's dark theme palette."""
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
IMG = ROOT / "docs" / "img"
W, H, S = 640, 360, 2  # rendered at 2x for smooth edges

BG_TOP, BG_BOTTOM = (7, 21, 34), (13, 32, 52)       # --primary-bg -> --secondary-bg
TEXT, MUTED = (237, 246, 255), (130, 160, 190)


def font(name, size):
    for candidate in (name, "segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size * S)
        except OSError:
            continue
    return ImageFont.load_default()


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


img = Image.new("RGB", (W * S, H * S))
d = ImageDraw.Draw(img)
for y in range(H * S):
    d.line([(0, y), (W * S, y)], fill=lerp(BG_TOP, BG_BOTTOM, y / (H * S)))

# Blueprint grid, echoing the icon background
grid = Image.new("RGBA", img.size, (0, 0, 0, 0))
g = ImageDraw.Draw(grid)
for x in range(0, W * S, 32 * S):
    g.line([(x, 0), (x, H * S)], fill=(22, 136, 239, 18), width=S)
for y in range(0, H * S, 32 * S):
    g.line([(0, y), (W * S, y)], fill=(22, 136, 239, 18), width=S)
img = Image.alpha_composite(img.convert("RGBA"), grid)

# Soft accent glow behind the icon
glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
gd = ImageDraw.Draw(glow)
cx, cy = 150 * S, 160 * S
for r in range(150 * S, 0, -4 * S):
    a = int(40 * (1 - r / (150 * S)) ** 2)
    gd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(22, 136, 239, a))
img = Image.alpha_composite(img, glow)

# Icon
icon = Image.open(IMG / "icon_1024.png").convert("RGBA").resize((120 * S, 120 * S), Image.LANCZOS)
img.alpha_composite(icon, (90 * S, 100 * S))

d = ImageDraw.Draw(img)
d.text((240 * S, 112 * S), "VLMPy", font=font("segoeuib.ttf", 52), fill=TEXT)
d.text((243 * S, 176 * S), "Vortex Lattice Method", font=font("segoeui.ttf", 18), fill=(87, 188, 244))
d.text((243 * S, 202 * S), "Aerodynamic analysis of wings", font=font("segoeui.ttf", 14), fill=MUTED)

# Qt draws the animated progress bar over this empty track.
bx0, bx1, by, bh = 60 * S, (W - 60) * S, 300 * S, 4 * S
d.rounded_rectangle([bx0, by, bx1, by + bh], radius=bh // 2, fill=(30, 52, 78))
d.text((bx1, 316 * S), "Author: Javier Lendínez", font=font("segoeui.ttf", 12), fill=MUTED, anchor="ra")

out = IMG / "splash.png"
img.convert("RGB").resize((W, H), Image.LANCZOS).save(out)
shutil.copyfile(out, ROOT / "src" / "vlm" / "static" / "images" / "splash.png")
print(out)
