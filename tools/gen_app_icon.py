"""
Generate the Windows app icon (multi-resolution .ico) from img/logo.png.

flutter_launcher_icons only emits a single-resolution Windows .ico, which
Windows cannot render at small sizes (16/32/48) and falls back to the default
app icon. This produces a proper multi-entry .ico instead.

Usage:  python tools/gen_app_icon.py
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "img" / "logo.png"
DST = ROOT / "app" / "windows" / "runner" / "resources" / "app_icon.ico"

SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def main() -> None:
    img = Image.open(SRC).convert("RGBA")
    DST.parent.mkdir(parents=True, exist_ok=True)
    img.save(DST, format="ICO", sizes=SIZES)
    print(f"Wrote {DST} with sizes {SIZES}")


if __name__ == "__main__":
    main()
