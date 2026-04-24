"""
Run this script once to generate favicon.ico and apple-touch-icon.png
from the SVG logo using Pillow (cairosvg or pillow-svg if available).
Falls back to drawing a simple crimson shield with gold 'T' if SVG rendering unavailable.

Usage: python static/img/generate_icons.py
"""
import os
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent

def generate_fallback_icons():
    """Draw simple TrialGuard icons using Pillow drawing primitives."""
    from PIL import Image, ImageDraw, ImageFont

    CRIMSON = (116, 0, 1)
    GOLD    = (211, 166, 37)
    DARK    = (26, 10, 0)

    def draw_shield_icon(size: int) -> Image.Image:
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        m = size * 0.1
        shield_pts = [
            (m, m),
            (size - m, m),
            (size - m, size * 0.6),
            (size / 2, size - m),
            (m, size * 0.6),
        ]
        d.polygon(shield_pts, fill=CRIMSON)
        d.polygon(shield_pts, outline=GOLD, width=max(1, size // 16))

        # Draw a simple "T" or cross
        cx, cy = size // 2, size // 2
        bar_w = max(2, size // 12)
        # Vertical bar
        d.rectangle([cx - bar_w, int(size * 0.25), cx + bar_w, int(size * 0.72)], fill=GOLD)
        # Horizontal bar
        d.rectangle([int(size * 0.28), cy - bar_w - size // 8, int(size * 0.72), cy - size // 8 + bar_w], fill=GOLD)
        return img

    # ── 32x32 favicon ──
    ico_img = draw_shield_icon(32)
    ico_img.save(OUTPUT_DIR / 'favicon.ico', format='ICO', sizes=[(32, 32)])
    print("✓ favicon.ico generated")

    # ── 180x180 apple-touch-icon ──
    touch_img = draw_shield_icon(180)
    # Add slight background circle
    bg = Image.new('RGBA', (180, 180), DARK + (255,))
    bg.paste(touch_img, (0, 0), touch_img)
    bg.save(OUTPUT_DIR / 'apple-touch-icon.png', format='PNG')
    print("✓ apple-touch-icon.png generated")


if __name__ == '__main__':
    try:
        import cairosvg
        svg_path = OUTPUT_DIR / 'logo.svg'
        cairosvg.svg2png(url=str(svg_path), write_to=str(OUTPUT_DIR / 'apple-touch-icon.png'),
                         output_width=180, output_height=180)
        cairosvg.svg2png(url=str(svg_path), write_to=str(OUTPUT_DIR / 'favicon_raw.png'),
                         output_width=32, output_height=32)
        from PIL import Image
        img = Image.open(OUTPUT_DIR / 'favicon_raw.png')
        img.save(OUTPUT_DIR / 'favicon.ico', format='ICO', sizes=[(32, 32)])
        os.remove(OUTPUT_DIR / 'favicon_raw.png')
        print("✓ Icons generated via cairosvg")
    except ImportError:
        print("cairosvg not available — using Pillow fallback")
        generate_fallback_icons()
