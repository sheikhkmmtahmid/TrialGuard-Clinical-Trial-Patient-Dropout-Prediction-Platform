"""
Pure-Python icon generator using only stdlib (zlib + struct).
Creates a TrialGuard favicon.ico and apple-touch-icon.png matching the SVG logo.
Run once: python static/img/make_icons_stdlib.py
"""
import math
import struct
import zlib
from pathlib import Path

OUT = Path(__file__).parent


def _png_chunk(name: bytes, data: bytes) -> bytes:
    c = struct.pack('>I', len(data)) + name + data
    return c + struct.pack('>I', zlib.crc32(c[4:]) & 0xFFFFFFFF)


def _build_png(pixels: list, width: int, height: int) -> bytes:
    """pixels: flat list of (R,G,B,A) tuples, row-major."""
    raw = b''
    for y in range(height):
        raw += b'\x00'
        for x in range(width):
            r, g, b, a = pixels[y * width + x]
            raw += bytes([r, g, b, a])
    compressed = zlib.compress(raw, 9)
    ihdr = struct.pack('>I', width) + struct.pack('>I', height) + bytes([8, 6, 0, 0, 0])
    return (
        b'\x89PNG\r\n\x1a\n'
        + _png_chunk(b'IHDR', ihdr)
        + _png_chunk(b'IDAT', compressed)
        + _png_chunk(b'IEND', b'')
    )


def _blend(bg, fg):
    """Alpha-composite fg over bg, both (R,G,B,A)."""
    a = fg[3] / 255.0
    return (
        int(fg[0] * a + bg[0] * (1 - a)),
        int(fg[1] * a + bg[1] * (1 - a)),
        int(fg[2] * a + bg[2] * (1 - a)),
        255,
    )


def _draw_shield(size: int):
    """
    Draw NHS-themed TrialGuard shield matching the SVG logo:
    - NHS Blue (#003087) background
    - NHS Bright Blue (#0072CE) shield with white border
    - Two white DNA sine-wave strands
    - NHS Aqua (#41B6E6) dot at top centre
    """
    NHS_BG  = (0,   48, 135, 255)   # #003087
    SHIELD  = (0,  114, 206, 255)   # #0072CE
    WHITE   = (255, 255, 255, 255)
    BORDER  = (255, 255, 255, 220)
    AQUA    = (65,  182, 230, 255)  # #41B6E6

    pixels = [NHS_BG] * (size * size)

    def pt(x, y, col):
        if 0 <= x < size and 0 <= y < size:
            pixels[y * size + x] = _blend(pixels[y * size + x], col)

    def fill_rect(x0, y0, x1, y1, col):
        for yy in range(max(0, y0), min(size, y1)):
            for xx in range(max(0, x0), min(size, x1)):
                pixels[yy * size + xx] = _blend(pixels[yy * size + xx], col)

    m     = max(1, size // 10)
    top   = m
    bot   = size - m
    left  = m
    right = size - m
    mid_x = size // 2
    peak  = int(size * 0.75)

    # Shield body
    for y in range(top, bot):
        if y <= peak:
            fill_rect(left, y, right, y + 1, SHIELD)
        else:
            t  = (y - peak) / max(1, bot - peak)
            lx = int(left  + (mid_x - left)  * t)
            rx = int(right - (right - mid_x) * t)
            fill_rect(lx, y, rx, y + 1, SHIELD)

    # White border (1-2px)
    bw = max(1, size // 20)
    for y in range(top, bot):
        if y <= peak:
            lx, rx = left, right
        else:
            t  = (y - peak) / max(1, bot - peak)
            lx = int(left  + (mid_x - left)  * t)
            rx = int(right - (right - mid_x) * t)
        for b in range(bw):
            pt(lx + b, y, BORDER)
            pt(rx - 1 - b, y, BORDER)
    fill_rect(left, top, right, top + bw, BORDER)

    # DNA wave strands: two sine curves with 1.5 periods across the shield width
    w      = right - left
    amp    = size * 0.07          # wave amplitude in pixels
    wave_w = max(1, size // 18)   # stroke half-width

    def wave_y(x_frac, phase, cy):
        return cy + amp * math.sin(x_frac * 2 * math.pi * 1.5 + phase)

    cy1 = top + (bot - top) * 0.36   # upper strand centre y
    cy2 = top + (bot - top) * 0.62   # lower strand centre y

    for px in range(left + bw + 1, right - bw - 1):
        x_frac = (px - left) / max(1, w)
        for cy, phase in [(cy1, 0), (cy2, math.pi)]:
            wy = wave_y(x_frac, phase, cy)
            for dy in range(-wave_w, wave_w + 1):
                iy = int(round(wy)) + dy
                alpha = max(0, 255 - abs(dy) * 80)
                pt(px, iy, (*WHITE[:3], alpha))

    # NHS Aqua dot at top centre
    dot_r = max(1, size // 14)
    dot_cx, dot_cy = mid_x, top - dot_r + bw
    for dy in range(-dot_r, dot_r + 1):
        for dx in range(-dot_r, dot_r + 1):
            if dx * dx + dy * dy <= dot_r * dot_r:
                pt(dot_cx + dx, dot_cy + dy, AQUA)

    return pixels


def build_png_file(size: int) -> bytes:
    return _build_png(_draw_shield(size), size, size)


def build_ico(size: int = 32) -> bytes:
    png_data = build_png_file(size)
    header = struct.pack('<HHH', 0, 1, 1)
    entry  = struct.pack(
        '<BBBBHHII',
        size if size < 256 else 0,
        size if size < 256 else 0,
        0, 0, 1, 32,
        len(png_data), 22,
    )
    return header + entry + png_data


if __name__ == '__main__':
    ico_bytes = build_ico(32)
    (OUT / 'favicon.ico').write_bytes(ico_bytes)
    print(f"OK favicon.ico  ({len(ico_bytes):,} bytes)")

    png_bytes = build_png_file(180)
    (OUT / 'apple-touch-icon.png').write_bytes(png_bytes)
    print(f"OK apple-touch-icon.png  ({len(png_bytes):,} bytes)")

    print("\nIcons generated successfully.")
