"""
Production Telegram Bot — Sticker/Emoji → GIF converter
aiogram 3.x | asyncio queue | SQLite | smart recolor | watermark

Fixes v2:
  - TGS fallback reads actual Lottie fill-colors + animates with them
  - Config messages are EDITED in-place (tracks config_msg_id in FSM)
  - /start has inline quick-action menu
  - Prompt messages deleted after user replies (clean chat)
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import sqlite3
import sys
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any, Optional

import aiohttp
import numpy as np
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Sticker,
)
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BOT_TOKEN: str       = os.getenv("BOT_TOKEN", "")
ADMIN_IDS: list[int] = [int(x) for x in os.getenv("ADMIN_IDS", "0").split(",") if x]
DB_PATH: str         = os.getenv("DB_PATH", "bot_data.db")
FONT_CACHE_DIR: Path = Path(os.getenv("FONT_CACHE_DIR", "fonts_cache"))
QUEUE_WORKERS: int   = int(os.getenv("QUEUE_WORKERS", "2"))

FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sticker_bot")

# ─────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────

DEFAULT_SETTINGS: dict[str, Any] = {
    "width":             1920,
    "height":            530,
    "fps":               60,
    "bg_color":          "#FF5E3B",
    "emoji_color":       "#FFFFFF",
    "watermark_text":    "",
    "watermark_enabled": False,
    "font_name":         "Montserrat",
    "output_format":     "GIF",
}

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────

def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def db_init() -> None:
    with db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id  INTEGER PRIMARY KEY,
                settings TEXT NOT NULL DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,
                created_at REAL    NOT NULL,
                status     TEXT    NOT NULL DEFAULT 'pending'
            )
        """)
        conn.commit()
    logger.info("DB ready: %s", DB_PATH)


def db_get_settings(user_id: int) -> dict[str, Any]:
    with db_connect() as conn:
        row = conn.execute(
            "SELECT settings FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
    if row:
        try:
            return {**DEFAULT_SETTINGS, **json.loads(row["settings"])}
        except Exception:
            pass
    return dict(DEFAULT_SETTINGS)


def db_save_settings(user_id: int, settings: dict[str, Any]) -> None:
    with db_connect() as conn:
        conn.execute("""
            INSERT INTO users (user_id, settings) VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET settings = excluded.settings
        """, (user_id, json.dumps(settings)))
        conn.commit()


def db_log_conversion(user_id: int, status: str = "pending") -> int:
    with db_connect() as conn:
        cur = conn.execute(
            "INSERT INTO conversions (user_id, created_at, status) VALUES (?, ?, ?)",
            (user_id, time.time(), status),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]


# ─────────────────────────────────────────────
# FSM
# ─────────────────────────────────────────────

class ConvState(StatesGroup):
    idle        = State()
    waiting_val = State()


# ─────────────────────────────────────────────
# FONTS
# ─────────────────────────────────────────────

FONT_TTF_MAP: dict[str, str] = {
    "Montserrat": "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf",
    "Roboto":     "https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Bold.ttf",
    "OpenSans":   "https://github.com/googlefonts/opensans/raw/main/fonts/ttf/OpenSans-Bold.ttf",
    "Lato":       "https://github.com/google/fonts/raw/main/ofl/lato/Lato-Bold.ttf",
    "Oswald":     "https://github.com/google/fonts/raw/main/ofl/oswald/static/Oswald-Bold.ttf",
}


async def fetch_font(font_name: str) -> Optional[Path]:
    safe   = re.sub(r"[^A-Za-z0-9_-]", "_", font_name)
    cached = FONT_CACHE_DIR / f"{safe}.ttf"
    if cached.exists():
        return cached
    url = FONT_TTF_MAP.get(font_name)
    if not url:
        return None
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url, timeout=aiohttp.ClientTimeout(total=20)) as r:
                if r.status == 200:
                    cached.write_bytes(await r.read())
                    logger.info("Font cached: %s", cached)
                    return cached
    except Exception as exc:
        logger.error("Font fetch: %s", exc)
    return None


def get_pil_font(
    path: Optional[Path], size: int
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if path and path.exists():
        try:
            return ImageFont.truetype(str(path), size)
        except Exception:
            pass
    return ImageFont.load_default()


# ─────────────────────────────────────────────
# COLOR HELPERS
# ─────────────────────────────────────────────

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hsv(r: float, g: float, b: float) -> tuple[float, float, float]:
    r, g, b = r/255, g/255, b/255
    mx, mn  = max(r, g, b), min(r, g, b)
    diff, v = mx - mn, mx
    s = diff / mx if mx else 0.0
    if diff == 0:
        h = 0.0
    elif mx == r:
        h = (60 * ((g - b) / diff)) % 360
    elif mx == g:
        h = 60 * ((b - r) / diff) + 120
    else:
        h = 60 * ((r - g) / diff) + 240
    return h, s, v


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    if s == 0:
        val = int(v * 255)
        return val, val, val
    h /= 60
    i = int(h)
    f = h - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    sectors = [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)]
    r, g, b = sectors[i % 6]
    return int(r*255), int(g*255), int(b*255)


# ─────────────────────────────────────────────
# SMART RECOLOR — HSV 5-tone, preserves luminance
# ─────────────────────────────────────────────

def smart_recolor_frame(
    frame: np.ndarray,
    target_hex: str,
    bg_hex: str,
) -> np.ndarray:
    tr, tg, tb    = hex_to_rgb(target_hex)
    br, bg_r, bb  = hex_to_rgb(bg_hex)
    t_h, t_s, _   = rgb_to_hsv(tr, tg, tb)

    if frame.shape[2] == 4:
        alpha = frame[:,:,3:4].astype(np.float32) / 255.0
        rgb   = frame[:,:,:3].astype(np.float32)
    else:
        alpha = np.ones((*frame.shape[:2], 1), dtype=np.float32)
        rgb   = frame[:,:,:3].astype(np.float32)

    lum = (0.2126*rgb[:,:,0] + 0.7152*rgb[:,:,1] + 0.0722*rgb[:,:,2]) / 255.0

    shades = [
        hsv_to_rgb(t_h, max(0.0, t_s - (1-lv)*0.3), lv)
        for lv in (0.15, 0.35, 0.55, 0.75, 0.95)
    ]

    out = np.zeros_like(rgb)
    n   = len(shades)
    for i, (sr, sg, sb) in enumerate(shades):
        mask = (lum >= i/n) & (lum < (i+1)/n)
        out[mask] = [sr, sg, sb]

    bg  = np.array([br, bg_r, bb], dtype=np.float32)
    res = out * alpha + bg * (1.0 - alpha)
    return np.clip(res, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# WATERMARK
# ─────────────────────────────────────────────

async def apply_watermark(
    img: Image.Image,
    text: str,
    font_name: str = "Montserrat",
    opacity: int = 178,
) -> Image.Image:
    if not text:
        return img
    font_path = await fetch_font(font_name)
    fsize     = max(12, img.width // 40)
    font      = get_pil_font(font_path, fsize)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    bbox    = draw.textbbox((0, 0), text, font=font)
    tw, th  = bbox[2]-bbox[0], bbox[3]-bbox[1]
    margin  = max(8, img.width // 80)
    x, y    = img.width - tw - margin, img.height - th - margin

    draw.text((x+2, y+2), text, font=font, fill=(0,   0,   0,   opacity//2))
    draw.text((x,   y),   text, font=font, fill=(255, 255, 255, opacity))

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


# ─────────────────────────────────────────────
# LOTTIE / TGS  — deep color + shape extraction
# ─────────────────────────────────────────────

def _k_to_rgb(k: Any) -> Optional[tuple[int, int, int]]:
    """
    Convert a Lottie color value to (R, G, B) 0-255.
    Handles:
      - Static:   k = [r, g, b, a]   (floats 0-1)
      - Animated: k = [{"s": [r,g,b,a], "t": …}, …]
    """
    if isinstance(k, list) and k:
        # Animated keyframes → take first keyframe value
        if isinstance(k[0], dict):
            for kf in k:
                sv = kf.get("s") or kf.get("e")
                if isinstance(sv, list) and len(sv) >= 3:
                    k = sv
                    break
            else:
                return None
        # Static [r, g, b, ...]
        if isinstance(k[0], (int, float)) and len(k) >= 3:
            return (
                min(255, int(abs(float(k[0])) * 255)),
                min(255, int(abs(float(k[1])) * 255)),
                min(255, int(abs(float(k[2])) * 255)),
            )
    return None


def _lottie_extract_colors(obj: Any) -> list[tuple[int, int, int]]:
    """
    Recursively walk a Lottie JSON tree collecting RGB colors from:
      fl  = fill         st  = stroke
      gf  = gradient fill   sc = solid color layer
    """
    colors: list[tuple[int, int, int]] = []

    if isinstance(obj, dict):
        ty = obj.get("ty")

        # ── Solid fill ────────────────────────
        if ty in ("fl", "st"):
            c = obj.get("c", {})
            k = c.get("k") if isinstance(c, dict) else c
            rgb = _k_to_rgb(k)
            if rgb:
                colors.append(rgb)

        # ── Solid color layer (sc) ────────────
        # Lottie solid layers store color as hex string in "sc"
        if "sc" in obj and isinstance(obj["sc"], str):
            try:
                rgb = hex_to_rgb(obj["sc"].lstrip("#").zfill(6))
                colors.append(rgb)
            except Exception:
                pass

        # ── Gradient fill / stroke ────────────
        if ty in ("gf", "gs"):
            g  = obj.get("g", {})
            np_ = g.get("p", 0)
            ks  = g.get("k", {})
            gk  = ks.get("k") if isinstance(ks, dict) else ks
            # Static gradient: flat list [pos, r, g, b, pos, r, g, b, ...]
            if isinstance(gk, list) and gk and isinstance(gk[0], (int, float)):
                step = 4
                for i in range(0, min(len(gk), np_ * step), step):
                    if i + 3 < len(gk):
                        colors.append((
                            min(255, int(abs(float(gk[i+1])) * 255)),
                            min(255, int(abs(float(gk[i+2])) * 255)),
                            min(255, int(abs(float(gk[i+3])) * 255)),
                        ))

        for v in obj.values():
            if isinstance(v, (dict, list)):
                colors.extend(_lottie_extract_colors(v))

    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                colors.extend(_lottie_extract_colors(item))

    return colors


# ── Lottie shape-path renderer (bezier → Pillow polygon) ──

def _lottie_get_shapes(layers: list[dict]) -> list[dict]:
    """Flatten all shape items from all layers."""
    shapes: list[dict] = []
    for layer in layers:
        lt = layer.get("ty")
        if lt == 4:  # shape layer
            for s in layer.get("shapes", []):
                shapes.append(s)
        _collect_shapes_recursive(layer.get("shapes", []), shapes)
    return shapes


def _collect_shapes_recursive(items: list, out: list) -> None:
    for item in items:
        if isinstance(item, dict):
            ty = item.get("ty")
            if ty in ("fl", "st", "sh", "rc", "el", "sr"):
                out.append(item)
            if ty == "gr":
                _collect_shapes_recursive(item.get("it", []), out)


def _lottie_bezier_to_points(
    vertices: list, in_tangents: list, out_tangents: list, n: int = 12
) -> list[tuple[float, float]]:
    """Approximate a Lottie bezier path as a flat polygon (n segments/curve)."""
    pts: list[tuple[float, float]] = []
    nv  = len(vertices)
    if nv < 2:
        return [(v[0], v[1]) for v in vertices]

    for i in range(nv):
        p0 = vertices[i]
        p3 = vertices[(i + 1) % nv]
        c1 = [p0[0] + out_tangents[i][0], p0[1] + out_tangents[i][1]]
        c2 = [p3[0] + in_tangents[(i + 1) % nv][0],
              p3[1] + in_tangents[(i + 1) % nv][1]]
        for j in range(n):
            t  = j / n
            mt = 1 - t
            x  = mt**3*p0[0] + 3*mt**2*t*c1[0] + 3*mt*t**2*c2[0] + t**3*p3[0]
            y  = mt**3*p0[1] + 3*mt**2*t*c1[1] + 3*mt*t**2*c2[1] + t**3*p3[1]
            pts.append((x, y))
    return pts


def _render_lottie_fallback_frame(
    w: int,
    h: int,
    frame_idx: int,
    total: int,
    palette: list[tuple[int, int, int]],
    lottie: dict,
) -> Image.Image:
    """
    Attempt to rasterise one Lottie frame without rlottie.
    Strategy:
      1. Try to render actual bezier paths from shape layers.
      2. Fall back to animated concentric rings in extracted palette.
    """
    img  = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    t    = frame_idx / max(1, total - 1)

    rendered_shapes = 0
    layers = lottie.get("layers", [])

    for layer in layers:
        if layer.get("ty") != 4:   # only shape layers
            continue
        # Collect fill color for this layer
        layer_colors = _lottie_extract_colors(layer)
        fill_color   = layer_colors[0] if layer_colors else (255, 255, 255)

        shapes = layer.get("shapes", [])
        for shape in shapes:
            ty = shape.get("ty")

            # ── Rectangle ────────────────────
            if ty == "rc":
                size_k = shape.get("s", {}).get("k", [w//2, h//2])
                pos_k  = shape.get("p", {}).get("k", [w//2, h//2])
                sw = size_k[0] if isinstance(size_k, list) else w//2
                sh = size_k[1] if isinstance(size_k, list) else h//2
                px = pos_k[0]  if isinstance(pos_k,  list) else w//2
                py = pos_k[1]  if isinstance(pos_k,  list) else h//2
                draw.rectangle(
                    [px-sw/2, py-sh/2, px+sw/2, py+sh/2],
                    fill=(*fill_color, 220),
                )
                rendered_shapes += 1

            # ── Ellipse ───────────────────────
            elif ty == "el":
                size_k = shape.get("s", {}).get("k", [w//2, h//2])
                pos_k  = shape.get("p", {}).get("k", [w//2, h//2])
                sw = size_k[0] if isinstance(size_k, list) else w//2
                sh = size_k[1] if isinstance(size_k, list) else h//2
                px = pos_k[0]  if isinstance(pos_k,  list) else w//2
                py = pos_k[1]  if isinstance(pos_k,  list) else h//2
                draw.ellipse(
                    [px-sw/2, py-sh/2, px+sw/2, py+sh/2],
                    fill=(*fill_color, 220),
                )
                rendered_shapes += 1

            # ── Bezier path ───────────────────
            elif ty == "sh":
                ks  = shape.get("ks", {})
                kv  = ks.get("k")
                if isinstance(kv, dict):
                    # Non-animated path
                    verts = kv.get("v", [])
                    ins   = kv.get("i", [[0,0]]*len(verts))
                    outs  = kv.get("o", [[0,0]]*len(verts))
                elif isinstance(kv, list) and kv and isinstance(kv[0], dict):
                    # Animated — pick closest keyframe
                    best = kv[0]
                    for kf in kv:
                        if isinstance(kf.get("t"), (int, float)):
                            if kf["t"] <= frame_idx:
                                best = kf
                    sv   = best.get("s", [best.get("v", {})])
                    path = sv[0] if isinstance(sv, list) and sv else {}
                    verts = path.get("v", [])
                    ins   = path.get("i", [[0,0]]*len(verts))
                    outs  = path.get("o", [[0,0]]*len(verts))
                else:
                    continue

                if len(verts) >= 2:
                    pts = _lottie_bezier_to_points(verts, ins, outs)
                    if len(pts) >= 3:
                        draw.polygon(pts, fill=(*fill_color, 220))
                        rendered_shapes += 1

            # ── Group — recurse ───────────────
            elif ty == "gr":
                sub_colors  = _lottie_extract_colors(shape)
                sub_fill    = sub_colors[0] if sub_colors else fill_color
                for sub in shape.get("it", []):
                    sty = sub.get("ty")
                    if sty == "el":
                        size_k = sub.get("s", {}).get("k", [w//2, h//2])
                        pos_k  = sub.get("p", {}).get("k", [w//2, h//2])
                        # Handle animated size/position
                        if isinstance(size_k, list) and size_k and isinstance(size_k[0], dict):
                            size_k = size_k[0].get("s", [w//2, h//2])
                        if isinstance(pos_k,  list) and pos_k  and isinstance(pos_k[0],  dict):
                            pos_k  = pos_k[0].get("s",  [w//2, h//2])
                        sw = size_k[0] if isinstance(size_k, list) and len(size_k)>0 else w//2
                        sh = size_k[1] if isinstance(size_k, list) and len(size_k)>1 else h//2
                        px = pos_k[0]  if isinstance(pos_k,  list) and len(pos_k)>0  else w//2
                        py = pos_k[1]  if isinstance(pos_k,  list) and len(pos_k)>1  else h//2
                        draw.ellipse(
                            [px-sw/2, py-sh/2, px+sw/2, py+sh/2],
                            fill=(*sub_fill, 220),
                        )
                        rendered_shapes += 1

    # ── If no shapes rendered — animated rings fallback ──────────────────
    if rendered_shapes == 0:
        n = max(1, len(palette))
        for i, col in enumerate(palette):
            outer = (n - i) / n
            inner = max(0.0, (n - i - 1) / n)
            pulse = 0.88 + 0.12 * abs(((t * 2 + i * 0.4) % 2.0) - 1.0)
            ro_x  = int(w / 2 * 0.85 * outer * pulse)
            ro_y  = int(h / 2 * 0.85 * outer * pulse)
            ri_x  = int(w / 2 * 0.85 * inner * pulse)
            ri_y  = int(h / 2 * 0.85 * inner * pulse)
            cx, cy = w // 2, h // 2
            a_val  = max(80, 240 - i * 40)
            draw.ellipse([cx-ro_x, cy-ro_y, cx+ro_x, cy+ro_y], fill=(*col, a_val))
            if i < n - 1 and ri_x > 2 and ri_y > 2:
                draw.ellipse([cx-ri_x, cy-ri_y, cx+ri_x, cy+ri_y], fill=(0, 0, 0, 0))

    return img


def _tgs_to_frames_fallback(data: bytes) -> tuple[list[Image.Image], int]:
    """
    Fallback TGS renderer (no rlottie):
    Decompress → parse Lottie JSON → extract real shapes + colors → render.
    """
    try:
        raw    = zlib.decompress(data, 16 + zlib.MAX_WBITS)
        lottie = json.loads(raw)
    except Exception as exc:
        logger.error("TGS decompress: %s", exc)
        # Return a single visible white frame so pipeline doesn't crash
        img = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        ImageDraw.Draw(img).ellipse([128, 128, 384, 384], fill=(255, 255, 255, 220))
        return [img], 30

    ip = int(lottie.get("ip", 0))
    op = int(lottie.get("op", 60))
    fr = float(lottie.get("fr", 30))
    w  = int(lottie.get("w", 512))
    h  = int(lottie.get("h", 512))
    n_frames = max(1, op - ip)

    # Extract palette (deduplicated, skip pure-black)
    raw_colors = _lottie_extract_colors(lottie)
    seen:    set[tuple[int,int,int]]  = set()
    palette: list[tuple[int,int,int]] = []
    for c in raw_colors:
        if c not in seen and max(c) > 10:   # ignore near-black
            seen.add(c)
            palette.append(c)
    if not palette:
        palette = [(255, 255, 255)]

    logger.info(
        "TGS fallback: %d frames @ %.0f fps  %dx%d  palette=%s",
        n_frames, fr, w, h, palette[:8],
    )

    frames = [
        _render_lottie_fallback_frame(w, h, i, n_frames, palette, lottie)
        for i in range(n_frames)
    ]
    return frames, int(fr)


# ── rlottie_python has priority ──────────────

try:
    import rlottie_python  # type: ignore

    def _tgs_to_frames(data: bytes) -> tuple[list[Image.Image], int]:
        with tempfile.NamedTemporaryFile(suffix=".tgs", delete=False) as f:
            f.write(data)
            tmp = f.name
        try:
            anim = rlottie_python.LottieAnimation.from_tgs(tmp)
            n    = anim.lottie_animation_get_totalframe()
            fps  = int(anim.lottie_animation_get_framerate())
            fw, fh = anim.lottie_animation_get_size()
            frames = []
            for i in range(n):
                buf = anim.lottie_animation_render(i, fw, fh)
                img = Image.frombuffer("RGBA", (fw, fh), bytes(buf), "raw", "RGBA")
                frames.append(img)
            return frames, fps
        finally:
            os.unlink(tmp)

    logger.info("rlottie_python: native TGS renderer active")

except ImportError:
    logger.warning("rlottie_python not found — using color-aware fallback")
    _tgs_to_frames = _tgs_to_frames_fallback  # type: ignore[assignment]


# ─────────────────────────────────────────────
# GIF PIPELINE
# ─────────────────────────────────────────────

def _resize_onto_canvas(
    img: Image.Image, w: int, h: int, bg: tuple[int, int, int]
) -> Image.Image:
    img.thumbnail((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), bg)
    ox = (w - img.width)  // 2
    oy = (h - img.height) // 2
    if img.mode == "RGBA":
        canvas.paste(img, (ox, oy), img)
    else:
        canvas.paste(img, (ox, oy))
    return canvas


def _frames_to_gif(
    frames: list[Image.Image],
    width: int,
    height: int,
    fps: int,
    bg_color: str,
    emoji_color: str,
) -> bytes:
    dur_ms  = max(1, int(1000 / fps))
    bg_rgb  = hex_to_rgb(bg_color)
    out: list[Image.Image] = []

    for frame in frames:
        if frame.mode != "RGBA":
            frame = frame.convert("RGBA")

        arr      = np.array(frame)
        recolored = smart_recolor_frame(arr, emoji_color, bg_color)

        base   = Image.new("RGB", (arr.shape[1], arr.shape[0]), bg_rgb)
        rc_img = Image.fromarray(recolored, "RGB")
        alpha  = Image.fromarray(arr[:,:,3], "L") if arr.shape[2] == 4 else None
        base.paste(rc_img, mask=alpha)

        final = _resize_onto_canvas(base, width, height, bg_rgb)
        out.append(final.convert("P", palette=Image.ADAPTIVE, dither=0))

    buf = io.BytesIO()
    if len(out) == 1:
        out[0].save(buf, format="GIF")
    else:
        out[0].save(
            buf, format="GIF", save_all=True,
            append_images=out[1:], loop=0,
            duration=dur_ms, optimize=True,
        )
    return buf.getvalue()


async def process_file(
    file_data: bytes,
    file_type: str,
    settings: dict[str, Any],
    watermark_font_path: Optional[Path] = None,
) -> bytes:
    width  = settings.get("width",  1920)
    height = settings.get("height", 530)
    fps    = settings.get("fps",    60)
    bg     = settings.get("bg_color",    "#FF5E3B")
    color  = settings.get("emoji_color", "#FFFFFF")
    wm_txt = settings.get("watermark_text",    "")
    wm_on  = settings.get("watermark_enabled", False)
    font   = settings.get("font_name", "Montserrat")

    frames:  list[Image.Image] = []
    src_fps: int               = fps

    if file_type == "tgs":
        frames, src_fps = _tgs_to_frames(file_data)

    elif file_type == "gif":
        with Image.open(io.BytesIO(file_data)) as gif:
            try:
                src_fps = int(1000 / gif.info.get("duration", 100))
            except ZeroDivisionError:
                src_fps = 10
            for i in range(getattr(gif, "n_frames", 1)):
                gif.seek(i)
                frames.append(gif.copy().convert("RGBA"))

    elif file_type in ("webm", "mp4"):
        try:
            import imageio  # type: ignore
            reader  = imageio.get_reader(io.BytesIO(file_data), format="ffmpeg")
            meta    = reader.get_meta_data()
            src_fps = int(meta.get("fps", fps))
            for frm in reader:
                frames.append(Image.fromarray(frm).convert("RGBA"))
            reader.close()
        except Exception:
            logger.warning("imageio not available — single blank frame for webm")
            frames = [Image.new("RGBA", (512, 512), (255, 255, 255, 200))]

    else:  # webp / png / jpg
        img    = Image.open(io.BytesIO(file_data))
        frames = [img.convert("RGBA")]

    # Sub-sample to target FPS
    if src_fps > 0 and src_fps != fps and len(frames) > 1:
        ratio   = src_fps / fps
        indices = [int(i * ratio) for i in range(int(len(frames) / ratio))]
        frames  = [frames[min(i, len(frames)-1)] for i in indices] or frames

    gif_bytes = await asyncio.get_event_loop().run_in_executor(
        None, _frames_to_gif, frames, width, height, fps, bg, color,
    )

    if wm_on and wm_txt:
        preview     = Image.open(io.BytesIO(gif_bytes))
        watermarked = await apply_watermark(preview, wm_txt, font)
        buf         = io.BytesIO()
        watermarked.save(buf, format="GIF")
        gif_bytes = buf.getvalue()

    return gif_bytes


# ─────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────

async def download_bytes(bot: Bot, file_id: str) -> bytes:
    f   = await bot.get_file(file_id)
    buf = io.BytesIO()
    await bot.download_file(f.file_path, buf)  # type: ignore[arg-type]
    return buf.getvalue()


async def get_sticker_set_files(
    bot: Bot, set_name: str
) -> list[tuple[bytes, str]]:
    try:
        ss = await bot.get_sticker_set(set_name)
    except Exception as exc:
        logger.error("get_sticker_set: %s", exc)
        return []
    results = []
    for sticker in ss.stickers[:5]:
        ftype = _sticker_file_type(sticker)
        data  = await download_bytes(bot, sticker.file_id)
        results.append((data, ftype))
    return results


def _sticker_file_type(sticker: Sticker) -> str:
    if sticker.is_animated:
        return "tgs"
    if sticker.is_video:
        return "webm"
    return "webp"


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────

def fmt_config(s: dict[str, Any]) -> str:
    wm     = "вкл" if s.get("watermark_enabled") else "выкл"
    wm_txt = s.get("watermark_text") or "—"
    return (
        "<blockquote>👇 <b>Конфигурация:</b></blockquote>\n\n"
        f"🖌 <b>Цвет фона:</b>   <code>{s['bg_color']}</code>\n"
        f"↔️ <b>Размер:</b>      {s['width']}×{s['height']}\n"
        f"🎞 <b>FPS:</b>         {s['fps']}\n"
        f"🎨 <b>Цвет emoji:</b>  <code>{s['emoji_color']}</code>\n"
        f"🔤 <b>Шрифт:</b>       {s['font_name']}\n"
        f"💧 <b>Watermark:</b>   {wm} · {wm_txt}"
    )


def kb_config(s: dict[str, Any]) -> InlineKeyboardMarkup:
    wm_lbl = "Watermark ✓" if s.get("watermark_enabled") else "Watermark"
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Цвет фона",  callback_data="set:bg_color"),
            InlineKeyboardButton(text="Размер",      callback_data="set:size"),
        ],
        [
            InlineKeyboardButton(text="FPS",         callback_data="set:fps"),
            InlineKeyboardButton(text="Цвет emoji",  callback_data="set:emoji_color"),
        ],
        [
            InlineKeyboardButton(text="Шрифт",       callback_data="set:font"),
            InlineKeyboardButton(text="Текст WM",    callback_data="set:wm_text"),
        ],
        [
            InlineKeyboardButton(text=wm_lbl,        callback_data="set:wm_toggle"),
        ],
        [
            InlineKeyboardButton(text="▶ Предпросмотр",   callback_data="action:preview"),
            InlineKeyboardButton(text="⚙ Конвертировать", callback_data="action:convert"),
        ],
    ])


def kb_start() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="⚙ Настройки",  callback_data="start:settings"),
            InlineKeyboardButton(text="❓ Помощь",     callback_data="start:help"),
        ],
        [
            InlineKeyboardButton(text="📊 Статистика", callback_data="start:stats"),
        ],
    ])


def kb_cancel() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✕ Отмена", callback_data="action:cancel_input")]
    ])


# ─────────────────────────────────────────────
# EDIT-OR-SEND  — единый способ обновить конфиг
# ─────────────────────────────────────────────

async def edit_or_send_config(
    bot: Bot,
    chat_id: int,
    msg_id: Optional[int],
    text: str,
    markup: InlineKeyboardMarkup,
) -> int:
    """
    Edit existing config message if possible; otherwise send a new one.
    Returns the message_id that is now showing the config.
    """
    if msg_id:
        try:
            await bot.edit_message_text(
                text,
                chat_id=chat_id,
                message_id=msg_id,
                reply_markup=markup,
                parse_mode="HTML",
            )
            return msg_id
        except Exception:
            pass  # message too old / deleted — fall through
    sent = await bot.send_message(
        chat_id, text, reply_markup=markup, parse_mode="HTML"
    )
    return sent.message_id


# ─────────────────────────────────────────────
# TASK QUEUE
# ─────────────────────────────────────────────

task_queue: asyncio.Queue = asyncio.Queue()


async def queue_worker(worker_id: int) -> None:
    logger.info("Queue worker #%d started", worker_id)
    while True:
        task = await task_queue.get()
        try:
            await _run_conversion(task)
        except Exception as exc:
            logger.exception("Worker #%d: %s", worker_id, exc)
            try:
                await task["bot"].send_message(task["user_id"], "❌ Ошибка конвертации.")
            except Exception:
                pass
        finally:
            task_queue.task_done()


async def _run_conversion(task: dict[str, Any]) -> None:
    bot: Bot         = task["bot"]
    uid: int         = task["user_id"]
    file_data: bytes = task["file_data"]
    file_type: str   = task["file_type"]
    settings: dict   = task["settings"]

    status = await bot.send_message(uid, "⚙️ Обрабатываю…")

    font_path = await fetch_font(settings.get("font_name", "Montserrat"))
    gif_bytes = await process_file(file_data, file_type, settings, font_path)

    try:
        await bot.delete_message(uid, status.message_id)
    except Exception:
        pass

    fname = f"result_{uid}_{int(time.time())}.gif"
    await bot.send_document(
        uid,
        BufferedInputFile(gif_bytes, filename=fname),
        caption="✅ Готово",
    )
    db_log_conversion(uid, "done")


# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────

router = Router()


# ── /start ──────────────────────────────────

@router.message(CommandStart())
async def cmd_start(msg: Message, state: FSMContext) -> None:
    await state.set_state(ConvState.idle)
    await state.update_data(config_msg_id=None)
    await msg.answer(
        "<blockquote>👇 <b>Отправь стикер, emoji или ссылку на пак.</b></blockquote>\n\n"
        "Поддерживаются: WEBP · GIF · TGS · WEBM, "
        "premium emoji, ссылки <code>t.me/addstickers/…</code>, ID emoji.",
        reply_markup=kb_start(),
        parse_mode="HTML",
    )


@router.callback_query(F.data.startswith("start:"))
async def cb_start_menu(call: CallbackQuery, state: FSMContext) -> None:
    key = call.data.split(":", 1)[1]  # type: ignore[union-attr]
    uid = call.from_user.id

    if key == "settings":
        s       = db_get_settings(uid)
        sd      = await state.get_data()
        new_mid = await edit_or_send_config(
            call.bot, call.message.chat.id,  # type: ignore[union-attr]
            sd.get("config_msg_id"),
            fmt_config(s), kb_config(s),
        )
        await state.update_data(config_msg_id=new_mid)
        await state.set_state(ConvState.idle)

    elif key == "help":
        await call.message.answer(  # type: ignore[union-attr]
            "📖 <b>Как пользоваться:</b>\n\n"
            "1. Пришли стикер / документ / emoji ID\n"
            "2. Настрой параметры в меню конфига\n"
            "3. Нажми <b>Предпросмотр</b> или <b>Конвертировать</b>\n\n"
            "<b>Вход:</b> WEBP · GIF · TGS · WEBM · PNG\n"
            "<b>Выход:</b> GIF\n\n"
            "<b>Premium emoji по ID:</b>\n"
            "Отправь числовой ID (15–19 цифр)",
            parse_mode="HTML",
        )

    elif key == "stats":
        with db_connect() as conn:
            users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            total = conn.execute("SELECT COUNT(*) FROM conversions").fetchone()[0]
        await call.message.answer(  # type: ignore[union-attr]
            f"📊 Пользователей: <b>{users}</b>\n"
            f"Конвертаций: <b>{total}</b>\n"
            f"Очередь: <b>{task_queue.qsize()}</b>",
            parse_mode="HTML",
        )

    await call.answer()


# ── STICKER ──────────────────────────────────

@router.message(F.sticker)
async def on_sticker(msg: Message, state: FSMContext) -> None:
    sticker = msg.sticker  # type: ignore[union-attr]
    ftype   = _sticker_file_type(sticker)
    data    = await download_bytes(msg.bot, sticker.file_id)  # type: ignore[arg-type]

    uid = msg.from_user.id  # type: ignore[union-attr]
    s   = db_get_settings(uid)
    sd  = await state.get_data()

    new_mid = await edit_or_send_config(
        msg.bot, msg.chat.id, sd.get("config_msg_id"),  # type: ignore[arg-type]
        fmt_config(s), kb_config(s),
    )
    await state.update_data(file_data=data, file_type=ftype, config_msg_id=new_mid)
    await state.set_state(ConvState.idle)


# ── DOCUMENT ─────────────────────────────────

@router.message(F.document)
async def on_document(msg: Message, state: FSMContext) -> None:
    doc  = msg.document  # type: ignore[union-attr]
    mime = doc.mime_type or ""
    name = (doc.file_name or "").lower()

    if   "gif"  in mime or name.endswith(".gif"):  ftype = "gif"
    elif "webm" in mime or name.endswith(".webm"): ftype = "webm"
    elif "webp" in mime or name.endswith(".webp"): ftype = "webp"
    elif "png"  in mime or name.endswith(".png"):  ftype = "webp"
    elif name.endswith(".tgs"):                     ftype = "tgs"
    else:
        await msg.answer("❌ Неподдерживаемый формат. GIF / WEBP / PNG / TGS / WEBM.")
        return

    data = await download_bytes(msg.bot, doc.file_id)  # type: ignore[arg-type]
    uid  = msg.from_user.id  # type: ignore[union-attr]
    s    = db_get_settings(uid)
    sd   = await state.get_data()

    new_mid = await edit_or_send_config(
        msg.bot, msg.chat.id, sd.get("config_msg_id"),  # type: ignore[arg-type]
        fmt_config(s), kb_config(s),
    )
    await state.update_data(file_data=data, file_type=ftype, config_msg_id=new_mid)
    await state.set_state(ConvState.idle)


# ── TEXT ─────────────────────────────────────

@router.message(F.text)
async def on_text(msg: Message, state: FSMContext) -> None:
    text      = (msg.text or "").strip()
    uid       = msg.from_user.id  # type: ignore[union-attr]
    cur_state = await state.get_state()

    # ── collecting a value ───────────────────
    if cur_state == ConvState.waiting_val:
        sd = await state.get_data()
        # Clean up prompt message
        if p := sd.get("prompt_msg_id"):
            try:
                await msg.bot.delete_message(msg.chat.id, p)  # type: ignore[union-attr]
            except Exception:
                pass
        try:
            await msg.delete()
        except Exception:
            pass
        await _apply_setting(msg, state, sd.get("editing_setting"), text)
        return

    # ── sticker pack link ────────────────────
    pack_m = re.search(r"t\.me/addstickers/([A-Za-z0-9_]+)", text)
    if pack_m:
        await msg.answer("🔍 Скачиваю набор…")
        files = await get_sticker_set_files(msg.bot, pack_m.group(1))  # type: ignore[arg-type]
        if not files:
            await msg.answer("❌ Не удалось загрузить набор.")
            return
        fd, ft = files[0]
        s  = db_get_settings(uid)
        sd = await state.get_data()
        new_mid = await edit_or_send_config(
            msg.bot, msg.chat.id, sd.get("config_msg_id"),  # type: ignore[arg-type]
            fmt_config(s), kb_config(s),
        )
        await state.update_data(file_data=fd, file_type=ft, config_msg_id=new_mid)
        await state.set_state(ConvState.idle)
        return

    # ── premium emoji HTML entity ─────────────
    ids = re.findall(r'emoji-id="(\d+)"', text)
    if ids:
        await _handle_emoji_id(msg, state, ids[0])
        return

    # ── raw numeric ID ────────────────────────
    if re.fullmatch(r"\d{15,20}", text):
        await _handle_emoji_id(msg, state, text)
        return

    await msg.answer(
        "Отправь стикер, ссылку на пак (<code>t.me/addstickers/NAME</code>) "
        "или ID premium emoji.",
        parse_mode="HTML",
    )


async def _handle_emoji_id(msg: Message, state: FSMContext, eid: str) -> None:
    uid = msg.from_user.id  # type: ignore[union-attr]
    try:
        stickers = await msg.bot.get_custom_emoji_stickers([eid])  # type: ignore[union-attr]
        if not stickers:
            await msg.answer("❌ Emoji не найден.")
            return
        sticker = stickers[0]
        ftype   = _sticker_file_type(sticker)
        data    = await download_bytes(msg.bot, sticker.file_id)  # type: ignore[arg-type]
        s       = db_get_settings(uid)
        sd      = await state.get_data()
        new_mid = await edit_or_send_config(
            msg.bot, msg.chat.id, sd.get("config_msg_id"),  # type: ignore[arg-type]
            fmt_config(s), kb_config(s),
        )
        await state.update_data(file_data=data, file_type=ftype, config_msg_id=new_mid)
        await state.set_state(ConvState.idle)
    except Exception as exc:
        logger.error("Emoji ID %s: %s", eid, exc)
        await msg.answer("❌ Не удалось загрузить premium emoji.")


# ── SETTINGS CALLBACKS ───────────────────────

SETTING_PROMPTS: dict[str, tuple[str, str]] = {
    "bg_color":    ("🖌 Цвет фона (HEX, напр. <code>#FF5E3B</code>):", "bg_color"),
    "size":        ("↔️ Размер (напр. <code>1920x530</code>):",        "size"),
    "fps":         ("🎞 FPS (1–120):",                                  "fps"),
    "emoji_color": ("🎨 Цвет emoji (HEX):",                             "emoji_color"),
    "font":        (
        "🔤 Шрифт:\n<code>Montserrat · Roboto · Lato · Oswald · OpenSans</code>",
        "font_name",
    ),
    "wm_text":     ("💧 Текст watermark:", "watermark_text"),
}


@router.callback_query(F.data.startswith("set:"))
async def cb_set(call: CallbackQuery, state: FSMContext) -> None:
    key = call.data.split(":", 1)[1]  # type: ignore[union-attr]
    uid = call.from_user.id

    if key == "wm_toggle":
        s = db_get_settings(uid)
        s["watermark_enabled"] = not s.get("watermark_enabled", False)
        db_save_settings(uid, s)
        sd  = await state.get_data()
        new_mid = await edit_or_send_config(
            call.bot, call.message.chat.id,  # type: ignore[union-attr]
            sd.get("config_msg_id"),
            fmt_config(s), kb_config(s),
        )
        await state.update_data(config_msg_id=new_mid)
        await call.answer()
        return

    if key in SETTING_PROMPTS:
        prompt_text, setting_key = SETTING_PROMPTS[key]
        pm = await call.message.answer(  # type: ignore[union-attr]
            prompt_text, reply_markup=kb_cancel(), parse_mode="HTML",
        )
        await state.update_data(editing_setting=setting_key, prompt_msg_id=pm.message_id)
        await state.set_state(ConvState.waiting_val)
        await call.answer()


@router.callback_query(F.data == "action:cancel_input")
async def cb_cancel_input(call: CallbackQuery, state: FSMContext) -> None:
    sd = await state.get_data()
    try:
        await call.message.delete()  # type: ignore[union-attr]
    except Exception:
        pass
    await state.set_state(ConvState.idle)
    uid = call.from_user.id
    s   = db_get_settings(uid)
    new_mid = await edit_or_send_config(
        call.bot, call.message.chat.id,  # type: ignore[union-attr]
        sd.get("config_msg_id"),
        fmt_config(s), kb_config(s),
    )
    await state.update_data(config_msg_id=new_mid)
    await call.answer()


async def _apply_setting(
    msg: Message,
    state: FSMContext,
    setting: Optional[str],
    value: str,
) -> None:
    uid = msg.from_user.id  # type: ignore[union-attr]
    s   = db_get_settings(uid)
    err: Optional[str] = None

    if setting == "size":
        m = re.match(r"(\d+)\s*[xX×]\s*(\d+)", value)
        if m:
            s["width"], s["height"] = int(m.group(1)), int(m.group(2))
        else:
            err = "❌ Формат: <code>1920x530</code>"

    elif setting == "fps":
        try:
            s["fps"] = max(1, min(120, int(value)))
        except ValueError:
            err = "❌ Введи число 1–120"

    elif setting in ("bg_color", "emoji_color"):
        if re.match(r"^#?[0-9A-Fa-f]{3,6}$", value):
            s[setting] = value if value.startswith("#") else f"#{value}"
        else:
            err = "❌ Неверный HEX (пример: <code>#FF5E3B</code>)"

    elif setting == "font_name":
        allowed = list(FONT_TTF_MAP.keys())
        if value in allowed:
            s["font_name"] = value
        else:
            err = f"❌ Доступные шрифты: <code>{' · '.join(allowed)}</code>"

    elif setting == "watermark_text":
        s["watermark_text"] = value[:100]

    if err:
        em = await msg.answer(err, parse_mode="HTML")
        async def _autodel():
            await asyncio.sleep(4)
            try:
                await em.delete()
            except Exception:
                pass
        asyncio.create_task(_autodel())
        await state.set_state(ConvState.idle)
        return

    db_save_settings(uid, s)
    await state.set_state(ConvState.idle)
    sd  = await state.get_data()
    new_mid = await edit_or_send_config(
        msg.bot, msg.chat.id,  # type: ignore[arg-type]
        sd.get("config_msg_id"),
        fmt_config(s), kb_config(s),
    )
    await state.update_data(config_msg_id=new_mid)


# ── PREVIEW ──────────────────────────────────

@router.callback_query(F.data == "action:preview")
async def cb_preview(call: CallbackQuery, state: FSMContext) -> None:
    uid = call.from_user.id
    sd  = await state.get_data()

    if "file_data" not in sd:
        await call.answer("Сначала отправь файл", show_alert=True)
        return

    await call.answer("Генерирую…")
    s         = db_get_settings(uid)
    prev_s    = {**s, "width": 320, "height": 90, "fps": 12}
    font_path = await fetch_font(s.get("font_name", "Montserrat"))

    status = await call.message.answer("⚙️ Превью…")  # type: ignore[union-attr]
    try:
        gif_bytes = await process_file(sd["file_data"], sd["file_type"], prev_s, font_path)
        try:
            await status.delete()
        except Exception:
            pass
        await call.message.answer_document(  # type: ignore[union-attr]
            BufferedInputFile(gif_bytes, "preview.gif"),
            caption="🖼 Превью (320×90, 12 FPS)",
        )
    except Exception as exc:
        logger.exception("Preview: %s", exc)
        await status.edit_text("❌ Ошибка генерации превью")


# ── CONVERT ───────────────────────────────────

@router.callback_query(F.data == "action:convert")
async def cb_convert(call: CallbackQuery, state: FSMContext) -> None:
    uid = call.from_user.id
    sd  = await state.get_data()

    if "file_data" not in sd:
        await call.answer("Сначала отправь файл", show_alert=True)
        return

    s = db_get_settings(uid)
    db_log_conversion(uid)

    await task_queue.put({
        "bot":       call.bot,
        "user_id":   uid,
        "file_data": sd["file_data"],
        "file_type": sd["file_type"],
        "settings":  s,
    })

    qsize = task_queue.qsize()
    await call.answer(f"Добавлено #{qsize}")
    await call.message.answer(  # type: ignore[union-attr]
        f"✅ Файл добавлен в очередь. Позиция: #{qsize}"
    )


# ── /stats (admin) ───────────────────────────

@router.message(Command("stats"))
async def cmd_stats(msg: Message) -> None:
    uid = msg.from_user.id  # type: ignore[union-attr]
    if ADMIN_IDS and uid not in ADMIN_IDS:
        return
    with db_connect() as conn:
        users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM conversions").fetchone()[0]
        done  = conn.execute(
            "SELECT COUNT(*) FROM conversions WHERE status='done'"
        ).fetchone()[0]
    await msg.answer(
        f"📊 <b>Статистика</b>\n"
        f"Пользователей: {users}\n"
        f"Конвертаций:   {total}  (выполнено: {done})\n"
        f"Очередь:       {task_queue.qsize()}",
        parse_mode="HTML",
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

PID_FILE = Path(os.getenv("PID_FILE", "bot.pid"))


def _acquire_pid_lock() -> None:
    """
    Prevent two bot instances from running simultaneously.
    Uses a PID file + flock (Linux/macOS).
    On Windows falls back to checking the PID file manually.
    """
    import fcntl  # noqa: F401 — available on Linux/macOS

    try:
        _acquire_pid_lock._fd = open(PID_FILE, "w")          # type: ignore[attr-defined]
        fcntl.flock(_acquire_pid_lock._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined]
        _acquire_pid_lock._fd.write(str(os.getpid()))         # type: ignore[attr-defined]
        _acquire_pid_lock._fd.flush()                         # type: ignore[attr-defined]
        logger.info("PID lock acquired (%s)", PID_FILE)
    except (ImportError, AttributeError):
        # Windows — simple PID file check
        if PID_FILE.exists():
            try:
                old_pid = int(PID_FILE.read_text().strip())
                # Check if process is alive
                os.kill(old_pid, 0)
                logger.error(
                    "Another bot instance is already running (PID %d). "
                    "Kill it first: kill %d",
                    old_pid, old_pid,
                )
                sys.exit(1)
            except (ProcessLookupError, ValueError):
                pass  # Stale PID file — overwrite
        PID_FILE.write_text(str(os.getpid()))
    except BlockingIOError:
        logger.error(
            "Another bot instance is already running!\n"
            "  Stop it first:  pkill -f bot.py\n"
            "  Or remove:       rm %s",
            PID_FILE,
        )
        sys.exit(1)


async def main() -> None:
    _acquire_pid_lock()
    db_init()

    bot = Bot(token=BOT_TOKEN)
    dp  = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    for i in range(QUEUE_WORKERS):
        asyncio.create_task(queue_worker(i + 1))

    logger.info("Bot started (workers=%d)", QUEUE_WORKERS)
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        try:
            PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
