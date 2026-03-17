"""
Telegram Sticker Colorizer Bot
Production-level aiogram 3.x bot
"""

import asyncio
import gzip
import io
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Optional

import aiohttp
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence
from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
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

load_dotenv()

# ─── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("bot")

# ─── Config ──────────────────────────────────────────────────────────────────

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
FONTS_DIR = "fonts"
DB_PATH = "users.db"
os.makedirs(FONTS_DIR, exist_ok=True)

# ─── Database ────────────────────────────────────────────────────────────────


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            settings_json TEXT NOT NULL DEFAULT '{}'
        )"""
    )
    conn.commit()
    return conn


def db_save_settings(user_id: int, settings: dict) -> None:
    with db_connect() as conn:
        conn.execute(
            "INSERT INTO users(user_id, settings_json) VALUES(?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET settings_json=excluded.settings_json",
            (user_id, json.dumps(settings)),
        )


def db_load_settings(user_id: int) -> Optional[dict]:
    with db_connect() as conn:
        row = conn.execute(
            "SELECT settings_json FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
    return json.loads(row[0]) if row else None


# ─── User Settings ───────────────────────────────────────────────────────────


@dataclass
class UserSettings:
    bg_color: str = "#FF5E3B"
    size: int = 512
    output_format: str = "GIF"
    tint_color: str = "#FFFFFF"
    watermark_enabled: bool = True
    watermark_font: str = "Montserrat"
    watermark_text: str = "example"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UserSettings":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


DEFAULT_SETTINGS = UserSettings()


def get_settings(data: dict) -> UserSettings:
    raw = data.get("settings")
    if raw:
        return UserSettings.from_dict(raw)
    return UserSettings()


# ─── FSM ─────────────────────────────────────────────────────────────────────


class BotStates(StatesGroup):
    idle = State()
    waiting_color = State()
    waiting_bg_color = State()
    waiting_size = State()
    waiting_watermark_text = State()
    waiting_watermark_font = State()


# ─── Keyboard builder ────────────────────────────────────────────────────────


def main_keyboard(s: UserSettings) -> InlineKeyboardMarkup:
    wm_label = "✓ Watermark" if s.watermark_enabled else "Watermark"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Цвет фона", callback_data="set_bg_color"),
                InlineKeyboardButton(text="Цвет тинта", callback_data="set_tint"),
            ],
            [
                InlineKeyboardButton(text="Размер", callback_data="set_size"),
                InlineKeyboardButton(text="Формат", callback_data="set_format"),
            ],
            [
                InlineKeyboardButton(text="Шрифт", callback_data="set_font"),
                InlineKeyboardButton(text="Текст", callback_data="set_wm_text"),
            ],
            [
                InlineKeyboardButton(text=wm_label, callback_data="toggle_wm"),
                InlineKeyboardButton(text="Предпросмотр", callback_data="preview"),
            ],
            [
                InlineKeyboardButton(text="▶ Конвертировать", callback_data="convert"),
            ],
        ]
    )


def format_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="WebP", callback_data="fmt_WEBP"),
                InlineKeyboardButton(text="PNG", callback_data="fmt_PNG"),
                InlineKeyboardButton(text="GIF", callback_data="fmt_GIF"),
            ],
            [InlineKeyboardButton(text="← Назад", callback_data="back")],
        ]
    )


def size_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="256", callback_data="size_256"),
                InlineKeyboardButton(text="512", callback_data="size_512"),
                InlineKeyboardButton(text="1024", callback_data="size_1024"),
            ],
            [InlineKeyboardButton(text="← Назад", callback_data="back")],
        ]
    )


def config_text(s: UserSettings) -> str:
    return (
        f"<b>Конфигурация</b>\n\n"
        f"Цвет фона: <code>{s.bg_color}</code>\n"
        f"Цвет тинта: <code>{s.tint_color}</code>\n"
        f"Размер: <code>{s.size}×{s.size}</code>\n"
        f"Формат: <code>{s.output_format}</code>\n"
        f"Watermark: <code>{'ON' if s.watermark_enabled else 'OFF'}</code>\n"
        f"Шрифт: <code>{s.watermark_font}</code>\n"
        f"Текст: <code>{s.watermark_text}</code>\n\n"
        f"Отправьте стикер или файл для обработки."
    )


# ─── Color Processing ────────────────────────────────────────────────────────


def hex_to_hue(hex_color: str) -> float:
    """Convert hex color to HSV hue (0–360)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    if delta == 0:
        return 0.0
    if cmax == r:
        hue = 60 * (((g - b) / delta) % 6)
    elif cmax == g:
        hue = 60 * (((b - r) / delta) + 2)
    else:
        hue = 60 * (((r - g) / delta) + 4)
    return hue % 360


def apply_monochrome_palette(
    img: Image.Image, tint_hue: float, bg_hex: str
) -> Image.Image:
    """
    Convert image to monochrome palette preserving lightness.

    Pipeline:
    1. Composite on background color
    2. Convert to float HSV via numpy
    3. Replace hue channel with tint_hue
    4. Desaturate shadows / boost highlights → quality palette
    5. Return RGBA image
    """
    # Ensure RGBA
    img = img.convert("RGBA")
    size = img.size

    # Parse background
    bg_hex = bg_hex.lstrip("#")
    bg_r = int(bg_hex[0:2], 16)
    bg_g = int(bg_hex[2:4], 16)
    bg_b = int(bg_hex[4:6], 16)

    # Composite on background
    background = Image.new("RGBA", size, (bg_r, bg_g, bg_b, 255))
    composited = Image.alpha_composite(background, img)
    rgb = composited.convert("RGB")

    arr = np.array(rgb, dtype=np.float32) / 255.0  # (H, W, 3)

    # Convert RGB → HSV manually (numpy, fast)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Value
    v = cmax

    # Saturation
    s = np.where(cmax == 0, 0.0, delta / cmax)

    # Hue (original, for reference — we'll replace it)
    h = np.zeros_like(r)
    mask_r = (cmax == r) & (delta != 0)
    mask_g = (cmax == g) & (delta != 0)
    mask_b = (cmax == b) & (delta != 0)
    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r])) % 360
    h[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120
    h[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240

    # ── Monochrome palette magic ──
    # Replace hue with tint hue
    new_h = np.full_like(h, tint_hue / 360.0 * 360.0)  # keep degrees

    # Modulate saturation based on brightness:
    # - shadows (dark areas): lower saturation → deeper, richer shadow
    # - midtones: full saturation of tint
    # - highlights (bright areas): reduced saturation → clean, airy highlights
    shadow_mask = v < 0.3
    highlight_mask = v > 0.75
    new_s = s.copy()
    new_s[shadow_mask] = s[shadow_mask] * 0.5 + 0.3   # shadows: moderate sat
    new_s[highlight_mask] = s[highlight_mask] * 0.4    # highlights: desaturated

    # Clamp saturation to [0, 1]
    new_s = np.clip(new_s, 0, 1)

    # Convert HSV → RGB
    new_h_norm = new_h / 360.0
    hi = (new_h_norm * 6).astype(np.int32)
    f_val = new_h_norm * 6 - hi
    p = v * (1 - new_s)
    q = v * (1 - f_val * new_s)
    t = v * (1 - (1 - f_val) * new_s)

    out_r = np.zeros_like(v)
    out_g = np.zeros_like(v)
    out_b = np.zeros_like(v)

    for i, (cr, cg, cb) in enumerate(
        [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    ):
        mask = hi % 6 == i
        out_r[mask] = cr[mask]
        out_g[mask] = cg[mask]
        out_b[mask] = cb[mask]

    out = np.stack([out_r, out_g, out_b], axis=2)
    out = np.clip(out * 255, 0, 255).astype(np.uint8)

    result = Image.fromarray(out, "RGB")

    # Re-apply original alpha
    alpha = img.split()[3]
    result.putalpha(alpha)

    return result


# ─── Watermark ───────────────────────────────────────────────────────────────

GOOGLE_FONTS_CACHE: dict[str, str] = {}


async def download_font(font_name: str) -> str:
    """
    Download font from Google Fonts GitHub mirror.
    Returns path to .ttf file.
    """
    safe_name = font_name.strip()
    slug = safe_name.replace(" ", "")
    path = os.path.join(FONTS_DIR, f"{slug}.ttf")

    if os.path.exists(path):
        log.info(f"Font cache hit: {path}")
        return path

    # Try Google Fonts API v2
    api_url = (
        f"https://fonts.googleapis.com/css2?family={safe_name.replace(' ', '+')}"
        f":wght@400&display=swap"
    )
    ttf_url: Optional[str] = None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 Chrome/120 Safari/537.36"
        )
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        try:
            async with session.get(api_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                css = await resp.text()
            # Extract .ttf / .woff2 URL from CSS
            urls = re.findall(r"src:\s*url\(([^)]+)\)", css)
            for u in urls:
                if u.endswith(".ttf") or "ttf" in u:
                    ttf_url = u
                    break
            if not ttf_url and urls:
                ttf_url = urls[0]
        except Exception as e:
            log.warning(f"Google Fonts CSS fetch failed: {e}")

        # Fallback: raw GitHub
        if not ttf_url:
            gh_url = (
                "https://raw.githubusercontent.com/google/fonts/main/ofl/"
                f"{slug.lower()}/{slug}-Regular.ttf"
            )
            try:
                async with session.get(gh_url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        ttf_url = gh_url
            except Exception as e:
                log.warning(f"GitHub font fallback failed: {e}")

        if ttf_url:
            try:
                async with session.get(
                    ttf_url, timeout=aiohttp.ClientTimeout(total=20)
                ) as resp:
                    data = await resp.read()
                with open(path, "wb") as f:
                    f.write(data)
                log.info(f"Font downloaded: {path}")
                return path
            except Exception as e:
                log.warning(f"Font download failed: {e}")

    # Return default PIL font path (no TTF)
    return ""


def apply_watermark(
    img: Image.Image, text: str, font_path: str, opacity: int = 180
) -> Image.Image:
    """
    Add watermark to bottom-right corner.
    Supports Cyrillic via TTF font.
    """
    img = img.convert("RGBA")
    w, h = img.size

    # Font size: ~7% of width
    font_size = max(12, int(w * 0.07))

    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Create watermark layer
    wm_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(wm_layer)

    # Measure text
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    padding = max(10, int(w * 0.03))
    x = w - tw - padding
    y = h - th - padding

    # Shadow for readability
    draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, opacity // 2))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, opacity))

    composited = Image.alpha_composite(img, wm_layer)
    return composited


# ─── Image processing ────────────────────────────────────────────────────────


def process_static_image(
    data: bytes,
    tint_hue: float,
    bg_hex: str,
    size: int,
    output_format: str,
    wm_text: Optional[str],
    font_path: str,
) -> bytes:
    """Process WEBP/PNG → monochrome palette."""
    img = Image.open(io.BytesIO(data)).convert("RGBA")
    img = img.resize((size, size), Image.LANCZOS)
    img = apply_monochrome_palette(img, tint_hue, bg_hex)

    if wm_text:
        img = apply_watermark(img, wm_text, font_path)

    buf = io.BytesIO()
    fmt = output_format.upper()
    if fmt == "GIF":
        fmt = "PNG"  # single frame → PNG
    img.save(buf, format=fmt)
    return buf.getvalue()


def process_gif(
    data: bytes,
    tint_hue: float,
    bg_hex: str,
    size: int,
    wm_text: Optional[str],
    font_path: str,
) -> bytes:
    """Process GIF frame-by-frame."""
    src = Image.open(io.BytesIO(data))
    frames = []
    durations = []

    for frame in ImageSequence.Iterator(src):
        duration = frame.info.get("duration", 50)
        f = frame.convert("RGBA").resize((size, size), Image.LANCZOS)
        f = apply_monochrome_palette(f, tint_hue, bg_hex)
        if wm_text:
            f = apply_watermark(f, wm_text, font_path)
        frames.append(f.convert("RGBA"))
        durations.append(duration)

    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=durations,
        disposal=2,
    )
    return buf.getvalue()


def process_tgs(
    data: bytes,
    tint_hue: float,
    wm_requested: bool,
) -> tuple[bytes, str]:
    """
    Process TGS (Lottie JSON in gzip).
    Returns (processed_bytes, warning_message).
    """
    warning = ""
    if wm_requested:
        warning = "Watermark не поддерживается для TGS."

    try:
        json_bytes = gzip.decompress(data)
        lottie = json.loads(json_bytes)
    except Exception as e:
        raise ValueError(f"Не удалось распаковать TGS: {e}")

    # Convert hue to normalized (0-1) for Lottie
    hue_norm = tint_hue / 360.0

    def shift_color(c: list[float]) -> list[float]:
        """Shift Lottie color array [r,g,b,a] preserving lightness."""
        if len(c) < 3:
            return c
        r, g, b = c[0], c[1], c[2]
        a = c[3] if len(c) > 3 else 1.0

        cmax = max(r, g, b)
        cmin = min(r, g, b)
        delta = cmax - cmin

        v = cmax
        s = 0.0 if cmax == 0 else delta / cmax

        # New HSV
        new_h = hue_norm
        new_s = min(s * 0.9 + 0.1, 1.0)
        new_v = v

        # HSV → RGB
        if new_s == 0:
            return [new_v, new_v, new_v, a]
        hi = int(new_h * 6) % 6
        f2 = new_h * 6 - int(new_h * 6)
        p = new_v * (1 - new_s)
        q = new_v * (1 - f2 * new_s)
        t = new_v * (1 - (1 - f2) * new_s)
        palette = [
            (new_v, t, p),
            (q, new_v, p),
            (p, new_v, t),
            (p, q, new_v),
            (t, p, new_v),
            (new_v, p, q),
        ]
        nr, ng, nb = palette[hi]
        return [nr, ng, nb, a]

    def traverse(node):
        if isinstance(node, dict):
            # Lottie color arrays appear as "c" keys with "k" values
            if "c" in node and isinstance(node["c"], dict) and "k" in node["c"]:
                k = node["c"]["k"]
                if isinstance(k, list) and len(k) >= 3 and isinstance(k[0], (int, float)):
                    node["c"]["k"] = shift_color(k)
            for v in node.values():
                traverse(v)
        elif isinstance(node, list):
            for item in node:
                traverse(item)

    traverse(lottie)

    result_json = json.dumps(lottie, separators=(",", ":")).encode()
    result = gzip.compress(result_json)
    return result, warning


# ─── Font download helper ────────────────────────────────────────────────────


async def ensure_font(font_name: str) -> str:
    if font_name in GOOGLE_FONTS_CACHE:
        return GOOGLE_FONTS_CACHE[font_name]
    path = await download_font(font_name)
    if path:
        GOOGLE_FONTS_CACHE[font_name] = path
    return path


# ─── Task Queue ──────────────────────────────────────────────────────────────


@dataclass
class ConvertTask:
    user_id: int
    chat_id: int
    message_id: int
    file_bytes: bytes
    file_type: str  # "webp", "png", "gif", "tgs", "webm"
    settings: UserSettings
    bot: "Bot" = field(repr=False, compare=False)


task_queue: asyncio.Queue[ConvertTask] = asyncio.Queue()


async def worker(worker_id: int):
    log.info(f"Worker {worker_id} started")
    while True:
        task: ConvertTask = await task_queue.get()
        log.info(f"Worker {worker_id} processing task for user {task.user_id}")
        try:
            await process_task(task)
        except Exception as e:
            log.exception(f"Worker {worker_id} error: {e}")
            try:
                await task.bot.send_message(
                    task.chat_id, f"Ошибка при обработке: {e}"
                )
            except Exception:
                pass
        finally:
            task_queue.task_done()


async def process_task(task: ConvertTask):
    s = task.settings
    tint_hue = hex_to_hue(s.tint_color)
    font_path = ""

    if s.watermark_enabled:
        font_path = await ensure_font(s.watermark_font)

    wm_text = s.watermark_text if s.watermark_enabled else None
    ft = task.file_type.lower()

    if ft in ("webp", "png"):
        result = process_static_image(
            task.file_bytes,
            tint_hue,
            s.bg_color,
            s.size,
            s.output_format,
            wm_text,
            font_path,
        )
        ext = s.output_format.lower()
        filename = f"sticker.{ext}"
        await task.bot.send_document(
            task.chat_id,
            BufferedInputFile(result, filename=filename),
            caption="✓ Готово",
        )

    elif ft == "gif":
        result = process_gif(
            task.file_bytes,
            tint_hue,
            s.bg_color,
            s.size,
            wm_text,
            font_path,
        )
        await task.bot.send_document(
            task.chat_id,
            BufferedInputFile(result, filename="sticker.gif"),
            caption="✓ Готово",
        )

    elif ft == "tgs":
        result, warning = process_tgs(
            task.file_bytes,
            tint_hue,
            s.watermark_enabled,
        )
        caption = "✓ Готово"
        if warning:
            caption += f"\n⚠ {warning}"
        await task.bot.send_document(
            task.chat_id,
            BufferedInputFile(result, filename="sticker.tgs"),
            caption=caption,
        )

    else:
        await task.bot.send_message(task.chat_id, f"Формат `{ft}` не поддерживается.")


# ─── File downloader ─────────────────────────────────────────────────────────


async def download_file(bot: Bot, file_id: str) -> bytes:
    file = await bot.get_file(file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    return buf.getvalue()


# ─── Router ──────────────────────────────────────────────────────────────────

router = Router()


# /start
@router.message(Command("start"))
async def cmd_start(msg: Message, state: FSMContext):
    await state.set_state(BotStates.idle)
    data = await state.get_data()
    s = get_settings(data)
    await state.update_data(settings=s.to_dict())
    db_save_settings(msg.from_user.id, s.to_dict())

    await msg.answer(
        "Создан для стильного оформления\n\n" + config_text(s),
        parse_mode=ParseMode.HTML,
        reply_markup=main_keyboard(s),
    )


# Receive sticker
@router.message(F.sticker)
async def on_sticker(msg: Message, state: FSMContext):
    sticker: Sticker = msg.sticker
    await state.update_data(
        pending_file_id=sticker.file_id,
        pending_file_type="tgs" if sticker.is_animated else (
            "gif" if sticker.is_video else "webp"
        ),
    )
    data = await state.get_data()
    s = get_settings(data)
    await msg.answer(
        "Стикер получен.\n\n" + config_text(s),
        parse_mode=ParseMode.HTML,
        reply_markup=main_keyboard(s),
    )


# Receive document/photo
@router.message(F.document)
async def on_document(msg: Message, state: FSMContext):
    doc = msg.document
    name = (doc.file_name or "").lower()
    if name.endswith(".tgs"):
        ft = "tgs"
    elif name.endswith(".gif"):
        ft = "gif"
    elif name.endswith(".webp"):
        ft = "webp"
    elif name.endswith(".png") or name.endswith(".jpg"):
        ft = "png"
    else:
        await msg.answer("Неподдерживаемый формат.")
        return

    await state.update_data(pending_file_id=doc.file_id, pending_file_type=ft)
    data = await state.get_data()
    s = get_settings(data)
    await msg.answer(
        "Файл получен.\n\n" + config_text(s),
        parse_mode=ParseMode.HTML,
        reply_markup=main_keyboard(s),
    )


# ─── Callbacks ───────────────────────────────────────────────────────────────


async def _update_config_message(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    s = get_settings(data)
    try:
        await call.message.edit_text(
            config_text(s),
            parse_mode=ParseMode.HTML,
            reply_markup=main_keyboard(s),
        )
    except Exception:
        pass
    await call.answer()


@router.callback_query(F.data == "back")
async def cb_back(call: CallbackQuery, state: FSMContext):
    await _update_config_message(call, state)


@router.callback_query(F.data == "toggle_wm")
async def cb_toggle_wm(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    s = get_settings(data)
    s.watermark_enabled = not s.watermark_enabled
    await state.update_data(settings=s.to_dict())
    db_save_settings(call.from_user.id, s.to_dict())
    await _update_config_message(call, state)


@router.callback_query(F.data == "set_format")
async def cb_set_format(call: CallbackQuery, state: FSMContext):
    await call.message.edit_text("Выберите формат:", reply_markup=format_keyboard())
    await call.answer()


@router.callback_query(F.data.startswith("fmt_"))
async def cb_fmt(call: CallbackQuery, state: FSMContext):
    fmt = call.data.split("_", 1)[1]
    data = await state.get_data()
    s = get_settings(data)
    s.output_format = fmt
    await state.update_data(settings=s.to_dict())
    db_save_settings(call.from_user.id, s.to_dict())
    await _update_config_message(call, state)


@router.callback_query(F.data == "set_size")
async def cb_set_size_menu(call: CallbackQuery, state: FSMContext):
    await call.message.edit_text("Выберите размер:", reply_markup=size_keyboard())
    await call.answer()


@router.callback_query(F.data.startswith("size_"))
async def cb_size(call: CallbackQuery, state: FSMContext):
    size = int(call.data.split("_", 1)[1])
    data = await state.get_data()
    s = get_settings(data)
    s.size = size
    await state.update_data(settings=s.to_dict())
    db_save_settings(call.from_user.id, s.to_dict())
    await _update_config_message(call, state)


@router.callback_query(F.data == "set_tint")
async def cb_set_tint(call: CallbackQuery, state: FSMContext):
    await state.set_state(BotStates.waiting_color)
    await call.message.answer("Введите HEX цвет тинта (например: #3A86FF):")
    await call.answer()


@router.callback_query(F.data == "set_bg_color")
async def cb_set_bg(call: CallbackQuery, state: FSMContext):
    await state.set_state(BotStates.waiting_bg_color)
    await call.message.answer("Введите HEX цвет фона (например: #FF5E3B):")
    await call.answer()


@router.callback_query(F.data == "set_wm_text")
async def cb_set_wm_text(call: CallbackQuery, state: FSMContext):
    await state.set_state(BotStates.waiting_watermark_text)
    await call.message.answer("Введите текст watermark:")
    await call.answer()


@router.callback_query(F.data == "set_font")
async def cb_set_font(call: CallbackQuery, state: FSMContext):
    await state.set_state(BotStates.waiting_watermark_font)
    await call.message.answer(
        "Введите название шрифта Google Fonts\n(например: Montserrat, Roboto, Oswald):"
    )
    await call.answer()


@router.callback_query(F.data == "preview")
async def cb_preview(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    file_id = data.get("pending_file_id")
    if not file_id:
        await call.answer("Сначала отправьте стикер.", show_alert=True)
        return

    await call.answer("Генерирую предпросмотр…")
    bot = call.bot
    s = get_settings(data)
    ft = data.get("pending_file_type", "webp")
    tint_hue = hex_to_hue(s.tint_color)
    font_path = await ensure_font(s.watermark_font) if s.watermark_enabled else ""
    wm_text = s.watermark_text if s.watermark_enabled else None

    try:
        file_bytes = await download_file(bot, file_id)
        if ft in ("webp", "png"):
            # Force PNG for preview
            result = process_static_image(
                file_bytes, tint_hue, s.bg_color, min(s.size, 256), "PNG", wm_text, font_path
            )
            await bot.send_photo(
                call.message.chat.id,
                BufferedInputFile(result, filename="preview.png"),
                caption="Предпросмотр",
            )
        elif ft == "gif":
            result = process_gif(file_bytes, tint_hue, s.bg_color, min(s.size, 256), wm_text, font_path)
            await bot.send_document(
                call.message.chat.id,
                BufferedInputFile(result, filename="preview.gif"),
                caption="Предпросмотр",
            )
        else:
            await bot.send_message(call.message.chat.id, "Предпросмотр недоступен для этого формата.")
    except Exception as e:
        log.exception(e)
        await bot.send_message(call.message.chat.id, f"Ошибка предпросмотра: {e}")


@router.callback_query(F.data == "convert")
async def cb_convert(call: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    file_id = data.get("pending_file_id")
    if not file_id:
        await call.answer("Сначала отправьте стикер.", show_alert=True)
        return

    await call.answer("Добавлено в очередь.")
    await call.message.answer("Файл добавлен в очередь. Ожидайте…")

    bot = call.bot
    s = get_settings(data)
    ft = data.get("pending_file_type", "webp")

    file_bytes = await download_file(bot, file_id)

    task = ConvertTask(
        user_id=call.from_user.id,
        chat_id=call.message.chat.id,
        message_id=call.message.message_id,
        file_bytes=file_bytes,
        file_type=ft,
        settings=s,
        bot=bot,
    )
    await task_queue.put(task)
    log.info(f"Task queued for user {call.from_user.id}, queue size: {task_queue.qsize()}")


# ─── FSM state handlers ──────────────────────────────────────────────────────


def _is_valid_hex(s: str) -> bool:
    return bool(re.fullmatch(r"#[0-9A-Fa-f]{6}", s.strip()))


@router.message(BotStates.waiting_color)
async def state_tint_color(msg: Message, state: FSMContext):
    val = msg.text.strip() if msg.text else ""
    if not _is_valid_hex(val):
        await msg.answer("Неверный формат. Введите HEX, например: #3A86FF")
        return
    data = await state.get_data()
    s = get_settings(data)
    s.tint_color = val.upper()
    await state.update_data(settings=s.to_dict())
    db_save_settings(msg.from_user.id, s.to_dict())
    await state.set_state(BotStates.idle)
    await msg.answer(config_text(s), parse_mode=ParseMode.HTML, reply_markup=main_keyboard(s))


@router.message(BotStates.waiting_bg_color)
async def state_bg_color(msg: Message, state: FSMContext):
    val = msg.text.strip() if msg.text else ""
    if not _is_valid_hex(val):
        await msg.answer("Неверный формат. Введите HEX, например: #FF5E3B")
        return
    data = await state.get_data()
    s = get_settings(data)
    s.bg_color = val.upper()
    await state.update_data(settings=s.to_dict())
    db_save_settings(msg.from_user.id, s.to_dict())
    await state.set_state(BotStates.idle)
    await msg.answer(config_text(s), parse_mode=ParseMode.HTML, reply_markup=main_keyboard(s))


@router.message(BotStates.waiting_watermark_text)
async def state_wm_text(msg: Message, state: FSMContext):
    val = msg.text.strip() if msg.text else ""
    if not val:
        await msg.answer("Текст не может быть пустым.")
        return
    data = await state.get_data()
    s = get_settings(data)
    s.watermark_text = val
    await state.update_data(settings=s.to_dict())
    db_save_settings(msg.from_user.id, s.to_dict())
    await state.set_state(BotStates.idle)
    await msg.answer(config_text(s), parse_mode=ParseMode.HTML, reply_markup=main_keyboard(s))


@router.message(BotStates.waiting_watermark_font)
async def state_wm_font(msg: Message, state: FSMContext):
    val = msg.text.strip() if msg.text else ""
    if not val:
        await msg.answer("Введите название шрифта.")
        return
    await msg.answer(f"Скачиваю шрифт «{val}»…")
    path = await ensure_font(val)
    if not path:
        await msg.answer(
            f"Не удалось загрузить «{val}». Шрифт по умолчанию (Montserrat) будет использован."
        )
        val = "Montserrat"
        await ensure_font(val)

    data = await state.get_data()
    s = get_settings(data)
    s.watermark_font = val
    await state.update_data(settings=s.to_dict())
    db_save_settings(msg.from_user.id, s.to_dict())
    await state.set_state(BotStates.idle)
    await msg.answer(config_text(s), parse_mode=ParseMode.HTML, reply_markup=main_keyboard(s))


# ─── Main ────────────────────────────────────────────────────────────────────


async def main():
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN не задан в .env")

    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    dp.include_router(router)

    # Pre-download default font
    asyncio.create_task(ensure_font("Montserrat"))

    # Start workers
    for i in range(2):
        asyncio.create_task(worker(i + 1))

    log.info("Bot started")
    await dp.start_polling(bot, skip_updates=True)


if __name__ == "__main__":
    asyncio.run(main())
