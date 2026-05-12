"""
Parse stock tips from text (and optionally OCR'd images).

Used by the bot's forwarded-message handler. Extracts:
- Symbol (matched against the known NSE universe)
- Action (BUY / SELL / SHORT)
- Entry / Target / Stop-Loss prices

Photo path requires Tesseract OCR. On Windows you may also need to set
TESSERACT_CMD in .env to point at tesseract.exe. On Linux it's on PATH
after `sudo apt install tesseract-ocr`.
"""
from __future__ import annotations

import logging
import os
import re
from io import BytesIO
from typing import Optional

logger = logging.getLogger(__name__)

# ---------- Regex patterns ----------

# Action verbs — order matters: more-specific patterns first
_ACTION_BUY = re.compile(
    r"\b(?:BUY|LONG|BULLISH|GO\s*LONG|ENTER\s*LONG|BUY\s*ABOVE)\b",
    re.IGNORECASE,
)
_ACTION_SELL = re.compile(
    r"\b(?:SELL|SHORT|BEARISH|GO\s*SHORT|ENTER\s*SHORT|SELL\s*BELOW)\b",
    re.IGNORECASE,
)

# Labelled prices — most reliable signal that this is a tip
_PRICE_ENTRY = re.compile(
    r"(?:@|\bat\b|\bentry\b|\bbuy\s*at\b|\bsell\s*at\b|\bcmp\b|\blevel\b)\s*:?\s*"
    r"₹?\s*([0-9]{2,6}(?:\.[0-9]{1,2})?)",
    re.IGNORECASE,
)
_PRICE_TARGET = re.compile(
    r"(?:\btarget\b|\btgt\b|\bt1\b|\btp\b|\btarget\s*1\b)\s*:?\s*"
    r"₹?\s*([0-9]{2,6}(?:\.[0-9]{1,2})?)",
    re.IGNORECASE,
)
_PRICE_TARGET2 = re.compile(
    r"(?:\bt2\b|\btarget\s*2\b|\btp2\b)\s*:?\s*"
    r"₹?\s*([0-9]{2,6}(?:\.[0-9]{1,2})?)",
    re.IGNORECASE,
)
_PRICE_SL = re.compile(
    r"(?:\bsl\b|\bstop\s*loss\b|\bstoploss\b|\bstop\b)\s*:?\s*"
    r"₹?\s*([0-9]{2,6}(?:\.[0-9]{1,2})?)",
    re.IGNORECASE,
)

# Candidate ticker tokens — uppercase, 2–15 chars, can contain digits/hyphen.
_TICKER_CANDIDATE = re.compile(r"\b[A-Z][A-Z0-9-]{1,14}\b")

# Words to ignore as ticker candidates (common false positives)
_NON_TICKER_WORDS = {
    "BUY", "SELL", "SHORT", "LONG", "TGT", "TARGET", "SL", "STOP", "LOSS",
    "CMP", "ENTRY", "EXIT", "BULLISH", "BEARISH", "BREAKOUT", "BREAKDOWN",
    "NIFTY", "BANKNIFTY", "SENSEX", "INTRADAY", "SWING", "BUYING", "SELLING",
    "AT", "ABOVE", "BELOW", "GO", "ENTER", "BOOK", "PROFIT", "VOL", "VOLUME",
    "QTY", "RSI", "MACD", "EMA", "SMA", "ATR", "VWAP", "CALL", "PUT",
    "PE", "CE", "STRIKE", "EXPIRY", "OI", "FII", "DII", "IPO", "PSU",
    "NEWS", "RESULTS", "Q1", "Q2", "Q3", "Q4", "FY25", "FY26",
}


def _first_match(pattern: re.Pattern, text: str) -> Optional[float]:
    m = pattern.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except (ValueError, IndexError):
        return None


def _find_symbol(text: str, known_tickers: set[str]) -> Optional[str]:
    """Pick the first uppercase token in text that matches a known ticker."""
    seen: list[str] = []
    for match in _TICKER_CANDIDATE.finditer(text):
        tok = match.group(0)
        if tok in _NON_TICKER_WORDS:
            continue
        if tok in known_tickers and tok not in seen:
            seen.append(tok)
    return seen[0] if seen else None


def extract_tip(text: str, known_tickers: set[str]) -> Optional[dict]:
    """
    Extract structured tip data from a plaintext message.
    Returns None if nothing tip-like found (no known ticker OR no action verb).
    """
    if not text or not text.strip():
        return None

    symbol = _find_symbol(text, known_tickers)
    if not symbol:
        return None

    has_buy = bool(_ACTION_BUY.search(text))
    has_sell = bool(_ACTION_SELL.search(text))

    if has_buy and not has_sell:
        action = "BUY"
    elif has_sell and not has_buy:
        action = "SELL"
    elif has_buy and has_sell:
        # Both keywords — ambiguous. Skip rather than guess wrong.
        return None
    else:
        # No action verb — probably not a trade tip
        return None

    return {
        "symbol": symbol,
        "action": action,
        "entry": _first_match(_PRICE_ENTRY, text),
        "target": _first_match(_PRICE_TARGET, text),
        "target2": _first_match(_PRICE_TARGET2, text),
        "sl": _first_match(_PRICE_SL, text),
        "raw_text": text.strip()[:500],
    }


# ---------- OCR path ----------

def _configure_tesseract() -> None:
    """Honor TESSERACT_CMD env var on Windows where tesseract isn't on PATH."""
    cmd = os.getenv("TESSERACT_CMD")
    if cmd:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = cmd


def extract_tip_from_image(image_bytes: bytes, known_tickers: set[str]) -> Optional[dict]:
    """
    OCR an image and extract a tip from the recognized text.
    Returns the tip dict (with `_ocr_text` key showing what was read), or None.
    Raises RuntimeError if Tesseract isn't installed.
    """
    try:
        from PIL import Image
        import pytesseract
    except ImportError as e:
        raise RuntimeError(f"OCR dependencies missing — pip install pytesseract Pillow ({e})")

    _configure_tesseract()

    try:
        img = Image.open(BytesIO(image_bytes))
    except Exception as e:
        raise RuntimeError(f"Failed to decode image: {e}")

    try:
        text = pytesseract.image_to_string(img)
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract binary not found.\n"
            "Windows: install from https://github.com/UB-Mannheim/tesseract/wiki "
            "and set TESSERACT_CMD in .env.\n"
            "Linux: sudo apt install tesseract-ocr"
        )

    tip = extract_tip(text, known_tickers)
    if tip:
        tip["_ocr_text"] = text.strip()[:500]
    return tip
