from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

# Simple language detection util focused on common Indian languages as a starting point.
# Heuristic-based to avoid heavy dependencies; can be swapped with a proper model later.
# Detects based on Unicode script ranges and basic keywords.

# Unicode blocks (partial):
# Devanagari: \u0900-\u097F (e.g., Hindi/Marathi)
# Tamil: \u0B80-\u0BFF
# Telugu: \u0C00-\u0C7F
# Bengali: \u0980-\u09FF
# Gujarati: \u0A80-\u0AFF
# Gurmukhi (Punjabi): \u0A00-\u0A7F
# Kannada: \u0C80-\u0CFF
# Malayalam: \u0D00-\u0D7F
# Odia: \u0B00-\u0B7F
# Latin: default to English


@dataclass
class DetectedLanguage:
    code: str  # e.g., 'hi', 'en', 'ta'
    confidence: float


def _script_ratio(text: str, start: int, end: int) -> float:
    if not text:
        return 0.0
    total = len(text)
    matches = sum(1 for ch in text if start <= ord(ch) <= end)
    return matches / max(total, 1)


def detect_language(text: str) -> Optional[DetectedLanguage]:
    """
    Very lightweight heuristic detector. Returns None if text is too short or ambiguous.
    """
    if not text or len(text.strip()) < 2:
        return None

    ratios: list[Tuple[str, float]] = []
    ratios.append(("hi", _script_ratio(text, 0x0900, 0x097F)))  # Devanagari
    ratios.append(("bn", _script_ratio(text, 0x0980, 0x09FF)))  # Bengali
    ratios.append(("pa", _script_ratio(text, 0x0A00, 0x0A7F)))  # Gurmukhi
    ratios.append(("gu", _script_ratio(text, 0x0A80, 0x0AFF)))  # Gujarati
    ratios.append(("ta", _script_ratio(text, 0x0B80, 0x0BFF)))  # Tamil
    ratios.append(("or", _script_ratio(text, 0x0B00, 0x0B7F)))  # Odia
    ratios.append(("te", _script_ratio(text, 0x0C00, 0x0C7F)))  # Telugu
    ratios.append(("kn", _script_ratio(text, 0x0C80, 0x0CFF)))  # Kannada
    ratios.append(("ml", _script_ratio(text, 0x0D00, 0x0D7F)))  # Malayalam

    # Choose the max script ratio
    code, conf = max(ratios, key=lambda x: x[1])
    if conf >= 0.2:  # at least 20% of characters in the script
        return DetectedLanguage(code=code, confidence=conf)

    # Default to English if mostly Latin
    latin_ratio = sum(1 for ch in text if (ord(ch) <= 0x007F)) / len(text)
    if latin_ratio >= 0.7:
        return DetectedLanguage(code="en", confidence=latin_ratio)

    return None


def choose_response_language(detected: Optional[DetectedLanguage], prefs_language: Optional[str], locale: Optional[str]) -> str:
    """
    Decide response language based on user preference > locale > detected.
    Returns language code string.
    """
    if prefs_language:
        return prefs_language
    if locale:
        return locale
    if detected:
        return detected.code
    return "auto"
