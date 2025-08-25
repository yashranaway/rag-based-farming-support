from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional


FEATURE_VISION = os.getenv("FEATURE_VISION", "0").lower() in {"1", "true", "yes"}


@dataclass
class ParsedDoc:
    text: str
    images: List[str]
    media_notes: Optional[str] = None


def parse_text(raw: str) -> ParsedDoc:
    """Trivial text passthrough; trims leading/trailing whitespace."""
    return ParsedDoc(text=raw.strip(), images=[])


def parse_html(raw: str) -> ParsedDoc:
    """Very lightweight HTML to text: removes tags, collapses whitespace.
    Not production grade; use Docling/BeautifulSoup later.
    """
    # Remove scripts/styles
    s = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.IGNORECASE)
    s = re.sub(r"<style[\s\S]*?</style>", " ", s, flags=re.IGNORECASE)
    # Replace block tags with newlines for basic separation
    s = re.sub(r"</?(p|div|br|li|ul|ol|h[1-6])[^>]*>", "\n", s, flags=re.IGNORECASE)
    # Remove remaining tags
    s = re.sub(r"<[^>]+>", " ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    images: List[str] = []
    media_notes = None
    if FEATURE_VISION:
        # Placeholder: pretend we extracted image placeholders
        images = ["image://placeholder-1", "image://placeholder-2"]
        media_notes = "Vision feature enabled (stub): images captured but not processed."
    return ParsedDoc(text=s, images=images, media_notes=media_notes)
