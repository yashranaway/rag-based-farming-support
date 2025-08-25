from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class NormalizedLocation:
    gps: Optional[Tuple[float, float]]
    pincode: Optional[str]
    district: Optional[str]
    region_tags: tuple[str, ...]


def _normalize_pincode(pin: Optional[str]) -> Optional[str]:
    if not pin:
        return None
    digits = ''.join(ch for ch in pin if ch.isdigit())
    if len(digits) < 5 or len(digits) > 8:
        return None
    return digits


def _normalize_gps(gps: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if not gps:
        return None
    lat, lon = gps
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return (round(lat, 6), round(lon, 6))


def normalize_location(
    gps: Optional[Tuple[float, float]] = None,
    pincode: Optional[str] = None,
    district: Optional[str] = None,
) -> NormalizedLocation:
    """
    Normalize inputs and derive simple region tags for retrieval filters.
    """
    ngps = _normalize_gps(gps)
    npin = _normalize_pincode(pincode)
    ndist = district.strip() if district and district.strip() else None

    tags: list[str] = []
    if ndist:
        tags.append(f"district:{ndist.lower()}")
    if npin:
        tags.append(f"pincode:{npin}")
    if ngps:
        tags.append("geo:present")

    return NormalizedLocation(gps=ngps, pincode=npin, district=ndist, region_tags=tuple(tags))
