from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from typing import List, Optional, Dict, Any


@dataclass
class WeatherSnapshot:
    timestamp: datetime
    temp_c: float
    rain_mm: float
    wind_kph: float
    alert: Optional[str] = None


class WeatherClient:
    """Mock-first weather client. Real implementation will call IMD/other APIs."""

    def __init__(self) -> None:
        pass

    def current_and_forecast(self, location: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now(UTC)
        return {
            "location": location,
            "current": WeatherSnapshot(timestamp=now, temp_c=30.2, rain_mm=0.0, wind_kph=9.0).__dict__,
            "forecast": [
                WeatherSnapshot(timestamp=now + timedelta(hours=6), temp_c=29.0, rain_mm=2.0, wind_kph=8.0).__dict__,
                WeatherSnapshot(timestamp=now + timedelta(hours=12), temp_c=27.5, rain_mm=10.0, wind_kph=12.0, alert="rain").__dict__,
            ],
        }


class MandiClient:
    """Mock-first mandi/eNAM client."""

    def __init__(self) -> None:
        self._prices = {
            ("tomato", "mumbai"): [{"market": "Vashi APMC", "price": 1800, "unit": "INR/quintal", "ts": datetime.now(UTC).isoformat()}],
        }

    def latest_prices(self, crop: str, region: str) -> List[Dict[str, Any]]:
        key = (crop.lower(), region.lower())
        return self._prices.get(key, [])


class SoilClient:
    """Mock-first soil defaults client."""

    def defaults_for_region(self, region: str) -> Dict[str, Any]:
        return {"ph": 6.8, "texture": "loam", "moisture": "medium", "region": region}


class GovtClient:
    """Mock-first government advisories client."""

    def __init__(self) -> None:
        self._advisories = [
            {
                "title": "Pest advisory: tomato leaf miner",
                "url": "https://example.gov/advisory/leaf-miner",
                "timestamp": datetime.now(UTC).isoformat(),
                "region": "maharashtra",
            }
        ]

    def latest_advisories(self, region: Optional[str] = None) -> List[Dict[str, Any]]:
        if not region:
            return self._advisories
        r = region.lower()
        return [a for a in self._advisories if a.get("region", "").lower() == r]
