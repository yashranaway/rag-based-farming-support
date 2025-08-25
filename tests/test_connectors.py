from datetime import UTC, datetime

from app.services.connectors import WeatherClient, MandiClient, SoilClient, GovtClient


def test_weather_client_mock():
    wc = WeatherClient()
    out = wc.current_and_forecast({"district": "Mumbai"})
    assert "current" in out and "forecast" in out
    assert isinstance(out["forecast"], list) and len(out["forecast"]) >= 1
    assert "temp_c" in out["current"]


def test_mandi_client_mock():
    mc = MandiClient()
    prices = mc.latest_prices("tomato", "mumbai")
    assert isinstance(prices, list)
    if prices:
        p = prices[0]
        assert "market" in p and "price" in p and "ts" in p


def test_soil_client_mock():
    sc = SoilClient()
    d = sc.defaults_for_region("maharashtra")
    assert d["ph"] and d["texture"] and d["region"] == "maharashtra"


def test_govt_client_mock():
    gc = GovtClient()
    all_adv = gc.latest_advisories()
    assert isinstance(all_adv, list) and len(all_adv) >= 1
    maha = gc.latest_advisories("maharashtra")
    assert all(a.get("region") == "maharashtra" for a in maha)
