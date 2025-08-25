from app.services.location import normalize_location


def test_normalize_location_all_fields():
    n = normalize_location(gps=(19.076, 72.8777), pincode=" 400001 ", district=" Mumbai ")
    assert n.gps == (19.076, 72.8777)
    assert n.pincode == "400001"
    assert n.district == "Mumbai"
    assert "district:mumbai" in n.region_tags
    assert "pincode:400001" in n.region_tags
    assert "geo:present" in n.region_tags


def test_normalize_location_invalid_gps_and_pin():
    n = normalize_location(gps=(200.0, 200.0), pincode="xx12")
    assert n.gps is None
    assert n.pincode is None
    assert all(tag not in n.region_tags for tag in ("geo:present", "pincode:"))


def test_normalize_location_minimal():
    n = normalize_location()
    assert n.gps is None and n.pincode is None and n.district is None
    assert n.region_tags == tuple()
