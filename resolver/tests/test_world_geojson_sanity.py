import json
from pathlib import Path


def iter_outer_ring_positions(geometry):
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if geom_type == "Polygon":
        ring = coords[0] if coords else []
        for lon, lat in ring:
            yield lon, lat
        return
    if geom_type == "MultiPolygon":
        for polygon in coords:
            ring = polygon[0] if polygon else []
            for lon, lat in ring:
                yield lon, lat


def has_iso3(properties):
    if not isinstance(properties, dict):
        return False
    iso3 = (
        properties.get("iso3")
        or properties.get("ISO_A3")
        or properties.get("ADM0_A3")
        or properties.get("adm0_a3")
    )
    return bool(iso3)


def get_iso3(properties):
    if not isinstance(properties, dict):
        return None
    iso3 = (
        properties.get("iso3")
        or properties.get("ISO_A3")
        or properties.get("ADM0_A3")
        or properties.get("adm0_a3")
    )
    if not iso3 or iso3 == "-99":
        return None
    return str(iso3).strip().upper()


def find_feature(features, iso3):
    iso3 = iso3.upper()
    for feature in features:
        properties = feature.get("properties") or {}
        if get_iso3(properties) == iso3:
            return feature
    return None


def bbox_center_from_geometry(geometry):
    min_lon = None
    max_lon = None
    min_lat = None
    max_lat = None
    for lon, lat in iter_outer_ring_positions(geometry):
        if not isinstance(lon, (int, float)) or not isinstance(lat, (int, float)):
            continue
        min_lon = lon if min_lon is None else min(min_lon, lon)
        max_lon = lon if max_lon is None else max(max_lon, lon)
        min_lat = lat if min_lat is None else min(min_lat, lat)
        max_lat = lat if max_lat is None else max(max_lat, lat)
    if None in (min_lon, max_lon, min_lat, max_lat):
        return None
    return ((min_lon + max_lon) / 2, (min_lat + max_lat) / 2)


def test_world_geojson_sanity():
    path = Path(__file__).resolve().parents[2] / "web" / "public" / "maps" / "world-countries-iso3.geojson"
    with path.open() as handle:
        data = json.load(handle)

    assert data.get("type") == "FeatureCollection"
    features = data.get("features", [])
    assert len(features) > 150

    polygon_features = [
        feature
        for feature in features
        if (feature.get("geometry") or {}).get("type") in {"Polygon", "MultiPolygon"}
    ]
    assert len(polygon_features) >= 100

    iso3_features = [feature for feature in features if has_iso3(feature.get("properties") or {})]
    assert len(iso3_features) >= 150

    latitudes = set()
    max_abs_lon = 0.0
    max_abs_lat = 0.0
    for feature in polygon_features[:50]:
        geometry = feature.get("geometry") or {}
        for lon, lat in iter_outer_ring_positions(geometry):
            if isinstance(lat, (float, int)):
                latitudes.add(round(lat, 4))
            if isinstance(lon, (float, int)):
                max_abs_lon = max(max_abs_lon, abs(lon))
            if isinstance(lat, (float, int)):
                max_abs_lat = max(max_abs_lat, abs(lat))

    assert len(latitudes) > 200
    assert max_abs_lon <= 180.5
    assert max_abs_lat <= 90.5

    sentinels = {
        "AFG": {"lon": (50, 80), "lat": (20, 45)},
        "AUS": {"lon": (110, 160), "lat": (-45, -10)},
        "USA": {"lon": (-130, -60), "lat": (20, 55)},
        "BRA": {"lon": (-80, -30), "lat": (-40, 10)},
        "CAN": {"lon": (-150, -40), "lat": (40, 85)},
    }
    iso3_codes = sorted(
        {
            code
            for feature in features
            for code in [get_iso3(feature.get("properties") or {})]
            if code
        }
    )
    for iso3, ranges in sentinels.items():
        feature = find_feature(features, iso3)
        if feature is None:
            prefix = iso3[:2]
            related = [code for code in iso3_codes if code.startswith(prefix)][:25]
            sample = iso3_codes[:25]
            assert feature is not None, (
                f"Expected ISO3 feature {iso3}. "
                f"Available ISO3 count={len(iso3_codes)} "
                f"sample={sample} related_prefix={related}"
            )
        geometry = feature.get("geometry") or {}
        center = bbox_center_from_geometry(geometry)
        assert center is not None, f"Expected geometry for {iso3}"
        lon, lat = center
        assert ranges["lon"][0] <= lon <= ranges["lon"][1]
        assert ranges["lat"][0] <= lat <= ranges["lat"][1]
