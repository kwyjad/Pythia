import json
from pathlib import Path


def iter_positions(coords):
    if isinstance(coords, (float, int)):
        return
    if coords and isinstance(coords[0], (float, int)):
        yield coords
        return
    for item in coords:
        yield from iter_positions(item)


def count_outer_ring_vertices(geometry):
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")
    if geom_type == "Polygon":
        return len(coords[0]) if coords else 0
    if geom_type == "MultiPolygon":
        return sum(len(polygon[0]) for polygon in coords if polygon)
    return 0


def test_world_geojson_sanity():
    path = Path(__file__).resolve().parents[2] / "web" / "public" / "maps" / "world-countries-iso3.geojson"
    with path.open() as handle:
        data = json.load(handle)

    assert data.get("type") == "FeatureCollection"
    features = data.get("features", [])
    assert len(features) > 150

    vertex_total = 0
    for feature in features:
        geometry = feature.get("geometry", {})
        vertex_total += count_outer_ring_vertices(geometry)

    avg_vertices = vertex_total / len(features)
    assert avg_vertices > 30

    lon_min = 180.0
    lon_max = -180.0
    lat_min = 90.0
    lat_max = -90.0
    for feature in features:
        geometry = feature.get("geometry", {})
        coords = geometry.get("coordinates")
        for lon, lat in iter_positions(coords):
            lon_min = min(lon_min, lon)
            lon_max = max(lon_max, lon)
            lat_min = min(lat_min, lat)
            lat_max = max(lat_max, lat)

    assert -180.0 <= lon_min <= 180.0
    assert -180.0 <= lon_max <= 180.0
    assert -90.0 <= lat_min <= 90.0
    assert -90.0 <= lat_max <= 90.0
