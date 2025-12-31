# Pythia Web

Minimal Next.js dashboard scaffold for the Pythia API.

## Setup

```bash
cd web
npm install
cp .env.example .env.local
```

Edit `.env.local` with your API base URL.

## Development

```bash
npm run dev
```

Open http://localhost:3000.

## Build & Lint

```bash
npm run build
npm run lint
```

## Maps

The overview map is generated from Natural Earth admin-0 country boundaries
(110m recommended). To refresh the assets:

1. Replace `web/public/maps/world-countries-iso3.geojson` with a Natural Earth
   admin-0 GeoJSON that includes ISO3 properties (`iso3`, `ISO_A3`, or
   `ADM0_A3`).
2. Run `python scripts/maps/build_world_svg.py`.
3. Commit both `world-countries-iso3.geojson` and the regenerated
   `world.svg`.

Web UI guardrails and resolver sanity tests will fail if the map asset is not
real geography.
