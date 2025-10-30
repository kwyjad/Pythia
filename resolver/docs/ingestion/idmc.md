# IDMC (IDU) Connector — Skeleton

*Status:* offline-only skeleton for deterministic tests.

- **Series:** `idp_displacement_new_idmc` (flow), `idp_displacement_stock_idmc` (stock)
- **Semantics:** flow = `new` (monthly), stock = `stock`
- **CLI:** `python -m resolver.ingestion.idmc.cli --skip-network`
- **Diagnostics:** writes to `diagnostics/ingestion/connectors.jsonl` and preview CSV at `diagnostics/ingestion/idmc/normalized_preview.csv`
- **Fixtures:** `resolver/ingestion/idmc/fixtures/*.csv`
