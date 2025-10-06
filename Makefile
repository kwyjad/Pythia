ctx:
	python tools/context_pack.py --base origin/main

ctx-changed:
	python tools/context_pack.py --base $(shell git describe --tags --abbrev=0 2>/dev/null || echo origin/main)

.PHONY: dev-setup dev-setup-online dev-setup-offline test-db

dev-setup:
	@echo ">> Installing DB test deps (offline-first)..."
	@$(MAKE) dev-setup-offline || $(MAKE) dev-setup-online
	@python -c "import duckdb; print('duckdb installed:', duckdb.__version__)"

dev-setup-offline:
	@echo ">> Attempting offline wheel install from tools/offline_wheels"
	python -m pip install --upgrade pip
	python -m pip install --no-index --find-links tools/offline_wheels -r tools/offline_wheels/constraints-db.txt

dev-setup-online:
	@echo ">> Falling back to online install"
	python -m pip install --upgrade pip
	python -m pip install -e ".[db]" || python -m pip install duckdb pytest

test-db:
	RESOLVER_API_BACKEND=db RESOLVER_DB_URL=duckdb:///resolver.dev.duckdb pytest -q resolver/tests/test_db_query_contract.py
