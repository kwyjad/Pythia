ctx:
	python tools/context_pack.py --base origin/main

ctx-changed:
	python tools/context_pack.py --base $(shell git describe --tags --abbrev=0 2>/dev/null || echo origin/main)

PY ?= $(if $(PY_BIN),$(PY_BIN),python)

.PHONY: dev-setup dev-setup-online dev-setup-offline test-db which-python

which-python:
	@echo "PY=$(PY)"
	@$(PY) -c "import sys; print('sys.executable', sys.executable)"

dev-setup:
        @./scripts/codex_bootstrap_db.sh

dev-setup-offline:
	@echo ">> Attempting offline wheel install from tools/offline_wheels"
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install --no-index --find-links tools/offline_wheels -r tools/offline_wheels/constraints-db.txt

dev-setup-online:
        @echo ">> Falling back to online install"
        $(PY) -m pip install --upgrade pip
        @if ! $(PY) -m pip install -e ".[db,test]"; then \
                $(PY) -m pip install -e .; \
                $(PY) -m pip install duckdb==0.10.3 httpx pytest; \
        fi

test-db:
	RESOLVER_API_BACKEND=db RESOLVER_DB_URL=duckdb:///resolver.dev.duckdb $(PY) -m pytest -q resolver/tests/test_db_query_contract.py
