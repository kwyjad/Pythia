# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Gate logic of the Cadre Harmonisé probe, with an injected transport.

The sandbox reaches neither api.ipcinfo.org nor data.humdata.org, so these
tests pin what the probe concludes from a given answer, never what the hosts
say today.
"""

from __future__ import annotations

import json
from datetime import date

from tools import probe_cadre_harmonise as probe

TODAY = date(2026, 10, 12)
COUNTRIES = ["MRT", "NER", "NGA"]

IPC_RECORDS = [
    {"country": "MR", "title": "Mauritania CH Mar 2026", "current_period_dates": "Mar 2026 - May 2026", "p3plus": 400000},
    {"country": "MR", "current_period_dates": "Oct 2025 - Dec 2025", "p3plus": 300000},
    {"country": "NE", "current_period_dates": "Oct 2024 - Dec 2024", "p3plus": 2000000},
]

HDX_SEARCH = {"result": {"results": [
    {"name": "cadre-harmonise", "title": "Cadre Harmonisé", "resources": [
        {"name": "cadre_harmonise_caf_ipc.csv", "format": "CSV", "url": "https://hdx.example/ch.csv"},
        {"name": "readme", "format": "PDF", "url": "https://hdx.example/readme.pdf"},
    ]},
    {"name": "unrelated", "title": "Rainfall", "resources": [
        {"name": "rain.csv", "format": "CSV", "url": "https://hdx.example/rain.csv"}]},
]}}

HDX_CSV = (
    "adm0_name,adm0_pcod3,adm1_name,reference_label,reference_year,chtype,phase35,population\n"
    "#country+name,#country+code,#adm1+name,#date,#date+year,#status,#affected,#population\n"
    "Niger,NER,Agadez,Jun-Aug,2026,current,100000,500000\n"
    "Niger,NER,Diffa,Jun-Aug,2026,current,150000,700000\n"
    "Niger,NER,Diffa,Jun-Aug,2026,projected,999999,700000\n"
    "Nigeria,NGA,Borno,Oct-Dec,2024,current,800000,5000000\n"
)


def make_transport(ipc=IPC_RECORDS, ipc_status=200, search=HDX_SEARCH, csv_text=HDX_CSV):
    calls = []

    def transport(url, params):
        calls.append(url)
        if url == probe.IPC_POPULATION:
            return ipc_status, "application/json", json.dumps(ipc).encode()
        if url == probe.HDX_SEARCH:
            return 200, "application/json", json.dumps(search).encode()
        if url == "https://hdx.example/ch.csv":
            return 200, "text/csv", csv_text.encode()
        return 404, "", b"not found"

    transport.calls = calls
    return transport


def test_period_end_month():
    assert probe.period_end_month("Mar 2026 - May 2026") == "2026-05"
    assert probe.period_end_month("June - August 2025") == "2025-08"
    assert probe.period_end_month("2025-06-01 - 2025-08-31") == "2025-08"
    assert probe.period_end_month("Oct-Dec", 2024) == "2024-12"
    assert probe.period_end_month("") is None
    assert probe.period_end_month("Oct-Dec") is None


def test_ipc_api_grades_newest_current_period():
    findings = {f.iso3: f for f in probe.ipc_api_findings(IPC_RECORDS, COUNTRIES, "2026-10")}
    assert findings["MRT"].status == "current"
    assert findings["MRT"].newest_period == "2026-05"
    assert findings["MRT"].lag_months == 5
    assert findings["MRT"].phase3_field == "p3plus"
    assert findings["NER"].status == "stale" and findings["NER"].lag_months == 22
    assert findings["NGA"].status == "absent"


def test_hdx_picks_only_ch_csv_resources():
    res = probe.pick_hdx_resources(HDX_SEARCH)
    assert [r["url"] for r in res] == ["https://hdx.example/ch.csv"]


def test_hdx_reads_current_rows_and_sums_admin1():
    findings, meta = probe.hdx_findings(HDX_CSV, COUNTRIES, "2026-10", "x")
    by = {f.iso3: f for f in findings}
    assert meta["iso3_column"] == "adm0_pcod3" and meta["phase3_column"] == "phase35"
    assert by["NER"].status == "current" and by["NER"].newest_period == "2026-08"
    assert by["NER"].phase3_value == 250000  # projected row excluded
    assert by["NGA"].status == "stale"
    assert by["MRT"].status == "absent"


def test_run_end_to_end_verdicts():
    report = probe.run(make_transport(), COUNTRIES, TODAY, api_key="k")
    assert report["verdict"] == {"MRT": "ipc_api_current", "NER": "hdx_current", "NGA": "no_current_source"}
    md = probe.render_markdown(report)
    assert "| MRT | ipc_api_current | ipc_api | current | 2026-05 | 5 | p3plus | 400000 |" in md


def test_missing_key_and_failed_routes_are_inconclusive_not_absent():
    report = probe.run(make_transport(), ["MRT"], TODAY, api_key="")
    assert report["ipc_api"]["status"] == "skipped"
    assert report["verdict"]["MRT"] == "inconclusive"

    report = probe.run(make_transport(ipc_status=500, search={"result": {"results": []}}),
                       ["NGA"], TODAY, api_key="k")
    assert report["ipc_api"]["status"] == "unreadable"
    assert report["hdx"]["status"] == "no_dataset"
    assert report["verdict"]["NGA"] == "inconclusive"


def test_key_never_reaches_the_report(monkeypatch):
    monkeypatch.setenv("IPC_API_KEY", "sekret-value-123")

    def transport(url, params):
        if url == probe.IPC_POPULATION:
            return 403, "text/plain", b"denied for key=sekret-value-123"
        return 500, "", b""

    report = probe.run(transport, ["MRT"], TODAY, api_key="sekret-value-123")
    assert "sekret-value-123" not in json.dumps(report)


def test_main_always_exits_zero(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "default_transport", make_transport())
    monkeypatch.setenv("IPC_API_KEY", "k")
    assert probe.main(["--out-dir", str(tmp_path), "--countries", "MRT,NER"]) == 0
    assert (tmp_path / "cadre_harmonise_probe.md").exists()

    def boom(url, params):
        raise RuntimeError("network gone")

    monkeypatch.setattr(probe, "default_transport", boom)
    assert probe.main(["--out-dir", str(tmp_path)]) == 0
