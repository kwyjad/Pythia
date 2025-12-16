# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import copy
import json
from typing import Dict, Iterable, List

from resolver.ingestion import reliefweb_client as rw


class DummyResponse:
    def __init__(self, status_code: int, json_data: Dict[str, object] | None = None, headers: Dict[str, str] | None = None):
        self.status_code = status_code
        self._json = json_data
        self.headers = headers or {}
        self._text = json.dumps(json_data, sort_keys=True) if json_data is not None else ""

    def json(self) -> Dict[str, object]:
        if self._json is None:
            raise ValueError("No JSON payload")
        return self._json

    @property
    def text(self) -> str:
        return self._text


class DummySession:
    def __init__(self, get_responses: Iterable[DummyResponse], post_responses: Iterable[DummyResponse]):
        self._get_iter = iter(get_responses)
        self._post_iter = iter(post_responses)
        self.get_calls: List[Dict[str, object]] = []
        self.post_calls: List[Dict[str, object]] = []

    def get(self, url: str, params: Dict[str, object] | None = None, timeout: float | None = None) -> DummyResponse:
        self.get_calls.append({"url": url, "params": params})
        try:
            return next(self._get_iter)
        except StopIteration as exc:  # pragma: no cover - defensive guard
            raise AssertionError("Unexpected GET call") from exc

    def post(self, url: str, json: Dict[str, object] | None = None, timeout: float | None = None) -> DummyResponse:
        snapshot = copy.deepcopy(json)
        self.post_calls.append({"url": url, "json": snapshot})
        try:
            return next(self._post_iter)
        except StopIteration as exc:  # pragma: no cover - defensive guard
            raise AssertionError("Unexpected POST call") from exc


def test_reliefweb_retries_without_type_field():
    url = "https://api.reliefweb.int/v2/reports?appname=test"
    payload: Dict[str, object] = {
        "fields": {"include": ["id", "type", "title"]},
        "filter": {
            "conditions": [
                {"field": "date.created", "value": {"from": "2025-01-01T00:00:00Z"}},
                {"field": "language", "value": "en"},
                {"field": "format", "value": ["Report"]},
            ]
        },
        "limit": 10,
        "offset": 0,
    }
    challenge_tracker = {"count": 0, "persisted": False}
    session = DummySession(
        get_responses=[DummyResponse(200, {"data": []})],
        post_responses=[
            DummyResponse(400, {"error": {"message": "Unrecognized field 'type'"}}),
            DummyResponse(200, {"data": [], "totalCount": 0}),
        ],
    )

    allowed_fields = ["id", "title", "type"]
    data, mode = rw.rw_request(
        session=session,
        url=url,
        payload=payload,
        since="2025-01-01T00:00:00Z",
        max_retries=2,
        retry_backoff=0.1,
        timeout=1.0,
        challenge_tracker=challenge_tracker,
        allowed_fields=allowed_fields,
    )

    assert mode == "post"
    assert data == {"data": [], "totalCount": 0}
    assert payload["fields"]["include"] == ["id", "title"]

    assert len(session.post_calls) == 2
    first_payload = session.post_calls[0]["json"]
    second_payload = session.post_calls[1]["json"]
    assert "type" in first_payload["fields"]["include"]
    assert "type" not in second_payload["fields"]["include"]
    assert challenge_tracker["persisted"] is False
