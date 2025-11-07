from resolver.ingestion.idmc.diagnostics import serialize_http_status_counts


def test_serialize_http_status_counts_clamps_buckets():
    counts = serialize_http_status_counts({"2xx": 3, "4xx": 1, "timeout": 7, "other": 9})

    assert counts == {"2xx": 3, "4xx": 1, "5xx": 0}


def test_serialize_http_status_counts_handles_none():
    counts = serialize_http_status_counts(None)

    assert counts == {"2xx": 0, "4xx": 0, "5xx": 0}
