from scripts.ci import summarize_connectors


def test_config_block_renders_with_expected_lines():
    entries = [
        {
            "connector_id": "dtm",
            "status": "ok",
            "extras": {
                "config": {
                    "config_source_label": "resolver",
                    "config_path_used": "resolver/config/dtm.yml",
                    "config_warnings": [],
                }
            },
        }
    ]

    markdown = summarize_connectors.build_markdown(entries)

    assert "## Config used" in markdown
    assert "Config source: resolver" in markdown
    assert "Config: resolver/config/dtm.yml" in markdown
