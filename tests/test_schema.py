from unittest.mock import MagicMock, patch

from text2cypher_composer.schema import (
    clear_schema_cache,
    format_schema,
    get_schema,
    get_structured_schema,
)

_FAKE_SCHEMA = {
    "node_props": {"miRNA": [{"property": "Label", "type": "STRING"}]},
    "rel_props": {},
    "relationships": [],
    "metadata": {},
}


def test_get_structured_schema_caches_per_graph():
    graph = MagicMock()
    with patch("text2cypher_composer.schema._fetch_structured_schema", return_value=_FAKE_SCHEMA) as fetch:
        first = get_structured_schema(graph, is_enhanced=True)
        second = get_structured_schema(graph, is_enhanced=True)

    assert first is second
    fetch.assert_called_once()


def test_get_structured_schema_cache_schema_false_always_refetches():
    graph = MagicMock()
    with patch("text2cypher_composer.schema._fetch_structured_schema", return_value=_FAKE_SCHEMA) as fetch:
        get_structured_schema(graph, is_enhanced=True, cache_schema=False)
        get_structured_schema(graph, is_enhanced=True, cache_schema=False)

    assert fetch.call_count == 2


def test_get_structured_schema_different_is_enhanced_dont_share_a_cache_entry():
    graph = MagicMock()
    with patch("text2cypher_composer.schema._fetch_structured_schema", return_value=_FAKE_SCHEMA) as fetch:
        get_structured_schema(graph, is_enhanced=True)
        get_structured_schema(graph, is_enhanced=False)

    assert fetch.call_count == 2


def test_get_structured_schema_different_graphs_dont_share_a_cache_entry():
    graph_a, graph_b = MagicMock(), MagicMock()
    with patch("text2cypher_composer.schema._fetch_structured_schema", return_value=_FAKE_SCHEMA) as fetch:
        get_structured_schema(graph_a, is_enhanced=True)
        get_structured_schema(graph_b, is_enhanced=True)

    assert fetch.call_count == 2


def test_clear_schema_cache_for_one_graph_forces_a_refetch():
    graph = MagicMock()
    with patch("text2cypher_composer.schema._fetch_structured_schema", return_value=_FAKE_SCHEMA) as fetch:
        get_structured_schema(graph, is_enhanced=True)
        clear_schema_cache(graph)
        get_structured_schema(graph, is_enhanced=True)

    assert fetch.call_count == 2


def test_clear_schema_cache_with_no_graph_clears_every_graph():
    graph_a, graph_b = MagicMock(), MagicMock()
    with patch("text2cypher_composer.schema._fetch_structured_schema", return_value=_FAKE_SCHEMA) as fetch:
        get_structured_schema(graph_a, is_enhanced=True)
        get_structured_schema(graph_b, is_enhanced=True)
        clear_schema_cache()
        get_structured_schema(graph_a, is_enhanced=True)
        get_structured_schema(graph_b, is_enhanced=True)

    assert fetch.call_count == 4


def test_get_schema_shares_the_cache_with_get_structured_schema():
    graph = MagicMock()
    with patch("text2cypher_composer.schema._fetch_structured_schema", return_value=_FAKE_SCHEMA) as fetch:
        text_via_get_schema = get_schema(graph, is_enhanced=True)
        structured_via_direct_call = get_structured_schema(graph, is_enhanced=True)

    assert fetch.call_count == 1
    assert text_via_get_schema == format_schema(_FAKE_SCHEMA, is_enhanced=True)
    assert structured_via_direct_call is _FAKE_SCHEMA
