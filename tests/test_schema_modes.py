from unittest.mock import MagicMock, patch

import pytest
from langchain_core.runnables import RunnableLambda

from text2cypher_composer.schema import format_schema
from text2cypher_composer.schema_modes import (
    SchemaSelection,
    exact_match_prune,
    expand_labels_by_hops,
    llm_prune,
    llm_prune_nodes_only,
    mask_entities,
    ner_exact_match_prune,
    resolve_cascade_mode_levels,
    resolve_schema_text,
    schema_delta,
    similarity_prune,
    similarity_prune_nodes_only,
    two_hop_expansion_prune,
)
from text2cypher_composer.techniques import CascadeModeLevel

STRUCTURED_SCHEMA = {
    "node_props": {
        "Gene": [{"property": "Label", "type": "STRING"}],
        "miRNA": [{"property": "Label", "type": "STRING"}],
        "Cancer": [{"property": "Label", "type": "STRING"}],
    },
    "relationships": [
        {"start": "Gene", "type": "TRANSCRIBED_TO", "end": "miRNA"},
        {"start": "miRNA", "type": "OVER_EXPRESSED_IN", "end": "Cancer"},
    ],
    "rel_props": {},
    "metadata": {},
}


def test_exact_match_prune_keeps_only_mentioned_labels():
    pruned = exact_match_prune(STRUCTURED_SCHEMA, "Which genes are related to cancer?")
    assert set(pruned["node_props"]) == {"Gene", "Cancer"}
    # neither endpoint of the Gene->miRNA or miRNA->Cancer relationship is fully selected
    assert pruned["relationships"] == []


def test_exact_match_prune_keeps_relationship_between_two_mentioned_labels():
    pruned = exact_match_prune(STRUCTURED_SCHEMA, "Which miRNAs are over expressed in cancer?")
    assert set(pruned["node_props"]) == {"miRNA", "Cancer"}
    assert pruned["relationships"] == [{"start": "miRNA", "type": "OVER_EXPRESSED_IN", "end": "Cancer"}]


def test_exact_match_prune_falls_back_to_full_schema_when_nothing_matches():
    pruned = exact_match_prune(STRUCTURED_SCHEMA, "What's the weather today?")
    assert pruned == STRUCTURED_SCHEMA


class _FakeEntity:
    def __init__(self, text, label, start_char):
        self.label_ = label
        self.start_char = start_char
        self.end_char = start_char + len(text)


class _FakeDoc:
    def __init__(self, ents):
        self.ents = ents


class _FakeNLP:
    """Minimal spaCy-like stand-in: masks a single hardcoded gene name as GENE."""

    def __call__(self, text):
        idx = text.find("MIR21")
        ents = [_FakeEntity("MIR21", "GENE", idx)] if idx != -1 else []
        return _FakeDoc(ents)


def test_mask_entities_replaces_entity_span_with_its_label():
    masked = mask_entities("Which miRNAs come from MIR21?", _FakeNLP())
    assert masked == "Which miRNAs come from GENE?"


def test_ner_exact_match_prune_uses_masked_question():
    # "MIR21" isn't a node label, so plain exact_match_prune would miss "Gene" here;
    # masking it to "GENE" still won't match "Gene" via substring — this instead
    # confirms masking ran and didn't crash the downstream exact-match pruning.
    pruned = ner_exact_match_prune(STRUCTURED_SCHEMA, "Which genes transcribe MIR21?", _FakeNLP())
    assert "Gene" in pruned["node_props"]


SIMILARITY_SCHEMA = {
    "node_props": {
        "Gene": [{"property": "symbol", "type": "STRING"}],
        # "Pathway" only becomes relevant via its "gene" property, not its own label —
        # exercises the "sneaks in via property match" case similarity_prune_nodes_only excludes.
        "Pathway": [{"property": "gene", "type": "STRING"}],
    },
    "relationships": [{"start": "Gene", "type": "PART_OF", "end": "Pathway"}],
    "rel_props": {"PART_OF": [{"property": "role", "type": "STRING"}]},
    "metadata": {},
}


class _FakeToken:
    """word-vector similarity == exact (lowercased) word match, for testing."""

    def __init__(self, text):
        self.text = text.lower()
        self.is_alpha = text.isalpha()
        self.has_vector = True

    def similarity(self, other):
        return 1.0 if self.text == other.text else 0.0


class _FakeSimilarityNLP:
    class _Vocab:
        def __getitem__(self, term):
            return _FakeToken(term)

    def __init__(self):
        self.vocab = self._Vocab()

    def __call__(self, text):
        return [_FakeToken(w) for w in text.split()]


def test_similarity_prune_full_lets_a_property_match_sneak_the_node_in():
    pruned = similarity_prune(SIMILARITY_SCHEMA, "Show me every gene entry", _FakeSimilarityNLP())
    # "Pathway" has no matching label, but its "gene" property does — so it's kept
    assert set(pruned["node_props"]) == {"Gene", "Pathway"}
    # the PART_OF relationship is kept via its *start* endpoint ("Gene") alone
    assert pruned["relationships"] == [{"start": "Gene", "type": "PART_OF", "end": "Pathway"}]


def test_similarity_prune_nodes_only_requires_the_label_itself_to_match():
    pruned = similarity_prune_nodes_only(SIMILARITY_SCHEMA, "Show me every gene entry", _FakeSimilarityNLP())
    # "Pathway"'s label doesn't match "gene" — property-only matches don't count here
    assert set(pruned["node_props"]) == {"Gene"}
    assert pruned["node_props"]["Gene"] == SIMILARITY_SCHEMA["node_props"]["Gene"]  # full properties kept
    # both endpoints must be selected labels — Pathway isn't, so the relationship is dropped
    assert pruned["relationships"] == []
    assert pruned["rel_props"] == {}


def test_similarity_prune_nodes_only_falls_back_to_full_schema_when_nothing_matches():
    pruned = similarity_prune_nodes_only(SIMILARITY_SCHEMA, "What's the weather today?", _FakeSimilarityNLP())
    assert pruned == SIMILARITY_SCHEMA


class _FakeStructuredOutputLLM:
    """Stands in for a chat model's `.with_structured_output(...)`, returning a fixed selection."""

    def __init__(self, selection: SchemaSelection):
        self._selection = selection

    def with_structured_output(self, schema, method=None):
        return RunnableLambda(lambda _inputs: self._selection)


def test_llm_prune_uses_relationship_types_directly():
    selection = SchemaSelection(node_labels=["Gene"], relationship_types=["PART_OF"])
    pruned = llm_prune(SIMILARITY_SCHEMA, "irrelevant text", _FakeStructuredOutputLLM(selection))
    # PART_OF is explicitly selected, so it's kept even though "Pathway" isn't
    assert pruned["relationships"] == [{"start": "Gene", "type": "PART_OF", "end": "Pathway"}]


def test_llm_prune_nodes_only_ignores_relationship_types():
    selection = SchemaSelection(node_labels=["Gene"], relationship_types=["PART_OF"])
    pruned = llm_prune_nodes_only(SIMILARITY_SCHEMA, "irrelevant text", _FakeStructuredOutputLLM(selection))
    assert set(pruned["node_props"]) == {"Gene"}
    # relationship_types is discarded — Pathway isn't a selected label, so PART_OF is dropped
    assert pruned["relationships"] == []


def test_resolve_cascade_mode_levels_exact_match_three_rungs_in_order():
    with patch("text2cypher_composer.schema_modes.get_structured_schema", return_value=STRUCTURED_SCHEMA) as get_structured, \
         patch("text2cypher_composer.schema_modes.get_schema") as get_schema:
        levels = resolve_cascade_mode_levels(
            MagicMock(), "exact_match", "Which miRNAs are over expressed in cancer?"
        )

    assert [level for level, _ in levels] == [
        CascadeModeLevel.NARROW,
        CascadeModeLevel.NODES_ONLY,
        CascadeModeLevel.FULL,
    ]
    narrow_text, nodes_only_text, full_text = (text for _, text in levels)
    assert "miRNA" in narrow_text and "Cancer" in narrow_text
    assert "miRNA" in nodes_only_text and "Cancer" in nodes_only_text
    # the "full" rung reuses the schema already fetched for narrow/nodes_only -- one Neo4j
    # round-trip per resolve_cascade_mode_levels() call, not a second get_schema() call
    assert full_text == format_schema(STRUCTURED_SCHEMA, is_enhanced=True)
    get_structured.assert_called_once()
    get_schema.assert_not_called()


def test_resolve_cascade_mode_levels_skip_narrow_only_has_two_rungs():
    with patch("text2cypher_composer.schema_modes.get_structured_schema", return_value=STRUCTURED_SCHEMA):
        levels = resolve_cascade_mode_levels(
            MagicMock(), "exact_match", "Which miRNAs are over expressed in cancer?", skip_narrow=True
        )

    assert [level for level, _ in levels] == [CascadeModeLevel.NODES_ONLY, CascadeModeLevel.FULL]


def test_resolve_cascade_mode_levels_forwards_cache_schema():
    with patch("text2cypher_composer.schema_modes.get_structured_schema", return_value=STRUCTURED_SCHEMA) as get_structured:
        resolve_cascade_mode_levels(
            MagicMock(), "exact_match", "some question", cache_schema=False
        )

    assert get_structured.call_args.kwargs["cache_schema"] is False


def test_resolve_schema_text_forwards_cache_schema():
    with patch("text2cypher_composer.schema_modes.get_schema", return_value="SCHEMA TEXT") as get_schema:
        resolve_schema_text(MagicMock(), "schema", "some question", cache_schema=False)

    assert get_schema.call_args.kwargs["cache_schema"] is False


def test_schema_delta_includes_new_labels_whole():
    old = {"node_props": {"Gene": [{"property": "Label", "type": "STRING"}]}, "relationships": [], "rel_props": {}, "metadata": {}}
    new = {
        "node_props": {
            "Gene": [{"property": "Label", "type": "STRING"}],
            "miRNA": [{"property": "Label", "type": "STRING"}],
        },
        "relationships": [],
        "rel_props": {},
        "metadata": {},
    }
    delta = schema_delta(new, old)
    assert delta["node_props"] == {"miRNA": [{"property": "Label", "type": "STRING"}]}


def test_schema_delta_narrows_shared_label_to_new_properties_only():
    old = {"node_props": {"miRNA": [{"property": "Label", "type": "STRING"}]}, "relationships": [], "rel_props": {}, "metadata": {}}
    new = {
        "node_props": {
            "miRNA": [
                {"property": "Label", "type": "STRING"},
                {"property": "sequence_size", "type": "INTEGER"},
            ],
        },
        "relationships": [],
        "rel_props": {},
        "metadata": {},
    }
    delta = schema_delta(new, old)
    assert delta["node_props"] == {"miRNA": [{"property": "sequence_size", "type": "INTEGER"}]}


def test_schema_delta_drops_a_label_with_nothing_new():
    schema = {"node_props": {"miRNA": [{"property": "Label", "type": "STRING"}]}, "relationships": [], "rel_props": {}, "metadata": {}}
    delta = schema_delta(schema, schema)
    assert delta["node_props"] == {}
    assert delta["relationships"] == []
    assert delta["rel_props"] == {}


def test_schema_delta_relationships_are_a_set_difference():
    old = {
        "node_props": {},
        "relationships": [{"start": "Gene", "type": "TRANSCRIBED_TO", "end": "miRNA"}],
        "rel_props": {},
        "metadata": {},
    }
    new = {
        "node_props": {},
        "relationships": [
            {"start": "Gene", "type": "TRANSCRIBED_TO", "end": "miRNA"},
            {"start": "miRNA", "type": "OVER_EXPRESSED_IN", "end": "Cancer"},
        ],
        "rel_props": {},
        "metadata": {},
    }
    delta = schema_delta(new, old)
    assert delta["relationships"] == [{"start": "miRNA", "type": "OVER_EXPRESSED_IN", "end": "Cancer"}]


def test_schema_delta_relationship_properties_narrowed_to_new_only():
    old = {
        "node_props": {},
        "relationships": [],
        "rel_props": {"TRANSCRIBED_TO": [{"property": "source", "type": "STRING"}]},
        "metadata": {},
    }
    new = {
        "node_props": {},
        "relationships": [],
        "rel_props": {
            "TRANSCRIBED_TO": [
                {"property": "source", "type": "STRING"},
                {"property": "confidence", "type": "FLOAT"},
            ],
        },
        "metadata": {},
    }
    delta = schema_delta(new, old)
    assert delta["rel_props"] == {"TRANSCRIBED_TO": [{"property": "confidence", "type": "FLOAT"}]}


CHAIN_SCHEMA = {
    "node_props": {
        "NodeA": [{"property": "Label", "type": "STRING"}],
        "NodeB": [{"property": "Label", "type": "STRING"}],
        "NodeC": [{"property": "Label", "type": "STRING"}],
        "NodeD": [{"property": "Label", "type": "STRING"}],
        "NodeE": [{"property": "Label", "type": "STRING"}],  # disconnected from the chain
    },
    "relationships": [
        {"start": "NodeA", "type": "REL_AB", "end": "NodeB"},
        {"start": "NodeB", "type": "REL_BC", "end": "NodeC"},
        {"start": "NodeC", "type": "REL_CD", "end": "NodeD"},
    ],
    "rel_props": {},
    "metadata": {},
}


def test_expand_labels_by_hops_one_hop():
    assert set(expand_labels_by_hops(CHAIN_SCHEMA, ["NodeA"], hops=1)) == {"NodeA", "NodeB"}


def test_expand_labels_by_hops_two_hops():
    assert set(expand_labels_by_hops(CHAIN_SCHEMA, ["NodeA"], hops=2)) == {"NodeA", "NodeB", "NodeC"}


def test_expand_labels_by_hops_three_hops_reaches_the_whole_chain():
    assert set(expand_labels_by_hops(CHAIN_SCHEMA, ["NodeA"], hops=3)) == {"NodeA", "NodeB", "NodeC", "NodeD"}


def test_expand_labels_by_hops_never_reaches_a_disconnected_label():
    assert "NodeE" not in expand_labels_by_hops(CHAIN_SCHEMA, ["NodeA"], hops=10)


def test_expand_labels_by_hops_treats_relationships_as_undirected_for_reachability():
    # NodeD has no outgoing relationship, only an incoming NodeC->NodeD one -- still reachable
    assert set(expand_labels_by_hops(CHAIN_SCHEMA, ["NodeD"], hops=1)) == {"NodeD", "NodeC"}


def test_two_hop_expansion_prune_keeps_only_reached_labels_and_relationships():
    pruned = two_hop_expansion_prune(CHAIN_SCHEMA, ["NodeA"], hops=2)
    assert set(pruned["node_props"].keys()) == {"NodeA", "NodeB", "NodeC"}
    kept_rel_types = {r["type"] for r in pruned["relationships"]}
    assert kept_rel_types == {"REL_AB", "REL_BC"}  # REL_CD's endpoint NodeD is out of reach


_CHAIN_NARROW = {
    "node_props": {"NodeA": [{"property": "Label", "type": "STRING"}]},
    "relationships": [],
    "rel_props": {},
    "metadata": {},
}


def test_resolve_cascade_mode_levels_delta_first_rung_is_full_narrow_schema():
    with patch("text2cypher_composer.schema_modes.get_structured_schema", return_value=CHAIN_SCHEMA), \
         patch("text2cypher_composer.schema_modes.exact_match_prune", return_value=_CHAIN_NARROW):
        levels = resolve_cascade_mode_levels(MagicMock(), "exact_match", "some question", strategy="delta")

    assert levels[0][1] == format_schema(_CHAIN_NARROW, is_enhanced=True)


def test_resolve_cascade_mode_levels_delta_second_rung_is_a_two_hop_expansion():
    with patch("text2cypher_composer.schema_modes.get_structured_schema", return_value=CHAIN_SCHEMA), \
         patch("text2cypher_composer.schema_modes.exact_match_prune", return_value=_CHAIN_NARROW):
        levels = resolve_cascade_mode_levels(MagicMock(), "exact_match", "some question", strategy="delta")

    nodes_only_text = levels[1][1]
    node_props_section = nodes_only_text.split("The relationships:")[0]
    # NodeB/NodeC are within 2 hops of NodeA (the narrow rung's only label) -- new at this rung
    assert "NodeB" in node_props_section
    assert "NodeC" in node_props_section
    # NodeA was already shown at the narrow rung -- not repeated
    assert "NodeA" not in node_props_section
    # NodeD is 3 hops away -- outside the 2-hop expansion entirely
    assert "NodeD" not in nodes_only_text


def test_resolve_cascade_mode_levels_delta_third_rung_shows_what_the_two_hop_expansion_missed():
    with patch("text2cypher_composer.schema_modes.get_structured_schema", return_value=CHAIN_SCHEMA), \
         patch("text2cypher_composer.schema_modes.exact_match_prune", return_value=_CHAIN_NARROW):
        levels = resolve_cascade_mode_levels(MagicMock(), "exact_match", "some question", strategy="delta")

    full_text = levels[2][1]
    node_props_section = full_text.split("The relationships:")[0]
    # NodeD (3 hops away) and NodeE (disconnected) are outside the 2-hop expansion -- new here
    assert "NodeD" in node_props_section
    assert "NodeE" in node_props_section
    assert "NodeA" not in node_props_section
    assert "NodeB" not in node_props_section
    assert "NodeC" not in node_props_section


def test_resolve_cascade_mode_levels_delta_skip_narrow_still_seeds_the_two_hop_expansion():
    with patch("text2cypher_composer.schema_modes.get_structured_schema", return_value=CHAIN_SCHEMA), \
         patch("text2cypher_composer.schema_modes.exact_match_prune", return_value=_CHAIN_NARROW):
        levels = resolve_cascade_mode_levels(
            MagicMock(), "exact_match", "some question", strategy="delta", skip_narrow=True
        )

    assert [level for level, _ in levels] == [CascadeModeLevel.NODES_ONLY, CascadeModeLevel.FULL]
    # narrow is never emitted as its own rung, but its label (NodeA) still seeded the expansion --
    # the first emitted rung is shown in full (nothing before it to delta against)
    first_text = levels[0][1]
    assert "NodeA" in first_text and "NodeB" in first_text and "NodeC" in first_text
    assert "NodeD" not in first_text


def test_resolve_cascade_mode_levels_rejects_non_pruning_modes():
    with pytest.raises(ValueError, match="nothing to prune"):
        resolve_cascade_mode_levels(MagicMock(), "schema", "some question")
    with pytest.raises(ValueError, match="nothing to prune"):
        resolve_cascade_mode_levels(MagicMock(), "enhanced", "some question")


def test_resolve_cascade_mode_levels_requires_nlp_for_similarity():
    with pytest.raises(ValueError, match="nlp"):
        resolve_cascade_mode_levels(MagicMock(), "similarity", "some question")


def test_resolve_cascade_mode_levels_requires_llm_for_llm_pruning():
    with pytest.raises(ValueError, match="llm_pruning"):
        resolve_cascade_mode_levels(MagicMock(), "llm_pruning", "some question")


def test_resolve_cascade_mode_levels_requires_ie_engine_for_ie_extraction():
    with pytest.raises(ValueError, match="ie_engine"):
        resolve_cascade_mode_levels(MagicMock(), "ie_extraction", "some question")
