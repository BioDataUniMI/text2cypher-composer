from unittest.mock import MagicMock, patch

import pytest
from langchain_core.runnables import RunnableLambda

from text2cypher_composer.schema import format_schema
from text2cypher_composer.schema_modes import (
    SchemaSelection,
    _format_seen_schema,
    _merge_structured,
    exact_match_prune,
    expand_labels_by_hops,
    llm_prune,
    llm_prune_nodes_only,
    mask_entities,
    narrow_top2_relationships,
    ner_exact_match_prune,
    resolve_cascade_mode_levels,
    resolve_schema_text,
    schema_delta,
    similarity_prune,
    similarity_prune_nodes_only,
    two_hop_expansion_prune,
)
from text2cypher_composer.techniques import DEFAULT_SCHEMA_COMPONENTS, CascadeModeLevel

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


_MULTI_REL_SCHEMA = {
    "node_props": {
        "Person": [{"property": "name", "type": "STRING"}],
        "Movie": [{"property": "title", "type": "STRING"}],
    },
    "relationships": [
        {"start": "Person", "type": "DIRECTED", "end": "Movie"},
        {"start": "Person", "type": "ACTED_IN", "end": "Movie"},
        {"start": "Person", "type": "REVIEWED", "end": "Movie"},
        {"start": "Person", "type": "PRODUCED", "end": "Movie"},
    ],
    "rel_props": {
        "DIRECTED": [{"property": "year", "type": "INTEGER"}],
        "ACTED_IN": [{"property": "role", "type": "STRING"}],
        "REVIEWED": [{"property": "rating", "type": "INTEGER"}],
        "PRODUCED": [{"property": "budget", "type": "FLOAT"}],
    },
    "metadata": {},
}


def test_narrow_top2_relationships_keeps_only_the_top2_most_lexically_similar_per_pair():
    pruned = narrow_top2_relationships(_MULTI_REL_SCHEMA, "Who acted in and reviewed this movie?")
    kept_types = {r["type"] for r in pruned["relationships"]}
    # ACTED_IN and REVIEWED both share tokens with the question -- DIRECTED/PRODUCED don't
    assert kept_types == {"ACTED_IN", "REVIEWED"}
    assert set(pruned["rel_props"].keys()) == {"ACTED_IN", "REVIEWED"}


def test_narrow_top2_relationships_leaves_a_pair_with_top_k_or_fewer_patterns_untouched():
    schema = {
        "node_props": {"A": [], "B": []},
        "relationships": [{"start": "A", "type": "R1", "end": "B"}],
        "rel_props": {},
        "metadata": {},
    }
    pruned = narrow_top2_relationships(schema, "anything")
    assert pruned["relationships"] == schema["relationships"]


def test_narrow_top2_relationships_ranks_independently_per_endpoint_pair():
    schema = {
        "node_props": {},
        "relationships": [
            {"start": "A", "type": "REL_ONE", "end": "B"},
            {"start": "A", "type": "REL_TWO", "end": "B"},
            {"start": "A", "type": "REL_THREE", "end": "B"},
            {"start": "C", "type": "REL_FOUR", "end": "D"},
        ],
        "rel_props": {},
        "metadata": {},
    }
    pruned = narrow_top2_relationships(schema, "no lexical overlap with any type")
    kept_types = {r["type"] for r in pruned["relationships"]}
    # A->B keeps its top 2 (ties broken by original order); C->D's only pattern is untouched
    assert kept_types == {"REL_ONE", "REL_TWO", "REL_FOUR"}


def test_narrow_top2_relationships_leaves_node_props_untouched():
    pruned = narrow_top2_relationships(_MULTI_REL_SCHEMA, "Who acted in this movie?")
    assert pruned["node_props"] == _MULTI_REL_SCHEMA["node_props"]


def test_merge_structured_unions_node_relationship_and_rel_props():
    a = {
        "node_props": {"NodeA": [{"property": "x", "type": "STRING"}]},
        "relationships": [{"start": "NodeA", "type": "R1", "end": "NodeB"}],
        "rel_props": {"R1": [{"property": "p", "type": "STRING"}]},
        "metadata": {},
    }
    b = {
        "node_props": {
            "NodeA": [{"property": "y", "type": "INTEGER"}],  # new property for a shared label
            "NodeC": [{"property": "z", "type": "STRING"}],  # a whole new label
        },
        "relationships": [
            {"start": "NodeA", "type": "R1", "end": "NodeB"},  # duplicate, not repeated
            {"start": "NodeB", "type": "R2", "end": "NodeC"},
        ],
        "rel_props": {"R1": [{"property": "p", "type": "STRING"}], "R2": []},
        "metadata": {},
    }
    merged = _merge_structured(a, b)
    assert merged["node_props"]["NodeA"] == [
        {"property": "x", "type": "STRING"},
        {"property": "y", "type": "INTEGER"},
    ]
    assert merged["node_props"]["NodeC"] == [{"property": "z", "type": "STRING"}]
    assert merged["relationships"] == [
        {"start": "NodeA", "type": "R1", "end": "NodeB"},
        {"start": "NodeB", "type": "R2", "end": "NodeC"},
    ]
    assert set(merged["rel_props"].keys()) == {"R1", "R2"}


def test_format_seen_schema_is_empty_for_an_empty_schema():
    empty = {"node_props": {}, "relationships": [], "rel_props": {}, "metadata": {}}
    assert _format_seen_schema(empty) == ""


def test_format_seen_schema_uses_the_compact_non_enhanced_style():
    seen = {
        "node_props": {"NodeA": [{"property": "Label", "type": "STRING"}]},
        "relationships": [],
        "rel_props": {},
        "metadata": {},
    }
    text = _format_seen_schema(seen)
    assert "Schema already shown" in text
    # the terse `NodeA {Label: STRING}` style, not the enhanced `- **NodeA**` bullet style
    assert "NodeA {Label: STRING}" in text
    assert "- **NodeA**" not in text


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


_CASCADE_NODES_ONLY = {
    "node_props": {
        "NodeA": [{"property": "Label", "type": "STRING"}],
        "NodeB": [{"property": "Label", "type": "STRING"}],
        "NodeC": [{"property": "Label", "type": "STRING"}],
    },
    "relationships": [
        # NodeA->NodeB has 3 patterns -- true_narrow_top2 must trim this pair down to 2
        {"start": "NodeA", "type": "LIKES", "end": "NodeB"},
        {"start": "NodeA", "type": "FOLLOWS", "end": "NodeB"},
        {"start": "NodeA", "type": "BLOCKS", "end": "NodeB"},
        # NodeB->NodeC has only 1 pattern -- untouched by the top-2 trim
        {"start": "NodeB", "type": "PARTNERS_WITH", "end": "NodeC"},
    ],
    "rel_props": {"LIKES": [], "FOLLOWS": [], "BLOCKS": [], "PARTNERS_WITH": []},
    "metadata": {},
}

_CASCADE_FULL_SCHEMA = {
    "node_props": {
        **_CASCADE_NODES_ONLY["node_props"],
        "NodeD": [{"property": "Label", "type": "STRING"}],
        "NodeE": [{"property": "Label", "type": "STRING"}],  # disconnected
    },
    "relationships": _CASCADE_NODES_ONLY["relationships"] + [
        {"start": "NodeC", "type": "MENTORS", "end": "NodeD"},
    ],
    "rel_props": {**_CASCADE_NODES_ONLY["rel_props"], "MENTORS": []},
    "metadata": {},
}

# Lexically favors LIKES/FOLLOWS ("likes"/"follows" tokens) over BLOCKS -- see
# test_narrow_top2_relationships_keeps_only_the_top2_most_lexically_similar_per_pair above for the
# same scoring mechanics.
_CASCADE_QUESTION = "Which NodeA likes and follows NodeB?"


def _fake_nodes_only_selection(_structured_schema, _question, components=DEFAULT_SCHEMA_COMPONENTS):
    """`resolve_cascade_mode_levels` now calls `exact_match_prune` only once under
    `strategy="delta"` (for `"nodes_only"` -- `"true_narrow_top2"` is derived from it, not from a
    separate `ALL_SCHEMA_COMPONENTS` call), so a single canned return covers it."""
    return _CASCADE_NODES_ONLY


def test_resolve_cascade_mode_levels_delta_first_rung_is_built_from_nodes_only_not_the_mode_narrow():
    with patch("text2cypher_composer.schema_modes.get_structured_schema", return_value=_CASCADE_FULL_SCHEMA), \
         patch("text2cypher_composer.schema_modes.exact_match_prune", side_effect=_fake_nodes_only_selection):
        levels = resolve_cascade_mode_levels(MagicMock(), "exact_match", _CASCADE_QUESTION, strategy="delta")

    narrow_text = levels[0][1]
    # same node labels as nodes_only's own selection -- not a separately-computed, possibly
    # differently-scoped mode narrow
    assert "NodeA" in narrow_text and "NodeB" in narrow_text and "NodeC" in narrow_text
    # top-2-per-pair trim: LIKES/FOLLOWS are lexically relevant to the question and survive ...
    assert "LIKES" in narrow_text
    assert "FOLLOWS" in narrow_text
    # ... BLOCKS is the 3rd, least relevant NodeA->NodeB pattern -- dropped
    assert "BLOCKS" not in narrow_text
    # NodeB->NodeC's only pattern is untouched by the trim
    assert "PARTNERS_WITH" in narrow_text


def test_resolve_cascade_mode_levels_delta_second_rung_reveals_what_top2_held_back():
    with patch("text2cypher_composer.schema_modes.get_structured_schema", return_value=_CASCADE_FULL_SCHEMA), \
         patch("text2cypher_composer.schema_modes.exact_match_prune", side_effect=_fake_nodes_only_selection):
        levels = resolve_cascade_mode_levels(MagicMock(), "exact_match", _CASCADE_QUESTION, strategy="delta")

    nodes_only_text = levels[1][1]
    assert "Schema already shown" in nodes_only_text
    assert "Additional schema" in nodes_only_text
    delta_section = nodes_only_text.split("Additional schema")[1]
    # nodes_only shares every node label/property with true_narrow_top2 (same underlying
    # selection) -- its whole job as a fallback is revealing BLOCKS, the one relationship pattern
    # the top-2 trim held back
    assert "BLOCKS" in delta_section
    assert "LIKES" not in delta_section
    assert "FOLLOWS" not in delta_section
    assert "PARTNERS_WITH" not in delta_section


def test_resolve_cascade_mode_levels_delta_third_rung_shows_what_nodes_only_missed():
    with patch("text2cypher_composer.schema_modes.get_structured_schema", return_value=_CASCADE_FULL_SCHEMA), \
         patch("text2cypher_composer.schema_modes.exact_match_prune", side_effect=_fake_nodes_only_selection):
        levels = resolve_cascade_mode_levels(MagicMock(), "exact_match", _CASCADE_QUESTION, strategy="delta")

    full_text = levels[2][1]
    # NodeD/NodeE and the relationship reaching NodeD are entirely outside nodes_only's selection
    assert "NodeD" in full_text
    assert "NodeE" in full_text
    assert "MENTORS" in full_text
    # everything true_narrow_top2 + nodes_only already showed (across *both* previous rungs, not
    # just the immediately previous one) is reminded via the compact inventory, not repeated
    assert "Schema already shown" in full_text
    delta_section = full_text.split("Additional schema")[1]
    assert "LIKES" not in delta_section
    assert "FOLLOWS" not in delta_section
    assert "BLOCKS" not in delta_section
    assert "PARTNERS_WITH" not in delta_section


def test_resolve_cascade_mode_levels_delta_skip_narrow_starts_at_nodes_only():
    with patch("text2cypher_composer.schema_modes.get_structured_schema", return_value=_CASCADE_FULL_SCHEMA), \
         patch("text2cypher_composer.schema_modes.exact_match_prune", side_effect=_fake_nodes_only_selection):
        levels = resolve_cascade_mode_levels(
            MagicMock(), "exact_match", _CASCADE_QUESTION, strategy="delta", skip_narrow=True
        )

    assert [level for level, _ in levels] == [CascadeModeLevel.NODES_ONLY, CascadeModeLevel.FULL]
    # narrow (true_narrow_top2) is never computed/emitted -- the first emitted rung is nodes_only,
    # shown in full (nothing shown before it to delta against or hold back), BLOCKS included
    first_text = levels[0][1]
    assert first_text == format_schema(_CASCADE_NODES_ONLY, is_enhanced=True)
    assert "BLOCKS" in first_text
    assert "NodeD" not in first_text

    second_text = levels[1][1]
    assert "NodeD" in second_text and "NodeE" in second_text
    assert "Schema already shown" in second_text


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
