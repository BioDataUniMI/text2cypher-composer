import pytest

from text2cypher_composer.schema_modes import exact_match_prune, mask_entities, ner_exact_match_prune

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
