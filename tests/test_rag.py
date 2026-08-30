from unittest.mock import MagicMock

from text2cypher_composer.rag import resolve_adaptive_rag_levels
from text2cypher_composer.techniques import RAGExpansionLevel


def test_resolve_adaptive_rag_levels_orders_and_sizes_rungs():
    dataset = MagicMock()
    dataset.n_results = 3
    dataset.collection.count.return_value = 7
    dataset.retrieve_examples.side_effect = lambda question, with_output, n_results: {
        "examples_text": f"EXAMPLES_{n_results}",
        "example_ids": [],
        "example_distances": [],
    }

    levels = resolve_adaptive_rag_levels(dataset, "question", with_output=False)

    assert [level for level, _ in levels] == [
        RAGExpansionLevel.MINIMAL,
        RAGExpansionLevel.MODERATE,
        RAGExpansionLevel.FULL,
    ]
    assert [retrieved["examples_text"] for _, retrieved in levels] == [
        "EXAMPLES_1",
        "EXAMPLES_3",
        "EXAMPLES_7",
    ]
    assert dataset.retrieve_examples.call_count == 3
