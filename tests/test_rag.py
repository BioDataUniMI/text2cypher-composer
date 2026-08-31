from unittest.mock import MagicMock

from text2cypher_composer.rag import resolve_adaptive_rag_levels
from text2cypher_composer.techniques import RAGExpansionLevel


def test_resolve_adaptive_rag_levels_orders_and_sizes_rungs():
    dataset = MagicMock()
    dataset.n_results = 3
    dataset.collection.count.return_value = 100  # large enough that no rung gets capped
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
    # MINIMAL = n_results (today's normal, non-adaptive count); MODERATE = 2x; FULL = 5x
    assert [retrieved["examples_text"] for _, retrieved in levels] == [
        "EXAMPLES_3",
        "EXAMPLES_6",
        "EXAMPLES_15",
    ]
    assert dataset.retrieve_examples.call_count == 3


def test_resolve_adaptive_rag_levels_never_exceeds_the_collection_size():
    dataset = MagicMock()
    dataset.n_results = 3
    dataset.collection.count.return_value = 8  # smaller than 2x (6) and 5x (15) of n_results
    dataset.retrieve_examples.side_effect = lambda question, with_output, n_results: {
        "examples_text": f"EXAMPLES_{n_results}",
        "example_ids": [],
        "example_distances": [],
    }

    levels = resolve_adaptive_rag_levels(dataset, "question", with_output=False)

    sizes = [int(retrieved["examples_text"].split("_")[1]) for _, retrieved in levels]
    assert sizes == [3, 6, 8]  # FULL is capped at the collection's actual size, never "everything"
    assert sizes == sorted(sizes)  # monotonically non-decreasing across rungs
