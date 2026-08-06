import pytest

from text2cypher_composer import (
    coverage_similarity,
    jaccard_similarity,
    jaro_winkler_similarity,
    normalized_levenshtein_similarity,
)


def test_string_metrics_exact_match():
    assert jaro_winkler_similarity("MATCH (n) RETURN n", "MATCH (n) RETURN n") == 1.0
    assert normalized_levenshtein_similarity("MATCH (n) RETURN n", "MATCH (n) RETURN n") == 1.0


def test_string_metrics_both_empty():
    assert jaro_winkler_similarity("", "") == 1.0
    assert jaro_winkler_similarity(None, None) == 1.0
    assert normalized_levenshtein_similarity("", "") == 1.0
    assert normalized_levenshtein_similarity(None, None) == 1.0


def test_string_metrics_one_empty_one_not():
    assert 0.0 <= jaro_winkler_similarity("", "abc") < 1.0
    assert normalized_levenshtein_similarity("", "abc") == 0.0


@pytest.mark.parametrize("cypher", ["MATCH (n) RETURN n", "MATCH (n) RETURN n ORDER BY n.name"])
def test_row_metrics_both_empty(cypher):
    assert jaccard_similarity(cypher, [], cypher, []) == 1.0
    assert coverage_similarity(cypher, [], cypher, []) == 1.0


def test_row_metrics_gt_empty_pred_nonempty():
    # vacuous ground truth: jaccard penalizes the unmatched extra row, coverage doesn't
    cypher = "MATCH (n) RETURN n"
    pred = [{"n": "extra"}]
    assert jaccard_similarity(cypher, [], cypher, pred) == 0.0
    assert coverage_similarity(cypher, [], cypher, pred) == 1.0


def test_row_metrics_gt_nonempty_pred_empty():
    cypher = "MATCH (n) RETURN n"
    gt = [{"n": "a"}]
    assert jaccard_similarity(cypher, gt, cypher, []) == 0.0
    assert coverage_similarity(cypher, gt, cypher, []) == 0.0


def test_row_metrics_exact_match():
    cypher = "MATCH (n) RETURN n"
    data = [{"n": "a"}, {"n": "b"}]
    assert jaccard_similarity(cypher, data, cypher, data) == 1.0
    assert coverage_similarity(cypher, data, cypher, data) == 1.0


def test_alignment_regression_reordered_rows():
    """Regression test for the bio2C 'coverage' alignment bug.

    The original notebook code computed a greedy row alignment but then
    indexed back into the *unaligned* ground-truth list whenever ground
    truth had more rows than the prediction — silently comparing mismatched
    rows. With gt=[Alice, Bob, Carol] and pred=[Carol, Alice] (a subset, out
    of order), the correct bipartite-matched Jaccard score is 2/3 matched
    rows contributing, not the ~0.0/0.33 a naive positional or
    first-available-slot match would give.
    """
    cypher = "MATCH (p:Person) RETURN p.name AS name"
    gt = [{"name": "Alice"}, {"name": "Bob"}, {"name": "Carol"}]
    pred = [{"name": "Carol"}, {"name": "Alice"}]

    score = jaccard_similarity(cypher, gt, cypher, pred)
    assert score == pytest.approx(2 / 3)


def test_order_sensitive_when_order_by_present():
    cypher = "MATCH (n) RETURN n ORDER BY n.name"
    gt = [{"n": "a"}, {"n": "b"}]
    pred = [{"n": "b"}, {"n": "a"}]  # right rows, wrong order
    assert jaccard_similarity(cypher, gt, cypher, pred) < 1.0
