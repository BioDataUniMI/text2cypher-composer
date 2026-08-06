"""Cypher/result comparison metrics: Jaro-Winkler, normalized Levenshtein, Jaccard, Coverage.

Jaro-Winkler and Levenshtein compare the generated and gold Cypher *text*.
Jaccard and Coverage instead compare the two queries' *result rows*, since a
textually different (but equivalent) query should still score well. Because
Neo4j does not guarantee row order without `ORDER BY`, rows are greedily
aligned by similarity before being compared when the gold query has none.

Adapted from the bio2C/Bloom evaluation notebooks, with the alignment bug
described in `_align_rows` fixed: the original attribute-recall ("coverage")
code computed a greedy alignment but then indexed back into the *unaligned*
ground-truth list whenever the ground truth had more rows than the
prediction, silently comparing mismatched rows in that case.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Hashable, List, Optional, Set

import textdistance

Row = Dict[str, Any]


def jaro_winkler_similarity(a: str, b: str) -> float:
    """Jaro-Winkler similarity between two strings, in [0, 1] (1 = exact match)."""
    return textdistance.jaro_winkler(a or "", b or "")


def normalized_levenshtein_similarity(a: str, b: str) -> float:
    """Levenshtein similarity between two strings, in [0, 1] (1 = exact match)."""
    a, b = a or "", b or ""
    if not a and not b:
        return 1.0
    return 1.0 - (textdistance.levenshtein(a, b) / max(len(a), len(b)))


def _norm_value(v: Any) -> Hashable:
    """Normalize a single value for equality (case/whitespace/numeric-string insensitive)."""
    if isinstance(v, str):
        s = v.strip()
        try:
            return float(s)
        except ValueError:
            return s.lower()
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, Mapping):
        return frozenset(_norm_value(x) for x in v.values())
    if isinstance(v, (list, tuple, set)):
        return frozenset(_norm_value(x) for x in v)
    return v if isinstance(v, Hashable) else str(v)


def _flat_values(v: Any) -> Set[Hashable]:
    """Recursively flatten a (possibly nested) value into normalized atomic values."""
    if isinstance(v, (str, int, float)):
        return {_norm_value(v)}
    if isinstance(v, Mapping):
        out: Set[Hashable] = set()
        for x in v.values():
            out |= _flat_values(x)
        return out
    if isinstance(v, (list, tuple, set)):
        out = set()
        for x in v:
            out |= _flat_values(x)
        return out
    return {_norm_value(v)}


def _row_value_set(row: Row) -> Set[Hashable]:
    """Row -> set of top-level normalized attribute values (used for Jaccard)."""
    return {_norm_value(v) for v in row.values()}


def _row_flat_values(row: Row) -> Set[Hashable]:
    """Row -> set of all recursively-flattened normalized values (used for Coverage)."""
    out: Set[Hashable] = set()
    for v in row.values():
        out |= _flat_values(v)
    return out


def _row_jaccard(a: Set[Hashable], b: Set[Hashable]) -> float:
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def _align_rows(gt_sets: List[Set[Hashable]], pred_sets: List[Set[Hashable]]) -> List[Optional[int]]:
    """Match each ground-truth row to its best predicted row via greedy bipartite matching.

    Returns `mapping` with `len(mapping) == len(gt_sets)`: `mapping[i]` is the
    index into `pred_sets` matched to `gt_sets[i]`, or `None` if predicted ran
    out of rows (fewer rows than ground truth).

    Repeatedly picks the single highest-similarity (gt, pred) pair among all
    not-yet-matched rows, on either side, rather than resolving ground-truth
    rows in a fixed order — a fixed order can force an early row to claim the
    only good match meant for a later row (e.g. gt=[Alice, Bob, Carol],
    pred=[Carol, Alice]: resolving Bob before Carol forces Bob onto Carol's
    slot, since Alice is taken and Carol hasn't matched yet, silently
    stranding the real Carol match). This is not provably a maximum-weight
    matching, but it is correct on that class of case and — unlike a
    "reorder the longer list to align with the shorter one" approach — never
    silently skips alignment based on which side happens to be longer.
    """
    n, m = len(gt_sets), len(pred_sets)
    mapping: List[Optional[int]] = [None] * n
    if n == 0 or m == 0:
        return mapping

    pairs = sorted(
        ((_row_jaccard(gt_sets[i], pred_sets[j]), i, j) for i in range(n) for j in range(m)),
        key=lambda t: t[0],
        reverse=True,
    )

    matched_gt: Set[int] = set()
    matched_pred: Set[int] = set()
    for _, i, j in pairs:
        if i in matched_gt or j in matched_pred:
            continue
        mapping[i] = j
        matched_gt.add(i)
        matched_pred.add(j)
        if len(matched_gt) == min(n, m):
            break
    return mapping


def _row_attr_recall(gt_row: Row, pred_values: Set[Hashable]) -> float:
    """Fraction of `gt_row`'s attributes whose (flattened) value appears in `pred_values`."""
    if not gt_row:
        return 1.0
    matched = sum(1 for v in gt_row.values() if _flat_values(v) & pred_values)
    return matched / len(gt_row)


def jaccard_similarity(
    true_cypher: str, true_data: List[Row], pred_cypher: str, pred_data: List[Row]
) -> float:
    """Intersection-over-union of (row-index, normalized-value) pairs between two result sets.

    Order-sensitive (compared position-by-position) if `true_cypher`
    contains an `ORDER BY`, otherwise rows are greedily aligned by
    similarity first. Extra, unmatched rows on either side count against
    the score.
    """
    order_sensitive = "order by" in (true_cypher or "").lower()
    gt_sets = [_row_value_set(r) for r in true_data]
    pred_sets = [_row_value_set(r) for r in pred_data]

    if order_sensitive:
        aligned_gt, aligned_pred = gt_sets, pred_sets
    else:
        mapping = _align_rows(gt_sets, pred_sets)
        matched = {j for j in mapping if j is not None}
        leftover = [j for j in range(len(pred_sets)) if j not in matched]
        aligned_gt = gt_sets + [set() for _ in leftover]
        aligned_pred = [pred_sets[j] if j is not None else set() for j in mapping] + [
            pred_sets[j] for j in leftover
        ]

    total_gt = {(i, x) for i, s in enumerate(aligned_gt) for x in s}
    total_pred = {(i, x) for i, s in enumerate(aligned_pred) for x in s}
    union = total_gt | total_pred
    return 1.0 if not union else len(total_gt & total_pred) / len(union)


def coverage_similarity(
    true_cypher: str, true_data: List[Row], pred_cypher: str, pred_data: List[Row]
) -> float:
    """Recall of the ground truth's attribute values, generously maxed with `jaccard_similarity`.

    For each ground-truth row, checks what fraction of its attributes have a
    matching value anywhere in the corresponding best-aligned predicted row
    — unlike `jaccard_similarity`, extra/unmatched predicted rows are not
    penalized. The final score is the max of this recall and
    `jaccard_similarity`, matching the bio2C benchmark's "coverage" metric.
    """
    order_sensitive = "order by" in (true_cypher or "").lower()

    if order_sensitive:
        n = max(len(true_data), len(pred_data))
        if n == 0:
            recall = 1.0
        else:
            row_scores = []
            for i in range(n):
                if i < len(true_data):
                    pred_vals = _row_flat_values(pred_data[i]) if i < len(pred_data) else set()
                    row_scores.append(_row_attr_recall(true_data[i], pred_vals))
                else:
                    pred_vals = _row_flat_values(pred_data[i])
                    row_scores.append(0.0 if pred_vals else 1.0)
            recall = sum(row_scores) / n
    else:
        gt_flat = [_row_flat_values(r) for r in true_data]
        pred_flat = [_row_flat_values(r) for r in pred_data]
        mapping = _align_rows(gt_flat, pred_flat)

        total_attrs = 0
        matched_attrs = 0.0
        for i, gt_row in enumerate(true_data):
            total_attrs += len(gt_row)
            j = mapping[i]
            pred_vals = pred_flat[j] if j is not None else set()
            matched_attrs += _row_attr_recall(gt_row, pred_vals) * len(gt_row)
        recall = 1.0 if total_attrs == 0 else matched_attrs / total_attrs

    return max(recall, jaccard_similarity(true_cypher, true_data, pred_cypher, pred_data))
