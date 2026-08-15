"""Cypher "rescue": retry generation with an error-aware fix-up prompt.

Ported from the miRNAKG rescue-prompt notebook: when the first generated
query is invalid (fails to execute) or comes back empty, a second prompt
— reusing the same schema/examples context as the original generation, plus
the bad query and an error message — asks the model to fix it. `error_message`
concatenates every signal `run()` has about why the query needs rescuing: the
native Neo4j error (if it failed to execute) and any Neo4j notifications
(warnings) observed during execution, exactly like the notebook's
`execute_query_with_warnings`, plus CyVer's validation report
(`validate_cypher`), which `run()` also computes for every query — combining
both its warning-level notifications and hard errors.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .validation import CypherValidationReport

_RESCUE_SYSTEM = (
    "You are an assistant that fixes invalid or empty Cypher queries. Given "
    "an input question, fix and return the Cypher query. No pre-amble."
)

_RESCUE_VANILLA = """Fix this Cypher statement to query a graph database.

The question that should have been answered is: {question}
The generated Cypher statement was: {query}
The error message is: {error_message}
Fixed Cypher query:"""

_RESCUE_SCHEMA = """Based on the Neo4j graph schema below,
fix this Cypher statement to query a graph database.
{enhanced_schema}

The question that should have been answered is: {question}
The generated Cypher statement was: {query}
The error message is: {error_message}
Fixed Cypher query:"""

_RESCUE_RAG = """Based on the examples below,
fix this Cypher statement to query a graph database.

Examples:
{examples}

The question that should have been answered is: {question}
The generated Cypher statement was: {query}
The error message is: {error_message}
Fixed Cypher query:"""

_RESCUE_SCHEMA_RAG = """Based on the Neo4j graph schema below,
fix this Cypher statement to query a graph database.
{enhanced_schema}

Examples:
{examples}

The question that should have been answered is: {question}
The generated Cypher statement was: {query}
The error message is: {error_message}
Fixed Cypher query:"""


def rescue_messages(uses_schema: bool, uses_rag: bool) -> List[tuple]:
    """The rescue prompt's messages, mirroring the same schema/examples context as generation."""
    if uses_schema and uses_rag:
        template = _RESCUE_SCHEMA_RAG
    elif uses_schema:
        template = _RESCUE_SCHEMA
    elif uses_rag:
        template = _RESCUE_RAG
    else:
        template = _RESCUE_VANILLA
    return [("system", _RESCUE_SYSTEM), ("human", template)]


def _format_metadata(label: str, entries: List[Dict[str, str]]) -> str:
    if not entries:
        return ""
    joined = "; ".join(f"{e.get('code', '')}: {e.get('description', '')}" for e in entries)
    return f"{label}: {joined}"


def build_error_message(
    executed: bool,
    result: Optional[List[Dict[str, Any]]],
    validation: CypherValidationReport,
    execution_error: Optional[str] = None,
    execution_warnings: Optional[List[str]] = None,
) -> str:
    """Build a rescue `error_message` from every signal available about the failure.

    Concatenates, in order: the native Neo4j error (if the query didn't
    execute), every Neo4j notification observed during execution (deprecated
    syntax, unknown labels/relationship-types/properties, cartesian products,
    ...), an "empty result set" note if it executed but returned nothing, and
    CyVer's validation report (both its warning-level notifications and hard
    errors).
    """
    parts = []
    if execution_error:
        parts.append(execution_error)
    if execution_warnings:
        parts.extend(execution_warnings)
    if executed and not result:
        parts.append("Empty result set.")
    for label, entries in (
        ("Syntax", validation.syntax_metadata),
        ("Schema", validation.schema_metadata),
        ("Properties", validation.properties_metadata),
    ):
        formatted = _format_metadata(label, entries)
        if formatted:
            parts.append(formatted)
    if not parts:
        parts.append("The query did not return the expected result.")
    return "\n".join(parts)


def needs_rescue(executed: bool, result: Optional[List[Dict[str, Any]]], validation: CypherValidationReport) -> bool:
    """Whether a query is "invalid or empty" and so worth a rescue attempt."""
    if not executed:
        return True
    if not result:
        return True
    if not validation.syntax_valid:
        return True
    return False
