"""Post-execution semantic self-verification: is a generated Cypher query actually right?

A different, orthogonal signal from `rescue.needs_rescue`'s mechanical checks (execution
success, non-empty result, CyVer syntax validity): a query can execute cleanly, return rows, and
pass CyVer, yet still not answer what was asked (wrong direction on a relationship, an aggregate
over the wrong property, a filter that's subtly too broad/narrow, ...). `verify_semantics` asks a
model to review `(question, cypher, result)` after the fact and judge whether it actually answers
the question — the same way a human reviewer would, and often the same model that generated the
query is able to catch its own mistake on a second look.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

_VERIFICATION_SYSTEM = (
    "You review a Cypher query written to answer a natural-language question against a Neo4j "
    "graph database. Given the question, the generated query, and the rows it returned, judge "
    "whether the query actually answers the question -- not just whether it runs."
)

_VERIFICATION_TEMPLATE = """Question: {question}
Cypher query: {cypher}
Query result (rows returned; "(no rows)" if empty): {result}
{criteria_block}
Does this query correctly answer the question?"""

_MAX_PREVIEW_ROWS = 5


class SemanticVerification(BaseModel):
    """A model's judgment of whether a generated Cypher query actually answers its question."""

    answers_question: bool = Field(
        description="Whether the Cypher query, and the rows it returned, actually answer the question."
    )
    reasoning: str = Field(description="A short explanation for the judgment.")


def verify_semantics(
    llm: Any,
    question: str,
    cypher: str,
    result: Optional[List[Dict[str, Any]]],
    criteria: Optional[str] = None,
) -> SemanticVerification:
    """Ask `llm` (via structured/JSON-schema output) whether `cypher` actually answers `question`.

    `result` is the query's returned rows (`None`/empty if it didn't execute or came back empty)
    — truncated to `_MAX_PREVIEW_ROWS` so a large result set doesn't blow the prompt budget.
    `criteria`, if given, is appended as extra evaluation guidance (e.g. "the answer must include
    units", "reject aggregations over the wrong property"). `llm` must support
    `.with_structured_output()` — pass it through `llm.resolve_pruning_model` first, the same
    resolution `schema_mode="llm_pruning"` uses for its judge model.
    """
    criteria_block = f"\nAdditional evaluation criteria: {criteria}\n" if criteria else ""
    prompt = ChatPromptTemplate.from_messages(
        [("system", _VERIFICATION_SYSTEM), ("human", _VERIFICATION_TEMPLATE)]
    )
    chain = prompt | llm.with_structured_output(SemanticVerification, method="json_schema")
    preview_rows = (result or [])[:_MAX_PREVIEW_ROWS]
    return chain.invoke(
        {
            "question": question,
            "cypher": cypher,
            "result": preview_rows if preview_rows else "(no rows)",
            "criteria_block": criteria_block,
        }
    )
