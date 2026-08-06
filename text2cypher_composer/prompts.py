from typing import Dict, List

from .techniques import Technique, TechniqueLike

SYSTEM_MESSAGE = "Given an input question, convert it to a Cypher query. No pre-amble."

_VANILLA = """Write a Cypher query that would answer the user's question.
Return only Cypher statement, no backticks, nothing else.

Question: {question}
Cypher query:"""

_SCHEMA = """Based on the Neo4j graph schema below,
write a Cypher query that would answer the user's question.
Return only Cypher statement, no backticks, nothing else.
{enhanced_schema}

Question: {question}
Cypher query:"""

_RAG = """Based on the examples below that provide useful
patterns for querying the graph database,
write a Cypher query that would answer the user's question.
Return only Cypher statement, no backticks, nothing else.

Examples:
{examples}

Question: {question}
Cypher query:"""

_RAG_O = """Based on the examples below that provide useful
patterns for querying the graph database (Neo4j outputs are also reported),
write a Cypher query that would answer the user's question.
Return only Cypher statement, no backticks, nothing else.

Examples:
{examples}

Question: {question}
Cypher query:"""

_SCHEMA_RAG = """Based on the Neo4j graph schema below,
write a Cypher query that would answer the user's question.
Return only Cypher statement, no backticks, nothing else.
{enhanced_schema}

Examples:
The following examples provide useful patterns for querying the graph database.
{examples}

Question: {question}
Cypher query:"""

_SCHEMA_RAG_O = """Based on the Neo4j graph schema below,
write a Cypher query that would answer the user's question.
Return only Cypher statement, no backticks, nothing else.
{enhanced_schema}

Examples:
The following examples provide useful patterns for querying
the graph database (Neo4j outputs are also reported).
{examples}

Question: {question}
Cypher query:"""

TEMPLATES = {
    Technique.VANILLA: _VANILLA,
    Technique.SCHEMA: _SCHEMA,
    Technique.RAG: _RAG,
    Technique.RAG_O: _RAG_O,
    Technique.SCHEMA_RAG: _SCHEMA_RAG,
    Technique.SCHEMA_RAG_O: _SCHEMA_RAG_O,
}


def messages_for(technique: Technique):
    return [("system", SYSTEM_MESSAGE), ("human", TEMPLATES[technique])]


def get_prompt_template(technique: TechniqueLike) -> List[Dict[str, str]]:
    """Return the parametric (unfilled) prompt for `technique`.

    Placeholders such as `{question}`, `{enhanced_schema}`, `{examples}` are
    left as literal text — this is the template, not an instantiated prompt.
    For the fully-instantiated prompt actually sent to the model on a given
    call, see `Text2CypherResult.prompt` (returned by `run()`).
    """
    t = Technique(technique)
    return [{"role": role, "content": content} for role, content in messages_for(t)]


def get_all_prompt_templates() -> Dict[str, List[Dict[str, str]]]:
    """Return `get_prompt_template` for every available technique, keyed by technique value."""
    return {t.value: get_prompt_template(t) for t in Technique}
