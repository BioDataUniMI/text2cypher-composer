from typing import Dict, Union

from langchain_community.graphs import Neo4jGraph

DatabaseLike = Union[Neo4jGraph, Dict[str, str]]


def resolve_database(database: DatabaseLike) -> Neo4jGraph:
    """Resolve `database` into a Neo4jGraph.

    Accepts an already-built Neo4jGraph, or a dict with
    uri/username/password/database keys (an empty dict falls back to the
    NEO4J_* environment variables).
    """
    if isinstance(database, Neo4jGraph):
        return database
    if isinstance(database, dict):
        kwargs = dict(database)
        if "uri" in kwargs and "url" not in kwargs:
            kwargs["url"] = kwargs.pop("uri")
        return Neo4jGraph(**kwargs)
    raise TypeError(
        "`database` must be a langchain_community.graphs.Neo4jGraph instance, "
        "or a dict with 'uri'/'username'/'password'(/'database') keys "
        "(an empty dict falls back to the NEO4J_* environment variables)."
    )
