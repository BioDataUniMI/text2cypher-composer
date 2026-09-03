"""Generates demo_text2cypher_composer.ipynb.

The demo notebook is never hand-edited: it's fully rebuilt from this script,
so it stays consistent (numbering, cross-references) as sections are added or
reordered. To change the notebook, edit this file and re-run it:

    python3 scripts/build_demo_notebook.py

See CONTRIBUTING.md for when this needs updating (e.g. adding a new
technique).
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""\
# Demo: `text2cypher_composer`

This notebook shows how to use the **`text2cypher_composer`** library to translate a
natural-language question into an executable **Cypher** query against a **Neo4j** database,
using the six prompting strategies from the bio2C benchmark:

| `technique`       | Uses schema | Uses `dataset` (RAG) |
|-------------------|:-----------:|:---------------------:|
| `"vanilla"`       |             |                        |
| `"Schema"`        |      ✓      |                        |
| `"RAG"`           |             |           ✓            |
| `"RAG+O"`         |             |           ✓            |
| `"Schema+RAG"`    |      ✓      |           ✓            |
| `"Schema+RAG+O"`  |      ✓      |           ✓            |

**Prerequisites:**
- the package installed, with the `rag` extra (this notebook builds and queries a RAG dataset in
  §5/§6): `pip install "text2cypher-composer[rag]"` — or, if you've cloned the repo and want an
  editable install instead, `pip install -e ".[rag]"` from the repo root
- a valid `OPENAI_API_KEY` environment variable (used both for the `"gpt-4o"` model and for the
  RAG embeddings)
- network access to the test Neo4j database used throughout the bio2C notebooks (credentials
  already included below, same as in the original notebooks)
""")

code("""\
# (If not done already) install the package, with the `rag` extra (this notebook builds and
# queries a RAG dataset in §5/§6) -- the base install alone is enough for non-RAG components.
# %pip install "text2cypher-composer[rag]"
#
# Cloned the repo instead and want an editable install that picks up local changes?
# %pip install -e ".[rag]"
""")

code("""\
import os

# The library uses ChatOpenAI (for the model) and OpenAIEmbeddings (for RAG),
# so a valid OpenAI API key is required.
#
# NOTE: this assigns the variable unconditionally, overriding any stale/invalid
# OPENAI_API_KEY that may already be exported in your shell or kernel environment.
# (os.environ.setdefault() would silently keep an existing value instead, which
# is a common source of confusing 401 "Incorrect API key" errors.)
os.environ["OPENAI_API_KEY"] = "...YOUR_OPENAI_API_KEY..."

from text2cypher_composer import run, show, RAGDataset, Technique
""")

md("""\
## 1. Neo4j database configuration

`database` can be:
- a dict with keys `uri` (or `url`), `username`, `password`, `database`;
- an empty dict `{}`, in which case the `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` /
  `NEO4J_DATABASE` environment variables are used instead;
- or an already-built `langchain_community.graphs.Neo4jGraph` instance.

Here we use the same (read-only) test database used in the evaluation notebooks under
`bio2C/evaluating_text2cypher`.
""")

code("""\
database = {
    "uri": "neo4j+s://helix.biodata.di.unimi.it:7687",
    "username": "Text2Cypher",
    "password": "Text2Cypher",
    "database": "mirnakgt2c",
}
""")

md("""\
## 2. The test natural-language question

We reuse the same example question already present in the original notebooks, so results are
easy to compare.
""")

code("""\
input_NL = "How many miRNAs have the keyword 'precursor' in the label and a sequence size under 100 nucleotides?"
""")

md("""\
`show()` (imported above, alongside `run`) is a pretty-printer for a `Text2CypherResult`: the
generated Cypher, its result rows, the always-populated CyVer validation report, and — when
present — rescue/`cascade_mode`/`adaptive_rag`/`self_verification` details. Pass
`show_prompt=True` to also print the exact fully-instantiated prompt(s) sent to the model.
""")

md("""\
## 3. `"vanilla"` technique

The model generates the Cypher query with no extra context: just the question.

Every `Text2CypherResult` also carries `prompt` — the exact messages sent to the model, with
all placeholders already substituted in (here just the question; schema/examples get filled in
too, for the techniques that use them — see §4 and §6). `show(..., show_prompt=True)` prints it.
""")

code("""\
result_vanilla = run(
    input_NL=input_NL,
    model="gpt-4o",
    database=database,
    technique="vanilla",
)
show(result_vanilla, show_prompt=True)
""")

md("""\
### 3.1 Previewing the prompt without running it (`dry_run`)

`dry_run=True` builds and returns the fully-instantiated `prompt` — schema resolved, RAG
examples retrieved, exactly as it would be for a real call — but stops there: no generation
call, no Cypher execution, no CyVer validation, no rescue. `cypher`/`initial_cypher`/`result`/
`validation` are all `None` and `executed` is `False`. Useful to sanity-check what a given
`technique`/`schema_mode`/`dataset` combination would actually send the model, without spending
an API call (or a database write, for techniques that execute) on it.
""")

code("""\
preview = run(
    input_NL=input_NL,
    model="gpt-4o",
    database=database,
    technique="vanilla",
    dry_run=True,
)
print(preview.dry_run, preview.cypher, preview.executed)
show(preview, show_prompt=True)
""")

md("""\
## 4. `"Schema"` technique: schema representation modes

The prompt is enriched with the graph's schema, extracted on the fly by querying the database
via APOC. `schema_mode` controls *how* that schema is derived/pruned before being placed in the
prompt — see the README for the full list. `"schema"` (plain, no per-property stats) is the
default when `schema_mode` is omitted.
""")

md("""\
### 4.1 `"schema"` (default) — plain schema, no per-property stats
""")

code("""\
result_schema = run(
    input_NL=input_NL,
    model="gpt-4o",
    database=database,
    technique="Schema",
    # schema_mode="schema" is the default -- equivalent to omitting it.
)
show(result_schema)

print("\\n--- Schema (first lines) ---")
print("\\n".join(result_schema.schema.splitlines()[:15]))
""")

md("""\
### 4.2 `"enhanced"` — adds per-property examples/min-max stats
""")

code("""\
result_enhanced = run(
    input_NL=input_NL,
    model="gpt-4o",
    database=database,
    technique="Schema",
    schema_mode="enhanced",
)
show(result_enhanced)

print("\\n--- Schema (first lines) ---")
print("\\n".join(result_enhanced.schema.splitlines()[:15]))
""")

md("""\
### 4.3 `"exact_match"` — pruned to labels mentioned in the question

Prunes the enhanced schema down to node labels mentioned (substring match) in the question,
plus relationships between two mentioned labels — falling back to the full schema if nothing
matches. No extra dependency needed. Compare the (usually much shorter) schema below to 4.1/4.2.
""")

code("""\
result_exact_match = run(
    input_NL=input_NL,
    model="gpt-4o",
    database=database,
    technique="Schema",
    schema_mode="exact_match",
)
show(result_exact_match)

print("\\n--- Pruned schema ---")
print(result_exact_match.schema)
""")

md("""\
### 4.3.1 `schema_components` — choosing which schema elements are matched

By default, `"exact_match"` (and `"ner_exact_match"`, `"ie_extraction"`) only match **entity
types** (node labels) against the question, as in 4.3 above. `schema_components` widens that to
relationship types and/or node/relationship properties — see `SchemaComponent`. Entity types
always anchor the selection; the other components only narrow further what's kept once a label
is selected. Compare each pruned schema below to 4.3's (entity types only).
""")

code("""\
from text2cypher_composer import list_schema_components

list_schema_components()
""")

code("""\
# Entity types (default) + relationship types: a relationship is now kept when its *type* is
# mentioned in the question, not just when both its endpoint labels are.
result_components_rel = run(
    input_NL=input_NL,
    model="gpt-4o",
    database=database,
    technique="Schema",
    schema_mode="exact_match",
    schema_components=["entity_types", "relationship_types"],
)
show(result_components_rel)

print("\\n--- Pruned schema (+ relationship_types) ---")
print(result_components_rel.schema)
""")

code("""\
# Entity types + node properties: each selected label's properties are narrowed down to the
# ones actually mentioned in the question (falling back to all of them if none are).
result_components_node_props = run(
    input_NL=input_NL,
    model="gpt-4o",
    database=database,
    technique="Schema",
    schema_mode="exact_match",
    schema_components=["entity_types", "node_properties"],
)
show(result_components_node_props)

print("\\n--- Pruned schema (+ node_properties) ---")
print(result_components_node_props.schema)
""")

code("""\
# Entity types + relationship properties: same narrowing, but for the properties of any kept
# relationship type.
result_components_rel_props = run(
    input_NL=input_NL,
    model="gpt-4o",
    database=database,
    technique="Schema",
    schema_mode="exact_match",
    schema_components=["entity_types", "relationship_properties"],
)
show(result_components_rel_props)

print("\\n--- Pruned schema (+ relationship_properties) ---")
print(result_components_rel_props.schema)
""")

md("""\
### 4.4 `"ner_exact_match"` — NER-masked pruning

Same as `"exact_match"`, but named entities in the question are first masked with their entity
type (so an entity *value*, e.g. a specific gene name, can't accidentally get confused with a
schema field name) via a **user-supplied** NLP pipeline passed as `nlp` — this library never
imports spaCy itself, it only calls whatever `nlp` object you pass in.

Demonstration-only cell (requires `pip install spacy` plus a downloaded biomedical NER model,
e.g. `en_ner_bionlp13cg_md`, not guaranteed available in this environment — so it is not
executed here).
""")

code("""\
# import spacy
# nlp = spacy.load("en_ner_bionlp13cg_md")  # a biomedical NER model
#
# result_ner = run(
#     input_NL=input_NL,
#     model="gpt-4o",
#     database=database,
#     technique="Schema",
#     schema_mode="ner_exact_match",
#     nlp=nlp,
# )
# show(result_ner)
""")

md("""\
### 4.5 `"similarity"` — word-vector-similarity pruning

Prunes the *base* schema to labels/types/properties whose word-vector similarity to the
question clears `similarity_threshold` (default `0.5`) — also via a user-supplied `nlp`, this
time one with word vectors (e.g. spaCy's `en_core_web_md`) rather than NER. Unlike
`"exact_match"`, there's no fallback: an unrelated question can prune to an empty schema.

Demonstration-only cell, for the same reason as 4.4 (needs `pip install spacy` plus a
downloaded model with word vectors).
""")

code("""\
# import spacy
# nlp = spacy.load("en_core_web_md")  # has word vectors
#
# result_similarity = run(
#     input_NL=input_NL,
#     model="gpt-4o",
#     database=database,
#     technique="Schema",
#     schema_mode="similarity",
#     nlp=nlp,
#     similarity_threshold=0.6,
# )
# show(result_similarity)
""")

md("""\
### 4.6 `"llm_pruning"` — let the model prune its own schema

Asks `model` itself which node labels and relationship types are relevant, via structured
(JSON-schema-mode) output — so it needs no extra NLP dependency, just a model that supports
`.with_structured_output()` (like `ChatOpenAI`). Hallucinated labels/types not actually in the
schema are silently dropped; it falls back to the full schema if nothing is selected.
""")

code("""\
result_llm_pruning = run(
    input_NL=input_NL,
    model="gpt-4o",
    database=database,
    technique="Schema",
    schema_mode="llm_pruning",
)
show(result_llm_pruning)

print("\\n--- LLM-pruned schema ---")
print(result_llm_pruning.schema)
""")

md("""\
### 4.7 `"ie_extraction"` — schema-grounded information extraction

Instead of substring-matching the question against schema names, `"ie_extraction"` runs
schema-grounded **information extraction** (NER for entity types, relation extraction for
relationship types, and — if requested via `schema_components` — attribute extraction for
properties) over the question, via a user-supplied `ie_engine`, and keeps exactly what it found.

`structured_schema_to_linkml` first converts the graph's structured schema into a
[LinkML](https://linkml.io/) YAML schema (class names match the Neo4j labels/types/properties
*verbatim* — no ontology grounding, since `ie_prune` only needs entity/relation/attribute
presence). `ie_engine` is then called as `ie_engine(schema_yaml, question)` and must return a
dict shaped like [SchemaLink](https://github.com/BioDataUniMI/schemalink-engine)'s output:
`{class_name: {"mentions": [{...}, ...]}}`.

`schemalink_ie_engine()` is a ready-made `ie_engine` backed by the real `schemalink-engine`
package (`pip install schemalink-engine`, then `schemalink api-key set sk-...`):

```python
from text2cypher_composer import schemalink_ie_engine

result = run(
    input_NL=input_NL,
    model="gpt-4o",
    database=database,
    technique="Schema",
    schema_mode="ie_extraction",
    schema_components=["entity_types"],
    ie_engine=schemalink_ie_engine(),  # include_node_types/include_relationship_types/
)                                      # include_properties booleans filter what's asked for
```

The cells below use a tiny **stand-in** `ie_engine` instead (a canned dict, not a real
extraction call) so this notebook runs without a SchemaLink install/OpenAI-billed extraction
call — the mechanics (what `ie_prune` does with the output) are identical either way. See the
README's `ie_engine` section for the exact contract `schemalink_ie_engine()` satisfies.
""")

code("""\
# A small mock structured schema (Gene/miRNA/transcribed_to, same shapes as §5.1's mock RAG
# dataset) -- used here instead of the live database schema so the property names below are
# guaranteed to match, with no dependency on what's actually in the graph.
mock_structured_schema = {
    "node_props": {
        "Gene": [{"property": "Label", "type": "STRING"}],
        "miRNA": [
            {"property": "Label", "type": "STRING"},
            {"property": "sequence_size", "type": "INTEGER"},
        ],
    },
    "rel_props": {
        "transcribed_to": [{"property": "source", "type": "STRING"}],
    },
    "relationships": [{"start": "Gene", "type": "transcribed_to", "end": "miRNA"}],
    "metadata": {},
}

from text2cypher_composer import structured_schema_to_linkml

print(structured_schema_to_linkml(
    mock_structured_schema,
    components=["entity_types", "relationship_types", "node_properties", "relationship_properties"],
))
""")

code("""\
from text2cypher_composer import ie_prune

def stub_ie_engine(schema_yaml, question):
    \"\"\"Stand-in for a real SchemaLink adapter -- same output shape, canned instead of extracted.\"\"\"
    return {
        "miRNA": {"mentions": [{"Label": "precursor", "sequence_size": "100"}]},
        "Gene": {"mentions": []},          # not mentioned in the question -> pruned out
        "transcribed_to": {"mentions": []},  # ditto
    }

ie_question = "How many miRNAs have the keyword 'precursor' in the label and a sequence size under 100 nucleotides?"

pruned = ie_prune(
    mock_structured_schema,
    ie_question,
    stub_ie_engine,
    components=["entity_types", "relationship_types", "node_properties"],
)
pruned
# -> only "miRNA" survives (the only class with mentions), narrowed to the "Label"/"sequence_size"
#    properties actually present in its mention -- "Gene" and "transcribed_to" are pruned out
#    entirely, same as an unmentioned label in "exact_match".
""")

md("""\
And through `run()`, exactly like every other `schema_mode` (using the same stand-in engine,
restricted to `entity_types` here since it only needs to get the live database's actual node
labels right, which — unlike the mock schema above — this demo doesn't control):
""")

code("""\
def live_stub_ie_engine(schema_yaml, question):
    return {"miRNA": {"mentions": [{"Label": "precursor"}]}}

result_ie = run(
    input_NL=input_NL,
    model="gpt-4o",
    database=database,
    technique="Schema",
    schema_mode="ie_extraction",
    schema_components=["entity_types"],
    ie_engine=live_stub_ie_engine,
)
show(result_ie)

print("\\n--- IE-pruned schema ---")
print(result_ie.schema)
""")

md("""\
### 4.8 Caching the schema across many `run()` calls (`cache_schema`)

Extracting a graph's schema from Neo4j isn't free: the base extraction alone is 3+ APOC-backed
queries, and the "enhanced" schema every pruning `schema_mode` above uses (§4.3-4.7) adds **one
more query per node label and per relationship type** on top of that. None of this changes across
the many questions of a benchmark run against the same database, so re-extracting it on every
`run()` call — multiplied by every `cascade_mode` rung (§9) — is pure overhead that dominates
wall-clock time once you're testing hundreds or thousands of questions.

`cache_schema=True` (the **default**) caches the extracted schema per `(database, is_enhanced,
sample)` and reuses it across every `run()` call made against the same graph instance — only the
extraction step is cached, not the per-question filtering/pruning or any LLM call, so this cuts
technical overhead, not model cost. To make the savings concrete without a live database, we wrap
`database` in a tiny proxy that counts how many times `.query(...)` actually reaches it:
""")

code("""\
from text2cypher_composer import clear_schema_cache, get_structured_schema

class CountingGraph:
    \"\"\"Wraps a Neo4jGraph and counts how many times .query() actually reaches it.\"\"\"
    def __init__(self, graph):
        self._graph = graph
        self.query_count = 0

    def query(self, *args, **kwargs):
        self.query_count += 1
        return self._graph.query(*args, **kwargs)

counting_database = CountingGraph(database)
clear_schema_cache(counting_database)  # start this demo from a clean cache

get_structured_schema(counting_database, is_enhanced=True)
print("Neo4j queries after the 1st call (cache miss):", counting_database.query_count)

get_structured_schema(counting_database, is_enhanced=True)
print("Neo4j queries after the 2nd call (cache hit):  ", counting_database.query_count)  # unchanged

get_structured_schema(counting_database, is_enhanced=True, cache_schema=False)
print("Neo4j queries after a cache_schema=False call: ", counting_database.query_count)  # grew again
""")

md("""\
Pass `cache_schema=False` to `run()` if your schema genuinely changes mid-experiment (e.g. writing
to the graph between questions), or call `clear_schema_cache(graph)` — or `clear_schema_cache()`
with no argument to drop every graph's cached schema — to invalidate an already-cached entry
instead of disabling caching outright. This also fixed a pre-existing inefficiency in
`cascade_mode` (§9): its `"full"` rung used to re-extract the schema a *second* time within the
same `run()` call instead of reusing what the `"narrow"`/`"nodes_only"` rungs already fetched —
that's now free regardless of `cache_schema`.
""")

md("""\
## 5. Building your own RAG dataset

The RAG techniques (§6) need a `RAGDataset`: a Chroma collection of embedded questions, plus
the sibling `CypherQueries/`/`Neo4jOutputs/` files holding each one's golden Cypher/output. If
you don't already have a bio2C-style benchmark on disk, you can build one from your own
examples in two steps. Here we do it with a small **mock** dataset (5 entries) — in practice
you'd use your own validated (question, cypher) pairs, and as many as you like.
""")

md("""\
### 5.1 A mock example set

A DataFrame with `"question"` and `"cypher"` columns — the same schema as the running example
(`Gene`/`miRNA`/`Cancer` nodes, `transcribed_to`/`over_expressed_in` relationships). Some of
these may return no rows depending on what's actually in the graph — that's fine, it doesn't
break anything below.
""")

code("""\
import pandas as pd

mock_df = pd.DataFrame([
    {
        "question": "Which cancers show over-expression of miRNA transcribed from the 'MIR411' gene?",
        "cypher": "MATCH (g:Gene {Label: 'MIR411'})-[:transcribed_to]->(m:miRNA)-[:over_expressed_in]->(c:Cancer) RETURN c.Label AS Cancer, m.Label AS miRNA",
    },
    {
        "question": "How many genes are there in total?",
        "cypher": "MATCH (g:Gene) RETURN count(g) AS GeneCount",
    },
    {
        "question": "List all distinct cancer labels.",
        "cypher": "MATCH (c:Cancer) RETURN DISTINCT c.Label AS Cancer",
    },
    {
        "question": "Which miRNAs are transcribed from the gene 'MIR21'?",
        "cypher": "MATCH (g:Gene {Label: 'MIR21'})-[:transcribed_to]->(m:miRNA) RETURN m.Label AS miRNA",
    },
    {
        "question": "How many miRNAs are over-expressed in 'lung cancer'?",
        "cypher": "MATCH (m:miRNA)-[:over_expressed_in]->(c:Cancer {Label: 'lung cancer'}) RETURN count(m) AS Count",
    },
])
mock_df
""")

md("""\
### 5.2 Generate the NLquestions/CypherQueries/Neo4jOutputs files

`build_rag_example_files` writes one `question_i.txt`/`cypher_i.txt` pair per row, plus an
`output_i.txt` obtained by actually running each query against `database` (a `LIMIT` is added
if missing; failures are recorded as an error message rather than raising).
""")

code("""\
from text2cypher_composer import build_rag_example_files

example_files = build_rag_example_files(
    mock_df,
    name="mock",              # subfolder name — call this again with a different name to add more groups
    database=database,
    root="mock_rag_dataset",  # NLquestions/CypherQueries/Neo4jOutputs are created under here
)
example_files
""")

md("""\
### 5.3 Index the questions into Chroma

`RAGDataset.index_from_root` embeds every file under `root/NLquestions/` and stores them in a
Chroma collection at `root/chroma_db`, then returns a ready-to-use `RAGDataset`.

The embedding backend is pluggable via `embedding_model`:
- an **OpenAI** embedding model id, e.g. `"text-embedding-3-large"` (the default) — needs
  `OPENAI_API_KEY`;
- a **HuggingFace/sentence-transformers** model id, e.g.
  `"sentence-transformers/all-mpnet-base-v2"` — runs locally, no API key needed. A `"/"` in the
  string is what selects this backend. Needs the optional `local-embeddings` extra
  (`pip install "text2cypher-composer[local-embeddings]"`);
- or an already-built LangChain `Embeddings` instance, for anything else.

**This does not re-embed from scratch every time.** If a collection already exists at
`root/chroma_db`, it's loaded and reused as-is by default — no API calls — so re-running this
cell (or a later session pointed at the same `root`) is instant. Pass `rebuild=True` only when
you actually want to re-embed, e.g. after adding new examples under `root/NLquestions/`.
""")

code("""\
dataset = RAGDataset.index_from_root(
    "mock_rag_dataset",
    embedding_model="text-embedding-3-large",
)
dataset
""")

md("""\
Using a local model instead just means a different `embedding_model` string (demonstration-only
here, since it needs the optional extra and downloads model weights on first use):
""")

code("""\
# dataset_local = RAGDataset.index_from_root(
#     "mock_rag_dataset_local",
#     embedding_model="sentence-transformers/all-mpnet-base-v2",
# )
""")

md("""\
**You never have to pass `embedding_model` again after indexing.** Whichever one was used is
recorded alongside the Chroma collection, so a later `RAGDataset.from_root(...)` — with
`embedding_model` omitted — automatically picks it back up, and `index_from_root` reusing the
collection does too:
""")

code("""\
print(dataset.recorded_embedding())

reloaded = RAGDataset.from_root("mock_rag_dataset")  # embedding_model not specified...
_ = reloaded.retrieve_examples(input_NL, with_output=False)  # ...still resolves it correctly
print(reloaded.embedding_model)
""")

md("""\
Passing a *different* `embedding_model` than what a collection was actually indexed with raises
`ValueError` instead of silently retrieving garbage (a query embedded with a different model than
its documents produces meaningless similarity scores) — and if a collection was indexed with an
OpenAI model but `OPENAI_API_KEY` isn't set when retrieval is attempted (e.g. a fresh session
where the key was never exported), that raises a clear `ValueError` too, rather than a cryptic
OpenAI auth error:
""")

code("""\
try:
    RAGDataset.index_from_root("mock_rag_dataset", embedding_model="sentence-transformers/all-mpnet-base-v2")
except ValueError as e:
    print("Mismatch caught as expected:", e)
""")

code("""\
# Re-running with the same root reuses the existing collection: no embedding calls, instant.
dataset_reloaded = RAGDataset.index_from_root(
    "mock_rag_dataset",
    embedding_model="text-embedding-3-large",
)

# Only pass rebuild=True when you actually want to re-embed (e.g. NLquestions/ changed):
# dataset_reloaded = RAGDataset.index_from_root("mock_rag_dataset", rebuild=True)
""")

md("""\
## 6. RAG-based techniques

The `"RAG"`, `"RAG+O"`, `"Schema+RAG"` and `"Schema+RAG+O"` techniques retrieve, from the
**vector store** built above, the most similar previously-seen questions together with their
"gold" Cypher query (and, for the `+O` variants, the corresponding Neo4j output too), to supply
as examples in the prompt.

`dataset` must be passed **only** for these techniques: passing it with `"vanilla"` or
`"Schema"`, or omitting it with a RAG technique, makes the function raise a `ValueError`.

We reuse the `dataset` built in §5. If you already have a bio2C-style benchmark on disk (like
`bio2C/evaluating_text2cypher`, with pre-built `chroma_db/`, `CypherQueries/`, `Neo4jOutputs/`),
point at it directly instead — no need to build anything:

```python
# dataset = RAGDataset.from_root("bio2C/evaluating_text2cypher")
```
""")

md("### 6.1 `\"RAG\"` — examples (question + Cypher) without output")

code("""\
result_rag = run(
    input_NL=input_NL,
    model="gpt-4o",
    database=database,
    technique="RAG",
    dataset=dataset,
)
show(result_rag)

print("\\nRetrieved examples:", result_rag.retrieved_examples["example_ids"])
print("Distances:", result_rag.retrieved_examples["example_distances"])
""")

md("### 6.2 `\"RAG+O\"` — examples including the gold query's Neo4j output")

code("""\
result_rag_o = run(
    input_NL=input_NL,
    model="gpt-4o",
    database=database,
    technique="RAG+O",
    dataset=dataset,
)
show(result_rag_o)
""")

md("### 6.3 `\"Schema+RAG\"` — enhanced schema + examples")

code("""\
result_schema_rag = run(
    input_NL=input_NL,
    model="gpt-4o",
    database=database,
    technique="Schema+RAG",
    dataset=dataset,
)
show(result_schema_rag)
""")

md("### 6.4 `\"Schema+RAG+O\"` — enhanced schema + examples + output")

code("""\
result_schema_rag_o = run(
    input_NL=input_NL,
    model="gpt-4o",
    database=database,
    technique="Schema+RAG+O",
    dataset=dataset,
)
show(result_schema_rag_o)
""")

md("""\
## 7. CyVer validation on an incorrect Cypher query

`run()` does two things with every generated query, regardless of technique:

1. it tries to **execute** it against Neo4j — `result.executed` tells you whether that succeeded,
   and `result.result` holds the returned rows if it did (`None` otherwise);
2. it **always** validates it with [**CyVer**](https://gitlab.com/netmode/CyVer) — a library that
   checks a Cypher query against the live graph schema — and attaches the report to
   `result.validation`, whether execution succeeded or not. The report gives a score in `[0, 1]`
   for:
   - **syntax validity** (`syntax_valid`, `syntax_metadata`);
   - **schema alignment** (`schema_score`, `schema_metadata`) — do the referenced labels,
     relationship types and paths actually exist in the graph?
   - **property-access correctness** (`properties_score`, `properties_metadata`) — do the
     accessed properties exist on the labels/relationships they're used on?

You already saw `result.validation` printed (as "CyVer validation report", all `1.0`/valid) for
every successful query in the sections above. To see it flag a **problem**, we plug in a fake
"model" — a
`Runnable` that ignores the question and always returns the same **deliberately broken** Cypher
query: it references a relationship type (`:TRANSCRIBED_FROM_TYPO`) that does not exist in the
schema, and accesses a property (`m.NotAProperty`) that does not exist on `miRNA` nodes either.
""")

code("""\
from langchain_core.runnables import RunnableLambda

# A fake "model": whatever the prompt is, always return this broken query.
broken_cypher = "MATCH (m:miRNA)-[:TRANSCRIBED_FROM_TYPO]->(g:Gene) RETURN m.NotAProperty"
broken_model = RunnableLambda(lambda _: broken_cypher)

result_broken = run(
    input_NL=input_NL,
    model=broken_model,
    database=database,
    technique="vanilla",
)
show(result_broken)
""")

md("""\
## 8. Rescuing a failed query

`rescue_prompt=True` (default `False`) retries a query that fails to execute or comes back empty
with a second "fix this query" prompt — reusing the same schema/examples context as the
technique, plus the bad query and an `error_message`. Ported from the miRNAKG rescue-prompt
notebook, and `error_message` concatenates every signal available about the failure: the native
Neo4j error (if the query didn't execute), an `"Empty result set."` note if it executed but
returned nothing, and **CyVer's validation report** (both its warning-level notifications and
hard errors), which `run()` already computes for every query. Raw Neo4j notifications
(deprecated syntax, unknown labels/relationship-types/properties, cartesian products, ... —
still captured separately on `result.execution_warnings`) are deliberately left out to save
tokens: CyVer's own validators already surface the ones that matter for a fix-up. `max_retries`
(default `1`) caps how many rescue attempts are made, stopping early once one succeeds.

We reuse the same deliberately-broken query from §7, via a fake "model" that returns it on the
*first* call only — every call after (i.e. the rescue attempt) returns a working query — so we
can see a rescue actually succeed.
""")

code("""\
call_count = {"n": 0}

def flaky_then_fixed(_):
    call_count["n"] += 1
    if call_count["n"] == 1:
        return broken_cypher  # the same broken query from §7
    return "MATCH (m:miRNA) RETURN m.Label AS miRNA LIMIT 5"

rescuable_model = RunnableLambda(flaky_then_fixed)

result_rescued = run(
    input_NL=input_NL,
    model=rescuable_model,
    database=database,
    technique="vanilla",
    rescue_prompt=True,
    max_retries=2,
)

print("Initial cypher:", result_rescued.initial_cypher)
print("Final cypher:  ", result_rescued.cypher)
print("Rescued:", result_rescued.rescued, "| attempts:", result_rescued.rescue_attempts)
show(result_rescued, show_prompt=True)  # show_prompt=True: prints the initial prompt *and* every rescue prompt
""")

md("""\
## 9. Cascading schema fallback (`cascade_mode`)

Schema pruning (§4.3-4.7) can over-prune: too little schema in the prompt, and the model can't
write a correct query at all. `cascade_mode=True` (default `False`) retries the *whole*
generation — a fresh prompt, not `rescue_prompt`'s error-aware fix-up — with progressively less
aggressive pruning whenever an attempt fails to execute or comes back empty, stopping early once
one rung succeeds:

1. **`"narrow"`**: node labels, relationship types, *and* properties all narrowed to the
   question — skipped entirely if `skip_narrow_schema_filter=True`.
2. **`"nodes_only"`**: only node labels are matched; relationships are kept via shared endpoints
   among the selected labels, and every property of a selected label/type is kept.
3. **`"full"`**: the unpruned schema — the final fallback, always tried last.

Only meaningful for a pruning `schema_mode` (`"exact_match"`, `"ner_exact_match"`, `"similarity"`,
`"llm_pruning"`, `"ie_extraction"`) — `"schema"`/`"enhanced"` have nothing to prune from.

**`cascade_mode` and `rescue_prompt`/`max_retries` are mutually exclusive** — two different retry
strategies for the same problem (a failing/empty query), not meant to be stacked. `run()` raises
`ValueError` if both are set:
""")

code("""\
try:
    run(
        input_NL=input_NL,
        model="gpt-4o",
        database=database,
        technique="Schema",
        schema_mode="exact_match",
        cascade_mode=True,
        rescue_prompt=True,
    )
except ValueError as e:
    print("Mutual-exclusivity caught as expected:", e)
""")

md("""\
We reuse the same deliberately-broken query from §7, via a fake "model" that only manages to
write a correct query starting from its *second* call — with `cascade_mode=True`, that second
call is the `"nodes_only"` rung (the `"narrow"` rung's attempt is the one that fails).
""")

code("""\
cascade_call_count = {"n": 0}

def flaky_until_less_pruned(_):
    cascade_call_count["n"] += 1
    if cascade_call_count["n"] == 1:
        return broken_cypher  # the same broken query from §7 -- fails on the "narrow" rung
    return "MATCH (m:miRNA) RETURN m.Label AS miRNA LIMIT 5"  # works from "nodes_only" onward

result_cascade = run(
    input_NL=input_NL,
    model=RunnableLambda(flaky_until_less_pruned),
    database=database,
    technique="Schema",
    schema_mode="exact_match",
    cascade_mode=True,
)

print("Rung used:   ", result_cascade.cascade_mode_level)      # "narrow" / "nodes_only" / "full"
print("Rungs tried: ", result_cascade.cascade_mode_attempts)   # 1..3
print("Final cypher:", result_cascade.cypher)
show(result_cascade)
""")

md("""\
`skip_narrow_schema_filter=True` skips straight to the `"nodes_only"` rung — useful once you know
the `"narrow"` rung tends to over-prune for your schema/questions and isn't worth the extra
generation call:
""")

code("""\
result_cascade_skip_narrow = run(
    input_NL=input_NL,
    model="gpt-4o",
    database=database,
    technique="Schema",
    schema_mode="exact_match",
    cascade_mode=True,
    skip_narrow_schema_filter=True,
)

print("Rung used:", result_cascade_skip_narrow.cascade_mode_level)  # never "narrow" here
show(result_cascade_skip_narrow)
""")

md("""\
### 9.1 Incremental delta cascade (`cascade_strategy="delta"`)

The cascade above repeats itself: `"narrow"`'s node labels/relationship types show up again,
folded into `"nodes_only"`'s bigger blob — so a schema element already sent to the model in an
earlier, failed rung gets paid for again in the next rung's prompt. `cascade_strategy="delta"`
(default `"standard"`) avoids that redundancy, but without the opposite failure mode either: since
every rung is still a fresh, self-contained prompt with no conversation history, showing a later
rung *only* the newly introduced elements would leave the model unable to reference a label/type
it only saw at an earlier rung. Two changes on top of `"standard"`:

1. **`"narrow"`** becomes `"true_narrow_top2"` — built from `"nodes_only"`'s own node-label
   selection (not the mode's own narrow pruning, which can itself already be too wide), with one
   extra trim: when a node-label pair is connected by more than 2 relationship types, only the 2
   most lexically similar to the question survive (`narrow_top2_relationships`, pure token
   overlap, no `nlp`/`llm` needed). So `"true_narrow_top2"` is a strict, cheap-to-fall-back-from
   subset of `"nodes_only"` — same labels/properties, fewer relationship choices.
2. **`"nodes_only"`/`"full"`** keep the same selection as `"standard"`, but their prompt now shows
   a **compact inventory** of everything a previous rung already showed (label/type/property names
   only, no examples — the terse non-enhanced `format_schema` style) plus only what's *newly
   introduced* at this rung, in full. The inventory accumulates across every previous rung, not
   just the one immediately before it.

Every rung stays a **fresh, independent, self-contained prompt** — no reference to a previous
rung's query or failure, deliberately unlike `rescue_prompt`'s fix-up mechanic, so the effect of
progressively revealing more schema can be studied on its own.

We reuse the exact same flaky-then-fixed scenario as `result_cascade` above, so the two
strategies' prompt token counts are directly comparable:
""")

code("""\
delta_call_count = {"n": 0}

def flaky_until_less_pruned_delta(_):
    delta_call_count["n"] += 1
    if delta_call_count["n"] == 1:
        return broken_cypher  # fails on the "narrow" rung, same as the standard cascade above
    return "MATCH (m:miRNA) RETURN m.Label AS miRNA LIMIT 5"

result_delta_cascade = run(
    input_NL=input_NL,
    model=RunnableLambda(flaky_until_less_pruned_delta),
    database=database,
    technique="Schema",
    schema_mode="exact_match",
    cascade_mode=True,
    cascade_strategy="delta",
)

print("Rung used:   ", result_delta_cascade.cascade_mode_level)
print("Rungs tried: ", result_delta_cascade.cascade_mode_attempts)
print("Final cypher:", result_delta_cascade.cypher)
show(result_delta_cascade)

print("\\nPrompt tokens per rung:")
print("  standard cascade:", result_cascade.cascade_mode_prompt_tokens)
print("  delta cascade:   ", result_delta_cascade.cascade_mode_prompt_tokens)
""")

md("""\
The winning (second) rung's prompt is still a fresh, self-contained "Schema" prompt — no
reference to the first rung's broken query, no error message — carrying a compact inventory of
what `"true_narrow_top2"` already showed plus only the schema newly introduced at `"nodes_only"`:
""")

code("""\
print(result_delta_cascade.prompt[-1]["content"])
""")

md("""\
## 10. Post-execution self-verification (`self_verification`)

`rescue_prompt` (§8) and `cascade_mode` (§9) both decide whether to retry using purely mechanical
checks: did the query fail to execute, come back empty, or fail CyVer's syntax check? None of
that catches a query that runs cleanly, returns rows, and is syntactically valid, yet still
doesn't answer what was actually asked — the wrong direction on a relationship, an aggregate over
the wrong property, a filter that's subtly too broad/narrow, and so on.

`self_verification=True` (default `False`) adds a **post-execution semantic check**: once an
attempt looks mechanically fine, a model reviews the question, the generated `cypher`, and the
rows it returned, and judges whether it actually answers the question. Often the same model that
generated a query is able to catch its own mistake on a second look. A mechanically-broken
attempt is retried as before, without spending a verification call on it — the semantic check
only runs once mechanical checks already pass, and its verdict becomes the retry decision
instead. Under `rescue_prompt`, a failed verdict's reasoning also flows into the fix-up prompt's
`error_message`; under `cascade_mode`, a failed verdict at one rung falls through to the next
exactly like a mechanical failure would.

**Requires `rescue_prompt=True` or `cascade_mode=True`** — there would otherwise be no retry for
a semantic verdict to inform:
""")

code("""\
try:
    run(
        input_NL=input_NL,
        model="gpt-4o",
        database=database,
        technique="vanilla",
        self_verification=True,
    )
except ValueError as e:
    print("Validation caught as expected:", e)
""")

md("""\
The cells below use a tiny **stub** verifier model instead of a real API call, so this notebook
runs without spending extra tokens on it — the mechanics (how a failed verdict feeds back into
`rescue_prompt`/`cascade_mode`) are identical either way. It mimics the one real requirement
`verify_semantics` has of `verification_model`: a `.with_structured_output(...)`-capable object
(see `verification.py`).
""")

code("""\
from langchain_core.runnables import RunnableLambda
from text2cypher_composer import SemanticVerification

class StubVerifierModel:
    \"\"\"A fake structured-output-capable model for self_verification -- no API calls.\"\"\"
    def __init__(self, judge):
        self.judge = judge  # judge(rendered_prompt_text) -> SemanticVerification

    def with_structured_output(self, schema, method=None):
        return RunnableLambda(lambda prompt_value: self.judge(prompt_value.to_string()))
""")

md("""\
### 10.1 With `rescue_prompt`

A fake "model" that answers with progressively more rows — first `LIMIT 5`, then `LIMIT 20` —
paired with a stub verifier that rejects the `LIMIT 5` answer as too narrow for the question and
approves `LIMIT 20`. Both queries execute cleanly (no mechanical failure at all): only the
semantic check tells them apart.
""")

code("""\
semantic_call_count = {"n": 0}

def flaky_semantics(_):
    semantic_call_count["n"] += 1
    limit = 5 if semantic_call_count["n"] == 1 else 20
    return f"MATCH (m:miRNA) RETURN m.Label AS miRNA LIMIT {limit}"

def judge_by_limit(prompt_text):
    if "LIMIT 5" in prompt_text:
        return SemanticVerification(
            answers_question=False, reasoning="the question asks for a broader sample than 5 rows"
        )
    return SemanticVerification(answers_question=True, reasoning="LIMIT 20 covers what was asked")

result_self_verified = run(
    input_NL=input_NL,
    model=RunnableLambda(flaky_semantics),
    database=database,
    technique="vanilla",
    rescue_prompt=True,
    self_verification=True,
    verification_model=StubVerifierModel(judge_by_limit),
)

print("Rescued:", result_self_verified.rescued, "| attempts:", result_self_verified.rescue_attempts)
print("Semantic verdict:", result_self_verified.self_verification_passed)
print("Reasoning:  ", result_self_verified.self_verification_reasoning)
show(result_self_verified)
""")

md("""\
### 10.2 With `cascade_mode`

Same idea, but a failed verdict now falls through to the next (less-pruned) rung instead of
triggering a fix-up prompt:
""")

code("""\
cascade_semantic_count = {"n": 0}

def flaky_cascade_semantics(_):
    cascade_semantic_count["n"] += 1
    limit = 5 if cascade_semantic_count["n"] == 1 else 20
    return f"MATCH (m:miRNA) RETURN m.Label AS miRNA LIMIT {limit}"

result_cascade_verified = run(
    input_NL=input_NL,
    model=RunnableLambda(flaky_cascade_semantics),
    database=database,
    technique="Schema",
    schema_mode="exact_match",
    cascade_mode=True,
    self_verification=True,
    verification_model=StubVerifierModel(judge_by_limit),
)

print("Rung used:", result_cascade_verified.cascade_mode_level)  # falls through past the "narrow" rung
print("Semantic verdict:", result_cascade_verified.self_verification_passed)
show(result_cascade_verified)
""")

md("""\
## 11. Adaptive RAG (`adaptive_rag`)

RAG retrieval (§6) can under-supply context too: too few retrieved examples, and the model may
never see the pattern it needs to write a correct query. `adaptive_rag=True` (default `False`) is
the RAG-side sibling of `cascade_mode` (§9): it retries the *whole* generation — a fresh prompt,
not `rescue_prompt`'s error-aware fix-up — with progressively **more** retrieved examples whenever
an attempt fails to execute or comes back empty, stopping early once one rung succeeds:

1. **`"minimal"`**: the dataset's configured `n_results` (§5's default, `3`) — today's normal,
   non-adaptive retrieval count, so the first try is identical to a plain (non-adaptive) call.
2. **`"moderate"`**: `min(2 * n_results, collection.count())`.
3. **`"full"`**: `min(5 * n_results, collection.count())` — the largest of the three, always
   tried last, but still a real, bounded cap — never *every* example in the collection, which
   for a real dataset (hundreds/thousands of examples) would blow the prompt budget for no
   benefit.

This is safe/informative by construction, not just "bigger": Chroma's top-k nearest-neighbor
retrieval is deterministic and monotonic (top-3 is always a strict prefix of top-6, top-6 of
top-15, ...), so a later rung's larger `n_results` only ever *adds* new, still-relevant
(similarity-ranked) examples on top of an earlier rung's — never duplicates or reorders them.

Only meaningful — and only allowed — for a RAG-using `technique` (`"RAG"`, `"RAG+O"`,
`"Schema+RAG"`, `"Schema+RAG+O"`).

**`adaptive_rag` and `cascade_mode`/`rescue_prompt`/`max_retries` are mutually exclusive** — three
different retry strategies for the same problem (a failing/empty query), not meant to be stacked.
`run()` raises `ValueError` if more than one is set:
""")

code("""\
try:
    run(
        input_NL=input_NL,
        model="gpt-4o",
        database=database,
        technique="RAG",
        dataset=dataset,
        adaptive_rag=True,
        cascade_mode=True,
    )
except ValueError as e:
    print("Mutual-exclusivity caught as expected:", e)
""")

md("""\
We reuse the same deliberately-broken query from §7 and the RAG `dataset` built in §5/§6, via a
fake "model" that only manages to write a correct query starting from its *second* call — with
`adaptive_rag=True`, that second call is the `"moderate"` rung (the `"minimal"` rung's attempt is
the one that fails).
""")

code("""\
adaptive_call_count = {"n": 0}

def flaky_until_more_examples(_):
    adaptive_call_count["n"] += 1
    if adaptive_call_count["n"] == 1:
        return broken_cypher  # the same broken query from §7 -- fails on the "minimal" rung
    return "MATCH (m:miRNA) RETURN m.Label AS miRNA LIMIT 5"  # works from "moderate" onward

result_adaptive = run(
    input_NL=input_NL,
    model=RunnableLambda(flaky_until_more_examples),
    database=database,
    technique="RAG",
    dataset=dataset,
    adaptive_rag=True,
)

print("Rung used:   ", result_adaptive.adaptive_rag_level)      # "minimal" / "moderate" / "full"
print("Rungs tried: ", result_adaptive.adaptive_rag_attempts)   # 1..3
print("Final cypher:", result_adaptive.cypher)
show(result_adaptive)
""")

md("""\
## 12. Fine-tuning your own model

Two paths to a model specialized on your own question→Cypher examples, ported from
`bio2C/evaluating_text2cypher/evaluating_text2cypher_gpt.ipynb` (dataset preparation) and
`bio2C/finetuning_LLaMa3-8B/Finetuning_llama3-8b.ipynb` (LoRA training):

- **Fine-tune GPT via OpenAI's fine-tuning GUI** — export a chat-format `.jsonl` and upload it
  at platform.openai.com/finetune; the resulting `ft:...` model id can be passed straight to
  `run(model=...)` like any other OpenAI model id.
- **LoRA-finetune a local model** (e.g. LLaMA) yourself with `finetune_lora`, then load the
  adapter back with `load_finetuned_model` — a ready-to-use `Runnable` for `run(model=...)`,
  the same "bring your own `Runnable`" path any LangChain-compatible chat model takes (not just
  a fine-tuned one — a plain base HuggingFace pipeline works too).

Both paths start from the same leveled gold dataset.
""")

md("""\
### 12.1 Loading a leveled gold dataset

bio2C organizes fine-tuning gold sets into "levels" (`nodeLevel`, `1hop`, `2hop`, `3hop`,
`hardLevel`, ...), each a JSON file of `{"question", "cypher"}` records. `load_finetune_levels`
loads and concatenates them, tagging every row with its source `level` (used below for a
stratified split) and a bio2C-style `ID`. We split `mock_df` (§5.1) into two toy "levels" and
write them to disk to demonstrate it for real.
""")

code("""\
import json, os, tempfile
from text2cypher_composer import load_finetune_levels

ft_source = mock_df[["question", "cypher"]]
ft_root = tempfile.mkdtemp(prefix="t2c_ft_demo_")

level_paths = {}
for level_name, chunk in [("nodeLevel", ft_source.iloc[:3]), ("1hop", ft_source.iloc[3:])]:
    level_path = os.path.join(ft_root, f"{level_name}.json")
    with open(level_path, "w", encoding="utf-8") as f:
        json.dump(chunk.to_dict("records"), f)
    level_paths[level_name] = level_path

ft_df = load_finetune_levels(level_paths)
ft_df
""")

md("""\
### 12.2 Sizing a token budget, and a stratified train/test split

`max_cypher_tokens` sizes a generation `max_tokens` budget from the longest gold query (useful
for fine-tuning as well as bulk evaluation, so it isn't needlessly large — that slows things
down). `split_finetune_dataset` mirrors bio2C's `groupby("level").sample(...)`: a train/test
split stratified per level.
""")

code("""\
from text2cypher_composer import max_cypher_tokens, split_finetune_dataset

print(max_cypher_tokens(ft_df))

train_df, test_df = split_finetune_dataset(ft_df, test_frac=0.4, random_state=42)
print(f"train: {len(train_df)}, test: {len(test_df)}")
train_df
""")

md("""\
### 12.3 Exporting for fine-tuning

`write_local_finetune_dataset` writes the format `finetune_lora` (§12.4) reads back in;
`build_gpt_finetune_jsonl` writes the chat-format `.jsonl` OpenAI's fine-tuning GUI expects —
one `{"messages": [...]}` line per example, with an optional system message.
""")

code("""\
from text2cypher_composer import write_local_finetune_dataset, build_gpt_finetune_jsonl

local_path = write_local_finetune_dataset(train_df, os.path.join(ft_root, "FTdataset_local.json"))
gpt_jsonl = build_gpt_finetune_jsonl(train_df, os.path.join(ft_root, "FTdataset_GPTviaGUI.jsonl"))

print(local_path)
print(gpt_jsonl)
with open(gpt_jsonl.path, encoding="utf-8") as f:
    print(f.readline())
""")

md("""\
### 12.4 LoRA-finetuning a local model

`finetune_lora` LoRA-finetunes a 4-bit-quantized base causal LM (`LoRATrainingConfig.base_model`
defaults to `meta-llama/Llama-3.1-8B`, matching the notebook) on `train_df`'s question/Cypher
pairs, saving the adapter to `config.output_dir`. Needs the optional `finetune` extra
(`pip install "text2cypher-composer[finetune]"` — torch, transformers, peft, datasets) and,
realistically, a GPU — demonstration-only cell, not executed here.

`meta-llama/Llama-3.1-8B` is **gated** on Hugging Face: accept its license on the model page and
run `huggingface-cli login` (or set `HF_TOKEN`) before calling `finetune_lora` with it, otherwise
`from_pretrained` raises a 401/403.
""")

code("""\
# from text2cypher_composer import LoRATrainingConfig, finetune_lora
#
# ft_result = finetune_lora(
#     train_df, config=LoRATrainingConfig(output_dir="./llama3_lora_mirnakgt2c")
# )
# print(ft_result.adapter_path, ft_result.n_examples)
""")

md("""\
### 12.5 Using the fine-tuned (or any local) model with `run()`

`load_finetuned_model` loads the adapter back as a `HuggingFacePipeline` `Runnable`,
pre-configured with its own generation parameters — pass it straight through as `run()`'s
`model` (used as-is, since it's already a `Runnable`), typically with `technique="vanilla"` to
match the notebook's inference-time prompt. A non-finetuned local pipeline (e.g. straight from
`bio2C/evaluating_text2cypher/evaluating_text2cypher_llama.ipynb`) works the same way — just
build the `HuggingFacePipeline` yourself instead of via `load_finetuned_model`. Also
demonstration-only (needs a GPU, the same Hugging Face login from §12.4, and the model weights
available).
""")

code("""\
# from text2cypher_composer import load_finetuned_model
#
# llama_ft = load_finetuned_model("meta-llama/Llama-3.1-8B", "./llama3_lora_mirnakgt2c")
# result_ft = run(
#     input_NL=input_NL,
#     model=llama_ft,           # <-- an already-built Runnable, instead of a string
#     database=database,
#     technique="vanilla",
# )
# show(result_ft)
""")

md("""\
## 13. Discovering available techniques and their prompts

A few introspection helpers, useful without needing a database/model/dataset at hand — e.g. to
build a UI, validate a `technique` string before calling `run()`, or just remember what each
technique needs and what it sends the model.
""")

md("### 13.1 `list_techniques()` — the accepted `technique` strings")

code("""\
from text2cypher_composer import list_techniques

list_techniques()
""")

md("""\
### 13.2 `describe_technique()` / `list_technique_info()` — what each technique needs

Tells you whether a technique uses the enhanced schema and/or RAG (and, if RAG, whether it's the
output-augmented `+O` variant) — i.e. whether `dataset` must be passed to `run()`.
""")

code("""\
from text2cypher_composer import describe_technique, list_technique_info

print(describe_technique("Schema+RAG+O"))
print()
for info in list_technique_info():
    print(info)
""")

md("""\
### 13.3 `get_prompt_template()` / `get_all_prompt_templates()` — the parametric prompts

The **unfilled** prompt for a technique — placeholders like `{question}`, `{enhanced_schema}`,
`{examples}` are left as literal text. This is the template; for the fully-instantiated prompt
actually sent to the model on a given call, use `Text2CypherResult.prompt` (§3's
`show_prompt=True`).
""")

code("""\
from text2cypher_composer import get_all_prompt_templates, get_prompt_template

for message in get_prompt_template("Schema+RAG+O"):
    print(f"[{message['role']}]")
    print(message["content"])
    print()

# Or grab every technique's template at once, keyed by technique value:
all_templates = get_all_prompt_templates()
print(list(all_templates.keys()))
""")

md("""\
## 14. Bulk evaluation against a gold test set

`evaluate_technique` runs a technique over a whole gold `(question, query)` set and reports
Jaro-Winkler, normalized Levenshtein, Jaccard, Coverage, and pass@k. We reuse the mock
`mock_df` from §5.1 as the gold set — `evaluate_technique` expects a `"query"` column, so we
rename `mock_df`'s `"cypher"` column back to it.

Jaccard/Coverage compare the two queries' **result rows**, not their Cypher text (a
differently-worded but equivalent query should still score well): rows are greedily matched by
similarity before comparing, since Neo4j doesn't guarantee row order without `ORDER BY`.

With `k=2`, each question gets 2 independent generation attempts; `pass@1`/`pass@2` say whether
the 1st, or either of the first 2, attempts exactly reproduced the gold result.

`rescue_prompt`/`max_retries` (§8) are forwarded to every attempt too; `report.to_dataframe()`
then carries `execution_error`/`execution_warnings` (populated regardless of `rescue_prompt`),
`rescued`/`rescue_attempts`/`rescue_error_messages`/`rescue_prompts` per question, and
`prompt_tokens`/`rescue_prompt_tokens` (via `tiktoken`) so you can see how much the rescue prompt
is pulling its weight — and costing in tokens — across a whole gold set, not just one query.
""")

code("""\
from text2cypher_composer import evaluate_technique

gold_df = mock_df.rename(columns={"cypher": "query"})

report = evaluate_technique(
    gold_df,
    model="gpt-4o",
    database=database,
    technique="vanilla",
    k=2,
    rescue_prompt=True,
    max_retries=2,
)

print(report.summary)
""")

code("""\
eval_df = report.to_dataframe()
eval_df
""")

code("""\
eval_df[[
    "question", "prompt_tokens", "execution_error", "execution_warnings",
    "rescued", "rescue_attempts", "rescue_error_messages", "rescue_prompts", "rescue_prompt_tokens",
]]
""")

md("""\
## 15. Summary

- `run()` always returns a `Text2CypherResult` with: `question`, `technique`, `model`,
  `initial_cypher` (what the model generated first), `cypher` (the final, possibly rescued
  query — already cleaned of any backticks/code fences), `prompt` (the exact messages sent to
  the model for the initial attempt, fully instantiated), `executed` (whether the *final* query
  ran successfully), `schema` (if used), and `retrieved_examples` (if RAG was used).
- If `executed` is `True`, `result` holds the rows returned by Neo4j; otherwise it's `None`.
- `validation` is **always** populated — for every query, successful or not — with a
  `CypherValidationReport` from CyVer (`syntax_valid`/`syntax_metadata`,
  `schema_score`/`schema_metadata`, `properties_score`/`properties_metadata`) for the final
  attempt.
- `schema_mode` (§4) controls how the schema is derived/pruned for schema-using techniques:
  `"schema"` (default), `"enhanced"`, `"exact_match"`, `"ner_exact_match"`, `"similarity"`
  (the latter two need a user-supplied `nlp` pipeline), `"llm_pruning"`, or `"ie_extraction"`
  (§4.7, needs an `ie_engine` — `schemalink_ie_engine()` is a ready-made one backed by the real
  `schemalink-engine` package, `pip install schemalink-engine`).
- `schema_components` (§4.3.1) narrows `"exact_match"`/`"ner_exact_match"`/`"ie_extraction"` down
  to (or, for `"ie_extraction"`, up from) entity types only, the default — any combination of
  `"entity_types"`, `"relationship_types"`, `"node_properties"`, `"relationship_properties"`
  (see `SchemaComponent`/`list_schema_components()`).
- `cache_schema=True` (the default, §4.8) caches a schema-using technique's extracted schema per
  `(database, is_enhanced, sample)` and reuses it across every `run()` call against the same graph
  instance — extraction is a fixed cost that doesn't change across a benchmark run, so this cuts
  the technical schema-extraction overhead (not LLM cost) that otherwise dominates wall-clock time
  once you're testing hundreds/thousands of questions. `cache_schema=False` always re-extracts;
  `clear_schema_cache(graph)`/`clear_schema_cache()` invalidate one or every cached graph.
- `dry_run=True` (§3.1) builds and returns `prompt` (schema/RAG resolved) without generating,
  executing, or validating anything — `cypher`/`result`/`validation` stay `None`. Incompatible
  with `rescue_prompt=True`.
- `execution_error`/`execution_warnings` (the native Neo4j error/notifications from the *final*
  attempt's actual execution) are always populated on every `Text2CypherResult`, independently of
  `rescue_prompt`.
- `rescue_prompt=True` (§8) retries a failed/empty query with a fix-up prompt whose
  `error_message` concatenates the native Neo4j error and CyVer's report (raw Neo4j notifications
  are left out — CyVer's own validators already surface the ones that matter), up to
  `max_retries` (default `1`, must be `>= 1`) times — `result.rescued`/`result.rescue_attempts`
  report whether and how many times that happened; `result.rescue_error_messages`/
  `result.rescue_prompts` hold, per attempt, the `error_message` sent and the exact
  fully-instantiated messages sent for it.
- `result.prompt_tokens` (via `tiktoken`, `None` if it isn't installed) is the initial prompt's
  token count — compare it across `technique`/`schema_mode` to see how many tokens schema
  filtering actually saves; `result.rescue_prompt_tokens` is the parallel per-attempt count for
  `rescue_prompts` — a list of `rescue_attempts` numbers, to see how many extra tokens
  `rescue_prompt` costs.
- `cascade_mode=True` (§9) retries a failed/empty query from scratch with progressively less
  aggressive schema pruning ("narrow" → "nodes_only" → "full", `skip_narrow_schema_filter=True`
  to start at "nodes_only"), stopping at the first rung that succeeds — a different retry
  strategy than `rescue_prompt`'s error-aware fix-up, so the two are mutually exclusive.
  `result.cascade_mode_level`/`result.cascade_mode_attempts`/`result.cascade_mode_prompts`/
  `result.cascade_mode_prompt_tokens` report which rung was used and what each tried rung cost.
- `cascade_strategy="delta"` (§9.1, requires `cascade_mode=True`) is the "Incremental delta
  cascade": `"narrow"` becomes `"true_narrow_top2"` — built from `"nodes_only"`'s own node-label
  selection rather than the mode's own (possibly already-wide) narrow pruning, keeping only the 2
  most lexically relevant relationship types per node-label pair — and every rung after the first
  shows a compact inventory of everything a previous rung already showed (label/type/property names, no examples)
  plus only the schema newly introduced at that rung — instead of repeating the full schema, or
  showing only the delta with no memory of what the model already saw. Every rung, including this
  one, stays a fresh, independent, self-contained prompt (no reference to a previous rung's query
  or failure, unlike `rescue_prompt`), keeping the schema-tightening effect isolated from any
  correction mechanic. `result.schema` then holds that rung's inventory + delta text, not the
  cumulative schema.
- `self_verification=True` (§10, requires `rescue_prompt` or `cascade_mode`) adds a post-execution
  semantic check on top of either retry strategy's mechanical one: once an attempt looks
  mechanically fine, a model reviews `(question, cypher, result)` and judges whether it actually
  answers the question — a failed verdict is folded into the same retry decision (feeding its
  reasoning into `rescue_prompt`'s fix-up prompt, or falling through to the next `cascade_mode`
  rung). `verification_model` (defaults to reusing `model`) and `verification_criteria` (extra
  free-text evaluation guidance) are optional. `result.self_verification_passed`/
  `result.self_verification_reasoning` report the final attempt's verdict, `None` if unused or if
  the final attempt was already mechanically broken.
- `adaptive_rag=True` (§11) is the RAG-side sibling of `cascade_mode`: it retries a failed/empty
  query from scratch with progressively more retrieved examples ("minimal" → "moderate" → "full",
  i.e. `n_results` of the dataset's configured default → 2x that → 5x that, each capped at the
  collection's actual size — never *every* example in the collection), stopping at the first rung
  that succeeds — mutually exclusive with both `cascade_mode` and `rescue_prompt`/`max_retries`
  (pick one retry strategy). `result.adaptive_rag_level`/
  `result.adaptive_rag_attempts`/`result.adaptive_rag_prompts`/`result.adaptive_rag_prompt_tokens`
  report which rung was used and what each tried rung cost.
- Fine-tuning (§12): `load_finetune_levels`/`max_cypher_tokens`/`split_finetune_dataset` prepare a
  leveled gold dataset, `write_local_finetune_dataset`/`build_gpt_finetune_jsonl` export it for
  `finetune_lora` (LoRA-finetune a local model) or OpenAI's fine-tuning GUI respectively; either
  way, the resulting model — a fine-tuned model id (string) or `load_finetuned_model`'s
  `Runnable` — plugs straight into `run(model=...)`.
- RAG embeddings (§5.3) are pluggable via `embedding_model`: OpenAI (default), HuggingFace/
  sentence-transformers (any id containing `"/"`, e.g. `"sentence-transformers/all-mpnet-base-v2"`,
  run locally), or an already-built `Embeddings` instance. Whichever one indexed a collection is
  recorded alongside it and reused automatically at retrieval — `embedding_model` never needs to
  be passed again — and a mismatched `embedding_model` (or a missing `OPENAI_API_KEY` for a
  collection that needs one) raises `ValueError` instead of silently corrupting retrieval.
- `RAGDataset`'s `chromadb` dependency is an optional extra (`pip install
  "text2cypher-composer[rag]"`) imported lazily, only inside `RAGDataset` itself — every non-RAG
  component (`"vanilla"`/`"Schema"`, `cascade_mode`, `rescue_prompt`, `self_verification`, ...)
  works from the base install alone.
- The available techniques are listed in `Technique`
  (`from text2cypher_composer import Technique`), or as plain strings via `list_techniques()`.
  `describe_technique()`/`list_technique_info()` tell you what each one needs, and
  `get_prompt_template()`/`get_all_prompt_templates()` show its unfilled prompt (§13).
- `evaluate_technique()` (§14) runs a technique over a gold test set and reports
  Jaro-Winkler, Levenshtein, Jaccard, Coverage, and pass@k as an `EvaluationReport`
  (`.summary` for dataset-level averages, `.to_dataframe()` for a per-question table).
  `rescue_prompt`/`max_retries` forward to every attempt, and `.to_dataframe()`/
  `save_evaluation_report()`'s `.pkl`/`.xlsx` then carry `execution_error`/`execution_warnings`,
  `rescued`/`rescue_attempts`/`rescue_error_messages`/`rescue_prompts`, and
  `prompt_tokens`/`rescue_prompt_tokens` per question.
""")

nb["cells"] = cells

path = Path(__file__).resolve().parent.parent / "demo_text2cypher_composer.ipynb"
with open(path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("written:", path)
