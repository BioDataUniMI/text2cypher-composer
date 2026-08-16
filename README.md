# text2cypher-composer

[![PyPI](https://img.shields.io/pypi/v/text2cypher-composer?label=PyPI&logo=pypi)](https://pypi.org/project/text2cypher-composer/)
[![Pypi total project downloads](https://static.pepy.tech/badge/text2cypher-composer)](https://pepy.tech/project/text2cypher-composer)

[![GitHub Action: Publish to PyPI](https://github.com/BioDataUniMI/text2cypher-composer/actions/workflows/publish.yml/badge.svg)](https://github.com/BioDataUniMI/text2cypher-composer/actions/workflows/publish.yml)

<!-- Quality/coverage badges below are placeholders — this project isn't registered on
     SonarCloud/Codacy yet. Replace the project key / badge ID in each URL (and set up the
     corresponding integration) before relying on them; until then they'll show as
     broken/"unknown". Coveralls' URL is already valid as-is (no token needed), it just
     needs Coveralls enabled for this repo. -->
[![SonarCloud Quality](https://sonarcloud.io/api/project_badges/measure?project=BioDataUniMI_text2cypher-composer&metric=alert_status)](https://sonarcloud.io/dashboard?id=BioDataUniMI_text2cypher-composer)
[![Codacy Maintainability](https://app.codacy.com/project/badge/Grade/REPLACE_WITH_CODACY_BADGE_ID)](https://www.codacy.com/gh/BioDataUniMI/text2cypher-composer/dashboard)
[![Coveralls Coverage](https://coveralls.io/repos/github/BioDataUniMI/text2cypher-composer/badge.svg?branch=main)](https://coveralls.io/github/BioDataUniMI/text2cypher-composer?branch=main)

Translate a natural-language question into an executable Cypher query and run it
against a Neo4j database, using the prompting strategies from the
[bio2C](bio2C/README.md) benchmark: `vanilla`, `Schema`, `RAG`, `RAG+O`,
`Schema+RAG`, `Schema+RAG+O`.

## Install

```bash
pip install text2cypher-composer
```

Cloned this repo instead and want an editable install that picks up local changes?
`pip install -e .` from the repo root does that (see [CONTRIBUTING.md](CONTRIBUTING.md)).

Set `OPENAI_API_KEY` in the environment if using an OpenAI model id or the
`RAGDataset` embedder (both default to OpenAI embeddings/chat models).

Using an Anthropic model id (e.g. `"claude-sonnet-5"`) needs `ANTHROPIC_API_KEY` set, plus an extra:

```bash
pip install "text2cypher-composer[anthropic]"
```

Using a Google Gemini model id (e.g. `"gemini-2.5-flash"`) needs `GOOGLE_API_KEY` set, plus an extra:

```bash
pip install "text2cypher-composer[google]"
```

Using a DeepSeek model id (e.g. `"deepseek-chat"`) needs `DEEPSEEK_API_KEY` set, plus an extra:

```bash
pip install "text2cypher-composer[deepseek]"
```

LoRA-finetuning a local model (`finetune_lora`/`load_finetuned_model`) needs an extra:

```bash
pip install "text2cypher-composer[finetune]"
```

Embedding a RAG dataset with a local HuggingFace/sentence-transformers model instead of OpenAI
(`RAGDataset.index_from_root(..., embedding_model="sentence-transformers/...")`) needs another:

```bash
pip install "text2cypher-composer[local-embeddings]"
```

Using `schema_mode="ie_extraction"` with the ready-made `schemalink_ie_engine()` (instead of
bringing your own `ie_engine`) needs an extra too, plus an OpenAI API key configured for it
(`schemalink api-key set sk-...`):

```bash
pip install "text2cypher-composer[schemalink]"
```

## Usage

```python
from text2cypher_composer import run, RAGDataset

result = run(
    input_NL="How many miRNAs have the keyword 'precursor' in the label and a sequence size under 100 nucleotides?",
    model="gpt-4o",
    database={
        "uri": "neo4j+s://<host>:7687",
        "username": "<user>",
        "password": "<password>",
        "database": "<database>",
    },
    technique="vanilla",
)

print(result.cypher)
print(result.prompt)                    # exact messages sent to the model, fully instantiated
print(result.executed, result.result)   # result.result is None if execution failed
print(result.validation)                # CyVer report, always populated
print(result.execution_error)           # native Neo4j error (code + message), or None
print(result.execution_warnings)        # Neo4j notifications observed during execution, always populated
print(result.prompt_tokens)             # prompt's tiktoken token count, or None if tiktoken isn't installed
```

### Techniques

| `technique`        | Uses schema | Uses `dataset` (RAG) |
|---------------------|:-----------:|:---------------------:|
| `"vanilla"`          |             |                        |
| `"Schema"`           |     ✓       |                        |
| `"RAG"`              |             |          ✓             |
| `"RAG+O"`            |             |          ✓             |
| `"Schema+RAG"`       |     ✓       |          ✓             |
| `"Schema+RAG+O"`     |     ✓       |          ✓             |

`"+O"` techniques additionally include the Neo4j output of each retrieved
example in the prompt. `dataset` must be provided if and only if `technique`
uses RAG.

### `schema_mode` (schema techniques only)

For `"Schema"`/`"Schema+RAG"`/`"Schema+RAG+O"`, `schema_mode` controls *how*
the schema is derived before being placed in the prompt — ported from the
miRNAKG schema-representation notebook ("The Impact of Schema Representation
in the Text2Cypher Task", https://doi.org/10.48550/arXiv.2505.05118), plus a
new LLM-driven pruning mode:

| `schema_mode`        | What it does                                                                 | Needs         |
|-----------------------|-------------------------------------------------------------------------------|---------------|
| `"schema"` (default)  | The plain schema as returned by Neo4j, no per-property stats.                 | —             |
| `"enhanced"`           | Adds per-property examples/min-max stats.                                     | —             |
| `"exact_match"`        | Prunes the enhanced schema to node labels mentioned (substring match) in the question, plus relationships between two mentioned labels. Falls back to the full schema if nothing matches. | — |
| `"ner_exact_match"`    | Same as `"exact_match"`, but named entities in the question are first masked with their entity type, so entity *values* (e.g. a gene name) don't get confused with schema field names. | `nlp` |
| `"similarity"`         | Prunes the base schema to labels/types/properties whose word-vector similarity to the question clears `similarity_threshold`. No fallback — an unrelated question can prune to nothing. | `nlp` |
| `"llm_pruning"`        | Asks `model` itself (via structured/JSON-schema output) which labels and relationship types are relevant. Falls back to the full schema if it selects none (and silently drops any hallucinated label/type not actually in the schema). | — |
| `"ie_extraction"`      | Runs schema-grounded information extraction (NER + relation extraction) over the question via a user-supplied `ie_engine`, and keeps exactly the entity types/relationship types/properties it found — no substring matching, unlike `"exact_match"`/`"ner_exact_match"`. | `ie_engine` |

`nlp` is a loaded NLP pipeline you bring yourself — this library never
imports spaCy (or any NLP library) directly, it only calls `nlp(text)` /
`nlp.vocab[...]` / token `.similarity()` on whatever you pass in, the same
"bring your own object" pattern `model` uses for non-OpenAI chat models:

```python
import spacy

# "ner_exact_match" needs named-entity recognition — a biomedical model, e.g.:
nlp = spacy.load("en_ner_bionlp13cg_md")

# "similarity" needs word vectors instead, e.g.:
nlp = spacy.load("en_core_web_md")

result = run(
    input_NL="...",
    model="gpt-4o",
    database={},
    technique="Schema",
    schema_mode="ner_exact_match",
    nlp=nlp,
)
```

`schema_mode`/`nlp`/`similarity_threshold`/`ie_engine` are disallowed for
techniques that don't use the schema (`"vanilla"`, `"RAG"`, `"RAG+O"`) —
`run()` raises `ValueError` if you pass them there. The available modes are
listed by `list_schema_modes()`, and the individual pruning functions
(`exact_match_prune`, `ner_exact_match_prune`, `similarity_prune`,
`similarity_prune_nodes_only`, `llm_prune`, `llm_prune_nodes_only`,
`ie_prune`, `mask_entities`, `structured_schema_to_linkml`) are exported
standalone too, for pruning a schema outside of a full `run()` call — see
`cascade_mode` below for the nodes-only variants' purpose.

### `schema_components` (`"exact_match"`/`"ner_exact_match"`/`"ie_extraction"` only)

By default, these three modes only match/extract **entity types** (node
labels) against the question — exactly as before. `schema_components` widens
that to any combination of the four schema element kinds in `SchemaComponent`:

```python
from text2cypher_composer import SchemaComponent, list_schema_components

list_schema_components()
# ["entity_types", "relationship_types", "node_properties", "relationship_properties"]
```

- `"entity_types"` (default): node labels mentioned in the question.
- `"relationship_types"`: relationship types mentioned in the question — for
  `"exact_match"`/`"ner_exact_match"`, this additionally keeps a relationship
  whose *type* (not just its endpoints) is mentioned, instead of only
  inferring kept relationships from which node labels were selected.
- `"node_properties"` / `"relationship_properties"`: narrows a selected
  label's/type's properties down to the ones actually mentioned, instead of
  keeping every property of a selected label/type (the default). A label/type
  with no property mentioned keeps all of its properties rather than none —
  same "don't over-prune" fallback `exact_match_prune` already uses when no
  label at all is mentioned.

Entity types always anchor the selection for `"exact_match"`/
`"ner_exact_match"` (their matches decide which node types — and, via shared
endpoints, which relationships — survive); the other components only narrow
further what's kept once a label is selected. For `"ie_extraction"`,
`schema_components` instead controls what's *asked of* the extraction engine
in the first place (see below).

```python
result = run(
    input_NL="...",
    model="gpt-4o",
    database={},
    technique="Schema",
    schema_mode="exact_match",
    schema_components=["entity_types", "relationship_types", "node_properties", "relationship_properties"],
)
```

### `ie_engine` (`"ie_extraction"` only)

`"ie_extraction"` asks a schema-grounded information-extraction engine —
rather than a substring-match heuristic — which entities, relations, and
(if requested via `schema_components`) attributes are present in the
question, then keeps exactly those. `structured_schema_to_linkml` first
converts the graph's structured schema into a
[LinkML](https://linkml.io/) YAML schema (no ontology grounding — just
classes/attributes named *exactly* like the Neo4j labels/types/properties,
since `ie_prune` only needs entity/relation/attribute presence, not links to
external ontology IDs); `ie_engine` is then called as
`ie_engine(schema_yaml, question)` and must return a dict shaped like
[SchemaLink](https://github.com/BioDataUniMI/schemalink-engine)'s output —
`{class_name: {"mentions": [{...}, ...]}}`, one entry per class actually
asked for (empty/absent `"mentions"` for a class found nowhere in the text):

```python
result = run(
    input_NL="...",
    model="gpt-4o",
    database={},
    technique="Schema",
    schema_mode="ie_extraction",
    schema_components=["entity_types", "relationship_types"],
    ie_engine=my_ie_engine,  # callable: (schema_yaml, question) -> dict
)
```

**`schemalink_ie_engine()` is a ready-made `ie_engine`** backed by the real
[SchemaLink](https://github.com/BioDataUniMI/schemalink-engine) package —
`pip install schemalink-engine` (or
`pip install "text2cypher-composer[schemalink]"`), then set an OpenAI API key
for it (`schemalink api-key set sk-...`):

```python
from text2cypher_composer import schemalink_ie_engine

result = run(
    input_NL="...",
    model="gpt-4o",
    database={},
    technique="Schema",
    schema_mode="ie_extraction",
    schema_components=["entity_types", "relationship_types"],
    ie_engine=schemalink_ie_engine(),
)
```

It bridges `ie_prune`'s `ie_engine(schema_yaml, question) -> dict` contract to
`schemalink_engine.pipeline.run_extraction_pipeline`, which works over
schema/text *file paths* and writes its output to a JSON file rather than
returning it — the adapter writes both to a scratch temp directory (so it
doesn't litter your cwd with the pipeline's `generated/`/`output/` working
directories) and reads the output back. It isn't thread-safe (the pipeline
`chdir`s into that scratch directory for the extraction's duration), so don't
call it from multiple threads concurrently.

`schemalink_ie_engine(include_node_types=True, include_relationship_types=True,
include_properties=True, with_dependencies=True, ground_entities=None)` — the
three booleans mirror `schema_components`'s entity/relationship/property split,
but coarser (node/relationship properties aren't split) and independent of
it: they filter the LinkML schema actually sent to SchemaLink on this call,
on top of whatever `schema_components` already restricted it to upstream —
e.g. `include_properties=False` still asks which entities/relationships are
mentioned, just not to also extract their property values. `with_dependencies`
toggles SchemaLink's dependency-aware extraction (each class's GPT call
conditioned on its dependencies' results — the default, and SchemaLink's own
default) vs. flat/independent extraction per class. `ground_entities` (e.g.
`{"mode": "auto"}`) grounds extracted entities to biomedical ontology IDs via
OAK — only for classes whose LinkML definition declares `annotators:`, and
downloads ontology databases on first use.

### `cascade_mode` / `skip_narrow_schema_filter` (pruning `schema_mode`s only)

Schema pruning can over-prune: too little schema in the prompt, and the model can't write a
correct query at all. `cascade_mode=True` (default `False`) retries the *whole* generation —
a fresh prompt, not `rescue_prompt`'s error-aware fix-up — with progressively less aggressive
pruning whenever an attempt fails to execute or comes back empty, stopping early once one
succeeds:

1. **`"narrow"`**: node labels, relationship types, *and* properties all narrowed to the question
   — `ALL_SCHEMA_COMPONENTS` for `"exact_match"`/`"ner_exact_match"`/`"ie_extraction"`; each
   mode's normal (most aggressive) pruning for `"similarity"`/`"llm_pruning"`. Skipped entirely
   if `skip_narrow_schema_filter=True`.
2. **`"nodes_only"`**: only node labels are matched — `DEFAULT_SCHEMA_COMPONENTS` for
   `"exact_match"`/`"ner_exact_match"`/`"ie_extraction"`; the new `similarity_prune_nodes_only`/
   `llm_prune_nodes_only` for `"similarity"`/`"llm_pruning"`. Relationships are kept when both
   endpoints are among the selected labels, and every property of a selected label/type is kept.
3. **`"full"`**: the unpruned schema — the final fallback, always tried last.

```python
result = run(
    input_NL="...",
    model="gpt-4o",
    database={},
    technique="Schema",
    schema_mode="exact_match",
    cascade_mode=True,
    # skip_narrow_schema_filter=True,  # start straight at "nodes_only" instead of "narrow"
)

print(result.cascade_mode_level)          # "narrow" / "nodes_only" / "full" — which one was used
print(result.cascade_mode_attempts)       # how many levels were tried (1..3)
print(result.cascade_mode_prompts)        # the fully-instantiated prompt tried at each level
print(result.cascade_mode_prompt_tokens)  # their token counts, parallel to cascade_mode_prompts
```

`result.cypher`/`result.executed`/`result.result`/`result.validation`/`result.schema`/
`result.prompt` always reflect the level that was ultimately returned (the first one that
succeeded, or `"full"` if none did) — same as `rescue_prompt`'s existing "reflects the final
attempt" convention.

**`cascade_mode` and `rescue_prompt`/`max_retries` are mutually exclusive** — two different retry
strategies for the same problem (a failing/empty query), not meant to be stacked. `run()` raises
`ValueError` if `cascade_mode=True` is combined with `rescue_prompt=True` or a non-default
`max_retries`; pick one.

Only meaningful for a pruning `schema_mode` (`"exact_match"`, `"ner_exact_match"`, `"similarity"`,
`"llm_pruning"`, `"ie_extraction"`) — `"schema"`/`"enhanced"` have nothing to prune from, and
`run()` raises `ValueError` if combined with `cascade_mode=True`, same as it does for
`skip_narrow_schema_filter=True` without `cascade_mode=True`, or `cascade_mode=True` with
`dry_run=True` (there's nothing generated yet to fail/fall back from).

### `database`

Either a `langchain_community.graphs.Neo4jGraph` instance, or a dict with
`uri`/`username`/`password`/`database` keys (an empty dict `{}` falls back to
the `NEO4J_URI`/`NEO4J_USERNAME`/`NEO4J_PASSWORD`/`NEO4J_DATABASE` environment
variables).

### `model`

**The chat backend is pluggable via `model`.** It accepts these kinds of values:

- an **OpenAI** chat model id, e.g. `"gpt-4o"`, `"gpt-4o-mini"`, or a fine-tuned `"ft:..."` id
  — needs `OPENAI_API_KEY`;
- an **Anthropic** chat model id, e.g. `"claude-sonnet-5"`, `"claude-opus-4-1"` — a `"claude"`
  prefix is what selects this backend. Needs `ANTHROPIC_API_KEY` and the optional `anthropic`
  extra: `pip install "text2cypher-composer[anthropic]"`;
- a **Google Gemini** chat model id, e.g. `"gemini-2.5-pro"`, `"gemini-2.5-flash"` — a `"gemini"`
  prefix selects this backend. Needs `GOOGLE_API_KEY` and the optional `google` extra:
  `pip install "text2cypher-composer[google]"`;
- a **DeepSeek** chat model id, e.g. `"deepseek-chat"`, `"deepseek-reasoner"` — a `"deepseek"`
  prefix selects this backend. Needs `DEEPSEEK_API_KEY` and the optional `deepseek` extra:
  `pip install "text2cypher-composer[deepseek]"`;
- or any other LangChain-compatible chat model / `Runnable` — e.g. a `HuggingFacePipeline`
  wrapping a local checkpoint (Llama, Mistral, Qwen, ..., base or fine-tuned via PEFT — see
  "Fine-tuning your own model" below), or a chat model from any other provider you've already
  built yourself — pre-configured with its own generation parameters, the same "bring your own
  object" pattern `embedding_model`/`nlp` use elsewhere in this library.

None of these prefixes overlap (`"gpt-"`/`"o1"`/`"ft:..."` vs. `"claude-"` vs. `"gemini-"` vs.
`"deepseek-"`), so a plain model id string is always resolved unambiguously — see
`llm_backend_for` if you want the exact rule.

### `rescue_prompt` / `max_retries`

`rescue_prompt=True` (default `False`) retries a query that fails to execute or comes back
empty with a second "fix this query" prompt — reusing the same schema/examples context as
`technique`, plus the bad query and an `error_message`. Ported from the miRNAKG rescue-prompt
notebook, and `error_message` concatenates every signal available about the failure, in order:

1. the **native Neo4j error** (code + message) if the query didn't execute at all;
2. an `"Empty result set."` note if it executed but returned nothing;
3. **CyVer's validation report** (`validation.syntax_metadata`/`schema_metadata`/
   `properties_metadata`, both its warning-level notifications and hard errors), which `run()`
   already computes for every query.

Raw Neo4j notifications (deprecated syntax, unknown labels/relationship-types/properties,
cartesian products, etc. — still captured separately on `result.execution_warnings`, see below)
are deliberately left out of `error_message` to save tokens: CyVer's schema/properties validators
already surface the notification codes that matter for a fix-up (unknown labels, relationship
types, property keys), so including both would just duplicate them in the prompt.

```python
result = run(
    input_NL="...",
    model="gpt-4o",
    database={},
    technique="vanilla",
    rescue_prompt=True,
    max_retries=2,  # optional, defaults to 1
)

print(result.initial_cypher)              # what the model generated first
print(result.cypher)                      # the final (possibly rescued) query
print(result.rescued, result.rescue_attempts)
print(result.rescue_error_messages)       # the error_message fed to each rescue attempt, in order
print(result.rescue_prompts)              # the fully-instantiated messages sent for each rescue attempt
print(result.rescue_prompt_tokens)        # token count of each rescue prompt — a list of rescue_attempts numbers
```

Each rescue attempt stops early once one succeeds (executes and returns a non-empty result);
`max_retries` caps how many are tried before giving up — it must be `>= 1` (`run()` raises
`ValueError` for `max_retries=0` or negative). `result.cypher`/`result.executed`/`result.result`/
`result.validation`/`result.execution_error`/`result.execution_warnings` always reflect the
*final* attempt; `result.initial_cypher` and `result.prompt` (the exact messages sent) always
reflect the *first* one.

`result.rescue_error_messages` and `result.rescue_prompts` are parallel lists — one entry per
rescue attempt (`rescue_prompts[i]` is the exact messages sent for `rescue_error_messages[i]`),
both empty if `rescued` is `False` — so you can inspect exactly what each fix-up prompt was told
went wrong, and what it was actually sent.

`result.execution_error`/`result.execution_warnings` — the native Neo4j error (if it didn't
execute) and any Neo4j notifications observed during execution — are populated for the *final*
attempt regardless of whether `rescue_prompt` is used at all; they're the same signals
`rescue_error_messages` is built from (see below), just always available, not only when rescuing.

`result.prompt_tokens` (the initial prompt's `tiktoken` token count) and
`result.rescue_prompt_tokens` (the parallel per-attempt count for `rescue_prompts` — a list of
`rescue_attempts` numbers) let you tally exactly how many extra tokens `rescue_prompt` costs on
top of the initial generation. If `tiktoken` isn't installed
(`pip install "text2cypher-composer[dataset-tools]"`), each count is `None` instead of raising —
`rescue_prompt_tokens` is then a list of `None`s the same length as `rescue_attempts`, not an
empty list (it's only empty when `rescued` is `False`).

### `dry_run`

`dry_run=True` (default `False`) builds and returns the fully-instantiated `prompt` — schema
resolved, RAG examples retrieved, exactly as it would be for a real call — but stops there: no
generation call, no Cypher execution, no CyVer validation, no rescue. `cypher`/`initial_cypher`/
`result`/`validation` are all `None`, and `executed` is `False`:

```python
preview = run(
    input_NL="...",
    model="gpt-4o",
    database={},
    technique="Schema+RAG",
    dataset=dataset,
    dry_run=True,
)

print(preview.prompt)         # the exact messages that *would* be sent to the model
print(preview.prompt_tokens)  # their tiktoken token count — still computed, no API call needed
print(preview.cypher)         # None — nothing was generated
```

Useful to sanity-check what a given `technique`/`schema_mode`/`dataset` combination would
actually send the model — including its token count, so `dry_run=True` alone is enough to compare
how many tokens different `schema_mode`s cost — without spending an API call (or a database
write, for techniques that execute) on it. Incompatible with `rescue_prompt=True` — `run()`
raises `ValueError` if both are passed, since `dry_run` generates nothing to rescue.

### `dataset` (RAG techniques only)

A `RAGDataset` pointing at a persisted Chroma collection of embedded NL
questions, plus the sibling `CypherQueries/` (and `Neo4jOutputs/` for `+O`
techniques) directories holding the corresponding golden Cypher/output for
each embedded question — mirroring the layout under
[`bio2C/evaluating_text2cypher`](bio2C/evaluating_text2cypher):

```python
from text2cypher_composer import RAGDataset

dataset = RAGDataset.from_root("bio2C/evaluating_text2cypher")
# equivalent to:
# RAGDataset(
#     chroma_path="bio2C/evaluating_text2cypher/chroma_db",
#     cypher_dir="bio2C/evaluating_text2cypher/CypherQueries",
#     output_dir="bio2C/evaluating_text2cypher/Neo4jOutputs",
# )

result = run(
    input_NL="...",
    model="gpt-4o",
    database={},  # uses NEO4J_* env vars
    technique="Schema+RAG+O",
    dataset=dataset,
)
```

A plain path string is also accepted for `dataset` and is resolved via
`RAGDataset.from_root`.

### Building your own RAG dataset

`build_rag_example_files` materializes the `NLquestions/`, `CypherQueries/`, and
`Neo4jOutputs/` directories a `RAGDataset` expects, from your own examples —
useful when you don't already have a bio2C-style benchmark on disk:

```python
import pandas as pd
from text2cypher_composer import build_rag_example_files, RAGDataset

df = pd.DataFrame([
    {"question": "How many genes are there?", "cypher": "MATCH (g:Gene) RETURN count(g) AS c"},
    {"question": "List all cancers.", "cypher": "MATCH (c:Cancer) RETURN c.Label AS Cancer"},
])

files = build_rag_example_files(
    df,
    name="my_examples",   # subfolder name, e.g. mirrors bio2C's "1hop"/"3hop"/...
    database={},           # uses NEO4J_* env vars; each query is executed to capture its output
    root="my_dataset",     # NLquestions/CypherQueries/Neo4jOutputs are created under here
)
```

Each row becomes a `question_i.txt`/`cypher_i.txt` pair (1-indexed), plus an
`output_i.txt` obtained by running the query against `database` (a `LIMIT` is
appended if the query doesn't already have one; long string values are
truncated to keep files compact). Call it multiple times with different
`name`s to build up several example groups under the same `root`.

This does **not** build the Chroma index itself. `RAGDataset.index_from_root`
does that: it embeds every file under `root/NLquestions/` and stores them in a
Chroma collection at `root/chroma_db`, then returns a ready-to-use
`RAGDataset`:

```python
dataset = RAGDataset.index_from_root(
    "my_dataset",
    embedding_model="text-embedding-3-large",  # the default — see below for alternatives
)

result = run(
    input_NL="...",
    model="gpt-4o",
    database={},
    technique="RAG",
    dataset=dataset,
)
```

**The embedding backend is pluggable via `embedding_model`.** It accepts three kinds of values:

- an **OpenAI** embedding model id, e.g. `"text-embedding-3-large"` (the default),
  `"text-embedding-3-small"` — needs `OPENAI_API_KEY`;
- a **HuggingFace/sentence-transformers** model id, e.g.
  `"sentence-transformers/all-mpnet-base-v2"` — runs locally, no API key or network calls
  needed at retrieval time. A `"/"` in the string is what selects this backend (that's how
  HuggingFace ids are conventionally shaped; OpenAI's never contain one). Needs the optional
  `local-embeddings` extra: `pip install "text2cypher-composer[local-embeddings]"`;
- or an already-built LangChain `Embeddings` instance, for any other backend — the same "bring
  your own object" pattern `model`/`nlp` use elsewhere in this library.

```python
dataset = RAGDataset.index_from_root(
    "my_dataset",
    embedding_model="sentence-transformers/all-mpnet-base-v2",
)
```

**You never have to pass `embedding_model` again after indexing.** Whichever one was used is
recorded alongside the Chroma collection (a small `<collection_name>.embedding.json` file next
to it under `root/chroma_db/`), so every later call — `index_from_root` reusing the collection,
or a plain `RAGDataset.from_root("my_dataset")` used directly for retrieval — automatically picks
it back up:

```python
dataset = RAGDataset.from_root("my_dataset")   # embedding_model omitted...
result = run(..., technique="RAG", dataset=dataset)  # ...retrieval still uses the model it was indexed with

dataset.recorded_embedding()
# EmbeddingMeta(backend='huggingface', model='sentence-transformers/all-mpnet-base-v2')
```

If you *do* pass an `embedding_model` that doesn't match what a collection was actually indexed
with, `run()`/`index_from_root()`/`retrieve_examples()` raise `ValueError` instead of silently
retrieving garbage — a query embedded with a different model than its documents produces
meaningless similarity scores. Similarly, if a collection was indexed with an OpenAI model and
`OPENAI_API_KEY` isn't set when retrieval is attempted (e.g. a later session where the key was
never exported), a clear `ValueError` is raised rather than a cryptic OpenAI auth error — an
already-built `Embeddings` instance can't be reconstructed from what's recorded either, so it
must be passed again explicitly (recorded as backend `"custom"`).

**It doesn't re-embed from scratch every time either.** If a collection
already exists at `root/chroma_db`, `index_from_root` loads and reuses it as-is
by default — no API calls — so calling it again later (a new session, a
restarted notebook, ...) just loads what's already there. Pass `rebuild=True`
only when you actually want to re-embed, e.g. after adding new examples under
`root/NLquestions/`:

```python
dataset = RAGDataset.index_from_root("my_dataset")               # loads existing, instant
dataset = RAGDataset.index_from_root("my_dataset", rebuild=True) # re-embeds from scratch
```

### Fine-tuning your own model

Two paths to a model specialized on your own question/Cypher examples, ported from
[`bio2C/evaluating_text2cypher/evaluating_text2cypher_gpt.ipynb`](bio2C/evaluating_text2cypher/evaluating_text2cypher_gpt.ipynb)
(dataset preparation) and
[`bio2C/finetuning_LLaMa3-8B/Finetuning_llama3-8b.ipynb`](bio2C/finetuning_LLaMa3-8B/Finetuning_llama3-8b.ipynb)
(LoRA training) — both start from the same leveled gold dataset.

**1. Prepare the dataset.** `load_finetune_levels` loads and concatenates bio2C-style leveled
gold JSON files (`nodeLevel.json`, `1hop.json`, ...; each a list of `{"question", "cypher"}`
records), tagging every row with its source `level` and a bio2C-style `ID`:

```python
from text2cypher_composer import load_finetune_levels, max_cypher_tokens, split_finetune_dataset

df = load_finetune_levels({
    "nodeLevel": "FTdataset/nodeLevel.json",
    "1hop": "FTdataset/1hop.json",
    "2hop": "FTdataset/2hop.json",
})

max_cypher_tokens(df)  # {'longest_cypher': ..., 'n_tokens': ..., 'max_tokens': ...} — needs `tiktoken`

train_df, test_df = split_finetune_dataset(df, test_frac=0.10, random_state=42)
```

**2. Export it.** `write_local_finetune_dataset` writes the format `finetune_lora` reads back in;
`build_gpt_finetune_jsonl` writes a chat-format `.jsonl` ready to upload to
[OpenAI's fine-tuning GUI](https://platform.openai.com/finetune) — one `{"messages": [...]}` line
per example, with an optional system message:

```python
from text2cypher_composer import write_local_finetune_dataset, build_gpt_finetune_jsonl

write_local_finetune_dataset(train_df, "FTdataset_local.json")
build_gpt_finetune_jsonl(train_df, "FTdataset_GPTviaGUI.jsonl")
```

The resulting `ft:...` model id from the GPT GUI plugs straight into `run(model="ft:...")`, like
any other OpenAI model id.

**3. Or LoRA-finetune a local model yourself**, e.g. LLaMA — `finetune_lora` needs the optional
`finetune` extra (`pip install "text2cypher-composer[finetune]"`: torch, transformers, peft,
datasets) and, realistically, a GPU. The default base model, `meta-llama/Llama-3.1-8B`, is
**gated** on Hugging Face: accept its license on the model page and run `huggingface-cli login`
(or set `HF_TOKEN`) before calling `finetune_lora`/`load_finetuned_model`, otherwise you'll hit a
401/403:

```python
from text2cypher_composer import LoRATrainingConfig, finetune_lora, load_finetuned_model

result = finetune_lora(
    train_df,  # "question"/"cypher" columns
    config=LoRATrainingConfig(output_dir="./llama3_lora_mirnakgt2c"),
)

# Load the adapter back as a ready-to-use Runnable:
llama_ft = load_finetuned_model("meta-llama/Llama-3.1-8B", result.adapter_path)

result = run(
    input_NL="...",
    model=llama_ft,      # <-- an already-built Runnable, instead of a string
    database={},
    technique="vanilla",
)
```

`LoRATrainingConfig` defaults match the notebook (4-bit-quantized `meta-llama/Llama-3.1-8B`,
LoRA rank 16 on the attention/MLP projections, 3 epochs) — override any field to change them.
`load_finetuned_model` (or a `HuggingFacePipeline` you build yourself, fine-tuned or not) works
the same "bring your own `Runnable`" way `model` already supports for any non-OpenAI chat model.

**`base_model` isn't limited to Llama.** `DEFAULT_TARGET_MODULES` (the LoRA adapter's
attention/MLP projection names) applies unchanged to any Llama-style architecture, which also
covers Mistral and Qwen2 checkpoints — just override `base_model` (and `output_dir`, so adapters
don't collide):

```python
result = finetune_lora(
    train_df,
    config=LoRATrainingConfig(
        base_model="Qwen/Qwen2.5-7B-Instruct",  # or "mistralai/Mistral-7B-Instruct-v0.3"
        output_dir="./qwen2.5_lora_mirnakgt2c",
    ),
)

qwen_ft = load_finetuned_model("Qwen/Qwen2.5-7B-Instruct", result.adapter_path)
```

Unlike `meta-llama/Llama-3.1-8B`, most Mistral/Qwen repos aren't gated — but that varies by
repo, so check the model page for whichever one you pick.

### Introspection

A few helpers that need no database/model/dataset — useful for building a UI
around `run()`, or for validating a `technique` string upfront:

```python
from text2cypher_composer import (
    list_techniques, describe_technique, list_technique_info,
    get_prompt_template, get_all_prompt_templates,
)

list_techniques()
# ['vanilla', 'Schema', 'RAG', 'RAG+O', 'Schema+RAG', 'Schema+RAG+O']

describe_technique("Schema+RAG+O")
# TechniqueInfo(technique='Schema+RAG+O', uses_schema=True, uses_rag=True, uses_output=True)

list_technique_info()  # describe_technique() for every technique

get_prompt_template("Schema+RAG")
# [{'role': 'system', 'content': '...'}, {'role': 'human', 'content': '...{enhanced_schema}...{examples}...{question}...'}]

get_all_prompt_templates()  # get_prompt_template() for every technique, keyed by technique value
```

`get_prompt_template`/`get_all_prompt_templates` return the **parametric**
prompt (placeholders left as literal text, e.g. `{question}`) — for the
fully-instantiated prompt actually sent to the model on a given call, use
`Text2CypherResult.prompt` (returned by `run()`).

### Bulk evaluation

`evaluate_technique` runs `technique` over a whole gold test set and reports
Jaro-Winkler, normalized Levenshtein, Jaccard, Coverage, and pass@k:

```python
import pandas as pd
from text2cypher_composer import evaluate_technique

gold_df = pd.DataFrame([
    {"question": "How many genes are there?", "query": "MATCH (g:Gene) RETURN count(g) AS c"},
    {"question": "List all cancers.", "query": "MATCH (c:Cancer) RETURN c.Label AS Cancer"},
])

report = evaluate_technique(
    gold_df,
    model="gpt-4o",
    database={},          # uses NEO4J_* env vars
    technique="Schema",
    k=3,                   # optional, defaults to 1
    rescue_prompt=True,    # optional, defaults to False — see `run()`'s rescue_prompt
    max_retries=2,         # optional, defaults to 1
)

print(report.summary)
# EvaluationSummary(technique='Schema', model='gpt-4o', n_questions=2, k=3,
#                    mean_jaro_winkler=..., mean_levenshtein=..., mean_jaccard=...,
#                    mean_coverage=..., pass_at_k={1: ..., 2: ..., 3: ...})

report.to_dataframe()  # one row per question, with a pass@1/pass@2/pass@3 column each
```

For each question, `k` independent Cypher completions are generated (via
`run()`, so every attempt goes through the full technique pipeline —
schema/RAG retrieval, execution, CyVer validation) and compared against the
gold query's actual result rows (obtained by executing `row["query"]`
against `database`). `jaro_winkler`/`levenshtein`/`jaccard`/`coverage` are
computed on the first attempt; `pass@j` (for every `j` in `1..k`) is whether
at least one of the first `j` attempts exactly reproduced the gold result
(`coverage == 1.0`). `k` defaults to `1` (only `pass@1` is reported); raise
it to also get `pass@2`, ..., `pass@k`.

`jaro_winkler_similarity`, `normalized_levenshtein_similarity`,
`jaccard_similarity`, and `coverage_similarity` are exported standalone too,
for scoring a single (gold, predicted) pair without a full bulk run. Jaccard
and Coverage compare two queries' *result rows* rather than their Cypher
text: since Neo4j doesn't guarantee row order without `ORDER BY`, rows are
greedily bipartite-matched by similarity before being compared whenever the
gold query has none.

`evaluate_technique` also forwards `rescue_prompt`/`max_retries` (default `False`/`1`, same as
`run()`) to every attempt.

Besides the metric/pass@j columns, `report.to_dataframe()` also carries, per question: any
columns `gold_df` had beyond `question`/`query` (e.g. bio2C's `"ID"`/`"level"`, if you built
`gold_df` with `load_finetune_levels`), `prompt`/`prompt_tokens` (the exact messages sent for the
first attempt and their `tiktoken` token count — `None` if `tiktoken` isn't installed; compare it
across `technique`/`schema_mode` rows to see how many tokens schema filtering saves),
`gold_data`/`predicted_data` (the gold/generated query's result rows), `execution_error`/
`execution_warnings` (the native Neo4j error/notifications from the first attempt's actual
execution, populated regardless of `rescue_prompt`), `rescued`/`rescue_attempts` (whether — and
how many retries — the first attempt needed to stop failing/coming back empty, when
`rescue_prompt=True`; `False`/`0` otherwise), `rescue_error_messages`/`rescue_prompts` (the
`error_message`/fully-instantiated messages sent for each retry) and `rescue_prompt_tokens`
(their token counts, to tally how many extra tokens `rescue_prompt` costs across the whole gold
set), and `retrieved_example_ids`/`retrieved_example_distances` for RAG techniques (`None`
otherwise).

**`save_evaluation_report` persists a report as a `.pkl`/`.xlsx` pair**, one per (model,
technique) — named `evaluating_text2cypher_{model}_{technique}.{pkl,xlsx}`:

```python
from text2cypher_composer import save_evaluation_report

paths = save_evaluation_report(report, "evaluating_cypher_jaccard RTT/gpt-4o-mini")
# {"pkl": PosixPath(".../evaluating_text2cypher_gpt-4o_Schema.pkl"),
#  "xlsx": PosixPath(".../evaluating_text2cypher_gpt-4o_Schema.xlsx")}
```

The `.pkl` holds `report.to_dataframe()` as-is — `prompt`/`gold_data`/`predicted_data`/
`retrieved_example_*` stay native Python lists/dicts, no extra dependency needed. The `.xlsx` is
the same table with those columns stringified (Excel has no list/dict type) and needs the
optional `excel` dependency:

```bash
pip install "text2cypher-composer[excel]"
```

## Contributing

This library **composes** a fixed set of techniques behind `run()` — adding a new one (a new
prompt template, plugged into the same schema/RAG/validation/rescue machinery every existing
technique shares) is the most natural contribution. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the step-by-step guide, including how to propose it and
open a pull request if you don't already have push access to this repository.
