# text2cypher-composer

[![PyPI](https://img.shields.io/pypi/v/text2cypher-composer?label=PyPI&logo=pypi)](https://pypi.org/project/text2cypher-composer/)
[![Pypi total project downloads](https://static.pepy.tech/badge/text2cypher-composer)](https://pepy.tech/project/text2cypher-composer)

[![GitHub Action: Publish to PyPI](https://github.com/BioDataUniMI/text2cypher-composer/actions/workflows/publish.yml/badge.svg)](https://github.com/BioDataUniMI/text2cypher-composer/actions/workflows/publish.yml)

<!-- Quality/coverage badges below are placeholders — this project isn't registered on
     SonarCloud/Codacy/Code Climate yet. Replace the project key / badge ID in each URL
     (and set up the corresponding integration) before relying on them; until then they'll
     show as broken/"unknown". Coveralls' URL is already valid as-is (no token needed), it
     just needs Coveralls enabled for this repo. -->
[![SonarCloud Quality](https://sonarcloud.io/api/project_badges/measure?project=BioDataUniMI_text2cypher-composer&metric=alert_status)](https://sonarcloud.io/dashboard?id=BioDataUniMI_text2cypher-composer)
[![Codacy Maintainability](https://app.codacy.com/project/badge/Grade/REPLACE_WITH_CODACY_BADGE_ID)](https://www.codacy.com/gh/BioDataUniMI/text2cypher-composer/dashboard)
[![Maintainability](https://api.codeclimate.com/v1/badges/REPLACE_WITH_CODECLIMATE_ID/maintainability)](https://codeclimate.com/github/BioDataUniMI/text2cypher-composer/maintainability)
[![Code Climate Coverage](https://api.codeclimate.com/v1/badges/REPLACE_WITH_CODECLIMATE_ID/test_coverage)](https://codeclimate.com/github/BioDataUniMI/text2cypher-composer/test_coverage)
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

`schema_mode`/`nlp`/`similarity_threshold` are disallowed for techniques
that don't use the schema (`"vanilla"`, `"RAG"`, `"RAG+O"`) — `run()` raises
`ValueError` if you pass them there. The available modes are listed by
`list_schema_modes()`, and the individual pruning functions
(`exact_match_prune`, `ner_exact_match_prune`, `similarity_prune`,
`llm_prune`, `mask_entities`) are exported standalone too, for pruning a
schema outside of a full `run()` call.

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
notebook, with one change: `error_message` is built from **CyVer's validation report** (both its
warning-level notifications and hard errors — `validation.syntax_metadata`/`schema_metadata`/
`properties_metadata`), which `run()` already computes for every query, rather than from a raw
Neo4j exception string:

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
```

Each rescue attempt stops early once one succeeds (executes and returns a non-empty result);
`max_retries` caps how many are tried before giving up. `result.cypher`/`result.executed`/
`result.result`/`result.validation` always reflect the *final* attempt; `result.initial_cypher`
and `result.prompt` (the exact messages sent) always reflect the *first* one.

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

print(preview.prompt)   # the exact messages that *would* be sent to the model
print(preview.cypher)   # None — nothing was generated
```

Useful to sanity-check what a given `technique`/`schema_mode`/`dataset` combination would
actually send the model, without spending an API call (or a database write, for techniques that
execute) on it. Incompatible with `rescue_prompt=True` — `run()` raises `ValueError` if both are
passed, since `dry_run` generates nothing to rescue.

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

## Contributing

This library **composes** a fixed set of techniques behind `run()` — adding a new one (a new
prompt template, plugged into the same schema/RAG/validation/rescue machinery every existing
technique shares) is the most natural contribution. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the step-by-step guide, including how to propose it and
open a pull request if you don't already have push access to this repository.
