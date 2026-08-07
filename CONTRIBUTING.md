# Contributing

`text2cypher_composer` **composes** a fixed set of Text2Cypher prompting
strategies (`vanilla`, `Schema`, `RAG`, `RAG+O`, `Schema+RAG`,
`Schema+RAG+O`) behind one `run()` entrypoint. The most natural — and most
welcome — contribution is a **new technique**. This guide walks through
exactly that, plus the mechanics of proposing it upstream if you don't have
push access to this repository (an "Issue" and a "Pull Request", explained
below).

Bug fixes, docs improvements, and anything else are welcome too — jump to
[Opening an issue or pull request](#opening-an-issue-or-pull-request) for
those.

## Setup

```bash
git clone https://github.com/BioDataUniMI/text2cypher-composer.git
cd text2cypher-composer
pip install -e ".[finetune,local-embeddings,dataset-tools,test]"
```

(`finetune`/`local-embeddings`/`dataset-tools` are only needed if you're
touching those areas; `test` installs `pytest` to run the test suite in
`tests/`.)

## Adding a new technique

A "technique" is: a prompt template, plus flags for whether it needs the
graph schema and/or a RAG example dataset. Everything else — schema
extraction, RAG retrieval, CyVer validation, the rescue-prompt retry loop —
is shared infrastructure that every technique gets for free. Concretely,
adding one touches these files:

1. **`text2cypher_composer/techniques.py`** — add a member to the
   `Technique` enum (its string value is what users pass as `run()`'s
   `technique=` argument), and, if applicable, add it to the
   `SCHEMA_TECHNIQUES` / `RAG_TECHNIQUES` / `OUTPUT_AUGMENTED_TECHNIQUES`
   sets. These sets are what drive everything downstream — `run()`'s
   `schema_mode`/`dataset` validation, `describe_technique()`, and (see next
   point) which rescue-prompt template gets used.

2. **`text2cypher_composer/prompts.py`** — add its template string to the
   `TEMPLATES` dict. Follow the existing placeholder convention:
   `{question}` always; `{enhanced_schema}` if (and only if) the technique is
   in `SCHEMA_TECHNIQUES`; `{examples}` if (and only if) it's in
   `RAG_TECHNIQUES`. `ChatPromptTemplate` fills these in via `run()`'s
   `format_kwargs`, so an unused or missing placeholder will error at call
   time — this is intentional; it's the sets from step 1 that decide which
   placeholders `run()` supplies.

3. **`text2cypher_composer/rescue.py`** — nothing to do here, usually.
   `rescue_messages()` dispatches purely on `uses_schema`/`uses_rag`, so a
   correctly-classified new technique (step 1) automatically gets a matching
   rescue template. Only touch this file if your technique needs a
   genuinely different rescue prompt than the schema/RAG combination it
   falls into.

4. **`README.md`** — add a row to the technique table (near the top), and
   any technique-specific behavior worth documenting (see how `schema_mode`
   or `rescue_prompt` are documented for the existing techniques).

5. **`demo_text2cypher_composer.ipynb`** — add a demo section. The notebook
   is generated, not hand-edited: add your section to
   [`scripts/build_demo_notebook.py`](scripts/build_demo_notebook.py) (look
   at how e.g. `"RAG"` or `"Schema+RAG"` are demoed for the shape to match),
   then regenerate it:

   ```bash
   python3 scripts/build_demo_notebook.py
   ```

   Since there's no live Neo4j/OpenAI access in most dev environments,
   sanity-check the generated notebook's structure/JSON validity rather than
   expecting to execute it end-to-end:

   ```python
   import nbformat
   nbformat.validate(nbformat.read("demo_text2cypher_composer.ipynb", as_version=4))
   ```

6. **`tests/`** — add a test confirming your technique is correctly wired:
   at minimum, that it appears in `list_techniques()`/`describe_technique()`
   with the right `uses_schema`/`uses_rag`/`uses_output` flags, and that
   `get_prompt_template()` produces the placeholders you expect. See
   `tests/test_techniques.py` for the existing pattern.

Run the test suite before opening a PR:

```bash
pytest
```

### Something that doesn't fit "a new technique"

Not every idea fits this shape — e.g. a new `schema_mode`, a new RAG
embedding backend, or a new evaluation metric are all real, valuable
contributions that touch different files (`schema_modes.py`, `embeddings.py`,
`metrics.py`, respectively). The same overall shape applies: implement it,
document it in `README.md`, demo it in the notebook, and add a test. If
you're unsure whether something fits the project's scope at all, open an
issue first (see below) before investing time in the implementation.

## Opening an issue or pull request

This project is hosted on GitHub at
[BioDataUniMI/text2cypher-composer](https://github.com/BioDataUniMI/text2cypher-composer).
Two things you can open there:

- **An issue** is a discussion ticket — "I'd like to add technique X, does
  this fit?", or a bug report. Good first step for anything nontrivial or
  if you're unsure it fits; not required for small, obviously-in-scope
  changes.
- **A pull request (PR)** is a proposed set of code changes submitted for
  review, which a maintainer can then merge into `main`. This is how an
  actual contribution — a new technique, a bug fix, a doc improvement — gets
  in.

If you don't have push access to this repository (the common case), the
flow is: **fork, branch, commit, push to your fork, open a PR from there.**

```bash
# 1. Fork the repo on GitHub (button on the repo page), then clone your fork:
git clone https://github.com/<your-username>/text2cypher-composer.git
cd text2cypher-composer
git remote add upstream https://github.com/BioDataUniMI/text2cypher-composer.git

# 2. Create a branch for your change:
git checkout -b add-my-technique

# 3. Make your changes (see "Adding a new technique" above), then:
git add <the files you changed>
git commit -m "Add MyTechnique: short description of what it does"
git push origin add-my-technique
```

Then go to your fork on GitHub — it'll offer a "Compare & pull request"
button against `BioDataUniMI/text2cypher-composer`'s `main` branch. Opening
that PR pre-fills a checklist (from
[`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md))
matching the steps above — fill it in so a reviewer can see at a glance
what's covered.

If you have push access to this repository directly (maintainers,
collaborators), you can skip the fork and just push a branch, then open the
PR from that branch instead.

To propose a new technique before writing any code, open an issue using the
"New technique proposal" template — it asks for exactly the information a
maintainer needs to give useful feedback before you invest time in an
implementation.

## Releasing (maintainers)

Pushing a tag matching `v*` (e.g. `v0.1.1`) triggers
[`.github/workflows/publish.yml`](.github/workflows/publish.yml), which builds
the package and publishes it to PyPI — no API token stored anywhere; it
authenticates via PyPI's **Trusted Publishing** (OIDC), scoped to this exact
repo/workflow.

**One-time setup** (already done for this repo, but needed again if the repo
moves or a workflow file gets renamed): on
[pypi.org](https://pypi.org) → the `text2cypher-composer` project (or, before
it exists yet, [pypi.org/manage/account/publishing](https://pypi.org/manage/account/publishing/)
for a "pending" publisher) → *Publishing* → add a new GitHub publisher with:

| Field | Value |
|---|---|
| Owner | `BioDataUniMI` |
| Repository name | `text2cypher-composer` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

**To cut a release:**

1. Bump `version` in `pyproject.toml` (PyPI never allows re-uploading the same version).
2. Commit and push that change.
3. Tag it and push the tag: `git tag v0.1.2 && git push origin v0.1.2`.
4. Watch the *Actions* tab — `build` runs first (and fails fast if the tag doesn't match
   `pyproject.toml`'s version), then `publish` uploads to PyPI.

### Testing a release on TestPyPI first

[`.github/workflows/publish-testpypi.yml`](.github/workflows/publish-testpypi.yml) does the same
build, but publishes to **TestPyPI** instead, and is **manually triggered** (Actions tab → "Publish
to TestPyPI" → *Run workflow*) rather than tag-triggered — TestPyPI also refuses to re-upload an
existing version, and most commits don't bump `pyproject.toml`'s version, so an automatic trigger
would fail far more often than it'd succeed.

This is a **separate** Trusted Publisher registration from the real-PyPI one above — same repo,
but its own workflow filename and environment, set up at
[test.pypi.org/manage/account/publishing](https://test.pypi.org/manage/account/publishing/):

| Field | Value |
|---|---|
| Owner | `BioDataUniMI` |
| Repository name | `text2cypher-composer` |
| Workflow name | `publish-testpypi.yml` |
| Environment name | `testpypi` |

If you already registered one on test.pypi.org pointing at `publish.yml` (the real-PyPI
workflow's filename), edit it to `publish-testpypi.yml` instead — the workflow filename in the
Trusted Publisher config must exactly match the file that's actually running, or the OIDC
exchange fails at publish time.

Once that's set up, running the workflow needs no local `twine`/token at all — though the manual
local flow (`python -m build && twine upload --repository testpypi dist/*`) still works too, if
you'd rather not push to GitHub first.
