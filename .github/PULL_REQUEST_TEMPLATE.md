## What does this PR do?

<!-- Short description. Link the issue it addresses, if any (e.g. "Closes #12"). -->

## If this adds a new technique

See CONTRIBUTING.md's "Adding a new technique" for details on each item.

- [ ] Added a `Technique` member in `text2cypher_composer/techniques.py`, and to
      `SCHEMA_TECHNIQUES`/`RAG_TECHNIQUES`/`OUTPUT_AUGMENTED_TECHNIQUES` as applicable
- [ ] Added its prompt template to `text2cypher_composer/prompts.py`'s `TEMPLATES`
- [ ] Added a row to the technique table in `README.md`
- [ ] Added a demo section via `scripts/build_demo_notebook.py`, and regenerated
      `demo_text2cypher_composer.ipynb`
- [ ] Added a test in `tests/` confirming it's wired correctly

## Checklist (all PRs)

- [ ] `pytest` passes
- [ ] `python3 -m py_compile text2cypher_composer/*.py` passes
- [ ] Relevant docs (`README.md`) updated
