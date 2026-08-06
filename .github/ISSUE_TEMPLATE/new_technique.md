---
name: New technique proposal
about: Propose a new Text2Cypher prompting technique for run()
title: "[Technique] "
labels: new-technique
---

<!--
See CONTRIBUTING.md's "Adding a new technique" section for the full guide.
This template just collects the info a maintainer needs to give useful
feedback before you invest time in an implementation.
-->

**Name of the technique**
<!-- e.g. "Schema+CoT" — becomes run()'s technique="..." string. -->

**What does it do differently from the existing six?**
<!-- vanilla / Schema / RAG / RAG+O / Schema+RAG / Schema+RAG+O -->

**Does it need the graph schema?**
<!-- yes/no — decides whether it belongs in SCHEMA_TECHNIQUES -->

**Does it need a RAG example dataset (`dataset=`)?**
<!-- yes/no — decides whether it belongs in RAG_TECHNIQUES -->

**Does it need the retrieved examples' Neo4j output too (a "+O" variant)?**
<!-- yes/no — decides whether it belongs in OUTPUT_AUGMENTED_TECHNIQUES -->

**Draft prompt template (if you have one)**
<!--
Following the {question} / {enhanced_schema} / {examples} placeholder
convention from text2cypher_composer/prompts.py.
-->

**Motivation / reference**
<!-- A paper, notebook, or experiment this is based on, if any. -->
