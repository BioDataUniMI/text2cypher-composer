"""Pretty-printing a `Text2CypherResult` — pulled out of the demo notebook so it's directly
importable instead of copy-pasted into every script/notebook that wants the same summary.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .core import Text2CypherResult


def _print_messages(messages: List[Dict[str, str]]) -> None:
    for message in messages:
        print(f"  [{message['role']}]")
        for line in message["content"].splitlines():
            print(f"    {line}")


def show(result: Text2CypherResult, show_prompt: bool = False) -> None:
    """Pretty-print a `Text2CypherResult`.

    CyVer validation (syntax validity, schema-alignment score, property-access
    score) is run on every generated query and is always printed, regardless
    of whether the query executed. The result rows are printed too, if the
    query executed successfully. `execution_error`/`execution_warnings` — the
    native Neo4j error/notifications from the *final* attempt's actual
    execution — are always printed too, independently of `rescue_prompt`.

    `result.prompt` is the exact messages sent for the *initial* generation
    attempt; if rescued, `result.rescue_prompts` holds one more fully-
    instantiated prompt per rescue attempt. Pass `show_prompt=True` to print
    all of them (not just the initial one). `result.prompt_tokens` (via
    `tiktoken`, `None` if it isn't installed) is the initial prompt's token
    count — handy for comparing prompt size across `technique`/`schema_mode`
    (how much schema filtering saves); `result.rescue_prompt_tokens` is the
    parallel per-attempt count for `rescue_prompts` (a list of
    `rescue_attempts` numbers when rescued — how many extra tokens
    `rescue_prompt` costs). `result.cascade_mode_level`/`cascade_mode_attempts`,
    `result.adaptive_rag_level`/`adaptive_rag_attempts`, and
    `result.self_verification_passed`/`self_verification_reasoning` are each
    printed too, if present.
    """
    print(f"Technique:        {result.technique}")
    print(f"Model:            {result.model}")
    print(f"Prompt tokens:    {result.prompt_tokens}")

    if show_prompt:
        all_prompts = [result.prompt] + result.rescue_prompts
        if len(all_prompts) == 1:
            print("\nFull instantiated prompt:")
            _print_messages(all_prompts[0])
        else:
            print(f"\nFull instantiated prompts ({len(all_prompts)}: 1 initial + {len(all_prompts) - 1} rescue):")
            for i, messages in enumerate(all_prompts):
                label = "initial" if i == 0 else f"rescue attempt {i}"
                print(f"  --- {label} ---")
                _print_messages(messages)

    if result.dry_run:
        print("\n[dry_run] Nothing was generated, executed, or validated — prompt only.")
        return

    print(f"\nGenerated Cypher:\n{result.cypher}\n")

    if result.executed:
        print(f"Result ({len(result.result)} rows):")
        for row in result.result[:5]:
            print(" ", row)
        if len(result.result) > 5:
            print(f"  ... and {len(result.result) - 5} more rows")
    else:
        print("Execution FAILED (see execution_error/CyVer report below for why).")

    if result.execution_error:
        print(f"\nExecution error: {result.execution_error}")
    if result.execution_warnings:
        print("Execution warnings (Neo4j notifications):")
        for warning in result.execution_warnings:
            print(f"  {warning}")

    v = result.validation
    print("\nCyVer validation report:")
    print(f"  Syntax valid:      {v.syntax_valid}")
    if v.syntax_metadata:
        print(f"  Syntax issues:     {v.syntax_metadata}")
    print(f"  Schema score:      {v.schema_score:.2f}  (1.0 = fully aligned with the graph schema)")
    if v.schema_metadata:
        print(f"  Schema issues:     {v.schema_metadata}")
    print(f"  Properties score:  {v.properties_score}")
    if v.properties_metadata:
        print(f"  Property issues:   {v.properties_metadata}")

    if result.rescue_error_messages:
        print(f"\nRescue attempts: {result.rescue_attempts}")
        print(f"Rescue prompt tokens: {result.rescue_prompt_tokens}")  # a list of rescue_attempts numbers
        for i, msg in enumerate(result.rescue_error_messages, start=1):
            print(f"  [attempt {i}] error_message sent to the fix-up prompt:")
            for line in msg.splitlines():
                print(f"    {line}")

    if result.cascade_mode_level:
        print(f"\nSchema fallback rung used: {result.cascade_mode_level}  (of {result.cascade_mode_attempts} tried)")
        print(f"Cascade prompt tokens: {result.cascade_mode_prompt_tokens}")  # one per rung tried

    if result.adaptive_rag_level:
        print(f"\nRAG expansion rung used: {result.adaptive_rag_level}  (of {result.adaptive_rag_attempts} tried)")
        print(f"Adaptive RAG prompt tokens: {result.adaptive_rag_prompt_tokens}")  # one per rung tried

    if result.self_verification_passed is not None:
        print(f"\nSelf-verification: {'passed' if result.self_verification_passed else 'failed'}")
        print(f"Reasoning: {result.self_verification_reasoning}")
