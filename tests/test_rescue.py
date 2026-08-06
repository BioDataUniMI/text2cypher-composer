from text2cypher_composer.rescue import build_error_message, needs_rescue, rescue_messages
from text2cypher_composer.validation import CypherValidationReport


def _validation(syntax_valid=True, syntax_metadata=None, schema_metadata=None, properties_metadata=None):
    return CypherValidationReport(
        syntax_valid=syntax_valid,
        syntax_metadata=syntax_metadata or [],
        schema_score=1.0,
        schema_metadata=schema_metadata or [],
        properties_score=1.0,
        properties_metadata=properties_metadata or [],
    )


def test_needs_rescue_false_when_executed_with_valid_nonempty_result():
    assert needs_rescue(True, [{"c": 1}], _validation()) is False


def test_needs_rescue_true_when_not_executed():
    assert needs_rescue(False, None, _validation()) is True


def test_needs_rescue_true_when_empty_result():
    assert needs_rescue(True, [], _validation()) is True


def test_needs_rescue_true_when_syntax_invalid_even_if_executed():
    assert needs_rescue(True, [{"c": 1}], _validation(syntax_valid=False)) is True


def test_build_error_message_reports_empty_result_set():
    msg = build_error_message(True, [], _validation())
    assert "Empty result set" in msg


def test_build_error_message_includes_cyver_warnings_and_errors():
    validation = _validation(
        syntax_valid=False,
        syntax_metadata=[{"code": "SYN001", "description": "bad syntax"}],
        schema_metadata=[{"code": "SCH001", "description": "unknown label"}],
    )
    msg = build_error_message(False, None, validation)
    assert "SYN001" in msg and "bad syntax" in msg
    assert "SCH001" in msg and "unknown label" in msg


def test_build_error_message_falls_back_when_nothing_to_report():
    msg = build_error_message(True, [{"c": 1}], _validation())
    assert msg  # non-empty, doesn't crash on the "no issues" case


def test_rescue_messages_dispatch_by_technique_flags():
    vanilla = rescue_messages(uses_schema=False, uses_rag=False)
    schema = rescue_messages(uses_schema=True, uses_rag=False)
    rag = rescue_messages(uses_schema=False, uses_rag=True)
    schema_rag = rescue_messages(uses_schema=True, uses_rag=True)

    # each combination gets a distinct template, and only the relevant placeholders
    templates = {"vanilla": vanilla[1][1], "schema": schema[1][1], "rag": rag[1][1], "schema_rag": schema_rag[1][1]}
    assert len({t for t in templates.values()}) == 4

    assert "{enhanced_schema}" not in templates["vanilla"] and "{examples}" not in templates["vanilla"]
    assert "{enhanced_schema}" in templates["schema"] and "{examples}" not in templates["schema"]
    assert "{enhanced_schema}" not in templates["rag"] and "{examples}" in templates["rag"]
    assert "{enhanced_schema}" in templates["schema_rag"] and "{examples}" in templates["schema_rag"]

    for template in templates.values():
        assert "{question}" in template
        assert "{query}" in template
        assert "{error_message}" in template
