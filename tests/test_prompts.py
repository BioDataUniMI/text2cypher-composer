from text2cypher_composer import Technique, get_all_prompt_templates, get_prompt_template
from text2cypher_composer.techniques import OUTPUT_AUGMENTED_TECHNIQUES, RAG_TECHNIQUES, SCHEMA_TECHNIQUES


def _content(technique):
    return "\n".join(m["content"] for m in get_prompt_template(technique))


def test_every_technique_has_a_template():
    all_templates = get_all_prompt_templates()
    assert set(all_templates.keys()) == {t.value for t in Technique}


def test_placeholders_match_technique_flags():
    for t in Technique:
        content = _content(t)
        assert "{question}" in content

        if t in SCHEMA_TECHNIQUES:
            assert "{enhanced_schema}" in content
        else:
            assert "{enhanced_schema}" not in content

        if t in RAG_TECHNIQUES:
            assert "{examples}" in content
        else:
            assert "{examples}" not in content


def test_output_augmented_prompts_mention_output():
    for t in OUTPUT_AUGMENTED_TECHNIQUES:
        assert "output" in _content(t).lower()


def test_get_prompt_template_accepts_string_or_enum():
    assert get_prompt_template("vanilla") == get_prompt_template(Technique.VANILLA)
