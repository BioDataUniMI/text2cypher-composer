from langchain_core.runnables import RunnableLambda

from text2cypher_composer.llm import llm_backend_for, resolve_model, resolve_pruning_model


def test_backend_detection():
    assert llm_backend_for("gpt-4o") == "openai"
    assert llm_backend_for("gpt-4o-mini") == "openai"
    assert llm_backend_for("ft:gpt-4o-mini:acme::abc123") == "openai"
    assert llm_backend_for("claude-sonnet-5") == "anthropic"
    assert llm_backend_for("claude-opus-4-1") == "anthropic"
    assert llm_backend_for("gemini-2.5-pro") == "google"
    assert llm_backend_for("gemini-2.5-flash") == "google"
    assert llm_backend_for("deepseek-chat") == "deepseek"
    assert llm_backend_for("deepseek-reasoner") == "deepseek"


def test_resolve_model_passes_through_custom_runnable():
    fake = RunnableLambda(lambda x: x)
    assert resolve_model(fake) is fake


def test_resolve_pruning_model_passes_through_custom_runnable():
    fake = RunnableLambda(lambda x: x)
    assert resolve_pruning_model(fake) is fake
