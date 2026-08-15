import pytest

from text2cypher_composer.tokens import count_message_tokens

pytest.importorskip("tiktoken")


def test_count_message_tokens_counts_all_messages():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "human", "content": "How many genes are there?"},
    ]
    n = count_message_tokens(messages, model="gpt-4o")
    assert isinstance(n, int)
    assert n > 0


def test_count_message_tokens_more_content_means_more_tokens():
    short = [{"role": "human", "content": "Hi"}]
    long = [{"role": "human", "content": "Hi " * 200}]
    assert count_message_tokens(long, model="gpt-4o") > count_message_tokens(short, model="gpt-4o")


def test_count_message_tokens_falls_back_for_unknown_model_name():
    messages = [{"role": "human", "content": "How many genes are there?"}]
    n = count_message_tokens(messages, model="RunnableLambda")
    assert isinstance(n, int)
    assert n > 0


def test_count_message_tokens_none_when_tiktoken_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("no tiktoken")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert count_message_tokens([{"role": "human", "content": "hi"}]) is None
