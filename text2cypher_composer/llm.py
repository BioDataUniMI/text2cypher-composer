from typing import Union

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

ModelLike = Union[str, Runnable]

DEFAULT_MAX_TOKENS = 1024
STOP_SEQUENCES = ["\nCypherResult:"]


def resolve_model(model: ModelLike) -> Runnable:
    """Resolve `model` into an invocable LangChain Runnable.

    A string is treated as an OpenAI chat model id (e.g. "gpt-4o",
    "gpt-4o-mini", or a fine-tuned "ft:..." id) and wrapped in ChatOpenAI.
    Anything else must already be a LangChain-compatible chat model /
    Runnable (e.g. a HuggingFacePipeline wrapping a local LLaMA checkpoint,
    base or fine-tuned via PEFT) and is used as-is, generation parameters
    included.
    """
    if isinstance(model, str):
        llm = ChatOpenAI(model_name=model, temperature=0)
        return llm.bind(stop=STOP_SEQUENCES, max_tokens=DEFAULT_MAX_TOKENS)
    if isinstance(model, Runnable):
        return model
    raise TypeError(
        "`model` must be an OpenAI model id (str) or a LangChain-compatible "
        "chat model / Runnable instance."
    )


def resolve_pruning_model(model: ModelLike) -> Runnable:
    """Resolve `model` into a chat model suitable for `.with_structured_output(...)`.

    Unlike `resolve_model`, a string is wrapped in a plain (unbound)
    ChatOpenAI — `resolve_model`'s `.bind(...)` wrapper is a RunnableBinding
    that may not reliably proxy `.with_structured_output()` across LangChain
    versions. Anything else is used as-is and must itself support
    `.with_structured_output()`.
    """
    if isinstance(model, str):
        return ChatOpenAI(model_name=model, temperature=0)
    if isinstance(model, Runnable):
        return model
    raise TypeError(
        "`model` must be an OpenAI model id (str) or a LangChain-compatible "
        "chat model / Runnable instance."
    )
