from typing import Union

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

ModelLike = Union[str, Runnable]

DEFAULT_MAX_TOKENS = 1024
STOP_SEQUENCES = ["\nCypherResult:"]


def llm_backend_for(model: str) -> str:
    """"openai", "anthropic", "google", or "deepseek", based on a deterministic rule on `model`.

    Each provider's model ids conventionally start with a distinct,
    non-overlapping prefix: Anthropic's with "claude-" (e.g.
    "claude-sonnet-5"), Google's with "gemini-" (e.g. "gemini-2.5-flash"),
    DeepSeek's with "deepseek-" (e.g. "deepseek-chat"); OpenAI's never do
    (e.g. "gpt-4o", "o1", a fine-tuned "ft:..." id) — so that prefix is what
    selects the backend, the same "bring your own object" pattern
    `embedding_backend_for` uses for `embedding_model`.
    """
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gemini"):
        return "google"
    if model.startswith("deepseek"):
        return "deepseek"
    return "openai"


def _build_chat_model(model: str, *, bind_generation_params: bool) -> Runnable:
    backend = llm_backend_for(model)
    if backend == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ImportError(
                f"model='{model}' is an Anthropic model, but the optional anthropic "
                'dependency isn\'t installed. Install it with `pip install '
                '"text2cypher-composer[anthropic]"`.'
            ) from e
        llm = ChatAnthropic(model=model, temperature=0)
    elif backend == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError(
                f"model='{model}' is a Google Gemini model, but the optional google "
                'dependency isn\'t installed. Install it with `pip install '
                '"text2cypher-composer[google]"`.'
            ) from e
        llm = ChatGoogleGenerativeAI(model=model, temperature=0)
    elif backend == "deepseek":
        try:
            from langchain_deepseek import ChatDeepSeek
        except ImportError as e:
            raise ImportError(
                f"model='{model}' is a DeepSeek model, but the optional deepseek "
                'dependency isn\'t installed. Install it with `pip install '
                '"text2cypher-composer[deepseek]"`.'
            ) from e
        llm = ChatDeepSeek(model=model, temperature=0)
    else:
        llm = ChatOpenAI(model_name=model, temperature=0)

    if bind_generation_params:
        return llm.bind(stop=STOP_SEQUENCES, max_tokens=DEFAULT_MAX_TOKENS)
    return llm


def resolve_model(model: ModelLike) -> Runnable:
    """Resolve `model` into an invocable LangChain Runnable.

    A string is treated as an OpenAI (e.g. "gpt-4o", "gpt-4o-mini", a
    fine-tuned "ft:..." id), Anthropic ("claude-" prefixed, e.g.
    "claude-sonnet-5"), Google ("gemini-" prefixed, e.g. "gemini-2.5-flash"),
    or DeepSeek ("deepseek-" prefixed, e.g. "deepseek-chat") chat model id —
    see `llm_backend_for` — and wrapped in the matching LangChain chat model.
    Anything else must already be a LangChain-compatible chat model /
    Runnable (e.g. a HuggingFacePipeline wrapping a local checkpoint, base or
    fine-tuned via PEFT) and is used as-is, generation parameters included.
    """
    if isinstance(model, str):
        return _build_chat_model(model, bind_generation_params=True)
    if isinstance(model, Runnable):
        return model
    raise TypeError(
        "`model` must be an OpenAI/Anthropic/Google/DeepSeek model id (str) or a "
        "LangChain-compatible chat model / Runnable instance."
    )


def resolve_pruning_model(model: ModelLike) -> Runnable:
    """Resolve `model` into a chat model suitable for `.with_structured_output(...)`.

    Unlike `resolve_model`, a string is wrapped in a plain (unbound) chat
    model — `resolve_model`'s `.bind(...)` wrapper is a RunnableBinding that
    may not reliably proxy `.with_structured_output()` across LangChain
    versions. Anything else is used as-is and must itself support
    `.with_structured_output()`.
    """
    if isinstance(model, str):
        return _build_chat_model(model, bind_generation_params=False)
    if isinstance(model, Runnable):
        return model
    raise TypeError(
        "`model` must be an OpenAI/Anthropic/Google/DeepSeek model id (str) or a "
        "LangChain-compatible chat model / Runnable instance."
    )
