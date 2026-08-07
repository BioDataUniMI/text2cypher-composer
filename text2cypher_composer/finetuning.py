"""Fine-tuning local (Hugging Face) models on your own question/Cypher dataset.

Ported from the miRNAKG LoRA fine-tuning notebook
(`bio2C/finetuning_LLaMa3-8B/Finetuning_llama3-8b.ipynb`): a small LoRA
adapter is trained on top of a frozen, 4-bit-quantized base causal LM, then
loaded back as a LangChain `Runnable` via `load_finetuned_model` — ready to
pass directly as `run()`'s `model` argument, the same "bring your own
Runnable" path `resolve_model` already supports for any non-OpenAI chat
model.

`config.base_model`/`base_model` is not restricted to `DEFAULT_BASE_MODEL` —
any Hugging Face causal-LM repo id works, since it's passed as-is to
`AutoModelForCausalLM.from_pretrained`/`AutoTokenizer.from_pretrained`. In
particular, `DEFAULT_TARGET_MODULES` (the LoRA adapter's attention/MLP
projection names) applies unchanged to any Llama-style architecture — which
covers Llama, Mistral, and Qwen2 checkpoints alike, e.g.
`"mistralai/Mistral-7B-Instruct-v0.3"` or `"Qwen/Qwen2.5-7B-Instruct"` — not
just Llama itself.

`finetune_lora`/`load_finetuned_model` need the optional fine-tuning
dependencies (`torch`, `transformers`, `peft`, `datasets`,
`langchain_huggingface`) — install with
`pip install "text2cypher-composer[finetune]"`. They're imported lazily
(only inside these two functions), so the rest of the library works without
them.

`DEFAULT_BASE_MODEL` (`meta-llama/Llama-3.1-8B`) is a gated Hugging Face
repo: you must accept its license on the model page and authenticate
locally (`huggingface-cli login`, or an `HF_TOKEN` env var) before calling
`finetune_lora`/`load_finetuned_model` with it — otherwise `from_pretrained`
raises a 401/403. This gating is specific to that particular repo, not to
fine-tuning in general — most Mistral/Qwen checkpoints are not gated, though
that varies by repo, so check the model page for whichever one you pick.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union

import pandas as pd

DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class LoRATrainingConfig:
    """LoRA fine-tuning hyperparameters — defaults match the miRNAKG notebook's settings."""

    base_model: str = DEFAULT_BASE_MODEL
    output_dir: str = "./llama3_lora_adapter"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
    load_in_4bit: bool = True
    max_length: int = 512
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    logging_dir: str = "./logs"
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 2


@dataclass
class LoRAFinetuneResult:
    base_model: str
    adapter_path: str
    n_examples: int
    train_output: Any


def build_prompt_completion_pairs(
    df: pd.DataFrame, question_col: str = "question", cypher_col: str = "cypher"
) -> List[dict]:
    """`{"prompt", "completion"}` pairs — the format `finetune_lora` trains on.

    Mirrors the notebook's framing: the prompt ends with "\\nCypher query:"
    and the completion is the Cypher query with a leading space, so the model
    learns to continue the prompt rather than restate it. The training loss
    is masked over the prompt tokens (see `finetune_lora`), so only this
    completion contributes to the gradient.
    """
    if question_col not in df.columns or cypher_col not in df.columns:
        raise ValueError(f"`df` must have '{question_col}' and '{cypher_col}' columns.")
    return [
        {"prompt": f"{row[question_col]}\nCypher query:", "completion": " " + str(row[cypher_col])}
        for _, row in df.iterrows()
    ]


def finetune_lora(
    df: pd.DataFrame,
    config: Optional[LoRATrainingConfig] = None,
    question_col: str = "question",
    cypher_col: str = "cypher",
) -> LoRAFinetuneResult:
    """LoRA-finetune `config.base_model` on `df`'s question/Cypher pairs, saving the adapter to disk.

    Ported from the miRNAKG LoRA fine-tuning notebook: a 4-bit-quantized base
    model, LoRA adapters on the attention/MLP projections, and a masked
    causal-LM loss (prompt tokens excluded, only the Cypher completion is
    trained on). The resulting adapter is written to `config.output_dir` —
    load it back with `load_finetuned_model`.
    """
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            default_data_collator,
        )
    except ImportError as e:
        raise ImportError(
            "finetune_lora requires the optional fine-tuning dependencies "
            "(torch, transformers, peft, datasets). Install them with "
            '`pip install "text2cypher-composer[finetune]"`.'
        ) from e

    config = config or LoRATrainingConfig()

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(load_in_4bit=True) if config.load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        token=True,
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    pairs = build_prompt_completion_pairs(df, question_col=question_col, cypher_col=cypher_col)
    dataset = Dataset.from_list(pairs)

    def preprocess(example):
        text = example["prompt"] + example["completion"]
        encoding = tokenizer(text, max_length=config.max_length, truncation=True, padding="max_length")
        labels = [tok if tok != tokenizer.pad_token_id else -100 for tok in encoding["input_ids"]]
        prompt_len = len(tokenizer(example["prompt"])["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len
        encoding["labels"] = labels
        return encoding

    tokenized_dataset = dataset.map(preprocess, remove_columns=["prompt", "completion"])

    training_args = TrainingArguments(
        output_dir=f"{config.output_dir}-checkpoints",
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        fp16=False,
        bf16=True,
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
    )
    train_output = trainer.train()

    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    return LoRAFinetuneResult(
        base_model=config.base_model,
        adapter_path=config.output_dir,
        n_examples=len(pairs),
        train_output=train_output,
    )


def load_finetuned_model(
    base_model: str,
    adapter_path: Union[str, Path],
    load_in_4bit: bool = True,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    top_p: float = 0.95,
    do_sample: bool = False,
):
    """Load a LoRA-finetuned local model as a LangChain `Runnable`, ready for `run(model=...)`.

    Requires the same optional dependencies as `finetune_lora`, plus
    `langchain_huggingface`. Returns a `HuggingFacePipeline` pre-configured
    with its own generation parameters — pass it straight through as `run()`'s
    `model` argument (used as-is, since it's already a `Runnable`), typically
    with `technique="vanilla"` to match the notebook's inference-time prompt.
    """
    try:
        import torch
        from langchain_huggingface import HuggingFacePipeline
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
    except ImportError as e:
        raise ImportError(
            "load_finetuned_model requires the optional fine-tuning dependencies "
            "(torch, transformers, peft, langchain_huggingface). Install them with "
            '`pip install "text2cypher-composer[finetune]"`.'
        ) from e

    quantization_config = BitsAndBytesConfig(load_in_4bit=True) if load_in_4bit else None
    base = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", quantization_config=quantization_config, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = PeftModel.from_pretrained(base, adapter_path)

    hf_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )
    return HuggingFacePipeline(pipeline=hf_pipe)
