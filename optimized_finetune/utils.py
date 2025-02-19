from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Pipeline,
    pipeline,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    LlamaPreTrainedModel,
    LlamaForCausalLM,
    PretrainedConfig,
)
from typing import Any, Literal
import torch
from config import config

def get_tokenizer(model_name: str, padding_side: Literal['left']|Literal['right'] = "right", *args, **kwargs) -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
        *args, **kwargs,
        )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side

    return tokenizer

def get_model(model_name: str, device_map: str = "auto", use_cache: bool = False, max_memory: str|dict = config.MAX_MEMORY, *args, **kwargs) -> PreTrainedModel:
    if isinstance(max_memory, str):
        max_memory = "".join(max_memory.strip().split(" ")).upper()
        max_memory = {0: max_memory}

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # Use fp16 for mixed precision
        use_cache=use_cache,
        max_memory=max_memory.update({"cpu": "34GB"}),
        *args, **kwargs
    )

    return model


def get_pipeline(model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=8192,
        temperature=0.3,
        top_p=0.95,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id
    )
    return pipe
