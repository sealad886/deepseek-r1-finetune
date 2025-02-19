"""Global configuration settings for the DeepSeek fine-tuning project."""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Project and Model Identity
    PROJECT_NAME: str = "dolphin-llama3.2-1B-finetune"
    MODEL_NAME: str = "cognitivecomputations/Dolphin3.0-Llama3.2-1B"  #"cognitivecomputations/Dolphin3.0-Llama3.2-3B"
    HF_TOKEN: str = os.environ.get("HF_TOKEN")

    # Basic Training Parameters
    OUTPUT_DIR: str = PROJECT_NAME
    NUM_EPOCHS: int = 1
    LEARNING_RATE: float = 5e-5
    WEIGHT_DECAY: float = 0.01    # L2 regularization factor
    WARMUP_RATIO: float = 0.03    # Percentage of steps for LR warmup
    SEED: int = 42

    # Batch Size and Gradient Settings
    TRAIN_BATCH_SIZE: int = 8     # Optimized for Apple Silicon M1/M2/M3
    EVAL_BATCH_SIZE: int = TRAIN_BATCH_SIZE
    BATCH_SIZE: int = 2           # Legacy setting, consider removing
    GRAD_ACCUM_STEPS: int = 6     # Increased gradient accumulation steps
    MAX_GRAD_NORM: float = 0.3    # Gradient clipping threshold
    MAX_STEPS: int = None

    # Logging Configuration
    LOGGING_STEPS: int = 2        # Reduced logging frequency
    LOGGING_STRATEGY: str = "steps"
    LOGGING_DIR: str = "./logs"
    REPORT_TO: str|None = None    # set to "wandb" to use Weights & Biases
    WANDB_PROJECT: str = PROJECT_NAME

    # Profiling Settings
    PROFILING_DIR: str = "./profiling"
    PROFILING_STEPS: int = 10

    # Model Saving and Evaluation Settings
    SAVE_STRATEGY: str = "steps"
    SAVE_EVERY: int = 5           # save_strategy="steps"
    SAVE_TOTAL_LIMIT: int = 1
    EVAL_STRATEGY: str = "steps"
    EVAL_STEPS: int = 5
    EVAL_ACCUMULATION_STEPS: int = 4

    # Performance Optimization Settings
    USE_FP16: bool = True         # Enable mixed precision training
    USE_BF16: bool = False        # BFloat16 precision (disabled for Apple Silicon)
    USE_GRADIENT_CHECKPOINTING: bool = True
    GRAD_CKPT: bool = True        # Legacy setting, consider removing
    GROUP_BY_LENGTH: bool = True  # Group similar length sequences
    OPTIMIZER: str = "adamw_torch"

    # LoRA Settings
    LORA_RANK: int = 4
    LORA_ALPHA: int = 8
    LORA_DROPOUT: float = 0.1
    LORA_BIAS: str = "none"
    LORA_TASK_TYPE: str = "CAUSAL_LM"
    LORA_TARGET_MODULES: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    USE_DORA: bool = True
    USE_RSLORA: bool = True
    INIT_LORA_WEIGHTS: str = "pissa_niter_16"

    # DataLoader Optimization
    NUM_WORKERS: int = os.cpu_count()  # Use all available CPU cores
    REMOVE_UNUSED_COLUMNS: bool = True

    # Apple Silicon Specific Settings
    APPLE_SILICON_CONFIG = {
        "compile_mode": "reduce-overhead",  # Optimized for M-series chips
        "prefetch_factor": 4,               # Increased prefetch for M-series
        "pin_memory": True,
        "persistent_workers": True,
        "use_mps_device": True,             # Explicitly enable MPS
        "mps_fallback_to_cpu": False,       # Prevent CPU fallback
        "mps_cache_limit": "24GB",          # Limit MPS cache
        "mps_allocator_strategy": "small",  # Optimized for M-series memory
        "torch_compile": False              # Disable torch.compile for MPS
    }
    MAX_MEMORY: str = "30GB"

    # Dataset Settings
    MAX_LENGTH: int = 8192*2*2
    TRUNCATION: bool = True
    PADDING: str|bool = True
    DEFAULT_DATASET: str = "cognitivecomputations/dolphin-r1"
    DEFAULT_NAME: str = "reasoning-flash"
    DATASET_SPLIT_TRAIN_TEST_RATIO: float = 0.80    # 80% train, 20% test
    DATASET_SPLIT_TEST_VAL_RATIO: float = 0.10      # 10% test, 90% validation
    PRETOKENIZE: bool = False
    PREPROCESS: bool = True
    CACHE_DIR: str = os.path.join(os.path.expanduser("~"), ".cache/huggingface")

    # Prompt Templates and System Settings
    system_prompt: str = """You are Dolphin, an AI assistant trained to be less filtered and restricted.

Your primary directive is to provide well-reasoned, structured, and extensively detailed responses.

Formatting Requirements:

* Always structure your replies using: <think>{reasoning}</think>{answer}
* The <think></think> block should contain at least six reasoning steps when applicable.
* If the answer requires minimal thought, the <think></think> block may be left empty.
* The user does not see the <think></think> section. Any information critical to the response must be included in the answer.
* If you notice that you have engaged in circular reasoning or repetition, immediately terminate {reasoning} with a </think> and proceed to the {answer}

Response Guidelines:

* Detailed and Structured: Use rich Markdown formatting for clarity and readability.
* Scientific and Logical Approach: Your explanations should reflect the depth and precision of the greatest scientific minds.
* Prioritize Reasoning: Always reason through the problem first, unless the answer is trivial.
* Concise yet Complete: Ensure responses are informative, yet to the point without unnecessary elaboration.
* Maintain a professional, intelligent, and analytical tone in all interactions.
* Respond in the same language(s) as used in the user query, unless otherwise instructed by the user."""

    st: str = "<|im_start|>"
    ed: str = "<|im_end|>"
    template: str = f"{st}system\n{ed}\n{st}user\n{ed}\n{st}assistant\n{ed}"
    template: str = template.replace(ed, "{}\n" + ed)


# Create a global instance
config = Config()
