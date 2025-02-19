import torch
import platform
import warnings
import os
import time
from contextlib import contextmanager
from datetime import datetime
import json
from tqdm import trange
from transformers import (
    TrainingArguments,
    logging,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from datasets import Dataset, DatasetDict
from transformers.tokenization_utils_fast import PreTokenizedInput
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from datasets import load_dataset, DatasetDict
from utils import get_model, get_tokenizer
from test_model import test_saved_model
from config import config as c
import preprocess


# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.set_verbosity_error()

def get_device() -> torch.device:
    if platform.processor() == 'arm' and torch.backends.mps.is_available():
        print("Using Apple Silicon MPS (Metal Performance Shaders)")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA GPU")
        device = torch.device("cuda")
    else:
        print("Using CPU")
        device = torch.device("cpu")
    return device

def setup_model() -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Setup model with memory-efficient configuration for Apple Silicon."""
    model_name = c.MODEL_NAME

    # Set tokenizer configuration
    tokenizer = get_tokenizer(model_name, 'right', return_special_tokens_mask=True)

    # Load model with optimized settings
    use_cache = not torch.backends.mps.is_available()
    model = get_model(
        model_name,
        device_map="mps" if torch.backends.mps.is_available() else "auto",
        use_cache=use_cache,
        max_memory=c.MAX_MEMORY
    )

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return model, tokenizer

def setup_trainer(model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast, train_dataset, eval_dataset, val_dataset=None):
    """Setup trainer with optimized settings for Apple Silicon."""

    # LoRA configuration for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        lora_alpha=c.LORA_ALPHA,
        lora_dropout=c.LORA_DROPOUT,
        r=c.LORA_RANK,
        bias=c.LORA_BIAS,
        task_type=c.LORA_TASK_TYPE,
        target_modules=c.LORA_TARGET_MODULES,
        use_dora=c.USE_DORA,
        use_rslora=c.USE_RSLORA,
        init_lora_weights=c.INIT_LORA_WEIGHTS,
    )

    # Training arguments configuration
    training_args = TrainingArguments(
        # Basic training parameters
        output_dir=c.OUTPUT_DIR,
        num_train_epochs=c.NUM_EPOCHS,
        learning_rate=c.LEARNING_RATE,
        weight_decay=c.WEIGHT_DECAY,
        warmup_ratio=c.WARMUP_RATIO,
        run_name=c.PROJECT_NAME,

        # Batch size and gradient settings
        per_device_train_batch_size=c.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=c.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=c.GRAD_ACCUM_STEPS,
        max_grad_norm=c.MAX_GRAD_NORM,
        max_steps=c.MAX_STEPS,

        # Logging configuration
        logging_steps=c.LOGGING_STEPS,
        logging_strategy=c.LOGGING_STRATEGY,
        logging_dir=c.LOGGING_DIR,
        report_to=c.REPORT_TO,

        # Model saving and evaluation settings
        save_strategy=c.SAVE_STRATEGY,
        save_steps=c.SAVE_EVERY,
        save_total_limit=c.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        evaluation_strategy=c.EVAL_STRATEGY,
        eval_steps=c.EVAL_STEPS,
        eval_accumulation_steps=c.EVAL_ACCUMULATION_STEPS,
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # Lower loss is better

        # Performance optimization settings
        fp16=c.USE_FP16,         # Mixed precision training
        bf16=c.USE_BF16,
        gradient_checkpointing=c.USE_GRADIENT_CHECKPOINTING,
        group_by_length=c.GROUP_BY_LENGTH,
        optim=c.OPTIMIZER,

        # DataLoader optimization
        dataloader_num_workers=c.NUM_WORKERS,
        remove_unused_columns=c.REMOVE_UNUSED_COLUMNS,

        # Apple Silicon specific optimizations
        dataloader_pin_memory=c.APPLE_SILICON_CONFIG["pin_memory"],
        dataloader_persistent_workers=c.APPLE_SILICON_CONFIG["persistent_workers"],
        dataloader_prefetch_factor=c.APPLE_SILICON_CONFIG["prefetch_factor"],
        dataloader_drop_last=False,
        torch_compile=not (mps:=torch.backends.mps.is_available()),  # Disable for MPS
        use_mps_device=mps,
        auto_find_batch_size=not mps,

        # Distributed training settings (disabled)
        deepspeed=None,
        local_rank=-1,
        ddp_find_unused_parameters=None,

        # UI settings
        disable_tqdm=False,
    )

    # Configure data collation for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt"
    )

    tokenizer

    # Initialize the SFT trainer with all configurations
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        args=training_args,
        data_collator=data_collator,
        processing_class=tokenizer, #(
        #     text=train_dataset["text"],
        #     truncation=c.TRUNCATION,
        #     padding=c.PADDING,
        #     max_length=c.MAX_LENGTH,
        #     return_tensors="pt",
        #     return_special_tokens_mask=True,
        #     return_length=True,
        #     return_attention_mask=True,
        # ),
    )

    # Add validation dataset for additional evaluation
    # trainer.val_dataset = val_dataset
    return trainer

def prepare_dataset(tokenizer: PreTrainedTokenizerFast, *, pretok: bool = c.PRETOKENIZE):
    if c.PREPROCESS:
        print("Preprocessing dataset...")
        dataset = preprocess.preprocess_dataset()
    else:
        dataset = load_dataset(c.DEFAULT_DATASET, c.DEFAULT_NAME, trust_remote_code=True)
    print(f"Dataset loaded with {len(dataset['train'])} training examples")

    def format_instruction(msg_dict: dict) -> str:
        return c.template.format(*[msg['content'] for msg in msg_dict])

    dataset = dataset.map(
        lambda x: {"text": (format_instruction(x["messages"]))},
        remove_columns=(remove_columns := ['reasoning', 'answer']),
        num_proc=c.NUM_WORKERS,
        desc="Format instruction",
    )

    # train_test = train_test.rename_column("messages", "labels").class_encode_column("labels")

    if 'test' not in dataset.keys():
        train_test = dataset['train'].train_test_split(
            test_size=(1 - c.DATASET_SPLIT_TRAIN_TEST_RATIO),
            train_size=c.DATASET_SPLIT_TRAIN_TEST_RATIO,
            seed=c.SEED,
            stratify_by_column="labels",
        )
    if not 'validate' in dataset.keys() and "test" in train_test.keys():
        test_validate_split: DatasetDict = train_test['test'].train_test_split(
            train_size=c.DATASET_SPLIT_TEST_VAL_RATIO,
            test_size=(1 - c.DATASET_SPLIT_TEST_VAL_RATIO),
            seed=c.SEED
        )

        train_test['test'] = test_validate_split['train']
        train_test['validate'] = test_validate_split['test']

    if 'test' not in train_test: train_test['test'] = Dataset()
    if 'validate' not in train_test: train_test['validate'] = Dataset()

    train_test = train_test.map(
        lambda x: tokenizer(
            x['text'],
            truncation=c.TRUNCATION,
            padding=c.PADDING,
            max_length=c.MAX_LENGTH,
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_length=True,
            return_attention_mask=True,
        ),
        num_proc=c.NUM_WORKERS,
        desc="Tokenizing split(s)",
        # remove_columns=["text"] if pretok else None,
        batched=bool(c.TRAIN_BATCH_SIZE),
    )

    # Ensure the necessary columns are present
    if 'input_ids' not in train_test['train'].column_names:
        raise ValueError("The 'input_ids' column is missing from the dataset.")
    if 'attention_mask' not in train_test['train'].column_names:
        raise ValueError("The 'attention_mask' column is missing from the dataset.")
    if 'labels' not in train_test['train'].column_names:
        raise ValueError("The 'labels' column is missing from the dataset.")

    print(f"\nUsing {len(train_test['train'])} examples for training, {len(train_test['test'])} for evaluation, and {len(train_test['validate'])} reserved for validation.")
    print("\nSample formatted data:")
    print(format_instruction(dataset['train'][0]['messages']))

    return (train_test['train'], train_test['test'], train_test['validate'])

def test_model(model_path):
    test_saved_model(model_path)

def main():
    """Main training function with progress updates."""
    try:
        # Model and tokenizer loading
        print("\n[1/5] Loading model and tokenizer...")
        model, tokenizer = setup_model()
        print(f"✓ Model '{c.MODEL_NAME}' and tokenizer loaded successfully")

        # Dataset preparation
        print("\n[2/5] Preparing dataset...")
        train_dataset, eval_dataset, val_dataset = prepare_dataset(tokenizer)
        print("✓ Dataset prepared and tokenized")

        # Trainer setup
        print("\n[3/5] Setting up trainer...")
        trainer = setup_trainer(model, tokenizer, train_dataset, eval_dataset, val_dataset)
        print("✓ Trainer configured and ready")

        # Training
        print("\n[4/5] Starting training...")
        print(f"Training for {c.NUM_EPOCHS} epochs with batch size {c.TRAIN_BATCH_SIZE}")
        trainer.train()
        print("✓ Training completed successfully")

        # Model saving
        print("\n[5/5] Saving model...")
        outpath = os.path.join(c.OUTPUT_DIR, "final_model")
        trainer.model.save_pretrained(outpath)
        tokenizer.save_pretrained(outpath)
        print(f"✓ Model saved to {outpath}")

        # Model testing
        print("\nTesting saved model...")
        test_saved_model(outpath)
        print("✓ Model testing completed")

    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user. Performing cleanup...")
        # Attempt to save current model state
        try:
            outpath = os.path.join(c.OUTPUT_DIR, "interrupted_model")
            print(f"Saving current model state to {outpath}...")
            trainer.model.save_pretrained(outpath)
            tokenizer.save_pretrained(outpath)
            print("✓ Model state saved successfully")
        except Exception as e:
            print(f"⚠️ Could not save model state: {str(e)}")
        print("Cleanup complete. Exiting gracefully.")
        return

    except Exception as e:
        print(f"\nError during pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
