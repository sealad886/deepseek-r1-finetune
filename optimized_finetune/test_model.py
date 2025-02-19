import os
import torch
import platform
import warnings
from transformers import logging
from utils import get_model, get_tokenizer, get_pipeline
from config import config as c

warnings.filterwarnings("ignore", category=UserWarning)
logging.set_verbosity_error()

def test_saved_model(model_path="./fine_tuned_model"):
    print("\nLoading model from:", model_path)
    os.makedirs("offload", exist_ok=True)

    try:
        kwargs = {
            "offload_folder": "offload",
            "offload_state_dict": True
        }

        # Apple Silicon specific optimizations
        if platform.processor() == 'arm':
            kwargs.update({
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.float16,
                "device_map": "mps",
                "max_memory": {"mps": c.APPLE_SILICON_CONFIG["mps_cache_limit"], "cpu": "10GB"}
            })

        model = get_model(model_path, **kwargs)
        tokenizer = get_tokenizer(model_path)

        # Set MPS-specific configurations
        if platform.processor() == 'arm':
            torch.mps.empty_cache()

        pipe = get_pipeline(model, tokenizer)

        test_cases = [
            """Please reason step by step:

A 45-year-old patient presents with sudden onset chest pain, shortness of breath, and anxiety. The pain is described as sharp and worsens with deep breathing. What is the most likely diagnosis and what immediate tests should be ordered?""",

            """Please reason step by step:

A 67-year-old diabetic patient presents with sudden onset of severe headache, confusion, and right-sided weakness. What is the most likely diagnosis and what immediate imaging should be performed?"""
        ]

        print("\nRunning test cases...")
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}:")
            print("Input:", test_case)
            print("\nGenerating response...")

            result = pipe(
                test_case,
                max_new_tokens=512,
                temperature=0.6,
                top_p=0.95,
                repetition_penalty=1.15
            )

            print("\nModel Response:", result[0]["generated_text"])
            print("\n" + "="*80)

    except Exception as e:
        print(f"\nError during testing: {str(e)}")
    finally:
        if os.path.exists("offload"):
            import shutil
            shutil.rmtree("offload")

if __name__ == "__main__":
    if not os.path.exists("./fine_tuned_model"):
        print("Error: Could not find fine-tuned model at './fine_tuned_model'")
        print("Please ensure the model was saved correctly during training.")
    else:
        test_saved_model()
