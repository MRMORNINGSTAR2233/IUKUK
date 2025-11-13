"""
Speculative Decoding Benchmark

This script benchmarks standard vs. speculative decoding, optimized
to run on an NVIDIA GPU (like RTX 3050) using CUDA, 4-bit quantization,
and Flash Attention 2.
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from datasets import load_dataset
from typing import List, Dict

# --- Configuration ---

# Step 2: Model Selection (NVIDIA RTX 3050 4GB/6GB VRAM Friendly)
TARGET_MODEL_NAME = "Qwen/Qwen2-1.5B"
DRAFT_MODEL_NAME = "Qwen/Qwen2-0.5B"

# Step 3: Dataset Selection
DATASET_NAME = "ai4bharat/IN22-Conv"
DATASET_CONFIG = "default"      # Use the 'default' config
DATASET_SPLIT = "test"          # Use the 'test' split
DATASET_PROMPT_KEY = "eng_Latn" # Use the 'eng_Latn' field as the prompt
NUM_SAMPLES_TO_BENCHMARK = 50   # Limit samples for a reasonable benchmark time
MAX_NEW_TOKENS = 50             # Max tokens to generate per prompt

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# --- High-Performance SpeculativeDecoder (with KV Cache) ---

class SpeculativeDecoderKV:
    """
    Implements speculative decoding using the Hugging Face `generate` method,
    which correctly handles KV caching.
    
    This class is configured to run on an NVIDIA GPU with CUDA.
    """

    def __init__(self, draft_model_name: str, target_model_name: str):
        """
        Initialize the speculative decoder, loading models for CUDA.
        """
        print("=" * 80)
        print("Loading Models for NVIDIA GPU (CUDA)...")
        print("=" * 80)
        
        if not torch.cuda.is_available():
            print("=" * 80)
            print("ERROR: CUDA is not available. This script is configured for an NVIDIA GPU.")
            print("Please check your PyTorch installation and NVIDIA driver.")
            print("=" * 80)
            raise SystemExit
        
        self.device = "cuda"
        print(f"Using device: {self.device}")
        
        # 4-bit quantization config to save VRAM on the 3050
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Use float16 for memory efficiency
        self.dtype = torch.float16

        # Try to use Flash Attention 2 for speed
        try:
            attn_implementation = "flash_attention_2"
            print("Attempting to use Flash Attention 2...")
        except ImportError:
            print("Flash Attention 2 not found. Falling back to 'sdpa'.")
            attn_implementation = "sdpa"

        # Load Draft Model
        print(f"Loading draft model: {draft_model_name}")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name,
            torch_dtype=self.dtype,
            quantization_config=bnb_config,
            attn_implementation=attn_implementation,
            device_map=self.device, # Automatically maps to CUDA
            trust_remote_code=True
        )

        # Load Target Model
        print(f"Loading target model: {target_model_name}")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name,
            torch_dtype=self.dtype,
            quantization_config=bnb_config,
            attn_implementation=attn_implementation,
            device_map=self.device, # Automatically maps to CUDA
            trust_remote_code=True
        )
        
        self.target_model.generation_config.do_sample = True

        # Load Tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"\nAll models loaded on {self.device}")

    def tokenize_prompts(self, prompts: List[str]) -> List[torch.Tensor]:
        """Tokenize a list of prompts and move them to the device."""
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        return inputs.to(self.device)

    def benchmark_prompts(
        self,
        prompts: List[str],
        max_new_tokens: int,
        use_speculative: bool = False,
        draft_length: int = 5
    ) -> Dict[str, float]:
        """
        Run a benchmark on a list of prompts, either baseline or speculative.
        
        This function loops through prompts one-by-one to respect the
        batch_size=1 limitation of assisted generation.
        """
        
        total_tokens_generated = 0
        
        # Create GenerationConfig
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True, # Required for speculative decoding
        )

        # Set assistant model for speculative decoding
        assistant = self.draft_model if use_speculative else None
        if use_speculative:
            gen_config.max_draft_length = draft_length

        # --- Warm-up run (single prompt) ---
        print("Running warm-up (single prompt)...")
        with torch.no_grad(): # Disable gradients for inference
            warmup_inputs = self.tokenize_prompts([prompts[0]])
            _ = self.target_model.generate(
                **warmup_inputs,
                generation_config=gen_config,
                assistant_model=assistant
            )
        
        # --- Timed run (looping one-by-one) ---
        print(f"Running benchmark (looping {len(prompts)} prompts)...")
        
        start_time = time.time() # Time the whole loop
        
        with torch.no_grad(): # Disable gradients for inference
            for prompt in prompts:
                # Tokenize one prompt (batch_size=1)
                inputs = self.tokenize_prompts([prompt])
                
                # Generate for one prompt
                outputs = self.target_model.generate(
                    **inputs,
                    generation_config=gen_config,
                    assistant_model=assistant
                )
                
                # Calculate tokens generated for this prompt
                input_length = inputs.input_ids.shape[1]
                total_tokens_generated += (len(outputs[0]) - input_length)

        
        torch.cuda.synchronize() # Ensure all CUDA operations are complete

        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Handle division by zero if total_time is too small
        tokens_per_sec = total_tokens_generated / total_time if total_time > 0 else 0
        
        return {
            "total_time": total_time,
            "tokens_per_sec": tokens_per_sec,
            "total_tokens": total_tokens_generated,
        }


def run_benchmark_challenge():
    """
    Orchestrates the full benchmark challenge (Steps 2-5).
    """
    
    # --- Load Models and Dataset ---
    print("=" * 80)
    print("Loading Models and Dataset...")
    print("=" * 80)
    
    decoder = None
    prompts = []
    
    try:
        decoder = SpeculativeDecoderKV(DRAFT_MODEL_NAME, TARGET_MODEL_NAME)
        
        print(f"\nLoading dataset: {DATASET_NAME} ({DATASET_CONFIG} config)...")
        dataset = load_dataset(
            DATASET_NAME,
            name=DATASET_CONFIG,
            split=DATASET_SPLIT
        )
        # Filter out potential None entries
        prompts = [p for p in dataset[DATASET_PROMPT_KEY] if p is not None]
        prompts = prompts[:NUM_SAMPLES_TO_BENCHMARK]
        
        if not prompts:
            print(f"Error: No prompts found. Column '{DATASET_PROMPT_KEY}' might be empty or all None.")
            return

        print(f"\nLoaded {len(prompts)} prompts from {DATASET_NAME} ({DATASET_CONFIG} config).")
    
    except Exception as e:
        print(f"\nError during loading: {e}")
        print("Please ensure you have an internet connection and dependencies installed.")
        return

    # --- Step 3: Benchmark Baseline (X) ---
    print("\n" + "=" * 80)
    print(f"Step 3: Benchmarking Baseline (Target Model: {TARGET_MODEL_NAME})")
    print(f"Generating {MAX_NEW_TOKENS} tokens for {len(prompts)} prompts...")
    print("=" * 80)
    
    baseline_results = decoder.benchmark_prompts(
        prompts,
        max_new_tokens=MAX_NEW_TOKENS,
        use_speculative=False
    )
    
    baseline_time = baseline_results["total_time"]
    baseline_tps = baseline_results["tokens_per_sec"]
    
    print("\n--- Baseline Results (X) ---")
    print(f"Total Time (X): {baseline_time:.2f} seconds")
    print(f"Tokens/Sec: {baseline_tps:.2f}")

    # --- Step 4 & 5: Benchmark Speculative (Y) vs. K ---
    print("\n" * 2 + "=" * 80)
    print(f"Steps 4 & 5: Benchmarking Speculative Decoding vs. K (Draft Length)")
    print(f"Target: {TARGET_MODEL_NAME} | Draft: {DRAFT_MODEL_NAME}")
    print("=" * 80)
    
    k_values = [3, 4, 5, 6, 8, 10] # Range of 'K' values to test
    results = []

    for k in k_values:
        print(f"\n--- Testing K = {k} ---")
        
        spec_results = decoder.benchmark_prompts(
            prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            use_speculative=True,
            draft_length=k
        )
        
        spec_time = spec_results["total_time"]
        spec_tps = spec_results["tokens_per_sec"]
        
        # Avoid division by zero if time is near-zero
        speedup = (baseline_time / spec_time) if spec_time > 0.001 else float('inf')
        
        print(f"Total Time (Y): {spec_time:.2f} s")
        print(f"Tokens/Sec: {spec_tps:.2f}")
        print(f"Speedup vs Baseline: {speedup:.2f}x")
        
        results.append({
            "K": k,
            "Time (Y)": spec_time,
            "Tokens/Sec": spec_tps,
            "Speedup": speedup
        })

    # --- Final Summary ---
    print("\n" * 2 + "=" * 80)
    print("Benchmark Challenge Summary")
    print("=" * 80)
    print(f"Hardware Device: {decoder.device.upper()} ({torch.cuda.get_device_name(0)})")
    print(f"Target Model: {TARGET_MODEL_NAME}")
    print(f"Draft Model: {DRAFT_MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME} (n={len(prompts)})")
    print(f"Tokens per prompt: {MAX_NEW_TOKENS}")
    print("\n--- Baseline (X) ---")
    print(f"Time: {baseline_time:.2f} s | Tokens/Sec: {baseline_tps:.2f}")
    
    print("\n--- Speculative (Y) vs. K ---")
    print(f"{'K (Draft Len)':<15} | {'Total Time':<12} | {'Tokens/Sec':<12} | {'Speedup':<8}")
    print("-" * 51)
    
    best_k = -1
    best_speedup = 0
    
    for res in results:
        print(f"{res['K']:<15} | {res['Time (Y)']:<12.2f} | {res['Tokens/Sec']:<12.2f} | {res['Speedup']:<8.2f}x")
        if res['Speedup'] > best_speedup:
            best_speedup = res['Speedup']
            best_k = res['K']

    print("-" * 51)
    print(f"Optimal K: {best_k} (Yielded {best_speedup:.2f}x speedup)")

if __name__ == "__main__":
    run_benchmark_challenge()


# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# # Install these libraries
# pip install transformers datasets accelerate bitsandbytes
# pip install flash-attn --no-build-isolation