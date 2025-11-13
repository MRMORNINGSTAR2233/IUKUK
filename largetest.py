"""
Speculative Decoding Benchmark

This script benchmarks standard vs. speculative decoding, optimized
for a high-VRAM NVIDIA GPU (A100 80GB).

This is the "winning" setup:
1.  Target Model: Gemma 2 27B-Instruct (8-bit) -> Creates MEMORY-BOUND system.
2.  Draft Model: Gemma 2 2B-Instruct (float16) -> Matched, bug-free draft.
3.  Attention: Uses Flash Attention 2 for maximum speed.
4.  Tokenizer: Uses the "single tokenizer" trick to bypass library bugs.
5.  std::bad_alloc Fix: Uses `low_cpu_mem_usage=True` for BOTH models.
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from datasets import load_dataset
from typing import List, Dict

# --- Configuration ---

# WARNING: This is a security risk. Do not share this code publicly.
# It is safer to use `huggingface-cli login` in your terminal.
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Step 2: Model Selection (The "Memory-Bound" Pair)
TARGET_MODEL_NAME = "google/gemma-2-27b-it"
DRAFT_MODEL_NAME = "google/gemma-2-2b-it"

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
            print("=" * 80)
            raise SystemExit
        
        self.device = "cuda"
        print(f"Using device: {self.device}")
        
        # 8-bit quantization config (FOR TARGET MODEL ONLY)
        # This is safer than 4-bit and avoids the `AttributeError`
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

        self.dtype = torch.float16

        # Set Attention Implementation
        try:
            import flash_attn
            attn_implementation = "flash_attention_2"
            print("Using 'flash_attention_2' implementation.")
        except ImportError:
            print("Flash Attention 2 not found. Falling back to 'sdpa'.")
            attn_implementation = "sdpa"


        # Load Draft Model (in float16, NO quantization)
        print(f"Loading draft model: {draft_model_name} (in float16)")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name,
            torch_dtype=self.dtype,
            attn_implementation=attn_implementation,
            device_map=self.device, # Automatically maps to CUDA
            trust_remote_code=True,
            token=HF_TOKEN,  # <-- Pass token
            low_cpu_mem_usage=True # <-- THE FIX for std::bad_alloc
        )

        # Load Target Model (with 8-bit quantization)
        print(f"Loading target model: {target_model_name} (in 8-bit)")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name,
            quantization_config=bnb_config, # <-- 8-bit
            attn_implementation=attn_implementation,
            device_map=self.device, # Automatically maps to CUDA
            trust_remote_code=True,
            token=HF_TOKEN,  # <-- Pass token
            low_cpu_mem_usage=True # <-- THE FIX for std::bad_alloc
        )
        
        self.target_model.generation_config.do_sample = True

        # --- FIX: Load ONLY ONE tokenizer ---
        # This forces `generate` to skip the buggy translator code path.
        print("Loading tokenizer (from target model)...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            target_model_name, 
            trust_remote_code=True,
            token=HF_TOKEN  # <-- Pass token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"\nAll models and tokenizer loaded on {self.device}")

    def tokenize_prompts(self, prompts: List[str]) -> List[torch.Tensor]:
        """Tokenize a list of prompts using the main tokenizer."""
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

        # --- Prepare kwargs for .generate() ---
        gen_kwargs = {
            "generation_config": gen_config,
            "assistant_model": assistant,
        }
        
        # --- Pass ONLY the main tokenizer ---
        if use_speculative:
            gen_kwargs["tokenizer"] = self.tokenizer

        # --- Warm-up run (single prompt) ---
        print("Running warm-up (single prompt)...")
        with torch.no_grad(): # Disable gradients for inference
            warmup_inputs = self.tokenize_prompts([prompts[0]])
            _ = self.target_model.generate(
                **warmup_inputs,
                **gen_kwargs # Pass the prepared kwargs
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
                    **gen_kwargs # Pass the same prepared kwargs
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
            split=DATASET_SPLIT,
            token=HF_TOKEN # <-- Pass token
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
        print("Did you run the install cell? pip install -U bitsandbytes accelerate transformers datasets flash-attn")
        print("Is your HF_TOKEN correct and do you have access to the Gemma models?")
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
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except:
        gpu_name = "NVIDIA GPU"
        
    print(f"Hardware Device: {decoder.device.upper()} ({gpu_name})")
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
        print(f"{'K (Draft Len)':<15} | {res['Time (Y)']:<12.2f} | {res['Tokens/Sec']:<12.2f} | {res['Speedup']:<8.2f}x")
        if res['Speedup'] > best_speedup:
            best_speedup = res['Speedup']
            best_k = res['K']

    print("-" * 51)
    if best_speedup > 1.0:
        print(f"Optimal K: {best_k} (Yielded {best_speedup:.2f}x speedup)")
    else:
        print("No speedup achieved. The overhead was greater than the gain.")

if __name__ == "__main__":
    run_benchmark_challenge()
