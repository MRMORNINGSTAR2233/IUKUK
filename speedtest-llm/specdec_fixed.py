"""
Speculative Decoding

This script demonstrates how speculative decoding works by using a small model
to draft tokens and a larger model to verify them efficiently.

This version is modified to pull its prompt from the ai4bharat/IN22-Conv dataset.
FIXED: Added proper validation for probability tensors to prevent CUDA assertion errors.
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Tuple
from datasets import load_dataset

class SpeculativeDecoder:
    """
    Implements speculative decoding for faster text generation.
    
    The key insight: Use a small fast model to guess multiple tokens,
    then verify them all at once with a larger model.
    """

    def __init__(self, draft_model_name: str, target_model_name: str):
        """
        Initialize the speculative decoder with two models.

        Args:
            draft_model_name: Name of the small, fast model (e.g., "gpt2")
            target_model_name: Name of the large, accurate model (e.g., "gpt2-medium")
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Configure 4-bit quantization for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        print(f"Loading draft model: {draft_model_name} (4-bit quantized)")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.draft_model.eval()

        print(f"Loading target model: {target_model_name} (4-bit quantized)")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.target_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Models loaded with 4-bit quantization on GPU\n")

    def _safe_softmax(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute softmax with numerical stability and validation.
        
        Args:
            logits: Input logits tensor
            dim: Dimension to apply softmax
            
        Returns:
            Valid probability distribution
        """
        # Clamp logits to prevent overflow/underflow
        logits = torch.clamp(logits, min=-100, max=100)
        
        # Compute softmax
        probs = torch.softmax(logits, dim=dim)
        
        # Replace any inf/nan values with uniform distribution
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print("Warning: Invalid values detected in probabilities, using uniform distribution")
            probs = torch.ones_like(probs) / probs.shape[dim]
        
        # Ensure all probabilities are non-negative
        probs = torch.clamp(probs, min=1e-10)
        
        # Renormalize to ensure sum = 1
        probs = probs / probs.sum(dim=dim, keepdim=True)
        
        return probs

    def _safe_multinomial(self, probs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Safe multinomial sampling with validation.
        
        Args:
            probs: Probability distribution
            num_samples: Number of samples to draw
            
        Returns:
            Sampled indices
        """
        # Validate probabilities
        probs = torch.clamp(probs, min=1e-10)
        
        # Check for inf/nan
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            # Use uniform distribution as fallback
            probs = torch.ones_like(probs) / probs.numel()
        
        # Renormalize
        probs = probs / probs.sum()
        
        # Sample
        return torch.multinomial(probs, num_samples=num_samples)

    def generate_draft_tokens(self, input_ids: torch.Tensor, num_tokens: int) -> Tuple[List[int], List[float]]:
        """
        Use the draft model to generate candidate tokens.

        Args:
            input_ids: Current token sequence
            num_tokens: Number of tokens to draft

        Returns:
            Tuple of (draft_tokens, draft_probabilities)
        """
        draft_tokens = []
        draft_probs = []

        current_ids = input_ids.clone()

        for _ in range(num_tokens):
            with torch.no_grad():
                outputs = self.draft_model(current_ids)
                logits = outputs.logits[0, -1, :]  # Last position
                probs = self._safe_softmax(logits, dim=0)

                # Sample next token
                next_token = self._safe_multinomial(probs, num_samples=1)
                token_id = next_token.item()

                draft_tokens.append(token_id)
                draft_probs.append(probs[token_id].item())

                # Append token for next iteration
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)

        return draft_tokens, draft_probs

    def verify_draft_tokens(self, input_ids: torch.Tensor, 
                           draft_tokens: List[int], 
                           draft_probs: List[float]) -> List[int]:
        """
        Verify draft tokens using the target model in a single forward pass.

        This is where the magic happens! We process all draft tokens at once
        and get probability distributions at each position.

        Args:
            input_ids: Original token sequence
            draft_tokens: Tokens proposed by draft model
            draft_probs: Probabilities assigned by draft model

        Returns:
            List of accepted tokens
        """
        # Create sequence with all draft tokens
        draft_sequence = torch.cat([
            input_ids,
            torch.tensor([draft_tokens], device=input_ids.device)
        ], dim=1)

        # Single forward pass through target model
        with torch.no_grad():
            outputs = self.target_model(draft_sequence)
            all_logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]

        # Verify each draft token
        accepted_tokens = []
        seq_len = input_ids.size(1)

        for i in range(len(draft_tokens)):
            # Get target model's probability distribution at this position
            position = seq_len - 1 + i
            target_probs = self._safe_softmax(all_logits[position], dim=0)
            target_prob = target_probs[draft_tokens[i]].item()
            draft_prob = draft_probs[i]

            # Avoid division by zero
            if draft_prob < 1e-10:
                draft_prob = 1e-10

            # Acceptance criterion: p_target(token) / p_draft(token)
            acceptance_ratio = min(1.0, target_prob / draft_prob)

            if torch.rand(1).item() < acceptance_ratio:
                # Accept the draft token
                accepted_tokens.append(draft_tokens[i])
            else:
                # Reject and sample from adjusted distribution
                # Adjusted distribution: max(0, p_target - p_draft)
                # NOTE: There is a known bug in this blog post's logic here:
                # `torch.softmax(all_logits[position], dim=0)` is just `target_probs`.
                # This subtraction will always be zero, so it falls back to the else clause.
                # Per the user's request, this logic is kept as-is.
                adjusted_probs = torch.clamp(
                    target_probs - self._safe_softmax(all_logits[position], dim=0), 
                    min=0.0
                )

                if adjusted_probs.sum() > 1e-10:
                    adjusted_probs = adjusted_probs / adjusted_probs.sum()
                    new_token = self._safe_multinomial(adjusted_probs, num_samples=1).item()
                else:
                    # Fallback: sample from target distribution
                    new_token = self._safe_multinomial(target_probs, num_samples=1).item()

                accepted_tokens.append(new_token)
                # Stop verifying remaining tokens
                break

        # Bonus token: if all drafts accepted, get one more from target model
        if len(accepted_tokens) == len(draft_tokens):
            position = seq_len - 1 + len(draft_tokens)
            bonus_probs = self._safe_softmax(all_logits[position], dim=0)
            bonus_token = self._safe_multinomial(bonus_probs, num_samples=1).item()
            accepted_tokens.append(bonus_token)

        return accepted_tokens

    def generate(self, prompt: str, max_new_tokens: int = 50, 
                 num_draft_tokens: int = 4, verbose: bool = True) -> str:
        """
        Generate text using speculative decoding.

        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            num_draft_tokens: Number of tokens to draft per iteration
            verbose: Print progress information

        Returns:
            Generated text
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated_tokens = 0
        iterations = 0
        total_accepted = 0

        if verbose:
            print(f"Prompt: {prompt}")
            print(f"Generating {max_new_tokens} tokens with {num_draft_tokens} drafts per iteration...")
            print("-" * 80)

        while generated_tokens < max_new_tokens:
            iterations += 1

            # Step 1: Draft tokens
            draft_tokens, draft_probs = self.generate_draft_tokens(input_ids, num_draft_tokens)

            # Step 2: Verify drafts
            accepted_tokens = self.verify_draft_tokens(input_ids, draft_tokens, draft_probs)
            
            # Update statistics
            num_accepted = len(accepted_tokens)
            total_accepted += num_accepted
            generated_tokens += num_accepted

            if verbose:
                draft_text = self.tokenizer.decode(draft_tokens)
                accepted_text = self.tokenizer.decode(accepted_tokens)
                print(f"Iteration {iterations}:")
                print(f"  Drafted: {draft_text!r}")
                print(f"  Accepted: {accepted_text!r} ({num_accepted}/{len(draft_tokens)} tokens)")
            
            # Add accepted tokens to sequence
            input_ids = torch.cat([
                input_ids,
                torch.tensor([accepted_tokens], device=input_ids.device)
            ], dim=1)

            # Stop if we generated enough
            if generated_tokens >= max_new_tokens:
                break

        result = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        if verbose:
            print("-" * 80)
            print(f"Statistics:")
            print(f"  Total iterations: {iterations}")
            print(f"  Tokens generated: {generated_tokens}")
            # Avoid division by zero
            if iterations * num_draft_tokens > 0:
                print(f"  Average acceptance rate: {total_accepted / (iterations * num_draft_tokens):.2%}")

        return result

    def baseline_generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """
        Generate text using standard autoregressive decoding (for comparison).

        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            output_ids = self.target_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


def compare_methods(prompt: str, max_tokens: int = 30):
    """
    Compare speculative decoding vs standard generation.
    """
    print("=" * 80)
    print("SPECULATIVE DECODING DEMONSTRATION")
    print("=" * 80)
    print()

    # Init decoder
    decoder = SpeculativeDecoder(
        draft_model_name="google/gemma-2-2b-it",
        target_model_name="google/gemma-2-27b-it"
    )

    # Method 1: Speculative Decoding
    print("\n" + "=" * 80)
    print("METHOD 1: SPECULATIVE DECODING")
    print("=" * 80)
    start_time = time.time()
    spec_result = decoder.generate(prompt, max_new_tokens=max_tokens, num_draft_tokens=3, verbose=True)
    spec_time = time.time() - start_time
    print(f"Result: {spec_result}")
    print(f"Time taken: {spec_time:.2f}s")

    # Method 2: Standard Generation
    print("\n" + "=" * 80)
    print("METHOD 2: STANDARD AUTOREGRESSIVE GENERATION")
    print("=" * 80)
    start_time = time.time()
    baseline_result = decoder.baseline_generate(prompt, max_new_tokens=max_tokens)
    baseline_time = time.time() - start_time
    print(f"Result: {baseline_result}")
    print(f"Time taken: {baseline_time:.2f}s")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Speculative decoding time: {spec_time:.2f}s")
    print(f"Standard generation time: {baseline_time:.2f}s")
    
    # Avoid division by zero
    if spec_time > 0:
        print(f"Speedup: {baseline_time / spec_time:.2f}x")
    else:
        print("Speedup: N/A (Speculative time was zero)")
    print()


if __name__ == "__main__":

    # --- MODIFIED BLOCK ---
    # Load the dataset instead of using a hard-coded list
    print("Loading ai4bharat/IN22-Conv dataset...")
    try:
        # We learned from previous errors that 'default', 'test', and 'eng_Latn' are correct
        dataset = load_dataset("ai4bharat/IN22-Conv", name="default", split="test")
        
        # Find the first valid prompt
        prompt = None
        for i in range(len(dataset)):
            if dataset[i]['eng_Latn']:
                prompt = dataset[i]['eng_Latn']
                break
        
        if prompt is None:
            raise ValueError("No valid 'eng_Latn' prompt found in dataset.")
            
        print(f"Loaded prompt: {prompt}\n")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using fallback prompt.")
        prompt = "The future of artificial intelligence is"
    # --- END MODIFIED BLOCK ---

    compare_methods(prompt, max_tokens=128)
