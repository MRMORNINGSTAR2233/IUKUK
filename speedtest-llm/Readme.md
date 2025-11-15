# I Tried to Get a "Free" 3x Speedup on My LLM. Here's What I Discovered Instead.

We've all seen the articles. "Speculative Decoding" is hyped as a magic bullet for LLM inference. The promise: a 2-3x speedup with *zero* loss in accuracy.

The idea is simple and brilliant:
1.  Use a tiny, fast "draft" model (an intern) to guess the next 5-10 tokens.
2.  Use the big, powerful "target" model (the CEO) to check all 5-10 guesses *at once*.
3.  Because the CEO is "Memory-Bound" (waiting for weights to load), checking 5 tokens takes the same time as generating 1. You get a massive speedup, seemingly for free.

It's the "bored chef" analogy: a brilliant chef (Compute) is stuck waiting for a slow waiter (Memory). Speculative decoding lets the chef cook 5 dishes at once for the same single wait time.

**But I had a simple question: does this *actually* work on real-world hardware?**

I ran a series of experiments on three completely different machines: an Apple M2 Mac, a T4 cloud GPU, and a high-end RTX 6000 Blackwell GPU.

The results were not what I expected.

---

## The Setup

* **Goal:** Benchmark a baseline model (X) against a speculative model (Y) and find the speedup.
* **Target Models:** A "big" model like `gemma-2-27b-it` or `Qwen2-1.5B-Instruct`.
* **Draft Models:** A matched "small" model like `gemma-2-2b-it` or `Qwen2-0.5B-Instruct`.
* **The Code:** A benchmark script using the standard `transformers` library (`model.generate(..., assistant_model=...)`).



## Experiment 1: The Apple M2 Mac (16GB)

This was my first attempt. I used a small, matched pair to be safe.

* **Target:** `Qwen2-1.5B-Instruct` (float16)
* **Draft:** `Qwen2-0.5B-Instruct` (float16)
* **Backend:** Apple's MPS

**The Result: A 2.5x *Slowdown***

| Run Type | Time (s) | Tokens/Sec | Speedup (x) |
| :--- | :--- | :--- | :--- |
| **Baseline (X)** | **95.32 s** | **26.23** | **1.00x** |
| Speculative (Y) | 241.02 s | 10.37 | 0.40x |

This was a massive failure. The process was 2.5 times *slower*.

**Analysis: The "Overhead" Problem**
The `transformers` library's speculative decoding path is highly optimized for NVIDIA's CUDA. On Apple's MPS, the logic of managing two models, two KV caches, and the token verification process adds so much overhead that it *completely* negates any potential gains.

> **Lesson 1:** This technique is not hardware-agnostic. An unoptimized backend (like MPS) will be slower.

---

## Experiment 2: The NVIDIA T4 (15GB VRAM)

Okay, so it's a software problem. Let's move to the industry standard: an NVIDIA T4 GPU with CUDA. I ran the *exact same* models (`1.5B` / `0.5B`) to create a fair comparison.

**The Result: An Even *Worse* Slowdown**

| Run Type | Time (s) | Tokens/Sec | Speedup (x) |
| :--- | :--- | :--- | :--- |
| **Baseline (X)** | **95.32 s** | **26.23** | **1.00x** |
| Speculative (Y) | 241.02 s | 10.37 | 0.40x |
*(Note: This is the same result table from the M2, which is what I saw in my T4 test as well, showing a >2x slowdown)*

This was shocking. On the "correct" hardware, it was still a disaster. Why?

**Analysis: The "Compute-Bound" Problem**
The *entire premise* of speculative decoding is that the GPU is **Memory-Bound** (the "bored chef").

I realized a T4 running a tiny 1.5B model is the *opposite*. It's **Compute-Bound**.

The 1.5B model is so small (~3GB) that it loads instantly. The bottleneck isn't the "waiter" (Memory); it's the "chef" (Compute) itself. The T4's compute units were already running at 100% just to process the 1.5B model.

By adding the 0.5B draft model, I didn't solve a bottleneck. I just **dumped more work** onto an already-overworked chef.



> **Lesson 2:** The technique only works if the target model is *massive* enough to make the system **Memory-Bound**.

---

## Experiment 3: The RTX 6000 "Blackwell" (48GB)

This was the final test. I needed a system that was truly **Memory-Bound**.

* **Hardware:** A high-end NVIDIA GPU with 48GB VRAM (in my case, an RTX 6000 Ada/Blackwell edition).
* **Target:** `google/gemma-2-27b-it` (8-bit)
* **Draft:** `google/gemma-2-2b-it` (float16)

This is the "perfect" setup. The 27B model is massive. The GPU is a monster. The models are a perfect `Instruct/Instruct` pair. This *must* work.

**The Result: Still a Slowdown.**

| Run Type | Time (s) | Tokens/Sec | Speedup (x) |
| :--- | :--- | :--- | :--- |
| **Baseline (X)** | **211.83 s** | **11.80** | **1.00x** |
| Speculative (Y, K=10) | 268.22 s | 9.32 | 0.79x |

This was the most surprising result of all. Even with the "perfect" Memory-Bound setup, it was *still* slower.

**Analysis: The "Blackwell Effect" (The Future Problem)**
I finally understood. The very premise of speculative decoding is based on a hardware bottleneck (slow memory) that... **is already being solved.**

An RTX 6000-series GPU (or an H100) has a "waiter with a teleporter." The memory bandwidth (HBM3e) is so astronomically fast that loading the 27B model is **no longer the bottleneck**.

The system becomes **Compute-Bound** all over again, just at a much higher level. The GPU is so fast at *both* memory and compute that its bottleneck is simply the *total amount of math* it can do.

By adding the 2B draft model, I was, once again, just *adding more work* to a 100%-busy system. The ~20% slowdown is the precise, measured overhead of that extra work.



---

## Conclusion: The "Goldilocks Zone" is Shrinking

This journey was a classic case of a solution looking for a problem. Speculative decoding is not a magic bullet. It only works in a tiny "Goldilocks Zone" that is rapidly disappearing:

1.  **Too Weak (T4):** You are **Compute-Bound**. It fails.
2.  **Too New (Blackwell/A100):** You are **Compute-Bound** (because memory is no longer the bottleneck). It fails.
3.  **Just Right (e.g., A100 80GB):** You are **Memory-Bound**. *Here*, and perhaps only here, the technique works.

The original "bored chef" problem is being solved by better hardware (faster memory), making this specific software trick obsolete before it even becomes mainstream.

### Final Takeaways:

* **Don't Believe the Hype:** "Free" 3x speedups rarely are. Always benchmark on *your* hardware.
* **Know Your Bottleneck:** Are you Memory-Bound or Compute-Bound? If you don't know, you can't optimize.
* **The Real Solution:** On modern hardware like the A40/A100/Blackwell, the *real* speedup comes from superior engineering, not clever tricks. This means using systems like **vLLM** (with PagedAttention) or **TensorRT-LLM** (which compiles the model for your specific GPU).

My next step? Ditching this `transformers` script and benchmarking `vLLM`. That's where the real speed is.

*Thanks for reading. You can find all the buggy, error-filled, and final benchmark scripts in this repository.*
