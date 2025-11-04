# Event-Structured Story Continuation with JSON-Constrained LLM

> [!IMPORTANT]
> This project is developed under the **01204466 Deep Learning** course of **Department of Computer Engineering**,
  **Faculity of Engineering**, **Kasetsart University**.

> **Project Developers**: *Kritchanat Thanapiphatsiri (6610501955)*

This project builds an interactive storytelling assistant that predicts the next narrative beat for a romance roleplay between two characters, Sparkle and Caelus. A quantized Qwen2.5-1.5B-Instruct model generates the continuation while a JSON schema enforces structural constraints so downstream systems can reason about narration, dialogue, thoughts, and imagined scenarios.

## Why Deep Learning?
- Story continuations require nuanced style transfer and long-range context that hand-written templates struggle to capture.
- Transformer-based large language models can integrate conversational cues, shared history, and safety instructions in a single forward pass.
- JSON-constrained decoding preserves the expressiveness of deep learning while guaranteeing outputs that are machine-readable.

## Model Architecture
- Qwen2.5-1.5B-Instruct (decoder-only transformer, 24 layers) loaded in 4-bit NF4 quantization via bitsandbytes.
- Recent events (up to 16) are linearised into a prompt with system rules that emphasise PG-13 tone and schema compliance.
- `lmformatenforcer` provides a prefix function to keep sampling on the JSON schema rails.
- A lightweight mapper turns the validated JSON back into local `Event` domain objects.

## Dataset and Preparation
- 18 handcrafted story threads (average 7.2 events each) label each continuation with event type and optional actor.
- Manual QA ensured a mix of narration, dialogue, inner thought, and imagination to stress every schema branch.
- Threads were split 5:1 into development and evaluation folds; development threads guided prompt tuning, while evaluation threads remained untouched for final scoring.

## Training & Inference
- We rely on the instruction-tuned Qwen checkpoint without additional gradient updates to stay within project compute limits.
- Prompt and decoding parameters were iteratively tuned on development threads until schema violations disappeared (<5%).
- Default runtime settings: temperature 0.4 for evaluation (deterministic), 0.7 for interactive sampling, top-p 0.95, repetition penalty 1.05.

## Evaluation
| Metric | Score | Notes |
| ------ | ----- | ----- |
| JSON validity | 100.00% | No outputs broke the schema on held-out threads. |
| Event-type accuracy | 100.00% | Predicted event categories matched the gold labels. |
| BLEU | 0.9758 | Calculated with `evaluate.load("sacrebleu")`. |

Qualitative checks confirm the continuations remain playful, coherent, and compliant with PG-13 policies.

## Repository Layout
- `main.py` – Complete inference pipeline, domain schema, and evaluation harness.
- `final_report.typ` – Typst source for the full written report.
- `final_report.pdf` – Compiled report for submission.
- `HOMEWORK.md` – Assignment brief.

## Reproducing Results Locally
1. Install dependencies (PyTorch with CUDA, `transformers`, `bitsandbytes`, `lm-format-enforcer`, `evaluate`, `typst`).
2. Authenticate with Hugging Face if required to download `Qwen/Qwen2.5-1.5B-Instruct`.
3. Run `python main.py` to perform a smoke test and print the predicted continuation.
4. Import `evaluate_model` from `main.py` to recompute the metrics on the curated threads.

> **Tip:** If GPU memory is limited, keep the default 4-bit quantization; otherwise, adjust `BitsAndBytesConfig` for 8-bit or full precision.

## Colab Notebook
Full development notebook (including dependency installation, model download, and prompt tuning logs): https://colab.research.google.com/drive/1SakNwh6FETC62Qrqem182OQH4pLskEMA

## Report
The final write-up lives in `final_report.typ` and compiled `final_report.pdf`. Convert the PDF into Markdown for classroom submission by copying the rendered sections into the GitHub repository README.
