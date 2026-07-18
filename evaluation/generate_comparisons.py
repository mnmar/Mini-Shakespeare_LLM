"""
Mini-Shakespeare GPT — Qualitative generation benchmarking (Assignment Part 5b).

For each of 4 short Shakespearean prompts, encodes with the project's
ByteTokenizer, generates exactly 150 tokens of completion with Model A and
Model B (loaded from the trained checkpoints), and decodes back to text.

Sampling uses the model's default generate() settings (temperature=1.0, no
top_k) — no cherry-picking or truncation of degenerate output. The
assignment explicitly expects local mini-models to produce fractured /
repetitive text and grades the analysis of those failures, not polished
poetry, so raw output is kept as-is.

Writes `outputs/generations.json`: one entry per prompt, with the full text
(prompt + completion) and the completion-only text for both models.

Run from the `evaluation/` folder:

    python generate_comparisons.py
"""

import json
import sys
from pathlib import Path

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
OUT_DIR = THIS_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT_DIR / "data"))
sys.path.insert(0, str(ROOT_DIR / "my-transformer"))

from tokenizer import ByteTokenizer     # noqa: E402
from model import GPTLanguageModel      # noqa: E402

CKPT_DIR = ROOT_DIR / "my-transformer" / "checkpoints"
MODEL_NAMES = ["A", "B"]
MAX_NEW_TOKENS = 150
GEN_SEED = 42

PROMPTS = [
    "To be, or not to ",
    "O Romeo, Romeo, wherefore art thou ",
    "Friends, Romans, countrymen, lend me your ",
    "Now is the winter of our discontent",
]


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name, device):
    ckpt_path = CKPT_DIR / f"model_{name.lower()}.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPTLanguageModel(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def generate_completion(model, tokenizer, prompt, device):
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(idx, max_new_tokens=MAX_NEW_TOKENS)

    full_ids = out[0].tolist()
    full_text = tokenizer.decode(full_ids)
    completion_text = tokenizer.decode(full_ids[len(ids):])
    return full_text, completion_text


def main():
    device = get_device()
    print(f"device: {device}")

    tokenizer = ByteTokenizer()
    models = {name: load_model(name, device) for name in MODEL_NAMES}
    print(f"loaded models: {list(models.keys())}")

    # Seed once, not per-call: reseeding before every generate() would replay
    # the exact same sampled RNG stream for every prompt, making completions
    # from weaker models look artificially identical across different
    # prompts. Seeding once keeps the whole run reproducible while letting
    # the RNG evolve naturally across prompts/models.
    torch.manual_seed(GEN_SEED)

    results = []
    for prompt in PROMPTS:
        entry = {"prompt": prompt}
        for name, model in models.items():
            full_text, completion_text = generate_completion(model, tokenizer, prompt, device)
            entry[f"model_{name.lower()}_full_text"] = full_text
            entry[f"model_{name.lower()}_completion"] = completion_text
            print(f"\n[Model {name}] prompt={prompt!r}")
            print(f"  completion: {completion_text!r}")
        results.append(entry)

    out_path = OUT_DIR / "generations.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
