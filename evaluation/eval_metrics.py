"""
Mini-Shakespeare GPT — Evaluation metrics (Assignment Part 5a).

Loads the two trained checkpoints (Model A, Model B) produced by
`my-transformer/train.py`, and for each model:

    1. Reconstructs the model from the config embedded in the checkpoint
       (no hardcoded/guessed hyperparameters).
    2. Computes a *deterministic, full-pass* cross-entropy loss over the
       entire held-out validation slice (non-overlapping windows, dropout
       off) and perplexity = exp(loss). This is a stricter number than the
       training-time estimate in the loss CSVs, which averages 50 randomly
       sampled batches (see `train.py::estimate_loss`) — both numbers are
       reported, clearly labeled, since they answer different questions.
    3. Records the model's actual hyperparameters and parameter count.

Writes `outputs/metrics.json`, `outputs/metrics.csv`, and
`outputs/perplexity_comparison.png`. This JSON/CSV pair is the single
source of truth the README and report/report.md pull their hyperparameter
and results tables from — nothing downstream should re-type these numbers.

Run from the `evaluation/` folder:

    python eval_metrics.py
"""

import csv
import json
import math
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

from data import prepare_data          # noqa: E402
from model import GPTLanguageModel      # noqa: E402
import train as train_module            # noqa: E402  (for shared hyperparameter constants)

CKPT_DIR = ROOT_DIR / "my-transformer" / "checkpoints"
LOG_DIR = ROOT_DIR / "my-transformer" / "logs"

MODEL_NAMES = ["A", "B"]


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name, device):
    """Load a checkpoint saved as {"config": GPTConfig, "state_dict": ...}.

    weights_only=False is required because the checkpoint embeds a
    GPTConfig dataclass instance (not just tensors) inside the "config"
    key — this is our own trusted checkpoint, produced by train.py in this
    same repo, so it's safe to disable the newer PyTorch default.
    """
    ckpt_path = CKPT_DIR / f"model_{name.lower()}.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GPTLanguageModel(config).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, config


def read_csv_final_row(name):
    """Return (step, train_loss, val_loss) from the last row of the training loss log."""
    log_path = LOG_DIR / f"model_{name.lower()}_losses.csv"
    with open(log_path, newline="") as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]
    return int(last["step"]), float(last["train_loss"]), float(last["val_loss"])


@torch.no_grad()
def full_pass_val_loss(model, val_data, block_size, device):
    """Deterministic cross-entropy loss over the ENTIRE held-out validation
    slice, using non-overlapping windows of length block_size (no random
    sampling, dropout off via model.eval()). This is the "final validation
    loss on a held-out evaluation text slice" the assignment asks for.
    """
    n_windows = (len(val_data) - 1) // block_size
    assert n_windows > 0, "validation slice is shorter than one block_size window"

    total_loss = 0.0
    for i in range(n_windows):
        start = i * block_size
        x = val_data[start:start + block_size].unsqueeze(0).to(device)
        y = val_data[start + 1:start + block_size + 1].unsqueeze(0).to(device)
        _, loss = model(x, y)
        total_loss += loss.item()

    return total_loss / n_windows


def main():
    device = get_device()
    print(f"device: {device}")

    _, val_data, tokenizer = prepare_data()
    print(f"val tokens: {len(val_data):,}  vocab: {tokenizer.vocab_size}")

    results = []
    for name in MODEL_NAMES:
        model, config = load_model(name, device)
        csv_step, csv_train_loss, csv_val_loss = read_csv_final_row(name)

        computed_val_loss = full_pass_val_loss(model, val_data, config.block_size, device)
        computed_perplexity = math.exp(computed_val_loss)
        csv_perplexity = math.exp(csv_val_loss)

        row = {
            "model": name,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "block_size": config.block_size,
            "vocab_size": config.vocab_size,
            "dropout": config.dropout,
            "batch_size": train_module.BATCH_SIZE,
            "learning_rate": train_module.LEARNING_RATE,
            "steps": csv_step,
            "num_params": model.num_params(),
            "training_time_final_train_loss": csv_train_loss,
            "training_time_final_val_loss": csv_val_loss,
            "training_time_perplexity": csv_perplexity,
            "full_pass_val_loss": computed_val_loss,
            "full_pass_perplexity": computed_perplexity,
        }
        results.append(row)

        print(f"\nModel {name}: params={row['num_params']:,}")
        print(f"  training-time (CSV, random-batch estimate) : val_loss={csv_val_loss:.4f}  perplexity={csv_perplexity:.2f}")
        print(f"  full-pass (this script, deterministic)      : val_loss={computed_val_loss:.4f}  perplexity={computed_perplexity:.2f}")

    json_path = OUT_DIR / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {json_path}")

    csv_path = OUT_DIR / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"saved -> {csv_path}")

    plot_perplexity_comparison(results, OUT_DIR / "perplexity_comparison.png")


def plot_perplexity_comparison(results, out_path):
    import matplotlib.pyplot as plt

    names = [r["model"] for r in results]
    perplexities = [r["full_pass_perplexity"] for r in results]

    plt.figure(figsize=(6, 5))
    bars = plt.bar(names, perplexities, color=["tab:blue", "tab:red"])
    for bar, p in zip(bars, perplexities):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{p:.2f}",
                  ha="center", va="bottom")
    plt.ylabel("Perplexity (full-pass, held-out val slice)")
    plt.title("Mini-Shakespeare GPT — Model A vs Model B perplexity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
