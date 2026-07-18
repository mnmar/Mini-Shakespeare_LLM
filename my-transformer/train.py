"""
Mini-Shakespeare GPT — Training & Experiments (Assignment Part 4).

Trains two model configurations on the Tiny Shakespeare byte-level dataset:

    Model A (baseline) — shallow / narrow, small context window
    Model B (scaled)   — deeper / wider, larger context window

For each model this script:
    1. Runs a fixed number of training steps (AdamW + cross-entropy).
    2. Periodically estimates train and validation loss (averaged over
       several batches, with dropout switched off) and logs it.
    3. Saves the loss history to `logs/model_a_losses.csv` /
       `logs/model_b_losses.csv` and a checkpoint to
       `checkpoints/model_a.pt` / `checkpoints/model_b.pt`.
    4. After both models are trained, plots a single comparison graph of
       train/val loss vs. step for A and B, saved to `logs/loss_curves.png`.

Run (from the `my-transformer/` folder):

    python train.py                  # trains both Model A and Model B
    python train.py --model a        # trains only Model A
    python train.py --model b        # trains only Model B
    python train.py --steps 3000     # override the number of training steps
    python train.py --device cpu     # force a specific device

Part 5 (evaluation/benchmarking: final val loss, perplexity, generation,
comparison against Gemini Flash) consumes the checkpoints and CSV logs
produced here — see the `evaluation/` folder.
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch

# The Task-2 data pipeline lives in the sibling ../data folder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from data import get_batch, prepare_data          # noqa: E402
from model import GPTConfig, GPTLanguageModel      # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
LOG_DIR = THIS_DIR / "logs"
CKPT_DIR = THIS_DIR / "checkpoints"
LOG_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Model A (baseline) vs Model B (scaled) — the only thing that differs
# between the two experiments is this config, per the assignment brief.
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "A": GPTConfig(
        vocab_size=256, block_size=64, n_layer=2, n_head=4, n_embd=128, dropout=0.1,
    ),
    "B": GPTConfig(
        vocab_size=256, block_size=128, n_layer=4, n_head=8, n_embd=256, dropout=0.1,
    ),
}

DEFAULT_STEPS = 3000
EVAL_INTERVAL = 100     # how often (in steps) to estimate train/val loss
EVAL_ITERS = 50         # batches averaged per loss estimate
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
SEED = 1337


def get_device(force=None):
    if force:
        return force
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, device, eval_iters=EVAL_ITERS):
    """Average loss over `eval_iters` random batches, for train and val splits.

    Dropout is switched off (model.eval()) so the estimate isn't noisy from
    stochastic regularization; a single batch would also be too noisy on its
    own, hence the averaging.
    """
    model.eval()
    out = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(data, batch_size=BATCH_SIZE, block_size=block_size)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out["train"], out["val"]


def train_model(name, config, steps, train_data, val_data, device):
    print(f"\n{'=' * 70}\nTraining Model {name}\n{config}\n{'=' * 70}")
    torch.manual_seed(SEED)
    model = GPTLanguageModel(config).to(device)
    print(f"parameters: {model.num_params():,} | device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    history = []  # (step, train_loss, val_loss)
    start = time.time()

    for step in range(steps + 1):
        if step % EVAL_INTERVAL == 0 or step == steps:
            train_loss, val_loss = estimate_loss(
                model, train_data, val_data, config.block_size, device
            )
            history.append((step, train_loss, val_loss))
            elapsed = time.time() - start
            print(
                f"  step {step:5d}/{steps} | train {train_loss:.4f} | "
                f"val {val_loss:.4f} | {elapsed:6.1f}s"
            )

        x, y = get_batch(train_data, batch_size=BATCH_SIZE, block_size=config.block_size)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    ckpt_path = CKPT_DIR / f"model_{name.lower()}.pt"
    torch.save({"config": config, "state_dict": model.state_dict()}, ckpt_path)
    print(f"saved checkpoint -> {ckpt_path}")

    log_path = LOG_DIR / f"model_{name.lower()}_losses.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss"])
        writer.writerows(history)
    print(f"saved loss log   -> {log_path}")

    final_train, final_val = history[-1][1], history[-1][2]
    perplexity = math.exp(final_val)
    print(
        f"final: train_loss={final_train:.4f}  val_loss={final_val:.4f}  "
        f"perplexity=e^val_loss={perplexity:.2f}"
    )

    return model, history


def plot_comparison(histories, out_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5.5))
    colors = {"A": "tab:blue", "B": "tab:red"}
    for name, history in histories.items():
        steps = [h[0] for h in history]
        train_losses = [h[1] for h in history]
        val_losses = [h[2] for h in history]
        c = colors.get(name, None)
        plt.plot(steps, train_losses, color=c, linestyle="--", label=f"Model {name} (train)")
        plt.plot(steps, val_losses, color=c, linestyle="-", label=f"Model {name} (val)")

    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy loss")
    plt.title("Mini-Shakespeare GPT — Model A vs Model B loss curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nsaved comparison plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Model A / Model B and log loss curves.")
    parser.add_argument("--model", choices=["a", "b", "both"], default="both")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--device", default=None, help="Force 'cpu', 'cuda', or 'mps'.")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"device: {device}")

    train_data, val_data, tokenizer = prepare_data()
    print(f"train tokens: {len(train_data):,}  val tokens: {len(val_data):,}  vocab: {tokenizer.vocab_size}")

    names = ["A", "B"] if args.model == "both" else [args.model.upper()]

    histories = {}
    for name in names:
        _, history = train_model(name, MODEL_CONFIGS[name], args.steps, train_data, val_data, device)
        histories[name] = history

    if len(histories) > 1:
        plot_comparison(histories, LOG_DIR / "loss_curves.png")
    else:
        print("\n(Only one model trained — run with --model both, or run the other model "
              "separately, to generate the comparison plot.)")


if __name__ == "__main__":
    main()
