"""
Tensor-shape verification for the Mini-Shakespeare GPT (Assignment Part 3).

This script is the "verify all tensor shapes" deliverable. It:

  1. Pulls a REAL batch from the Task-2 data pipeline (get_batch).
  2. Walks a batch through every architectural block in isolation and asserts
     the shape after each stage, printing a running (B, T, C) trace.
  3. Runs a full forward pass and checks that the *initial* loss is close to
     ln(vocab_size) — the value you must get if the untrained logits are
     uniform. This catches init / masking bugs immediately.
  4. Proves the causal mask never leaks the future: scrambling the tokens after
     position t must not change the predictions at positions 0..t.
  5. Generates 150 tokens from a Shakespearean prompt and decodes them.
  6. (Sanity check) overfits a single batch for a few hundred steps to prove
     that gradients flow and the loss actually goes down.

Run:
    python verify_shapes.py
"""

import math
import sys
from pathlib import Path

import torch

# The Task-2 data pipeline lives in the sibling ../data folder. Add it to the
# import path so `from data import ...` resolves when running from my-transformer/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from data import get_batch, prepare_data
from model import (
    GPTConfig,
    Head,
    MultiHeadAttention,
    FeedForward,
    Block,
    GPTLanguageModel,
)


def banner(text):
    print("\n" + "=" * 68)
    print(text)
    print("=" * 68)


def check(label, tensor, expected):
    """Assert a tensor's shape and print a tidy trace line."""
    actual = tuple(tensor.shape)
    status = "OK " if actual == expected else "FAIL"
    print(f"  [{status}] {label:<38} {str(actual):<22} expected {expected}")
    assert actual == expected, f"{label}: got {actual}, expected {expected}"


def main():
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # A small config so the script runs fast. These numbers are illustrative —
    # the training teammate picks the real Model A / Model B values.
    cfg = GPTConfig(
        vocab_size=256, block_size=64, n_layer=2, n_head=4, n_embd=128, dropout=0.1
    )
    B = 4                      # batch size for this check
    T = cfg.block_size         # use the full context length
    C = cfg.n_embd
    H = cfg.n_head
    hs = cfg.head_size
    V = cfg.vocab_size

    banner("CONFIG")
    print(f"  device={device}")
    print(f"  B(batch)={B}  T(seq)={T}  C(n_embd)={C}  H(heads)={H}  hs(head_size)={hs}  V(vocab)={V}")

    # ---------------------------------------------------------------------
    # 1. Real batch from the Task-2 pipeline
    # ---------------------------------------------------------------------
    banner("1. INPUT BATCH  (from data.get_batch)")
    train_data, val_data, tokenizer = prepare_data()
    x, y = get_batch(train_data, batch_size=B, block_size=T)
    x, y = x.to(device), y.to(device)
    check("x (input ids)", x, (B, T))
    check("y (target ids)", y, (B, T))
    assert x.dtype == torch.long and y.dtype == torch.long
    print(f"  dtypes: x={x.dtype}, y={y.dtype}  (must be int64/long for embedding lookup)")

    # ---------------------------------------------------------------------
    # 2. Embeddings
    # ---------------------------------------------------------------------
    banner("2. EMBEDDINGS  (token + position)")
    model = GPTLanguageModel(cfg).to(device)
    tok_emb = model.token_embedding(x)                          # (B, T, C)
    pos = torch.arange(T, device=device)
    pos_emb = model.position_embedding(pos)                     # (T, C)
    check("token_embedding(x)", tok_emb, (B, T, C))
    check("position_embedding(arange T)", pos_emb, (T, C))
    emb = tok_emb + pos_emb                                     # broadcast over batch
    check("tok_emb + pos_emb (broadcast)", emb, (B, T, C))

    # ---------------------------------------------------------------------
    # 3. Self-attention: one head
    # ---------------------------------------------------------------------
    banner("3. SELF-ATTENTION  (single Head)")
    head = Head(cfg).to(device)
    k = head.key(emb); q = head.query(emb); v = head.value(emb)
    check("key(x) / query(x) / value(x)", k, (B, T, hs))
    att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(hs))      # scaled scores
    check("att = q @ k^T / sqrt(hs)", att, (B, T, T))
    head_out = head(emb)
    check("Head(x) output", head_out, (B, T, hs))
    print(f"  note: att is the (T x T) attention matrix — each row is a softmax over the past.")

    # ---------------------------------------------------------------------
    # 4. Multi-head attention
    # ---------------------------------------------------------------------
    banner("4. MULTI-HEAD ATTENTION")
    mha = MultiHeadAttention(cfg).to(device)
    concat = torch.cat([h(emb) for h in mha.heads], dim=-1)    # H*(B,T,hs) -> (B,T,C)
    check("concat of H heads", concat, (B, T, C))
    mha_out = mha(emb)
    check("MultiHeadAttention(x) output", mha_out, (B, T, C))
    print(f"  note: {H} heads x {hs} head_size = {H * hs} = C({C}); concat then projected.")

    # ---------------------------------------------------------------------
    # 5. Feed-forward
    # ---------------------------------------------------------------------
    banner("5. FEED-FORWARD  (MLP, 4x expansion)")
    ff = FeedForward(cfg).to(device)
    hidden = ff.act(ff.fc(emb))                                # (B, T, 4C)
    check("fc(x) -> GELU  (expand to 4C)", hidden, (B, T, 4 * C))
    ff_out = ff(emb)
    check("FeedForward(x) output", ff_out, (B, T, C))

    # ---------------------------------------------------------------------
    # 6. Transformer block
    # ---------------------------------------------------------------------
    banner("6. TRANSFORMER BLOCK  (attn + ffn + residuals)")
    block = Block(cfg).to(device)
    block_out = block(emb)
    check("Block(x) output", block_out, (B, T, C))
    print(f"  note: input and output shapes match — blocks are stackable ({cfg.n_layer} deep here).")

    # ---------------------------------------------------------------------
    # 7. Full forward pass: logits + loss
    # ---------------------------------------------------------------------
    banner("7. FULL FORWARD  (embeddings -> blocks -> ln_f -> lm_head)")
    logits, loss = model(x, y)
    check("logits", logits, (B, T, V))
    print(f"  [OK ] loss is a scalar: shape={tuple(loss.shape)}  value={loss.item():.4f}")

    # The head/LM should start out predicting ~uniformly over V tokens, so the
    # cross-entropy at step 0 should be about ln(V). Far from this => a bug.
    expected_loss = math.log(V)
    print(f"  expected initial loss ~ ln({V}) = {expected_loss:.4f}")
    assert abs(loss.item() - expected_loss) < 0.7, (
        "initial loss is far from ln(vocab); check init / masking / logits."
    )
    print(f"  [OK ] initial loss is within tolerance of ln(vocab).")
    print(f"  parameters (tied): {model.num_params():,}")

    # ---------------------------------------------------------------------
    # 8. Correctness: causal masking + weight tying
    # ---------------------------------------------------------------------
    banner("8. CORRECTNESS  (causal mask must not leak the future)")
    model.eval()  # turn OFF dropout so two forward passes are directly comparable
    with torch.no_grad():
        logits_ref, _ = model(x)                                    # (B, T, V)
        t = T // 2
        x_future = x.clone()
        # Scramble every token STRICTLY AFTER position t with random bytes.
        x_future[:, t + 1:] = torch.randint(
            0, V, x_future[:, t + 1:].shape, device=device
        )
        logits_future, _ = model(x_future)
        # A token may attend only to itself and the past, so the predictions at
        # positions 0..t must be identical even though the future changed.
        unchanged = torch.allclose(
            logits_ref[:, : t + 1], logits_future[:, : t + 1], atol=1e-5
        )
        assert unchanged, "CAUSAL MASK LEAK: future tokens changed past predictions!"
    print(f"  [OK ] scrambled all tokens after position {t}; logits[:, :{t + 1}] unchanged")
    print(f"        -> causal mask is correct: no information leaks from the future.")
    assert model.lm_head.weight is model.token_embedding.weight
    print(f"  [OK ] lm_head.weight IS token_embedding.weight -> weight tying active.")
    model.train()  # restore default mode

    # ---------------------------------------------------------------------
    # 9. Generation
    # ---------------------------------------------------------------------
    banner("9. GENERATION  (autoregressive, 150 tokens)")
    prompt = "To be, or not to "
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    check("prompt context", context, (1, len(prompt)))
    out = model.generate(context, max_new_tokens=150, temperature=1.0, top_k=50)
    check("generate() output", out, (1, len(prompt) + 150))
    decoded = tokenizer.decode(out[0].tolist())
    print("  decoded sample (UNTRAINED model — expected to be gibberish):")
    print("  " + "-" * 60)
    for line in decoded.splitlines():
        print("  | " + line)
    print("  " + "-" * 60)

    # ---------------------------------------------------------------------
    # 10. Sanity check: can it overfit a single batch?
    # ---------------------------------------------------------------------
    banner("10. SANITY CHECK  (overfit ONE batch -> loss must drop)")
    xb, yb = get_batch(train_data, batch_size=B, block_size=T)
    xb, yb = xb.to(device), yb.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    _, first_loss = model(xb, yb)
    print(f"  step   0 loss = {first_loss.item():.4f}")
    for step in range(1, 301):
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"  step {step:3d} loss = {loss.item():.4f}")
    assert loss.item() < first_loss.item() - 1.0, "model failed to overfit one batch"
    print("  [OK ] loss dropped sharply -> backprop works, the model can learn.")

    banner("ALL SHAPE CHECKS PASSED")


if __name__ == "__main__":
    main()
