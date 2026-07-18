"""
Mini-Shakespeare GPT — Transformer architecture (Assignment Part 3).

A small, from-scratch GPT-style *decoder-only* Transformer in PyTorch. Every
architectural block the assignment asks for lives in this file:

    - Token + positional embeddings ....... GPTLanguageModel.__init__ / .forward
    - Self-attention (one causal head) .... class Head
    - Multi-head attention ................ class MultiHeadAttention
    - Feed-forward / MLP .................. class FeedForward
    - Transformer block .................. class Block
    - Final language-model head .......... GPTLanguageModel.lm_head
    - Autoregressive text generation ..... GPTLanguageModel.generate

We deliberately build the blocks ourselves (no nn.Transformer /
nn.MultiheadAttention), using only PyTorch primitives (Linear, Embedding,
LayerNorm, Dropout) plus the `@` / matmul operator. This mirrors the microGPT
blueprint but replaces its scalar Python loops with vectorized tensor ops.

This module depends ONLY on torch, so it is portable: the training and
evaluation teammates import `GPTConfig` and `GPTLanguageModel` from here.

Shape convention used in every comment below:
    B  = batch size
    T  = time / sequence length  (number of tokens in the context, T <= block_size)
    C  = channels / embedding dimension  (config.n_embd)
    H  = number of attention heads  (config.n_head)
    hs = head size = C // H
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class GPTConfig:
    """All architecture knobs live here.

    The training teammate builds Model A / Model B simply by passing different
    numbers into this object — the model code below never hardcodes a size.
    """
    vocab_size: int = 256    # byte-level tokenizer -> fixed at 256 (0..255)
    block_size: int = 64     # context length: the maximum T the model can see
    n_layer: int = 2         # number of stacked Transformer blocks (depth)
    n_head: int = 4          # number of attention heads per block
    n_embd: int = 128        # embedding dimension C (must be divisible by n_head)
    dropout: float = 0.1     # dropout probability (use 0.0 to disable)
    tie_weights: bool = True # share the token embedding matrix with the LM head

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head}) "
            f"so the C channels split evenly across heads."
        )

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head


# ---------------------------------------------------------------------------
# Self-attention: ONE causal head
# ---------------------------------------------------------------------------
class Head(nn.Module):
    """A single head of causal (masked) self-attention.

    Each token emits a query, a key and a value. A token's query is compared
    against every *earlier* token's key to decide how much to read from that
    token's value. "Causal" = a token may only look left (at itself and the
    past), never at the future — which is what makes this a language model.

        forward:  x (B, T, C)  ->  out (B, T, hs)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        C = config.n_embd
        hs = config.head_size

        # Three linear projections C -> hs. bias=False is the GPT convention.
        self.key = nn.Linear(C, hs, bias=False)     # "what do I contain?"
        self.query = nn.Linear(C, hs, bias=False)   # "what am I looking for?"
        self.value = nn.Linear(C, hs, bias=False)   # "what do I pass on if attended to?"

        # Causal mask: a lower-triangular matrix of ones. Registered as a buffer
        # (not a Parameter) so it is saved and moved to the GPU with .to(device),
        # but is never updated by the optimizer.
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size)),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)     # (B, T, hs)
        q = self.query(x)   # (B, T, hs)
        v = self.value(x)   # (B, T, hs)

        # Attention scores = scaled dot-product of every query with every key.
        # (B, T, hs) @ (B, hs, T) -> (B, T, T). Entry [b, i, j] = how much
        # token i attends to token j. We divide by sqrt(hs) so the scores keep
        # unit variance and softmax does not saturate into one-hot too early.
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))  # (B, T, T)

        # Causal masking: zero out the future by setting the upper triangle to
        # -inf *before* softmax (exp(-inf) = 0). Slice tril to the current T so
        # sequences shorter than block_size still work (e.g. during generation).
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))   # (B, T, T)
        att = F.softmax(att, dim=-1)     # (B, T, T) -> rows are probabilities summing to 1
        att = self.dropout(att)

        # Weighted sum of value vectors: each token becomes a blend of the value
        # vectors it attended to. (B, T, T) @ (B, T, hs) -> (B, T, hs).
        out = att @ v        # (B, T, hs)
        return out


# ---------------------------------------------------------------------------
# Multi-head attention: several heads in parallel
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """Runs `n_head` independent self-attention heads and combines them.

    Different heads can specialise on different relationships (e.g. one tracks
    the previous character, another tracks matching quotes). We concatenate
    their outputs back to width C and mix them with a linear projection.

        forward:  x (B, T, C)  ->  out (B, T, C)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        # H separate heads, each producing (B, T, hs). H * hs == C.
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        # Output projection lets the heads' outputs interact before the residual.
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Concatenate all heads along the channel dim:
        # H tensors of (B, T, hs) -> (B, T, H*hs) = (B, T, C).
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, C)
        out = self.dropout(self.proj(out))                    # (B, T, C)
        return out


# ---------------------------------------------------------------------------
# Feed-forward (MLP) applied per token
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    """Position-wise MLP: the same 2-layer network applied to every token
    independently. Attention moves information *between* tokens; this block
    lets each token "think" on the information it just gathered.

    Expands to 4*C (extra capacity), applies a non-linearity, projects back.

        forward:  x (B, T, C)  ->  out (B, T, C)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        C = config.n_embd
        self.fc = nn.Linear(C, 4 * C)     # (B, T, C)  -> (B, T, 4C)
        self.act = nn.GELU()              # smooth non-linearity used by GPT-2/3
        self.proj = nn.Linear(4 * C, C)   # (B, T, 4C) -> (B, T, C)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.act(self.fc(x))          # (B, T, 4C)
        x = self.dropout(self.proj(x))    # (B, T, C)
        return x


# ---------------------------------------------------------------------------
# Transformer block: attention + feed-forward, each with a residual
# ---------------------------------------------------------------------------
class Block(nn.Module):
    """One Transformer block. We use the *pre-norm* formulation (LayerNorm
    applied before each sub-layer), like GPT-2, which trains more stably than
    the original post-norm design.

        x = x + attention(norm(x))     # communicate between tokens
        x = x + feedforward(norm(x))   # compute within each token

    The `x + ...` are residual (skip) connections: they give gradients a clean
    highway back through the network, letting us stack many blocks.

        forward:  x (B, T, C)  ->  out (B, T, C)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))    # (B, T, C)
        x = x + self.ffwd(self.ln2(x))    # (B, T, C)
        return x


# ---------------------------------------------------------------------------
# The full GPT language model
# ---------------------------------------------------------------------------
class GPTLanguageModel(nn.Module):
    """Decoder-only Transformer for next-token (next-byte) prediction.

    Pipeline:
        idx (B, T) token ids
          -> token embedding + positional embedding      (B, T, C)
          -> n_layer Transformer blocks                  (B, T, C)
          -> final LayerNorm                             (B, T, C)
          -> language-model head                         (B, T, vocab_size)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # --- Embeddings ---
        # Token embedding: a lookup table mapping each of the 256 byte ids to a
        # learned C-dim vector ("what is this token?").
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # Positional embedding: a lookup table mapping each position 0..T-1 to a
        # learned C-dim vector ("where is this token?"). Attention alone is
        # order-agnostic, so we must inject position explicitly.
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # --- Transformer blocks (the depth of the model) ---
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)   # final LayerNorm before the head

        # --- Language-model head ---
        # Projects each token's C-vector to a score (logit) for every possible
        # next token in the vocabulary.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: reuse the token-embedding matrix as the output head.
        # Input and output live in the same vocabulary, so sharing the matrix
        # saves parameters and usually helps generalisation (GPT-2 does this).
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # Initialise all weights (GPT-2 style), then apply a special scaled init
        # to the residual-path projections (see _init_weights / loop below).
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            # The two projections that write into a residual stream ("proj.weight")
            # are scaled by 1/sqrt(2 * n_layer). Because residual outputs
            # accumulate across all 2*n_layer sub-layers, this keeps the variance
            # of the residual stream from growing with depth (GPT-2 trick).
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        # Small Gaussian init (std 0.02) is the standard GPT-2 choice; a large
        # init would make the softmax logits explode before any training.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        """Total number of trainable parameters (tied weights counted once)."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) LongTensor of token ids (each in 0..vocab_size-1)
        targets: (B, T) LongTensor of the next token at each position, or None

        Returns (logits, loss):
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy if targets given, else None
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"sequence length T={T} exceeds block_size={self.config.block_size}; "
            f"crop the context before calling forward."
        )

        # Look up token and position embeddings and add them.
        tok_emb = self.token_embedding(idx)                          # (B, T, C)
        pos = torch.arange(T, device=idx.device)                     # (T,)
        pos_emb = self.position_embedding(pos)                       # (T, C)
        x = self.drop(tok_emb + pos_emb)     # (B, T, C); pos_emb broadcasts over B

        # Pass through the stack of Transformer blocks.
        for block in self.blocks:
            x = block(x)                                             # (B, T, C)
        x = self.ln_f(x)                                            # (B, T, C)

        # Project to vocabulary logits.
        logits = self.lm_head(x)                                    # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # cross_entropy wants (N, C) logits vs (N,) class indices, so we
            # flatten the batch and time dimensions together: N = B*T.
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively extend `idx` by `max_new_tokens` tokens.

        idx: (B, T) starting context of token ids.
        Returns: (B, T + max_new_tokens).

        temperature: >1 flattens the distribution (more random), <1 sharpens it.
        top_k:       if set, sample only from the k most likely next tokens
                     (truncates the long low-probability tail).
        """
        # Dropout must be off while sampling; remember the mode and restore it.
        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            # The model only has block_size positional embeddings, so we can
            # never feed more than block_size tokens: crop to the last window.
            idx_cond = idx[:, -self.config.block_size:]              # (B, <=block_size)

            logits, _ = self(idx_cond)                              # (B, T, vocab_size)
            # We only care about the prediction at the LAST position.
            logits = logits[:, -1, :] / temperature                # (B, vocab_size)

            if top_k is not None:
                # Keep the top-k logits, set the rest to -inf so they get 0 prob.
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)                       # (B, k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)                      # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)     # (B, 1) sampled id
            idx = torch.cat((idx, idx_next), dim=1)                # (B, T+1)

        if was_training:
            self.train()
        return idx


if __name__ == "__main__":
    # Tiny self-test so `python model.py` proves the file is internally consistent.
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=256, block_size=32, n_layer=2, n_head=4, n_embd=64)
    model = GPTLanguageModel(cfg)

    B, T = 3, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    logits, loss = model(idx, targets)
    print("logits:", tuple(logits.shape), "| loss:", round(loss.item(), 4))
    print("parameters:", f"{model.num_params():,}")

    out = model.generate(idx, max_new_tokens=20)
    print("generate:", tuple(idx.shape), "->", tuple(out.shape))
    print("model.py self-test passed.")
