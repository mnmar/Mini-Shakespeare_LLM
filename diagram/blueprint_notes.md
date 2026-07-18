Step 1: Understanding the Blueprint (microGPT) 

What it is: microgpt.py is a ~200-line, dependency-free Python implementation of a GPT-style transformer. It trains and runs inference on a character-level dataset (by default, a list of names) with no external libraries — not even NumPy.

Core components:

Custom Autograd Engine (Value class) — Every scalar number is wrapped in a Value object storing data, grad, its parent nodes (children), and local gradients (_local_grads). Operations like __add, __mul_, .exp(), .log(), .relu() each define how gradients flow backward. The .backward() method does a topological sort of the computation graph, then applies the chain rule in reverse — this is a scalar-level mirror of what Zhou's article explains conceptually.
Tokenizer — Character-level: every unique character in the dataset becomes a token ID, plus one special BOS (beginning/end of sequence) token. Vocab size = unique characters + 1.
Parameters (state_dict) — Includes token embeddings (wte), position embeddings (wpe), an LM head, and per-layer weights: attn_wq/wk/wv/wo (attention projections) and mlp_fc1/fc2 (feedforward). Weights are initialized as small Gaussian-random Value matrices.
Forward pass (gpt() function):
Combine token + position embeddings, apply rmsnorm.
Multi-head attention: project into Q, K, V; split into n_head heads; compute scaled dot-product attention per head using a running keys/values cache (this is how causality is enforced — each step only sees past tokens, without an explicit mask); concatenate heads and project back.
Residual connection around attention.
MLP block: linear (16→64) → ReLU → linear (64→16), also wrapped in a residual connection.
Final linear layer (lm_head) maps back to vocab-size logits.
Training loop — For each document, tokenize with BOS on both ends, run the model step-by-step through the sequence, compute cross-entropy loss (-log(prob of correct next token)), call .backward(), and update all parameters with Adam (manually implemented, including bias-corrected moving averages) over 1000 steps with linear learning-rate decay.
Inference — Autoregressive sampling: feed BOS, get logits, apply temperature-scaled softmax, sample a token, feed it back in, repeat until BOS is generated again (end of sequence) or block_size is reached.
Key takeaway for the assignment: This file proves that a full GPT — autograd, embeddings, multi-head attention, residuals, MLP, Adam optimizer, and autoregressive sampling — can be built with plain Python control flow and no matrix libraries. Everything your team will do with PyTorch later (tensors, batching, nn.Module, autograd) is just an efficiency layer over these exact same operations.