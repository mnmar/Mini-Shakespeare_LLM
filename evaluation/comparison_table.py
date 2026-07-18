"""
Mini-Shakespeare GPT — 3-model comparison table (Assignment Part 5b).

Merges Model A, Model B (from `outputs/generations.json`, produced by
`generate_comparisons.py`) and Gemini Flash (manually pasted into
`gemini_template.md`, since no Gemini API key is used anywhere here) into
one markdown table, one row per prompt.

Columns:
    - Model A / Model B / Gemini completions
    - Degenerate repetition loop: AUTO-DETECTED (repeated word n-grams) —
      an objective signal, not a judgment call.
    - Structural stability / Shakespearean styling accuracy: left as
      "TODO (manual)" for every model, including Gemini — these are
      subjective calls the assignment expects a human to make, and are
      never fabricated here.

If `gemini_template.md` still has empty/placeholder completions, that
prompt's Gemini cell is marked "PENDING" (not blank, not invented) and a
warning is printed.

Run from the `evaluation/` folder, after `generate_comparisons.py` and
after (optionally) filling in `gemini_template.md`:

    python comparison_table.py
"""

import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "outputs"
GEMINI_TEMPLATE_PATH = THIS_DIR / "gemini_template.md"
GENERATIONS_PATH = OUT_DIR / "generations.json"

PLACEHOLDER = "(PASTE COMPLETION HERE)"


def load_generations():
    with open(GENERATIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def parse_gemini_template(path):
    """Extract the fenced completion block under each '## Prompt N: "..."' heading.

    Returns a list of strings in prompt order, one per '## Prompt' section;
    an entry is "" if that section's block is empty or still the placeholder.
    """
    text = path.read_text(encoding="utf-8")

    sections = re.split(r"^## Prompt \d+:.*$", text, flags=re.MULTILINE)[1:]

    completions = []
    for section in sections:
        match = re.search(r"```\s*\n(.*?)\n```", section, flags=re.DOTALL)
        content = match.group(1).strip() if match else ""
        if content == PLACEHOLDER or not content:
            completions.append("")
        else:
            completions.append(content)
    return completions


def detect_repetition(text, n_min=3, n_max=6, min_repeats=3):
    """Objective degenerate-repetition signal: does any word n-gram (n=3..6)
    repeat at least `min_repeats` times anywhere in the text?
    """
    words = text.split()
    for n in range(n_min, n_max + 1):
        counts = {}
        for i in range(len(words) - n + 1):
            gram = tuple(words[i:i + n])
            counts[gram] = counts.get(gram, 0) + 1
        worst_gram, worst_count = max(counts.items(), key=lambda kv: kv[1], default=(None, 0))
        if worst_count >= min_repeats:
            return True, f'{n}-gram "{" ".join(worst_gram)}" repeated {worst_count}x'
    return False, "none detected"


def cell(text):
    """Markdown-table-safe cell: escape pipes, turn newlines into <br>."""
    if not text:
        return ""
    return text.replace("|", "\\|").replace("\n", "<br>")


def build_table(generations, gemini_completions, gemini_pending):
    lines = []
    lines.append("| Prompt | Model A completion | Model B completion | Gemini Flash completion | "
                  "Degenerate repetition loop (auto) | Structural stability (manual) | "
                  "Shakespearean styling accuracy (manual) |")
    lines.append("|---|---|---|---|---|---|---|")

    for i, entry in enumerate(generations):
        prompt = entry["prompt"]
        a_text = entry["model_a_completion"]
        b_text = entry["model_b_completion"]
        gemini_text = gemini_completions[i] if i < len(gemini_completions) else ""

        a_rep, a_detail = detect_repetition(a_text)
        b_rep, b_detail = detect_repetition(b_text)

        gemini_cell = "**PENDING**" if not gemini_text else cell(gemini_text)
        gemini_rep_note = ""
        if gemini_text:
            g_rep, g_detail = detect_repetition(gemini_text)
            gemini_rep_note = f"Gemini: {'YES — ' + g_detail if g_rep else 'no'}"

        rep_cell_parts = [
            f"A: {'YES — ' + a_detail if a_rep else 'no'}",
            f"B: {'YES — ' + b_detail if b_rep else 'no'}",
        ]
        if gemini_rep_note:
            rep_cell_parts.append(gemini_rep_note)
        rep_cell = "<br>".join(rep_cell_parts)

        lines.append(
            f"| {cell(prompt)} | {cell(a_text)} | {cell(b_text)} | {gemini_cell} | "
            f"{rep_cell} | TODO (manual) | TODO (manual) |"
        )

    return "\n".join(lines)


def main():
    if not GENERATIONS_PATH.exists():
        print(f"ERROR: {GENERATIONS_PATH} not found — run generate_comparisons.py first.")
        sys.exit(1)

    generations = load_generations()
    gemini_completions = parse_gemini_template(GEMINI_TEMPLATE_PATH)

    gemini_pending = [i for i, c in enumerate(gemini_completions) if not c]
    if gemini_pending or len(gemini_completions) < len(generations):
        print(f"WARNING: gemini_template.md is still empty/unfilled for "
              f"{len(gemini_pending) + max(0, len(generations) - len(gemini_completions))} "
              f"of {len(generations)} prompt(s). Those cells are marked PENDING, not fabricated. "
              f"Fill in gemini_template.md and re-run this script to complete the table.")

    table_md = build_table(generations, gemini_completions, gemini_pending)

    out_path = OUT_DIR / "comparison_table.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Model A vs Model B vs Gemini Flash — Comparison Table\n\n")
        f.write(table_md)
        f.write("\n")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
