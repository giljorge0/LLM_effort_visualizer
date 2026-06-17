#!/usr/bin/env python3
"""
train_predictor.py

Builds a small ground-truth dataset of (prompt -> reasoning tokens) by calling
the Anthropic API with extended thinking enabled, extracts the same surface
features used by the in-browser heuristic in thinking_effort_v2.html, and
fits a linear regression to predict reasoning effort from prompt text alone.

Usage:
    pip install requests scikit-learn --break-system-packages
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 train_predictor.py --prompts prompts.json --out results.json --weights weights.json

The output `weights.json` can be pasted directly into the "predictor weights"
box in thinking_effort_v2.html.

Notes on cost: each prompt makes one API call with extended thinking enabled.
With the default budget (4000 tokens) and ~55 starter prompts, this is on the
order of a few dollars depending on model pricing. Use --limit to test on a
small subset first, and --resume to avoid re-paying for prompts already done.
"""

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.request
import urllib.error

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Feature extraction — MUST match the JS implementation in
# thinking_effort_v2.html exactly, so fitted weights transfer directly.
# ---------------------------------------------------------------------------

LOGIC_WORDS = ['prove', 'derive', 'why', 'how', 'explain', 'analyze', 'analyse',
               'compare', 'calculate', 'solve', 'optimi', 'design', 'debug',
               'reasoning', 'algorithm', 'trade-off', 'tradeoff', 'implications',
               'justify', 'evaluate', 'generalis', 'generaliz']

CONSTRAINT_WORDS = ['must', 'only', 'except', 'exclude', 'without', 'all but',
                     'neither', 'unless', 'given that', 'at least', 'at most',
                     'no more than', 'cannot', 'not allowed']

AMBIGUITY_WORDS = ['could', 'might', 'depends', 'maybe', 'possibly', 'arguably', 'perhaps']

PIVOT_PHRASES = ['wait', 'actually', 'hold on', 'let me reconsider', 'on second thought',
                 'alternatively', 'hmm', 'no wait', 'let me re-check', 'let me recheck',
                 "i made an error", "that's wrong", "that is wrong", 'let me redo',
                 'scratch that', 'rethink', 'on reflection', 'let me double-check',
                 'let me double check', 'this is incorrect', 'i need to reconsider']

CODE_MARKER_RE = re.compile(r'```|function |def |class |import |for\(|for \(|while\(')


def count_occurrences(text, terms):
    lower = text.lower()
    count = 0
    for term in terms:
        count += len(re.findall(re.escape(term), lower))
    return count


def extract_features(prompt):
    words = prompt.split()
    word_count = len(words)
    question_marks = prompt.count('?')
    logic_words = count_occurrences(prompt, LOGIC_WORDS)
    constraint_words = count_occurrences(prompt, CONSTRAINT_WORDS)
    ambiguity = count_occurrences(prompt, AMBIGUITY_WORDS)
    numbered_items = len(re.findall(r'\b\d+[\.\)]', prompt))
    and_count = len(re.findall(r'\band\b', prompt.lower()))
    multipart = numbered_items + max(0, and_count - 1)
    code_markers = 1 if CODE_MARKER_RE.search(prompt) else 0
    numbers = len(re.findall(r'\d+', prompt))

    return {
        "log_words": math.log(word_count + 1),
        "question_marks": question_marks,
        "logic_words": logic_words,
        "constraint_words": constraint_words,
        "multipart": multipart,
        "ambiguity": ambiguity,
        "code_markers": code_markers,
        "numbers": numbers,
    }


def estimate_tokens_from_text(text):
    """Matches the JS estimate: word_count * 1.3"""
    if not text:
        return 0
    return round(len(text.split()) * 1.3)


def churn_score(thinking_text, think_tokens):
    pivots = count_occurrences(thinking_text, PIVOT_PHRASES)
    per100 = (pivots / think_tokens * 100) if think_tokens > 0 else 0
    return pivots, round(per100, 1)


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_claude(prompt, api_key, budget_tokens, max_retries=4):
    body = json.dumps({
        "model": MODEL,
        "max_tokens": budget_tokens + 2000,
        "thinking": {"type": "enabled", "budget_tokens": budget_tokens},
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            if e.code == 429 or e.code >= 500:
                wait = 2 ** attempt * 2
                print(f"  HTTP {e.code}, retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {e.code}: {err_body}")
    raise RuntimeError("Max retries exceeded")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", default="prompts.json", help="Input prompt dataset")
    parser.add_argument("--out", default="results.json", help="Output dataset with features + labels")
    parser.add_argument("--weights", default="weights.json", help="Output fitted predictor weights")
    parser.add_argument("--budget", type=int, default=4000, help="Thinking token budget per call")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N prompts")
    parser.add_argument("--resume", action="store_true", help="Skip prompts already in --out")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Set ANTHROPIC_API_KEY in your environment first.")

    with open(args.prompts) as f:
        prompts = json.load(f)

    if args.limit:
        prompts = prompts[:args.limit]

    results = []
    done_prompts = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
        done_prompts = {r["prompt"] for r in results}
        print(f"Resuming — {len(results)} prompts already done.")

    for i, item in enumerate(prompts):
        prompt = item["prompt"]
        category = item.get("category", "")

        if prompt in done_prompts:
            continue

        print(f"[{i+1}/{len(prompts)}] {category}: {prompt[:60]}...")

        try:
            data = call_claude(prompt, api_key, args.budget)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            continue

        if "error" in data:
            print(f"  API error: {data['error']}", file=sys.stderr)
            continue

        content = data.get("content", [])
        thinking_block = next((b for b in content if b.get("type") == "thinking"), None)
        text_block = next((b for b in content if b.get("type") == "text"), None)

        thinking_text = thinking_block.get("thinking", "") if thinking_block else ""
        answer_text = text_block.get("text", "") if text_block else ""

        think_tokens = estimate_tokens_from_text(thinking_text)
        out_tokens = estimate_tokens_from_text(answer_text)
        pivots, churn_per100 = churn_score(thinking_text, think_tokens)
        features = extract_features(prompt)

        results.append({
            "prompt": prompt,
            "category": category,
            "features": features,
            "think_tokens": think_tokens,
            "out_tokens": out_tokens,
            "pivots": pivots,
            "churn_per100": churn_per100,
        })

        # checkpoint after every call so --resume works and partial progress isn't lost
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

        time.sleep(0.5)  # be polite to the API

    print(f"\nCollected {len(results)} labeled examples -> {args.out}")

    if len(results) < 5:
        sys.exit("Need at least 5 examples to fit a regression. Run with more prompts.")

    fit_and_save_weights(results, args.weights)


def fit_and_save_weights(results, out_path):
    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score
        import numpy as np
    except ImportError:
        sys.exit("Install scikit-learn: pip install scikit-learn --break-system-packages")

    feature_keys = ["log_words", "question_marks", "logic_words", "constraint_words",
                     "multipart", "ambiguity", "code_markers", "numbers"]

    X = np.array([[r["features"][k] for k in feature_keys] for r in results])
    y = np.array([r["think_tokens"] for r in results])

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    scores = cross_val_score(model, X, y, cv=min(5, len(results)), scoring="r2")
    print(f"\nCross-validated R^2: {scores.mean():.3f} (+/- {scores.std():.3f})")
    print("Note: a low or negative R^2 here is itself a finding — it would mean")
    print("surface prompt features don't reliably predict reasoning depth.")

    weights = {"bias": float(model.intercept_)}
    for k, coef in zip(feature_keys, model.coef_):
        weights[k] = float(coef)

    with open(out_path, "w") as f:
        json.dump(weights, f, indent=2)

    print(f"\nFitted weights written to {out_path}")
    print(json.dumps(weights, indent=2))

    # Quick deceptive-simplicity scan using the fitted model itself
    preds = model.predict(X)
    print("\n--- Deceptive-simplicity candidates (predicted low, actual high) ---")
    flagged = False
    for r, pred, actual in zip(results, preds, y):
        if pred < 250 and actual > pred * 3 and actual > 600:
            flagged = True
            print(f"  predicted {pred:.0f} vs actual {actual}: {r['prompt'][:70]}")
    if not flagged:
        print("  (none found in this dataset)")

    print("\n--- Highest cognitive churn (pivots per 100 reasoning tokens) ---")
    top_churn = sorted(results, key=lambda r: r["churn_per100"], reverse=True)[:5]
    for r in top_churn:
        print(f"  {r['churn_per100']:.1f} per 100 tok ({r['pivots']} pivots): {r['prompt'][:70]}")


if __name__ == "__main__":
    main()
