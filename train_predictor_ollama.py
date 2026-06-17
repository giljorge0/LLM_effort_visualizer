#!/usr/bin/env python3
"""
train_predictor_ollama.py

Builds a labeled dataset of (prompt -> generation_time_ms, tokens/sec, churn)
by calling a local Ollama instance, then fits a linear regression to predict
generation time from prompt surface features alone.

Usage:
    pip install requests scikit-learn --break-system-packages
    python3 train_predictor_ollama.py --prompts prompts.json --out results.json --weights weights.json

Output weights.json can be pasted into the "predictor weights (advanced)" box
in thinking_effort_ollama.html.

Options:
    --host        Ollama base URL (default: http://localhost:11434)
    --model       Model to use (default: mistral)
    --max-tokens  Max tokens per response (default: 256 — keep low for speed)
    --limit N     Only run first N prompts (test mode)
    --resume      Skip prompts already in --out
    --out         Output labeled dataset JSON
    --weights     Output fitted weights JSON
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

# ---------------------------------------------------------------------------
# Feature extraction — MUST match JS in thinking_effort_ollama.html exactly
# ---------------------------------------------------------------------------

LOGIC_WORDS = ['prove','derive','why','how','explain','analyze','analyse','compare',
               'calculate','solve','optimi','design','debug','reasoning','algorithm',
               'trade-off','tradeoff','implications','justify','evaluate','generalis','generaliz']
CONSTRAINT_WORDS = ['must','only','except','exclude','without','all but','neither',
                    'unless','given that','at least','at most','no more than','cannot','not allowed']
AMBIGUITY_WORDS = ['could','might','depends','maybe','possibly','arguably','perhaps']
PIVOT_PHRASES = ['wait','actually','hold on','let me reconsider','on second thought',
                 'alternatively','hmm','no wait','let me re-check','let me recheck',
                 "i made an error","that's wrong","that is wrong",'let me redo',
                 'scratch that','rethink','on reflection','let me double-check',
                 'let me double check','this is incorrect','i need to reconsider']
CODE_MARKER_RE = re.compile(r'```|function |def |class |import |for\(|for \(|while\(')


def count_occ(text, terms):
    lower = text.lower()
    return sum(len(re.findall(re.escape(t), lower)) for t in terms)


def extract_features(prompt):
    words = prompt.split()
    and_count = len(re.findall(r'\band\b', prompt.lower()))
    numbered = len(re.findall(r'\b\d+[\.\)]', prompt))
    return {
        "log_words":        math.log(len(words) + 1),
        "question_marks":   prompt.count('?'),
        "logic_words":      count_occ(prompt, LOGIC_WORDS),
        "constraint_words": count_occ(prompt, CONSTRAINT_WORDS),
        "multipart":        numbered + max(0, and_count - 1),
        "ambiguity":        count_occ(prompt, AMBIGUITY_WORDS),
        "code_markers":     1 if CODE_MARKER_RE.search(prompt) else 0,
        "numbers":          len(re.findall(r'\d+', prompt)),
    }


def churn_score(text, out_tokens):
    pivots = count_occ(text, PIVOT_PHRASES)
    per100 = (pivots / out_tokens * 100) if out_tokens > 0 else 0.0
    return pivots, round(per100, 1)


# ---------------------------------------------------------------------------
# Ollama call — non-streaming so we get the full stats block cleanly
# ---------------------------------------------------------------------------

def call_ollama(prompt, host, model, max_tokens, retries=3):
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.7}
    }).encode("utf-8")

    req = urllib.request.Request(
        host.rstrip('/') + '/api/generate',
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  connection error, retrying in {wait}s: {e}", file=sys.stderr)
                time.sleep(wait)
            else:
                raise RuntimeError(f"Could not reach Ollama at {host}: {e}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host",       default="http://localhost:11434")
    ap.add_argument("--model",      default="mistral")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="Keep low (256) for speed during dataset collection")
    ap.add_argument("--prompts",    default="prompts.json")
    ap.add_argument("--out",        default="results_ollama.json")
    ap.add_argument("--weights",    default="weights_ollama.json")
    ap.add_argument("--limit",      type=int, default=None)
    ap.add_argument("--resume",     action="store_true")
    args = ap.parse_args()

    # verify connection
    try:
        r = urllib.request.urlopen(args.host.rstrip('/') + '/api/tags', timeout=5)
        tags = json.loads(r.read().decode())
        names = [m['name'] for m in tags.get('models', [])]
        print(f"Connected to Ollama. Available models: {', '.join(names) or '(none)'}")
        if args.model not in names and not any(n.startswith(args.model) for n in names):
            print(f"Warning: '{args.model}' not found in model list above.", file=sys.stderr)
    except Exception as e:
        sys.exit(f"Cannot reach Ollama at {args.host}: {e}\nIs Ollama running? Try: ollama serve")

    with open(args.prompts) as f:
        prompts = json.load(f)

    if args.limit:
        prompts = prompts[:args.limit]

    results = []
    done = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
        done = {r["prompt"] for r in results}
        print(f"Resuming — {len(results)} already done.")

    feature_keys = ["log_words","question_marks","logic_words","constraint_words",
                    "multipart","ambiguity","code_markers","numbers"]

    for i, item in enumerate(prompts):
        prompt   = item["prompt"]
        category = item.get("category", "")

        if prompt in done:
            continue

        print(f"[{i+1}/{len(prompts)}] {category}: {prompt[:65]}...")

        t0 = time.time()
        try:
            data = call_ollama(prompt, args.host, args.model, args.max_tokens)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            continue
        wall_ms = (time.time() - t0) * 1000

        # Ollama returns durations in nanoseconds
        eval_duration_ns    = data.get("eval_duration", 0)
        prompt_eval_ns      = data.get("prompt_eval_duration", 0)
        total_duration_ns   = data.get("total_duration", 0)
        eval_count          = data.get("eval_count", 0)
        prompt_eval_count   = data.get("prompt_eval_count", 0)
        response_text       = data.get("response", "")

        gen_ms = eval_duration_ns / 1e6 if eval_duration_ns > 0 else wall_ms
        tps    = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else None
        pivots, churn_per100 = churn_score(response_text, eval_count or 1)
        features = extract_features(prompt)

        tps_str = f"{tps:.1f}" if tps else "?"
        print(f"  gen_ms={gen_ms:.0f}  tps={tps_str}  "
              f"out_tokens={eval_count}  churn={churn_per100}")

        results.append({
            "prompt":           prompt,
            "category":         category,
            "features":         features,
            "gen_ms":           gen_ms,
            "wall_ms":          wall_ms,
            "eval_count":       eval_count,
            "prompt_eval_count": prompt_eval_count,
            "tps":              tps,
            "pivots":           pivots,
            "churn_per100":     churn_per100,
            "response_snippet": response_text[:200],
        })

        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDataset: {len(results)} examples -> {args.out}")

    if len(results) < 5:
        sys.exit("Need at least 5 examples to fit. Add more prompts.")

    fit_and_save(results, feature_keys, args.weights)


def fit_and_save(results, feature_keys, out_path):
    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score
        import numpy as np
    except ImportError:
        sys.exit("pip install scikit-learn --break-system-packages")

    X = np.array([[r["features"][k] for k in feature_keys] for r in results])
    y = np.array([r["gen_ms"] for r in results])

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    n_splits = min(5, len(results))
    if n_splits >= 2:
        scores = cross_val_score(model, X, y, cv=n_splits, scoring="r2")
        print(f"\nCross-validated R² (predicting gen_ms): {scores.mean():.3f} ± {scores.std():.3f}")
        print("Low R² = surface features don't predict generation time = the finding itself.")
    else:
        model.fit(X, y)

    weights = {"bias": float(model.intercept_)}
    for k, c in zip(feature_keys, model.coef_):
        weights[k] = float(c)

    with open(out_path, "w") as f:
        json.dump(weights, f, indent=2)

    print(f"\nWeights -> {out_path}")
    print(json.dumps(weights, indent=2))

    preds = model.predict(X)
    print("\n--- Deceptive-simplicity (predicted fast, actually slow) ---")
    flagged = False
    for r, pred, actual in zip(results, preds, y):
        if pred < 1500 and actual > pred * 2.5 and actual > 3000:
            flagged = True
            print(f"  pred {pred:.0f}ms vs actual {actual:.0f}ms: {r['prompt'][:70]}")
    if not flagged:
        print("  (none in this dataset)")

    print("\n--- Highest churn (backtracking per 100 output tokens) ---")
    for r in sorted(results, key=lambda r: r["churn_per100"], reverse=True)[:5]:
        print(f"  {r['churn_per100']:.1f}/100tok ({r['pivots']} pivots): {r['prompt'][:70]}")


if __name__ == "__main__":
    main()
