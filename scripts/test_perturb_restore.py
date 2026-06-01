#!/usr/bin/env python3
"""
Smoke-test for the randopt_server /perturb and /restore endpoints.

Steps:
  1. Baseline inference (unperturbed)
  2. /perturb  with a fixed seed
  3. Perturbed inference
  4. /restore  with the same seed
  5. Post-restore inference  ← should match baseline

Run:
  python3 scripts/test_perturb_restore.py [--base-url http://localhost:8000]
"""

import argparse
import json
import sys
import requests

# ---------------------------------------------------------------------------
MODEL_NAME = "qwen-qwen25-32b-instruct"
PROMPT = "What is 2 + 2? Answer with just the number."
SEED = 12345
SIGMA = 0.001
MAX_TOKENS = 16
# ---------------------------------------------------------------------------


def chat(base_url: str, prompt: str) -> str:
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def perturb(base_url: str, seed: int, sigma: float) -> None:
    resp = requests.post(
        f"{base_url}/perturb",
        json={"seed": seed, "sigma": sigma},
        timeout=120,
    )
    resp.raise_for_status()
    print(f"  /perturb response: {resp.json()}")


def restore(base_url: str, seed: int, sigma: float) -> None:
    resp = requests.post(
        f"{base_url}/restore",
        json={"seed": seed, "sigma": sigma},
        timeout=120,
    )
    resp.raise_for_status()
    print(f"  /restore response: {resp.json()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    url = args.base_url.rstrip("/")

    print(f"Target: {url}")
    print(f"Prompt: {PROMPT!r}\n")

    print("Step 1 — baseline (unperturbed)")
    baseline = chat(url, PROMPT)
    print(f"  response: {baseline!r}\n")

    print(f"Step 2 — /perturb seed={SEED} sigma={SIGMA}")
    perturb(url, SEED, SIGMA)

    print("Step 3 — perturbed inference")
    perturbed = chat(url, PROMPT)
    print(f"  response: {perturbed!r}\n")

    print(f"Step 4 — /restore seed={SEED} sigma={SIGMA}")
    restore(url, SEED, SIGMA)

    print("Step 5 — post-restore inference")
    restored = chat(url, PROMPT)
    print(f"  response: {restored!r}\n")

    # -----------------------------------------------------------------------
    print("=" * 60)
    print(f"baseline  : {baseline!r}")
    print(f"perturbed : {perturbed!r}  {'(different ✓)' if perturbed != baseline else '(same — perturbation had no visible effect)'}")
    print(f"restored  : {restored!r}  {'(matches baseline ✓)' if restored == baseline else '(MISMATCH — restore may be imperfect)'}")

    if restored != baseline:
        print("\nWARNING: restored output differs from baseline.")
        print("This can be expected with fp16 rounding; check logits rather than greedy tokens for a stronger test.")
        sys.exit(1)


if __name__ == "__main__":
    main()
