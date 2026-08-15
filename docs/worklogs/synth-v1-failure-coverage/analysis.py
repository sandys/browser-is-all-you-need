#!/usr/bin/env python3
"""Regenerate the headroom and variance findings from atomic-ledger.jsonl.

    python3 docs/worklogs/synth-v1-failure-coverage/analysis.py

Two questions, both answered from the ledger alone:

  headroom  Is GRPO sharpening latent capability or building new capability?
  variance  Is the healthcheck trial's 16/26 multi-turn a real effect?
"""

from __future__ import annotations

import collections
import json
import pathlib
import statistics
from math import comb

LEDGER = pathlib.Path(__file__).resolve().parent / "atomic-ledger.jsonl"
ARCHIVED = ["a1", "a2", "a3", "a4"]
HEALTHCHECK = "healthcheck-20260815"


def fisher(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test on a 2x2 table."""
    n = a + b + c + d
    observed = comb(a + b, a) * comb(c + d, c) / comb(n, a + c)
    total = 0.0
    for i in range(min(a + b, a + c) + 1):
        k = a + c - i
        if k < 0 or k > c + d:
            continue
        p = comb(a + b, i) * comb(c + d, k) / comb(n, a + c)
        if p <= observed + 1e-12:
            total += p
    return total


def load():
    rows = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    return [r for r in rows if r["checkpoint_epoch"] == 50]


def headroom(ep50) -> None:
    trials = ARCHIVED + [HEALTHCHECK]
    tasks = sorted({r["task_id"] for r in ep50})
    first = {(r["trial"], r["task_id"]): r["turns"][0]["outcome"] == "pass" for r in ep50}
    final = {(r["trial"], r["task_id"]): r["turns"][-1]["outcome"] == "pass" for r in ep50}

    print("== headroom ==")
    print(f"{'task':<28} {'turn1':<7} {'final':<7}")
    for task in tasks:
        t1 = "".join("P" if first.get((t, task)) else "." for t in trials)
        tf = "".join("P" if final.get((t, task)) else "." for t in trials)
        print(f"  {task:<26} {t1:<7} {tf:<7}")

    for label, table in (("pass@1", first), ("final ", final)):
        scores = [sum(table.get((t, k), False) for k in tasks) for t in trials]
        union = sum(any(table.get((t, k)) for t in trials) for k in tasks)
        never = sum(not any(table.get((t, k)) for t in trials) for k in tasks)
        always = sum(all(table.get((t, k), False) for t in trials) for k in tasks)
        mean = statistics.mean(scores)
        print(
            f"\n{label}: per-trial {scores}  mean {mean:.2f}/26  union {union}/26"
            f"  headroom {union - mean:.2f}  always {always}  never {never}"
        )

    # A GRPO group yields zero advantage when all samples agree. Shrink the
    # observed rate so 0/5 and 5/5 are not asserted as probability 0 and 1.
    useful = 0.0
    for task in tasks:
        passes = sum(1 for t in trials if first.get((t, task)))
        p = (passes + 0.5) / (len(trials) + 1.0)
        useful += 1 - (p**8 + (1 - p) ** 8)
    print(f"\ntask families yielding mixed GRPO groups at 8 samples: {useful:.1f}/26 ({useful / 26:.0%})")


def variance(ep50) -> None:
    print("\n== variance ==")
    causal = [r for r in ep50 if r["evidence_grade"] == "causal" and len(r["turns"]) > 1]

    # Does context exhaustion on the feedback turn explain non-recovery?
    tab = collections.Counter()
    for r in causal:
        if r["turns"][0]["outcome"] != "fail":
            continue
        t2 = r["turns"][1]
        tab[(bool(t2.get("context_exhaustion")), t2["outcome"] == "pass")] += 1
    a, b = tab[(True, True)], tab[(True, False)]
    c, d = tab[(False, True)], tab[(False, False)]
    print(f"  turn-2 exhausted recovers {a}/{a + b} ({a / max(1, a + b):.0%});"
          f" clean recovers {c}/{c + d} ({c / max(1, c + d):.0%});"
          f" Fisher p={fisher(a, b, c, d):.3f}")

    def counts(trials, index):
        rows = [r for r in ep50 if r["trial"] in trials]
        passed = sum(1 for r in rows if r["turns"][index]["outcome"] == "pass")
        return passed, len(rows) - passed

    def recovery(trials):
        rows = [r for r in ep50 if r["trial"] in trials and len(r["turns"]) > 1
                and r["turns"][0]["outcome"] == "fail"]
        return sum(1 for r in rows if r["turns"][-1]["outcome"] == "pass"), len(rows)

    for label, index in (("pass@1", 0), ("final ", -1)):
        ha, hb = counts([HEALTHCHECK], index)
        aa, ab = counts(ARCHIVED, index)
        print(f"  unconditional {label}: healthcheck {ha}/{ha + hb}  archived {aa}/{aa + ab}"
              f"  Fisher p={fisher(ha, hb, aa, ab):.3f}")

    hr, hf = recovery([HEALTHCHECK])
    ar, af = recovery(ARCHIVED)
    print(f"  conditional recovery: healthcheck {hr}/{hf}  archived {ar}/{af}"
          f"  Fisher p={fisher(hr, hf - hr, ar, af - ar):.3f}   <-- confounded")

    # The denominator is itself a random draw: tasks the archived trials never
    # failed on turn 1 become near-free recoveries when healthcheck does fail them.
    failed_t1 = {r["task_id"] for r in ep50 if r["trial"] in ARCHIVED
                 and r["turns"][0]["outcome"] == "fail"}
    stripped = [r for r in ep50 if r["trial"] == HEALTHCHECK and len(r["turns"]) > 1
                and r["turns"][0]["outcome"] == "fail" and r["task_id"] in failed_t1]
    sr = sum(1 for r in stripped if r["turns"][-1]["outcome"] == "pass")
    free = sorted({r["task_id"] for r in ep50 if r["trial"] == HEALTHCHECK
                   and r["turns"][0]["outcome"] == "fail"} - failed_t1)
    print(f"  archived never failed turn 1 on: {free}")
    print(f"  recovery excluding those: healthcheck {sr}/{len(stripped)}  archived {ar}/{af}"
          f"  Fisher p={fisher(sr, len(stripped) - sr, ar, af - ar):.3f}")
    print("\n  Verdict: no evidence of a config difference. Report final pass rate,\n"
          "  not recovery rate -- its denominator moves between trials.")


if __name__ == "__main__":
    rows = load()
    headroom(rows)
    variance(rows)
