#!/usr/bin/env python3
"""Issue #110 T1 — r3 rollout forensics.

Regenerates every number in README.md from the committed ledgers:

    python3 docs/worklogs/issue110-r3-rollout-forensics/analysis.py

Inputs (sha256-pinned below, all derived from the durable GCS run prefix
issue110-bankacct-grpo-r3-20260816t090200z):
  gate_records.jsonl  — 320 admission-gate reward records (verbatim GCS copy)
  eval_records.jsonl  — 64 reward records extracted from rollout_dumps/grpo_eval_0.pt
  train_records.jsonl — 256 reward records extracted from rollout_dumps/grpo_0.pt
  gate.json           — gate verdict (verbatim GCS copy)
  heldout-result.json — fixed26 bank-account single-attempt result (verbatim GCS copy)

Source .pt provenance:
  rollout_dumps/grpo_0.pt      sha256 87c14287e8f32320a007c3808929f4e1cb59b130b7a28a2c1ceca4d0e2713d3e
  rollout_dumps/grpo_eval_0.pt sha256 7b61d970cbda8b4b2fa2658caa07e53171d49887ed28216be358997fddb354b8
Extraction: torch.load(...)["samples"][i]["reward"], sorted by (problem_id,
sample_index), one JSON object per line with sorted keys.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent

PINNED = {
    "gate_records.jsonl": "1974605dfdcba46aaaf19ff55871576ea89c73e544a396d7d719be110685b963",
    "eval_records.jsonl": "69196f542080c6cbafe18a301111dc7410ddbe53eab2ed38c977665fa4b9ce77",
    "train_records.jsonl": "6719132967d351efb9aab65ec2e8b219e2554d79befed8e25438fa290f6e9bbb",
    "gate.json": "1f925127ed0edbbf8ede6e300b18fdb78841ef0390562602d68eb7159cf3d2bf",
    "heldout-result.json": "490321133a6e64cfda43af48ee2f47a55b4cc02c48bc0026013e0c70ecc7149a",
}

EAGAIN_MARK = "Resource temporarily unavailable"
KINDS = ("full-solve", "build-link-repair", "state-repair", "feedback-repair")


def check_hashes() -> None:
    for name, expected in PINNED.items():
        digest = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        if digest != expected:
            sys.exit(f"hash mismatch for {name}: {digest}")
    print(f"input hashes verified: {len(PINNED)} files\n")


def load_jsonl(name: str) -> list[dict]:
    with open(HERE / name) as f:
        return [json.loads(line) for line in f]


def classify(r: dict) -> str:
    """Failure mechanism, refined by whether an assertion failed before EAGAIN."""
    if not r["format_valid"]:
        return "format"
    if r["compile_error"]:
        return "compile"
    if r["all_tests_pass"]:
        return "pass"
    test_log = r.get("logs", {}).get("test", "") or ""
    eagain = EAGAIN_MARK in test_log or "system_error" in test_log
    asserted = "FAILED:" in test_log
    if eagain and not asserted:
        return "infra-robbed"  # zero failed assertions, killed by thread-spawn EAGAIN
    if eagain:
        return "semantic+eagain"  # already failing before the EAGAIN abort
    if r["timeout"]:
        return "timeout"
    return "semantic"


def kind_of(r: dict) -> str:
    return r["problem_id"].split("--")[1]


def table(counter: Counter, total: int) -> str:
    return "  ".join(f"{k}={v} ({v / total:.1%})" for k, v in counter.most_common())


def main() -> None:
    check_hashes()
    gate_records = load_jsonl("gate_records.jsonl")
    eval_records = load_jsonl("eval_records.jsonl")
    train_records = load_jsonl("train_records.jsonl")
    gate = json.load(open(HERE / "gate.json"))
    heldout = json.load(open(HERE / "heldout-result.json"))
    assert (len(gate_records), len(eval_records), len(train_records)) == (320, 64, 256)

    # ---- F2: EAGAIN prevalence in the admission gate -------------------------
    cls = Counter(classify(r) for r in gate_records)
    print("== Gate (320 no-update rollouts): failure mechanism ==")
    print(table(cls, 320))
    eagain_all = cls["infra-robbed"] + cls["semantic+eagain"]
    test_stage = 320 - cls["compile"] - cls["format"]
    assert (eagain_all, cls["infra-robbed"], cls["semantic+eagain"]) == (121, 80, 41)
    print(
        f"EAGAIN deaths: {eagain_all}/320 = {eagain_all / 320:.1%} of all rollouts, "
        f"{eagain_all}/{test_stage} = {eagain_all / test_stage:.1%} of test-stage rollouts"
    )
    print(
        f"infra-robbed (zero failed assertions before EAGAIN): {cls['infra-robbed']}; "
        f"already-failing before EAGAIN: {cls['semantic+eagain']}"
    )
    rewards = Counter(r["reward"] for r in gate_records if classify(r) == "infra-robbed")
    print(f"reward assigned to infra-robbed rollouts: {dict(rewards)} (compile failure = -0.5)\n")

    # ---- F3: both halves of hard gate 2 were vacuous -------------------------
    print("== Gate-2 vacuity ==")
    named = sum(
        1
        for r in gate_records
        if "FAILED: concurrent_transactions" in (r.get("logs", {}).get("test") or "")
    )
    print(
        f"detector hits for 'FAILED: concurrent_transactions': {named} "
        f"(gate reported observed_failures={gate['concurrency_load_check']['observed_failures']}) "
        f"vs {eagain_all} EAGAIN deaths"
    )
    loads = [r["reward_worker_load"] for r in gate_records]
    at_max = sum(1 for x in loads if x >= 32)
    print(
        f"load median={sorted(loads)[len(loads) // 2]}, records at load>=32: {at_max}/320; "
        f"gate bucketed high_load_records={gate['concurrency_load_check']['high_load_records']} "
        "(median split is degenerate when the modal load is the max)"
    )
    stage = [r for r in gate_records if classify(r) in ("pass", "infra-robbed", "semantic+eagain", "semantic", "timeout")]
    hi = [r for r in stage if r["reward_worker_load"] >= 32]
    lo = [r for r in stage if r["reward_worker_load"] < 32]
    for label, part in (("load=32", hi), ("load<32", lo)):
        e = sum(1 for r in part if classify(r) in ("infra-robbed", "semantic+eagain"))
        print(f"  {label}: EAGAIN {e}/{len(part)} = {e / len(part):.1%} of test-stage rollouts")
    print("  -> not graded by recorded load: hard thread/pid ceiling, not load-proportional flake\n")

    # ---- Gate counterfactual: does admission survive EAGAIN invalidation? ----
    groups: dict[str, list[str]] = defaultdict(list)
    for r in gate_records:
        groups[r["problem_id"]].append(classify(r))
    mixed_reported = sum(1 for cs in groups.values() if 0 < cs.count("pass") < len(cs))
    valid_mixed = 0
    for cs in groups.values():
        kept = [c for c in cs if c != "infra-robbed"]
        if kept and 0 < kept.count("pass") < len(kept):
            valid_mixed += 1
    print("== Gate counterfactual (infra-robbed treated as infrastructure-invalid) ==")
    print(
        f"mixed groups reported: {mixed_reported}/40 = {mixed_reported / 40:.1%}; "
        f"correctness-mixed after exclusion: {valid_mixed}/40 = {valid_mixed / 40:.1%} "
        "(threshold 30% -> admission still passes)\n"
    )

    # ---- F1: eval_0 is the pre-update policy ---------------------------------
    print("== eval dump identity ==")
    gate_val = sorted(
        (r for r in gate_records if r["split"] == "validation"),
        key=lambda r: (r["problem_id"], r["sample_index"]),
    )
    ev = sorted(eval_records, key=lambda r: (r["problem_id"], r["sample_index"]))
    same = sum(1 for a, b in zip(gate_val, ev) if a["response"] == b["response"])
    digest = lambda rows: hashlib.sha256("\n".join(r["response"] for r in rows).encode()).hexdigest()[:16]
    assert same == 64 and digest(gate_val) == digest(ev)
    print(
        f"identical responses gate-validation vs grpo_eval_0: {same}/64 "
        f"(joint sha256 prefix {digest(ev)})"
    )
    print(
        "-> the pass@1=0.0625 table in the issue is the FROZEN pre-update policy "
        "(eval at rollout 0); no post-update validation exists in the artifacts\n"
    )

    # ---- F5: per-kind reality -------------------------------------------------
    print("== Per-kind mechanism table (gate rollouts; n per cell: train 64, validation 16) ==")
    per = defaultdict(Counter)
    for r in gate_records:
        per[(r["split"], kind_of(r))][classify(r)] += 1
    header = f"{'split':11s} {'kind':18s}" + "".join(
        f"{c:>15s}" for c in ("pass", "infra-robbed", "compile", "semantic*", "format")
    )
    print(header)
    for split in ("train", "validation"):
        for k in KINDS:
            c = per[(split, k)]
            sem = c["semantic"] + c["semantic+eagain"]
            row = [c["pass"], c["infra-robbed"], c["compile"], sem, c["format"]]
            print(f"{split:11s} {k:18s}" + "".join(f"{v:>15d}" for v in row))
    print("(*semantic includes rollouts that failed an assertion and then hit EAGAIN)\n")

    for split, total in (("train", 256), ("validation", 64)):
        rows = [r for r in gate_records if r["split"] == split]
        p = sum(1 for r in rows if classify(r) == "pass")
        robbed = sum(1 for r in rows if classify(r) == "infra-robbed")
        print(
            f"{split}: observed pass {p}/{total} = {p / total:.1%}; "
            f"infra-clean rate {p}/{total - robbed} = {p / (total - robbed):.1%}; "
            f"upper bound if all robbed pass: {(p + robbed) / total:.1%}"
        )
    print()

    # ---- F4: training-batch contamination ------------------------------------
    print("== Training batch (256 samples, the single optimizer update) ==")
    tcls = Counter(classify(r) for r in train_records)
    print(table(tcls, 256))
    print(f"reward values: {dict(Counter(r['reward'] for r in train_records))}")
    tg: dict[str, list[dict]] = defaultdict(list)
    for r in train_records:
        tg[r["problem_id"]].append(r)
    uniform = contaminated = noise_only = 0
    for rows in tg.values():
        if len({r["reward"] for r in rows}) == 1:
            uniform += 1
            continue
        cs = [classify(r) for r in rows]
        if "infra-robbed" in cs:
            contaminated += 1
        kept = [r["reward"] for r, c in zip(rows, cs) if c != "infra-robbed"]
        if len(set(kept)) <= 1:
            noise_only += 1
    print(
        f"groups: {len(tg)}; uniform-reward (no gradient): {uniform}; "
        f"variance-bearing: {len(tg) - uniform}, of which containing infra-robbed: {contaminated}, "
        f"variance ONLY from infra-robbed: {noise_only}"
    )
    robbed_train = tcls["infra-robbed"]
    print(
        f"infra-robbed samples fed to the optimizer: {robbed_train}/256 = {robbed_train / 256:.1%} "
        "(each scored 0.0 -> negative advantage against in-group passes)\n"
    )

    # ---- F6: heldout ----------------------------------------------------------
    print("== Held-out fixed26 attempt ==")
    print(
        f"pass@1={heldout['pass_at_1']} failure_class={heldout['failure_class']} "
        f"infra_valid={heldout['infra_valid']}"
    )
    print(f"summary: {heldout['failure_summary']}")
    header_text = (HERE / "heldout-bank_account.h").read_text()
    assert "int balance();" in header_text and "int balance;" in header_text
    print(
        "receipt heldout-bank_account.h shows method 'int balance();' and member "
        "'int balance;' in one class — legal in Java, illegal in C++. SFT variants "
        "use accessor!=member naming (e.g. funds()/value_), so the protective "
        "convention did not transfer through the single near-null update."
    )


if __name__ == "__main__":
    main()
