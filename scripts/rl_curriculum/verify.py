"""Prove the issue #110 step-3 oracle contract, then emit the manifests.

For every task the oracle must accept its reference and reject:
  * the starter the policy is handed,
  * the intended failure mutation,
  * a distinct compiling semantic error.

Also enforces lineage disjointness from the held-out `circular-buffer` target:
no fixed26 family name and no symbol from the Exercism circular_buffer API may
appear in any prompt, starter, or oracle.

    python3 scripts/rl_curriculum/verify.py
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import episodes
import harness

OUT = harness.REPO / "data" / "rl-curriculum" / "out"

FIXED26 = [
    "all-your-base", "allergies", "bank-account", "binary-search-tree", "circular-buffer",
    "clock", "complex-numbers", "crypto-square", "diamond", "dnd-character", "gigasecond",
    "grade-school", "kindergarten-garden", "knapsack", "linked-list", "meetup",
    "parallel-letter-frequency", "perfect-numbers", "phone-number", "queen-attack",
    "robot-name", "space-age", "spiral-matrix", "sublist", "yacht", "zebra-puzzle",
]

# Symbols owned by the held-out target's public API. Reusing any of them would
# make the curriculum a renamed copy rather than a disjoint domain.
TARGET_SYMBOLS = ["circular_buffer", "overwrite(", "domain_error"]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def prompt_for(task: str, meta: dict) -> str:
    files = sorted(meta["files"], key=lambda n: (not n.endswith(".cpp"), n))
    return f"""# Instructions

{meta['blurb']}

## Build environment

- Compiled as C++17 with `-Wall -Wextra -Wpedantic -Werror`, so any warning fails the build.
- Only `{meta['slug']}.h` and `{meta['slug']}.cpp` are editable. The test file is fixed and hidden.


Use the above instructions to modify the supplied files: {' '.join(files)}
Don't change the names of existing functions or classes, as they may be referenced from other code like unit tests, etc.
Only use standard libraries, don't suggest installing any packages."""


def main() -> int:
    tasks = harness.load_tasks()
    compiler = harness.compiler_id()
    print(f"compiler: {compiler}\n")

    failures: list[str] = []
    records = []

    for task in sorted(tasks):
        meta = tasks[task]
        reference = harness.reference_files(task)

        stage, _ = harness.evaluate(task, reference)
        if stage != harness.PASS:
            failures.append(f"{task}: oracle REJECTS its own reference ({stage})")
            continue
        print(f"{task}: oracle accepts reference")

        ctrl = episodes.CONTROLS[task]
        stage, _ = harness.evaluate(task, ctrl.apply(reference))
        control_ok = stage in harness.REJECTIONS
        if not control_ok:
            failures.append(f"{ctrl.episode_id}: distinct semantic control NOT rejected")
        print(f"  control {ctrl.episode_id}: {'rejected' if control_ok else 'ACCEPTED'} ({stage})")

        for ep in [e for e in episodes.EPISODES if e.task == task]:
            try:
                starter = ep.apply(reference)
            except episodes.InjectionError as exc:
                failures.append(str(exc))
                continue
            stage, _ = harness.evaluate(task, starter)
            ok = stage in harness.REJECTIONS
            if not ok:
                failures.append(f"{ep.episode_id}: starter NOT rejected by oracle")
            print(f"  tier {ep.tier} {ep.episode_id:<44} {'rejected' if ok else 'ACCEPTED'} ({stage})")
            records.append({
                "episode_id": ep.episode_id,
                "task": task,
                "tier": ep.tier,
                "invariant": ep.invariant,
                "intent": ep.intent,
                "prompt_sha256": sha(prompt_for(task, meta)),
                "starter_sha256": {n: sha(b) for n, b in sorted(starter.items())},
                "reference_sha256": {n: sha(b) for n, b in sorted(reference.items())},
                "oracle_sha256": sha((harness.TASKS / task / "test" / meta["test"]).read_text()),
                "starter_rejected_as": stage,
                "control_episode": ctrl.episode_id,
                "control_rejected": control_ok,
            })

    # Lineage disjointness across everything the policy can see.
    blob = json.dumps([
        prompt_for(t, tasks[t]) for t in tasks
    ] + [
        (harness.TASKS / t / "test" / tasks[t]["test"]).read_text() for t in tasks
    ] + [
        b for t in tasks for b in harness.reference_files(t).values()
    ])
    name_hits = sorted({n for n in FIXED26 if n in blob or n.replace("-", "_") in blob})
    sym_hits = sorted({s for s in TARGET_SYMBOLS if s in blob})
    if name_hits:
        failures.append(f"lineage: fixed26 names present: {name_hits}")
    if sym_hits:
        failures.append(f"lineage: held-out target symbols present: {sym_hits}")

    print(f"\nlineage: fixed26 names {name_hits or 'none'}; target symbols {sym_hits or 'none'}")

    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f in failures:
            print("  ", f)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "episodes.json").write_text(json.dumps(records, indent=2) + "\n")
    manifest = {
        "curriculum": "rl-curriculum-v1",
        "issue": 110,
        "purpose": "circular-state pilot ladder; held-out target circular-buffer",
        "domains": sorted(tasks),
        "episodes": len(records),
        "tiers": {str(t): sum(1 for r in records if r["tier"] == t) for t in (1, 2, 3)},
        "compiler": compiler,
        "compile_flags": harness.FLAGS,
        "eval_compiler_for_comparison": "GNU 13.3.0 (observed in archived fixed26 run logs)",
        "step3_contract": {
            "oracle_accepts_reference": True,
            "oracle_rejects_starter": True,
            "oracle_rejects_intended_mutation": True,
            "oracle_rejects_distinct_semantic_error": True,
        },
        "lineage_disjoint_from_fixed26": True,
        "lineage_disjoint_from_target_api": True,
        "episodes_sha256": sha((OUT / "episodes.json").read_text()),
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nSTEP-3 CONTRACT SATISFIED — {len(records)} episodes")
    print(json.dumps(manifest["tiers"], indent=None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
