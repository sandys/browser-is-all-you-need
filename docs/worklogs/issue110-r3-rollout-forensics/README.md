# Issue #110 T1 — r3 rollout forensics

Row-level audit of the `issue110-bankacct-grpo-r3-20260816t090200z` run: the
no-update admission gate, the single-update training batch, the "post-update"
internal validation, and the held-out fixed26 attempt. Zero GPU spend.

Every number below regenerates with one command:

```
python3 docs/worklogs/issue110-r3-rollout-forensics/analysis.py
```

## Inputs

| file | rows | sha256 |
|---|---:|---|
| `gate_records.jsonl` (verbatim GCS `bank_account_rollout/records.jsonl`) | 320 | `1974605dfdcba46aaaf19ff55871576ea89c73e544a396d7d719be110685b963` |
| `eval_records.jsonl` (extracted from `rollout_dumps/grpo_eval_0.pt`) | 64 | `69196f542080c6cbafe18a301111dc7410ddbe53eab2ed38c977665fa4b9ce77` |
| `train_records.jsonl` (extracted from `rollout_dumps/grpo_0.pt`) | 256 | `6719132967d351efb9aab65ec2e8b219e2554d79befed8e25438fa290f6e9bbb` |
| `gate.json` (verbatim GCS copy) | — | `1f925127ed0edbbf8ede6e300b18fdb78841ef0390562602d68eb7159cf3d2bf` |
| `heldout-result.json` (verbatim GCS copy) | — | `490321133a6e64cfda43af48ee2f47a55b4cc02c48bc0026013e0c70ecc7149a` |
| `heldout-bank_account.h` (verbatim GCS copy) | — | `fa725df6f59798eaf88ed2ea23267a5a6a37161faa11529be1ef6b48376e87fc` |

Source `.pt` provenance (durable in the GCS run prefix):
`grpo_0.pt` sha256 `87c14287e8f32320a007c3808929f4e1cb59b130b7a28a2c1ceca4d0e2713d3e`,
`grpo_eval_0.pt` sha256 `7b61d970cbda8b4b2fa2658caa07e53171d49887ed28216be358997fddb354b8`.
Extraction: `torch.load(f)["samples"][i]["reward"]`, sorted by
`(problem_id, sample_index)`, one sorted-key JSON object per line.

## F1 — The "internal validation after the update" was never post-update

All 64 responses in `grpo_eval_0.pt` are byte-identical to the admission gate's
validation slice (64/64, joint response sha256 prefix `5a0676a3e968e690`;
`rollout_id=0`, `weight_versions=['default']`). The pass@1 = 0.0625 table in the
issue therefore describes the **frozen epoch-50 policy before the optimizer
update**. No post-update validation rollouts exist in the durable artifacts.
"Did the single update move any episode kind" is **unmeasured**; the only
post-update evaluation of any sort is the held-out 0/1.

## F2 — Sandbox thread exhaustion was scored as model failure

121/320 gate rollouts (37.8%; **62.1% of the 195 that reached the test
stage**) aborted with pthread-create EAGAIN inside the hidden oracle's
1,000-thread block:

```
terminate called after throwing an instance of 'std::system_error'
  what():  Resource temporarily unavailable
```

The oracle prints a `FAILED: <assertion>` line for each failing check before
its summary. **80 of the 121 EAGAIN deaths contain zero FAILED lines** — every
functional assertion had passed when the sandbox killed them ("infra-robbed").
The other 41 had already failed an assertion (38× `restart_resets_zero`) and
are genuine semantic failures with a polluted exit.

All 121 were classified `tests_failed` with reward **0.0** — *above* a compile
failure's −0.5 — because `_is_infrastructure_error`
(`src/glm47_posttraining/aider_polyglot/harness.py:40`) matches only four
Docker-daemon strings. This violates the issue contract: "sandbox failure …
infrastructure-invalid and produce no model gradient."

**Correction (post-forensics):** the ceiling is the repo's own
`--pids-limit 128` in `docker_base_args`
(`src/glm47_posttraining/cpp_perf/sandbox.py`), applied to every docker
sandbox stage. A 1,000-thread oracle under a 128-pid budget survives only when
thread exit outpaces thread spawn — a scheduler race, which is why the failure
was flaky (~60% of test-stage rollouts) rather than deterministic, and why the
recorded worker load did not grade it. The CPU proof passed because it ran the
local bubblewrap backend, which sets no pid cap. An earlier revision of this
report blamed the training image's container runtime; that was wrong — the
grep that "found no rlimits" did not cover `src/glm47_posttraining/cpp_perf/`.

## F3 — Both halves of hard gate 2 were vacuous

- The concurrency detector (`scripts/rl_curriculum/check_bank_account_rollout.py:87`)
  greps for `FAILED: concurrent_transactions` — an assertion an EAGAIN abort
  never lets print. Hits in the gate records: **0**, against 121 EAGAIN deaths.
- The load check median-splits `reward_worker_load`; the median is the modal
  maximum (32, with 204/320 records at it) and the "high" bucket
  (`load > median`) is empty: `high_load_records=0`, so no correlation was
  computable even in principle.
- In the raw data EAGAIN is not graded by recorded load (54.1% of test-stage
  rollouts at load 32 vs 79.0% below): a hard thread/pid ceiling, not a
  load-proportional flake, so capping verifier parallelism alone is not the fix.

Counterfactual: treating infra-robbed records as infrastructure-invalid, the
correctness-bearing mixed-group fraction is **20/40 = 50%** (reported: 57.5%) —
still above the 30% bar. The curriculum's reward contrast is real; admission
was not manufactured by the flake.

## F4 — The single optimizer update trained on contaminated advantages

The training batch (`grpo_0.pt`, 256 samples, 32 groups) contains **61
infra-robbed samples (23.8%)**. 22 of the 30 variance-bearing groups contain at
least one; in 4 groups the reward variance comes *only* from robbed-vs-rest.
Inside any group with a genuine pass (reward 1.0), a robbed sample's 0.0
becomes a negative advantage on likely-correct code, and the reward ordering
EAGAIN(0.0) > compile(−0.5) rewards dying at the thread block over failing to
compile. The output adapter `662b04d9…` is one update of this gradient:
nothing about it is trustworthy enough to build on, and nothing was measured
after it (F1).

## F5 — Per-kind reality (pre-update, gate rollouts)

| split | kind | pass | infra-robbed | compile | semantic* | format |
|---|---|---:|---:|---:|---:|---:|
| train | full-solve | 15 | 14 | 20 | 10 | 4 |
| train | build-link-repair | 7 | 13 | 34 | 3 | 7 |
| train | state-repair | 2 | 3 | 0 | 41 | 18 |
| train | feedback-repair | 20 | 31 | 10 | 1 | 2 |
| validation | full-solve | 4 | 8 | 2 | 0 | 2 |
| validation | build-link-repair | 0 | 2 | 14 | 0 | 0 |
| validation | state-repair | 0 | 4 | 0 | 11 | 1 |
| validation | feedback-repair | 0 | 5 | 10 | 0 | 1 |

(n = 64 per train cell row, 16 per validation cell row; *semantic includes
assertion failures that later hit EAGAIN.)

The **0.0625 anomaly dissolves into three named causes**, each row-backed:

1. **Mislabeled eval (F1):** the number never measured the updated policy.
2. **EAGAIN robbery (F2):** validation observed 4/64 = 6.2%; infra-clean rate
   4/45 = 8.9%; upper bound counting robbed as passes 23/64 = 35.9%. Train:
   observed 17.2%, bounds 22.6–41.0%. Feedback-repair train is the extreme
   case: 20 passes + 31 robbed of 64 → potentially ~80% true pass rate.
3. **Repair kinds are genuinely hard; memorization only shows on full-solve.**
   Validation full-solve is 4 passes + 8 robbed of 16 (up to 75%), consistent
   with SFT memorization. But build-link-repair is 0/16 with 14 stuck at
   compile (the planted build defect unfixed), and state-repair is 0/16 with 11
   semantic failures on exactly `restart_resets_zero` (the planted stale-state
   defect). Knowing the solution file does not equal fixing a planted defect in
   it. Train state-repair confirms: 2/64 pass, 41 semantic — the curriculum's
   hardest kind by far.

## F6 — Held-out failure is a Java-idiom intrusion, not negative transfer

The fixed26 attempt (infra-valid, `heldout-result.json`) died on a compile
error kept through three Aider reflections: `int balance();` (required API)
next to `int balance;` (chosen member) in one class — legal in Java, illegal in
C++. The SFT variants consistently use accessor≠member naming (`funds()` /
`value_`, `credits()` / `value_`), which is *protective* against exactly this
collision. The habit did not transfer through the single near-null update; the
RL batch did not cause the failure. See `heldout-bank_account.h`.

## Consequences for T2/T5

1. **Before any further GPU spend:** classify pthread-EAGAIN aborts
   (`std::system_error` + rc 134) as `SandboxInfrastructureError` in
   `harness.py`, raise and pin the sandbox pids budget for the 1,000-thread
   oracle, and fix both gate-2 defects (EAGAIN-aware detector, non-degenerate
   load split). *Implemented in the infra-fix commit accompanying this report;
   see `tests/`.*
2. **Add a reward-path canary:** prove the sandbox can hold the oracle's peak
   thread count before any rollout batch. *Implemented: the sandbox preflight
   now holds 1,000 concurrent threads and fails closed.*
3. **Rerun the admission gate after the fix** — true pass rates roughly double,
   so the mixture/difficulty picture (and the 57.5% figure) changes.
4. **Warm-start the T5 multi-update run from the epoch-50 adapter, not
   `662b04d9…`** (F4: one update of 23.8%-contaminated gradient, post-update
   state unmeasured).
5. **T2 as declared measures `662b04d9…`** — a near-null, noise-trained
   adapter. The held-out eval path itself was infra-valid, so T2 is not blocked
   by the sandbox fix, but consider re-pointing its N=4 at the epoch-50
   baseline (thickening 2/5 into a 9-trial baseline) or deferring the powered
   eval to the post-T5 checkpoint.

## T1 acceptance mapping

- Committed one-command script regenerating every number, inputs pinned by
  sha256: `analysis.py` (this directory).
- Per-kind table: F5. Pre/post comparison: **impossible — no post-update
  rollouts exist (F1)**, which is itself the finding, with row-level evidence.
- Named cause of 0.0625: three causes in F5, each row-backed.
- GPU spend: zero (all inputs pulled from the existing GCS prefix).
