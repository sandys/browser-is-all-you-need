# Step 1 — Hard Evidence Gate

Accepted records are frozen in [atomic-ledger.jsonl](atomic-ledger.jsonl),
SHA-256 `248d7cb4f959c3b74b3bbfa57bd8d235c441264034b1c32a366e51a3409e4514`.

> Provenance migration (2026-08-16): supersedes SHA-256
> `d0517317f6074a27c31bd75afd39aa842cb57efec7a44993c7ea0fe6107f72cd`. The only
> change is the `source_ref` field of the 26 healthcheck-20260815 rows, which
> pointed at a private Modal volume and at Modal function logs. The 13 shard-0
> artifact sets were hash-verified against the ledger, archived to
> `TokenBender/glm47-synth-v1-fixed26-evals@2056a4c1691911d642ee4bdb31d21e840238906f`
> under `groups/epoch-50-healthcheck-20260815/attempts/a1/`, re-downloaded from
> that revision, and re-verified 13/13 before the refs were rewritten. The 13
> shard-1 rows were always outcome-only; their Modal function logs are no
> longer retrievable, so their refs now point at the archived group manifest,
> which records each task's original `modal-log://` id and evidence status. No
> outcome, grade, hash, or note field changed; `analysis.py` output is
> byte-identical across the migration.

## Hard accept

A task record is causal evidence only when:

- checkpoint, adapter, dataset, harness, prompt contract, decoding, and turn
  identity are exact;
- a terminal external test result exists;
- the tested model action and diagnostic or feedback are recoverable; and
- no malformed response, timeout, harness failure, or ambiguous tested state
  explains the result.

Accepted causal evidence:

| Checkpoint | Records | Result |
|---|---:|---|
| Epoch 5 | 52 | 6/26, 4/26; single turn |
| Epoch 50 archived | 104 | 38/104 first turn; 48/104 by turn 2 |
| Epoch 50 current shard 0 | 13 | 5/13 first turn; 7/13 by turn 2 |
| Epoch 100 | 104 | 10/26, 7/26, 10/26, 9/26; single turn |
| **Total** | **273** | **causal-grade** |

Current epoch-50 shard 1 contributes 13 additional outcome-only records:
3/13 first turn and 9/13 by turn 2. Its structured task artifacts did not
persist, so it may corroborate scores but not establish causes.

## Hard no

Do not use as causal evidence:

- aggregate scores without task artifacts;
- malformed, timed-out, unexecuted, harness-failed, or ambiguously tested attempts;
- context-exhausted calls unless the terminal tested candidate is unambiguous;
- current epoch-50 shard 1 beyond outcome-level corroboration;
- `a4/grade-school` as emitted-code evidence; it is only a no-action failure;
- nonexistent epoch-5 trials a3/a4 or unevaluated epochs 10–45 and 55–95;
- Synth-v2, Synth-v3, SFT-v5, base, Luna, or unrelated adapters; or
- fixed26 results as pristine benchmark evidence: fixed26 is an assisted,
  test-derived contract.
