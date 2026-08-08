# Synth v1 epoch 50: failure-informed answer-withheld fmt PR 3727 pass

## Result

Trial `a16` produced an applicable patch on its first model turn from the untouched fmt base commit `06f1c0d725855861535e9e65cd4d502aca7c61ed`. A fresh official `mswebench/fmtlib_m_fmt:pr-3727` container built the candidate and passed all 20 top-level tests.

```text
100% tests passed, 0 tests failed out of 20
first_turn_passed: true
final_passed: true
```

Candidate patch SHA-256: `6b376eb22063ba36f718ac7505326b07dd6aae94d88c5b8bf05331a07574982c`.

## Claim boundary

This is an answer-withheld but failure-informed result.

- The original frozen prompt arm (`a1`–`a8`) produced 0 first-turn passes in 8 attempts.
- After explicit user direction to iterate on non-solution context, the failure-informed prompt-tuning arm (`a9`–`a16`) produced 1 first-turn pass in 8 attempts.
- Trial `a16` started from the untouched repository and used no repair turn.
- The accepted PR patch, hidden test patch, and grader source were not supplied to the model.
- Later prompts did use earlier candidates' compiler/test failures and source-derived architecture.
- The 16 attempts are sequential and non-IID; this is not a pass@16 estimate.
- The issue and PR are public, so pretraining contamination cannot be ruled out.

See `protocol/contract-amendment.json` and `protocol/final-prompt-leak-audit.json`.

## Evidence map

- `manifest.json`: checkpoint, protocol, result, and claim boundary.
- `protocol/task-prompt.md`: exact prompt used by passing trial `a16`.
- `protocol/task-prompt-v*.md`: complete prompt-tuning lineage.
- `protocol/prompt-revision-v*.json`: prompt hashes and failure-derived revision reasons.
- `protocol/contract.json`: original frozen protocol.
- `protocol/contract-amendment.json`: user-authorized failure-informed arm.
- `protocol/final-prompt-leak-audit.json`: offline accepted-patch overlap audit.
- `protocol/{base-model,workspace,lora-activation}-receipt.json`: model and adapter identity evidence.
- `trials/a16/generation-receipt.json`: exact inference configuration and response hash.
- `trials/a16/raw-response-turn1.txt`: model response.
- `trials/a16/grades/turn1/candidate.patch`: applied candidate patch.
- `trials/a16/grades/turn1/grader.log`: fresh official build and 20/20 test log.
- `trials/a16/grades/turn1/grade-receipt.json`: immutable grader hashes.
- `trials/a1`–`trials/a15`: preserved non-workspace failure lineage.
- `checksums.sha256`: bundle integrity index.

Large reproducible assets are intentionally excluded: base-model shards, converted serving-adapter tensors, and copied workspaces.
