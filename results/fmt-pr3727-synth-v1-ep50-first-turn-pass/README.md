# Synth v1 epoch 50 — fmt PR 3727

## Result

A fresh first-turn candidate from the published **Synth v1 epoch 50** checkpoint passed the official `mswebench/fmtlib_m_fmt:pr-3727` evaluator:

- Build: **passed**
- CTest: **20/20 passed**
- Repair turn: **not used**
- Passing trial: `trials/exact-a2`
- Candidate patch SHA-256: `2e00944c8f75ab95ae095b45c2e89132f057757a3058bb62522d2154219cbf57`
- Grader log SHA-256: `d4edf46151466d2256802bae05728dde373420fde51dac34cdf9d88494edc563`

This was a **success-first assisted implementation test**, not an answer-blind discovery benchmark. The model received the complete pre-fix header, exact behavioral requirements, and the authoritative public production diff. The passing candidate is not byte-identical to the upstream reference, but it passed the hidden regression patch and the full 20-target suite.

## Configuration

- Base: `zai-org/GLM-4.7-Flash@7dd20894a642a0aa287e9827cb1a1f7f91386b67`
- Adapter: `TokenBender/glm47-synth-v1-100ep@a8d386d6b5bedf761cbe4effe2fdc36a97ad62bc`
- Checkpoint: iteration `649`, LoRA rank `16`
- Aider: commit `5dc9490bb35f9729ef2c95d00a19ccd30c26339c`, whole-file format, repo map disabled
- Sampling: thinking enabled, temperature `0.7`, top-p `1.0`, maximum completion `32,768`
- Model prompt: `model-prompt.txt`, SHA-256 `76f562cb819409a8e63da114eec41e4856d19ac87d548bd0cb96b271b235c2e4`

## Evidence map

- Frozen contract: `contract.json`
- Machine-readable summary: `manifest.json`
- Exact model input: `model-prompt.txt`
- Task packet: `task-prompt.md`
- Public production reference: `production-reference.patch`
- Adapter activation proof: `lora-activation-receipt.json`
- Base identity: `base-model-receipt.json`
- Final run receipt: `trials/exact-a2/run-receipt.json`
- Candidate patch: `trials/exact-a2/grades/turn1/candidate.patch`
- Complete grader output: `trials/exact-a2/grades/turn1/grader.log`
- Failed first-turn attempts: `trials/a1`, `trials/guided-a1`, and `trials/exact-a1`
- Checksums: `checksums.sha256`

Large reproducible assets are intentionally excluded: the 62.4 GB base snapshot, the 454 MB converted serving adapter, and disposable workspace copies. Their immutable identities and conversion procedures are preserved.

## Worklog

- Master: [tokenbender/browser-is-all-you-need#95](https://github.com/tokenbender/browser-is-all-you-need/issues/95)
- Experiment: [tokenbender/browser-is-all-you-need#96](https://github.com/tokenbender/browser-is-all-you-need/issues/96)
