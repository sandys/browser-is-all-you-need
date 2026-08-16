# Synth-v1 Failure Coverage

- Step 1: complete — [hard gate](01-evidence-admissibility.md) +
  [atomic ledger](atomic-ledger.jsonl)
- Ledger: 286 task-trial rows, 370 tested turns; 273 causal-grade records and
  13 outcome-only records
- SHA-256: `248d7cb4f959c3b74b3bbfa57bd8d235c441264034b1c32a366e51a3409e4514`
  (supersedes `d0517317…` — 2026-08-16 provenance migration of the 26
  healthcheck rows from Modal refs to
  `TokenBender/glm47-synth-v1-fixed26-evals@2056a4c1`; see the migration note
  in [01-evidence-admissibility.md](01-evidence-admissibility.md))
- Step 2: complete — [mechanism coverage](mechanism-coverage.jsonl)
- Exact SFT corpus: `TokenBender/glm47-synth-v1-dataset@c586446fd309a1c2488b2953f77f3f370a73913c`,
  260 rows / 26 families; train SHA-256
  `3472d76169e52bd0859c181d63de24a060c4c7f2d3d8a004ceb6090498f1ddc1`
- Coverage: 149 mechanisms / 246 causal failed turns — 75 / 143
  covered-not-learned, 66 / 88 partial, 8 / 15 uncovered, 0 unresolved
- Feedback gap: no SFT row contains compiler/test feedback followed by repair;
  among 62 failed feedback turns, 57 are partial and 5 uncovered, with 0
  directly covered
- Uncovered turns: 13 inference-time context-budget terminations, which are not
  SFT-actionable, plus 2 complex-number stream-insertion dependency failures
- Coverage SHA-256: `e25156ca6553bbbe710725eb54ac39d7a4befaed16ee1e099a5c62cad76e3945`
- Headroom: across the 5 ep50 trials, mean pass@1 is 9.2/26 but the union is
  17/26, and mean final 12.8/26 against a union of 21/26 — only 1 task passes on
  turn 1 in every trial and 9 never do, so ~77% of task families would yield
  mixed GRPO groups at 8 samples
- The healthcheck 16/26 vs archived 12/26 gap does not survive: conditional
  recovery rate is confounded by which tasks land in its denominator, and the
  unconditional comparison is 16/26 vs 48/104 at p=0.19. Report final pass rate,
  never recovery rate
- Both findings above regenerate from the committed ledger:
  `python3 docs/worklogs/synth-v1-failure-coverage/analysis.py`
