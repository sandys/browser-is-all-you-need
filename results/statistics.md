# Fixed26 statistics

| Result | Pass@1 trials | Pass@1 mean, SD, range, 95% CI (out of 26) | Multi turn with feedback (turn=2) trials | Multi turn mean, SD, range, 95% CI (out of 26) | Conditional turn-2 recovery |
| --- | --- | --- | --- | --- | --- |
| GLM-4.7-Flash base | 0, 1, 1, 0 | 0.5/26; SD 0.58; range 0-1; CI 0-1.25 | 4, 5, 6, 3 | 4.5/26; SD 1.29; range 3-6; CI 2.25-6.75 | 16/102 (15.7%); CI 7.8-24.8% |
| Synth v1, epoch 50 | 10, 9, 10, 9 | 9.5/26; SD 0.58; range 9-10; CI 5.75-13.5 | 11, 13, 12, 12 | 12/26; SD 0.82; range 11-13; CI 8.25-15.75 | 10/66 (15.2%); CI 7.4-24.4% |
| SFT v5, Aider-format | 8, 4, 6, 6 | 6/26; SD 1.63; range 4-8; CI 3-9.25 | 12, 10, 11, 8 | 10.25/26; SD 1.71; range 8-12; CI 7-13.75 | 17/80 (21.2%); CI 11.9-31.6% |
| Luna | 6, 5, 5, 9 | 6.25/26; SD 1.89; range 5-9; CI 3.25-9.5 | 18, 16, 15, 18 | 16.75/26; SD 1.5; range 15-18; CI 12.75-20.5 | 42/79 (53.2%); CI 36.8-70% |

The 95% intervals use 100,000 deterministic task-clustered bootstrap replicates (seed 20260805). Each replicate resamples the 26 task IDs and preserves all four outcomes for each selected task. The 104 task-trial records are not treated as independent Bernoulli observations.

Per-task frequencies are in [per_task_success.csv](per_task_success.csv). Run `python3 results/compute_statistics.py` from the repository root to recompute all outputs.
