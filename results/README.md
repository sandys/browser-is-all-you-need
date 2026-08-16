# Fixed26 results

## Results

| Result | Pass@1 | Multi turn with feedback (turn=2) | Trials | Samples |
| --- | ---: | ---: | ---: | ---: |
| [GLM-4.7-Flash base](base-fixed26-20260711/) | 0.5/26 mean | 4.5/26 mean | 4 | 104 |
| [Synth v1, epoch 50](synth-v1-ep50-9.5-mean/) | 9.5/26 mean | 12/26 mean | 4 | 104 |
| [SFT v5, Aider-format](sft-v5-aiderfmt-1117-4trials/) | 6/26 mean | 10.25/26 mean | 4 | 104 |
| [Luna](luna-fixed26-20260805/) | 6.25/26 mean | 16.75/26 mean | 4 | 104 |

| Result | Pass@1 SD, range, 95% CI (out of 26) | Multi turn with feedback (turn=2) SD, range, 95% CI (out of 26) | Conditional turn-2 recovery |
| --- | --- | --- | ---: |
| GLM-4.7-Flash base | 0.58; 0-1; 0-1.25 | 1.29; 3-6; 2.25-6.75 | 16/102 (15.7%; CI 7.8-24.8%) |
| Synth v1, epoch 50 | 0.58; 9-10; 5.75-13.5 | 0.82; 11-13; 8.25-15.75 | 10/66 (15.2%; CI 7.4-24.4%) |
| SFT v5, Aider-format | 1.63; 4-8; 3-9.25 | 1.71; 8-12; 7-13.75 | 17/80 (21.2%; CI 11.9-31.6%) |
| Luna | 1.89; 5-9; 3.25-9.5 | 1.5; 15-18; 12.75-20.5 | 42/79 (53.2%; CI 36.8-70%) |

Statistics: [method and summary](statistics.md) · [per-task frequencies](per_task_success.csv) · [recompute](compute_statistics.py)

## Focused upstream evaluations

- [Synth v1 epoch 50: failure-informed answer-withheld fmt PR 3727 pass](fmt-pr3727-synth-v1-ep50-answer-free-first-turn-pass/) — trial `a16` passed 20/20 official tests on its first model turn from the untouched base. Original frozen arm: 0/8; subsequent failure-informed prompt-tuning arm: 1/8.
- [Synth v1 epoch 50: exact-reference fmt PR 3727 reproduction](fmt-pr3727-synth-v1-ep50-first-turn-pass/) — separate success-first evaluation that disclosed the accepted implementation.

## How to reproduce

```bash
cd results/base-fixed26-20260711/reproduction
export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=local-eval
./run.sh
```

```bash
cd results/synth-v1-ep50-9.5-mean/reproduction
./run.sh
```

```bash
cd results/sft-v5-aiderfmt-1117-4trials/reproduction
./run.sh
```

```bash
cd results/luna-fixed26-20260805/reproduction
export OPENROUTER_API_KEY=...
./run.sh
```
