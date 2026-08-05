# GLM-4.7-Flash base

## Results

- Pass@1: **0.5/26 mean** (`0`, `1`, `1`, `0`)
- Pass@1 variability: **SD 0.58; range 0-1; 95% CI 0-1.25**
- Multi turn with feedback (turn=2): **4.5/26 mean** (`4`, `5`, `6`, `3`)
- Multi turn variability: **SD 1.29; range 3-6; 95% CI 2.25-6.75**
- Conditional turn-2 recovery: **16/102 (15.7%; 95% CI 7.8-24.8%)**
- Trials: **4**
- Samples: **104**
- Statistics: [summary](../statistics.md) · [per-task frequencies](../per_task_success.csv)

## How to reproduce

```bash
cd reproduction
export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=local-eval
./run.sh
```
