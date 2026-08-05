# Luna

## Results

- Pass@1 trials: `6, 5, 5, 9`
- Mean Pass@1: **6.25/26**
- Pass@1 variability: **SD 1.89; range 5-9; 95% CI 3.25-9.5**
- Multi turn with feedback (turn=2) trials: `18, 16, 15, 18`
- Mean multi turn with feedback (turn=2): **16.75/26**
- Multi turn variability: **SD 1.5; range 15-18; 95% CI 12.75-20.5**
- Conditional turn-2 recovery: **42/79 (53.2%; 95% CI 36.8-70%)**
- Samples: **104**
- Statistics: [summary](../statistics.md) · [per-task frequencies](../per_task_success.csv)

## How to reproduce

```bash
cd reproduction
export OPENROUTER_API_KEY=...
./run.sh
```
