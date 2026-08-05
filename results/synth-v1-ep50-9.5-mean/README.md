# Synth v1, epoch 50

## Results

- Pass@1 trials: `10, 9, 10, 9`
- Mean Pass@1: **9.5/26**
- Pass@1 variability: **SD 0.58; range 9-10; 95% CI 5.75-13.5**
- Multi turn with feedback (turn=2) trials: `11, 13, 12, 12`
- Mean multi turn with feedback (turn=2): **12/26**
- Multi turn variability: **SD 0.82; range 11-13; 95% CI 8.25-15.75**
- Conditional turn-2 recovery: **10/66 (15.2%; 95% CI 7.4-24.4%)**
- Samples: **104**
- Statistics: [summary](../statistics.md) · [per-task frequencies](../per_task_success.csv)
- Reproducibility bundle: [TokenBender/glm47-synth-v1-reproducibility](https://huggingface.co/TokenBender/glm47-synth-v1-reproducibility/tree/17a73a21f38512f2af9a2bf8b41e283dc31430d1)

## How to reproduce

```bash
cd reproduction
./run.sh
```
