# Adaptive, Task-Specific C++ Rubric Reward

## Purpose

This reward is an execution-derived scalar for GRPO. It is deliberately not a
second learning algorithm. Task category chooses a fixed rubric template,
analysis supplies one risk per active rubric, and the scorer returns one
weighted sum.

The implementation is split between:

- `src/glm47_posttraining/cpp_perf/rubrics.py`: templates, normalization,
  weighted sum, pass-rate extraction, and EMA updates.
- `src/glm47_posttraining/cpp_perf/reward.py`: response parsing and rubric
  scores.
- `src/glm47_posttraining/integrations/miles_polyglot_cpp.py`: task-specific
  risk loading and flattened reward records.
- `scripts/update_rubric_risks.py`: evaluation-epoch risk update.

## 1. Static task categories

Category selection does not change during training.

| Category | Active rubrics |
| --- | --- |
| `standard` | Correctness, Reasoning, Memory Safety, C++ Quality |
| `state_concurrency` | Correctness, Reasoning, Memory Safety, Thread Safety, C++ Quality |
| `performance_intensive` | Correctness, Reasoning, Memory Safety, Runtime, C++ Quality |

Each task record contains `rubric_category`. Older records with the topic
`State & concurrency` are resolved to `state_concurrency` for compatibility.
An inactive rubric has no weight and cannot affect the reward.

## 2. Risk normalization

For active rubric set `A`, each task has non-negative risks `r_i`:

\[
w_i = \frac{r_i}{\sum_{j \in A} r_j}
\]

Rules:

1. Only active rubrics are normalized.
2. If `rubric_risks` is empty, all active rubrics receive uniform weights.
3. If at least one risk is supplied, an omitted active rubric receives zero
   risk and therefore zero weight.
4. Negative risks and unknown rubric names are rejected by schema validation.
5. If all selected risks are zero, the scorer safely falls back to uniform
   weights.

The final reward is exactly:

\[
R = \sum_{i \in A} w_i S_i
\]

There is no difficulty multiplier, learned reward model, cross-rubric
optimizer, or direct use of the underlying analysis parameters.

## 3. Rubric scores

### 3.1 Format gate and Reasoning

The response must contain exactly one `<reasoning>...</reasoning>` block,
followed by exactly one fenced C++ block containing every required editable
file marker.

| Outcome | Reasoning score | Terminal behavior |
| --- | ---: | --- |
| Invalid response or multi-file structure | -1.0 | Hard reward override of -1.0; sandbox is not run |
| Rollout reports context exhaustion | -0.5 | Weighted reward from available rubric scores; sandbox is not run |
| Code is repeated inside reasoning | -0.5 | Continue to executable rubrics |
| Reasoning exceeds 1,000 tokens | -0.5 | Continue to executable rubrics |
| Reasoning is below 400 tokens | +0.1 | Continue |
| Reasoning is 400–1,000 tokens | 0.0 | Continue |

An exact rollout-provided reasoning-token count is preferred. A deterministic
lexical count is used when the rollout API does not provide one.

### 3.2 Correctness

Correctness comes from Catch2 test-case and assertion counts.

| Outcome | Correctness score |
| --- | ---: |
| All cases and all assertions pass | +1.0 |
| Partial result | `-0.2 + 0.7 × (0.5 × case_pass_ratio + 0.5 × assertion_pass_ratio)` |
| Execution timeout | -0.5 |
| Compilation prevents testing | 0.0 |

The partial branch ranges from -0.2 toward, but not including, +0.5. A task is
counted as passing the correctness rubric for EMA purposes only at score 1.0.

### 3.3 Memory Safety

| Outcome | Memory-safety score |
| --- | ---: |
| Normal or sanitizer compilation failure | -0.5 |
| ASan, UBSan, or leak-sanitizer diagnostic | -0.5 |
| Sanitized execution completes cleanly | +0.1 |
| Sanitizer is not reached | 0.0 |

Sanitizer evidence comes from executing an instrumented binary. Merely
compiling with sanitizer flags is not considered a pass.

### 3.4 C++ Quality

This is a conservative static delta against the supplied starter files.

| Outcome | C++-quality score |
| --- | ---: |
| Candidate introduces heavy standard containers passed by value | -0.1 |
| Candidate introduces raw `new` or `delete` | -0.1 |
| No new configured anti-pattern is introduced | +0.1 |

Comparing candidate and starter counts avoids penalizing an API signature that
the task requires. This rubric is a small heuristic, not a full semantic C++
style checker.

### 3.5 Thread Safety

This rubric is active only for `state_concurrency` tasks.

| Outcome | Thread-safety score |
| --- | ---: |
| TSan test execution is clean | +0.1 |
| TSan reports a race or concurrency defect | -0.5 |
| TSan execution times out or deadlocks | -0.5 |
| TSan did not actually run | 0.0 |

The scorer does not award thread-safety credit for finding `std::mutex`,
`std::atomic`, or similar keywords. A state task without executed TSan evidence
is marked `correct_thread_sanitizer_not_run` and is not reported as fully
verified by the Miles/evaluation adapter.

### 3.6 Runtime

This rubric is active only for `performance_intensive` tasks.

| Outcome | Runtime score |
| --- | ---: |
| Measured candidate CPU time is below measured reference CPU time | +0.1 |
| Candidate is not faster | 0.0 |
| Either measurement is unavailable | 0.0 |

The generic Catch2 harness exposes candidate/reference CPU-time fields, but
does not invent them. A dataset-specific runner may populate those fields using
a controlled benchmark protocol. Wall-clock time is never substituted for
missing CPU-time evidence.

## 4. Category examples

### Standard task with uniform risks

Active scores for a concise, fully correct, sanitizer-clean, idiomatic result:

| Rubric | Weight | Score | Contribution |
| --- | ---: | ---: | ---: |
| Correctness | 0.25 | 1.0 | 0.250 |
| Reasoning | 0.25 | 0.1 | 0.025 |
| Memory Safety | 0.25 | 0.1 | 0.025 |
| C++ Quality | 0.25 | 0.1 | 0.025 |
| **Reward** | | | **0.325** |

### State task with task-specific risks

Suppose risks are correctness 0.55, thread safety 0.30, reasoning 0.10, memory
safety 0.05, and C++ quality 0.00. They already sum to one. A candidate scoring
1.0, 0.0, 0.1, 0.1, and 0.1 respectively receives:

\[
R = 0.55(1.0) + 0.30(0.0) + 0.10(0.1) + 0.05(0.1) + 0(0.1) = 0.565
\]

A clean TSan score raises this reward by `0.30 × 0.1 = 0.03`.

### Performance task

If correctness and runtime each have weight 0.5, a correct candidate with a
verified runtime improvement receives `0.5(1.0) + 0.5(0.1) = 0.55`. Without
runtime evidence it receives 0.5.

## 5. Evaluation-epoch adaptation

For each active rubric, evaluation records are converted to a pass rate. The
failure rate is:

\[
Failure_i = 1 - PassRate_i
\]

Risk is updated once after an evaluation epoch:

\[
r_i^{new} = \alpha r_i^{old} + (1-\alpha)Failure_i,\quad \alpha=0.8
\]

Rubric pass thresholds are: correctness 1.0; reasoning, memory safety, thread
safety, runtime, and C++ quality 0.1. Missing active evidence does not become a
pass.

Run:

```bash
PYTHONPATH=src python3 scripts/update_rubric_risks.py \
  path/to/task_taxonomy.json \
  path/to/evaluation.records.jsonl \
  path/to/task_taxonomy.json \
  --alpha 0.8
```

The write is atomic. Reward workers watch the configured sidecar modification
time and load the next snapshot without modifying GRPO.

## 6. Reward record

Each scored rollout exposes:

- scalar `reward`/`score` and terminal `reason`;
- `rubric_category` and `active_rubrics`;
- `rubric_risks`, normalized `rubric_weights`, and all `rubric_scores`;
- flattened `rubric_<name>` and `rubric_weight_<name>` fields;
- format and reasoning-token evidence;
- compilation, Catch2, ASan/UBSan, timeout, and TSan evidence;
- candidate/reference runtime fields when measured;
- bounded diagnostic observations and candidate size.

This record is sufficient to reconstruct every contribution to the scalar
reward.

## 7. Multi-attempt helper

The optional retry helper applies a post-reward discount only after all active
verification gates pass: attempt 1 uses 1.0, attempt 2 uses 0.8, and later
attempts use 0.5. The main Miles reward callback remains single-attempt unless
the caller explicitly uses this helper.
