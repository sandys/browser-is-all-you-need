#!/usr/bin/env python3
"""Recompute Fixed26 statistics from the archived task-level result records."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
BOOTSTRAP_SEED = 20260805
BOOTSTRAP_REPLICATES = 100_000

EXPECTED = {
    "GLM-4.7-Flash base": {
        "trials": ["a1", "a2", "a3", "a4"],
        "pass_at_1": [0, 1, 1, 0],
        "multi_turn": [4, 5, 6, 3],
    },
    "Synth v1, epoch 50": {
        "trials": ["a1", "a2", "a3", "a4"],
        "pass_at_1": [10, 9, 10, 9],
        "multi_turn": [11, 13, 12, 12],
    },
    "SFT v5, Aider-format": {
        "trials": ["a2", "a4", "a5", "a8"],
        "pass_at_1": [8, 4, 6, 6],
        "multi_turn": [12, 10, 11, 8],
    },
    "Luna": {
        "trials": ["a1", "a2", "a3", "a4"],
        "pass_at_1": [6, 5, 5, 9],
        "multi_turn": [18, 16, 15, 18],
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def aider_outcome(payload: dict) -> tuple[str, bool, bool]:
    task = payload.get("testcase")
    outcomes = payload.get("tests_outcomes")
    require(isinstance(task, str) and task, "missing Aider testcase")
    require(
        isinstance(outcomes, list)
        and 1 <= len(outcomes) <= 2
        and all(isinstance(value, bool) for value in outcomes),
        f"invalid tests_outcomes for {task}: {outcomes!r}",
    )
    turn_1 = outcomes[0]
    cumulative = any(outcomes)
    return task, turn_1, cumulative


def load_aider_directory(path: Path) -> dict[str, tuple[bool, bool]]:
    rows: dict[str, tuple[bool, bool]] = {}
    for result_path in sorted(path.glob("*/.aider.results.json")):
        task, turn_1, cumulative = aider_outcome(json.loads(result_path.read_text()))
        require(task not in rows, f"duplicate task {task} in {path}")
        rows[task] = (turn_1, cumulative)
    return rows


def load_aider_archive(path: Path) -> dict[str, tuple[bool, bool]]:
    rows: dict[str, tuple[bool, bool]] = {}
    with tarfile.open(path, "r:gz") as archive:
        members = sorted(
            [
                member
                for member in archive.getmembers()
                if member.isfile() and member.name.endswith("/.aider.results.json")
            ],
            key=lambda member: member.name,
        )
        for member in members:
            extracted = archive.extractfile(member)
            require(extracted is not None, f"could not read {member.name} from {path}")
            task, turn_1, cumulative = aider_outcome(json.load(extracted))
            require(task not in rows, f"duplicate task {task} in {path}")
            rows[task] = (turn_1, cumulative)
    return rows


def load_base() -> dict[str, dict[str, tuple[bool, bool]]]:
    base = ROOT / "base-fixed26-20260711" / "extracted"
    return {
        trial: load_aider_directory(base / trial / "responses")
        for trial in EXPECTED["GLM-4.7-Flash base"]["trials"]
    }


def load_archived_trials(
    directory: str, trials: Iterable[str]
) -> dict[str, dict[str, tuple[bool, bool]]]:
    base = ROOT / directory / "trials"
    return {
        trial: load_aider_archive(base / trial / "responses.tar.gz")
        for trial in trials
    }


def load_luna() -> dict[str, dict[str, tuple[bool, bool]]]:
    path = ROOT / "luna-fixed26-20260805" / "datasets" / "terminal_eval_only_samples.jsonl"
    trials: dict[str, dict[str, tuple[bool, bool]]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        trial = f"a{row['trial_index']}"
        task = row["task_id"]
        turn_1 = row["selection"]["turn_1_passed"]
        cumulative = row["passed"]
        require(isinstance(turn_1, bool) and isinstance(cumulative, bool), f"bad Luna row {task}")
        require(not turn_1 or cumulative, f"Luna cumulative regression for {trial}/{task}")
        require(task not in trials[trial], f"duplicate Luna row {trial}/{task}")
        trials[trial][task] = (turn_1, cumulative)
    return dict(trials)


def percentile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_intervals(
    trials: dict[str, dict[str, tuple[bool, bool]]], tasks: list[str]
) -> dict[str, list[float]]:
    rng = random.Random(BOOTSTRAP_SEED)
    trial_names = list(trials)
    pass_means: list[float] = []
    multi_means: list[float] = []
    recovery_rates: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        pass_total = 0
        multi_total = 0
        failures = 0
        recoveries = 0
        for trial in trial_names:
            for task in sampled:
                turn_1, cumulative = trials[trial][task]
                pass_total += int(turn_1)
                multi_total += int(cumulative)
                failures += int(not turn_1)
                recoveries += int((not turn_1) and cumulative)
        pass_means.append(pass_total / len(trial_names))
        multi_means.append(multi_total / len(trial_names))
        recovery_rates.append(recoveries / failures if failures else 0.0)
    pass_means.sort()
    multi_means.sort()
    recovery_rates.sort()
    return {
        "pass_at_1_mean_score": [percentile(pass_means, 0.025), percentile(pass_means, 0.975)],
        "multi_turn_mean_score": [percentile(multi_means, 0.025), percentile(multi_means, 0.975)],
        "conditional_turn_2_recovery_rate": [
            percentile(recovery_rates, 0.025),
            percentile(recovery_rates, 0.975),
        ],
    }


def summarize(
    name: str, trials: dict[str, dict[str, tuple[bool, bool]]]
) -> tuple[dict, list[dict]]:
    expected = EXPECTED[name]
    require(list(trials) == expected["trials"], f"trial order mismatch for {name}: {list(trials)}")
    task_sets = [set(rows) for rows in trials.values()]
    require(all(len(tasks) == 26 for tasks in task_sets), f"{name} does not have 26 tasks per trial")
    require(all(tasks == task_sets[0] for tasks in task_sets[1:]), f"task-set drift in {name}")
    tasks = sorted(task_sets[0])

    pass_scores = [sum(trials[trial][task][0] for task in tasks) for trial in trials]
    multi_scores = [sum(trials[trial][task][1] for task in tasks) for trial in trials]
    require(pass_scores == expected["pass_at_1"], f"Pass@1 score drift for {name}: {pass_scores}")
    require(multi_scores == expected["multi_turn"], f"multi-turn score drift for {name}: {multi_scores}")

    turn_1_failures = sum(26 - score for score in pass_scores)
    turn_2_recoveries = sum(multi - first for first, multi in zip(pass_scores, multi_scores))
    per_trial_recovery = [
        {
            "trial": trial,
            "recoveries": multi - first,
            "turn_1_failures": 26 - first,
            "rate": (multi - first) / (26 - first),
        }
        for trial, first, multi in zip(trials, pass_scores, multi_scores)
    ]

    per_task = []
    for task in tasks:
        first = sum(trials[trial][task][0] for trial in trials)
        multi = sum(trials[trial][task][1] for trial in trials)
        failures = 4 - first
        recoveries = multi - first
        per_task.append(
            {
                "result": name,
                "task": task,
                "pass_at_1_successes_out_of_4": first,
                "multi_turn_successes_out_of_4": multi,
                "turn_2_recoveries": recoveries,
                "turn_1_failures": failures,
                "conditional_turn_2_recovery_rate": recoveries / failures if failures else None,
            }
        )

    summary = {
        "trial_ids": list(trials),
        "pass_at_1": {
            "scores": pass_scores,
            "mean": statistics.mean(pass_scores),
            "range": [min(pass_scores), max(pass_scores)],
            "sample_standard_deviation": statistics.stdev(pass_scores),
        },
        "multi_turn_with_feedback_turn_2": {
            "scores": multi_scores,
            "mean": statistics.mean(multi_scores),
            "range": [min(multi_scores), max(multi_scores)],
            "sample_standard_deviation": statistics.stdev(multi_scores),
        },
        "conditional_turn_2_recovery": {
            "recoveries": turn_2_recoveries,
            "turn_1_failures": turn_1_failures,
            "rate": turn_2_recoveries / turn_1_failures,
            "per_trial": per_trial_recovery,
        },
        "bootstrap_95_percent_intervals": bootstrap_intervals(trials, tasks),
    }
    return summary, per_task


def fmt(value: float, digits: int = 2) -> str:
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".")


def write_markdown(statistics_payload: dict) -> None:
    lines = [
        "# Fixed26 statistics",
        "",
        "| Result | Pass@1 trials | Pass@1 mean, SD, range, 95% CI (out of 26) | Multi turn with feedback (turn=2) trials | Multi turn mean, SD, range, 95% CI (out of 26) | Conditional turn-2 recovery |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, values in statistics_payload["results"].items():
        first = values["pass_at_1"]
        multi = values["multi_turn_with_feedback_turn_2"]
        recovery = values["conditional_turn_2_recovery"]
        intervals = values["bootstrap_95_percent_intervals"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    ", ".join(map(str, first["scores"])),
                    f"{fmt(first['mean'])}/26; SD {fmt(first['sample_standard_deviation'])}; "
                    f"range {first['range'][0]}-{first['range'][1]}; "
                    f"CI {fmt(intervals['pass_at_1_mean_score'][0])}-{fmt(intervals['pass_at_1_mean_score'][1])}",
                    ", ".join(map(str, multi["scores"])),
                    f"{fmt(multi['mean'])}/26; SD {fmt(multi['sample_standard_deviation'])}; "
                    f"range {multi['range'][0]}-{multi['range'][1]}; "
                    f"CI {fmt(intervals['multi_turn_mean_score'][0])}-{fmt(intervals['multi_turn_mean_score'][1])}",
                    f"{recovery['recoveries']}/{recovery['turn_1_failures']} "
                    f"({fmt(100 * recovery['rate'], 1)}%); "
                    f"CI {fmt(100 * intervals['conditional_turn_2_recovery_rate'][0], 1)}-"
                    f"{fmt(100 * intervals['conditional_turn_2_recovery_rate'][1], 1)}%",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The 95% intervals use 100,000 deterministic task-clustered bootstrap replicates "
            "(seed 20260805). Each replicate resamples the 26 task IDs and preserves all four "
            "outcomes for each selected task. The 104 task-trial records are not treated as "
            "independent Bernoulli observations.",
            "",
            "Per-task frequencies are in [per_task_success.csv](per_task_success.csv). "
            "Run `python3 results/compute_statistics.py` from the repository root to recompute "
            "all outputs.",
        ]
    )
    (ROOT / "statistics.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    datasets = {
        "GLM-4.7-Flash base": load_base(),
        "Synth v1, epoch 50": load_archived_trials(
            "synth-v1-ep50-9.5-mean", EXPECTED["Synth v1, epoch 50"]["trials"]
        ),
        "SFT v5, Aider-format": load_archived_trials(
            "sft-v5-aiderfmt-1117-4trials", EXPECTED["SFT v5, Aider-format"]["trials"]
        ),
        "Luna": load_luna(),
    }
    summaries = {}
    task_rows = []
    canonical_tasks = None
    for name, trials in datasets.items():
        summary, per_task = summarize(name, trials)
        summaries[name] = summary
        task_rows.extend(per_task)
        tasks = [row["task"] for row in per_task]
        if canonical_tasks is None:
            canonical_tasks = tasks
        else:
            require(tasks == canonical_tasks, f"cross-result task-set drift for {name}")

    payload = {
        "schema_version": 1,
        "task_count": 26,
        "trial_count_per_result": 4,
        "sample_count_per_result": 104,
        "bootstrap": {
            "method": "task-clustered percentile bootstrap",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "cluster": "task_id",
            "preserved_within_cluster": "all four trial outcomes",
        },
        "results": summaries,
    }
    (ROOT / "statistics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    columns = [
        "result",
        "task",
        "pass_at_1_successes_out_of_4",
        "multi_turn_successes_out_of_4",
        "turn_2_recoveries",
        "turn_1_failures",
        "conditional_turn_2_recovery_rate",
    ]
    with (ROOT / "per_task_success.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(task_rows)
    write_markdown(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
