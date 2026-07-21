#!/usr/bin/env python3
"""Build Polyglot C++ complexity, prompt-length, and sampling metadata."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from glm47_posttraining.cpp_perf.dataset import build_aider_prompt, load_aider_tasks


DECISION_RE = re.compile(r"\b(?:if|for|while|case|catch)\b|&&|\|\||\?(?![?:])")
COMMENT_OR_LITERAL_RE = re.compile(
    r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
    re.DOTALL,
)
CATEGORY_BOOSTS = {
    "State & concurrency": 1.4,
    "Time & date": 1.4,
    "Text & parsing": 1.2,
    "Logic, grids & games": 1.2,
    "Algorithms & data structures": 1.0,
    "Numerical reasoning": 0.8,
}


def compute_cyclomatic_complexity(cpp_text: str) -> int:
    """Return a deterministic lexical McCabe proxy for C++ decision points."""

    cleaned = COMMENT_OR_LITERAL_RE.sub(" ", cpp_text)
    return 1 + len(DECISION_RE.findall(cleaned))


def fallback_token_count(text: str) -> int:
    """Dependency-free approximation used only when no GLM tokenizer is supplied."""

    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def tokenizer_counter(tokenizer_path: str) -> Callable[[str], int]:
    if not tokenizer_path:
        return fallback_token_count
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("--tokenizer requires transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def quantile_ranks(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    denominator = max(1, len(ordered) - 1)
    return {task_id: index / denominator for index, (task_id, _value) in enumerate(ordered)}


def empirical_failure_rates(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("cells", payload.get("records", payload)) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("empirical results must contain a list of records/cells")
    outcomes: dict[str, list[float]] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("task_id"):
            continue
        if "try1_success" in row:
            failed = 0.0 if bool(row["try1_success"]) else 1.0
        elif "all_tests_pass" in row:
            failed = 0.0 if bool(row["all_tests_pass"]) else 1.0
        else:
            continue
        outcomes.setdefault(str(row["task_id"]), []).append(failed)
    return {task_id: sum(values) / len(values) for task_id, values in outcomes.items()}


def build_taxonomy(
    tasks_dir: Path,
    output_json: Path,
    *,
    empirical_results_json: Path | None = None,
    tokenizer_path: str = "",
) -> dict[str, dict[str, object]]:
    loaded = load_aider_tasks(tasks_dir)
    if not loaded:
        raise ValueError(f"no Polyglot task JSON found under {tasks_dir}")
    count_tokens = tokenizer_counter(tokenizer_path)
    failures = empirical_failure_rates(empirical_results_json)

    complexities: dict[str, float] = {}
    lengths: dict[str, float] = {}
    base: dict[str, dict[str, object]] = {}
    for _path, task in loaded:
        implementation_and_tests = "\n".join(
            list(task.solution_files.values()) + list(task.test_files.values())
        )
        complexity = compute_cyclomatic_complexity(implementation_and_tests)
        prompt_length = count_tokens(build_aider_prompt(task))
        complexities[task.task_id] = float(complexity)
        lengths[task.task_id] = float(prompt_length)
        base[task.task_id] = {
            "task_id": task.task_id,
            "topic_category": task.topic_category,
            "rubric_category": task.effective_rubric_category,
            "rubric_risks": task.rubric_risks,
            "difficulty": task.difficulty,
            "cyclomatic_complexity": complexity,
            "prompt_token_length": prompt_length,
            "empirical_failure_rate": failures.get(task.task_id, 0.5),
        }

    complexity_quantiles = quantile_ranks(complexities)
    length_quantiles = quantile_ranks(lengths)
    taxonomy: dict[str, dict[str, object]] = {}
    for task_id, record in base.items():
        failure = float(record["empirical_failure_rate"])
        difficulty_index = (
            0.50 * failure
            + 0.25 * length_quantiles[task_id]
            + 0.25 * complexity_quantiles[task_id]
        )
        category_boost = CATEGORY_BOOSTS.get(str(record["topic_category"]), 1.0)
        priority = max(0.05, difficulty_index * category_boost)
        taxonomy[task_id] = {
            **record,
            "complexity_quantile": round(complexity_quantiles[task_id], 6),
            "length_quantile": round(length_quantiles[task_id], 6),
            "difficulty_index_D_i": round(difficulty_index, 6),
            "priority_weight_P_i": round(priority, 6),
            "tokenizer": tokenizer_path or "lexical-fallback",
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(taxonomy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return taxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--empirical-results", type=Path, default=None)
    parser.add_argument(
        "--tokenizer",
        default="",
        help="Local GLM tokenizer/model path; lexical counting is used when omitted",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    taxonomy = build_taxonomy(
        args.tasks_dir,
        args.output_json,
        empirical_results_json=args.empirical_results,
        tokenizer_path=args.tokenizer,
    )
    print(f"Wrote {len(taxonomy)} task records to {args.output_json}")


if __name__ == "__main__":
    main()
