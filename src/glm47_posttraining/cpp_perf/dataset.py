"""Build model prompts and targets from validated PIE C++ tasks."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Mapping, Sequence

from .schema import AiderPolyglotTask, CppTask


DATA_SOURCE = "pie-cpp-perf"
AIDER_DATA_SOURCE = "aider-polyglot-cpp-v1"
FILE_MARKER_PREFIX = "// ===== FILE: "


def build_prompt(task: CppTask) -> str:
    """Return the C++ optimization instruction shown to the model."""

    visible_tests = "\n\n".join(
        f"Input:\n{case.input}\nExpected output:\n{case.expected}" for case in task.unit_tests
    )
    return (
        "Optimize the following C++20 program while preserving its exact behavior.\n"
        "Return exactly one <reasoning>...</reasoning> block followed by exactly one fenced cpp code block.\n"
        "Do not mention or rely on hidden tests. The program must read stdin and write stdout.\n\n"
        "Visible tests:\n"
        f"{visible_tests}\n\n"
        "Program:\n"
        f"```cpp\n{task.prompt_code.rstrip()}\n```"
    )


def sft_output(task: CppTask) -> str:
    """Return the supervised target using the PIE fast solution."""

    return (
        "<reasoning>The optimized program preserves the tested behavior while reducing work.</reasoning>\n"
        f"```cpp\n{task.oracle_solution.rstrip()}\n```"
    )


def load_tasks(tasks_dir: str | Path) -> list[tuple[Path, CppTask]]:
    """Load all task JSON files from a directory tree."""

    root = Path(tasks_dir)
    tasks: list[tuple[Path, CppTask]] = []
    for path in sorted(root.rglob("*.json")):
        if path.name.startswith("_glm47_"):
            continue
        tasks.append((path, CppTask.read_json(path)))
    return tasks


def build_aider_prompt(task: AiderPolyglotTask) -> str:
    """Build a test-blind, multi-file prompt matching Aider's edit task."""

    files = "\n\n".join(
        f"## Editable File (`{name}`)\n```cpp\n{content.rstrip()}\n```"
        for name, content in task.solution_files.items()
    )
    expected = ", ".join(f"`{name}`" for name in task.solution_files)
    return (
        f"# C++ Task: {task.task_id}\n\n"
        f"## Instructions\n{task.instructions_md.strip()}\n\n"
        f"{files}\n\n"
        "Implement the exercise without changing tests or build files. Return exactly one "
        "<reasoning>...</reasoning> block followed by exactly one fenced cpp block. "
        "Inside that block, provide every editable file in this form:\n\n"
        f"{FILE_MARKER_PREFIX}<relative-path> =====\n<complete file content>\n\n"
        f"The required files are: {expected}. Do not omit unchanged files."
    )


def render_aider_solution(files: Mapping[str, str]) -> str:
    """Serialize a complete multi-file edit into the single-block RL protocol."""

    chunks = []
    for name, content in files.items():
        chunks.append(f"{FILE_MARKER_PREFIX}{name} =====\n{content.rstrip()}")
    return "\n\n".join(chunks) + "\n"


def aider_sft_output(task: AiderPolyglotTask) -> str:
    """Render the Exercism example implementation as the supervised target."""

    files = {
        name: task.oracle_files.get(name, stub)
        for name, stub in task.solution_files.items()
    }
    return (
        "<reasoning>Implement the specified interfaces and preserve the exercise's file layout.</reasoning>\n"
        f"```cpp\n{render_aider_solution(files)}```"
    )


def load_aider_tasks(tasks_dir: str | Path) -> list[tuple[Path, AiderPolyglotTask]]:
    """Load serialized Aider Polyglot task JSON files."""

    root = Path(tasks_dir)
    tasks: list[tuple[Path, AiderPolyglotTask]] = []
    for path in sorted(root.rglob("*.json")):
        if path.name.startswith("_"):
            continue
        tasks.append((path, AiderPolyglotTask.read_json(path)))
    return tasks


class ComplexityWireSampler:
    """Construct batches with a stable difficulty-wire distribution.

    Counts use largest-remainder allocation, so every returned batch has the
    requested size. Empty tiers are redistributed across populated tiers.
    """

    TIER_RATIOS = {
        "tier1_easy": 0.20,
        "tier2_medium": 0.35,
        "tier3_hard": 0.30,
        "tier4_extreme": 0.15,
    }

    def __init__(
        self,
        tasks: Sequence[AiderPolyglotTask],
        taxonomy: Mapping[str, Mapping[str, object]],
        *,
        seed: int = 0,
    ) -> None:
        if not tasks:
            raise ValueError("ComplexityWireSampler requires at least one task")
        self._random = random.Random(seed)
        self.buckets: dict[str, list[AiderPolyglotTask]] = {
            tier: [] for tier in self.TIER_RATIOS
        }
        for task in tasks:
            meta = taxonomy.get(task.task_id, {})
            difficulty = float(meta.get("difficulty_index_D_i", task.difficulty_index))
            self.buckets[self.tier_for_difficulty(difficulty)].append(task)

    @staticmethod
    def tier_for_difficulty(difficulty: float) -> str:
        if difficulty <= 0.35:
            return "tier1_easy"
        if difficulty <= 0.65:
            return "tier2_medium"
        if difficulty <= 0.85:
            return "tier3_hard"
        return "tier4_extreme"

    def sample_batch(self, batch_size: int = 32) -> list[AiderPolyglotTask]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        populated = [tier for tier, values in self.buckets.items() if values]
        if not populated:
            raise ValueError("wire sampler has no populated tiers")
        ratio_total = sum(self.TIER_RATIOS[tier] for tier in populated)
        quotas = {
            tier: batch_size * self.TIER_RATIOS[tier] / ratio_total
            for tier in populated
        }
        counts = {tier: math.floor(quota) for tier, quota in quotas.items()}
        remaining = batch_size - sum(counts.values())
        remainders = sorted(
            populated,
            key=lambda tier: (quotas[tier] - counts[tier], self.TIER_RATIOS[tier]),
            reverse=True,
        )
        for tier in remainders[:remaining]:
            counts[tier] += 1

        sampled: list[AiderPolyglotTask] = []
        for tier in populated:
            sampled.extend(self._random.choices(self.buckets[tier], k=counts[tier]))
        self._random.shuffle(sampled)
        return sampled


def priority_weighted_sample(
    tasks: Sequence[AiderPolyglotTask],
    taxonomy: Mapping[str, Mapping[str, object]],
    *,
    count: int,
    gamma: float = 1.5,
    seed: int = 0,
) -> list[AiderPolyglotTask]:
    """Sample tasks in proportion to positive ``P_i ** gamma`` weights."""

    if not tasks:
        raise ValueError("priority sampling requires at least one task")
    if count < 0 or gamma < 0:
        raise ValueError("count and gamma must be non-negative")
    weights = []
    for task in tasks:
        meta = taxonomy.get(task.task_id, {})
        priority = float(meta.get("priority_weight_P_i", task.priority_weight))
        weights.append(max(priority, 1e-12) ** gamma)
    return random.Random(seed).choices(list(tasks), weights=weights, k=count)
