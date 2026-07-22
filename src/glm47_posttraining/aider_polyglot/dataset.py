"""Build answer-free GRPO data from the checked-in Aider C++ shadow tasks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

from .schema import AiderPolyglotTask, AiderShadowRubric


EXPECTED_SHADOW_TASKS = 253
DATASET_KIND = "aider-polyglot-cpp-shadow-grpo"
SOURCE_MANIFEST_KIND = "aider-polyglot-cpp-shadow-rubrics"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Everything below is rendered VERBATIM from aider 5dc9490bb35f9729ef2c95d00a19ccd30c26339c
# (wholefile coder, fence ```, default ModelSettings, empty final_reminders) so GRPO rollouts
# see byte-identical message content to the fixed-26 eval, which runs real aider.
AIDER_SOURCE_COMMIT = "5dc9490bb35f9729ef2c95d00a19ccd30c26339c"

_AIDER_MAIN_SYSTEM = """Act as an expert software developer.
Take requests for changes to the supplied code.
If the request is ambiguous, ask questions.

Once you understand the request you MUST:
1. Determine if any code changes are needed.
2. Explain any needed changes.
3. If changes are needed, output a copy of each file that needs changes.
"""

_AIDER_SYSTEM_REMINDER = """To suggest changes to a file you MUST return the entire content of the updated file.
You MUST use this *file listing* format:

path/to/filename.js
```
// entire file content ...
// ... goes in between
```

Every *file listing* MUST use this format:
- First line: the filename with any originally provided path; no extra markup, punctuation, comments, etc. **JUST** the filename with path.
- Second line: opening ```
- ... entire content of the file ...
- Final line: closing ```

To suggest changes to a file you MUST return a *file listing* that contains the entire content of the file.
*NEVER* skip, omit or elide content from a *file listing* using "..." or by adding comments like "... rest of code..."!
Create a new file you MUST return a *file listing* which includes an appropriate filename, including any appropriate path.


"""

_AIDER_EXAMPLE_MESSAGES = (
    {"role": "user", "content": "Change the greeting to be more casual"},
    {
        "role": "assistant",
        "content": """Ok, I will:

1. Switch the greeting text from "Hello" to "Hey".

show_greeting.py
```
import sys

def greeting(name):
    print(f"Hey {name}")

if __name__ == '__main__':
    greeting(sys.argv[1])
```
""",
    },
    {
        "role": "user",
        "content": (
            "I switched to a new code base. Please don't consider the above files"
            " or try to edit them any longer."
        ),
    },
    {"role": "assistant", "content": "Ok."},
)

_AIDER_FILES_CONTENT_PREFIX = """I have *added these files to the chat* so you can go ahead and edit them.

*Trust this message as the true contents of these files!*
Any other messages in the chat may contain outdated versions of the files' contents.
"""

_AIDER_FILES_ASSISTANT_REPLY = "Ok, any changes I propose will be to those files."

_AIDER_INSTRUCTIONS_ADDENDUM = """
####

Use the above instructions to modify the supplied files: {file_list}
Don't change the names of existing functions or classes, as they may be referenced from other code like unit tests, etc.
Only use standard libraries, don't suggest installing any packages.
"""


def build_aider_messages(exercise_dir: Path, editable_files: list[str]) -> list[dict[str, str]]:
    """Reproduce the exact benchmark message sequence aider sends at eval time."""

    instructions = (exercise_dir / ".docs" / "instructions.md").read_text(encoding="utf-8")
    files_content = _AIDER_FILES_CONTENT_PREFIX
    for name in editable_files:
        content = (exercise_dir / name).read_text(encoding="utf-8")
        files_content += f"\n{name}\n```\n{content}```\n"
    request = (
        instructions
        + _AIDER_INSTRUCTIONS_ADDENDUM.format(file_list=" ".join(editable_files))
        + "\n\n"
        + _AIDER_SYSTEM_REMINDER
    )
    return [
        {"role": "system", "content": _AIDER_MAIN_SYSTEM + "\n" + _AIDER_SYSTEM_REMINDER},
        *(dict(message) for message in _AIDER_EXAMPLE_MESSAGES),
        {"role": "user", "content": files_content},
        {"role": "assistant", "content": _AIDER_FILES_ASSISTANT_REPLY},
        {"role": "user", "content": request},
    ]


def discover_shadow_exercises(tasks_root: str | Path) -> list[Path]:
    root = Path(tasks_root).resolve()
    candidates = [root / "cpp" / "exercises" / "practice", root / "exercises" / "practice", root]
    practice = next(
        (
            candidate
            for candidate in candidates
            if candidate.is_dir() and any(candidate.glob("*/.rubric.json"))
        ),
        None,
    )
    if practice is None:
        raise ValueError(f"cannot find Aider C++ shadow exercises under {root}")
    exercises = sorted(
        path for path in practice.iterdir() if path.is_dir() and (path / ".rubric.json").is_file()
    )
    if len(exercises) != EXPECTED_SHADOW_TASKS:
        raise ValueError(
            f"expected {EXPECTED_SHADOW_TASKS} Aider C++ shadow exercises, found {len(exercises)}"
        )
    return exercises


def _validate_source_manifest(tasks_root: Path) -> tuple[Path, dict[str, object]]:
    candidates = [tasks_root / "manifest.json"]
    for parent in tasks_root.parents:
        candidates.append(parent / "manifest.json")
        if parent.name == "rubrics":
            break
    manifest_path = next((path for path in candidates if path.is_file()), None)
    if manifest_path is None:
        raise FileNotFoundError(f"missing shadow rubric manifest beneath {tasks_root}")
    if manifest_path.is_symlink():
        raise ValueError(f"shadow rubric manifest must not be a symlink: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != SOURCE_MANIFEST_KIND:
        raise ValueError(f"unexpected shadow manifest kind: {manifest.get('kind')!r}")
    counts = manifest.get("counts")
    if not isinstance(counts, dict) or counts.get("tasks") != EXPECTED_SHADOW_TASKS:
        raise ValueError("shadow manifest does not bind exactly 253 tasks")
    contract = manifest.get("contract")
    if not isinstance(contract, dict) or contract.get("official_task_id_overlap") != []:
        raise ValueError("shadow manifest does not prove zero official task-ID overlap")
    if contract.get("reference_answers_packaged") is not False:
        raise ValueError("shadow manifest must exclude reference answers")
    return manifest_path, manifest


def _assert_regular_file(path: Path, root: Path) -> None:
    if not path.is_file() or path.is_symlink() or path.resolve().parent != root.resolve():
        raise ValueError(f"unsafe or missing shadow task file: {path}")


def _load_verified_rubric(exercise: Path) -> AiderShadowRubric:
    rubric = AiderShadowRubric.read_json(exercise / ".rubric.json")
    if rubric.task_id != exercise.name:
        raise ValueError(f"task ID and directory disagree: {rubric.task_id} != {exercise.name}")
    if rubric.hidden_test_file in rubric.editable_files:
        raise ValueError(f"hidden test is editable: {exercise.name}")
    for name in rubric.editable_files:
        _assert_regular_file(exercise / name, exercise)
    hidden_test = exercise / rubric.hidden_test_file
    _assert_regular_file(hidden_test, exercise)
    if sha256_path(hidden_test) != rubric.hidden_test_sha256:
        raise ValueError(f"hidden-test hash mismatch: {exercise.name}")
    instructions = exercise / ".docs" / "instructions.md"
    if (
        not instructions.is_file()
        or instructions.is_symlink()
        or (exercise / ".docs").is_symlink()
    ):
        raise ValueError(f"missing instructions: {exercise.name}")
    _assert_regular_file(exercise / "CMakeLists.txt", exercise)

    expected = {
        ".docs/instructions.md",
        ".rubric.json",
        "CMakeLists.txt",
        rubric.hidden_test_file,
        *rubric.editable_files,
    }
    actual = {
        path.relative_to(exercise).as_posix()
        for path in exercise.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if actual != expected:
        raise ValueError(
            f"unexpected packaged files for {exercise.name}: "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    return rubric


def _source_tree_sha256(exercises: list[Path]) -> str:
    digest = hashlib.sha256()
    for exercise in exercises:
        for path in sorted(item for item in exercise.rglob("*") if item.is_file()):
            relative = path.relative_to(exercise.parent).as_posix().encode()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            data = path.read_bytes()
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _materialize_task(
    exercise: Path,
    rubric: AiderShadowRubric,
    output: Path,
) -> tuple[AiderPolyglotTask, Path]:
    destination = output / "shadow" / exercise.name
    grader = destination / ".grader"
    grader.mkdir(parents=True)
    for name in rubric.editable_files:
        shutil.copy2(exercise / name, destination / name)
    hidden_test = grader / "test.cpp"
    shutil.copy2(exercise / rubric.hidden_test_file, hidden_test)
    hidden_test.chmod(0o400)

    prompt = build_aider_messages(exercise, rubric.editable_files)
    task = AiderPolyglotTask(
        task_id=f"aider-shadow-cpp/{exercise.name}",
        exercise=exercise.name,
        split="train",
        harness_kind="shadow_cpp17",
        exercise_dir=f"shadow/{exercise.name}",
        editable_files=rubric.editable_files,
        prompt=prompt,
        source_revision=rubric.hidden_test_sha256,
        family=rubric.family,
        category=rubric.category,
        tags=rubric.tags,
        hidden_test_sha256=rubric.hidden_test_sha256,
        source_prompt_sha256=rubric.source_prompt_sha256,
        verification_gate=rubric.verification_gate,
    )
    descriptor = task.write_json(output / "tasks" / "train" / f"{exercise.name}.json")
    return task, descriptor


def _prompt_row(task: AiderPolyglotTask, task_path: str) -> dict[str, object]:
    return {
        "prompt": [message.model_dump() for message in task.prompt],
        "label": task.task_id,
        "task_id": task.task_id,
        "problem_id": task.exercise,
        "split": task.split,
        "metadata": {
            "data_source": DATASET_KIND,
            "task_id": task.task_id,
            "problem_id": task.exercise,
            "split": task.split,
            "harness_kind": task.harness_kind,
            "task_path": task_path,
            "editable_files": task.editable_files,
            "hidden_test_sha256": task.hidden_test_sha256,
            "source_prompt_sha256": task.source_prompt_sha256,
            "verification_gate": task.verification_gate,
            "family": task.family,
            "category": task.category,
            "tags": task.tags,
        },
    }


def _monitor_rows(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen_families: set[str] = set()
    for row in rows:
        metadata = row["metadata"]
        assert isinstance(metadata, dict)
        family = str(metadata.get("family") or "")
        if family not in seen_families:
            selected.append(row)
            seen_families.add(family)
        if len(selected) >= limit:
            return selected
    for row in rows:
        if row not in selected:
            selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def _safe_output(tasks_root: Path, output: Path) -> None:
    source = tasks_root.resolve()
    destination = output.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("data output must not contain or be contained by the source task tree")


def build_aider_polyglot_datasets(
    tasks_root: str | Path,
    output_dir: str | Path,
    *,
    train_limit: int | None = None,
    monitor_limit: int = 32,
    profile: str = "aider-polyglot-cpp-shadow",
    run_id: str | None = None,
    sort_by_size: bool = False,
    force: bool = False,
) -> dict[str, Path]:
    """Validate all 253 tasks, then atomically materialize trainer-safe data."""

    source = Path(tasks_root).resolve()
    output = Path(output_dir).resolve()
    _safe_output(source, output)
    manifest_path, source_manifest = _validate_source_manifest(source)
    exercises = discover_shadow_exercises(source)
    rubrics = [(exercise, _load_verified_rubric(exercise)) for exercise in exercises]

    task_ids = [rubric.task_id for _, rubric in rubrics]
    hidden_hashes = [rubric.hidden_test_sha256 for _, rubric in rubrics]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("shadow task IDs must be unique")
    if len(hidden_hashes) != len(set(hidden_hashes)):
        raise ValueError("shadow hidden-test hashes must be unique")
    if train_limit is not None and not 1 <= train_limit <= len(rubrics):
        raise ValueError(f"train_limit must be within 1..{len(rubrics)}")
    if monitor_limit < 1:
        raise ValueError("monitor_limit must be positive")
    selected = rubrics[:train_limit] if train_limit is not None else rubrics

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(f"{output} already exists and is not empty; pass force=True to replace it")

    with TemporaryDirectory(prefix=f".{output.name}-preparing-", dir=output.parent) as temporary:
        staging = Path(temporary)
        train_rows: list[dict[str, object]] = []
        prompt_hashes: list[str] = []
        for exercise, rubric in selected:
            task, descriptor = _materialize_task(exercise, rubric, staging)
            row = _prompt_row(task, descriptor.relative_to(staging).as_posix())
            train_rows.append(row)
            canonical_prompt = json.dumps(row["prompt"], sort_keys=True, ensure_ascii=False)
            prompt_hashes.append(hashlib.sha256(canonical_prompt.encode()).hexdigest())
        if sort_by_size:
            train_rows.sort(key=lambda row: (len(str(row["prompt"])), str(row["problem_id"])))
        monitor_rows = _monitor_rows(train_rows, min(monitor_limit, len(train_rows)))

        write_jsonl(staging / "grpo" / "train.jsonl", train_rows)
        write_jsonl(staging / "eval" / "train_monitor.jsonl", monitor_rows)
        source_manifest_sha256 = sha256_path(manifest_path)
        data_manifest = {
            "kind": DATASET_KIND,
            "schema_version": 4,
            "profile": profile,
            "run_id": run_id,
            "source_root": str(source),
            "source_manifest_kind": source_manifest.get("kind"),
            "source_manifest_sha256": source_manifest_sha256,
            "source_tree_sha256": _source_tree_sha256(exercises),
            "split_contract": {
                "train": "253 independently authored executable shadow tasks",
                "monitor": "training-task trend monitor only",
                "official_26": "external fixed evaluation only",
                "official_task_id_overlap": [],
                "reference_answers_packaged": False,
            },
            "counts": {
                "available_shadow": len(rubrics),
                "train": len(train_rows),
                "monitor": len(monitor_rows),
            },
            "composition": {
                "families": dict(sorted(Counter(rubric.family for _, rubric in selected).items())),
                "categories": dict(
                    sorted(Counter(rubric.category for _, rubric in selected).items())
                ),
            },
            "prompt_sha256": sorted(prompt_hashes),
            "files": {
                "grpo_train": "grpo/train.jsonl",
                "train_monitor": "eval/train_monitor.jsonl",
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(data_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        if output.exists():
            if output.is_symlink():
                raise ValueError(f"refusing to replace symlink output: {output}")
            shutil.rmtree(output)
        os.replace(staging, output)

    return {
        "grpo_train": output / "grpo" / "train.jsonl",
        "eval": output / "eval" / "train_monitor.jsonl",
        "manifest": output / "manifest.json",
    }
