#!/usr/bin/env python3
import hashlib
import json
import tarfile
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "manifest.json").read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def instruction_prompt(path: Path) -> str:
    lines = path.read_text(errors="replace").splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("####"))
    selected = []
    for line in lines[start:]:
        if line.startswith("####") or not line.strip():
            selected.append(line)
        else:
            break
    while selected and not selected[-1].strip():
        selected.pop()
    return "\n".join(selected) + "\n"


task_sets = []
prompt_hashes = {}
pass_at_1_scores = []
feedback_at_2_scores = []

for label, trial in MANIFEST["source"]["trials"].items():
    trial_root = ROOT / "trials" / label
    extracted = ROOT / "extracted" / label
    receipt = json.loads((trial_root / "run_receipt.json").read_text())

    assert receipt["model_revision"] == MANIFEST["model"]["revision"]
    assert receipt["aider_commit"] == MANIFEST["evaluation"]["aider_commit"]
    assert receipt["polyglot_commit"] == MANIFEST["evaluation"]["polyglot_commit"]
    assert sha256(trial_root / "responses.tar.gz") == trial["archive_sha256"]
    assert sha256(trial_root / "run_receipt.json") == trial["run_receipt_sha256"]

    archived_files = 0
    with tarfile.open(trial_root / "responses.tar.gz", "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            archived_files += 1
            archived = archive.extractfile(member)
            assert archived is not None
            extracted_path = extracted / member.name
            assert extracted_path.is_file()
            assert hashlib.sha256(archived.read()).hexdigest() == sha256(extracted_path)
    assert archived_files == 52

    if label != "a1":
        request_command = json.dumps(receipt["full"]["command"])
        assert "lora_path" not in request_command
        assert "extra_body" not in request_command

    results = sorted(extracted.rglob(".aider.results.json"))
    chats = sorted(extracted.rglob(".aider.chat.history.md"))
    assert len(results) == len(chats) == 26

    outcomes = []
    tasks = []
    for path in results:
        record = json.loads(path.read_text())
        tasks.append(record.get("testcase") or path.parent.name)
        outcomes.append(record["tests_outcomes"])
    task_sets.append(sorted(tasks))

    pass_at_1 = sum(bool(item and item[0]) for item in outcomes)
    feedback_at_2 = sum(bool(item and any(item[:2])) for item in outcomes)
    pass_at_1_scores.append(pass_at_1)
    feedback_at_2_scores.append(feedback_at_2)

    for path in chats:
        task = path.parent.name
        prompt_hashes.setdefault(task, set()).add(
            hashlib.sha256(instruction_prompt(path).encode()).hexdigest()
        )

assert all(tasks == task_sets[0] for tasks in task_sets[1:])
assert len(task_sets[0]) == 26
assert len(prompt_hashes) == 26
assert all(len(hashes) == 1 for hashes in prompt_hashes.values())
assert pass_at_1_scores == MANIFEST["evaluation"]["pass_at_1_scores"]
assert feedback_at_2_scores == MANIFEST["evaluation"]["cumulative_feedback_at_2_scores"]
assert mean(pass_at_1_scores) == MANIFEST["evaluation"]["pass_at_1_mean"]
assert mean(feedback_at_2_scores) == MANIFEST["evaluation"]["cumulative_feedback_at_2_mean"]
assert sum(len(tasks) for tasks in task_sets) == MANIFEST["sample_count"] == 104

print(json.dumps({
    "status": "verified",
    "trials": 4,
    "samples": 104,
    "pass_at_1_scores": pass_at_1_scores,
    "pass_at_1_mean": mean(pass_at_1_scores),
    "feedback_at_2_scores": feedback_at_2_scores,
    "feedback_at_2_mean": mean(feedback_at_2_scores),
    "identical_task_sets": True,
    "identical_instruction_prompts": True,
}, indent=2))
