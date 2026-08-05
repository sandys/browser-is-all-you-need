#!/usr/bin/env python3
import hashlib
import io
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


task_sets = []
pass_at_1_scores = []
feedback_at_2_scores = []

for label in MANIFEST["source"]["trials"]:
    trial_root = ROOT / "trials" / label
    receipt_path = trial_root / "source_run_receipt.json"
    archive_path = trial_root / "responses.tar.gz"
    receipt = json.loads(receipt_path.read_text())

    assert sha256(receipt_path) == MANIFEST["source_receipts"][label]
    assert sha256(archive_path) == MANIFEST["archives"][label]
    assert receipt["status"] == "complete"
    assert receipt["tries"] == 2
    assert receipt["temperature"] == 0.7
    assert receipt["top_p"] == 1.0
    assert receipt["aider_commit"] == MANIFEST["evaluation"]["aider_commit"]
    assert receipt["polyglot_commit"] == MANIFEST["evaluation"]["polyglot_commit"]
    assert receipt["adapter_sha256"] == MANIFEST["checkpoint"]["adapter_sha256"]
    assert receipt["training_data_manifest_sha256"] == MANIFEST["training_data"]["manifest_sha256"]
    assert receipt["validation"]["terminal_tasks"] == 26
    assert receipt["validation"]["unique_testcases"] == 26

    result_records = []
    chat_tasks = []
    with tarfile.open(archive_path, "r:gz") as archive:
        files = [member for member in archive.getmembers() if member.isfile()]
        names = [member.name for member in files]
        assert len([name for name in names if name.endswith("/.aider.results.json")]) == 26
        assert len([name for name in names if name.endswith("/.aider.chat.history.md")]) == 26
        assert not any("/test/" in name or "/.meta/" in name for name in names)
        assert not any(name.endswith("_test.cpp") or name.endswith("tests-main.cpp") for name in names)

        for member in files:
            if member.name.endswith("/.aider.results.json"):
                extracted = archive.extractfile(member)
                assert extracted is not None
                result_records.append(json.load(io.TextIOWrapper(extracted, encoding="utf-8")))
            elif member.name.endswith("/.aider.chat.history.md"):
                chat_tasks.append(Path(member.name).parent.name)

    tasks = sorted(record["testcase"] for record in result_records)
    assert tasks == sorted(chat_tasks)
    assert len(tasks) == len(set(tasks)) == 26
    task_sets.append(tasks)

    pass_at_1 = sum(
        bool(record["tests_outcomes"] and record["tests_outcomes"][0])
        for record in result_records
    )
    feedback_at_2 = sum(
        bool(record["tests_outcomes"] and any(record["tests_outcomes"][:2]))
        for record in result_records
    )
    assert pass_at_1 == receipt["validation"]["pass_at_1"]
    assert feedback_at_2 == receipt["validation"]["multi_turn_with_error_feedback_at_2"]
    pass_at_1_scores.append(pass_at_1)
    feedback_at_2_scores.append(feedback_at_2)

assert all(tasks == task_sets[0] for tasks in task_sets[1:])
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
    "benchmark_tests_in_archives": False,
}, indent=2))
