#!/usr/bin/env python3
"""Verify the exact official bank-account drill environment end to end on GCC."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from glm47_posttraining.aider_polyglot.bank_account_official_drill import (  # noqa: E402
    CURRICULUM_NAME,
    OFFICIAL_PROMPT_SHA256,
    REFERENCE,
    TEST,
    build_bank_account_official_drill,
    episodes,
    evaluate_files,
    imitation_targets,
)
from glm47_posttraining.aider_polyglot.dataset import (  # noqa: E402
    build_aider_polyglot_datasets,
)
from glm47_posttraining.aider_polyglot.parser import parse_whole_file_response  # noqa: E402
from glm47_posttraining.aider_polyglot.reward import compute_aider_reward  # noqa: E402
from glm47_posttraining.aider_polyglot.schema import (  # noqa: E402
    AiderPolyglotTask,
    AiderTestResult,
)
from verify_bank_account_epoch50_signatures import (  # noqa: E402
    EXPECTED_TESTS as OFFICIAL_TEST_NAMES,
    evaluate as evaluate_official_tests,
    load_harness as load_official_harness,
)


EXPECTED_EPISODES = {
    "full-solve": ("official-full-solve", "compile", None),
    "missing-stdexcept-repair": ("missing-standard-header", "compile", None),
    "stale-reopen-repair": ("stale-state-on-reopen", "semantic", 13),
    "closed-balance-repair": ("closed-account-balance-guard", "semantic", 7),
    "missing-definitions-repair": ("missing-method-definitions", "link-or-odr", None),
    "name-collision-repair": ("method-member-name-collision", "compile", None),
    "compiler-feedback-repair": ("feedback-missing-standard-header", "compile", None),
    "test-feedback-repair": ("feedback-stale-state-on-reopen", "semantic", 13),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def response_for(files: dict[str, str]) -> str:
    return "\n\n".join(f"{name}\n```cpp\n{body}\n```" for name, body in files.items())


def result_adapter(_path: Path, files: dict[str, str], *, compiler: str) -> AiderTestResult:
    result = evaluate_files(files, compiler=compiler)
    if result.stage == "pass":
        return AiderTestResult(status="passed", tests_passed=17, tests_total=17)
    if result.stage == "semantic":
        passed = max(0, (result.failed_ordinal or 1) - 1)
        return AiderTestResult(
            status="tests_failed",
            tests_passed=passed,
            tests_total=17,
            candidate_returncode=result.failed_ordinal,
            logs={"test": result.diagnostic},
        )
    return AiderTestResult(
        status="compile_failed",
        candidate_returncode=1,
        logs={result.stage: result.diagnostic},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()

    with TemporaryDirectory(prefix="bank-account-official-env-") as temporary:
        root = Path(temporary)
        rubric_root = root / "rubrics"
        dataset_root = root / "dataset"
        manifest = build_bank_account_official_drill(
            rubric_root,
            compiler=args.compiler,
            require_gcc=True,
        )
        require(manifest["curriculum_id"].endswith("self-imitation-rl-v1"), "wrong curriculum")
        require(manifest["counts"] == {"tasks": 8, "train": 8, "validation": 0}, "wrong counts")
        contract = manifest["contract"]
        require(contract["official_task_id_overlap"] == ["bank-account"], "official overlap lost")
        require(contract["official_training_authorized"] is True, "training authorization missing")
        require(contract["strict_binary_reward"] is True, "binary reward contract missing")
        require(contract["zero_held_out"] is True, "zero-held-out decision missing")
        require(contract["official_prompt_sha256"] == OFFICIAL_PROMPT_SHA256, "prompt SHA drifted")

        paths = build_aider_polyglot_datasets(
            rubric_root,
            dataset_root,
            monitor_limit=8,
            profile=CURRICULUM_NAME,
            run_id="cpu-verification",
            sort_by_size=True,
            imitation_targets=imitation_targets(),
        )
        data_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        require(
            data_manifest["counts"]
            == {
                "available_shadow": 8,
                "monitor": 8,
                "sft_train": 8,
                "train": 8,
                "validation": 0,
            },
            f"serialized counts drifted: {data_manifest['counts']}",
        )
        require(
            data_manifest["split_contract"]["official_26"]
            == "explicitly authorized training target",
            "official training role was not serialized",
        )
        require(
            data_manifest["split_contract"]["reference_answers_packaged"] is True,
            "authorized imitation targets were not declared",
        )
        require(
            data_manifest["split_contract"]["source_reference_answers_packaged"] is False,
            "answer-free source task contract drifted",
        )

        rows = [
            json.loads(line)
            for line in paths["grpo_train"].read_text(encoding="utf-8").splitlines()
            if line
        ]
        require(len(rows) == 8, "expected eight serialized train rows")
        require(len({row["task_id"] for row in rows}) == 8, "duplicate task IDs")
        serialized = paths["grpo_train"].read_text(encoding="utf-8")
        require(REFERENCE["bank_account.cpp"] not in serialized, "reference answer leaked")

        sft_rows = [
            json.loads(line)
            for line in paths["sft_train"].read_text(encoding="utf-8").splitlines()
            if line
        ]
        require(len(sft_rows) == 8, "expected eight serialized imitation rows")
        require(
            {row["task_id"] for row in sft_rows} == {row["task_id"] for row in rows},
            "SFT and GRPO task identities disagree",
        )
        targets = imitation_targets()
        official_harness = load_official_harness()
        episode_by_task = {
            f"bank-account-official--{episode.episode_kind}": episode
            for episode in episodes()
        }
        for row in sft_rows:
            source_task_id = row["problem_id"]
            messages = row["messages"]
            require(messages[-1]["role"] == "assistant", "SFT row lacks assistant target")
            response = messages[-1]["content"]
            require(response == targets[source_task_id], "serialized imitation target drifted")
            task = AiderPolyglotTask.read_json(dataset_root / row["metadata"]["task_path"])
            parsed = parse_whole_file_response(response, task.editable_files)
            require(parsed.format_valid, f"{task.task_id}: imitation response format invalid")
            starter = dict(episode_by_task[source_task_id].starter)
            starter.update(parsed.files)
            require(
                set(starter) == set(REFERENCE)
                and all(starter[name].rstrip() == REFERENCE[name].rstrip() for name in REFERENCE),
                f"{task.task_id}: imitation action misses reference",
            )
            target_result = evaluate_files(starter, compiler=args.compiler)
            require(target_result.stage == "pass", f"{task.task_id}: imitation target failed")
            official_result = evaluate_official_tests(
                args.compiler,
                official_harness,
                starter,
                root / "official-catch2",
                source_task_id,
            )
            require(
                official_result.stage == "pass"
                and official_result.tests_passed == len(OFFICIAL_TEST_NAMES),
                f"{task.task_id}: imitation target failed exact official tests",
            )
            require(TEST not in response, f"{task.task_id}: hidden-test filename leaked")
            require(
                row["metadata"]["imitation_target_sha256"]
                == hashlib.sha256(response.encode("utf-8")).hexdigest(),
                f"{task.task_id}: imitation target hash drifted",
            )

        receipts = {
            value["episode_kind"]: value
            for value in (
                json.loads(line)
                for line in (rubric_root / "verification.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            )
        }
        require(set(receipts) == set(EXPECTED_EPISODES), "episode inventory drifted")
        for episode, (signature, stage, ordinal) in EXPECTED_EPISODES.items():
            receipt = receipts[episode]
            require(receipt["failure_signature"] == signature, f"{episode}: signature drifted")
            require(receipt["starter_rejected_as"] == stage, f"{episode}: stage drifted")
            require(receipt["starter_failed_ordinal"] == ordinal, f"{episode}: ordinal drifted")

        reward_records = []
        reference_response = response_for(REFERENCE)
        for row in rows:
            metadata = row["metadata"]
            task_path = dataset_root / metadata["task_path"]
            task = AiderPolyglotTask.read_json(task_path)
            require(task.objective_group == "bank-account-drill", "objective metadata missing")
            require(task.failure_signature == metadata["failure_signature"], "signature routing drifted")
            require("strict-binary-reward" in task.tags, "strict reward tag missing")

            parsed = parse_whole_file_response(reference_response, task.editable_files)
            require(parsed.format_valid, f"{task.task_id}: reference response format invalid")
            passed = compute_aider_reward(
                task,
                dataset_root / task.exercise_dir,
                reference_response,
                runner=lambda path, files: result_adapter(path, files, compiler=args.compiler),
                strict_binary=True,
            )
            require(passed.reward == 1.0 and passed.reason == "passed", f"{task.task_id}: pass reward wrong")

            empty = compute_aider_reward(
                task,
                dataset_root / task.exercise_dir,
                "",
                runner=lambda path, files: result_adapter(path, files, compiler=args.compiler),
                strict_binary=True,
            )
            require(empty.reward == 0.0, f"{task.task_id}: no-edit reward is not zero")
            reward_records.append(
                {
                    "task_id": task.task_id,
                    "episode_kind": task.episode_kind,
                    "failure_signature": task.failure_signature,
                    "pass_reward": passed.reward,
                    "no_edit_reward": empty.reward,
                }
            )

        representative = AiderPolyglotTask.read_json(
            dataset_root / rows[0]["metadata"]["task_path"]
        )
        malformed = "bank_account.cpp\n```cpp\nnamespace Bankaccount {```"
        bad_format = compute_aider_reward(
            representative,
            dataset_root / representative.exercise_dir,
            malformed,
            runner=lambda path, files: result_adapter(path, files, compiler=args.compiler),
            strict_binary=True,
        )
        require(bad_format.reward == 0.0, "malformed/backtick response reward is not zero")

        summary = {
            "status": "passed",
            "kind": "bank-account-official-drill-environment-verification",
            "compiler": manifest["compiler"],
            "curriculum": CURRICULUM_NAME,
            "tasks": len(rows),
            "imitation_rows": len(sft_rows),
            "exact_official_imitation_passes": len(sft_rows),
            "zero_held_out": True,
            "strict_binary_reward": True,
            "reference_reward": 1.0,
            "no_edit_reward": 0.0,
            "malformed_response_reward": bad_format.reward,
            "episode_receipts": [receipts[name] for name in sorted(receipts)],
            "reward_routing": reward_records,
            "limitations": [
                "The ordinal oracle reproduces the official 17 behavioral cases but is not byte-identical to the Catch2 source.",
                "The official task does not test zero-valued deposit or withdrawal.",
                "The 1,000-thread concurrency case remains probabilistic.",
                "Verified-success replay is implemented as a separate direct SFT stage, not a custom mixed-loss trainer.",
                "KL routing, complete optimizer-state persistence, and in-process trainer phases remain out of scope.",
            ],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        print("BANK_ACCOUNT_OFFICIAL_DRILL_ENVIRONMENT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
