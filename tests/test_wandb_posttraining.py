from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

from glm47_posttraining.integrations.wandb_posttraining import (
    COMPARISON_TABLE_COLUMNS,
    EVAL_TABLE_COLUMNS,
    FAILURE_TABLE_COLUMNS,
    MILES_CHECKPOINT_TABLE_COLUMNS,
    MILES_METRIC_TABLE_COLUMNS,
    MILES_REWARD_OUTCOME_TABLE_COLUMNS,
    MILES_SAMPLE_TABLE_COLUMNS,
    MILES_SYNC_TABLE_COLUMNS,
    PIPELINE_TABLE_COLUMNS,
    build_eval_table_rows,
    build_failure_bucket_rows,
    log_comparison_run,
    log_eval_run,
    log_pipeline_milestone,
    log_stage_finalization,
    redact_sensitive,
    select_artifact_paths,
)


class FakeTable:
    def __init__(self, *, columns: list[str], data: list[list[Any]]) -> None:
        self.columns = columns
        self.data = data


class FakeArtifact:
    def __init__(self, name: str, *, type: str, metadata: dict[str, Any]) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files: list[tuple[str, str]] = []

    def add_file(self, path: str, *, name: str) -> None:
        self.files.append((path, name))


class FakeRun:
    def __init__(self, run_id: str) -> None:
        self.id = run_id
        self.url = f"https://wandb.invalid/runs/{run_id}"
        self.summary: dict[str, Any] = {}
        self.logs: list[dict[str, Any]] = []
        self.artifacts: list[FakeArtifact] = []
        self.metrics: list[tuple[str, dict[str, Any]]] = []
        self.finished = False

    def define_metric(self, name: str, **kwargs: Any) -> None:
        self.metrics.append((name, kwargs))

    def log(self, payload: dict[str, Any]) -> None:
        self.logs.append(payload)

    def log_artifact(self, artifact: FakeArtifact) -> None:
        self.artifacts.append(artifact)

    def finish(self) -> None:
        self.finished = True


class FakeWandb:
    Table = FakeTable
    Artifact = FakeArtifact

    def __init__(self, *, seed_summary: dict[str, Any] | None = None) -> None:
        self.init_calls: list[dict[str, Any]] = []
        self.runs: list[FakeRun] = []
        self.seed_summary = seed_summary or {}

    def init(self, **kwargs: Any) -> FakeRun:
        self.init_calls.append(kwargs)
        run = FakeRun(kwargs["id"])
        run.summary.update(self.seed_summary)
        self.runs.append(run)
        return run


def _records() -> list[dict[str, Any]]:
    return [
        {
            "label": "base",
            "task_id": "task-1",
            "problem_id": "problem-1",
            "split": "validation",
            "sample_index": 0,
            "reason": "correct_and_faster",
            "reward": 1.25,
            "all_tests_pass": True,
            "tests_passed": 5,
            "tests_total": 5,
            "runtime_cpu_ns": 100,
            "reference_runtime_cpu_ns": 200,
            "runtime_speedup": 2.0,
        },
        {
            "label": "base",
            "task_id": "task-2",
            "split": "validation",
            "sample_index": 0,
            "reason": "compile_error",
            "reward": 0.0,
            "all_tests_pass": False,
            "compile_error": True,
        },
    ]


def _generations() -> list[dict[str, Any]]:
    return [
        {
            "label": "base",
            "task_id": "task-1",
            "sample_index": 0,
            "response": "int main() { return 0; }",
            "completion_tokens": 12,
            "prompt_tokens": 50,
            "truncated": False,
            "finish_reason": "stop",
        }
    ]


def _summary(label: str, score: float) -> dict[str, Any]:
    return {
        "label": label,
        "task_count": 2,
        "sample_count": 2,
        "pass_rate": score,
        "valid_format_rate": 1.0,
        "correct_and_faster_rate": score,
        "mean_best_reward": score,
        "mean_correct_faster_speedup": 1.5,
        "missing_runtime_count": 0,
        "missing_runtime_rate": 0.0,
        "compile_error_rate": 0.0,
        "sanitizer_error_rate": 0.0,
        "timeout_rate": 0.0,
        "truncated_ratio": 0.0,
        "mean_completion_tokens": 12.0,
    }


def test_eval_and_failure_tables_have_stable_public_schemas() -> None:
    eval_rows = build_eval_table_rows(
        _records(),
        _generations(),
        experiment_id="experiment-1",
        timing_status="verified",
    )
    assert len(eval_rows) == 2
    assert all(len(row) == len(EVAL_TABLE_COLUMNS) for row in eval_rows)
    first = dict(zip(EVAL_TABLE_COLUMNS, eval_rows[0], strict=True))
    assert first["response_preview"] == "int main() { return 0; }"
    assert first["runtime_speedup"] == 2.0
    assert first["timing_status"] == "verified"

    failure_rows = build_failure_bucket_rows(_records(), experiment_id="experiment-1")
    assert all(len(row) == len(FAILURE_TABLE_COLUMNS) for row in failure_rows)
    by_bucket = {
        dict(zip(FAILURE_TABLE_COLUMNS, row, strict=True))["bucket"]: row for row in failure_rows
    }
    assert set(by_bucket) == {"compile_error", "correct_and_faster"}


def test_redaction_covers_nested_values_urls_and_free_form_errors() -> None:
    redacted = redact_sensitive(
        {
            "WANDB_API_KEY": "secret-value",
            "max_tokens": 1536,
            "nested": {"password": "hunter2"},
            "url": "https://user:pass@example.test/path",
            "error": "failed with ACCESS_TOKEN=abcdef and retry",
        }
    )

    assert redacted["WANDB_API_KEY"] == "<redacted>"
    assert redacted["max_tokens"] == 1536
    assert redacted["nested"]["password"] == "<redacted>"
    assert redacted["url"] == "https://<redacted>@example.test/path"
    assert redacted["error"] == "failed with ACCESS_TOKEN=<redacted> and retry"


def test_artifact_selection_excludes_secrets_weights_and_symlinks(tmp_path: Path) -> None:
    summary = tmp_path / "base.summary.json"
    summary.write_text("{}\n", encoding="utf-8")
    secret = tmp_path / "wandb_api_key.txt"
    secret.write_text("do-not-upload\n", encoding="utf-8")
    weights = tmp_path / "adapter.safetensors"
    weights.write_bytes(b"weights")
    link = tmp_path / "summary-link.json"
    link.symlink_to(summary)
    leaked_log = tmp_path / "run.log"
    leaked_log.write_text("WANDB_API_KEY=do-not-upload\n", encoding="utf-8")

    selected, skipped = select_artifact_paths([summary, secret, weights, link, leaked_log, summary])

    assert selected == [summary]
    assert {item["name"]: item["reason"] for item in skipped} == {
        "wandb_api_key.txt": "sensitive_name",
        "adapter.safetensors": "unsupported_suffix",
        "summary-link.json": "symlink",
        "run.log": "sensitive_content",
    }


def test_eval_logger_emits_tables_summary_and_manifest_artifact(tmp_path: Path) -> None:
    source = tmp_path / "base.summary.json"
    source.write_text(json.dumps(_summary("base", 0.5)), encoding="utf-8")
    wandb = FakeWandb()

    result = log_eval_run(
        wandb,
        project="glm47-pie-cpp-posttraining",
        entity=None,
        experiment_id="experiment-1",
        run_id="experiment-1-base-eval",
        name="base eval",
        group="experiment-1",
        job_type="eval",
        mode="offline",
        timing_status="verified",
        summary=_summary("base", 0.5),
        records=_records(),
        generations=_generations(),
        config={"WANDB_API_KEY": "secret", "max_tokens": 1536},
        artifact_paths=[source],
        manifest_dir=tmp_path,
    )

    init = wandb.init_calls[0]
    run = wandb.runs[0]
    assert result["run_id"] == "experiment-1-base-eval"
    assert init["group"] == "experiment-1"
    assert init["job_type"] == "eval"
    assert init["resume"] == "allow"
    assert init["config"]["WANDB_API_KEY"] == "<redacted>"
    assert init["config"]["max_tokens"] == 1536
    assert run.logs[0]["tables/eval_samples"].columns == list(EVAL_TABLE_COLUMNS)
    assert run.logs[0]["tables/failure_buckets"].columns == list(FAILURE_TABLE_COLUMNS)
    assert run.summary["observability/timing_status"] == "verified"
    assert run.summary["stage/status"] == "success"
    assert run.artifacts[0].type == "eval"
    artifact_names = {name for _, name in run.artifacts[0].files}
    assert "base.summary.json" in artifact_names
    assert "experiment-1-base-eval.artifact_manifest.json" in artifact_names
    assert run.finished is True


def test_pipeline_milestones_resume_one_stable_run_and_redact_failures(tmp_path: Path) -> None:
    receipt = tmp_path / "run.sft.json"
    receipt.write_text("{}\n", encoding="utf-8")
    wandb = FakeWandb()

    log_pipeline_milestone(
        wandb,
        project="glm47-pie-cpp-posttraining",
        entity=None,
        experiment_id="experiment-1",
        stage="sft",
        event="started",
        status="started",
        mode="offline",
        event_time=100.0,
    )
    log_pipeline_milestone(
        wandb,
        project="glm47-pie-cpp-posttraining",
        entity=None,
        experiment_id="experiment-1",
        stage="sft",
        event="failed",
        status="failed",
        mode="offline",
        wall_s=12.5,
        receipt=receipt,
        error="WANDB_API_KEY=should-not-leak",
        event_time=112.5,
    )

    assert [call["id"] for call in wandb.init_calls] == [
        "experiment-1-pipeline",
        "experiment-1-pipeline",
    ]
    assert all(call["resume"] == "allow" for call in wandb.init_calls)
    table = wandb.runs[1].logs[0]["tables/pipeline_sft_failed"]
    assert table.columns == list(PIPELINE_TABLE_COLUMNS)
    row = dict(zip(PIPELINE_TABLE_COLUMNS, table.data[0], strict=True))
    assert row["status"] == "failed"
    assert row["error"] == "WANDB_API_KEY=<redacted>"
    assert wandb.runs[1].artifacts[0].type == "stage-receipt"


def test_stage_finalizer_resumes_miles_run_and_curates_receipt(tmp_path: Path) -> None:
    receipt = tmp_path / "run_receipt.txt"
    receipt.write_text(
        "status=success\n"
        "wall_s=11005\n"
        "max_memory_used_mib=25737\n"
        "max_tokens_per_gpu=24576\n"
        "save_dir=/runs/checkpoints/sft_lora_r16\n",
        encoding="utf-8",
    )
    vram = tmp_path / "vram_usage.csv"
    vram.write_text("timestamp,index,memory.used\nnow,0,25737\n", encoding="utf-8")
    wandb = FakeWandb(seed_summary={"train/loss": 0.12, "_runtime": 11005})

    log_stage_finalization(
        wandb,
        project="glm47-pie-cpp-posttraining",
        entity=None,
        experiment_id="experiment-1",
        run_id="experiment-1-sft",
        group="experiment-1",
        stage="sft",
        status="success",
        mode="offline",
        timing_status="verified",
        receipt=receipt,
        artifact_paths=[vram],
        manifest_dir=tmp_path,
    )

    init = wandb.init_calls[0]
    run = wandb.runs[0]
    assert init["id"] == "experiment-1-sft"
    assert init["resume"] == "allow"
    assert init["job_type"] == "sft"
    assert init["config"]["stage_receipt"]["max_tokens_per_gpu"] == 24576
    assert run.summary["stage/status"] == "success"
    assert run.summary["stage/wall_s"] == 11005
    assert run.summary["stage/max_memory_used_mib"] == 25737
    assert run.summary["stage/checkpoint_or_adapter"] == "/runs/checkpoints/sft_lora_r16"
    assert run.summary["stage/final_metrics/train-loss"] == 0.12
    assert run.artifacts[0].type == "stage-run"
    assert {name for _, name in run.artifacts[0].files} >= {
        "run_receipt.txt",
        "vram_usage.csv",
        "experiment-1-sft.artifact_manifest.json",
    }


def test_stage_finalizer_publishes_native_miles_evidence_tables(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    checkpoint = tmp_path / "checkpoints" / "grpo_lora_r16"
    adapter = checkpoint / "iter_0000000" / "adapter"
    rollout_state = checkpoint / "rollout"
    adapter.mkdir(parents=True)
    rollout_state.mkdir()
    (adapter / "adapter_model.bin").write_bytes(b"adapter")
    (adapter / "training_state_rank0.pt").write_bytes(b"state")
    (rollout_state / "global_dataset_state_dict_0.pt").write_bytes(b"cursor")

    receipt = tmp_path / "run_receipt.txt"
    receipt.write_text(
        "status=success\n"
        "wall_s=653\n"
        "max_memory_used_mib=75957\n"
        "rollout_max_response_len=1536\n"
        "eval_max_response_len=1536\n"
        f"save_dir={checkpoint}\n",
        encoding="utf-8",
    )
    run_log = tmp_path / "run.log"
    run_log.write_text(
        "metrics.py:67 - perf 0: {'perf/rollout_time': 55.3, 'perf/tokens_per_gpu_per_sec': 400.8}\n"
        "log_utils.py:54 - passrate 0: {'passrate/pass@1': np.float64(0.25)}\n"
        "log_utils.py:54 - step 0: {'train/loss': -1e-9, 'train/grad_norm': 0.21}\n"
        "metrics.py:46 - eval 0: {'eval/pie_cpp': -0.41, 'eval/pie_cpp/truncated_ratio': 0.5}\n",
        encoding="utf-8",
    )

    dump_dir = tmp_path / "rollout_dumps"
    dump_dir.mkdir()
    (dump_dir / "grpo_0.pt").write_bytes(b"rollout")
    (dump_dir / "grpo_eval_0.pt").write_bytes(b"eval")

    def fake_load(path: Path, **_: Any) -> dict[str, Any]:
        is_eval = "_eval_" in Path(path).name
        response_length = 1536 if is_eval else 128
        return {
            "rollout_id": 0,
            "samples": [
                {
                    "group_index": 0,
                    "index": 0,
                    "prompt": "optimize this program",
                    "response": "```cpp\nint main(){}\n```",
                    "response_length": response_length,
                    "label": "pie-task",
                    "status": "completed",
                    "metadata": {"task_id": "task-1", "problem_id": "p1", "split": "validation"},
                    "prefix_cache_info": {"cached_tokens": 10, "total_prompt_tokens": 100},
                    "reward": {
                        "reward": 1.0,
                        "reason": "correct",
                        "all_tests_pass": True,
                        "format_valid": True,
                        "compile_error": False,
                        "sanitizer_error": False,
                        "timeout": False,
                        "tests_passed": 2,
                        "tests_total": 2,
                        "runtime_cpu_ns": 100,
                        "reference_runtime_cpu_ns": 120,
                        "runtime_speedup": 1.2,
                    },
                }
            ],
        }

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(load=fake_load))

    sync_dir = tmp_path / "weight_sync"
    sync_dir.mkdir()
    for sync, digest, total in ((1, "before", 10.0), (2, "after", 9.5)):
        for rank in range(2):
            (sync_dir / f"sync{sync:02d}_rank{rank}.json").write_text(
                json.dumps(
                    {
                        "sync": sync,
                        "rank": rank,
                        "n_tensors": 9741,
                        "sha256": digest,
                        "total_sum_abs": total,
                    }
                ),
                encoding="utf-8",
            )

    wandb = FakeWandb()
    log_stage_finalization(
        wandb,
        project="glm47-pie-cpp-posttraining",
        entity=None,
        experiment_id="experiment-1",
        run_id="experiment-1-grpo",
        group="experiment-1",
        stage="grpo",
        status="success",
        mode="offline",
        timing_status="verified",
        receipt=receipt,
        artifact_paths=[run_log],
        manifest_dir=tmp_path,
        run_log=run_log,
        rollout_dump_dir=dump_dir,
        sync_metrics_dir=sync_dir,
        checkpoint_dir=checkpoint,
    )

    run = wandb.runs[0]
    table_log = next(payload for payload in run.logs if "tables/stage_metrics" in payload)
    assert table_log["tables/stage_metrics"].columns == list(MILES_METRIC_TABLE_COLUMNS)
    assert table_log["tables/rollout_samples"].columns == list(MILES_SAMPLE_TABLE_COLUMNS)
    assert table_log["tables/eval_samples"].columns == list(MILES_SAMPLE_TABLE_COLUMNS)
    assert table_log["tables/reward_outcomes"].columns == list(MILES_REWARD_OUTCOME_TABLE_COLUMNS)
    assert table_log["tables/weight_sync"].columns == list(MILES_SYNC_TABLE_COLUMNS)
    assert table_log["tables/checkpoint_manifest"].columns == list(MILES_CHECKPOINT_TABLE_COLUMNS)
    assert run.summary["passrate/pass@1"] == 0.25
    assert run.summary["eval/pie_cpp"] == -0.41
    assert run.summary["sync/all_ranks_match"] is True
    assert run.summary["sync/updated_after_train"] is True
    assert run.summary["evidence/rollout_rows_total"] == 1
    assert run.summary["evidence/eval_rows_total"] == 1
    assert run.summary["evidence/checkpoint_file_count"] == 3
    assert wandb.init_calls[0]["config"]["proof_surface_schema"] == 2
    artifact_names = {name for _, name in run.artifacts[0].files}
    assert "experiment-1-grpo.checkpoint_manifest.json" in artifact_names
    assert "experiment-1-grpo.evidence_summary.json" in artifact_names


def test_comparison_logger_publishes_uplift_gate_and_table(tmp_path: Path) -> None:
    summaries = [_summary("base", 0.2), _summary("sft", 0.3), _summary("grpo", 0.5)]
    paths = []
    for summary in summaries:
        path = tmp_path / f"{summary['label']}.summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        paths.append(path)
    wandb = FakeWandb()

    log_comparison_run(
        wandb,
        project="glm47-pie-cpp-posttraining",
        entity=None,
        experiment_id="experiment-1",
        run_id="experiment-1-comparison",
        mode="offline",
        timing_status="verified",
        summaries=summaries,
        summary_paths=paths,
        output_dir=tmp_path,
    )

    run = wandb.runs[0]
    table = run.logs[0]["tables/checkpoint_comparison"]
    assert table.columns == list(COMPARISON_TABLE_COLUMNS)
    assert len(table.data) == 3
    assert run.summary["comparison/uplift_gate"]["passed"] is True
    assert run.summary["comparison/best_correct_and_faster"] == "grpo"
    assert run.artifacts[0].type == "comparison"
