from __future__ import annotations

import json

from typer.testing import CliRunner

from w8_biayn.cli import app


def write_credentials(path):
    path.write_text(json.dumps({"type": "service_account", "project_id": "proj"}), encoding="utf-8")


def test_cli_help_exposes_new_surface_only():
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "cpp" in result.output
    assert "data" in result.output
    assert "benchmarks" in result.output
    assert "gcp" in result.output
    assert "ops" in result.output
    assert "eval" in result.output
    assert "slime" in result.output
    assert "domdiff" not in result.output
    assert "harbor" not in result.output
    assert "miniwob" not in result.output
    assert "webarena" not in result.output


def test_cli_ops_wraps_backend_debug_commands(tmp_path):
    credentials = tmp_path / "sa.json"
    write_credentials(credentials)

    status = CliRunner().invoke(
        app,
        ["ops", "status", "--credentials", str(credentials), "--refresh", "--dry-run"],
    )
    logs = CliRunner().invoke(
        app,
        [
            "ops",
            "logs",
            "w8-biayn-cpp-grpo-rtest",
            "1",
            "--credentials",
            str(credentials),
            "--tail",
            "200",
            "--dry-run",
        ],
    )
    queue = CliRunner().invoke(
        app,
        ["ops", "queue", "w8-biayn-cpp-grpo-rtest", "--credentials", str(credentials), "--dry-run"],
    )
    cancel = CliRunner().invoke(
        app,
        ["ops", "cancel", "w8-biayn-cpp-grpo-rtest", "1", "--credentials", str(credentials), "--dry-run"],
    )
    down = CliRunner().invoke(
        app,
        ["ops", "down", "w8-biayn-cpp-grpo-rtest", "--credentials", str(credentials), "--dry-run"],
    )
    gpus = CliRunner().invoke(
        app,
        ["ops", "gpus", "A100", "--credentials", str(credentials), "--all-regions", "--dry-run"],
    )
    legacy_logs = CliRunner().invoke(
        app,
        [
            "logs",
            "w8-biayn-cpp-grpo-rtest",
            "1",
            "--credentials",
            str(credentials),
            "--tail",
            "50",
            "--status",
            "--dry-run",
        ],
    )
    legacy_down = CliRunner().invoke(
        app,
        ["down", "w8-biayn-cpp-grpo-rtest", "--credentials", str(credentials), "--no-yes", "--dry-run"],
    )

    assert status.exit_code == 0, status.output
    assert "sky status --refresh" in status.output
    assert logs.exit_code == 0, logs.output
    assert "sky logs --tail 200 --no-follow w8-biayn-cpp-grpo-rtest 1" in logs.output
    assert queue.exit_code == 0, queue.output
    assert "sky queue --output table w8-biayn-cpp-grpo-rtest" in queue.output
    assert cancel.exit_code == 0, cancel.output
    assert "sky cancel w8-biayn-cpp-grpo-rtest 1 --yes" in cancel.output
    assert down.exit_code == 0, down.output
    assert "sky down -y w8-biayn-cpp-grpo-rtest" in down.output
    assert gpus.exit_code == 0, gpus.output
    assert "sky gpus list A100 --infra gcp --all-regions --output table" in gpus.output
    assert legacy_logs.exit_code == 0, legacy_logs.output
    assert "sky logs --status --tail 50 --no-follow w8-biayn-cpp-grpo-rtest 1" in legacy_logs.output
    assert legacy_down.exit_code == 0, legacy_down.output
    assert "sky down w8-biayn-cpp-grpo-rtest" in legacy_down.output
    assert "sky down -y" not in legacy_down.output


def test_cli_measure_coverage_resumes_existing_report(tmp_path, monkeypatch):
    prepared = tmp_path / "pie-full"
    prepared.mkdir()
    (prepared / "train.jsonl").write_text(
        json.dumps({"problem_id": "p1", "v0": "slow", "v1": "fast", "cpu_time_v1": 100}) + "\n",
        encoding="utf-8",
    )
    report = prepared / "coverage-report.json"
    report.write_text(
        json.dumps(
            {
                "splits": ["train"],
                "problems": {
                    "p1": {
                        "coverage": {"line": 0.96, "branch": 0.86},
                        "tests_total": 2,
                        "tests_passed": 2,
                        "ok": True,
                        "reason": "ok",
                    }
                },
                "accepted": 1,
                "rejected": 0,
            }
        ),
        encoding="utf-8",
    )
    out = prepared / "coverage.json"

    def fail_measure(*_args, **_kwargs):
        raise AssertionError("already measured problem should be skipped")

    monkeypatch.setattr("w8_biayn.cli.measure_cpp_coverage", fail_measure)

    result = CliRunner().invoke(
        app,
        [
            "data",
            "pie",
            "measure-coverage",
            "--prepared-root",
            str(prepared),
            "--out",
            str(out),
            "--report-out",
            str(report),
            "--split",
            "train",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "resuming_completed: 1" in result.output
    assert json.loads(out.read_text(encoding="utf-8")) == {"p1": {"line": 0.96, "branch": 0.86}}


def test_cli_cpp_task_build_and_reward_dry_run(tmp_path):
    tsv = tmp_path / "pie.tsv"
    tsv.write_text("problem_id\tv0\tv1\tcpu_time_v1\np1\tslow\tfast\t100\n", encoding="utf-8")
    tests = tmp_path / "tests.json"
    tests.write_text(
        json.dumps(
            {
                "p1": {
                    "unit_tests": [{"input": "1\n", "expected": "1\n"}],
                    "hidden_tests": [{"input": "2\n", "expected": "2\n"}],
                    "coverage": {"line": 0.96, "branch": 0.86},
                }
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "tasks"

    build_result = CliRunner().invoke(
        app,
        ["cpp", "task", "build", "--pie-tsv", str(tsv), "--tests-json", str(tests), "--out", str(out)],
    )
    assert build_result.exit_code == 0, build_result.output
    task_path = out / "pie_cpp_000001.json"
    assert task_path.exists()

    candidate = tmp_path / "candidate.cpp"
    candidate.write_text("int main(){return 0;}\n", encoding="utf-8")
    harness_result = CliRunner().invoke(
        app,
        ["cpp", "harness", "run", "--task", str(task_path), "--candidate", str(candidate), "--dry-run"],
    )
    assert harness_result.exit_code == 0, harness_result.output
    assert "C++ performance harness dry run" in harness_result.output

    model_output = tmp_path / "output.md"
    model_output.write_text("<reasoning>x</reasoning>\n```cpp\nint main(){return 0;}\n```\n", encoding="utf-8")
    reward_result = CliRunner().invoke(
        app,
        ["cpp", "reward", "score", "--task", str(task_path), "--model-output", str(model_output), "--dry-run"],
    )
    assert reward_result.exit_code == 0, reward_result.output
    assert "valid_format: True" in reward_result.output

    preflight_result = CliRunner().invoke(app, ["cpp", "harness", "preflight", "--dry-run"])
    assert preflight_result.exit_code == 0, preflight_result.output
    assert "C++ runtime preflight dry run" in preflight_result.output
    assert "python3 /tmp/w8_runtime_bench.py" in preflight_result.output


def test_cli_benchmarks_list_uses_cpp_ladder():
    result = CliRunner().invoke(app, ["benchmarks", "list"])

    assert result.exit_code == 0
    assert "pie-one-smoke" in result.output
    assert "harbor" not in result.output.lower()


def test_cli_eval_cpp_aggregates_records(tmp_path):
    records = tmp_path / "base.jsonl"
    records.write_text(
        '{"task_id":"t1","reward":1.2,"reason":"correct","all_tests_pass":true,'
        '"compile_error":false,"sanitizer_error":false,"timeout":false,'
        '"runtime_cpu_ns":50,"reference_runtime_cpu_ns":100}\n',
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["eval", "cpp", "--records", f"base={records}"])

    assert result.exit_code == 0, result.output
    assert '"best_correct_and_faster": "base"' in result.output
    assert '"uplift_gate"' in result.output
    assert '"verdict": "insufficient_labels"' in result.output


def test_cli_eval_raw_report_writes_markdown_assets_and_curves(tmp_path):
    run_id = "rtest"
    run_root = tmp_path / "runs" / run_id
    eval_dir = run_root / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "base.records.jsonl").write_text(
        '{"task_id":"t1","problem_id":"p1","reward":-1.0,"reason":"invalid_format",'
        '"all_tests_pass":false,"compile_error":false,"sanitizer_error":false,"timeout":false,'
        '"runtime_cpu_ns":null,"reference_runtime_cpu_ns":null,"tests_passed":0,"tests_total":0}\n',
        encoding="utf-8",
    )
    (eval_dir / "sft.records.jsonl").write_text(
        '{"task_id":"t1","problem_id":"p1","reward":0.2,"reason":"recoverable_format_missing_runtime",'
        '"all_tests_pass":true,"compile_error":false,"sanitizer_error":false,"timeout":false,'
        '"runtime_cpu_ns":null,"reference_runtime_cpu_ns":null,"tests_passed":2,"tests_total":2}\n',
        encoding="utf-8",
    )
    (eval_dir / "grpo.records.jsonl").write_text(
        '{"task_id":"t1","problem_id":"p1","reward":1.5,"reason":"correct",'
        '"all_tests_pass":true,"compile_error":false,"sanitizer_error":false,"timeout":false,'
        '"runtime_cpu_ns":50,"reference_runtime_cpu_ns":100,"tests_passed":2,"tests_total":2}\n',
        encoding="utf-8",
    )
    (run_root / "metrics.api.json").write_text(
        json.dumps(
            {
                "schema_version": "w8-mlflow-metrics-v1",
                "metrics": {
                    "backend": "mlflow_api",
                    "tracking_state": "metrics_available",
                    "latest_step": 2,
                    "metric_count": 2,
                    "metric_row_count": 4,
                    "series": {
                        "loss/avg_final_rewards": [
                            {"step": 1, "timestamp_ms": 1000, "value": 0.1},
                            {"step": 2, "timestamp_ms": 2000, "value": 0.5},
                        ],
                        "policy/grad_norm": [
                            {"step": 1, "timestamp_ms": 1000, "value": 0.2},
                            {"step": 2, "timestamp_ms": 2000, "value": 0.1},
                        ],
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_root / "status.json").write_text(
        json.dumps(
            {
                "dataset": {"gcs_prefix": "gs://bucket/datasets/cpp-perf/cpp-perf-v1/full-official/rdata/skyrl"},
                "pipelines": [
                    {
                        "pipeline": "cpp-grpo",
                        "artifacts": {
                            "checkpoint": {
                                "latest": {"prefix": "gs://bucket/runs/cpp-perf/rtest/cpp-grpo/ckpts/global_step_1"}
                            }
                        },
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "raw.md"
    assets = tmp_path / "raw_assets"

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "raw-report",
            "--run-id",
            run_id,
            "--run-root",
            str(run_root),
            "--out",
            str(out),
            "--assets-dir",
            str(assets),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Raw Run Evidence: `rtest`" in text
    assert "uplift_gate" in text
    assert "Held-out eval outcome rates" in text
    assert (assets / "uplift-summary-recomputed.json").exists()
    assert (assets / "eval_summary.csv").exists()
    assert (assets / "missing_runtime_tasks.csv").read_text(encoding="utf-8").count("t1") == 1
    assert "<svg" in (assets / "eval_outcome_rates.svg").read_text(encoding="utf-8")
    assert "<svg" in (assets / "train_reward_pass.svg").read_text(encoding="utf-8")
    summary = json.loads((assets / "uplift-summary-recomputed.json").read_text(encoding="utf-8"))
    assert summary["uplift_gate"]["verdict"] == "held_out_lift_but_gate_failed"


def test_cli_slime_doctor_reports_missing_clone(tmp_path):
    result = CliRunner().invoke(app, ["slime", "doctor", "--repo-root", str(tmp_path)])

    assert result.exit_code == 1
    assert "SLIME upstream clone is absent" in result.output
    assert "uv run w8-biayn upstreams clone slime" in result.output


def test_cli_slime_doctor_accepts_expected_layout(tmp_path):
    root = tmp_path / ".cache" / "upstreams" / "slime"
    root.mkdir(parents=True)
    for filename in ("README.md", "train.py", "train_async.py"):
        (root / filename).write_text("placeholder\n", encoding="utf-8")
    for dirname in ("slime", "examples", "docs"):
        (root / dirname).mkdir()

    result = CliRunner().invoke(app, ["slime", "doctor", "--repo-root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "SLIME upstream doctor" in result.output
    assert "train_async.py" in result.output


def test_cli_slime_setup_generates_launcher_and_bootstrap(tmp_path, monkeypatch):
    root = tmp_path / ".cache" / "upstreams" / "slime"
    root.mkdir(parents=True)
    for filename in ("README.md", "train.py", "train_async.py"):
        (root / filename).write_text("placeholder\n", encoding="utf-8")
    for dirname in ("slime", "examples", "docs"):
        (root / dirname).mkdir()

    monkeypatch.setattr("w8_biayn.cli.upstreams.clone_or_update", lambda name, repo_root='.', dry_run=False: root)

    result = CliRunner().invoke(app, ["slime", "setup", "--repo-root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "SLIME setup files generated" in result.output
    assert "run-container.sh" in result.output
    assert "bootstrap-inside-container.sh" in result.output
    assert (tmp_path / ".w8-biayn" / "slime" / "run-container.sh").exists()
    assert (tmp_path / ".w8-biayn" / "slime" / "bootstrap-inside-container.sh").exists()
