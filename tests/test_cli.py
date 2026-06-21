from __future__ import annotations

import json

import yaml
from typer.testing import CliRunner

from w8_biayn import cli as cli_module
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
    assert "domdiff" not in result.output
    assert "harbor" not in result.output
    assert "miniwob" not in result.output
    assert "webarena" not in result.output


def test_cli_config_render_cpp_smoke(tmp_path):
    credentials = tmp_path / "sa.json"
    write_credentials(credentials)
    output = tmp_path / "cpp.sky.yaml"

    result = CliRunner().invoke(
        app,
        [
            "config",
            "render",
            "cpp-smoke",
            "--credentials",
            str(credentials),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    rendered = output.read_text(encoding="utf-8")
    assert config["name"] == "w8-biayn-cpp-smoke"
    assert config["resources"]["infra"] == "gcp"
    assert config["envs"]["W8_BIAYN_MODEL"] == "zai-org/GLM-5.1"
    assert config["envs"]["CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"] == "/tmp/w8-gcp-service-account.json"
    assert config["envs"]["GOOGLE_CLOUD_PROJECT"] == "proj"
    assert "vllm" in config["run"]
    assert "SkyRL" in config["setup"]
    assert "domdiff" not in rendered.lower()
    assert "harbor" not in rendered.lower()


def test_cli_launch_cpp_smoke_dry_run_prints_sky_command(tmp_path):
    credentials = tmp_path / "sa.json"
    write_credentials(credentials)

    result = CliRunner().invoke(
        app,
        [
            "launch",
            "cpp-smoke",
            "--credentials",
            str(credentials),
            "--run-id",
            "rtest",
            "--detach-run",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "run_id: rtest" in result.output
    assert "sky launch -c w8-biayn-cpp-smoke-rtest" in result.output
    assert "--detach-run" in result.output
    assert ".w8-biayn/rendered/cpp-smoke.sky.yaml" in result.output


def test_cli_rejects_low_utilization_multinode_grpo_render(tmp_path):
    credentials = tmp_path / "sa.json"
    output = tmp_path / "cpp-grpo.sky.yaml"
    write_credentials(credentials)

    result = CliRunner().invoke(
        app,
        [
            "config",
            "render",
            "cpp-grpo",
            "--credentials",
            str(credentials),
            "--output",
            str(output),
            "--accelerators",
            "A100:8",
            "--num-nodes",
            "2",
            "--train-batch-size",
            "32",
            "--n-samples-per-prompt",
            "4",
            "--max-env-workers",
            "128",
        ],
    )

    assert result.exit_code != 0
    assert "effective samples per step 128 < 256" in result.output
    assert not output.exists()


def test_cli_allows_tuned_multinode_grpo_render(tmp_path):
    credentials = tmp_path / "sa.json"
    output = tmp_path / "cpp-grpo.sky.yaml"
    write_credentials(credentials)

    result = CliRunner().invoke(
        app,
        [
            "config",
            "render",
            "cpp-grpo",
            "--credentials",
            str(credentials),
            "--output",
            str(output),
            "--accelerators",
            "A100:8",
            "--num-nodes",
            "2",
            "--train-batch-size",
            "32",
            "--n-samples-per-prompt",
            "8",
            "--max-env-workers",
            "256",
            "--micro-train-batch-size-per-gpu",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    rendered = output.read_text(encoding="utf-8")
    assert "trainer.train_batch_size=32" in rendered
    assert "generator.n_samples_per_prompt=8" in rendered
    assert "environment.skyrl_gym.max_env_workers=256" in rendered
    assert "trainer.micro_train_batch_size_per_gpu=2" in rendered
    assert "trainer.algorithm.use_kl_loss=true" in rendered
    assert "trainer.algorithm.use_entropy_loss=true" in rendered
    assert "generator.inference_engine.gpu_memory_utilization=0.7" in rendered
    assert "trainer.logger=[console,mlflow]" in rendered
    assert "mlflow server" in rendered


def test_cli_render_accepts_console_only_tracking_backend(tmp_path):
    credentials = tmp_path / "sa.json"
    output = tmp_path / "cpp-grpo.sky.yaml"
    write_credentials(credentials)

    result = CliRunner().invoke(
        app,
        [
            "config",
            "render",
            "cpp-grpo",
            "--credentials",
            str(credentials),
            "--output",
            str(output),
            "--tracking-backend",
            "console",
        ],
    )

    assert result.exit_code == 0, result.output
    rendered = output.read_text(encoding="utf-8")
    assert "trainer.logger=console" in rendered
    assert "mlflow server" not in rendered


def test_cli_render_rejects_unknown_tracking_backend(tmp_path):
    credentials = tmp_path / "sa.json"
    output = tmp_path / "cpp-grpo.sky.yaml"
    write_credentials(credentials)

    result = CliRunner().invoke(
        app,
        [
            "config",
            "render",
            "cpp-grpo",
            "--credentials",
            str(credentials),
            "--output",
            str(output),
            "--tracking-backend",
            "sqlite",
        ],
    )

    assert result.exit_code != 0
    assert "unknown tracking backend" in result.output
    assert not output.exists()


def test_cli_rejects_small_disk_for_multinode_grpo_resume(tmp_path):
    credentials = tmp_path / "sa.json"
    output = tmp_path / "cpp-grpo.sky.yaml"
    write_credentials(credentials)

    result = CliRunner().invoke(
        app,
        [
            "config",
            "render",
            "cpp-grpo",
            "--credentials",
            str(credentials),
            "--output",
            str(output),
            "--accelerators",
            "A100:8",
            "--num-nodes",
            "2",
            "--train-batch-size",
            "32",
            "--n-samples-per-prompt",
            "8",
            "--max-env-workers",
            "256",
            "--resume-from",
            "latest",
            "--disk-size",
            "1024",
        ],
    )

    assert result.exit_code != 0
    assert "multi-node GRPO resume requires --disk-size 2048 or larger" in result.output
    assert not output.exists()


def test_cli_allows_low_utilization_multinode_grpo_with_explicit_override(tmp_path):
    credentials = tmp_path / "sa.json"
    output = tmp_path / "cpp-grpo.sky.yaml"
    write_credentials(credentials)

    result = CliRunner().invoke(
        app,
        [
            "config",
            "render",
            "cpp-grpo",
            "--credentials",
            str(credentials),
            "--output",
            str(output),
            "--accelerators",
            "A100:8",
            "--num-nodes",
            "2",
            "--train-batch-size",
            "32",
            "--n-samples-per-prompt",
            "4",
            "--max-env-workers",
            "128",
            "--allow-low-multinode-utilization",
        ],
    )

    assert result.exit_code == 0, result.output
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert config["envs"]["W8_BIAYN_ALLOW_LOW_MULTINODE_UTILIZATION"] == "true"
    assert config["envs"]["W8_BIAYN_EFFECTIVE_SAMPLES_PER_STEP"] == "128"


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


def test_cli_ops_run_status_emits_dashboard_json(tmp_path):
    credentials = tmp_path / "sa.json"
    out = tmp_path / "status.json"
    write_credentials(credentials)

    result = CliRunner().invoke(
        app,
        [
            "ops",
            "run-status",
            "--run-id",
            "rtest",
            "--credentials",
            str(credentials),
            "--dataset-gcs-prefix",
            "gs://proj-w8-biayn/datasets/cpp-perf/cpp-perf-v1/full-official/rdata/skyrl",
            "--pipeline",
            "cpp-sft",
            "--expected-sft-final-step",
            "1074",
            "--node-health",
            "--out",
            str(out),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == payload
    assert payload["schema_version"] == "w8-run-status-v1"
    assert payload["run_id"] == "rtest"
    assert payload["summary"]["current_pipeline"] == "cpp-sft"
    assert payload["summary"]["resources"]["total_instance_count"] == 0
    assert payload["dataset"]["gcs_prefix"].endswith("/rdata/skyrl")
    assert payload["dataset"]["checks"][0]["command"] == [
        "gcloud",
        "storage",
        "cat",
        "gs://proj-w8-biayn/datasets/cpp-perf/cpp-perf-v1/full-official/rdata/skyrl/_w8_data_manifest.json",
    ]
    assert payload["pipelines"][0]["pipeline"] == "cpp-sft"
    assert payload["pipelines"][0]["cluster"] == "w8-biayn-cpp-sft-rtest"
    assert payload["pipelines"][0]["phase"]["group"] == "unknown"
    assert payload["pipelines"][0]["progress"]["pipeline"] == "cpp-sft"
    assert payload["pipelines"][0]["speed_comparison"]["reason"] == "no_baseline_status"
    assert payload["pipelines"][0]["commands"]["queue"] == [
        "uv",
        "run",
        "w8-biayn",
        "ops",
        "queue",
        "w8-biayn-cpp-sft-rtest",
    ]
    assert payload["pipelines"][0]["artifacts"]["export"]["expected_final_step"] == 1074
    assert payload["pipelines"][0]["node_health"]["skipped"] is True
    assert payload["pipelines"][0]["checks"][0]["command"] == [
        "sky",
        "queue",
        "--output",
        "json",
        "w8-biayn-cpp-sft-rtest",
    ]
    assert str(credentials) in payload["cleanup"]["commands"]["dry_run"]
    assert payload["cleanup"]["commands"]["execute"][-1] == "--execute"


def test_cli_ops_run_status_loads_baseline_status_for_dashboard_json(tmp_path, monkeypatch):
    credentials = tmp_path / "sa.json"
    baseline = tmp_path / "single-node-status.json"
    write_credentials(credentials)
    baseline.write_text(
        json.dumps(
            {
                "run_id": "rsingle",
                "pipelines": [
                    {
                        "pipeline": "cpp-grpo",
                        "progress": {
                            "throughput": {"rollout_samples_per_second": 6.59},
                            "grpo_config": {"total_gpu_count": 8},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_build_run_status(**kwargs):
        captured.update(kwargs)
        return {
            "schema_version": "w8-run-status-v1",
            "run_id": kwargs["run_id"],
            "summary": {
                "current_pipeline": "cpp-grpo",
                "speed_comparison": {
                    "available": True,
                    "primary": {
                        "verdict": "slower",
                        "rollout_speedup_factor": 0.6889,
                        "cost_verdict": "cost_inefficient",
                    },
                },
            },
            "pipelines": [
                {
                    "pipeline": "cpp-grpo",
                    "speed_comparison": {
                        "available": True,
                        "primary": {
                            "source": str(baseline),
                            "verdict": "slower",
                            "rollout_speedup_factor": 0.6889,
                            "cost_verdict": "cost_inefficient",
                        },
                    },
                }
            ],
        }

    monkeypatch.setattr(cli_module, "build_run_status", fake_build_run_status)

    result = CliRunner().invoke(
        app,
        [
            "ops",
            "run-status",
            "--run-id",
            "rtest",
            "--credentials",
            str(credentials),
            "--pipeline",
            "cpp-grpo",
            "--baseline-status",
            str(baseline),
            "--check-retries",
            "2",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert captured["command_retries"] == 2
    assert captured["baseline_statuses"][0]["_status_source"] == str(baseline)
    assert captured["baseline_statuses"][0]["run_id"] == "rsingle"
    comparison = payload["pipelines"][0]["speed_comparison"]["primary"]
    assert comparison["source"] == str(baseline)
    assert comparison["rollout_speedup_factor"] == 0.6889
    assert comparison["verdict"] == "slower"
    assert comparison["cost_verdict"] == "cost_inefficient"


def test_cli_ops_metrics_dry_run_emits_mlflow_schema(tmp_path):
    credentials = tmp_path / "sa.json"
    write_credentials(credentials)

    result = CliRunner().invoke(
        app,
        [
            "ops",
            "metrics",
            "--run-id",
            "rtest",
            "--pipeline",
            "cpp-grpo",
            "--credentials",
            str(credentials),
            "--metric",
            "policy/policy_entropy",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == "w8-mlflow-metrics-v1"
    assert payload["source"] == "auto"
    assert payload["mlflow_db_uri"].endswith("/runs/cpp-perf/rtest/cpp-grpo/tracking/mlflow/mlflow.db")
    assert payload["metrics"]["reason"] == "dry_run"
    assert payload["checks"][0]["name"] == "ssh_tunnel:mlflow_api"
    assert payload["checks"][0]["command"][0:3] == ["ssh", "-N", "-L"]
    assert payload["checks"][1]["command"][0:3] == ["gcloud", "storage", "cp"]


def test_cli_ops_grpo_readiness_emits_json(tmp_path):
    credentials = tmp_path / "sa.json"
    rendered = tmp_path / "cpp-grpo.sky.yaml"
    out = tmp_path / "readiness.json"
    write_credentials(credentials)

    render_result = CliRunner().invoke(
        app,
        [
            "config",
            "render",
            "cpp-grpo",
            "--credentials",
            str(credentials),
            "--output",
            str(rendered),
            "--accelerators",
            "A100:8",
            "--num-nodes",
            "2",
            "--train-batch-size",
            "32",
            "--n-samples-per-prompt",
            "8",
            "--max-env-workers",
            "256",
            "--max-ckpts-to-keep",
            "8",
            "--hf-save-interval",
            "10000",
        ],
    )
    assert render_result.exit_code == 0, render_result.output

    result = CliRunner().invoke(
        app,
        [
            "ops",
            "grpo-readiness",
            "--rendered-config",
            str(rendered),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == "w8-grpo-readiness-v1"
    assert payload["overall"] == "pass"
    assert json.loads(out.read_text(encoding="utf-8")) == payload


def test_cli_multinode_grpo_launch_blocks_failed_readiness(tmp_path, monkeypatch):
    credentials = tmp_path / "sa.json"
    write_credentials(credentials)

    def fake_readiness(_path):
        return {
            "schema_version": "w8-grpo-readiness-v1",
            "overall": "fail",
            "checks": [
                {
                    "id": "network.gloo_concrete_interface",
                    "severity": "critical",
                    "ok": False,
                    "message": "bad gloo",
                }
            ],
        }

    def fail_run_command(*_args, **_kwargs):
        raise AssertionError("launch should block before backend command")

    monkeypatch.setattr(cli_module, "build_grpo_readiness", fake_readiness)
    monkeypatch.setattr(cli_module, "run_command", fail_run_command)

    result = CliRunner().invoke(
        app,
        [
            "launch",
            "cpp-grpo",
            "--credentials",
            str(credentials),
            "--run-id",
            "rtest",
            "--accelerators",
            "A100:8",
            "--num-nodes",
            "2",
            "--train-batch-size",
            "32",
            "--n-samples-per-prompt",
            "8",
            "--max-env-workers",
            "256",
            "--max-ckpts-to-keep",
            "8",
            "--hf-save-interval",
            "10000",
            "--dry-run",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["overall"] == "fail"
    assert payload["checks"][0]["id"] == "network.gloo_concrete_interface"


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


def test_cli_render_cpp_grpo_defaults_to_training_model_and_a100(tmp_path):
    credentials = tmp_path / "sa.json"
    write_credentials(credentials)
    output = tmp_path / "grpo.sky.yaml"

    result = CliRunner().invoke(
        app,
        [
            "config",
            "render",
            "cpp-grpo",
            "--credentials",
            str(credentials),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert config["resources"]["accelerators"] == "A100:8"
    assert config["envs"]["W8_BIAYN_MODEL"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert "python -m w8_biayn.integrations.skyrl_cpp_perf_main" in config["run"]
    assert "environment.env_class=cpp-perf" in config["run"]
    assert "trainer.micro_train_batch_size_per_gpu=1" in config["run"]
    assert "trainer.algorithm.use_kl_loss=true" in config["run"]
    assert "trainer.algorithm.kl_loss_coef=0.001" in config["run"]
    assert "trainer.algorithm.use_entropy_loss=true" in config["run"]
    assert "trainer.algorithm.entropy_loss_coef=0.001" in config["run"]
    assert "generator.inference_engine.gpu_memory_utilization=0.7" in config["run"]
    assert "w8-biayn cpp harness preflight" in config["run"]
    assert "-v /var/run/docker.sock:/var/run/docker.sock" in config["run"]
    assert "-v /tmp:/tmp" in config["run"]


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


def test_cli_data_build_skyrl_and_cache_dry_run(tmp_path):
    task_dir = tmp_path / "tasks"
    data_root = tmp_path / "skyrl"
    tsv = tmp_path / "pie.tsv"
    tsv.write_text(
        "problem_id\tv0\tv1\tcpu_time_v1\n"
        "p1\tint main(){return 0;}\tint main(){return 0;}\t100\n"
        "p2\tint main(){return 0;}\tint main(){return 0;}\t200\n",
        encoding="utf-8",
    )
    tests = tmp_path / "tests.json"
    tests.write_text(
        json.dumps(
            {
                "p1": {
                    "unit_tests": [{"input": "1\n", "expected": "1\n"}],
                    "hidden_tests": [{"input": "2\n", "expected": "2\n"}],
                    "coverage": {"line": 0.96, "branch": 0.86},
                },
                "p2": {
                    "unit_tests": [{"input": "3\n", "expected": "3\n"}],
                    "hidden_tests": [{"input": "4\n", "expected": "4\n"}],
                    "coverage": {"line": 0.96, "branch": 0.86},
                },
            }
        ),
        encoding="utf-8",
    )

    train_result = CliRunner().invoke(
        app,
        [
            "data",
            "pie",
            "build-tasks",
            "--pairs",
            str(tsv),
            "--tests-json",
            str(tests),
            "--out",
            str(task_dir / "train"),
            "--split",
            "train",
            "--limit",
            "1",
        ],
    )
    val_result = CliRunner().invoke(
        app,
        [
            "data",
            "pie",
            "build-tasks",
            "--pairs",
            str(tsv),
            "--tests-json",
            str(tests),
            "--out",
            str(task_dir / "validation"),
            "--split",
            "validation",
        ],
    )
    assert train_result.exit_code == 0, train_result.output
    assert val_result.exit_code == 0, val_result.output

    skyrl_result = CliRunner().invoke(
        app,
        ["data", "skyrl", "build", "--tasks-dir", str(task_dir), "--out", str(data_root)],
    )
    assert skyrl_result.exit_code == 0, skyrl_result.output
    assert (data_root / "grpo" / "train.parquet").exists()
    assert (data_root / "sft" / "train.jsonl").exists()

    credentials = tmp_path / "sa.json"
    write_credentials(credentials)
    upload_result = CliRunner().invoke(
        app,
        [
            "data",
            "cache",
            "upload",
            "--path",
            str(data_root),
            "--credentials",
            str(credentials),
            "--dry-run",
        ],
    )
    assert upload_result.exit_code == 0, upload_result.output
    assert "gcloud storage rsync --recursive" in upload_result.output
    assert "gs://proj-w8-biayn/datasets/cpp-perf/cpp-perf-v1/skyrl" in upload_result.output


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
