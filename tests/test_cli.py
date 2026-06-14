from __future__ import annotations

import json

import yaml
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
    assert "vllm" in config["run"]
    assert "SkyRL" in config["setup"]
    assert "domdiff" not in rendered.lower()
    assert "harbor" not in rendered.lower()


def test_cli_launch_cpp_smoke_dry_run_prints_sky_command(tmp_path):
    credentials = tmp_path / "sa.json"
    write_credentials(credentials)

    result = CliRunner().invoke(
        app,
        ["launch", "cpp-smoke", "--credentials", str(credentials), "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert "sky launch -c w8-biayn-cpp-smoke" in result.output
    assert ".w8-biayn/rendered/cpp-smoke.sky.yaml" in result.output


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
    assert "C++ perf-counter preflight dry run" in preflight_result.output
    assert "instructions:u" in preflight_result.output


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
