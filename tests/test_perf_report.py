from __future__ import annotations

import json

from typer.testing import CliRunner

from w8_biayn.cli import app
from w8_biayn.perf.report import load_run_metrics, parse_metrics_text, summarize


def test_parse_metrics_text_extracts_namespaced_numbers():
    text = """
    step 5 | timing/step: 749.0343 timing/generate=670.2853
    timing/train_critic_and_policy: 40.5535
    reward/avg_pass_at_2 = 1.0000
    vllm/generation_throughput_tok_s: 12.5
    learning_rate=1e-6 trainer.flash_attn=false
    """
    metrics = parse_metrics_text(text)

    assert metrics["timing/step"] == 749.0343
    assert metrics["timing/generate"] == 670.2853
    assert metrics["reward/avg_pass_at_2"] == 1.0
    assert metrics["vllm/generation_throughput_tok_s"] == 12.5
    # Non-namespaced keys are ignored.
    assert "learning_rate" not in metrics
    assert "trainer.flash_attn" not in metrics


def test_summarize_computes_fractions():
    metrics = {
        "timing/step": 749.0343,
        "timing/generate": 670.2853,
        "timing/train_critic_and_policy": 40.5535,
    }
    summary = summarize(metrics)

    assert abs(summary["generation_fraction"] - 670.2853 / 749.0343) < 1e-9
    assert abs(summary["train_fraction"] - 40.5535 / 749.0343) < 1e-9


def test_load_run_metrics_reads_json_and_logs(tmp_path):
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "timing": {"step": 749.0, "generate": 670.0},
                "reward": {"avg_raw_reward": 0.3354},
                "config": {"lr": 1e-6},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "train.log").write_text("vllm/generation_throughput_tok_s=42.0\n", encoding="utf-8")

    metrics = load_run_metrics(tmp_path)

    assert metrics["timing/step"] == 749.0
    assert metrics["reward/avg_raw_reward"] == 0.3354
    assert metrics["vllm/generation_throughput_tok_s"] == 42.0
    # Config keys are not in a surfaced namespace.
    assert "config/lr" not in metrics


def test_cli_perf_report(tmp_path):
    (tmp_path / "metrics.json").write_text(
        json.dumps({"timing": {"step": 749.0, "generate": 670.0, "train_critic_and_policy": 40.0}}),
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["perf", "report", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "timing/step" in result.output
    assert "generation_fraction" in result.output


def test_cli_perf_report_rejects_missing_dir():
    result = CliRunner().invoke(app, ["perf", "report", "/no/such/run/dir"])

    assert result.exit_code != 0
    assert "run directory not found" in result.output


def test_cli_perf_profile_renders_profile(tmp_path):
    sa = tmp_path / "sa.json"
    sa.write_text(
        json.dumps({"type": "service_account", "project_id": "proj", "private_key": "X"}),
        encoding="utf-8",
    )
    out = tmp_path / "perf.sky.yaml"
    result = CliRunner().invoke(
        app,
        [
            "perf",
            "profile",
            "harbor-r3",
            "--optimization-profile",
            "a100-kernel-lab",
            "--credentials",
            str(sa),
            "-o",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "W8_BIAYN_KERNELS" in out.read_text(encoding="utf-8")
