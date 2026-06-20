"""Patch pinned SkyRL GRPO logging to emit dashboard-friendly health JSON."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

DEFAULT_SKYRL_TRAINER_PATH = Path("skyrl/train/trainer.py")
PATCH_MARKER = "Emit W8 GRPO learning-signal metrics for run-status dashboards."
IMPORT_MATH_RE = re.compile(r"(?m)^import math\s*$")
IMPORT_JSON_RE = re.compile(r"(?m)^import json\s*$")
ADVANTAGE_METRICS_RE = re.compile(
    r"(?m)^(?P<indent>[ \t]*)avg_advantages: float = valid_advantages\.mean\(\)\.item\(\)\n"
    r"(?P=indent)avg_advantages_abs: float = valid_advantages\.abs\(\)\.mean\(\)\.item\(\)\s*$"
)
LOG_PAYLOAD_RE = re.compile(
    r"(?m)^(?P<indent>[ \t]*)self\.tracker\.log\(log_payload, step=self\.global_step, commit=True\)\s*$"
)


def patch_skyrl_grpo_health(path: str | Path = DEFAULT_SKYRL_TRAINER_PATH) -> str:
    trainer_path = Path(path)
    text = trainer_path.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        return "already_patched"

    text = _ensure_json_import(text, trainer_path)

    advantage_match = ADVANTAGE_METRICS_RE.search(text)
    if advantage_match is None or LOG_PAYLOAD_RE.search(text) is None:
        raise SystemExit(
            "SkyRL GRPO health patch target not found in "
            f"{trainer_path}; pinned SkyRL source may have changed"
        )

    indent = advantage_match.group("indent")
    advantage_patch = "\n".join(
        [
            f"{indent}avg_advantages: float = valid_advantages.mean().item()",
            f"{indent}avg_advantages_abs: float = valid_advantages.abs().mean().item()",
            f"{indent}zero_advantage_token_fraction: float = (",
            f"{indent}    valid_advantages.abs().le(1e-8).float().mean().item()",
            f"{indent}    if valid_advantages.numel()",
            f"{indent}    else 0.0",
            f"{indent})",
            f"{indent}# {PATCH_MARKER}",
            f"{indent}_w8_return_sums = return_sums.detach().float().cpu().tolist()",
            f'{indent}_w8_uids = list(data.metadata["uids"])[: num_samples - pad_size]',
            f"{indent}if self.cfg.generator.step_wise_trajectories:",
            f"{indent}    _w8_keep_mask = is_last_step[: num_samples - pad_size].detach().cpu().tolist()",
            f"{indent}    _w8_group_items = [",
            f"{indent}        (_w8_uid, _w8_reward)",
            f"{indent}        for _w8_uid, _w8_reward, _w8_keep in zip(_w8_uids, _w8_return_sums, _w8_keep_mask)",
            f"{indent}        if _w8_keep",
            f"{indent}    ]",
            f"{indent}else:",
            f"{indent}    _w8_group_items = list(zip(_w8_uids, _w8_return_sums))",
            f"{indent}_w8_group_rewards = defaultdict(list)",
            f"{indent}for _w8_uid, _w8_reward in _w8_group_items:",
            f"{indent}    _w8_group_rewards[_w8_uid].append(float(_w8_reward))",
            f"{indent}_w8_group_vars = [float(np.var(_w8_values)) for _w8_values in _w8_group_rewards.values() if _w8_values]",
            f"{indent}reward_group_variance_mean: float = float(np.mean(_w8_group_vars)) if _w8_group_vars else 0.0",
            f"{indent}reward_group_variance_max: float = float(np.max(_w8_group_vars)) if _w8_group_vars else 0.0",
            f"{indent}zero_variance_group_count = sum(1 for _w8_var in _w8_group_vars if _w8_var <= 1e-12)",
            f"{indent}reward_group_count = len(_w8_group_vars)",
            f"{indent}zero_variance_group_fraction: float = (",
            f"{indent}    zero_variance_group_count / reward_group_count if reward_group_count else 0.0",
            f"{indent})",
        ]
    )
    text = text[: advantage_match.start()] + advantage_patch + text[advantage_match.end() :]

    log_match = LOG_PAYLOAD_RE.search(text)
    if log_match is None:
        raise SystemExit(
            "SkyRL GRPO health logging insertion target not found in "
            f"{trainer_path}; pinned SkyRL source may have changed"
        )
    log_indent = log_match.group("indent")
    log_patch = "\n".join(
        [
            f"{log_indent}_w8_health_metrics = {{",
            f"{log_indent}    str(_w8_key): (_w8_value.item() if hasattr(_w8_value, \"item\") else _w8_value)",
            f"{log_indent}    for _w8_key, _w8_value in log_payload.items()",
            f"{log_indent}    if isinstance(_w8_value, (int, float)) or hasattr(_w8_value, \"item\")",
            f"{log_indent}}}",
            f'{log_indent}logger.info("W8_GRPO_HEALTH " + json.dumps({{',
            f'{log_indent}    "schema_version": "w8-grpo-health-v1",',
            f'{log_indent}    "step": self.global_step,',
            f'{log_indent}    "metrics": _w8_health_metrics,',
            f"{log_indent}}}, sort_keys=True))",
            f"{log_indent}self.tracker.log(log_payload, step=self.global_step, commit=True)",
        ]
    )
    text = text[: log_match.start()] + log_patch + text[log_match.end() :]

    metadata_replacements = {
        '"avg_advantages_abs": avg_advantages_abs,': "\n".join(
            [
                '"avg_advantages_abs": avg_advantages_abs,',
                '                "zero_advantage_token_fraction": zero_advantage_token_fraction,',
                '                "reward_group_variance_mean": reward_group_variance_mean,',
                '                "reward_group_variance_max": reward_group_variance_max,',
                '                "zero_variance_group_fraction": zero_variance_group_fraction,',
                '                "reward_group_count": reward_group_count,',
            ]
        ),
        '"loss/avg_raw_advantages_abs": avg_advantages_abs,': "\n".join(
            [
                '"loss/avg_raw_advantages_abs": avg_advantages_abs,',
                '                "w8/zero_advantage_token_fraction": zero_advantage_token_fraction,',
                '                "w8/reward_group_variance_mean": reward_group_variance_mean,',
                '                "w8/reward_group_variance_max": reward_group_variance_max,',
                '                "w8/zero_variance_group_fraction": zero_variance_group_fraction,',
                '                "w8/reward_group_count": reward_group_count,',
            ]
        ),
    }
    for target, replacement in metadata_replacements.items():
        if target not in text:
            raise SystemExit(
                "SkyRL GRPO health patch metric insertion target not found in "
                f"{trainer_path}; pinned SkyRL source may have changed"
            )
        text = text.replace(target, replacement, 1)

    trainer_path.write_text(text, encoding="utf-8")
    return "patched"


def _ensure_json_import(text: str, trainer_path: Path) -> str:
    if IMPORT_JSON_RE.search(text):
        return text
    match = IMPORT_MATH_RE.search(text)
    if match is None:
        raise SystemExit(
            "SkyRL GRPO health patch target has no top-level import math in "
            f"{trainer_path}; pinned SkyRL source may have changed"
        )
    return text[: match.start()] + "import json\n" + text[match.start() :]


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch SkyRL GRPO trainer health logging.")
    parser.add_argument("path", nargs="?", default=str(DEFAULT_SKYRL_TRAINER_PATH), help="Path to SkyRL trainer.py")
    args = parser.parse_args()

    result = patch_skyrl_grpo_health(args.path)
    print(f"SkyRL GRPO health logging patch: {result} ({args.path})", flush=True)


if __name__ == "__main__":
    main()
