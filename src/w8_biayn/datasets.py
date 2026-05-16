"""Dataset preparation for BrowserGym tasks using the rLLM record shape."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Literal

SYSTEM_PROMPT = """You are controlling a browser through BrowserGym.

At each turn, emit exactly one executable BrowserGym action wrapped in <action>...</action>.
Do not explain the action. Use the observation to decide the next action.
"""

Benchmark = Literal["miniwob", "webarena"]


def _import_browsergym(benchmark: Benchmark) -> None:
    if benchmark == "miniwob":
        import browsergym.miniwob  # noqa: F401
    elif benchmark == "webarena":
        try:
            import browsergym.async_webarena  # type: ignore # noqa: F401
        except Exception:
            import browsergym.webarena  # type: ignore # noqa: F401
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")


def _env_ids(benchmark: Benchmark) -> list[str]:
    import gymnasium as gym

    _import_browsergym(benchmark)
    prefixes = {
        "miniwob": ("browsergym/miniwob",),
        "webarena": ("browsergym_async/webarena", "browsergym/webarena"),
    }[benchmark]
    ids = [env_id for env_id in gym.envs.registry.keys() if env_id.startswith(prefixes)]
    if not ids:
        raise RuntimeError(f"No BrowserGym {benchmark} environments are registered")
    return sorted(ids)


def make_record(benchmark: Benchmark, env_id: str, split: str, index: int) -> dict:
    """Return one training/eval record in the rLLM BrowserGym shape."""
    return {
        "data_source": benchmark,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ""},
        ],
        "ability": "web",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "split": split,
            "index": index,
            "env_id": env_id,
        },
    }


def split_env_ids(env_ids: Iterable[str], train_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    ids = list(env_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    train_size = max(1, min(len(ids), int(len(ids) * train_ratio)))
    if train_size == len(ids) and len(ids) > 1:
        train_size -= 1
    return ids[:train_size], ids[train_size:]


def prepare_dataset(
    benchmark: Benchmark,
    output_dir: str | Path,
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
    limit: int | None = None,
) -> tuple[Path, Path]:
    """Prepare BrowserGym dataset parquet files."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Dataset preparation requires pandas and pyarrow") from exc

    ids = _env_ids(benchmark)
    if limit is not None:
        ids = ids[:limit]
    train_ids, val_ids = split_env_ids(ids, train_ratio, seed)

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    train_path = output_path / "train.parquet"
    val_path = output_path / "validation.parquet"
    metadata_path = output_path / "metadata.json"

    train_records = [make_record(benchmark, env_id, "train", i) for i, env_id in enumerate(train_ids)]
    val_records = [make_record(benchmark, env_id, "validation", i) for i, env_id in enumerate(val_ids)]
    pd.DataFrame(train_records).to_parquet(train_path)
    pd.DataFrame(val_records).to_parquet(val_path)
    metadata_path.write_text(
        json.dumps(
            {
                "benchmark": benchmark,
                "train_count": len(train_records),
                "validation_count": len(val_records),
                "seed": seed,
                "train_ratio": train_ratio,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return train_path, val_path

