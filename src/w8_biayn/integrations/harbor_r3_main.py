"""rLLM/Tinker Harbor DOMDiff training entrypoint for `w8-biayn launch r3`."""

from __future__ import annotations

from typing import Any

import hydra
from omegaconf import OmegaConf

from w8_biayn.harbor.agent_flow import HarborPromptAgentFlow
from w8_biayn.harbor.evaluator import HarborDomdiffEvaluator
from w8_biayn.harbor.tasks import register_harbor_dataset


def _to_container(value: Any, default: Any) -> Any:
    if value is None:
        return default
    return OmegaConf.to_container(value, resolve=True)


def _tinker_backend_cls():
    from rllm.trainer.tinker.tinker_backend import TinkerBackend

    return TinkerBackend


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config) -> None:
    from rllm.data.dataset import DatasetRegistry
    from rllm.experimental.unified_trainer import UnifiedTrainer

    dataset_name = OmegaConf.select(config, "w8.harbor.dataset_name", default="w8_harbor_domdiff_smoke")
    task_root = OmegaConf.select(config, "w8.harbor.task_root", default=None)
    task_ids = _to_container(OmegaConf.select(config, "w8.harbor.task_ids", default=None), None)
    max_samples = OmegaConf.select(config, "w8.harbor.max_samples", default=None)
    chromiumrl_url = str(OmegaConf.select(config, "w8.harbor.chromiumrl_url", default="") or "")
    oracle = _as_bool(OmegaConf.select(config, "w8.harbor.oracle", default=True))
    keep_containers = _as_bool(OmegaConf.select(config, "w8.harbor.keep_containers", default=False))

    if task_ids is not None:
        task_ids = tuple(str(task_id) for task_id in task_ids)

    if DatasetRegistry.dataset_exists(dataset_name, "train"):
        train_dataset = DatasetRegistry.load_dataset(dataset_name, "train")
        val_dataset = DatasetRegistry.load_dataset(dataset_name, "test")
    else:
        train_dataset, val_dataset = register_harbor_dataset(
            dataset_name,
            task_root=task_root,
            task_ids=task_ids,
        )

    if max_samples is not None:
        train_dataset = train_dataset.select(range(min(int(max_samples), len(train_dataset))))
        val_dataset = val_dataset.select(range(min(int(max_samples), len(val_dataset))))

    trainer = UnifiedTrainer(
        backend_cls=_tinker_backend_cls(),
        config=config,
        agent_flow=HarborPromptAgentFlow(),
        evaluator=HarborDomdiffEvaluator(
            chromiumrl_url=chromiumrl_url,
            oracle=oracle,
            keep_containers=keep_containers,
        ),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        backend_name="tinker",
    )
    try:
        trainer.fit()
    finally:
        trainer.shutdown()


if __name__ == "__main__":
    main()
