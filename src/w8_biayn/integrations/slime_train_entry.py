from __future__ import annotations

import logging
import os
from pathlib import Path

import ray


LOGGER = logging.getLogger(__name__)


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _patch_final_train_sleep() -> None:
    if not _truthy_env("W8_SLIME_SKIP_FINAL_TRAIN_SLEEP"):
        return

    from slime.backends.megatron_utils import actor as actor_module

    original_train = actor_module.MegatronTrainRayActor.train
    original_save_model = actor_module.MegatronTrainRayActor.save_model

    if getattr(original_train, "_w8_final_sleep_patch", False):
        return

    def should_skip_final_sleep(self, rollout_id: int) -> bool:  # type: ignore[no-untyped-def]
        return (
            self.args.offload_train
            and self.role == "actor"
            and rollout_id == self.args.num_rollout - 1
            and _truthy_env("W8_SLIME_SKIP_FINAL_TRAIN_SLEEP")
        )

    def patched_train(self, rollout_id: int, rollout_data_ref, external_data=None):  # type: ignore[no-untyped-def]
        if self.args.debug_rollout_only:
            return None

        if self.args.offload_train:
            self.wake_up()

        with actor_module.timer("data_preprocess"):
            rollout_data = self._get_rollout_data(rollout_data_ref)

        try:
            if self.role == "critic":
                return self.train_critic(rollout_id, rollout_data)
            self.train_actor(rollout_id, rollout_data, external_data=external_data)
            return None
        finally:
            if self.args.offload_train:
                del rollout_data
                if should_skip_final_sleep(self, rollout_id):
                    self._w8_final_train_sleep_skipped = True
                    if actor_module.is_megatron_main_rank():
                        LOGGER.info("Skipping final actor sleep before checkpoint for one-rollout measurement.")
                else:
                    self.sleep()

    def patched_save_model(self, rollout_id: int, force_sync: bool = False) -> None:  # type: ignore[no-untyped-def]
        if not should_skip_final_sleep(self, rollout_id) or not getattr(
            self, "_w8_final_train_sleep_skipped", False
        ):
            return original_save_model(self, rollout_id, force_sync=force_sync)

        if self.args.debug_rollout_only:
            return

        if self.args.async_save:
            from megatron.training.async_utils import maybe_finalize_async_save

            maybe_finalize_async_save(blocking=True)

        actor_module.save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)

        if force_sync and self.args.async_save:
            maybe_finalize_async_save(blocking=True)

        if self.args.save_hf is not None and self.role == "actor":
            actor_module.save_hf_model_to_path(
                self.args,
                Path(self.args.save_hf.format(rollout_id=rollout_id)),
                self.model,
            )

    patched_train._w8_final_sleep_patch = True  # type: ignore[attr-defined]
    actor_module.MegatronTrainRayActor.train = patched_train
    actor_module.MegatronTrainRayActor.save_model = patched_save_model


def train(args):  # type: ignore[no-untyped-def]
    from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
    from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
    from slime.utils.misc import should_run_periodic_action

    configure_logger()
    placement_groups = create_placement_groups(args)
    init_tracking(args)

    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, placement_groups["rollout"])
    actor_model, critic_model = create_training_models(args, placement_groups, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def clear_or_keep_train_memory(actor_trains_this_step: bool) -> None:
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    def save(rollout_id: int) -> None:
        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if actor_trains_this_step:
            actor_model.save_model(rollout_id, force_sync=rollout_id == args.num_rollout - 1)
        if args.use_critic:
            critic_model.save_model(rollout_id, force_sync=rollout_id == args.num_rollout - 1)
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    skip_final_sleep = _truthy_env("W8_SLIME_SKIP_FINAL_TRAIN_SLEEP")

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        final_rollout = rollout_id == args.num_rollout - 1
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps

        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains_this_step:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(rollout_id)

        if skip_final_sleep and final_rollout:
            continue

        clear_or_keep_train_memory(actor_trains_this_step)
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


def main() -> None:
    from slime.utils.arguments import parse_args

    _patch_final_train_sleep()
    train(parse_args())


if __name__ == "__main__":
    main()
