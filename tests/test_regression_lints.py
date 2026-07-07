"""Regression lints: one guard per bug the GPU smoke campaign paid to find.

Every lint names its incident and the anti-pattern that caused it. These are
deliberately blunt text-level checks over the load-bearing files -- if one
fires, read its docstring before "fixing" the assert: each of these cost real
money and hours on a paid A100 box.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LAUNCHER = (ROOT / "src/w8_biayn/cloud_launch.py").read_text(encoding="utf-8")
DRIVER = (ROOT / "src/w8_biayn/integrations/swe_agent_driver.py").read_text(encoding="utf-8")
HOOK = (ROOT / "src/w8_biayn/integrations/slime_swe_agent_cpp_perf.py").read_text(encoding="utf-8")
MILESTONE = (ROOT / "scripts/wandb_milestone.py").read_text(encoding="utf-8")
UPSTREAMS = (ROOT / "src/w8_biayn/upstreams.py").read_text(encoding="utf-8")
LANES = {
    "glm47": (ROOT / "examples/slime/glm47_cpp_perf/glm47_cpp_perf.sh").read_text(encoding="utf-8"),
    "agentic": (ROOT / "examples/slime/glm47_swe_agent_cpp_perf/glm47_swe_agent_cpp_perf.sh").read_text(
        encoding="utf-8"
    ),
}


def test_lint_orphan_box_needs_cluster_side_autostop() -> None:
    """Incident: a SIGKILLed launcher never ran its finally-teardown; a spot box
    idled UP ~6h (~$150). The teardown must not depend on the launcher process:
    sky.launch arms an idle autostop that TERMINATES (down=True)."""

    assert "idle_minutes_to_autostop=(options.idle_minutes_to_autostop or None)" in LAUNCHER
    assert "down=options.idle_minutes_to_autostop > 0" in LAUNCHER


def test_lint_no_network_host_in_the_training_container() -> None:
    """Incident: --network host made Ray's GCS bind the host IP while the lane
    pins node-ip 127.0.0.1 -> 'No available agent to submit job'. SWE-agent runs
    in-process (swerex LocalDeployment); nothing needs host networking."""

    assert "--network host" not in LAUNCHER
    assert '"deployment": {"type": "local"}' in DRIVER


def test_lint_sweagent_install_is_constraint_pinned() -> None:
    """Incident: `pip install -e sweagent` bumped tokenizers past transformers'
    pin and broke the trainer; --ignore-installed made it worse. The editable
    install must be constrained to the image's frozen environment."""

    install_lines = [line for line in LANES["agentic"].splitlines() if "pip install" in line]
    assert any("-e" in line and "--constraint" in line for line in install_lines)
    assert not any("--ignore-installed" in line for line in install_lines)


def test_lint_stage_runs_have_deterministic_ids_and_names() -> None:
    """Incident: pinned slime ignores --wandb-run-id and names every run after
    the group -- all stages of a launch displayed identically in W&B."""

    for lane in LANES.values():
        assert 'export WANDB_RUN_ID="${SLIME_WANDB_RUN_ID:-${RUN_ID}-${STAGE}}"' in lane
        assert 'export SLIME_WANDB_RUN_NAME="${WANDB_RUN_ID}"' in lane
    entry = (ROOT / "src/w8_biayn/integrations/slime_train_entry.py").read_text(encoding="utf-8")
    assert "init_tracking(args)\n    _apply_wandb_run_name()" in entry


def test_lint_milestones_never_log_raw_unix_timestamps() -> None:
    """Incident: pipeline/<event>_at = time.time() rendered as N one-point
    panels at Y~1.75e9. Events belong in tables/summary; curves get elapsed
    seconds on a define_metric'd step axis."""

    assert "elapsed_seconds" in MILESTONE
    assert 'run.log({f"pipeline/{event}_at": now})' not in MILESTONE
    assert "_unix\"] = now" not in MILESTONE


def test_lint_grpo_group_size_is_never_hardcoded_to_one() -> None:
    """Incident: SLIME_GRPO_N_SAMPLES_PER_PROMPT=1 hardcoded in the container
    script -> advantage = reward - mean(group of 1) = 0 for every sample ->
    zero learning signal, NaN kl."""

    assert "SLIME_GRPO_N_SAMPLES_PER_PROMPT=1" not in LAUNCHER
    assert '"$W8_GLM47_GRPO_N_SAMPLES_PER_PROMPT"' in LAUNCHER
    for lane in LANES.values():
        assert 'GRPO_N_SAMPLES_PER_PROMPT="${SLIME_GRPO_N_SAMPLES_PER_PROMPT:-8}"' in lane
        assert "$((GRPO_ROLLOUT_BATCH_SIZE * GRPO_N_SAMPLES_PER_PROMPT))" in lane


def test_lint_session_id_rides_in_the_request_body() -> None:
    """Incident-class: adapter session routing falls back bearer -> body; if
    LiteLLM ever drops Authorization on a custom api_base, turns file under
    'default' and finish_session drains empty."""

    assert '"user": sid' in DRIVER
    assert '"extra_body": {"metadata": {"session_id": sid}}' in DRIVER


def test_lint_repo_copies_are_unique_and_cleaned() -> None:
    """Incident: deterministic repo basenames collided at the shared FS root
    across stages (copytree without dirs_exist_ok) -> FileExistsError storms."""

    assert "_materialize_repo(work / _unique_repo_name(sid)" in DRIVER
    assert 'shutil.rmtree(Path("/") / repo.name, ignore_errors=True)' in DRIVER


def test_lint_swerex_local_upload_is_patched_idempotent() -> None:
    """Incident: SWE-agent uploads tool bundles to the FIXED /root/tools/{bundle}
    through swerex's bare copytree -- every episode after the first died with
    FileExistsError BEFORE its first model call (16/16 abort storms)."""

    assert "_patch_swerex_local_upload()" in DRIVER
    assert "dirs_exist_ok=True" in DRIVER


def test_lint_hf_export_gates_require_real_weight_shards() -> None:
    """Incident: an index.json-without-shards restore satisfied the existence
    gate and SGLang hung ~85 minutes loading a weightless model."""

    for lane in LANES.values():
        assert '! -name "*.index.json"' in lane
        assert '-o -name "model.safetensors.index.json"' not in lane
    assert "pruning weightless restored HF export" in LAUNCHER


def test_lint_persist_is_loud_readable_and_retried() -> None:
    """Incidents: (a) the persist rsync's >/dev/null || echo swallowed failures,
    poisoning GCS with shard-less exports; (b) the container writes exports as
    root, unreadable to the host-side gcloud rsync -- 'Unable to read file' on
    every shard while small json files uploaded fine."""

    assert 'chmod -R a+rX "$W8_REMOTE_RUN_ROOT"' in LAUNCHER
    assert "CHECKPOINT PERSIST FAILED after retries" in LAUNCHER
    persist_block = LAUNCHER.split("Loud, retried persist")[1].split("W8_REMOTE_EXPORT_ROOT")[0]
    assert ">/dev/null 2>&1" not in persist_block


def test_lint_every_trainer_sample_carries_round_number() -> None:
    """Incident: slime --log-multi-turn does sample.metadata['round_number'] (a
    direct KeyError) on every trainer-reaching sample; abort husks reach the
    trainer and killed a whole GRPO stage."""

    assert '"round_number": 0' in HOOK  # abort husks
    assert '"round_number": int(extract.steps or 0)' in HOOK  # graded episodes


def test_lint_exactly_one_trainable_sample_per_episode() -> None:
    """Incident: GLM's chat template strips <think> from history, so every turn
    re-tokenizes past the adapter's fork threshold -> 3 samples per episode ->
    48 rewards where GRPO's (prompts x n_samples) reshape expected 16."""

    assert "samples = samples[:1]" in HOOK
    assert "fork_samples_dropped" in HOOK
    assert 'SWE_FORK_MERGE_MAX_RESPONSE_TOKENS="${SLIME_SWE_FORK_MERGE_MAX_RESPONSE_TOKENS:-4096}"' in LANES["agentic"]


def test_lint_abort_reasons_keep_the_exception_message() -> None:
    """Incident: abort reasons carried only the exception TYPE
    ('exception:FileExistsError'); the colliding path was invisible everywhere
    and cost a full paid smoke to localize."""

    assert "error=str(exc)" in HOOK
    assert 'sample.metadata["abort_error"] = error[:500]' in HOOK


def test_lint_pinned_upstream_fetch_skips_network_when_pin_is_local() -> None:
    """Incident: `slime setup` ran an unconditional git fetch on a fresh cloud
    node; a transient GitHub outage killed the paid job even though the pinned
    commit was already in the workdir-synced clone."""

    assert "_commit_present(destination, upstream.pin)" in UPSTREAMS
    assert "rev-parse" in UPSTREAMS and "--verify" in UPSTREAMS


def test_lint_network_failures_must_bubble_up() -> None:
    """Incident: local DNS died mid-provision and the launch stalled in silent
    googleapiclient retries; the operator found out by asking. Launches must
    preflight reachability before spending and watchdog it throughout."""

    assert "net_health.preflight()" in LAUNCHER
    assert "net_health.NetWatchdog(on_event=_launch_event)" in LAUNCHER
    net = (ROOT / "src/w8_biayn/net_health.py").read_text(encoding="utf-8")
    assert "net_degraded" in net and "net_recovered" in net and "net_still_degraded" in net


def test_lint_vanished_cluster_is_terminal_not_retryable() -> None:
    """Incident: a spot preemption left the job-status poll retrying
    ClusterDoesNotExist for ~2 hours. A gone cluster (or persistent tracking
    failure) must return CLUSTER_LOST so the provisioning retry loop resumes."""

    assert 'return "CLUSTER_LOST"' in LAUNCHER
    assert "ClusterDoesNotExist" in LAUNCHER
    assert "max_status_failures" in LAUNCHER
    assert '_launch_event("cluster_lost"' in LAUNCHER


def test_lint_teardown_is_tag_keyed_and_reaper_exists() -> None:
    """Incident-class: paid boxes must always be findable and killable by run
    id, independent of any launcher state."""

    assert "labels=_resource_labels(options.run_id)" in LAUNCHER
    cli = (ROOT / "src/w8_biayn/cli.py").read_text(encoding="utf-8")
    assert 'ops_app.command("down-run")' in cli or "down-run" in cli
