#!/usr/bin/env python3
"""Fail-closed verification of the completed Luna Low fixed26 evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = Path(
    os.environ.get(
        "LUNA_VERIFY_ARTIFACT_ROOT",
        ROOT / "artifacts" / "gpt56-luna-low-fixed26-v2contract-20260803",
    )
).resolve()
EXPECTED_MODEL = "gpt-5.6-luna"
EXPECTED_ALIAS = "openai/gpt-5.6-luna"
EXPECTED_EFFORT = os.environ.get("LUNA_EXPECTED_REASONING_EFFORT", "low")
EXPECTED_TREE = os.environ.get(
    "LUNA_EXPECTED_TREE_SHA256",
    "c0541864071b5df862e735aa6063d121c8154df3dbd652ef3b9d2ce101ba515e",
)
EXPECTED_SCORER = os.environ.get(
    "LUNA_MODAL_SCORER_APP_NAME", "luna-low-fixed26-cpu-scorer-v3-boost"
)
EXPECTED_SCORER_VARIANT = os.environ.get("LUNA_FIXED26_SCORER_VARIANT", "v2")
EXPECTED_SCORER_PROVIDER = os.environ.get(
    "LUNA_VERIFY_SCORER_PROVIDER", "Modal"
)
EXPECTED_CONTRACT = os.environ.get(
    "LUNA_EXPECTED_OVERLAY_VERSION", "fixed26-contract-v2"
)
LABEL_PREFIX = os.environ.get("LUNA_VERIFY_LABEL_PREFIX", "gpt56-luna-low-fixed26-v2c")
STRICT_ONE_SHOT = os.environ.get("LUNA_VERIFY_STRICT_ONE_SHOT") == "1"
EXPECTED_LOCAL_REJECTED_PREFLIGHTS = int(
    os.environ.get("LUNA_VERIFY_EXPECTED_LOCAL_REJECTED_PREFLIGHTS", "0")
)
SINGLE_ONLY = os.environ.get("LUNA_VERIFY_SINGLE_ONLY") == "1"
FEEDBACK_ONLY = os.environ.get("LUNA_VERIFY_FEEDBACK_ONLY") == "1"
REQUIRE_AUTHORIZED_FEEDBACK = (
    os.environ.get("LUNA_VERIFY_REQUIRE_AUTHORIZED_FEEDBACK") == "1"
)
SINGLE_TRIALS = int(os.environ.get("LUNA_VERIFY_SINGLE_TRIALS", "4"))
TRANSPORT = os.environ.get("LUNA_VERIFY_TRANSPORT", "codex")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def receipt_order(path: Path) -> int:
    return int(path.stem.rsplit("-", 2)[-2])


def main() -> int:
    require(SINGLE_TRIALS >= 1, "single-trial count must be positive")
    require(not (SINGLE_ONLY and FEEDBACK_ONLY), "single-only and feedback-only conflict")
    labels = [] if FEEDBACK_ONLY else [
        f"{LABEL_PREFIX}-single-a{i}" for i in range(1, SINGLE_TRIALS + 1)
    ]
    if not SINGLE_ONLY:
        labels += [f"{LABEL_PREFIX}-feedback-a{i}" for i in range(1, 5)]
    expected_tries = {label: (1 if "single" in label else 2) for label in labels}
    task_set: set[str] | None = None
    attempt_summaries = []
    expected_matrix_calls = 0
    total_modal_receipts = 0

    for label in labels:
        attempt = read_json(ARTIFACT / "attempt-receipts" / f"{label}.json")
        require(attempt.get("status") == "complete" and attempt.get("returncode") == 0,
                f"attempt did not complete cleanly: {label}")
        require(attempt.get("actual_upstream_model") == EXPECTED_MODEL, f"model drift: {label}")
        require(attempt.get("reasoning_effort") == EXPECTED_EFFORT, f"effort drift: {label}")
        require(attempt.get("tries") == expected_tries[label], f"tries drift: {label}")

        practice = ARTIFACT / "benchmark" / label / "cpp" / "exercises" / "practice"
        result_paths = sorted(practice.glob("*/.aider.results.json"))
        require(len(result_paths) == 26, f"expected 26 results for {label}, got {len(result_paths)}")
        results = {read_json(path)["testcase"]: read_json(path) for path in result_paths}
        require(len(results) == 26, f"duplicate task identity in {label}")
        if task_set is None:
            task_set = set(results)
        require(set(results) == task_set, f"task set drift in {label}")

        modal_by_task: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
        for path in (ARTIFACT / "modal-receipts" / label).glob("*.json"):
            receipt = read_json(path)
            modal_by_task[receipt["task"]].append((path, receipt))
        require(set(modal_by_task) == set(results), f"Modal task set drift in {label}")

        pass_turn_1 = 0
        pass_cumulative = 0
        outcome_count = 0
        model_error_outputs = 0
        malformed_responses = 0
        for task, result in results.items():
            outcomes = result.get("tests_outcomes")
            require(isinstance(outcomes, list) and 1 <= len(outcomes) <= expected_tries[label],
                    f"invalid outcomes for {label}/{task}: {outcomes!r}")
            require(result.get("model") == EXPECTED_ALIAS, f"Aider model drift: {label}/{task}")
            require(result.get("reasoning_effort") == EXPECTED_EFFORT,
                    f"result effort drift: {label}/{task}")
            require(result.get("test_timeouts") == 0, f"test timeout: {label}/{task}")
            error_outputs = result.get("num_error_outputs")
            malformed = result.get("num_malformed_responses")
            require(
                isinstance(error_outputs, int) and 0 <= error_outputs <= len(outcomes),
                f"invalid model-error count: {label}/{task}: {error_outputs!r}",
            )
            require(
                isinstance(malformed, int) and 0 <= malformed <= error_outputs,
                f"invalid malformed-response count: {label}/{task}: {malformed!r}",
            )
            if not STRICT_ONE_SHOT:
                require(error_outputs == 0, f"model error output: {label}/{task}")
                require(malformed == 0, f"malformed response: {label}/{task}")
            elif error_outputs:
                # With reflections disabled, a malformed first response is a
                # terminal model miss, not invalid evidence and never grounds
                # for a replacement completion.
                require(not all(outcomes), f"malformed output scored as pass: {label}/{task}")
            model_error_outputs += error_outputs
            malformed_responses += malformed
            require(result.get("num_exhausted_context_windows") == 0,
                    f"context exhaustion: {label}/{task}")
            if STRICT_ONE_SHOT:
                require(result.get("num_user_asks") == 0,
                        f"hidden harness interaction: {label}/{task}")
                require(len(result.get("chat_hashes", [])) == len(outcomes),
                        f"model-call/test-attempt mismatch: {label}/{task}")

            ordered = [value for _, value in sorted(modal_by_task[task], key=lambda item: receipt_order(item[0]))]
            require(len(ordered) == len(outcomes),
                    f"scorer-call count mismatch for {label}/{task}: {len(ordered)} != {len(outcomes)}")
            statuses = [receipt.get("status") == "passed" for receipt in ordered]
            require(statuses == outcomes,
                    f"scorer outcome mismatch for {label}/{task}: {statuses} != {outcomes}")
            for receipt in ordered:
                environment = receipt.get("environment", {})
                source_tree = receipt.get("source_tree", {})
                require(environment.get("app") == EXPECTED_SCORER, f"scorer drift: {label}/{task}")
                require(
                    environment.get("provider") == EXPECTED_SCORER_PROVIDER,
                    f"scorer provider drift: {label}/{task}",
                )
                if EXPECTED_SCORER_VARIANT != "v2":
                    require(
                        receipt.get("contract_variant") == EXPECTED_SCORER_VARIANT,
                        f"scorer contract drift: {label}/{task}",
                    )
                require(environment.get("network_blocked") is True,
                        f"scorer network not blocked: {label}/{task}")
                require(bool(environment.get("boost_date_time_package")),
                        f"Boost missing: {label}/{task}")
                require(source_tree.get("sha256") == EXPECTED_TREE and
                        source_tree.get("expected_sha256") == EXPECTED_TREE,
                        f"source tree drift: {label}/{task}")

            final_receipt = ordered[-1]
            for candidate in final_receipt.get("candidate_files", []):
                candidate_path = practice / task / candidate["path"]
                require(candidate_path.is_file(), f"missing final candidate: {label}/{task}/{candidate['path']}")
                require(sha256(candidate_path) == candidate["sha256"],
                        f"candidate hash mismatch: {label}/{task}/{candidate['path']}")

            pass_turn_1 += int(bool(outcomes[0]))
            pass_cumulative += int(any(outcomes))
            outcome_count += len(outcomes)

        expected_matrix_calls += outcome_count
        modal_count = sum(len(values) for values in modal_by_task.values())
        total_modal_receipts += modal_count
        attempt_summaries.append({
            "label": label,
            "tries": expected_tries[label],
            "tasks": 26,
            "pass_turn_1": pass_turn_1,
            "pass_cumulative": pass_cumulative,
            "model_calls": outcome_count,
            "modal_score_receipts": modal_count,
            "model_error_outputs": model_error_outputs,
            "malformed_responses": malformed_responses,
        })

    call_receipts = sorted((ARTIFACT / "transport" / "calls").glob("call-*/receipt.json"))
    if STRICT_ONE_SHOT:
        require(
            len(call_receipts)
            == expected_matrix_calls + 1 + EXPECTED_LOCAL_REJECTED_PREFLIGHTS,
            "strict receipt count mismatch: "
            f"{len(call_receipts)} != {expected_matrix_calls} scored + 1 smoke + "
            f"{EXPECTED_LOCAL_REJECTED_PREFLIGHTS} locally rejected preflight(s)",
        )
    else:
        # Aider's default whole-edit harness can make multiple candidate-model
        # calls before one compile/test invocation.
        require(len(call_receipts) > expected_matrix_calls,
                f"expected more transport than scorer calls: {len(call_receipts)} <= {expected_matrix_calls}")
    total_usage = defaultdict(int)
    accepted_call_receipts = []
    rejected_local_preflights = []
    authorized_feedback_calls = []
    for path in call_receipts:
        call = read_json(path)
        if call.get("status") == "rejected":
            # Preserve and classify a locally rejected preflight instead of
            # deleting it from the evidence ledger. It is not a model call.
            leak = call.get("leak_audit", {})
            require(
                call.get("upstream_http_attempts") == 0
                and call.get("model_returned") is None
                and call.get("provider_returned") is None
                and leak.get("message_count") == 1
                and leak.get("prompt_chars", 10_000) < 200
                and call.get("violations")
                == [f"unexpected Aider model '{EXPECTED_ALIAS}'"],
                f"rejected receipt is not a harmless local preflight: {path.parent.name}",
            )
            rejected_local_preflights.append(path.parent.name)
            continue
        require(call.get("status") == "accepted", f"unknown transport status: {path.parent.name}")
        accepted_call_receipts.append(path)
        require(call.get("reasoning_effort") == EXPECTED_EFFORT,
                f"transport effort drift: {path.parent.name}")
        require(call.get("violations") == [], f"transport violation: {path.parent.name}")
        if TRANSPORT == "openrouter":
            require(call.get("model_requested") == EXPECTED_ALIAS,
                    f"OpenRouter request model drift: {path.parent.name}")
            require(call.get("model_returned") in {
                EXPECTED_ALIAS, "openai/gpt-5.6-luna-20260709"
            }, f"OpenRouter response model drift: {path.parent.name}")
            require(call.get("provider_returned") == "OpenAI",
                    f"OpenRouter provider drift: {path.parent.name}")
            require(call.get("upstream_http_attempts") == 1,
                    f"not exactly one upstream call: {path.parent.name}")
            require(call.get("tools_present") is False and
                    call.get("plugins_present") is False and
                    call.get("web_search_enabled") is False,
                    f"external capability enabled: {path.parent.name}")
            require(call.get("provider_fallbacks_allowed") is False and
                    call.get("provider_data_collection") == "deny" and
                    call.get("provider_zdr_required") is False,
                    f"routing privacy drift: {path.parent.name}")
            require(call.get("leak_audit", {}).get("status") == "passed" and
                    call.get("leak_audit", {}).get("violations") == [],
                    f"hidden-data audit failed: {path.parent.name}")
            feedback_authorization = call.get("feedback_authorization", {})
            if feedback_authorization.get("present"):
                require(
                    feedback_authorization.get("status") == "authorized"
                    and feedback_authorization.get("task")
                    and feedback_authorization.get("matched_scorer_receipts")
                    and feedback_authorization.get("matched_candidate_files"),
                    f"unbound feedback call: {path.parent.name}",
                )
                authorized_feedback_calls.append(path.parent.name)
        else:
            require(call.get("model") == EXPECTED_MODEL,
                    f"transport model drift: {path.parent.name}")
            require(call.get("codex_exit_status") == 0,
                    f"Codex failure: {path.parent.name}")
            require(call.get("timed_out") is False,
                    f"Codex timeout: {path.parent.name}")
            argv = call.get("codex_argv", [])
            require(EXPECTED_MODEL in argv, f"model absent from argv: {path.parent.name}")
        for key, value in call.get("usage", {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            total_usage[key] += value

    require(
        len(rejected_local_preflights) == EXPECTED_LOCAL_REJECTED_PREFLIGHTS,
        "locally rejected preflight count mismatch: "
        f"{len(rejected_local_preflights)} != {EXPECTED_LOCAL_REJECTED_PREFLIGHTS}",
    )
    require(
        len(accepted_call_receipts) == expected_matrix_calls + 1,
        "accepted model-call count mismatch: "
        f"{len(accepted_call_receipts)} != {expected_matrix_calls} scored + 1 smoke",
    )

    require(total_modal_receipts == expected_matrix_calls,
            f"Modal receipt total mismatch: {total_modal_receipts} != {expected_matrix_calls}")
    single = [value for value in attempt_summaries if value["tries"] == 1]
    feedback = [value for value in attempt_summaries if value["tries"] == 2]
    summary = {
        "schema_version": 1,
        "kind": "gpt56-luna-low-fixed26-verification-summary",
        "status": "passed",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": EXPECTED_MODEL,
        "reasoning_effort": EXPECTED_EFFORT,
        "contract": EXPECTED_CONTRACT,
        "strict_one_shot": STRICT_ONE_SHOT,
        "single_only": SINGLE_ONLY,
        "feedback_only": FEEDBACK_ONLY,
        "transport": TRANSPORT,
        "tasks_per_attempt": 26,
        "attempt_count": len(labels),
        "attempts": attempt_summaries,
        "single_turn": {
            "scores": [value["pass_turn_1"] for value in single],
            "mean": (
                sum(value["pass_turn_1"] for value in single) / len(single)
                if single else None
            ),
        },
        "feedback_turn_2": {
            "turn_1_scores": [value["pass_turn_1"] for value in feedback],
            "turn_1_mean": (
                sum(value["pass_turn_1"] for value in feedback) / len(feedback)
                if feedback else None
            ),
            "cumulative_scores": [value["pass_cumulative"] for value in feedback],
            "cumulative_mean": (
                sum(value["pass_cumulative"] for value in feedback) / len(feedback)
                if feedback else None
            ),
        },
        "benchmark_transport_calls": len(accepted_call_receipts) - 1,
        "scored_candidate_attempts": expected_matrix_calls,
        "identity_smoke_calls": 1,
        "transport_call_receipts": len(call_receipts),
        "accepted_transport_calls": len(accepted_call_receipts),
        "local_rejected_preflight_calls": len(rejected_local_preflights),
        "local_rejected_preflight_receipts": rejected_local_preflights,
        "modal_score_receipts": total_modal_receipts,
        "authorized_feedback_calls": len(authorized_feedback_calls),
        "transport_violations": 0,
        "transport_timeouts": 0,
        "test_timeouts": 0,
        "model_error_outputs": sum(
            value["model_error_outputs"] for value in attempt_summaries
        ),
        "malformed_responses": sum(
            value["malformed_responses"] for value in attempt_summaries
        ),
        "total_usage": dict(sorted(total_usage.items())),
        "source_tree_sha256": EXPECTED_TREE,
        "runner_receipt_sha256": sha256(ARTIFACT / "matrix_runner_receipt.json"),
    }
    if REQUIRE_AUTHORIZED_FEEDBACK:
        expected_feedback_calls = expected_matrix_calls - 26 * len(feedback)
        require(
            len(authorized_feedback_calls) == expected_feedback_calls,
            "authorized feedback count mismatch: "
            f"{len(authorized_feedback_calls)} != {expected_feedback_calls}",
        )
    destination = ARTIFACT / "verification_summary.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
