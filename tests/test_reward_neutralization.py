#!/usr/bin/env python3
"""Infra-invalid samples must carry zero GRPO advantage (issue #110).

Run: PYTHONPATH=src python3 tests/test_reward_neutralization.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from glm47_posttraining.integrations.miles_aider_polyglot import (  # noqa: E402
    neutralize_infrastructure_scores,
)


def _record(problem, sample, score, *, infra=False, rollout=0):
    return {
        "problem_id": problem,
        "rollout_id": rollout,
        "sample_index": sample,
        "score": score,
        "reward": score,
        "infrastructure_error": infra,
    }


def test_invalid_samples_move_to_group_mean() -> None:
    records = [
        _record("p1", 0, 1.0),
        _record("p1", 1, 0.0),
        _record("p1", 2, -0.5),
        _record("p1", 3, 0.0, infra=True),
    ]
    neutralize_infrastructure_scores(records)
    group_scores = [r["score"] for r in records]
    anchor = records[3]["score"]
    # Anchor equals the valid members' mean, so the group mean equals the
    # anchor and the invalid sample's group-normalized advantage is zero.
    assert abs(anchor - (1.0 + 0.0 - 0.5) / 3) < 1e-12
    assert abs(sum(group_scores) / len(group_scores) - anchor) < 1e-12
    assert records[3]["score_neutralized"] is True
    # The audited reward value is untouched.
    assert records[3]["reward"] == 0.0


def test_valid_samples_untouched() -> None:
    records = [_record("p1", i, float(i)) for i in range(4)]
    neutralize_infrastructure_scores(records)
    assert [r["score"] for r in records] == [0.0, 1.0, 2.0, 3.0]
    assert not any("score_neutralized" in r for r in records)


def test_all_invalid_group_collapses_to_uniform() -> None:
    records = [_record("p1", i, 0.0, infra=True) for i in range(4)]
    neutralize_infrastructure_scores(records)
    scores = {r["score"] for r in records}
    assert scores == {0.0}  # identical scores: no advantage, no gradient


def test_groups_are_isolated() -> None:
    records = [
        _record("p1", 0, 1.0),
        _record("p1", 1, 0.0, infra=True),
        _record("p2", 0, -1.0),
        _record("p2", 1, 0.0, infra=True),
    ]
    neutralize_infrastructure_scores(records)
    assert records[1]["score"] == 1.0
    assert records[3]["score"] == -1.0


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
