from __future__ import annotations

from pathlib import Path


DOCS = (
    Path("README.md"),
    Path(".agents/REPO_GUIDE.md"),
    Path(".agents/skills/w8-biayn-framework/SKILL.md"),
)

REQUIRED_SLIME_GUARDRAILS = (
    "SLIME",
    "Moonlight",
    "GLM",
    "Megatron",
    "SGLang",
    "examples/slime/moonlight_cpp_perf",
    "examples/slime/glm47_cpp_perf",
    "src/w8_biayn/integrations/slime_cpp_perf.py",
    "child-process CPU time",
    "visible and hidden tests",
    "Do not write a custom trainer",
    "Do not reintroduce SkyRL/rLLM as the active",
    "git rm --cached",
    "RUN_REPORT",
)


def test_docs_keep_active_slime_guardrails_documented() -> None:
    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        missing = [phrase for phrase in REQUIRED_SLIME_GUARDRAILS if phrase not in text]
        assert not missing, f"{path} is missing guardrails: {missing}"
