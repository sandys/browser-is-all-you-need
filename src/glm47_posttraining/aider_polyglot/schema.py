"""Persistent task and result models for Aider Polyglot C++ RL."""

from __future__ import annotations

import json
from pathlib import Path, PurePath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AiderChatMessage(BaseModel):
    """One turn of the exact chat aider sends to the model."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class AiderPolyglotTask(BaseModel):
    """One relocatable shadow-training or official-evaluation C++ task."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    task_id: str
    exercise: str
    language: Literal["cpp"] = "cpp"
    split: Literal["train", "validation"]
    harness_kind: Literal["shadow_cpp17", "official_cmake"]
    exercise_dir: str
    editable_files: list[str]
    prompt: list[AiderChatMessage]
    source_revision: str | None = None
    family: str | None = None
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    hidden_test_sha256: str | None = None
    source_prompt_sha256: str | None = None
    verification_gate: str | None = None

    @field_validator("prompt")
    @classmethod
    def _validate_prompt(cls, messages: list[AiderChatMessage]) -> list[AiderChatMessage]:
        if not messages or messages[-1].role != "user":
            raise ValueError("prompt must be a non-empty chat ending with a user turn")
        return messages

    @field_validator("editable_files")
    @classmethod
    def _validate_editable_files(cls, names: list[str]) -> list[str]:
        if not names:
            raise ValueError("Aider task requires at least one editable file")
        if len(set(names)) != len(names):
            raise ValueError("editable file names must be unique")
        for name in names:
            path = PurePath(name)
            if path.is_absolute() or len(path.parts) != 1 or name in {".", ".."}:
                raise ValueError(f"unsafe editable file name: {name}")
            if not name.endswith((".cpp", ".h", ".hpp", ".cc")):
                raise ValueError(f"unsupported editable file: {name}")
        return names

    @field_validator("exercise_dir")
    @classmethod
    def _validate_exercise_dir(cls, value: str) -> str:
        path = PurePath(value)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError(f"unsafe exercise directory: {value}")
        return value

    @classmethod
    def read_json(cls, path: str | Path) -> "AiderPolyglotTask":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return output


class AiderTestResult(BaseModel):
    """Outcome of applying a whole-file response and running the official tests."""

    status: Literal[
        "passed",
        "tests_failed",
        "compile_failed",
        "candidate_timeout",
        "infrastructure_error",
    ]
    tests_passed: int = Field(default=0, ge=0)
    tests_total: int = Field(default=0, ge=0)
    candidate_returncode: int | None = None
    logs: dict[str, str] = Field(default_factory=dict)

    @property
    def all_tests_pass(self) -> bool:
        return (
            self.status == "passed"
            and self.tests_total > 0
            and self.tests_passed == self.tests_total
        )

    @property
    def fraction_tests_passed(self) -> float:
        if self.tests_total <= 0:
            return 0.0
        return self.tests_passed / self.tests_total

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), indent=2, sort_keys=True)


class AiderShadowRubric(BaseModel):
    """Checked-in, answer-free contract for one shadow training exercise."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    task_id: str
    language: Literal["cpp"] = "cpp"
    editable_files: list[str]
    hidden_test_file: str
    hidden_test_sha256: str
    source_prompt_sha256: str
    reference_answer_packaged: Literal[False]
    verification_stage: Literal["passed"]
    verification_gate: str
    family: str
    category: str
    tags: list[str] = Field(default_factory=list)

    @field_validator("editable_files")
    @classmethod
    def _validate_editables(cls, names: list[str]) -> list[str]:
        return AiderPolyglotTask._validate_editable_files(names)

    @field_validator("hidden_test_file")
    @classmethod
    def _validate_hidden_test(cls, name: str) -> str:
        path = PurePath(name)
        if path.is_absolute() or len(path.parts) != 1 or not name.endswith("_test.cpp"):
            raise ValueError(f"unsafe hidden test file: {name}")
        return name

    @field_validator("hidden_test_sha256", "source_prompt_sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        lowered = value.lower()
        if len(lowered) != 64 or any(character not in "0123456789abcdef" for character in lowered):
            raise ValueError("expected lowercase SHA-256")
        return lowered

    @classmethod
    def read_json(cls, path: str | Path) -> "AiderShadowRubric":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
