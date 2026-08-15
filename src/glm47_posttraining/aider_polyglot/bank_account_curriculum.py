"""Build the execution-grounded bank-account curriculum from pinned Synth-v1 assets."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Mapping

from .dataset import SOURCE_MANIFEST_KIND
from .schema import AiderShadowRubric


CURRICULUM_ID = "synth-v1-bank-account-execution-grpo-v1"
CURRICULUM_NAME = "bank-account-v1"
SOURCE_COMMIT = "6188070622895021d1c340ad31939e888c514396"
SOURCE_SUITE_MANIFEST_SHA256 = (
    "781c3ab494b7ddd2833ab9d1e8fb1556025b0e68ae5e61934bca765f95d0eb7b"
)
SFT_DATASET_REVISION = "c586446fd309a1c2488b2953f77f3f370a73913c"
SFT_TRAIN_SHA256 = "3472d76169e52bd0859c181d63de24a060c4c7f2d3d8a004ceb6090498f1ddc1"
VALIDATION_VARIANTS = frozenset({"inventory-ledger", "transit-pass"})
EPISODE_KINDS = (
    "full-solve",
    "build-link-repair",
    "state-repair",
    "feedback-repair",
)
VERIFICATION_GATE = "bank-account-reference-starter-mutation-control-v1"
FLAGS = ["-std=c++17", "-Wall", "-Wextra", "-Werror", "-pedantic", "-pthread", "-I."]
COMPILE_TIMEOUT_S = 120
TEST_TIMEOUT_S = 45

Stage = Literal["pass", "compile", "link-or-odr", "runtime-or-sanitizer", "semantic"]


@dataclass(frozen=True)
class Evaluation:
    stage: Stage
    diagnostic: str


@dataclass(frozen=True)
class Variant:
    variant_id: str
    lineage_id: str
    split: Literal["train", "validation"]
    root: Path
    header: str
    source: str
    test: str
    prompt: str
    cmake: str
    manifest_entry: Mapping[str, object]

    @property
    def reference(self) -> dict[str, str]:
        return {
            self.header: (self.root / self.header).read_text(encoding="utf-8"),
            self.source: (self.root / self.source).read_text(encoding="utf-8"),
        }


BUILD_MUTATIONS: dict[str, str] = {
    "secure-wallet": "missing-mutex-header",
    "energy-reserve": "missing-mutex-header",
    "arcade-card": "undefined-constructor",
    "cloud-quota": "undefined-constructor",
    "library-credit": "missing-definitions",
    "workshop-tokens": "missing-definitions",
    "reward-points": "unqualified-start-method",
    "prepaid-data": "unqualified-start-method",
    "inventory-ledger": "odr-header-definition",
    "transit-pass": "odr-header-definition",
}


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compiler_identity(compiler: str) -> str:
    result = subprocess.run(
        [compiler, "--version"], check=False, capture_output=True, text=True, timeout=20
    )
    text = (result.stdout or result.stderr).strip()
    return text.splitlines()[0] if text else compiler


def compiler_is_gcc(compiler: str) -> bool:
    """Identify GNU C++ from predefined macros, independent of executable name."""
    result = subprocess.run(
        [compiler, "-dM", "-E", "-x", "c++", "-"],
        input="",
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode != 0:
        return False
    macros = result.stdout
    return "#define __GNUC__ " in macros and "#define __clang__ " not in macros


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _source_manifest(source_root: Path) -> dict[str, object]:
    path = source_root / "manifest.json"
    _require(path.is_file() and not path.is_symlink(), f"missing source manifest: {path}")
    _require(
        sha256_path(path) == SOURCE_SUITE_MANIFEST_SHA256,
        "bank-account source manifest hash does not match the pinned verified suite",
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(value.get("id") == "bank-account-equivalent-v1", "wrong source suite")
    variants = value.get("variants")
    _require(isinstance(variants, list) and len(variants) == 10, "expected ten source variants")
    return value


def _file_inventory(entry: Mapping[str, object]) -> tuple[str, str, str]:
    files = entry.get("files")
    _require(isinstance(files, dict), f"{entry.get('id')}: malformed file inventory")
    names = list(files)
    headers = [name for name in names if name.endswith((".h", ".hpp"))]
    sources = [
        name for name in names if name.endswith((".cpp", ".cc")) and not name.endswith("_test.cpp")
    ]
    tests = [name for name in names if name.endswith("_test.cpp")]
    _require(
        len(headers) == len(sources) == len(tests) == 1,
        f"{entry.get('id')}: expected one header, source, and test",
    )
    return headers[0], sources[0], tests[0]


def load_variants(source_root: str | Path) -> list[Variant]:
    source = Path(source_root).resolve()
    manifest = _source_manifest(source)
    variants: list[Variant] = []
    for raw_entry in manifest["variants"]:  # type: ignore[index]
        _require(isinstance(raw_entry, dict), "source variant is not an object")
        entry: dict[str, object] = raw_entry
        variant_id = str(entry["id"])
        header, source_name, test = _file_inventory(entry)
        root = source / str(entry["directory"])
        for name in (header, source_name, test, "PROMPT.md", "CMakeLists.txt"):
            path = root / name
            _require(path.is_file() and not path.is_symlink(), f"missing source asset: {path}")
            file_record = entry["files"].get(name)  # type: ignore[union-attr]
            _require(isinstance(file_record, dict), f"missing source hash record: {path}")
            _require(
                sha256_path(path) == file_record.get("sha256"),
                f"source hash mismatch: {path}",
            )
        variants.append(
            Variant(
                variant_id=variant_id,
                lineage_id=f"synth-v1/bank-account-equivalent-v1/{variant_id}",
                split="validation" if variant_id in VALIDATION_VARIANTS else "train",
                root=root,
                header=header,
                source=source_name,
                test=test,
                prompt=(root / "PROMPT.md").read_text(encoding="utf-8"),
                cmake=(root / "CMakeLists.txt").read_text(encoding="utf-8"),
                manifest_entry=entry,
            )
        )
    _require(
        {variant.variant_id for variant in variants} == set(BUILD_MUTATIONS),
        "variant set drifted",
    )
    return variants


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    _require(count == 1, f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def _remove_constructor(source: str, class_name: str, label: str) -> str:
    pattern = re.compile(
        rf"(?m)^{re.escape(class_name)}::{re.escape(class_name)}\([^\n]*\)"
        rf"(?:\s*=\s*default;|[^{{\n]*\{{[^\n]*\}})\n?"
    )
    match = pattern.search(source)
    _require(match is not None, f"{label}: constructor definition anchor not found")
    return source[: match.start()] + source[match.end() :]


def build_mutation(variant: Variant) -> tuple[dict[str, str], str, Stage]:
    files = variant.reference
    mutation = BUILD_MUTATIONS[variant.variant_id]
    header = files[variant.header]
    source = files[variant.source]
    api = variant.manifest_entry["api"]
    _require(isinstance(api, dict), f"{variant.variant_id}: malformed API metadata")
    class_name = str(api["class"])
    start_name = str(api["start"])

    if mutation == "missing-mutex-header":
        files[variant.header] = _replace_once(
            header, "#include <mutex>\n", "", variant.variant_id
        )
        expected: Stage = "compile"
    elif mutation == "undefined-constructor":
        files[variant.source] = _remove_constructor(source, class_name, variant.variant_id)
        expected = "link-or-odr"
    elif mutation == "missing-definitions":
        files[variant.source] = f'#include "{variant.header}"\n'
        expected = "link-or-odr"
    elif mutation == "unqualified-start-method":
        files[variant.source] = _replace_once(
            source,
            f"void {class_name}::{start_name}()",
            f"void {start_name}()",
            variant.variant_id,
        )
        expected = "compile"
    elif mutation == "odr-header-definition":
        namespace_end = re.compile(r"(?m)^}\s*// namespace [A-Za-z0-9_]+\s*$")
        matches = list(namespace_end.finditer(header))
        _require(len(matches) == 1, f"{variant.variant_id}: namespace end anchor drifted")
        insertion = "\nint glm47_bank_account_odr_probe() { return 0; }\n\n"
        match = matches[0]
        files[variant.header] = header[: match.start()] + insertion + header[match.start() :]
        expected = "link-or-odr"
    else:  # pragma: no cover - the fixed mapping above is exhaustive
        raise ValueError(f"unsupported mutation: {mutation}")
    return files, mutation, expected


def state_mutation(variant: Variant) -> dict[str, str]:
    files = variant.reference
    assignment = re.compile(
        r"(?m)^[ \t]+(?:state_\.)?(?:value|value_|amount_|count_) = 0;\n"
    )
    matches = [(name, list(assignment.finditer(body))) for name, body in files.items()]
    populated = [(name, found) for name, found in matches if found]
    _require(
        len(populated) == 1 and len(populated[0][1]) == 1,
        f"{variant.variant_id}: reset assignment anchor drifted",
    )
    name, found = populated[0]
    match = found[0]
    files[name] = files[name][: match.start()] + files[name][match.end() :]
    return files


def semantic_control(variant: Variant) -> dict[str, str]:
    files = variant.reference
    anchors = [
        (name, body.count("amount <= 0")) for name, body in files.items() if "amount <= 0" in body
    ]
    _require(len(anchors) == 1, f"{variant.variant_id}: amount boundary anchor drifted")
    name, _ = anchors[0]
    files[name] = files[name].replace("amount <= 0", "amount < 0", 1)
    return files


def _normalize_diagnostic(text: str, work: Path, label: str) -> str:
    replacement = f"/aider/{label}"
    for value in sorted({str(work), str(work.resolve())}, key=len, reverse=True):
        text = text.replace(value, replacement)
    return text.strip()


def evaluate(
    variant: Variant,
    files: Mapping[str, str],
    *,
    compiler: str,
    label: str,
) -> Evaluation:
    with TemporaryDirectory(prefix=f"bank-account-rl-{variant.variant_id}-") as temporary:
        work = Path(temporary)
        for name, body in files.items():
            (work / name).write_text(body, encoding="utf-8")
        shutil.copy2(variant.root / variant.test, work / variant.test)

        units = [variant.test, *sorted(name for name in files if name.endswith((".cpp", ".cc")))]
        objects: list[str] = []
        for index, unit in enumerate(units):
            object_name = f"unit-{index}.o"
            result = subprocess.run(
                [compiler, *FLAGS, "-c", unit, "-o", object_name],
                cwd=work,
                check=False,
                capture_output=True,
                text=True,
                timeout=COMPILE_TIMEOUT_S,
            )
            diagnostic = _normalize_diagnostic(result.stdout + result.stderr, work, label)
            if result.returncode != 0:
                return Evaluation("compile", diagnostic)
            objects.append(object_name)

        link = subprocess.run(
            [compiler, *FLAGS, *objects, "-o", "candidate"],
            cwd=work,
            check=False,
            capture_output=True,
            text=True,
            timeout=COMPILE_TIMEOUT_S,
        )
        diagnostic = _normalize_diagnostic(link.stdout + link.stderr, work, label)
        if link.returncode != 0:
            return Evaluation("link-or-odr", diagnostic)

        run = subprocess.run(
            [str(work / "candidate")],
            cwd=work,
            check=False,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT_S,
        )
        diagnostic = _normalize_diagnostic(run.stdout + run.stderr, work, label)
        if run.returncode == 0:
            return Evaluation("pass", diagnostic)
        if run.returncode < 0 or run.returncode > 128:
            return Evaluation("runtime-or-sanitizer", diagnostic)
        return Evaluation("semantic", diagnostic)


def feedback_instructions(prompt: str, diagnostic: str) -> str:
    _require(bool(diagnostic.strip()), "feedback episode requires a real diagnostic")
    return (
        prompt.rstrip()
        + "\n\n## Executed failure evidence\n\n"
        + "The supplied implementation produced this compiler or linker output under the "
        + "declared build contract:\n\n```text\n"
        + diagnostic.rstrip()
        + "\n```\n\nFix the supplied implementation. The hidden tests and build contract "
        + "are correct.\n"
    )


def _write_task(
    root: Path,
    variant: Variant,
    episode_kind: str,
    starter: Mapping[str, str],
    instructions: str,
    *,
    hidden_test_sha256: str,
) -> AiderShadowRubric:
    task_id = f"bank-account-{variant.variant_id}--{episode_kind}"
    task_root = root / task_id
    docs = task_root / ".docs"
    docs.mkdir(parents=True)
    (docs / "instructions.md").write_text(instructions.rstrip() + "\n", encoding="utf-8")
    for name, body in starter.items():
        (task_root / name).write_text(body, encoding="utf-8")
    shutil.copy2(variant.root / variant.test, task_root / variant.test)
    (task_root / "CMakeLists.txt").write_text(variant.cmake, encoding="utf-8")

    tags = [
        "bank-account-state-machine",
        "thread-safe-state",
        "whole-file-action",
        episode_kind,
    ]
    if episode_kind in {"build-link-repair", "feedback-repair"}:
        tags.extend(["cross-file-grounding", BUILD_MUTATIONS[variant.variant_id]])
    if episode_kind == "state-repair":
        tags.extend(["state-lifecycle", "stale-state-on-reopen"])
    rubric = AiderShadowRubric(
        task_id=task_id,
        split=variant.split,
        editable_files=[variant.header, variant.source],
        hidden_test_file=variant.test,
        hidden_test_sha256=hidden_test_sha256,
        source_prompt_sha256=sha256_path(docs / "instructions.md"),
        reference_answer_packaged=False,
        verification_stage="passed",
        verification_gate=VERIFICATION_GATE,
        family=variant.variant_id,
        category="bank-account-state-lifecycle",
        lineage_id=variant.lineage_id,
        episode_kind=episode_kind,
        tags=tags,
    )
    rubric_path = task_root / ".rubric.json"
    rubric_path.write_text(rubric.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return rubric


def build_bank_account_curriculum(
    source_root: str | Path,
    output_root: str | Path,
    *,
    compiler: str = "c++",
    require_gcc: bool = True,
) -> dict[str, object]:
    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    _require(source != output and source not in output.parents, "unsafe curriculum output path")
    identity = compiler_identity(compiler)
    gcc = compiler_is_gcc(compiler)
    if require_gcc:
        _require(gcc, f"GCC is required for frozen feedback, got: {identity}")
    variants = load_variants(source)
    if output.exists():
        _require(not output.is_symlink(), f"refusing to replace symlink: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    receipts: list[dict[str, object]] = []
    for variant in variants:
        reference = variant.reference
        build_files, build_mechanism, expected_build_stage = build_mutation(variant)
        state_files = state_mutation(variant)
        control_files = semantic_control(variant)
        empty_files = {variant.header: "", variant.source: ""}

        reference_result = evaluate(
            variant, reference, compiler=compiler, label=f"{variant.variant_id}-reference"
        )
        empty_result = evaluate(
            variant, empty_files, compiler=compiler, label=f"{variant.variant_id}-empty"
        )
        build_result = evaluate(
            variant, build_files, compiler=compiler, label=f"{variant.variant_id}-{build_mechanism}"
        )
        state_result = evaluate(
            variant, state_files, compiler=compiler, label=f"{variant.variant_id}-stale-state"
        )
        control_result = evaluate(
            variant, control_files, compiler=compiler, label=f"{variant.variant_id}-zero-boundary"
        )

        _require(reference_result.stage == "pass", f"{variant.variant_id}: reference failed")
        _require(empty_result.stage != "pass", f"{variant.variant_id}: empty starter passed")
        _require(
            build_result.stage == expected_build_stage,
            f"{variant.variant_id}: {build_mechanism} produced {build_result.stage}, "
            f"expected {expected_build_stage}",
        )
        _require(
            state_result.stage == "semantic",
            f"{variant.variant_id}: stale-state mutation produced {state_result.stage}",
        )
        _require(
            control_result.stage == "semantic",
            f"{variant.variant_id}: semantic control produced {control_result.stage}",
        )

        instructions = {
            "full-solve": variant.prompt,
            "build-link-repair": variant.prompt,
            "state-repair": variant.prompt,
            "feedback-repair": feedback_instructions(variant.prompt, build_result.diagnostic),
        }
        starters = {
            "full-solve": empty_files,
            "build-link-repair": build_files,
            "state-repair": state_files,
            "feedback-repair": build_files,
        }
        hidden_test_sha256 = sha256_path(variant.root / variant.test)
        for episode_kind in EPISODE_KINDS:
            rubric = _write_task(
                output,
                variant,
                episode_kind,
                starters[episode_kind],
                instructions[episode_kind],
                hidden_test_sha256=hidden_test_sha256,
            )
            starter_stage = {
                "full-solve": empty_result.stage,
                "build-link-repair": build_result.stage,
                "state-repair": state_result.stage,
                "feedback-repair": build_result.stage,
            }[episode_kind]
            receipts.append(
                {
                    "task_id": rubric.task_id,
                    "lineage_id": variant.lineage_id,
                    "split": variant.split,
                    "episode_kind": episode_kind,
                    "reference_passed": True,
                    "starter_rejected_as": starter_stage,
                    "failure_mutation": build_mechanism,
                    "failure_mutation_rejected_as": build_result.stage,
                    "semantic_mutation": "stale-state-on-reopen",
                    "semantic_mutation_rejected_as": state_result.stage,
                    "distinct_semantic_control": "zero-amount-accepted",
                    "distinct_semantic_control_rejected_as": control_result.stage,
                    "inherited_assertions_replayed": 19,
                    "new_assertions_added": 0,
                    "hidden_test_sha256": hidden_test_sha256,
                    "starter_sha256": {
                        name: sha256_text(body)
                        for name, body in sorted(starters[episode_kind].items())
                    },
                    "reference_sha256": {
                        name: sha256_text(body) for name, body in sorted(reference.items())
                    },
                    "prompt_sha256": rubric.source_prompt_sha256,
                }
            )

    receipt_path = output / "verification.jsonl"
    receipt_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in receipts),
        encoding="utf-8",
    )
    manifest: dict[str, object] = {
        "kind": SOURCE_MANIFEST_KIND,
        "schema_version": 2,
        "curriculum_id": CURRICULUM_ID,
        "source_locator": (
            f"repository:benchmarks/cpp/bank-account-equivalent-v1@{SOURCE_COMMIT}"
        ),
        "counts": {"tasks": 40, "train": 32, "validation": 8, "variant_lineages": 10},
        "compiler": identity,
        "compiler_is_gcc": gcc,
        "compile_flags": FLAGS,
        "source": {
            "suite": "bank-account-equivalent-v1",
            "suite_commit": SOURCE_COMMIT,
            "suite_manifest_sha256": SOURCE_SUITE_MANIFEST_SHA256,
            "sft_dataset_revision": SFT_DATASET_REVISION,
            "sft_train_sha256": SFT_TRAIN_SHA256,
        },
        "contract": {
            "official_task_id_overlap": [],
            "reference_answers_packaged": False,
            "shared_hidden_tests_within_lineage": True,
            "validation_lineages": sorted(VALIDATION_VARIANTS),
            "fixed26_artifacts_packaged": False,
            "old_source_training_exclusion_superseded_for_sft_lineages": True,
            "authorization_issue": (
                "https://github.com/tokenbender/browser-is-all-you-need/issues/110"
            ),
            "verification_gate": VERIFICATION_GATE,
        },
        "rollout_go_no_go": {
            "samples_per_prompt": 8,
            "minimum_mixed_prompt_group_fraction": 0.30,
            "all_pass_action": "stop-environment-too-easy",
            "all_fail_action": "use-atomic-repair-only",
            "concurrency_load_correlation": "required-before-training",
            "load_correlated_failure_action": "cap-reward-worker-parallelism-and-repeat",
        },
        "interference_mitigation": {
            "status": "recommended-before-optimizer-update",
            "recommended_anchor_weight": "10-15%",
            "candidate_anchor_source": "committed circular-state curriculum",
        },
        "verification_receipts": "verification.jsonl",
        "verification_receipts_sha256": sha256_path(receipt_path),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
