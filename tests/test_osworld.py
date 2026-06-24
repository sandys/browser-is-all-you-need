from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from w8_biayn import osworld


def make_osworld_tree(tmp_path):
    root = tmp_path / ".cache" / "upstreams" / "OSWorld"
    task_dir = root / "evaluation_examples" / "examples" / "os"
    task_dir.mkdir(parents=True)
    (root / "run.py").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='osworld'\n", encoding="utf-8")
    agent_dir = root / "mm_agents"
    agent_dir.mkdir()
    (agent_dir / "agent.py").write_text(
        """import os\n\ndef call_llm(self, payload):\n    if self.model == \"azure-gpt-4o\":\n        pass\n    elif self.model.startswith(\"gpt\"):\n        # Support custom OpenAI base URL via environment variable\n        base_url = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com')\n        api_url = f\"{base_url}/chat/completions\" if base_url.endswith('/v1') else f\"{base_url}/v1/chat/completions\"\n        headers = {\n            \"Content-Type\": \"application/json\",\n            \"Authorization\": f\"Bearer {os.environ['OPENAI_API_KEY']}\"\n        }\n        logger.info(\"Generating content with GPT model: %s\", self.model)\n        response = requests.post(api_url, headers=headers, json=payload)\n\n        if response.status_code != 200:\n            if response.json()['error']['code'] == "context_length_exceeded":\n                logger.error("Context length exceeded. Retrying with a smaller context.")\n                payload["messages"] = [payload["messages"][0]] + payload["messages"][-1:]\n                retry_response = requests.post(\n                    api_url,\n                    headers=headers,\n                    json=payload\n                )\n                if retry_response.status_code != 200:\n                    logger.error(\n                        "Failed to call LLM even after attempt on shortening the history: " + retry_response.text)\n                    return ""\n\n            logger.error("Failed to call LLM: " + response.text)\n            time.sleep(5)\n            return ""\n        else:\n            return response.json()['choices'][0]['message']['content']\n\ndef trim_accessibility_tree(linearized_accessibility_tree, max_tokens):\n    enc = tiktoken.encoding_for_model("gpt-4")\n    tokens = enc.encode(linearized_accessibility_tree)\n    if len(tokens) > max_tokens:\n        linearized_accessibility_tree = enc.decode(tokens[:max_tokens])\n        linearized_accessibility_tree += "[...]\\n"\n    return linearized_accessibility_tree\n\ndef parse_code_from_string(input_string):\n    input_string = "\\\\n".join([line.strip() for line in input_string.split(';') if line.strip()])\n    if input_string.strip() in ['WAIT', 'DONE', 'FAIL']:\n        return [input_string.strip()]\n    return []\n""",
        encoding="utf-8",
    )
    (root / ".venv").mkdir()
    task_id = "e0df059f-28a6-4169-924f-b9623e7184cc"
    (task_dir / f"{task_id}.json").write_text(
        json.dumps(
            {
                "id": task_id,
                "instruction": "Rename todo_list_Jan_1 to todo_list_Jan_2",
                "proxy": False,
                "fixed_ip": False,
                "possibility_of_env_change": "low",
            }
        ),
        encoding="utf-8",
    )
    for extra_task_id, instruction in (
        ("28cc3b7e-b194-4bc9-8353-d04c0f4d56d2", "Turn the system volume up to max"),
        ("bedcedc4-4d72-425e-ad62-21960b11fe0d", "Turn off dim screen when inactive"),
    ):
        (task_dir / f"{extra_task_id}.json").write_text(
            json.dumps(
                {
                    "id": extra_task_id,
                    "instruction": instruction,
                    "proxy": False,
                    "fixed_ip": False,
                    "possibility_of_env_change": "low",
                }
            ),
            encoding="utf-8",
        )
    (task_dir / "not-smoke.json").write_text(
        json.dumps(
            {
                "id": "not-smoke",
                "instruction": "Needs network",
                "proxy": True,
                "fixed_ip": False,
                "possibility_of_env_change": "high",
            }
        ),
        encoding="utf-8",
    )
    return root




def write_proxy_config(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {
                    "host": "gw.example.com",
                    "port": 823,
                    "username": "user",
                    "password": "pass",
                    "protocol": "http",
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


def write_patchable_run_py(root: Path) -> Path:
    run_py = root / "run.py"
    run_py.write_text(
        '''import os

env = DesktopEnv(
    provider_name=args.provider_name,
    path_to_vm=args.path_to_vm,
    action_space=agent.action_space,
    screen_size=(args.screen_width, args.screen_height),
    headless=args.headless,
    os_type="Ubuntu",
    require_a11y_tree=args.observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"],
    vm_secret_mounts=args.vm_secret_mount,
)
''',
        encoding="utf-8",
    )
    return run_py


def test_osworld_a11y_compaction_patch_is_idempotent(tmp_path):
    make_osworld_tree(tmp_path)
    agent_path = osworld.upstream_agent_path(tmp_path)

    assert osworld.ensure_a11y_compaction_support(repo_root=tmp_path) is True
    patched = agent_path.read_text(encoding="utf-8")
    assert osworld.A11Y_COMPACTION_PATCH_MARKER in patched
    assert "OSWORLD_A11Y_TREE_MAX_ITEMS" in patched
    assert "OSWORLD_A11Y_IOU_THRESHOLD" in patched
    assert "accessibility tree truncated by item cap" in patched
    assert osworld.ensure_a11y_compaction_support(repo_root=tmp_path) is False


def test_osworld_a11y_compaction_env_validates_values():
    assert osworld.a11y_compaction_environment(
        a11y_tree_max_items=123, a11y_iou_threshold=0.35
    ) == {
        "OSWORLD_A11Y_TREE_MAX_ITEMS": "123",
        "OSWORLD_A11Y_IOU_THRESHOLD": "0.35",
    }
    with pytest.raises(ValueError, match="a11y_tree_max_items"):
        osworld.a11y_compaction_environment(a11y_tree_max_items=0)
    with pytest.raises(ValueError, match="a11y_iou_threshold"):
        osworld.a11y_compaction_environment(a11y_iou_threshold=-0.1)


def test_parse_task_requires_domain_and_id():
    with pytest.raises(ValueError):
        osworld.parse_task("missing-domain")

    task = osworld.parse_task("os/e0df059f-28a6-4169-924f-b9623e7184cc")

    assert task.domain == "os"
    assert task.task_id == "e0df059f-28a6-4169-924f-b9623e7184cc"


def test_osworld_validate_reports_upstream_and_task(tmp_path, monkeypatch):
    make_osworld_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    kvm = tmp_path / "kvm"
    kvm.touch()
    monkeypatch.setattr(osworld, "KVM_DEVICE", kvm)
    monkeypatch.setattr(osworld.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(
        osworld.subprocess,
        "run",
        lambda *args, **kwargs: osworld.subprocess.CompletedProcess(args, 0, "", ""),
    )

    rows = osworld.validate(repo_root=tmp_path)
    by_check = {row.check: row for row in rows}

    assert by_check["OSWorld upstream"].status == "ok"
    assert by_check["OSWorld run.py"].status == "ok"
    assert by_check["task config"].status == "ok"
    assert by_check["OSWorld Docker image"].status == "ok"
    assert not osworld.has_errors(rows)


def test_osworld_validate_blocks_missing_required_prereqs(tmp_path, monkeypatch):
    make_osworld_tree(tmp_path)
    monkeypatch.setattr(osworld, "KVM_DEVICE", tmp_path / "missing-kvm")
    monkeypatch.setattr(osworld.shutil, "which", lambda tool: None)

    rows = osworld.validate(repo_root=tmp_path)
    by_check = {row.check: row for row in rows}

    assert by_check["docker"].status == "missing"
    assert by_check["OSWorld Docker image"].status == "skipped"
    assert by_check["/dev/kvm"].status == "missing"
    assert osworld.has_errors(rows)


def test_osworld_list_tasks_discovers_upstream_configs(tmp_path):
    make_osworld_tree(tmp_path)

    tasks = osworld.list_tasks(domain="os", repo_root=tmp_path)

    assert [task.task for task in tasks] == [
        "os/28cc3b7e-b194-4bc9-8353-d04c0f4d56d2",
        "os/bedcedc4-4d72-425e-ad62-21960b11fe0d",
        "os/e0df059f-28a6-4169-924f-b9623e7184cc",
        "os/not-smoke",
    ]
    assert tasks[2].instruction == "Rename todo_list_Jan_1 to todo_list_Jan_2"


def test_osworld_list_tasks_filters_smoke_candidates(tmp_path):
    make_osworld_tree(tmp_path)

    tasks = osworld.list_tasks(domain="os", smoke_candidates=True, repo_root=tmp_path)

    assert [task.task for task in tasks] == [
        "os/28cc3b7e-b194-4bc9-8353-d04c0f4d56d2",
        "os/bedcedc4-4d72-425e-ad62-21960b11fe0d",
        "os/e0df059f-28a6-4169-924f-b9623e7184cc",
    ]


def test_osworld_run_record_round_trip(tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}\n", encoding="utf-8")

    results = tmp_path / "results"

    path = osworld.write_run_record(
        run_id="osworld-test",
        command="run",
        tasks=("os/e0df059f-28a6-4169-924f-b9623e7184cc",),
        metadata=metadata,
        results=results,
        status="completed",
        observation_type="screenshot",
        model="qwen3-vl-2b",
        repo_root=tmp_path,
    )

    assert path == osworld.run_record_path("osworld-test", repo_root=tmp_path)
    record = osworld.read_run_record(repo_root=tmp_path)
    assert record["run_id"] == "osworld-test"
    assert record["results"] == str(results)
    assert record["model"] == "qwen3-vl-2b"
    assert osworld.list_run_records(repo_root=tmp_path)[0]["status"] == "completed"


def test_osworld_run_record_records_mlflow_metadata(tmp_path, monkeypatch):
    root = make_osworld_tree(tmp_path)
    monkeypatch.setattr(osworld, "upstream_path", lambda *_args, **_kwargs: root)

    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}\n", encoding="utf-8")
    results = tmp_path / "results"

    path = osworld.write_run_record(
        run_id="osworld-mlflow",
        command="run",
        tasks=("os/e0df059f-28a6-4169-924f-b9623e7184cc",),
        metadata=metadata,
        results=results,
        status="completed",
        observation_type="screenshot",
        model="qwen3-vl-2b",
        mlflow_tracking_uri="file:///tmp/mlruns",
        mlflow_run_id="mlflow-run-id",
        mlflow_run_name="mlflow-run-name",
        mlflow_experiment_name="mlflow-exp",
        mlflow_enabled=True,
        repo_root=tmp_path,
    )

    assert path == osworld.run_record_path("osworld-mlflow", repo_root=tmp_path)
    record = osworld.read_run_record(repo_root=tmp_path)
    assert record["mlflow_tracking_uri"] == "file:///tmp/mlruns"
    assert record["mlflow_run_id"] == "mlflow-run-id"
    assert record["mlflow_run_name"] == "mlflow-run-name"
    assert record["mlflow_experiment_name"] == "mlflow-exp"
    assert record["mlflow_enabled"] is True


def test_osworld_summarize_task_results(tmp_path):
    task = osworld.parse_task("os/e0df059f-28a6-4169-924f-b9623e7184cc")
    result_file = osworld.task_result_path(task, repo_root=tmp_path)
    result_file.parent.mkdir(parents=True)
    result_file.write_text("0.0\n", encoding="utf-8")

    summary = osworld.summarize_task_results((task.key,), repo_root=tmp_path)

    assert summary.completed == 1
    assert summary.successes == 0
    assert summary.failures == 1
    assert summary.average_score == 0.0


def test_osworld_read_task_result_parses_score(tmp_path):
    task = osworld.parse_task("os/e0df059f-28a6-4169-924f-b9623e7184cc")
    result_file = osworld.task_result_path(task, repo_root=tmp_path)
    result_file.parent.mkdir(parents=True)
    result_file.write_text("1.0\n", encoding="utf-8")

    result = osworld.read_task_result(task.key, repo_root=tmp_path)

    assert result.task == task.key
    assert result.status == "success"
    assert result.score == 1.0
    assert result.result_file == result_file


def test_osworld_read_task_result_reports_missing(tmp_path):
    result = osworld.read_task_result("os/e0df059f-28a6-4169-924f-b9623e7184cc", repo_root=tmp_path)

    assert result.status == "missing"
    assert result.score is None


def test_osworld_smoke_plan_renders_upstream_run_command(tmp_path):
    make_osworld_tree(tmp_path)

    plan = osworld.smoke_plan(repo_root=tmp_path)

    assert "# OSWorld smoke dry run" in plan
    assert "cd " in plan
    assert "uv run python run.py" in plan
    assert "--provider_name docker" in plan
    assert "--test_all_meta_path" in plan
    assert "os/e0df059f-28a6-4169-924f-b9623e7184cc" in plan


def test_osworld_has_no_packaged_tasksets_in_fork():
    assert osworld.list_tasksets() == ()


def test_osworld_select_task_keys_supports_custom_taskset_path(tmp_path):
    make_osworld_tree(tmp_path)
    taskset = tmp_path / "taskset.json"
    taskset.write_text(
        json.dumps({"os": ["e0df059f-28a6-4169-924f-b9623e7184cc"]}),
        encoding="utf-8",
    )

    task_keys = osworld.select_task_keys(taskset=str(taskset), repo_root=tmp_path)

    assert task_keys == ("os/e0df059f-28a6-4169-924f-b9623e7184cc",)


def test_osworld_run_plan_selects_taskset(tmp_path):
    make_osworld_tree(tmp_path)
    taskset = tmp_path / "taskset.json"
    taskset.write_text(
        json.dumps({"os": ["e0df059f-28a6-4169-924f-b9623e7184cc"]}),
        encoding="utf-8",
    )

    plan = osworld.run_plan(taskset=str(taskset), repo_root=tmp_path)

    assert "tasks=os/e0df059f-28a6-4169-924f-b9623e7184cc" in plan
    assert "run-taskset-taskset.json" in plan


def test_osworld_run_plan_selects_domain_with_limit(tmp_path):
    make_osworld_tree(tmp_path)

    plan = osworld.run_plan(domain="os", limit=1, repo_root=tmp_path)

    assert "# OSWorld run dry run" in plan
    assert "tasks=os/28cc3b7e-b194-4bc9-8353-d04c0f4d56d2" in plan
    assert "os/bedcedc4-4d72-425e-ad62-21960b11fe0d" not in plan
    assert "run-os.json" in plan
    assert "--domain os" in plan


def test_osworld_select_task_keys_requires_one_selector(tmp_path):
    make_osworld_tree(tmp_path)

    with pytest.raises(ValueError):
        osworld.select_task_keys(repo_root=tmp_path)
    with pytest.raises(ValueError):
        osworld.select_task_keys(tasks=("os/a",), domain="os", repo_root=tmp_path)


def test_osworld_smoke_plan_renders_tiny_suite(tmp_path):
    make_osworld_tree(tmp_path)

    plan = osworld.smoke_plan(suite="tiny", repo_root=tmp_path)

    assert "suite=tiny" in plan
    assert "tasks=os/e0df059f-28a6-4169-924f-b9623e7184cc" in plan
    assert "os/28cc3b7e-b194-4bc9-8353-d04c0f4d56d2" in plan
    assert "os/bedcedc4-4d72-425e-ad62-21960b11fe0d" in plan
    assert "suite-tiny.json" in plan
    assert "--domain os" in plan


def test_osworld_write_task_metadata_groups_domains(tmp_path):
    tasks = [
        osworld.TaskRef("os", "task-a"),
        osworld.TaskRef("os", "task-b"),
        osworld.TaskRef("chrome", "task-c"),
    ]
    path = tmp_path / "metadata.json"

    osworld.write_task_metadata(tasks, path)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "os": ["task-a", "task-b"],
        "chrome": ["task-c"],
    }


def test_osworld_write_one_task_metadata(tmp_path):
    task = osworld.parse_task("os/e0df059f-28a6-4169-924f-b9623e7184cc")

    path = osworld.write_one_task_metadata(task, repo_root=tmp_path)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "os": ["e0df059f-28a6-4169-924f-b9623e7184cc"]
    }


def test_osworld_cleanup_docker_provider_containers_stops_running_osworld_containers(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:2] == ["docker", "ps"]:
            return osworld.subprocess.CompletedProcess(args, 0, "abc123\ndef456\n", "")
        return osworld.subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(osworld.shutil, "which", lambda tool: "/usr/bin/docker")
    monkeypatch.setattr(osworld.subprocess, "run", fake_run)

    stopped = osworld.cleanup_docker_provider_containers()

    assert stopped == ("abc123", "def456")
    assert calls[0] == [
        "docker",
        "ps",
        "--filter",
        "ancestor=happysixd/osworld-docker",
        "--format",
        "{{.ID}}",
    ]
    assert calls[1] == ["docker", "stop", "abc123", "def456"]


def test_osworld_upstream_command_cleans_docker_provider_on_keyboard_interrupt(monkeypatch, capsys):
    monkeypatch.setattr(osworld, "run_command", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(osworld, "cleanup_docker_provider_containers", lambda: ("abc123",))

    with pytest.raises(KeyboardInterrupt):
        osworld.run_upstream_command_with_cleanup(["uv", "run", "python", "run.py"], cwd=Path("."), provider="docker")

    assert "Stopped interrupted OSWorld Docker container(s): abc123" in capsys.readouterr().out


def test_osworld_run_records_interrupted_status(tmp_path, monkeypatch):
    make_osworld_tree(tmp_path)
    monkeypatch.setattr(
        osworld,
        "run_upstream_command_with_cleanup",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        osworld.run(suite="tiny", repo_root=tmp_path)

    assert osworld.read_run_record(repo_root=tmp_path)["status"] == "interrupted"


def clear_local_provider_env(monkeypatch):
    for name in (
        "LOCAL_OPENAI_BASE_URL",
        "LOCAL_OPENAI_API_KEY",
        "QWEN_BASE_URL",
        "QWEN_API_KEY",
        "KIMI_BASE_URL",
        "KIMI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_osworld_uses_local_provider_for_model_prefixes_and_env_vars(monkeypatch):
    clear_local_provider_env(monkeypatch)

    assert osworld.uses_local_openai_provider("kimi-k2.6")
    assert osworld.uses_local_openai_provider("KIMI-K2.6")
    assert osworld.uses_local_openai_provider("qwen3-vl-4b")
    assert osworld.uses_local_openai_provider("QWEN2.5-VL-3B")
    assert not osworld.uses_local_openai_provider("gpt-4o")

    monkeypatch.setenv("QWEN_BASE_URL", "http://127.0.0.1:8001/v1")
    assert osworld.uses_local_openai_provider("gpt-4o")

    monkeypatch.delenv("QWEN_BASE_URL", raising=False)
    monkeypatch.setenv("LOCAL_OPENAI_BASE_URL", "http://127.0.0.1:8001/v1")
    assert osworld.uses_local_openai_provider("gpt-4o")

    monkeypatch.delenv("LOCAL_OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("KIMI_BASE_URL", "http://127.0.0.1:8001/v1")
    assert osworld.uses_kimi_openai_provider("gpt-4o")


def test_osworld_ensure_local_provider_patches_upstream_agent_for_qwen(tmp_path, monkeypatch):
    root = make_osworld_tree(tmp_path)
    clear_local_provider_env(monkeypatch)
    agent_path = root / "mm_agents" / "agent.py"

    patched = osworld.ensure_local_openai_provider(model="qwen3-vl-4b", repo_root=tmp_path)
    source = agent_path.read_text(encoding="utf-8")

    assert patched is True
    assert osworld.LOCAL_OPENAI_PROVIDER_PATCH_MARKER in source
    assert osworld.LOCAL_OPENAI_PROVIDER_PATCH_V2_MARKER in source
    assert 'self.model.lower().startswith(("kimi", "qwen"))' in source
    assert 'os.environ.get("LOCAL_OPENAI_BASE_URL")' in source
    assert 'os.environ.get("QWEN_BASE_URL")' in source
    assert 'os.environ.get("KIMI_BASE_URL")' in source
    assert 'os.environ.get("LOCAL_OPENAI_API_KEY")' in source
    assert 'os.environ.get("QWEN_API_KEY")' in source
    assert 'os.environ.get("KIMI_API_KEY")' in source
    assert '"Authorization": f"Bearer {api_key}"' in source
    assert "requests.post(api_url, headers=headers, json=payload)" in source
    assert "response_text = response.text" in source
    assert "response.json()['error']['code']" not in source
    assert "if isinstance(content, list):" in source

    before = source
    patched_again = osworld.ensure_local_openai_provider(model="qwen3-vl-4b", repo_root=tmp_path)

    assert patched_again is False
    assert agent_path.read_text(encoding="utf-8") == before


def test_osworld_ensure_local_provider_upgrades_existing_v1_patch(tmp_path, monkeypatch):
    root = make_osworld_tree(tmp_path)
    clear_local_provider_env(monkeypatch)
    agent_path = root / "mm_agents" / "agent.py"

    assert osworld.ensure_local_openai_provider(model="qwen3-vl-4b", repo_root=tmp_path)
    source = agent_path.read_text(encoding="utf-8")
    v1_source = source.replace(
        f"{osworld.LOCAL_OPENAI_PROVIDER_PATCH_MARKER}\n            {osworld.LOCAL_OPENAI_PROVIDER_PATCH_V2_MARKER}",
        osworld.LOCAL_OPENAI_PROVIDER_PATCH_MARKER,
    ).replace('            if response.status_code != 200:\n                response_text = response.text\n                try:\n                    error_doc = response.json()\n                except ValueError:\n                    error_doc = {}\n                error_obj = error_doc.get("error") if isinstance(error_doc, dict) else None\n                error_code = error_obj.get("code") if isinstance(error_obj, dict) else None\n                if error_code == "context_length_exceeded":\n                    logger.error("Context length exceeded. Retrying with a smaller context.")\n                    payload["messages"] = [payload["messages"][0]] + payload["messages"][-1:]\n                    retry_response = requests.post(\n                        api_url,\n                        headers=headers,\n                        json=payload\n                    )\n                    if retry_response.status_code != 200:\n                        logger.error(\n                            "Failed to call LLM even after attempt on shortening the history: " + retry_response.text)\n                        return ""\n\n                logger.error("Failed to call LLM: " + response_text)\n                time.sleep(5)\n                return ""\n            else:\n                response_doc = response.json()\n                content = response_doc["choices"][0]["message"].get("content", "")\n                if isinstance(content, list):\n                    return "".join(\n                        part.get("text", "") for part in content if isinstance(part, dict)\n                    )\n                return content or ""\n', '            if response.status_code != 200:\n                if response.json()[\'error\'][\'code\'] == "context_length_exceeded":\n                    logger.error("Context length exceeded. Retrying with a smaller context.")\n                    payload["messages"] = [payload["messages"][0]] + payload["messages"][-1:]\n                    retry_response = requests.post(\n                        api_url,\n                        headers=headers,\n                        json=payload\n                    )\n                    if retry_response.status_code != 200:\n                        logger.error(\n                            "Failed to call LLM even after attempt on shortening the history: " + retry_response.text)\n                        return ""\n\n                logger.error("Failed to call LLM: " + response.text)\n                time.sleep(5)\n                return ""\n            else:\n                return response.json()[\'choices\'][0][\'message\'][\'content\']\n')
    agent_path.write_text(v1_source, encoding="utf-8")

    assert osworld.ensure_local_openai_provider(model="qwen3-vl-4b", repo_root=tmp_path)
    upgraded = agent_path.read_text(encoding="utf-8")
    assert osworld.LOCAL_OPENAI_PROVIDER_PATCH_V2_MARKER in upgraded
    assert "response_text = response.text" in upgraded
    assert "response.json()['error']['code']" not in upgraded


def test_osworld_ensure_local_provider_patches_upstream_agent_for_generic_env(tmp_path, monkeypatch):
    root = make_osworld_tree(tmp_path)
    clear_local_provider_env(monkeypatch)
    monkeypatch.setenv("LOCAL_OPENAI_BASE_URL", "http://127.0.0.1:8001/v1")

    patched = osworld.ensure_local_openai_provider(model="gpt-4o", repo_root=tmp_path)
    source = (root / "mm_agents" / "agent.py").read_text(encoding="utf-8")

    assert patched is True
    assert osworld.LOCAL_OPENAI_PROVIDER_PATCH_MARKER in source
    assert 'os.environ.get("LOCAL_OPENAI_BASE_URL")' in source


def test_osworld_ensure_local_provider_does_not_patch_default_gpt(tmp_path, monkeypatch):
    root = make_osworld_tree(tmp_path)
    clear_local_provider_env(monkeypatch)
    agent_path = root / "mm_agents" / "agent.py"
    before = agent_path.read_text(encoding="utf-8")

    patched = osworld.ensure_local_openai_provider(model="gpt-4o", repo_root=tmp_path)

    assert patched is False
    assert agent_path.read_text(encoding="utf-8") == before


def test_osworld_run_applies_local_provider_patch_for_qwen(tmp_path, monkeypatch):
    root = make_osworld_tree(tmp_path)
    calls = []
    clear_local_provider_env(monkeypatch)

    def fake_run_upstream(args, **kwargs):
        calls.append(args)

    monkeypatch.setattr(osworld, "run_upstream_command_with_cleanup", fake_run_upstream)

    _, task_keys, run_id = osworld.run(suite="tiny", model="qwen3-vl-4b", repo_root=tmp_path)

    source = (root / "mm_agents" / "agent.py").read_text(encoding="utf-8")
    assert osworld.LOCAL_OPENAI_PROVIDER_PATCH_MARKER in source
    assert calls
    assert "--model" in calls[0]
    assert "qwen3-vl-4b" in calls[0]
    assert task_keys == osworld.tasks_for_suite("tiny")
    assert run_id is not None
    assert osworld.read_run_record(repo_root=tmp_path)["status"] == "completed"


def test_osworld_run_uses_timestamped_result_dir(tmp_path, monkeypatch):
    make_osworld_tree(tmp_path)
    calls = []
    clear_local_provider_env(monkeypatch)
    monkeypatch.setattr(osworld, "make_run_id", lambda prefix="osworld": "osworld-fixed")
    monkeypatch.setattr(osworld, "run_upstream_command_with_cleanup", lambda args, **kwargs: calls.append(args))

    metadata, task_keys, run_id = osworld.run(suite="tiny", model="qwen3-vl-2b", repo_root=tmp_path)

    result_dir = osworld.run_results_path("osworld-fixed", repo_root=tmp_path)
    assert run_id == "osworld-fixed"
    assert metadata == osworld.run_metadata_path("osworld-fixed", "run-tiny", repo_root=tmp_path)
    assert result_dir.exists()
    assert calls
    assert calls[0][calls[0].index("--result_dir") + 1] == str(result_dir.resolve())
    assert calls[0][calls[0].index("--test_all_meta_path") + 1] == str(metadata.resolve())
    record = osworld.read_run_record("osworld-fixed", repo_root=tmp_path)
    assert record["results"] == str(result_dir)
    assert record["model"] == "qwen3-vl-2b"
    assert record["observation_type"] == "screenshot"


def test_osworld_proxy_environment_validates_default_upstream_config(tmp_path):
    root = make_osworld_tree(tmp_path)
    proxy_path = write_proxy_config(root / osworld.DEFAULT_PROXY_CONFIG)

    env = osworld.proxy_environment(enable_proxy=True, repo_root=tmp_path)

    assert env["OSWORLD_ENABLE_PROXY"] == "1"
    assert env["PROXY_CONFIG_FILE"] == str(proxy_path.resolve())
    assert env["OSWORLD_CLIENT_PASSWORD"] == osworld.DEFAULT_CLIENT_PASSWORD


def test_osworld_proxy_environment_rejects_missing_config(tmp_path):
    make_osworld_tree(tmp_path)

    with pytest.raises(FileNotFoundError, match="proxy config"):
        osworld.proxy_environment(enable_proxy=True, repo_root=tmp_path)


def test_osworld_validate_reports_proxy_config_when_enabled(tmp_path, monkeypatch):
    root = make_osworld_tree(tmp_path)
    proxy_path = write_proxy_config(root / osworld.DEFAULT_PROXY_CONFIG)
    monkeypatch.setattr(osworld, "KVM_DEVICE", tmp_path / "missing-kvm")
    monkeypatch.setattr(osworld.shutil, "which", lambda tool: None)

    rows = osworld.validate(enable_proxy=True, repo_root=tmp_path)
    by_check = {row.check: row for row in rows}

    assert by_check["proxy config"].status == "ok"
    assert by_check["proxy config"].detail == str(proxy_path)


def test_osworld_ensure_proxy_run_support_patches_upstream_run_py(tmp_path):
    root = make_osworld_tree(tmp_path)
    run_py = write_patchable_run_py(root)

    patched = osworld.ensure_proxy_run_support(repo_root=tmp_path)
    source = run_py.read_text(encoding="utf-8")

    assert patched is True
    assert osworld.PROXY_RUN_PATCH_MARKER in source
    assert 'enable_proxy=os.environ.get("OSWORLD_ENABLE_PROXY") == "1"' in source
    assert 'client_password=os.environ.get("OSWORLD_CLIENT_PASSWORD", "password")' in source

    before = source
    patched_again = osworld.ensure_proxy_run_support(repo_root=tmp_path)

    assert patched_again is False
    assert run_py.read_text(encoding="utf-8") == before


def test_osworld_run_with_proxy_sets_env_and_records_config(tmp_path, monkeypatch):
    root = make_osworld_tree(tmp_path)
    write_patchable_run_py(root)
    proxy_path = write_proxy_config(tmp_path / "proxy.json")
    calls = []
    clear_local_provider_env(monkeypatch)

    def fake_run_upstream(args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(osworld, "run_upstream_command_with_cleanup", fake_run_upstream)

    _, _task_keys, _run_id = osworld.run(
        suite="tiny",
        model="qwen3-vl-4b",
        enable_proxy=True,
        proxy_config_file=proxy_path,
        client_password="password",
        repo_root=tmp_path,
    )

    assert calls
    env = calls[0][1]["env"]
    assert env["OSWORLD_ENABLE_PROXY"] == "1"
    assert env["PROXY_CONFIG_FILE"] == str(proxy_path.resolve())
    assert env["OSWORLD_CLIENT_PASSWORD"] == "password"
    record = osworld.read_run_record(repo_root=tmp_path)
    assert record["enable_proxy"] is True
    assert record["proxy_config_file"] == str(proxy_path.resolve())


def write_osworld_task(root: Path, domain: str, task_id: str, *, proxy: bool = False, env_change: str = "low") -> Path:
    task_dir = root / "evaluation_examples" / "examples" / domain
    task_dir.mkdir(parents=True, exist_ok=True)
    task_path = task_dir / f"{task_id}.json"
    task_path.write_text(
        json.dumps(
            {
                "id": task_id,
                "instruction": f"Task {task_id}",
                "proxy": proxy,
                "fixed_ip": False,
                "possibility_of_env_change": env_change,
            }
        ),
        encoding="utf-8",
    )
    return task_path


def test_osworld_benchmark_plan_renders_custom_taskset_by_domain(tmp_path):
    taskset = tmp_path / "taskset.json"
    taskset.write_text(
        json.dumps(
            {
                "chrome": ["chrome-task"],
                "os": ["e0df059f-28a6-4169-924f-b9623e7184cc"],
            }
        ),
        encoding="utf-8",
    )

    plan = osworld.benchmark_plan(
        taskset=str(taskset),
        limit_per_domain=1,
        model="qwen3-vl-8b",
        repo_root=tmp_path,
    )

    assert f"taskset={taskset}" in plan
    assert "taskset_tasks=2" in plan
    assert "domain=chrome tasks=1" in plan
    assert "domain=os tasks=1" in plan
    assert "--task chrome/chrome-task" in plan
    assert "--domain chrome" not in plan


def test_osworld_benchmark_rejects_domain_and_taskset_together(tmp_path):
    taskset = tmp_path / "taskset.json"
    taskset.write_text(
        json.dumps({"os": ["e0df059f-28a6-4169-924f-b9623e7184cc"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="taskset"):
        osworld.benchmark_plan(
            domains=("os",),
            taskset=str(taskset),
            repo_root=tmp_path,
        )


def test_osworld_benchmark_plan_renders_domain_commands(tmp_path):
    root = make_osworld_tree(tmp_path)
    write_osworld_task(root, "chrome", "chrome-task")

    plan = osworld.benchmark_plan(
        domains=("os", "chrome"),
        limit_per_domain=1,
        model="qwen3-vl-8b",
        repo_root=tmp_path,
    )

    assert "# OSWorld benchmark dry run" in plan
    assert "domains=os, chrome" in plan
    assert "domain=os tasks=1" in plan
    assert "domain=chrome tasks=1" in plan
    assert "uv run w8-biayn osworld run --domain os --limit 1 --provider docker" in plan
    assert "--model qwen3-vl-8b" in plan
    assert "--max-tokens 256" in plan
    assert "--max-trajectory-length 1" in plan
    assert "--a11y-tree-max-items 300" in plan
    assert "--a11y-iou-threshold 0.2" in plan


def test_osworld_benchmark_plan_renders_local_openai_endpoint(tmp_path):
    root = make_osworld_tree(tmp_path)
    write_osworld_task(root, "chrome", "chrome-task")

    plan = osworld.benchmark_plan(
        domains=("chrome",),
        limit_per_domain=1,
        model="qwen2.5-vl-7b",
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        repo_root=tmp_path,
    )

    assert "base_url=http://127.0.0.1:8000/v1" in plan
    assert "api_key=set" in plan
    assert "--base-url http://127.0.0.1:8000/v1" in plan
    assert "--api-key EMPTY" in plan


def test_osworld_benchmark_plan_smoke_candidates_renders_explicit_tasks(tmp_path):
    root = make_osworld_tree(tmp_path)
    write_osworld_task(root, "chrome", "safe-task", proxy=False, env_change="low")
    write_osworld_task(root, "chrome", "proxy-task", proxy=True, env_change="high")

    plan = osworld.benchmark_plan(
        domains=("chrome",),
        smoke_candidates=True,
        repo_root=tmp_path,
    )

    assert "domain=chrome tasks=1" in plan
    assert "--task chrome/safe-task" in plan
    assert "chrome/proxy-task" not in plan
    assert "note=--smoke-candidates is applied by benchmark task selection" in plan


def test_osworld_benchmark_plan_skips_empty_selected_domains(tmp_path):
    root = make_osworld_tree(tmp_path)
    write_osworld_task(root, "chrome", "proxy-task", proxy=True, env_change="high")

    plan = osworld.benchmark_plan(
        domains=("chrome",),
        smoke_candidates=True,
        repo_root=tmp_path,
    )

    assert "domain=chrome tasks=0" in plan
    assert "skip=no selected tasks" in plan
    assert "uv run w8-biayn osworld run" not in plan


def test_osworld_benchmark_runs_selected_domains(tmp_path, monkeypatch):
    root = make_osworld_tree(tmp_path)
    write_osworld_task(root, "chrome", "chrome-task")
    calls = []
    call_kwargs = []
    clear_local_provider_env(monkeypatch)

    def fake_run_upstream(args, **kwargs):
        calls.append(args)
        call_kwargs.append(kwargs)

    monkeypatch.setattr(osworld, "run_upstream_command_with_cleanup", fake_run_upstream)

    result = osworld.benchmark(
        domains=("chrome", "os"),
        limit_per_domain=1,
        model="gpt-4o",
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        repo_root=tmp_path,
    )

    assert result is not None
    assert [domain.domain for domain in result.domains] == ["chrome", "os"]
    assert [len(domain.tasks) for domain in result.domains] == [1, 1]
    assert len(calls) == 2
    assert call_kwargs[0]["env"]["LOCAL_OPENAI_BASE_URL"] == "http://127.0.0.1:8000/v1"
    assert call_kwargs[0]["env"]["LOCAL_OPENAI_API_KEY"] == "EMPTY"
    assert call_kwargs[0]["env"]["OSWORLD_A11Y_TREE_MAX_ITEMS"] == "300"
    assert call_kwargs[0]["env"]["OSWORLD_A11Y_IOU_THRESHOLD"] == "0.2"
    assert "--max_trajectory_length" in calls[0]
    assert "1" == calls[0][calls[0].index("--max_trajectory_length") + 1]
    assert result.total_tasks == 2
    assert result.completed == 0


def test_osworld_benchmark_records_mlflow_fields(tmp_path, monkeypatch):
    root = make_osworld_tree(tmp_path)
    write_osworld_task(root, "chrome", "chrome-task")
    monkeypatch.setattr(osworld, "upstream_path", lambda *_args, **_kwargs: root)
    clear_local_provider_env(monkeypatch)

    monkeypatch.setattr(
        osworld,
        "run_upstream_command_with_cleanup",
        lambda *args, **kwargs: None,
    )

    result = osworld.benchmark(
        domains=("os", "chrome"),
        limit_per_domain=1,
        model="gpt-4o",
        mlflow_tracking_uri="http://127.0.0.1:5000",
        mlflow_experiment="ci-exp",
        mlflow_run_name="ci-run",
        repo_root=tmp_path,
    )

    assert result is not None
    records = osworld.list_run_records(repo_root=tmp_path)
    assert len(records) == 2
    for record in records:
        assert record["mlflow_tracking_uri"] == "http://127.0.0.1:5000"
        assert record["mlflow_run_name"] == "ci-run"
        assert record["mlflow_experiment_name"] == "ci-exp"
        assert record["mlflow_enabled"] is True


def test_osworld_benchmark_reports_progress(tmp_path, monkeypatch):
    make_osworld_tree(tmp_path)
    write_osworld_task(tmp_path / ".cache" / "upstreams" / "OSWorld", "chrome", "chrome-task")
    snapshots = []

    def fake_run_upstream(args, **kwargs):
        result_dir = Path(args[args.index("--result_dir") + 1])
        observation_type = args[args.index("--observation_type") + 1]
        model = args[args.index("--model") + 1]
        metadata_path = Path(args[args.index("--test_all_meta_path") + 1])
        task_groups = json.loads(metadata_path.read_text(encoding="utf-8"))
        for domain, task_ids in task_groups.items():
            for task_id in task_ids:
                time.sleep(0.02)
                result_path = osworld.task_result_path(
                    osworld.TaskRef(domain, task_id),
                    observation_type=observation_type,
                    model=model,
                    result_dir=result_dir,
                    repo_root=tmp_path,
                )
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text("1.0\n", encoding="utf-8")

    monkeypatch.setattr(osworld, "run_upstream_command_with_cleanup", fake_run_upstream)

    result = osworld.benchmark(
        domains=("chrome", "os"),
        limit_per_domain=1,
        model="gpt-4o",
        progress_callback=snapshots.append,
        progress_poll_seconds=0.005,
        repo_root=tmp_path,
    )

    assert result is not None
    assert snapshots
    assert snapshots[0].completed_tasks == 0
    assert snapshots[-1].completed_tasks == 2
    assert snapshots[-1].remaining_tasks == 0
    assert any(snapshot.completed_tasks == 1 and snapshot.remaining_tasks == 1 for snapshot in snapshots)
    assert any(snapshot.eta_seconds is not None for snapshot in snapshots if snapshot.completed_tasks > 0)
