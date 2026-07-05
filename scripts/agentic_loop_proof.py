#!/usr/bin/env python3
"""Local, no-GPU proof that the SWE-agent driver actually runs end-to-end.

Stands up a mock OpenAI ``/v1/chat/completions`` endpoint that returns a scripted
``thought_action`` sequence (rewrite ``candidate.cpp`` to a faster closed form,
then ``submit``), points ``swe_agent_driver.run_swe_agent_and_extract`` at it on
a tiny ``CppTask``, and grades the extracted file with
``run_in_directory_prewritten``. This exercises the never-run driver + SWE-agent
execution + swerex Docker spawn + file extraction + grader path with **no GPU and
no served model** — the Stage 1 mechanics proof.

SWE-agent 1.x must be an editable clone (its ``config/`` + ``tools/`` are not
packaged), so run via::

    uv run --with-editable /path/to/SWE-agent python scripts/agentic_loop_proof.py

Exits 0 on PASS (agent edited the file AND the graded file passes all tests).
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from w8_biayn.cpp_perf.sandbox import DEFAULT_DOCKER_IMAGE, run_in_directory_prewritten  # noqa: E402
from w8_biayn.cpp_perf.schema import CppTask, ReferencePerformance, TestCase, TestCoverage  # noqa: E402
from w8_biayn.integrations import swe_agent_driver  # noqa: E402

SLOW_CPP = (
    "#include <iostream>\n"
    "int main(){long long n; std::cin>>n; long long s=0; "
    "for(long long i=0;i<n;i++) s+=i; std::cout<<s<<\"\\n\"; return 0;}\n"
)
FAST_CPP = (
    "#include <iostream>\n"
    "int main(){long long n; std::cin>>n; std::cout<<n*(n-1)/2<<\"\\n\"; return 0;}\n"
)


def make_task() -> CppTask:
    # reference/oracle is the SLOW loop, so the agent's closed-form candidate
    # should both pass tests and beat the reference (speedup > 1).
    return CppTask(
        task_id="proof_0001",
        problem_id="proof",
        prompt_code=SLOW_CPP,
        unit_tests=[TestCase(input="1000\n", expected="499500\n")],
        hidden_tests=[TestCase(input="100\n", expected="4950\n")],
        oracle_solution=SLOW_CPP,
        test_coverage=TestCoverage(line=0.96, branch=0.86),
        reference=ReferencePerformance(value=1000),
        split="test",
    )


class MockLLM(BaseHTTPRequestHandler):
    turn = 0
    lock = threading.Lock()

    def log_message(self, *args):  # silence
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        with MockLLM.lock:
            MockLLM.turn += 1
            turn = MockLLM.turn
        if turn == 1:
            content = (
                "I'll rewrite candidate.cpp with a faster closed-form solution.\n\n"
                "```\ncat > candidate.cpp <<'CPPEOF'\n" + FAST_CPP + "CPPEOF\n```"
            )
        else:
            # every later turn (including any submit-review prompt) just submits
            content = "The file is optimized and correct. Submitting.\n\n```\nsubmit\n```"
        body = json.dumps(
            {
                "id": "chatcmpl-proof",
                "object": "chat.completion",
                "created": 0,
                "model": "proof",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockLLM)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[proof] mock LLM serving on http://127.0.0.1:{port}/v1")

    task = make_task()
    edited = False
    harness = None
    with tempfile.TemporaryDirectory(prefix="w8-proof-") as work:
        limits = swe_agent_driver.SweAgentLimits(per_instance_call_limit=6, total_execution_timeout=300)
        print("[proof] running SWE-agent ...")
        result = swe_agent_driver.run_swe_agent_and_extract(
            task,
            task.prompt_code,
            adapter_url=f"http://127.0.0.1:{port}/v1",
            sid="proof-sid",
            temperature=0.0,
            top_p=1.0,
            image=swe_agent_driver.DEFAULT_SWE_AGENT_IMAGE,
            limits=limits,
            work_root=work,
        )
        print(f"[proof] exit_status={result.exit_status} steps={result.steps} llm_turns={MockLLM.turn}")
        print(f"[proof] extracted candidate.cpp:\n{result.candidate_code}")
        edited = result.candidate_code.strip() != task.prompt_code.strip()
        print(f"[proof] agent edited the file: {edited}")

        scratch = Path(work) / "grade"
        scratch.mkdir()
        (scratch / "candidate.cpp").write_text(result.candidate_code, encoding="utf-8")
        harness = run_in_directory_prewritten(task, scratch, image=DEFAULT_DOCKER_IMAGE)
        print(
            f"[proof] graded: all_tests_pass={harness.all_tests_pass} "
            f"tests={harness.tests_passed}/{harness.tests_total} "
            f"compile_error={harness.compile_error} sanitizer_error={harness.sanitizer_error} "
            f"runtime_speedup={harness.runtime_speedup}"
        )
        if harness.logs:
            for key, val in harness.logs.items():
                print(f"[proof] log[{key}]:\n{val[:1500]}")

    server.shutdown()
    ok = edited and harness is not None and harness.all_tests_pass
    print(f"[proof] RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
