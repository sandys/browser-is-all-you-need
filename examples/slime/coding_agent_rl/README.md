# Repo-Owned Coding-Agent RL

This directory is the repo-owned adaptation surface for the upstream SLIME coding-agent example. The current state is intentionally narrow:

- `generate.py` and `swe.py` are copied from the upstream example as the starting baseline.
- `run_qwen36_35b_a3b_smoke.sh` is a local smoke-launcher scaffold for later reduction and wiring.
- Docker sandbox support lives under `src/w8_biayn/slime_integration/` and will be wired into these files next.

Reference upstream source:

- `.cache/upstreams/slime/examples/coding_agent_rl/README.md`
- `.cache/upstreams/slime/examples/coding_agent_rl/generate.py`
- `.cache/upstreams/slime/examples/coding_agent_rl/swe.py`
