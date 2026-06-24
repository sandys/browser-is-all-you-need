# Edit TOML config

- Domain: `vs_code`
- Task ID: `7b77f2a1-1e64-4c6b-9b29-0dcbcf847015`
- Snapshot: `vs_code`
- Source: `custom/w8-biayn`
- Proxy: `False`

## Instruction

Edit pyproject.toml so project.name is custom-osworld-demo, project.version is 0.2.0, tool.ospw.max_turns is 15, and tool.ospw.observation is screenshot_a11y_tree.

## Evaluator

This task uses OSWorld's `check_include_exclude` evaluator with a `vm_command_line` Python check. The command prints `Evaluation successful.` only when the expected final VM state is present, and `Evaluation failed.` otherwise.
