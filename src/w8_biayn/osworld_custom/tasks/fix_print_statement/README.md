# Fix Print Statement

- Domain: `vscode`
- Task ID: `7b77f2a1-1e64-4c6b-9b29-0dcbcf847051`
- Snapshot: `vscode`
- Source: `custom/w8-biayn`
- Proxy: `False`

## Instruction

The script uses Python 2 style print without parentheses. Fix it to Python 3 syntax so the script prints: hello world

## Evaluator

This task uses OSWorld's `check_include_exclude` evaluator with a `vm_command_line` Python check. The command prints `Evaluation successful.` only when the expected final VM state is present, and `Evaluation failed.` otherwise.
