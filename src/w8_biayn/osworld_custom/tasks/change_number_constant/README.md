# Change Number Constant

- Domain: `vscode`
- Task ID: `7b77f2a1-1e64-4c6b-9b29-0dcbcf847089`
- Snapshot: `vscode`
- Source: `custom/w8-biayn`
- Proxy: `False`

## Instruction

Change MAX_RETRIES = 3 to MAX_RETRIES = 5 in config.py. Save the file.

## Evaluator

This task uses OSWorld's `check_include_exclude` evaluator with a `vm_command_line` Python check. The command prints `Evaluation successful.` only when the expected final VM state is present, and `Evaluation failed.` otherwise.
