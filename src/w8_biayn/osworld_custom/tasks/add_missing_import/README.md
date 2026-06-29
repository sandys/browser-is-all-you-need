# Add Missing Import

- Domain: `vscode`
- Task ID: `7b77f2a1-1e64-4c6b-9b29-0dcbcf847026`
- Snapshot: `vscode`
- Source: `custom/w8-biayn`
- Proxy: `False`

## Instruction

The script uses json.dumps but is missing the import statement. Add import json at the top of the file so the script runs without errors.

## Evaluator

This task uses OSWorld's `check_include_exclude` evaluator with a `vm_command_line` Python check. The command prints `Evaluation successful.` only when the expected final VM state is present, and `Evaluation failed.` otherwise.
