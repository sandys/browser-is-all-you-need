# Add Shebang Line

- Domain: `vscode`
- Task ID: `7b77f2a1-1e64-4c6b-9b29-0dcbcf847050`
- Snapshot: `vscode`
- Source: `custom/w8-biayn`
- Proxy: `False`

## Instruction

Add #!/usr/bin/env python3 as the very first line of script.py. Save the file.

## Evaluator

This task uses OSWorld's `check_include_exclude` evaluator with a `vm_command_line` Python check. The command prints `Evaluation successful.` only when the expected final VM state is present, and `Evaluation failed.` otherwise.
