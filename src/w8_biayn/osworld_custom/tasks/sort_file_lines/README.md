# Sort File Lines

- Domain: `os`
- Task ID: `7b77f2a1-1e64-4c6b-9b29-0dcbcf847054`
- Snapshot: `os`
- Source: `custom/w8-biayn`
- Proxy: `False`

## Instruction

Sort the lines in Desktop/input.txt alphabetically and write the result to Desktop/sorted.txt.

## Evaluator

This task uses OSWorld's `check_include_exclude` evaluator with a `vm_command_line` Python check. The command prints `Evaluation successful.` only when the expected final VM state is present, and `Evaluation failed.` otherwise.
