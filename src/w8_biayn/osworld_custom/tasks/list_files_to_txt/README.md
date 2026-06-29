# List Files To Txt

- Domain: `os`
- Task ID: `7b77f2a1-1e64-4c6b-9b29-0dcbcf847081`
- Snapshot: `os`
- Source: `custom/w8-biayn`
- Proxy: `False`

## Instruction

List all filenames in Desktop/assets/ and write them (one per line, sorted) to Desktop/files.txt.

## Evaluator

This task uses OSWorld's `check_include_exclude` evaluator with a `vm_command_line` Python check. The command prints `Evaluation successful.` only when the expected final VM state is present, and `Evaluation failed.` otherwise.
