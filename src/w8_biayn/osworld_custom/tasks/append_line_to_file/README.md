# Append Line To File

- Domain: `os`
- Task ID: `7b77f2a1-1e64-4c6b-9b29-0dcbcf847047`
- Snapshot: `os`
- Source: `custom/w8-biayn`
- Proxy: `False`

## Instruction

Using the terminal, append the line Done. to the end of Desktop/log.txt without overwriting the existing content.

## Evaluator

This task uses OSWorld's `check_include_exclude` evaluator with a `vm_command_line` Python check. The command prints `Evaluation successful.` only when the expected final VM state is present, and `Evaluation failed.` otherwise.
