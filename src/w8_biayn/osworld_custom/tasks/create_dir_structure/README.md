# Create Dir Structure

- Domain: `os`
- Task ID: `7b77f2a1-1e64-4c6b-9b29-0dcbcf847037`
- Snapshot: `os`
- Source: `custom/w8-biayn`
- Proxy: `False`

## Instruction

Read Desktop/manifest.txt and create every directory path listed under Desktop/project/. Each line in the manifest is a relative path to create.

## Evaluator

This task uses OSWorld's `check_include_exclude` evaluator with a `vm_command_line` Python check. The command prints `Evaluation successful.` only when the expected final VM state is present, and `Evaluation failed.` otherwise.
