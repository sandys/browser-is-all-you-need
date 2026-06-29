# Clean release folder

- Domain: `os`
- Task ID: `7b77f2a1-1e64-4c6b-9b29-0dcbcf847012`
- Snapshot: `os`
- Source: `custom/w8-biayn`
- Proxy: `False`

## Instruction

Create Desktop/release_clean from Desktop/release_src, preserving folder structure. Copy only real release files and exclude .tmp and .bak files.

## Evaluator

This task uses OSWorld's `check_include_exclude` evaluator with a `vm_command_line` Python check. The command prints `Evaluation successful.` only when the expected final VM state is present, and `Evaluation failed.` otherwise.
