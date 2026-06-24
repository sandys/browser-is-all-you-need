# Meeting minutes cleanup

- Domain: `libreoffice_writer`
- Task ID: `7b77f2a1-1e64-4c6b-9b29-0dcbcf847019`
- Snapshot: `libreoffice_writer`
- Source: `custom/w8-biayn`
- Proxy: `False`

## Instruction

Edit the meeting minutes so the title is Project Meeting Minutes - June 2026, the product name is Pipeshift OSWorld, there is a ## Action Items section, and all TODO markers are removed. Save the file.

## Evaluator

This task uses OSWorld's `check_include_exclude` evaluator with a `vm_command_line` Python check. The command prints `Evaluation successful.` only when the expected final VM state is present, and `Evaluation failed.` otherwise.
