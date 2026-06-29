# Fix invoice script

- Domain: `vs_code`
- Task ID: `7b77f2a1-1e64-4c6b-9b29-0dcbcf847014`
- Snapshot: `vs_code`
- Source: `custom/w8-biayn`
- Proxy: `False`

## Instruction

In invoice.py, fix the tax calculation so the script prints Subtotal: 25.00, Tax: 2.50, and Total: 27.50.

## Evaluator

This task uses OSWorld's `check_include_exclude` evaluator with a `vm_command_line` Python check. The command prints `Evaluation successful.` only when the expected final VM state is present, and `Evaluation failed.` otherwise.
