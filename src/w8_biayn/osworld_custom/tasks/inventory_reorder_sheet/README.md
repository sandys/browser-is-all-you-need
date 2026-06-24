# Inventory reorder sheet

- Domain: `libreoffice_calc`
- Task ID: `7b77f2a1-1e64-4c6b-9b29-0dcbcf847017`
- Snapshot: `libreoffice_calc`
- Source: `custom/w8-biayn`
- Proxy: `False`

## Instruction

In the inventory spreadsheet, fill the Status column with Reorder when Stock is below Threshold, otherwise OK. Save the file.

## Evaluator

This task uses OSWorld's `check_include_exclude` evaluator with a `vm_command_line` Python check. The command prints `Evaluation successful.` only when the expected final VM state is present, and `Evaluation failed.` otherwise.
