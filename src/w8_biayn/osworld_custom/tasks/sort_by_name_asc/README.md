# Sort By Name Asc

- Domain: `libreoffice_calc`
- Task ID: `7b77f2a1-1e64-4c6b-9b29-0dcbcf847101`
- Snapshot: `libreoffice_calc`
- Source: `custom/w8-biayn`
- Proxy: `False`

## Instruction

Sort the spreadsheet rows alphabetically A to Z by the Name column. Save the file.

## Evaluator

This task uses OSWorld's `check_include_exclude` evaluator with a `vm_command_line` Python check. The command prints `Evaluation successful.` only when the expected final VM state is present, and `Evaluation failed.` otherwise.
