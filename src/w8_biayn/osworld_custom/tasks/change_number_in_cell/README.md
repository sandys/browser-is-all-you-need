# Change Number In Cell

- Domain: `libreoffice_calc`
- Task ID: `7b77f2a1-1e64-4c6b-9b29-0dcbcf847097`
- Snapshot: `libreoffice_calc`
- Source: `custom/w8-biayn`
- Proxy: `False`

## Instruction

In the spreadsheet, change the Budget value for Marketing from 100 to 150. Save the file.

## Evaluator

This task uses OSWorld's `check_include_exclude` evaluator with a `vm_command_line` Python check. The command prints `Evaluation successful.` only when the expected final VM state is present, and `Evaluation failed.` otherwise.
