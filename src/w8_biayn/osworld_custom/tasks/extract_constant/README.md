# Extract Constant

- Domain: `vscode`
- Task ID: `7b77f2a1-1e64-4c6b-9b29-0dcbcf847029`
- Snapshot: `vscode`
- Source: `custom/w8-biayn`
- Proxy: `False`

## Instruction

The script uses the magic number 0.08 for a tax rate in three places. Extract it to a module-level constant named TAX_RATE and update all usages. The script must still print Total with tax: 378.00.

## Evaluator

This task uses OSWorld's `check_include_exclude` evaluator with a `vm_command_line` Python check. The command prints `Evaluation successful.` only when the expected final VM state is present, and `Evaluation failed.` otherwise.
