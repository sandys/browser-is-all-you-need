# Merge JSON contacts

- Domain: `os`
- Task ID: `7b77f2a1-1e64-4c6b-9b29-0dcbcf847013`
- Snapshot: `os`
- Source: `custom/w8-biayn`
- Proxy: `False`

## Instruction

Merge Desktop/contacts/contacts_a.json and Desktop/contacts/contacts_b.json into Desktop/contacts_merged.json. De-duplicate by email and sort the final list by email.

## Evaluator

This task uses OSWorld's `check_include_exclude` evaluator with a `vm_command_line` Python check. The command prints `Evaluation successful.` only when the expected final VM state is present, and `Evaluation failed.` otherwise.
