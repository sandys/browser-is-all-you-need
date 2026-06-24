# Export thumbnail pair

- Domain: `gimp`
- Task ID: `7b77f2a1-1e64-4c6b-9b29-0dcbcf847020`
- Snapshot: `gimp`
- Source: `custom/w8-biayn`
- Proxy: `False`

## Instruction

Open the image on the Desktop and export two PNG files on the Desktop: thumb_64.png with dimensions 64 by 64 pixels and thumb_256.png with dimensions 256 by 256 pixels.

## Evaluator

This task uses OSWorld's `check_include_exclude` evaluator with a `vm_command_line` Python check. The command prints `Evaluation successful.` only when the expected final VM state is present, and `Evaluation failed.` otherwise.
