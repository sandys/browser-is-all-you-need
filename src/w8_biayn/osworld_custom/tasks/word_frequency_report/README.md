# Word Frequency Report

- Domain: `os`
- Task ID: `7b77f2a1-1e64-4c6b-9b29-0dcbcf847040`
- Snapshot: `os`
- Source: `custom/w8-biayn`
- Proxy: `False`

## Instruction

Count word frequencies in Desktop/article.txt (case-insensitive, ignore punctuation) and write the top 5 words and their counts to Desktop/word_report.txt, one per line in the format: word: count, sorted by count descending.

## Evaluator

This task uses OSWorld's `check_include_exclude` evaluator with a `vm_command_line` Python check. The command prints `Evaluation successful.` only when the expected final VM state is present, and `Evaluation failed.` otherwise.
