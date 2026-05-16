# Authoring Playbook

Use this playbook when creating or revising a skill.

## 1. Define Scope

Write one sentence: "This skill helps an agent do X when Y." If the sentence needs multiple unrelated verbs, split the work into multiple skills.

Good scopes:

- "Create and maintain Terraform modules for this infrastructure repository."
- "Review pull requests using this team's risk checklist."
- "Generate branded sales decks using company templates."

Poor scopes:

- "Help with engineering."
- "Know everything about this company."
- "Use all our internal tools."

## 2. Choose The Canonical Location

For a repository skill, create `.agents/skills/<skill-name>`.

For a personal skill, create `$HOME/.agents/skills/<skill-name>`.

For Claude-only behavior, `.claude/skills/<skill-name>` is valid, but prefer `.agents/skills` plus symlinks when the same skill should work in multiple tools.

## 3. Design Progressive Disclosure

Keep `SKILL.md` as the routing layer:

- What to do first.
- What decisions to make.
- Which bundled file to read for each branch.
- What validation to run.

Move detail out:

- Long command matrices go in `references/commands.md`.
- API specs go in `references/api.md`.
- Examples go in `references/examples.md`.
- Reusable code goes in `scripts/`.
- Output templates go in `assets/`.

## 4. Write The Description

The description is the matching mechanism. It must answer:

- What capability does this skill provide?
- When should an agent invoke it?
- What user phrases, file types, tools, or tasks imply relevance?

Pattern:

```yaml
description: Create and maintain <domain> using <specific tools/files>. Use when the user asks to <task 1>, <task 2>, or work with <trigger files/terms>.
```

Strong example:

```yaml
description: Create, validate, and install portable Agent Skills with SKILL.md folders for Claude Code, Gemini CLI, Codex, and compatible AI agents. Use when working on skill frontmatter, discovery paths, progressive disclosure, bundled resources, or cross-tool symlinks.
```

Weak example:

```yaml
description: Helps with skills.
```

## 5. Write Agent-Facing Instructions

Use imperative steps:

- Inspect existing files before editing.
- Prefer the repo's existing conventions.
- Do not duplicate canonical resources.
- Run the validator after changes.

Avoid motivational prose, marketing copy, and long explanations of obvious concepts.

## 6. Explain A Whole Framework

When the skill teaches a framework, include these sections or references:

- Inventory: important directories, commands, config files, services, and generated outputs.
- Mental model: lifecycle, state transitions, data flow, ownership boundaries.
- Happy paths: common workflows with exact commands and expected results.
- Extension paths: how to add a module, adapter, page, endpoint, command, or integration.
- Guardrails: what not to change, what must stay synchronized, and which generated files are off limits.
- Validation: test commands, lint commands, smoke checks, and manual checks.
- Recovery: common errors, logs to inspect, and rollback or cleanup steps.

If a section is longer than a screen, move it to `references/` and link it from `SKILL.md`.

## 7. Validate

Run the bundled validator:

```bash
python3 .agents/skills/agent-skills-framework/scripts/validate_skill.py .agents/skills/<skill-name>
```

Also run any skill-specific scripts or checks the skill tells agents to run.

Inspect the result from a fresh agent's perspective:

- Would the description trigger for realistic user requests?
- Does `SKILL.md` tell the agent what to inspect first?
- Are all referenced files present?
- Are scripts executable and documented by usage text?
- Are symlinks links to the canonical folder rather than copies?

## Common Mistakes

- Creating a broad "company knowledge" skill instead of focused skills.
- Putting all documentation in `SKILL.md` instead of using references.
- Hiding trigger guidance in the body instead of the description.
- Copying the same skill into `.claude`, `.gemini`, and `.agents` instead of symlinking.
- Adding README or install docs inside the skill folder.
- Using product-specific frontmatter in a skill intended to be portable.
- Adding scripts that require undeclared tools or silently mutate files.
