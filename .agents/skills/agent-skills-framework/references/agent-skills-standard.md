# Agent Skills Standard

Use this reference for the portable Agent Skills format. Verify upstream docs when exact client behavior matters, because individual AI tools may add product-specific extensions.

## Required Shape

A skill is a directory with a required `SKILL.md` file:

```text
skill-name/
├── SKILL.md
├── references/
├── scripts/
├── assets/
└── agents/
    └── openai.yaml
```

Only `SKILL.md` is required. The other directories exist only when they directly support the skill.

## SKILL.md

`SKILL.md` starts with YAML frontmatter between `---` markers, followed by Markdown instructions.

Minimal portable frontmatter:

```yaml
---
name: skill-name
description: Explain what this skill does and exactly when an agent should use it.
---
```

`name` requirements:

- Match the parent folder name.
- Use lowercase letters, digits, and hyphens only.
- Use 1 to 64 characters.
- Do not start or end with a hyphen.
- Do not use consecutive hyphens.

`description` requirements:

- Use 1 to 1024 characters for the open standard.
- Describe both what the skill does and when to use it.
- Include trigger terms a user is likely to write.
- Put trigger guidance in the description, not only in the body.

Open-standard optional fields include `license`, `compatibility`, `metadata`, and experimental `allowed-tools`. Product-specific fields may work only in one client. Prefer only `name` and `description` unless the user explicitly needs an extension.

## Progressive Disclosure

Design every skill around progressive disclosure:

1. Discovery: the client loads `name` and `description` for all available skills.
2. Activation: the agent reads the full `SKILL.md` only when the task matches.
3. Resource access: the agent reads references, runs scripts, or uses assets only when needed.

Keep `SKILL.md` concise. A useful target is under 500 lines and under about 5000 tokens.

## Bundled Resources

Use `references/` for docs that are useful only in specific situations. Keep files focused and link them directly from `SKILL.md`.

Use `scripts/` for deterministic work that should not be reimplemented in every session: validation, conversion, extraction, schema checks, formatting, or API wrappers. Scripts should print clear errors and avoid hidden side effects.

Use `assets/` for files the agent should use in outputs rather than read as instructions: templates, starter projects, images, diagrams, schemas, sample data, fonts, or config snippets.

Use `agents/openai.yaml` for optional Codex UI metadata. Keep it separate from portable behavior.

## File Reference Rules

Reference bundled files with paths relative to the skill root:

```markdown
Read [references/api.md](references/api.md) when implementing API calls.
Run `scripts/validate.py <path>` before finishing.
```

Avoid chains where `SKILL.md` points to a reference that points to another required reference. Link important files directly from `SKILL.md`.

## Security

Audit skills before enabling them, especially scripts and assets. Do not hardcode secrets, tokens, private endpoints, or credentials. Treat downloaded skills like code dependencies.

When a skill can cause side effects, instruct the agent to inspect current state, explain planned actions, and run validation. For dangerous workflows, prefer explicit user invocation and product-specific safeguards where available.
