---
name: agent-skills-framework
description: Create, update, audit, install, and use portable Agent Skills/Claude Skills with SKILL.md folders for Claude Code, Gemini CLI, Codex, and compatible AI agents. Use when the user asks about skills, skill discovery paths, cross-tool symlinks, frontmatter, bundled scripts/references/assets, progressive disclosure, or making an AI agent understand a repeatable framework or workflow.
---

# Agent Skills Framework

Use this skill to build portable Agent Skills: self-contained directories that teach an AI agent repeatable workflows, domain knowledge, or tool procedures without loading all details into context up front.

## Core Workflow

1. Identify the job the skill should make repeatable. Keep one skill focused on one capability or closely related workflow.
2. Choose a canonical source folder. Prefer `.agents/skills/<skill-name>` for repo-scoped team skills, or `$HOME/.agents/skills/<skill-name>` for personal skills shared across tools.
3. Make the folder name, frontmatter `name`, and invocation name identical: lowercase letters, digits, and hyphens only.
4. Create `SKILL.md` with YAML frontmatter and concise Markdown instructions.
5. Add only useful bundled resources:
   - `references/` for detailed docs the agent should load only when needed.
   - `scripts/` for deterministic helpers the agent can run.
   - `assets/` for templates, examples, images, schemas, boilerplate, or other output resources.
   - `agents/openai.yaml` for optional Codex UI metadata.
6. If multiple AI CLI tools should discover the same skill, symlink their tool-specific skill paths to the canonical folder. Do not copy the skill.
7. Validate the structure, referenced files, and symlinks before finishing.

## Authoring Rules

Use only `name` and `description` in frontmatter unless a product-specific or open-standard optional field is clearly required. Portability is highest when frontmatter stays minimal.

Write `description` as the trigger surface. Include what the skill does and when to use it. Front-load important keywords because some clients shorten descriptions in the initial skill list.

Keep `SKILL.md` under 500 lines. Put long explanations, specs, examples, and path matrices in `references/`, and link them from `SKILL.md` with clear "read this when..." guidance.

Write instructions for an AI agent, not for a human reader. Prefer imperative steps, decision rules, validation checks, and examples of successful behavior.

Do not add README, changelog, installation guide, or unrelated docs inside a skill folder. Those files dilute discovery and make agents load or inspect noise.

## Teaching A Framework

When a skill must explain an entire framework, cover the operational surface an agent needs to act correctly:

1. Purpose: what the framework is for and what problems it solves.
2. Entry points: files, commands, APIs, CLIs, or directories the agent should inspect first.
3. Core model: the entities, lifecycle, invariants, and how pieces connect.
4. Common workflows: create, update, install, run, test, debug, publish.
5. Extension points: how to add new modules, adapters, resources, or integrations.
6. Failure modes: common mistakes, unsafe shortcuts, and how to recover.
7. Validation: concrete commands or checks the agent must run.

Put the framework map in `SKILL.md` only if it is short. Move deeper API details, examples, command matrices, and troubleshooting into `references/`.

## Repository Skills

When creating skills for a repository, also check for repo-level instructions such as `AGENTS.md` and user-facing docs such as `README.md`. If the skill describes how to work in that repo, require agents to keep those files and any architecture/workflow diagrams current with implementation changes.

Prefer one repo-specific skill for the project workflow instead of expanding a generic skill until it becomes project-specific.

## Cross-Tool Discovery

Use `.agents/skills/<skill-name>` as the canonical repo path when possible. Then symlink compatibility paths:

```bash
mkdir -p .claude/skills .gemini/skills
ln -s ../../.agents/skills/<skill-name> .claude/skills/<skill-name>
ln -s ../../.agents/skills/<skill-name> .gemini/skills/<skill-name>
```

For exact Claude Code, Gemini CLI, and Codex discovery locations, plus safe symlink commands, read [references/discovery-and-symlinks.md](references/discovery-and-symlinks.md).

## Validation

After creating or changing a skill, run:

```bash
python3 .agents/skills/agent-skills-framework/scripts/validate_skill.py .agents/skills/<skill-name>
```

If validating this skill from another current working directory, resolve the script path from the skill root.

Also inspect symlinks:

```bash
find .agents .claude .gemini -maxdepth 4 \( -type l -o -type f \) -print
```

## References

- Read [references/agent-skills-standard.md](references/agent-skills-standard.md) for the portable skill format, frontmatter constraints, progressive disclosure model, and security notes.
- Read [references/authoring-playbook.md](references/authoring-playbook.md) when designing a new skill or converting a large process into a skill.
- Read [references/discovery-and-symlinks.md](references/discovery-and-symlinks.md) before installing skills across Claude Code, Gemini CLI, and Codex.
