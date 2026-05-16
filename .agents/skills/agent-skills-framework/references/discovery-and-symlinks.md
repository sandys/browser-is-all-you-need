# Discovery And Symlinks

Use this reference when installing one skill for multiple AI CLI tools. Prefer one canonical source folder and symlink every compatibility path to it.

## Recommended Canonical Layout

For repository-shared skills:

```text
.agents/skills/<skill-name>/
```

For user-wide skills:

```text
$HOME/.agents/skills/<skill-name>/
```

`.agents/skills` is the best common source because Codex reads it directly and Gemini CLI treats it as an interoperable alias.

## Tool Discovery Paths

Claude Code:

- Personal skills: `$HOME/.claude/skills/<skill-name>/`
- Project skills: `.claude/skills/<skill-name>/`
- Added directories: `.claude/skills/` inside a directory passed with Claude Code's `--add-dir` is loaded automatically.

Gemini CLI:

- User skills: `$HOME/.gemini/skills/<skill-name>/`
- User alias: `$HOME/.agents/skills/<skill-name>/`
- Workspace skills: `.gemini/skills/<skill-name>/`
- Workspace alias: `.agents/skills/<skill-name>/`
- Precedence from low to high: built-in, extension, user, workspace.
- Within user or workspace tiers, `.agents/skills` takes precedence over `.gemini/skills`.

Codex:

- Repo skills: `.agents/skills/<skill-name>/` from the current working directory and parent directories up to the repo root.
- User skills: `$HOME/.agents/skills/<skill-name>/`
- Admin skills: `/etc/codex/skills/<skill-name>/`
- System skills: bundled with Codex.
- Codex follows symlinked skill folders.

## Workspace Symlinks

From the repository root, keep the real skill in `.agents/skills` and link Claude/Gemini workspace paths to it:

```bash
skill="<skill-name>"
mkdir -p .claude/skills .gemini/skills
ln -s "../../.agents/skills/$skill" ".claude/skills/$skill"
ln -s "../../.agents/skills/$skill" ".gemini/skills/$skill"
```

Use relative targets for committed workspace symlinks so the repository can move.

## User-Wide Symlinks

Use absolute targets for home-directory symlinks:

```bash
skill="<skill-name>"
canonical="/absolute/path/to/.agents/skills/$skill"
mkdir -p "$HOME/.agents/skills" "$HOME/.claude/skills" "$HOME/.gemini/skills"
ln -s "$canonical" "$HOME/.agents/skills/$skill"
ln -s "$canonical" "$HOME/.claude/skills/$skill"
ln -s "$canonical" "$HOME/.gemini/skills/$skill"
```

If the canonical folder already lives in `$HOME/.agents/skills`, do not create a self-link for Codex/Gemini alias. Link only the tool-specific paths that need compatibility.

## Safe Replacement Rules

Before creating a link:

1. If the destination does not exist, create the symlink.
2. If the destination is already a symlink, inspect it with `readlink`.
3. If the destination is a real directory or file, do not replace it without explicit user approval.

Safe shell helper:

```bash
link_skill() {
  src="$1"
  dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [ -L "$dest" ]; then
    rm "$dest"
    ln -s "$src" "$dest"
  elif [ -e "$dest" ]; then
    printf 'Refusing to replace non-symlink: %s\n' "$dest" >&2
    return 1
  else
    ln -s "$src" "$dest"
  fi
}
```

## Verification

Check workspace links:

```bash
find .agents .claude .gemini -maxdepth 4 \( -type l -o -type f \) -print
readlink .claude/skills/<skill-name>
readlink .gemini/skills/<skill-name>
```

Check user links:

```bash
find "$HOME/.agents/skills" "$HOME/.claude/skills" "$HOME/.gemini/skills" -maxdepth 1 -type l -name '<skill-name>' -print
```

A correct setup has one real skill folder and every other installation path points at it.
