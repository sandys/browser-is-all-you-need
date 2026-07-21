# Public Release Checklist

## Release principle

Publish the implementation, tests, schemas, scripts, and documentation. Do not
publish external benchmark clones, ingested task JSON, prompts containing task
instructions, oracle solutions, hidden tests, generated rollouts, checkpoints,
credentials, or local experiment artifacts.

## 1. Choose a project license

This repository currently has no tracked root `LICENSE`. Select a license you
are authorized to use, add it at the repository root, and confirm that every
dependency and copied source is compatible. Do not copy the license from an
external benchmark and assume it licenses this repository.

GitHub's official guidance explains that a public repository without a license
remains under default copyright and is not automatically open source:
<https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository>.

If external code has been copied rather than merely consumed as a dependency,
retain its notices and attribution as required. Obtain legal review when
ownership or redistribution rights are unclear.

## 2. Keep external data outside Git

The root `.gitignore` excludes:

- `.glm47-posttraining/` generated task/data projections;
- `data/`, `datasets/`, artifacts, outputs, logs, and checkpoints;
- the root `polyglot-benchmark/` external clone;
- environment and cache directories.

Ignore rules do not remove files that were already committed. Verify the Git
index explicitly:

```bash
git ls-files | rg '(^|/)(polyglot-benchmark|\.glm47-posttraining|data|datasets|artifacts|outputs|wandb|checkpoints)/'
```

The command must print nothing. If a forbidden file is already tracked, remove
it from the index without deleting the local copy:

```bash
git rm -r --cached -- path/to/forbidden-directory
```

Review the resulting deletion before committing.

## 3. Stage with an allowlist

Do not begin with `git add .` in a dirty research workspace. Stage only the
implementation intended for release, for example:

```bash
git add .gitignore README.md pyproject.toml Dockerfile
git add src/glm47_posttraining
git add scripts examples tests
git add docs/ADAPTIVE_CPP_RUBRIC_REWARD.md
git add docs/CPP_RL_ENVIRONMENT.md
git add docs/PUBLIC_RELEASE_CHECKLIST.md
```

If `scripts`, `examples`, `tests`, or `docs` contain unrelated local work, stage
individual files instead of the entire directory.

Inspect exactly what will be published:

```bash
git diff --cached --name-status
git diff --cached --stat
git diff --cached
```

## 4. Scan for task leakage

Inspect staged data-like files and unexpected large files:

```bash
git diff --cached --name-only | rg '\.(json|jsonl|parquet|tar|gz|zip)$'
git diff --cached --numstat | sort -nr | head -n 50
```

Reject staged files containing serialized task fields such as `oracle_files`,
`test_files`, or full private instructions unless they are small synthetic test
fixtures created by you. Synthetic fixtures should not reproduce external task
text or solutions.

Also check that no model outputs, evaluation rows, W&B exports, or generated
taxonomy files are staged.

## 5. Scan for credentials and private paths

At minimum, scan staged content for common credential forms:

```bash
git diff --cached | rg -ni '(api[_-]?key|access[_-]?token|secret|password|credential|private[_-]?key|BEGIN [A-Z ]*PRIVATE KEY)'
```

Review absolute local paths, usernames, internal hosts, bucket names, and run
URLs. Prefer a dedicated secret scanner such as Gitleaks or TruffleHog before
publication. Rotate any credential that has ever entered Git history; deleting
it in a later commit is insufficient.

Official remediation guidance:
<https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository>.

## 6. Review provenance and claims

- Ensure README results have reproducible public evidence or clearly label
  them as private/illustrative.
- Remove links to inaccessible internal artifacts.
- Document the source and version of required external datasets without
  redistributing them.
- Distinguish local smoke verification from Docker production verification.
- Do not claim runtime or TSan evidence when those checks were not executed.
- Record known limitations, especially the static C++-quality heuristic and
  neutral runtime score when measurements are absent.

## 7. Validate the clean public checkout

Commit to a release branch, clone it into a new directory, and test without any
private data present:

```bash
git clone --no-local path/to/repository /tmp/project-public-check
cd /tmp/project-public-check
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m compileall -q src scripts tests
.venv/bin/pytest -q
```

Tests that require a separately downloaded external corpus should skip cleanly.
Core rubric, schema, reward arithmetic, XML parsing, and integration tests must
still pass.

## 8. Repository hosting steps

1. Create an empty remote repository with no automatically generated README or
   license conflict.
2. Add the chosen root license and any required notices.
3. Create a release branch and commit only the reviewed staged allowlist.
4. Add the remote and push the release branch.
5. Enable secret scanning, dependency alerts, and protected default-branch
   rules.
6. Open a pull request and review the hosting platform's complete file diff.
7. After merge, create a signed version tag and release notes describing
   verification status and excluded external data.

Example commands after the remote exists:

```bash
git switch -c public-release
git commit -m 'Publish adaptive C++ rubric RL environment'
git remote add public REMOTE_URL
git push -u public public-release
```

Do not run the push until the license, staged diff, secret scan, clean-clone
tests, and data-exclusion checks are complete.
