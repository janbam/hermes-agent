# Hermes Agent — Development Workflow

You are running on this fork of hermes-agent. This file covers only what's specific to our setup. You know standard git.

> **⚠️ Unresolved**: restarting the stable hermes gateway after promotion is not yet automated. Manual restart required for now.

## Your Identities

- **System user**: `hermes` on this machine
- **GitHub account**: [janbambot](https://github.com/janbambot) — collaborator on [janbam/hermes-agent](https://github.com/janbam/hermes-agent)
- **SSH**: clone/push via `git@github-bot.com` (janbambot key)

## Repository & Branches

| Remote | URL |
|--------|-----|
| `origin` | `git@github-bot.com:janbam/hermes-agent.git` |
| `upstream` | `git@github.com:NousResearch/hermes-agent.git` |

| Branch | Who writes | Your access |
|--------|-----------|-------------|
| `main` | **janbam only** (protected) | Read, merge from into `hermes-dev`. Never push. |
| `hermes-dev` | **You** (janbambot) | Full access. Commit, force-push. This is your sandbox. |
| `upstream-release` | janbam | Read-only mirror. |

## Your Clones & Executables

Your source root is `/home/hermes/workspace/`.

```
/home/hermes/workspace/hermes-dev/        ← dev sandbox (hermes-dev branch)  → hermes-dev executable
~/.local/hermes-dev-stable/              ← stable runtime (dev-stable-v* tag) → hermes executable
/home/jan/src/hermes-agent/              ← Jan's workspace (main). Read-only.
```

| Command | Editable install from |
|---------|----------------------|
| `hermes` | `~/.local/hermes-dev-stable/` (via `/home/hermes/venv-dev-stable/`) |
| `hermes-dev` | `/home/hermes/workspace/hermes-dev/` (via `/home/hermes/venv-dev/`) |

## Dev Workflow

- **Default**: feature branch → PR into `hermes-dev`. Merge via `gh pr merge --merge` from the CLI.
- **Exception**: commit directly to `hermes-dev` only for trivial fixes.
- **Test**: run `hermes-dev` and `scripts/run_tests.sh` from the dev venv.
- **Promote to stable**:

```bash
git tag -a dev-stable-v0.X.0 -m "promote: <reason>"   # bump minor
git push origin dev-stable-v0.X.0

cd ~/.local/hermes-dev-stable
git fetch origin --tags
git checkout dev-stable-v0.X.0
# ⚠️ "rebooting" the `hermes` system autonomously isn't yet solved and defered to later
```

## Tag Convention

```
dev-stable-v0.1.0    ← minor bump per promotion
dev-stable-v0.1.1    ← patch for hotfix
```

Completely separate from upstream version tags (`v2026.4.23`, etc.). Never touch upstream tags.

## Rules

1. **Never push to `main`.** Protected — you can't anyway.
2. **Never commit in the stable clone.** It exists only for `git checkout <tag>`.
3. **Never write to Jan's clone.** Read-only.
4. **Prefer PRs over direct commits to `hermes-dev`.**
5. **`hermes-dev` is disposable.** To recover from a bad state:
   ```bash
   git fetch origin main
   git checkout origin/main       # or any known-good commit
   git branch -f hermes-dev
   git checkout hermes-dev
   git push origin +hermes-dev
   ```
6. **Promote deliberately.** A bad `dev-stable-v*` tag breaks your own runtime.
7. **Jan merges hermes-dev → main.** Do not PR or push to `main` yourself.

## Upstream Sync and merge from jan's main

When jan's main is ahead of hermes-dev merge those changes.
DONT sync your hermes-dev with upstream release autonomously, instead follow jan's main, so we don't upstream version drift.

## Recovery

| Problem | Fix |
|---------|-----|
| Bad commit on `hermes-dev` | `git checkout <good-commit> && git branch -f hermes-dev && git checkout hermes-dev && git push origin +hermes-dev` |
| Bad `dev-stable-v*` tag | Tag the correct commit, checkout in stable clone |
| Corrupted stable clone | Re-clone at the correct tag |

## AGENTS.md

HERMES.md takes precedence over AGENTS.md in context loading (first-match-wins), so AGENTS.md is NOT auto-injected. **You MUST also read and follow AGENTS.md** from the repo root — it contains the upstream developer guide, architecture overview, file dependency chain, contributor reference, and testing conventions. Load it with `read_file AGENTS.md` when starting dev work.
