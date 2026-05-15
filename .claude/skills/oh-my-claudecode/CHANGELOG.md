# oh-my-claudecode v4.14.0: ultragoal port, autopilot/team launch hardening, plugin skill registry

## Release Notes

Release with **1 new feature**, **11 bug fixes / hardening changes** across **12 merged PRs**.

### Highlights

- **feat(ultragoal): port OMX ultragoal to OMC** (#2995) — durable multi-goal workflow with `omc ultragoal` CLI, persisted plan/ledger artifacts under `.omc/ultragoal`, and Claude Code `/goal` handoff guidance. Checkpointing fails closed: only the active in-progress microgoal can be completed, failed, or blocked.
- **Preserve Claude Code across OMC updates** (#2997) — `omc update` now snapshots the global `@anthropic-ai/claude-code` install before running and restores it if the OMC update path removes it; users without a prior global Claude Code install are left untouched.
- **Keep plugin skill registry concise** (#2989) — re-register every bundled plugin skill now that plugin cache sync compacts native `skills/*/SKILL.md` files into small registry shims, while preserving full on-demand skill bodies under `skill-bodies/*/SKILL.md` (loaded via a Windows-safe `omc-full-body` pointer at invocation time).

### New Features

- **feat(ultragoal): port OMX ultragoal to OMC** (#2995) — @probepark

### Bug Fixes & Hardening

- **Fix team launch fixed worker plans** (#3011) — honor explicit `N:agent` worker specs for fixed/pre-authored team launch plans, add `--no-decompose`, fail closed when explicit worker count and decomposed scope count disagree, ignore dead tmux-backed stale team state during spawn gating.
- **fix(pre-tool-enforcer): cover extra naming slop boundary** (#3013) — neutral domain qualifiers like `extra`/`additional` (e.g. `extraSecretFetch`, `extraSecrets []extraSecretFetch`) no longer trigger the SLOP fallback warning.
- **Harden HUD cache stale lock cleanup** (#3003) — resolve the HUD cache dir before cleanup, GC stale render locks early, remove only zero-byte HUD temp/error files (preserve non-empty diagnostics), and clean up render locks and stdout temp files on background-refresh EXIT/signal without blocking the statusLine hot path.
- **Fix autopilot state cleanup stop-hook loop** (#3001) — make persistent-mode autopilot enforcement read `phase ?? current_phase ?? "unspecified"`, clean orphan autopilot routing echoes, prevent `state_clear` from clearing a live singleton autopilot owned by another session, and write/read autopilot `phase`/`current_phase` aliases for compatibility.
- **Preserve Claude Code across OMC updates** (#2997) — see Highlights.
- **Fix doctor warning for synced omc-reference fallback** (#2993) — suppress doctor legacy-skill collision warnings only when `~/.claude/skills/omc-reference/SKILL.md` byte-for-byte matches the bundled fallback; modified copies and non-contract `omc-reference.md` legacy files still warn.
- **Suppress persistent reinforcement after oversized tool redirects** (#2990) — focused Stop classifier requires both a `tool-results/*.txt` pointer and oversize/redirect wording; reinforcement is suppressed only for the first few consecutive redirect stops, then normal Ralph/todo stall protection resumes.
- **Keep plugin skill registry concise** (#2989) — see Highlights.
- **Fix doctor package version lookup** (#2982).
- **Fix launch credential mirroring** (#2979).
- **Fix cancel-ralph skill alias** (#2974).

### Stats

- **12 PRs merged** | **1 new feature** | **11 bug fixes / hardening changes**

### Install / Update

```bash
npm install -g oh-my-claude-sisyphus@4.14.0
```

Or reinstall the plugin:
```bash
claude /install-plugin oh-my-claudecode
```

**Full Changelog**: https://github.com/Yeachan-Heo/oh-my-claudecode/compare/v4.13.7...v4.14.0

## Contributors

Thank you to all contributors who made this release possible!

@probepark @Yeachan-Heo
