# Dungeon Master — Project Guidelines

A standalone application. **This project is independent — it has no relationship to any
other project on this machine.** Do not import code, patterns, or context from elsewhere.

## What it is
Dungeon Master — a tool/assistant for running tabletop RPG (D&D-style) sessions.
(Refine this vision in README.md as the direction firms up: e.g. campaign & NPC
management, encounter/initiative tracking, dice, rules lookup, AI-narrated adventures.)

## Tech stack
_To be decided in the first working session._ Once chosen, record it here (language,
framework, how to run, how to test) so future sessions have the setup at a glance.

## Coding standards
- Keep it simple. Build the smallest thing that works, then iterate.
- No premature abstraction, no speculative features, no dead code.
- Comment only where the *why* is non-obvious.
- Prefer editing existing files over adding new ones.
- Validate at boundaries (user input, external APIs); trust internal calls.

## Docs
- `README.md` — the vision, setup, and how to run it.
- Add an `ARCHITECTURE.md` once the design has real structure worth explaining.

## Autonomy
- Run ordinary dev commands (build, test, run, git) without asking.
- Confirm before anything irreversible or outward-facing (publishing, deploying, deleting).
