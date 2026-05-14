# Agent Working Rules (Project-specific)

## Environment
- All experiments must run inside a Python virtual environment.
- All dependencies must be installed inside that virtual environment only.
- Do not install dependencies globally.

## Experiment Logging
- After each completed improvement or experiment, append a record to `log.md`.
- Each record must include:
  - What was changed
  - Result (metrics, observations, or failure details)

## Execution Policy
- Do not add broad fallback logic or “catch-all”兜底 behavior unless explicitly requested.
- Keep changes targeted, explicit, and traceable to the experiment goal.
