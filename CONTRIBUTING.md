# Contributing

## Branching model

- `main`: protected stable branch
- `codex/feat/*`: feature branches
- `codex/release/*`: release-prep branches

## Pull request workflow

1. Create a `codex/feat/*` branch from `main`.
2. Run local checks:
   - `ruff check .`
   - `pytest -q`
   - `python -m cavity_ml --help`
3. Open a PR with a clear summary, linked issue, and test evidence.
4. Wait for CI checks to pass before merge.

## Commit style

Use concise imperative commit messages, e.g. `Add data reconstruction validator`.

## Security and privacy

Do not commit:
- private datasets
- personal identity information
- secrets/tokens
- model binaries larger than policy allows
