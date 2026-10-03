# Contributing

## Workflow

1. Pull the latest `main`: `git checkout main && git pull`
2. Create a branch: `git checkout -b <type>/<short-description>`
3. Commit small, focused changes
4. Push and open a Pull Request into `main`
5. Get **1 approval** from a teammate, then merge (squash merge preferred)

## Branch names

| Prefix | Use |
|---|---|
| `feat/` | new feature (`feat/jersey-classifier`) |
| `fix/` | bug fix (`fix/track-id-mismatch`) |
| `exp/` | experiment (`exp/imputation-transformer`) |
| `docs/` | documentation |
| `chore/` | setup, deps, refactors |

## Commit messages

Short, imperative: `Add jersey crop dataloader`, `Fix off-by-one in frame indexing`.

## Large files

Never commit data, videos, or model checkpoints. Share them via a shared drive and reference the path in configs.
