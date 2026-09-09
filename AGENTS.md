# Agent instructions for QuantEcon.py

Guidance for AI coding agents working in this repository. Human contributors should start from
[CONTRIBUTING.md](CONTRIBUTING.md) and the rendered guide it links to; everything there applies to
agents too — this file only adds the essentials and repo-specific conventions.

## Quickstart

- Environment: `conda env create -f environment.yml` (creates `qe`), activate it, then
  `flit install --symlink`. Environment creation takes several minutes — let it finish.
- Tests: `pytest quantecon` (bare `pytest` also works — `pytest.ini` scopes collection to
  `quantecon/`). The full suite takes minutes; run `pytest quantecon/<module>/tests/...` while
  iterating. Much of the library is Numba-jitted, so the first call of anything compiled is slow —
  don't mistake JIT compilation for a hang. Slow tests can be skipped with `-m "not slow"`.
- Lint (same selects as CI): `flake8 --select=F401,F405,E231 quantecon`
- The `ci/` directory holds CI-only assets (for example the WASM smoke suite in `ci/wasm/`). They
  are not part of the shipped package and are run by explicit path in workflows — keep imports of
  CI-only dependencies (such as `playwright`) out of anything bare `pytest` collects.

## Working procedure

- Make small, self-contained pull requests against `main`; each must be green in CI and safe to
  release on its own. Do not create long-lived feature branches — multi-phase work merges to `main`
  incrementally. See "Multi-phase projects and releases" in
  [docs/source/contributing.rst](docs/source/contributing.rst) for the full procedure and its
  rationale.
- Keep `main` releasable: publishing to PyPI is automated on `v*` tags, so anything merged can ship
  at any time. Do not change default behaviour of library functions unless that is the reviewed
  purpose of the change.
- Larger campaigns are tracked as an umbrella issue plus sub-issues under a milestone; read the
  umbrella issue before working on a sub-issue, and record findings in the issues — they are the
  durable record.
- Release notes live on [GitHub releases](https://github.com/QuantEcon/QuantEcon.py/releases);
  `CHANGELOG.md` only points there. Do not add per-PR changelog entries.
