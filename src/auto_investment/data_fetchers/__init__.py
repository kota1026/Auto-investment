"""Real-data fetchers for Phase 2.

Each module has a `fetch(...)` that hits the live API and a `load(...)`
that reads from local parquet cache. Callers should always go through
`load_or_fetch(...)` so the fast path is the cache.

The split exists because the sandbox where Claude Code runs may not have
network egress to crypto APIs; the user runs `scripts/fetch_real_data.py`
on their own machine to populate `data/` and commits (or `.gitignore`s)
the cache.
"""

CACHE_VERSION = 1
