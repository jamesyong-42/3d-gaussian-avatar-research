# Contributing

Start with [the README](README.md), [development guide](docs/development.md) and [research boundaries](docs/research.md). The original code's license is still undecided; discuss intended external contributions with the maintainer before submitting substantial work.

Keep changes focused. Run `python scripts/test.py` and `python scripts/check-publication.py`; run browser checks for UI, rendering or receiver changes. Preserve native numerical behavior when optimizing and retain explicit failure/fallback reporting. Do not silently prune skin weights, reinterpret expression channels, or promote an experimental model without independent geometry and control checks.

Use synthetic fixtures in public tests. Never attach personal photos, generated human assets, restricted model templates, checkpoints, credentials or unreviewed browser traces. Report reproducible environment details without private paths. Clearly separate measured results, paper claims, inferences and future plans.

Update `docs/index.html` when architecture decisions change. Keep diagrams readable on mobile and source links correct. Only that audited HTML is deployed to Pages; the API and local data never belong in the Pages artifact.
