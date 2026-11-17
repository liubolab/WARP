# OpenWAM source package

This package contains the main OpenWAM source tree: Python packages, model and
data configurations, training/deployment scripts, benchmark clients, Docker
files, tests, usage documentation, repository images, and RoboDojo verification
records.

Large downloaded assets are intentionally excluded: model checkpoints,
benchmark datasets and videos, generated outputs, experiment results, caches,
and third-party build products. Download assets with the scripts in
`scripts/download_assets/` after installing the dependencies described in
`README.md`.

The package preserves source files as stored. The accompanying
`OPENWAM_SOURCE_MANIFEST_20261117.json` records a SHA-256 digest for every
packaged file. The repository's separate `research/` archives contain the
geometry-prior experiment code and saved results.
