# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with RHEL AI midstream extensions.

## [0.14.1+rhaiv.1] - 2026-03-05

### Changed

**New versioning scheme**: This release adopts the RHEL AI midstream versioning pattern `X.Y.Z+rhaiv.N` (e.g., `0.14.1+rhaiv.1`).

- Automatic version management via setuptools-scm (derived from git tags)
- Build system migrated from flit_core to setuptools + setuptools-scm
- Tags are now immutable - never deleted or moved
- Build counter increments for fixes, resets on upstream version bumps

### For Users

- **Installation unchanged**: `pip install garak` continues to work
- **Version pinning**: Use full version string (e.g., `garak==0.14.1+rhaiv.1`) for reproducible builds
- **Version checking**: `uv run python -c "import garak; print(garak.__version__)"`

See the [Versioning](#versioning) section in README for details.

---

## [0.14.1.pre1] - 2026-03-04 and earlier

Previous releases used traditional versioning without the RHEL AI midstream pattern.
