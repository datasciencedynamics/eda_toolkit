# Changelog

All notable changes to `eda_toolkit` are documented here.

## [Unreleased]

### Added
- `eda_toolkit.about()` prints the package logo, overview, and version.
- `CITATION.cff` with author ORCIDs and the Zenodo DOI.
- GitHub Actions test workflow running the suite on Python 3.10 through 3.13.

### Changed
- Dependency pins loosened to minimum versions only. The library is now
  tested against current releases of numpy (2.x), pandas (3.x), matplotlib,
  seaborn, plotly (7.x), scikit-learn, scipy, tqdm, nbformat, and xlsxwriter.
- `pyproject.toml` is the single source of package metadata; `setup.py` is a
  thin shim.
- Repository URLs updated to `datasciencedynamics/eda_toolkit`.

### Fixed
- `summarize_all_combinations` table-of-contents hyperlinks now use
  xlsxwriter's documented `internal:` syntax, which is required by
  xlsxwriter 3.3 and later.

### Removed
- Importing `eda_toolkit` no longer overrides Python's built-in `help()`.
  Use `eda_toolkit.about()` for the package overview.

## [0.0.30]

Last release prior to this changelog.
