# Releases

`release.yml` builds the wheel and source distribution on Ubuntu. A pull request that changes a packaged file, or a
manual run, only builds and checks them. Pushing a `v*.*.*` tag builds, checks and publishes them.

## Prepare

1. Open a pull request that sets the new version in `pyproject.toml`, `CITATION.cff` and `docs/source/conf.py`,
   renames `## [Unreleased]` in `CHANGELOG.md` to `## [<version>] - <YYYY-MM-DD>`, and sets `date-released` in
   `CITATION.cff` to the same date. `tests/test_release.py` fails until these agree.
2. Check the archives locally in a clean checkout:

   ```bash
   python -m pip install build==1.2.2.post1 twine==7.0.0
   python -m build
   python -m twine check --strict dist/*
   python tools/check_release.py distributions dist
   ```

3. Merge the pull request through the merge queue.

## Publish

1. Wait for the push runs on `main` at the version commit to pass. The tag workflow publishes only if
   `full-tests.yml`, `backend-tests.yml`, `translator-tests.yml`, `docs.yml`, `examples-test.yml`,
   `native-host-loader.yml`, `deferred-native-compilation.yml` and `demo-project-testing.yml` each have a successful
   push run at that exact commit. The last three run on `main` only when their paths change, and every one of those
   path lists includes `pyproject.toml`, so tag the commit that changed the version.
2. Tag that commit `v<version>` and push the tag. The workflow rejects a tag that is not on `main` or does not match
   the package version.
3. The build job checks the archive metadata and contents, installs each archive in its own environment on Python
   3.11 and 3.12, and runs the CLI and a translation to DirectX, OpenGL and Metal from each install. The PyPI job
   uploads those archives with the `PYPI_TOKEN` secret. The GitHub job checks that PyPI serves the same files, then
   creates the release from them with the version's changelog section as its notes. Zenodo archives the release
   and takes its version from the tag.
4. Check both files on PyPI and on the GitHub release.

If a required run is still in progress, the tag workflow fails before publishing; rerun it once the run has passed.
Rerunning after an interrupted upload is safe: PyPI skips files it already has, and the GitHub job updates the
existing release. Never move a published tag or replace a published version; release a new version instead.
