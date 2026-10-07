# Releases

The `Release` workflow builds a wheel and source distribution on Ubuntu. Pull
requests affecting packaging and manual workflow runs validate both archives and
install each in a separate environment. These runs never publish packages or
create a GitHub Release. Native runtime execution remains in the platform test
workflows; the package check does not add Windows or macOS runners.

## Prepare

1. Set the same version in `setup.py`, `CITATION.cff` and `docs/source/conf.py`.
2. Add a dated section to `CHANGELOG.md` and update the citation release date.
   Keep compiler coverage, numerical execution and project integration claims
   separate. Do not describe incomplete external-project ports as supported
   replacement backends.
3. Run the release tests and all-file pre-commit checks. Build both archives with
   `python -m build`, then run `python -m twine check --strict dist/*` and
   `python tools/check_release.py distributions dist` in a clean output directory.
4. Require the pull request's package check and applicable numerical, project,
   platform and unit checks to pass before merging the checkpoint to `main`.

## Publish

1. Wait for validation of the merged commit on `main`. The tag workflow requires
   successful main-push runs of `full-tests.yml`, `backend-tests.yml`,
   `translator-tests.yml`, `docs.yml` and `examples-test.yml` at that exact commit.
   A successful pull-request merge ref or an earlier main commit is not accepted.
2. Tag the validated commit as `v<version>` and push that tag. The workflow rejects
   tags outside `main` and tags that disagree with the package version.
3. The build checks archive metadata, every packaged source file, the native
   DirectX worker, isolated wheel/source installs, all three CLI entry points and
   translation to DirectX, OpenGL and Metal. It uploads the checked archives once.
4. The PyPI job downloads those archives and publishes using the existing
   `PYPI_TOKEN` repository secret. After that job succeeds, the GitHub job uses
   the same archives and the version's changelog section to create the release.
5. Verify both package files on PyPI and both attached files on GitHub. Record the
   release run and commit in the checkpoint pull request.

If main checks are still queued, the tag workflow fails without publishing.
Rerun it after the required checks succeed. An interrupted upload can be retried:
PyPI skips existing files, and GitHub updates the existing release. Never move an
already-published tag or replace a published PyPI version with different source;
use a new version for corrections.
