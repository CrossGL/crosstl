"""Write one page per language from the published coverage report before Sphinx reads the sources.

The report comes from ``CROSSTL_COVERAGE`` when it is set, either a directory with the files of the ``coverage``
branch or the URL they are published under, and otherwise from :data:`tools.coverage.docs.PUBLISHED`. A build that
cannot read it writes an index page that links to ``COVERAGE.md`` instead, unless ``CROSSTL_COVERAGE_REQUIRED=1``
asks it to fail, as the Documentation workflow does.
"""

from __future__ import annotations

import os
from pathlib import Path

from sphinx.application import Sphinx
from sphinx.errors import ExtensionError
from sphinx.util import logging

from tools.coverage import docs

LOGGER = logging.getLogger(__name__)
# The pages go here, under the documentation sources; git ignores the directory.
DIRECTORY = "languages"
# The document name of each language's backend guide under ``backends/``, for the languages that have one.
guides = docs.guides


def generate(srcdir: Path, source: str, required: bool) -> list[Path]:
    """Write the pages into ``srcdir/languages`` from the report at ``source`` and return the files written."""
    try:
        data = docs.load(source)
        files = docs.pages(data, guides(srcdir, docs.languages(data)))
    except (OSError, ValueError, KeyError, TypeError) as error:
        if required:
            raise ExtensionError(
                f"cannot read the coverage report from {source}: {error}"
            ) from error
        LOGGER.info(
            "coverage pages: cannot read %s (%s); writing an index without them",
            source,
            error,
        )
        files = {
            "index.md": docs.unavailable(source, f"{type(error).__name__}: {error}")
        }
    return docs.write(srcdir / DIRECTORY, files)


def _builder_inited(app: Sphinx) -> None:
    source = os.environ.get("CROSSTL_COVERAGE") or docs.PUBLISHED
    required = os.environ.get("CROSSTL_COVERAGE_REQUIRED") == "1"
    written = generate(Path(app.srcdir), source, required)
    LOGGER.info("coverage pages: %d written from %s", len(written), source)


def setup(app: Sphinx) -> dict:
    app.connect("builder-inited", _builder_inited)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
