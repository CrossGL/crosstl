"""Retain compiled native modules under their content hash.

Threads that compile the same module retain it under the same name, and the
inline dispatch audit reads a retained module back while other threads may
still be retaining it. A module is therefore written to a private temporary
file and renamed onto its name, so a reader sees a whole module or none.
"""

import hashlib
import os
import tempfile
from pathlib import Path


def retain(directory, data, suffix):
    """Store ``data`` as ``<sha256><suffix>`` in ``directory``; return its record."""
    if not data:
        raise RuntimeError("Native compiler produced an empty module")
    digest = hashlib.sha256(data).hexdigest()
    directory = Path(directory)
    destination = directory / (digest + suffix)
    if not destination.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".retain-", suffix=suffix, dir=directory
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
            # The name holds the same bytes whichever thread renames last.
            os.replace(temporary, destination)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
    return {"file": str(destination), "sha256": digest}
