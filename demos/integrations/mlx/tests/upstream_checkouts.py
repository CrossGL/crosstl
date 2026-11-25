"""Read the upstream MLX checkouts that the MLX workflow makes.

Every job checks out MLX through ``demos.integrations.mlx.fetch_upstream``,
which fixes the bytes (``core.autocrlf`` off), the commit and a clean tree.
Workflow tests therefore assert which directory, commit variable and sparse
directories each step passes to it.
"""

import re

KERNEL_TREE = ("mlx/backend/metal/kernels",)
_FETCH = re.compile(
    r"python -m demos\.integrations\.mlx\.fetch_upstream"
    r' (?P<directory>\S+) "(?P<commit>\$\w+)"'
    r"(?: --sparse (?P<sparse>[^\n]+?))?[ \t]*$",
    re.MULTILINE,
)


def upstream_fetches(text):
    """Return ``(directory, commit variable, sparse directories)`` per fetch."""
    return [
        (
            match["directory"],
            match["commit"],
            tuple(match["sparse"].split()) if match["sparse"] else (),
        )
        for match in _FETCH.finditer(text)
    ]


def fetches(text, commit, directory=None, sparse=KERNEL_TREE):
    """Whether ``text`` fetches ``commit`` with these sparse directories."""
    return any(
        found_commit == commit
        and found_sparse == sparse
        and directory in (None, found_directory)
        for found_directory, found_commit, found_sparse in upstream_fetches(text)
    )
