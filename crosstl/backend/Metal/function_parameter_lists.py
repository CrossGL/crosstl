"""Locate the parameter list of a function definition.

Reference-alias lowering scans a function body for the declarations of the
receivers its aliases bind to. Starting that scan at the parameter list makes
parameters visible as receivers, the same way block-scope declarations are.
"""

from __future__ import annotations


def parameter_list_start(masked: str, declaration_start: int, body_open: int) -> int:
    """Return the offset of the ``(`` that opens the parameter list.

    ``masked`` is the source with comments and literals blanked out,
    ``declaration_start`` the first character of the function's declaration
    and ``body_open`` the offset of its ``{``. The list must close right
    before the body. Otherwise ``body_open`` is returned, so the caller scans
    the body alone.
    """
    close = body_open - 1
    while close >= declaration_start and masked[close].isspace():
        close -= 1
    if close < declaration_start or masked[close] != ")":
        return body_open
    depth = 0
    for index in range(close, declaration_start - 1, -1):
        character = masked[index]
        if character == ")":
            depth += 1
        elif character == "(":
            depth -= 1
            if depth == 0:
                return index
    return body_open
