"""Function pointer and function reference template parameters.

A non-type template parameter can be a function pointer or reference::

    template <typename T, void (*load)(device const T *, short, thread float &), short N>

Its name sits inside the declarator rather than at the end of it, so the usual "last identifier is the name"
rule finds nothing and the parameter used to be filed as an anonymous constraint. Explicit template
arguments are bound positionally, so every argument after it then bound to the wrong parameter.
"""

import re
from typing import Optional, Tuple

_FUNCTION_DECLARATOR = re.compile(
    r"(?P<result>[^()]+?)\s*\(\s*(?P<kind>[*&])\s*(?P<name>[A-Za-z_]\w*)\s*\)"
)


def template_function_pointer_parameter(declarator: str) -> Optional[Tuple[str, str]]:
    """Return ``(name, declared_type)`` for ``R (*name)(params)`` or ``R (&name)(params)``.

    ``declared_type`` is the abstract declarator, such as ``void (*)(device const T *, short)``. Anything
    else, including an unnamed function pointer, returns ``None``.
    """
    text = " ".join(declarator.split())
    if not text.endswith(")"):
        return None
    close = len(text) - 1
    depth = 0
    params_open = None
    for index in range(close, -1, -1):
        character = text[index]
        if character == ")":
            depth += 1
        elif character == "(":
            depth -= 1
            if depth == 0:
                params_open = index
                break
    if params_open is None:
        return None
    match = _FUNCTION_DECLARATOR.fullmatch(text[:params_open].rstrip())
    if match is None:
        return None
    result = match.group("result").strip()
    if not result:
        return None
    parameters = text[params_open + 1 : close].strip()
    return match.group("name"), f"{result} ({match.group('kind')})({parameters})"
