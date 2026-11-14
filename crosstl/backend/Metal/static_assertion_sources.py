"""Find the conditions of the static assertions that concrete Metal code keeps.

A ``static_assert`` at namespace or structure scope outside any template is concrete code: the CrossGL importer
emits it, and its condition is lowered like any expression. A condition such as
``(complex_t<half>{1.0h, 2.0h} * complex_t<half>{3.0h, 4.0h}).real == -5.0h`` therefore needs the free operator
templates instantiated for the types it names, even when no concrete function uses those types. Passes that decide
which instantiations are reachable from concrete code read these conditions alongside the concrete function bodies.

An assertion inside a template declaration is not concrete: each instantiation materializes its own copy.
"""

import re
from typing import Any, List, Tuple

_STATIC_ASSERT_RE = re.compile(r"\bstatic_assert\s*\(")


def concrete_static_assertion_conditions(
    preprocessor: Any, code: str, template_spans: List[Tuple[int, int]]
) -> List[str]:
    """The argument text of every ``static_assert`` in ``code`` outside ``template_spans``, in source order.

    The text includes an optional message argument; callers scan it for type names, and a message is a string
    literal that names none. A ``static_assert`` written inside a comment or a string literal is not an assertion.
    An assertion whose parentheses do not close is skipped here; the parser reports it.
    """
    if "static_assert" not in code:
        return []
    masked = preprocessor._mask_comments_and_literals(code)
    conditions: List[str] = []
    for match in _STATIC_ASSERT_RE.finditer(masked):
        if preprocessor._containing_span(match.start(), template_spans) is not None:
            continue
        open_paren = match.end() - 1
        close_paren = preprocessor._find_matching_delimiter(code, open_paren, "(", ")")
        if close_paren is None:
            continue
        conditions.append(code[open_paren + 1 : close_paren])
    return conditions
