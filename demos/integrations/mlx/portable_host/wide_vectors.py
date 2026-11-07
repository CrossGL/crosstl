"""Launch geometry of the wide (``v2``) unary, binary and ternary vectors.

``unary_v2``, ``binary_vv2``, ``binary_sv2``, ``binary_vs2`` and
``ternary_v2`` in the pinned ``unary.h``, ``binary.h`` and ``ternary.h`` read
``N = WorkPerThread<T>::n`` (eight bytes of ``T``) consecutive elements per
invocation, starting at ``N * (x + grid.x * y)`` of a 2-D grid, and stop at
their int64 ``size``. An invocation with ``N > 1`` past the end writes
nothing; one with ``N == 1`` writes its element unconditionally, so a
one-element kernel needs a grid of exactly ``size`` invocations.

The portable host bounds every launch axis at 65535 one-thread groups. For a
vector of more than 65535 elements, ``backend.cpp`` launches the wide kernel
over at most ``MAX_ROWS`` rows of 65535 groups, so each launch moves at most
``8 * 65535 * MAX_ROWS`` bytes per operand; a one-element kernel's launch
covers whole rows, and the remaining fewer than 65535 elements run through
the narrow kernel. ``launches`` mirrors that split exactly.
"""

__all__ = (
    "MAX_GROUPS",
    "MAX_ROWS",
    "grid",
    "launches",
    "max_elements",
    "next_elements",
)

MAX_GROUPS = 65535
MAX_ROWS = 16
_WIDTHS = frozenset({1, 2, 4, 8})


def _require_width(per_thread):
    if type(per_thread) is not int or per_thread not in _WIDTHS:
        raise ValueError("Wide vectors read one, two, four or eight elements")


def max_elements(per_thread):
    """The most elements one wide launch covers."""
    _require_width(per_thread)
    return MAX_GROUPS * MAX_ROWS * per_thread


def grid(elements, per_thread):
    """The ``(columns, rows)`` of one-thread groups a wide launch uses."""
    _require_width(per_thread)
    if type(elements) is not int or not 0 < elements <= max_elements(per_thread):
        raise ValueError("Wide vector size exceeds the host contract")
    threads = -(-elements // per_thread)
    columns = min(threads, MAX_GROUPS)
    rows = -(-threads // columns)
    if per_thread == 1 and columns * rows != threads:
        raise ValueError("A one-element wide vector needs whole rows of groups")
    return columns, rows


def next_elements(remaining, per_thread):
    """The elements of the wide launch that starts ``remaining`` from the end.

    Only a vector with more than 65535 elements left launches wide.
    """
    if type(remaining) is not int or remaining <= MAX_GROUPS:
        raise ValueError("Only vectors of more than 65535 elements launch wide")
    elements = min(remaining, max_elements(per_thread))
    if per_thread == 1:
        elements -= elements % MAX_GROUPS
    return elements


def launches(count, per_thread, *, wide=True):
    """``(first, elements, wide)`` of every launch over ``count`` elements.

    ``wide`` says the host serves the entry's wide kernel; without it, every
    launch is a narrow one of at most 65535 elements.
    """
    _require_width(per_thread)
    result, first = [], 0
    while first < count:
        remaining = count - first
        if wide and remaining > MAX_GROUPS:
            elements = next_elements(remaining, per_thread)
            result.append((first, elements, True))
        else:
            elements = min(remaining, MAX_GROUPS)
            result.append((first, elements, False))
        first += elements
    return result
