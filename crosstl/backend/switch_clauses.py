"""Source-ordered switch clauses for the native-source importers.

The importer ASTs keep a switch's ``case`` clauses in ``cases`` and the
``default`` clause separately, as ``default_case``. A ``default`` label may sit
anywhere among the cases, and a clause without a jump falls through into the
next one in source order, so its position is part of the program's meaning::

    switch (x) {
        case 4:
        default: r = 1;   // x == 4 runs r = 1, then r = 2
        case 5:  r = 2;
    }

Each parser records the default's position as ``default_index``, the number of
case clauses before it. The importer code generators emit the clauses through
``ordered_switch_clauses`` so the CrossGL switch keeps that order. An empty
``default:`` is kept too, because it can fall through into a later case.
"""

DEFAULT_INDEX_ATTRIBUTE = "default_index"


def record_default_position(switch_node, default_index):
    """Store where the ``default`` clause appeared among the case clauses."""
    if default_index is not None:
        setattr(switch_node, DEFAULT_INDEX_ATTRIBUTE, int(default_index))
    return switch_node


def switch_default_statements(switch_node):
    """The statements of the ``default`` clause, or None when there is none."""
    for name in ("default_case", "default", "default_body"):
        statements = getattr(switch_node, name, None)
        if statements is not None:
            return statements
    return None


def ordered_switch_clauses(switch_node):
    """Return ``(value, statements)`` pairs in source order.

    ``value`` is None for the ``default`` clause. A switch node without a
    recorded position, such as one built by a lowering pass, gets its default
    after the cases.
    """
    cases = list(getattr(switch_node, "cases", None) or [])
    clauses = [
        (case.value, list(_case_statements(case)))
        for case in cases
        if getattr(case, "value", None) is not None
    ]
    # Cases that already carry a default (value None) keep their own position.
    if len(clauses) != len(cases):
        return [(case.value, list(_case_statements(case))) for case in cases]
    default = switch_default_statements(switch_node)
    if default is None:
        return clauses
    index = getattr(switch_node, DEFAULT_INDEX_ATTRIBUTE, None)
    if index is None or index > len(clauses):
        index = len(clauses)
    if not isinstance(default, list):
        default = [default]
    clauses.insert(max(int(index), 0), (None, list(default)))
    return clauses


def _case_statements(case):
    statements = getattr(case, "statements", None)
    if statements is None:
        statements = getattr(case, "body", None)
    return statements or []
