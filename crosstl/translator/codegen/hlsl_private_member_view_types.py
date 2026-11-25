"""Type ``&local.member`` private-pointer views through the caller's locals.

The HLSL private pointer pre-pass classifies every pointer argument before any
function is emitted, so ``self.local_variable_types`` does not yet describe the
calling function. Without it, the type of ``box.bits`` falls back to a lookup
of ``bits`` across every structure, which has no answer when two structures
declare ``bits`` with different types (``uint`` and ``uint[2]``), and a scalar
member view is then misread as an array backing.

``hlsl_typed_member_view_root`` resolves the chain against the caller's
unambiguous local declarations (the pre-pass's ``local_types``) for the
duration of the check, so the root's declared structure decides the member.
"""

from contextlib import contextmanager

from .private_member_views import hlsl_member_view_root, member_view_chain


@contextmanager
def _local_types_in_scope(generator, local_types):
    previous = generator.local_variable_types
    scoped = dict(previous or {})
    for name in sorted(local_types):
        type_name = generator.type_name_string(local_types[name])
        if type_name:
            scoped[name] = type_name
    generator.local_variable_types = scoped
    try:
        yield
    finally:
        generator.local_variable_types = previous


def hlsl_typed_member_view_root(generator, operand, excluded_names, local_types):
    """``hlsl_member_view_root`` with the chain's root typed by ``local_types``.

    ``local_types`` maps the calling function's local names to their declared
    types; names declared with conflicting types are already left out, so the
    root keeps its default resolution in that case.
    """
    root, _members = member_view_chain(operand)
    if root is None or root not in (local_types or {}):
        return hlsl_member_view_root(generator, operand, excluded_names)
    with _local_types_in_scope(generator, {root: local_types[root]}):
        return hlsl_member_view_root(generator, operand, excluded_names)
