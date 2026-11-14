"""Calls that bind two private pointer views to one backing object.

HLSL has no private pointers. A private pointer parameter becomes a copy of its
backing array (or scalar) plus a base offset: an ``inout`` copy when the callee
may write through it, and a read-only copy when it is ``const``. Two views of
one backing object in one call are therefore two independent copies.

- ``inout`` copies are written back one after another when the callee returns,
  so the copy written back last overwrites every write made through the other
  view, even when the two views address disjoint elements.
- A read-only copy is taken at the call, so it never observes a write made
  through a writable view of the same object during the call.

Only calls whose views of one object are all read-only keep the source
semantics. Any other aliasing call must fail closed; this module finds it.

Two views share a backing object when they name the same caller local, or the
same caller pointer parameter. Views rooted at two different caller
parameters cannot alias, because every call that binds those parameters is
itself checked here.
"""

#: Binding root kinds that name a caller local (array, word view or scalar).
LOCAL_ROOT_KINDS = frozenset({"local", "local_word_view", "scalar"})

#: Qualifiers that make an HLSL private pointer parameter a read-only copy.
READ_ONLY_QUALIFIERS = frozenset({"const", "in"})


def backing_key(binding):
    """Identity of the caller object a view binding names, or ``None``."""
    root = binding.get("root")
    if not root:
        return None
    root_kind = binding.get("root_kind")
    if root_kind == "parameter":
        return ("parameter", root)
    if root_kind in LOCAL_ROOT_KINDS:
        return ("local", root)
    return None


def hlsl_writable_pointer_parameters(generator, pointer_parameters):
    """Names of the private pointer parameters HLSL passes as ``inout``.

    ``pointer_parameters`` maps each function name to its private pointer
    parameter nodes. A parameter is writable unless it is declared ``const``
    or ``in``, or its pointee type is immutable.
    """
    writable = {}
    for function_name, parameters in pointer_parameters.items():
        names = set()
        for parameter in parameters:
            qualifiers = generator.hlsl_parameter_direction_qualifiers(parameter)
            if qualifiers & READ_ONLY_QUALIFIERS:
                continue
            parameter_type = getattr(
                parameter, "param_type", getattr(parameter, "vtype", None)
            )
            # The same test HLSLCodeGen.hlsl_parameter_qualifiers uses to add
            # ``inout`` to a private pointer parameter.
            if not getattr(parameter_type, "is_mutable", True):
                continue
            names.add(parameter.name)
        writable[function_name] = frozenset(names)
    return writable


def view_is_writable(parameter_name, binding, writable_parameters):
    """True when the callee receives this view as an ``inout`` copy."""
    if parameter_name not in writable_parameters:
        return False
    # A word view of a local is always passed as a read-only copy.
    return binding.get("root_kind") != "local_word_view"


def hlsl_aliased_writable_view(view_calls, writable_parameters):
    """First call that binds a writable view and another view to one object.

    ``view_calls`` maps each caller name to its recorded private pointer
    calls; ``writable_parameters`` maps each callee name to the parameters it
    receives as ``inout`` copies. Returns ``(call, first_parameter,
    second_parameter, root)`` for the first such call in deterministic order,
    or ``None`` when every shared object is only read.
    """
    for caller_name in sorted(view_calls):
        for call in view_calls[caller_name]:
            callee_writable = writable_parameters.get(call["callee"], frozenset())
            groups = {}
            for parameter_name, binding in call["bindings"].items():
                key = backing_key(binding)
                if key is not None:
                    groups.setdefault(key, []).append((parameter_name, binding))
            for (_kind, root), views in groups.items():
                if len(views) < 2:
                    continue
                if not any(
                    view_is_writable(name, binding, callee_writable)
                    for name, binding in views
                ):
                    continue
                return call, views[0][0], views[1][0], root
    return None
