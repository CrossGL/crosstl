"""Module-scope private variables become kernel locals passed by reference on CUDA and HIP.

CrossGL's module-scope ``static`` variable has one copy per invocation, which every function the invocation calls
reads and writes: SPIR-V ``Private`` storage, a WGSL ``var<private>``, an HLSL ``static`` global. Stage-scope private
variables and function-scope ``static`` locals reach the same form (:mod:`.stage_private_variables`,
:mod:`.static_locals`). CUDA and HIP have no per-thread storage at namespace scope. A namespace-scope variable
without a memory-space specifier is a host variable that device code cannot use, and a ``__device__`` variable is one
copy shared by every thread of the grid (CUDA C++ Programming Guide, "Variable Memory Space Specifiers"; HIP
follows CUDA here). The CUDA and HIP generators declared such a variable as written, so ``static int D = 4;`` became a
host variable that nvcc refuses to let the kernel read.

For those targets :func:`lower_device_private_globals` declares each such variable, with its initializer, as a
local at the start of every entry point that reaches it, directly or through the functions it calls, and passes it
by reference (an ``inout`` parameter, which both generators emit as a C++ reference) to every other function that
reaches it. The parameters come first, in declaration order, so a defaulted trailing parameter stays last, and every
function of one name receives the same ones, so each call passes the same arguments whichever overload it
resolves to. A variable whose name some function redeclares, as a parameter or a local, gets a fresh name first.
A variable without an initializer starts from zero, as an HLSL ``static`` does; a structure or array one is
value-initialized with ``{}``. A ``static`` variable whose type is not a plain value (an opaque type, a pointer, an
unsized array) is refused with :class:`DevicePrivateGlobalUnsupportedError` when a function uses it.
"""

from __future__ import annotations

from copy import deepcopy

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from ..stage_utils import SHADER_STAGE_BY_NAME, normalize_stage_name
from .hub_types import iter_functions
from .stage_private_variables import (
    _aliases,
    _is_value_type,
    _parameters,
    _qualifiers,
    _redeclares,
    _walk,
    _without_aliases,
)
from .static_locals import _declared_names, _fresh_name, _Renamer, _zero_value

__all__ = [
    "TARGETS",
    "DevicePrivateGlobalUnsupportedError",
    "lower_device_private_globals",
]

TARGETS = frozenset({"cuda", "hip"})

_ENTRY_QUALIFIERS = frozenset({"kernel", "__global__"})


class DevicePrivateGlobalUnsupportedError(TranslationDiagnostic):
    """A module-scope private variable that CUDA or HIP device code cannot hold per thread."""

    def __init__(self, message, target):
        super().__init__(
            message,
            target=target,
            code=f"{CODE_PREFIX}device-private-global-unsupported",
        )


def lower_device_private_globals(program, target) -> int:
    """Make module-scope private variables kernel locals passed by reference; returns how many moved."""
    target = str(target or "").lower()
    if not isinstance(program, A.ShaderNode) or target not in TARGETS:
        return 0
    globals_ = list(getattr(program, "global_variables", None) or [])
    statics = [variable for variable in globals_ if _is_static_variable(variable)]
    if not statics:
        return 0
    structs = {
        getattr(struct, "name", None): struct
        for struct in getattr(program, "structs", None) or []
    }
    aliases = _aliases(globals_, {})
    functions = [f for f in iter_functions(program) if isinstance(f, A.FunctionNode)]
    privates = []
    for variable in statics:
        if _is_value_type(getattr(variable, "var_type", None), structs, aliases, set()):
            variable.var_type = variable.vtype = _without_aliases(
                variable.var_type, aliases, frozenset()
            )
            privates.append(variable)
        elif any(_uses(function, variable.name) for function in functions):
            raise DevicePrivateGlobalUnsupportedError(
                f"The module-scope private variable '{variable.name}' has a type that is not a plain value, "
                f"so {target.upper()} device code cannot hold a copy of it per thread",
                target,
            )
    if not privates:
        return 0
    _make_names_unique(program, functions, privates)
    order = [variable.name for variable in privates]
    entry_ids = _entry_ids(program)
    helpers = [function for function in functions if id(function) not in entry_ids]
    needs = _helper_needs(helpers, set(order))
    initializer_needs = {
        variable.name: _reached(getattr(variable, "initial_value", None), needs, order)
        for variable in privates
    }
    for function in functions:
        _pass_arguments(getattr(function, "body", None), needs, order)
    for variable in privates:
        _pass_arguments(getattr(variable, "initial_value", None), needs, order)
    for helper in helpers:
        required = needs.get(helper.name) or set()
        helper.parameters[:0] = [
            A.ParameterNode(
                variable.name, deepcopy(variable.var_type), qualifiers=["inout"]
            )
            for variable in privates
            if variable.name in required
        ]
        if hasattr(helper, "params"):
            helper.params = helper.parameters
    for function in functions:
        if id(function) in entry_ids:
            required = _reached(getattr(function, "body", None), needs, order)
            _declare_locals(function, privates, required, initializer_needs)
    private_ids = {id(variable) for variable in privates}
    program.global_variables = [
        variable for variable in globals_ if id(variable) not in private_ids
    ]
    return len(privates)


def _is_static_variable(variable):
    """A mutable module-scope ``static`` variable with no storage, interface or resource qualifier."""
    if type(variable) is not A.VariableNode or not getattr(variable, "name", None):
        return False
    if not getattr(variable, "is_mutable", True) or _qualifiers(variable) != {"static"}:
        return False
    return not (
        getattr(variable, "semantic", None)
        or getattr(variable, "attributes", None)
        or getattr(variable, "resource_qualifiers", None)
        or getattr(variable, "is_type_alias", False)
    )


def _entry_ids(program):
    """The ids of the functions the generators emit as entry points: stage entries and stage-attributed ones."""
    stages = getattr(program, "stages", None) or {}
    stages = stages.values() if isinstance(stages, dict) else stages
    ids = {
        id(stage.entry_point)
        for stage in stages
        if getattr(stage, "entry_point", None) is not None
    }
    for function in getattr(program, "functions", None) or []:
        names = list(getattr(function, "qualifiers", None) or [])
        names += [
            getattr(a, "name", None)
            for a in getattr(function, "attributes", None) or []
        ]
        if any(
            str(name).lower() in _ENTRY_QUALIFIERS
            or normalize_stage_name(name) in SHADER_STAGE_BY_NAME
            for name in names
            if name
        ):
            ids.add(id(function))
    return ids


def _make_names_unique(program, functions, privates):
    """Rename each variable whose name a function redeclares, so a parameter or local can carry it."""
    taken = _declared_names(program)
    for variable in privates:
        old = variable.name
        if not any(_redeclares(function, old) for function in functions):
            continue
        new = _fresh_name(old, taken)
        renamer = _Renamer(None, old, new)
        for function in functions:
            if any(getattr(p, "name", None) == old for p in _parameters(function)):
                continue
            body = getattr(function, "body", None)
            if isinstance(body, list):
                renamer._sequence(body, active=True)
            elif isinstance(body, A.ASTNode):
                renamer._node(body, active=True)
        for other in privates:
            value = getattr(other, "initial_value", None)
            if isinstance(value, A.ASTNode):
                renamer._node(value, active=True)
        variable.name = new


def _callee_name(call):
    callee = getattr(call, "function", None)
    name = getattr(callee, "name", callee)
    return name if isinstance(name, str) else None


def _direct_uses_and_calls(root, names):
    used, calls = set(), set()
    for node in _walk(root):
        if isinstance(node, A.IdentifierNode) and node.name in names:
            used.add(node.name)
        elif isinstance(node, A.FunctionCallNode):
            name = _callee_name(node)
            if name is not None:
                calls.add(name)
    return used, calls


def _uses(function, name):
    return name in _direct_uses_and_calls(getattr(function, "body", None), {name})[0]


def _helper_needs(helpers, names):
    """The variables each helper name reaches, directly or through the helpers it calls, to a fixed point."""
    direct = {}
    for helper in helpers:
        used, calls = _direct_uses_and_calls(getattr(helper, "body", None), names)
        entry = direct.setdefault(helper.name, (set(), set()))
        entry[0].update(used)
        entry[1].update(calls)
    needs = {name: set(used) for name, (used, _) in direct.items()}
    changed = True
    while changed:
        changed = False
        for name, (_, calls) in direct.items():
            reached = set().union(*(needs.get(callee, set()) for callee in calls))
            if not reached <= needs[name]:
                needs[name] |= reached
                changed = True
    return needs


def _reached(root, needs, order):
    used, calls = _direct_uses_and_calls(root, set(order))
    for callee in calls:
        used |= needs.get(callee, set())
    return used


def _pass_arguments(root, needs, order):
    """Give each call of a helper that reaches variables those variables, first and in declaration order."""
    for node in _walk(root):
        if not isinstance(node, A.FunctionCallNode):
            continue
        required = needs.get(_callee_name(node))
        if required:
            # In place: ``args`` is the same list.
            node.arguments[:0] = [
                A.IdentifierNode(name) for name in order if name in required
            ]


def _declare_locals(entry, privates, required, initializer_needs):
    """Declare the variables an entry point reaches, and those their initializers read, at its start."""
    pending = list(required)
    while pending:
        for name in initializer_needs.get(pending.pop(), ()):
            if name not in required:
                required.add(name)
                pending.append(name)
    declarations = []
    for variable in privates:
        if variable.name not in required:
            continue
        value = getattr(variable, "initial_value", None)
        if value is None:
            value = _zero_value(variable.var_type) or A.ArrayLiteralNode([])
        else:
            value = deepcopy(value)
        declarations.append(
            A.VariableNode(variable.name, deepcopy(variable.var_type), value)
        )
    if not declarations:
        return
    body = getattr(entry, "body", None)
    if isinstance(body, list):
        body[:0] = declarations
    elif isinstance(body, A.BlockNode):
        body.statements[:0] = declarations
    else:
        entry.body = A.BlockNode(declarations + ([body] if body else []))
