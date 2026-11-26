"""Kernel arguments of several compute entries in one HLSL program.

CUDA, HIP and OpenCL C kernels take their buffers and scalars as parameters. Before a DirectX target sees them,
:func:`crosstl.backend.OpenCL.target_lowering.normalize_opencl_intermediate_for_target` hoists each kernel's
parameters to program scope, a resource per buffer or image parameter and one ``<kernel>_Args`` constant buffer for
the scalars, and marks every hoisted declaration with the kernel it came from (the ``entry_argument`` annotation).
HLSL has one global namespace: the members of a ``cbuffer`` are global names, so two cbuffers that both declare
``width`` redefine it, and a resource name denotes one register. Two kernels whose parameters share a name then
collide:

- ``int width`` in two kernels becomes a member of two cbuffers;
- ``part`` bound at ``u0`` in one kernel and at ``u1`` in another, or with another element type, is two resources
  under one name.

:func:`drop_other_entry_arguments` removes the other kernels' declarations from a program scoped to one entry
(``entry_point=``), before declarations are pruned by reachability: by name, kernel ``copy``'s ``width`` reaches
kernel ``copy2``'s ``width`` too. :func:`separate_entry_argument_names` keeps every kernel of a whole program by
renaming a later kernel's colliding declarations to ``<kernel>_<name>``, together with that kernel's references to
them. A declaration identical in every field to the earlier kernel's (same type, binding and qualifiers) stays shared:
the generator merges the two into one resource, which both kernels then read.
"""

from __future__ import annotations

from copy import deepcopy

from .. import ast as A
from .stage_utils import declaration_signature

__all__ = [
    "ENTRY_ARGUMENT_ANNOTATION",
    "drop_other_entry_arguments",
    "separate_entry_argument_names",
]

# Set by crosstl.backend.OpenCL.target_lowering on every declaration hoisted from a kernel parameter.
ENTRY_ARGUMENT_ANNOTATION = "entry_argument"

_DECLARATION_GROUPS = ("global_variables", "cbuffers")
_SKIPPED_FIELDS = frozenset(
    {"parent", "annotations", "source_location", "expression_type"}
)
# A call's callee is a function name, never a reference to a hoisted argument.
_CALLEE_FIELDS = frozenset({"function", "name"})


def _entry_of(declaration):
    """The kernel ``declaration`` was hoisted from, or ``None`` for an ordinary module declaration."""
    get_annotation = getattr(declaration, "get_annotation", None)
    if not callable(get_annotation):
        return None
    entry = get_annotation(ENTRY_ARGUMENT_ANNOTATION)
    return entry if isinstance(entry, str) and entry else None


def drop_other_entry_arguments(program, entry_name):
    """Remove from ``program`` the declarations hoisted from kernels other than ``entry_name``."""
    for attribute in _DECLARATION_GROUPS:
        declarations = getattr(program, attribute, None)
        if not declarations:
            continue
        setattr(
            program,
            attribute,
            [
                declaration
                for declaration in declarations
                if _entry_of(declaration) in (None, entry_name)
            ],
        )


def _kernel_order(program):
    """The kernels that own hoisted declarations, in the order of their entries, then of their declarations."""
    order = []
    for _stage_type, stage in getattr(program, "stages", {}).items():
        name = getattr(getattr(stage, "entry_point", None), "name", None)
        if isinstance(name, str) and name not in order:
            order.append(name)
    for function in getattr(program, "functions", []) or []:
        name = getattr(function, "name", None)
        if isinstance(name, str) and name not in order:
            order.append(name)
    for attribute in _DECLARATION_GROUPS:
        for declaration in getattr(program, attribute, []) or []:
            entry = _entry_of(declaration)
            if entry is not None and entry not in order:
                order.append(entry)
    return order


def _kernel_declarations(program, kernel):
    """``(node, signature)`` for each name ``kernel``'s hoisted declarations put in the global namespace.

    A resource's signature is its declaration's structure; a cbuffer member has none, since a member of another
    cbuffer is always a second declaration of the name.
    """
    for declaration in getattr(program, "global_variables", []) or []:
        if _entry_of(declaration) == kernel:
            yield declaration, declaration_signature(declaration)
    for cbuffer in getattr(program, "cbuffers", []) or []:
        if _entry_of(cbuffer) == kernel:
            for member in getattr(cbuffer, "members", []) or []:
                yield member, None


def _taken_names(program):
    """Every name ``program`` declares or reads, so a new name collides with none of them."""
    taken = set()
    stack = [program]
    seen = set()
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(item)
            continue
        if isinstance(item, dict):
            stack.extend(item.values())
            continue
        if not isinstance(item, A.ASTNode) or id(item) in seen:
            continue
        seen.add(id(item))
        for field in ("name", "identifier"):
            value = getattr(item, field, None)
            if isinstance(value, str):
                taken.add(value)
        stack.extend(
            value for field, value in vars(item).items() if field not in _SKIPPED_FIELDS
        )
    return taken


def _rename_plan(program):
    """``{kernel: [(declaration, new_name)]}`` for the declarations a later kernel must rename."""
    owners = {}
    plan = {}
    taken = None
    for kernel in _kernel_order(program):
        for node, signature in _kernel_declarations(program, kernel):
            name = getattr(node, "name", None)
            if not isinstance(name, str) or not name:
                continue
            owner = owners.get(name)
            if owner is None:
                owners[name] = (kernel, signature)
                continue
            owner_kernel, owner_signature = owner
            if owner_kernel == kernel:
                continue
            if signature is not None and signature == owner_signature:
                continue
            if taken is None:
                taken = _taken_names(program)
            fresh = f"{kernel}_{name}"
            index = 2
            while fresh in taken:
                fresh = f"{kernel}_{name}_{index}"
                index += 1
            taken.add(fresh)
            plan.setdefault(kernel, []).append((node, fresh))
    return plan


def _children(node):
    """The fields of ``node`` that can hold nodes, each distinct object once (a call keeps its args twice)."""
    seen = set()
    call = isinstance(node, A.FunctionCallNode)
    for field, value in vars(node).items():
        if field in _SKIPPED_FIELDS or (call and field in _CALLEE_FIELDS):
            continue
        if id(value) in seen:
            continue
        seen.add(id(value))
        yield value


class _References:
    """Renames the references a function makes to module names, leaving names its locals shadow alone.

    CrossGL scopes a local from the end of its declarator, as GLSL does, so its initializer still reads the
    module name.
    """

    def statements(self, items, renames):
        renames = dict(renames)
        for item in items:
            if isinstance(item, A.VariableNode):
                self.declaration(item, renames)
            else:
                self.visit(item, renames)

    def declaration(self, node, renames):
        self.visit(getattr(node, "initial_value", None), renames)
        self.visit(getattr(node, "var_type", None), renames)
        renames.pop(getattr(node, "name", None), None)

    def visit(self, node, renames):
        if not renames:
            return
        if isinstance(node, list):
            for item in node:
                self.visit(item, renames)
            return
        if not isinstance(node, A.ASTNode):
            return
        if isinstance(node, A.IdentifierNode):
            new_name = renames.get(node.name)
            if new_name is not None:
                node.name = new_name
                node.identifier = new_name
            return
        if isinstance(node, A.BlockNode):
            self.statements(node.statements or [], renames)
            return
        if isinstance(node, A.VariableNode):
            # A declaration in a position of its own (a for-init, a condition) scopes to that statement.
            self.declaration(node, dict(renames))
            return
        if isinstance(node, A.ForNode):
            scope = dict(renames)
            if isinstance(node.init, A.VariableNode):
                self.declaration(node.init, scope)
            else:
                self.visit(node.init, scope)
            for field in ("condition", "update", "body"):
                self.visit(getattr(node, field, None), scope)
            return
        for child in _children(node):
            if isinstance(child, list) and any(
                isinstance(item, A.VariableNode) for item in child
            ):
                self.statements(child, renames)
            else:
                self.visit(child, renames)

    def function(self, function, renames):
        parameters = getattr(function, "parameters", None) or getattr(
            function, "params", None
        )
        scope = dict(renames)
        for parameter in parameters or []:
            scope.pop(getattr(parameter, "name", None), None)
        self.visit(getattr(function, "body", None), scope)


def _kernel_functions(program, kernel):
    for _stage_type, stage in getattr(program, "stages", {}).items():
        entry = getattr(stage, "entry_point", None)
        if getattr(entry, "name", None) == kernel:
            yield entry
    for function in getattr(program, "functions", []) or []:
        if getattr(function, "name", None) == kernel:
            yield function


def separate_entry_argument_names(program):
    """Return ``program``, or a copy in which no two kernels' hoisted arguments share a global name."""
    if not any(
        _entry_of(declaration) is not None
        for attribute in _DECLARATION_GROUPS
        for declaration in getattr(program, attribute, []) or []
    ):
        return program
    plan = _rename_plan(program)
    if not plan:
        return program
    memo = {}
    renamed = deepcopy(program, memo)
    references = _References()
    for kernel, declarations in plan.items():
        renames = {}
        for node, fresh in declarations:
            copied = memo[id(node)]
            renames[copied.name] = fresh
            copied.name = fresh
        for function in _kernel_functions(renamed, kernel):
            references.function(function, renames)
    return renamed
