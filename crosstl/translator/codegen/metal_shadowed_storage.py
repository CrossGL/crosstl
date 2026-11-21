"""Rename function-local declarations that share a name with program-scope storage, for Metal.

Metal has no program-scope buffers, textures, samplers, threadgroup memory or uniform values. Each entry function
receives them as parameters, and each helper function that reaches one, directly or through a callee, receives it
as a parameter appended under the variable's own name. A parameter or local of a function with the same name then
either collides with that appended parameter or hides it from code that reads the program-scope variable. CUDA and
HIP kernels are often written this way: ``__device__ int twice(int o)`` next to ``__global__ void k(int* o)``.

:func:`rename_metal_shadowing_declarations` gives every such declaration a fresh name and renames the references
in its scope, following the C and GLSL scope rules: a parameter is visible in the whole body, and a local from just
after its declarator (its own initializer still reads the outer name) to the end of its block. A declaration of a
resource type keeps its name; it hides the outer variable in its scope as before. The program is copied before
anything is renamed.
"""

from __future__ import annotations

from copy import deepcopy

from .. import ast as A
from ..semantics.rewrite import child_slots, get_child
from ..semantics.scoped_walk import parameter_type, program_stages

__all__ = ["rename_metal_shadowing_declarations", "metal_program_storage_names"]

# Stage interface values are not passed to helpers by name, and constants stay at program scope.
_INTERFACE_QUALIFIERS = frozenset(
    {"in", "out", "inout", "attribute", "varying", "const", "constant"}
)

# Statements that open a scope. A ``switch`` body is one scope; its ``case`` labels open none.
_SCOPE_NODES = (
    A.BlockNode,
    A.ForNode,
    A.IfNode,
    A.WhileNode,
    A.DoWhileNode,
    A.SwitchNode,
)


def _qualifier_names(node):
    return {
        str(qualifier).lower() for qualifier in getattr(node, "qualifiers", []) or []
    }


def _storage_variable_names(variables):
    for node in variables or []:
        name = getattr(node, "name", None)
        if isinstance(name, str) and not (
            _qualifier_names(node) & _INTERFACE_QUALIFIERS
        ):
            yield name


def _cbuffer_member_names(cbuffers):
    for cbuffer in cbuffers or []:
        for member in getattr(cbuffer, "members", []) or []:
            name = getattr(member, "name", None)
            if isinstance(name, str):
                yield name


def metal_program_storage_names(ast):
    """Names of the program- and stage-scope storage Metal passes to functions by name."""
    names = set(_storage_variable_names(getattr(ast, "global_variables", None)))
    names.update(_cbuffer_member_names(getattr(ast, "cbuffers", None)))
    for stage in program_stages(ast):
        names.update(_storage_variable_names(getattr(stage, "local_variables", None)))
        names.update(_cbuffer_member_names(getattr(stage, "local_cbuffers", None)))
    return names


def _program_functions(ast):
    """Program functions, stage helpers and stage entries, each once, in a fixed order."""
    candidates = list(getattr(ast, "functions", None) or [])
    for stage in program_stages(ast):
        candidates.extend(getattr(stage, "local_functions", None) or [])
        candidates.append(getattr(stage, "entry_point", None))
    functions, seen = [], set()
    for function in candidates:
        if isinstance(function, A.FunctionNode) and id(function) not in seen:
            seen.add(id(function))
            functions.append(function)
    return functions


def _callee_name(node):
    function = getattr(node, "function", None)
    return function if isinstance(function, str) else getattr(function, "name", None)


class _ScopedWalk:
    """Walks one function body in source order, with the names each scope declares.

    ``targets`` are the names to rename in this function; every other name in ``names`` is only tracked, so that a
    reference that no declaration of the function covers counts as a read of the program-scope variable.
    """

    def __init__(self, names, is_resource_type, used=None):
        self.names = names
        self.is_resource_type = is_resource_type
        self.used = used
        self.renames = {}
        self.targets = frozenset()
        self.scopes = []
        self.outer_reads = set()
        self.callees = set()
        self.declared = set()

    def fresh(self, name):
        # Every renamed declaration of one name takes the same new name: they keep the nesting the original
        # declarations had, so each still hides the same outer ones.
        renamed = self.renames.get(name)
        if renamed is None:
            renamed = f"{name}_"
            while renamed in self.used:
                renamed = f"{renamed}_"
            self.used.add(renamed)
            self.renames[name] = renamed
        return renamed

    def lookup(self, name):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def declare(self, node, declared_type):
        name = getattr(node, "name", None)
        if name not in self.names:
            return
        renamed = name
        if not self.is_resource_type(declared_type):
            self.declared.add(name)
            if name in self.targets:
                renamed = self.fresh(name)
                node.name = renamed
        # A declaration that keeps its name hides the program-scope variable.
        self.scopes[-1][name] = renamed

    def run(self, function, targets=frozenset()):
        self.targets = targets
        self.outer_reads, self.callees, self.declared = set(), set(), set()
        self.scopes = [{}]
        for parameter in function.parameters or []:
            self.declare(parameter, parameter_type(parameter))
        self.walk(getattr(function, "body", None))
        self.scopes = []

    def walk(self, node):
        if node is None or isinstance(node, (A.FunctionNode, A.StructNode)):
            return
        scoped = isinstance(node, _SCOPE_NODES)
        if scoped:
            self.scopes.append({})
        if isinstance(node, A.FunctionCallNode):
            callee = _callee_name(node)
            if callee:
                self.callees.add(callee)
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            if isinstance(child, A.IdentifierNode):
                # A callee names a function or type, never a variable.
                if not (isinstance(node, A.FunctionCallNode) and field == "function"):
                    self.reference(child)
                continue
            self.walk(child)
        if isinstance(node, A.VariableNode):
            self.declare(node, node.var_type)
        if scoped:
            self.scopes.pop()

    def reference(self, node):
        name = node.name
        if name not in self.names:
            return
        renamed = self.lookup(name)
        if renamed is None:
            self.outer_reads.add(name)
        elif renamed != name:
            node.name = renamed
            node.identifier = renamed


def _entry_functions(ast):
    return [
        stage.entry_point
        for stage in program_stages(ast)
        if isinstance(getattr(stage, "entry_point", None), A.FunctionNode)
    ]


def _always_renamed_names(ast, codegen):
    """Program-scope resources and every threadgroup variable.

    Metal refused a function that declared one of these names whether or not it read the variable, so renaming
    the declaration in every function changes no output Metal generated before.
    """
    names = set(codegen.collect_global_resource_names(ast))
    variables = list(getattr(ast, "global_variables", None) or [])
    for stage in program_stages(ast):
        variables.extend(getattr(stage, "local_variables", None) or [])
    for node in variables:
        name = getattr(node, "name", None)
        if isinstance(name, str) and codegen.is_program_scope_groupshared_global(node):
            names.add(name)
    return names


def _rename_targets(ast, functions, names, is_resource_type, always_renamed):
    """The names to rename in each function, keyed by ``id(function)``.

    A function renames its declarations of a name it reaches (it reads the program-scope variable, or calls a
    function that does), since Metal appends that variable to its parameters under the same name. An entry
    function renames every declaration of a storage name: Metal declares the storage among its parameters. Every
    function renames the names of program-scope resources, which Metal refused to generate before.
    """
    walk = _ScopedWalk(names, is_resource_type)
    reads, calls, declared = {}, {}, {}
    for function in functions:
        walk.run(function)
        reads[id(function)] = set(walk.outer_reads)
        calls[id(function)] = set(walk.callees)
        declared[id(function)] = set(walk.declared)
    by_name = {}
    for function in functions:
        by_name.setdefault(function.name, []).append(function)
    reached = {key: set(value) for key, value in reads.items()}
    changed = True
    while changed:
        changed = False
        for function in functions:
            before = len(reached[id(function)])
            for callee in sorted(calls[id(function)]):
                for target in by_name.get(callee, []):
                    reached[id(function)] |= reached[id(target)]
            changed = changed or len(reached[id(function)]) != before
    entries = {id(function) for function in _entry_functions(ast)}
    targets = {}
    for function in functions:
        wanted = reached[id(function)] | always_renamed
        if id(function) in entries:
            wanted = names
        found = declared[id(function)] & wanted
        if found:
            targets[id(function)] = frozenset(found)
    return targets


def rename_metal_shadowing_declarations(ast, codegen):
    """Return ``ast``, or a renamed copy when a function declares a name of program-scope storage it reaches."""
    names = metal_program_storage_names(ast)
    if not names:
        return ast
    is_resource_type = codegen.is_resource_parameter_type
    targets = _rename_targets(
        ast,
        _program_functions(ast),
        names,
        is_resource_type,
        _always_renamed_names(ast, codegen),
    )
    if not targets:
        return ast
    planned = [targets.get(id(function)) for function in _program_functions(ast)]
    ast = deepcopy(ast)
    walk = _ScopedWalk(
        names, is_resource_type, codegen.collect_metal_program_identifier_names(ast)
    )
    # The copy lists its functions in the order of the original.
    for function, function_targets in zip(_program_functions(ast), planned):
        if function_targets:
            walk.run(function, function_targets)
    return ast
