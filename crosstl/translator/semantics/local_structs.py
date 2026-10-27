"""Structures declared in a block become module-scope structures.

GLSL, HLSL, MSL, CUDA, HIP, Slang and OpenCL C accept a structure declared in a function body, visible from its
declaration to the end of its block; WGSL and Mojo declare structures only at module scope. The CrossGL parser
reads such a declaration as a statement (a :class:`~crosstl.translator.ast.StructNode` in the block). A type has
no run-time state, so moving its declaration out of the function changes nothing the program computes:
:func:`hoist_local_structs` moves each one to the structures of its stage, or of the program for a module-scope
function, under a fresh name ``<function>_<Struct>``, and renames the references to it from the declaration to the
end of its block (variable and member types, constructor calls, casts and patterns). A nested block that declares
the same name again hides it there, as a variable of that name does. Every generator then reads an ordinary
structure.

A structure whose members name something declared in the function (a parameter, a local constant used as an
array size, a local type alias, or a generic parameter) cannot leave the function, because those names are not in
scope at module scope; it is refused with :class:`LocalStructUnsupportedError`.
"""

import re

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .rewrite import child_slots, get_child

__all__ = ["LocalStructUnsupportedError", "hoist_local_structs"]

_SKIPPED_FIELDS = frozenset({"parent", "annotations", "source_location"})

# String-valued fields that hold a type's name rather than a declared name.
_TYPE_NAME_FIELDS = frozenset(
    {
        "var_type",
        "param_type",
        "return_type",
        "constructor_type",
        "type_name",
        "target_type",
        "element_type",
        "vtype",
        "member_type",
    }
)


class LocalStructUnsupportedError(TranslationDiagnostic):
    """A structure declared in a block that cannot move to module scope."""

    def __init__(self, message, target):
        super().__init__(
            message,
            target=target,
            code=f"{CODE_PREFIX}local-struct-unsupported",
        )


def hoist_local_structs(program, target) -> int:
    """Move structures declared in function bodies to module scope; returns how many moved."""
    if not isinstance(program, A.ShaderNode):
        return 0
    target = str(target or "")
    taken = None
    moved = 0
    for function, destination in _functions_with_destinations(program):
        body = getattr(function, "body", None)
        if body is None:
            continue
        found = _local_structs(body)
        if not found:
            continue
        if taken is None:
            taken = _declared_names(program)
        function_name = str(getattr(function, "name", "") or "function")
        local_names = _function_scope_names(function)
        for container, struct in found:
            source_name = struct.name
            _check_movable(struct, local_names, function_name, target)
            new_name = _fresh_name(f"{function_name}_{source_name}", taken)
            _TypeRenamer(struct, source_name, new_name).walk(body)
            container.remove(struct)
            struct.name = new_name
            struct.local_declaration_name = source_name
            destination.append(struct)
            moved += 1
    return moved


def _functions_with_destinations(program):
    """``(function, structure list)`` pairs: a stage's functions hoist into that stage's structures."""
    program.structs = list(getattr(program, "structs", None) or [])
    for function in getattr(program, "functions", None) or []:
        yield function, program.structs
    stages = getattr(program, "stages", None) or {}
    for stage in list(stages.values()) if isinstance(stages, dict) else list(stages):
        functions = list(getattr(stage, "local_functions", None) or [])
        if getattr(stage, "entry_point", None) is not None:
            functions.append(stage.entry_point)
        if not functions:
            continue
        stage.local_structs = list(getattr(stage, "local_structs", None) or [])
        for function in functions:
            yield function, stage.local_structs


def _local_structs(body):
    """``(statement list, structure)`` for each structure declared in ``body``, in source order."""
    found = []

    def visit(node):
        for field, key in child_slots(node):
            child = get_child(node, field, key)
            if isinstance(child, A.StructNode):
                if isinstance(key, int):
                    found.append((getattr(node, field), child))
                    # A structure declared inside a nested one's members is not a statement.
                continue
            visit(child)

    if isinstance(body, list):
        holder = A.BlockNode(body)
        visit(holder)
        return [
            (body if container is holder.statements else container, node)
            for container, node in found
        ]
    visit(body)
    return found


def _function_scope_names(function):
    """Names declared by ``function`` (parameters, generic parameters and locals) that a structure may not use."""
    names = set()
    for parameter in getattr(function, "parameters", None) or []:
        name = getattr(parameter, "name", None)
        if isinstance(name, str):
            names.add(name)
    for generic in getattr(function, "generic_params", None) or []:
        name = getattr(generic, "name", generic)
        if isinstance(name, str):
            names.add(name)

    def visit(node):
        if isinstance(node, A.StructNode):
            return
        if isinstance(node, A.VariableNode) and isinstance(node.name, str):
            names.add(node.name)
        for field, key in child_slots(node):
            visit(get_child(node, field, key))

    body = getattr(function, "body", None)
    for item in body if isinstance(body, list) else [body]:
        if isinstance(item, A.ASTNode):
            visit(item)
    return names


def _check_movable(struct, local_names, function_name, target):
    own = {
        getattr(generic, "name", generic)
        for generic in getattr(struct, "generic_params", None) or []
    }
    blocked = local_names - own
    for name in sorted(_referenced_names(struct) & blocked):
        raise LocalStructUnsupportedError(
            f"The structure '{struct.name}' declared in '{function_name}' uses '{name}', which "
            f"'{function_name}' declares, so it cannot be declared at module scope",
            target,
        )


def _referenced_names(node):
    names = set()

    def visit(current):
        if isinstance(current, (A.IdentifierNode, A.NamedType)):
            name = getattr(current, "name", None)
            if isinstance(name, str):
                names.add(name)
        for field, value in vars(current).items():
            if field in _TYPE_NAME_FIELDS and isinstance(value, str):
                names.add(value)
        for _field, children in _fields(current):
            for child in children if isinstance(children, list) else [children]:
                visit(child)

    visit(node)
    return names


def _fields(node):
    """``(field, child or list of children)`` for each child of ``node``, type nodes included, each once.

    :func:`~.rewrite.child_slots` leaves out the fields that hold types, which are what refers to a structure.
    """
    seen = set()
    for field, value in vars(node).items():
        if field in _SKIPPED_FIELDS or id(value) in seen:
            continue
        if isinstance(value, A.ASTNode):
            seen.add(id(value))
            yield field, value
        elif isinstance(value, (list, tuple)):
            seen.add(id(value))
            items = [item for item in value if isinstance(item, A.ASTNode)]
            if items:
                yield field, items
        elif isinstance(value, dict) and not isinstance(value, A.StageMap):
            seen.add(id(value))
            for item in value.values():
                if isinstance(item, A.ASTNode):
                    yield field, item


class _TypeRenamer:
    """Renames the references to one local structure, from its declaration to the end of its block.

    A later structure or variable of the same name in a nested block, or in a ``for`` header, hides it there.
    """

    def __init__(self, declaration, old, new):
        self.declaration = declaration
        self.old = old
        self.new = new

    def walk(self, body):
        if isinstance(body, list):
            self._sequence(body, active=False)
        else:
            self._node(body, active=False)

    def _sequence(self, items, active):
        """Visit siblings in order; returns whether the name refers to the declaration after them."""
        for item in items:
            if item is self.declaration:
                # Its own members may refer to it (``struct Node { Node* next; }``).
                self._node(item, True)
                active = True
                continue
            if not isinstance(item, A.ASTNode):
                continue
            if self._hides(item):
                if isinstance(item, A.VariableNode):
                    self._node(item, active)
                active = False
                continue
            self._node(item, active)
        return active

    def _hides(self, node):
        return (
            isinstance(node, (A.StructNode, A.VariableNode))
            and getattr(node, "name", None) == self.old
        )

    def _rename_fields(self, node):
        if isinstance(node, (A.IdentifierNode, A.NamedType)) and node.name == self.old:
            node.name = self.new
            if isinstance(node, A.IdentifierNode):
                node.identifier = self.new
        if isinstance(node, A.FunctionCallNode):
            if node.function == self.old:
                node.function = self.new
            if node.name == self.old:
                node.name = self.new
        for field in _TYPE_NAME_FIELDS:
            if getattr(node, field, None) == self.old and isinstance(
                getattr(node, field), str
            ):
                setattr(node, field, self.new)

    def _node(self, node, active):
        if active:
            self._rename_fields(node)
        if isinstance(node, A.StructNode) and node is not self.declaration:
            if node.name == self.old:
                return
        for field, children in _fields(node):
            if isinstance(children, list):
                # A statement list or a for header's declarations: later items see earlier declarations.
                active_after = self._sequence(children, active)
                if isinstance(node, A.ForNode) and field == "init":
                    active = active_after
            else:
                self._node(children, active)
                if isinstance(node, A.ForNode) and field == "init":
                    if self._hides(children):
                        active = False


def _declared_names(program):
    names = set()

    def visit(node):
        for field in ("name", "identifier"):
            value = getattr(node, field, None)
            if isinstance(value, str):
                names.add(value)
        for field, key in child_slots(node):
            visit(get_child(node, field, key))

    visit(program)
    return names


def _fresh_name(base, taken):
    base = re.sub(r"\W", "_", base)
    name = base
    suffix = 1
    while name in taken:
        name = f"{base}_{suffix}"
        suffix += 1
    taken.add(name)
    return name
