"""Normalize storage views of single-scalar wrapper records to scalar views.

A record whose only member is one 32-bit scalar has that scalar's size and
alignment, and a pointer to such a record is pointer-interconvertible with a
pointer to its member (C++ [basic.compound]/4, which the Metal shading
language inherits). A reinterpreted view ``(device W*)p`` whose elements are
only ever reached as ``view[i].m``, ``view->m`` or ``(*view).m`` therefore
addresses exactly the words that ``(device S*)p`` addresses as ``view[i]``.

Rewriting those views to scalar views lets every target reuse its scalar
reinterpretation lowering. A view that escapes in any other way, such as an
element read as a whole record, pointer arithmetic or a reassignment, is left
unchanged so the target's existing record-view diagnostics still apply.
"""

import copy

from ..ast import (
    ArrayAccessNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    MemberAccessNode,
    NamedType,
    PointerAccessNode,
    PointerReinterpretNode,
    PointerType,
    PrimitiveType,
    StructNode,
    UnaryOpNode,
    VariableNode,
)
from .pointer_reinterpret import scalar_storage_layout

_SKIPPED_ATTRIBUTES = frozenset({"parent", "annotations", "source_location"})
_STORAGE_BUFFER_TYPES = frozenset(
    {
        "Buffer",
        "RWBuffer",
        "StructuredBuffer",
        "RWStructuredBuffer",
        "RasterizerOrderedStructuredBuffer",
    }
)


def scalar_wrapper_member(struct_node):
    """Return ``(member_name, member_type)`` for ``struct { S m; }``.

    ``S`` must be a 32-bit scalar. Records with attributes, generic
    parameters, bases, methods, member defaults or member attributes are not
    wrappers, because any of those can change layout or meaning.
    """

    if not isinstance(struct_node, StructNode):
        return None
    if (
        getattr(struct_node, "generic_params", None)
        or getattr(struct_node, "attributes", None)
        or getattr(struct_node, "inheritance", None)
        or getattr(struct_node, "methods", None)
    ):
        return None
    members = list(getattr(struct_node, "members", ()) or ())
    if len(members) != 1:
        return None
    member = members[0]
    if (
        getattr(member, "default_value", None) is not None
        or getattr(member, "attributes", None)
        or getattr(member, "resource_qualifiers", None)
    ):
        return None
    member_type = getattr(member, "member_type", None)
    if not isinstance(member_type, PrimitiveType):
        return None
    layout = scalar_storage_layout(getattr(member_type, "name", None))
    if layout is None or layout.bit_width != 32:
        return None
    name = getattr(member, "name", None)
    if not isinstance(name, str) or not name:
        return None
    return name, member_type


def scalar_wrapper_members(ast):
    """Map each wrapper record name visible in ``ast`` to its member."""

    structs = list(getattr(ast, "structs", ()) or ())
    stages = getattr(ast, "stages", None)
    stage_values = stages.values() if hasattr(stages, "values") else stages or ()
    for stage in stage_values:
        structs.extend(getattr(stage, "local_structs", ()) or ())
    wrappers = {}
    ambiguous = set()
    for struct in structs:
        name = getattr(struct, "name", None)
        if not isinstance(name, str):
            continue
        if name in wrappers or name in ambiguous:
            # Two records with one spelling cannot be resolved by name.
            wrappers.pop(name, None)
            ambiguous.add(name)
            continue
        member = scalar_wrapper_member(struct)
        if member is not None:
            wrappers[name] = member
    return wrappers


def _children(node):
    if isinstance(node, (list, tuple)):
        return list(node)
    if isinstance(node, dict):
        return [child for child in node.values() if child is not None]
    if not hasattr(node, "__dict__"):
        return []
    return [
        child
        for key, child in vars(node).items()
        if key not in _SKIPPED_ATTRIBUTES
        and not isinstance(child, (str, int, float, bool))
        and child is not None
    ]


def _walk(node, parent=None, seen=None):
    """Yield ``(node, parent)`` once per node in deterministic field order."""

    seen = set() if seen is None else seen
    stack = [(node, parent)]
    while stack:
        current, owner = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if not isinstance(current, (list, tuple, dict)):
            yield current, owner
            next_owner = current
        else:
            next_owner = owner
        stack.extend((child, next_owner) for child in reversed(_children(current)))


def _replace_child(parent, old, new):
    """Replace every reference to ``old`` held directly by ``parent``."""

    replaced = False
    for key, value in list(vars(parent).items()):
        if key in _SKIPPED_ATTRIBUTES:
            continue
        if value is old:
            setattr(parent, key, new)
            replaced = True
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if item is old:
                    value[index] = new
                    replaced = True
    return replaced


def _wrapper_pointee(pointer_type, wrappers):
    if not isinstance(pointer_type, PointerType):
        return None
    pointee = getattr(pointer_type, "pointee_type", None)
    if not isinstance(pointee, NamedType) or getattr(pointee, "generic_args", None):
        return None
    name = getattr(pointee, "name", None)
    return name if name in wrappers else None


def _scalar_pointer(pointer_type, member_type):
    scalar = copy.copy(pointer_type)
    scalar.pointee_type = copy.deepcopy(member_type)
    return scalar


def _member_use(node, parent, grandparent, member):
    """Return ``(replaced, replacement)`` when ``node`` is a member-only use."""

    if (
        isinstance(parent, ArrayAccessNode)
        and parent.array is node
        and isinstance(grandparent, MemberAccessNode)
        and grandparent.object is parent
        and grandparent.member == member
    ):
        return grandparent, parent
    if (
        isinstance(parent, PointerAccessNode)
        and parent.pointer_expr is node
        and parent.member == member
    ):
        return parent, UnaryOpNode("*", node)
    if (
        isinstance(parent, UnaryOpNode)
        and getattr(parent, "op", None) == "*"
        and parent.operand is node
        and isinstance(grandparent, MemberAccessNode)
        and grandparent.object is parent
        and grandparent.member == member
    ):
        return grandparent, parent
    return None


def _function_bodies(ast):
    seen = set()
    for node, _ in _walk(ast):
        if isinstance(node, FunctionNode) and id(node) not in seen:
            seen.add(id(node))
            yield node


def _call_name(call):
    function = getattr(call, "function", None)
    name = getattr(function, "name", function)
    return name if isinstance(name, str) else None


class _View:
    """A wrapper pointer declaration that may become a scalar pointer."""

    def __init__(self, function, declaration, wrapper):
        self.function = function
        self.declaration = declaration
        self.wrapper = wrapper
        self.uses = []
        self.forwards = []
        self.arguments = []
        self.valid = True


class _Program:
    def __init__(self, ast, wrappers):
        self.wrappers = wrappers
        self.functions = list(_function_bodies(ast))
        self.by_name = {}
        for function in self.functions:
            self.by_name.setdefault(function.name, []).append(function)
        stages = getattr(ast, "stages", None)
        stage_values = stages.values() if hasattr(stages, "values") else stages or ()
        self.entries = {
            id(getattr(stage, "entry_point", None)) for stage in stage_values
        }
        self.parents = {}
        self.nodes = {}
        self.declarations = {}
        for function in self.functions:
            nodes = []
            for node, parent in _walk(function):
                self.parents[id(node)] = parent
                nodes.append(node)
            self.nodes[id(function)] = nodes
            declared = {}
            for node in nodes:
                if isinstance(node, VariableNode):
                    declared.setdefault(node.name, []).append(node)
            for parameter in getattr(function, "parameters", ()) or ():
                declared.setdefault(getattr(parameter, "name", None), []).append(
                    parameter
                )
            self.declarations[id(function)] = declared
        self.views = {}
        self.parameter_views = {}

    def declaration(self, function, name):
        declared = self.declarations[id(function)].get(name) or []
        return declared[0] if len(declared) == 1 else None

    def unique_function(self, name):
        functions = self.by_name.get(name) or []
        return functions[0] if len(functions) == 1 else None

    def collect_views(self):
        for function in self.functions:
            declared = self.declarations[id(function)]
            for name, items in declared.items():
                if len(items) != 1:
                    continue
                item = items[0]
                if isinstance(item, VariableNode):
                    initial = getattr(item, "initial_value", None)
                    wrapper = _wrapper_pointee(
                        getattr(item, "var_type", None), self.wrappers
                    )
                    if (
                        wrapper is None
                        or not isinstance(initial, PointerReinterpretNode)
                        or _wrapper_pointee(initial.target_type, self.wrappers)
                        != wrapper
                    ):
                        continue
                elif (
                    id(function) not in self.entries
                    and not getattr(function, "generic_params", None)
                    and self.unique_function(function.name) is function
                ):
                    wrapper = _wrapper_pointee(
                        getattr(item, "param_type", None), self.wrappers
                    )
                    if wrapper is None:
                        continue
                else:
                    continue
                view = _View(function, item, wrapper)
                self.views[(id(function), name)] = view
                if not isinstance(item, VariableNode):
                    index = function.parameters.index(item)
                    self.parameter_views[(function.name, index)] = view

    def classify_uses(self):
        for function in self.functions:
            for node in self.nodes[id(function)]:
                if not isinstance(node, IdentifierNode):
                    continue
                view = self.views.get((id(function), node.name))
                if view is None or not view.valid:
                    continue
                parent = self.parents.get(id(node))
                grandparent = (
                    self.parents.get(id(parent)) if parent is not None else None
                )
                member = self.wrappers[view.wrapper][0]
                use = _member_use(node, parent, grandparent, member)
                if use is not None:
                    view.uses.append(use)
                    continue
                callee_view = self.forwarded_parameter(node, parent, view)
                if callee_view is not None:
                    view.forwards.append(callee_view)
                    continue
                view.valid = False

    def forwarded_parameter(self, node, parent, view):
        if not isinstance(parent, FunctionCallNode):
            return None
        arguments = list(getattr(parent, "arguments", ()) or ())
        positions = [index for index, item in enumerate(arguments) if item is node]
        name = _call_name(parent)
        if len(positions) != 1 or name is None:
            return None
        callee_view = self.parameter_views.get((name, positions[0]))
        if callee_view is None or callee_view.wrapper != view.wrapper:
            return None
        return callee_view

    def classify_call_sites(self):
        """Every argument of a rewritten parameter must be a wrapper pointer."""

        for function in self.functions:
            for node in self.nodes[id(function)]:
                if isinstance(node, IdentifierNode) and node.name in self.by_name:
                    parent = self.parents.get(id(node))
                    if not (
                        isinstance(parent, FunctionCallNode)
                        and getattr(parent, "function", None) is node
                    ):
                        # The function escapes as a value; its callers are unknown.
                        for (name, _), view in self.parameter_views.items():
                            if name == node.name:
                                view.valid = False
                if not isinstance(node, FunctionCallNode):
                    continue
                name = _call_name(node)
                arguments = list(getattr(node, "arguments", ()) or ())
                for (callee, index), view in self.parameter_views.items():
                    if callee != name:
                        continue
                    argument = arguments[index] if index < len(arguments) else None
                    if not self.wrapper_pointer_argument(function, argument, view):
                        view.valid = False
                        continue
                    view.arguments.append(self.views.get((id(function), argument.name)))

    def wrapper_pointer_argument(self, function, argument, view):
        if not isinstance(argument, IdentifierNode):
            return False
        declaration = self.declaration(function, argument.name)
        declared_type = getattr(
            declaration, "var_type", getattr(declaration, "param_type", None)
        )
        if _wrapper_pointee(declared_type, self.wrappers) == view.wrapper:
            return True
        # A storage buffer of wrapper records, such as an entry argument.
        arguments = list(getattr(declared_type, "generic_args", ()) or ())
        return (
            isinstance(declared_type, NamedType)
            and declared_type.name in _STORAGE_BUFFER_TYPES
            and len(arguments) == 1
            and isinstance(arguments[0], NamedType)
            and arguments[0].name == view.wrapper
            and not arguments[0].generic_args
        )

    def solve(self):
        changed = True
        while changed:
            changed = False
            for view in self.views.values():
                if not view.valid:
                    continue
                forwards_to_record = any(not callee.valid for callee in view.forwards)
                # Rewrite a parameter only for a caller that passes a view;
                # callers that pass records already have the declared type.
                parameter_without_views = not isinstance(
                    view.declaration, VariableNode
                ) and not any(
                    argument is not None and argument.valid
                    for argument in view.arguments
                )
                if forwards_to_record or parameter_without_views:
                    view.valid = False
                    changed = True

    def rewrite(self):
        changed = False
        # Views were collected in deterministic AST order.
        for view in self.views.values():
            if not view.valid:
                continue
            member_type = self.wrappers[view.wrapper][1]
            declaration = view.declaration
            if isinstance(declaration, VariableNode):
                scalar = _scalar_pointer(declaration.var_type, member_type)
                declaration.var_type = scalar
                declaration.vtype = scalar
                _retarget_cast(declaration.initial_value, member_type)
            else:
                self.wrap_call_site_arguments(view, member_type)
                scalar = _scalar_pointer(declaration.param_type, member_type)
                declaration.param_type = scalar
                if isinstance(getattr(declaration, "vtype", None), PointerType):
                    declaration.vtype = scalar
            for replaced, replacement in view.uses:
                _replace_child(self.parents[id(replaced)], replaced, replacement)
            changed = True
        return changed

    def wrap_call_site_arguments(self, view, member_type):
        index = view.function.parameters.index(view.declaration)
        target_type = _scalar_pointer(view.declaration.param_type, member_type)
        for function in self.functions:
            for node in self.nodes[id(function)]:
                if (
                    not isinstance(node, FunctionCallNode)
                    or _call_name(node) != view.function.name
                ):
                    continue
                argument = node.arguments[index]
                source = self.views.get((id(function), argument.name))
                if source is not None and source.valid:
                    continue
                # A wrapper pointer is pointer-interconvertible with its member.
                cast = PointerReinterpretNode(argument, copy.copy(target_type))
                _replace_child(node, argument, cast)
                self.parents[id(cast)] = node


def _retarget_cast(cast, member_type):
    scalar = _scalar_pointer(cast.target_type, member_type)
    cast.target_type = scalar
    for key in ("expression_type", "vtype"):
        if isinstance(getattr(cast, key, None), PointerType):
            setattr(cast, key, scalar)


def normalize_scalar_wrapper_views(ast):
    """Rewrite member-only views of single-scalar wrappers in place.

    Local views initialized by a reinterpretation and helper parameters may
    be rewritten. A view may be forwarded to a helper parameter that is also
    rewritten; a helper parameter whose caller passes another wrapper pointer
    receives that pointer through an explicit scalar reinterpretation.
    Returns whether any view was rewritten.
    """

    wrappers = scalar_wrapper_members(ast)
    if not wrappers:
        return False
    program = _Program(ast, wrappers)
    program.collect_views()
    if not program.views:
        return False
    program.classify_uses()
    program.classify_call_sites()
    program.solve()
    return program.rewrite()
