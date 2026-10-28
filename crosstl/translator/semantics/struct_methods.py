"""Struct member functions, lowered to free functions for every target.

A CrossGL structure may declare member functions. An instance method reads and writes its receiver's members by
name, or through ``this``, and calls the structure's other methods unqualified; a ``static`` method has no receiver
and is called as ``Owner::name(...)``::

    struct Span {
        float lo;
        float hi;
        float width() { return hi - lo; }
        void widen(float d) { hi += d; }
    };
    ...
    float w = s.width();
    s.widen(3.0);

:func:`lower_struct_methods` gives every target each method as a free function named ``Owner_name`` (a fresh name
when that one is taken), declared ahead of the functions of the scope that declares the structure. An instance
method takes its receiver as a leading parameter: ``in`` when its body cannot write the receiver, ``inout`` when it
may, through an assignment or increment of a member, a member passed to an ``out`` or ``inout`` parameter or to a
writing built-in, or a call of a writing method. ``recv.name(args)`` becomes ``Owner_name(recv, args)``. Within one
invocation ``inout``'s copy-in/copy-out matches the reference receiver of HLSL, MSL and C++, except when the method
also reaches the receiver's storage through another name.

Refused with :class:`StructMethodUnsupportedError`: methods of generic structures, a writing method called on a
value that is not an lvalue, a static method called through a receiver with side effects, and a call of a member
function whose receiver type cannot be inferred.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import TypeScope, _struct_name
from .rewrite import child_slots, clone, get_child, is_pure, set_child
from .scoped_walk import ScopedWalker, program_stages

__all__ = ["StructMethodUnsupportedError", "lower_struct_methods"]

_RECEIVER = "this"
_WRITING_QUALIFIERS = frozenset({"out", "inout"})
_WRITING_BUILTINS = frozenset({"modf", "frexp", "sincos", "remquo"})
_WRITING_BUILTIN_PREFIXES = ("atomic", "Interlocked", "imageAtomic")


class StructMethodUnsupportedError(TranslationDiagnostic):
    """A structure member function, or a call of one, that no target can be given as a free function."""

    code = CODE_PREFIX + "struct-method-unsupported"

    def __init__(self, reason, *, target=None):
        super().__init__(reason, target=target)


@dataclass
class _Method:
    owner: str
    definition: A.FunctionNode
    static: bool
    free_name: str = ""
    receiver: str = ""
    writes: bool = False


@dataclass
class _Owner:
    struct: A.StructNode
    functions: list
    members: frozenset
    methods: dict = field(default_factory=dict)  # name -> [_Method, ...] (overloads)


def _walk(node):
    """Every structural node below and including ``node``, in source order."""
    if not isinstance(node, A.ASTNode):
        return
    yield node
    for slot in list(child_slots(node)):
        yield from _walk(get_child(node, *slot))


def _taken_names(program):
    names = set()
    for node in _walk(program):
        for attribute in ("name", "member"):
            value = getattr(node, attribute, None)
            if isinstance(value, str):
                names.add(value)
    return names


def _fresh(base, taken):
    candidate, index = base, 1
    while candidate in taken:
        index += 1
        candidate = f"{base}_{index}"
    taken.add(candidate)
    return candidate


def _is_static(function):
    return "static" in (getattr(function, "qualifiers", None) or [])


def _owners(program):
    """``_Owner`` for each structure that declares member functions, with the function list it belongs to."""
    if getattr(program, "functions", None) is None:
        program.functions = []
    scopes = [(getattr(program, "structs", None) or [], program.functions)]
    for stage in program_stages(program):
        if getattr(stage, "local_functions", None) is None:
            stage.local_functions = []
        scopes.append(
            (getattr(stage, "local_structs", None) or [], stage.local_functions)
        )
    owners = {}
    for structs, functions in scopes:
        for struct in structs:
            if not getattr(struct, "methods", None):
                continue
            members = frozenset(
                member.name
                for member in struct.members or []
                if isinstance(getattr(member, "name", None), str)
            )
            owners[struct.name] = _Owner(struct, functions, members)
    return owners


def _lvalue_root(expr):
    while isinstance(expr, (A.MemberAccessNode, A.ArrayAccessNode, A.SwizzleNode)):
        if isinstance(expr, A.MemberAccessNode):
            expr = expr.object_expr
        elif isinstance(expr, A.ArrayAccessNode):
            expr = expr.array_expr
        else:
            expr = expr.vector_expr
    return expr


def _is_lvalue(expr):
    return isinstance(_lvalue_root(expr), A.IdentifierNode)


class _MethodBody:
    """Walks one method body with its block-scoped declarations, resolving names that refer to the receiver."""

    def __init__(self, owner, method):
        self.owner = owner
        self.method = method

    def run(self, visit):
        self.shadowed = [{p.name for p in self.method.definition.parameters or []}]
        self.visit = visit
        self._walk(self.method.definition.body, None, None)

    def names_receiver(self, expr):
        """True when ``expr`` is rooted at the receiver: ``this`` or an unshadowed member name."""
        root = _lvalue_root(expr)
        if not isinstance(root, A.IdentifierNode):
            return False
        if root.name == _RECEIVER:
            return True
        return root.name in self.owner.members and not self.is_shadowed(root.name)

    def is_shadowed(self, name):
        return any(name in frame for frame in self.shadowed)

    def _walk(self, node, parent, slot):
        if not isinstance(node, A.ASTNode) or isinstance(
            node, (A.FunctionNode, A.StructNode)
        ):
            return
        scoped = isinstance(node, (A.BlockNode, A.ForNode))
        if scoped:
            self.shadowed.append(set())
        for child_slot in list(child_slots(node)):
            self._walk(get_child(node, *child_slot), node, child_slot)
        if isinstance(node, A.VariableNode):
            self.shadowed[-1].add(node.name)
        self.visit(node, parent, slot)
        if scoped:
            self.shadowed.pop()


class _Lowering:
    def __init__(self, program, target, owners):
        self.program = program
        self.target = target
        self.owners = owners
        self.user_functions = {}
        for function in _program_functions(program):
            self.user_functions.setdefault(function.name, []).append(function)

    # Analysis ---------------------------------------------------------------------------------------------
    def index(self):
        taken = _taken_names(self.program)
        for owner in self.owners.values():
            if getattr(owner.struct, "generic_params", None):
                raise StructMethodUnsupportedError(
                    f"generic structure '{owner.struct.name}' declares member functions; CrossGL lowers the "
                    "member functions of concrete structures only",
                    target=self.target,
                )
            free_names = {}
            for definition in owner.struct.methods:
                name = definition.name
                if name not in free_names:
                    free_names[name] = _fresh(f"{owner.struct.name}_{name}", taken)
                method = _Method(
                    owner.struct.name,
                    definition,
                    _is_static(definition),
                    free_names[name],
                )
                if not method.static:
                    method.receiver = _fresh("receiver", set(taken))
                owner.methods.setdefault(name, []).append(method)

    def methods_named(self, name):
        return [
            method
            for owner in self.owners.values()
            for method in owner.methods.get(name, [])
        ]

    def analyze_writes(self):
        """Mark the instance methods that may write their receiver, to a fixed point over method calls."""
        changed = True
        while changed:
            changed = False
            for owner in self.owners.values():
                for overloads in owner.methods.values():
                    for method in overloads:
                        if method.static or method.writes:
                            continue
                        if self._body_writes(owner, method):
                            method.writes = True
                            changed = True

    def _body_writes(self, owner, method):
        found = []
        body = _MethodBody(owner, method)

        def visit(node, parent, slot):
            if isinstance(node, A.AssignmentNode) and body.names_receiver(node.target):
                found.append(node)
            elif (
                isinstance(node, A.UnaryOpNode)
                and node.operator in ("++", "--")
                and body.names_receiver(node.operand)
            ):
                found.append(node)
            elif isinstance(node, A.FunctionCallNode) and self._call_writes(
                owner, body, node
            ):
                found.append(node)

        body.run(visit)
        return bool(found)

    def _call_writes(self, owner, body, call):
        function = call.function
        if isinstance(function, A.IdentifierNode):
            name = function.name
            sibling = owner.methods.get(name)
            if sibling and any(m.writes for m in sibling if not m.static):
                return True
            for index, argument in enumerate(call.arguments or []):
                if body.names_receiver(argument) and self._argument_written(
                    name, index
                ):
                    return True
            return False
        if isinstance(function, A.MemberAccessNode) and body.names_receiver(
            function.object_expr
        ):
            # A method of a member, or of ``this``: writing when any structure's method of that name writes.
            return any(m.writes for m in self.methods_named(function.member))
        return False

    def _argument_written(self, callee, index):
        functions = self.user_functions.get(callee)
        if functions:
            for function in functions:
                parameters = function.parameters or []
                if index < len(parameters) and _WRITING_QUALIFIERS & set(
                    parameters[index].qualifiers or []
                ):
                    return True
            return False
        return callee in _WRITING_BUILTINS or callee.startswith(
            _WRITING_BUILTIN_PREFIXES
        )

    # Rewriting --------------------------------------------------------------------------------------------
    def free_functions(self):
        """Declare each method as a free function ahead of its scope's functions, in declaration order."""
        by_scope = {}
        for owner in self.owners.values():
            for definition in owner.struct.methods:
                method = next(
                    m
                    for m in owner.methods[definition.name]
                    if m.definition is definition
                )
                by_scope.setdefault(id(owner.functions), (owner.functions, []))[
                    1
                ].append(self._free_function(owner, method))
            owner.struct.methods = []
        for functions, declared in by_scope.values():
            functions[:0] = declared

    def _free_function(self, owner, method):
        function = clone(method.definition)
        function.name = method.free_name
        function.qualifiers = [q for q in function.qualifiers or [] if q != "static"]
        rewritten = _Method(
            method.owner, function, method.static, method.free_name, method.receiver
        )
        body = _MethodBody(owner, rewritten)
        body.run(
            lambda node, parent, slot: self._rewrite_in_method(body, node, parent, slot)
        )
        if not method.static:
            receiver = A.ParameterNode(
                method.receiver,
                A.NamedType(owner.struct.name),
                qualifiers=["inout"] if method.writes else [],
            )
            function.parameters = [receiver, *(function.parameters or [])]
        return function

    def _rewrite_in_method(self, body, node, parent, slot):
        method = body.method
        if isinstance(node, A.IdentifierNode) and parent is not None:
            if isinstance(parent, A.FunctionCallNode) and slot == ("function", None):
                self._rewrite_unqualified_call(body, parent)
                return
            if node.name == _RECEIVER and not body.is_shadowed(_RECEIVER):
                if method.static:
                    raise StructMethodUnsupportedError(
                        f"static member function '{method.owner}::{method.definition.name}' uses 'this'",
                        target=self.target,
                    )
                set_child(parent, *slot, A.IdentifierNode(method.receiver))
            elif node.name in body.owner.members and not body.is_shadowed(node.name):
                if method.static:
                    raise StructMethodUnsupportedError(
                        f"static member function '{method.owner}::{method.definition.name}' reads member "
                        f"'{node.name}' without an object",
                        target=self.target,
                    )
                set_child(
                    parent,
                    *slot,
                    A.MemberAccessNode(A.IdentifierNode(method.receiver), node.name),
                )

    def _rewrite_unqualified_call(self, body, call):
        overloads = body.owner.methods.get(call.function.name)
        if not overloads:
            return
        callee = overloads[0]
        arguments = list(call.arguments or [])
        if not all(m.static for m in overloads):
            if body.method.static:
                raise StructMethodUnsupportedError(
                    f"static member function '{body.method.owner}::{body.method.definition.name}' calls "
                    f"member function '{call.function.name}' without an object",
                    target=self.target,
                )
            arguments.insert(0, A.IdentifierNode(body.method.receiver))
        _retarget(call, callee.free_name, arguments)

    def call_sites(self):
        _CallSites(self).run_program()


class _CallSites(ScopedWalker):
    """Rewrites ``recv.name(args)`` and ``Owner::name(args)`` calls of member functions in every function body."""

    def __init__(self, lowering):
        super().__init__(lowering.program)
        self.lowering = lowering
        self.qualified = {
            f"{owner_name}::{name}": overloads
            for owner_name, owner in lowering.owners.items()
            for name, overloads in owner.methods.items()
        }

    def visit(self, node):
        if not isinstance(node, A.FunctionCallNode):
            return
        function = node.function
        if isinstance(function, A.IdentifierNode):
            overloads = self.qualified.get(function.name)
            if overloads is None:
                return
            if not all(m.static for m in overloads):
                raise StructMethodUnsupportedError(
                    f"member function '{function.name}' is called without an object",
                    target=self.lowering.target,
                )
            _retarget(node, overloads[0].free_name, list(node.arguments or []))
        elif isinstance(function, A.MemberAccessNode):
            self._member_call(node, function)

    def _member_call(self, node, function):
        lowering = self.lowering
        candidates = lowering.methods_named(function.member)
        if not candidates:
            return
        receiver = function.object_expr
        receiver_type = _receiver_type(self.scope, receiver)
        owner = lowering.owners.get(_struct_name(receiver_type))
        if owner is None:
            if receiver_type is not None:
                return  # A method of a type without member functions, such as a texture's.
            raise StructMethodUnsupportedError(
                f"cannot infer the type of the object whose member function '{function.member}' is called",
                target=lowering.target,
            )
        overloads = owner.methods.get(function.member)
        if not overloads:
            return
        arguments = list(node.arguments or [])
        if all(m.static for m in overloads):
            if not is_pure(receiver):
                raise StructMethodUnsupportedError(
                    f"static member function '{owner.struct.name}::{function.member}' is called through an "
                    "object expression with side effects",
                    target=lowering.target,
                )
        else:
            if any(m.writes for m in overloads) and not _is_lvalue(receiver):
                raise StructMethodUnsupportedError(
                    f"member function '{owner.struct.name}::{function.member}' writes its object, which is "
                    "not an lvalue",
                    target=lowering.target,
                )
            arguments.insert(0, receiver)
        _retarget(node, overloads[0].free_name, arguments)


def _receiver_type(scope: TypeScope, expr):
    declared = scope.node_type(expr)
    if declared is not None:
        return declared
    if isinstance(expr, A.FunctionCallNode) and isinstance(
        expr.function, A.IdentifierNode
    ):
        name = expr.function.name
        if name in scope.structs:
            return A.NamedType(name)
        return scope.functions.get(name)
    if isinstance(expr, A.ConstructorNode):
        return expr.constructor_type
    if isinstance(expr, A.CastNode):
        return expr.target_type
    if isinstance(expr, A.TernaryOpNode):
        return _receiver_type(scope, expr.true_expr) or _receiver_type(
            scope, expr.false_expr
        )
    return None


def _retarget(call, name, arguments):
    callee = A.IdentifierNode(name)
    call.function = callee
    call.name = callee
    call.arguments = arguments
    call.args = arguments


def _program_functions(program):
    for function in getattr(program, "functions", None) or []:
        if isinstance(function, A.FunctionNode):
            yield function
    for stage in program_stages(program):
        for function in getattr(stage, "local_functions", None) or []:
            if isinstance(function, A.FunctionNode):
                yield function


def lower_struct_methods(program, target):
    """Lower every structure member function of ``program`` to a free function; rewrites the AST in place."""
    owners = _owners(program)
    if not owners:
        return
    lowering = _Lowering(program, target, owners)
    lowering.index()
    lowering.analyze_writes()
    lowering.free_functions()
    lowering.call_sites()
