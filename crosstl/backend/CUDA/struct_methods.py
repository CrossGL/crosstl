"""Struct member functions of CUDA and HIP programs as CrossGL free functions.

CrossGL structs have data members only. The CUDA parser keeps each member function defined in a struct body that it
can lower (see :mod:`crosstl.backend.CUDA.parsing.struct_methods`); :func:`lower_struct_methods` turns each one into
a free function named ``<Struct>_<method>`` whose first parameter is the receiver, as the HLSL importer does for
HLSL struct methods:

* the receiver is a ``const S &`` parameter for a ``const`` member function and an ``S &`` parameter otherwise, so
  :class:`~crosstl.backend.CUDA.reference_parameters.ReferenceParameters` imports it as ``inout`` exactly when the
  function writes a data member, and as a value otherwise;
* inside the body, a data member named alone or through ``this->``, ``*this``, and a call of another kept member
  function of the same struct name the receiver ([class.mfct.non.static], [expr.prim.this]);
* a member call ``x.f(...)`` or ``p->f(...)`` whose receiver has a struct type with a kept ``f`` becomes
  ``S_f(x, ...)`` or ``S_f(p[0], ...)``, choosing among overloads by argument count.

Only the member functions that some call of the program reaches are lowered; the others are left out, as the
importer did before, so a body the pass cannot carry is refused only when the program uses it.

A member call the pass cannot resolve to a kept function is refused when it names a member function of any struct
of the program, rather than written through as a member call that no target declares: so are a call of a skipped
member function (a constructor, an operator, a template, a static or virtual member, or one defined outside the
class), a receiver whose type the pass cannot determine, a local or parameter that hides a member, ``this`` used as
a pointer value, a member function that returns a reference, and a body construct the pass does not know.
"""

from __future__ import annotations

import re

from ..common_ast import (
    ArrayAccessNode,
    AssignmentNode,
    ASTNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    CaseNode,
    CastNode,
    ContinueNode,
    DesignatedInitializerNode,
    DoWhileNode,
    EnumNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    InitializerListNode,
    MemberAccessNode,
    PostfixOpNode,
    RangeForNode,
    ReturnNode,
    StaticAssertNode,
    StructNode,
    SwitchNode,
    SyncNode,
    TernaryOpNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
    VectorConstructorNode,
)
from .CudaAst import AtomicOperationNode, CudaBuiltinNode, SharedMemoryNode
from .keyword_renames import source_identifier_names

__all__ = ["lower_struct_methods"]

RECEIVER_NAME = "this_"

# The attributes of each node kind that hold expressions or statements. A member function body may only contain
# these kinds; any other is refused, because the pass cannot tell where it names data members.
_EXPRESSION_FIELDS = {
    ArrayAccessNode: ("array", "index"),
    AssignmentNode: ("left", "right"),
    AtomicOperationNode: ("args",),
    BinaryOpNode: ("left", "right"),
    BlockNode: ("statements",),
    BreakNode: (),
    CaseNode: ("body", "statements"),
    CastNode: ("expression",),
    ContinueNode: (),
    CudaBuiltinNode: (),
    DesignatedInitializerNode: ("value",),
    DoWhileNode: ("body", "condition"),
    ForNode: ("init", "condition", "update", "body"),
    FunctionCallNode: ("args",),
    IfNode: ("condition", "if_body", "else_body", "if_chain", "else_if_chain"),
    InitializerListNode: ("elements",),
    MemberAccessNode: ("object",),
    PostfixOpNode: ("operand",),
    RangeForNode: ("iterable", "body"),
    ReturnNode: ("value",),
    StaticAssertNode: ("condition",),
    SwitchNode: ("expression", "cases", "default_case", "default"),
    SyncNode: ("args",),
    TernaryOpNode: ("condition", "true_expr", "false_expr"),
    UnaryOpNode: ("operand",),
    VariableNode: ("value",),
    VectorConstructorNode: ("args",),
}

_TYPE_WORDS = frozenset(
    {"const", "volatile", "struct", "class", "union", "__restrict__", "__restrict"}
)
_ARRAY_SUFFIX = re.compile(r"\[[^\[\]]*\]\s*$")


def lower_struct_methods(program, refuse):
    """Lower the kept member functions of ``program`` in place; ``refuse(reason)`` records each refusal."""
    if not isinstance(program, ASTNode):
        return
    method_names = set(getattr(program, "struct_method_names", None) or [])
    if not method_names:
        return
    _StructMethodLowering(program, method_names, refuse).run()


def _walk(node):
    """Every AST node under ``node``, ``node`` included."""
    pending, seen = [node], set()
    while pending:
        current = pending.pop()
        if isinstance(current, (list, tuple)):
            pending.extend(reversed(current))
            continue
        if not isinstance(current, ASTNode) or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend(
            value
            for value in reversed(list(vars(current).values()))
            if isinstance(value, (ASTNode, list, tuple))
        )


def _replace(value, rewrite):
    """``value`` with ``rewrite`` applied to each node or string of a list or tuple, or to ``value`` itself."""
    if isinstance(value, list):
        return [_replace(item, rewrite) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace(item, rewrite) for item in value)
    return rewrite(value)


def _is_auto(type_text):
    words = str(type_text or "").replace("&", " ").replace("*", " ").split()
    return "auto" in words or any(word.startswith("decltype") for word in words)


def _split_type(type_text, aliases):
    """``(record, indirection)`` of a declared type: its base name and how many ``*`` or ``[N]`` it has."""
    text = str(type_text or "").replace("&", " ")
    indirection = 0
    while True:
        stripped = _ARRAY_SUFFIX.sub("", text)
        if stripped == text:
            break
        text, indirection = stripped, indirection + 1
    indirection += text.count("*")
    words = [
        word
        for word in text.replace("*", " ").split()
        if word not in _TYPE_WORDS and not word.startswith("__align__")
    ]
    if len(words) != 1:
        return None, indirection
    name = words[0]
    for _ in range(8):
        target = aliases.get(name)
        if target is None:
            break
        record, extra = _split_type(target, {})
        if record is None:
            return None, indirection
        name, indirection = record, indirection + extra
    return name, indirection


class _Method:
    """A kept member function and the free function it becomes."""

    def __init__(self, owner, function, free_name):
        self.owner = owner
        self.function = function
        self.free_name = free_name
        params = list(getattr(function, "params", None) or [])
        self.max_args = len(params)
        self.min_args = sum(
            1 for param in params if getattr(param, "value", None) is None
        )

    def accepts(self, count):
        return self.min_args <= count <= self.max_args


class _MemberContext:
    """Rewrites one member function body so that it names its receiver explicitly."""

    def __init__(self, lowering, method):
        self.lowering = lowering
        self.method = method
        self.owner = method.owner
        self.label = f"member function '{method.owner}::{method.function.name}'"
        fields = lowering.fields.get(self.owner, {})
        own_methods = lowering.methods.get(self.owner, {})
        params = {
            str(param.name)
            for param in getattr(method.function, "params", None) or []
            if isinstance(param, VariableNode)
        }
        locals_ = {
            str(node.name)
            for node in _walk(getattr(method.function, "body", None))
            if isinstance(node, (VariableNode, SharedMemoryNode, RangeForNode))
        }
        # A parameter hides a data member of the same name throughout the body ([basic.scope.param]); the member
        # stays reachable through ``this->``.
        self.fields = set(fields) - params
        self.static_fields = lowering.static_fields.get(self.owner, set()) - params
        hidden = sorted(
            (locals_ & (set(fields) | set(own_methods))) | (params & set(own_methods))
        )
        if hidden:
            self.refuse(
                f"declares a local or parameter that hides member '{hidden[0]}'; CrossGL names the "
                "receiver's members explicitly"
            )

    def refuse(self, reason):
        self.lowering.refuse(f"CUDA {self.label} {reason}")

    def rewrite(self, value):
        lowering = self.lowering
        if isinstance(value, str):
            if value in self.static_fields:
                self.refuse(
                    f"names static member '{value}', which a CrossGL struct cannot hold"
                )
            elif value in self.fields:
                return MemberAccessNode(lowering.receiver, value)
            elif value == "this":
                self.refuse("uses 'this' as a pointer value, which CrossGL cannot form")
            return value
        if not isinstance(value, ASTNode):
            return value
        if (
            isinstance(value, UnaryOpNode)
            and value.op == "*"
            and value.operand == "this"
        ):
            return lowering.receiver
        if (
            isinstance(value, MemberAccessNode)
            and value.object == "this"
            and value.is_pointer
        ):
            if value.member in lowering.fields.get(self.owner, {}):
                return MemberAccessNode(lowering.receiver, value.member)
            self.refuse(f"names 'this->{value.member}', which is not a data member")
            return value
        if isinstance(value, FunctionCallNode):
            callee = value.name
            if callee == "lambda":
                self.refuse(
                    "contains a lambda, which the member function import does not handle"
                )
                return value
            if (
                isinstance(callee, MemberAccessNode)
                and callee.object == "this"
                and callee.is_pointer
            ):
                callee = value.name = str(callee.member)
            if isinstance(callee, str) and callee in lowering.method_names:
                if callee in lowering.methods.get(self.owner, {}):
                    args = _replace(list(value.args or []), self.rewrite)
                    return self.own_call(callee, args)
                if callee not in lowering.function_returns:
                    self.refuse(
                        f"calls member function '{callee}', which is not defined in the body of "
                        f"'{self.owner}' as a function CrossTL imports"
                    )
                    return value
        fields = _EXPRESSION_FIELDS.get(type(value))
        if fields is None:
            self.refuse(
                f"contains a {type(value).__name__}, which the member function import does not handle"
            )
            return value
        for field in fields:
            if hasattr(value, field):
                setattr(value, field, _replace(getattr(value, field), self.rewrite))
        if isinstance(value, FunctionCallNode) and not isinstance(value.name, str):
            value.name = _replace(value.name, self.rewrite)
        return value

    def own_call(self, name, args):
        """A call of member function ``name`` on the receiver, inside a member function of the same struct."""
        candidates = [
            method
            for method in self.lowering.methods[self.owner][name]
            if method.accepts(len(args))
        ]
        if len(candidates) == 1:
            return FunctionCallNode(
                candidates[0].free_name, [self.lowering.receiver, *args]
            )
        self.refuse(
            f"calls member function '{name}' with {len(args)} arguments, which no single imported member "
            f"function of '{self.owner}' takes"
        )
        return FunctionCallNode(name, args)


class _StructMethodLowering:
    def __init__(self, program, method_names, refuse):
        self.program = program
        self.method_names = method_names
        self.refuse = refuse
        self.used_names = set(source_identifier_names(program))
        self.receiver = self.unused_name(RECEIVER_NAME)
        self.aliases = {
            str(alias.name): str(alias.alias_type)
            for alias in getattr(program, "typedefs", None) or []
            if isinstance(alias, TypeAliasNode)
        }
        self.records = {}
        for struct in getattr(program, "structs", None) or []:
            if isinstance(struct, StructNode) and not isinstance(struct, EnumNode):
                self.records.setdefault(str(struct.name), struct)
        self.fields = {
            name: {
                str(member.name): member.vtype
                for member in getattr(struct, "members", None) or []
                if isinstance(member, VariableNode)
            }
            for name, struct in self.records.items()
        }
        self.static_fields = {
            name: {
                str(member.name)
                for member in getattr(struct, "members", None) or []
                if isinstance(member, VariableNode)
                and "static" in (getattr(member, "qualifiers", None) or [])
            }
            for name, struct in self.records.items()
        }
        self.methods = {}
        self.function_returns = {}
        for function in getattr(program, "functions", None) or []:
            self.function_returns.setdefault(str(function.name), []).append(
                function.return_type
            )
        self.global_types = {
            str(variable.name): variable.vtype
            for variable in getattr(program, "global_variables", None) or []
            if isinstance(variable, VariableNode)
        }

    def unused_name(self, name):
        candidate, suffix = name, 0
        while candidate in self.used_names:
            suffix += 1
            candidate = f"{name}{suffix}"
        self.used_names.add(candidate)
        return candidate

    def run(self):
        kept_by_owner = {}
        for struct in self.records.values():
            kept_by_owner[str(struct.name)] = list(
                getattr(struct, "methods", None) or []
            )
            struct.methods = []
        reachable = self.reachable(kept_by_owner)
        lowered = []
        for struct in self.records.values():
            kept = [
                function
                for function in kept_by_owner[str(struct.name)]
                if (str(struct.name), str(function.name)) in reachable
            ]
            by_name = {}
            for function in kept:
                by_name.setdefault(str(function.name), []).append(function)
            owner = str(struct.name)
            self.methods[owner] = {
                name: [
                    _Method(
                        owner,
                        function,
                        self.unused_name(
                            f"{owner}_{name}"
                            if len(functions) == 1
                            else f"{owner}_{name}_{index}"
                        ),
                    )
                    for index, function in enumerate(functions)
                ]
                for name, functions in by_name.items()
            }
            lowered.append((struct, kept))
        # Order before lowering: lowering replaces the calls the order is read from.
        lowered = [
            (struct, self.callees_first(str(struct.name), kept))
            for struct, kept in lowered
        ]
        free_functions = {}
        for struct, kept in lowered:
            owner = str(struct.name)
            for function in kept:
                method = self.method_for(owner, function)
                free_functions[id(function)] = self.lower(method)
        functions = list(getattr(self.program, "functions", None) or [])
        offset = 0
        for struct, kept in sorted(
            lowered, key=lambda item: getattr(item[0], "function_index", 0) or 0
        ):
            ordered = kept
            index = min(
                len(functions), (getattr(struct, "function_index", 0) or 0) + offset
            )
            new = [free_functions[id(function)] for function in ordered]
            functions[index:index] = new
            offset += len(new)
        self.program.functions = functions
        for function in functions:
            self.rewrite_calls(function)
        for kernel in getattr(self.program, "kernels", None) or []:
            self.rewrite_calls(kernel)

    def reachable(self, kept_by_owner):
        """``(owner, name)`` of every kept member function that a call of the program can reach.

        A member function nothing calls is left out, as before, so that a body the import cannot carry is refused
        only when the program uses it. Calls are matched by name, which keeps every overload a call might select.
        """
        owners_of = {}
        for owner, kept in kept_by_owner.items():
            for function in kept:
                owners_of.setdefault(str(function.name), set()).add(owner)
        marked, pending = set(), []

        def mark(owner, name):
            if (owner, name) not in marked and name in {
                str(function.name) for function in kept_by_owner.get(owner, [])
            }:
                marked.add((owner, name))
                pending.append((owner, name))

        def scan(body, owner):
            for node in _walk(body):
                if not isinstance(node, FunctionCallNode):
                    continue
                callee = node.name
                if isinstance(callee, MemberAccessNode):
                    name = str(callee.member).partition("<")[0].strip()
                    if callee.object == "this" and owner is not None:
                        mark(owner, name)
                    else:
                        for candidate in sorted(owners_of.get(name, ())):
                            mark(candidate, name)
                elif isinstance(callee, str) and owner is not None:
                    mark(owner, callee)

        for function in [
            *(getattr(self.program, "functions", None) or []),
            *(getattr(self.program, "kernels", None) or []),
        ]:
            scan(getattr(function, "body", None), None)
        while pending:
            owner, name = pending.pop()
            for function in kept_by_owner[owner]:
                if str(function.name) == name:
                    scan(getattr(function, "body", None), owner)
        return marked

    def method_for(self, owner, function):
        """The lowered method of ``function``, which :meth:`run` registered under ``owner``."""
        return next(
            method
            for method in self.methods[owner][str(function.name)]
            if method.function is function
        )

    def callees_first(self, owner, kept):
        """``kept`` ordered so that a member function follows the member functions of ``owner`` it calls."""
        own = self.methods.get(owner, {})
        ordered, visiting, done = [], set(), set()

        def visit(function):
            if id(function) in done or id(function) in visiting:
                return
            visiting.add(id(function))
            for node in _walk(getattr(function, "body", None)):
                if isinstance(node, FunctionCallNode) and isinstance(node.name, str):
                    for method in own.get(node.name, []):
                        visit(method.function)
            visiting.discard(id(function))
            done.add(id(function))
            ordered.append(function)

        for function in kept:
            visit(function)
        return ordered

    # Member function bodies.

    def lower(self, method):
        function, owner = method.function, method.owner
        context = _MemberContext(self, method)
        if "&" in str(function.return_type or ""):
            context.refuse(
                "returns a reference, and a CrossGL function returns a value"
            )
        receiver_type = (
            f"const {owner} &"
            if getattr(function, "struct_method_is_const", False)
            else f"{owner} &"
        )
        receiver = VariableNode(receiver_type, self.receiver, qualifiers=[])
        body = _replace(function.body, context.rewrite)
        free = FunctionNode(
            function.return_type,
            method.free_name,
            [receiver, *list(function.params or [])],
            body,
            list(getattr(function, "qualifiers", None) or []),
            list(getattr(function, "attributes", None) or []),
        )
        for key, value in vars(function).items():
            if key not in vars(free) and not key.startswith("struct_method_"):
                setattr(free, key, value)
        return free

    # Member calls everywhere.

    def rewrite_calls(self, function):
        scope = dict(self.global_types)
        scope.update(
            {
                str(param.name): param.vtype
                for param in getattr(function, "params", None) or []
                if isinstance(param, VariableNode)
            }
        )
        declared = {}
        for node in _walk(getattr(function, "body", None)):
            if isinstance(node, (VariableNode, SharedMemoryNode, RangeForNode)):
                vtype = node.vtype
                if _is_auto(vtype):
                    # ``auto`` takes the initializer's type, when the pass can read it.
                    vtype = self.type_of(getattr(node, "value", None), scope)
                declared.setdefault(str(node.name), set()).add(
                    None if vtype is None else str(vtype)
                )
        for name, types in declared.items():
            scope[name] = next(iter(types)) if len(types) == 1 else None
        label = f"function '{function.name}'"

        def rewrite(value):
            if not isinstance(value, ASTNode):
                return value
            for key, item in list(vars(value).items()):
                if isinstance(item, (ASTNode, list, tuple)):
                    setattr(value, key, _replace(item, rewrite))
            if isinstance(value, FunctionCallNode) and isinstance(
                value.name, MemberAccessNode
            ):
                return self.member_call(value, scope, label)
            return value

        function.body = _replace(getattr(function, "body", None), rewrite)

    def member_call(self, call, scope, label):
        access = call.name
        name, _, template_arguments = str(access.member).partition("<")
        name = name.strip()
        if name not in self.method_names:
            return call
        receiver_type = self.type_of(access.object, scope)
        record, indirection = _split_type(receiver_type, self.aliases)
        if (
            record is not None
            and "<" in record
            and record.split("<")[0].strip() in self.records
        ):
            # A specialization of a struct template, whose member functions are not imported.
            record = record.split("<")[0].strip()
        if (
            receiver_type is not None
            and record not in self.records
            and not _is_auto(record)
        ):
            # A member call on a value that is not a struct of the program, such as a cooperative group.
            return call
        wanted = 1 if access.is_pointer else 0
        if record is None or indirection != wanted:
            self.refuse(
                f"CUDA member call '{name}' in {label}: the type of its receiver is not a struct of the program "
                "that CrossTL can determine, so the member function cannot be imported"
            )
            return call
        args = list(call.args or [])
        candidates = [
            method
            for method in self.methods.get(record, {}).get(name, [])
            if method.accepts(len(args)) and not template_arguments
        ]
        if len(candidates) != 1:
            self.refuse(
                f"CUDA member call '{record}::{name}' in {label}: no single member function defined in the "
                f"body of '{record}' takes {len(args)} arguments (constructors, operators, templates, static, "
                "virtual and out-of-class member functions are not imported)"
            )
            return call
        receiver = access.object
        if access.is_pointer:
            receiver = ArrayAccessNode(receiver, "0")
        return FunctionCallNode(candidates[0].free_name, [receiver, *args])

    def type_of(self, expression, scope):
        """The declared type text of ``expression``, or ``None`` when the pass cannot tell."""
        if isinstance(expression, str):
            return scope.get(expression)
        if isinstance(expression, MemberAccessNode):
            record, indirection = _split_type(
                self.type_of(expression.object, scope), self.aliases
            )
            if record not in self.fields or indirection != (
                1 if expression.is_pointer else 0
            ):
                return None
            return self.fields[record].get(str(expression.member))
        if isinstance(expression, ArrayAccessNode):
            text = self.type_of(expression.array, scope)
            if text is None:
                return None
            stripped = _ARRAY_SUFFIX.sub("", str(text))
            if stripped != str(text):
                return stripped
            if "*" in str(text):
                head, _, tail = str(text).rpartition("*")
                return head + tail
            return None
        if isinstance(expression, UnaryOpNode) and expression.op == "*":
            text = self.type_of(expression.operand, scope)
            if text is None or "*" not in str(text):
                return None
            head, _, tail = str(text).rpartition("*")
            return head + tail
        if isinstance(expression, CastNode):
            return expression.target_type
        if isinstance(expression, FunctionCallNode) and isinstance(
            expression.name, str
        ):
            returns = self.function_returns.get(expression.name, [])
            if len(returns) == 1:
                return returns[0]
            if expression.name in self.records:
                return expression.name
        return None
