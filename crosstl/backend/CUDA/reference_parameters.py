"""Reference parameters of CUDA and HIP functions.

A reference parameter binds the caller's object (C++17 [dcl.init.ref], [expr.call]), so the callee's writes through
it reach the caller. The CUDA and HIP parsers keep a parameter's declarator in its type text, as in ``float &``,
``const Pair &``, ``float & [2]``, ``T &&`` or ``float * &``. :class:`ReferenceParameters` reads that text for every
function of a program and decides how each parameter imports:

* a reference the callee writes, directly, through a member or element, by ``++``/``--``, by taking its address,
  by passing it to a written reference parameter or, for an lvalue reference, by calling a method on it, imports
  as ``inout``. CrossGL's
  ``inout`` copies the argument in at the call and back out at the return;
* a reference the callee only reads, and every ``const T&``, imports as a value parameter, which reads the same value;
* a written rvalue reference (``T&&``) and a written reference to a pointer (``T*&``) raise
  :class:`ReferenceParameterError`: ``inout`` needs an lvalue argument, and CrossGL cannot rebind a caller's pointer.

Copy-in and copy-out matches a reference only while nothing else reaches the argument's object during the call. A
call that passes an argument to an ``inout`` parameter names either storage of the caller's own (a by-value local or
parameter whose address the caller never takes, a member or element of one, or an ``inout`` parameter of the caller)
or an element or member of shared, global or pointed-to memory, as in ``grow(out[i], k)``. The reference arguments of
a call must not overlap. Memory that other threads and other arguments can reach binds an ``inout`` parameter only
when the call cannot tell a copy from a reference:

* the callee is *isolated*: it and every function it calls take no pointer or array parameter, name no writable
  ``__device__``, ``__shared__`` or ``__managed__`` variable, call no barrier, fence, atomic, warp, clock or
  cache-operator built-in, no method and no namespaced function, and hold no inline assembly. Its only way to the
  argument's memory is then the reference itself, and with no synchronization in the call another thread can observe
  the object only through a data race, whose result the CUDA memory model leaves undefined (CUDA C++ Programming
  Guide, "Memory Consistency"), so a copy written back at the return is one of the behaviours a reference allows;
* every other reference argument of the call is storage of the caller's own, because two pointers may name the same
  object and the copies would then miss each other's writes;
* the argument's indices have no side effects, so the write-back names the element the call read.

Any other argument in such memory raises :class:`ReferenceParameterError`.
"""

from __future__ import annotations

import re

from ...translator.diagnostics import CODE_PREFIX, TranslationDiagnostic
from ..common_ast import (
    ArrayAccessNode,
    AssignmentNode,
    ASTNode,
    CastNode,
    EnumNode,
    FunctionCallNode,
    FunctionNode,
    MemberAccessNode,
    RangeForNode,
    StructNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
)

__all__ = ["ReferenceParameterError", "ReferenceParameters", "reference_kind"]

_OPEN = {"<": ">", "(": ")", "[": "]"}
_CLOSE = {">", ")", "]"}
_ALIAS_WORDS = re.compile(r"(\s+|[&*<>(),\[\]])")
_STORAGE_QUALIFIERS = frozenset(
    {"__shared__", "__device__", "__constant__", "__managed__", "static", "extern"}
)
_READ_ONLY_QUALIFIERS = frozenset({"__constant__", "const", "constexpr"})
# Built-in calls through which a callee synchronizes with, or reaches memory of, other threads: barriers, fences,
# atomics, warp and wave collectives, clocks, cache-operator loads and stores, cooperative groups and inline assembly.
_SHARED_STATE_CALL = re.compile(
    r"sync|fence|barrier|atomic|shfl|ballot|vote|match|reduce|activemask|^__any|^__all|lane|wave|warp|permute|"
    r"readfirst|clock|timer|^__ld|^__st|nontemporal|cooperative|grid|group|asm|::",
    re.IGNORECASE,
)
# Statements and expressions through which a callee synchronizes, calls a method or callable value, launches work,
# allocates, declares block-shared storage or runs inline assembly.
_SHARED_STATE_NODE = re.compile(
    r"Sync|Asm|MethodCall|^CallNode$|Launch|^NewNode$|^DeleteNode$|SharedMemory"
)
_SIDE_EFFECT_OPS = ("++", "--")
_WORD = re.compile(r"[A-Za-z_]\w*")


class ReferenceParameterError(TranslationDiagnostic):
    """A CUDA or HIP reference parameter or argument that CrossGL cannot express."""

    code = f"{CODE_PREFIX}reference-parameter-unsupported"


def _top_level_tokens(type_text):
    """The ``&``, ``&&``, ``*`` and word tokens of ``type_text`` outside template, call and array brackets."""
    tokens = []
    depth = 0
    word = []
    text = str(type_text or "")
    index = 0

    def flush():
        if word and depth == 0:
            tokens.append("".join(word))
        word.clear()

    while index < len(text):
        char = text[index]
        if char in _OPEN:
            flush()
            depth += 1
        elif char in _CLOSE:
            flush()
            depth = max(depth - 1, 0)
        elif depth:
            pass
        elif char == "&":
            flush()
            if text.startswith("&&", index):
                tokens.append("&&")
                index += 1
            else:
                tokens.append("&")
        elif char == "*":
            flush()
            tokens.append("*")
        elif char.isspace():
            flush()
        else:
            word.append(char)
        index += 1
    flush()
    return tokens


def reference_kind(type_text):
    """``None``, ``"const"``, ``"lvalue"``, ``"rvalue"`` or ``"pointer"`` for a parameter declared ``type_text``."""
    tokens = _top_level_tokens(type_text)
    references = [index for index, token in enumerate(tokens) if token in ("&", "&&")]
    if not references:
        return None
    if len(references) > 1:
        return "rvalue"
    before = tokens[: references[0]]
    if before and before[-1] == "const":
        return "const"
    if tokens[references[0]] == "&&":
        return "rvalue"
    if "*" in before:
        return "pointer"
    if "const" in before:
        return "const"
    return "lvalue"


def _parameter_type(parameter):
    if isinstance(parameter, dict):
        return str(parameter.get("type", "") or "")
    return str(getattr(parameter, "vtype", "") or "")


def _parameter_name(parameter):
    if isinstance(parameter, dict):
        return str(parameter.get("name", "") or "")
    return str(getattr(parameter, "name", "") or "")


def _parameters(function):
    return list(getattr(function, "params", None) or [])


def _walk(node, seen=None):
    """Every backend AST node under ``node``, parents first."""
    seen = set() if seen is None else seen
    if isinstance(node, (list, tuple)):
        for item in node:
            yield from _walk(item, seen)
        return
    if not isinstance(node, ASTNode) or id(node) in seen:
        return
    seen.add(id(node))
    yield node
    for value in vars(node).values():
        if isinstance(value, (ASTNode, list, tuple)):
            yield from _walk(value, seen)


def _program_parts(program):
    """The functions of ``program``, nested ones included, the names it declares outside them, the read-only ones,
    and the records with a pointer or reference member mapped to the record types their members name.
    """
    functions, names, read_only, records = [], set(), set(), {}
    pending = [program]
    seen = set()
    while pending:
        node = pending.pop()
        if isinstance(node, (list, tuple)):
            pending.extend(reversed(node))
            continue
        if not isinstance(node, ASTNode) or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, FunctionNode):
            functions.append(node)
            continue
        if isinstance(node, EnumNode):
            continue
        if isinstance(node, StructNode):
            member_types = [
                str(getattr(member, "vtype", "") or "")
                for member in getattr(node, "members", None) or []
                if isinstance(member, VariableNode)
            ]
            records[str(node.name)] = (
                any(_reaches_memory(text) for text in member_types),
                {word for text in member_types for word in _WORD.findall(text)},
            )
            # Members are not module-scope variables; member functions and nested records still count.
            pending.extend(
                item
                for item in _walk(node)
                if isinstance(item, (FunctionNode, StructNode)) and item is not node
            )
            continue
        if isinstance(node, VariableNode):
            names.add(str(node.name))
            qualifiers = {str(q) for q in getattr(node, "qualifiers", None) or []}
            vtype = str(getattr(node, "vtype", "") or "")
            # A pointer variable is read-only only as a pointer: the memory it points to stays writable.
            if not _reaches_memory(vtype) and (
                getattr(node, "is_const", False)
                or (qualifiers | set(vtype.split())) & _READ_ONLY_QUALIFIERS
            ):
                read_only.add(str(node.name))
        pending.extend(
            value
            for value in vars(node).values()
            if isinstance(value, (ASTNode, list, tuple))
        )
    return functions, names, read_only, records


def _reaches_memory(type_text):
    """Whether a value of type ``type_text`` holds a pointer, a reference or an array, which reach memory."""
    return any(mark in str(type_text) for mark in ("*", "&", "["))


def _pointer_records(records):
    """The record names whose values hold a pointer or reference, directly or through a member of record type."""
    holding = {name for name, (direct, _) in records.items() if direct}
    changed = True
    while changed:
        changed = False
        for name, (_, words) in records.items():
            if name not in holding and words & holding:
                holding.add(name)
                changed = True
    return holding


def _path(argument):
    """``(root, path)`` of an lvalue built from a name, ``.`` members and indices; ``None`` otherwise."""
    path = []
    node = argument
    while True:
        if isinstance(node, str):
            return node.strip(), tuple(reversed(path))
        if isinstance(node, MemberAccessNode):
            if (
                getattr(node, "is_pointer", False)
                or getattr(node, "operator", ".") != "."
            ):
                return None
            path.append(("member", str(node.member)))
            node = node.object
        elif isinstance(node, ArrayAccessNode):
            index = node.index.strip() if isinstance(node.index, str) else None
            path.append(("index", index))
            node = node.array
        else:
            return None


def _root(expression):
    """The name an lvalue expression starts from, through members, elements, ``->`` and ``*``."""
    node = expression
    while True:
        if isinstance(node, str):
            return node.strip()
        if isinstance(node, MemberAccessNode):
            node = node.object
        elif isinstance(node, ArrayAccessNode):
            node = node.array
        elif isinstance(node, UnaryOpNode) and node.op == "*":
            node = node.operand
        else:
            return None


def _overlap(first, second):
    for a, b in zip(first, second):
        if a == b and a[1] is not None:
            continue
        if a[0] == b[0] == "member":
            return False
        if (
            a[0] == b[0] == "index"
            and a[1] is not None
            and b[1] is not None
            and a[1].isdigit()
            and b[1].isdigit()
        ):
            return False
        return True
    return True


def _short_name(name):
    if isinstance(name, MemberAccessNode):
        name = name.member
    if not isinstance(name, str):
        return None
    return name.rsplit("::", 1)[-1].strip()


def _is_storage_type(type_text):
    tokens = _top_level_tokens(type_text)
    return "*" not in tokens and "&" not in tokens and "&&" not in tokens


def _identifiers(node):
    """Every word the nodes under ``node`` hold as text: the names they read, write and call, and their literals."""
    for item in _walk(node):
        for value in vars(item).values():
            values = value if isinstance(value, (list, tuple)) else (value,)
            for text in values:
                if isinstance(text, str):
                    yield from _WORD.findall(text)


def _side_effect_free(expression):
    """Whether evaluating ``expression`` twice reads what evaluating it once does: no call, assignment or ``++``/``--``."""
    for node in _walk(expression):
        if isinstance(node, (FunctionCallNode, AssignmentNode)):
            return False
        if any(step in str(getattr(node, "op", "")) for step in _SIDE_EFFECT_OPS):
            return False
    return True


class ReferenceParameters:
    """How the reference parameters of every function of a CUDA or HIP program import; see the module docstring."""

    def __init__(self, program, *, target="cgl"):
        self.target = target
        self.aliases = {
            str(node.name): str(node.alias_type)
            for node in _walk(program)
            if isinstance(node, TypeAliasNode)
            and not getattr(node, "array_sizes", None)
            and not getattr(node, "declarator_type_suffix", "")
        }
        self.functions, self.module_names, read_only, records = _program_parts(program)
        self.writable_module_names = self.module_names - read_only
        self.pointer_records = _pointer_records(records)
        self.by_name = {}
        for function in self.functions:
            name = _short_name(getattr(function, "name", ""))
            self.by_name.setdefault(name, []).append(function)
        self.kinds = {
            id(function): [
                reference_kind(self.resolve(_parameter_type(parameter)))
                for parameter in _parameters(function)
            ]
            for function in self.functions
        }
        self.written = {id(function): set() for function in self.functions}
        self._find_writes()
        self._check_kinds()
        self.isolated = self._find_isolated()
        for function in self.functions:
            self._check_calls(function)

    def resolve(self, type_text):
        """``type_text`` with the program's type aliases replaced by their declarators, as in ``Ref`` -> ``State &``."""
        text = str(type_text or "")
        for _ in range(8):
            parts = _ALIAS_WORDS.split(text)
            if not any(part in self.aliases for part in parts):
                break
            # Only alias names are replaced; every other word and separator is the rest of the declarator.
            text = "".join(
                self.aliases[part] if part in self.aliases else part for part in parts
            )
        return text

    # Modes -------------------------------------------------------------------------------------------------------
    def mode(self, function, index):
        """``"inout"`` when parameter ``index`` of ``function`` writes through to the caller, else ``None``."""
        kinds = self.kinds.get(id(function))
        if kinds is None:
            kind = reference_kind(
                self.resolve(_parameter_type(_parameters(function)[index]))
            )
            if kind in ("rvalue", "pointer"):
                self._refuse_kind(_parameter_name(_parameters(function)[index]), kind)
            return "inout" if kind == "lvalue" else None
        if kinds[index] == "lvalue" and index in self.written[id(function)]:
            return "inout"
        return None

    def _inout_indices(self, name, count):
        indices = set()
        for candidate in self.by_name.get(_short_name(name), ()):
            if len(_parameters(candidate)) == count:
                indices.update(
                    index
                    for index in self.written[id(candidate)]
                    if self.kinds[id(candidate)][index]
                    in ("lvalue", "rvalue", "pointer")
                )
        return indices

    def _reference_indices(self, name, count):
        indices = set()
        for candidate in self.by_name.get(_short_name(name), ()):
            if len(_parameters(candidate)) == count:
                indices.update(
                    index
                    for index, kind in enumerate(self.kinds[id(candidate)])
                    if kind is not None
                )
        return indices

    # Writes ------------------------------------------------------------------------------------------------------
    def _direct_writes(self, function):
        """Parameter names written in ``function``'s body, and calls whose arguments may write a parameter."""
        names = {
            _parameter_name(parameter): index
            for index, parameter in enumerate(_parameters(function))
            if self.kinds[id(function)][index] is not None
        }
        written, calls = set(), []
        if not names:
            return written, calls

        def record(target, method=False):
            position = self._written_position(function, names, target)
            if position is None:
                return
            # A method call may mutate its object, so it writes an lvalue reference. On a forwarding or rvalue
            # reference it is taken as a call, as for the callables generic code invokes through ``F&& f``.
            if method and self.kinds[id(function)][position] != "lvalue":
                return
            written.add(position)

        for node in _walk(getattr(function, "body", None)):
            if isinstance(node, AssignmentNode):
                record(node.left)
            elif hasattr(node, "operand") and any(
                step in str(getattr(node, "op", "")) for step in ("++", "--")
            ):
                record(node.operand)
            elif isinstance(node, UnaryOpNode) and node.op == "&":
                record(node.operand)
            elif isinstance(node, FunctionCallNode):
                if isinstance(node.name, MemberAccessNode):
                    record(node.name.object, method=True)
                calls.append(node)
        return written, (names, calls)

    def _written_position(self, function, names, target):
        """The index of the parameter that writing ``target`` writes, if any.

        Writing through a reference to a pointer (``p[i]``, ``*p``, ``p->m``) writes the pointee, not the pointer.
        """
        root = _root(target)
        position = names.get(root)
        if position is None:
            return None
        whole = isinstance(target, str)
        if self.kinds[id(function)][position] == "pointer" and not whole:
            return None
        return position

    def _find_writes(self):
        pending = {}
        for function in self.functions:
            written, calls = self._direct_writes(function)
            self.written[id(function)] = written
            pending[id(function)] = calls
        changed = True
        while changed:
            changed = False
            for function in self.functions:
                calls = pending[id(function)]
                if not calls:
                    continue
                names, nodes = calls
                for node in nodes:
                    arguments = list(getattr(node, "args", None) or [])
                    inout = self._inout_indices(node.name, len(arguments))
                    for index in inout:
                        if index >= len(arguments):
                            continue
                        position = self._written_position(
                            function, names, arguments[index]
                        )
                        if (
                            position is not None
                            and position not in self.written[id(function)]
                        ):
                            self.written[id(function)].add(position)
                            changed = True

    def _check_kinds(self):
        for function in self.functions:
            for index in sorted(self.written[id(function)]):
                kind = self.kinds[id(function)][index]
                if kind in ("rvalue", "pointer"):
                    self._refuse_kind(
                        _parameter_name(_parameters(function)[index]), kind
                    )

    def _refuse_kind(self, name, kind):
        if kind == "rvalue":
            reason = f"parameter '{name}' is an rvalue reference the function writes, which CrossGL cannot express"
        else:
            reason = (
                f"parameter '{name}' is a reference to a pointer the function writes; CrossGL cannot rebind "
                "the caller's pointer"
            )
        raise ReferenceParameterError(reason, target=self.target)

    # Isolated callees --------------------------------------------------------------------------------------------
    def _locally_isolated(self, function):
        """Whether ``function``'s own parameters and body reach no memory but its references (module docstring)."""
        for parameter in _parameters(function):
            raw = self.resolve(_parameter_type(parameter))
            if "*" in _top_level_tokens(raw) or "[" in raw or "volatile" in raw:
                return False
            if set(_WORD.findall(raw)) & self.pointer_records:
                return False
        body = getattr(function, "body", None)
        for node in _walk(body):
            if _SHARED_STATE_NODE.search(type(node).__name__):
                return False
            if isinstance(node, CastNode) and "*" in str(node.target_type):
                return False
            if isinstance(node, FunctionCallNode):
                name = node.name
                if not isinstance(name, str):
                    return False
                if _short_name(name) not in self.by_name and _SHARED_STATE_CALL.search(
                    name
                ):
                    return False
        return not any(
            word in self.writable_module_names for word in _identifiers(body)
        )

    def _find_isolated(self):
        """The ids of the isolated functions: locally isolated, and calling only isolated functions of the program."""
        isolated = {
            id(function)
            for function in self.functions
            if self._locally_isolated(function)
        }
        changed = True
        while changed:
            changed = False
            for function in self.functions:
                if id(function) not in isolated:
                    continue
                for node in _walk(getattr(function, "body", None)):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    callees = self.by_name.get(_short_name(node.name), ())
                    if any(id(callee) not in isolated for callee in callees):
                        isolated.discard(id(function))
                        changed = True
                        break
        return isolated

    def _copies_shared_argument(self, node, index, arguments, references, local):
        """Whether argument ``index`` of call ``node``, in memory others can reach, may bind its ``inout`` by copy."""
        callees = [
            callee
            for callee in self.by_name.get(_short_name(node.name), ())
            if len(_parameters(callee)) == len(arguments)
        ]
        if not callees or any(id(callee) not in self.isolated for callee in callees):
            return False
        if not _side_effect_free(arguments[index]):
            return False
        for other, argument in enumerate(arguments):
            address = isinstance(argument, UnaryOpNode) and argument.op == "&"
            if other == index or (other not in references and not address):
                continue
            path = _path(argument.operand if address else argument)
            if path is None or path[0] not in local:
                return False
        return True

    # Call sites --------------------------------------------------------------------------------------------------
    def _local_storage(self, function):
        """Names that denote storage only ``function`` itself reaches."""
        local, other = set(), set(self.module_names)
        for index, parameter in enumerate(_parameters(function)):
            name = _parameter_name(parameter)
            raw = self.resolve(_parameter_type(parameter))
            if self.mode(function, index) == "inout" or (
                _is_storage_type(raw) and "[" not in raw
            ):
                local.add(name)
            else:
                other.add(name)
        for node in _walk(getattr(function, "body", None)):
            if isinstance(node, VariableNode):
                qualifiers = {str(q) for q in getattr(node, "qualifiers", None) or []}
                raw = self.resolve(str(getattr(node, "vtype", "") or ""))
                if (
                    type(node) is VariableNode
                    and not (qualifiers & _STORAGE_QUALIFIERS)
                    and _is_storage_type(raw)
                ):
                    local.add(str(node.name))
                else:
                    other.add(str(node.name))
            elif isinstance(node, RangeForNode):
                raw = self.resolve(str(node.vtype or ""))
                (local if _is_storage_type(raw) else other).add(str(node.name))
            elif isinstance(node, UnaryOpNode) and node.op == "&":
                root = _root(node.operand)
                if root is not None:
                    other.add(root)
        return local - other

    def _reference_names(self, function):
        """Names ``function`` declares as references, which a copy in and out of the referred object cannot bind."""
        names = {
            _parameter_name(parameter)
            for index, parameter in enumerate(_parameters(function))
            if self.mode(function, index) != "inout"
            and reference_kind(self.resolve(_parameter_type(parameter))) is not None
        }
        for node in _walk(getattr(function, "body", None)):
            if isinstance(node, (VariableNode, RangeForNode)):
                raw = self.resolve(str(getattr(node, "vtype", "") or ""))
                if reference_kind(raw) is not None or "&" in raw:
                    names.add(str(node.name))
        return names

    def _check_calls(self, function):
        local = reference_locals = None
        for node in _walk(getattr(function, "body", None)):
            if not isinstance(node, FunctionCallNode):
                continue
            arguments = list(getattr(node, "args", None) or [])
            inout = self._inout_indices(node.name, len(arguments))
            if not inout:
                continue
            if local is None:
                local = self._local_storage(function)
                reference_locals = self._reference_names(function)
            references = self._reference_indices(node.name, len(arguments))
            name = _short_name(node.name)
            reached = []
            for index, argument in enumerate(arguments):
                address = isinstance(argument, UnaryOpNode) and argument.op == "&"
                if index not in references and not address:
                    continue
                path = _path(argument.operand if address else argument)
                if (
                    index in inout
                    and (path is None or path[0] not in local)
                    and (
                        path is None
                        or path[0] in reference_locals
                        or not self._copies_shared_argument(
                            node, index, arguments, references, local
                        )
                    )
                ):
                    raise ReferenceParameterError(
                        f"argument {index + 1} of '{name}' binds a reference parameter to storage that other "
                        "threads or other arguments can reach; CrossGL passes it in and out by copy",
                        target=self.target,
                    )
                if path is None:
                    continue
                for other in reached:
                    if other[0] == path[0] and _overlap(other[1], path[1]):
                        raise ReferenceParameterError(
                            f"arguments of '{name}' pass overlapping parts of '{path[0]}' by reference; CrossGL "
                            "passes each in and out by copy",
                            target=self.target,
                        )
                reached.append(path)
