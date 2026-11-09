"""HLSL namespaces: declarations take their qualified names, and every reference resolves to its declaration.

HLSL 2021 has C++ namespaces (DXC, "HLSL 2021 language features"). The parser flattened a ``namespace ns { ... }``
block into the enclosing scope and dropped its name, while references kept their spelling: ``ns::neg(x)`` called a
function the program no longer declared and was copied into CrossGL verbatim, ``ns::Pair`` named a ``ns_Pair``
nobody declared, and two namespaces declaring ``neg`` produced two indistinguishable functions.

The parser now records each declaration's namespace path (:func:`qualify_namespace_declarations`) and the
``using namespace`` directives and ``using ns::name`` declarations at file or namespace scope
(:func:`using_directive_path`). :func:`resolve_namespaces` then gives every namespaced declaration a flat CrossGL
name (``ns_inner_twice``), and rewrites every reference: qualified names (``ns::inner::twice``, ``inner::twice``
inside ``ns``, ``::neg``) and unqualified ones that C++ lookup finds in an enclosing namespace or through a using
directive. Locals and parameters shadow namespace members.

Resources keep their declared names, since a host binds them by name; two namespaces that declare the same resource
name are refused rather than merged. Constant-buffer members are file-scope names even inside a namespace, as DXC
declares them.
"""

from __future__ import annotations

import re

from ..common_ast import (
    FunctionCallNode,
    FunctionNode,
    MemberAccessNode,
    MethodCallNode,
    StructNode,
    VariableNode,
)

__all__ = [
    "qualify_namespace_declarations",
    "resolve_namespaces",
    "source_name",
    "using_directive_path",
]

# The ShaderNode collections whose entries a namespace block can declare.
_COLLECTIONS = (
    "functions",
    "structs",
    "global_variables",
    "cbuffers",
    "enums",
    "typedefs",
)
# A possibly qualified name: ``x``, ``ns::x``, ``::x``; not a member (``.x``), a literal suffix (``1.0f``) or a
# member of a template specialization (``::zero`` in ``Leg<T>::zero``).
_QUALIFIED_NAME = re.compile(
    r"(?<![\w.:>])(::\s*)?[A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)*"
)
# Fields that hold names of declarations, members, operators or metadata rather than references.
_SKIPPED_FIELDS = frozenset(
    {
        "name",
        "member",
        "method",
        "semantic",
        "op",
        "operator",
        "qualifiers",
        "attributes",
        "register",
        "source_location",
        "parent",
        "hlsl_namespace",
        "packoffset",
    }
)
# Fields that hold a type spelling.
_TYPE_FIELDS = frozenset(
    {"vtype", "return_type", "target_type", "alias_type", "type_name", "vector_type"}
)


def using_directive_path(tokens):
    """The namespace or member path a ``using`` directive names, from its identifier tokens."""
    return tuple(str(token) for token in tokens if str(token) not in ("", "::"))


def qualify_namespace_declarations(namespace_ast, path, using_directives):
    """Prefix ``path`` to the namespace of every declaration in a parsed namespace block.

    ``using_directives`` are the block's own ``(scope, kind, target)`` records; they are returned with ``path``
    prefixed to their scope, for the enclosing parser to keep.
    """
    path = tuple(path)
    for collection in _COLLECTIONS:
        for declaration in getattr(namespace_ast, collection, None) or []:
            inner = tuple(getattr(declaration, "hlsl_namespace", ()) or ())
            declaration.hlsl_namespace = path + inner
    return [
        (path + tuple(scope), kind, tuple(target))
        for scope, kind, target in using_directives or []
    ]


def _is_interface_variable(variable):
    """Whether a global keeps its declared name: a resource or uniform the host binds by name."""
    qualifiers = {str(item) for item in getattr(variable, "qualifiers", None) or []}
    return "static" not in qualifiers and "groupshared" not in qualifiers


class _Namespaces:
    def __init__(self, ast, refuse):
        self.ast = ast
        self.refuse = refuse
        self.table = {}
        self.directives = [
            (tuple(scope), kind, tuple(target))
            for scope, kind, target in getattr(ast, "hlsl_using_directives", None) or []
        ]
        self.namespaces = set()
        # The names of file-scope declarations.
        self.global_names = set()

    def _qualify_out_of_line_definitions(self):
        """Give ``float3 A::B::f() { ... }`` the namespace ``A::B`` its prototype was declared in."""
        for collection in _COLLECTIONS:
            for declaration in getattr(self.ast, collection, None) or []:
                if getattr(declaration, "hlsl_namespace", None):
                    self.namespaces.add(tuple(declaration.hlsl_namespace))
        known = {
            path[:depth]
            for path in self.namespaces
            for depth in range(1, len(path) + 1)
        }
        for function in getattr(self.ast, "functions", None) or []:
            name = getattr(function, "name", None)
            if getattr(function, "hlsl_namespace", None) or not isinstance(name, str):
                continue
            parts = tuple(part.strip() for part in name.strip().strip(":").split("::"))
            if len(parts) > 1 and parts[:-1] in known:
                function.hlsl_namespace = parts[:-1]
                function.name = parts[-1]

    def collect(self):
        """Map each namespaced declaration's qualified name to its CrossGL name."""
        self._qualify_out_of_line_definitions()
        taken = set()
        namespaced = []
        for collection in _COLLECTIONS:
            for declaration in getattr(self.ast, collection, None) or []:
                path = tuple(getattr(declaration, "hlsl_namespace", ()) or ())
                name = getattr(declaration, "name", None)
                if not isinstance(name, str):
                    continue
                if path:
                    namespaced.append((collection, declaration, path, name))
                else:
                    taken.add(name)
                    self.global_names.add(name)
        for cbuffer in getattr(self.ast, "cbuffers", None) or []:
            # DXC declares a constant buffer's members at file scope even inside a namespace: ``res::bias`` is
            # "no member named 'bias' in namespace 'res'".
            for member in getattr(cbuffer, "members", None) or []:
                if isinstance(getattr(member, "name", None), str):
                    self.global_names.add(member.name)
        interface = {}
        for collection, declaration, path, name in namespaced:
            for depth in range(1, len(path) + 1):
                self.namespaces.add(path[:depth])
            key = path + (name,)
            if key in self.table:
                continue  # an overload or a redeclaration shares the name
            if collection == "global_variables" and _is_interface_variable(declaration):
                if name in taken or name in interface:
                    other = interface.get(name, ())
                    self.refuse(
                        f"HLSL resource '{'::'.join(key)}' has the same name as "
                        f"'{'::'.join(other + (name,)) if other else name}'; CrossGL binds "
                        "resources by their declared name, so namespaces cannot tell them apart"
                    )
                interface[name] = path
                self.table[key] = name
                continue
            flat = "_".join(key)
            while flat in taken:
                flat += "_"
            taken.add(flat)
            self.table[key] = flat
        self._collect_enumerators()
        return namespaced

    def _collect_enumerators(self):
        """An unscoped enumerator is a member of its enumeration's scope: ``On`` reads ``Mode::On`` in CrossGL."""
        for enum in getattr(self.ast, "enums", None) or []:
            name = getattr(enum, "name", None)
            if getattr(enum, "is_scoped", False) or not isinstance(name, str):
                continue
            path = tuple(getattr(enum, "hlsl_namespace", ()) or ())
            flat = self.table.get(path + (name,), name)
            for member in getattr(enum, "members", None) or []:
                member_name = member[0] if isinstance(member, tuple) else member
                if isinstance(member_name, str):
                    self.table.setdefault(
                        path + (member_name,), f"{flat}::{member_name}"
                    )

    def lookup(self, text, context, locals_):
        """The CrossGL name of ``text``; a member of a namespaced type (``a::Kind::Y``) keeps its member part."""
        found = self._lookup_name(text, context, locals_)
        if found is not None:
            return found
        parts = text.strip().split("::")
        for length in range(len(parts) - 1, 0, -1):
            prefix = "::".join(parts[:length])
            if not prefix.strip():
                break
            owner = self._lookup_name(prefix, context, locals_)
            if owner is not None:
                return "::".join([owner] + [part.strip() for part in parts[length:]])
        return None

    def _lookup_name(self, text, context, locals_):
        """The CrossGL name of the possibly qualified ``text`` seen from namespace ``context``, else ``None``.

        Unqualified and relative names follow C++ lookup: the innermost enclosing namespace that declares the name
        wins. A ``using namespace`` directive makes the nominated namespace's members visible from the nearest
        namespace that encloses both it and the directive; a ``using ns::name`` declaration declares ``name`` in
        the directive's own namespace. Two different declarations found at the same scope are ambiguous, which is
        refused, since only overload resolution across namespaces could choose between them.
        """
        absolute = text.lstrip().startswith("::")
        parts = tuple(part.strip() for part in text.strip().strip(":").split("::"))
        if absolute:
            return self.table.get(parts)
        if len(parts) == 1 and parts[0] in locals_:
            return None
        for depth in range(len(context), -1, -1):
            prefix = context[:depth]
            hits = []
            found = self.table.get(prefix + parts)
            if found is not None:
                hits.append(found)
            if depth == 0 and len(parts) == 1 and parts[0] in self.global_names:
                # A file-scope declaration keeps its spelling.
                hits.append(None)
            hits.extend(self._directive_hits(prefix, context, parts))
            if not hits:
                continue
            distinct = list(dict.fromkeys(hits))
            if len(distinct) > 1:
                self.refuse(
                    f"HLSL name '{text.strip()}' is ambiguous: it names declarations in more than one "
                    "namespace (through a using directive), and CrossGL keeps no overload set across namespaces"
                )
            return distinct[0]
        return None

    def _directive_hits(self, prefix, context, parts):
        """Declarations that ``using`` directives make visible at namespace ``prefix`` for a lookup from ``context``."""
        hits = []
        for scope, kind, target in self.directives:
            if context[: len(scope)] != scope:
                continue  # the directive is not in effect where the name is read
            target = self._namespace_path(target, scope)
            if kind == "namespace":
                common = 0
                while (
                    common < min(len(scope), len(target))
                    and scope[common] == target[common]
                ):
                    common += 1
                if common != len(prefix) or prefix != scope[:common]:
                    continue
                found = self.table.get(target + parts)
            else:
                if scope != prefix or len(parts) != 1 or target[-1:] != parts:
                    continue
                found = self.table.get(target)
            if found is not None:
                hits.append(found)
        return hits

    def _namespace_path(self, target, scope):
        """``target`` as seen from ``scope``: relative to an enclosing namespace when one declares it."""
        for depth in range(len(scope), -1, -1):
            candidate = scope[:depth] + target
            if candidate in self.namespaces or candidate in self.table:
                return candidate
        return target

    def resolve_text(self, text, context, locals_):
        if not isinstance(text, str) or not text:
            return text

        def replace(match):
            spelling = match.group(0)
            found = self.lookup(spelling, context, locals_)
            if found is not None:
                return found
            # ``::max`` and ``::g`` name file-scope declarations, which keep their names.
            return re.sub(r"^::\s*", "", spelling)

        return _QUALIFIED_NAME.sub(replace, text)

    def resolve(self, value, context, scopes):
        """Rewrite the references under ``value``; ``scopes`` is the stack of local name sets."""
        if isinstance(value, list):
            scopes.append(set())
            for index, item in enumerate(value):
                if isinstance(item, str):
                    value[index] = self.resolve_text(item, context, _names(scopes))
                else:
                    self.resolve(item, context, scopes)
            scopes.pop()
            return
        if isinstance(value, VariableNode):
            value.vtype = self.resolve_text(value.vtype, context, _names(scopes))
            self._resolve_fields(value, context, scopes, skip={"vtype"})
            if isinstance(value.name, str) and scopes:
                scopes[-1].add(value.name)
            return
        if isinstance(value, FunctionCallNode) and isinstance(value.name, str):
            value.name = self.resolve_text(value.name, context, _names(scopes))
        elif isinstance(value, FunctionCallNode):
            self.resolve(value.name, context, scopes)
        if isinstance(value, MethodCallNode):
            obj = value.object
            if isinstance(obj, str):
                value.object = self.resolve_text(obj, context, _names(scopes))
        if isinstance(value, MemberAccessNode) and isinstance(value.object, str):
            value.object = self.resolve_text(value.object, context, _names(scopes))
        if hasattr(value, "__dict__") and not isinstance(value, type):
            self._resolve_fields(value, context, scopes)

    def _resolve_fields(self, node, context, scopes, skip=frozenset()):
        for field, item in list(vars(node).items()):
            if field in _SKIPPED_FIELDS or field in skip:
                continue
            if (
                isinstance(node, (MemberAccessNode, MethodCallNode))
                and field == "object"
            ):
                if isinstance(item, str):
                    continue
            if isinstance(item, str):
                if field in _TYPE_FIELDS or _QUALIFIED_NAME.fullmatch(item.strip()):
                    setattr(
                        node, field, self.resolve_text(item, context, _names(scopes))
                    )
            elif isinstance(item, (list, tuple)) or hasattr(item, "__dict__"):
                if isinstance(item, tuple):
                    continue
                self.resolve(item, context, scopes)

    def resolve_function(self, function, class_scope=frozenset()):
        context = tuple(getattr(function, "hlsl_namespace", ()) or ())
        params = list(getattr(function, "params", None) or [])
        function.return_type = self.resolve_text(function.return_type, context, set())
        scopes = [set(class_scope)]
        for param in params:
            if isinstance(param, VariableNode):
                param.vtype = self.resolve_text(param.vtype, context, set())
                if param.value is not None:
                    self.resolve(param.value, context, scopes)
                if isinstance(param.name, str):
                    scopes[0].add(param.name)
        self.resolve(getattr(function, "body", None), context, scopes)


def source_name(declaration):
    """The name ``declaration`` is spelled with in the HLSL source: ``ns::inner::twice`` once flattened."""
    return getattr(declaration, "hlsl_source_name", None) or declaration.name


def _names(scopes):
    names = set()
    for scope in scopes:
        names |= scope
    return names


def resolve_namespaces(ast, refuse):
    """Flatten the namespaced declarations of ``ast`` and resolve every reference to them, in place.

    ``refuse(reason)`` records an import refusal. Returns how many declarations were renamed.
    """
    resolver = _Namespaces(ast, refuse)
    namespaced = resolver.collect()
    # Walk even without namespaces: ``::g`` is spelled ``g`` in CrossGL.
    for function in getattr(ast, "functions", None) or []:
        if isinstance(function, FunctionNode):
            resolver.resolve_function(function)
    for struct in list(getattr(ast, "structs", None) or []) + list(
        getattr(ast, "cbuffers", None) or []
    ):
        context = tuple(getattr(struct, "hlsl_namespace", ()) or ())
        members = list(getattr(struct, "members", None) or [])
        methods = list(getattr(struct, "methods", None) or [])
        # A method finds its structure's members before the namespace's.
        class_scope = {
            item.name
            for item in members + methods
            if isinstance(getattr(item, "name", None), str)
        }
        for member in members + methods:
            if isinstance(member, VariableNode):
                member.vtype = resolver.resolve_text(member.vtype, context, set())
            elif isinstance(member, FunctionNode):
                member.hlsl_namespace = context
                resolver.resolve_function(member, class_scope)
    for collection in ("global_variables", "typedefs", "enums"):
        for declaration in getattr(ast, collection, None) or []:
            context = tuple(getattr(declaration, "hlsl_namespace", ()) or ())
            resolver.resolve(declaration, context, [set()])
    renamed = 0
    for _collection, declaration, path, name in namespaced:
        flat = resolver.table.get(path + (name,))
        if flat is not None and flat != name:
            declaration.hlsl_source_name = "::".join(path + (name,))
            declaration.name = flat
            renamed += 1
        if isinstance(declaration, StructNode):
            for member in getattr(declaration, "members", None) or []:
                if isinstance(member, FunctionNode):
                    member.hlsl_namespace = path
    return renamed
