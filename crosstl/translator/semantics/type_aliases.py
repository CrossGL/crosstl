"""Type aliases: every use of a ``typedef`` name becomes the type it names.

CrossGL reads ``typedef float real;`` as a declaration (a :class:`~crosstl.translator.ast.VariableNode` with
``is_type_alias``) at program scope, in a stage, or in a block. An alias has no run-time state and no identity of
its own: a value of type ``real`` is a ``float``. :func:`resolve_type_aliases` therefore writes the aliased type in
place of the alias wherever it names a type (declarations, parameters, return types, structure members, generic
arguments), turns a constructor or cast through the alias (``real(x)``, ``(real)x``) into one through the aliased
type, and drops the alias declarations. An alias of an alias resolves to the type at the end of the chain.

The MSL, HLSL and GLSL generators declare a program-scope alias themselves, so for those targets the program's
aliases are kept and only those of stages and blocks are resolved; every other generator then reads no alias at all.
A block alias is visible from its declaration to the end of its block, and a variable, parameter or structure of the
same name hides it, as in C. Two forms cannot be resolved faithfully and are refused with
:class:`TypeAliasUnsupportedError`: an alias carrying a qualifier or attribute (``typedef const float cf;``), whose
meaning would have to move onto each declaration, and a constructor call through an alias of a type CrossGL has no
constructor spelling for (a pointer, a resource, a generic type).
"""

from __future__ import annotations

from copy import deepcopy

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = ["TypeAliasUnsupportedError", "resolve_type_aliases", "type_constructor_name"]

# Targets whose generators declare program-scope aliases themselves.
NATIVE_PROGRAM_ALIAS_TARGETS = frozenset(
    {
        "metal",
        "directx",
        "hlsl",
        "dx11",
        "dx12",
        "d3d11",
        "d3d12",
        "opengl",
        "glsl",
        "webgl",
        "gles",
        "essl",
    }
)
# CrossGL output keeps the program as written.
_CARRYING = frozenset({"cgl", "crossgl", ""})
_SKIPPED_FIELDS = frozenset({"parent", "annotations", "source_location"})
# String-valued fields that hold a type's name rather than a declared name.
_TYPE_NAME_FIELDS = frozenset(
    {
        "var_type",
        "param_type",
        "return_type",
        "const_type",
        "member_type",
        "constructor_type",
        "type_name",
        "target_type",
        "element_type",
        "vtype",
    }
)
# Calls whose first argument names a type.
_TYPE_OPERAND_CALLS = frozenset({"sizeof", "alignof", "metal_sizeof"})
_VECTOR_PREFIX = {"float": "", "int": "i", "uint": "u", "bool": "b", "double": "d"}
_MATRIX_PREFIX = {"float": "", "double": "d"}


class TypeAliasUnsupportedError(TranslationDiagnostic):
    """A type alias whose uses cannot be rewritten as the type it names."""

    def __init__(self, message, target):
        super().__init__(
            message,
            target=target,
            code=f"{CODE_PREFIX}type-alias-unsupported",
        )


def type_constructor_name(type_value):
    """The CrossGL name a constructor call of ``type_value`` takes (``float``, ``vec3``, ``Pair``, ``float[4]``).

    Returns ``None`` for a type CrossGL constructs under no plain name: a pointer, a reference, a function type or
    a type with generic arguments.
    """
    if isinstance(type_value, str):
        return type_value.strip() or None
    if isinstance(type_value, A.PrimitiveType):
        return type_value.name
    if isinstance(type_value, A.NamedType):
        return None if type_value.generic_args else type_value.name
    if isinstance(type_value, A.VectorType):
        element = _scalar_name(type_value.element_type)
        prefix = _VECTOR_PREFIX.get(element)
        if prefix is None or not isinstance(type_value.size, int):
            return None
        return f"{prefix}vec{type_value.size}"
    if isinstance(type_value, A.MatrixType):
        element = _scalar_name(type_value.element_type)
        prefix = _MATRIX_PREFIX.get(element)
        if prefix is None:
            return None
        cols, rows = type_value.cols, type_value.rows
        shape = f"{cols}" if cols == rows else f"{cols}x{rows}"
        return f"{prefix}mat{shape}"
    if isinstance(type_value, A.ArrayType):
        element = type_constructor_name(type_value.element_type)
        size = _array_size(type_value.size)
        if element is None or size is None:
            return None
        return f"{element}[{size}]"
    return None


def _scalar_name(type_value):
    if isinstance(type_value, A.PrimitiveType):
        return type_value.name
    if isinstance(type_value, str):
        return type_value
    return None


def _array_size(size):
    if size is None:
        return ""
    if isinstance(size, int) and not isinstance(size, bool):
        return str(size)
    if isinstance(size, A.LiteralNode):
        value = size.value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, str) and value.isdigit():
            return value
        return None
    if isinstance(size, A.IdentifierNode) and isinstance(size.name, str):
        return size.name
    return None


def _is_alias(node):
    return isinstance(node, A.VariableNode) and bool(
        getattr(node, "is_type_alias", False)
    )


def _alias_name(type_value):
    """The name a type node refers to when it could be an alias: a plain name without generic arguments."""
    if isinstance(type_value, A.NamedType) and not type_value.generic_args:
        return type_value.name
    return None


def _callee_name(call):
    function = call.function
    if isinstance(function, str):
        return function
    if isinstance(function, A.IdentifierNode) and isinstance(function.name, str):
        return function.name
    return None


def _children(node):
    """``(field, value)`` for each field of ``node`` holding a node or a list of nodes."""
    for field, value in list(vars(node).items()):
        if field in _SKIPPED_FIELDS:
            continue
        if isinstance(value, (A.ASTNode, list)):
            yield field, value
        elif isinstance(value, dict) and not isinstance(value, A.StageMap):
            yield field, value


# Statement lists: an alias or a declaration in one changes the scope of the items after it.
_STATEMENT_FIELDS = frozenset({"statements", "body", "init"})
# Fields of a function or ``for`` loop where a parameter or header declaration hides an alias.
_BODY_FIELDS = frozenset({"body", "condition", "update", "increment"})


class _Resolver:
    """Walks the program once, each node under the aliases in scope there (name to resolved type)."""

    def __init__(self, target):
        self.target = target
        self.visited = set()
        self.dropped = 0

    def declare(self, alias, env):
        """Resolve ``alias``'s own type under ``env``; returns the scope that follows the declaration."""
        self.visited.add(id(alias))
        if (
            alias.qualifiers
            or alias.attributes
            or getattr(alias, "resource_qualifiers", None)
        ):
            raise TypeAliasUnsupportedError(
                f"type alias '{alias.name}' carries a qualifier or attribute, which would have to move onto every "
                "declaration that names it",
                self.target,
            )
        aliased = self.type_replacement(alias.var_type, env)
        if aliased is None:
            aliased = alias.var_type
            self.walk_fields(alias, env)
        self.dropped += 1
        scope = {name: value for name, value in env.items() if name != alias.name}
        if _alias_name(aliased) != alias.name:
            # ``typedef struct S S;`` names the structure itself and changes nothing.
            scope[alias.name] = aliased
        return scope

    def declarations(self, items, env):
        """Resolve and remove the aliases of a declaration list, in order; returns the scope after them."""
        kept = []
        for item in items:
            if _is_alias(item):
                env = self.declare(item, env)
            else:
                kept.append(item)
        items[:] = kept
        return env

    def node(self, node, env):
        if not isinstance(node, A.ASTNode) or id(node) in self.visited:
            return
        self.visited.add(id(node))
        self.walk_fields(node, env)

    def walk_fields(self, node, env):
        if env:
            self.rename_strings(node, env)
            if isinstance(node, A.FunctionCallNode):
                self.call(node, env)
        body_env = self.hidden_by_header(node, env)
        for field, value in _children(node):
            scope = body_env if field in _BODY_FIELDS else env
            if isinstance(value, list):
                self.sequence(node, field, value, scope)
            elif isinstance(value, dict):
                for key, item in list(value.items()):
                    replacement = self.type_replacement(item, scope)
                    if replacement is not None:
                        value[key] = replacement
                    else:
                        self.node(item, scope)
            else:
                replacement = self.type_replacement(value, scope)
                if replacement is not None:
                    self.replace(node, value, replacement)
                else:
                    self.node(value, scope)

    @staticmethod
    def hidden_by_header(node, env):
        """The scope of a function or ``for`` body: without the names its parameters or header declare."""
        if not env:
            return env
        declared = []
        if isinstance(node, A.FunctionNode):
            declared = list(getattr(node, "parameters", None) or [])
        elif isinstance(node, A.ForNode):
            init = getattr(node, "init", None)
            declared = init if isinstance(init, list) else [init]
        names = {
            item.name
            for item in declared
            if isinstance(item, (A.VariableNode, A.ParameterNode))
            and not _is_alias(item)
            and item.name in env
        }
        if not names:
            return env
        return {name: value for name, value in env.items() if name not in names}

    def sequence(self, owner, field, items, env):
        """Visit a list in order; in a statement list, aliases and hiding declarations scope the items after them."""
        statements = field in _STATEMENT_FIELDS
        scope = env
        kept = []
        for item in items:
            if statements and _is_alias(item):
                scope = self.declare(item, scope)
                continue
            replacement = self.type_replacement(item, scope)
            if replacement is not None:
                self.replace(owner, item, replacement, lists=False)
                kept.append(replacement)
                continue
            kept.append(item)
            self.node(item, scope)
            if statements and self.hides(item, scope):
                scope = {
                    name: value for name, value in scope.items() if name != item.name
                }
        if len(kept) != len(items) or any(a is not b for a, b in zip(kept, items)):
            items[:] = kept

    @staticmethod
    def hides(item, env):
        return (
            isinstance(item, (A.VariableNode, A.StructNode))
            and not _is_alias(item)
            and getattr(item, "name", None) in env
        )

    @staticmethod
    def type_replacement(value, env):
        name = _alias_name(value)
        if name is None or name not in env:
            return None
        return deepcopy(env[name])

    @staticmethod
    def replace(node, old, new, lists=True):
        """Replace ``old`` in every field of ``node`` that holds it, legacy alias fields included."""
        for field, value in list(vars(node).items()):
            if field in _SKIPPED_FIELDS:
                continue
            if value is old:
                setattr(node, field, new)
            elif lists and isinstance(value, list):
                for position, item in enumerate(value):
                    if item is old:
                        value[position] = new

    def rename_strings(self, node, env):
        for field in _TYPE_NAME_FIELDS:
            value = getattr(node, field, None)
            if isinstance(value, str) and value in env:
                spelled = type_constructor_name(env[value])
                setattr(
                    node,
                    field,
                    spelled if spelled is not None else deepcopy(env[value]),
                )

    def call(self, node, env):
        name = _callee_name(node)
        if name in env:
            spelled = self.constructor(name, env)
            if isinstance(node.function, A.IdentifierNode):
                node.function.name = spelled
                node.function.identifier = spelled
            else:
                for field in ("function", "name"):
                    if getattr(node, field, None) == name:
                        setattr(node, field, spelled)
        if name in _TYPE_OPERAND_CALLS and node.arguments:
            operand = node.arguments[0]
            if isinstance(operand, A.IdentifierNode) and operand.name in env:
                spelled = self.constructor(operand.name, env)
                operand.name = spelled
                operand.identifier = spelled

    def constructor(self, name, env):
        spelled = type_constructor_name(env[name])
        if spelled is None:
            raise TypeAliasUnsupportedError(
                f"type alias '{name}' names a type CrossGL has no constructor name for, so '{name}(...)' cannot be "
                "written through the aliased type",
                self.target,
            )
        return spelled


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values()) if isinstance(stages, dict) else list(stages)


def _declares_alias(program):
    seen = set()
    pending = [program, *_stages(program)]
    while pending:
        node = pending.pop()
        if not isinstance(node, A.ASTNode) or id(node) in seen:
            continue
        seen.add(id(node))
        if _is_alias(node):
            return True
        for _, value in _children(node):
            if isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
            else:
                pending.append(value)
    return False


def _resolve_alias_chains(resolver, variables):
    """Keep program-scope aliases, each naming the type at the end of its chain (``typedef real3 point;``)."""
    chain = {}
    for alias in variables:
        if not _is_alias(alias):
            continue
        if alias.qualifiers or alias.attributes:
            # The generator declares a qualified alias as written.
            resolver.visited.add(id(alias))
            continue
        dropped = resolver.dropped
        chain = resolver.declare(alias, chain)
        resolver.dropped = dropped
        aliased = chain.get(alias.name)
        if aliased is not None and aliased is not alias.var_type:
            resolver.replace(alias, alias.var_type, aliased)


def resolve_type_aliases(program, target) -> int:
    """Write the aliased type for every use of a type alias and drop the aliases; returns how many were dropped.

    For the MSL, HLSL and GLSL targets the program-scope aliases stay for their generators to declare.
    """
    if not isinstance(program, A.ShaderNode):
        return 0
    target = str(target or "").lower()
    if target in _CARRYING or not _declares_alias(program):
        return 0
    resolver = _Resolver(target)
    variables = getattr(program, "global_variables", None)
    env = {}
    if target in NATIVE_PROGRAM_ALIAS_TARGETS:
        _resolve_alias_chains(resolver, variables or [])
    elif variables:
        env = resolver.declarations(variables, env)
    resolver.node(program, env)
    for stage in _stages(program):
        stage_env = resolver.declarations(stage.local_variables, env)
        resolver.node(stage, stage_env)
    return resolver.dropped
