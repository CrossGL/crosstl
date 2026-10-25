"""Function-scope ``static`` variables become module-scope private variables.

A ``static`` local keeps its value from one call of its function to the next within an invocation, and is
initialized once (HLSL "Variable Syntax", Storage_Class ``static``; DXC places it in a zero-initialized internal
global). Every generator emitted it as an ordinary local, so ``int next() { static int c = 0; return ++c; }``
returned 1 on every call. HLSL and CrossGL already carry module-scope ``static`` variables to each target (a GLSL
global, a WGSL ``var<private>``, a Metal ``thread`` variable passed to the functions that use it), so
:func:`hoist_static_locals` moves each mutable ``static`` local there under a fresh name and renames its uses in
the declaration's scope. A local without an initializer starts from zero, as in HLSL. An initializer that is not a
constant expression, and a structure or array without an initializer, are refused with
:class:`StaticLocalUnsupportedError`, because a module-scope initializer cannot read the function's values.
"""

import re

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import iter_functions
from .rewrite import child_slots, get_child

__all__ = ["StaticLocalUnsupportedError", "hoist_static_locals"]

_TYPE_CONSTRUCTOR = re.compile(
    r"(?:[biud]|f16|i16|u16|i64|u64)?(?:vec[234]|mat[234](?:x[234])?)"
    r"|float|double|half|int|uint|bool|float16_t|float32_t|float64_t"
    r"|int16_t|int32_t|int64_t|uint16_t|uint32_t|uint64_t"
)
_VECTOR_PREFIXES = {
    "float": "vec",
    "int": "ivec",
    "uint": "uvec",
    "bool": "bvec",
    "double": "dvec",
}
_MATRIX_PREFIXES = {"float": "mat", "double": "dmat"}
_ZERO_LITERALS = {
    "float": 0.0,
    "double": 0.0,
    "half": 0.0,
    "int": 0,
    "uint": 0,
    "bool": False,
}


class StaticLocalUnsupportedError(TranslationDiagnostic):
    """A function-scope ``static`` variable that cannot become a module-scope variable."""

    def __init__(self, message, target):
        super().__init__(
            message,
            target=target,
            code=f"{CODE_PREFIX}static-local-unsupported",
        )


def hoist_static_locals(program, target) -> int:
    """Move mutable function-scope ``static`` variables to module scope; returns how many moved."""
    if not isinstance(program, A.ShaderNode):
        return 0
    target = str(target or "")
    constants = _constant_names(program)
    taken = _declared_names(program)
    hoisted = []
    for function in iter_functions(program):
        body = getattr(function, "body", None)
        if body is None:
            continue
        for container, declaration in _static_locals(body):
            function_name = str(getattr(function, "name", "") or "function")
            context = f"static variable '{declaration.name}' of '{function_name}'"
            _check_initializer(declaration, constants, context, target)
            new_name = _fresh_name(f"{function_name}_{declaration.name}", taken)
            _Renamer(declaration, declaration.name, new_name).walk(body)
            container.remove(declaration)
            declaration.name = new_name
            hoisted.append(declaration)
    program.global_variables = list(getattr(program, "global_variables", None) or [])
    program.global_variables.extend(hoisted)
    return len(hoisted)


def _static_locals(body):
    """``(statement list, declaration)`` for each mutable ``static`` declaration, in source order."""
    found = []

    def visit(node):
        for field, key in child_slots(node):
            child = get_child(node, field, key)
            if key is not None and _is_static_local(child):
                found.append((getattr(node, field), child))
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


def _is_static_local(node):
    if not isinstance(node, A.VariableNode):
        return False
    qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", None) or []}
    return "static" in qualifiers and "const" not in qualifiers


def _check_initializer(declaration, constants, context, target):
    value = getattr(declaration, "initial_value", None)
    if value is None:
        zero = _zero_value(getattr(declaration, "var_type", None))
        if zero is None:
            raise StaticLocalUnsupportedError(
                f"The {context} has no initializer, and its type "
                f"'{_type_name(declaration.var_type)}' is not a scalar, vector or matrix to start "
                "from zero",
                target,
            )
        declaration.initial_value = zero
        return
    if not _is_constant(value, constants):
        raise StaticLocalUnsupportedError(
            f"The initializer of the {context} is not a constant expression, so it cannot be "
            "evaluated once at module scope",
            target,
        )


def _is_constant(node, constants):
    if isinstance(node, A.LiteralNode):
        return True
    if isinstance(node, A.IdentifierNode):
        return node.name in constants
    if isinstance(node, A.FunctionCallNode):
        callee = getattr(node, "function", None)
        name = getattr(callee, "name", callee)
        if not (isinstance(name, str) and _TYPE_CONSTRUCTOR.fullmatch(name)):
            return False
        return all(_is_constant(arg, constants) for arg in node.arguments or [])
    if isinstance(node, A.ConstructorNode):
        return all(_is_constant(arg, constants) for arg in node.arguments or [])
    if isinstance(node, A.ArrayLiteralNode):
        return all(_is_constant(item, constants) for item in node.elements or [])
    if isinstance(node, A.UnaryOpNode):
        return node.operator in {"-", "+", "!", "~"} and _is_constant(
            node.operand, constants
        )
    if isinstance(node, A.BinaryOpNode):
        return _is_constant(node.left, constants) and _is_constant(
            node.right, constants
        )
    if isinstance(node, A.TernaryOpNode):
        return all(
            _is_constant(part, constants)
            for part in (node.condition, node.true_expr, node.false_expr)
        )
    if isinstance(node, A.CastNode):
        return _is_constant(node.expression, constants)
    return False


class _Renamer:
    """Renames the identifiers that refer to one declaration, from the declaration to the end of its scope.

    A later declaration of the same name in a nested block, or in a ``for`` header, hides it there.
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
                active = True
                continue
            if isinstance(item, A.ASTNode):
                self._node(item, active)
                if isinstance(item, A.VariableNode) and item.name == self.old:
                    active = False
        return active

    def _node(self, node, active):
        if isinstance(node, A.IdentifierNode):
            if active and node.name == self.old:
                node.name = self.new
                node.identifier = self.new
            return
        fields = {}
        for field, key in child_slots(node):
            fields.setdefault(field, []).append(key)
        for field, keys in fields.items():
            value = getattr(node, field)
            if isinstance(value, list):
                # A statement list or a for header's declarations: later items see earlier declarations.
                active_after = self._sequence(
                    [value[key] for key in keys if isinstance(key, int)], active
                )
                if isinstance(node, A.ForNode) and field == "init":
                    active = active_after
            else:
                child = get_child(node, field, None)
                self._node(child, active)
                if isinstance(child, A.VariableNode) and child.name == self.old:
                    if isinstance(node, A.ForNode) and field == "init":
                        active = False


def _constant_names(program):
    names = {
        getattr(constant, "name", None)
        for constant in getattr(program, "constants", None) or []
    }
    for variable in getattr(program, "global_variables", None) or []:
        qualifiers = {
            str(q).lower() for q in getattr(variable, "qualifiers", None) or []
        }
        if "const" in qualifiers or "constexpr" in qualifiers:
            names.add(getattr(variable, "name", None))
    names.discard(None)
    return names


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


def _zero_value(type_node):
    """A zero of a scalar, vector or matrix type, or None for other types."""
    if isinstance(type_node, A.PrimitiveType):
        return _zero_literal(type_node.name)
    element = getattr(getattr(type_node, "element_type", None), "name", None)
    if isinstance(type_node, A.VectorType):
        prefix = _VECTOR_PREFIXES.get(element)
        spelling = None if prefix is None else f"{prefix}{type_node.size}"
    elif isinstance(type_node, A.MatrixType):
        prefix = _MATRIX_PREFIXES.get(element)
        shape = (
            f"{type_node.cols}"
            if type_node.rows == type_node.cols
            else f"{type_node.cols}x{type_node.rows}"
        )
        spelling = None if prefix is None else f"{prefix}{shape}"
    else:
        return None
    literal = _zero_literal(element)
    if spelling is None or literal is None:
        return None
    # A one-scalar vector constructor fills every component and a matrix one its diagonal: all zero here.
    return A.FunctionCallNode(A.IdentifierNode(spelling), [literal])


def _zero_literal(type_name):
    if type_name not in _ZERO_LITERALS:
        return None
    return A.LiteralNode(_ZERO_LITERALS[type_name], A.PrimitiveType(type_name))


def _type_name(type_node):
    name = getattr(type_node, "name", None)
    return name if isinstance(name, str) else (str(type_node) if type_node else None)
