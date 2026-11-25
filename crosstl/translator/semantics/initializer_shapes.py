"""Declarations whose initializer is a vector or matrix of a different shape.

CrossGL follows GLSL here: a declaration's initializer has the declared type, or converts to it implicitly, and
GLSL 4.60 section 4.1.10 converts only between component types of the same shape. ``vec4 m = a;`` with ``vec3 a``
and ``mat3x4 m = a;`` with ``mat4x3 a`` are invalid. CrossTL used to translate them anyway: the Vulkan generator
wrote an ``OpStore`` of a value whose type is not the pointer's, which ``spirv-val`` rejects, and the text
generators copied the declaration, which their own compilers reject.

:func:`refuse_initializer_shape_mismatches` refuses such a declaration in a CrossGL program with
:class:`InitializerShapeMismatchError`, which names the variable, its type and the initializer's type. Only a
vector or matrix initializer of another vector or matrix shape is refused. A scalar initializer (``vec4 v = 1.0;``)
is a separate question and is left alone, as is an initializer whose type the pass cannot tell. Programs imported
from other languages are left alone too: HLSL truncates a wider vector implicitly, and the importers carry that.

Declarations are checked in function bodies, at module scope (constants and global variables) and at stage
scope. An initializer that is matrix arithmetic has the shape GLSL 4.60 section 5.10 gives it: ``matCxR * vecC``
is ``vecR``, ``vecR * matCxR`` is ``vecC``, ``matC1xR1 * matC2xC1`` is ``matC2xR1``, and matrix ``+``, ``-``,
``/``, unary minus and scaling by a scalar keep the matrix's shape. Further arithmetic keeps that shape: unary
``-`` and ``+`` keep the operand's shape, a ternary has its branches' shape when both agree, and ``+``, ``-``,
``*`` and ``/`` with a scalar, or between two vectors with the same number of components, keep the vector's
shape, so ``vec4 a = m43 * v4 + v3;`` is refused too.

The shape of a call is known only for a constructor, a DXR transform call and a function of the program. Other
calls are left alone, and so is an expression that contains one: a texel read into a narrower vector
(``vec2 t = imageLoad(rg, p);``) takes the leading components, which CrossGL allows.

The DXR transform calls have the types :mod:`.dxr_ray_interfaces` gives them: ``ObjectToWorld3x4()`` and
``ObjectToWorld()`` are ``mat4x3`` and ``ObjectToWorld4x3()`` is ``mat3x4``, and the same for ``WorldToObject``.
A zero-parameter function the program defines under one of those names has its own return type
(:func:`.dxr_ray_interfaces.defined_function_names`).
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from ..matrix_shape import crossgl_matrix_shape, crossgl_matrix_spelling
from .dxr_ray_interfaces import defined_function_names
from .hub_types import HubType, call_name, type_from_name, type_from_node
from .rewrite import child_slots, get_child
from .scoped_walk import ScopedWalker, program_stages

__all__ = ["InitializerShapeMismatchError", "refuse_initializer_shape_mismatches"]

# Source languages whose declarations are CrossGL's own.
_CROSSGL_SOURCES = frozenset({"cgl", "crossgl"})
# DXR transform call -> its CrossGL matrix type (columns, rows).
_DXR_TRANSFORMS = {
    "ObjectToWorld": (4, 3),
    "ObjectToWorld3x4": (4, 3),
    "ObjectToWorld4x3": (3, 4),
    "WorldToObject": (4, 3),
    "WorldToObject3x4": (4, 3),
    "WorldToObject4x3": (3, 4),
}
_VECTOR_PREFIXES = {"i": "ivec", "u": "uvec", "b": "bvec"}
_FLOAT_VECTOR_PREFIXES = {16: "f16vec", 32: "vec", 64: "dvec"}


class InitializerShapeMismatchError(TranslationDiagnostic):
    """A declaration's initializer is a vector or matrix of a different shape from the declared type."""

    # The program is invalid CrossGL whatever the target, so the code names no target.
    code = CODE_PREFIX + "initializer-shape-mismatch"

    def __init__(self, target, name, declared, initializer, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = ()
        self.name = name
        self.declared_type = declared
        self.initializer_type = initializer
        super().__init__(
            f"declaration of '{name}' has type '{declared}' but its initializer has type '{initializer}'; "
            "a vector or matrix initializer must have the declared shape",
            target=target,
        )


def refuse_initializer_shape_mismatches(program, source, target) -> None:
    """Raise :class:`InitializerShapeMismatchError` for the first mismatched declaration of a CrossGL program."""
    if not isinstance(program, A.ShaderNode):
        return
    if str(source or "").lower() not in _CROSSGL_SOURCES:
        return
    _ShapeWalker(program, target).run_program()


class _ShapeWalker(ScopedWalker):
    def __init__(self, program, target):
        super().__init__(program)
        self.target = target
        self.defined = defined_function_names(program)

    def run_program(self):
        # Module-scope constants and globals, then each stage's own variables, then the function bodies.
        for constant in getattr(self.program, "constants", None) or []:
            if isinstance(constant, A.ConstantNode) and constant.value is not None:
                self.check_declaration(constant, constant.const_type, constant.value)
        for variable in getattr(self.program, "global_variables", None) or []:
            self.check_variable(variable)
        for stage in program_stages(self.program):
            self.scope.push()
            for variable in getattr(stage, "local_variables", None) or []:
                self.check_variable(variable)
                if getattr(variable, "name", None) is not None:
                    self.scope.declare(variable.name, variable.var_type)
                    self.scope.declare_block_members(variable)
            self.scope.pop()
        super().run_program()

    def _walk(self, node):
        self.check_variable(node)
        super()._walk(node)

    def check_variable(self, node):
        if isinstance(node, A.VariableNode) and node.initial_value is not None:
            self.check_declaration(node, node.var_type, node.initial_value)

    def visit(self, node):
        return

    def check_declaration(self, node, declared_type, initial_value):
        declared = _type_shape(declared_type)
        if declared is None or declared[0] == "scalar":
            return
        initializer = self.value_shape(initial_value)
        if initializer is None or initializer[0] == "scalar":
            return
        if declared[:-1] == initializer[:-1]:
            return
        raise InitializerShapeMismatchError(
            self.target,
            node.name,
            declared[-1],
            initializer[-1],
            source_location=getattr(node, "source_location", None),
        )

    def value_shape(self, expr):
        """``("vector", lanes, spelling)``, ``("matrix", columns, rows, spelling)``, ``("scalar", ...)`` or None."""
        if isinstance(expr, (A.IdentifierNode, A.MemberAccessNode)):
            declared = self.scope.node_type(expr)
            if declared is not None:
                return _type_shape(declared)
        if isinstance(expr, A.ConstructorNode):
            return _type_shape(expr.constructor_type)
        if isinstance(expr, A.CastNode):
            return _type_shape(expr.target_type)
        if isinstance(expr, A.FunctionCallNode):
            name = call_name(expr)
            transform = _DXR_TRANSFORMS.get(name)
            if (
                transform is not None
                and not expr.arguments
                and name not in self.defined
            ):
                return _matrix_shape("float", *transform)
            if name in self.scope.functions:
                returned = self.scope.functions[name]
                return _type_shape(returned) if returned is not None else None
            matrix = crossgl_matrix_shape(name)
            if matrix is not None:
                return _matrix_shape(*matrix)
            return _hub_shape(type_from_name(name))
        arithmetic = self.arithmetic_shape(expr)
        if arithmetic is not None:
            return arithmetic
        if _contains_call(expr):
            return None
        return _hub_shape(self.scope.infer(expr))

    def arithmetic_shape(self, expr):
        """The shape of vector or matrix arithmetic (GLSL 4.60 section 5.10), or None.

        Vector operands combine too, so a matrix product inside further arithmetic keeps its shape:
        ``-(m43 * v4)``, ``(m43 * v4) * 2.0``, ``m43 * v4 + v3`` and ``flag ? m43 * v4 : v3`` are ``vec3``.
        """
        if isinstance(expr, A.UnaryOpNode) and expr.operator in {"-", "+"}:
            operand = self.value_shape(expr.operand)
            return operand if _is_vector_or_matrix(operand) else None
        if isinstance(expr, A.TernaryOpNode):
            branches = self.value_shape(expr.true_expr), self.value_shape(
                expr.false_expr
            )
            if (
                _is_vector_or_matrix(branches[0])
                and branches[0][:-1] == (branches[1] or ())[:-1]
            ):
                return branches[0]
            return None
        if not isinstance(expr, A.BinaryOpNode) or expr.operator not in {
            "+",
            "-",
            "*",
            "/",
        }:
            return None
        left, right = self.value_shape(expr.left), self.value_shape(expr.right)
        if left is None or right is None:
            return None
        if left[0] == "scalar" and right[0] == "scalar":
            return None
        if left[0] == "scalar":
            return right
        if right[0] == "scalar":
            return left
        if not (_is_matrix(left) or _is_matrix(right)):
            # Component-wise vector arithmetic needs two vectors with the same number of components.
            return left if left[:-1] == right[:-1] else None
        if expr.operator != "*":
            # Component-wise matrix arithmetic needs two matrices of one shape.
            return left if left[:-1] == right[:-1] else None
        if _is_matrix(left) and _is_matrix(right):
            # matC1xR1 * matC2xC1 is matC2xR1.
            if left[1] != right[2]:
                return None
            return (
                "matrix",
                right[1],
                left[2],
                _matrix_spelling_like(left, right[1], left[2]),
            )
        if _is_matrix(left):
            # matCxR * vecC is vecR.
            if left[1] != right[1]:
                return None
            return ("vector", left[2], _vector_spelling_like(left, left[2]))
        # vecR * matCxR is vecC.
        if right[2] != left[1]:
            return None
        return ("vector", right[1], _vector_spelling_like(right, right[1]))


def _is_matrix(shape):
    return shape is not None and shape[0] == "matrix"


def _is_vector_or_matrix(shape):
    return shape is not None and shape[0] in {"vector", "matrix"}


def _matrix_spelling_like(matrix, columns, rows):
    scalar = "double" if matrix[-1].startswith("d") else "float"
    return crossgl_matrix_spelling(columns, rows, scalar)


def _vector_spelling_like(matrix, lanes):
    return f"{'d' if matrix[-1].startswith('d') else ''}vec{lanes}"


def _contains_call(node):
    if isinstance(node, A.FunctionCallNode):
        return True
    return any(
        _contains_call(get_child(node, field, key)) for field, key in child_slots(node)
    )


def _type_shape(type_node):
    if isinstance(type_node, A.MatrixType):
        scalar = getattr(type_node.element_type, "name", "float")
        return _matrix_shape(scalar, type_node.cols, type_node.rows)
    if isinstance(type_node, (A.NamedType, A.PrimitiveType)) and not getattr(
        type_node, "generic_args", None
    ):
        matrix = crossgl_matrix_shape(type_node.name)
        if matrix is not None:
            return _matrix_shape(*matrix)
        return _hub_shape(type_from_name(type_node.name))
    return _hub_shape(type_from_node(type_node))


def _matrix_shape(scalar, columns, rows):
    columns, rows = int(columns), int(rows)
    return ("matrix", columns, rows, crossgl_matrix_spelling(columns, rows, scalar))


def _hub_shape(hub: HubType | None):
    if hub is None:
        return None
    if hub.lanes == 1:
        return ("scalar", 1, "scalar")
    if hub.kind == "f":
        prefix = _FLOAT_VECTOR_PREFIXES.get(hub.bits, "vec")
    else:
        prefix = _VECTOR_PREFIXES[hub.kind]
    return ("vector", hub.lanes, f"{prefix}{hub.lanes}")
