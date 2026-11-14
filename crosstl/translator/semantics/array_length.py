"""``x.length()`` of a sized array, a vector or a matrix becomes its constant count.

GLSL's ``length()`` method on a sized array is a constant integral expression (GLSL 4.60 section 5.9). On a vector
it is the number of components and on a matrix the number of columns (sections 5.5 and 5.6); glslang folds all three
to a constant without evaluating the object. HLSL, MSL, SPIR-V and the other targets have no such method: the
emitters printed the call unchanged, which DXC and the Metal compiler reject ("member reference base type 'const
float [25]' is not a structure or union", "illegal vector component name 'l'"), or refused it as a call through a
callee expression. This pass replaces the call, for every target but GLSL, with an ``int`` literal of that count.

The object's type is found with lexical scopes (:class:`.hub_types.TypeScope`): globals, constants, stage variables,
parameters and block-scoped locals, through element and member access; a vector's width is inferred from any
expression whose value type is known (a swizzle, a call, arithmetic). An array size that is a literal or a constant
with a literal value is used.

A specialization constant's literal is only its default: the host replaces it when it builds the pipeline, and
``length()`` of an array it sizes is that specialized count (GL_KHR_vulkan_glsl, "Specialization Constants"; an
``OpTypeArray`` whose length is an ``OpSpecConstant`` in SPIR-V). So an array sized by an expression over program
constants, specialization constants among them, has its ``length()`` replaced by ``int(<that expression>)``, which
every target evaluates with the specialized values. The expression is copied to the call only when each name in it
still means the program constant there; where a local, parameter or member shadows one, the call is refused with
:class:`ArrayLengthUnresolvedError`. Any other ``length()`` (a runtime-sized array, a buffer) is left for the
emitters, which lower or refuse it.
"""

from __future__ import annotations

import copy

from .. import ast as A
from ..codegen.specialization_constants import declares_specialization_id
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import TypeScope
from .rewrite import child_slots, get_child, set_child

__all__ = ["ArrayLengthUnresolvedError", "fold_array_lengths"]

# Targets whose arrays have a native ``length()`` method.
_NATIVE = frozenset({"opengl", "glsl", "webgl", "essl", "gles", "cgl", "crossgl"})
_SCOPES = (A.BlockNode, A.ForNode, A.ForInNode, A.FunctionNode)
# Operators a constant integral expression may apply; none has a side effect.
_CONSTANT_BINARY_OPERATORS = frozenset(
    {
        "+",
        "-",
        "*",
        "/",
        "%",
        "<<",
        ">>",
        "&",
        "|",
        "^",
        "&&",
        "||",
        "==",
        "!=",
        "<",
        ">",
        "<=",
        ">=",
    }
)
_CONSTANT_UNARY_OPERATORS = frozenset({"+", "-", "~", "!"})


class ArrayLengthUnresolvedError(TranslationDiagnostic):
    """``length()`` of an array whose size names a program constant that a declaration shadows at the call."""

    def __init__(self, target, name):
        self.missing_capabilities = (f"{target}.array-length-shadowed-size",)
        super().__init__(
            f"length() of an array sized by '{name}' is called where another declaration named '{name}' hides "
            "the program constant, so the count cannot be written there; rename the declaration",
            target=target,
            code=f"{CODE_PREFIX}{target}-array-length-shadowed-size",
        )


def _literal_int(node):
    if isinstance(node, A.LiteralNode) and not isinstance(node.value, bool):
        if isinstance(node.value, int):
            return node.value
        text = str(node.value).strip().rstrip("uU")
        if text.isdigit():
            return int(text)
    return None


class _Folder:
    def __init__(self, program, target):
        self.program = program
        self.target = target
        self.scope = TypeScope(program)
        self.constants = {}
        self.program_constant_types = {}
        self.folded = 0

    def size(self, array_type):
        if not isinstance(array_type, A.ArrayType) or array_type.size is None:
            return None
        size = array_type.size
        if isinstance(size, int) and not isinstance(size, bool):
            return size
        value = _literal_int(size)
        if value is None and isinstance(size, A.IdentifierNode):
            value = self.constants.get(size.name)
        return value

    def symbolic_size(self, array_type):
        """``int(size)`` for an array sized by an expression over program constants visible here, else ``None``."""
        if not isinstance(array_type, A.ArrayType) or array_type.size is None:
            return None
        size = array_type.size
        if isinstance(size, str):
            size = A.IdentifierNode(size)
        names = self.constant_names(size)
        if names is None:
            return None
        for name in sorted(names):
            if self.scope.lookup(name) is not self.program_constant_types[name]:
                raise ArrayLengthUnresolvedError(self.target, name)
        return A.FunctionCallNode(A.IdentifierNode("int"), [copy.deepcopy(size)])

    def constant_names(self, node):
        """The program constants ``node`` reads when it reads nothing but literals and those, else ``None``."""
        if isinstance(node, A.LiteralNode):
            return None if isinstance(node.value, bool) else set()
        if isinstance(node, A.IdentifierNode):
            return {node.name} if node.name in self.program_constant_types else None
        if isinstance(node, A.BinaryOpNode):
            if str(node.operator) not in _CONSTANT_BINARY_OPERATORS:
                return None
            parts = (node.left, node.right)
        elif isinstance(node, A.UnaryOpNode):
            if node.is_postfix or str(node.operator) not in _CONSTANT_UNARY_OPERATORS:
                return None
            parts = (node.operand,)
        elif isinstance(node, A.TernaryOpNode):
            parts = (node.condition, node.true_expr, node.false_expr)
        else:
            return None
        names = set()
        for part in parts:
            found = self.constant_names(part)
            if found is None:
                return None
            names |= found
        return names

    def run(self):
        for constant in getattr(self.program, "constants", None) or []:
            self.scope.declare(constant.name, constant.const_type)
            if constant.const_type is not None:
                self.program_constant_types[constant.name] = constant.const_type
            if declares_specialization_id(constant):
                # The host replaces the default when it builds the pipeline, so the count is not this literal.
                continue
            value = _literal_int(getattr(constant, "value", None))
            if value is not None:
                self.constants[constant.name] = value
        for cbuffer in getattr(self.program, "cbuffers", None) or []:
            self.scope.declare_cbuffer(cbuffer)
        self.declare_variables(getattr(self.program, "global_variables", None) or [])
        for function in getattr(self.program, "functions", None) or []:
            self.visit(function)
        stages = getattr(self.program, "stages", None) or {}
        for stage in stages.values() if isinstance(stages, dict) else stages:
            self.scope.push()
            for cbuffer in getattr(stage, "local_cbuffers", None) or []:
                self.scope.declare_cbuffer(cbuffer)
            self.declare_variables(getattr(stage, "local_variables", None) or [])
            for function in getattr(stage, "local_functions", None) or []:
                self.visit(function)
            if getattr(stage, "entry_point", None) is not None:
                self.visit(stage.entry_point)
            self.scope.pop()
        return self.folded

    def declare_variables(self, variables):
        for variable in variables:
            if isinstance(variable, A.VariableNode):
                self.scope.declare(variable.name, variable.var_type)
                if "const" in {
                    str(q) for q in variable.qualifiers or []
                } and not declares_specialization_id(variable):
                    value = _literal_int(getattr(variable, "initial_value", None))
                    if value is not None:
                        self.constants[variable.name] = value

    def length(self, node):
        """The count ``node`` (a ``length()`` call) evaluates to: an ``int``, an ``int(size)`` call, or ``None``."""
        if not isinstance(node, A.FunctionCallNode) or node.arguments:
            return None
        callee = node.function
        if not (isinstance(callee, A.MemberAccessNode) and callee.member == "length"):
            return None
        receiver = callee.object_expr
        declared = self.scope.node_type(receiver)
        if isinstance(declared, A.ArrayType):
            count = self.size(declared)
            return count if count is not None else self.symbolic_size(declared)
        if isinstance(declared, A.MatrixType):
            # A matrix's length() is its column count (GLSL 4.60 section 5.6).
            return int(declared.cols) if str(declared.cols).isdigit() else None
        value = self.scope.infer(receiver)
        if value is not None and value.lanes > 1:
            # A vector's length() is its component count (GLSL 4.60 section 5.5).
            return value.lanes
        return None

    def visit(self, node):
        scoped = isinstance(node, _SCOPES)
        if scoped:
            self.scope.push()
        if isinstance(node, A.FunctionNode):
            for parameter in node.parameters or []:
                self.scope.declare(parameter.name, parameter.param_type)
            self.visit_children(node, skip=("parameters", "return_type"))
        else:
            self.visit_children(node)
        if isinstance(node, A.VariableNode):
            self.scope.declare(node.name, node.var_type)
        if scoped:
            self.scope.pop()

    def visit_children(self, node, skip=()):
        for field, key in list(child_slots(node)):
            if field in skip or field in ("var_type", "param_type", "const_type"):
                continue
            child = get_child(node, field, key)
            count = self.length(child)
            if count is not None:
                if isinstance(count, int):
                    count = A.LiteralNode(count, A.PrimitiveType("int"))
                set_child(node, field, key, count)
                self.folded += 1
                continue
            self.visit(child)


def fold_array_lengths(program, target) -> int:
    """Replace ``x.length()`` of constant-size arrays, vectors and matrices by the count; returns how many."""
    if not isinstance(program, A.ShaderNode) or str(target or "").lower() in _NATIVE:
        return 0
    return _Folder(program, str(target or "").lower()).run()
