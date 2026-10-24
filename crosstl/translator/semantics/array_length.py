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
with a literal value is used; any other ``length()`` (a runtime-sized array, a buffer, a size that is a
specialization constant) is left for the emitters, which lower or refuse it.
"""

from __future__ import annotations

from .. import ast as A
from .hub_types import TypeScope
from .rewrite import child_slots, get_child, set_child

__all__ = ["fold_array_lengths"]

# Targets whose arrays have a native ``length()`` method.
_NATIVE = frozenset({"opengl", "glsl", "webgl", "essl", "gles", "cgl", "crossgl"})
_SCOPES = (A.BlockNode, A.ForNode, A.ForInNode, A.FunctionNode)


def _literal_int(node):
    if isinstance(node, A.LiteralNode) and not isinstance(node.value, bool):
        if isinstance(node.value, int):
            return node.value
        text = str(node.value).strip().rstrip("uU")
        if text.isdigit():
            return int(text)
    return None


class _Folder:
    def __init__(self, program):
        self.program = program
        self.scope = TypeScope(program)
        self.constants = {}
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

    def run(self):
        for constant in getattr(self.program, "constants", None) or []:
            self.scope.declare(constant.name, constant.const_type)
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
                if "const" in {str(q) for q in variable.qualifiers or []}:
                    value = _literal_int(getattr(variable, "initial_value", None))
                    if value is not None:
                        self.constants[variable.name] = value

    def length(self, node):
        """The constant count ``node`` (a ``length()`` call) evaluates to, or ``None``."""
        if not isinstance(node, A.FunctionCallNode) or node.arguments:
            return None
        callee = node.function
        if not (isinstance(callee, A.MemberAccessNode) and callee.member == "length"):
            return None
        receiver = callee.object_expr
        declared = self.scope.node_type(receiver)
        if isinstance(declared, A.ArrayType):
            return self.size(declared)
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
                set_child(
                    node, field, key, A.LiteralNode(count, A.PrimitiveType("int"))
                )
                self.folded += 1
                continue
            self.visit(child)


def fold_array_lengths(program, target) -> int:
    """Replace ``x.length()`` of constant-size arrays, vectors and matrices by the count; returns how many."""
    if not isinstance(program, A.ShaderNode) or str(target or "").lower() in _NATIVE:
        return 0
    return _Folder(program).run()
