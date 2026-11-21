"""SPIR-V lowering of the GLSL interpolation functions.

GLSL 4.60 section 8.13.2 defines ``interpolateAtCentroid(interpolant)``, ``interpolateAtSample(interpolant, int
sample)`` and ``interpolateAtOffset(interpolant, vec2 offset)``. Each reads a fragment input at another location in
the pixel. The interpolant is an input variable, an element of an input array, or a component selection of one.
CrossGL uses the same functions, and its interpolant may also be a member of an entry point's struct parameter,
which SPIR-V output declares as one ``Input`` variable per member and copies into the parameter's local struct.

SPIR-V expresses them with the ``GLSL.std.450`` instructions ``InterpolateAtCentroid`` (76), ``InterpolateAtSample``
(77) and ``InterpolateAtOffset`` (78). Their ``Interpolant`` operand is a pointer into the ``Input`` storage class
whose pointee is the result type, a 32-bit float scalar or vector. They are valid only in the ``Fragment``
execution model and need the ``InterpolationFunction`` capability. ``Sample`` is a 32-bit integer scalar and
``Offset`` a two-component 32-bit float vector.

The lowering therefore resolves the interpolant to the ``Input`` variable that holds it, never to the parameter's
local copy. A multi-component selection such as ``v.xy`` interpolates ``v`` and then selects the components, as
glslang does; a single component is addressed with an access chain. An interpolant this cannot resolve to an
``Input`` pointer, such as a local value, a parameter the entry point writes, or a value read in a helper
function, is refused.
"""

from __future__ import annotations

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    UnaryOpNode,
    VariableNode,
)
from .spirv_exponent_builtins import OUTPUT_OPERAND_FUNCTIONS

__all__ = [
    "SPIRV_INTERPOLATION_FUNCTIONS",
    "record_entry_input_variable",
    "spirv_interpolation_call",
]

# CrossGL name -> (GLSL.std.450 instruction, type of the second operand or None).
SPIRV_INTERPOLATION_FUNCTIONS = {
    "interpolateAtCentroid": ("InterpolateAtCentroid", None),
    "interpolateAtSample": ("InterpolateAtSample", "int"),
    "interpolateAtOffset": ("InterpolateAtOffset", "vec2"),
}

_INCREMENTS = {"++", "--", "PRE_INCREMENT", "PRE_DECREMENT"}
_INCREMENTS |= {"POST_INCREMENT", "POST_DECREMENT"}


def record_entry_input_variable(codegen, param_name, member_name, variable):
    """Remember that ``param_name.member_name`` (or ``param_name``) of the current entry point is ``variable``."""
    if variable is None or variable.type.storage_class != "Input":
        return
    variables = getattr(codegen, "entry_input_interface_variables", None)
    if variables is None:
        variables = codegen.entry_input_interface_variables = {}
    key = (codegen.current_function_name, param_name, member_name)
    variables[key] = variable


def spirv_interpolation_call(codegen, function_name, arguments, spirv_id_class):
    """Emit the ``GLSL.std.450`` instruction for ``function_name(arguments...)``."""
    instruction, operand_type = SPIRV_INTERPOLATION_FUNCTIONS[function_name]
    expected = 1 if operand_type is None else 2
    if len(arguments) != expected:
        return _refuse(
            codegen,
            spirv_id_class,
            f"{function_name} requires {expected} operand(s); got {len(arguments)}",
        )
    if codegen.current_execution_model != "Fragment":
        return _refuse(
            codegen,
            spirv_id_class,
            f"{function_name} reads a fragment input and is valid only in a fragment "
            "entry point (GLSL.std.450 requires the Fragment execution model)",
        )
    interpolant, swizzle = _interpolant(codegen, arguments[0])
    if interpolant is None:
        return _refuse(
            codegen,
            spirv_id_class,
            f"{function_name} needs an interpolant that is a fragment input variable, "
            "an element of one or a component selection of one; SPIR-V reads it "
            "through a pointer to the Input storage class",
        )
    value_type = codegen.pointer_pointee_type(interpolant)
    if value_type is None or not _is_float32_value(codegen, interpolant, value_type):
        return _refuse(
            codegen,
            spirv_id_class,
            f"{function_name} needs a 32-bit float scalar or vector interpolant",
        )
    operands = [f"%{interpolant.id}"]
    if operand_type is not None:
        operand = _operand(codegen, arguments[1], operand_type)
        if operand is None:
            return _refuse(
                codegen,
                spirv_id_class,
                f"{function_name} cannot convert its second operand to {operand_type}",
            )
        operands.append(f"%{operand.id}")
    codegen.require_capability("InterpolationFunction")
    result_id = codegen.get_id()
    codegen.emit(
        f"%{result_id} = OpExtInst %{value_type.id} %{codegen.glsl_std450_id} "
        f"{instruction} {' '.join(operands)}"
    )
    codegen.value_types[result_id] = value_type
    result = spirv_id_class(result_id, value_type.type)
    if swizzle is None:
        return result
    return _select(codegen, result, value_type, swizzle, spirv_id_class)


def _refuse(codegen, spirv_id_class, reason):
    codegen.refuse_lowering(reason)
    float_type = codegen.register_primitive_type("float")
    return spirv_id_class(codegen.get_id(), float_type.type)


def _is_float32_value(codegen, pointer, value_type):
    if pointer.id in getattr(codegen, "interface_original_value_types", {}):
        return False
    return codegen.scalar_or_vector_component_type(value_type.type) == "float"


def _operand(codegen, expression, operand_type):
    value = codegen.process_expression(expression)
    if value is None:
        return None
    if operand_type == "int":
        target = codegen.register_primitive_type("int")
    else:
        target = codegen.register_vector_type(
            codegen.register_primitive_type("float"), 2
        )
    converted = codegen.convert_value_to_type(value, target)
    if converted is None or converted.type.base_type != target.type.base_type:
        return None
    return converted


def _interpolant(codegen, expression):
    """The ``Input`` pointer that holds ``expression`` and the swizzle still to apply, or ``(None, None)``."""
    pointer = _input_pointer(codegen, expression)
    if pointer is not None:
        return pointer, None
    if isinstance(expression, MemberAccessNode):
        base = _input_pointer(codegen, expression.object)
        if base is not None:
            base_type = codegen.pointer_pointee_type(base)
            if base_type is not None and codegen.vector_swizzle_indices(
                base_type.type.base_type, str(expression.member)
            ):
                return base, str(expression.member)
    return None, None


def _input_pointer(codegen, expression):
    entry = _entry_input_variable(codegen, expression)
    if entry is not None:
        return entry
    if _entry_parameter_root(codegen, expression) is not None:
        # A read through an entry parameter's local copy: only one component of a member's Input variable is
        # addressable here.
        if not isinstance(expression, MemberAccessNode):
            return None
        base = _entry_input_variable(codegen, expression.object)
        if base is None:
            return None
        return codegen.vector_component_reference_pointer(base, str(expression.member))
    pointer = codegen.assignable_pointer_from_expression(expression)
    if pointer is None or pointer.type.storage_class != "Input":
        return None
    return pointer


def _entry_input_variable(codegen, expression):
    """The ``Input`` variable of an unwritten entry parameter, or of one member of it, that ``expression`` names."""
    variables = getattr(codegen, "entry_input_interface_variables", None) or {}
    function = codegen.current_function_name
    if isinstance(expression, (IdentifierNode, VariableNode)):
        key = (function, expression.name, None)
    elif isinstance(expression, MemberAccessNode) and isinstance(
        expression.object, (IdentifierNode, VariableNode)
    ):
        key = (function, expression.object.name, str(expression.member))
    else:
        return None
    variable = variables.get(key)
    if variable is None or key[1] in _written_names(codegen):
        return None
    return variable


def _entry_parameter_root(codegen, expression):
    """The entry parameter ``expression`` reads through, if it reads one whose members are Input variables."""
    variables = getattr(codegen, "entry_input_interface_variables", None) or {}
    function = codegen.current_function_name
    root = _root_name(expression)
    if root is None:
        return None
    if any(key[0] == function and key[1] == root for key in variables):
        return root
    return None


def _root_name(expression):
    while True:
        if isinstance(expression, (IdentifierNode, VariableNode)):
            return expression.name
        if isinstance(expression, MemberAccessNode):
            expression = expression.object
        elif isinstance(expression, ArrayAccessNode):
            expression = getattr(expression, "array", None)
        else:
            return None


def _written_names(codegen):
    """Names the current function may change: assignment and increment targets and call arguments."""
    function = getattr(codegen, "current_function_node", None)
    cache = getattr(codegen, "_interpolation_written_names", None)
    if cache is not None and cache[0] is function:
        return cache[1]
    names = set()
    for node in _walk(getattr(function, "body", None)):
        if isinstance(node, AssignmentNode):
            names.add(_root_name(getattr(node, "left", getattr(node, "target", None))))
        elif isinstance(node, UnaryOpNode) and node.op in _INCREMENTS | {"&"}:
            names.add(_root_name(node.operand))
        elif isinstance(node, FunctionCallNode):
            callee = getattr(node, "name", None)
            callee = getattr(callee, "name", callee)
            if not _may_write_arguments(codegen, callee):
                continue
            for argument in getattr(node, "args", None) or []:
                names.add(_root_name(argument))
    names.discard(None)
    codegen._interpolation_written_names = (function, names)
    return names


def _may_write_arguments(codegen, callee):
    """Whether a call to ``callee`` may write an argument: a program function, or a built-in with an out operand."""
    if not isinstance(callee, str) or callee in SPIRV_INTERPOLATION_FUNCTIONS:
        return False
    if callee in OUTPUT_OPERAND_FUNCTIONS or callee.startswith(
        ("atomic", "Interlocked")
    ):
        return True
    return codegen.has_function_reference(callee)


def _walk(node):
    stack, seen = [node], set()
    while stack:
        current = stack.pop()
        if current is None or isinstance(current, (str, int, float, bool)):
            continue
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, (list, tuple)):
            stack.extend(current)
            continue
        yield current
        if hasattr(current, "child_nodes"):
            stack.extend(current.child_nodes())


def _select(codegen, value, value_type, swizzle, spirv_id_class):
    info = codegen.vector_swizzle_info(value_type.type.base_type, swizzle)
    indices, component_type, result_type = info
    result_id = codegen.get_id()
    literals = " ".join(str(index) for index in indices)
    codegen.emit(
        f"%{result_id} = OpVectorShuffle %{result_type.id} %{value.id} %{value.id} "
        f"{literals}"
    )
    codegen.value_types[result_id] = result_type
    return spirv_id_class(result_id, result_type.type)
