"""OpenCL-specific CrossGL AST lowering for target backend code generation."""

from crosstl.translator.ast import (
    ArrayType,
    AttributeNode,
    FunctionNode,
    LiteralNode,
    NamedType,
    PrimitiveType,
    ShaderStage,
    StageNode,
    StructMemberNode,
    StructNode,
    VariableNode,
    VectorType,
)

_TARGET_SCALAR_TYPE_ALIASES = {
    "f32": "float",
    "f64": "double",
    "i32": "int",
    "u32": "uint",
}
_STAGE_CONTROL_ATTRIBUTES = {"compute", "workgroup_size", "local_size"}
_SYNTHETIC_BUILTIN_ALIASES = {
    "thread_id": "gl_GlobalInvocationID",
    "block_id": "gl_WorkGroupID",
    "thread_local_id": "gl_LocalInvocationID",
    "block_dim": "gl_WorkGroupSize",
}


def _clone_target_type(type_node):
    if isinstance(type_node, PrimitiveType):
        return PrimitiveType(
            _TARGET_SCALAR_TYPE_ALIASES.get(type_node.name, type_node.name),
            size_bits=type_node.size_bits,
        )
    if isinstance(type_node, VectorType):
        return VectorType(
            _clone_target_type(type_node.element_type),
            type_node.size,
        )
    if isinstance(type_node, ArrayType):
        return ArrayType(
            _clone_target_type(type_node.element_type),
            type_node.size,
        )
    if isinstance(type_node, NamedType):
        return NamedType(
            type_node.name,
            [_clone_target_type(arg) for arg in type_node.generic_args],
        )
    return type_node


def _attribute_literal_value(argument):
    if hasattr(argument, "value"):
        return argument.value
    if hasattr(argument, "name"):
        return argument.name
    return argument


def _attribute_int_value(attribute):
    arguments = getattr(attribute, "arguments", []) or []
    if not arguments:
        return None
    value = _attribute_literal_value(arguments[0])
    try:
        return int(str(value), 0)
    except (TypeError, ValueError):
        return None


def _binding_attribute(binding):
    return AttributeNode(
        "binding",
        [LiteralNode(binding, PrimitiveType("int"))],
    )


def _parameter_binding(parameter):
    for attribute in getattr(parameter, "attributes", []) or []:
        if str(getattr(attribute, "name", "")).lower() != "binding":
            continue
        binding = _attribute_int_value(attribute)
        if binding is not None:
            return binding
    return None


def _stage_execution_config(function):
    for attribute in getattr(function, "attributes", []) or []:
        name = str(getattr(attribute, "name", "")).lower()
        if name not in {"workgroup_size", "local_size", "numthreads"}:
            continue
        values = [
            str(_attribute_literal_value(argument))
            for argument in (getattr(attribute, "arguments", []) or [])[:3]
        ]
        if len(values) == 3:
            return {"numthreads": tuple(values)}
    return {}


def _target_function_attributes(function):
    return [
        attribute
        for attribute in getattr(function, "attributes", []) or []
        if str(getattr(attribute, "name", "")).lower() not in _STAGE_CONTROL_ATTRIBUTES
    ]


def _is_synthetic_builtin_alias(statement):
    if not isinstance(statement, VariableNode):
        return False
    expected_builtin = _SYNTHETIC_BUILTIN_ALIASES.get(statement.name)
    if expected_builtin is None:
        return False
    initializer = getattr(statement, "initial_value", None)
    return getattr(initializer, "name", None) == expected_builtin


def _strip_synthetic_builtin_aliases(function):
    body = getattr(function, "body", None)
    statements = getattr(body, "statements", None)
    if not isinstance(statements, list):
        return
    while statements and _is_synthetic_builtin_alias(statements[0]):
        statements.pop(0)


def _structured_buffer_type(parameter, writable=True):
    param_type = getattr(parameter, "param_type", None)
    if isinstance(param_type, ArrayType):
        element_type = param_type.element_type
    else:
        element_type = param_type
    buffer_name = "RWStructuredBuffer" if writable else "StructuredBuffer"
    return NamedType(buffer_name, [_clone_target_type(element_type)])


def _stage_local_variable(parameter):
    param_type = _clone_target_type(getattr(parameter, "param_type", None))
    variable = VariableNode(
        parameter.name,
        param_type,
        qualifiers=["workgroup"],
    )
    variable.is_var_address_space = True
    variable.address_space_qualifiers = ["workgroup"]
    return variable


def _target_kernel_cbuffer(function_name, scalar_parameters, binding):
    members = [
        StructMemberNode(
            parameter.name,
            _clone_target_type(getattr(parameter, "param_type", None)),
        )
        for parameter in scalar_parameters
    ]
    cbuffer = StructNode(
        f"{function_name}_Args",
        members,
        attributes=[_binding_attribute(binding)],
    )
    cbuffer.is_cbuffer = True
    return cbuffer


def _resource_qualifiers(parameter):
    return {
        str(qualifier).lower()
        for qualifier in getattr(parameter, "resource_qualifiers", []) or []
    }


def _is_writable_storage_parameter(parameter):
    qualifiers = _resource_qualifiers(parameter)
    return not qualifiers.intersection({"read", "readonly", "read_only"})


def _is_target_compute_function(function):
    if not isinstance(function, FunctionNode):
        return False
    return any(
        str(getattr(attribute, "name", "")).lower() == "compute"
        for attribute in getattr(function, "attributes", []) or []
    )


def normalize_opencl_intermediate_for_target(cgl_ast):
    """Lower OpenCL bridge compute parameters to neutral CrossGL resources."""

    functions = list(getattr(cgl_ast, "functions", []) or [])
    target_functions = []
    remaining_functions = []

    for function in functions:
        if _is_target_compute_function(function):
            target_functions.append(function)
        else:
            remaining_functions.append(function)

    if not target_functions:
        return cgl_ast

    cgl_ast.functions = remaining_functions

    for function in target_functions:
        global_variables = []
        local_variables = []
        scalar_parameters = []
        next_binding = 0

        for parameter in getattr(function, "parameters", []) or []:
            qualifiers = _resource_qualifiers(parameter)
            if "workgroup" in qualifiers:
                local_variables.append(_stage_local_variable(parameter))
                continue
            if "storage" in qualifiers:
                binding = _parameter_binding(parameter)
                if binding is None:
                    binding = next_binding
                next_binding = max(next_binding, binding + 1)
                global_variables.append(
                    VariableNode(
                        parameter.name,
                        _structured_buffer_type(
                            parameter,
                            writable=_is_writable_storage_parameter(parameter),
                        ),
                        attributes=[_binding_attribute(binding)],
                    )
                )
                continue
            scalar_parameters.append(parameter)

        if scalar_parameters:
            cgl_ast.cbuffers.append(
                _target_kernel_cbuffer(
                    function.name,
                    scalar_parameters,
                    next_binding,
                )
            )

        cgl_ast.global_variables.extend(global_variables)
        function.parameters = []
        function.attributes = _target_function_attributes(function)
        _strip_synthetic_builtin_aliases(function)
        cgl_ast.stages.append(
            ShaderStage.COMPUTE,
            StageNode(
                ShaderStage.COMPUTE,
                function,
                local_variables=local_variables,
                execution_config=_stage_execution_config(function),
            ),
        )

    return cgl_ast
