"""OpenCL-specific CrossGL AST lowering for target backend code generation."""

from dataclasses import dataclass
from typing import Tuple

from crosstl.translator.ast import (
    ArrayAccessNode,
    ArrayType,
    AssignmentNode,
    AttributeNode,
    BinaryOpNode,
    BlockNode,
    ExpressionStatementNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    NamedType,
    PrimitiveType,
    ReturnNode,
    ShaderStage,
    StageNode,
    StructMemberNode,
    StructNode,
    VariableNode,
    VectorType,
)

OPENCL_TARGET_UNSUPPORTED_CODE = "opencl.target.unsupported"
OPENCL_TARGET_UNRESOLVED_HELPER_CODE = "opencl.target.unresolved-helper"
OPENCL_TARGET_POINTER_HELPER_CODE = "opencl.target.pointer-helper-parameter"
OPENCL_TARGET_BUILTIN_CODE = "opencl.target.unsupported-builtin"
DIRECTX_BINDING_PROVENANCE_ANNOTATION = "directx.binding_provenance"
DIRECTX_RELOCATABLE_BINDING_ANNOTATION = "directx.relocatable_binding"
_TARGET_SCALAR_TYPE_ALIASES = {
    "f32": "float",
    "f64": "double",
    "i32": "int",
    "u32": "uint",
    "i64": "int64_t",
    "u64": "uint64_t",
}
_STAGE_CONTROL_ATTRIBUTES = {"compute", "workgroup_size", "local_size"}
_EVENT_LOCAL_MEMORY_BUILTINS = {
    "async_work_group_copy",
    "wait_group_events",
}
_DYNAMIC_LOCAL_MEMORY_FALLBACK_SIZE = 1024
_SYNTHETIC_BUILTIN_ALIASES = {
    "thread_id": "gl_GlobalInvocationID",
    "block_id": "gl_WorkGroupID",
    "thread_local_id": "gl_LocalInvocationID",
    "block_dim": "gl_WorkGroupSize",
}
_IMAGE_RESOURCE_TYPE_NAMES = {
    "image1D",
    "image1DArray",
    "image2D",
    "image2DArray",
    "image2DMS",
    "image2DMSArray",
    "image3D",
    "iimage1D",
    "iimage1DArray",
    "iimage2D",
    "iimage2DArray",
    "iimage2DMS",
    "iimage2DMSArray",
    "iimage3D",
    "uimage1D",
    "uimage1DArray",
    "uimage2D",
    "uimage2DArray",
    "uimage2DMS",
    "uimage2DMSArray",
    "uimage3D",
}
_SAMPLER_RESOURCE_TYPE_NAMES = {
    "sampler",
    "sampler1D",
    "sampler1DArray",
    "sampler2D",
    "sampler2DArray",
    "sampler2DMS",
    "sampler2DMSArray",
    "sampler2DShadow",
    "sampler2DArrayShadow",
    "sampler3D",
    "samplerCube",
    "samplerCubeArray",
    "samplerCubeShadow",
    "samplerCubeArrayShadow",
}


class OpenCLTargetUnsupportedError(ValueError):
    """Raised when OpenCL target lowering would emit invalid target code."""

    project_diagnostic_code = OPENCL_TARGET_UNSUPPORTED_CODE
    missing_capabilities = ("opencl.target-lowering",)

    def __init__(self, message, *, contracts=()):
        super().__init__(message)
        self.contracts = tuple(contracts)


@dataclass(frozen=True)
class OpenCLTargetUnsupportedContract:
    """One unsupported OpenCL contract discovered before target codegen."""

    code: str
    kind: str
    name: str
    signature: str
    reason: str
    action: str
    missing_capabilities: Tuple[str, ...]

    def format_project_message(self, target_backend=None):
        target = _target_backend_label(target_backend)
        subject = self.signature or self.name
        return (
            f"OpenCL target lowering cannot lower {subject} to {target}: "
            f"{self.reason}. Suggested action: {self.action}"
        )

    def to_json(self):
        return {
            "code": self.code,
            "kind": self.kind,
            "name": self.name,
            "signature": self.signature,
            "reason": self.reason,
            "action": self.action,
            "missingCapabilities": list(self.missing_capabilities),
        }


def _target_backend_label(target_backend):
    target = str(target_backend or "target").strip()
    return target or "target"


def _type_label(type_node):
    if isinstance(type_node, PrimitiveType):
        return type_node.name
    if isinstance(type_node, VectorType):
        return f"vec{type_node.size}<{_type_label(type_node.element_type)}>"
    if isinstance(type_node, ArrayType):
        element = _type_label(type_node.element_type)
        if type_node.size is None:
            return f"array<{element}>"
        return f"array<{element}, {type_node.size}>"
    if isinstance(type_node, NamedType):
        if not type_node.generic_args:
            return type_node.name
        args = ", ".join(_type_label(arg) for arg in type_node.generic_args)
        return f"{type_node.name}<{args}>"
    return str(type_node)


def _is_void_type(type_node):
    return isinstance(type_node, PrimitiveType) and type_node.name == "void"


def _function_statements(function):
    body = getattr(function, "body", None)
    statements = getattr(body, "statements", None)
    return statements if isinstance(statements, list) else None


def _is_empty_nonvoid_function(function):
    if not isinstance(function, FunctionNode) or _is_void_type(function.return_type):
        return False
    statements = _function_statements(function)
    return statements is not None and len(statements) == 0


def _is_pointer_named_type(type_node):
    return isinstance(type_node, NamedType) and type_node.name == "ptr"


def _pointer_parameter_labels(function):
    labels = []
    for parameter in getattr(function, "parameters", []) or []:
        param_type = getattr(parameter, "param_type", None)
        if _is_pointer_named_type(param_type):
            labels.append(f"{parameter.name}: {_type_label(param_type)}")
    return labels


def _function_signature_label(function):
    params = []
    for parameter in getattr(function, "parameters", []) or []:
        params.append(f"{parameter.name}: {_type_label(parameter.param_type)}")
    return (
        f"{function.name}({', '.join(params)}) -> "
        f"{_type_label(function.return_type)}"
    )


def _function_call_name(call):
    function = getattr(call, "function", None)
    if isinstance(function, IdentifierNode):
        return function.name
    if isinstance(function, str):
        return function
    return getattr(function, "name", None)


def _collect_called_function_names(node):
    calls = set()
    for child in node.walk():
        if isinstance(child, FunctionCallNode):
            name = _function_call_name(child)
            if name:
                calls.add(str(name))
    return calls


def _collect_called_function_names_from_functions(functions):
    called_names = set()
    for function in functions:
        called_names.update(_collect_called_function_names(function))
    return called_names


def _format_list(values):
    return ", ".join(sorted(set(values)))


def _opencl_target_contracts(cgl_ast):
    functions = [
        function
        for function in getattr(cgl_ast, "functions", []) or []
        if isinstance(function, FunctionNode)
    ]
    for stage in getattr(cgl_ast, "stages", {}).values():
        entry_point = getattr(stage, "entry_point", None)
        if isinstance(entry_point, FunctionNode):
            functions.append(entry_point)
    called_names = _collect_called_function_names_from_functions(functions)

    contracts = []
    for function in functions:
        if (
            not _is_empty_nonvoid_function(function)
            or function.name not in called_names
        ):
            continue
        contracts.append(
            OpenCLTargetUnsupportedContract(
                code=OPENCL_TARGET_UNRESOLVED_HELPER_CODE,
                kind="unresolved-helper-declaration",
                name=function.name,
                signature=_function_signature_label(function),
                reason=(
                    "non-void helper declaration has no body, so reduction "
                    "semantics would be lost before target codegen"
                ),
                action=(
                    "provide an OpenCL helper body, materialize a known reduction "
                    "helper specialization, or keep the source on an OpenCL path"
                ),
                missing_capabilities=("opencl.helper-resolution",),
            )
        )

    for function in functions:
        labels = _pointer_parameter_labels(function)
        for label in labels:
            contracts.append(
                OpenCLTargetUnsupportedContract(
                    code=OPENCL_TARGET_POINTER_HELPER_CODE,
                    kind="pointer-helper-parameter",
                    name=function.name,
                    signature=f"{function.name}({label})",
                    reason=(
                        "helper pointer parameters do not preserve OpenCL "
                        "address-space and local-memory aliasing semantics in "
                        "target codegen"
                    ),
                    action=(
                        "rewrite the helper as kernel-local logic, pass scalar "
                        "values explicitly, or keep the source on an OpenCL path"
                    ),
                    missing_capabilities=("opencl.local-pointer-helper",),
                )
            )

    for builtin_name in sorted(called_names.intersection(_EVENT_LOCAL_MEMORY_BUILTINS)):
        contracts.append(
            OpenCLTargetUnsupportedContract(
                code=OPENCL_TARGET_BUILTIN_CODE,
                kind="event-local-memory-builtin",
                name=builtin_name,
                signature=f"{builtin_name}(...)",
                reason=(
                    "event/local-memory builtin semantics are not preserved by "
                    "target lowering"
                ),
                action=(
                    "replace the builtin with explicit target synchronization and "
                    "copy logic, or keep the source on an OpenCL path"
                ),
                missing_capabilities=("opencl.event-local-memory",),
            )
        )

    return tuple(contracts)


def validate_opencl_intermediate_for_target(cgl_ast, target_backend=None):
    """Reject OpenCL intermediates that target codegen cannot lower correctly."""

    contracts = _opencl_target_contracts(cgl_ast)
    if not contracts:
        return

    unsupported = []
    empty_helpers = [
        contract.name
        for contract in contracts
        if contract.kind == "unresolved-helper-declaration"
    ]
    if empty_helpers:
        unsupported.append(
            "unresolved non-void helper declarations without bodies: "
            f"{_format_list(empty_helpers)}"
        )

    pointer_helpers = [
        contract.signature
        for contract in contracts
        if contract.kind == "pointer-helper-parameter"
    ]
    if pointer_helpers:
        unsupported.append(
            "OpenCL pointer helper parameters are not representable in target "
            f"codegen: {_format_list(pointer_helpers)}"
        )

    event_local_calls = [
        contract.name
        for contract in contracts
        if contract.kind == "event-local-memory-builtin"
    ]
    if event_local_calls:
        unsupported.append(
            "OpenCL event/local-memory builtins require semantics not preserved by "
            f"target lowering: {_format_list(event_local_calls)}"
        )

    target = _target_backend_label(target_backend)
    details = "; ".join(unsupported)
    raise OpenCLTargetUnsupportedError(
        f"{OPENCL_TARGET_UNSUPPORTED_CODE}: cannot lower OpenCL source to "
        f"{target} because target lowering would produce invalid artifacts: {details}",
        contracts=contracts,
    )


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


def _mark_directx_relocatable_binding(node, provenance):
    node.add_annotation(DIRECTX_RELOCATABLE_BINDING_ANNOTATION, True)
    node.add_annotation(DIRECTX_BINDING_PROVENANCE_ANNOTATION, provenance)
    return node


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
    if isinstance(param_type, ArrayType) and param_type.size is None:
        param_type = ArrayType(
            param_type.element_type,
            _DYNAMIC_LOCAL_MEMORY_FALLBACK_SIZE,
        )
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
    return _mark_directx_relocatable_binding(
        cbuffer,
        f"OpenCL target-lowered scalar parameter block {function_name}_Args",
    )


def _parameter_type_base_name(parameter):
    param_type = getattr(parameter, "param_type", None)
    while isinstance(param_type, ArrayType):
        param_type = param_type.element_type
    if isinstance(param_type, NamedType):
        return param_type.name
    return ""


def _is_resource_parameter(parameter):
    return _parameter_type_base_name(parameter) in (
        _IMAGE_RESOURCE_TYPE_NAMES | _SAMPLER_RESOURCE_TYPE_NAMES
    )


def _has_binding_attribute(parameter):
    return _parameter_binding(parameter) is not None


def _target_resource_variable(function_name, parameter, binding):
    attributes = list(getattr(parameter, "attributes", []) or [])
    if not _has_binding_attribute(parameter):
        attributes.append(_binding_attribute(binding))
    variable = VariableNode(
        parameter.name,
        _clone_target_type(getattr(parameter, "param_type", None)),
        attributes=attributes,
    )
    variable.resource_qualifiers = list(
        getattr(parameter, "resource_qualifiers", []) or []
    )
    return _mark_directx_relocatable_binding(
        variable,
        f"OpenCL target-lowered resource parameter {function_name}.{parameter.name}",
    )


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


def _is_workgroup_resource_parameter(parameter):
    return "workgroup" in _resource_qualifiers(parameter)


def _workgroup_parameter_names(functions):
    names = set()
    for function in functions:
        if not _is_target_compute_function(function):
            continue
        for parameter in getattr(function, "parameters", []) or []:
            if _is_workgroup_resource_parameter(parameter):
                names.add(parameter.name)
    return names


def _called_workgroup_pointer_parameters(functions, workgroup_names):
    function_by_name = {function.name: function for function in functions}
    parameter_keys = set()
    for function in functions:
        for child in function.walk():
            if not isinstance(child, FunctionCallNode):
                continue
            callee_name = _function_call_name(child)
            callee = function_by_name.get(callee_name)
            if callee is None:
                continue
            parameters = getattr(callee, "parameters", []) or []
            arguments = getattr(child, "arguments", getattr(child, "args", [])) or []
            for index, argument in enumerate(arguments):
                if index >= len(parameters):
                    continue
                if getattr(argument, "name", None) not in workgroup_names:
                    continue
                parameter = parameters[index]
                if _is_pointer_named_type(getattr(parameter, "param_type", None)):
                    parameter_keys.add((callee.name, index))
    return parameter_keys


def _specialize_workgroup_pointer_helpers(functions):
    workgroup_names = _workgroup_parameter_names(functions)
    if not workgroup_names:
        return

    parameter_keys = _called_workgroup_pointer_parameters(functions, workgroup_names)
    if not parameter_keys:
        return

    for function in functions:
        for index, parameter in enumerate(getattr(function, "parameters", []) or []):
            if (function.name, index) not in parameter_keys:
                continue
            param_type = getattr(parameter, "param_type", None)
            if not _is_pointer_named_type(param_type):
                continue
            parameter.param_type = ArrayType(
                param_type.generic_args[0],
                _DYNAMIC_LOCAL_MEMORY_FALLBACK_SIZE,
            )
            qualifiers = list(getattr(parameter, "qualifiers", []) or [])
            if "workgroup" not in {str(qualifier).lower() for qualifier in qualifiers}:
                qualifiers.append("workgroup")
            parameter.qualifiers = qualifiers


def _prune_unused_empty_helpers(functions):
    called_names = _collect_called_function_names_from_functions(functions)
    return [
        function
        for function in functions
        if not _is_empty_nonvoid_function(function) or function.name in called_names
    ]


def _is_call_statement(statement, builtin_name):
    if not isinstance(statement, ExpressionStatementNode):
        return False
    expression = getattr(statement, "expression", None)
    return (
        isinstance(expression, FunctionCallNode)
        and _function_call_name(expression) == builtin_name
    )


def _identifier(name):
    return IdentifierNode(str(name))


def _local_invocation_index():
    return _identifier("lid")


def _array_access_with_offset(array_expr, index_expr):
    if (
        isinstance(array_expr, BinaryOpNode)
        and array_expr.operator == "+"
        and isinstance(array_expr.left, IdentifierNode)
    ):
        return ArrayAccessNode(
            array_expr.left,
            BinaryOpNode(array_expr.right, "+", index_expr),
        )
    return ArrayAccessNode(array_expr, index_expr)


def _lower_async_work_group_copy(statement):
    expression = getattr(statement, "expression", None)
    arguments = getattr(expression, "arguments", getattr(expression, "args", [])) or []
    if len(arguments) < 3:
        return statement

    local_index = _local_invocation_index()
    destination = arguments[0]
    source = arguments[1]
    count = arguments[2]
    assignment = AssignmentNode(
        ArrayAccessNode(destination, local_index),
        _array_access_with_offset(source, local_index),
    )
    return IfNode(
        BinaryOpNode(local_index, "<", count),
        BlockNode([assignment]),
    )


def _event_token_name(argument):
    name = getattr(argument, "name", None)
    if name:
        return name
    operand = getattr(argument, "operand", None)
    return getattr(operand, "name", None)


def _collect_event_local_memory_tokens(statements):
    names = set()
    for statement in statements:
        for child in statement.walk():
            if not isinstance(child, FunctionCallNode):
                continue
            call_name = _function_call_name(child)
            arguments = getattr(child, "arguments", getattr(child, "args", [])) or []
            if call_name == "async_work_group_copy" and len(arguments) >= 4:
                token_name = _event_token_name(arguments[3])
                if token_name:
                    names.add(token_name)
            elif call_name == "wait_group_events":
                for argument in arguments[1:]:
                    token_name = _event_token_name(argument)
                    if token_name:
                        names.add(token_name)
    return names


def _normalize_opencl_event_local_memory_statement(statement, event_tokens):
    if isinstance(statement, VariableNode) and statement.name in event_tokens:
        return []
    if _is_call_statement(statement, "wait_group_events"):
        return []
    if _is_call_statement(statement, "async_work_group_copy"):
        return [_lower_async_work_group_copy(statement)]

    if isinstance(statement, BlockNode):
        _normalize_opencl_event_local_memory_statements(
            statement.statements,
            event_tokens,
        )
    elif isinstance(statement, IfNode):
        _normalize_opencl_event_local_memory_branch(statement.then_branch, event_tokens)
        if statement.else_branch is not None:
            _normalize_opencl_event_local_memory_branch(
                statement.else_branch,
                event_tokens,
            )
    elif isinstance(statement, ForNode):
        _normalize_opencl_event_local_memory_branch(statement.body, event_tokens)
    return [statement]


def _normalize_opencl_event_local_memory_branch(branch, event_tokens):
    statements = getattr(branch, "statements", None)
    if isinstance(statements, list):
        _normalize_opencl_event_local_memory_statements(statements, event_tokens)


def _normalize_opencl_event_local_memory_statements(statements, event_tokens):
    normalized = []
    for statement in list(statements):
        normalized.extend(
            _normalize_opencl_event_local_memory_statement(statement, event_tokens)
        )
    statements[:] = normalized


def _normalize_opencl_event_local_memory(functions):
    for function in functions:
        statements = _function_statements(function)
        if isinstance(statements, list):
            event_tokens = _collect_event_local_memory_tokens(statements)
            _normalize_opencl_event_local_memory_statements(statements, event_tokens)


def _is_signed_int_type(type_node):
    return isinstance(type_node, PrimitiveType) and type_node.name in {"i32", "int"}


def _is_unsigned_int_type(type_node):
    return isinstance(type_node, PrimitiveType) and type_node.name in {"u32", "uint"}


def _function_by_name(functions, name):
    for function in functions:
        if getattr(function, "name", None) == name:
            return function
    return None


def _is_opencl_sdk_reduce_op_helper(function):
    parameters = getattr(function, "parameters", []) or []
    return (
        isinstance(function, FunctionNode)
        and function.name == "op"
        and _is_empty_nonvoid_function(function)
        and _is_signed_int_type(getattr(function, "return_type", None))
        and len(parameters) == 2
        and all(
            _is_signed_int_type(getattr(parameter, "param_type", None))
            for parameter in parameters
        )
    )


def _is_opencl_sdk_reduce_shape(functions):
    reduce_function = _function_by_name(functions, "reduce")
    if reduce_function is None or not _is_target_compute_function(reduce_function):
        return False

    called_names = _collect_called_function_names(reduce_function)
    if not {"op", "read_local", "zmin"}.issubset(called_names):
        return False

    parameter_names = {
        getattr(parameter, "name", None)
        for parameter in getattr(reduce_function, "parameters", []) or []
    }
    if not {"front", "back", "length", "zero_elem"}.issubset(parameter_names):
        return False
    if not any(str(name or "").startswith("shared") for name in parameter_names):
        return False

    read_local = _function_by_name(functions, "read_local")
    if read_local is None:
        return False

    return any(
        _is_pointer_named_type(getattr(parameter, "param_type", None))
        for parameter in getattr(read_local, "parameters", []) or []
    )


def _materialize_opencl_sdk_reduce_helpers(functions):
    if not _is_opencl_sdk_reduce_shape(functions):
        return

    op_helper = _function_by_name(functions, "op")
    if not _is_opencl_sdk_reduce_op_helper(op_helper):
        return

    # The OpenCL-SDK reduce host appends this default min reducer at build time.
    lhs, rhs = (parameter.name for parameter in op_helper.parameters)
    op_helper.body = BlockNode(
        [
            ReturnNode(
                FunctionCallNode(
                    IdentifierNode("min"),
                    [IdentifierNode(lhs), IdentifierNode(rhs)],
                )
            )
        ]
    )


def _collect_variable_types(statements):
    variable_types = {}
    for statement in statements:
        for child in statement.walk():
            if isinstance(child, VariableNode) and child.name:
                variable_types.setdefault(child.name, getattr(child, "var_type", None))
    return variable_types


def _normalize_opencl_for_loop_counter_statement(statement, variable_types):
    if isinstance(statement, ForNode):
        init = getattr(statement, "init", None)
        initial_value = getattr(init, "initial_value", None)
        source_type = variable_types.get(getattr(initial_value, "name", None))
        if (
            isinstance(init, VariableNode)
            and _is_signed_int_type(getattr(init, "var_type", None))
            and _is_unsigned_int_type(source_type)
        ):
            init.var_type = source_type
            init.vtype = source_type
        _normalize_opencl_for_loop_counter_branch(statement.body, variable_types)
    elif isinstance(statement, BlockNode):
        _normalize_opencl_for_loop_counter_statements(
            statement.statements,
            variable_types,
        )
    elif isinstance(statement, IfNode):
        _normalize_opencl_for_loop_counter_branch(statement.then_branch, variable_types)
        if statement.else_branch is not None:
            _normalize_opencl_for_loop_counter_branch(
                statement.else_branch,
                variable_types,
            )


def _normalize_opencl_for_loop_counter_branch(branch, variable_types):
    statements = getattr(branch, "statements", None)
    if isinstance(statements, list):
        _normalize_opencl_for_loop_counter_statements(statements, variable_types)


def _normalize_opencl_for_loop_counter_statements(statements, variable_types):
    for statement in statements:
        _normalize_opencl_for_loop_counter_statement(statement, variable_types)


def _normalize_opencl_for_loop_counter_types(functions):
    for function in functions:
        statements = _function_statements(function)
        if isinstance(statements, list):
            _normalize_opencl_for_loop_counter_statements(
                statements,
                _collect_variable_types(statements),
            )


def normalize_opencl_intermediate_for_target(cgl_ast):
    """Lower OpenCL bridge compute parameters to neutral CrossGL resources."""

    functions = list(getattr(cgl_ast, "functions", []) or [])
    _materialize_opencl_sdk_reduce_helpers(functions)
    _specialize_workgroup_pointer_helpers(functions)
    _normalize_opencl_for_loop_counter_types(functions)
    _normalize_opencl_event_local_memory(functions)
    functions = _prune_unused_empty_helpers(functions)
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
                    _mark_directx_relocatable_binding(
                        VariableNode(
                            parameter.name,
                            _structured_buffer_type(
                                parameter,
                                writable=_is_writable_storage_parameter(parameter),
                            ),
                            attributes=[_binding_attribute(binding)],
                        ),
                        (
                            "OpenCL target-lowered storage parameter "
                            f"{function.name}.{parameter.name}"
                        ),
                    )
                )
                continue
            if _is_resource_parameter(parameter):
                binding = _parameter_binding(parameter)
                if binding is None:
                    binding = next_binding
                next_binding = max(next_binding, binding + 1)
                global_variables.append(
                    _target_resource_variable(function.name, parameter, binding)
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
