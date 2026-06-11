"""OpenCL-specific CrossGL AST lowering for target backend code generation."""

from dataclasses import dataclass
from typing import Tuple

from crosstl.translator.ast import (
    ArrayType,
    AttributeNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
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
}
_STAGE_CONTROL_ATTRIBUTES = {"compute", "workgroup_size", "local_size"}
_EVENT_LOCAL_MEMORY_BUILTINS = {
    "async_work_group_copy",
    "wait_group_events",
}
_SYNTHETIC_BUILTIN_ALIASES = {
    "thread_id": "gl_GlobalInvocationID",
    "block_id": "gl_WorkGroupID",
    "thread_local_id": "gl_LocalInvocationID",
    "block_dim": "gl_WorkGroupSize",
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


def _format_list(values):
    return ", ".join(sorted(set(values)))


def _opencl_target_contracts(cgl_ast):
    functions = [
        function
        for function in getattr(cgl_ast, "functions", []) or []
        if isinstance(function, FunctionNode)
    ]
    called_names = set()
    for function in functions:
        called_names.update(_collect_called_function_names(function))

    contracts = []
    for function in functions:
        if not _is_empty_nonvoid_function(function):
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
