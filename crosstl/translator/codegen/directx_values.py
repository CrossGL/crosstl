"""Constants, errors and helpers shared by the modules of the HLSL generator."""

import re

from ..ast import (
    CooperativeMatrixType,
    NamedType,
)
from .enum_utils import generic_type_parts

# Specialization contract for a workgroup pointer parameter bound to ``nullptr``.
# ``nullptr`` cannot name a groupshared array and real backings always have a
# positive extent, so the pair cannot collide with a concrete contract.
HLSL_NULL_WORKGROUP_POINTER_CONTRACT = ("nullptr", 0)
# DXR system-value intrinsics and their result types (DirectX Raytracing specification, "System value intrinsics").
HLSL_RAY_SYSTEM_VALUE_TYPES = {
    "DispatchRaysIndex": "uint3",
    "DispatchRaysDimensions": "uint3",
    "WorldRayOrigin": "float3",
    "WorldRayDirection": "float3",
    "ObjectRayOrigin": "float3",
    "ObjectRayDirection": "float3",
    "RayTMin": "float",
    "RayTCurrent": "float",
    "RayFlags": "uint",
    "InstanceIndex": "uint",
    "InstanceID": "uint",
    "GeometryIndex": "uint",
    "PrimitiveIndex": "uint",
    "HitKind": "uint",
    "ObjectToWorld": "float3x4",
    "ObjectToWorld3x4": "float3x4",
    "ObjectToWorld4x3": "float4x3",
    "WorldToObject": "float3x4",
    "WorldToObject3x4": "float3x4",
    "WorldToObject4x3": "float4x3",
}
# Private pointer parameters lower to ``array, int base``.  A Metal rebase such
# as ``x_thread += 8 * i`` becomes ``x_thread_base += int(8 * i)``; the analysis
# tracks the rebased view start as an interval under this tuple key, which can
# never collide with an identifier name in the same interval map.
HLSL_PRIVATE_POINTER_OFFSET_KEY = "private-pointer-view-offset"
HLSL_PRIVATE_POINTER_MAX_REBASE_ITERATIONS = 4096
HLSL_PRIVATE_POINTER_MAX_REBASE_SIMULATION_STEPS = 1 << 16
HLSL_PRIVATE_POINTER_BASE_MAX = (1 << 31) - 1


class DirectXUnresolvedSourceTypeError(ValueError):
    """Diagnostic exception for source expressions with missing type metadata."""

    project_diagnostic_code = "project.translate.unresolved-source-type"
    missing_capabilities = ("directx.type-inference",)

    def __init__(self, message):
        super().__init__(message)


class DirectXContextualConversionError(ValueError):
    """Raised when a source-defined numeric conversion has no HLSL equivalent."""

    project_diagnostic_code = (
        "project.translate.directx-contextual-conversion-unrepresentable"
    )
    missing_capabilities = ("directx.contextual-conversion-lowering",)

    def __init__(
        self,
        message,
        *,
        source_type,
        target_type,
        reason,
        source_location=None,
    ):
        super().__init__(message)
        self.source_type = source_type
        self.target_type = target_type
        self.reason = reason
        self.source_location = source_location


class DirectXStructConversionError(ValueError):
    """Raised when a registered source structure conversion is not faithful."""

    project_diagnostic_code = (
        "project.translate.directx-struct-construction-unsupported"
    )
    missing_capabilities = ("directx.struct-conversion-construction",)

    def __init__(
        self,
        message,
        *,
        destination_type=None,
        source_type=None,
        conversion_kind=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.destination_type = destination_type
        self.source_type = source_type
        self.conversion_kind = conversion_kind
        self.reason = reason
        self.source_location = source_location


class DirectXBFloat16UnsupportedError(ValueError):
    """Raised when HLSL cannot preserve source bfloat16 semantics exactly."""

    project_diagnostic_code = "project.translate.directx-bfloat16-unsupported"
    missing_capabilities = ("directx.exact-bfloat16-lowering",)

    def __init__(
        self,
        message,
        *,
        target_profile=None,
        operation=None,
        source_type=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.target_profile = target_profile
        self.operation = operation
        self.source_type = source_type
        self.reason = reason
        self.source_location = source_location


class DirectXNative16BitUnsupportedError(ValueError):
    """Raised when a DirectX profile cannot preserve exact 16-bit types."""

    project_diagnostic_code = "project.translate.directx-native-16bit-unsupported"
    missing_capabilities = ("directx.native-16bit-types",)

    def __init__(
        self,
        message,
        *,
        target_profile,
        source_type,
        mapped_type,
        reason,
        source_location=None,
    ):
        super().__init__(message)
        self.target_profile = target_profile
        self.source_type = source_type
        self.mapped_type = mapped_type
        self.reason = reason
        self.source_location = source_location


class DirectXRayTracingStageProfileError(ValueError):
    """Raised when a DirectX profile below Direct3D 12 receives a DXR stage."""

    project_diagnostic_code = "project.translate.directx-ray-tracing-stage-unsupported"
    missing_capabilities = ("directx.ray-tracing",)

    def __init__(self, message, *, target_profile, shader_type, shader_attribute):
        super().__init__(message)
        self.target_profile = target_profile
        self.shader_type = shader_type
        self.shader_attribute = shader_attribute


class DirectXWaveSizeProfileError(ValueError):
    """Raised when a DirectX profile below Direct3D 12 receives WaveSize."""

    project_diagnostic_code = "project.translate.directx-wave-size-unsupported"
    missing_capabilities = ("directx.wave-size",)

    def __init__(self, message, *, target_profile, wave_size, shader_model):
        super().__init__(message)
        self.target_profile = target_profile
        self.wave_size = wave_size
        self.shader_model = shader_model


class DirectXMeshStageProfileError(ValueError):
    """Raised when a DirectX profile below Direct3D 12 receives a mesh stage."""

    project_diagnostic_code = "project.translate.directx-mesh-stage-unsupported"
    missing_capabilities = ("directx.mesh-shading",)

    def __init__(self, message, *, target_profile, shader_type, shader_attribute):
        super().__init__(message)
        self.target_profile = target_profile
        self.shader_type = shader_type
        self.shader_attribute = shader_attribute


class DirectXMeshOutputCountsError(ValueError):
    """Raised when a mesh stage does not call SetMeshOutputCounts exactly once, outside loops and outside
    thread-varying control flow."""

    project_diagnostic_code = "project.translate.directx-mesh-output-counts-unsupported"
    missing_capabilities = ("directx.mesh-output-counts",)


class DirectXWorkgroupSizeError(ValueError):
    """Raised when an HLSL compute workgroup size is not representable."""

    project_diagnostic_code = "project.translate.workgroup-size-invalid"
    missing_capabilities = ("execution.workgroup-size-specialization",)
    check_kind = "execution-specialization"

    def __init__(self, message, *, workgroup_size, reason):
        super().__init__(message)
        self.workgroup_size = tuple(workgroup_size)
        self.reason = reason


class DirectXSoftwareSubgroupError(ValueError):
    """Raised when barrier-backed HLSL subgroup emulation is not provably safe."""

    project_diagnostic_code = "project.translate.directx-software-subgroup-invalid"
    missing_capabilities = ("directx.software-subgroup-lowering",)
    check_kind = "execution-specialization"

    def __init__(
        self,
        message,
        *,
        software_subgroup_width=None,
        workgroup_size=None,
        operation=None,
        reason,
        source_location=None,
    ):
        super().__init__(message)
        self.software_subgroup_width = software_subgroup_width
        self.workgroup_size = (
            tuple(workgroup_size) if workgroup_size is not None else None
        )
        self.operation = operation
        self.reason = reason
        self.source_location = source_location


class DirectXEntryPointSelectionError(ValueError):
    """Raised when a requested standalone DirectX entry cannot be selected."""

    project_diagnostic_code = "project.translate.directx-entry-point-unavailable"
    missing_capabilities = ("artifact.entry-point-selection",)

    def __init__(
        self,
        message,
        *,
        entry_point=None,
        available_entry_points=None,
        reason=None,
        stage=None,
        source_location=None,
    ):
        super().__init__(message)
        self.entry_point = entry_point
        self.available_entry_points = tuple(available_entry_points or ())
        self.reason = reason
        self.stage = stage
        self.source_location = source_location


class DirectXTextureOffsetError(ValueError):
    """Raised when an HLSL texture offset cannot preserve source semantics."""

    project_diagnostic_code = (
        "project.translate.directx-texture-offset-immediate-required"
    )
    missing_capabilities = ("directx.runtime-texture-offset-lowering",)

    def __init__(
        self,
        message,
        *,
        operation,
        target_method,
        blocking_expression,
        function_name,
        reason,
        source_location=None,
    ):
        super().__init__(message)
        self.operation = operation
        self.target_method = target_method
        self.blocking_expression = blocking_expression
        self.function_name = function_name
        self.reason = reason
        self.source_location = source_location
        self.operation_context = {
            "sourceOperation": operation,
            "targetMethod": target_method,
            "function": function_name,
            "blockingExpression": blocking_expression,
            "reason": reason,
        }


class DirectXForInIterableError(ValueError):
    """Raised when a CrossGL for-in binding has no faithful HLSL lowering."""

    project_diagnostic_code = "project.translate.directx-for-in-iterable-unsupported"
    missing_capabilities = ("directx.fixed-array-for-in-lowering",)

    def __init__(
        self,
        message,
        *,
        pattern=None,
        iterable_type=None,
        binding_type=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.pattern = pattern
        self.iterable_type = iterable_type
        self.binding_type = binding_type
        self.reason = reason
        self.source_location = source_location


class DirectXTrailingZeroBuiltinError(ValueError):
    """Raised when a Clang trailing-zero builtin has no faithful HLSL lowering."""

    project_diagnostic_code = "project.translate.directx-trailing-zero-unsupported"
    missing_capabilities = ("directx.trailing-zero-builtin-lowering",)

    def __init__(
        self,
        message,
        *,
        builtin_name=None,
        operand_type=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.builtin_name = builtin_name
        self.operand_type = operand_type
        self.reason = reason
        self.source_location = source_location


class DirectXSemanticArraySizeError(ValueError):
    """Diagnostic exception for stage semantic arrays with unknown extents."""

    project_diagnostic_code = "project.translate.directx-semantic-array-size"
    missing_capabilities = ("directx.semantic-array-size",)

    def __init__(self, message):
        super().__init__(message)


class DirectXAggregateInitializerError(ValueError):
    """Raised when an aggregate initializer has no faithful HLSL lowering."""

    project_diagnostic_code = "project.translate.directx-aggregate-initializer-invalid"
    missing_capabilities = ("directx.aggregate-initializer-lowering",)

    def __init__(
        self,
        message,
        *,
        expected_type=None,
        element_count=None,
        expected_element_count=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.expected_type = expected_type
        self.element_count = element_count
        self.expected_element_count = expected_element_count
        self.reason = reason
        self.source_location = source_location


class DirectXCompileTimeGlobalError(ValueError):
    """Raised when a file-scope constant has no faithful HLSL initializer."""

    project_diagnostic_code = "project.translate.directx-compile-time-global-invalid"
    missing_capabilities = ("directx.compile-time-global-initializer",)

    def __init__(
        self,
        message,
        *,
        variable_name=None,
        expression_kind=None,
        detail=None,
        operation=None,
        operand_type=None,
        result_type=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.variable_name = variable_name
        self.expression_kind = expression_kind
        self.detail = detail
        self.operation = operation
        self.operand_type = operand_type
        self.result_type = result_type
        self.reason = reason
        self.source_location = source_location


class DirectXAggregateConditionalError(ValueError):
    """Raised when an aggregate conditional has no faithful HLSL lowering."""

    project_diagnostic_code = (
        "project.translate.directx-aggregate-conditional-unsupported"
    )
    missing_capabilities = ("directx.aggregate-conditional-lowering",)

    def __init__(
        self,
        message,
        *,
        aggregate_type=None,
        target="HLSL",
        context=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.aggregate_type = aggregate_type
        self.target = target
        self.context = context
        self.reason = reason
        self.source_location = source_location


class DirectXUnionLayoutError(ValueError):
    """Raised when source union storage cannot be represented faithfully in HLSL."""

    project_diagnostic_code = "project.translate.directx-union-layout-unsupported"
    missing_capabilities = ("directx.union-storage-aliasing",)

    def __init__(
        self,
        message,
        *,
        union_name=None,
        member_name=None,
        member_type=None,
        size=None,
        alignment=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.union_name = union_name
        self.member_name = member_name
        self.member_type = member_type
        self.size = size
        self.alignment = alignment
        self.reason = reason
        self.source_location = source_location


class DirectXMappedOverloadError(ValueError):
    """Raised when source overload binding cannot survive HLSL type mapping."""

    project_diagnostic_code = "project.translate.directx-mapped-overload-ambiguous"
    missing_capabilities = ("directx.mapped-overload-resolution",)

    def __init__(
        self,
        message,
        *,
        function_name=None,
        argument_types=None,
        candidates=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.function_name = function_name
        self.argument_types = tuple(argument_types or ())
        self.candidates = tuple(candidates or ())
        self.reason = reason
        self.source_location = source_location


class DirectXRecursiveFunctionError(ValueError):
    """Raised when HLSL would retain an unsupported recursive call."""

    project_diagnostic_code = "project.translate.directx-recursion-unsupported"
    missing_capabilities = ("directx.function-recursion",)

    def __init__(
        self,
        message,
        *,
        function_name=None,
        cycle=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.function_name = function_name
        self.cycle = tuple(cycle or ())
        self.reason = reason
        self.source_location = source_location


class DirectXMinimumPrecisionIntegerArithmeticError(ValueError):
    """Raised when minimum-precision integer arithmetic cannot be lowered."""

    project_diagnostic_code = (
        "project.translate.directx-minimum-precision-integer-arithmetic-unsupported"
    )
    missing_capabilities = ("directx.minimum-precision-integer-arithmetic-lowering",)

    def __init__(
        self,
        message,
        *,
        operator=None,
        left_type=None,
        right_type=None,
        expected_type=None,
        context=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.operator = operator
        self.left_type = left_type
        self.right_type = right_type
        self.expected_type = expected_type
        self.context = context
        self.reason = reason
        self.source_location = source_location


class DirectXNative16BitArithmeticError(ValueError):
    """Raised when native 16-bit arithmetic cannot be represented in HLSL."""

    project_diagnostic_code = (
        "project.translate.directx-native-16bit-arithmetic-unrepresentable"
    )
    missing_capabilities = ("directx.native-16bit-arithmetic-lowering",)

    def __init__(
        self,
        message,
        *,
        operator=None,
        left_type=None,
        right_type=None,
        operation_type=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.operator = operator
        self.left_type = left_type
        self.right_type = right_type
        self.operation_type = operation_type
        self.reason = reason
        self.source_location = source_location


class DirectXBooleanCompoundAssignmentError(ValueError):
    """Raised when a boolean compound assignment has no faithful HLSL form."""

    project_diagnostic_code = (
        "project.translate.directx-boolean-compound-assignment-unrepresentable"
    )
    missing_capabilities = ("directx.boolean-compound-assignment-lowering",)

    def __init__(
        self,
        message,
        *,
        operator=None,
        target_type=None,
        right_type=None,
        operation_type=None,
        context=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.operator = operator
        self.target_type = target_type
        self.right_type = right_type
        self.operation_type = operation_type
        self.context = context
        self.reason = reason
        self.source_location = source_location


class DirectXComplexCompoundAssignmentError(ValueError):
    """Raised when a complex compound assignment has no faithful HLSL form."""

    project_diagnostic_code = (
        "project.translate.directx-complex-compound-assignment-unrepresentable"
    )
    missing_capabilities = ("directx.complex-compound-assignment-lowering",)

    def __init__(
        self,
        message,
        *,
        operator=None,
        target_type=None,
        right_type=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.operator = operator
        self.target_type = target_type
        self.right_type = right_type
        self.reason = reason
        self.source_location = source_location


class DirectXBooleanOrderedIntrinsicError(ValueError):
    """Raised when boolean min/max operands have no faithful HLSL lowering."""

    project_diagnostic_code = (
        "project.translate.directx-boolean-ordered-intrinsic-unrepresentable"
    )
    missing_capabilities = ("directx.boolean-ordered-intrinsic-lowering",)

    def __init__(
        self,
        message,
        *,
        operation=None,
        operand_types=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.operation = operation
        self.operand_types = tuple(operand_types or ())
        self.reason = reason
        self.source_location = source_location


class DirectXCopySignError(ValueError):
    """Raised when ``copysign`` has no exact 32-bit HLSL lowering."""

    project_diagnostic_code = "project.translate.directx-copysign-unrepresentable"
    missing_capabilities = ("directx.copysign-lowering",)

    def __init__(
        self,
        message,
        *,
        operation="copysign",
        operand_types=None,
        result_type=None,
        target_profile=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.operation = operation
        self.operand_types = tuple(operand_types or ())
        self.result_type = result_type
        self.target_profile = target_profile
        self.reason = reason
        self.source_location = source_location


class DirectXSignBitError(ValueError):
    """Raised when ``signbit`` has no exact 32-bit HLSL lowering."""

    project_diagnostic_code = "project.translate.directx-signbit-unrepresentable"
    missing_capabilities = ("directx.signbit-lowering",)

    def __init__(
        self,
        message,
        *,
        operation="signbit",
        operand_type=None,
        result_type=None,
        target_profile=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.operation = operation
        self.operand_type = operand_type
        self.result_type = result_type
        self.target_profile = target_profile
        self.reason = reason
        self.source_location = source_location


class DirectXInverseHyperbolicError(ValueError):
    """Raised when an inverse-hyperbolic call has no faithful HLSL lowering."""

    project_diagnostic_code = (
        "project.translate.directx-inverse-hyperbolic-unrepresentable"
    )
    missing_capabilities = ("directx.inverse-hyperbolic-lowering",)

    def __init__(
        self,
        message,
        *,
        operation=None,
        operand_type=None,
        result_type=None,
        target_profile=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.operation = operation
        self.operand_type = operand_type
        self.result_type = result_type
        self.target_profile = target_profile
        self.reason = reason
        self.source_location = source_location


class DirectXPrivatePointerParameterError(ValueError):
    """Raised when a private pointer cannot become a fixed HLSL array."""

    project_diagnostic_code = "project.translate.directx-private-pointer-unsupported"
    missing_capabilities = ("directx.private-pointer-parameter-lowering",)

    def __init__(
        self,
        message,
        *,
        function_name=None,
        parameter_name=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.function_name = function_name
        self.parameter_name = parameter_name
        self.reason = reason
        self.source_location = source_location


class DirectXWorkgroupPointerError(ValueError):
    """Raised when a workgroup pointer has no concrete HLSL array contract."""

    project_diagnostic_code = "project.translate.directx-workgroup-pointer-unsupported"
    missing_capabilities = ("directx.workgroup-pointer-lowering",)

    def __init__(
        self,
        message,
        *,
        function_name=None,
        parameter_name=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.function_name = function_name
        self.parameter_name = parameter_name
        self.reason = reason
        self.source_location = source_location


class DirectXResourcePointerParameterError(ValueError):
    """Raised when a storage-pointer parameter has no faithful HLSL pair."""

    project_diagnostic_code = (
        "project.translate.directx-resource-pointer-parameter-unsupported"
    )
    missing_capabilities = ("directx.resource-pointer-parameter-lowering",)

    def __init__(
        self,
        message,
        *,
        function_name=None,
        parameter_name=None,
        address_space=None,
        expected_access=None,
        actual_access=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.function_name = function_name
        self.parameter_name = parameter_name
        self.address_space = address_space
        self.expected_access = expected_access
        self.actual_access = actual_access
        self.reason = reason
        self.source_location = source_location


class DirectXResourcePointerArrayError(ValueError):
    """Raised when a storage-pointer array has no faithful HLSL lowering."""

    project_diagnostic_code = (
        "project.translate.directx-resource-pointer-array-unsupported"
    )
    missing_capabilities = ("directx.resource-pointer-array-lowering",)

    def __init__(
        self,
        message,
        *,
        function_name=None,
        array_name=None,
        address_space=None,
        backing_root=None,
        conflicting_root=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.function_name = function_name
        self.array_name = array_name
        self.address_space = address_space
        self.backing_root = backing_root
        self.conflicting_root = conflicting_root
        self.reason = reason
        self.source_location = source_location


class DirectXCooperativeMatrixUnsupportedError(ValueError):
    """Raised when cooperative matrix semantics cannot be preserved in HLSL."""

    project_diagnostic_code = "project.translate.directx-cooperative-matrix-unsupported"
    missing_capabilities = ("directx.cooperative-matrix-lowering",)

    def __init__(
        self,
        message,
        *,
        operation=None,
        matrix_type=None,
        fragment_layout=None,
        subgroup_size=None,
        elements_per_lane=None,
        fragment_provenance=None,
        fragment_mapping=None,
        fragment_mapping_provenance=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        if fragment_layout is None:
            fragment_layout = getattr(matrix_type, "fragment_layout", None)
        if subgroup_size is None:
            subgroup_size = getattr(matrix_type, "subgroup_size", None)
        if elements_per_lane is None:
            elements_per_lane = getattr(matrix_type, "elements_per_lane", None)
        if fragment_provenance is None:
            fragment_provenance = getattr(matrix_type, "fragment_provenance", None)
        if fragment_mapping is None:
            fragment_mapping = getattr(matrix_type, "fragment_mapping", None)
        if fragment_mapping_provenance is None:
            fragment_mapping_provenance = getattr(
                matrix_type, "fragment_mapping_provenance", None
            )

        self.operation = operation
        self.matrix_type = matrix_type
        self.fragment_layout = fragment_layout
        self.subgroup_size = subgroup_size
        self.elements_per_lane = elements_per_lane
        self.fragment_provenance = fragment_provenance
        self.fragment_mapping = fragment_mapping
        self.fragment_mapping_provenance = fragment_mapping_provenance
        self.reason = reason
        details = {
            "fragmentLayout": _directx_cooperative_matrix_detail_value(fragment_layout),
            "subgroupSize": _directx_cooperative_matrix_detail_value(subgroup_size),
            "elementsPerLane": _directx_cooperative_matrix_detail_value(
                elements_per_lane
            ),
            "fragmentProvenance": _directx_cooperative_matrix_detail_value(
                fragment_provenance
            ),
            "fragmentMapping": _directx_cooperative_matrix_detail_value(
                fragment_mapping
            ),
            "fragmentMappingProvenance": _directx_cooperative_matrix_detail_value(
                fragment_mapping_provenance
            ),
        }
        self.details = {
            name: value for name, value in details.items() if value is not None
        }
        self.source_location = source_location


def _directx_cooperative_matrix_detail_value(value):
    if isinstance(value, bool) or value is None or value == "":
        return None
    literal_value = getattr(value, "value", None)
    if literal_value is not None:
        return literal_value
    name = getattr(value, "name", None)
    if name is not None:
        return name
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def _directx_cooperative_matrix_fragment_contract(matrix_type):
    """Return the lane-fragment contract retained by a cooperative matrix type."""
    return {
        "fragment_layout": getattr(matrix_type, "fragment_layout", None),
        "subgroup_size": getattr(matrix_type, "subgroup_size", None),
        "elements_per_lane": getattr(matrix_type, "elements_per_lane", None),
        "fragment_provenance": getattr(matrix_type, "fragment_provenance", None),
        "fragment_mapping": getattr(matrix_type, "fragment_mapping", None),
        "fragment_mapping_provenance": getattr(
            matrix_type, "fragment_mapping_provenance", None
        ),
    }


def _directx_cooperative_matrix_contract_type(matrix_type):
    if isinstance(matrix_type, CooperativeMatrixType):
        return matrix_type

    base_name, generic_args = generic_type_parts(str(matrix_type).strip())
    if base_name.rsplit("::", 1)[-1] != "CooperativeMatrix" or not (
        3 <= len(generic_args) <= 12
    ):
        return matrix_type

    defaults = [
        "subgroup",
        "unspecified",
        "unspecified",
        "unspecified",
        "unspecified",
        "unspecified",
        "unspecified",
        "unspecified",
        "unspecified",
    ]
    generic_args = [*generic_args, *defaults[len(generic_args) - 3 :]]

    def dimension(value):
        value = str(value).strip()
        return int(value) if re.fullmatch(r"[+-]?\d+", value) else NamedType(value)

    def optional_label(value):
        value = str(value).strip()
        return None if value == "unspecified" else value

    def optional_dimension(value):
        value = str(value).strip()
        return None if value == "unspecified" else dimension(value)

    try:
        return CooperativeMatrixType(
            NamedType(str(generic_args[0]).strip()),
            dimension(generic_args[1]),
            dimension(generic_args[2]),
            str(generic_args[3]).strip(),
            str(generic_args[4]).strip(),
            str(generic_args[5]).strip(),
            optional_label(generic_args[6]),
            optional_dimension(generic_args[7]),
            optional_dimension(generic_args[8]),
            optional_label(generic_args[9]),
            optional_label(generic_args[10]),
            optional_label(generic_args[11]),
        )
    except ValueError:
        return matrix_type


def _directx_cooperative_matrix_fragment_contract_message(contract):
    """Render retained requirements without supplying target-specific defaults."""

    def render(value):
        normalized = _directx_cooperative_matrix_detail_value(value)
        return "unspecified" if normalized is None else str(normalized)

    return (
        "required fragment contract: "
        f"fragment_layout={render(contract['fragment_layout'])}, "
        f"subgroup_size={render(contract['subgroup_size'])}, "
        f"elements_per_lane={render(contract['elements_per_lane'])}, "
        f"fragment_provenance={render(contract['fragment_provenance'])}, "
        f"fragment_mapping={render(contract['fragment_mapping'])}, "
        "fragment_mapping_provenance="
        f"{render(contract['fragment_mapping_provenance'])}"
    )


class DirectXAtomicFenceLoweringError(ValueError):
    """Raised when a CrossGL atomic fence has no equivalent HLSL contract."""

    project_diagnostic_code = "project.translate.directx-atomic-fence-unsupported"
    missing_capabilities = ("directx.atomic-thread-fence-contract-lowering",)

    def __init__(
        self,
        reason,
        *,
        memory_flags=None,
        memory_order=None,
        thread_scope=None,
        source_location=None,
    ):
        self.reason = reason
        self.memory_flags = memory_flags
        self.memory_order = memory_order
        self.thread_scope = thread_scope
        self.source_location = source_location
        contract = (
            f"flags={memory_flags or '<missing>'}, "
            f"order={memory_order or '<missing>'}, "
            f"scope={thread_scope or '<missing>'}"
        )
        super().__init__(
            "Cannot lower CrossGL atomicThreadFence to HLSL without changing "
            f"its semantics ({contract}): {reason.replace('-', ' ')}"
        )


class DirectXSpecializationConstantError(ValueError):
    """Raised when a deferred specialization input cannot be represented in HLSL."""

    project_diagnostic_code = (
        "project.translate.directx-specialization-constant-unsupported"
    )
    missing_capabilities = ("directx.specialization-constant-variant",)

    def __init__(
        self,
        message,
        *,
        constant_name=None,
        constant_id=None,
        constant_type=None,
        reason=None,
        source_location=None,
        conflicting_name=None,
        conflicting_id=None,
        conflicting_type=None,
    ):
        super().__init__(message)
        self.constant_name = constant_name
        self.constant_id = constant_id
        self.constant_type = constant_type
        self.reason = reason
        self.source_location = source_location
        self.conflicting_name = conflicting_name
        self.conflicting_id = conflicting_id
        self.conflicting_type = conflicting_type


class _DirectXResourceRegisterAllocator(dict):
    """Track resource allocations with merged occupancy ranges per namespace."""

    def __init__(self):
        super().__init__()
        self._allocation_keys = set()
        self._occupied_ranges = {}
        self.relocation_cursors = {}
        self.range_inspection_count = 0

    def _lower_bound(self, ranges, start):
        lower = 0
        upper = len(ranges)
        while lower < upper:
            middle = (lower + upper) // 2
            self.range_inspection_count += 1
            if ranges[middle][0] < start:
                lower = middle + 1
            else:
                upper = middle
        return lower

    def _occupied_range_conflicts(self, namespace, start, end):
        ranges = self._occupied_ranges.get(namespace, ())
        index = self._lower_bound(ranges, start)
        if index:
            self.range_inspection_count += 1
            if ranges[index - 1][1] >= start:
                return True
        if index < len(ranges):
            self.range_inspection_count += 1
            if ranges[index][0] <= end:
                return True
        return False

    def range_is_available(self, namespace, start, end):
        return not self._occupied_range_conflicts(namespace, start, end)

    def has_allocation(self, namespace, start, end, name):
        return (namespace, start, end, name) in self._allocation_keys

    def conflicting_allocation(self, namespace, start, end):
        if not self._occupied_range_conflicts(namespace, start, end):
            return None
        for record in self.get(namespace, ()):
            self.range_inspection_count += 1
            if start <= record[1] and record[0] <= end:
                return record
        return None

    def next_available(self, namespace, start, count):
        ranges = self._occupied_ranges.get(namespace, ())
        binding = start
        index = self._lower_bound(ranges, binding)
        if index:
            self.range_inspection_count += 1
            previous_end = ranges[index - 1][1]
            if previous_end >= binding:
                binding = previous_end + 1
        while index < len(ranges):
            self.range_inspection_count += 1
            used_start, used_end = ranges[index]
            if used_start > binding + count - 1:
                break
            binding = max(binding, used_end + 1)
            index += 1
        return binding

    def add_allocation(self, namespace, start, end, name, provenance):
        self.setdefault(namespace, []).append((start, end, name, provenance))
        self._allocation_keys.add((namespace, start, end, name))

        ranges = self._occupied_ranges.setdefault(namespace, [])
        index = self._lower_bound(ranges, start)
        merged_start = start
        merged_end = end
        if index:
            self.range_inspection_count += 1
            previous_start, previous_end = ranges[index - 1]
            if previous_end + 1 >= merged_start:
                merged_start = previous_start
                merged_end = max(merged_end, previous_end)
                index -= 1
                del ranges[index]
        while index < len(ranges):
            self.range_inspection_count += 1
            used_start, used_end = ranges[index]
            if used_start > merged_end + 1:
                break
            merged_end = max(merged_end, used_end)
            del ranges[index]
        ranges.insert(index, (merged_start, merged_end))

    def advance_relocation_cursor(self, namespace, start, count):
        self.relocation_cursors[namespace] = max(
            self.relocation_cursors.get(namespace, 0), start + count
        )
