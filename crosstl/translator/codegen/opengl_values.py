"""Constants, errors and helpers shared by the modules of the GLSL generator."""

import re

from ..ast import (
    CooperativeMatrixType,
    NamedType,
)
from ..semantics.helpers import float_to_integer_helper_names
from .enum_utils import generic_type_parts

# Opens the branch a SPIR-V compilation of the emitted GLSL takes (GL_ARB_gl_spirv, GL_KHR_vulkan_glsl).
GLSL_SPIRV_GUARD = "#if defined(GL_SPIRV) || defined(VULKAN)\n"
GLSL_FLOAT_TO_INTEGER_RULE_HELPERS = float_to_integer_helper_names()


class OpenGLTemplateTypeError(ValueError):
    """Raised when OpenGL generation would emit an unresolved template type."""

    project_diagnostic_code = "project.translate.opengl-template-type-unresolved"
    missing_capabilities = ("template.specialization",)

    def __init__(self, message, *, unresolved_parameter=None, enclosing_type=None):
        super().__init__(message)
        self.unresolved_parameter = unresolved_parameter
        self.enclosing_generated_type = enclosing_type


class OpenGLWorkgroupSizeError(ValueError):
    """Raised when an OpenGL compute workgroup size is not representable."""

    project_diagnostic_code = "project.translate.workgroup-size-invalid"
    missing_capabilities = ("execution.workgroup-size-specialization",)
    check_kind = "execution-specialization"

    def __init__(self, message, *, workgroup_size, reason):
        super().__init__(message)
        self.workgroup_size = tuple(workgroup_size)
        self.reason = reason


class OpenGLSubgroupWidthError(ValueError):
    """Raised when an exact OpenGL subgroup-width guard is not representable."""

    project_diagnostic_code = "project.translate.subgroup-width-invalid"
    missing_capabilities = ("execution.subgroup-width-specialization",)
    check_kind = "execution-specialization"

    def __init__(self, message, *, subgroup_width=None, reason=None):
        super().__init__(message)
        self.subgroup_width = subgroup_width
        self.reason = reason


class OpenGLSoftwareSubgroupError(ValueError):
    """Raised when explicit OpenGL software subgroup lowering is not provable."""

    project_diagnostic_code = "project.translate.opengl-software-subgroup-invalid"
    missing_capabilities = ("execution.software-subgroup",)
    check_kind = "execution-specialization"

    def __init__(
        self,
        message,
        *,
        software_subgroup_width=None,
        workgroup_size=None,
        operation=None,
        reason=None,
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


class OpenGLEntryPointSelectionError(ValueError):
    """Raised when a requested standalone OpenGL entry cannot be selected."""

    project_diagnostic_code = "project.translate.opengl-entry-point-unavailable"
    missing_capabilities = ("artifact.entry-point-selection",)

    def __init__(
        self,
        message,
        *,
        entry_point=None,
        available_entry_points=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.entry_point = entry_point
        self.available_entry_points = tuple(available_entry_points or ())
        self.reason = reason
        self.source_location = source_location


class OpenGLMappedOverloadError(ValueError):
    """Raised when source overload binding cannot survive GLSL type mapping."""

    project_diagnostic_code = "project.translate.opengl-mapped-overload-ambiguous"
    missing_capabilities = ("opengl.mapped-overload-resolution",)

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


class OpenGLRecursiveFunctionError(ValueError):
    """Raised when emitted GLSL helpers contain a recursive call cycle."""

    project_diagnostic_code = "project.translate.opengl-recursion-unsupported"
    missing_capabilities = ("opengl.function-recursion",)

    def __init__(
        self,
        message,
        *,
        cycle=None,
        source_edge=None,
        source_location=None,
    ):
        super().__init__(message)
        self.cycle = tuple(cycle or ())
        self.reason = "recursive-call-cycle"
        self.source_edge = tuple(source_edge or ())
        self.source_location = source_location


class OpenGLAggregateInitializerError(ValueError):
    """Raised when a bare aggregate cannot be represented safely in GLSL."""

    project_diagnostic_code = "project.translate.opengl-aggregate-initializer-invalid"
    missing_capabilities = ("opengl.aggregate-initializer-lowering",)

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


class OpenGLScalarConversionError(ValueError):
    """Raised when a source scalar conversion has no faithful GLSL form."""

    project_diagnostic_code = "project.translate.opengl-scalar-conversion-unsupported"
    missing_capabilities = ("opengl.scalar-conversion-lowering",)

    def __init__(
        self,
        message,
        *,
        source_type=None,
        target_type=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.source_type = source_type
        self.target_type = target_type
        self.reason = reason
        self.source_location = source_location


class OpenGLArithmeticConversionError(ValueError):
    """Raised when source arithmetic has no faithful GLSL conversion plan."""

    project_diagnostic_code = "project.translate.opengl-arithmetic-conversion-invalid"
    missing_capabilities = ("opengl.arithmetic-conversion-lowering",)

    def __init__(
        self,
        message,
        *,
        operator=None,
        operand_types=None,
        attempted_common_type=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.operator = operator
        self.operand_types = tuple(operand_types or ())
        self.attempted_common_type = attempted_common_type
        self.common_type = attempted_common_type
        self.reason = reason
        self.source_location = source_location


class OpenGLBooleanCompoundAssignmentError(ValueError):
    """Raised when a boolean compound assignment cannot be lowered safely."""

    project_diagnostic_code = (
        "project.translate.opengl-boolean-compound-assignment-invalid"
    )
    missing_capabilities = ("opengl.boolean-compound-assignment-lowering",)

    def __init__(
        self,
        message,
        *,
        operator=None,
        target=None,
        target_type=None,
        right_type=None,
        operation_type=None,
        context=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.operator = operator
        self.target = target
        self.target_type = target_type
        self.right_type = right_type
        self.operation_type = operation_type
        self.context = context
        self.reason = reason
        self.source_location = source_location


class OpenGLBooleanOrderedIntrinsicError(ValueError):
    """Raised when boolean min/max operands have no faithful GLSL lowering."""

    project_diagnostic_code = (
        "project.translate.opengl-boolean-ordered-intrinsic-unrepresentable"
    )
    missing_capabilities = ("opengl.boolean-ordered-intrinsic-lowering",)

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


class OpenGLTrailingZeroBuiltinError(ValueError):
    """Raised when a Clang trailing-zero builtin has no faithful GLSL lowering."""

    project_diagnostic_code = "project.translate.opengl-trailing-zero-unsupported"
    missing_capabilities = ("opengl.trailing-zero-builtin-lowering",)

    def __init__(
        self,
        message,
        *,
        builtin_name=None,
        operand_type=None,
        argument_count=None,
        target_profile=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.builtin_name = builtin_name
        self.operand_type = operand_type
        self.argument_count = argument_count
        self.target_profile = target_profile
        self.reason = reason
        self.source_location = source_location


class OpenGLCopySignError(ValueError):
    """Raised when ``copysign`` has no exact 32-bit GLSL lowering."""

    project_diagnostic_code = "project.translate.opengl-copysign-unrepresentable"
    missing_capabilities = ("opengl.copysign-lowering",)

    def __init__(
        self,
        message,
        *,
        operation="copysign",
        operand_types=None,
        result_type=None,
        target_profile="glsl-460",
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


class OpenGLFloatingRemainderError(ValueError):
    """Raised when floating remainder has no exact GLSL representation."""

    project_diagnostic_code = "project.translate.opengl-fmod-unrepresentable"
    missing_capabilities = ("opengl.floating-remainder-lowering",)

    def __init__(
        self, message, *, operand_types, target_profile, reason, source_location=None
    ):
        super().__init__(message)
        self.operation = "fmod"
        self.operand_types = tuple(operand_types)
        self.target_profile = target_profile
        self.reason = reason
        self.source_location = source_location


class OpenGLMetalMathError(ValueError):
    """Raised when canonical Metal math cannot be preserved in GLSL."""

    project_diagnostic_code = "project.translate.opengl-metal-math-unrepresentable"
    missing_capabilities = ("opengl.metal-math-lowering",)

    def __init__(
        self,
        message,
        *,
        operation,
        operand_types=(),
        target_profile=None,
        reason,
        source_location=None,
    ):
        super().__init__(message)
        self.operation = operation
        self.operand_types = tuple(operand_types)
        self.target_profile = target_profile
        self.reason = reason
        self.source_location = source_location


class OpenGLSignBitError(ValueError):
    """Raised when ``signbit`` has no exact 32-bit GLSL lowering."""

    project_diagnostic_code = "project.translate.opengl-signbit-unrepresentable"
    missing_capabilities = ("opengl.signbit-lowering",)

    def __init__(
        self,
        message,
        *,
        operation="signbit",
        operand_type=None,
        result_type=None,
        target_profile="glsl-460",
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


class OpenGLIsFiniteError(ValueError):
    """Raised when ``isfinite`` has no exact 32-bit GLSL lowering."""

    project_diagnostic_code = "project.translate.opengl-isfinite-unrepresentable"
    missing_capabilities = ("opengl.isfinite-lowering",)

    def __init__(
        self,
        message,
        *,
        operation="isfinite",
        operand_type=None,
        result_type=None,
        target_profile="glsl-460",
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


class OpenGLCompoundAssignmentError(ValueError):
    """Raised when a source compound assignment has no faithful GLSL form."""

    project_diagnostic_code = "project.translate.opengl-compound-assignment-invalid"
    missing_capabilities = ("opengl.compound-assignment-lowering",)

    def __init__(
        self,
        message,
        *,
        operator=None,
        target=None,
        target_type=None,
        value_type=None,
        common_type=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.operator = operator
        self.target = target
        self.target_type = target_type
        self.value_type = value_type
        self.common_type = common_type
        self.reason = reason
        self.source_location = source_location


class OpenGLComplexArithmeticError(ValueError):
    """Raised when complex arithmetic has no faithful GLSL lowering."""

    project_diagnostic_code = "project.translate.opengl-complex-arithmetic-unsupported"
    missing_capabilities = ("opengl.complex-arithmetic-lowering",)

    def __init__(
        self,
        message,
        *,
        operator=None,
        operand_types=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.operator = operator
        self.operand_types = tuple(operand_types or ())
        self.reason = reason
        self.source_location = source_location


class OpenGLStructConstructionError(ValueError):
    """Raised when a source structure conversion has no faithful GLSL form."""

    project_diagnostic_code = "project.translate.opengl-struct-construction-unsupported"
    missing_capabilities = ("opengl.struct-conversion-construction",)

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


class OpenGLPreciseQualifierError(ValueError):
    """Raised when a GLSL target profile has no ``precise`` qualifier.

    Source arithmetic that must be evaluated as written (precise functions and
    values) needs desktop GLSL 4.00 or GLSL ES 3.20.
    """

    project_diagnostic_code = "project.translate.opengl-precise-unsupported"
    missing_capabilities = ("opengl.precise-qualifier",)

    def __init__(self, message, *, version_line=None):
        super().__init__(message)
        self.version_line = version_line


class OpenGLIndexTypeError(ValueError):
    """Raised when a source subscript has no legal GLSL index representation."""

    project_diagnostic_code = "project.translate.opengl-index-type-unsupported"
    missing_capabilities = ("opengl.index-width-normalization",)

    def __init__(
        self,
        message,
        *,
        index_type=None,
        target_index_type=None,
        target_profile=None,
        indexed_value=None,
        index_expression=None,
        range_status=None,
        source_range=None,
        accepted_range=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.index_type = index_type
        self.target_index_type = target_index_type
        self.target_profile = target_profile
        self.indexed_value = indexed_value
        self.index_expression = index_expression
        self.range_status = range_status
        self.source_range = source_range
        self.accepted_range = accepted_range
        self.reason = reason
        self.source_location = source_location


class OpenGLWorkgroupPointerError(ValueError):
    """Raised when a workgroup pointer has no concrete GLSL array contract."""

    project_diagnostic_code = "project.translate.opengl-workgroup-pointer-unsupported"
    missing_capabilities = ("opengl.workgroup-pointer-lowering",)

    def __init__(
        self,
        message,
        *,
        function_name=None,
        parameter_name=None,
        backing_name=None,
        offset_expression=None,
        materialization_name=None,
        reason=None,
        entry_point=None,
        asserted_range=None,
        source_location=None,
    ):
        super().__init__(message)
        self.function_name = function_name
        self.parameter_name = parameter_name
        self.backing_name = backing_name
        self.offset_expression = offset_expression
        self.materialization_name = materialization_name
        self.reason = reason
        self.entry_point = entry_point
        self.asserted_range = asserted_range
        self.source_location = source_location


class OpenGLPrivatePointerParameterError(ValueError):
    """Raised when a private pointer cannot become a fixed GLSL array."""

    project_diagnostic_code = "project.translate.opengl-private-pointer-unsupported"
    missing_capabilities = ("opengl.private-pointer-parameter-lowering",)

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


class OpenGLStoragePointerError(ValueError):
    """Raised when an SSBO-backed pointer has no faithful GLSL lowering."""

    project_diagnostic_code = "project.translate.opengl-storage-pointer-unsupported"
    missing_capabilities = ("opengl.storage-pointer-lowering",)

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


class OpenGLFixedArrayResourceError(ValueError):
    """Raised when a fixed array cannot bind a runtime storage resource."""

    project_diagnostic_code = (
        "project.translate.opengl-fixed-array-resource-unsupported"
    )
    missing_capabilities = ("opengl.fixed-array-resource-specialization",)

    def __init__(
        self,
        message,
        *,
        function_name=None,
        parameter_name=None,
        resource_names=None,
        fixed_extent=None,
        required_access=None,
        actual_access=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.function_name = function_name
        self.parameter_name = parameter_name
        self.resource_names = tuple(sorted(resource_names or ()))
        self.fixed_extent = fixed_extent
        self.required_access = required_access
        self.actual_access = actual_access
        self.reason = reason
        self.source_location = source_location


class OpenGLReferenceParameterError(ValueError):
    """Raised when a mutable reference call cannot be represented in GLSL."""

    project_diagnostic_code = "project.translate.opengl-reference-argument-invalid"
    missing_capabilities = ("opengl.reference-parameter-lowering",)

    def __init__(
        self,
        message,
        *,
        function_name=None,
        parameter_name=None,
        argument=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.function_name = function_name
        self.parameter_name = parameter_name
        self.argument = argument
        self.reason = reason
        self.source_location = source_location


class OpenGLSpecializationConstantError(ValueError):
    """Raised when a specialization constant has no faithful GLSL contract."""

    project_diagnostic_code = "project.translate.opengl-specialization-constant-invalid"
    missing_capabilities = ("opengl.specialization-constant-lowering",)

    def __init__(
        self,
        message,
        *,
        constant_name=None,
        specialization_id=None,
        constant_type=None,
        conflicting_name=None,
        conflicting_id=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.constant_name = constant_name
        self.specialization_id = specialization_id
        self.constant_type = constant_type
        self.conflicting_name = conflicting_name
        self.conflicting_id = conflicting_id
        self.reason = reason
        self.source_location = source_location


class OpenGLGlobalInitializerError(ValueError):
    """Raised when a runtime-dependent global initializer cannot be lowered."""

    project_diagnostic_code = "project.translate.opengl-global-initializer-invalid"
    missing_capabilities = ("opengl.global-initializer-lowering",)

    def __init__(
        self,
        message,
        *,
        variable_name=None,
        dependencies=None,
        cycle=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.variable_name = variable_name
        self.dependencies = tuple(sorted(dependencies or ()))
        self.cycle = tuple(cycle or ())
        self.reason = reason
        self.source_location = source_location


class OpenGLCompileTimeGlobalError(ValueError):
    """Raised when immutable global data has no GLSL constant expression."""

    project_diagnostic_code = "project.translate.opengl-compile-time-global-invalid"
    missing_capabilities = ("opengl.compile-time-global-lowering",)

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


class OpenGLForInIterableError(ValueError):
    """Raised when a CrossGL for-in binding has no faithful GLSL lowering."""

    project_diagnostic_code = "project.translate.opengl-for-in-iterable-unsupported"
    missing_capabilities = ("opengl.fixed-array-for-in-lowering",)

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


class OpenGLAtomicFenceLoweringError(ValueError):
    """Raised when OpenGL GLSL cannot preserve an atomic fence contract."""

    project_diagnostic_code = "project.translate.opengl-atomic-fence-unsupported"
    missing_capabilities = ("opengl.atomic-thread-fence-contract-lowering",)

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
            "Cannot lower CrossGL atomicThreadFence to OpenGL GLSL without "
            f"changing its semantics ({contract}): {reason.replace('-', ' ')}"
        )


class OpenGLResourceMemoryQualifierError(ValueError):
    """Raised when GLSL cannot preserve a typed resource-memory contract."""

    project_diagnostic_code = (
        "project.translate.opengl-resource-memory-qualifier-unsupported"
    )
    missing_capabilities = ("opengl.resource-memory-qualifier-lowering",)

    def __init__(
        self,
        message,
        *,
        resource_name=None,
        qualifier_kind=None,
        requested_scope=None,
        target_qualifier=None,
        target_scope=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.resource_name = resource_name
        self.qualifier_kind = qualifier_kind
        self.qualifier = qualifier_kind
        self.requested_scope = requested_scope
        self.scope = requested_scope
        self.requested_contract = (
            f"{qualifier_kind}({requested_scope})"
            if qualifier_kind and requested_scope
            else qualifier_kind
        )
        self.target_qualifier = target_qualifier
        self.target_scope = target_scope
        self.target_mapping = (
            f"{target_qualifier}({target_scope})"
            if target_qualifier and target_scope
            else target_qualifier
        )
        self.reason = reason
        self.source_location = source_location


class OpenGLCooperativeMatrixError(ValueError):
    """Raised when a cooperative matrix has no faithful OpenGL lowering."""

    project_diagnostic_code = "project.translate.opengl-cooperative-matrix-unsupported"
    missing_capabilities = ("opengl.cooperative-matrix-lowering",)

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
        self.source_location = source_location
        details = {
            "fragmentLayout": _opengl_cooperative_matrix_detail_value(fragment_layout),
            "subgroupSize": _opengl_cooperative_matrix_detail_value(subgroup_size),
            "elementsPerLane": _opengl_cooperative_matrix_detail_value(
                elements_per_lane
            ),
            "fragmentProvenance": _opengl_cooperative_matrix_detail_value(
                fragment_provenance
            ),
            "fragmentMapping": _opengl_cooperative_matrix_detail_value(
                fragment_mapping
            ),
            "fragmentMappingProvenance": _opengl_cooperative_matrix_detail_value(
                fragment_mapping_provenance
            ),
        }
        self.details = {
            name: value for name, value in details.items() if value is not None
        }


def _opengl_cooperative_matrix_detail_value(value):
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


def _opengl_cooperative_matrix_fragment_contract(matrix_type):
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


def _opengl_cooperative_matrix_contract_type(matrix_type):
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


def _opengl_cooperative_matrix_fragment_contract_message(contract):
    """Render retained requirements without supplying target-specific defaults."""

    def render(value):
        normalized = _opengl_cooperative_matrix_detail_value(value)
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


# GLSL types of the fragment built-ins an entry parameter of another integer type converts from.
GLSL_CONVERTED_FRAGMENT_BUILTIN_TYPES = {
    "gl_FragFullyCoveredNV": "bool",
    "gl_ShadingRateEXT": "int",
    "gl_Layer": "int",
    "gl_ViewportIndex": "int",
}
