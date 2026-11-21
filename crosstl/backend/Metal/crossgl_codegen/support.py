"""Exceptions and patterns shared by the members of the Metal to CrossGL converter."""

import re

_PLAIN_TYPE_NAME = re.compile(r"[A-Za-z_]\w*")


class MetalCallableAliasLoweringError(ValueError):
    """Raised when a callable alias survives as a runtime value."""

    project_diagnostic_code = "project.translate.metal-callable-alias-unsupported"
    missing_capabilities = ("metal.runtime-callable-alias-lowering",)

    def __init__(
        self,
        alias_name,
        signature,
        usage,
        reason,
        source_location=None,
    ):
        self.alias_name = alias_name
        self.signature = signature
        self.usage = usage
        self.reason = reason
        self.source_location = source_location
        super().__init__(
            f"Cannot lower Metal callable alias '{alias_name}' ({signature}) "
            f"used by {usage}: {reason}"
        )


class MetalStaticConstantResolutionError(ValueError):
    """Raised when a referenced Metal static constant cannot be materialized."""

    project_diagnostic_code = "project.translate.metal-static-constant-unresolved"
    missing_capabilities = ("metal.static-constant-materialization",)

    def __init__(self, owner, member, reason):
        self.owner = owner
        self.member = member
        self.reason = reason
        super().__init__(
            f"Cannot materialize Metal static constant {owner}::{member}: {reason}"
        )


class MetalBuiltinOverloadResolutionError(ValueError):
    """Raised when a Metal call cannot be bound to one user overload."""

    project_diagnostic_code = "project.translate.metal-builtin-overload-ambiguous"
    missing_capabilities = ("metal.builtin-overload-resolution",)

    def __init__(self, function_name, argument_types, candidates):
        self.function_name = function_name
        self.argument_types = tuple(argument_types)
        self.candidates = tuple(candidates)
        signature = ", ".join(self.argument_types)
        super().__init__(
            "Cannot resolve Metal built-in/user overload binding for "
            f"{function_name}({signature}); candidates are "
            f"{', '.join(self.candidates)}"
        )


class MetalBuiltinResultTypeResolutionError(ValueError):
    """Raised when a Metal builtin call has no unique source result type."""

    project_diagnostic_code = "project.translate.metal-builtin-result-unresolved"
    missing_capabilities = ("metal.builtin-result-type-inference",)
    maximum_candidates = 8

    def __init__(
        self,
        function_name,
        argument_types,
        candidates,
        reason,
        source_location=None,
    ):
        self.function_name = function_name
        self.argument_types = tuple(argument_types)
        all_candidates = tuple(candidates)
        self.candidates = all_candidates[: self.maximum_candidates]
        self.omitted_candidate_count = max(
            0, len(all_candidates) - len(self.candidates)
        )
        self.reason = reason
        self.source_location = source_location
        signature = ", ".join(self.argument_types)
        candidate_text = ", ".join(self.candidates) or "<none>"
        if self.omitted_candidate_count:
            candidate_text += f", and {self.omitted_candidate_count} more"
        super().__init__(
            "Cannot infer Metal builtin result type for "
            f"{function_name}({signature}): {reason}; viable signatures are "
            f"{candidate_text}"
        )


class MetalPreciseMathLoweringError(ValueError):
    """Raised when a precise Metal math call has no portable implementation."""

    project_diagnostic_code = "project.translate.metal-precise-math-unsupported"
    missing_capabilities = ("metal.precise-math-lowering",)

    def __init__(
        self,
        operation,
        operand_type,
        reason,
        source_location=None,
    ):
        self.operation = operation
        self.operand_type = operand_type
        self.reason = reason
        self.source_location = source_location
        super().__init__(
            f"Cannot preserve Metal precise::{operation} for "
            f"'{operand_type or '<unknown>'}': {reason}"
        )


class MetalRoundLoweringError(ValueError):
    """Raised when source rounding cannot be represented by the portable helper."""

    project_diagnostic_code = "project.translate.metal-round-unsupported"
    missing_capabilities = ("metal.round-lowering",)

    def __init__(self, operand_type, reason, source_location=None):
        self.operand_type = operand_type
        self.reason = reason
        self.source_location = source_location
        super().__init__(
            f"Cannot preserve Metal round for '{operand_type or '<unknown>'}': {reason}"
        )


class MetalDivisionProfileError(ValueError):
    """Raised when a selected division profile cannot be applied faithfully."""

    project_diagnostic_code = "project.translate.metal-division-profile-unsupported"
    missing_capabilities = ("metal.division-profile-lowering",)

    def __init__(self, profile, operand_type, reason, source_location=None):
        self.profile = profile
        self.operand_type = operand_type
        self.reason = reason
        self.source_location = source_location
        super().__init__(
            f"Cannot apply Metal binary32 division profile '{profile}' to "
            f"'{operand_type or '<unknown>'}': {reason}"
        )


class MetalMultiplicationProfileError(ValueError):
    """Raised when an explicit multiplication policy cannot be represented."""

    project_diagnostic_code = (
        "project.translate.metal-multiplication-profile-unsupported"
    )
    missing_capabilities = ("metal.multiplication-profile-lowering",)

    def __init__(self, profile, operand_type, reason, source_location=None):
        self.profile = profile
        self.operand_type = operand_type
        self.reason = reason
        self.source_location = source_location
        super().__init__(
            f"Cannot apply Metal binary32 multiplication profile '{profile}' to "
            f"'{operand_type or '<unknown>'}': {reason}"
        )


class MetalContractionProfileError(ValueError):
    """Raised when a selected floating-point contraction cannot be represented."""

    project_diagnostic_code = "project.translate.metal-contraction-profile-unsupported"
    missing_capabilities = ("metal.contraction-profile-lowering",)

    def __init__(self, profile, operand_type, reason, source_location=None):
        self.profile = profile
        self.operand_type = operand_type
        self.reason = reason
        self.source_location = source_location
        super().__init__(
            f"Cannot apply Metal floating contraction profile '{profile}' to "
            f"'{operand_type or '<unknown>'}': {reason}"
        )


class MetalComparisonProfileError(ValueError):
    """Raised when a selected comparison profile cannot be represented."""

    project_diagnostic_code = "project.translate.metal-comparison-profile-unsupported"
    missing_capabilities = ("metal.comparison-profile-lowering",)

    def __init__(self, profile, operand_type, reason, source_location=None):
        self.profile = profile
        self.operand_type = operand_type
        self.reason = reason
        self.source_location = source_location
        super().__init__(
            f"Cannot apply Metal binary32 comparison profile '{profile}' to "
            f"'{operand_type or '<unknown>'}': {reason}"
        )


class MetalAdditiveProfileError(ValueError):
    """Raised when an explicit addition/subtraction policy cannot be represented."""

    project_diagnostic_code = "project.translate.metal-additive-profile-unsupported"
    missing_capabilities = ("metal.additive-profile-lowering",)

    def __init__(self, profile, operand_type, reason, source_location=None):
        self.profile = profile
        self.operand_type = operand_type
        self.reason = reason
        self.source_location = source_location
        super().__init__(
            f"Cannot apply Metal binary32 additive profile '{profile}' to "
            f"'{operand_type or '<unknown>'}': {reason}"
        )


class MetalRemainderProfileError(ValueError):
    """Raised when a selected binary32 remainder profile cannot be represented."""

    project_diagnostic_code = "project.translate.metal-remainder-profile-unsupported"
    missing_capabilities = ("metal.remainder-profile-lowering",)

    def __init__(self, profile, operand_type, reason, source_location=None):
        self.profile = profile
        self.operand_type = operand_type
        self.reason = reason
        self.source_location = source_location
        super().__init__(
            f"Cannot apply Metal binary32 remainder profile '{profile}' to "
            f"'{operand_type or '<unknown>'}': {reason}"
        )


class MetalHalfRemainderProfileError(ValueError):
    """Raised when an explicit half remainder profile cannot be represented."""

    project_diagnostic_code = (
        "project.translate.metal-half-remainder-profile-unsupported"
    )
    missing_capabilities = ("metal.half-remainder-profile-lowering",)

    def __init__(self, profile, operand_type, reason, source_location=None):
        self.profile = profile
        self.operand_type = operand_type
        self.reason = reason
        self.source_location = source_location
        super().__init__(
            f"Cannot apply Metal binary16 remainder profile '{profile}' to "
            f"'{operand_type or '<unknown>'}': {reason}"
        )


class MetalHalfWideningProfileError(ValueError):
    """Raised when a selected half conversion policy cannot be represented."""

    project_diagnostic_code = (
        "project.translate.metal-half-widening-profile-unsupported"
    )
    missing_capabilities = ("metal.half-widening-profile-lowering",)

    def __init__(self, operand_type, source_location=None):
        self.source_location = source_location
        super().__init__(
            "Metal canonical-NaN half widening requires a function and one to four "
            f"lanes, got '{operand_type}'"
        )


class MetalStandardLibraryWrapperLoweringError(ValueError):
    """Raised when a materialized Metal standard-library wrapper has no target op."""

    project_diagnostic_code = "project.translate.metal-stdlib-wrapper-unsupported"
    missing_capabilities = ("metal.standard-library-wrapper-lowering",)

    def __init__(
        self,
        function_name,
        implementation_intrinsics,
        source_location=None,
    ):
        self.function_name = function_name
        self.implementation_intrinsics = tuple(implementation_intrinsics)
        self.source_location = source_location
        intrinsic_text = ", ".join(self.implementation_intrinsics)
        super().__init__(
            "Cannot lower materialized Metal standard-library wrapper "
            f"'{function_name}': no canonical operation represents "
            f"{intrinsic_text}"
        )


class MetalSourceOverloadResolutionError(ValueError):
    """Raised when a Metal call cannot be bound from source argument types."""

    project_diagnostic_code = "project.translate.metal-source-overload-unresolved"
    missing_capabilities = ("metal.source-overload-resolution",)

    def __init__(
        self,
        function_name,
        argument_types,
        candidates,
        reason,
        source_location=None,
    ):
        self.function_name = function_name
        self.argument_types = tuple(argument_types)
        self.candidates = tuple(candidates)
        self.reason = reason
        self.source_location = source_location
        signature = ", ".join(self.argument_types)
        candidate_text = ", ".join(self.candidates) or "<none>"
        super().__init__(
            "Cannot preserve Metal source overload binding for "
            f"{function_name}({signature}): {reason}; candidates are "
            f"{candidate_text}"
        )


class MetalArithmeticTypeResolutionError(ValueError):
    """Raised when source arithmetic has no valid common operand type."""

    project_diagnostic_code = "project.translate.metal-arithmetic-type-invalid"
    missing_capabilities = ("metal.source-arithmetic-types",)

    def __init__(self, operator, operand_types, reason, source_location=None):
        self.operator = operator
        self.operand_types = tuple(operand_types)
        self.reason = reason
        self.source_location = source_location
        super().__init__(
            f"Cannot preserve Metal operator '{operator}' for "
            f"'{operand_types[0]}' and '{operand_types[1]}': {reason}"
        )


class MetalAutoTypeInferenceError(ValueError):
    """Raised when a Metal auto initializer has no determinate value type."""

    project_diagnostic_code = "project.translate.metal-auto-type-unresolved"
    missing_capabilities = ("metal.auto-local-type-inference",)

    def __init__(
        self,
        variable_name,
        callable_name,
        return_type,
        reason,
        source_location=None,
        unresolved_parameters=(),
    ):
        self.variable_name = variable_name
        self.callable_name = callable_name
        self.return_type = return_type
        self.reason = reason
        self.source_location = source_location
        self.unresolved_parameters = tuple(unresolved_parameters)
        origin = (
            f"selected callable '{callable_name}' returning '{return_type}'"
            if callable_name is not None
            else f"initializer of type '{return_type}'"
        )
        super().__init__(
            f"Cannot infer Metal auto local '{variable_name}' from {origin}: {reason}"
        )


class MetalAddressProvenanceError(ValueError):
    """Raised when an address expression has no provable storage provenance."""

    project_diagnostic_code = "project.translate.metal-address-provenance-unresolved"
    missing_capabilities = ("metal.address-provenance-inference",)

    def __init__(
        self,
        operand_kind,
        reason,
        source_location=None,
        base_type=None,
    ):
        self.operand_kind = operand_kind
        self.reason = reason
        self.source_location = source_location
        self.base_type = base_type
        type_detail = f" with base type '{base_type}'" if base_type else ""
        super().__init__(
            f"Cannot infer Metal address provenance for {operand_kind}{type_detail}: "
            f"{reason}"
        )


class MetalStructMethodCallResolutionError(ValueError):
    """Raised when a lowered sibling method call cannot be rebound safely."""

    project_diagnostic_code = "project.translate.metal-struct-method-call-unresolved"
    missing_capabilities = ("metal.struct-method-call-lowering",)

    def __init__(
        self,
        owner,
        method_name,
        argument_types,
        candidates,
        reason,
        source_location=None,
    ):
        self.owner = owner
        self.method_name = method_name
        self.argument_types = tuple(argument_types)
        self.candidates = tuple(candidates)
        self.reason = reason
        self.source_location = source_location
        signature = ", ".join(self.argument_types)
        candidate_text = ", ".join(self.candidates) or "<none>"
        super().__init__(
            "Cannot resolve lowered Metal struct method call "
            f"{owner}::{method_name}({signature}): {reason}; candidates are "
            f"{candidate_text}; qualify the intended call before lowering or "
            "make the overload argument types exact."
        )


class MetalOutOfLineCallOperatorLoweringError(ValueError):
    """Raised when a declared call-operator body has no unique lowered helper."""

    project_diagnostic_code = (
        "project.translate.metal-out-of-line-call-operator-unresolved"
    )
    missing_capabilities = ("metal.out-of-line-call-operator-lowering",)

    def __init__(
        self,
        owner,
        signature,
        candidates,
        reason,
        *,
        declaration_location=None,
        definition_location=None,
        candidate_locations=(),
    ):
        self.owner = owner
        self.method_name = "operator()"
        self.signature = signature
        self.candidates = tuple(candidates)
        self.reason = reason
        self.declaration_location = declaration_location
        self.definition_location = definition_location
        self.candidate_locations = tuple(candidate_locations)
        self.source_location = definition_location or declaration_location
        candidate_text = ", ".join(self.candidates) or "<none>"
        super().__init__(
            "Cannot bind out-of-line Metal call operator "
            f"'{owner}::{signature}' to a lowered helper: {reason}; "
            f"candidates are {candidate_text}"
        )


class MetalConstructorContractError(ValueError):
    """Raised when an explicit Metal constructor cannot be preserved."""

    project_diagnostic_code = "project.translate.metal-constructor-unrepresentable"
    missing_capabilities = ("metal.explicit-constructor-lowering",)

    def __init__(
        self,
        owner,
        argument_types,
        candidates,
        reason,
        source_location=None,
    ):
        self.owner = owner
        self.argument_types = tuple(argument_types)
        self.candidates = tuple(candidates)
        self.reason = reason
        self.source_location = source_location
        signature = ", ".join(self.argument_types)
        candidate_text = ", ".join(self.candidates) or "<none>"
        super().__init__(
            "Cannot preserve Metal constructor contract for "
            f"{owner}({signature}): {reason}; candidates are {candidate_text}"
        )


class MetalSizeofResolutionError(ValueError):
    """Raised when a concrete Metal object size cannot be represented safely."""

    project_diagnostic_code = "project.translate.metal-sizeof-unresolved"
    missing_capabilities = ("metal.sizeof-materialization",)

    def __init__(self, operand, reason, source_location=None):
        self.operand = operand
        self.reason = reason
        self.source_location = source_location
        super().__init__(f"Cannot materialize Metal sizeof({operand}): {reason}")


class MetalTemplateArgumentResolutionError(ValueError):
    """Raised when a required non-type template argument is not concrete."""

    project_diagnostic_code = "project.translate.metal-template-argument-unresolved"
    missing_capabilities = ("metal.value-template-argument-materialization",)

    def __init__(
        self,
        function_name,
        parameter_name,
        argument_expression,
        selected_call,
        reason,
        argument_kind,
        source_location=None,
        owner=None,
        member=None,
        requested_specialization=None,
        enclosing_function=None,
        enclosing_specialization=None,
        nested_helper=None,
        enclosing_context=None,
    ):
        self.function_name = function_name
        self.parameter_name = parameter_name
        self.argument_expression = argument_expression
        self.argument_kind = argument_kind
        self.default_expression = (
            argument_expression if argument_kind == "default" else None
        )
        self.explicit_argument = (
            argument_expression
            if argument_kind in {"explicit", "explicit_type"}
            else None
        )
        self.selected_call = selected_call
        self.owner = owner
        self.member = member
        self.requested_specialization = requested_specialization or selected_call
        self.enclosing_function = enclosing_function
        self.enclosing_specialization = enclosing_specialization
        self.nested_helper = nested_helper
        self.enclosing_context = enclosing_context
        self.reason = reason
        self.source_location = source_location
        if argument_kind == "missing":
            argument = f"required argument for '{parameter_name}'"
        elif argument_kind == "overload":
            argument = f"overload set for '{function_name}'"
        elif argument_kind == "explicit_type":
            argument = (
                f"explicit type argument for '{parameter_name}' "
                f"({argument_expression})"
            )
        elif argument_kind == "constexpr_body":
            argument = f"constexpr body ({argument_expression})"
        else:
            argument = (
                f"{argument_kind} argument for '{parameter_name}' "
                f"({argument_expression})"
            )
        message = (
            "Cannot materialize Metal function template "
            f"'{function_name}' for call '{selected_call}': {argument} {reason}"
        )
        if owner is not None and member is not None:
            message += f" while resolving {owner}::{member}"
        if enclosing_specialization is not None:
            helper = nested_helper or selected_call
            context = enclosing_context or "constexpr call"
            message += (
                f" while resolving {context} '{helper}' in specialization "
                f"'{enclosing_specialization}'"
            )
        super().__init__(message)


class MetalAliasTemplateResolutionError(ValueError):
    """Raised when a reachable Metal alias template cannot be made concrete."""

    project_diagnostic_code = "project.translate.metal-alias-template-unresolved"
    missing_capabilities = ("metal.alias-template-pack-materialization",)

    def __init__(
        self,
        alias_name,
        requested_signature,
        reason,
        *,
        source_location=None,
        dependency_chain=(),
        parameter_pack=None,
        limit=None,
        limit_source=None,
        required_work_items=None,
    ):
        self.alias_name = alias_name
        self.requested_signature = requested_signature
        self.reason = reason
        self.source_location = source_location
        self.dependency_chain = tuple(dependency_chain)
        self.parameter_pack = parameter_pack
        self.limit = limit
        self.limit_source = limit_source
        self.required_work_items = required_work_items
        chain = (
            f" ({' -> '.join(self.dependency_chain)})" if self.dependency_chain else ""
        )
        super().__init__(
            "Cannot materialize Metal alias template "
            f"'{requested_signature}': {reason}{chain}"
        )


class MetalScalarAliasResolutionError(ValueError):
    """Raised when a concrete scalar alias has no unique declared meaning."""

    project_diagnostic_code = "project.translate.metal-scalar-alias-unresolved"
    missing_capabilities = ("metal.scalar-alias-resolution",)

    def __init__(
        self, alias_name, reason, *, source_location=None, dependency_chain=()
    ):
        self.alias_name = alias_name
        self.reason = reason
        self.source_location = source_location
        self.dependency_chain = tuple(dependency_chain)
        super().__init__(f"Cannot resolve Metal scalar alias '{alias_name}': {reason}")


class MetalStructAliasResolutionError(ValueError):
    """Raised when a struct-scoped Metal alias has no unique concrete type."""

    project_diagnostic_code = "project.translate.metal-struct-alias-unresolved"
    missing_capabilities = ("metal.struct-alias-resolution",)

    def __init__(
        self,
        owner,
        alias_name,
        reason,
        *,
        candidate_identities=(),
        candidate_locations=(),
        source_location=None,
    ):
        self.owner = owner
        self.alias_name = alias_name
        self.requested_signature = f"{owner}::{alias_name}"
        self.reason = reason
        self.candidate_identities = tuple(candidate_identities)
        self.candidate_locations = tuple(candidate_locations)
        self.source_location = source_location
        candidates = ", ".join(self.candidate_identities) or "<none>"
        super().__init__(
            "Cannot materialize Metal struct alias "
            f"'{self.requested_signature}': {reason}; candidates are {candidates}"
        )


class MetalFunctionLocalTypeResolutionError(ValueError):
    """Raised when a function-local aggregate cannot be made concrete."""

    project_diagnostic_code = "project.translate.metal-local-type-unresolved"
    missing_capabilities = ("metal.function-local-type-materialization",)

    def __init__(
        self,
        function_name,
        type_name,
        reason,
        *,
        source_location=None,
        unresolved_dependencies=(),
    ):
        self.function_name = function_name
        self.type_name = type_name
        self.reason = reason
        self.source_location = source_location
        self.unresolved_dependencies = tuple(unresolved_dependencies)
        dependency_text = (
            f" ({', '.join(self.unresolved_dependencies)})"
            if self.unresolved_dependencies
            else ""
        )
        super().__init__(
            "Cannot materialize function-local Metal type "
            f"'{type_name}' in '{function_name}': {reason}{dependency_text}"
        )


class MetalCallableLoweringError(ValueError):
    """Raised when a Metal callback cannot be lowered without changing semantics."""

    project_diagnostic_code = "project.translate.metal-callable-unsupported"
    missing_capabilities = ("metal.captured-callback-lowering",)

    def __init__(
        self,
        helper,
        reason,
        source_location=None,
        capture=None,
        enclosing_function=None,
        suggested_action=None,
    ):
        self.helper = helper
        self.reason = reason
        self.source_location = source_location
        self.capture = capture
        self.enclosing_function = enclosing_function
        self.suggested_action = suggested_action
        super().__init__(f"Cannot lower Metal callback passed to {helper}: {reason}")


class MetalWideVectorLoweringError(ValueError):
    """Raised when a concrete Metal vector cannot be lowered as an aggregate."""

    project_diagnostic_code = "project.translate.metal-wide-vector-unsupported"
    missing_capabilities = ("metal.wide-vector-aggregate-lowering",)

    def __init__(
        self,
        vector_type,
        reason,
        source_location=None,
        operation=None,
    ):
        self.vector_type = vector_type
        self.reason = reason
        self.source_location = source_location
        self.operation = operation
        detail = f" for operation '{operation}'" if operation else ""
        super().__init__(
            f"Cannot lower Metal wide vector '{vector_type}'{detail}: {reason}"
        )


class MetalCooperativeMatrixFragmentLoweringError(ValueError):
    """Raised when a whole-lane matrix fragment cast is not provably exact."""

    project_diagnostic_code = (
        "project.translate.metal-cooperative-matrix-fragment-unsupported"
    )
    missing_capabilities = ("metal.cooperative-matrix-fragment-lowering",)

    def __init__(
        self,
        matrix_type,
        fragment_type,
        direction,
        reason,
        source_location=None,
        qualifiers=(),
    ):
        self.matrix_type = matrix_type
        self.fragment_type = fragment_type
        self.direction = direction
        self.reason = reason
        self.source_location = source_location
        self.qualifiers = tuple(qualifiers)
        super().__init__(
            "Cannot lower Metal cooperative-matrix whole-fragment "
            f"{direction} from '{matrix_type}' through '{fragment_type}': {reason}"
        )


class MetalIndexedComponentTypeResolutionError(ValueError):
    """Raised when an indexed Metal component has no provable source type."""

    project_diagnostic_code = (
        "project.translate.metal-indexed-component-type-unresolved"
    )
    missing_capabilities = ("metal.indexed-component-type-inference",)

    def __init__(
        self,
        base_type,
        access_kind,
        reason,
        source_location=None,
        index_expression=None,
    ):
        self.base_type = base_type
        self.access_kind = access_kind
        self.reason = reason
        self.source_location = source_location
        self.index_expression = index_expression
        type_name = base_type or "<unknown>"
        index_detail = (
            f" at index '{index_expression}'" if index_expression is not None else ""
        )
        super().__init__(
            "Cannot resolve Metal indexed component type for "
            f"'{type_name}' ({access_kind}){index_detail}: {reason}"
        )


class MetalStageEntryArrayResourceError(ValueError):
    """Raised when a Metal entry array has no faithful resource contract."""

    project_diagnostic_code = "project.translate.metal-entry-array-resource-invalid"
    missing_capabilities = ("metal.stage-entry-array-resource-lowering",)

    def __init__(
        self,
        parameter_name,
        array_dimensions,
        reason,
        source_location=None,
    ):
        self.parameter_name = parameter_name
        self.array_dimensions = tuple(array_dimensions or ())
        self.reason = reason
        self.source_location = source_location
        super().__init__(
            "Cannot lower Metal stage-entry array resource "
            f"'{parameter_name}': {reason.replace('-', ' ')}"
        )


class MetalAtomicFenceLoweringError(ValueError):
    """Raised when a Metal fence contract cannot be preserved in CrossGL."""

    project_diagnostic_code = "project.translate.metal-atomic-fence-unsupported"
    missing_capabilities = ("metal.atomic-thread-fence-contract-lowering",)

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
            "Cannot lower Metal atomic_thread_fence without changing its "
            f"semantics ({contract}): {reason.replace('-', ' ')}"
        )


class MetalAtomicStoreLoweringError(ValueError):
    """Raised when an atomic store's ordering cannot be preserved."""

    project_diagnostic_code = "project.translate.metal-atomic-store-unsupported"
    missing_capabilities = ("metal.atomic-store-contract-lowering",)

    def __init__(self, reason, source_location=None):
        self.reason = reason
        self.source_location = source_location
        super().__init__(
            "Cannot lower Metal atomic_store_explicit without changing its "
            f"semantics: {reason}"
        )


class MetalAtomicLoadLoweringError(ValueError):
    """Raised when an atomic load's ordering cannot be preserved."""

    project_diagnostic_code = "project.translate.metal-atomic-load-unsupported"
    missing_capabilities = ("metal.atomic-load-contract-lowering",)

    def __init__(self, reason, source_location=None):
        self.reason = reason
        self.source_location = source_location
        super().__init__(
            "Cannot lower Metal atomic_load_explicit without changing its "
            f"semantics: {reason}"
        )


class MetalAtomicFetchSubLoweringError(ValueError):
    """Raised when an atomic subtraction has no CrossGL atomic addition of the same width."""

    project_diagnostic_code = "project.translate.metal-atomic-fetch-sub-unsupported"
    missing_capabilities = ("metal.atomic-fetch-sub-lowering",)

    def __init__(self, reason, source_location=None):
        self.reason = reason
        self.source_location = source_location
        super().__init__(f"Cannot lower Metal atomic_fetch_sub_explicit: {reason}")


class MetalAtomicCompareExchangeLoweringError(ValueError):
    """Raised when a compare-exchange contract cannot be preserved."""

    project_diagnostic_code = (
        "project.translate.metal-atomic-compare-exchange-unsupported"
    )
    missing_capabilities = ("metal.atomic-compare-exchange-contract-lowering",)

    def __init__(self, reason, source_location=None):
        self.reason = reason
        self.source_location = source_location
        super().__init__(
            "Cannot lower Metal atomic_compare_exchange_weak_explicit without "
            f"changing its semantics: {reason}"
        )


class MetalConstexprBranchLoweringError(ValueError):
    """Raised when compile-time branch selection remains unresolved."""

    project_diagnostic_code = "project.translate.metal-constexpr-branch-unresolved"
    missing_capabilities = ("metal.constexpr-branch-selection",)

    def __init__(self, condition, source_location=None):
        self.source_location = source_location
        super().__init__(
            f"Cannot select Metal if constexpr branch for unresolved condition '{condition}'"
        )
