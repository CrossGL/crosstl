"""Preprocessor support for Metal source imports."""

import operator
import re
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from crosstl.backend.DirectX.macro_expansion import HAS_INCLUDE_MACROS, METAL_MACROS
from crosstl.backend.DirectX.preprocessor import HLSLPreprocessor, Macro
from crosstl.translator.refusals import MetalSourceRefusal

from .alias_template_types import (
    expand_concrete_alias_template_references,
    fold_expression_decltypes,
)
from .array_element_receivers import array_element_receiver_end, array_rank_at
from .bool_dispatch_callbacks import expand_bool_dispatch_callbacks
from .call_result_types import CallResultScope, infer_call_result_type
from .class_argument_precedence import class_identity_template_outranks
from .concrete_qualified_aliases import resolve_concrete_qualified_type_aliases
from .constrained_call_carriers import constrained_call_carriers
from .constrained_template_retention import (
    DeferredConstrainedCall as _DeferredConstrainedCall,
)
from .constrained_template_retention import (
    defines_function,
    removable_constrained_templates,
)
from .defaulted_template_arguments import canonical_struct_template_arguments
from .dependent_result_templates import dependent_result_template_names
from .directives import MetalDirectiveMixin
from .floating_point import NO_CONTRACTION_PRAGMA, contraction_requirement
from .function_specialization import (
    _explicit_template_function_specialization_for_selected_overload,
    _metal_function_parameter_declarations,
)
from .instantiation_memo import memoized_instantiations
from .integral_constant_arithmetic import fold_integral_constant_function_arithmetic
from .integral_constant_tags import (
    concretize_integral_constant_tags,
    library_tag_value_type,
)
from .member_address_types import addressed_member_pointer_type
from .member_lowering_layers import lower_members_with_late_specializations
from .namespaced_specializations import place_in_namespace, specialization_reference
from .narrow_float_constants import metal_predefined_constant_type
from .operator_argument_types import infer_conditional_type, infer_unary_sign_type
from .overload_precedence import exact_template_outranks
from .pointer_conversions import pointer_conversion_type
from .preprocessing.access_type_inference import MetalAccessTypeInferenceMixin
from .preprocessing.call_rewriting import MetalCallRewriteMixin
from .preprocessing.declared_types import MetalDeclaredTypeMixin
from .preprocessing.function_definitions import MetalFunctionDefinitionMixin
from .preprocessing.integral_constant_parameters import (
    MetalIntegralConstantParameterMixin,
)
from .preprocessing.internal_template_calls import MetalInternalTemplateCallMixin
from .preprocessing.local_declarations import MetalLocalDeclarationMixin
from .preprocessing.local_variables import MetalLocalVariableMixin
from .preprocessing.pointer_promotion import MetalPointerPromotionMixin
from .preprocessing.receiver_methods import MetalReceiverMethodMixin
from .preprocessing.reference_aliases import MetalReferenceAliasMixin
from .preprocessing.source_text import MetalSourceTextMixin
from .preprocessing.stateless_globals import MetalStatelessGlobalMixin
from .preprocessing.static_aggregates import MetalStaticAggregateMixin
from .preprocessing.static_constexpr_calls import MetalStaticConstexprCallMixin
from .preprocessing.static_members import MetalStaticMemberMixin
from .preprocessing.static_scalars import MetalStaticScalarMixin
from .preprocessing.struct_member_lowering import MetalStructMemberLoweringMixin
from .preprocessing.struct_methods import MetalStructMethodMixin
from .preprocessing.struct_scoped_aliases import MetalStructScopedAliasMixin
from .preprocessing.struct_specializations import MetalStructSpecializationMixin
from .preprocessing.support import (
    CLANG_FEATURE_TEST_MACROS,
    CONTAINING_SPAN_CACHE_LIMIT,
    DEFAULT_EXPLICIT_TEMPLATE_SPECIALIZATION_LIMIT,
    IDENTIFIER_RE,
    LEXICAL_SCOPE_CACHE_LIMIT,
    METAL_ANGLE_OPERATOR_RE,
    METAL_ENTRY_FUNCTION_RE,
    METAL_PREDEFINED_MACROS,
    METAL_STRING_EXPRESSION_PATTERN,
    MLX_INSTANTIATE_KERNEL_RE,
    PRESERVED_INCLUDE_SENTINEL,
    SCAN_STOP_RE,
    SOURCE_ANALYSIS_CACHE_LIMIT,
    MetalMacroExpansionError,
    MetalStatelessGlobalElisionError,
    MetalStaticAssertionError,
    MetalStructMethodError,
    MetalTemplateSpecializationError,
    MetalTypeTraitError,
    _MetalAddressableValueType,
    _MetalBufferTypeView,
    _MetalConstexprFunction,
    _MetalContainingSpanLookup,
    _MetalDataMember,
    _MetalFunctionDefinition,
    _MetalIntegralConstantBinding,
    _MetalLexicalScopeLookup,
    _MetalOutOfLineMemberDefinition,
    _MetalPositionedBufferTypes,
    _MetalReachabilityTypeContext,
    _MetalReceiverContract,
    _MetalSourceAnalysis,
    _MetalStructDefinition,
    _MetalStructMethod,
    _MetalTemplateFunction,
    _MetalTemplateStruct,
    _MetalTypeAliasBinding,
    _MLXKernelInstantiation,
)
from .preprocessing.template_call_materialization import (
    MetalTemplateCallMaterializationMixin,
)
from .preprocessing.template_instantiations import MetalTemplateInstantiationMixin
from .preprocessing.template_parameters import MetalTemplateParameterMixin
from .preprocessing.template_struct_arguments import MetalTemplateStructArgumentMixin
from .preprocessing.temporary_functors import MetalTemporaryFunctorMixin
from .preprocessing.type_inference import MetalTypeInferenceMixin
from .preprocessing.type_traits import MetalTypeTraitMixin
from .reference_binding import member_call_result
from .return_type_enablers import fold_concrete_enablers
from .simd_group_template_overloads import (
    canonicalize_struct_template_ids,
    only_simd_group_overloads,
    overloads_simd_group_function,
)
from .source_analysis_cache import SourceAnalysisCache
from .specialized_member_types import specialized_member_type_template_names
from .static_member_constants import (
    materialized_struct_for_template_id,
    static_member_constant_substituter,
)
from .struct_homonyms import distinct_struct_homonyms
from .swizzle_types import infer_swizzle_type
from .template_carried_reachability import template_carried_reachable_spans
from .template_constants import unresolved_constant_references
from .unreachable_constrained_callers import unreachable_caller_scope
from .unreferenced_constrained_templates import (
    prune_unreferenced_constrained_templates,
)
from .unused_specialization_members import prune_unreferenced_specialization_members
from .variable_templates import materialize_value_variable_templates

# The preprocessor members live in the ``preprocessing`` subpackage. These names stay importable from here.
__all__ = [
    "CONTAINING_SPAN_CACHE_LIMIT",
    "DEFAULT_EXPLICIT_TEMPLATE_SPECIALIZATION_LIMIT",
    "ENTRY_DISCOVERY_CACHE_LIMIT",
    "LEXICAL_SCOPE_CACHE_LIMIT",
    "METAL_ANGLE_OPERATOR_RE",
    "METAL_ENTRY_FUNCTION_RE",
    "METAL_PREDEFINED_MACROS",
    "MetalMacroExpansionError",
    "MetalPreprocessor",
    "MetalStatelessGlobalElisionError",
    "MetalStaticAssertionError",
    "MetalStructMethodError",
    "MetalTemplateSpecializationError",
    "MetalTypeTraitError",
    "PRESERVED_INCLUDE_SENTINEL",
    "SCAN_STOP_RE",
    "SOURCE_ANALYSIS_CACHE_LIMIT",
    "discover_metal_entry_points",
]

ENTRY_DISCOVERY_CACHE_LIMIT = 8


class MetalPreprocessor(
    MetalTemplateCallMaterializationMixin,
    MetalStatelessGlobalMixin,
    MetalStructMemberLoweringMixin,
    MetalStructScopedAliasMixin,
    MetalLocalDeclarationMixin,
    MetalTypeTraitMixin,
    MetalStructMethodMixin,
    MetalStaticMemberMixin,
    MetalStaticAggregateMixin,
    MetalStaticScalarMixin,
    MetalStaticConstexprCallMixin,
    MetalPointerPromotionMixin,
    MetalLocalVariableMixin,
    MetalCallRewriteMixin,
    MetalReceiverMethodMixin,
    MetalTemporaryFunctorMixin,
    MetalIntegralConstantParameterMixin,
    MetalInternalTemplateCallMixin,
    MetalReferenceAliasMixin,
    MetalDeclaredTypeMixin,
    MetalTypeInferenceMixin,
    MetalAccessTypeInferenceMixin,
    MetalTemplateInstantiationMixin,
    MetalTemplateStructArgumentMixin,
    MetalStructSpecializationMixin,
    MetalFunctionDefinitionMixin,
    MetalTemplateParameterMixin,
    MetalSourceTextMixin,
    MetalDirectiveMixin,
    HLSLPreprocessor,
):
    """Small Metal preprocessor used before lexing imported source files."""

    macro_dialect = METAL_MACROS

    def __init__(
        self,
        include_paths: Optional[List[str]] = None,
        defines: Optional[Dict[str, str]] = None,
        strict: bool = False,
        max_expansion_depth: int = 64,
        max_template_specializations: int = (
            DEFAULT_EXPLICIT_TEMPLATE_SPECIALIZATION_LIMIT
        ),
        template_specialization_limit_source: Optional[str] = None,
        promote_derived_pointer_members: bool = False,
        group_non_type_template_substitutions: bool = True,
    ):
        super().__init__(
            include_paths=include_paths,
            defines=defines,
            strict=strict,
            max_expansion_depth=max_expansion_depth,
        )
        if not isinstance(promote_derived_pointer_members, bool):
            raise MetalSourceRefusal(
                "Metal promote_derived_pointer_members must be a boolean"
            )
        self.promote_derived_pointer_members = promote_derived_pointer_members
        if not isinstance(group_non_type_template_substitutions, bool):
            raise MetalSourceRefusal(
                "Metal group_non_type_template_substitutions must be a boolean"
            )
        self.group_non_type_template_substitutions = (
            group_non_type_template_substitutions
        )
        if isinstance(max_template_specializations, bool):
            raise MetalSourceRefusal(
                "Metal max_template_specializations must be a non-negative integer"
            )
        try:
            specialization_limit = operator.index(max_template_specializations)
        except TypeError as exc:
            raise MetalSourceRefusal(
                "Metal max_template_specializations must be a non-negative integer"
            ) from exc
        if specialization_limit < 0:
            raise MetalSourceRefusal(
                "Metal max_template_specializations must be a non-negative integer"
            )
        self.max_template_specializations = specialization_limit
        self.template_specialization_limit_source = (
            template_specialization_limit_source or "max_template_specializations"
        )
        self._materialized_struct_specializations: Dict[
            str, Tuple[str, Tuple[str, ...]]
        ] = {}
        self._materialized_struct_specialization_namespaces: Dict[str, str] = {}
        self._materialized_struct_primary_templates: Dict[str, _MetalTemplateStruct] = (
            {}
        )
        self._known_member_function_return_types: Dict[str, str] = {}
        self._call_result_scope = CallResultScope()
        # (source, rewrite) for each active call-site scan, innermost last;
        # rewrite(start, end) returns source[start:end] with its member calls
        # rewritten in place.
        self._call_argument_rewriters: List[Tuple[str, Callable[[int, int], str]]] = []
        self._inferred_bitcast_shadowed = False
        self._instantiated_template_member_calls: Dict[
            str,
            Tuple[
                _MetalStructDefinition,
                _MetalStructMethod,
                Dict[str, str],
                Optional[Dict[str, _MetalStructDefinition]],
            ],
        ] = {}
        self._materialized_function_names: Set[str] = set()
        self._materialized_function_templates: Set[Tuple[str, str]] = set()
        # Explicit free-function template calls materialized by `preprocess`,
        # shared by its passes so a call first reached after member lowering
        # reuses the helper an earlier pass emitted instead of redefining it.
        self._explicit_template_function_names: Dict[
            Tuple[str, Tuple[str, ...]], str
        ] = {}
        self._active_static_constexpr_functions: Dict[
            str, List[_MetalConstexprFunction]
        ] = {}
        self._source_type_alias_bindings: Dict[str, List[_MetalTypeAliasBinding]] = {}
        # The source whose aggregates give `sizeof(S)` its value in integral
        # constant initializers evaluated without a position-aware resolver,
        # and the structs, file-scope aliases and templates parsed from it.
        self._active_sizeof_source = ""
        self._sizeof_context_cache: Dict[str, Tuple[object, ...]] = {}
        self._integral_constant_binary_operators: Set[str] = set()
        self._integral_constant_contract_verified = False
        self._integral_constant_conversion_contract_verified = False
        self._int_alias_contract_verified = False
        self._const_for_loop_contract_verified = False
        self._const_for_loop_expansion_work = 0
        self._fail_closed_static_assertions = False
        self._static_constexpr_helper_values = {}
        self._static_constexpr_helper_resolution_stack = []
        self._source_analysis_cache: SourceAnalysisCache[_MetalSourceAnalysis] = (
            SourceAnalysisCache(SOURCE_ANALYSIS_CACHE_LIMIT)
        )
        self._local_data_member_memo: Dict[
            str, Optional[Tuple[str, _MetalDataMember]]
        ] = {}
        # The materialization scan alternates between multiple span lists at
        # nearly every source position. Cache their sorted lookup indexes, but
        # retain only a fixed number of lists as generated source snapshots are
        # replaced during specialization.
        self._containing_span_cache: Dict[int, _MetalContainingSpanLookup] = {}
        self._lexical_scope_lookup_cache: Dict[int, _MetalLexicalScopeLookup] = {}
        self.macros.setdefault(
            "TARGET_OS_SIMULATOR",
            Macro(name="TARGET_OS_SIMULATOR", replacement="0"),
        )
        # The Metal compiler always defines these, and shared host/shader headers test them to hide host-only
        # code (#ifndef __METAL_VERSION__) or choose language features (#if __METAL_VERSION__ >= 230). The
        # importer reads sources as MSL 3.1 unless the caller defines another version.
        for name, value in METAL_PREDEFINED_MACROS.items():
            self.macros.setdefault(name, Macro(name=name, replacement=value))
        # __has_include and __has_include_next are the shared expander's built-ins (MacroDialect.has_include):
        # #if looks the header up, and anywhere else they are refused, as Clang rejects them there.
        for name in sorted(CLANG_FEATURE_TEST_MACROS - set(HAS_INCLUDE_MACROS)):
            self.macros.setdefault(name, Macro(name=name, replacement="0"))

    def preprocess(self, code: str, file_path: Optional[str] = None) -> str:
        self._source_analysis_cache.clear()
        self._local_data_member_memo.clear()
        # Span lists are per-source; drop any cache entries from a previous
        # source so retained references cannot pin freed lists across runs.
        self._containing_span_cache.clear()
        self._lexical_scope_lookup_cache.clear()
        self._materialized_struct_specializations.clear()
        self._materialized_struct_specialization_namespaces.clear()
        self._materialized_struct_primary_templates.clear()
        self._known_member_function_return_types.clear()
        self._instantiated_template_member_calls.clear()
        self._materialized_function_names.clear()
        self._materialized_function_templates.clear()
        self._explicit_template_function_names.clear()
        self._active_static_constexpr_functions.clear()
        self._source_type_alias_bindings.clear()
        self._active_sizeof_source = ""
        self._sizeof_context_cache = {}
        self._integral_constant_binary_operators.clear()
        self._integral_constant_contract_verified = False
        self._integral_constant_conversion_contract_verified = False
        self._int_alias_contract_verified = False
        self._const_for_loop_contract_verified = False
        self._const_for_loop_expansion_work = 0
        self._fail_closed_static_assertions = False
        self._static_constexpr_helper_values = {}
        self._static_constexpr_helper_resolution_stack = []
        code = self._strip_leading_compiler_diagnostics(code)
        processed = super().preprocess(code, file_path=file_path)
        processed = self._canonicalize_metal_library_matrix_typedefs(processed)
        processed = distinct_struct_homonyms(processed)
        if re.search(r"\b(?:fp|FP_CONTRACT)\b", processed):
            # Inspect active expanded tokens before materialization changes the
            # lexical scope of a directive or prunes its containing template.
            from .MetalLexer import MetalLexer

            # A setting CrossGL cannot express is refused here. A directive that requires @precise functions is
            # restated at the end of the unit, so the parser still finds it after template rewriting.
            if contraction_requirement(
                MetalLexer(processed, preprocess=False).token_generator()
            ):
                processed = (
                    processed.rstrip("\n") + f"\n#pragma {NO_CONTRACTION_PRAGMA}\n"
                )
        self._configure_inferred_bitcast_ownership(processed)
        self._configure_integral_constant_contracts(processed)
        # Generic-lambda callbacks of a Boolean dispatch helper become their two
        # instantiations, so `p.value` is a constant for the passes below.
        processed = expand_bool_dispatch_callbacks(self, processed)
        # Assertions over constexpr aggregates and operators, such as MLX's
        # `(complex_t<half>{1, 2} * complex_t<half>{3, 4}).real == -5`, are decided while the struct templates and
        # their operators are still spelled as in the source, as template materialization decides them; an
        # undecided one is left for the translator's check (translator/semantics/static_assertions.py).
        processed = self._evaluate_static_assertions(processed, keep_unresolved=True)
        processed = self._materialize_project_template_instantiations(processed)
        # Concrete helper bodies must exist while their receiver's methods are
        # still present; lowering the struct first loses those later call sites.
        processed = self._materialize_explicit_template_function_calls(
            processed, include_struct_members=False
        )
        processed = self._materialize_explicit_template_struct_instantiations(processed)
        processed = self._materialize_free_operator_overloads(processed)
        processed = self._elide_stateless_compile_time_globals(processed)
        # Member lowering can introduce fresh concrete struct-template spellings
        # in emitted free-function signatures.  In particular, a non-type member
        # parameter substituted from `Int<(N)>` is not visible to the first pass
        # above.  Materialize that generated layer before checking for residual
        # function templates; equivalent parenthesized arguments share the
        # original concrete specialization through normalized keys.  A layer
        # struct that still declares members is lowered by a retry.
        processed = lower_members_with_late_specializations(
            self,
            processed,
            lambda source: self._materialize_explicit_template_struct_instantiations(
                self._lower_struct_member_functions(source)
            ),
            self._materialize_explicit_template_struct_instantiations,
            MetalStructMethodError,
        )
        processed = self._materialize_explicit_template_function_calls(processed)
        # This source reaches the CrossGL generic path, which cannot defer a SIMD-group
        # overload to each instantiation; the project materializer binds its own.
        constrained_source = processed
        processed = self._materialize_inferred_constrained_template_function_calls(
            processed, simd_group_overloads=True
        )
        if processed != constrained_source:
            # A materialized constrained helper spells the struct templates of its signature with the bound
            # arguments (`device mlx_atomic<float>*`), which the struct pass above has not seen.
            processed = self._materialize_explicit_template_struct_instantiations(
                processed
            )
        processed = self._prune_unreferenced_materialized_function_templates(processed)
        # A constrained overload nothing calls would reach CrossGL without its constraint.
        processed = prune_unreferenced_constrained_templates(self, processed)
        processed = prune_unreferenced_specialization_members(self, processed)
        processed = materialize_value_variable_templates(self, processed)
        processed = self._substitute_local_integral_constant_array_extents(processed)
        # A dependent member type substituted before its owner was materialized
        # names a concrete struct's alias now; spell the alias target instead.
        processed = resolve_concrete_qualified_type_aliases(self, processed)
        # Materialized instantiations are concrete: fold their type traits (is_same<float4, T0>::value with T0
        # bound) as for the source's own concrete code. No later stage folds a trait, so one left in concrete
        # code is refused for every target rather than reaching the parser as an identifier.
        processed = self._fold_concrete_type_trait_values(
            processed, require_resolution=True
        )
        # With the traits folded, a materialized return-type enabler names its type, and so does a decltype of
        # built-in operands that a dependent result type leaves (dependent_result_templates.py). A concrete library
        # tag (metal::bool_constant<true>) names a declared tag struct (integral_constant_tags.py).
        # An alias template whose arguments are built-in types names one type too (alias_template_types.py).
        processed = fold_expression_decltypes(
            self,
            expand_concrete_alias_template_references(
                self, fold_concrete_enablers(self, processed)
            ),
        )
        processed = concretize_integral_constant_tags(self, processed)
        return processed.replace(PRESERVED_INCLUDE_SENTINEL, "#include ")

    def preprocess_for_entry_discovery(
        self, code: str, file_path: Optional[str] = None
    ) -> str:
        """Expand active source directives without materializing templates."""
        self._source_analysis_cache.clear()
        self._containing_span_cache.clear()
        self._lexical_scope_lookup_cache.clear()
        code = self._strip_leading_compiler_diagnostics(code)
        self._discovering_entry_points = True
        try:
            return super().preprocess(code, file_path=file_path)
        finally:
            self._discovering_entry_points = False

    def _materialize_constrained_template_calls_in_scope(
        self, code: str, *, simd_group_overloads: bool = False
    ) -> str:
        """Resolve ordinary calls to constrained free function templates.

        Metal overload sets commonly provide an SFINAE fallback next to a native
        overload, for example MLX's integral-only ``isnan(T)`` beside
        ``metal::isnan(float)``.  The CrossGL generic-function path cannot retain
        anonymous ``enable_if_t`` parameters, so allowing such a declaration to
        reach it silently turns the fallback into an unconstrained helper.

        Resolve only calls in reachable concrete function bodies.  A uniquely
        enabled constrained candidate is materialized and the call is renamed;
        when every viable constrained candidate is disabled, the original call
        remains for native overload resolution.  Unknown argument types,
        unrecognized constraints, multiple enabled candidates, and competing
        overloads whose precedence is not proven fail closed.
        Residual template declarations that participated in this resolution are
        removed so a target generic specializer cannot later discard the SFINAE
        contract.
        """

        templates = self._find_template_functions(code)
        # A template named like a SIMD-group function competes with the library per
        # instantiation too; deduction is its only constraint.
        constrained = [
            template
            for template in templates
            if template.template_constraints
            or (simd_group_overloads and overloads_simd_group_function(template))
        ]
        # Templates whose result is a decltype of their parameters are carriers (dependent_result_templates.py),
        # and so are templates that name a member type a specialization may redefine (specialized_member_types.py).
        dependent_results = dependent_result_template_names(
            self, code, templates
        ) | specialized_member_type_template_names(self, code, templates)
        if not constrained and not dependent_results:
            return code

        # A template method retained inside a residual aggregate declaration is
        # not a free overload. Member-template lowering owns those declarations.
        aggregate_spans = sorted(
            [struct.span for struct in self._find_concrete_struct_definitions(code)]
            + [struct.span for struct in self._find_template_structs(code)]
        )
        constrained = [
            template
            for template in constrained
            if self._containing_span(template.span[0], aggregate_spans) is None
        ]
        if not constrained and not dependent_results:
            return code

        by_name: Dict[str, List[_MetalTemplateFunction]] = {}
        for template in constrained:
            by_name.setdefault(template.name, []).append(template)

        template_spans = self._find_template_declaration_spans(code)
        functions = self._find_non_template_function_definitions(code, template_spans)
        all_body_spans = [function.body_span for function in functions]
        # Unconstrained templates whose bodies call a constrained template are
        # instantiated here too, so those calls bind before the templates go.
        free_templates = [
            template
            for template in templates
            if self._containing_span(template.span[0], aggregate_spans) is None
        ]
        carriers = constrained_call_carriers(
            free_templates,
            set(by_name),
            {function.name for function in functions},
            [
                reference
                for reference in self._find_static_constexpr_calls(
                    code, {template.name for template in free_templates}
                )
                if not self._function_reference_is_member_call(
                    code, reference.span[0], set()
                )
            ],
            dependent_results,
        )
        carrier_names = {template.name for template in carriers}
        for template in carriers:
            by_name[template.name] = [template]
        masked_code = self._mask_comments_and_literals(code)
        # Unreachable bodies join while unreachable_constrained_callers.py resolves them softly.
        reachable_spans = unreachable_caller_scope(self).extend(
            self._reachable_function_spans(code, template_spans)
        )
        if reachable_spans is not None:
            # Concrete functions called from reached template bodies are reached too.
            reachable_set = set(
                template_carried_reachable_spans(
                    masked_code, reachable_spans, functions, free_templates
                )
            )
            directly_reachable = set(reachable_spans)
            carried_body_spans = sorted(
                function.body_span
                for function in functions
                if function.span in reachable_set
                and function.span not in directly_reachable
            )
            functions = [
                function for function in functions if function.span in reachable_set
            ]
        else:
            carried_body_spans = []
        body_spans = [function.body_span for function in functions]
        if not body_spans:
            return code

        references = self._find_static_constexpr_calls(code, set(by_name))
        calls = [
            call
            for call in references
            if self._containing_span(call.span[0], body_spans) is not None
            and not self._function_reference_is_member_call(code, call.span[0], set())
        ]
        if not calls:
            return code

        structs = self._find_concrete_struct_definitions(code)
        template_structs = self._find_template_structs(code)
        context = self._reachability_type_context(code, template_spans, structs)
        # Bodies are only ever appended, and every context view is resolved at a
        # call position. A context of a shorter prefix therefore answers queries
        # before its end; it is rebuilt only when a call in a newer body needs it.
        context_end = len(code)
        namespace_visibility = self._metal_namespace_visibility(code)
        boolean_templates = self._find_boolean_variable_templates(code)
        # A namespace-scope explicit specialization is no overload: it replaces the
        # body of the primary it specializes once that primary is selected
        # (C++14 [temp.expl.spec]/12), so it never competes with the primaries.
        explicit_specializations = (
            self._find_explicit_template_function_specializations(
                code, include_namespaced=True
            )
        )
        explicit_spans = sorted(
            specialization["span"]
            for specialization in explicit_specializations.values()
        )
        competing_declarations = self._find_free_function_overload_declarations(
            code,
            references,
            excluded_spans=sorted(
                aggregate_spans
                + all_body_spans
                + explicit_spans
                + [(template.body_start, template.span[1]) for template in templates]
                + [template.span for template in constrained]
                + [template.span for template in carriers]
            ),
            template_spans=template_spans,
            namespace_visibility=namespace_visibility,
        )
        replacements: List[Tuple[int, int, str]] = []
        materialized_names: Dict[Tuple[Tuple[int, int], Tuple[str, ...]], str] = {}
        generated_name_owners: Dict[str, Tuple[Tuple[int, int], Tuple[str, ...]]] = {}
        handled_template_spans: Set[Tuple[int, int]] = set()
        native_fallback_names: Set[str] = set()
        deferred_call_ids: Set[int] = set()
        deferred_carrier_calls: List[_DeferredConstrainedCall] = []
        carrier_spans = {template.span for template in carriers}
        unconstrained_template_names = {
            template.name
            for template in templates
            if not template.template_constraints
            and template.span not in carrier_spans
            and not (simd_group_overloads and overloads_simd_group_function(template))
        }
        reserved_identifiers = set(IDENTIFIER_RE.findall(masked_code))

        # Append concrete bodies without rewriting the source until the worklist
        # is exhausted. Original declaration offsets and specialization ownership
        # then stay stable, including when a new body calls an earlier helper.
        # Lookup uses the declaring template, not the appended body's position.
        dependencies = {}
        pending_calls = [
            (call, call.span[0], call.span[0], None, False) for call in calls
        ]
        for (
            call,
            lookup_position,
            instantiation_position,
            owner_key,
            dependent,
        ) in pending_calls:
            visible_candidates = [
                template
                for template in by_name.get(call.name, [])
                if template.span[0] <= lookup_position
                and self._metal_template_function_call_visible(
                    template,
                    call.qualified_name,
                    lookup_position,
                    namespace_visibility,
                    None,
                )
            ]
            dependent_lookup = (
                dependent
                and self._metal_call_reference_namespace(call.qualified_name) is None
            )
            if not visible_candidates and not dependent_lookup:
                continue
            potential_candidates = (
                by_name.get(call.name, []) if dependent_lookup else visible_candidates
            )
            if not any(
                parameters is not None
                and self._callable_accepts_argument_count(
                    parameters, len(call.arguments)
                )
                for parameters in (
                    self._template_function_parameter_text(template)
                    for template in potential_candidates
                )
            ):
                continue

            if call.argument_open >= context_end:
                context = self._reachability_type_context(code, template_spans, structs)
                context_end = len(code)
            buffer_view = self._flatten_types_at(
                context.buffer_element_types, call.argument_open
            )
            local_view = self._flatten_types_at(
                context.local_variable_types, call.argument_open
            )
            field_types = self._struct_field_types_at(
                context.receiver_variable_types,
                context.structs_by_name,
                call.argument_open,
            )
            argument_types = [
                self._infer_argument_type(
                    argument,
                    buffer_view,
                    local_view,
                    field_types,
                    context.structs_by_name,
                )
                for argument in call.arguments
            ]
            if any(argument_type is None for argument_type in argument_types):
                if call.name in carrier_names:
                    # The generic path instantiates it; templates it needs stay.
                    continue
                if owner_key is None and only_simd_group_overloads(visible_candidates):
                    # Unconstrained templates keep this call for a later binder,
                    # such as the project materializer, which types it.
                    deferred_call_ids.add(id(call))
                    continue
                if (
                    owner_key is None
                    and self._containing_span(call.span[0], carried_body_spans)
                    is not None
                ):
                    # Only a template reaches this body; a later pass, which
                    # sees the instantiated caller, binds the call.
                    deferred_call_ids.add(id(call))
                    continue
                if owner_key is not None and owner_key[0] in carrier_spans:
                    # A later pass binds the call in the instantiated carrier,
                    # which keeps the templates lookup from its body can see.
                    deferred_carrier_calls.append(
                        _DeferredConstrainedCall(
                            call.name, call.qualified_name, lookup_position
                        )
                    )
                    continue
                if owner_key is None and unreachable_caller_scope(self).defer(
                    call.span[0]
                ):
                    # An argument may be typed by a call this pass binds; a later round decides.
                    deferred_call_ids.add(id(call))
                    continue
                self._raise_constrained_free_function_error(
                    code,
                    call,
                    "one or more call argument types could not be inferred "
                    "conservatively",
                )

            associated_namespaces = set()
            if dependent_lookup:
                if call.name in local_view:
                    self._raise_constrained_free_function_error(
                        code,
                        call,
                        "a local declaration hides the dependent function name",
                    )
                associated_namespaces = self._constrained_call_associated_namespaces(
                    argument_types,
                    structs,
                    template_structs,
                    position=lookup_position,
                    namespace_visibility=namespace_visibility,
                )
                if associated_namespaces is None and call.name in carrier_names:
                    continue
                if associated_namespaces is None:
                    self._raise_constrained_free_function_error(
                        code,
                        call,
                        "argument-dependent lookup has unproven type ownership",
                    )
                for template in by_name.get(call.name, []):
                    if (
                        template.span[0] <= instantiation_position
                        and template.namespace in associated_namespaces
                        and template not in visible_candidates
                    ):
                        visible_candidates.append(template)

            argument_count_candidates: List[Tuple[_MetalTemplateFunction, str]] = []
            for template in visible_candidates:
                parameters = self._template_function_parameter_text(template)
                if parameters is None or not self._callable_accepts_argument_count(
                    parameters, len(call.arguments)
                ):
                    continue
                argument_count_candidates.append((template, parameters))
            if not argument_count_candidates:
                continue

            enabled: List[Tuple[_MetalTemplateFunction, Dict[str, str]]] = []
            saw_unrecognized_constraint = False
            for template, parameters in argument_count_candidates:
                method = self._template_function_method_signature(template, parameters)
                bindings = self._bind_template_method_parameters(
                    method,
                    list(argument_types),
                    explicit_template_arguments=list(call.template_arguments or []),
                    structs_by_name=context.structs_by_name,
                    argument_type_aliases=context.source_type_aliases,
                    argument_type_position=call.argument_open,
                    argument_type_fallback_position=call.argument_open,
                )
                if bindings is None:
                    continue
                try:
                    candidate_enabled = all(
                        self._evaluate_template_constraint(
                            constraint,
                            bindings,
                            structs=structs,
                            type_aliases=context.source_type_aliases,
                            position=call.span[0],
                            boolean_templates=boolean_templates,
                            namespace_visibility=namespace_visibility,
                            lookup_position=template.span[0],
                        )
                        for constraint in template.template_constraints
                    )
                except self._UnrecognizedConstraint:
                    saw_unrecognized_constraint = True
                    continue
                handled_template_spans.add(template.span)
                if candidate_enabled:
                    enabled.append((template, bindings))

            if saw_unrecognized_constraint:
                self._raise_constrained_free_function_error(
                    code,
                    call,
                    "a viable SFINAE constraint is not recognized",
                )
            if len(enabled) > 1:
                self._raise_constrained_free_function_error(
                    code,
                    call,
                    "multiple constrained overloads are enabled",
                )
            if not enabled and call.name in carrier_names:
                # Deduction failed; the generic path instantiates the carrier.
                continue
            if not enabled:
                # Every recognized constrained candidate is disabled. Preserve
                # the spelling so a native overload (such as metal::isnan) can
                # resolve it after the fallback declaration is removed.
                self._require_library_fallback_lookup(
                    code,
                    masked_code,
                    call,
                    lookup_position,
                    visible_candidates,
                    argument_types,
                    namespace_visibility,
                    has_other_declarations=bool(competing_declarations.get(call.name))
                    or call.name in unconstrained_template_names,
                )
                native_fallback_names.add(call.name)
                continue

            for declaration in competing_declarations.get(call.name, []):
                if declaration.span[0] > max(lookup_position, instantiation_position):
                    continue
                if (
                    call.template_arguments is not None
                    and self._containing_span(declaration.span[0], template_spans)
                    is None
                ):
                    continue
                if not self._callable_accepts_argument_count(
                    ", ".join(declaration.arguments), len(call.arguments)
                ):
                    continue
                declaration_namespace = self._metal_call_reference_namespace(
                    declaration.qualified_name
                )
                lexical_namespace = self._namespace_at(
                    list(namespace_visibility.namespace_spans), declaration.span[0]
                )
                namespaces = (
                    self._metal_namespace_reference_candidates(
                        declaration_namespace,
                        lexical_namespace,
                        globally_qualified=declaration_namespace.startswith("::"),
                    )
                    if declaration_namespace is not None
                    else [lexical_namespace]
                )
                if any(
                    (
                        declaration.span[0] <= lookup_position
                        and self._metal_namespace_declaration_visible(
                            namespace,
                            self._metal_call_reference_namespace(call.qualified_name),
                            lookup_position,
                            namespace_visibility,
                        )
                    )
                    or (
                        declaration.span[0] <= instantiation_position
                        and namespace in associated_namespaces
                    )
                    for namespace in namespaces
                ) and not (
                    exact_template_outranks(
                        self,
                        self._template_function_parameter_text(enabled[0][0]),
                        enabled[0][1],
                        argument_types,
                        declaration.arguments,
                    )
                    or class_identity_template_outranks(
                        self,
                        code,
                        context,
                        enabled[0][0],
                        enabled[0][1],
                        argument_types,
                        declaration,
                        call.argument_open,
                    )
                ):
                    self._raise_constrained_free_function_error(
                        code,
                        call,
                        "a competing free-function overload has unproven precedence",
                    )

            template, bindings = enabled[0]
            if self._library_overload_selected(
                code,
                masked_code,
                call,
                lookup_position,
                visible_candidates,
                argument_types,
                namespace_visibility,
                template,
                bindings,
            ):
                # A non-template library overload matches as well as the
                # template does and wins; the call stays for it.
                native_fallback_names.add(call.name)
                continue
            ordered_arguments = tuple(
                bindings[name] for name in template.template_parameters
            )
            parameters = self._template_function_parameter_text(template)
            specialization = (
                _explicit_template_function_specialization_for_selected_overload(
                    preprocessor=self,
                    explicit_specializations={
                        (str(value["name"]), key[1], key[2]): value
                        for key, value in explicit_specializations.items()
                        if value["namespace"] == template.namespace
                        and value["span"][0] <= instantiation_position
                    },
                    template=template,
                    function_name=template.name,
                    arguments=ordered_arguments,
                    parameter_declarations=_metal_function_parameter_declarations(
                        self, f"{template.name}({parameters})"
                    ),
                    declaration_source=code,
                    declaration_type_aliases=context.source_type_aliases,
                    argument_alias_contexts=(
                        (context.source_type_aliases, call.argument_open, code),
                    ),
                )
            )
            key = (template.span, ordered_arguments)
            if owner_key is not None:
                pending_dependencies = [key]
                visited = set()
                while pending_dependencies:
                    dependency = pending_dependencies.pop()
                    if dependency == owner_key:
                        self._raise_constrained_free_function_error(
                            code,
                            call,
                            "recursive constrained helper specialization is unsupported",
                        )
                    if dependency not in visited:
                        visited.add(dependency)
                        pending_dependencies.extend(dependencies.get(dependency, ()))
                dependencies.setdefault(owner_key, set()).add(key)
            specialized_name = materialized_names.get(key)
            if specialized_name is None:
                specialized_name = unreachable_caller_scope(
                    self
                ).reusable_instantiation(
                    template, tuple(ordered_arguments), reserved_identifiers
                )
                if specialized_name is not None:
                    materialized_names[key] = specialized_name
            if specialized_name is None:
                unique_count = len(materialized_names) + 1
                if unique_count > self.max_template_specializations:
                    requested = self._template_specialization_signature(
                        template.name, list(ordered_arguments)
                    )
                    suggested_action = (
                        "raise max_template_specializations for this source pattern "
                        "or reduce the number of reachable constrained specializations"
                    )
                    raise MetalTemplateSpecializationError(
                        "Metal constrained template specialization limit exceeded "
                        f"while materializing '{requested}'; {unique_count} unique "
                        f"concrete signatures requested, limit {self.max_template_specializations} "
                        f"from {self.template_specialization_limit_source}. "
                        f"Suggested action: {suggested_action}.",
                        limit=self.max_template_specializations,
                        limit_source=self.template_specialization_limit_source,
                        unique_specialization_count=unique_count,
                        requested_signature=requested,
                        suggested_action=suggested_action,
                    )
                preferred_name = self._template_specialization_identifier(
                    template.name, list(ordered_arguments)
                )
                prior_owner = generated_name_owners.get(preferred_name)
                if prior_owner is not None and prior_owner != key:
                    self._raise_constrained_free_function_error(
                        code,
                        call,
                        "enabled specializations would produce the same concrete "
                        "helper name",
                    )
                specialized_name = preferred_name
                suffix = 1
                while specialized_name in reserved_identifiers:
                    specialized_name = f"{preferred_name}_{suffix}"
                    suffix += 1
                if specialization is not None:
                    materialized = (
                        self._materialize_explicit_template_function_specialization(
                            specialization, specialized_name
                        )
                    )
                else:
                    materialized = self._materialize_template_function_with_name(
                        template,
                        list(ordered_arguments),
                        specialized_name,
                        host_name=None,
                    )
                if not materialized:
                    self._raise_constrained_free_function_error(
                        code,
                        call,
                        "the uniquely enabled overload could not be materialized",
                    )
                # Type the body's arguments from the structs the source declares.
                materialized = canonicalize_struct_template_ids(
                    self,
                    materialized,
                    {struct.name for struct in structs},
                )
                reserved_identifiers.add(specialized_name)
                generated_name_owners[preferred_name] = key
                materialized_names[key] = specialized_name
                unreachable_caller_scope(self).record_instantiation(
                    template, tuple(ordered_arguments), specialized_name
                )
                placed, declaration_offset = place_in_namespace(
                    template.namespace, materialized
                )
                materialization_start = len(code) + 2 + declaration_offset
                code += "\n\n" + placed
                body_start = self._find_next_top_level_char(materialized, 0, "{")
                owner_position = (
                    specialization["span"][0]
                    if specialization is not None
                    else template.body_start
                )
                dependent_parameters = {
                    name
                    for name, declared_type in zip(
                        self._parameter_identifier_names(parameters),
                        self._parameter_declared_types(parameters),
                    )
                    if set(IDENTIFIER_RE.findall(declared_type))
                    & set(template.template_parameters)
                }
                for nested in self._find_static_constexpr_calls(code, set(by_name)):
                    if nested.span[0] <= materialization_start + body_start:
                        continue
                    if self._function_reference_is_member_call(
                        code, nested.span[0], set()
                    ):
                        continue
                    pending_calls.append(
                        (
                            nested,
                            owner_position,
                            instantiation_position,
                            key,
                            specialization is None
                            and any(
                                self._strip_enclosing_parens(argument.strip())
                                in dependent_parameters
                                for argument in nested.arguments
                            ),
                        )
                    )

            replacements.append(
                (
                    call.span[0],
                    call.argument_open,
                    specialization_reference(template.namespace, specialized_name),
                )
            )

        if not handled_template_spans:
            return code
        # Calls this pass does not bind, in template bodies and in functions no
        # entry point reaches yet, keep the templates they can see for a later pass.
        processed = {id(call) for call in calls} - deferred_call_ids
        unbound_references = [
            reference
            for reference in references
            if id(reference) not in processed
            and not self._function_reference_is_member_call(
                code, reference.span[0], set()
            )
            and not defines_function(masked_code, reference.span[1])
        ]
        removable = removable_constrained_templates(
            handled_template_spans,
            [*constrained, *carriers],
            unbound_references,
            native_fallback_names,
            lambda template, reference: self._metal_call_reference_namespace(
                reference.qualified_name
            )
            is None
            or self._metal_template_function_call_visible(
                template,
                reference.qualified_name,
                reference.span[0],
                namespace_visibility,
                None,
            ),
            self._containing_span,
            deferred_carrier_calls,
        )
        replacements.extend((start, end, "") for start, end in removable)
        resolved = self._apply_text_replacements(code, replacements)
        return resolved

    class _UnrecognizedConstraint(Exception):
        """A SFINAE constraint outside the small recognized set — clean-fail."""

    def _elide_used_resolved_static_data_members(
        self,
        code: str,
        eligible_struct_names: Set[str],
    ) -> str:
        """Inline used compile-time fields from lowered data-only functors.

        Metal permits ``static constexpr constant`` fields on source functors,
        but CrossGL structs are data-only and the round trip would otherwise
        render those fields as instance references (for example
        ``constant float& init [[static]]``).  Only concrete, fully lowered
        structs named by the caller participate.  A declaration is removed only
        when every observed use is a qualified owner access or an access through
        a receiver whose lexical type resolves back to that exact struct.
        Mutable, cyclic/unresolved, constructor-bearing, and ambiguously accessed
        members remain untouched.
        """
        if not eligible_struct_names or "static" not in code:
            return code

        structs = self._find_concrete_struct_definitions(code)
        if not structs:
            return code
        struct_spans = [struct.span for struct in structs]

        owner_name_counts: Dict[str, int] = {}
        for struct in structs:
            owner_name_counts[struct.name] = owner_name_counts.get(struct.name, 0) + 1

        structs_by_owner: Dict[str, _MetalStructDefinition] = {}
        for struct in structs:
            qualified_name = struct.qualified_name or struct.name
            structs_by_owner[qualified_name] = struct
            if owner_name_counts[struct.name] == 1:
                structs_by_owner[struct.name] = struct

        candidate_members: Dict[
            Tuple[str, str],
            Tuple[_MetalStructDefinition, _MetalDataMember, str],
        ] = {}
        blocked_members: Set[Tuple[str, str]] = set()
        hard_blocked_members: Set[Tuple[str, str]] = set()
        storage_required_members: Set[Tuple[str, str]] = set()
        storage_use_spans: Dict[
            Tuple[str, str],
            List[Tuple[int, int]],
        ] = {}
        for struct in structs:
            qualified_name = struct.qualified_name or struct.name
            if (
                struct.name not in eligible_struct_names
                and qualified_name not in eligible_struct_names
            ):
                continue
            # Constructors and residual member declarations can still observe
            # object identity or fields internally.  Do not alter such structs.
            if (
                struct.methods
                or struct.template_methods
                or struct.constructors
                or struct.member_prototype_names
            ):
                continue
            resolved = self._resolved_static_data_member_initializers(struct)
            for member in struct.data_members:
                initializer = resolved.get(member.name)
                qualifiers = set(IDENTIFIER_RE.findall(member.type_text))
                if (
                    initializer is None
                    or member.span is None
                    or "static" not in qualifiers
                    or not qualifiers.intersection({"const", "constexpr"})
                ):
                    continue
                declaration = code[member.span[0] : member.span[1]]
                declarator, _default = self._split_top_level_assignment(declaration)
                if len(self._split_top_level_commas(declarator)) != 1:
                    continue
                key = (qualified_name, member.name)
                candidate_members[key] = (struct, member, initializer)
                # A retained member initializer that still depends on this field
                # makes removing the declaration unsafe.  Materialized structs
                # normally substitute such dependencies before this phase.
                for other in struct.data_members:
                    if (
                        other is member
                        or other.default is None
                        or member.name not in self._static_initializer_dependencies(
                            other.default,
                            [member.name],
                        )
                    ):
                        continue
                    blocked_members.add(key)
                    hard_blocked_members.add(key)

        if not candidate_members:
            return code

        keys_by_member_name: Dict[str, Set[Tuple[str, str]]] = {}
        for key in candidate_members:
            keys_by_member_name.setdefault(key[1], set()).add(key)

        ignored_spans = sorted(
            self._find_comment_and_literal_spans(code)
            + struct_spans
            + self._find_template_declaration_spans(code)
        )
        masked_code = self._mask_comments_and_literals(code)
        replacements: List[Tuple[int, int, str]] = []
        used_members: Set[Tuple[str, str]] = set()

        qualified_pattern = re.compile(
            r"(?<![A-Za-z0-9_])(?P<owner>[A-Za-z_][A-Za-z0-9_]*"
            r"(?:\s*::\s*[A-Za-z_][A-Za-z0-9_]*)*)\s*::\s*"
            r"(?P<member>[A-Za-z_][A-Za-z0-9_]*)\b"
        )
        for match in qualified_pattern.finditer(code):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            owner_name = re.sub(r"\s*::\s*", "::", match.group("owner"))
            owner = structs_by_owner.get(owner_name)
            if owner is None:
                continue
            key = (owner.qualified_name or owner.name, match.group("member"))
            candidate = candidate_members.get(key)
            if candidate is None:
                continue
            if self._static_member_use_requires_declared_storage(
                masked_code,
                match.start(),
            ):
                blocked_members.add(key)
                storage_required_members.add(key)
                storage_use_spans.setdefault(key, []).append(match.span())
                used_members.add(key)
                continue
            initializer = self._resolve_qualified_static_constant_expression(
                candidate[2],
                structs_by_owner,
                resolving={key},
            )
            replacements.append(
                (
                    match.start(),
                    match.end(),
                    self._static_initializer_reference(initializer),
                )
            )
            used_members.add(key)

        local_variable_types = self._collect_local_variable_types(code, struct_spans)
        parameter_buffer_types: _MetalPositionedBufferTypes = {}
        self._collect_function_parameter_types(
            code,
            struct_spans,
            parameter_buffer_types,
            local_variable_types,
        )
        indirect_struct_types = self._collect_struct_variable_types(
            code,
            {struct.name for struct in structs},
            struct_spans,
            include_indirect=True,
        )
        receiver_types: Dict[str, List[Tuple[int, str]]] = {
            name: list(entries) for name, entries in local_variable_types.items()
        }
        for name, entries in indirect_struct_types.items():
            receiver_types.setdefault(name, []).extend(entries)
        for name, entries in parameter_buffer_types.items():
            receiver_types.setdefault(name, []).extend(
                (position, subscriptable.pointer_type)
                for position, subscriptable in entries
            )
        for entries in receiver_types.values():
            entries.sort(key=lambda item: item[0])

        lexical_scopes = self._find_lexical_brace_scopes(code)
        local_shadows = self._local_variable_shadow_scopes(code)

        def active_receiver_type(
            receiver_name: str,
            position: int,
        ) -> Optional[str]:
            best_position = -1
            best_type: Optional[str] = None
            for declaration_position, declared_type in receiver_types.get(
                receiver_name, ()
            ):
                if declaration_position > position:
                    break
                scope_start, scope_end = self._innermost_lexical_scope(
                    lexical_scopes,
                    declaration_position,
                    len(code),
                )
                if not (scope_start <= position < scope_end):
                    continue
                if declaration_position >= best_position:
                    best_position = declaration_position
                    best_type = declared_type
            if best_type is None:
                return None

            active_shadows = [
                declaration_position
                for declaration_position, scope_start, scope_end in local_shadows.get(
                    receiver_name, ()
                )
                if declaration_position <= position
                and scope_start <= position < scope_end
            ]
            if active_shadows:
                shadow_position = max(active_shadows)
                same_declaration = (
                    best_position <= shadow_position
                    and re.search(r"[;{}]", code[best_position:shadow_position]) is None
                )
                if shadow_position > best_position and not same_declaration:
                    return None
            return best_type

        candidate_structs_by_type: Dict[str, _MetalStructDefinition] = {}
        for struct, _member, _initializer in candidate_members.values():
            qualified_name = struct.qualified_name or struct.name
            candidate_structs_by_type[qualified_name] = struct
            if owner_name_counts[struct.name] == 1:
                candidate_structs_by_type[struct.name] = struct

        receiver_pattern = re.compile(
            r"\b(?P<receiver>[A-Za-z_][A-Za-z0-9_]*)\s*"
            r"(?P<access>\.|->)\s*"
            r"(?P<member>[A-Za-z_][A-Za-z0-9_]*)\b"
        )
        for match in receiver_pattern.finditer(code):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            member_name = match.group("member")
            member_keys = keys_by_member_name.get(member_name)
            if not member_keys:
                continue
            declared_type = active_receiver_type(
                match.group("receiver"),
                match.start(),
            )
            if declared_type is None:
                blocked_members.update(member_keys)
                hard_blocked_members.update(member_keys)
                continue
            indirection_match = re.search(r"(?P<indirection>[*&]+)\s*$", declared_type)
            indirection = (
                indirection_match.group("indirection")
                if indirection_match is not None
                else ""
            )
            base_type = (
                declared_type[: indirection_match.start()].strip()
                if indirection_match is not None
                else declared_type
            )
            base_type = self._normalize_inferred_type(base_type)
            owner = candidate_structs_by_type.get(base_type)
            if owner is None:
                # A proven scalar or different aggregate is a same-named shadow,
                # not an ambiguous use of this static field.
                continue
            key = (owner.qualified_name or owner.name, member_name)
            candidate = candidate_members.get(key)
            if candidate is None:
                continue
            if (match.group("access") == "->") != ("*" in indirection):
                blocked_members.add(key)
                hard_blocked_members.add(key)
                continue
            if self._static_member_use_requires_declared_storage(
                masked_code,
                match.start(),
            ):
                blocked_members.add(key)
                storage_required_members.add(key)
                storage_use_spans.setdefault(key, []).append(match.span())
                used_members.add(key)
                continue
            initializer = self._resolve_qualified_static_constant_expression(
                candidate[2],
                structs_by_owner,
                resolving={key},
            )
            replacements.append(
                (
                    match.start(),
                    match.end(),
                    self._static_initializer_reference(initializer),
                )
            )
            used_members.add(key)

        hoisted_members = storage_required_members - hard_blocked_members
        occupied_names = set(IDENTIFIER_RE.findall(masked_code))
        insertion_text: Dict[int, List[str]] = {}
        for key in sorted(hoisted_members):
            struct, member, initializer = candidate_members[key]
            owner_name = struct.qualified_name or struct.name
            base_name = "_crosstl_metal_static_" + re.sub(
                r"[^A-Za-z0-9_]", "_", f"{owner_name}_{member.name}"
            )
            generated_name = base_name
            suffix = 2
            while generated_name in occupied_names:
                generated_name = f"{base_name}_{suffix}"
                suffix += 1
            occupied_names.add(generated_name)

            namespace = owner_name.rpartition("::")[0]
            reference_name = (
                f"{namespace}::{generated_name}" if namespace else generated_name
            )
            for start, end in storage_use_spans.get(key, ()):
                replacements.append((start, end, reference_name))

            global_type = re.sub(
                r"\b(?:static|constexpr)\b",
                " ",
                member.type_text,
            )
            global_type = re.sub(r"\s+", " ", global_type).strip()
            if "constant" not in set(IDENTIFIER_RE.findall(global_type)):
                global_type = f"constant {global_type}"
            resolved_initializer = self._resolve_qualified_static_constant_expression(
                initializer,
                structs_by_owner,
                resolving={key},
            )
            declaration = (
                f"\n{global_type} {generated_name}{member.array_suffix} = "
                f"{resolved_initializer};\n"
            )
            insertion_text.setdefault(struct.span[1], []).append(declaration)

        for position, declarations in sorted(insertion_text.items()):
            replacements.append((position, position, "".join(declarations)))

        removable_members = (used_members - blocked_members) | hoisted_members
        for key in sorted(removable_members):
            _struct, member, _initializer = candidate_members[key]
            assert member.span is not None
            replacements.append((member.span[0], member.span[1], ""))

        if not replacements:
            return code
        return self._apply_text_replacements(code, replacements)

    def _try_rewrite_call_at(
        self,
        code: str,
        ident_start: int,
        ident: str,
        ident_end: int,
        struct_names: Set[str],
        methods_by_struct: Dict[str, Dict[str, _MetalStructMethod]],
        template_methods_by_struct: Dict[str, Dict[str, List[_MetalStructMethod]]],
        operator_call_structs: Set[str],
        variable_types: Dict[str, List[Tuple[int, str]]],
        structs_by_name: Dict[str, _MetalStructDefinition],
        buffer_element_types: _MetalPositionedBufferTypes,
        local_variable_types: Dict[str, List[Tuple[int, str]]],
        instantiated_template_functions: Dict[str, str],
        field_variable_types: Optional[Dict[str, List[Tuple[int, str]]]] = None,
        field_structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
        struct_type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]] = None,
        local_integral_constants: Optional[
            Dict[str, List[_MetalIntegralConstantBinding]]
        ] = None,
        type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]] = None,
        type_alias_fallback_position: Optional[int] = None,
        implicit_receiver_contract: Optional[_MetalReceiverContract] = None,
    ) -> Optional[Tuple[int, str]]:
        # `field_variable_types` / `field_structs_by_name` cover EVERY struct/union
        # type (including method-less data carriers) and are used only to resolve
        # `obj.member` / `obj.member[i]` call arguments; call REWRITING still uses
        # the method-driven `variable_types` / `structs_by_name`.
        if field_variable_types is None:
            field_variable_types = variable_types
        if field_structs_by_name is None:
            field_structs_by_name = structs_by_name
        nested_member_rewrite = self._try_rewrite_nested_member_call_at(
            code,
            ident_start,
            ident,
            ident_end,
            methods_by_struct,
            template_methods_by_struct,
            operator_call_structs,
            variable_types,
            structs_by_name,
            buffer_element_types,
            local_variable_types,
            instantiated_template_functions,
            field_variable_types,
            field_structs_by_name,
            local_integral_constants,
            struct_type_aliases,
            type_aliases,
            type_alias_fallback_position,
            implicit_receiver_contract,
        )
        if nested_member_rewrite is not None:
            return nested_member_rewrite
        # Member access on a previous object (`a.var.m(...)`) is not a struct
        # variable we tracked; skip when the identifier is a member access.
        if self._is_member_identifier_context(code, ident_start):
            return None

        j = ident_end
        while j < len(code) and code[j].isspace():
            j += 1
        if j >= len(code):
            return None

        conversion_type = ident
        conversion_open = j
        if ident == "static_cast" and code[j] == "<":
            angle_close = self._find_matching_angle(code, j)
            if angle_close is not None:
                conversion_type = code[j + 1 : angle_close]
                conversion_open = angle_close + 1
        elif ident in {"vec", "vector", "metal"}:
            type_end = j
            if ident == "metal" and code[j : j + 2] == "::":
                type_end = j + 2
                while type_end < len(code) and code[type_end].isspace():
                    type_end += 1
                _name, consumed = self._read_identifier(code, type_end)
                type_end += consumed
                conversion_type = code[ident_start:type_end]
                conversion_open = type_end
                while type_end < len(code) and code[type_end].isspace():
                    type_end += 1
            if type_end < len(code) and code[type_end] == "<":
                angle_close = self._find_matching_angle(code, type_end)
                if angle_close is not None:
                    conversion_type = code[ident_start : angle_close + 1]
                    conversion_open = angle_close + 1
        while conversion_open < len(code) and code[conversion_open].isspace():
            conversion_open += 1
        if conversion_open < len(code) and code[conversion_open] == "(":
            conversion = self._try_rewrite_implicit_conversion_call(
                code,
                ident_start,
                conversion_type,
                conversion_open,
                buffer_element_types,
                local_variable_types,
                field_variable_types,
                field_structs_by_name,
                type_aliases,
                local_integral_constants,
            )
            if conversion is not None:
                return conversion

        # Qualified static call: `S::m(args)` -> `S__m(args)`. A local alias may
        # name the concrete materialized struct, so bind it to the nearest alias
        # declaration that precedes this call site.
        qualified_struct = None
        if struct_type_aliases is not None:
            qualified_struct = self._resolve_struct_type_alias_at(
                struct_type_aliases, ident, ident_start
            )
        if qualified_struct is None and ident in struct_names:
            qualified_struct = ident
        qualifier_end = j
        if qualified_struct is None and code[j] == "<":
            # A template-id owner such as `Frag<float,1>::m(args)` names the
            # struct materialized for exactly those arguments.
            angle_close = self._find_matching_angle(code, j)
            if angle_close is not None:
                concrete = materialized_struct_for_template_id(
                    self, code[ident_start : angle_close + 1]
                )
                if concrete in struct_names:
                    qualifier_end = angle_close + 1
                    while qualifier_end < len(code) and code[qualifier_end].isspace():
                        qualifier_end += 1
                    if code[qualifier_end : qualifier_end + 2] == "::":
                        qualified_struct = concrete
        if (
            qualified_struct is not None
            and code[qualifier_end : qualifier_end + 2] == "::"
        ):
            j = qualifier_end
            k = j + 2
            while k < len(code) and code[k].isspace():
                k += 1
            member, consumed = self._read_identifier(code, k)
            if member == "template":
                # `S::template m<A>(args)`: the disambiguator precedes the name.
                k += consumed
                while k < len(code) and code[k].isspace():
                    k += 1
                member, consumed = self._read_identifier(code, k)
            if not member:
                return None
            after = k + consumed
            call_suffix = self._member_template_call_suffix(code, after)
            if call_suffix is None:
                return None
            arg_open, explicit_template_arguments = call_suffix
            template_overloads = template_methods_by_struct.get(
                qualified_struct, {}
            ).get(member)
            owner_struct = structs_by_name.get(qualified_struct)
            method = None
            if owner_struct is not None and explicit_template_arguments is None:
                method = self._select_concrete_method_for_call(
                    code,
                    owner_struct,
                    member,
                    arg_open,
                    buffer_element_types,
                    local_variable_types,
                    field_variable_types,
                    field_structs_by_name,
                    allow_template_fallback=bool(template_overloads),
                    template_overloads=template_overloads,
                    require_static=True,
                    type_alias_fallback_position=type_alias_fallback_position,
                )
            if method is not None and explicit_template_arguments is None:
                arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
                if arg_close is None:
                    return None
                self._reject_reference_returning_method_call(
                    code,
                    owner_struct,
                    method,
                    arg_open,
                    arg_close,
                )
                return arg_open, method.free_name
            # A static TEMPLATE member call `S::m(args)`: instantiate from args.
            if template_overloads:
                rewrite = self._instantiate_template_member_call(
                    code,
                    structs_by_name[qualified_struct],
                    template_overloads,
                    receiver=None,
                    arg_open=arg_open,
                    buffer_element_types=buffer_element_types,
                    local_variable_types=local_variable_types,
                    instantiated_template_functions=instantiated_template_functions,
                    structs_by_name=field_structs_by_name,
                    variable_types=field_variable_types,
                    template_methods_by_struct=template_methods_by_struct,
                    methods_by_struct=methods_by_struct,
                    operator_call_structs=operator_call_structs,
                    rewrite_structs_by_name=structs_by_name,
                    explicit_template_arguments=explicit_template_arguments,
                    local_integral_constants=local_integral_constants,
                    argument_type_aliases=type_aliases,
                    argument_type_fallback_position=type_alias_fallback_position,
                )
                if rewrite is not None:
                    return rewrite
            return None

        # Temporary construction followed by `operator()`:
        # `S{}(args)` / `S()(args)` -> `S__operator_call__...__temporary(args)`.
        # Nested temporaries are rewritten bottom-up and composed into the outer
        # replacement. Only proven stateless default construction is lowered;
        # constructor state or effects require ordered lifetime materialization.
        temporary_call = self._temporary_functor_call(code, j)
        if temporary_call is not None and self._temporary_functor_is_candidate(
            code, ident_start, temporary_call
        ):
            struct = structs_by_name.get(ident)
            if struct is None or ident not in operator_call_structs:
                self._raise_temporary_functor_error(
                    code,
                    ident,
                    ident_start,
                    temporary_call,
                    reason="temporary-functor-type-unresolved",
                    detail=(
                        "the temporary type does not resolve to a concrete struct "
                        "with operator()"
                    ),
                )
            if not self._temporary_functor_is_stateless_default(
                code, struct, temporary_call
            ):
                self._raise_temporary_functor_error(
                    code,
                    ident,
                    ident_start,
                    temporary_call,
                    reason="temporary-functor-constructor-stateful",
                    detail=(
                        "constructor arguments, object state, or constructor effects "
                        "cannot be sequenced safely by the stateless wrapper"
                    ),
                )

            rewritten_arguments = self._rewrite_nested_temporary_functor_calls(
                code,
                temporary_call.argument_open + 1,
                temporary_call.argument_close,
                struct_names,
                methods_by_struct,
                template_methods_by_struct,
                operator_call_structs,
                variable_types,
                structs_by_name,
                buffer_element_types,
                local_variable_types,
                instantiated_template_functions,
                field_variable_types,
                field_structs_by_name,
                local_integral_constants,
                struct_type_aliases,
                type_aliases,
                type_alias_fallback_position,
            )
            template_overloads = template_methods_by_struct.get(ident, {}).get(
                "operator()"
            )
            method = self._select_concrete_method_for_call(
                code,
                struct,
                "operator()",
                temporary_call.argument_open,
                buffer_element_types,
                local_variable_types,
                field_variable_types,
                field_structs_by_name,
                allow_template_fallback=bool(template_overloads),
                template_overloads=template_overloads,
                require_static=False,
                type_alias_fallback_position=type_alias_fallback_position,
                receiver_contract=_MetalReceiverContract(
                    struct_name=ident,
                    value_category="rvalue",
                    source_type=f"thread {ident}",
                ),
                call_start=ident_start,
            )
            if method is not None:
                return self._build_temporary_operator_call_rewrite(
                    code,
                    struct,
                    method,
                    temporary_call.argument_open,
                    instantiated_template_functions,
                    rewritten_arguments=rewritten_arguments,
                )
            if template_overloads:
                return self._instantiate_temporary_template_operator_call(
                    code,
                    struct,
                    template_overloads,
                    temporary_call.argument_open,
                    buffer_element_types,
                    local_variable_types,
                    instantiated_template_functions,
                    structs_by_name=field_structs_by_name,
                    variable_types=field_variable_types,
                    template_methods_by_struct=template_methods_by_struct,
                    methods_by_struct=methods_by_struct,
                    operator_call_structs=operator_call_structs,
                    rewrite_structs_by_name=structs_by_name,
                    local_integral_constants=local_integral_constants,
                    argument_type_aliases=type_aliases,
                    argument_type_fallback_position=type_alias_fallback_position,
                    rewritten_arguments=rewritten_arguments,
                )
            self._raise_temporary_functor_error(
                code,
                ident,
                ident_start,
                temporary_call,
                reason="temporary-functor-overload-unresolved",
                detail="no concrete operator() overload matches the call arguments",
            )

        # Prefer the lexically visible declaration over the legacy positioned
        # type map. If the current declaration cannot be resolved, it must shadow
        # same-named receivers from outer or sibling functions.
        receiver_declaration, struct_type = self._receiver_struct_type_at(
            code,
            ident,
            ident_start,
            struct_names,
            struct_type_aliases,
        )
        if receiver_declaration is None:
            struct_type = self._resolve_declared_type_at(
                variable_types,
                ident,
                ident_start,
            )
        receiver_contract = None
        if struct_type is not None:
            receiver_contract = (
                implicit_receiver_contract
                if ident == "self" and implicit_receiver_contract is not None
                else self._receiver_contract_for_named_value(
                    code,
                    struct_type,
                    ident,
                    ident_start,
                    type_aliases=struct_type_aliases,
                )
            )

        # An element of an array of structs is a receiver like a named variable:
        # `tiles[k].m(args)` -> `S__m(tiles[k], args)`.
        receiver = ident
        if (
            code[j] == "["
            and struct_type is not None
            and receiver_declaration is not None
            and not re.search(r"[*&]", receiver_declaration.raw_type_text)
        ):
            element_end = array_element_receiver_end(
                code, j, array_rank_at(code, receiver_declaration.declaration_end)
            )
            if element_end is not None:
                receiver = code[ident_start:element_end]
                j = element_end
                while j < len(code) and code[j].isspace():
                    j += 1

        # Instance method call: `var.m(args)` -> `S__m(var, args)`.
        if code[j] == ".":
            k = j + 1
            while k < len(code) and code[k].isspace():
                k += 1
            member, consumed = self._read_identifier(code, k)
            if not member:
                return None
            if member == "template":
                k += consumed
                while k < len(code) and code[k].isspace():
                    k += 1
                member, consumed = self._read_identifier(code, k)
                if not member:
                    return None
            after = k + consumed
            call_suffix = self._member_template_call_suffix(code, after)
            if call_suffix is None:
                # A data-member access (`var.field`) — leave untouched.
                return None
            arg_open, explicit_template_arguments = call_suffix
            if struct_type is None:
                if self._receiver_has_builtin_member_type_at(
                    code, ident, ident_start, receiver_declaration
                ):
                    return None
                arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
                location_end = arg_close + 1 if arg_close is not None else arg_open + 1
                receiver_type = (
                    receiver_declaration.raw_type_text.strip()
                    if receiver_declaration is not None
                    else "<unknown>"
                )
                raise MetalStructMethodError(
                    "Cannot lower concrete member call "
                    f"'{ident}.{member}' because receiver declaration "
                    f"'{receiver_type} {ident}' does not resolve to a concrete "
                    "struct in its lexical scope.",
                    method_name=member,
                    requested_signature=f"{ident}.{member}",
                    suggested_action=(
                        "make the receiver alias resolve to a concrete struct "
                        "within the current function scope"
                    ),
                    source_location=self._source_location_for_offsets(
                        code,
                        ident_start,
                        location_end,
                    ),
                    missing_capabilities=("metal.member-overload-resolution",),
                    reason="concrete-receiver-type-unresolved",
                    receiver_type=receiver_type,
                )
            owner_struct = structs_by_name.get(struct_type)
            reference_methods = (
                self._concrete_reference_methods(owner_struct, member, code, arg_open)
                if owner_struct is not None
                else []
            )
            if reference_methods and explicit_template_arguments is None:
                receiver_is_const = (
                    None
                    if receiver_contract is not None
                    and "constant" in receiver_contract.address_spaces
                    else (
                        receiver_contract.is_readonly
                        if receiver_contract is not None
                        else self._simple_receiver_constness(
                            code,
                            struct_type,
                            ident,
                            ident_start,
                            type_aliases=struct_type_aliases,
                        )
                    )
                )
                method = self._select_direct_reference_method(
                    code,
                    owner_struct,
                    reference_methods,
                    arg_open,
                    receiver_is_const=receiver_is_const,
                )
                return self._build_direct_reference_accessor_rewrite(
                    code,
                    owner_struct,
                    method,
                    receiver=receiver,
                    arg_open=arg_open,
                    call_start=ident_start,
                )
            template_overloads = template_methods_by_struct.get(struct_type, {}).get(
                member
            )
            method = None
            if owner_struct is not None and explicit_template_arguments is None:
                method = self._select_concrete_method_for_call(
                    code,
                    owner_struct,
                    member,
                    arg_open,
                    buffer_element_types,
                    local_variable_types,
                    field_variable_types,
                    field_structs_by_name,
                    allow_template_fallback=bool(template_overloads),
                    template_overloads=template_overloads,
                    require_static=False,
                    type_alias_fallback_position=type_alias_fallback_position,
                    receiver_contract=receiver_contract,
                    call_start=ident_start,
                )
            if method is not None and explicit_template_arguments is None:
                return self._build_instance_call_rewrite(
                    code,
                    receiver,
                    method,
                    arg_open,
                    struct_name=owner_struct,
                )
            if owner_struct is not None and explicit_template_arguments is None:
                static_method = self._select_concrete_method_for_call(
                    code,
                    owner_struct,
                    member,
                    arg_open,
                    buffer_element_types,
                    local_variable_types,
                    field_variable_types,
                    field_structs_by_name,
                    allow_template_fallback=bool(template_overloads),
                    template_overloads=template_overloads,
                    require_static=True,
                    type_alias_fallback_position=type_alias_fallback_position,
                    call_start=ident_start,
                )
                if static_method is not None:
                    arg_close = self._find_matching_delimiter(
                        code,
                        arg_open,
                        "(",
                        ")",
                    )
                    if arg_close is None:
                        return None
                    self._reject_reference_returning_method_call(
                        code,
                        owner_struct,
                        static_method,
                        arg_open,
                        arg_close,
                    )
                    return arg_open, static_method.free_name
            # An instance TEMPLATE member call `var.m(args)`.
            if template_overloads:
                return self._instantiate_template_member_call(
                    code,
                    structs_by_name[struct_type],
                    template_overloads,
                    receiver=receiver,
                    receiver_is_const=(
                        None
                        if receiver_contract is not None
                        and "constant" in receiver_contract.address_spaces
                        else (
                            receiver_contract.is_readonly
                            if receiver_contract is not None
                            else self._simple_receiver_constness(
                                code,
                                struct_type,
                                ident,
                                ident_start,
                                type_aliases=struct_type_aliases,
                            )
                        )
                    ),
                    arg_open=arg_open,
                    buffer_element_types=buffer_element_types,
                    local_variable_types=local_variable_types,
                    instantiated_template_functions=instantiated_template_functions,
                    structs_by_name=field_structs_by_name,
                    variable_types=field_variable_types,
                    template_methods_by_struct=template_methods_by_struct,
                    methods_by_struct=methods_by_struct,
                    operator_call_structs=operator_call_structs,
                    rewrite_structs_by_name=structs_by_name,
                    explicit_template_arguments=explicit_template_arguments,
                    local_integral_constants=local_integral_constants,
                    argument_type_aliases=type_aliases,
                    argument_type_fallback_position=type_alias_fallback_position,
                )
            return None

        # Functor call: `var(args)` -> `S__operator_call(var, ...)`.
        if code[j] == "(" and struct_type in operator_call_structs:
            template_overloads = template_methods_by_struct.get(struct_type, {}).get(
                "operator()"
            )
            owner_struct = structs_by_name.get(struct_type)
            method = (
                self._select_concrete_method_for_call(
                    code,
                    owner_struct,
                    "operator()",
                    j,
                    buffer_element_types,
                    local_variable_types,
                    field_variable_types,
                    field_structs_by_name,
                    allow_template_fallback=bool(template_overloads),
                    template_overloads=template_overloads,
                    require_static=False,
                    type_alias_fallback_position=type_alias_fallback_position,
                    receiver_contract=receiver_contract,
                    call_start=ident_start,
                )
                if owner_struct is not None
                else None
            )
            if method is not None:
                return self._build_instance_call_rewrite(
                    code,
                    ident,
                    method,
                    j,
                    struct_name=owner_struct,
                )
            # A template `operator()` functor call `var(args)`.
            if template_overloads:
                return self._instantiate_template_member_call(
                    code,
                    structs_by_name[struct_type],
                    template_overloads,
                    receiver=ident,
                    arg_open=j,
                    buffer_element_types=buffer_element_types,
                    local_variable_types=local_variable_types,
                    instantiated_template_functions=instantiated_template_functions,
                    structs_by_name=field_structs_by_name,
                    variable_types=field_variable_types,
                    template_methods_by_struct=template_methods_by_struct,
                    methods_by_struct=methods_by_struct,
                    operator_call_structs=operator_call_structs,
                    rewrite_structs_by_name=structs_by_name,
                    local_integral_constants=local_integral_constants,
                    argument_type_aliases=type_aliases,
                    argument_type_fallback_position=type_alias_fallback_position,
                )

        return None

    @staticmethod
    def _resolve_struct_type_alias_at(
        type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
        name: str,
        position: int,
    ) -> Optional[str]:
        binding = MetalPreprocessor._resolve_type_alias_binding_at(
            type_aliases,
            name,
            position,
        )
        return None if binding is None else binding.target

    def _infer_argument_type(
        self,
        argument: str,
        buffer_element_types: _MetalBufferTypeView,
        local_variable_types: Dict[str, str],
        struct_field_types: Optional[Dict[str, Dict[str, str]]] = None,
        structs_by_name: Optional[Dict[str, "_MetalStructDefinition"]] = None,
    ) -> Optional[str]:
        # Conservatively infer the concrete type of a call-argument expression.
        # Returns None (un-inferable) for anything outside the recognized shapes.
        # `structs_by_name`, when supplied, lets construction / functor-call
        # shapes (`T{...}`, `F{}(args)`) be typed from the tracked struct set; it
        # is optional so the rule stays a pure, unit-testable function.
        expr = self._strip_template_argument_comments(argument).strip()
        if not expr:
            return None
        # Strip a single fully-enclosing paren group: `(expr)` has the type of
        # `expr` (but `T(expr)` is a cast handled below, so only strip when the
        # paren is at position 0).
        while (
            expr.startswith("(")
            and self._find_matching_delimiter(expr, 0, "(", ")") == len(expr) - 1
        ):
            expr = expr[1:-1].strip()
            if not expr:
                return None

        # `metal::bool_constant<true>{}` has its library tag type (integral_constant_tags.py).
        library_tag = library_tag_value_type(self, expr)
        if library_tag is not None:
            return library_tag

        def infer_operand(operand: str) -> Optional[str]:
            return self._infer_argument_type(
                operand,
                buffer_element_types,
                local_variable_types,
                struct_field_types,
                structs_by_name,
            )

        # A conditional expression has lower precedence than every shape below.
        conditional = infer_conditional_type(self, expr, infer_operand)
        if conditional is not None:
            return conditional

        # Built-in scalar postfix updates retain the declared value type. Keep
        # the source expression unchanged so specialization does not consume or
        # duplicate its mutation; user-defined update operators remain unresolved.
        postfix = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\+\+|--)", expr)
        if postfix is not None:
            declared = local_variable_types.get(postfix.group(1), "")
            if re.search(
                r"\bconst\b", getattr(declared, "pointer_type", None) or declared
            ):
                return None
            normalized = self._normalize_inferred_expression_type(declared)
            scalar = self._scalar_and_width(normalized or "")
            if scalar is not None and scalar[1] == 1 and scalar[0] != "bool":
                return normalized
            return None

        bitcast = re.match(
            r"(?:metal\s*::\s*)?as_type\s*<([A-Za-z_][A-Za-z0-9_]*)>\s*\(", expr
        )
        if bitcast is not None:
            if (
                self._inferred_bitcast_shadowed
                or "as_type" in local_variable_types
                or "as_type" in buffer_element_types
            ):
                return None
            target = self._normalize_inferred_expression_type(bitcast.group(1))
            start = bitcast.end() - 1
            end = self._find_matching_delimiter(expr, start, "(", ")")
            if end == len(expr) - 1 and self._is_metal_scalar_or_vector_type(
                target or ""
            ):
                arguments = self._split_top_level_commas(expr[start + 1 : end])
                if len(arguments) == 1 and arguments[0].strip():
                    return target
            return None

        # Address-of requires declaration provenance, whether a local value or
        # a buffer/array element. Value-only type records cannot prove storage.
        if expr.startswith("&"):
            operand = self._strip_enclosing_parens(expr[1:].strip())
            if IDENTIFIER_RE.fullmatch(operand):
                declared = local_variable_types.get(operand)
                if isinstance(declared, _MetalAddressableValueType):
                    return declared.pointer_type
            member_pointer = addressed_member_pointer_type(
                self, operand, local_variable_types, struct_field_types
            )
            if member_pointer is not None:
                return member_pointer
            return self._infer_addressed_buffer_element_pointer_type(
                expr,
                buffer_element_types,
                local_variable_types,
                struct_field_types,
                structs_by_name,
            )

        # Cast: `static_cast<T>(expr)` -> T. A pointer target is inferable only
        # when it names exactly one explicit Metal address space.
        static_cast = re.match(r"static_cast\s*<(?P<type>.+)>\s*\(", expr, re.DOTALL)
        if static_cast is not None:
            angle_start = expr.find("<")
            angle_end = self._find_matching_angle(expr, angle_start)
            if angle_end is not None:
                paren_start = angle_end + 1
                while paren_start < len(expr) and expr[paren_start].isspace():
                    paren_start += 1
                if paren_start < len(expr) and expr[paren_start] == "(":
                    paren_end = self._find_matching_delimiter(
                        expr, paren_start, "(", ")"
                    )
                    if paren_end == len(expr) - 1:
                        return self._normalize_inferred_expression_type(
                            expr[angle_start + 1 : angle_end]
                        )

        # A pointer conversion has its target type ([expr.reinterpret.cast]/1,
        # [expr.cast]/1): `reinterpret_cast<device T*>(p)` and `(device T*)p`
        # are `device T*`, so a call or an `auto` local built from one deduces
        # template arguments like an explicitly typed pointer. Only pointer
        # targets naming exactly one Metal address space are admitted.
        pointer_cast = pointer_conversion_type(self, expr)
        if pointer_cast is not None:
            return pointer_cast

        # Dereference of one explicit C-style pointer cast has the cast's
        # pointee type: `*(thread Value*)(&bits)` -> `Value`. A dereference of a
        # tracked pointer/array is also exact: `*row`, `*(row + index)`, and the
        # MLX reduction shape `*row++` all retain the declared pointee type. The
        # postfix update is admitted only on one bare tracked pointer and remains
        # in the rewritten call expression, so its side effect is neither erased
        # nor duplicated. Prefix updates, arbitrary calls, nested dereferences,
        # and unbalanced/compound tails remain unresolved rather than guessed.
        if expr.startswith("*") and not expr.startswith("**"):
            cast = expr[1:].lstrip()
            cast_type = self._c_style_pointer_cast_type(cast)
            if cast_type is not None:
                return self._pointer_pointee_value_type(cast_type)

            operand = expr[1:].strip()
            parenthesized = False
            if operand.startswith("("):
                operand_close = self._find_matching_delimiter(operand, 0, "(", ")")
                if operand_close == len(operand) - 1:
                    operand = operand[1:-1].strip()
                    parenthesized = True

            postfix = re.fullmatch(
                r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<update>\+\+|--)",
                operand,
            )
            if postfix is not None:
                name = postfix.group("name")
                pointer_type = self._buffer_pointer_type(buffer_element_types, name)
                if pointer_type is None and name in local_variable_types:
                    pointer_type = self._normalize_known_address_space_pointer_type(
                        local_variable_types[name]
                    )
                if pointer_type is not None:
                    pointee = self._pointer_pointee_value_type(pointer_type)
                    if pointee is not None:
                        return pointee
            elif IDENTIFIER_RE.fullmatch(operand):
                element_type = self._buffer_element_type(buffer_element_types, operand)
                if element_type is not None:
                    return self._normalize_inferred_type(element_type)
                pointer_type = self._normalize_known_address_space_pointer_type(
                    local_variable_types.get(operand, "")
                )
                if pointer_type is not None:
                    pointee = self._pointer_pointee_value_type(pointer_type)
                    if pointee is not None:
                        return pointee
            elif parenthesized:
                pointer_type = self._infer_pointer_arithmetic_type(
                    operand,
                    buffer_element_types,
                    local_variable_types,
                    struct_field_types,
                    structs_by_name,
                )
                if pointer_type is not None:
                    pointee = self._pointer_pointee_value_type(pointer_type)
                    if pointee is not None:
                        return pointee

        # Literal types.
        literal_type = self._infer_literal_type(expr)
        if literal_type is not None:
            return literal_type
        # A unary sign preserves these numeric literal types. Do not infer
        # user-defined operators or compound expressions from their first token.
        if expr[0] in {"+", "-"}:
            signed_literal_type = self._infer_literal_type(expr[1:].strip())
            if signed_literal_type not in {None, "bool"}:
                return signed_literal_type

        # Functor construction-and-call temporary `F{}(args)` / `F{...}(args)` /
        # `F()(args)` -> the result type of `F::operator()` for those arguments.
        # Checked before the functional-cast rule (which recognizes a single
        # `T(expr)` group) because this shape has TWO trailing groups.
        functor_call = self._infer_functor_construction_call_type(
            expr,
            buffer_element_types,
            local_variable_types,
            struct_field_types,
            structs_by_name,
        )
        if functor_call is not None:
            return functor_call

        # Construction of a recognized type by brace init `T{...}` (no trailing
        # call) -> T. Covers `complex64_t{re, im}` and vector aggregate inits.
        construction = self._infer_braced_construction_type(expr, structs_by_name)
        if construction is not None:
            return construction

        # Parenthesized construction of a tracked struct, `T(args...)`, has type
        # T when one balanced argument group spans the whole expression.  Keep
        # this narrower than generic function-call inference: an untracked name,
        # qualified tail, or second call group remains unresolved.  This admits
        # implicit scalar conversion from a temporary such as
        # `float(fp8_e8m0(scale))` without guessing arbitrary call return types.
        functional_construction = re.match(
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(", expr
        )
        if functional_construction is not None and structs_by_name is not None:
            type_name = functional_construction.group("name")
            paren_start = expr.find("(", functional_construction.end("name"))
            paren_end = self._find_matching_delimiter(expr, paren_start, "(", ")")
            resolved_type = self._inferred_aggregate_type(type_name, structs_by_name)
            if paren_end == len(expr) - 1 and resolved_type is not None:
                return resolved_type

        # Resolve one declared array/pointer layer per balanced subscript. Do not
        # collapse an array of pointers to its pointee before the second index.
        bracket = self._find_next_top_level_char(expr, 0, "[", track_angles=False)
        if bracket is not None and expr.endswith("]"):
            element = self._infer_subscript_base_element_type(
                expr[:bracket].strip(), buffer_element_types, struct_field_types
            )
            if element is None:
                element = self._vector_local_component_type(
                    expr[:bracket].strip(), local_variable_types
                )
            while element is not None:
                close = self._find_matching_delimiter(expr, bracket, "[", "]")
                if close is None or not expr[bracket + 1 : close].strip():
                    break
                if close == len(expr) - 1:
                    return element
                bracket = close + 1
                while bracket < len(expr) and expr[bracket].isspace():
                    bracket += 1
                if bracket >= len(expr) or expr[bracket] != "[":
                    break
                element = self._subscript_declared_element_type(element)

        element_member = self._infer_array_element_member_type(
            expr,
            buffer_element_types,
            local_variable_types,
            struct_field_types,
            structs_by_name,
        )
        if element_member is not None:
            return element_member

        # A SIMD/quad group built-in that returns its first argument's type
        # (`simd_shuffle_and_fill_up(x, ...)`, `simd_prefix_inclusive_sum(x)`).
        group_builtin = self._infer_group_builtin_call_type(
            expr,
            buffer_element_types,
            local_variable_types,
            struct_field_types,
            structs_by_name,
        )
        if group_builtin is not None:
            return group_builtin

        known_member_call = self._infer_known_member_function_call_type(expr)
        if known_member_call is not None:
            return known_member_call

        # `receiver.method(args)` on a tracked struct (reference_binding.py).
        receiver_call = member_call_result(
            self, expr, local_variable_types, structs_by_name
        )
        if receiver_call is not None:
            return receiver_call[0]

        # A call to a source function whose declarations agree on one return type.
        call_result = infer_call_result_type(
            self, expr, self._call_result_scope, local_variable_types, structs_by_name
        )
        if call_result is not None:
            return call_result

        # Functional cast `T(expr)` where T is a recognized scalar/vector type.
        cast = re.match(r"(?P<type>[A-Za-z_][A-Za-z0-9_]*)\s*\(", expr)
        if cast is not None:
            type_name = cast.group("type")
            paren_start = expr.find("(")
            paren_end = self._find_matching_delimiter(expr, paren_start, "(", ")")
            if (
                paren_end == len(expr) - 1
                and type_name in self._METAL_SCALAR_VECTOR_TYPES
            ):
                return self._normalize_inferred_type(type_name)

        # A bare tracked pointer/array preserves its complete qualified pointer
        # type. Binding later peels the pointee only when the DECLARED parameter
        # itself is a pointer (`device U*`); a by-value `Pointer` binds the full
        # type. Structured entries without a proven address space fail closed.
        if IDENTIFIER_RE.fullmatch(expr) and expr in buffer_element_types:
            return self._buffer_pointer_expression_type(buffer_element_types, expr)

        # Bare local variable -> its declared type.
        if IDENTIFIER_RE.fullmatch(expr) and expr in local_variable_types:
            return self._normalize_inferred_expression_type(local_variable_types[expr])
        # A predefined constant that no local declares has its library type.
        if IDENTIFIER_RE.fullmatch(expr):
            predefined = metal_predefined_constant_type(expr)
            if predefined is not None:
                return self._normalize_inferred_expression_type(predefined)

        # Member access `obj.member` / `obj->member` -> the declared field type.
        # Pointer fields are not value-typeable as a bare access, so a trailing
        # `*` marker is rejected here.
        member_field_type = self._infer_member_access_type(expr, struct_field_types)
        if member_field_type is not None:
            return member_field_type

        # Built-in Metal vector component/swizzle access (`dims.y`, `color.rgb`)
        # -> the scalar element type or a vector of the selected width. The base
        # expression must be a typed local/parameter and every selected component
        # must exist in that vector, so unknown and out-of-range accesses remain
        # un-inferable.
        vector_member_type = self._infer_vector_member_access_type(
            expr, local_variable_types
        )
        if vector_member_type is not None:
            return vector_member_type
        swizzle_type = infer_swizzle_type(
            self,
            expr,
            lambda base: self._infer_argument_type(
                base,
                buffer_element_types,
                local_variable_types,
                struct_field_types,
                structs_by_name,
            ),
        )
        if swizzle_type is not None:
            return swizzle_type

        # Pointer offset `base + integer`, `integer + base`, or `base - integer`
        # preserves the pointer's complete qualified type for template deduction.
        pointer_arithmetic = self._infer_pointer_arithmetic_type(
            expr,
            buffer_element_types,
            local_variable_types,
            struct_field_types,
            structs_by_name,
        )
        if pointer_arithmetic is not None:
            return pointer_arithmetic
        if self._contains_bare_pointer_operand(expr, buffer_element_types):
            return None

        # Binary arithmetic `a + b`, `a - b`, `a * b`, `a / b`, `a % b` (checked last, as a
        # fallback for a COMPOUND expression once every atomic shape above has been
        # ruled out) -> the conservative result type of the usual arithmetic
        # conversion between the two operand types.
        binary = self._infer_binary_arithmetic_type(
            expr,
            buffer_element_types,
            local_variable_types,
            struct_field_types,
            structs_by_name,
        )
        if binary is not None:
            return binary

        return infer_unary_sign_type(self, expr, infer_operand)

    def _find_project_template_instantiations(
        self, code: str
    ) -> List[_MLXKernelInstantiation]:
        # Every entry of a translation unit scans the same expanded text.
        return memoized_instantiations(
            self,
            code,
            self._scan_project_template_instantiations,
            exact_type=MetalPreprocessor,
        )

    def _materialize_template_function_with_name(
        self,
        template: _MetalTemplateFunction,
        template_arguments: List[str],
        function_identifier: str,
        host_name: Optional[str],
    ) -> str:
        if not self._template_arguments_satisfy_parameters(
            template,
            template_arguments,
        ):
            return ""
        substitutions, variadic_bindings = self._template_argument_bindings(
            template,
            template_arguments,
        )
        if not substitutions:
            return ""

        if variadic_bindings:
            materialized_source = self._expand_variadic_function_parameters(
                template.source,
                variadic_bindings,
            )
        else:
            materialized_source = template.source
        materialized = self._replace_identifiers(materialized_source, substitutions)
        materialized = (
            self._reconstruct_materialized_function_array_parameter_declarators(
                materialized_source,
                materialized,
                substitutions,
            )
        )
        materialized = fold_integral_constant_function_arithmetic(self, materialized)
        materialized = self._rename_function_definition(
            materialized,
            template.name,
            function_identifier,
        )
        return_type = self._materialized_template_function_return_type(
            template,
            substitutions,
        )
        if return_type is not None:
            self._record_known_member_function_return_type(
                function_identifier,
                return_type,
            )

        if host_name is not None:
            insertion = f'[[host_name("{host_name}")]]\n'
            materialized = insertion + materialized.lstrip()
        else:
            body_start = self._find_next_top_level_char(materialized, 0, "{")
            declaration = materialized[:body_start]
            if _metal_entry_stage(self, declaration) is None:
                masked = self._mask_comments_and_literals(declaration)
                if template.internal_linkage and not re.search(r"\bstatic\b", masked):
                    materialized = "static " + materialized.lstrip()
                if not re.search(r"\b(?:inline|constexpr)\b", masked):
                    # Implicit instantiations may appear in multiple modules;
                    # anonymous-namespace ownership must remain private as well.
                    materialized = "inline " + materialized.lstrip()
        if not materialized.endswith("\n"):
            materialized += "\n"
        self._materialized_function_names.add(function_identifier)
        self._materialized_function_templates.add((template.namespace, template.source))
        return materialized

    def _find_explicit_template_function_calls(
        self,
        code: str,
        templates_by_name: Dict[str, _MetalTemplateFunction],
        excluded_spans: List[Tuple[int, int]],
        explicit_specialization_keys: Set[Tuple[str, Tuple[str, ...]]],
        included_spans: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Tuple[str, List[str], Tuple[int, int]]]:
        calls: List[Tuple[str, List[str], Tuple[int, int]]] = []
        namespace_visibility = self._metal_namespace_visibility(code)
        concrete_structs = self._find_concrete_struct_definitions(code)
        template_owners: Dict[Tuple[int, int], _MetalStructDefinition] = {}
        for template in templates_by_name.values():
            owners = [
                struct
                for struct in concrete_structs
                if struct.body_span[0] <= template.span[0]
                and template.span[1] <= struct.body_span[1]
            ]
            if owners:
                template_owners[template.span] = min(
                    owners,
                    key=lambda struct: struct.body_span[1] - struct.body_span[0],
                )
        i = 0
        while i < len(code):
            if code[i] in "\"'":
                _literal, consumed = self._read_string(code, i)
                i += consumed
                continue
            if code.startswith("//", i):
                end = code.find("\n", i)
                if end == -1:
                    break
                i = end + 1
                continue
            if code.startswith("/*", i):
                end = code.find("*/", i + 2)
                if end == -1:
                    break
                i = end + 2
                continue
            span = self._containing_span(i, excluded_spans)
            if span is not None:
                i = span[1]
                continue
            if (
                included_spans is not None
                and self._containing_span(i, included_spans) is None
            ):
                i = self._next_scan_stop(code, i + 1, excluded_spans, included_spans)
                continue
            if code[i].isalpha() or code[i] == "_":
                ident, consumed = self._read_identifier(code, i)
                if ident == "operator":
                    i += consumed
                    continue
                if ident not in templates_by_name:
                    i += consumed
                    continue
                j = i + consumed
                while j < len(code) and code[j].isspace():
                    j += 1
                if j >= len(code) or code[j] != "<":
                    i += consumed
                    continue
                angle_end = self._find_matching_angle(code, j)
                if angle_end is None:
                    i += consumed
                    continue
                k = angle_end + 1
                while k < len(code) and code[k].isspace():
                    k += 1
                if k >= len(code) or code[k] != "(":
                    i += consumed
                    continue
                template_arguments = self._split_top_level_commas(
                    code[j + 1 : angle_end]
                )
                key = self._template_specialization_key(ident, template_arguments)
                if key in explicit_specialization_keys:
                    i = angle_end + 1
                    continue
                span_start = self._scoped_identifier_start(code, i)
                qualified_name = re.sub(
                    r"\s*::\s*",
                    "::",
                    code[span_start : i + consumed].strip(),
                )
                template = templates_by_name[ident]
                if not self._metal_template_function_call_visible(
                    template,
                    qualified_name,
                    span_start,
                    namespace_visibility,
                    template_owners.get(template.span),
                ):
                    i = angle_end + 1
                    continue
                calls.append((ident, template_arguments, (span_start, angle_end + 1)))
                i = angle_end + 1
                continue
            i = self._next_scan_stop(code, i + 1, excluded_spans, included_spans)
        if not calls:
            return calls

        local_binding_owner_spans = list(included_spans or [])
        if not local_binding_owner_spans:
            local_binding_owner_spans = [
                function.body_span
                for function in self._find_non_template_function_definitions(
                    code,
                    excluded_spans,
                )
            ]
        for struct in concrete_structs:
            local_binding_owner_spans.extend(
                method.span for method in [*struct.methods, *struct.template_methods]
            )
            local_binding_owner_spans.extend(
                constructor.span
                for constructor in struct.constructors
                if constructor.span is not None
            )

        local_type_aliases = self._collect_local_type_alias_bindings(
            code,
            local_binding_owner_spans,
        )
        source_type_aliases = self._collect_local_type_alias_bindings(
            code,
            [(0, len(code))],
        )
        explicit_free_specializations = (
            self._find_explicit_template_function_specializations(code)
        )
        constexpr_functions = self._materialization_constexpr_function_index(code)
        local_integral_constants = self._collect_local_integral_constant_bindings(
            code,
            local_binding_owner_spans,
            local_type_aliases,
            constexpr_functions,
            include_file_scope=True,
            # `constexpr short TM = Tile::kRows;` is a constant expression too.
            static_member_constants=static_member_constant_substituter(
                self, concrete_structs
            ),
        )
        resolved_calls: List[Tuple[str, List[str], Tuple[int, int]]] = []
        for function_name, template_arguments, span in calls:
            visible_local_constants = self._local_integral_constant_bindings_at(
                local_integral_constants,
                span[0],
            )
            resolved_arguments = self._resolve_template_argument_types_at(
                template_arguments,
                local_type_aliases,
                span[0],
                self._visible_local_declared_types(visible_local_constants),
            )
            local_constants = {
                name: binding.value
                for name, binding in visible_local_constants.items()
                if binding.value is not None
            }
            if local_constants:
                resolved_arguments = [
                    self._substitute_template_argument_static_constants(
                        argument,
                        local_constants,
                    )
                    for argument in resolved_arguments
                ]
            resolved_arguments = self._select_conditional_template_arguments(
                resolved_arguments
            )
            unresolved_constexpr_calls: Set[str] = set()
            constexpr_resolved_arguments: List[str] = []
            for argument in resolved_arguments:
                resolved_argument, _evaluation, unresolved_calls = (
                    self._resolve_static_constexpr_calls(
                        argument,
                        constexpr_functions,
                        span[0],
                    )
                )
                constexpr_resolved_arguments.append(
                    self._normalize_template_argument_text(resolved_argument)
                )
                unresolved_constexpr_calls.update(unresolved_calls)
            if unresolved_constexpr_calls:
                # A nested helper is only a concrete non-type argument after the
                # constexpr evaluator has proved one unique lexical candidate and
                # value.  Leave ambiguous, runtime-dependent, or cyclic calls in
                # source so the residual-template diagnostic remains fail-closed.
                continue
            resolved_arguments = constexpr_resolved_arguments
            unresolved_local_constants = unresolved_constant_references(
                resolved_arguments,
                (
                    name
                    for name, binding in visible_local_constants.items()
                    if binding.value is None and not binding.type_alias
                ),
            )
            if unresolved_local_constants:
                requested_signature = self._template_specialization_signature(
                    function_name,
                    resolved_arguments,
                )
                constant_names = ", ".join(
                    f"'{name}'" for name in unresolved_local_constants
                )
                suggested_action = (
                    "make each function-local template argument a constexpr "
                    "integral expression composed from concrete values"
                )
                raise MetalTemplateSpecializationError(
                    "Metal function template materialization left function-local "
                    f"constant {constant_names} unresolved in "
                    f"'{requested_signature}'. Suggested action: "
                    f"{suggested_action}.",
                    requested_signature=requested_signature,
                    suggested_action=suggested_action,
                    source_location=self._source_location_for_offsets(
                        code,
                        span[0],
                        span[1],
                    ),
                    unresolved_local_constants=tuple(unresolved_local_constants),
                    callee_template=function_name,
                    requested_arguments=tuple(resolved_arguments),
                )
            template = templates_by_name[function_name]
            canonical_call_arguments = (
                self._canonicalize_template_function_arguments_at(
                    template,
                    resolved_arguments,
                    source_type_aliases,
                    span[0],
                )
            )
            if canonical_call_arguments is None:
                requested_signature = self._template_specialization_signature(
                    function_name,
                    resolved_arguments,
                )
                suggested_action = (
                    "declare every call-site type alias before use and remove "
                    "cyclic alias declarations"
                )
                raise MetalTemplateSpecializationError(
                    "Metal function template materialization cannot determine "
                    f"whether '{requested_signature}' has an explicit "
                    "specialization because a call-site type alias is unresolved "
                    "at its lexical position. Suggested action: "
                    f"{suggested_action}.",
                    requested_signature=requested_signature,
                    suggested_action=suggested_action,
                    source_location=self._source_location_for_offsets(
                        code,
                        span[0],
                        span[1],
                    ),
                    callee_template=function_name,
                    requested_arguments=tuple(resolved_arguments),
                )

            alias_equivalent_specialization = False
            for specialization in explicit_free_specializations.values():
                if specialization["name"] != function_name:
                    continue
                if not specialization["templateArgumentsExplicit"]:
                    # `template <> R f(P)` and `template <> R f<>(P)` deduce
                    # their template arguments from the concrete function type.
                    # Only the project-level exact-signature selector has enough
                    # type information to decide which overload owns the body.
                    alias_equivalent_specialization = True
                    break
                specialization_position = int(specialization["span"][0])
                canonical_specialization_arguments = (
                    self._canonicalize_template_function_arguments_at(
                        template,
                        specialization["arguments"],
                        source_type_aliases,
                        specialization_position,
                    )
                )
                if canonical_specialization_arguments is None:
                    requested_signature = self._template_specialization_signature(
                        function_name,
                        resolved_arguments,
                    )
                    suggested_action = (
                        "declare every specialization type alias before use and "
                        "remove cyclic alias declarations"
                    )
                    raise MetalTemplateSpecializationError(
                        "Metal function template materialization cannot determine "
                        f"whether '{requested_signature}' has an explicit "
                        "specialization because a specialization type alias is "
                        "unresolved at its lexical position. Suggested action: "
                        f"{suggested_action}.",
                        requested_signature=requested_signature,
                        suggested_action=suggested_action,
                        source_location=specialization.get("sourceLocation"),
                        callee_template=function_name,
                        requested_arguments=tuple(resolved_arguments),
                    )
                explicit_count = min(
                    len(canonical_specialization_arguments),
                    len(canonical_call_arguments),
                )
                if (
                    canonical_specialization_arguments[:explicit_count]
                    == canonical_call_arguments[:explicit_count]
                ):
                    alias_equivalent_specialization = True
                    break
            if alias_equivalent_specialization:
                # Deduced/default arguments need the full-signature selector.
                # Another overload may own the explicit body, in which case that
                # selector safely materializes the chosen primary instead.
                continue

            resolved_key = self._template_specialization_key(
                function_name,
                resolved_arguments,
            )
            if resolved_key in explicit_specialization_keys:
                continue
            resolved_calls.append((function_name, resolved_arguments, span))
        return resolved_calls

    def _find_explicit_template_struct_instantiations(
        self,
        code: str,
        struct_templates_by_name: Dict[str, _MetalTemplateStruct],
        excluded_spans: List[Tuple[int, int]],
        explicit_specialization_keys: Set[Tuple[str, Tuple[str, ...]]],
        partial_specializations: Dict[
            str, List[Tuple[_MetalTemplateStruct, List[str]]]
        ],
        *,
        allow_partial_object_specializations: bool = False,
        allow_unqualified_remove_cv_t: bool = False,
    ) -> Tuple[
        List[Tuple[str, List[str], Tuple[int, int]]],
        Dict[
            Tuple[str, Tuple[str, ...]],
            Tuple[_MetalTemplateStruct, Dict[str, str]],
        ],
    ]:
        # Struct counterpart of _find_explicit_template_function_calls: locate
        # concrete `StructName<args>` TYPE references (variable declarations, base
        # classes, casts, nested template arguments, ...). Unlike a function call
        # the reference need not be followed by "(", so that trailing guard is
        # dropped. References inside template declarations (excluded_spans) are
        # skipped: they are only materialized once their enclosing template is.
        concrete_structs = self._find_concrete_struct_definitions(code)
        namespace_spans = self._find_namespace_spans(code)
        concrete_structs_by_name = {struct.name: struct for struct in concrete_structs}
        owner_contexts: List[
            Tuple[
                _MetalStructDefinition,
                List[Tuple[int, int]],
                Dict[str, str],
            ]
        ] = []
        for struct in concrete_structs:
            constants = self._resolved_static_data_member_initializers(struct)
            if not constants and not struct.type_aliases:
                continue
            method_spans = [
                method.span for method in [*struct.methods, *struct.template_methods]
            ]
            method_spans.extend(
                constructor.span
                for constructor in struct.constructors
                if constructor.span is not None
            )
            owner_contexts.append((struct, method_spans, constants))

        local_binding_owner_spans = [
            function.body_span
            for function in self._find_non_template_function_definitions(
                code, excluded_spans
            )
        ]
        for struct in concrete_structs:
            local_binding_owner_spans.extend(
                method.span for method in [*struct.methods, *struct.template_methods]
            )
            local_binding_owner_spans.extend(
                constructor.span
                for constructor in struct.constructors
                if constructor.span is not None
            )
        local_type_aliases = self._collect_local_type_alias_bindings(
            code,
            local_binding_owner_spans,
        )
        partial_match_type_aliases = self._collect_local_type_alias_bindings(
            code,
            [(0, len(code))],
            skip_spans=excluded_spans,
        )
        constexpr_functions = self._materialization_constexpr_function_index(code)
        local_integral_constants = self._collect_local_integral_constant_bindings(
            code,
            local_binding_owner_spans,
            local_type_aliases,
            constexpr_functions,
            include_file_scope=True,
        )

        instantiations: List[Tuple[str, List[str], Tuple[int, int]]] = []
        selected_partial_specializations: Dict[
            Tuple[str, Tuple[str, ...]],
            Tuple[_MetalTemplateStruct, Dict[str, str]],
        ] = {}
        discovered_materialized_structs: Dict[str, Dict[Tuple[str, ...], str]] = {}
        i = 0
        while i < len(code):
            if code[i] in "\"'":
                _literal, consumed = self._read_string(code, i)
                i += consumed
                continue
            if code.startswith("//", i):
                end = code.find("\n", i)
                if end == -1:
                    break
                i = end + 1
                continue
            if code.startswith("/*", i):
                end = code.find("*/", i + 2)
                if end == -1:
                    break
                i = end + 2
                continue
            span = self._containing_span(i, excluded_spans)
            if span is not None:
                i = span[1]
                continue
            if code[i].isalpha() or code[i] == "_":
                ident, consumed = self._read_identifier(code, i)
                if ident not in struct_templates_by_name:
                    i += consumed
                    continue
                j = i + consumed
                while j < len(code) and code[j].isspace():
                    j += 1
                if j >= len(code) or code[j] != "<":
                    i += consumed
                    continue
                angle_end = self._find_matching_angle(code, j)
                if angle_end is None:
                    i += consumed
                    continue
                primary_template = struct_templates_by_name[ident]
                reference_start = self._scoped_identifier_start(code, i)
                reference_name = re.sub(
                    r"\s*::\s*",
                    "::",
                    code[reference_start : i + consumed].strip(),
                )
                if "::" in reference_name:
                    globally_qualified = reference_name.startswith("::")
                    normalized_reference = reference_name.lstrip(":")
                    expected_name = (
                        f"{primary_template.namespace}::{ident}"
                        if primary_template.namespace
                        else ident
                    )
                    visible_names = self._namespace_lookup_names(
                        normalized_reference,
                        self._namespace_at(namespace_spans, i),
                        globally_qualified,
                    )
                    if expected_name not in visible_names:
                        i = angle_end + 1
                        continue
                template_arguments = self._split_top_level_commas(
                    code[j + 1 : angle_end]
                )
                if self._template_arguments_contain_qualified_struct_alias(
                    template_arguments,
                    set(struct_templates_by_name),
                ):
                    i = j + 1
                    continue
                visible_local_constants = self._local_integral_constant_bindings_at(
                    local_integral_constants,
                    i,
                )
                template_arguments = self._resolve_template_argument_types_at(
                    template_arguments,
                    local_type_aliases,
                    i,
                    self._visible_local_declared_types(visible_local_constants),
                )
                local_constants = {
                    name: binding.value
                    for name, binding in visible_local_constants.items()
                    if binding.value is not None
                }
                if local_constants:
                    template_arguments = [
                        self._substitute_template_argument_static_constants(
                            argument,
                            local_constants,
                        )
                        for argument in template_arguments
                    ]
                template_arguments = self._select_conditional_template_arguments(
                    template_arguments
                )
                unresolved_local_constants = unresolved_constant_references(
                    template_arguments,
                    (
                        name
                        for name, binding in visible_local_constants.items()
                        if binding.value is None and not binding.type_alias
                    ),
                )
                if unresolved_local_constants:
                    requested_signature = self._template_specialization_signature(
                        ident, template_arguments
                    )
                    constant_names = ", ".join(
                        f"'{name}'" for name in unresolved_local_constants
                    )
                    suggested_action = (
                        "make each function-local template argument a constexpr "
                        "integral expression composed from concrete values"
                    )
                    raise MetalTemplateSpecializationError(
                        "Metal nested struct template materialization left "
                        f"function-local constant {constant_names} unresolved in "
                        f"'{requested_signature}'. Suggested action: "
                        f"{suggested_action}.",
                        requested_signature=requested_signature,
                        suggested_action=suggested_action,
                        source_location=self._source_location_for_offsets(
                            code,
                            self._scoped_identifier_start(code, i),
                            angle_end + 1,
                        ),
                        nested_struct_name=ident,
                        unresolved_local_constants=tuple(unresolved_local_constants),
                    )
                containing_contexts = [
                    context
                    for context in owner_contexts
                    if context[0].span[0] <= i < context[0].span[1]
                ]
                if containing_contexts and any(
                    self._containing_span(i, context[1]) is not None
                    for context in containing_contexts
                ):
                    for owner, _method_spans, context_constants in sorted(
                        containing_contexts,
                        key=lambda context: (context[0].span[1] - context[0].span[0]),
                    ):
                        shadowed: Set[str] = set()
                        member_template_parameters: Set[str] = set()
                        for method in [*owner.methods, *owner.template_methods]:
                            if method.span[0] <= i < method.span[1]:
                                shadowed.update(method.parameter_names)
                                shadowed.update(self._local_variable_names(method.body))
                                member_template_parameters.update(
                                    method.template_parameters
                                )
                        for constructor in owner.constructors:
                            if (
                                constructor.span is not None
                                and constructor.span[0] <= i < constructor.span[1]
                            ):
                                shadowed.update(constructor.param_names)
                                shadowed.update(
                                    self._local_variable_names(constructor.body)
                                )
                        # A member function body sees the class's constant
                        # static data members unless a parameter or local
                        # declaration hides them ([basic.scope.class]), so a
                        # local `Tile<float, TM>` names the owner's `TM`.
                        # Innermost owners are visited first and replace the
                        # names they declare before an enclosing owner can.
                        visible_constants = {
                            name: value
                            for name, value in context_constants.items()
                            if name not in shadowed
                            and name not in member_template_parameters
                        }
                        if visible_constants:
                            template_arguments = [
                                self._substitute_template_argument_static_constants(
                                    argument, visible_constants
                                )
                                for argument in template_arguments
                            ]
                        visible_aliases = set(owner.type_aliases) - shadowed
                        if not visible_aliases:
                            continue
                        template_arguments = [
                            (
                                self._canonicalize_struct_scoped_type(
                                    argument,
                                    owner,
                                    concrete_structs_by_name,
                                    excluded_aliases=shadowed,
                                )
                                if set(IDENTIFIER_RE.findall(argument))
                                & visible_aliases
                                else argument
                            )
                            for argument in template_arguments
                        ]
                if containing_contexts and not any(
                    self._containing_span(i, context[1]) is not None
                    for context in containing_contexts
                ):
                    resolved_owner_alias = False
                    constants: Dict[str, str] = {}
                    for _owner, _method_spans, context_constants in sorted(
                        containing_contexts,
                        key=lambda context: (context[0].span[1] - context[0].span[0]),
                        reverse=True,
                    ):
                        constants.update(context_constants)
                    template_arguments = [
                        self._substitute_template_argument_static_constants(
                            argument, constants
                        )
                        for argument in template_arguments
                    ]
                    owner_aliases: Set[str] = set()
                    # Resolve innermost aliases first so normal owner shadowing is
                    # preserved while a lifted specialization becomes self-contained.
                    for owner, _method_spans, _context_constants in sorted(
                        containing_contexts,
                        key=lambda context: (context[0].span[1] - context[0].span[0]),
                    ):
                        if not owner.type_aliases:
                            continue
                        aliases = set(owner.type_aliases)
                        referenced_aliases = aliases & {
                            identifier
                            for argument in template_arguments
                            for identifier in IDENTIFIER_RE.findall(argument)
                        }
                        if not referenced_aliases:
                            continue
                        owner_aliases.update(referenced_aliases)
                        resolved_arguments: List[str] = []
                        for argument in template_arguments:
                            if (
                                set(IDENTIFIER_RE.findall(argument))
                                & referenced_aliases
                            ):
                                resolved_argument = (
                                    self._canonicalize_struct_scoped_type(
                                        argument,
                                        owner,
                                        concrete_structs_by_name,
                                    )
                                )
                                resolved_owner_alias = resolved_owner_alias or (
                                    resolved_argument != argument
                                )
                                resolved_arguments.append(resolved_argument)
                            else:
                                resolved_arguments.append(argument)
                        template_arguments = resolved_arguments
                        owner_aliases.update(
                            aliases
                            & {
                                identifier
                                for argument in template_arguments
                                for identifier in IDENTIFIER_RE.findall(argument)
                            }
                        )
                    template_arguments = [
                        self._substitute_template_argument_static_constants(
                            argument, constants
                        )
                        for argument in template_arguments
                    ]
                    unresolved_aliases = sorted(
                        owner_aliases
                        & {
                            identifier
                            for argument in template_arguments
                            for identifier in IDENTIFIER_RE.findall(argument)
                        }
                    )
                    if unresolved_aliases:
                        alias_owner = next(
                            owner
                            for owner, _method_spans, _context_constants in sorted(
                                containing_contexts,
                                key=lambda context: (
                                    context[0].span[1] - context[0].span[0]
                                ),
                            )
                            if set(owner.type_aliases) & set(unresolved_aliases)
                        )
                        owner_specialization = (
                            self._materialized_struct_specializations.get(
                                alias_owner.name
                            )
                        )
                        owner_struct_name = (
                            owner_specialization[0]
                            if owner_specialization is not None
                            else alias_owner.name
                        )
                        requested_signature = self._template_specialization_signature(
                            ident, template_arguments
                        )
                        alias_names = ", ".join(
                            f"'{alias}'" for alias in unresolved_aliases
                        )
                        suggested_action = (
                            "make each owner-scoped template argument resolve to a "
                            "concrete type before instantiating the nested struct"
                        )
                        raise MetalTemplateSpecializationError(
                            "Metal nested struct template materialization left "
                            f"owner-scoped alias {alias_names} dependent in "
                            f"'{requested_signature}'. Suggested action: "
                            f"{suggested_action}.",
                            requested_signature=requested_signature,
                            suggested_action=suggested_action,
                            source_location=self._source_location_for_offsets(
                                code,
                                self._scoped_identifier_start(code, i),
                                angle_end + 1,
                            ),
                            owner_struct_name=owner_struct_name,
                            owner_aliases=tuple(sorted(owner_aliases)),
                            nested_struct_name=ident,
                        )
                    # Reuse a concrete identifier discovered earlier in this scan.
                    # This avoids injecting another template reference and forcing
                    # an additional whole-source materialization pass.
                    if resolved_owner_alias and discovered_materialized_structs:
                        template_arguments = [
                            self._replace_discovered_struct_template_references(
                                argument,
                                discovered_materialized_structs,
                            )
                            for argument in template_arguments
                        ]
                template_arguments = [
                    self._canonicalize_qualified_struct_type_aliases(
                        argument,
                        concrete_structs,
                    )
                    for argument in template_arguments
                ]
                template_arguments = [
                    self._resolve_concrete_qualified_template_struct_aliases(
                        argument,
                        struct_templates_by_name,
                        partial_specializations,
                        partial_match_type_aliases,
                        concrete_structs,
                        i,
                        source_code=code,
                        allow_unqualified_remove_cv_t=(allow_unqualified_remove_cv_t),
                    )
                    for argument in template_arguments
                ]
                if self._template_arguments_contain_qualified_struct_alias(
                    template_arguments,
                    set(struct_templates_by_name),
                ):
                    i = angle_end + 1
                    continue
                resolved_arguments = (
                    self._template_arguments_with_resolved_defaults(
                        primary_template,
                        template_arguments,
                    )
                    or template_arguments
                )
                partial_match_arguments = [
                    self._canonicalize_metal_standard_type_aliases(
                        self._canonicalize_qualified_struct_type_aliases(
                            self._resolve_type_aliases_at(
                                argument,
                                partial_match_type_aliases,
                                i,
                            ),
                            concrete_structs,
                        ),
                        allow_unqualified_remove_cv_t=(allow_unqualified_remove_cv_t),
                    )
                    for argument in resolved_arguments
                ]
                key = self._struct_specialization_comparison_key(
                    ident,
                    partial_match_arguments,
                )
                if key in explicit_specialization_keys:
                    sink = getattr(self, "_explicit_struct_reference_sink", None)
                    if sink is not None:
                        sink.append(
                            (
                                ident,
                                list(template_arguments),
                                key,
                                (self._scoped_identifier_start(code, i), angle_end + 1),
                            )
                        )
                    i = angle_end + 1
                    continue
                matching_partials: List[Tuple[_MetalTemplateStruct, Dict[str, str]]] = (
                    []
                )
                has_unmaterializable_partial = False
                for (
                    partial_template,
                    specialization_arguments,
                ) in partial_specializations.get(ident, []):
                    canonical_specialization_arguments = [
                        self._canonical_partial_struct_specialization_pattern(
                            partial_template,
                            argument,
                            partial_match_type_aliases,
                            allow_unqualified_remove_cv_t=(
                                allow_unqualified_remove_cv_t
                            ),
                        )
                        for argument in specialization_arguments
                    ]
                    if not self._template_arguments_may_match_partial_struct_specialization(
                        partial_template,
                        canonical_specialization_arguments,
                        partial_match_arguments,
                        code=code,
                        position=i,
                    ):
                        continue
                    bindings = self._partial_struct_specialization_bindings(
                        partial_template,
                        canonical_specialization_arguments,
                        partial_match_arguments,
                        code=code,
                        position=i,
                    )
                    if bindings is None:
                        has_unmaterializable_partial = True
                    else:
                        matching_partials.append((partial_template, bindings))
                self._require_unique_constrained_struct_partial(
                    ident, resolved_arguments, matching_partials
                )
                if matching_partials or has_unmaterializable_partial:
                    resolved_constraint = len(matching_partials) == 1 and any(
                        re.search(r"\benable_if(?:_t)?\s*<", argument)
                        for argument in (
                            self._template_struct_specialization_arguments(
                                matching_partials[0][0]
                            )
                            or []
                        )
                    )
                    qualified_start = angle_end + 1
                    while (
                        qualified_start < len(code) and code[qualified_start].isspace()
                    ):
                        qualified_start += 1
                    qualified_member_owner = code.startswith("::", qualified_start)
                    if (
                        (
                            not qualified_member_owner
                            and not allow_partial_object_specializations
                            and not resolved_constraint
                        )
                        or has_unmaterializable_partial
                        or len(matching_partials) != 1
                    ):
                        i = angle_end + 1
                        continue
                    template_arguments = resolved_arguments
                    selection_key = self._template_specialization_key(
                        ident,
                        template_arguments,
                    )
                    selected_partial_specializations[selection_key] = matching_partials[
                        0
                    ]
                written_arguments = template_arguments
                if not (matching_partials or has_unmaterializable_partial):
                    template_arguments = canonical_struct_template_arguments(
                        self, primary_template, template_arguments
                    )
                span_start = self._scoped_identifier_start(code, i)
                instantiations.append(
                    (ident, template_arguments, (span_start, angle_end + 1))
                )
                if self._template_arguments_satisfy_parameters(
                    primary_template, template_arguments
                ):
                    specialization_key = self._template_specialization_key(
                        ident, template_arguments
                    )
                    specialized_identifier = self._template_specialization_identifier(
                        ident, list(specialization_key[1])
                    )
                    discovered = discovered_materialized_structs.setdefault(ident, {})
                    discovered[specialization_key[1]] = specialized_identifier
                    discovered[
                        self._template_specialization_key(ident, written_arguments)[1]
                    ] = specialized_identifier
                i = angle_end + 1
                continue
            i = self._next_scan_stop(code, i + 1, excluded_spans)
        return instantiations, selected_partial_specializations

    def _reachable_function_spans(
        self,
        code: str,
        excluded_spans: List[Tuple[int, int]],
    ) -> Optional[List[Tuple[int, int]]]:
        functions = self._find_non_template_function_definitions(code, excluded_spans)
        if not functions:
            return None

        roots = [function for function in functions if function.is_entry]
        if not roots:
            return None

        by_name: Dict[str, List[_MetalFunctionDefinition]] = {}
        qualified_functions = [
            function
            for function in functions
            if self._qualified_function_owner(
                code[function.span[0] : function.body_span[0] - 1]
            )
            is not None
        ]
        structs = (
            self._find_concrete_struct_definitions(code) if qualified_functions else []
        )
        struct_names = {struct.name for struct in structs}
        by_owner: Dict[str, List[_MetalOutOfLineMemberDefinition]] = {}
        for function in functions:
            by_name.setdefault(function.name, []).append(function)
        for function in qualified_functions:
            member = self._out_of_line_member_definition(
                code,
                function,
                struct_names,
            )
            if member is not None:
                by_owner.setdefault(member.owner_name, []).append(member)

        known_names = set(by_name)
        known_owners = set(by_owner)
        member_type_context: Optional[_MetalReachabilityTypeContext] = None
        reachable: Set[Tuple[int, int]] = set()
        pending = list(roots)
        while pending:
            function = pending.pop()
            if function.span in reachable:
                continue
            reachable.add(function.span)
            body_start, body_end = function.body_span
            for referenced in self._find_function_references(
                code[body_start:body_end],
                known_names,
                known_owners,
            ):
                pending.extend(by_name.get(referenced, ()))
            function_source = code[function.span[0] : function.span[1]]
            for owner in self._find_identifier_references(
                function_source,
                known_owners,
            ):
                if member_type_context is None:
                    member_type_context = self._reachability_type_context(
                        code,
                        excluded_spans,
                        structs,
                    )
                candidates = by_owner.get(owner, ())
                selected = self._reachable_out_of_line_member_functions(
                    code,
                    function,
                    owner,
                    candidates,
                    member_type_context,
                )
                pending.extend(candidate.function for candidate in selected)

        return [function.span for function in functions if function.span in reachable]

    def _expand_macros(
        self,
        text: str,
        line_num: int,
        in_expression: bool,
        file_path: Optional[str] = None,
        disabled_macros: Optional[Set[str]] = None,
    ) -> str:
        if in_expression:
            text = self._expand_clang_feature_test_macros(text, file_path)
        if not in_expression:
            incomplete = self._find_incomplete_function_macro_call(text)
            if incomplete is not None:
                macro_name, start = incomplete
                self._raise_incomplete_function_macro_call(
                    text,
                    macro_name,
                    start,
                    line_num,
                    file_path,
                )
        return super()._expand_macros(
            text, line_num, in_expression, file_path, disabled_macros
        )

    def _handle_include(self, rest: str, file_path: Optional[str]) -> Optional[str]:
        included = super()._handle_include(rest, file_path)
        if included is not None or self.strict:
            return included
        include_target = rest.strip()
        if include_target.startswith("<") and include_target.endswith(">"):
            return f"{PRESERVED_INCLUDE_SENTINEL}{include_target}"
        return None


_METAL_ENTRY_STAGE_PATTERNS = (
    (re.compile(r"\[\[\s*kernel\s*\]\]|\b(?:kernel|compute)\b"), "compute"),
    (re.compile(r"\bvertex\b"), "vertex"),
    (re.compile(r"\bfragment\b"), "fragment"),
    (re.compile(r"\bmesh\b"), "mesh"),
    (re.compile(r"\bobject\b"), "object"),
    (re.compile(r"\bamplification\b"), "amplification"),
    (re.compile(r"\bintersection\b"), "ray_intersection"),
    (re.compile(r"\banyhit\b"), "ray_any_hit"),
    (re.compile(r"\bclosesthit\b"), "ray_closest_hit"),
    (re.compile(r"\bmiss\b"), "ray_miss"),
    (re.compile(r"\bcallable\b"), "ray_callable"),
)


def _metal_entry_stage(
    preprocessor: MetalPreprocessor, declaration: str
) -> Optional[str]:
    masked = preprocessor._mask_comments_and_literals(declaration)
    for pattern, stage in _METAL_ENTRY_STAGE_PATTERNS:
        if pattern.search(masked):
            return stage
    return None


def _source_entry_location(source: str, start: int, end: int):
    from crosstl.translator.entry_discovery import SourceEntryLocation, source_position

    line, column = source_position(source, start)
    return SourceEntryLocation(
        line=line,
        column=column,
        offset=start,
        length=max(0, end - start),
    )


def _static_metal_host_name(preprocessor: MetalPreprocessor, expression: str) -> str:
    if (
        re.fullmatch(
            rf"\s*{METAL_STRING_EXPRESSION_PATTERN}\s*",
            expression,
            re.DOTALL,
        )
        is None
    ):
        return ""
    return preprocessor._evaluate_metal_string_expression(expression)


def _unresolved_metal_host_name_diagnostics(
    preprocessor: MetalPreprocessor,
    source: str,
    excluded_spans: Sequence[Tuple[int, int]],
):
    from crosstl.translator.entry_discovery import SourceEntryDiscoveryDiagnostic

    diagnostics = []
    for match in re.finditer(r"\bhost_name\s*\(", source):
        if preprocessor._containing_span(match.start(), excluded_spans) is not None:
            continue
        open_paren = source.find("(", match.start(), match.end())
        close_paren = preprocessor._find_matching_delimiter(
            source, open_paren, "(", ")"
        )
        if close_paren is None:
            continue
        expression = source[open_paren + 1 : close_paren]
        if _static_metal_host_name(preprocessor, expression):
            continue
        diagnostics.append(
            SourceEntryDiscoveryDiagnostic(
                severity="warning",
                code="source.entry-discovery.unresolved-host-name",
                message=(
                    "Metal host_name expression cannot be resolved to a static "
                    "entry-point name."
                ),
                location=_source_entry_location(source, match.start(), close_paren + 1),
                missing_capabilities=("source.entry-point-name-resolution",),
                details={"expression": expression.strip()},
            )
        )

    for match in MLX_INSTANTIATE_KERNEL_RE.finditer(source):
        if preprocessor._containing_span(match.start(), excluded_spans) is not None:
            continue
        open_paren = source.find("(", match.start(), match.end())
        arguments, consumed = preprocessor._parse_macro_args(source, open_paren)
        if not consumed or not arguments:
            continue
        if _static_metal_host_name(preprocessor, arguments[0]):
            continue
        diagnostics.append(
            SourceEntryDiscoveryDiagnostic(
                severity="warning",
                code="source.entry-discovery.unresolved-host-name",
                message=(
                    "Metal instantiate_kernel name cannot be resolved to a static "
                    "entry-point name."
                ),
                location=_source_entry_location(
                    source, match.start(), open_paren + consumed
                ),
                missing_capabilities=("source.entry-point-name-resolution",),
                details={"expression": arguments[0].strip()},
            )
        )
    return diagnostics


def discover_metal_entry_points(
    code: str,
    file_path: Optional[str] = None,
    *,
    include_paths: Optional[Sequence[str]] = None,
    defines: Optional[Mapping[str, str]] = None,
    source_options: Optional[Mapping[str, Any]] = None,
):
    """Discover Metal entry points without materializing or translating kernels."""
    options = dict(source_options or {})
    preprocessor = MetalPreprocessor(
        include_paths=list(include_paths or ()),
        defines=dict(defines or {}),
        strict=bool(options.get("strict_preprocessor", False)),
    )
    source = code.lstrip("\ufeff")
    if options.get("preprocess", True):
        source = preprocessor.preprocess_for_entry_discovery(
            source, file_path=file_path
        )

    # Always expand current inputs before lookup; cached records own no file state.
    return replace(
        deepcopy(_discover_preprocessed_metal_entries(source)),
        source_path=file_path or "<memory>",
    )


@lru_cache(maxsize=ENTRY_DISCOVERY_CACHE_LIMIT)
def _discover_preprocessed_metal_entries(source: str):
    from crosstl.translator.entry_discovery import (
        ENTRY_DISCOVERY_AVAILABLE,
        SourceEntryDiscovery,
        SourceEntryDiscoveryDiagnostic,
        SourceEntryPoint,
        SourceEntryProvenance,
    )

    preprocessor = MetalPreprocessor()
    comment_and_literal_spans = preprocessor._find_comment_and_literal_spans(source)
    template_functions = preprocessor._find_template_functions(source)
    template_entries = {
        template.name: stage
        for template in template_functions
        if (stage := _metal_entry_stage(preprocessor, template.source)) is not None
    }

    candidates = []
    for instantiation in preprocessor._find_project_template_instantiations(source):
        if (
            preprocessor._containing_span(
                instantiation.span[0], comment_and_literal_spans
            )
            is not None
        ):
            continue
        declaration = source[instantiation.span[0] : instantiation.span[1]]
        if "host_name" in preprocessor._mask_comments_and_literals(
            declaration
        ) and not preprocessor._host_name_from_attributes(declaration):
            continue
        instantiate_match = MLX_INSTANTIATE_KERNEL_RE.search(declaration)
        if instantiate_match is not None:
            open_paren = declaration.find(
                "(", instantiate_match.start(), instantiate_match.end()
            )
            arguments, consumed = preprocessor._parse_macro_args(
                declaration, open_paren
            )
            if (
                not consumed
                or not arguments
                or not _static_metal_host_name(preprocessor, arguments[0])
            ):
                continue
        stage = _metal_entry_stage(preprocessor, declaration) or template_entries.get(
            instantiation.function_name
        )
        if stage is None:
            continue
        candidates.append(
            (
                instantiation.span[0],
                SourceEntryPoint(
                    name=instantiation.host_name,
                    stage=stage,
                    location=_source_entry_location(source, *instantiation.span),
                    provenance=SourceEntryProvenance(
                        kind="host-named-materialization",
                        declared_name=instantiation.function_name,
                        template_arguments=tuple(instantiation.template_arguments),
                    ),
                ),
            )
        )

    excluded_spans = list(preprocessor._find_template_declaration_spans(source))
    excluded_spans.extend(
        definition.span
        for definition in preprocessor._find_concrete_struct_definitions(source)
    )
    excluded_spans.extend(comment_and_literal_spans)
    for function in preprocessor._find_non_template_function_definitions(
        source, sorted(excluded_spans)
    ):
        declaration_end = max(function.span[0], function.body_span[0] - 1)
        declaration = source[function.span[0] : declaration_end]
        stage = _metal_entry_stage(preprocessor, declaration)
        if stage is None:
            continue
        host_name = preprocessor._host_name_from_attributes(declaration)
        if "host_name" in preprocessor._mask_comments_and_literals(declaration):
            if not host_name:
                continue
        candidates.append(
            (
                function.span[0],
                SourceEntryPoint(
                    name=host_name or function.name,
                    stage=stage,
                    location=_source_entry_location(source, *function.span),
                    provenance=SourceEntryProvenance(
                        kind="concrete",
                        declared_name=function.name,
                    ),
                ),
            )
        )

    entries = []
    diagnostics = list(
        _unresolved_metal_host_name_diagnostics(
            preprocessor, source, comment_and_literal_spans
        )
    )
    entries_by_name = {}
    for _position, entry in sorted(
        candidates, key=lambda item: (item[0], item[1].name)
    ):
        existing = entries_by_name.get(entry.name)
        if existing is None:
            entries_by_name[entry.name] = entry
            entries.append(entry)
            continue
        if existing.stage == entry.stage and existing.provenance == entry.provenance:
            continue
        diagnostics.append(
            SourceEntryDiscoveryDiagnostic(
                severity="warning",
                code="source.entry-discovery.duplicate-exported-name",
                message=(
                    f"Metal entry-point name '{entry.name}' is declared more than "
                    "once; the first declaration is retained."
                ),
                location=entry.location,
                missing_capabilities=("source.entry-point-identity",),
                details={
                    "entryPoint": entry.name,
                    "firstOffset": existing.location.offset,
                    "duplicateOffset": entry.location.offset,
                },
            )
        )

    return SourceEntryDiscovery(
        source_backend="metal",
        source_path="<memory>",
        status=ENTRY_DISCOVERY_AVAILABLE,
        entries=tuple(entries),
        diagnostics=tuple(diagnostics),
    )
