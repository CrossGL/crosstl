"""Preprocessor support for Metal source imports."""

import ast
import operator
import os
import re
from bisect import bisect_right
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Set, Tuple, TypeVar, Union

from crosstl.backend.DirectX.preprocessor import HLSLPreprocessor, Macro

from .type_layout import metal_type_layout, metal_type_size

PRESERVED_INCLUDE_SENTINEL = "__CROSSGL_METAL_PRESERVED_INCLUDE__ "
CLANG_FEATURE_TEST_MACROS = {
    "__has_attribute",
    "__has_builtin",
    "__has_extension",
    "__has_feature",
    "__has_include",
    "__has_include_next",
}
COMPILER_DIAGNOSTIC_START_RE = re.compile(
    r'^\s*(?:<[^>\n]+>[^:\n]*|"[^"\n]+"|[^:\n]+):\d+:\d+:?\s+' r"(?:warning|note):"
)
MSL_SOURCE_START_RE = re.compile(
    r"^\s*(?:"
    r"#include\b|#pragma\b|using\b|namespace\b|template\b|"
    r"struct\b|class\b|enum\b|typedef\b|constant\b|"
    r"\[\[|kernel\b|vertex\b|fragment\b|"
    r"(?:inline|static|constexpr|const|device|thread|threadgroup|void|"
    r"float|half|double|int|uint|long|ulong|short|ushort|char|uchar|bool)\b"
    r")"
)
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
METAL_STRING_LITERAL_PATTERN = r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'"
METAL_STRING_EXPRESSION_PATTERN = (
    rf"(?:{METAL_STRING_LITERAL_PATTERN})"
    rf"(?:\s*(?:{METAL_STRING_LITERAL_PATTERN}))*"
)
METAL_ENTRY_FUNCTION_RE = re.compile(
    r"\b(?:kernel|vertex|fragment|compute|mesh|object|amplification|"
    r"intersection|anyhit|closesthit|miss|callable)\b|"
    r"\[\[\s*kernel\s*\]\]"
)
MLX_INSTANTIATE_KERNEL_RE = re.compile(r"\binstantiate_kernel\s*\(")
MLX_HOST_NAME_DECL_RE = re.compile(
    r"\btemplate\s+\[\[\s*host_name\s*\(\s*(?P<host>"
    + METAL_STRING_EXPRESSION_PATTERN
    + r")\s*\)\s*\]\]\s*"
    r"\[\[\s*kernel\s*\]\]\s*decltype\s*\(\s*(?P<function>"
    r"[A-Za-z_][A-Za-z0-9_:]*)\s*<",
    re.DOTALL,
)
DEFAULT_EXPLICIT_TEMPLATE_SPECIALIZATION_LIMIT = 512
SOURCE_ANALYSIS_CACHE_LIMIT = 8
CONTAINING_SPAN_CACHE_LIMIT = 64


class MetalStructMethodError(ValueError):
    """A struct member-function call that cannot be lowered safely.

    Raised when a called member method cannot be instantiated or when its object
    or return-value semantics cannot be represented without changing behavior.
    It deliberately propagates out of ``preprocess`` so project translation
    reports a structured failure instead of emitting a dangling call or a
    validating artifact with value-copy semantics.
    """

    project_diagnostic_code = "project.translate.metal-struct-method"
    missing_capabilities = ("struct.template-method",)

    def __init__(
        self,
        message: str,
        *,
        struct_name: Optional[str] = None,
        method_name: Optional[str] = None,
        requested_signature: Optional[str] = None,
        suggested_action: Optional[str] = None,
        source_location: Optional[object] = None,
        missing_capabilities: Optional[Tuple[str, ...]] = None,
        reason: Optional[str] = None,
        return_type: Optional[str] = None,
    ):
        super().__init__(message)
        self.struct_name = struct_name
        self.method_name = method_name
        self.requested_signature = requested_signature
        self.suggested_action = suggested_action
        self.source_location = source_location
        self.reason = reason
        self.return_type = return_type
        if missing_capabilities is not None:
            self.missing_capabilities = missing_capabilities


class MetalStatelessGlobalElisionError(ValueError):
    """A compile-time object that cannot be erased without changing behavior."""

    project_diagnostic_code = "project.translate.metal-stateless-global-unsafe"
    missing_capabilities = ("metal.stateless-global-elision",)

    def __init__(
        self,
        message: str,
        *,
        global_name: str,
        type_name: str,
        reason: str,
        source_location: object,
        method_name: Optional[str] = None,
        blocking_use: Optional[str] = None,
        effect: Optional[str] = None,
        suggested_action: Optional[str] = None,
    ):
        super().__init__(message)
        self.global_name = global_name
        self.type_name = type_name
        self.reason = reason
        self.source_location = source_location
        self.method_name = method_name
        self.blocking_use = blocking_use
        self.effect = effect
        self.suggested_action = suggested_action


class MetalMacroExpansionError(ValueError):
    """A function-like macro invocation that cannot be expanded safely."""

    project_diagnostic_code = "project.translate.metal-macro-expansion-invalid"
    missing_capabilities = ("metal.function-macro-expansion",)

    def __init__(
        self,
        message: str,
        *,
        macro_name: str,
        reason: str,
        source_location: object,
        suggested_action: Optional[str] = None,
    ):
        super().__init__(message)
        self.macro_name = macro_name
        self.reason = reason
        self.source_location = source_location
        self.suggested_action = suggested_action


class MetalTemplateSpecializationError(ValueError):
    project_diagnostic_code = "project.translate.metal-template-specialization"
    missing_capabilities = ("template.specialization",)

    def __init__(
        self,
        message: str,
        *,
        limit: Optional[int] = None,
        limit_source: Optional[str] = None,
        unique_specialization_count: Optional[int] = None,
        required_work_items: Optional[int] = None,
        requested_signature: Optional[str] = None,
        suggested_action: Optional[str] = None,
        source_location: Optional[object] = None,
        owner_struct_name: Optional[str] = None,
        owner_aliases: Optional[Tuple[str, ...]] = None,
        nested_struct_name: Optional[str] = None,
        unresolved_local_constants: Optional[Tuple[str, ...]] = None,
        caller_specialization: Optional[str] = None,
        callee_template: Optional[str] = None,
        requested_arguments: Optional[Tuple[str, ...]] = None,
    ):
        super().__init__(message)
        self.limit = limit
        self.limit_source = limit_source
        self.unique_specialization_count = unique_specialization_count
        self.required_work_items = required_work_items
        self.requested_signature = requested_signature
        self.suggested_action = suggested_action
        self.source_location = source_location
        self.owner_struct_name = owner_struct_name
        self.owner_aliases = owner_aliases
        self.nested_struct_name = nested_struct_name
        self.unresolved_local_constants = unresolved_local_constants
        self.caller_specialization = caller_specialization
        self.callee_template = callee_template
        self.requested_arguments = requested_arguments


class MetalStaticAssertionError(ValueError):
    """A source compile-time assertion that cannot be discharged safely."""

    project_diagnostic_code = "project.translate.metal-static-assertion"
    missing_capabilities = ("compile-time.static-assertion",)

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        expression: str,
        resolved_expression: str,
        assertion_message: Optional[str],
        unresolved_dependencies: Tuple[str, ...],
        source_location: object,
        suggested_action: str,
    ):
        super().__init__(message)
        self.reason = reason
        self.expression = expression
        self.resolved_expression = resolved_expression
        self.assertion_message = assertion_message
        self.unresolved_dependencies = unresolved_dependencies
        self.source_location = source_location
        self.suggested_action = suggested_action


@dataclass
class _MetalTemplateFunction:
    name: str
    template_parameters: List[str]
    span: Tuple[int, int]
    body_start: int
    source: str
    variadic_template_parameters: Set[str] = field(default_factory=set)
    template_parameter_defaults: Dict[str, str] = field(default_factory=dict)
    template_parameter_types: Dict[str, str] = field(default_factory=dict)
    template_type_traits: Dict[str, Dict[str, object]] = field(default_factory=dict)
    namespace: str = ""
    materializations: List[str] = field(default_factory=list)


@dataclass
class _MetalTemplateStruct:
    name: str
    template_parameters: List[str]
    span: Tuple[int, int]
    source: str
    variadic_template_parameters: Set[str] = field(default_factory=set)
    template_parameter_defaults: Dict[str, str] = field(default_factory=dict)
    template_type_traits: Dict[str, Dict[str, object]] = field(default_factory=dict)
    namespace: str = ""


@dataclass(frozen=True)
class _MLXKernelInstantiation:
    host_name: str
    function_name: str
    template_arguments: List[str]
    span: Tuple[int, int]


@dataclass(frozen=True)
class _MetalFunctionDefinition:
    name: str
    span: Tuple[int, int]
    body_span: Tuple[int, int]
    is_entry: bool


@dataclass(frozen=True)
class _MetalTypeAliasBinding:
    declaration_position: int
    scope_start: int
    scope_end: int
    target: str


@dataclass(frozen=True)
class _MetalIntegralConstantBinding:
    declaration_position: int
    scope_start: int
    scope_end: int
    value: Optional[str]


@dataclass
class _MetalSourceAnalysis:
    comment_and_literal_spans: Optional[List[Tuple[int, int]]] = None
    lexical_brace_scopes: Optional[List[Tuple[int, int]]] = None
    template_declaration_spans: Optional[List[Tuple[int, int]]] = None
    namespace_spans: Optional[List[Tuple[int, int, str]]] = None


@dataclass(frozen=True)
class _MetalStaticAssertion:
    span: Tuple[int, int]
    condition_span: Tuple[int, int]
    condition: str
    message: Optional[str]


@dataclass(frozen=True)
class _MetalSubscriptableType:
    """Element type plus a proven address-space-qualified pointer type."""

    element_type: str
    pointer_type: Optional[str] = None


_MetalBufferType = Union[str, _MetalSubscriptableType]
_MetalPositionedBufferTypes = Dict[str, List[Tuple[int, _MetalBufferType]]]
_MetalBufferTypeView = Dict[str, _MetalBufferType]
_DeclaredTypeT = TypeVar("_DeclaredTypeT")


@dataclass(frozen=True)
class _TemplateParameter:
    """One entry of a parsed `template <...>` parameter list.

    A parameter is either a BINDABLE type parameter (`typename T` / `class T`,
    optionally variadic / with a default) or an anonymous non-type SFINAE
    CONSTRAINT parameter such as ``metal::enable_if_t<sizeof(T) < 8, bool> =
    true``. Only bindable type parameters take part in template-argument binding;
    constraints are recorded verbatim so an overload set can be disambiguated by
    evaluating each constraint against the concrete type arguments.
    """

    name: Optional[str]
    is_type_parameter: bool
    is_variadic: bool = False
    default: Optional[str] = None
    constraint_text: Optional[str] = None
    declared_type: Optional[str] = None

    @property
    def is_constraint(self) -> bool:
        # A non-type parameter whose declarator is anonymous (no bindable name)
        # is a SFINAE enabler we keep only for overload selection.
        return not self.is_type_parameter and self.constraint_text is not None


@dataclass(frozen=True)
class _MetalTypeAliasTemplate:
    parameters: List[_TemplateParameter]
    target: str


@dataclass
class _MetalStructMethod:
    """A member function found inside a concrete struct/class definition."""

    name: str
    free_name: str
    is_static: bool
    is_operator_call: bool
    return_type: str
    parameters: str
    parameter_names: List[str]
    body: str
    span: Tuple[int, int]
    is_const: bool = False
    receiver_address_spaces: Tuple[str, ...] = ()
    # Template member methods carry their template parameter names; an empty list
    # marks an ordinary (non-template) member function. The raw `RetType` /
    # `parameters` text still contains the template parameter identifiers; they
    # are substituted with concrete types at instantiation time.
    template_parameters: List[str] = field(default_factory=list)
    template_parameter_types: Dict[str, str] = field(default_factory=dict)
    variadic_template_parameters: Set[str] = field(default_factory=set)
    template_parameter_defaults: Dict[str, str] = field(default_factory=dict)
    # SFINAE constraint texts harvested from anonymous non-type template
    # parameters (e.g. `metal::enable_if_t<sizeof(T) < 8, bool>`). They are NOT
    # bindable parameters; they are evaluated against the concrete type bindings
    # to pick the unique enabled overload when a method name is overloaded.
    template_constraints: List[str] = field(default_factory=list)
    # The return type BEFORE any return-type SFINAE wrapper is stripped (the
    # second SFINAE layer, e.g. `metal::enable_if_t<is_integral_v<T>, T>`). Kept
    # so the wrapped form can be recorded as an extra constraint while the
    # emitted free function uses the unwrapped value type.
    return_type_constraint: Optional[str] = None

    @property
    def is_template(self) -> bool:
        return bool(self.template_parameters)


@dataclass
class _MetalDataMember:
    """One data member of a struct, kept in DECLARATION order.

    Unlike the unordered ``data_member_names`` set and the normalized
    ``data_member_types`` map, this preserves the full declared type text
    (address space + cv + pointer, e.g. ``const device float2*``), the trailing
    array suffix (``[N]`` for array members), and any default initializer. It is
    what the pointer-member scalar-replacement path needs to explode a struct
    into individual per-member parameters and per-member locals.
    """

    name: str
    type_text: str
    default: Optional[str] = None
    array_suffix: str = ""

    @property
    def is_pointer(self) -> bool:
        return "*" in self.type_text

    @property
    def is_array(self) -> bool:
        return bool(self.array_suffix)


@dataclass
class _MetalConstructor:
    """A struct constructor captured for pointer-member scalar replacement.

    Records the constructor's parameter names (in order), the member
    initializer-list mapping (member name -> init expression, where the
    expression usually names a constructor parameter), and the constructor body
    (statements that compute members such as ``threads_per_tg = ...;``).

    The remaining fields carry everything the pointer-member promotion path
    needs to REWRITE the constructor in place (drop the pointer parameters and
    their initializer-list entries): the absolute ``span`` of the whole
    constructor in the source, the ``prefix`` text before the parameter list
    (leading macros + the constructor name), the raw parameter and
    initializer-list text, and the FULL declared type of each parameter (kept in
    ``param_names`` order) so a pointer parameter can be told apart from a scalar
    one.
    """

    param_names: List[str]
    init_map: Dict[str, str]
    body: str
    span: Optional[Tuple[int, int]] = None
    prefix: str = ""
    params_text: str = ""
    init_text: str = ""
    param_types: List[str] = field(default_factory=list)
    is_constexpr: bool = False
    receiver_address_spaces: Tuple[str, ...] = ()
    template_parameters: List[str] = field(default_factory=list)
    template_parameter_defaults: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _MetalTemporaryFunctorCall:
    """A temporary construction immediately followed by ``operator()``."""

    constructor_open: int
    constructor_close: int
    argument_open: int
    argument_close: int


@dataclass(frozen=True)
class _MetalCompileTimeGlobal:
    name: str
    type_name: str
    namespace: str
    arguments: Tuple[str, ...]
    span: Tuple[int, int]
    name_span: Tuple[int, int]
    receiver_address_space: str
    prior_declaration_spans: Tuple[Tuple[int, int], ...] = ()


@dataclass
class _MetalStructDefinition:
    """A concrete (non-template) struct/class with its members split out."""

    name: str
    span: Tuple[int, int]
    body_span: Tuple[int, int]
    data_member_names: Set[str]
    methods: List[_MetalStructMethod]
    has_operator_call: bool
    qualified_name: str = ""
    # Data-member name -> its declared element type (the value type a `self.x`
    # access yields, with array extents stripped). Populated best-effort for
    # members whose type is recognizable; missing entries are simply
    # un-inferable. Used to type a `obj.member` / `obj.member[i]` call argument.
    data_member_types: Dict[str, str] = field(default_factory=dict)
    # Template member methods are kept separate from `methods`: they have no
    # single concrete signature to emit up front, so they are instantiated on
    # demand from their call sites' inferred argument types.
    template_methods: List[_MetalStructMethod] = field(default_factory=list)
    # Data members in DECLARATION order with full type text / defaults, plus the
    # struct's constructors. Only needed by (and only populated for) the
    # pointer-member scalar-replacement path, but always captured so the info is
    # available without a second parse.
    data_members: List[_MetalDataMember] = field(default_factory=list)
    constructors: List[_MetalConstructor] = field(default_factory=list)
    base_clause: str = ""
    aggregate_kind: str = "struct"
    layout_supported: bool = True
    member_prototype_names: Set[str] = field(default_factory=set)
    # Struct-scoped `using` / `typedef` aliases. These remain in the data-only
    # struct, but lowered method bodies need the same lexical type context when
    # resolving qualified calls and concrete return/parameter types.
    type_aliases: Dict[str, str] = field(default_factory=dict)
    # Struct-scoped alias templates retain their bindable parameter contract so
    # a concrete owner can expand `Owner::alias<args>` before target codegen.
    type_alias_templates: Dict[str, _MetalTypeAliasTemplate] = field(
        default_factory=dict
    )


@dataclass
class _PointerPromotionPlan:
    """How to promote a struct's device/threadgroup pointer members out of the
    struct so the residual struct is a pure scalar aggregate (legal under
    SPIR-V logical addressing).

    A pointer member cannot live inside a Function-storage struct value, so the
    plain member-function lowering (``S self`` passed by value) squashes it to a
    scalar and every ``self.ptr[i]`` becomes an access chain into a scalar
    (invalid SPIR-V). Instead we drop pointer members from the struct, thread
    them through each lowered method as ordinary pointer PARAMETERS (which the
    SPIR-V backend already inlines, exactly like a free function that takes a
    ``device T*``), and pass each construction's pointer expression directly at
    the call site. The scalar members (including ones the constructor computes)
    stay in the struct and keep flowing through the value ``self``.
    """

    struct_name: str
    # Pointer members in declaration order (these leave the struct).
    pointer_members: List[_MetalDataMember]
    # For each pointer member (same order), the index of the constructor
    # parameter that initializes it, so a construction call's argument at that
    # position is the pointer expression to forward at call sites.
    pointer_ctor_arg_indices: List[int]
    constructor: _MetalConstructor

    @property
    def pointer_member_names(self) -> List[str]:
        return [member.name for member in self.pointer_members]

    @property
    def pointer_parameter_decls(self) -> List[str]:
        # ``<type> <name>`` for each promoted pointer member, in order.
        return [f"{member.type_text} {member.name}" for member in self.pointer_members]


class MetalPreprocessor(HLSLPreprocessor):
    """Small Metal preprocessor used before lexing imported source files."""

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
    ):
        super().__init__(
            include_paths=include_paths,
            defines=defines,
            strict=strict,
            max_expansion_depth=max_expansion_depth,
        )
        if isinstance(max_template_specializations, bool):
            raise ValueError(
                "Metal max_template_specializations must be a non-negative integer"
            )
        try:
            specialization_limit = operator.index(max_template_specializations)
        except TypeError as exc:
            raise ValueError(
                "Metal max_template_specializations must be a non-negative integer"
            ) from exc
        if specialization_limit < 0:
            raise ValueError(
                "Metal max_template_specializations must be a non-negative integer"
            )
        self.max_template_specializations = specialization_limit
        self.template_specialization_limit_source = (
            template_specialization_limit_source or "max_template_specializations"
        )
        self._materialized_struct_specializations: Dict[
            str, Tuple[str, Tuple[str, ...]]
        ] = {}
        self._known_member_function_return_types: Dict[str, str] = {}
        self._instantiated_template_member_calls: Dict[
            str,
            Tuple[
                _MetalStructDefinition,
                _MetalStructMethod,
                Dict[str, str],
                Optional[Dict[str, _MetalStructDefinition]],
            ],
        ] = {}
        self._source_type_alias_bindings: Dict[str, List[_MetalTypeAliasBinding]] = {}
        self._integral_constant_binary_operators: Set[str] = set()
        self._integral_constant_contract_verified = False
        self._integral_constant_conversion_contract_verified = False
        self._int_alias_contract_verified = False
        self._const_for_loop_contract_verified = False
        self._const_for_loop_expansion_work = 0
        self._fail_closed_static_assertions = False
        self._source_analysis_cache: Dict[str, _MetalSourceAnalysis] = {}
        # The materialization scan alternates between multiple span lists at
        # nearly every source position. Cache their sorted lookup indexes, but
        # retain only a fixed number of lists as generated source snapshots are
        # replaced during specialization.
        self._containing_span_cache: Dict[
            int,
            Tuple[
                List[Tuple[int, int]],
                int,
                Optional[Tuple[int, int]],
                Optional[Tuple[int, int]],
                Optional[List[int]],
            ],
        ] = {}
        self.macros.setdefault(
            "TARGET_OS_SIMULATOR",
            Macro(name="TARGET_OS_SIMULATOR", replacement="0"),
        )
        for name in CLANG_FEATURE_TEST_MACROS:
            self.macros.setdefault(name, Macro(name=name, replacement="0"))

    def preprocess(self, code: str, file_path: Optional[str] = None) -> str:
        self._source_analysis_cache.clear()
        # Span lists are per-source; drop any cache entries from a previous
        # source so retained references cannot pin freed lists across runs.
        self._containing_span_cache.clear()
        self._materialized_struct_specializations.clear()
        self._known_member_function_return_types.clear()
        self._instantiated_template_member_calls.clear()
        self._source_type_alias_bindings.clear()
        self._integral_constant_binary_operators.clear()
        self._integral_constant_contract_verified = False
        self._integral_constant_conversion_contract_verified = False
        self._int_alias_contract_verified = False
        self._const_for_loop_contract_verified = False
        self._const_for_loop_expansion_work = 0
        self._fail_closed_static_assertions = False
        code = self._strip_leading_compiler_diagnostics(code)
        processed = super().preprocess(code, file_path=file_path)
        self._configure_integral_constant_contracts(processed)
        processed = self._materialize_project_template_instantiations(processed)
        processed = self._materialize_explicit_template_struct_instantiations(processed)
        processed = self._elide_stateless_compile_time_globals(processed)
        processed = self._lower_struct_member_functions(processed)
        processed = self._materialize_explicit_template_function_calls(processed)
        return processed.replace(PRESERVED_INCLUDE_SENTINEL, "#include ")

    def _source_analysis(self, code: str) -> _MetalSourceAnalysis:
        cache = self._source_analysis_cache
        analysis = cache.pop(code, None)
        if analysis is None:
            analysis = _MetalSourceAnalysis()
        cache[code] = analysis
        if len(cache) > SOURCE_ANALYSIS_CACHE_LIMIT:
            del cache[next(iter(cache))]
        return analysis

    def _strip_leading_compiler_diagnostics(self, code: str) -> str:
        lines = code.splitlines(keepends=True)
        saw_diagnostic = False

        for index, line in enumerate(lines):
            if MSL_SOURCE_START_RE.match(line):
                if saw_diagnostic:
                    return "".join(lines[index:])
                return code

            stripped = line.strip()
            if not stripped:
                continue
            if COMPILER_DIAGNOSTIC_START_RE.match(line):
                saw_diagnostic = True
                continue
            if saw_diagnostic and (
                line.startswith((" ", "\t")) or stripped.startswith("^")
            ):
                continue
            return code

        return code

    def _materialize_project_template_instantiations(
        self,
        code: str,
        *,
        enforce_specialization_limit: bool = True,
        host_names: Optional[Set[str]] = None,
    ) -> str:
        if not enforce_specialization_limit:
            # The project pipeline disables this limit only for template-hostile
            # targets, where residual source assertions cannot be emitted.
            self._fail_closed_static_assertions = True
        all_instantiations = self._find_project_template_instantiations(code)
        if not all_instantiations:
            return code
        instantiations = (
            all_instantiations
            if host_names is None
            else [
                instantiation
                for instantiation in all_instantiations
                if instantiation.host_name in host_names
            ]
        )
        # Explicit project instantiations (`instantiate_kernel(...)` and the
        # equivalent `template [[host_name(...)]] ... decltype(func<args>)
        # func<args>;` declarations) are the source's own enumerated request for
        # concrete kernels; their number is bounded by the source text rather than
        # by combinatorial discovery. `enforce_specialization_limit` keeps the
        # historical safety bail for template-FRIENDLY consumers (e.g. the direct
        # MetalLexer/`preprocess` path targeting CGL, which can keep the residual
        # templates when a source declares an unusually large instantiation
        # family). Template-HOSTILE targets cannot emit residual templates, so the
        # project pipeline disables the bail (`enforce_specialization_limit=False`)
        # and materializes every explicit instantiation selected for the artifact;
        # otherwise bailing here leaves the kernels and every helper template
        # reachable only from them unmaterialized and mis-reported as "missing
        # template arguments".
        if (
            enforce_specialization_limit
            and len(instantiations) > self.max_template_specializations
        ):
            return code

        templates = self._find_template_functions(code)
        if not templates or not instantiations:
            return self._apply_text_replacements(
                code,
                [
                    (instantiation.span[0], instantiation.span[1], "")
                    for instantiation in all_instantiations
                ],
            )

        templates_by_name = {template.name: template for template in templates}
        replacements: List[Tuple[int, int, str]] = [
            (instantiation.span[0], instantiation.span[1], "")
            for instantiation in all_instantiations
        ]
        seen: Set[Tuple[str, Tuple[str, ...], str]] = set()

        for instantiation in instantiations:
            template = templates_by_name.get(instantiation.function_name)
            if template is None:
                continue
            template_arguments = self._template_arguments_with_defaults(
                code,
                template,
                instantiation.template_arguments,
            )
            key = (
                instantiation.function_name,
                tuple(
                    self._normalize_template_argument_text(argument)
                    for argument in template_arguments
                ),
                instantiation.host_name,
            )
            if key in seen:
                continue
            seen.add(key)
            materialized = self._materialize_template_function(
                code, template, instantiation
            )
            if materialized:
                template.materializations.append(materialized)

        for template in templates:
            if template.materializations:
                replacement = "\n\n".join(template.materializations)
                replacements.append((template.span[0], template.span[1], replacement))

        return self._apply_text_replacements(code, replacements)

    def _materialize_explicit_template_function_calls(
        self,
        code: str,
        *,
        work_budget: Optional[object] = None,
        materialized_names: Optional[Dict[Tuple[str, Tuple[str, ...]], str]] = None,
    ) -> str:
        materialized_names = (
            materialized_names if materialized_names is not None else {}
        )
        working = code

        while True:
            templates = self._find_template_functions(working)
            if not templates:
                return working

            templates_by_name = {template.name: template for template in templates}
            template_spans = self._find_template_declaration_spans(working)
            reachable_function_spans = self._reachable_function_spans(
                working, template_spans
            )
            explicit_specialization_keys = (
                self._find_explicit_template_specialization_keys(working)
            )
            calls = self._find_explicit_template_function_calls(
                working,
                templates_by_name,
                template_spans,
                explicit_specialization_keys,
                reachable_function_spans,
            )
            if not calls:
                return working

            replacements: List[Tuple[int, int, str]] = []
            new_materializations: List[str] = []
            for (
                function_name,
                template_arguments,
                spans,
            ) in self._dedupe_explicit_template_function_calls(calls):
                key = self._template_specialization_key(
                    function_name, template_arguments
                )
                template = templates_by_name[function_name]
                if not self._template_arguments_satisfy_parameters(
                    template,
                    template_arguments,
                ):
                    continue
                specialized_name = materialized_names.get(key)
                if specialized_name is not None:
                    replacements.extend(
                        (span[0], span[1], specialized_name) for span in spans
                    )
                    continue
                unique_count = len(materialized_names) + 1
                if len(materialized_names) >= self.max_template_specializations:
                    requested_signature = self._template_specialization_signature(
                        function_name, template_arguments
                    )
                    suggested_action = (
                        "raise max_template_specializations for this source pattern "
                        "or backend, or reduce explicit template helper "
                        "instantiations"
                    )
                    raise MetalTemplateSpecializationError(
                        "Metal template specialization limit exceeded while "
                        f"materializing '{requested_signature}'; "
                        f"{unique_count} unique concrete signatures requested, "
                        f"limit {self.max_template_specializations} from "
                        f"{self.template_specialization_limit_source}. "
                        f"Suggested action: {suggested_action}.",
                        limit=self.max_template_specializations,
                        limit_source=self.template_specialization_limit_source,
                        unique_specialization_count=unique_count,
                        requested_signature=requested_signature,
                        suggested_action=suggested_action,
                    )
                if work_budget is not None:
                    consume_unique = getattr(work_budget, "consume_unique", None)
                    work_key = ("function", key[0], *key[1])
                    context = (
                        "reachable concrete helper "
                        f"'{self._template_specialization_signature(function_name, template_arguments)}'"
                    )
                    if callable(consume_unique):
                        consume_unique(
                            work_key,
                            offset=spans[0][0],
                            length=max(spans[0][1] - spans[0][0], 0),
                            context=context,
                        )
                    else:
                        work_budget.consume(
                            1,
                            offset=spans[0][0],
                            length=max(spans[0][1] - spans[0][0], 0),
                            context=context,
                        )
                specialized_name = self._template_specialization_identifier(
                    function_name, list(key[1])
                )
                materialized = self._materialize_template_function_with_name(
                    template,
                    template_arguments,
                    specialized_name,
                    host_name=None,
                )
                if materialized:
                    replacements.extend(
                        (span[0], span[1], specialized_name) for span in spans
                    )
                    materialized_names[key] = specialized_name
                    new_materializations.append(materialized)

            if not replacements and not new_materializations:
                return working

            working = self._apply_text_replacements(working, replacements)
            if new_materializations:
                working = working.rstrip() + "\n\n" + "\n\n".join(new_materializations)
                if not working.endswith("\n"):
                    working += "\n"

    def _materialize_explicit_template_struct_instantiations(
        self, code: str, *, work_budget: Optional[object] = None
    ) -> str:
        # Struct counterpart of _materialize_explicit_template_function_calls
        # (issue #1354). Replaces concrete `StructName<args>` type references with
        # a materialized concrete struct, iterating so nested instantiations
        # (e.g. BlockLoader referencing BlockMMA<...>) resolve. Regression-safe:
        # any failure returns the unmodified source, so a kernel that previously
        # reached the template-materialization diagnostic is left unchanged rather
        # than emitting a half-rewritten translation.
        try:
            return self._materialize_explicit_template_struct_instantiations_impl(
                code,
                work_budget=work_budget,
            )
        except MetalTemplateSpecializationError:
            raise
        except Exception:
            return code

    def _materialize_explicit_template_struct_instantiations_impl(
        self, code: str, *, work_budget: Optional[object] = None
    ) -> str:
        materialized_names: Dict[Tuple[str, Tuple[str, ...]], str] = {}
        working = code
        # Each iteration resolves one "layer" of instantiations; the bound mirrors
        # the specialization budget so deeply nested or pathological inputs cannot
        # loop unbounded.
        max_iterations = self.max_template_specializations + 1
        for _ in range(max_iterations):
            templates = self._find_template_structs(working)
            if not templates:
                return working
            primary_templates = [
                template
                for template in templates
                if self._template_struct_specialization_arguments(template) is None
            ]
            templates_by_name = {
                template.name: template for template in primary_templates
            }
            if not templates_by_name:
                return working
            explicit_specialization_keys = (
                self._find_explicit_struct_specialization_keys(
                    working,
                    templates_by_name,
                )
            )
            partial_specializations: Dict[
                str, List[Tuple[_MetalTemplateStruct, List[str]]]
            ] = {}
            for template in templates:
                specialization_arguments = (
                    self._template_struct_specialization_arguments(template)
                )
                if specialization_arguments is None:
                    continue
                partial_specializations.setdefault(template.name, []).append(
                    (template, specialization_arguments)
                )
            excluded_spans = self._find_template_declaration_spans(working)
            (
                instantiations,
                selected_partial_specializations,
            ) = self._find_explicit_template_struct_instantiations(
                working,
                templates_by_name,
                excluded_spans,
                explicit_specialization_keys,
                partial_specializations,
            )
            if not instantiations:
                return working

            replacements: List[Tuple[int, int, str]] = []
            new_materializations: List[str] = []
            for (
                struct_name,
                template_arguments,
                spans,
            ) in self._dedupe_explicit_template_function_calls(instantiations):
                key = self._template_specialization_key(struct_name, template_arguments)
                partial_selection = selected_partial_specializations.get(key)
                primary_template = templates_by_name[struct_name]
                if partial_selection is None and not (
                    self._template_arguments_satisfy_parameters(
                        primary_template, template_arguments
                    )
                ):
                    continue
                selected_template = primary_template
                selected_bindings: Dict[str, str]
                if partial_selection is None:
                    selected_bindings, _variadic_bindings = (
                        self._template_argument_bindings(
                            primary_template,
                            template_arguments,
                        )
                    )
                else:
                    selected_template, selected_bindings = partial_selection

                alias_replacements: List[Tuple[int, int, str]] = []
                materialization_spans: List[Tuple[int, int]] = []
                for span in spans:
                    qualified_alias = self._qualified_template_struct_alias_at(
                        working,
                        span,
                    )
                    if qualified_alias is None:
                        materialization_spans.append(span)
                        continue
                    alias_name, alias_span = qualified_alias
                    alias_target = self._concrete_template_struct_scalar_alias_target(
                        selected_template,
                        selected_bindings,
                        alias_name,
                    )
                    if alias_target is None:
                        materialization_spans.append(span)
                        continue
                    alias_replacements.append(
                        (alias_span[0], alias_span[1], alias_target)
                    )
                replacements.extend(alias_replacements)
                if not materialization_spans:
                    continue

                specialized_name = materialized_names.get(key)
                if specialized_name is not None:
                    replacements.extend(
                        (span[0], span[1], specialized_name)
                        for span in materialization_spans
                    )
                    continue
                if len(materialized_names) >= self.max_template_specializations:
                    # Stay regression-safe instead of raising: leave the residual
                    # `Name<...>` so the existing template-materialization
                    # diagnostic fires exactly as it does today.
                    return code
                if work_budget is not None:
                    consume_unique = getattr(work_budget, "consume_unique", None)
                    work_key = ("struct", key[0], *key[1])
                    context = (
                        "reachable concrete struct "
                        f"'{self._template_specialization_signature(struct_name, template_arguments)}'"
                    )
                    first_span = materialization_spans[0]
                    if callable(consume_unique):
                        consume_unique(
                            work_key,
                            offset=first_span[0],
                            length=max(first_span[1] - first_span[0], 0),
                            context=context,
                        )
                    else:
                        work_budget.consume(
                            1,
                            offset=first_span[0],
                            length=max(first_span[1] - first_span[0], 0),
                            context=context,
                        )
                specialized_name = self._template_specialization_identifier(
                    struct_name, list(key[1])
                )
                if partial_selection is None:
                    materialized = self._materialize_template_struct_with_name(
                        primary_template,
                        template_arguments,
                        specialized_name,
                    )
                else:
                    partial_template, partial_bindings = partial_selection
                    materialized = self._materialize_partial_template_struct_with_name(
                        partial_template,
                        partial_bindings,
                        specialized_name,
                    )
                if materialized:
                    replacements.extend(
                        (span[0], span[1], specialized_name)
                        for span in materialization_spans
                    )
                    materialized_names[key] = specialized_name
                    self._materialized_struct_specializations[specialized_name] = (
                        struct_name,
                        tuple(key[1]),
                    )
                    new_materializations.append(materialized)

            if not replacements and not new_materializations:
                return working

            working = self._apply_text_replacements(working, replacements)
            if new_materializations:
                working = working.rstrip() + "\n\n" + "\n\n".join(new_materializations)
                if not working.endswith("\n"):
                    working += "\n"
        return working

    def _prune_unreferenced_template_struct_declarations(self, code: str) -> str:
        declarations = self._find_all_template_struct_declaration_spans(code)
        if not declarations:
            return code

        declarations_by_name: Dict[str, List[Tuple[int, int]]] = {}
        for name, span in declarations:
            declarations_by_name.setdefault(name, []).append(span)
        declaration_spans = sorted(span for _name, span in declarations)
        owners_by_span = {span: name for name, span in declarations}
        ignored_spans = self._find_comment_and_literal_spans(code)

        dependencies: Dict[str, Set[str]] = {
            name: set() for name in declarations_by_name
        }
        live_names: Set[str] = set()
        names_pattern = "|".join(
            re.escape(name)
            for name in sorted(declarations_by_name, key=len, reverse=True)
        )
        reference_pattern = re.compile(rf"\b(?P<name>{names_pattern})\s*<")
        for match in reference_pattern.finditer(code):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            referenced_name = match.group("name")
            containing = [
                span for span in declaration_spans if span[0] <= match.start() < span[1]
            ]
            if not containing:
                live_names.add(referenced_name)
                continue
            owner_span = min(containing, key=lambda span: span[1] - span[0])
            owner_name = owners_by_span[owner_span]
            if owner_name != referenced_name:
                dependencies[owner_name].add(referenced_name)

        pending = list(live_names)
        while pending:
            owner_name = pending.pop()
            for dependency in dependencies.get(owner_name, set()):
                if dependency in live_names:
                    continue
                live_names.add(dependency)
                pending.append(dependency)

        replacements = [
            (span[0], span[1], "")
            for name, span in declarations
            if name not in live_names
        ]
        if not replacements:
            return code
        return self._apply_text_replacements(code, replacements)

    def _find_all_template_struct_declaration_spans(
        self, code: str
    ) -> List[Tuple[str, Tuple[int, int]]]:
        declarations: List[Tuple[str, Tuple[int, int]]] = []
        for span in self._find_template_declaration_spans(code):
            angle_start = code.find("<", span[0], span[1])
            if angle_start == -1:
                continue
            angle_end = self._find_matching_template_param_angle(code, angle_start)
            if angle_end is None or angle_end >= span[1]:
                continue
            header = code[angle_end + 1 : span[1]]
            match = re.match(
                r"\s*(?:\[\[[^\]]*\]\]\s*)*(?:struct|class)\s+"
                r"(?P<name>[A-Za-z_][A-Za-z0-9_:]*)\b",
                header,
                re.DOTALL,
            )
            if match is None:
                continue
            declarations.append((match.group("name").split("::")[-1], span))
        return declarations

    def _elide_stateless_compile_time_globals(self, code: str) -> str:
        """Erase compile-time objects only after proving all observable effects.

        This runs before generic member lowering because empty variadic methods
        otherwise lose their bodies before overload selection. Any uncertain
        layout, lifetime operation, receiver use, or method body raises a
        source-located diagnostic instead of producing a dangling target call.
        """
        structs = self._find_concrete_struct_definitions(code)
        if not structs:
            return code
        globals_ = self._find_compile_time_struct_globals(code, structs)
        if not globals_:
            return code

        structs_by_name = {struct.qualified_name: struct for struct in structs}
        namespace_spans = self._find_namespace_spans(code)
        declaration_spans = [
            span
            for global_ in globals_
            for span in (global_.span, *global_.prior_declaration_spans)
        ]
        global_names = {global_.name for global_ in globals_}
        replacements: List[Tuple[int, int, str]] = []
        for global_ in globals_:
            struct = structs_by_name[global_.type_name]
            if not self._stateless_global_layout_is_proven_empty(code, struct):
                continue
            self._prove_stateless_global_constructor(
                code, global_, struct, global_names
            )
            self._prove_stateless_global_destructor(code, global_, struct)
            replacements.extend(
                self._stateless_global_call_replacements(
                    code,
                    global_,
                    struct,
                    namespace_spans,
                    declaration_spans,
                    global_names,
                )
            )
            replacements.extend(
                (start, end, "") for start, end in global_.prior_declaration_spans
            )
            replacements.append((global_.span[0], global_.span[1], ""))
        return self._apply_text_replacements(code, replacements)

    def _find_compile_time_struct_globals(
        self,
        code: str,
        structs: List[_MetalStructDefinition],
    ) -> List[_MetalCompileTimeGlobal]:
        spellings = {
            spelling
            for struct in structs
            for spelling in (struct.name, struct.qualified_name)
        }
        if not spellings:
            return []
        type_pattern = "|".join(
            self._qualified_type_pattern(spelling)
            for spelling in sorted(spellings, key=len, reverse=True)
        )
        declaration_pattern = re.compile(
            r"(?<![A-Za-z0-9_])"
            r"(?P<qualifiers>(?:(?:static|inline|constexpr|const|constant|"
            r"extern|typedef|thread_local|volatile)\s+)+)"
            rf"(?P<type>(?:::)?(?:{type_pattern}))\s+"
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
        )
        ignored_spans = self._find_comment_and_literal_spans(code)
        masked_code = self._mask_comments_and_literals(code)
        lexical_scopes = self._find_lexical_brace_scopes(code)
        namespace_spans = self._find_namespace_spans(code)
        namespace_scopes = {(start, end) for start, end, _name in namespace_spans}
        structs_by_qualified = {struct.qualified_name: struct for struct in structs}
        globals_: List[_MetalCompileTimeGlobal] = []

        for match in declaration_pattern.finditer(code):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            line_start = masked_code.rfind("\n", 0, match.start()) + 1
            line_prefix = masked_code[line_start : match.start()]
            if line_prefix.strip():
                previous = match.start() - 1
                while previous >= line_start and masked_code[previous].isspace():
                    previous -= 1
                if previous >= line_start and masked_code[previous] not in ";{}":
                    continue
            scope = self._innermost_lexical_scope(
                lexical_scopes, match.start(), len(code)
            )
            if scope != (0, len(code)) and scope not in namespace_scopes:
                continue
            qualifiers = set(IDENTIFIER_RE.findall(match.group("qualifiers")))
            if not ({"constant", "constexpr"} & qualifiers):
                continue
            if qualifiers & {"extern", "typedef", "thread_local", "volatile"}:
                continue
            namespace = self._namespace_at(namespace_spans, match.start())
            source_type = re.sub(r"\s*::\s*", "::", match.group("type"))
            if source_type.startswith("::"):
                source_type = source_type[2:]
            struct = structs_by_qualified.get(source_type)
            if struct is None:
                candidates = [
                    candidate
                    for candidate in structs
                    if candidate.name == source_type
                    and self._struct_namespace(candidate) == namespace
                ]
                if len(candidates) != 1:
                    self._raise_stateless_global_error(
                        code,
                        global_name=match.group("name"),
                        type_name=source_type,
                        reason="ambiguous-type",
                        start=match.start("type"),
                        end=match.end("type"),
                        blocking_use=match.group("type"),
                    )
                struct = candidates[0]

            parsed = self._parse_compile_time_global_initializer(
                code,
                match.end("name"),
                {source_type, struct.name, struct.qualified_name},
            )
            if parsed is None:
                continue
            arguments, declaration_end = parsed
            prior_declaration_spans = self._find_prior_compile_time_global_declarations(
                code,
                before=match.start(),
                name=match.group("name"),
                expected_type_names={
                    source_type,
                    struct.name,
                    struct.qualified_name,
                },
                namespace=namespace,
                lexical_scopes=lexical_scopes,
                namespace_scopes=namespace_scopes,
                namespace_spans=namespace_spans,
            )
            globals_.append(
                _MetalCompileTimeGlobal(
                    name=match.group("name"),
                    type_name=struct.qualified_name,
                    namespace=namespace,
                    arguments=tuple(arguments),
                    span=(match.start(), declaration_end),
                    name_span=(match.start("name"), match.end("name")),
                    receiver_address_space="constant",
                    prior_declaration_spans=tuple(prior_declaration_spans),
                )
            )
        return globals_

    def _find_prior_compile_time_global_declarations(
        self,
        code: str,
        *,
        before: int,
        name: str,
        expected_type_names: Set[str],
        namespace: str,
        lexical_scopes: List[Tuple[int, int]],
        namespace_scopes: Set[Tuple[int, int]],
        namespace_spans: List[Tuple[int, int, str]],
    ) -> List[Tuple[int, int]]:
        type_pattern = "|".join(
            self._qualified_type_pattern(spelling)
            for spelling in sorted(expected_type_names, key=len, reverse=True)
        )
        pattern = re.compile(
            r"(?<![A-Za-z0-9_])"
            r"(?P<qualifiers>(?:(?:static|inline|constexpr|const|constant|"
            r"extern|typedef|thread_local|volatile)\s+)+)"
            rf"(?P<type>(?:::)?(?:{type_pattern}))\s+"
            rf"{re.escape(name)}\s*;"
        )
        masked = self._mask_comments_and_literals(code)
        declarations: List[Tuple[int, int]] = []
        for match in pattern.finditer(masked, 0, before):
            qualifiers = set(IDENTIFIER_RE.findall(match.group("qualifiers")))
            if "extern" not in qualifiers or not (
                {"constant", "constexpr"} & qualifiers
            ):
                continue
            scope = self._innermost_lexical_scope(
                lexical_scopes, match.start(), len(code)
            )
            if scope != (0, len(code)) and scope not in namespace_scopes:
                continue
            declaration_namespace = self._namespace_at(namespace_spans, match.start())
            if declaration_namespace != namespace:
                continue
            source_type = re.sub(r"\s*::\s*", "::", match.group("type"))
            if source_type.startswith("::"):
                source_type = source_type[2:]
            if source_type not in expected_type_names:
                continue
            declarations.append(match.span())
        return declarations

    @staticmethod
    def _qualified_type_pattern(type_name: str) -> str:
        return r"\s*::\s*".join(re.escape(part) for part in type_name.split("::"))

    @staticmethod
    def _struct_namespace(struct: _MetalStructDefinition) -> str:
        if "::" not in struct.qualified_name:
            return ""
        return struct.qualified_name.rsplit("::", 1)[0]

    def _parse_compile_time_global_initializer(
        self, code: str, name_end: int, expected_type_names: Set[str]
    ) -> Optional[Tuple[List[str], int]]:
        cursor = self._skip_cpp_trivia(code, name_end)
        if cursor >= len(code):
            return None
        if code[cursor] == ";":
            return [], cursor + 1
        if code[cursor] in "({":
            opener = code[cursor]
            closer = ")" if opener == "(" else "}"
            end = self._find_matching_delimiter(code, cursor, opener, closer)
            if end is None:
                return None
            after = self._skip_cpp_trivia(code, end + 1)
            if after >= len(code) or code[after] != ";":
                return None
            text = code[cursor + 1 : end]
            # `Type value();` is a function declaration, not an object.
            if opener == "(" and (
                not text.strip()
                or self._looks_like_function_parameter_declarations(text)
            ):
                return None
            return self._split_initializer_arguments(text), after + 1
        if code[cursor] != "=":
            return None
        declaration_end = self._statement_end(code, cursor)
        if declaration_end is None:
            return None
        initializer = code[cursor + 1 : declaration_end - 1].strip()
        if initializer.startswith("{"):
            end = self._find_matching_delimiter(initializer, 0, "{", "}")
            if end == len(initializer) - 1:
                return (
                    self._split_initializer_arguments(initializer[1:end]),
                    declaration_end,
                )
        constructor = re.match(
            r"(?P<type>(?:::)?[A-Za-z_][A-Za-z0-9_]*"
            r"(?:\s*::\s*[A-Za-z_]\w*)*)\s*(?P<open>[({])",
            initializer,
        )
        if constructor is None:
            return None
        constructor_type = re.sub(r"\s*::\s*", "::", constructor.group("type"))
        if constructor_type.startswith("::"):
            constructor_type = constructor_type[2:]
        if constructor_type not in expected_type_names:
            return None
        opener = constructor.group("open")
        open_index = constructor.end() - 1
        closer = ")" if opener == "(" else "}"
        end = self._find_matching_delimiter(initializer, open_index, opener, closer)
        if end != len(initializer) - 1:
            return None
        return (
            self._split_initializer_arguments(initializer[open_index + 1 : end]),
            declaration_end,
        )

    def _split_initializer_arguments(self, text: str) -> List[str]:
        return [
            argument.strip()
            for argument in self._split_top_level_commas(text)
            if argument.strip()
        ]

    def _looks_like_function_parameter_declarations(self, text: str) -> bool:
        parameters = self._split_top_level_commas(text)
        if not parameters:
            return True
        for parameter in parameters:
            declarator, _default = self._split_top_level_assignment(parameter)
            declarator = declarator.strip()
            if not declarator or declarator in {"true", "false", "nullptr"}:
                return False
            if any(character in declarator for character in "\"'(){}"):
                return False
            if re.search(r"[+\-/%|^!?=]", declarator):
                return False
            if IDENTIFIER_RE.search(declarator) is None:
                return False
        return True

    def _stateless_global_layout_is_proven_empty(
        self,
        code: str,
        struct: _MetalStructDefinition,
    ) -> bool:
        if struct.base_clause or struct.data_members:
            return False
        body = code[struct.body_span[0] : struct.body_span[1]]
        return self._first_unproven_struct_state_declaration(body) is None

    def _first_unproven_struct_state_declaration(self, body: str) -> Optional[str]:
        i = 0
        while i < len(body):
            i = self._skip_cpp_trivia(body, i)
            if i >= len(body):
                return None
            access = re.match(r"(?:public|private|protected)\s*:", body[i:])
            if access is not None:
                i += access.end()
                continue

            declaration_start = i
            if re.match(r"template\s*<", body[i:]):
                angle_start = body.find("<", i)
                angle_end = self._find_matching_template_param_angle(body, angle_start)
                if angle_end is None:
                    return body[declaration_start:].strip()
                i = angle_end + 1

            brace = self._find_next_top_level_char(body, i, "{")
            semicolon = self._find_next_top_level_char(body, i, ";")
            if brace is not None and (semicolon is None or brace < semicolon):
                header = body[i:brace]
                brace_end = self._find_matching_brace(body, brace)
                if brace_end is None:
                    return body[declaration_start:].strip()
                if not self._stateless_member_header_is_function(header):
                    end = self._statement_end(body, declaration_start) or brace_end
                    return body[declaration_start:end].strip()
                i = brace_end
                trailing = self._skip_cpp_trivia(body, i)
                if trailing < len(body) and body[trailing] == ";":
                    i = trailing + 1
                continue
            if semicolon is None:
                trailing = body[declaration_start:].strip()
                return trailing or None
            declaration = body[i:semicolon].strip()
            if not declaration:
                i = semicolon + 1
                continue
            if self._declaration_is_type_alias(declaration):
                i = semicolon + 1
                continue
            if re.match(r"static_assert\s*\(", declaration):
                i = semicolon + 1
                continue
            if self._stateless_member_declaration_is_function(declaration):
                i = semicolon + 1
                continue
            return body[declaration_start : semicolon + 1].strip()
        return None

    def _stateless_member_header_is_function(self, header: str) -> bool:
        if re.search(r"\bvirtual\b", header):
            return False
        paren_start = self._function_parameter_start(header)
        if paren_start is None:
            return False
        prefix = header[:paren_start]
        if "=" in prefix or re.search(r"\[[^\]]*\]\s*$", prefix):
            return False
        return bool(
            re.search(
                r"(?:~\s*)?[A-Za-z_][A-Za-z0-9_]*\s*$|"
                r"\boperator(?:\s*\(\s*\)|\s*[^\s]+)\s*$",
                prefix,
            )
        )

    def _stateless_member_declaration_is_function(self, declaration: str) -> bool:
        if re.search(r"\bvirtual\b", declaration):
            return False
        if re.search(r"\(\s*[*&]", declaration):
            return False
        declarator, default = self._split_top_level_assignment(declaration)
        if default is not None and default.strip() not in {"default", "delete"}:
            return False
        return self._declaration_is_method_prototype(declarator)

    def _prove_stateless_global_constructor(
        self,
        code: str,
        global_: _MetalCompileTimeGlobal,
        struct: _MetalStructDefinition,
        global_names: Set[str],
    ) -> None:
        for argument in global_.arguments:
            referenced_globals = (
                set(IDENTIFIER_RE.findall(self._mask_comments_and_literals(argument)))
                & global_names
            )
            if referenced_globals:
                self._raise_stateless_global_error(
                    code,
                    global_=global_,
                    reason="initializer-observes-global-identity",
                    start=global_.name_span[0],
                    end=global_.span[1],
                    blocking_use=", ".join(sorted(referenced_globals)),
                    effect="object-identity",
                )
            if not self._stateless_initializer_argument_is_proven_constant(argument):
                self._raise_stateless_global_error(
                    code,
                    global_=global_,
                    reason="initializer-effect-unproven",
                    start=global_.name_span[0],
                    end=global_.span[1],
                    blocking_use=argument,
                    effect="initializer-evaluation",
                )

        if struct.name in struct.member_prototype_names:
            self._raise_stateless_global_error(
                code,
                global_=global_,
                reason="constructor-body-unresolved",
                start=global_.name_span[0],
                end=global_.span[1],
                blocking_use=struct.name,
            )

        matching = [
            constructor
            for constructor in struct.constructors
            if self._callable_accepts_argument_count(
                constructor.params_text, len(global_.arguments)
            )
            and global_.receiver_address_space in constructor.receiver_address_spaces
        ]
        if not matching:
            has_declared_constructor = bool(struct.constructors) or (
                struct.name in struct.member_prototype_names
            )
            if not global_.arguments and not has_declared_constructor:
                return
            self._raise_stateless_global_error(
                code,
                global_=global_,
                reason="constructor-unresolved",
                start=global_.name_span[0],
                end=global_.span[1],
                blocking_use=f"{struct.qualified_name}({', '.join(global_.arguments)})",
            )
        for constructor in matching:
            if not constructor.is_constexpr:
                self._raise_stateless_global_error(
                    code,
                    global_=global_,
                    reason="constructor-not-constexpr",
                    start=(
                        constructor.span[0]
                        if constructor.span
                        else global_.name_span[0]
                    ),
                    end=(
                        constructor.span[1]
                        if constructor.span
                        else global_.name_span[1]
                    ),
                    blocking_use=constructor.prefix.rstrip("("),
                )
            if constructor.init_text.strip():
                self._raise_stateless_global_error(
                    code,
                    global_=global_,
                    reason="constructor-initializer-not-empty",
                    start=(
                        constructor.span[0]
                        if constructor.span
                        else global_.name_span[0]
                    ),
                    end=(
                        constructor.span[1]
                        if constructor.span
                        else global_.name_span[1]
                    ),
                    blocking_use=constructor.init_text,
                    effect="object-state",
                )
            effect = self._stateless_body_effect(constructor.body, allow_return=False)
            if effect is not None:
                self._raise_stateless_global_error(
                    code,
                    global_=global_,
                    reason="constructor-has-effects",
                    start=(
                        constructor.span[0]
                        if constructor.span
                        else global_.name_span[0]
                    ),
                    end=(
                        constructor.span[1]
                        if constructor.span
                        else global_.name_span[1]
                    ),
                    blocking_use=constructor.body.strip(),
                    effect=effect,
                )

    def _prove_stateless_global_destructor(
        self,
        code: str,
        global_: _MetalCompileTimeGlobal,
        struct: _MetalStructDefinition,
    ) -> None:
        body_start, body_end = struct.body_span
        body = code[body_start:body_end]
        masked = self._mask_comments_and_literals(body)
        scopes = self._find_lexical_brace_scopes(body)
        pattern = re.compile(rf"~\s*{re.escape(struct.name)}\s*\(")
        for match in pattern.finditer(masked):
            if self._innermost_lexical_scope(scopes, match.start(), len(body)) != (
                0,
                len(body),
            ):
                continue
            open_paren = masked.find("(", match.start(), match.end())
            close_paren = self._find_matching_delimiter(body, open_paren, "(", ")")
            if close_paren is None:
                self._raise_stateless_global_error(
                    code,
                    global_=global_,
                    reason="destructor-unresolved",
                    start=body_start + match.start(),
                    end=body_start + match.end(),
                )
            cursor = self._skip_cpp_trivia(body, close_paren + 1)
            while True:
                qualifier = IDENTIFIER_RE.match(body, cursor)
                if qualifier is None or qualifier.group(0) not in {
                    "const",
                    "constant",
                    "device",
                    "thread",
                    "threadgroup",
                    "noexcept",
                }:
                    break
                cursor = self._skip_cpp_trivia(body, qualifier.end())
            if cursor < len(body) and body[cursor] == "{":
                close_brace = self._find_matching_delimiter(body, cursor, "{", "}")
                if close_brace is None:
                    effect = "unresolved-body"
                    close_brace = cursor
                else:
                    effect = self._stateless_body_effect(
                        body[cursor + 1 : close_brace], allow_return=False
                    )
                if effect is None:
                    continue
                end = close_brace + 1
            else:
                declaration_end = body.find(";", cursor)
                if declaration_end == -1:
                    declaration_end = cursor
                suffix = body[cursor:declaration_end].strip()
                if re.fullmatch(r"=\s*default", suffix):
                    continue
                effect = "deleted-body" if "delete" in suffix else "unresolved-body"
                end = declaration_end + 1
            self._raise_stateless_global_error(
                code,
                global_=global_,
                reason="destructor-has-effects",
                start=body_start + match.start(),
                end=body_start + end,
                blocking_use=body[match.start() : end].strip(),
                effect=effect,
            )

    def _stateless_global_call_replacements(
        self,
        code: str,
        global_: _MetalCompileTimeGlobal,
        struct: _MetalStructDefinition,
        namespace_spans: List[Tuple[int, int, str]],
        declaration_spans: List[Tuple[int, int]],
        global_names: Set[str],
    ) -> List[Tuple[int, int, str]]:
        masked = self._mask_comments_and_literals(code)
        replacements: List[Tuple[int, int, str]] = []
        position = min(
            (end for _start, end in global_.prior_declaration_spans),
            default=global_.span[1],
        )
        pattern = re.compile(rf"\b{re.escape(global_.name)}\b")
        while True:
            match = pattern.search(masked, position)
            if match is None:
                break
            position = match.end()
            if self._containing_span(match.start(), declaration_spans) is not None:
                continue
            receiver_start = self._stateless_global_receiver_start(
                code, match.start(), global_, namespace_spans
            )
            if receiver_start is None:
                continue
            call = self._parse_stateless_receiver_call(code, match.end())
            if call is None:
                self._raise_stateless_global_error(
                    code,
                    global_=global_,
                    reason="identity-observed",
                    start=receiver_start,
                    end=match.end(),
                    blocking_use=code[receiver_start : match.end()],
                    effect="object-identity",
                )
            method_name, arguments, call_end, is_statement = call
            if not is_statement:
                self._raise_stateless_global_error(
                    code,
                    global_=global_,
                    reason="call-result-observed",
                    start=receiver_start,
                    end=call_end,
                    method_name=method_name,
                    blocking_use=code[receiver_start:call_end],
                    effect="returned-value",
                )
            if method_name in struct.member_prototype_names:
                self._raise_stateless_global_error(
                    code,
                    global_=global_,
                    reason="method-body-unresolved",
                    start=receiver_start,
                    end=call_end,
                    method_name=method_name,
                    blocking_use=code[receiver_start:call_end],
                )
            candidates = [
                method
                for method in [*struct.methods, *struct.template_methods]
                if method.name == method_name
                and not method.is_static
                and method.is_const
                and global_.receiver_address_space in method.receiver_address_spaces
                and self._callable_accepts_argument_count(
                    method.parameters, len(arguments)
                )
            ]
            if not candidates:
                self._raise_stateless_global_error(
                    code,
                    global_=global_,
                    reason="method-overload-unresolved",
                    start=receiver_start,
                    end=call_end,
                    method_name=method_name,
                    blocking_use=code[receiver_start:call_end],
                )
            for method in candidates:
                if self._normalize_inferred_type(method.return_type) != "void":
                    self._raise_stateless_global_error(
                        code,
                        global_=global_,
                        reason="method-returns-value",
                        start=receiver_start,
                        end=call_end,
                        method_name=method_name,
                        blocking_use=method.return_type,
                        effect="returned-value",
                    )
                effect = self._stateless_body_effect(method.body, allow_return=True)
                if effect is not None:
                    self._raise_stateless_global_error(
                        code,
                        global_=global_,
                        reason="method-has-effects",
                        start=receiver_start,
                        end=call_end,
                        method_name=method_name,
                        blocking_use=method.body.strip(),
                        effect=effect,
                    )
            for argument in arguments:
                referenced_globals = (
                    set(
                        IDENTIFIER_RE.findall(
                            self._mask_comments_and_literals(argument)
                        )
                    )
                    & global_names
                )
                if referenced_globals:
                    self._raise_stateless_global_error(
                        code,
                        global_=global_,
                        reason="identity-observed",
                        start=receiver_start,
                        end=call_end,
                        method_name=method_name,
                        blocking_use=", ".join(sorted(referenced_globals)),
                        effect="object-identity",
                    )
            evaluated = [
                argument
                for argument in arguments
                if self._stateless_argument_requires_evaluation(argument)
            ]
            replacement = self._render_elided_call_arguments(evaluated)
            replacements.append((receiver_start, call_end, replacement))
            position = call_end
        return replacements

    def _stateless_global_receiver_start(
        self,
        code: str,
        name_start: int,
        global_: _MetalCompileTimeGlobal,
        namespace_spans: List[Tuple[int, int, str]],
    ) -> Optional[int]:
        line_start = code.rfind("\n", 0, name_start) + 1
        prefix = code[line_start:name_start]
        qualified = re.search(
            r"(?P<qualification>(?:::)?(?:[A-Za-z_]\w*\s*::\s*)+)$", prefix
        )
        if qualified is not None:
            qualification = re.sub(
                r"\s*::\s*", "::", qualified.group("qualification")
            ).strip(":")
            if qualification != global_.namespace:
                return None
            return line_start + qualified.start("qualification")
        previous = name_start - 1
        while previous >= 0 and code[previous].isspace():
            previous -= 1
        if previous >= 0 and code[previous] == ".":
            return None
        if previous >= 1 and code[previous - 1 : previous + 1] == "->":
            return None
        namespace = self._namespace_at(namespace_spans, name_start)
        if global_.namespace and not (
            namespace == global_.namespace
            or namespace.startswith(global_.namespace + "::")
        ):
            return None
        return name_start

    def _parse_stateless_receiver_call(
        self, code: str, name_end: int
    ) -> Optional[Tuple[str, List[str], int, bool]]:
        cursor = self._skip_cpp_trivia(code, name_end)
        if cursor >= len(code) or code[cursor] != ".":
            return None
        cursor = self._skip_cpp_trivia(code, cursor + 1)
        method = IDENTIFIER_RE.match(code, cursor)
        if method is None:
            return None
        method_name = method.group(0)
        cursor = self._skip_cpp_trivia(code, method.end())
        if cursor < len(code) and code[cursor] == "<":
            angle_end = self._find_matching_angle(code, cursor)
            if angle_end is None:
                return None
            cursor = self._skip_cpp_trivia(code, angle_end + 1)
        if cursor >= len(code) or code[cursor] != "(":
            return None
        close_paren = self._find_matching_delimiter(code, cursor, "(", ")")
        if close_paren is None:
            return None
        arguments = self._split_initializer_arguments(code[cursor + 1 : close_paren])
        call_end = self._skip_cpp_trivia(code, close_paren + 1)
        is_statement = call_end < len(code) and code[call_end] == ";"
        return (
            method_name,
            arguments,
            call_end + 1 if is_statement else close_paren + 1,
            is_statement,
        )

    def _callable_accepts_argument_count(self, parameters: str, count: int) -> bool:
        parts = [
            part.strip()
            for part in self._split_top_level_commas(parameters)
            if part.strip() and part.strip() != "void"
        ]
        variadic = any("..." in part for part in parts)
        fixed = [part for part in parts if "..." not in part]
        required = sum(
            1
            for parameter in fixed
            if self._split_top_level_assignment(parameter)[1] is None
        )
        return required <= count and (variadic or count <= len(fixed))

    def _stateless_initializer_argument_is_proven_constant(self, argument: str) -> bool:
        masked = self._mask_comments_and_literals(argument)
        if re.search(r"\+\+|--|\b(?:new|delete|throw|asm)\b", masked):
            return False
        if re.search(
            r"(?:<<|>>|[+\-*/%&|^])=|(?<![=!<>+\-*/%&|^])=(?!=)",
            masked,
        ):
            return False
        if re.search(r"\b[A-Za-z_]\w*\s*(?:<[^;{}()]*>)?\s*\(", masked):
            return False
        return not bool(re.search(r"\[|\]|->|\.", masked))

    def _stateless_body_effect(self, body: str, *, allow_return: bool) -> Optional[str]:
        masked = self._mask_comments_and_literals(body).strip()
        if not masked:
            return None
        if allow_return and re.fullmatch(r"(?:return\s*;\s*)+", masked):
            return None
        if re.search(r"\b(?:atomic\w*|barrier|threadgroup_barrier)\b", masked):
            return "atomic-or-synchronization"
        if re.search(r"\b(?:trap|abort|assert|discard_fragment)\b", masked):
            return "trap-or-termination"
        if re.search(
            r"\+\+|--|(?:<<|>>|[+\-*/%&|^])=|" r"(?<![=!<>+\-*/%&|^])=(?!=)",
            masked,
        ):
            return "write"
        if re.search(r"\breturn\s+[^;]", masked):
            return "returned-value"
        if re.search(r"\b[A-Za-z_]\w*\s*(?:<[^;{}()]*>)?\s*\(", masked):
            return "function-call"
        return "unproven-statement"

    def _stateless_argument_requires_evaluation(self, argument: str) -> bool:
        text = argument.strip()
        while text.startswith("(") and text.endswith(")"):
            close = self._find_matching_delimiter(text, 0, "(", ")")
            if close != len(text) - 1:
                break
            text = text[1:-1].strip()
        literal = METAL_STRING_EXPRESSION_PATTERN
        scalar = (
            r"(?:true|false|nullptr|[-+]?(?:0[xX][0-9A-Fa-f]+|"
            r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
            r"[uUlLfFhH]*)"
        )
        return re.fullmatch(rf"(?:{literal}|{scalar})", text) is None

    @staticmethod
    def _render_elided_call_arguments(arguments: List[str]) -> str:
        if not arguments:
            return ";"
        if len(arguments) == 1:
            return f"{arguments[0].strip()};"
        statements = "\n".join(f"  {argument.strip()};" for argument in arguments)
        return f"{{\n{statements}\n}}"

    def _raise_stateless_global_error(
        self,
        code: str,
        *,
        reason: str,
        start: int,
        end: int,
        global_: Optional[_MetalCompileTimeGlobal] = None,
        global_name: Optional[str] = None,
        type_name: Optional[str] = None,
        method_name: Optional[str] = None,
        blocking_use: Optional[str] = None,
        effect: Optional[str] = None,
    ) -> None:
        resolved_name = (
            global_.name if global_ is not None else global_name or "<unknown>"
        )
        resolved_type = (
            global_.type_name if global_ is not None else type_name or "<unknown>"
        )
        if blocking_use is not None:
            blocking_use = re.sub(r"\s+", " ", blocking_use).strip()
            if len(blocking_use) > 240:
                blocking_use = blocking_use[:237].rstrip() + "..."
        raise MetalStatelessGlobalElisionError(
            f"Cannot safely elide compile-time global '{resolved_name}' of type "
            f"'{resolved_type}': {reason.replace('-', ' ')}.",
            global_name=resolved_name,
            type_name=resolved_type,
            reason=reason,
            source_location=self._source_location_for_offsets(code, start, end),
            method_name=method_name,
            blocking_use=blocking_use,
            effect=effect,
            suggested_action=(
                "Keep the object state and effects target-representable, or make the "
                "selected constructor, destructor, and constant receiver methods "
                "empty with unobserved object identity."
            ),
        )

    def _lower_struct_member_functions(self, code: str) -> str:
        # CrossGL structs are data-only, so the Metal frontend has historically
        # dropped struct member functions while keeping the now-dangling
        # `obj.method(...)` call sites (invalid output). This pass lowers each
        # concrete (non-template) struct's member functions to FREE functions and
        # rewrites the corresponding call sites BEFORE lexing/parsing. A CALLED
        # template member method is instantiated from its call-site argument
        # types and lowered to a concrete free function too. It runs after the
        # struct-template materializer so concrete structs produced from
        # templates (e.g. `Sum_float`) are lowered too. Regression-safe: an
        # UNEXPECTED failure returns the unmodified source, and the pass no-ops
        # when there are no struct methods (method-free structs stay
        # byte-identical). A DELIBERATE clean-fail for an unresolvable template
        # member method call (MetalStructMethodError) is re-raised so the
        # pipeline reports the kernel as a translation FAILURE instead of leaving
        # a dangling call.
        evaluated = (
            self._evaluate_static_assertions(code)
            if self._fail_closed_static_assertions
            else code
        )
        try:
            return self._lower_struct_member_functions_impl(evaluated)
        except MetalStructMethodError:
            raise
        except Exception:
            return evaluated

    def _lower_struct_member_functions_impl(self, code: str) -> str:
        template_declaration_spans = self._find_template_declaration_spans(code)
        self._source_type_alias_bindings = self._collect_local_type_alias_bindings(
            code,
            [(0, len(code))],
            skip_spans=template_declaration_spans,
        )
        structs = self._find_concrete_struct_definitions(code)
        self._reject_unresolved_temporary_functor_calls(
            code,
            {struct.name: struct for struct in structs},
            template_declaration_spans,
        )
        if not structs:
            return code
        # Only structs that actually declare at least one member function (a
        # concrete method OR a template method) need rewriting; everything else
        # stays untouched so method-free structs and existing kernels are
        # byte-identical.
        structs_with_methods = [
            struct for struct in structs if struct.methods or struct.template_methods
        ]
        if not structs_with_methods:
            return self._canonicalize_qualified_struct_alias_templates(code, structs)

        # Structs with device/threadgroup pointer members cannot be lowered by
        # passing `self` by value (a pointer cannot live in a Function-storage
        # struct), so they take the pointer-member promotion path instead. A
        # struct is only promoted when it is actually CONSTRUCTED in a form we can
        # rewrite (every pointer expression is sourced from a construction
        # argument); otherwise it falls back to the ordinary path unchanged.
        all_struct_spans = [struct.span for struct in structs]
        promotion_plans: Dict[str, _PointerPromotionPlan] = {}
        for struct in structs_with_methods:
            plan = self._pointer_promotion_plan(struct)
            if plan is not None:
                promotion_plans[struct.name] = plan
        promoted_pointer_args: Dict[str, List[Tuple[int, str, List[str]]]] = {}
        promoted_construction_replacements: List[Tuple[int, int, str]] = []
        promoted_names: Set[str] = set()
        if promotion_plans:
            (
                promoted_pointer_args,
                promoted_construction_replacements,
                promoted_names,
            ) = self._scan_pointer_struct_constructions(
                code, promotion_plans, all_struct_spans
            )
        promoted_structs = [
            struct for struct in structs_with_methods if struct.name in promoted_names
        ]
        # The ordinary lowering runs on every struct that is NOT promoted.
        structs_with_methods = [
            struct
            for struct in structs_with_methods
            if struct.name not in promoted_names
        ]

        struct_names = {struct.name for struct in structs_with_methods}
        # ALL concrete struct/union definitions (with or without methods), keyed
        # by name, so a member-access call argument (`obj.field` / `obj.field[i]`)
        # can resolve `field`'s type even when `obj`'s struct carries no methods
        # (e.g. a plain data carrier or a union). Kept separate from the
        # method-driven maps used for call rewriting.
        field_structs_by_name: Dict[str, _MetalStructDefinition] = {
            struct.name: struct for struct in structs
        }
        # Concrete methods keyed by struct name for direct call-site rewriting.
        methods_by_struct: Dict[str, Dict[str, _MetalStructMethod]] = {}
        # Template methods keyed by struct name then method name to the LIST of
        # overloads sharing that name (e.g. the two SFINAE `simd_reduce`
        # overloads). A single-overload name is just a one-element list, so the
        # non-overloaded path is unchanged; an overload set is resolved by
        # evaluating each candidate's SFINAE constraints at the call site.
        template_methods_by_struct: Dict[str, Dict[str, List[_MetalStructMethod]]] = {}
        operator_call_structs: Set[str] = set()
        structs_by_name: Dict[str, _MetalStructDefinition] = {}
        for struct in structs_with_methods:
            structs_by_name[struct.name] = struct
            method_map: Dict[str, _MetalStructMethod] = {}
            for method in struct.methods:
                method_map[method.name] = method
                if method.is_operator_call:
                    operator_call_structs.add(struct.name)
            methods_by_struct[struct.name] = method_map
            template_map: Dict[str, List[_MetalStructMethod]] = {}
            for method in struct.template_methods:
                template_map.setdefault(method.name, []).append(method)
                if method.is_operator_call:
                    operator_call_structs.add(struct.name)
            template_methods_by_struct[struct.name] = template_map

        for struct in structs_with_methods:
            for method in struct.methods:
                return_type = self._canonicalize_struct_scoped_type(
                    method.return_type, struct, field_structs_by_name
                )
                self._record_known_member_function_return_type(
                    method.free_name, return_type
                )

        instantiated_template_functions: Dict[str, str] = {}
        replacements: List[Tuple[int, int, str]] = []
        free_functions: List[str] = []
        for struct in structs_with_methods:
            data_only, lowered = self._render_lowered_struct(
                code,
                struct,
                instantiated_template_functions=instantiated_template_functions,
                template_methods_by_struct=template_methods_by_struct,
                methods_by_struct=methods_by_struct,
                operator_call_structs=operator_call_structs,
                structs_by_name=field_structs_by_name,
            )
            replacements.append((struct.span[0], struct.span[1], data_only))
            free_functions.extend(lowered)

        # Pointer-member-promoted structs: render the residual scalar struct,
        # emit each method as a free function taking `self` + the promoted pointer
        # parameters, and rewrite construction call sites (drop pointer arguments)
        # / method call sites (forward the pointer expressions).
        for struct in promoted_structs:
            plan = promotion_plans[struct.name]
            data_only, lowered = self._render_promoted_struct(code, struct, plan)
            replacements.append((struct.span[0], struct.span[1], data_only))
            free_functions.extend(lowered)
        replacements.extend(promoted_construction_replacements)
        if promoted_structs:
            replacements.extend(
                self._rewrite_promoted_call_sites(
                    code,
                    promoted_structs,
                    promotion_plans,
                    promoted_pointer_args,
                    all_struct_spans,
                )
            )

        # Rewrite call sites across the rest of the source (outside the structs
        # we are replacing, so receiver-less internal references are handled when
        # the method body is emitted, not here). Template-method call sites are
        # instantiated here; each unique (struct, method, bindings) instance adds
        # one concrete free function, deduplicated across call sites. The scan
        # skips EVERY method-bearing struct span (ordinary and promoted) so a
        # promoted struct's soon-to-be-replaced body is never rewritten in place.
        struct_spans = sorted(
            [struct.span for struct in structs_with_methods]
            + [struct.span for struct in promoted_structs]
        )
        # Member-access field resolution must see ALL struct/union spans (so a
        # field declaration is never mistaken for a local variable), and the
        # names of every struct/union type (so `Type var;` locals of a method-less
        # carrier are tracked).
        all_struct_names = {struct.name for struct in structs}
        call_replacements = self._rewrite_struct_member_call_sites(
            code,
            struct_names,
            methods_by_struct,
            template_methods_by_struct,
            operator_call_structs,
            struct_spans,
            field_structs_by_name,
            instantiated_template_functions,
            field_structs_by_name=field_structs_by_name,
            all_struct_spans=all_struct_spans,
            all_struct_names=all_struct_names,
        )
        replacements.extend(call_replacements)
        free_functions.extend(instantiated_template_functions.values())

        rewritten = self._apply_text_replacements(code, replacements)
        if free_functions:
            rewritten = rewritten.rstrip() + "\n\n" + "\n\n".join(free_functions)
            if not rewritten.endswith("\n"):
                rewritten += "\n"
        return self._canonicalize_qualified_struct_alias_templates(rewritten, structs)

    def _reject_unresolved_temporary_functor_calls(
        self,
        code: str,
        structs_by_name: Dict[str, _MetalStructDefinition],
        template_declaration_spans: List[Tuple[int, int]],
    ) -> None:
        ignored_spans = sorted(
            self._find_comment_and_literal_spans(code) + template_declaration_spans
        )
        cursor = 0
        while cursor < len(code):
            ignored = self._containing_span(cursor, ignored_spans)
            if ignored is not None:
                cursor = ignored[1]
                continue
            if not (code[cursor].isalpha() or code[cursor] == "_"):
                cursor += 1
                continue
            ident, consumed = self._read_identifier(code, cursor)
            ident_end = cursor + consumed
            if ident == "operator":
                cursor = ident_end
                continue
            constructor_start = ident_end
            while constructor_start < len(code) and code[constructor_start].isspace():
                constructor_start += 1
            temporary = self._temporary_functor_call(code, constructor_start)
            struct = structs_by_name.get(ident)
            if (
                temporary is not None
                and self._temporary_functor_is_candidate(code, cursor, temporary)
                and (
                    struct is None
                    or not (
                        struct.has_operator_call
                        or "operator()" in struct.member_prototype_names
                    )
                )
            ):
                self._raise_temporary_functor_error(
                    code,
                    ident,
                    cursor,
                    temporary,
                    reason="temporary-functor-type-unresolved",
                    detail=(
                        "the temporary type does not resolve to a concrete struct "
                        "with operator()"
                    ),
                )
            cursor = ident_end

    def _find_concrete_struct_definitions(
        self, code: str
    ) -> List[_MetalStructDefinition]:
        # Locate every concrete (non-template) `struct/class/union Name { ... };`
        # and split its body into data members and method definitions. Template
        # structs/classes (`template <...> struct ...`) are skipped wholesale:
        # those are handled by the materializer, and their (possibly template)
        # methods are out of scope for lowering. Unions are captured for their
        # data members only (they carry no methods we lower) so a `union var;`
        # local and its members are type-resolvable for call-argument inference.
        template_spans = self._find_template_declaration_spans(code)
        namespace_spans = self._find_namespace_spans(code)
        definitions: List[_MetalStructDefinition] = []
        for match in re.finditer(r"\b(?P<aggregate_kind>struct|class|union)\s+", code):
            start = match.start()
            if self._containing_span(start, template_spans) is not None:
                continue
            name_start = match.end()
            name, consumed = self._read_identifier(code, name_start)
            if not name or not consumed:
                continue
            after_name = name_start + consumed
            # Distinguish a definition (`struct Name { ... }`) from a forward
            # declaration / variable usage (`struct Name x;`). Skip anything that
            # is not immediately a `{` or a base-class clause `: ... {`.
            body_start = self._find_next_top_level_char(code, after_name, "{")
            semicolon = self._find_next_top_level_char(code, after_name, ";")
            if body_start is None or (semicolon is not None and semicolon < body_start):
                continue
            between = code[after_name:body_start]
            # Only a base-class clause may appear between the name and the body.
            stripped_between = between.strip()
            if stripped_between and not stripped_between.startswith(":"):
                continue
            body_end_after = self._find_matching_brace(code, body_start)
            if body_end_after is None:
                continue
            # The struct definition span includes the trailing semicolon so the
            # data-only replacement keeps the declaration well-formed.
            span_end = body_end_after
            trailing = code.find(";", body_end_after)
            if trailing != -1 and code[body_end_after:trailing].strip() == "":
                span_end = trailing + 1

            body = code[body_start + 1 : body_end_after - 1]
            parent = min(
                (
                    definition
                    for definition in definitions
                    if definition.span[0] < start < definition.span[1]
                ),
                key=lambda definition: definition.span[1] - definition.span[0],
                default=None,
            )
            namespace = self._namespace_at(namespace_spans, start)
            qualified_name = (
                f"{parent.qualified_name}::{name}"
                if parent is not None and parent.qualified_name
                else f"{namespace}::{name}" if namespace else name
            )
            member_prototype_names: Set[str] = set()
            (
                data_member_names,
                data_member_types,
                methods,
                template_methods,
                ordered_members,
                constructors,
                layout_supported,
            ) = self._split_struct_body(
                name,
                body,
                body_start + 1,
                member_prototype_names=member_prototype_names,
            )
            definitions.append(
                _MetalStructDefinition(
                    name=name,
                    span=(start, span_end),
                    body_span=(body_start + 1, body_end_after - 1),
                    data_member_names=data_member_names,
                    methods=methods,
                    has_operator_call=any(m.is_operator_call for m in methods)
                    or any(m.is_operator_call for m in template_methods),
                    qualified_name=qualified_name,
                    template_methods=template_methods,
                    data_member_types=data_member_types,
                    data_members=ordered_members,
                    constructors=constructors,
                    base_clause=stripped_between,
                    aggregate_kind=match.group("aggregate_kind"),
                    layout_supported=layout_supported,
                    member_prototype_names=member_prototype_names,
                    type_aliases=self._collect_struct_scope_type_aliases(body),
                    type_alias_templates=self._collect_struct_scope_type_alias_templates(
                        body
                    ),
                )
            )
        return definitions

    def _collect_struct_scope_type_aliases(self, body: str) -> Dict[str, str]:
        if "using" not in body and "typedef" not in body:
            return {}
        aliases: Dict[str, str] = {}
        ignored_spans = self._find_comment_and_literal_spans(body)
        lexical_scopes = self._find_lexical_brace_scopes(body)

        def is_struct_scope(position: int) -> bool:
            if self._containing_span(position, ignored_spans) is not None:
                return False
            return self._innermost_lexical_scope(
                lexical_scopes, position, len(body)
            ) == (0, len(body))

        for match in re.finditer(
            r"\busing\s+(?P<alias>[A-Za-z_]\w*)\s*=\s*" r"(?P<target>[^;{}]+?)\s*;",
            body,
            re.DOTALL,
        ):
            if not is_struct_scope(match.start()):
                continue
            target = self._normalize_template_argument_text(match.group("target"))
            if target:
                aliases[match.group("alias")] = target

        for match in re.finditer(
            r"\btypedef\s+(?P<target>[^;{}]+?)\s+" r"(?P<alias>[A-Za-z_]\w*)\s*;",
            body,
            re.DOTALL,
        ):
            if not is_struct_scope(match.start()):
                continue
            target = self._normalize_template_argument_text(match.group("target"))
            if target:
                aliases[match.group("alias")] = target
        return aliases

    def _collect_struct_scope_type_alias_templates(
        self, body: str
    ) -> Dict[str, _MetalTypeAliasTemplate]:
        if "template" not in body or "using" not in body:
            return {}
        templates: Dict[str, _MetalTypeAliasTemplate] = {}
        ignored_spans = self._find_comment_and_literal_spans(body)
        lexical_scopes = self._find_lexical_brace_scopes(body)

        for match in re.finditer(r"\btemplate\s*<", body):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            if self._innermost_lexical_scope(
                lexical_scopes, match.start(), len(body)
            ) != (0, len(body)):
                continue
            angle_start = body.find("<", match.start(), match.end())
            angle_end = self._find_matching_angle(body, angle_start)
            if angle_end is None:
                continue
            cursor = angle_end + 1
            while cursor < len(body) and body[cursor].isspace():
                cursor += 1
            alias_match = re.match(
                r"using\s+(?P<alias>[A-Za-z_]\w*)\s*=\s*",
                body[cursor:],
            )
            if alias_match is None:
                continue
            target_start = cursor + alias_match.end()
            target_end = body.find(";", target_start)
            if target_end == -1:
                continue
            parameters = self._parse_template_parameter_list(
                body[angle_start + 1 : angle_end]
            )
            target = self._normalize_template_argument_text(
                body[target_start:target_end]
            )
            if parameters and target:
                templates[alias_match.group("alias")] = _MetalTypeAliasTemplate(
                    parameters=parameters,
                    target=target,
                )
        return templates

    def _materialize_struct_scoped_alias_template(
        self,
        alias_template: _MetalTypeAliasTemplate,
        template_arguments: List[str],
        struct: _MetalStructDefinition,
    ) -> Optional[str]:
        parameters = alias_template.parameters
        if len(template_arguments) > len(parameters) or any(
            parameter.name is None or parameter.is_variadic for parameter in parameters
        ):
            return None

        bindings: Dict[str, str] = {}
        for index, parameter in enumerate(parameters):
            argument = (
                template_arguments[index]
                if index < len(template_arguments)
                else parameter.default
            )
            if argument is None or parameter.name is None:
                return None
            argument = self._normalize_template_argument_text(
                self._replace_identifiers(argument, bindings)
            )
            if not argument:
                return None
            bindings[parameter.name] = argument

        target = self._normalize_template_argument_text(
            self._replace_identifiers(alias_template.target, bindings)
        )
        parameter_names = {
            parameter.name for parameter in parameters if parameter.name is not None
        }
        if set(IDENTIFIER_RE.findall(target)) & parameter_names:
            return None

        constants = self._resolved_static_data_member_initializers(struct)
        if constants:
            target = self._resolve_qualified_static_constant_expression(
                target,
                {struct.name: struct},
            )
            target = self._substitute_template_argument_static_constants(
                target, constants
            )
        unresolved_static_constants = {
            member.name
            for member in struct.data_members
            if "static" in IDENTIFIER_RE.findall(member.type_text)
            and set(IDENTIFIER_RE.findall(member.type_text)).intersection(
                {"const", "constant", "constexpr"}
            )
            and member.name not in constants
        }
        if set(IDENTIFIER_RE.findall(target)) & unresolved_static_constants:
            return None

        target = re.sub(r"^typename\s+", "", target).strip()
        target = re.sub(r"::template\s+", "::", target)
        owner = struct.qualified_name or struct.name
        for alias in sorted(struct.type_alias_templates, key=len, reverse=True):
            target = re.sub(
                rf"(?<![A-Za-z0-9_:]){re.escape(alias)}(?=\s*<)",
                f"{owner}::{alias}",
                target,
            )
        return self._normalize_template_argument_text(target) or None

    def _canonicalize_struct_scoped_alias_templates(
        self,
        type_text: str,
        struct: _MetalStructDefinition,
        *,
        excluded_aliases: Optional[Set[str]] = None,
        qualified_only: bool = False,
        preserve_formatting: bool = False,
        allow_unqualified_owner: bool = True,
    ) -> str:
        templates = struct.type_alias_templates
        text = (
            str(type_text)
            if preserve_formatting
            else self._normalize_template_argument_text(type_text)
        )
        if not text or not templates:
            return text

        owner_names = {struct.qualified_name or struct.name}
        if allow_unqualified_owner:
            owner_names.add(struct.name)
        materialized_owner = self._materialized_struct_specializations.get(struct.name)
        if materialized_owner is not None:
            owner_names.add(materialized_owner[0])
        owner_pattern = "|".join(
            r"\s*::\s*".join(re.escape(part) for part in owner.split("::"))
            for owner in sorted(owner_names, key=len, reverse=True)
        )

        for _ in range(len(templates) + 2):
            ignored_spans = self._find_comment_and_literal_spans(text)
            replacements: List[Tuple[int, int, str]] = []
            for alias, alias_template in templates.items():
                patterns = [
                    re.compile(
                        rf"(?<![A-Za-z0-9_])(?:typename\s+)?"
                        rf"(?:::)?(?:{owner_pattern})"
                        rf"(?:\s*<[^<>{{}}()]*>)?\s*::\s*"
                        rf"(?:template\s+)?{re.escape(alias)}\s*<"
                    )
                ]
                if not qualified_only and alias not in (excluded_aliases or set()):
                    patterns.append(
                        re.compile(rf"(?<![A-Za-z0-9_:]){re.escape(alias)}\s*<")
                    )
                for pattern in patterns:
                    for match in pattern.finditer(text):
                        if (
                            self._containing_span(match.start(), ignored_spans)
                            is not None
                        ):
                            continue
                        angle_start = text.rfind("<", match.start(), match.end())
                        angle_end = self._find_matching_angle(text, angle_start)
                        if angle_end is None:
                            continue
                        replacement = self._materialize_struct_scoped_alias_template(
                            alias_template,
                            self._split_top_level_commas(
                                text[angle_start + 1 : angle_end]
                            ),
                            struct,
                        )
                        if replacement is None:
                            continue
                        span = (match.start(), angle_end + 1)
                        if any(
                            not (span[1] <= start or span[0] >= end)
                            for start, end, _replacement in replacements
                        ):
                            continue
                        replacements.append((span[0], span[1], replacement))
            if not replacements:
                break
            text = self._apply_text_replacements(text, replacements)
        if preserve_formatting:
            return text
        return self._normalize_template_argument_text(text)

    def _canonicalize_qualified_struct_alias_templates(
        self,
        source: str,
        structs: List[_MetalStructDefinition],
    ) -> str:
        rewritten = source
        owner_name_counts: Dict[str, int] = {}
        for struct in structs:
            owner_name_counts[struct.name] = owner_name_counts.get(struct.name, 0) + 1
        for struct in structs:
            if struct.type_alias_templates:
                rewritten = self._canonicalize_struct_scoped_alias_templates(
                    rewritten,
                    struct,
                    qualified_only=True,
                    preserve_formatting=True,
                    allow_unqualified_owner=owner_name_counts[struct.name] == 1,
                )
        return rewritten

    def _canonicalize_struct_scoped_type(
        self,
        type_text: str,
        struct: _MetalStructDefinition,
        structs_by_name: Optional[Dict[str, _MetalStructDefinition]],
        excluded_aliases: Optional[Set[str]] = None,
    ) -> str:
        text = self._normalize_template_argument_text(type_text)
        known_structs = structs_by_name or {struct.name: struct}
        text = self._canonicalize_struct_scoped_alias_templates(
            text,
            struct,
            excluded_aliases=excluded_aliases,
        )
        text = self._canonicalize_qualified_struct_alias_templates(
            text,
            list(known_structs.values()),
        )
        all_type_aliases = dict(struct.type_aliases)
        type_aliases = {
            alias: target
            for alias, target in all_type_aliases.items()
            if alias not in (excluded_aliases or set())
        }
        if not text or not all_type_aliases:
            return text
        text = re.sub(r"\btypename\s+", "", text)
        text = re.sub(r"::template\s+", "::", text)
        owner_names = {struct.name}
        materialized_owner = self._materialized_struct_specializations.get(struct.name)
        if materialized_owner is not None:
            owner_names.add(materialized_owner[0])
        owner_pattern = "|".join(
            re.escape(owner) for owner in sorted(owner_names, key=len, reverse=True)
        )
        for alias in sorted(all_type_aliases, key=len, reverse=True):
            qualified_alias = re.compile(
                rf"(?<![A-Za-z0-9_])"
                rf"(?:(?:[A-Za-z_]\w*)\s*::\s*)*"
                rf"(?:{owner_pattern})(?:\s*<[^<>{{}}()]*>)?\s*::\s*"
                rf"(?:template\s+)?{re.escape(alias)}\b"
            )
            qualified_target = re.sub(
                r"^typename\s+", "", all_type_aliases[alias]
            ).strip()
            qualified_target = re.sub(r"::template\s+", "::", qualified_target)
            text = qualified_alias.sub(qualified_target, text)

        def struct_alias_target(alias: str, seen: Set[str]) -> Optional[str]:
            if alias in seen:
                return None
            target = type_aliases.get(alias)
            if target is None:
                return None
            target = re.sub(r"^typename\s+", "", target).strip()
            if not IDENTIFIER_RE.fullmatch(target):
                return None
            if target in known_structs:
                return target
            return struct_alias_target(target, {*seen, alias})

        for _ in range(len(type_aliases) + 2):
            previous = text
            for alias in sorted(type_aliases, key=len, reverse=True):
                owner = struct_alias_target(alias, set())
                if owner is not None:
                    text = re.sub(
                        rf"(?<![A-Za-z0-9_:]){re.escape(alias)}(?=::)",
                        owner,
                        text,
                    )
                text = re.sub(
                    rf"(?<![A-Za-z0-9_:]){re.escape(alias)}(?=\s*<)",
                    f"{struct.name}::{alias}",
                    text,
                )
                target = type_aliases[alias]
                target = re.sub(r"^typename\s+", "", target).strip()
                target = re.sub(r"::template\s+", "::", target)
                text = re.sub(
                    rf"(?<![A-Za-z0-9_:]){re.escape(alias)}(?![A-Za-z0-9_<])",
                    target,
                    text,
                )
            if text == previous:
                break
        text = self._canonicalize_qualified_struct_alias_templates(
            text,
            list(known_structs.values()),
        )
        return self._normalize_template_argument_text(text)

    def _canonicalize_struct_scoped_parameters(
        self,
        parameters: str,
        struct: _MetalStructDefinition,
        structs_by_name: Optional[Dict[str, _MetalStructDefinition]],
    ) -> str:
        if not (set(IDENTIFIER_RE.findall(parameters)) & set(struct.type_aliases)):
            return parameters.strip()
        canonical: List[str] = []
        for parameter in self._split_top_level_commas(parameters):
            if not parameter.strip():
                continue
            declaration, default = self._split_top_level_assignment(parameter)
            attributes = self._metal_attributes(declaration)
            plain_declaration = self._strip_metal_attributes(declaration).strip()
            name = self._declared_data_member_name(plain_declaration)
            member = (
                self._parse_ordered_data_member(name, plain_declaration)
                if name is not None
                else None
            )
            if member is not None:
                type_text = self._canonicalize_struct_scoped_type(
                    member.type_text, struct, structs_by_name
                )
                declaration = f"{type_text} {member.name}{member.array_suffix}"
                if attributes:
                    declaration += " " + " ".join(attributes)
            else:
                declaration = self._canonicalize_struct_scoped_type(
                    declaration, struct, structs_by_name
                )
            if default is not None:
                declaration = f"{declaration} = {default}"
            canonical.append(declaration)
        return ", ".join(canonical)

    def _local_type_alias_shadow_scopes(
        self, body: str
    ) -> Dict[str, List[Tuple[int, int, int]]]:
        aliases: Dict[str, List[Tuple[int, int, int]]] = {}
        declarations: List[Tuple[int, str]] = []
        for match in re.finditer(r"\busing\s+([A-Za-z_]\w*)\s*=\s*[^;{}]+;", body):
            declarations.append((match.start(), match.group(1)))
        for match in re.finditer(r"\btypedef\s+[^;{}]+\s+([A-Za-z_]\w*)\s*;", body):
            declarations.append((match.start(), match.group(1)))
        if not declarations:
            return aliases
        ignored_spans = self._find_comment_and_literal_spans(body)
        lexical_scopes = self._find_lexical_brace_scopes(body)
        for declaration_position, alias in sorted(declarations):
            if self._containing_span(declaration_position, ignored_spans) is not None:
                continue
            scope_start, scope_end = self._innermost_lexical_scope(
                lexical_scopes, declaration_position, len(body)
            )
            aliases.setdefault(alias, []).append(
                (declaration_position, scope_start, scope_end)
            )
        return aliases

    def _collect_local_type_alias_bindings(
        self,
        code: str,
        owner_spans: List[Tuple[int, int]],
        *,
        skip_spans: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, List[_MetalTypeAliasBinding]]:
        raw_aliases: List[Tuple[int, str, str]] = []

        for match in re.finditer(
            r"\busing\s+(?P<alias>[A-Za-z_]\w*)\s*=\s*" r"(?P<target>[^;{}]+?)\s*;",
            code,
            re.DOTALL,
        ):
            if self._containing_span(match.start(), owner_spans) is None:
                continue
            raw_aliases.append(
                (match.start(), match.group("alias"), match.group("target"))
            )

        for match in re.finditer(
            r"\btypedef\s+(?P<target>[^;{}]+?)\s+" r"(?P<alias>[A-Za-z_]\w*)\s*;",
            code,
            re.DOTALL,
        ):
            if self._containing_span(match.start(), owner_spans) is None:
                continue
            raw_aliases.append(
                (match.start(), match.group("alias"), match.group("target"))
            )

        if not raw_aliases:
            return {}
        ignored_spans = sorted(
            [*(skip_spans or []), *self._find_comment_and_literal_spans(code)]
        )
        raw_aliases = [
            alias
            for alias in raw_aliases
            if self._containing_span(alias[0], ignored_spans) is None
        ]
        if not raw_aliases:
            return {}
        aliases: Dict[str, List[_MetalTypeAliasBinding]] = {}
        lexical_scopes = self._find_lexical_brace_scopes(code)
        for declaration_position, alias, target in sorted(raw_aliases):
            resolved_target = self._resolve_type_aliases_at(
                target,
                aliases,
                declaration_position,
            )
            scope_start, scope_end = self._innermost_lexical_scope(
                lexical_scopes, declaration_position, len(code)
            )
            aliases.setdefault(alias, []).append(
                _MetalTypeAliasBinding(
                    declaration_position=declaration_position,
                    scope_start=scope_start,
                    scope_end=scope_end,
                    target=resolved_target,
                )
            )
        return aliases

    def _resolve_type_aliases_at(
        self,
        type_text: str,
        aliases: Dict[str, List[_MetalTypeAliasBinding]],
        position: int,
    ) -> str:
        replacements = {
            alias: resolved
            for alias in aliases
            if (
                resolved := self._resolve_struct_type_alias_at(aliases, alias, position)
            )
            is not None
        }
        resolved = self._normalize_template_argument_text(type_text)
        for _ in range(len(replacements)):
            candidate = self._replace_identifiers(resolved, replacements)
            if candidate == resolved:
                break
            resolved = candidate
        return self._normalize_template_argument_text(resolved)

    def _canonicalize_type_aliases_at(
        self,
        type_text: str,
        aliases: Dict[str, List[_MetalTypeAliasBinding]],
        position: int,
        *,
        excluded_aliases: Optional[Set[str]] = None,
    ) -> Optional[str]:
        """Resolve aliases visible at ``position`` without crossing a cycle.

        Alias targets are resolved at their own declaration positions. This
        preserves lexical shadowing and prevents a later declaration from
        retroactively satisfying an unresolved or forward alias target.
        """
        normalized = self._normalize_template_argument_text(type_text)
        if not normalized:
            return None

        cache: Dict[Tuple[str, int, int, int], Optional[str]] = {}

        def canonicalize(
            text: str,
            context_position: int,
            resolving: Set[Tuple[str, int, int, int]],
        ) -> Optional[str]:
            candidate = self._normalize_template_argument_text(text)
            if not candidate:
                return None
            replacements: Dict[str, str] = {}
            for identifier in dict.fromkeys(IDENTIFIER_RE.findall(candidate)):
                if identifier in (excluded_aliases or set()):
                    continue
                binding = self._resolve_type_alias_binding_at(
                    aliases,
                    identifier,
                    context_position,
                )
                if binding is None:
                    if any(
                        future.declaration_position > context_position
                        and future.scope_start <= context_position < future.scope_end
                        for future in aliases.get(identifier, [])
                    ):
                        return None
                    continue
                key = (
                    identifier,
                    binding.declaration_position,
                    binding.scope_start,
                    binding.scope_end,
                )
                if key in resolving:
                    return None
                if key not in cache:
                    cache[key] = canonicalize(
                        binding.target,
                        binding.declaration_position,
                        {*resolving, key},
                    )
                resolved = cache[key]
                if resolved is None:
                    return None
                replacements[identifier] = resolved
            return self._normalize_template_argument_text(
                self._replace_identifiers(candidate, replacements)
            )

        return canonicalize(normalized, position, set())

    def _collect_local_integral_constant_bindings(
        self,
        code: str,
        owner_spans: List[Tuple[int, int]],
        type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]] = None,
    ) -> Dict[str, List[_MetalIntegralConstantBinding]]:
        raw_declarations: List[Tuple[int, str, _MetalDataMember]] = []
        seen_declarations: Set[Tuple[int, str]] = set()
        for owner_start, owner_end in sorted(owner_spans):
            owner_text = code[owner_start:owner_end]
            search_start = 0
            for statement in self._iter_simple_declarations(owner_text):
                relative_start = owner_text.find(statement, search_start)
                if relative_start == -1:
                    continue
                search_start = relative_start + len(statement)
                stripped = statement.strip()
                if not stripped:
                    continue
                declaration_position = (
                    owner_start
                    + relative_start
                    + len(statement)
                    - len(statement.lstrip())
                )
                name = self._declared_local_name(stripped)
                if name is None:
                    continue
                member = self._parse_ordered_data_member(name, stripped)
                if member is None:
                    continue
                declaration_key = (declaration_position, name)
                if declaration_key in seen_declarations:
                    continue
                seen_declarations.add(declaration_key)
                raw_declarations.append((declaration_position, name, member))

        if not raw_declarations:
            return {}
        bindings: Dict[str, List[_MetalIntegralConstantBinding]] = {}
        lexical_scopes = self._find_lexical_brace_scopes(code)
        ignored_type_tokens = {
            "const",
            "constant",
            "constexpr",
            "device",
            "register",
            "static",
            "thread",
            "threadgroup",
            "volatile",
        }
        for declaration_position, name, member in sorted(raw_declarations):
            scope_start, scope_end = self._innermost_lexical_scope(
                lexical_scopes, declaration_position, len(code)
            )
            value: Optional[str] = None
            type_tokens = IDENTIFIER_RE.findall(member.type_text)
            value_type_tokens = [
                token for token in type_tokens if token not in ignored_type_tokens
            ]
            if (
                member.default is not None
                and "constexpr" in type_tokens
                and len(value_type_tokens) == 1
                and value_type_tokens[0] in self._METAL_INTEGRAL_SCALAR_TYPES
                and not member.array_suffix
                and "*" not in member.type_text
                and "&" not in member.type_text
            ):
                visible_constants = self._local_integral_constants_at(
                    bindings, declaration_position
                )
                # The declared name is already in scope in its own initializer;
                # leaving it unresolved prevents an outer binding from making a
                # self-reference look like a valid constant expression.
                visible_constants.pop(name, None)
                initializer = self._substitute_template_argument_static_constants(
                    member.default,
                    visible_constants,
                )
                if type_aliases is not None:
                    initializer = self._replace_concrete_sizeof_expressions(
                        initializer,
                        lambda type_text: self._resolve_type_aliases_at(
                            type_text,
                            type_aliases,
                            declaration_position,
                        ),
                    )
                else:
                    initializer = self._replace_concrete_sizeof_expressions(initializer)
                value = self._proven_integral_constant_value(
                    value_type_tokens[0], initializer
                )
            bindings.setdefault(name, []).append(
                _MetalIntegralConstantBinding(
                    declaration_position=declaration_position,
                    scope_start=scope_start,
                    scope_end=scope_end,
                    value=value,
                )
            )
        return bindings

    @staticmethod
    def _local_integral_constant_bindings_at(
        bindings: Dict[str, List[_MetalIntegralConstantBinding]],
        position: int,
    ) -> Dict[str, _MetalIntegralConstantBinding]:
        visible: Dict[str, _MetalIntegralConstantBinding] = {}
        for name, candidates in bindings.items():
            best: Optional[_MetalIntegralConstantBinding] = None
            for binding in candidates:
                if binding.declaration_position > position:
                    break
                if not (binding.scope_start <= position < binding.scope_end):
                    continue
                if (
                    best is None
                    or binding.declaration_position > best.declaration_position
                ):
                    best = binding
            if best is not None:
                visible[name] = best
        return visible

    @classmethod
    def _local_integral_constants_at(
        cls,
        bindings: Dict[str, List[_MetalIntegralConstantBinding]],
        position: int,
    ) -> Dict[str, str]:
        return {
            name: binding.value
            for name, binding in cls._local_integral_constant_bindings_at(
                bindings, position
            ).items()
            if binding.value is not None
        }

    def _strip_function_parameter_defaults(self, parameters: str) -> str:
        declarations: List[str] = []
        for parameter in self._split_top_level_commas(parameters):
            if not parameter.strip():
                continue
            declaration, _default = self._split_top_level_assignment(parameter)
            declarations.append(declaration.strip())
        return ", ".join(declarations)

    def _canonicalize_struct_scoped_local_declarations(
        self,
        body: str,
        struct: _MetalStructDefinition,
        structs_by_name: Optional[Dict[str, _MetalStructDefinition]],
    ) -> str:
        replacements: List[Tuple[int, int, str]] = []
        local_aliases = self._local_type_alias_shadow_scopes(body)
        search_start = 0
        for statement in self._iter_simple_declarations(body):
            statement_start = body.find(statement, search_start)
            if statement_start == -1:
                continue
            search_start = statement_start + len(statement)
            if self._declaration_is_type_alias(statement):
                continue
            name = self._declared_local_name(statement)
            if name is None:
                continue

            leading = len(statement) - len(statement.lstrip())
            declarator = self._strip_top_level_default_value(statement[leading:])
            paren = self._function_parameter_start(declarator)
            if paren is not None:
                declarator = declarator[:paren].rstrip()
            while declarator.endswith("]"):
                open_bracket = declarator.rfind("[")
                if open_bracket == -1:
                    break
                declarator = declarator[:open_bracket].rstrip()
            name_match = re.search(rf"\b{re.escape(name)}\s*$", declarator)
            if name_match is None:
                continue

            type_start = statement_start + leading
            type_end = type_start + name_match.start()
            type_text = body[type_start:type_end]
            shadowed_aliases = self._shadowed_type_aliases_at(local_aliases, type_start)
            canonical = self._canonicalize_struct_scoped_type(
                type_text,
                struct,
                structs_by_name,
                excluded_aliases=shadowed_aliases,
            )
            if canonical and canonical != type_text.strip():
                replacements.append((type_start, type_end, f"{canonical} "))
        return self._apply_text_replacements(body, replacements)

    @staticmethod
    def _shadowed_type_aliases_at(
        local_aliases: Dict[str, List[Tuple[int, int, int]]], position: int
    ) -> Set[str]:
        return {
            alias
            for alias, bindings in local_aliases.items()
            if any(
                declaration_position <= position and scope_start <= position < scope_end
                for declaration_position, scope_start, scope_end in bindings
            )
        }

    def _canonicalize_struct_scoped_cast_types(
        self,
        body: str,
        struct: _MetalStructDefinition,
        structs_by_name: Optional[Dict[str, _MetalStructDefinition]],
    ) -> str:
        alias_names = set(struct.type_aliases)
        if not alias_names or not (set(IDENTIFIER_RE.findall(body)) & alias_names):
            return body

        ignored_spans = self._find_comment_and_literal_spans(body)
        local_aliases = self._local_type_alias_shadow_scopes(body)
        replacements: List[Tuple[int, int, str]] = []

        def canonical_type(type_start: int, type_end: int) -> Optional[str]:
            type_text = body[type_start:type_end]
            if not (set(IDENTIFIER_RE.findall(type_text)) & alias_names):
                return None
            canonical = self._canonicalize_struct_scoped_type(
                type_text,
                struct,
                structs_by_name,
                excluded_aliases=self._shadowed_type_aliases_at(
                    local_aliases, type_start
                ),
            )
            if not canonical or canonical == type_text.strip():
                return None
            return canonical

        def add_replacement(type_start: int, type_end: int) -> None:
            canonical = canonical_type(type_start, type_end)
            if canonical is not None:
                replacements.append((type_start, type_end, canonical))

        named_cast_pattern = re.compile(
            r"\b(?:static_cast|reinterpret_cast|const_cast)\s*<"
        )
        for match in named_cast_pattern.finditer(body):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            angle_start = body.find("<", match.start(), match.end())
            angle_end = self._find_matching_angle(body, angle_start)
            if angle_end is not None:
                add_replacement(angle_start + 1, angle_end)

        # Parenthesized expressions overlap C-style cast syntax. Only rewrite a
        # span that resolves to a pointer/reference type and has a concrete cast
        # operand; ambiguous scalar groups remain unchanged.
        allowed_c_style_identifiers = {
            "atomic",
            "const",
            "constant",
            "device",
            "packed",
            "restrict",
            "thread",
            "threadgroup",
            "threadgroup_imageblock",
            "typename",
            "volatile",
            "read",
            "write",
            "read_write",
            "__restrict",
            "__restrict__",
        }
        for alias_target in struct.type_aliases.values():
            allowed_c_style_identifiers.update(IDENTIFIER_RE.findall(alias_target))
        alias_pattern = "|".join(
            re.escape(alias) for alias in sorted(alias_names, key=len, reverse=True)
        )
        c_style_cast_pattern = re.compile(
            rf"\((?P<type>[^(){{}};]*\b(?:{alias_pattern})\b[^(){{}};]*)\)"
        )
        for match in c_style_cast_pattern.finditer(body):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            type_text = match.group("type")
            identifiers = set(IDENTIFIER_RE.findall(type_text))
            if not (identifiers & alias_names):
                continue
            if re.search(r"[=;{}?!|^/%+]", type_text):
                continue
            canonical = canonical_type(match.start("type"), match.end("type"))
            if canonical is None or not ({"*", "&"} & set(canonical)):
                continue
            if set(IDENTIFIER_RE.findall(canonical)) - allowed_c_style_identifiers:
                continue
            operand_start = match.end()
            while operand_start < len(body) and body[operand_start].isspace():
                operand_start += 1
            if operand_start >= len(body) or not (
                body[operand_start].isalnum()
                or body[operand_start] in {"_", "(", "{", "*", "&"}
            ):
                continue
            replacements.append((match.start("type"), match.end("type"), canonical))

        return self._apply_text_replacements(body, replacements)

    def _split_struct_body(
        self,
        struct_name: str,
        body: str,
        body_offset: int,
        member_prototype_names: Optional[Set[str]] = None,
    ) -> Tuple[
        Set[str],
        Dict[str, str],
        List[_MetalStructMethod],
        List[_MetalStructMethod],
        List[_MetalDataMember],
        List[_MetalConstructor],
        bool,
    ]:
        # Walk a struct body separating DATA members from METHOD definitions.
        # A method is a declarator followed by `(params)` then `{...}`; everything
        # else terminated by `;` (or an access-specifier label) is data. Template
        # member functions (`template <...> ...`) are collected SEPARATELY: they
        # are instantiated on demand from their call sites rather than emitted up
        # front (a template method has no single concrete signature).
        data_member_names: Set[str] = set()
        data_member_types: Dict[str, str] = {}
        methods: List[_MetalStructMethod] = []
        template_methods: List[_MetalStructMethod] = []
        # Ordered data members and constructors are needed by the pointer-member
        # scalar-replacement path (declaration order and ctor mapping matter).
        ordered_members: List[_MetalDataMember] = []
        constructors: List[_MetalConstructor] = []
        layout_supported = True
        if member_prototype_names is None:
            member_prototype_names = set()
        i = 0
        n = len(body)
        while i < n:
            ch = body[i]
            if ch.isspace():
                i += 1
                continue
            if body.startswith("//", i):
                end = body.find("\n", i)
                i = n if end == -1 else end + 1
                continue
            if body.startswith("/*", i):
                end = body.find("*/", i + 2)
                i = n if end == -1 else end + 2
                continue
            # Access specifiers (public:/private:/protected:) are labels, not
            # members; consume up to and including the colon.
            label = re.match(r"(public|private|protected)\s*:", body[i:])
            if label:
                i += label.end()
                continue

            # A `template <...>` member function: capture its definition so a
            # CALLED instance can be instantiated from its call-site argument
            # types. A bare prototype (no body) has nothing to lower and is
            # skipped; only definitions with a body are recorded.
            if re.match(r"template\s*<", body[i:]):
                angle_start = body.find("<", i)
                angle_end = self._find_matching_template_param_angle(body, angle_start)
                if angle_end is None:
                    layout_supported = False
                    break
                method_body_start = self._find_next_top_level_char(
                    body, angle_end + 1, "{"
                )
                semicolon = self._find_next_top_level_char(body, angle_end + 1, ";")
                if method_body_start is not None and (
                    semicolon is None or method_body_start < semicolon
                ):
                    method_body_end = self._find_matching_brace(body, method_body_start)
                    if method_body_end is not None:
                        template_method = self._parse_struct_template_method(
                            struct_name,
                            body,
                            i,
                            angle_start,
                            angle_end,
                            method_body_start,
                            method_body_end,
                            body_offset,
                        )
                        if template_method is not None:
                            if template_method.is_template:
                                template_methods.append(template_method)
                            else:
                                methods.append(template_method)
                        else:
                            constructor = self._parse_struct_constructor(
                                struct_name,
                                body,
                                angle_end + 1,
                                method_body_start,
                                method_body_end,
                                body_offset,
                            )
                            if constructor is not None:
                                parameter_text = body[angle_start + 1 : angle_end]
                                constructor.template_parameters = (
                                    self._template_parameter_names(parameter_text)
                                )
                                constructor.template_parameter_defaults = (
                                    self._template_parameter_defaults(parameter_text)
                                )
                                constructor.span = (
                                    body_offset + i,
                                    body_offset + method_body_end,
                                )
                                constructors.append(constructor)
                    if method_body_end is None:
                        layout_supported = False
                        i = n
                    else:
                        i = method_body_end
                elif semicolon is not None:
                    prototype_name = self._struct_member_prototype_name(
                        body[i:semicolon]
                    )
                    if prototype_name is not None:
                        member_prototype_names.add(prototype_name)
                    i = semicolon + 1
                else:
                    layout_supported = False
                    break
                continue

            # Find the next statement boundary: either a `{` (method body or
            # in-struct initializer braces) or a `;` (data member / declaration).
            brace = self._find_next_top_level_char(body, i, "{")
            semicolon = self._find_next_top_level_char(body, i, ";")
            if brace is not None and (semicolon is None or brace < semicolon):
                method = self._parse_struct_method(struct_name, body, i, brace)
                brace_end = self._find_matching_brace(body, brace)
                if brace_end is None:
                    layout_supported = False
                    break
                if method is not None:
                    method.span = (body_offset + i, body_offset + brace_end)
                    methods.append(method)
                    i = brace_end
                    # An optional trailing `;` after a method body.
                    j = i
                    while j < n and body[j].isspace():
                        j += 1
                    if j < n and body[j] == ";":
                        i = j + 1
                    continue
                # `_parse_struct_method` declined this brace-delimited construct.
                # It is either an OUT-OF-SCOPE method definition (constructor,
                # destructor, conversion/comparison operator, ...) or a data
                # member with a brace initializer (`size_t n_{0};`). A
                # method-shaped construct has a top-level parameter list before
                # the body and is left in place untouched so the parser's
                # existing struct-method skipping drops it; a brace-initialized
                # member is recorded as a data member.
                if self._brace_construct_is_method_definition(body[i:brace]):
                    # A constructor is captured (for pointer-member scalar
                    # replacement); other declined constructs (destructor,
                    # conversion/comparison operators) are simply left in place.
                    constructor = self._parse_struct_constructor(
                        struct_name, body, i, brace, brace_end, body_offset
                    )
                    if constructor is not None:
                        constructors.append(constructor)
                    i = brace_end
                    # Consume an optional trailing `;`.
                    j = i
                    while j < n and body[j].isspace():
                        j += 1
                    if j < n and body[j] == ";":
                        i = j + 1
                    continue
                decl_semicolon = self._find_next_top_level_char(body, brace_end, ";")
                if decl_semicolon is None:
                    layout_supported = False
                    i = brace_end
                    continue
                declaration = body[i:decl_semicolon]
                declaration_prefix = body[i:brace].strip()
                trailing_declarator = body[brace_end:decl_semicolon].strip()
                nested_type = re.match(
                    r"^(?:struct|class|union|enum)\s+[A-Za-z_]\w*\b",
                    declaration_prefix,
                )
                if nested_type is None or trailing_declarator:
                    layout_supported &= self._record_data_member(
                        declaration,
                        data_member_names,
                        data_member_types,
                        ordered_members,
                    )
                i = decl_semicolon + 1
                continue
            if semicolon is None:
                layout_supported = False
                break
            # A declaration terminated by `;`. It may still be a method
            # PROTOTYPE (declarator + params + ;) with no body — those have no
            # definition to lower, so record any data member name otherwise.
            declaration = body[i:semicolon]
            if self._declaration_is_method_prototype(declaration):
                prototype_name = self._struct_member_prototype_name(declaration)
                if prototype_name is not None:
                    member_prototype_names.add(prototype_name)
                if re.search(
                    r"\b(?:alignas|__attribute__|__declspec)\s*\(", declaration
                ):
                    layout_supported = False
                if re.search(r"\(\s*[*&]", declaration):
                    layout_supported = False
            elif not self._declaration_is_type_alias(declaration):
                type_declaration = re.fullmatch(
                    r"\s*(?:struct|class|union|enum)\s+[A-Za-z_]\w*\s*",
                    declaration,
                )
                if type_declaration is None:
                    layout_supported &= self._record_data_member(
                        declaration,
                        data_member_names,
                        data_member_types,
                        ordered_members,
                    )
            i = semicolon + 1
        return (
            data_member_names,
            data_member_types,
            methods,
            template_methods,
            ordered_members,
            constructors,
            layout_supported,
        )

    def _record_data_member(
        self,
        declaration: str,
        data_member_names: Set[str],
        data_member_types: Dict[str, str],
        ordered_members: List[_MetalDataMember],
    ) -> bool:
        # Record a data member into the name set, the (normalized) type map, and
        # the ordered list with its FULL declared type text / default / array
        # suffix (used by the pointer-member scalar-replacement path).
        name = self._declared_data_member_name(declaration)
        if not name:
            return False
        data_member_names.add(name)
        self._record_data_member_type(data_member_types, name, declaration)
        member = self._parse_ordered_data_member(name, declaration)
        if member is not None:
            ordered_members.append(member)
            return True
        return False

    def _parse_ordered_data_member(
        self, name: str, declaration: str
    ) -> Optional[_MetalDataMember]:
        # Split `[qualifiers] Type name [array] [= default]` into its full type
        # text (preserving address space / cv / pointer), trailing array suffix,
        # and default initializer. Returns None when the type cannot be isolated.
        lhs, default = self._split_top_level_assignment(declaration)
        lhs = lhs.strip()
        # Peel trailing array extents (`data[N][M]`) off the declarator.
        array_suffix = ""
        while lhs.endswith("]"):
            open_bracket = lhs.rfind("[")
            if open_bracket == -1:
                break
            array_suffix = lhs[open_bracket:] + array_suffix
            lhs = lhs[:open_bracket].rstrip()
        # Remove the trailing member name to leave the (full) type text.
        type_text = re.sub(rf"\b{re.escape(name)}\s*$", "", lhs).strip()
        type_text = re.sub(r"\s+", " ", type_text)
        if not type_text:
            return None
        return _MetalDataMember(
            name=name,
            type_text=type_text,
            default=default,
            array_suffix=array_suffix,
        )

    def _split_top_level_assignment(self, text: str) -> Tuple[str, Optional[str]]:
        # Split at the first top-level `=` that is a plain assignment (not part of
        # `==`/`<=`/`>=`/`!=` or a compound operator), returning (lhs, rhs) with
        # rhs None when there is no such assignment. Used to peel a data member's
        # default initializer from its declarator.
        paren = bracket = brace = angle = 0
        i = 0
        while i < len(text):
            ch = text[i]
            if ch in "\"'":
                _literal, consumed = self._read_string(text, i)
                i += consumed
                continue
            if text.startswith("//", i):
                break
            if text.startswith("/*", i):
                end = text.find("*/", i + 2)
                i = len(text) if end == -1 else end + 2
                continue
            if ch == "(":
                paren += 1
            elif ch == ")":
                paren = max(0, paren - 1)
            elif ch == "[":
                bracket += 1
            elif ch == "]":
                bracket = max(0, bracket - 1)
            elif ch == "{":
                brace += 1
            elif ch == "}":
                brace = max(0, brace - 1)
            elif ch == "<":
                angle += 1
            elif ch == ">":
                angle = max(0, angle - 1)
            elif ch == "=" and paren == bracket == brace == angle == 0:
                nxt = text[i + 1] if i + 1 < len(text) else ""
                prv = text[i - 1] if i > 0 else ""
                if nxt != "=" and prv not in "=<>!+-*/%&|^~":
                    return text[:i].strip(), text[i + 1 :].strip()
            i += 1
        return text.strip(), None

    def _parse_struct_constructor(
        self,
        struct_name: str,
        body: str,
        decl_start: int,
        brace: int,
        brace_end: int,
        body_offset: int = 0,
    ) -> Optional[_MetalConstructor]:
        # Parse a constructor `[macros] Name(params) : init_list { body }`.
        # `Name` is a bare identifier with NO return type (that is what makes it a
        # constructor rather than a method); for a materialized template struct it
        # is the ORIGINAL template name, not the renamed struct. Destructors and
        # operators are rejected. Returns None for any non-constructor construct.
        header = body[decl_start:brace]
        paren_start = self._function_parameter_start(header)
        if paren_start is None:
            return None
        paren_end = self._find_matching_delimiter(header, paren_start, "(", ")")
        if paren_end is None:
            return None
        before = header[:paren_start].rstrip()
        is_constexpr = bool(re.search(r"\b(?:constexpr|C10_METAL_CONSTEXPR)\b", before))
        name_region = self._strip_function_qualifier_macros(before)
        name_region = re.sub(r"\b(?:constexpr|explicit|inline)\b", " ", name_region)
        name_region = re.sub(r"\s+", " ", name_region).strip()
        # A genuine constructor declarator is a single identifier: no return type
        # tokens, no `~` destructor, no `operator`.
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name_region or ""):
            return None
        if name_region == "operator":
            return None
        params_text = header[paren_start + 1 : paren_end]
        param_names = self._parameter_identifier_names(params_text)
        param_types = self._parameter_declared_types(params_text)
        # The member initializer list sits between `)` and `{`, introduced by `:`.
        between = header[paren_end + 1 :].strip()
        init_text = ""
        init_map: Dict[str, str] = {}
        colon = self._find_next_top_level_char(between, 0, ":")
        receiver_text = between if colon is None else between[:colon]
        if colon is not None:
            init_text = between[colon + 1 :].strip()
            init_map = self._parse_constructor_init_list(init_text)
        receiver_address_spaces = self._receiver_address_spaces(receiver_text)
        constructor_body = body[brace + 1 : brace_end - 1]
        # `prefix` is everything up to and including `(` (leading macros + the
        # constructor name), so the pointer-member promotion path can rebuild the
        # constructor header after dropping pointer parameters.
        prefix = header[: paren_start + 1]
        return _MetalConstructor(
            param_names=param_names,
            init_map=init_map,
            body=constructor_body,
            span=(body_offset + decl_start, body_offset + brace_end),
            prefix=prefix,
            params_text=params_text,
            init_text=init_text,
            param_types=param_types,
            is_constexpr=is_constexpr,
            receiver_address_spaces=receiver_address_spaces,
        )

    def _parameter_declared_types(self, params_text: str) -> List[str]:
        # Return the FULL declared type text of each parameter (in order),
        # preserving address space / cv / pointer (e.g. ``const device float2*``)
        # and dropping the parameter name / default. Mirrors
        # ``_parameter_identifier_names`` but keeps the type rather than the name,
        # so a pointer parameter can be recognized by its ``*``.
        types: List[str] = []
        for parameter in self._split_top_level_commas(params_text):
            parameter = parameter.strip()
            if not parameter:
                continue
            # Drop any default value.
            lhs, _default = self._split_top_level_assignment(parameter)
            lhs = lhs.strip()
            # Peel trailing array extents that belong to the declarator.
            while lhs.endswith("]"):
                open_bracket = lhs.rfind("[")
                if open_bracket == -1:
                    break
                lhs = lhs[:open_bracket].rstrip()
            name_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", lhs)
            if name_match is None:
                types.append(re.sub(r"\s+", " ", lhs).strip())
                continue
            type_text = lhs[: name_match.start()].strip()
            # A bare-type parameter (`threadgroup float*`) leaves nothing after
            # stripping the "name": keep the whole text as the type instead.
            if not type_text:
                type_text = lhs
            types.append(re.sub(r"\s+", " ", type_text).strip())
        return types

    def _parse_constructor_init_list(self, init_text: str) -> Dict[str, str]:
        # Parse `member(expr), member2{expr2}, ...` into {member: expr}. Each
        # entry initializes one member; the expression usually names a constructor
        # parameter (`in(in_)`) but any expression is captured verbatim.
        init_map: Dict[str, str] = {}
        for item in self._split_top_level_commas(init_text):
            item = item.strip()
            if not item:
                continue
            match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*([\(\{])", item)
            if match is None:
                continue
            member = match.group(1)
            open_char = match.group(2)
            close_char = ")" if open_char == "(" else "}"
            open_index = match.end() - 1
            close_index = self._find_matching_delimiter(
                item, open_index, open_char, close_char
            )
            if close_index is None:
                continue
            init_map[member] = item[open_index + 1 : close_index].strip()
        return init_map

    def _record_data_member_type(
        self, data_member_types: Dict[str, str], name: str, declaration: str
    ) -> None:
        # Best-effort capture of a data member's element type from its
        # declaration text (`float bias`, `T data[N]`, `device float* ptr`,
        # `bool4 b`). The type is the declaration with the trailing declarator
        # (name + any array extents / default value) removed; pointer members
        # keep a `*` marker so a `self.ptr[i]` access can still resolve. A type
        # we cannot isolate is simply omitted (left un-inferable).
        element_type = self._data_member_element_type(declaration)
        if element_type:
            data_member_types[name] = element_type

    def _data_member_element_type(self, declaration: str) -> Optional[str]:
        text = self._strip_top_level_default_value(declaration).strip()
        if not text:
            return None
        # Drop trailing array extents so a `T data[N]` member yields element T.
        while text.endswith("]"):
            open_bracket = text.rfind("[")
            if open_bracket == -1:
                break
            text = text[:open_bracket].rstrip()
        # Strip the trailing member name to leave the type text.
        type_text = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\s*$", "", text).strip()
        normalized = self._normalize_inferred_type(type_text)
        # A pointer member (`device T* ptr`) collapses to the pointee marked with
        # a single trailing `*` so a subscript access can be element-typed.
        if not normalized:
            return None
        return normalized

    def _parse_struct_template_method(
        self,
        struct_name: str,
        body: str,
        decl_start: int,
        angle_start: int,
        angle_end: int,
        method_body_start: int,
        method_body_end: int,
        body_offset: int,
    ) -> Optional[_MetalStructMethod]:
        # Parse a `template <...> RetType name(params) { body }` member function.
        # The declarator AFTER the template-parameter list is parsed with the
        # same machinery as a non-template method; the template parameter names
        # are recorded so the method can be instantiated from a call site.
        parameter_text = body[angle_start + 1 : angle_end]
        template_parameters = self._template_parameter_names(parameter_text)
        if not template_parameters:
            method = self._parse_struct_method(
                struct_name, body, angle_end + 1, method_body_start
            )
            if method is None:
                return None
            method.span = (body_offset + decl_start, body_offset + method_body_end)
            return method
        method = self._parse_struct_method(
            struct_name, body, angle_end + 1, method_body_start
        )
        if method is None:
            return None
        method.template_parameters = template_parameters
        method.template_parameter_types = {
            parameter.name: parameter.declared_type
            for parameter in self._parse_template_parameter_list(parameter_text)
            if parameter.name is not None
            and not parameter.is_type_parameter
            and parameter.declared_type is not None
        }
        method.variadic_template_parameters = self._variadic_template_parameter_names(
            parameter_text
        )
        method.template_parameter_defaults = self._template_parameter_defaults(
            parameter_text
        )
        method.template_constraints = self._template_parameter_constraints(
            parameter_text
        )
        # A SECOND SFINAE layer can hide on the RETURN TYPE, e.g. Sum/Min/Max:
        #   metal::enable_if_t<metal::is_integral_v<T>, T> simd_reduce_impl(T val)
        # Unwrap it so the emitted free function returns the real value type while
        # the wrapped constraint joins the overload-selection set.
        return_constraint, unwrapped = self._split_return_type_sfinae(
            method.return_type
        )
        if return_constraint is not None:
            method.return_type = unwrapped
            method.return_type_constraint = return_constraint
            method.template_constraints = [
                *method.template_constraints,
                return_constraint,
            ]
        # The span covers the whole `template <...> ... { ... }` definition so the
        # data-only struct removes it cleanly.
        method.span = (body_offset + decl_start, body_offset + method_body_end)
        return method

    def _split_return_type_sfinae(self, return_type: str) -> Tuple[Optional[str], str]:
        # Detect a return-type SFINAE wrapper `[metal::]enable_if_t<COND, TYPE>`
        # (the second SFINAE layer used by Sum/Min/Max's `simd_reduce_impl`). On a
        # match return (COND, TYPE) — the enabling condition and the unwrapped
        # value type; otherwise (None, return_type) unchanged. Only a return type
        # that is EXACTLY an `enable_if_t<...>` template-id is unwrapped, so an
        # ordinary `T` / `float4` return is left intact (never guess).
        text = return_type.strip()
        match = re.match(r"^(?:typename\s+)?(?:metal\s*::\s*)?enable_if_t\s*<", text)
        if match is None:
            return None, return_type
        angle_start = text.find("<", match.end() - 1)
        # Use the SFINAE-aware angle/split so a condition containing a comparison
        # (`sizeof(T) < 8`) balances correctly.
        angle_end = self._find_matching_template_param_angle(text, angle_start)
        if angle_end is None or text[angle_end + 1 :].strip():
            # Trailing tokens after the angle (e.g. a pointer/qualifier) are not a
            # simple wrapper; leave it for clean-fail rather than mis-parse.
            return None, return_type
        arguments = self._split_template_parameter_list(
            text[angle_start + 1 : angle_end]
        )
        if len(arguments) != 2:
            return None, return_type
        condition = arguments[0].strip()
        value_type = arguments[1].strip()
        if not condition or not value_type:
            return None, return_type
        return condition, value_type

    # ------------------------------------------------------------------ #
    # SFINAE constraint evaluation for overload selection.               #
    # ------------------------------------------------------------------ #
    # Only the two constraint families MLX `reduce` needs are understood:
    # `sizeof(T) <cmp> N` and `[!]is_integral_v<T>`. Any other constraint shape
    # is a clean-fail (the overload set is left unresolved so the existing
    # diagnostic fires) — the contract is to NEVER guess or mis-select.

    # Byte sizes of the scalar element types (vectors scale by component count).
    _METAL_SCALAR_TYPE_SIZES: Dict[str, int] = {
        "bool": 1,
        "char": 1,
        "uchar": 1,
        "int8_t": 1,
        "uint8_t": 1,
        "short": 2,
        "ushort": 2,
        "int16_t": 2,
        "uint16_t": 2,
        "half": 2,
        "float16_t": 2,
        "bfloat16_t": 2,
        "int": 4,
        "uint": 4,
        "int32_t": 4,
        "uint32_t": 4,
        "float": 4,
        "long": 8,
        "ulong": 8,
        "int64_t": 8,
        "uint64_t": 8,
        "double": 8,
        "size_t": 8,
    }
    # Integral scalar element types for `is_integral_v<T>`.
    _METAL_INTEGRAL_SCALAR_TYPES: Set[str] = {
        "bool",
        "char",
        "uchar",
        "int8_t",
        "uint8_t",
        "short",
        "ushort",
        "int16_t",
        "uint16_t",
        "int",
        "uint",
        "int32_t",
        "uint32_t",
        "long",
        "ulong",
        "int64_t",
        "uint64_t",
        "size_t",
    }
    _METAL_SIGNED_SCALAR_TYPES: Set[str] = {
        "char",
        "int8_t",
        "short",
        "int16_t",
        "int",
        "int32_t",
        "long",
        "int64_t",
        "half",
        "float16_t",
        "bfloat16_t",
        "float",
        "double",
    }

    # Metal SIMD-group / quad-group built-ins that return the SAME type as their
    # first argument (a shuffle/broadcast moves a value between lanes; a prefix /
    # reduction combines values of the operand type). Recognizing them lets an
    # argument like `simd_shuffle_and_fill_up(val, init, i)` be typed from `val`,
    # so a SFINAE member method that threads such a call through a sibling method
    # (scan's `operator()(val, simd_shuffle_and_fill_up(val, ...))`) can resolve.
    _METAL_FIRST_ARG_TYPED_GROUP_BUILTINS: Set[str] = {
        "simd_shuffle",
        "simd_shuffle_up",
        "simd_shuffle_down",
        "simd_shuffle_xor",
        "simd_shuffle_rotate_up",
        "simd_shuffle_rotate_down",
        "simd_shuffle_and_fill_up",
        "simd_shuffle_and_fill_down",
        "simd_broadcast",
        "simd_broadcast_first",
        "simd_prefix_inclusive_sum",
        "simd_prefix_exclusive_sum",
        "simd_prefix_inclusive_product",
        "simd_prefix_exclusive_product",
        "simd_sum",
        "simd_product",
        "simd_prod",
        "simd_min",
        "simd_max",
        "simd_and",
        "simd_or",
        "simd_xor",
        "quad_shuffle",
        "quad_shuffle_up",
        "quad_shuffle_down",
        "quad_shuffle_xor",
        "quad_shuffle_and_fill_up",
        "quad_shuffle_and_fill_down",
        "quad_broadcast",
        "quad_sum",
        "quad_prefix_inclusive_sum",
        "quad_prefix_exclusive_sum",
    }

    class _UnrecognizedConstraint(Exception):
        """A SFINAE constraint outside the small recognized set — clean-fail."""

    def _scalar_and_width(self, type_text: str) -> Optional[Tuple[str, int]]:
        # Decompose a scalar/vector type into (scalar base, component width). A
        # trailing 2/3/4 on a known scalar denotes a vector; concrete
        # `metal::vec<T, N>` spellings carry the same information. Returns None
        # when the base or vector extent is not concrete and recognized.
        text = self._normalize_inferred_type(type_text)
        if not text or " " in text:
            return None
        generic_match = re.fullmatch(r"(?:::)?(?:metal::)?vec<(.+)>", text)
        if generic_match is not None:
            arguments = self._split_top_level_commas(generic_match.group(1))
            if len(arguments) != 2:
                return None
            base = self._normalize_inferred_type(arguments[0])
            if base not in self._METAL_SCALAR_TYPE_SIZES:
                return None
            width_text = self._proven_integral_constant_value("int", arguments[1])
            if width_text is None:
                return None
            width = int(width_text)
            return (base, width) if width > 0 else None
        match = re.fullmatch(
            r"(?P<base>[A-Za-z_][A-Za-z0-9_]*?)(?P<width>[234])?", text
        )
        if match is None:
            return None
        base = match.group("base")
        if base not in self._METAL_SCALAR_TYPE_SIZES:
            # A width suffix that is actually part of the name (e.g. an unknown
            # type) — treat the whole token as the base.
            if text in self._METAL_SCALAR_TYPE_SIZES:
                return text, 1
            return None
        width = int(match.group("width")) if match.group("width") else 1
        return base, width

    def _sizeof_concrete_type(self, type_text: str) -> Optional[int]:
        return metal_type_size(type_text)

    def _is_integral_concrete_type(self, type_text: str) -> Optional[bool]:
        decomposed = self._scalar_and_width(type_text)
        if decomposed is None:
            return None
        base, _width = decomposed
        return base in self._METAL_INTEGRAL_SCALAR_TYPES

    def _is_signed_concrete_type(self, type_text: str) -> Optional[bool]:
        decomposed = self._scalar_and_width(type_text)
        if decomposed is None:
            return None
        base, _width = decomposed
        return base in self._METAL_SIGNED_SCALAR_TYPES

    def _evaluate_template_constraint(
        self, constraint: str, bindings: Dict[str, str]
    ) -> bool:
        # Evaluate a single SFINAE constraint for the concrete `bindings` and
        # return whether the overload is ENABLED. Raises _UnrecognizedConstraint
        # for any constraint shape outside the recognized set so the caller can
        # clean-fail rather than guess.
        text = constraint.strip()
        # Unwrap an `[metal::]enable_if_t<COND, type>` enabler down to COND; a
        # bare `enable_if_t<COND>` (one argument) unwraps to COND too. The
        # presence of the alias means "enabled iff COND".
        enable_match = re.match(
            r"^(?:typename\s+)?(?:metal\s*::\s*)?enable_if_t\s*<", text
        )
        if enable_match is not None:
            angle_start = text.find("<", enable_match.end() - 1)
            # The enabler's condition may embed a comparison `<` (`sizeof(T) < 8`),
            # so balance the angle and split its arguments with the SFINAE-aware
            # helpers rather than the generic ones.
            angle_end = self._find_matching_template_param_angle(text, angle_start)
            if angle_end is None or text[angle_end + 1 :].strip():
                raise self._UnrecognizedConstraint(constraint)
            arguments = self._split_template_parameter_list(
                text[angle_start + 1 : angle_end]
            )
            if not arguments:
                raise self._UnrecognizedConstraint(constraint)
            return self._evaluate_boolean_constraint(arguments[0].strip(), bindings)
        return self._evaluate_boolean_constraint(text, bindings)

    def _evaluate_boolean_constraint(
        self, expression: str, bindings: Dict[str, str]
    ) -> bool:
        expr = expression.strip()
        if not expr:
            raise self._UnrecognizedConstraint(expression)
        # Leading negation.
        if expr.startswith("!"):
            return not self._evaluate_boolean_constraint(expr[1:], bindings)
        # Strip a single fully-enclosing paren group.
        while (
            expr.startswith("(")
            and self._find_matching_delimiter(expr, 0, "(", ")") == len(expr) - 1
        ):
            expr = expr[1:-1].strip()
            if not expr:
                raise self._UnrecognizedConstraint(expression)
            if expr.startswith("!"):
                return not self._evaluate_boolean_constraint(expr[1:], bindings)

        disjunction = self._split_top_level_boolean_constraint(expr, ("||", "|"))
        if disjunction is not None:
            return any(
                self._evaluate_boolean_constraint(part, bindings)
                for part in disjunction
            )
        conjunction = self._split_top_level_boolean_constraint(expr, ("&&", "&"))
        if conjunction is not None:
            return all(
                self._evaluate_boolean_constraint(part, bindings)
                for part in conjunction
            )

        # `is_integral_v<T>` (optionally `metal::`).
        integral_match = re.fullmatch(
            r"(?:metal\s*::\s*)?is_integral_v\s*<\s*(?P<arg>[^<>]+?)\s*>", expr
        )
        if integral_match is not None:
            concrete = self._resolve_constraint_type(
                integral_match.group("arg"), bindings
            )
            result = self._is_integral_concrete_type(concrete)
            if result is None:
                raise self._UnrecognizedConstraint(expression)
            return result

        # `is_signed_v<T>` (optionally `metal::`).
        signed_match = re.fullmatch(
            r"(?:metal\s*::\s*)?is_signed_v\s*<\s*(?P<arg>[^<>]+?)\s*>", expr
        )
        if signed_match is not None:
            concrete = self._resolve_constraint_type(
                signed_match.group("arg"), bindings
            )
            result = self._is_signed_concrete_type(concrete)
            if result is None:
                raise self._UnrecognizedConstraint(expression)
            return result

        # `sizeof(T) <cmp> N`.
        sizeof_match = re.fullmatch(
            r"sizeof\s*\(\s*(?P<arg>[^()]+?)\s*\)\s*"
            r"(?P<op><=|>=|==|!=|<|>)\s*(?P<rhs>[0-9]+)",
            expr,
        )
        if sizeof_match is not None:
            concrete = self._resolve_constraint_type(
                sizeof_match.group("arg"), bindings
            )
            size = self._sizeof_concrete_type(concrete)
            if size is None:
                raise self._UnrecognizedConstraint(expression)
            rhs = int(sizeof_match.group("rhs"))
            return self._compare(size, sizeof_match.group("op"), rhs)

        raise self._UnrecognizedConstraint(expression)

    def _split_top_level_boolean_constraint(
        self, expression: str, operators: Tuple[str, ...]
    ) -> Optional[List[str]]:
        parts: List[str] = []
        depth = 0
        angle_depth = 0
        start = 0
        i = 0
        while i < len(expression):
            ch = expression[i]
            if ch == "(":
                depth += 1
                i += 1
                continue
            if ch == ")":
                depth = max(0, depth - 1)
                i += 1
                continue
            if ch == "<":
                angle_depth += 1
                i += 1
                continue
            if ch == ">":
                angle_depth = max(0, angle_depth - 1)
                i += 1
                continue
            if depth == 0 and angle_depth == 0:
                for operator_text in operators:
                    if expression.startswith(operator_text, i):
                        parts.append(expression[start:i].strip())
                        i += len(operator_text)
                        start = i
                        break
                else:
                    i += 1
                continue
            i += 1
        if not parts:
            return None
        parts.append(expression[start:].strip())
        if any(not part for part in parts):
            raise self._UnrecognizedConstraint(expression)
        return parts

    def _resolve_constraint_type(self, arg: str, bindings: Dict[str, str]) -> str:
        # Resolve a constraint's type operand (`T`) to its concrete binding,
        # leaving an already-concrete type as-is.
        text = arg.strip()
        return bindings.get(text, text)

    def _compare(self, left: int, op: str, right: int) -> bool:
        comparisons = {
            "<": operator.lt,
            ">": operator.gt,
            "<=": operator.le,
            ">=": operator.ge,
            "==": operator.eq,
            "!=": operator.ne,
        }
        return comparisons[op](left, right)

    def _select_constrained_overload(
        self,
        candidates: List[Tuple[_MetalStructMethod, Dict[str, str]]],
    ) -> Optional[Tuple[_MetalStructMethod, Dict[str, str]]]:
        # Pick the unique overload whose SFINAE constraints ALL evaluate true for
        # that overload's bindings. Returns None when zero or more than one
        # overload is enabled, or when any constraint is unrecognized — every
        # such case is a clean failure (never guess / never mis-select). An
        # overload with NO constraints is considered always-enabled.
        enabled: List[Tuple[_MetalStructMethod, Dict[str, str]]] = []
        for overload, bindings in candidates:
            try:
                if all(
                    self._evaluate_template_constraint(constraint, bindings)
                    for constraint in overload.template_constraints
                ):
                    enabled.append((overload, bindings))
            except self._UnrecognizedConstraint:
                return None
        if len(enabled) == 1:
            return enabled[0]
        return None

    def _parse_struct_method(
        self, struct_name: str, body: str, decl_start: int, brace: int
    ) -> Optional[_MetalStructMethod]:
        # Parse a `RetType name(params) <qualifiers> { body }` declarator that
        # starts at decl_start with its body opening at `brace`. Returns None when
        # the construct is not actually an instance/static member function we can
        # lower (e.g. a constructor, destructor, or a brace-initialized member).
        header = body[decl_start:brace]
        nested_type_header = re.match(r"\s*(?:struct|class|union)\b", header)
        elaborated_return_method = re.match(
            r"\s*(?:struct|class|union)\s+"
            r"[A-Za-z_][A-Za-z0-9_:]*\s+"
            r"[A-Za-z_][A-Za-z0-9_]*\s*\(",
            header,
        )
        if nested_type_header is not None and elaborated_return_method is None:
            return None

        # `operator()` is special: the declarator itself contains parentheses, so
        # the actual parameter list is the paren group that FOLLOWS the empty
        # `operator()` token rather than the first top-level `(`.
        operator_match = re.search(r"\boperator\s*\(\s*\)", header)
        is_operator_call = False
        if operator_match is not None:
            paren_start = self._function_parameter_start(header[operator_match.end() :])
            if paren_start is None:
                return None
            paren_start += operator_match.end()
            paren_end = self._find_matching_delimiter(header, paren_start, "(", ")")
            if paren_end is None:
                return None
            is_operator_call = True
            method_name = "operator()"
            signature_prefix = header[: operator_match.start()].rstrip()
            parameters = header[paren_start + 1 : paren_end]
        else:
            paren_start = self._function_parameter_start(header)
            if paren_start is None:
                return None
            paren_end = self._find_matching_delimiter(header, paren_start, "(", ")")
            if paren_end is None:
                return None
            before_params = header[:paren_start].rstrip()
            parameters = header[paren_start + 1 : paren_end]
            name_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", before_params)
            if name_match is None:
                return None
            method_name = name_match.group(1)
            signature_prefix = before_params[: name_match.start()].rstrip()
            # `operator+`, `operator==`, ... (overloaded operators other than the
            # call operator) are not lowered: they are out of scope and the
            # existing operator-overload handling/diagnostics still apply.
            if signature_prefix.endswith("operator") or method_name == "operator":
                return None
            # Constructors/destructors have no return type and a name equal to the
            # struct; leave them in place (out of scope, data-only structs do not
            # model them and call sites use them as types, not methods).
            if method_name == struct_name or signature_prefix.endswith("~"):
                return None
            # A bare `name(params) {` with no return type is a constructor-like
            # declarator we do not lower.
            if not signature_prefix:
                return None
            # A "return type" consisting only of a macro qualifier (e.g.
            # `METAL_FUNC`) with no actual type names a constructor whose name no
            # longer matches its struct. This happens for a template struct that
            # was materialized (renamed `Foo` -> `Foo_float`) while its
            # constructor keeps the original template name `Foo`, so the
            # `method_name == struct_name` check above cannot see it. Leave such
            # constructors in place like every other constructor rather than
            # emitting a malformed `METAL_FUNC Foo_float__Foo(...)` free function.
            if not self._strip_function_qualifier_macros(signature_prefix):
                return None

        # Detect and strip a leading `static` qualifier from the return type.
        is_static = bool(re.search(r"(^|\s)static(\s|$)", signature_prefix))
        return_type = re.sub(r"\bstatic\b", " ", signature_prefix)
        # Strip storage/qualifier keywords that are meaningless on a free
        # function return type while preserving the actual type tokens.
        return_type = re.sub(r"\binline\b", " ", return_type)
        return_type = re.sub(r"\bconstexpr\b", " ", return_type)
        return_type = self._strip_function_qualifier_macros(return_type)
        if not return_type:
            return None
        receiver_qualifiers = header[paren_end + 1 :]
        is_const = bool(re.search(r"\bconst\b", receiver_qualifiers))
        receiver_address_spaces = self._receiver_address_spaces(receiver_qualifiers)

        method_body_end = self._find_matching_brace(body, brace)
        if method_body_end is None:
            return None
        method_body = body[brace + 1 : method_body_end - 1]

        parameter_names = self._parameter_identifier_names(parameters)
        free_name = self._struct_member_free_name(
            struct_name, method_name, is_operator_call
        )
        return _MetalStructMethod(
            name=method_name,
            free_name=free_name,
            is_static=is_static,
            is_operator_call=is_operator_call,
            return_type=return_type,
            parameters=parameters.strip(),
            parameter_names=parameter_names,
            body=method_body,
            span=(decl_start, method_body_end),
            is_const=is_const,
            receiver_address_spaces=receiver_address_spaces,
        )

    def _strip_function_qualifier_macros(self, prefix: str) -> str:
        # Remove Metal function-qualifier MACROS (`METAL_FUNC`, `STEEL_CONST`,
        # `C10_METAL_CONSTEXPR`) that may precede a member function's return type
        # and return the remaining, whitespace-collapsed text. An empty result
        # means the declarator carried no actual return type (a constructor).
        stripped = re.sub(
            r"\b(?:METAL_FUNC|STEEL_CONST|C10_METAL_CONSTEXPR)\b",
            " ",
            prefix,
        )
        return re.sub(r"\s+", " ", stripped).strip()

    def _struct_member_free_name(
        self, struct_name: str, method_name: str, is_operator_call: bool
    ) -> str:
        if is_operator_call:
            return f"{struct_name}__operator_call"
        return f"{struct_name}__{method_name}"

    def _brace_construct_is_method_definition(self, header: str) -> bool:
        # Decide whether the text preceding a `{` body is a function declarator
        # (constructor/destructor/operator/regular method) rather than a data
        # member with a brace initializer. A method declarator has a top-level
        # parameter list `(...)`; a brace-initialized member (`size_t n_{0}`) has
        # the member name immediately before the brace with no parameter list.
        # A constructor initializer list (`Foo(v) : a(v)`) also has parens, which
        # is exactly why such constructs are classified as methods.
        return self._function_parameter_start(header) is not None

    def _declaration_is_method_prototype(self, declaration: str) -> bool:
        # A declaration with a top-level parameter list and a name immediately
        # before it is a function prototype rather than a data member. Inspect only
        # the declarator side of a top-level assignment: parentheses in a static
        # member initializer such as `16 / sizeof(T)` are not function parameters.
        declarator, _default = self._split_top_level_assignment(declaration)
        paren_start = self._function_parameter_start(declarator)
        if paren_start is None:
            return False
        paren_end = self._find_matching_delimiter(declarator, paren_start, "(", ")")
        if paren_end is None:
            return False
        before = declarator[:paren_start].rstrip()
        if re.search(r"\boperator\s*\(\s*\)\s*$", before):
            return True
        return re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*$", before) is not None

    def _struct_member_prototype_name(self, declaration: str) -> Optional[str]:
        text = declaration.strip()
        if text.startswith("template"):
            angle_start = text.find("<")
            angle_end = self._find_matching_template_param_angle(text, angle_start)
            if angle_end is None:
                return None
            text = text[angle_end + 1 :].strip()
        declarator, _default = self._split_top_level_assignment(text)
        if re.search(r"\boperator\s*\(\s*\)\s*\(", declarator):
            return "operator()"
        paren_start = self._function_parameter_start(declarator)
        if paren_start is None:
            return None
        before = declarator[:paren_start].rstrip()
        destructor = re.search(r"~\s*([A-Za-z_][A-Za-z0-9_]*)\s*$", before)
        if destructor is not None:
            return f"~{destructor.group(1)}"
        match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", before)
        return match.group(1) if match is not None else None

    @staticmethod
    def _receiver_address_spaces(text: str) -> Tuple[str, ...]:
        address_spaces = ("constant", "device", "thread", "threadgroup")
        words = set(IDENTIFIER_RE.findall(text))
        return tuple(space for space in address_spaces if space in words)

    @staticmethod
    def _declaration_is_type_alias(declaration: str) -> bool:
        return bool(re.match(r"\s*(?:using\b[^=]*=|typedef\b)", declaration))

    def _declared_data_member_name(self, declaration: str) -> Optional[str]:
        # Extract the declared identifier from a data-member declaration such as
        # `float bias`, `T data[N]`, `device float* ptr`, or
        # `static constexpr constant U init = U(0)`.
        text = self._strip_top_level_default_value(declaration).strip()
        if not text:
            return None
        # Drop any trailing array extents so the bare name is left.
        while text.endswith("]"):
            open_bracket = text.rfind("[")
            if open_bracket == -1:
                break
            text = text[:open_bracket].rstrip()
        match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", text)
        if match is None:
            return None
        name = match.group(1)
        # Guard against returning a type keyword when no name is present.
        if name in {"struct", "class", "void"}:
            return None
        return name

    def _parameter_identifier_names(self, parameters: str) -> List[str]:
        names: List[str] = []
        for parameter in self._split_top_level_commas(parameters):
            name = self._declared_data_member_name(parameter)
            if name and name != "void":
                names.append(name)
        return names

    def _render_lowered_struct(
        self,
        code: str,
        struct: _MetalStructDefinition,
        *,
        instantiated_template_functions: Optional[Dict[str, str]] = None,
        template_methods_by_struct: Optional[
            Dict[str, Dict[str, List[_MetalStructMethod]]]
        ] = None,
        methods_by_struct: Optional[Dict[str, Dict[str, _MetalStructMethod]]] = None,
        operator_call_structs: Optional[Set[str]] = None,
        structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
    ) -> Tuple[str, List[str]]:
        # Produce (data_only_struct_text, [free_function_text, ...]) for one
        # struct. The data-only struct keeps the original header (including any
        # base clause) and all NON-method statements; methods are removed and
        # re-emitted as free functions.
        struct_text = code[struct.span[0] : struct.span[1]]
        header_body_start = self._find_next_top_level_char(struct_text, 0, "{")
        header = struct_text[: header_body_start + 1]
        # Body content sits between the struct span body bounds; recompute
        # relative to the struct_text slice.
        body_abs_start, body_abs_end = struct.body_span
        body_rel_start = body_abs_start - struct.span[0]
        body_rel_end = body_abs_end - struct.span[0]
        body = struct_text[body_rel_start:body_rel_end]

        # Remove method spans (relative to body) to build the data-only body.
        # Both concrete and template methods are stripped from the struct: the
        # struct becomes pure data and every method (concrete now, template on
        # demand) is re-emitted as a free function.
        removals: List[Tuple[int, int]] = []
        for method in [*struct.methods, *struct.template_methods]:
            rel_start = method.span[0] - body_abs_start
            rel_end = method.span[1] - body_abs_start
            # Extend over a trailing `;` after the method body if present.
            tail = rel_end
            while tail < len(body) and body[tail].isspace():
                tail += 1
            if tail < len(body) and body[tail] == ";":
                rel_end = tail + 1
            removals.append((rel_start, rel_end))
        data_body = self._remove_spans(body, removals)
        data_body = self._collapse_blank_lines(data_body)

        data_only = header + data_body + "}"
        data_only = data_only.rstrip()
        if not data_only.endswith(";"):
            data_only += ";"

        free_functions = [
            self._emit_free_function(
                struct,
                method,
                instantiated_template_functions=instantiated_template_functions,
                template_methods_by_struct=template_methods_by_struct,
                methods_by_struct=methods_by_struct,
                operator_call_structs=operator_call_structs,
                structs_by_name=structs_by_name,
            )
            for method in struct.methods
            if not self._method_returns_lvalue_reference(method)
        ]
        return data_only, free_functions

    def _emit_free_function(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        *,
        instantiated_template_functions: Optional[Dict[str, str]] = None,
        template_methods_by_struct: Optional[
            Dict[str, Dict[str, List[_MetalStructMethod]]]
        ] = None,
        methods_by_struct: Optional[Dict[str, Dict[str, _MetalStructMethod]]] = None,
        operator_call_structs: Optional[Set[str]] = None,
        structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
    ) -> str:
        # Emit `RetType S__m(thread S& self, <params>) { body' }` for an instance
        # method,
        # or `RetType S__m(<params>) { body' }` for a static method. References to
        # the struct's data members inside the body are rewritten to `self.x`.
        return_type = self._canonicalize_struct_scoped_type(
            method.return_type, struct, structs_by_name
        )
        self._record_known_member_function_return_type(method.free_name, return_type)
        specialized_body = self._specialize_concrete_method_body(
            struct, method, method.body, structs_by_name
        )
        rewritten_body = self._rewrite_method_body(
            struct, replace(method, body=specialized_body)
        )
        rewritten_body = self._canonicalize_struct_scoped_local_declarations(
            rewritten_body, struct, structs_by_name
        )
        rewritten_body = self._canonicalize_struct_scoped_cast_types(
            rewritten_body, struct, structs_by_name
        )
        if instantiated_template_functions is not None and template_methods_by_struct:
            rewritten_body = self._lower_internal_template_member_calls(
                struct,
                method,
                method.parameters,
                rewritten_body,
                instantiated_template_functions,
                template_methods_by_struct,
                methods_by_struct=methods_by_struct,
                operator_call_structs=operator_call_structs,
                rewrite_structs_by_name=structs_by_name,
            )
        params = self._canonicalize_struct_scoped_parameters(
            method.parameters, struct, structs_by_name
        )
        if method.is_static:
            new_params = params if params and params != "void" else ""
        else:
            self_param = self._instance_receiver_parameter(struct, method)
            if params and params != "void":
                new_params = f"{self_param}, {params}"
            else:
                new_params = self_param
        return f"{return_type} {method.free_name}({new_params}) {{{rewritten_body}}}"

    @staticmethod
    def _instance_receiver_parameter(
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
    ) -> str:
        const_qualifier = "const " if method.is_const else ""
        return f"thread {const_qualifier}{struct.name}& self"

    def _resolved_static_data_member_initializers(
        self, struct: _MetalStructDefinition
    ) -> Dict[str, str]:
        members: Dict[str, _MetalDataMember] = {}
        initializers: Dict[str, str] = {}
        for member in struct.data_members:
            qualifiers = set(IDENTIFIER_RE.findall(member.type_text))
            if (
                member.default is None
                or "static" not in qualifiers
                or not qualifiers.intersection({"const", "constant", "constexpr"})
            ):
                continue
            initializer = member.default.strip()
            if not initializer:
                continue
            initializer = self._canonicalize_struct_scoped_type(
                initializer,
                struct,
                None,
            )
            members[member.name] = member
            initializers[member.name] = initializer
        if not members:
            return {}

        states: Dict[str, str] = {}
        resolved: Dict[str, str] = {}
        member_names = list(members)

        def resolve(name: str, depth: int = 0) -> Optional[str]:
            state = states.get(name)
            if state == "resolved":
                return resolved[name]
            if state in {"visiting", "failed"}:
                return None
            if depth >= self._STATIC_INITIALIZER_MAX_DEPENDENCY_DEPTH:
                states[name] = "failed"
                return None

            states[name] = "visiting"
            initializer = initializers[name]
            dependency_mapping: Dict[str, str] = {}
            for dependency in self._static_initializer_dependencies(
                initializer, member_names
            ):
                dependency_value = resolve(dependency, depth + 1)
                if dependency_value is None:
                    states[name] = "failed"
                    return None
                dependency_mapping[dependency] = f"({dependency_value})"
            if dependency_mapping:
                initializer = self._substitute_bare_member_references(
                    initializer, dependency_mapping
                )
            initializer = self._fold_static_integral_initializer(
                members[name], initializer
            )
            resolved[name] = initializer
            states[name] = "resolved"
            return initializer

        for name in member_names:
            resolve(name)
        return resolved

    def _substitute_static_data_member_initializers(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        body: str,
    ) -> str:
        # A lowered free function cannot resolve a bare reference to a static
        # data member that remains nested in the data-only struct. Inline the
        # initializer for initialized static constants, but only at bare,
        # unshadowed uses. Mutable static members are deliberately excluded.
        shadowed = set(method.parameter_names)
        shadowed.update(self._local_variable_names(body))
        mapping = {
            name: f"({initializer})"
            for name, initializer in self._resolved_static_data_member_initializers(
                struct
            ).items()
            if name not in shadowed
        }
        if not mapping:
            return body
        return self._substitute_bare_member_references(body, mapping)

    def _evaluate_static_assertions(self, code: str) -> str:
        if "static_assert" not in code:
            return code

        template_spans = self._find_template_declaration_spans(code)
        assertions = self._find_concrete_static_assertions(code, template_spans)
        if not assertions:
            return code

        owner_spans = [(0, len(code))]
        type_aliases = self._collect_local_type_alias_bindings(
            code,
            owner_spans,
            skip_spans=template_spans,
        )
        constants = self._collect_local_integral_constant_bindings(
            code,
            owner_spans,
            type_aliases,
        )
        structs = self._find_concrete_struct_definitions(code)
        replacements: List[Tuple[int, int, str]] = []
        for assertion in assertions:
            condition_position = assertion.condition_span[0]
            visible_bindings = self._local_integral_constant_bindings_at(
                constants,
                condition_position,
            )
            visible_constants = {
                name: binding.value
                for name, binding in visible_bindings.items()
                if binding.value is not None
            }
            resolved_condition = self._substitute_template_argument_static_constants(
                assertion.condition,
                visible_constants,
            )
            resolved_condition = self._replace_concrete_sizeof_expressions(
                resolved_condition,
                lambda type_text: self._resolve_type_aliases_at(
                    type_text,
                    type_aliases,
                    condition_position,
                ),
                lambda type_text: self._concrete_aggregate_type_layout(
                    type_text,
                    structs,
                    type_aliases,
                    condition_position,
                ),
            )
            folded, value = self._evaluate_static_integral_expression(
                resolved_condition
            )
            if folded and value:
                replacements.append((*assertion.span, ""))
                continue

            source_message = (
                self._evaluate_metal_string_expression(assertion.message)
                if assertion.message is not None
                else None
            )
            compact_condition = re.sub(r"\s+", " ", assertion.condition).strip()
            compact_resolved = re.sub(r"\s+", " ", resolved_condition).strip()
            if folded:
                suggested_action = (
                    "correct the source condition or select specialization values "
                    "that satisfy the compile-time constraint"
                )
                detail = (
                    f"Metal static assertion failed because '{compact_resolved}' "
                    "evaluated to false"
                )
                reason = "assertion-failed"
                unresolved_dependencies: Tuple[str, ...] = ()
            else:
                unresolved_dependencies = self._static_assertion_dependencies(
                    resolved_condition,
                    visible_bindings,
                )
                suggested_action = (
                    "materialize all template parameters and express assertion "
                    "dependencies as constexpr integral values before translation"
                )
                dependency_detail = (
                    "; unresolved dependencies: " + ", ".join(unresolved_dependencies)
                    if unresolved_dependencies
                    else ""
                )
                detail = (
                    "Metal static assertion could not be evaluated at translation "
                    f"time: '{compact_resolved}'{dependency_detail}"
                )
                reason = "condition-unresolved"
            if source_message:
                detail += f". Source message: {source_message}"
            detail += f". Suggested action: {suggested_action}."
            raise MetalStaticAssertionError(
                detail,
                reason=reason,
                expression=compact_condition,
                resolved_expression=compact_resolved,
                assertion_message=source_message,
                unresolved_dependencies=unresolved_dependencies,
                source_location=self._source_location_for_offsets(
                    code,
                    assertion.span[0],
                    assertion.span[1],
                ),
                suggested_action=suggested_action,
            )

        return self._apply_text_replacements(code, replacements)

    def _find_concrete_static_assertions(
        self,
        code: str,
        template_spans: List[Tuple[int, int]],
    ) -> List[_MetalStaticAssertion]:
        ignored_spans = self._find_comment_and_literal_spans(code)
        assertions: List[_MetalStaticAssertion] = []
        for match in re.finditer(r"\bstatic_assert\b", code):
            start = match.start()
            if self._containing_span(start, ignored_spans) is not None:
                continue
            if self._containing_span(start, template_spans) is not None:
                continue

            open_paren = match.end()
            while open_paren < len(code) and code[open_paren].isspace():
                open_paren += 1
            if open_paren >= len(code) or code[open_paren] != "(":
                self._raise_malformed_static_assertion(code, start, match.end())
            close_paren = self._find_matching_delimiter(
                code,
                open_paren,
                "(",
                ")",
            )
            if close_paren is None:
                self._raise_malformed_static_assertion(code, start, len(code))

            end = close_paren + 1
            while end < len(code) and code[end].isspace():
                end += 1
            if end >= len(code) or code[end] != ";":
                self._raise_malformed_static_assertion(
                    code,
                    start,
                    close_paren + 1,
                )
            end += 1
            condition, assertion_message, condition_offset = (
                self._split_static_assertion_arguments(
                    code[open_paren + 1 : close_paren]
                )
            )
            if not condition:
                self._raise_malformed_static_assertion(code, start, end)
            condition_start = open_paren + 1 + condition_offset
            assertions.append(
                _MetalStaticAssertion(
                    span=(start, end),
                    condition_span=(
                        condition_start,
                        condition_start + len(condition),
                    ),
                    condition=condition,
                    message=assertion_message,
                )
            )
        return assertions

    def _split_static_assertion_arguments(
        self, arguments: str
    ) -> Tuple[str, Optional[str], int]:
        comma_positions: List[int] = []
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        i = 0
        while i < len(arguments):
            if arguments[i] in "\"'":
                _literal, consumed = self._read_string(arguments, i)
                i += consumed
                continue
            if arguments.startswith("//", i):
                end = arguments.find("\n", i)
                i = len(arguments) if end == -1 else end + 1
                continue
            if arguments.startswith("/*", i):
                end = arguments.find("*/", i + 2)
                i = len(arguments) if end == -1 else end + 2
                continue
            character = arguments[i]
            if character == "(":
                paren_depth += 1
            elif character == ")":
                paren_depth = max(0, paren_depth - 1)
            elif character == "[":
                bracket_depth += 1
            elif character == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif character == "{":
                brace_depth += 1
            elif character == "}":
                brace_depth = max(0, brace_depth - 1)
            elif (
                character == ","
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
            ):
                comma_positions.append(i)
            i += 1

        for comma in reversed(comma_positions):
            candidate_message = arguments[comma + 1 :].strip()
            if re.fullmatch(
                METAL_STRING_EXPRESSION_PATTERN,
                candidate_message,
                re.DOTALL,
            ):
                raw_condition = arguments[:comma]
                condition = raw_condition.strip()
                return (
                    condition,
                    candidate_message,
                    len(raw_condition) - len(raw_condition.lstrip()),
                )
        condition = arguments.strip()
        return condition, None, len(arguments) - len(arguments.lstrip())

    def _raise_malformed_static_assertion(
        self, code: str, start: int, end: int
    ) -> None:
        suggested_action = (
            "provide a well-formed static_assert(condition[, message]) statement"
        )
        raise MetalStaticAssertionError(
            "Metal static assertion could not be parsed. Suggested action: "
            f"{suggested_action}.",
            reason="assertion-malformed",
            expression="",
            resolved_expression="",
            assertion_message=None,
            unresolved_dependencies=(),
            source_location=self._source_location_for_offsets(code, start, end),
            suggested_action=suggested_action,
        )

    def _static_assertion_dependencies(
        self,
        expression: str,
        visible_bindings: Dict[str, _MetalIntegralConstantBinding],
    ) -> Tuple[str, ...]:
        masked = self._mask_comments_and_literals(expression)
        ignored = {
            "alignof",
            "false",
            "sizeof",
            "true",
        }
        dependencies = {
            identifier
            for identifier in IDENTIFIER_RE.findall(masked)
            if identifier not in ignored
        }
        dependencies.update(
            name
            for name, binding in visible_bindings.items()
            if binding.value is None
            and re.search(rf"\b{re.escape(name)}\b", masked) is not None
        )
        return tuple(sorted(dependencies))

    def _substitute_struct_scoped_alias_owners(
        self,
        source: str,
        struct: _MetalStructDefinition,
    ) -> str:
        if not struct.type_aliases:
            return source
        ignored_spans = self._find_comment_and_literal_spans(source)
        replacements: List[Tuple[int, int, str]] = []
        for alias in sorted(struct.type_aliases, key=len, reverse=True):
            target = self._canonicalize_struct_scoped_type(alias, struct, None)
            if not target or target == alias:
                continue
            for match in re.finditer(
                rf"\b{re.escape(alias)}\b(?=\s*::)",
                source,
            ):
                if self._containing_span(match.start(), ignored_spans) is None:
                    replacements.append((match.start(), match.end(), target))
        return self._apply_text_replacements(source, replacements)

    _STATIC_INITIALIZER_MAX_DEPENDENCY_DEPTH = 128

    def _static_initializer_dependencies(
        self, initializer: str, member_names: List[str]
    ) -> List[str]:
        # Reuse the body substitution scanner to identify only BARE dependency
        # references. Names in comments, strings, calls, or member accesses do
        # not form initializer-map edges.
        dependencies: List[str] = []
        for name in member_names:
            probe = self._substitute_bare_member_references(
                initializer, {name: f"({name})"}
            )
            if probe != initializer:
                dependencies.append(name)
        return dependencies

    def _fold_static_integral_initializer(
        self, member: _MetalDataMember, initializer: str
    ) -> str:
        # Fold only a small, integer-only expression language. Calls, casts,
        # names, floating-point values, and unsupported operators leave the
        # original initializer untouched and are still substituted verbatim.
        type_tokens = IDENTIFIER_RE.findall(member.type_text)
        base_type = next(
            (
                token
                for token in reversed(type_tokens)
                if token in self._METAL_INTEGRAL_SCALAR_TYPES
            ),
            None,
        )
        if base_type is None:
            return initializer
        value = self._proven_integral_constant_value(base_type, initializer)
        return initializer if value is None else value

    def _proven_integral_constant_value(
        self, base_type: str, initializer: str
    ) -> Optional[str]:
        if base_type not in self._METAL_INTEGRAL_SCALAR_TYPES:
            return None
        folded, value = self._evaluate_static_integral_expression(initializer)
        if not folded or value is None:
            return None
        if base_type == "bool":
            return "true" if bool(value) else "false"

        bits = self._METAL_SCALAR_TYPE_SIZES[base_type] * 8
        if base_type in self._METAL_SIGNED_SCALAR_TYPES:
            minimum = -(1 << (bits - 1))
            maximum = (1 << (bits - 1)) - 1
        else:
            minimum = 0
            maximum = (1 << bits) - 1
        integer_value = int(value)
        if not minimum <= integer_value <= maximum:
            return None
        return str(integer_value)

    _STATIC_FOLD_MIN_INT = -(1 << 31)
    _STATIC_FOLD_MAX_INT = (1 << 31) - 1
    _MAX_CONCRETE_AGGREGATE_SIZE = (1 << 63) - 1

    def _concrete_aggregate_type_layout(
        self,
        type_text: str,
        structs: List[_MetalStructDefinition],
        type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
        position: int,
        resolving: Optional[Set[str]] = None,
    ) -> Optional[Tuple[int, int]]:
        resolved_type = self._resolve_type_aliases_at(
            type_text,
            type_aliases,
            position,
        )
        layout = metal_type_layout(resolved_type)
        if layout is not None:
            return layout

        normalized = self._normalize_inferred_type(resolved_type)
        normalized = re.sub(r"^(?:struct|class|union)\s+", "", normalized)
        normalized = normalized.lstrip(":")
        if (
            re.fullmatch(
                r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*",
                normalized,
            )
            is None
        ):
            return None

        candidates = [
            struct
            for struct in structs
            if normalized
            in {
                struct.name,
                struct.qualified_name.lstrip(":"),
            }
        ]
        if len(candidates) != 1:
            return None
        struct = candidates[0]
        if not struct.layout_supported or struct.base_clause:
            return None

        aggregate_name = struct.qualified_name or struct.name
        active = set(resolving or ())
        if aggregate_name in active:
            return None
        active.add(aggregate_name)

        name_counts: Dict[str, int] = {}
        for definition in structs:
            name_counts[definition.name] = name_counts.get(definition.name, 0) + 1
        structs_by_name = {
            definition.name: definition
            for definition in structs
            if name_counts[definition.name] == 1
        }

        offset = 0
        aggregate_alignment = 1
        for member in struct.data_members:
            qualifiers = set(IDENTIFIER_RE.findall(member.type_text))
            if "static" in qualifiers:
                continue
            if "*" in member.type_text or "&" in member.type_text:
                return None

            member_type = self._canonicalize_struct_scoped_type(
                member.type_text,
                struct,
                structs_by_name,
            )
            member_layout = self._concrete_aggregate_type_layout(
                member_type,
                structs,
                type_aliases,
                struct.span[0],
                active,
            )
            if member_layout is None:
                return None
            member_size, member_alignment = member_layout
            array_extents = self._concrete_array_extents(member.array_suffix)
            if array_extents is None:
                return None
            for extent in array_extents:
                stride = self._align_concrete_aggregate_offset(
                    member_size,
                    member_alignment,
                )
                if stride > self._MAX_CONCRETE_AGGREGATE_SIZE // extent:
                    return None
                member_size = stride * extent

            if struct.aggregate_kind == "union":
                offset = max(offset, member_size)
            else:
                offset = self._align_concrete_aggregate_offset(
                    offset,
                    member_alignment,
                )
                offset += member_size
            if offset > self._MAX_CONCRETE_AGGREGATE_SIZE:
                return None
            aggregate_alignment = max(aggregate_alignment, member_alignment)

        aggregate_size = self._align_concrete_aggregate_offset(
            offset,
            aggregate_alignment,
        )
        if aggregate_size == 0:
            aggregate_size = 1
        if aggregate_size > self._MAX_CONCRETE_AGGREGATE_SIZE:
            return None
        return aggregate_size, aggregate_alignment

    def _concrete_array_extents(self, suffix: str) -> Optional[List[int]]:
        if not suffix:
            return []
        extents: List[int] = []
        cursor = 0
        for match in re.finditer(r"\[\s*([^\[\]]+?)\s*\]", suffix):
            if suffix[cursor : match.start()].strip():
                return None
            folded, value = self._evaluate_static_integral_expression(match.group(1))
            if not folded or value is None or value <= 0:
                return None
            extents.append(int(value))
            cursor = match.end()
        if not extents or suffix[cursor:].strip():
            return None
        return extents

    @staticmethod
    def _align_concrete_aggregate_offset(offset: int, alignment: int) -> int:
        return ((offset + alignment - 1) // alignment) * alignment

    def _replace_concrete_sizeof_expressions(
        self,
        expression: str,
        type_resolver=None,
        layout_resolver=None,
    ) -> str:
        resolver = type_resolver or (lambda type_text: type_text)

        def replace(match):
            operand = self._normalize_template_argument_text(match.group("operand"))
            resolved_operand = resolver(operand)
            layout = (
                layout_resolver(resolved_operand)
                if layout_resolver is not None
                else metal_type_layout(resolved_operand)
            )
            size = layout[0] if layout is not None else None
            return str(size) if size is not None else match.group(0)

        return re.sub(
            r"\bsizeof\s*\(\s*(?P<operand>[^()]+?)\s*\)",
            replace,
            expression,
        )

    def _evaluate_static_integral_expression(
        self, expression: str
    ) -> Tuple[bool, Optional[int]]:
        # Translate C++ boolean spellings to Python solely for parsing, then
        # interpret a strict AST whitelist. The 32-bit intermediate bound avoids
        # folding expressions whose C++ overflow/promotion behavior is unclear.
        text = self._replace_concrete_sizeof_expressions(
            self._strip_template_argument_comments(expression)
        ).strip()
        if not text or len(text) > 512:
            return False, None
        if re.search(r"\b(?:if|else)\b", text):
            return False, None
        if self._static_integral_expression_has_ambiguous_precedence(text):
            return False, None
        text = self._rewrite_static_integral_conditional_expressions(text)
        if text is None:
            return False, None
        text = re.sub(r"\btrue\b", "True", text)
        text = re.sub(r"\bfalse\b", "False", text)
        text = text.replace("&&", " and ").replace("||", " or ")
        text = re.sub(r"!(?!=)", " not ", text)
        try:
            parsed = ast.parse(f"({text.strip()})", mode="eval")
        except (SyntaxError, ValueError):
            return False, None

        allowed_nodes = (
            ast.Expression,
            ast.Constant,
            ast.UnaryOp,
            ast.UAdd,
            ast.USub,
            ast.Not,
            ast.BinOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.LShift,
            ast.RShift,
            ast.BitAnd,
            ast.BitOr,
            ast.BitXor,
            ast.BoolOp,
            ast.And,
            ast.Or,
            ast.IfExp,
            ast.Compare,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.Name,
            ast.Load,
        )
        nodes = list(ast.walk(parsed))
        if len(nodes) > 128 or any(
            not isinstance(node, allowed_nodes) for node in nodes
        ):
            return False, None
        return self._evaluate_static_integral_node(parsed.body)

    def _static_integral_expression_has_ambiguous_precedence(
        self, expression: str, *, depth: int = 0
    ) -> bool:
        # Python groups bitwise operators above comparisons and `not` below
        # arithmetic, unlike C++. Reject only unparenthesized mixed groups;
        # explicit parentheses split the recursive checks and remain foldable.
        if depth > 32:
            return True
        question = self._top_level_conditional_question(expression)
        if question is not None:
            colon = self._matching_conditional_colon(expression, question)
            if colon is None:
                return True
            return any(
                self._static_integral_expression_has_ambiguous_precedence(
                    part, depth=depth + 1
                )
                for part in (
                    expression[:question],
                    expression[question + 1 : colon],
                    expression[colon + 1 :],
                )
            )

        top_level: List[str] = []
        cursor = 0
        while cursor < len(expression):
            if expression[cursor] in "\"'":
                _literal, consumed = self._read_string(expression, cursor)
                cursor += consumed
                continue
            if expression[cursor] == "(":
                close = self._find_matching_delimiter(expression, cursor, "(", ")")
                if (
                    close is None
                    or self._static_integral_expression_has_ambiguous_precedence(
                        expression[cursor + 1 : close], depth=depth + 1
                    )
                ):
                    return True
                cursor = close + 1
                continue
            operator_text = next(
                (
                    operator_text
                    for operator_text in (
                        "<<",
                        ">>",
                        "<=",
                        ">=",
                        "==",
                        "!=",
                        "&&",
                        "||",
                    )
                    if expression.startswith(operator_text, cursor)
                ),
                None,
            )
            if operator_text is not None:
                top_level.append(operator_text)
                cursor += len(operator_text)
                continue
            if expression[cursor] in "+-*/%<>&|^!":
                top_level.append(expression[cursor])
            cursor += 1

        logical_operands: List[List[str]] = [[]]
        for operator_text in top_level:
            if operator_text in {"&&", "||"}:
                logical_operands.append([])
            else:
                logical_operands[-1].append(operator_text)

        comparisons = {"<", "<=", ">", ">=", "==", "!="}
        bitwise = {"&", "|", "^"}
        for operand_operators in logical_operands:
            comparison_count = sum(
                operator_text in comparisons for operator_text in operand_operators
            )
            if comparison_count > 1:
                return True
            if comparisons.intersection(operand_operators) and bitwise.intersection(
                operand_operators
            ):
                return True
            if "!" in operand_operators and any(
                operator_text != "!" for operator_text in operand_operators
            ):
                return True
        return False

    def _rewrite_static_integral_conditional_expressions(
        self, expression: str, *, depth: int = 0
    ) -> Optional[str]:
        if depth > 32:
            return None
        question = self._top_level_conditional_question(expression)
        if question is not None:
            colon = self._matching_conditional_colon(expression, question)
            if colon is None:
                return None
            condition = self._rewrite_static_integral_conditional_expressions(
                expression[:question], depth=depth + 1
            )
            true_value = self._rewrite_static_integral_conditional_expressions(
                expression[question + 1 : colon], depth=depth + 1
            )
            false_value = self._rewrite_static_integral_conditional_expressions(
                expression[colon + 1 :], depth=depth + 1
            )
            if condition is None or true_value is None or false_value is None:
                return None
            if (
                not condition.strip()
                or not true_value.strip()
                or not false_value.strip()
            ):
                return None
            return f"(({true_value}) if ({condition}) else ({false_value}))"

        rewritten: List[str] = []
        cursor = 0
        while cursor < len(expression):
            if expression[cursor] in "\"'":
                literal, consumed = self._read_string(expression, cursor)
                rewritten.append(literal)
                cursor += consumed
                continue
            if expression[cursor] != "(":
                rewritten.append(expression[cursor])
                cursor += 1
                continue
            close = self._find_matching_delimiter(expression, cursor, "(", ")")
            if close is None:
                return None
            inner = self._rewrite_static_integral_conditional_expressions(
                expression[cursor + 1 : close], depth=depth + 1
            )
            if inner is None:
                return None
            rewritten.append(f"({inner})")
            cursor = close + 1
        return "".join(rewritten)

    def _top_level_conditional_question(self, expression: str) -> Optional[int]:
        depth = 0
        i = 0
        while i < len(expression):
            if expression[i] in "\"'":
                _literal, consumed = self._read_string(expression, i)
                i += consumed
                continue
            if expression[i] in "([{":
                depth += 1
            elif expression[i] in ")]}" and depth:
                depth -= 1
            elif expression[i] == "?" and depth == 0:
                return i
            i += 1
        return None

    def _matching_conditional_colon(
        self, expression: str, question: int
    ) -> Optional[int]:
        depth = 0
        nested_conditionals = 0
        i = question + 1
        while i < len(expression):
            if expression[i] in "\"'":
                _literal, consumed = self._read_string(expression, i)
                i += consumed
                continue
            token = expression[i]
            if token in "([{":
                depth += 1
            elif token in ")]}" and depth:
                depth -= 1
            elif depth == 0 and token == "?":
                nested_conditionals += 1
            elif depth == 0 and token == ":":
                if (i > 0 and expression[i - 1] == ":") or (
                    i + 1 < len(expression) and expression[i + 1] == ":"
                ):
                    i += 1
                    continue
                if nested_conditionals:
                    nested_conditionals -= 1
                else:
                    return i
            i += 1
        return None

    def _evaluate_static_integral_node(
        self, node: ast.AST
    ) -> Tuple[bool, Optional[int]]:
        if isinstance(node, ast.Constant):
            if type(node.value) not in {bool, int}:
                return False, None
            value = int(node.value)
            if not self._static_fold_value_is_bounded(value):
                return False, None
            return True, value

        if isinstance(node, ast.UnaryOp):
            valid, operand = self._evaluate_static_integral_node(node.operand)
            if not valid or operand is None:
                return False, None
            if isinstance(node.op, ast.UAdd):
                value = operand
            elif isinstance(node.op, ast.USub):
                value = -operand
            elif isinstance(node.op, ast.Not):
                return True, int(not operand)
            else:
                return False, None
            if not self._static_fold_value_is_bounded(value):
                return False, None
            return True, value

        if isinstance(node, ast.BinOp):
            left_valid, left = self._evaluate_static_integral_node(node.left)
            right_valid, right = self._evaluate_static_integral_node(node.right)
            if not left_valid or not right_valid or left is None or right is None:
                return False, None
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            elif isinstance(node.op, (ast.Div, ast.Mod)):
                if right == 0:
                    return False, None
                quotient = abs(left) // abs(right)
                if (left < 0) != (right < 0):
                    quotient = -quotient
                value = (
                    quotient
                    if isinstance(node.op, ast.Div)
                    else left - quotient * right
                )
            elif isinstance(node.op, (ast.LShift, ast.RShift)):
                if left < 0 or right < 0 or right >= 31:
                    return False, None
                value = (
                    left << right if isinstance(node.op, ast.LShift) else left >> right
                )
            elif isinstance(node.op, ast.BitAnd):
                if left < 0 or right < 0:
                    return False, None
                value = left & right
            elif isinstance(node.op, ast.BitOr):
                if left < 0 or right < 0:
                    return False, None
                value = left | right
            elif isinstance(node.op, ast.BitXor):
                if left < 0 or right < 0:
                    return False, None
                value = left ^ right
            else:
                return False, None
            if not self._static_fold_value_is_bounded(value):
                return False, None
            return True, value

        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                for item in node.values:
                    valid, value = self._evaluate_static_integral_node(item)
                    if not valid or value is None:
                        return False, None
                    if not value:
                        return True, 0
                return True, 1
            if isinstance(node.op, ast.Or):
                for item in node.values:
                    valid, value = self._evaluate_static_integral_node(item)
                    if not valid or value is None:
                        return False, None
                    if value:
                        return True, 1
                return True, 0
            return False, None

        if isinstance(node, ast.IfExp):
            valid, condition = self._evaluate_static_integral_node(node.test)
            if not valid or condition is None:
                return False, None
            body_valid, body = self._evaluate_static_integral_node(node.body)
            else_valid, else_value = self._evaluate_static_integral_node(node.orelse)
            if not body_valid or body is None or not else_valid or else_value is None:
                return False, None
            return True, body if condition else else_value

        if isinstance(node, ast.Compare):
            valid, left = self._evaluate_static_integral_node(node.left)
            if not valid or left is None:
                return False, None
            comparisons = {
                ast.Eq: operator.eq,
                ast.NotEq: operator.ne,
                ast.Lt: operator.lt,
                ast.LtE: operator.le,
                ast.Gt: operator.gt,
                ast.GtE: operator.ge,
            }
            for op_node, comparator in zip(node.ops, node.comparators):
                valid, right = self._evaluate_static_integral_node(comparator)
                comparison = comparisons.get(type(op_node))
                if not valid or right is None or comparison is None:
                    return False, None
                if not comparison(left, right):
                    return True, 0
                left = right
            return True, 1

        return False, None

    def _static_fold_value_is_bounded(self, value: int) -> bool:
        return self._STATIC_FOLD_MIN_INT <= value <= self._STATIC_FOLD_MAX_INT

    # ------------------------------------------------------------------ #
    # Pointer-member promotion (device/threadgroup pointer members).      #
    # ------------------------------------------------------------------ #

    def _promoted_pointer_param_name(self, member_name: str) -> str:
        # A fresh, collision-proof name for a promoted pointer parameter. Using a
        # prefix (rather than the raw member name) avoids clashing with a local
        # in the method body and sidesteps backends where `in`/`out` are reserved.
        return f"crosstl_ptr_{member_name}"

    def _pointer_promotion_plan(
        self, struct: _MetalStructDefinition
    ) -> Optional[_PointerPromotionPlan]:
        # Decide whether `struct` can have its pointer members promoted out (see
        # _PointerPromotionPlan). Returns a plan, or None to leave the struct on
        # the ordinary self-passing lowering path. Every check is conservative:
        # any shape we are not certain we can rewrite faithfully bails to None so
        # a struct is never translated worse than it is today.
        data_members = struct.data_members
        if not data_members:
            return None
        pointer_members = [member for member in data_members if member.is_pointer]
        if not pointer_members:
            return None
        # A pointer member captured as an array (`device T* taps[4]`) or a
        # template/operator-call struct is out of scope.
        if any(member.is_array for member in pointer_members):
            return None
        if struct.template_methods or struct.has_operator_call:
            return None
        # Exactly one constructor, which must source every pointer member from a
        # pointer parameter, so a construction argument is the pointer expression.
        if len(struct.constructors) != 1:
            return None
        constructor = struct.constructors[0]
        if constructor.span is None:
            return None
        pointer_member_names = {member.name for member in pointer_members}
        pointer_ctor_arg_indices: List[int] = []
        for member in pointer_members:
            init_expr = constructor.init_map.get(member.name)
            if init_expr is None:
                return None
            init_expr = init_expr.strip()
            if init_expr not in constructor.param_names:
                return None
            index = constructor.param_names.index(init_expr)
            if index >= len(constructor.param_types):
                return None
            if "*" not in constructor.param_types[index]:
                return None
            pointer_ctor_arg_indices.append(index)
        # The constructor body must not reference any pointer parameter or pointer
        # member (both are about to disappear); otherwise dropping them is unsafe.
        pointer_param_names = {
            constructor.param_names[index] for index in pointer_ctor_arg_indices
        }
        body_identifiers = set(IDENTIFIER_RE.findall(constructor.body))
        if body_identifiers & pointer_param_names:
            return None
        if body_identifiers & pointer_member_names:
            return None
        # No method may declare a parameter or local that collides with a pointer
        # member name (the promoted parameter would shadow / be shadowed by it).
        for method in struct.methods:
            if method.is_static:
                continue
            names = set(method.parameter_names)
            names.update(self._local_variable_names(method.body))
            if names & pointer_member_names:
                return None
        return _PointerPromotionPlan(
            struct_name=struct.name,
            pointer_members=pointer_members,
            pointer_ctor_arg_indices=pointer_ctor_arg_indices,
            constructor=constructor,
        )

    def _render_promoted_struct(
        self,
        code: str,
        struct: _MetalStructDefinition,
        plan: _PointerPromotionPlan,
    ) -> Tuple[str, List[str]]:
        # Produce (data_only_struct_text, [free_function_text, ...]) for a struct
        # whose pointer members are promoted. The residual struct keeps only its
        # SCALAR members (regenerated from captured metadata, in declaration
        # order) plus a constructor rewritten to drop the pointer parameters and
        # their initializer-list entries; every method is re-emitted as a free
        # function that takes `self` plus the promoted pointer parameters.
        #
        # The header (up to and including `{`, so any base clause is preserved) is
        # taken verbatim from the source; `body_span[0]` is the first body
        # character, i.e. just past the opening brace.
        header = code[struct.span[0] : struct.body_span[0]].rstrip()
        member_lines: List[str] = []
        for member in struct.data_members:
            if member.is_pointer:
                continue
            declaration = f"  {member.type_text} {member.name}{member.array_suffix}"
            if member.default is not None:
                declaration += f" = {member.default}"
            declaration += ";"
            member_lines.append(declaration)
        rebuilt_constructor = self._rebuild_promoted_constructor(plan)
        body_parts = "\n".join(member_lines)
        if body_parts:
            body_parts += "\n"
        data_only = f"{header}\n{body_parts}  {rebuilt_constructor}\n}};"
        free_functions = [
            self._emit_promoted_free_function(struct, method, plan)
            for method in struct.methods
        ]
        return data_only, free_functions

    def _rebuild_promoted_constructor(self, plan: _PointerPromotionPlan) -> str:
        # Rebuild the constructor without its pointer parameters or their
        # initializer-list entries; the scalar parameters, remaining initializers
        # and the body (which computes scalar members) are preserved verbatim.
        constructor = plan.constructor
        drop_indices = set(plan.pointer_ctor_arg_indices)
        kept_params: List[str] = []
        params = [
            param.strip()
            for param in self._split_top_level_commas(constructor.params_text)
        ]
        for index, param in enumerate(params):
            if not param:
                continue
            if index in drop_indices:
                continue
            kept_params.append(param)
        pointer_member_names = set(plan.pointer_member_names)
        kept_inits: List[str] = []
        if constructor.init_text:
            for item in self._split_top_level_commas(constructor.init_text):
                item = item.strip()
                if not item:
                    continue
                match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*[\({]", item)
                if match is not None and match.group(1) in pointer_member_names:
                    continue
                kept_inits.append(item)
        rebuilt = f"{constructor.prefix}{', '.join(kept_params)})"
        if kept_inits:
            rebuilt += " : " + ", ".join(kept_inits)
        rebuilt += f" {{{constructor.body}}}"
        return rebuilt

    def _emit_promoted_free_function(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        plan: _PointerPromotionPlan,
    ) -> str:
        # Emit an instance helper with a reference receiver and promoted pointer
        # parameters, or the ordinary static form for a static method.
        return_type = self._canonicalize_struct_scoped_type(
            method.return_type, struct, None
        )
        rewritten_body = self._rewrite_promoted_method_body(struct, method, plan)
        rewritten_body = self._rewrite_promoted_internal_method_calls(
            struct,
            plan,
            rewritten_body,
        )
        rewritten_body = self._canonicalize_struct_scoped_local_declarations(
            rewritten_body, struct, None
        )
        rewritten_body = self._canonicalize_struct_scoped_cast_types(
            rewritten_body, struct, None
        )
        params = self._canonicalize_struct_scoped_parameters(
            method.parameters, struct, None
        )
        if method.is_static:
            new_params = params if params and params != "void" else ""
        else:
            parts = [self._instance_receiver_parameter(struct, method)]
            for member in plan.pointer_members:
                member_type = self._canonicalize_struct_scoped_type(
                    member.type_text, struct, None
                )
                parts.append(
                    f"{member_type} "
                    f"{self._promoted_pointer_param_name(member.name)}"
                )
            if params and params != "void":
                parts.append(params)
            new_params = ", ".join(parts)
        return (
            f"{return_type} {method.free_name}({new_params}) " f"{{{rewritten_body}}}"
        )

    def _rewrite_promoted_internal_method_calls(
        self,
        struct: _MetalStructDefinition,
        plan: _PointerPromotionPlan,
        body: str,
    ) -> str:
        """Forward the implicit receiver and promoted pointers to sibling helpers."""
        siblings: Dict[str, List[_MetalStructMethod]] = {}
        for method in struct.methods:
            siblings.setdefault(method.name, []).append(method)

        result: List[str] = []
        index = 0
        while index < len(body):
            if body[index] in "\"'":
                literal, consumed = self._read_string(body, index)
                result.append(literal)
                index += consumed
                continue
            if body.startswith("//", index):
                end = body.find("\n", index)
                if end == -1:
                    result.append(body[index:])
                    break
                result.append(body[index:end])
                index = end
                continue
            if body.startswith("/*", index):
                end = body.find("*/", index + 2)
                if end == -1:
                    result.append(body[index:])
                    break
                result.append(body[index : end + 2])
                index = end + 2
                continue
            if not (body[index].isalpha() or body[index] == "_"):
                result.append(body[index])
                index += 1
                continue

            ident, consumed = self._read_identifier(body, index)
            ident_end = index + consumed
            rewrite = self._try_rewrite_promoted_internal_method_call(
                struct,
                plan,
                body,
                ident,
                index,
                ident_end,
                siblings,
            )
            if rewrite is not None:
                end, replacement = rewrite
                result.append(replacement)
                index = end
                continue
            result.append(ident)
            index = ident_end
        return "".join(result)

    def _try_rewrite_promoted_internal_method_call(
        self,
        struct: _MetalStructDefinition,
        plan: _PointerPromotionPlan,
        body: str,
        ident: str,
        ident_start: int,
        ident_end: int,
        siblings: Dict[str, List[_MetalStructMethod]],
    ) -> Optional[Tuple[int, str]]:
        method_name = ident
        method_name_end = ident_end

        if ident == "self":
            cursor = ident_end
            while cursor < len(body) and body[cursor].isspace():
                cursor += 1
            if body.startswith("->", cursor):
                cursor += 2
            elif cursor < len(body) and body[cursor] == ".":
                cursor += 1
            else:
                return None
            while cursor < len(body) and body[cursor].isspace():
                cursor += 1
            if cursor >= len(body) or not (
                body[cursor].isalpha() or body[cursor] == "_"
            ):
                return None
            method_name, consumed = self._read_identifier(body, cursor)
            method_name_end = cursor + consumed
        else:
            previous = ident_start - 1
            while previous >= 0 and body[previous].isspace():
                previous -= 1
            if previous >= 0 and (
                body[previous] == "."
                or (previous >= 1 and body[previous - 1 : previous + 1] in {"->", "::"})
            ):
                return None

        if method_name == "operator":
            empty_open = method_name_end
            while empty_open < len(body) and body[empty_open].isspace():
                empty_open += 1
            if empty_open >= len(body) or body[empty_open] != "(":
                return None
            empty_close = self._find_matching_delimiter(body, empty_open, "(", ")")
            if empty_close is None or body[empty_open + 1 : empty_close].strip():
                return None
            method_name = "operator()"
            method_name_end = empty_close + 1

        overloads = siblings.get(method_name, [])
        if len(overloads) != 1:
            return None
        call_suffix = self._member_template_call_suffix(body, method_name_end)
        if call_suffix is None:
            return None
        arg_open, explicit_template_arguments = call_suffix
        if explicit_template_arguments is not None:
            return None
        arg_close = self._find_matching_delimiter(body, arg_open, "(", ")")
        if arg_close is None:
            return None

        sibling = overloads[0]
        self._reject_reference_returning_method_call(
            body,
            struct,
            sibling,
            arg_open,
            arg_close,
        )
        raw_args = body[arg_open + 1 : arg_close].strip()
        forwarded: List[str] = []
        if not sibling.is_static:
            forwarded.append("self")
            forwarded.extend(
                self._promoted_pointer_param_name(member.name)
                for member in plan.pointer_members
            )
        if raw_args:
            forwarded.append(raw_args)
        return arg_close + 1, f"{sibling.free_name}({', '.join(forwarded)})"

    def _rewrite_promoted_method_body(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        plan: _PointerPromotionPlan,
    ) -> str:
        # Like _rewrite_method_body, but pointer members resolve to the promoted
        # pointer PARAMETER (bare name) rather than `self.<member>`; scalar
        # members still resolve to `self.<member>`.
        body = method.body
        body = re.sub(r"\(\s*\*\s*this\s*\)\s*(?=\.|->)", "self", body)
        body = re.sub(r"\bthis\s*->\s*", "self.", body)
        body = re.sub(r"\bthis\s*\.\s*", "self.", body)
        body = re.sub(r"\bthis\b", "self", body)
        body = self._substitute_static_data_member_initializers(struct, method, body)
        if method.is_static:
            return body
        pointer_member_names = plan.pointer_member_names
        # `self.<ptr>` (typically from a normalized `this->ptr`) becomes the bare
        # promoted parameter.
        for name in pointer_member_names:
            body = re.sub(
                rf"\bself\s*\.\s*{re.escape(name)}\b",
                self._promoted_pointer_param_name(name),
                body,
            )
        shadowed = set(method.parameter_names)
        shadowed.update(self._local_variable_names(body))
        pointer_name_set = set(pointer_member_names)
        mapping: Dict[str, str] = {}
        for name in struct.data_member_names:
            if name in shadowed:
                continue
            if name in pointer_name_set:
                mapping[name] = self._promoted_pointer_param_name(name)
            else:
                mapping[name] = f"self.{name}"
        if not mapping:
            return body
        return self._substitute_bare_member_references(body, mapping)

    def _substitute_bare_member_references(
        self, body: str, mapping: Dict[str, str]
    ) -> str:
        # Walk identifiers and replace bare member references per `mapping`, while
        # leaving member accesses on OTHER objects (`obj.x`, `obj->x`), already
        # rewritten `self.x`, and function-name uses untouched. Generalizes
        # `_qualify_member_references` to an arbitrary member -> replacement map.
        result: List[str] = []
        i = 0
        n = len(body)
        while i < n:
            ch = body[i]
            if ch in "\"'":
                literal, consumed = self._read_string(body, i)
                result.append(literal)
                i += consumed
                continue
            if body.startswith("//", i):
                end = body.find("\n", i)
                if end == -1:
                    result.append(body[i:])
                    break
                result.append(body[i:end])
                i = end
                continue
            if body.startswith("/*", i):
                end = body.find("*/", i + 2)
                if end == -1:
                    result.append(body[i:])
                    break
                result.append(body[i : end + 2])
                i = end + 2
                continue
            if ch.isalpha() or ch == "_":
                ident, consumed = self._read_identifier(body, i)
                replacement = mapping.get(ident)
                if (
                    replacement is not None
                    and not self._is_member_identifier_context(body, i)
                    and not self._identifier_is_declaration_or_call(
                        body, i, i + consumed
                    )
                ):
                    result.append(replacement)
                else:
                    result.append(ident)
                i += consumed
                continue
            result.append(ch)
            i += 1
        return "".join(result)

    def _rewrite_method_body(
        self, struct: _MetalStructDefinition, method: _MetalStructMethod
    ) -> str:
        # Rewrite member references inside a method body to `self.<member>`:
        #   - `this->x` / `this.x` / `(*this).x` -> `self.x`; bare `this` -> `self`
        #   - a bare identifier equal to a data-member name that is NOT a
        #     parameter and NOT a local variable declared in the body -> self.x
        # For a static method there is no `self`, so only the `this` forms (which
        # cannot legally appear) are normalized and bare members are left as-is.
        body = method.body
        # Normalize the various `this` spellings to a single `self` token first.
        body = re.sub(r"\(\s*\*\s*this\s*\)\s*(?=\.|->)", "self", body)
        body = re.sub(r"\bthis\s*->\s*", "self.", body)
        body = re.sub(r"\bthis\s*\.\s*", "self.", body)
        body = re.sub(r"\bthis\b", "self", body)
        body = self._substitute_static_data_member_initializers(struct, method, body)
        if method.is_static:
            return body

        shadowed = set(method.parameter_names)
        shadowed.update(self._local_variable_names(body))
        members = struct.data_member_names - shadowed
        if not members:
            return body
        return self._qualify_member_references(body, members)

    def _qualify_member_references(self, body: str, members: Set[str]) -> str:
        # Walk identifiers and prefix bare member references with `self.`, while
        # leaving member accesses on OTHER objects (`obj.x`, `obj->x`) and
        # already-qualified `self.x` untouched.
        result: List[str] = []
        i = 0
        n = len(body)
        while i < n:
            ch = body[i]
            if ch in "\"'":
                literal, consumed = self._read_string(body, i)
                result.append(literal)
                i += consumed
                continue
            if body.startswith("//", i):
                end = body.find("\n", i)
                if end == -1:
                    result.append(body[i:])
                    break
                result.append(body[i:end])
                i = end
                continue
            if body.startswith("/*", i):
                end = body.find("*/", i + 2)
                if end == -1:
                    result.append(body[i:])
                    break
                result.append(body[i : end + 2])
                i = end + 2
                continue
            if ch.isalpha() or ch == "_":
                ident, consumed = self._read_identifier(body, i)
                if (
                    ident in members
                    and not self._is_member_identifier_context(body, i)
                    and not self._identifier_is_declaration_or_call(
                        body, i, i + consumed
                    )
                ):
                    result.append(f"self.{ident}")
                else:
                    result.append(ident)
                i += consumed
                continue
            result.append(ch)
            i += 1
        return "".join(result)

    def _identifier_is_declaration_or_call(
        self, body: str, start: int, end: int
    ) -> bool:
        # A member name is left untouched when it is used as a function name
        # (`member(...)`) — a data member of struct type is not callable in our
        # model, but a same-named free function might be — to stay conservative.
        j = end
        while j < len(body) and body[j].isspace():
            j += 1
        return j < len(body) and body[j] == "("

    def _local_variable_names(self, body: str) -> Set[str]:
        # Collect identifiers introduced as locals inside a method body so member
        # references shadowed by a local are not rewritten to `self.x`. This is a
        # best-effort scan of `Type name ...;` / `Type name = ...;` declarations
        # at any brace depth, plus simple `for (Type name ...)` headers.
        names: Set[str] = set()
        for statement in self._iter_simple_declarations(body):
            name = self._declared_local_name(statement)
            if name:
                names.add(name)
        return names

    def _iter_simple_declarations(self, body: str) -> List[str]:
        # Yield candidate declaration statements: the text since the previous
        # statement/scope boundary up to each top-level (relative) `;` or the
        # initializer clauses of `for (...)` headers.
        statements: List[str] = []
        i = 0
        n = len(body)
        segment_start = 0
        paren_depth = 0
        while i < n:
            ch = body[i]
            if ch in "\"'":
                _literal, consumed = self._read_string(body, i)
                i += consumed
                continue
            if body.startswith("//", i):
                end = body.find("\n", i)
                i = n if end == -1 else end + 1
                segment_start = i
                continue
            if body.startswith("/*", i):
                end = body.find("*/", i + 2)
                i = n if end == -1 else end + 2
                segment_start = i
                continue
            if ch == "(":
                # Capture `for (` initializer declarations specially.
                preceding = body[segment_start:i]
                if re.search(r"\bfor\s*$", preceding):
                    close = self._find_matching_delimiter(body, i, "(", ")")
                    if close is not None:
                        header = body[i + 1 : close]
                        first_clause = header.split(";", 1)[0]
                        statements.append(first_clause)
                        i = close + 1
                        segment_start = i
                        continue
                paren_depth += 1
                i += 1
                continue
            if ch == ")":
                paren_depth = max(0, paren_depth - 1)
                i += 1
                continue
            if paren_depth == 0 and ch in ";{}":
                statements.append(body[segment_start:i])
                i += 1
                segment_start = i
                continue
            i += 1
        if segment_start < n:
            statements.append(body[segment_start:n])
        return statements

    def _declared_local_name(self, statement: str) -> Optional[str]:
        # Recognize `Type name`, `Type name = ...`, `Type name(...)` where the
        # leading token(s) look like a type. Reject pure expressions/assignments.
        text = statement.strip()
        if not text:
            return None
        # Strip an initializer to isolate the declarator.
        declarator = self._strip_top_level_default_value(text)
        # A call/paren-initialized declarator: `Type name(args)`.
        paren = self._function_parameter_start(declarator)
        if paren is not None:
            declarator = declarator[:paren].rstrip()
        # Drop trailing array extents.
        while declarator.endswith("]"):
            open_bracket = declarator.rfind("[")
            if open_bracket == -1:
                break
            declarator = declarator[:open_bracket].rstrip()
        tokens = IDENTIFIER_RE.findall(declarator)
        # Need at least a type token and a name token; a single token is an
        # expression (e.g. `i++` stripped) rather than a declaration.
        if len(tokens) < 2:
            return None
        # Reject obvious non-declarations (control-flow keywords as the leader).
        if tokens[0] in {"return", "if", "else", "while", "for", "switch", "do"}:
            return None
        name_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", declarator)
        if name_match is None:
            return None
        return name_match.group(1)

    def _declared_local_type(self, statement: str, name: str) -> Optional[str]:
        declarator = self._strip_top_level_default_value(statement.strip())
        paren = self._function_parameter_start(declarator)
        if paren is not None:
            declarator = declarator[:paren].rstrip()
        while declarator.endswith("]"):
            open_bracket = declarator.rfind("[")
            if open_bracket == -1:
                break
            declarator = declarator[:open_bracket].rstrip()
        type_text = re.sub(rf"\b{re.escape(name)}\s*$", "", declarator).strip()
        normalized = self._normalize_inferred_type(type_text)
        return normalized or None

    def _rewrite_struct_member_call_sites(
        self,
        code: str,
        struct_names: Set[str],
        methods_by_struct: Dict[str, Dict[str, _MetalStructMethod]],
        template_methods_by_struct: Dict[str, Dict[str, List[_MetalStructMethod]]],
        operator_call_structs: Set[str],
        struct_spans: List[Tuple[int, int]],
        structs_by_name: Dict[str, _MetalStructDefinition],
        instantiated_template_functions: Dict[str, str],
        field_structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
        all_struct_spans: Optional[List[Tuple[int, int]]] = None,
        all_struct_names: Optional[Set[str]] = None,
    ) -> List[Tuple[int, int, str]]:
        # Rewrite call sites across the source (outside the struct definitions we
        # are replacing):
        #   var.m(args)  -> S__m(var, args)            (instance method)
        #   var(args)    -> S__operator_call(var, ...)  (var has operator())
        #   S::m(args)   -> S__m(args)                  (qualified static call)
        # A template member method call is instantiated from its argument types
        # and rewritten to a concrete free function. Local variable struct types
        # are tracked per the declarations that introduce them; scalar/vector
        # local types and buffer element types feed template-argument inference.
        # All struct/union spans and type names are needed so member-access field
        # resolution sees method-less carriers too; fall back to the method-driven
        # views when the caller does not provide them.
        field_structs_by_name = field_structs_by_name or structs_by_name
        all_struct_spans = (
            all_struct_spans if all_struct_spans is not None else struct_spans
        )
        all_struct_names = (
            all_struct_names if all_struct_names is not None else struct_names
        )
        # A materializer leaves the ORIGINAL template declarations in place after
        # emitting their concrete instantiations. Those residual template bodies
        # still reference unbound template parameters (`U a`, `static_cast<U>(b)`),
        # so their aliases and call sites must not participate in concrete
        # inference. Concrete and template spans are disjoint.
        scan_skip_spans = all_struct_spans
        template_declaration_spans = self._find_template_declaration_spans(code)
        if template_declaration_spans:
            scan_skip_spans = sorted(all_struct_spans + template_declaration_spans)
        # Member specialization uses the same proven lexical constants as nested
        # struct materialization; unresolved or runtime bindings remain absent.
        local_constant_owner_spans = [
            function.body_span
            for function in self._find_non_template_function_definitions(
                code,
                scan_skip_spans,
            )
        ]
        local_constant_type_aliases = self._collect_local_type_alias_bindings(
            code,
            local_constant_owner_spans,
        )
        local_integral_constants = self._collect_local_integral_constant_bindings(
            code,
            local_constant_owner_spans,
            local_constant_type_aliases,
        )
        struct_type_aliases = self._collect_struct_type_aliases(
            code,
            all_struct_names,
            scan_skip_spans,
            field_structs_by_name,
        )

        variable_types = self._collect_struct_variable_types(
            code, struct_names, struct_spans
        )
        # Struct-typed locals for EVERY struct/union type (used for `obj.member`
        # inference), resolved over the full struct span set so member-access
        # works for data carriers without methods.
        field_variable_types = self._collect_struct_variable_types(
            code,
            all_struct_names,
            all_struct_spans,
            include_indirect=True,
        )
        aliased_variable_types = self._collect_aliased_struct_variable_types(
            code, struct_type_aliases, scan_skip_spans
        )
        for name, entries in aliased_variable_types.items():
            field_entries = field_variable_types.setdefault(name, [])
            field_entries.extend(
                entry for entry in entries if entry not in field_entries
            )
            field_entries.sort(key=lambda item: item[0])

            method_entries = variable_types.setdefault(name, [])
            method_entries.extend(
                entry
                for entry in entries
                if entry[1] in struct_names and entry not in method_entries
            )
            method_entries.sort(key=lambda item: item[0])
        buffer_element_types = self._collect_buffer_element_types(
            code, all_struct_spans
        )
        local_variable_types = self._collect_local_variable_types(
            code, all_struct_spans
        )
        for name, entries in field_variable_types.items():
            local_entries = local_variable_types.setdefault(name, [])
            for entry in entries:
                if entry not in local_entries:
                    local_entries.append(entry)
            local_entries.sort(key=lambda item: item[0])
        # Fold the enclosing functions' parameters into the same position-ordered
        # maps so a call argument that is a parameter (or a subscript of one) is
        # inferable.
        self._collect_function_parameter_types(
            code,
            all_struct_spans,
            buffer_element_types,
            local_variable_types,
        )
        # Type `auto` locals from their initializer LAST, so the inference can use
        # every buffer/parameter/scalar/struct local collected above (and the
        # tracked struct set for construction / functor-call initializers).
        self._collect_auto_local_variable_types(
            code,
            all_struct_spans,
            buffer_element_types,
            local_variable_types,
            field_structs_by_name,
        )
        replacements: List[Tuple[int, int, str]] = []
        i = 0
        n = len(code)
        while i < n:
            ch = code[i]
            if ch in "\"'":
                _literal, consumed = self._read_string(code, i)
                i += consumed
                continue
            if code.startswith("//", i):
                end = code.find("\n", i)
                i = n if end == -1 else end + 1
                continue
            if code.startswith("/*", i):
                end = code.find("*/", i + 2)
                i = n if end == -1 else end + 2
                continue
            span = self._containing_span(i, scan_skip_spans)
            if span is not None:
                i = span[1]
                continue
            if ch.isalpha() or ch == "_":
                ident, consumed = self._read_identifier(code, i)
                ident_end = i + consumed
                rewrite = self._try_rewrite_call_at(
                    code,
                    i,
                    ident,
                    ident_end,
                    struct_names,
                    methods_by_struct,
                    template_methods_by_struct,
                    operator_call_structs,
                    variable_types,
                    structs_by_name,
                    buffer_element_types,
                    local_variable_types,
                    instantiated_template_functions,
                    field_variable_types=field_variable_types,
                    field_structs_by_name=field_structs_by_name,
                    struct_type_aliases=struct_type_aliases,
                    local_integral_constants=local_integral_constants,
                    type_aliases=self._source_type_alias_bindings,
                )
                if rewrite is not None:
                    end, replacement = rewrite
                    replacements.append((i, end, replacement))
                    i = end
                    continue
                i = ident_end
                continue
            i += 1
        return replacements

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
        if qualified_struct is not None and code[j : j + 2] == "::":
            k = j + 2
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
                    require_static=True,
                    type_alias_fallback_position=type_alias_fallback_position,
                )
            if method is not None and explicit_template_arguments is None:
                arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
                if arg_close is None:
                    return None
                self._reject_reference_returning_method_call(
                    code,
                    structs_by_name.get(qualified_struct, qualified_struct),
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
                require_static=False,
                type_alias_fallback_position=type_alias_fallback_position,
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

        # Resolve the variable's struct type from the NEAREST declaration at or
        # before this call site (deterministic; correct across sibling kernels
        # that reuse a variable name for different functor types).
        struct_type = self._resolve_declared_type_at(variable_types, ident, ident_start)
        if struct_type is None:
            return None

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
            owner_struct = structs_by_name.get(struct_type)
            reference_methods = (
                self._concrete_reference_methods(owner_struct, member)
                if owner_struct is not None
                else []
            )
            if reference_methods and explicit_template_arguments is None:
                receiver_is_const = self._simple_receiver_constness(
                    code,
                    struct_type,
                    ident,
                    ident_start,
                    type_aliases=struct_type_aliases,
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
                    receiver=ident,
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
                    require_static=False,
                    type_alias_fallback_position=type_alias_fallback_position,
                )
            if method is not None and explicit_template_arguments is None:
                return self._build_instance_call_rewrite(
                    code,
                    ident,
                    method,
                    arg_open,
                    struct_name=structs_by_name.get(struct_type, struct_type),
                )
            # An instance TEMPLATE member call `var.m(args)`.
            if template_overloads:
                return self._instantiate_template_member_call(
                    code,
                    structs_by_name[struct_type],
                    template_overloads,
                    receiver=ident,
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
                    require_static=False,
                    type_alias_fallback_position=type_alias_fallback_position,
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
                    struct_name=structs_by_name.get(struct_type, struct_type),
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

    def _try_rewrite_nested_member_call_at(
        self,
        code: str,
        ident_start: int,
        ident: str,
        ident_end: int,
        methods_by_struct: Dict[str, Dict[str, _MetalStructMethod]],
        template_methods_by_struct: Dict[str, Dict[str, List[_MetalStructMethod]]],
        operator_call_structs: Set[str],
        variable_types: Dict[str, List[Tuple[int, str]]],
        structs_by_name: Dict[str, _MetalStructDefinition],
        buffer_element_types: _MetalPositionedBufferTypes,
        local_variable_types: Dict[str, List[Tuple[int, str]]],
        instantiated_template_functions: Dict[str, str],
        field_variable_types: Dict[str, List[Tuple[int, str]]],
        field_structs_by_name: Dict[str, _MetalStructDefinition],
        local_integral_constants: Optional[
            Dict[str, List[_MetalIntegralConstantBinding]]
        ],
        struct_type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]],
        type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]],
        type_alias_fallback_position: Optional[int],
    ) -> Optional[Tuple[int, str]]:
        """Rewrite a method call whose receiver is a nested struct field."""
        if self._is_member_identifier_context(code, ident_start):
            return None

        declared_type = self._resolve_declared_type_at(
            field_variable_types, ident, ident_start
        ) or self._resolve_declared_type_at(variable_types, ident, ident_start)
        receiver_info = self._nested_member_receiver_info(
            declared_type, field_structs_by_name
        )
        if receiver_info is None:
            return None
        current_struct_name, required_access = receiver_info

        receiver_end = ident_end
        cursor = ident_end
        while True:
            while cursor < len(code) and code[cursor].isspace():
                cursor += 1
            if code.startswith("->", cursor):
                access = "->"
                cursor += 2
            elif cursor < len(code) and code[cursor] == ".":
                access = "."
                cursor += 1
            else:
                return None
            if access != required_access:
                return None

            while cursor < len(code) and code[cursor].isspace():
                cursor += 1
            member, consumed = self._read_identifier(code, cursor)
            if not member:
                return None
            cursor += consumed
            explicit_template_marker = member == "template"
            if explicit_template_marker:
                while cursor < len(code) and code[cursor].isspace():
                    cursor += 1
                member, consumed = self._read_identifier(code, cursor)
                if not member:
                    return None
                cursor += consumed

            call_suffix = self._member_template_call_suffix(code, cursor)
            if call_suffix is not None:
                arg_open, explicit_template_arguments = call_suffix
                receiver = code[ident_start:receiver_end].strip()
                if access == "->":
                    receiver = f"{receiver}[0]"
                owner_struct = structs_by_name.get(current_struct_name)
                reference_methods = (
                    self._concrete_reference_methods(owner_struct, member)
                    if owner_struct is not None
                    else []
                )
                if (
                    reference_methods
                    and explicit_template_arguments is None
                    and receiver == ident
                    and access == "."
                ):
                    method = self._select_direct_reference_method(
                        code,
                        owner_struct,
                        reference_methods,
                        arg_open,
                        receiver_is_const=self._simple_receiver_constness(
                            code,
                            current_struct_name,
                            ident,
                            ident_start,
                            type_aliases=struct_type_aliases,
                        ),
                    )
                    return self._build_direct_reference_accessor_rewrite(
                        code,
                        owner_struct,
                        method,
                        receiver=receiver,
                        arg_open=arg_open,
                        call_start=ident_start,
                    )
                template_overloads = template_methods_by_struct.get(
                    current_struct_name, {}
                ).get(member)
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
                        require_static=False,
                        type_alias_fallback_position=type_alias_fallback_position,
                    )
                if method is not None and explicit_template_arguments is None:
                    return self._build_instance_call_rewrite(
                        code,
                        receiver,
                        method,
                        arg_open,
                        struct_name=structs_by_name.get(
                            current_struct_name,
                            current_struct_name,
                        ),
                    )
                if template_overloads:
                    struct = structs_by_name.get(current_struct_name)
                    if struct is None:
                        return None
                    return self._instantiate_template_member_call(
                        code,
                        struct,
                        template_overloads,
                        receiver=receiver,
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
            if explicit_template_marker:
                return None

            current_struct = field_structs_by_name.get(current_struct_name)
            if current_struct is None:
                return None
            field_type = current_struct.data_member_types.get(member)
            receiver_info = self._nested_member_receiver_info(
                field_type, field_structs_by_name
            )
            if receiver_info is None:
                return None
            current_struct_name, required_access = receiver_info
            receiver_end = cursor

    def _nested_member_receiver_info(
        self,
        type_text: Optional[str],
        structs_by_name: Dict[str, _MetalStructDefinition],
    ) -> Optional[Tuple[str, str]]:
        """Return a concrete receiver struct and its required access operator."""
        normalized = self._normalize_inferred_type(type_text or "")
        if not normalized:
            return None
        indirection_match = re.search(r"(?P<indirection>[*&]+)\s*$", normalized)
        indirection = (
            indirection_match.group("indirection")
            if indirection_match is not None
            else ""
        )
        if indirection not in {"", "*", "&"}:
            return None
        struct_name = (
            normalized[: indirection_match.start()].strip()
            if indirection_match is not None
            else normalized
        )
        if struct_name not in structs_by_name:
            return None
        return struct_name, "->" if indirection == "*" else "."

    def _concrete_method_matches_call(
        self,
        method: _MetalStructMethod,
        code: str,
        arg_open: int,
        buffer_element_types: _MetalPositionedBufferTypes,
        local_variable_types: Dict[str, List[Tuple[int, str]]],
        variable_types: Dict[str, List[Tuple[int, str]]],
        structs_by_name: Dict[str, _MetalStructDefinition],
        type_alias_fallback_position: Optional[int] = None,
    ) -> bool:
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            return False
        raw_args = code[arg_open + 1 : arg_close]
        call_arguments = [
            argument
            for argument in self._split_top_level_commas(raw_args)
            if argument.strip()
        ]
        declared_types = [
            self._normalize_inferred_type(
                self._normalize_function_parameter_type_text(parameter)
            )
            for parameter in self._split_top_level_commas(method.parameters)
            if parameter.strip()
        ]
        if self._source_type_alias_bindings:
            declared_types = [
                self._canonicalize_type_aliases_at(
                    declared_type,
                    self._source_type_alias_bindings,
                    method.span[0],
                )
                or declared_type
                for declared_type in declared_types
            ]
        if len(declared_types) != len(call_arguments):
            return False

        buffer_view = self._flatten_types_at(buffer_element_types, arg_open)
        local_view = self._flatten_types_at(local_variable_types, arg_open)
        struct_field_types = self._struct_field_types_at(
            variable_types,
            structs_by_name,
            arg_open,
        )
        inferred_types = []
        for argument in call_arguments:
            inferred = self._infer_argument_type(
                argument,
                buffer_view,
                local_view,
                struct_field_types,
                structs_by_name,
            )
            if inferred is None:
                return False
            inferred_type = self._normalize_inferred_type(inferred)
            if self._source_type_alias_bindings:
                inferred_type = (
                    self._canonicalize_type_aliases_at(
                        inferred_type,
                        self._source_type_alias_bindings,
                        arg_open,
                    )
                    or inferred_type
                )
                if (
                    type_alias_fallback_position is not None
                    and inferred_type == self._normalize_inferred_type(inferred)
                ):
                    inferred_type = (
                        self._canonicalize_type_aliases_at(
                            inferred_type,
                            self._source_type_alias_bindings,
                            type_alias_fallback_position,
                        )
                        or inferred_type
                    )
            inferred_types.append(inferred_type)
        return declared_types == inferred_types

    def _select_concrete_method_for_call(
        self,
        code: str,
        struct: _MetalStructDefinition,
        method_name: str,
        arg_open: int,
        buffer_element_types: _MetalPositionedBufferTypes,
        local_variable_types: Dict[str, List[Tuple[int, str]]],
        variable_types: Dict[str, List[Tuple[int, str]]],
        structs_by_name: Dict[str, _MetalStructDefinition],
        *,
        allow_template_fallback: bool,
        require_static: Optional[bool] = None,
        type_alias_fallback_position: Optional[int] = None,
    ) -> Optional[_MetalStructMethod]:
        candidates = [
            method
            for method in struct.methods
            if method.name == method_name
            and (require_static is None or method.is_static is require_static)
        ]
        if not candidates:
            return None
        if len(candidates) == 1:
            candidate = candidates[0]
            if not allow_template_fallback or self._concrete_method_matches_call(
                candidate,
                code,
                arg_open,
                buffer_element_types,
                local_variable_types,
                variable_types,
                structs_by_name,
                type_alias_fallback_position,
            ):
                return candidate
            return None

        matches = [
            method
            for method in candidates
            if self._concrete_method_matches_call(
                method,
                code,
                arg_open,
                buffer_element_types,
                local_variable_types,
                variable_types,
                structs_by_name,
                type_alias_fallback_position,
            )
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches and allow_template_fallback:
            return None
        if any(self._method_returns_lvalue_reference(method) for method in candidates):
            # Reference-return lowering owns the fail-closed diagnostic and its
            # lvalue-identity analysis. Preserve that path when ordinary value
            # overload ranking cannot identify one candidate first.
            return candidates[-1]

        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        raw_args = (
            code[arg_open + 1 : arg_close].strip() if arg_close is not None else "..."
        )
        reason = "no exact concrete overload matches"
        if len(matches) > 1:
            reason = "multiple exact concrete overloads match"
        requested_signature = f"{struct.name}::{method_name}({raw_args})"
        raise MetalStructMethodError(
            "Cannot lower concrete member overload "
            f"'{struct.name}::{method_name}': {reason}. Requested call: "
            f"{requested_signature}.",
            struct_name=struct.name,
            method_name=method_name,
            requested_signature=requested_signature,
            suggested_action=(
                "cast each call argument to the intended source overload type"
            ),
            source_location=(
                self._source_location_for_offsets(code, arg_open, arg_close + 1)
                if arg_close is not None
                else None
            ),
            reason="concrete-overload-ambiguous",
        )

    @staticmethod
    def _concrete_method_overload_ordinal(
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
    ) -> Optional[int]:
        overloads = [
            candidate
            for candidate in struct.methods
            if candidate.free_name == method.free_name
        ]
        if len(overloads) < 2:
            return None
        return next(
            index
            for index, candidate in enumerate(overloads, start=1)
            if candidate is method
        )

    def _temporary_functor_call(
        self, code: str, constructor_start: int
    ) -> Optional[_MetalTemporaryFunctorCall]:
        if constructor_start >= len(code):
            return None
        if code[constructor_start] == "{":
            constructor_close = self._find_matching_delimiter(
                code,
                constructor_start,
                "{",
                "}",
            )
        elif code[constructor_start] == "(":
            constructor_close = self._find_matching_delimiter(
                code,
                constructor_start,
                "(",
                ")",
            )
        else:
            return None
        if constructor_close is None:
            return None
        arg_open = constructor_close + 1
        while arg_open < len(code) and code[arg_open].isspace():
            arg_open += 1
        if arg_open >= len(code) or code[arg_open] != "(":
            return None
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            return None
        return _MetalTemporaryFunctorCall(
            constructor_open=constructor_start,
            constructor_close=constructor_close,
            argument_open=arg_open,
            argument_close=arg_close,
        )

    def _temporary_functor_is_candidate(
        self,
        code: str,
        identifier_start: int,
        temporary: _MetalTemporaryFunctorCall,
    ) -> bool:
        constructor_text = code[
            temporary.constructor_open + 1 : temporary.constructor_close
        ].strip()
        # Function-pointer declarations share the `Type(...)(...)` delimiter
        # shape with a constructed callable. Exclude both named pointer
        # declarators (`(*Fn)`) and anonymous alias declarators (`(*)`).
        if re.fullmatch(
            r"(?:(?:[A-Za-z_]\w*)\s*::\s*)?\*\s*" r"(?:const\s*)?(?:[A-Za-z_]\w*)?",
            constructor_text,
        ):
            return False

        # A direct function-type typedef has no `*`: `typedef void (Fn)(...)`.
        # Limit this exclusion to the current declaration so an expression such
        # as `Missing(state)(value)` remains fail-closed.
        boundary = max(
            code.rfind(token, 0, identifier_start) for token in (";", "{", "}")
        )
        declaration_prefix = self._mask_comments_and_literals(
            code[boundary + 1 : identifier_start]
        )
        if re.search(r"\btypedef\b", declaration_prefix):
            return False

        return True

    def _temporary_functor_is_stateless_default(
        self,
        code: str,
        struct: _MetalStructDefinition,
        temporary: _MetalTemporaryFunctorCall,
    ) -> bool:
        constructor_arguments = code[
            temporary.constructor_open + 1 : temporary.constructor_close
        ]
        if constructor_arguments.strip() or struct.data_member_names:
            return False
        if not struct.constructors:
            return True

        viable_default_constructors: List[_MetalConstructor] = []
        for constructor in struct.constructors:
            parameters = [
                parameter
                for parameter in self._split_top_level_commas(constructor.params_text)
                if parameter.strip()
            ]
            if all(
                self._split_top_level_assignment(parameter)[1] is not None
                for parameter in parameters
            ):
                viable_default_constructors.append(constructor)
        if len(viable_default_constructors) != 1:
            return False
        constructor = viable_default_constructors[0]
        return not constructor.init_text.strip() and not constructor.body.strip()

    def _raise_temporary_functor_error(
        self,
        code: str,
        struct_name: str,
        call_start: int,
        temporary: _MetalTemporaryFunctorCall,
        *,
        reason: str,
        detail: str,
    ) -> None:
        call_end = temporary.argument_close + 1
        requested_signature = re.sub(r"\s+", " ", code[call_start:call_end]).strip()
        raise MetalStructMethodError(
            f"Cannot lower temporary function object '{requested_signature}': "
            f"{detail}.",
            struct_name=struct_name,
            method_name="operator()",
            requested_signature=requested_signature,
            suggested_action=(
                "materialize the function object as a typed local before the call, "
                "or provide a concrete stateless functor with a uniquely resolvable "
                "operator() overload"
            ),
            source_location=self._source_location_for_offsets(
                code, call_start, call_end
            ),
            reason=reason,
        )

    def _rewrite_nested_temporary_functor_calls(
        self,
        code: str,
        start: int,
        end: int,
        struct_names: Set[str],
        methods_by_struct: Dict[str, Dict[str, _MetalStructMethod]],
        template_methods_by_struct: Dict[str, Dict[str, List[_MetalStructMethod]]],
        operator_call_structs: Set[str],
        variable_types: Dict[str, List[Tuple[int, str]]],
        structs_by_name: Dict[str, _MetalStructDefinition],
        buffer_element_types: _MetalPositionedBufferTypes,
        local_variable_types: Dict[str, List[Tuple[int, str]]],
        instantiated_template_functions: Dict[str, str],
        field_variable_types: Dict[str, List[Tuple[int, str]]],
        field_structs_by_name: Dict[str, _MetalStructDefinition],
        local_integral_constants: Optional[
            Dict[str, List[_MetalIntegralConstantBinding]]
        ],
        struct_type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]],
        type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]],
        type_alias_fallback_position: Optional[int],
    ) -> str:
        replacements: List[Tuple[int, int, str]] = []
        cursor = start
        while cursor < end:
            if code[cursor] in "\"'":
                _literal, consumed = self._read_string(code, cursor)
                cursor += consumed
                continue
            if code.startswith("//", cursor):
                line_end = code.find("\n", cursor)
                cursor = end if line_end == -1 else min(end, line_end + 1)
                continue
            if code.startswith("/*", cursor):
                comment_end = code.find("*/", cursor + 2)
                cursor = end if comment_end == -1 else min(end, comment_end + 2)
                continue
            if not (code[cursor].isalpha() or code[cursor] == "_"):
                cursor += 1
                continue

            ident, consumed = self._read_identifier(code, cursor)
            ident_end = cursor + consumed
            constructor_start = ident_end
            while constructor_start < end and code[constructor_start].isspace():
                constructor_start += 1
            nested = self._temporary_functor_call(code, constructor_start)
            if nested is None or nested.argument_close >= end:
                cursor = ident_end
                continue

            rewrite = self._try_rewrite_call_at(
                code,
                cursor,
                ident,
                ident_end,
                struct_names,
                methods_by_struct,
                template_methods_by_struct,
                operator_call_structs,
                variable_types,
                structs_by_name,
                buffer_element_types,
                local_variable_types,
                instantiated_template_functions,
                field_variable_types=field_variable_types,
                field_structs_by_name=field_structs_by_name,
                struct_type_aliases=struct_type_aliases,
                local_integral_constants=local_integral_constants,
                type_aliases=type_aliases,
                type_alias_fallback_position=type_alias_fallback_position,
            )
            if rewrite is None:
                self._raise_temporary_functor_error(
                    code,
                    ident,
                    cursor,
                    nested,
                    reason="temporary-functor-overload-unresolved",
                    detail="the nested operator() call could not be materialized",
                )
            rewrite_end, replacement = rewrite
            if rewrite_end > end:
                cursor = ident_end
                continue
            replacements.append((cursor, rewrite_end, replacement))
            cursor = rewrite_end

        if not replacements:
            return code[start:end]
        parts: List[str] = []
        position = start
        for replacement_start, replacement_end, replacement in replacements:
            parts.append(code[position:replacement_start])
            parts.append(replacement)
            position = replacement_end
        parts.append(code[position:end])
        return "".join(parts)

    def _build_temporary_operator_call_rewrite(
        self,
        code: str,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        arg_open: int,
        instantiated_template_functions: Dict[str, str],
        *,
        rewritten_arguments: Optional[str] = None,
    ) -> Optional[Tuple[int, str]]:
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            return None
        self._reject_reference_returning_method_call(
            code,
            struct,
            method,
            arg_open,
            arg_close,
        )
        overload_ordinal = self._concrete_method_overload_ordinal(struct, method)
        wrapper_name = (
            f"{method.free_name}__metal_overload_{overload_ordinal}__temporary"
            if overload_ordinal is not None
            else f"{method.free_name}__temporary"
        )
        if wrapper_name not in instantiated_template_functions:
            helper_source = self._emit_free_function(struct, method)
            wrapper = self._emit_temporary_operator_wrapper(
                struct.name,
                method.free_name,
                wrapper_name,
                helper_source,
            )
            if wrapper is None:
                return None
            instantiated_template_functions[wrapper_name] = wrapper
        return self._build_template_call_rewrite(
            code,
            None,
            wrapper_name,
            arg_open,
            arg_close,
            raw_arguments=rewritten_arguments,
        )

    def _instantiate_temporary_template_operator_call(
        self,
        code: str,
        struct: _MetalStructDefinition,
        overloads: List[_MetalStructMethod],
        arg_open: int,
        buffer_element_types: _MetalPositionedBufferTypes,
        local_variable_types: Dict[str, List[Tuple[int, str]]],
        instantiated_template_functions: Dict[str, str],
        structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
        variable_types: Optional[Dict[str, List[Tuple[int, str]]]] = None,
        template_methods_by_struct: Optional[
            Dict[str, Dict[str, List[_MetalStructMethod]]]
        ] = None,
        methods_by_struct: Optional[Dict[str, Dict[str, _MetalStructMethod]]] = None,
        operator_call_structs: Optional[Set[str]] = None,
        rewrite_structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
        local_integral_constants: Optional[
            Dict[str, List[_MetalIntegralConstantBinding]]
        ] = None,
        argument_type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]] = None,
        argument_type_fallback_position: Optional[int] = None,
        rewritten_arguments: Optional[str] = None,
    ) -> Optional[Tuple[int, str]]:
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            return None
        raw_args = code[arg_open + 1 : arg_close]
        call_arguments = [
            argument
            for argument in self._split_top_level_commas(raw_args)
            if argument.strip()
        ]
        representative = overloads[0]
        signature = self._template_member_call_signature(
            struct,
            representative,
            None,
            raw_args,
        )
        call_start = code.rfind(struct.name, 0, arg_open)
        if call_start < 0:
            call_start = arg_open
        call_location = self._source_location_for_offsets(
            code,
            call_start,
            arg_close + 1,
        )

        buffer_view = self._flatten_types_at(buffer_element_types, arg_open)
        local_view = self._flatten_types_at(local_variable_types, arg_open)
        struct_field_types = self._struct_field_types_at(
            variable_types,
            structs_by_name,
            arg_open,
        )
        local_constant_values = self._local_integral_constants_at(
            local_integral_constants or {},
            arg_open,
        )
        concrete_argument_types: List[str] = []
        for argument in call_arguments:
            inference_argument = self._substitute_template_argument_static_constants(
                argument,
                local_constant_values,
            )
            inferred = self._infer_argument_type(
                inference_argument,
                buffer_view,
                local_view,
                struct_field_types,
                structs_by_name,
            )
            if inferred is None:
                raise MetalStructMethodError(
                    "Cannot lower template member method "
                    f"'{struct.name}::{representative.name}': the type of call "
                    f"argument '{argument.strip()}' could not be inferred "
                    f"conservatively. Requested call: {signature}.",
                    struct_name=struct.name,
                    method_name=representative.name,
                    requested_signature=signature,
                    suggested_action=(
                        "pass the argument as a buffer-element access, a typed "
                        "local variable, a literal, or an explicit cast so its "
                        "type can be inferred, or specialize the method manually"
                    ),
                    source_location=call_location,
                )
            concrete_argument_types.append(inferred)

        try:
            free_name = self._instantiate_template_member_overload(
                struct,
                overloads,
                concrete_argument_types,
                signature,
                instantiated_template_functions,
                template_methods_by_struct,
                methods_by_struct=methods_by_struct,
                operator_call_structs=operator_call_structs,
                rewrite_structs_by_name=rewrite_structs_by_name,
                argument_type_aliases=argument_type_aliases,
                argument_type_position=arg_open,
                argument_type_fallback_position=argument_type_fallback_position,
            )
        except MetalStructMethodError as exc:
            if exc.source_location is None:
                exc.source_location = call_location
            raise
        helper_source = instantiated_template_functions.get(free_name)
        if helper_source is None:
            return None
        wrapper_name = f"{free_name}__temporary"
        if wrapper_name not in instantiated_template_functions:
            wrapper = self._emit_temporary_operator_wrapper(
                struct.name,
                free_name,
                wrapper_name,
                helper_source,
            )
            if wrapper is None:
                return None
            instantiated_template_functions[wrapper_name] = wrapper
        selection = self._instantiated_template_member_calls.get(free_name)
        if selection is not None:
            self._instantiated_template_member_calls[wrapper_name] = selection
        return self._build_template_call_rewrite(
            code,
            None,
            wrapper_name,
            arg_open,
            arg_close,
            raw_arguments=rewritten_arguments,
        )

    def _emit_temporary_operator_wrapper(
        self,
        struct_name: str,
        free_name: str,
        wrapper_name: str,
        helper_source: str,
    ) -> Optional[str]:
        match = re.match(
            rf"\s*(?P<return>.*?)\s+{re.escape(free_name)}\s*"
            r"\((?P<params>.*?)\)\s*\{",
            helper_source,
            re.DOTALL,
        )
        if match is None:
            return None
        return_type = re.sub(r"\s+", " ", match.group("return")).strip()
        parameters = [
            parameter.strip()
            for parameter in self._split_top_level_commas(match.group("params"))
            if parameter.strip()
        ]
        if not parameters:
            return None
        if self._declared_data_member_name(parameters[0]) != "self":
            return None
        wrapper_parameters = parameters[1:]
        wrapper_parameter_text = ", ".join(wrapper_parameters)
        argument_names = self._parameter_identifier_names(wrapper_parameter_text)
        call_arguments = ", ".join(["self", *argument_names])
        call = f"{free_name}({call_arguments})"
        if return_type == "void":
            body = f"{{ {struct_name} self; {call}; }}"
        else:
            body = f"{{ {struct_name} self; return {call}; }}"
        return f"{return_type} {wrapper_name}({wrapper_parameter_text}) {body}"

    def _instantiate_template_member_call(
        self,
        code: str,
        struct: _MetalStructDefinition,
        overloads: List[_MetalStructMethod],
        receiver: Optional[str],
        arg_open: int,
        buffer_element_types: _MetalPositionedBufferTypes,
        local_variable_types: Dict[str, List[Tuple[int, str]]],
        instantiated_template_functions: Dict[str, str],
        structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
        variable_types: Optional[Dict[str, List[Tuple[int, str]]]] = None,
        template_methods_by_struct: Optional[
            Dict[str, Dict[str, List[_MetalStructMethod]]]
        ] = None,
        methods_by_struct: Optional[Dict[str, Dict[str, _MetalStructMethod]]] = None,
        operator_call_structs: Optional[Set[str]] = None,
        rewrite_structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
        explicit_template_arguments: Optional[List[str]] = None,
        local_integral_constants: Optional[
            Dict[str, List[_MetalIntegralConstantBinding]]
        ] = None,
        argument_type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]] = None,
        argument_type_fallback_position: Optional[int] = None,
    ) -> Optional[Tuple[int, str]]:
        # Instantiate a CALLED template member method from its call-site argument
        # types and rewrite the call to the concrete free function. `overloads`
        # holds every method sharing the called name; the unique overload whose
        # SFINAE constraints are satisfied by the inferred type bindings is
        # selected. A call that cannot be resolved (un-inferable argument,
        # non-binding template parameters, or ambiguous/unrecognized constraints)
        # raises MetalStructMethodError — a clean translation failure — rather
        # than leaving a dangling call or mis-selecting an overload.
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            return None
        raw_args = code[arg_open + 1 : arg_close]
        call_arguments = [
            argument
            for argument in self._split_top_level_commas(raw_args)
            if argument.strip()
        ]
        # All overloads of one name share a declared-parameter shape; bind using a
        # representative for argument typing and the diagnostic signature.
        representative = overloads[0]
        signature = self._template_member_call_signature(
            struct, representative, receiver, raw_args
        )
        call_start = code.rfind(representative.name, 0, arg_open)
        if call_start < 0:
            call_start = arg_open
        call_location = self._source_location_for_offsets(
            code, call_start, arg_close + 1
        )

        # Resolve the position-ordered declaration maps to the FLAT views valid at
        # this call site (nearest preceding declaration wins). Passing flat dicts
        # keeps `_infer_argument_type` a pure, unit-testable function while the
        # per-call-site resolution stays deterministic and scope-correct.
        buffer_view = self._flatten_types_at(buffer_element_types, arg_open)
        local_view = self._flatten_types_at(local_variable_types, arg_open)
        struct_field_types = self._struct_field_types_at(
            variable_types, structs_by_name, arg_open
        )
        local_constant_values = self._local_integral_constants_at(
            local_integral_constants or {},
            arg_open,
        )

        # Infer the concrete type of every call argument; one un-inferable
        # argument is a clean failure. The source call stays unchanged; only the
        # specialization identity and emitted helper signature use proven values.
        concrete_argument_types: List[str] = []
        for argument in call_arguments:
            inference_argument = self._substitute_template_argument_static_constants(
                argument,
                local_constant_values,
            )
            inferred = self._infer_argument_type(
                inference_argument,
                buffer_view,
                local_view,
                struct_field_types,
                structs_by_name,
            )
            if inferred is None:
                raise MetalStructMethodError(
                    "Cannot lower template member method "
                    f"'{struct.name}::{representative.name}': the type of call "
                    f"argument '{argument.strip()}' could not be inferred "
                    f"conservatively. Requested call: {signature}.",
                    struct_name=struct.name,
                    method_name=representative.name,
                    requested_signature=signature,
                    suggested_action=(
                        "pass the argument as a buffer-element access, a typed "
                        "local variable, a literal, or an explicit cast so its "
                        "type can be inferred, or specialize the method manually"
                    ),
                    source_location=call_location,
                )
            concrete_argument_types.append(inferred)

        resolved_explicit_template_arguments = (
            [
                self._substitute_template_argument_static_constants(
                    argument,
                    local_constant_values,
                )
                for argument in explicit_template_arguments
            ]
            if explicit_template_arguments is not None
            else None
        )

        try:
            free_name = self._instantiate_template_member_overload(
                struct,
                overloads,
                concrete_argument_types,
                signature,
                instantiated_template_functions,
                template_methods_by_struct,
                methods_by_struct=methods_by_struct,
                operator_call_structs=operator_call_structs,
                rewrite_structs_by_name=rewrite_structs_by_name,
                explicit_template_arguments=resolved_explicit_template_arguments,
                argument_type_aliases=argument_type_aliases,
                argument_type_position=arg_open,
                argument_type_fallback_position=argument_type_fallback_position,
            )
        except MetalStructMethodError as exc:
            if exc.source_location is None:
                exc.source_location = call_location
            raise
        # Static-ness comes from the overload set; all overloads of one name agree
        # on static-ness in the patterns we handle, so the representative decides.
        receiver_name = None if representative.is_static else receiver
        return self._build_template_call_rewrite(
            code, receiver_name, free_name, arg_open, arg_close
        )

    @staticmethod
    def _source_location_for_offsets(code: str, start: int, end: int) -> Dict[str, int]:
        start = max(0, min(start, len(code)))
        end = max(start, min(end, len(code)))
        line = code.count("\n", 0, start) + 1
        end_line = code.count("\n", 0, end) + 1
        previous_newline = code.rfind("\n", 0, start)
        end_previous_newline = code.rfind("\n", 0, end)
        return {
            "line": line,
            "column": start - previous_newline,
            "offset": start,
            "length": end - start,
            "endLine": end_line,
            "endColumn": end - end_previous_newline,
            "endOffset": end,
        }

    def _instantiate_template_member_overload(
        self,
        struct: _MetalStructDefinition,
        overloads: List[_MetalStructMethod],
        concrete_argument_types: List[Optional[str]],
        signature: str,
        instantiated_template_functions: Dict[str, str],
        template_methods_by_struct: Optional[
            Dict[str, Dict[str, List[_MetalStructMethod]]]
        ],
        methods_by_struct: Optional[Dict[str, Dict[str, _MetalStructMethod]]] = None,
        operator_call_structs: Optional[Set[str]] = None,
        rewrite_structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
        explicit_template_arguments: Optional[List[str]] = None,
        argument_type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]] = None,
        argument_type_position: Optional[int] = None,
        argument_type_fallback_position: Optional[int] = None,
    ) -> str:
        # Bind, select the enabled overload, materialize it (recursively lowering
        # any internal template-member calls in its body), and return the
        # concrete free-function name. Raises MetalStructMethodError on any
        # unresolved/ambiguous case.
        representative = overloads[0]
        candidates: List[Tuple[_MetalStructMethod, Dict[str, str]]] = []
        for overload in overloads:
            bindings = self._bind_template_method_parameters(
                overload,
                concrete_argument_types,
                explicit_template_arguments=explicit_template_arguments,
                owner_struct=struct,
                structs_by_name=rewrite_structs_by_name,
                argument_type_aliases=argument_type_aliases,
                argument_type_position=argument_type_position,
                argument_type_fallback_position=argument_type_fallback_position,
            )
            if bindings is not None:
                candidates.append((overload, bindings))
        if not candidates:
            raise MetalStructMethodError(
                "Cannot lower template member method "
                f"'{struct.name}::{representative.name}': its template parameters "
                f"{representative.template_parameters} did not bind consistently "
                f"from the inferred argument types {concrete_argument_types}. "
                f"Requested call: {signature}.",
                struct_name=struct.name,
                method_name=representative.name,
                requested_signature=signature,
                suggested_action=(
                    "ensure each template parameter appears in a parameter type "
                    "that matches the inferred argument types, or specialize the "
                    "method manually"
                ),
            )

        selected_candidate = self._select_constrained_overload(candidates)
        if selected_candidate is None:
            raise MetalStructMethodError(
                "Cannot lower template member method "
                f"'{struct.name}::{representative.name}': no unique overload is "
                "compatible with the inferred argument types and enabled by its "
                "SFINAE constraints (zero or several matched, or a constraint "
                "is unsupported). "
                f"Requested call: {signature}.",
                struct_name=struct.name,
                method_name=representative.name,
                requested_signature=signature,
                suggested_action=(
                    "specialize the method manually, or restrict the call to a "
                    "type whose constraint is recognized (sizeof / is_integral_v)"
                ),
            )
        selected, bindings = selected_candidate
        resolved_return_type = self._replace_identifiers(
            selected.return_type,
            bindings,
        )
        resolved_return_type = self._canonicalize_struct_scoped_type(
            resolved_return_type,
            struct,
            rewrite_structs_by_name,
        )
        self._reject_selected_reference_return(
            struct,
            selected,
            resolved_return_type,
            signature,
        )
        ordered_arguments = [bindings[name] for name in selected.template_parameters]
        free_name = self._template_member_free_name(struct, selected, ordered_arguments)
        self._instantiated_template_member_calls[free_name] = (
            struct,
            selected,
            bindings,
            rewrite_structs_by_name,
        )
        if free_name not in instantiated_template_functions:
            # Reserve the name first so a (pathological) self-recursive method does
            # not loop forever; the body is filled in below.
            instantiated_template_functions[free_name] = ""
            instantiated_template_functions[free_name] = (
                self._emit_template_member_free_function(
                    struct,
                    selected,
                    bindings,
                    free_name,
                    instantiated_template_functions=instantiated_template_functions,
                    template_methods_by_struct=template_methods_by_struct,
                    methods_by_struct=methods_by_struct,
                    operator_call_structs=operator_call_structs,
                    rewrite_structs_by_name=rewrite_structs_by_name,
                )
            )
        return free_name

    def _template_member_call_signature(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        receiver: Optional[str],
        raw_args: str,
    ) -> str:
        target = receiver if receiver is not None else struct.name
        if method.is_operator_call:
            return f"{target}({raw_args.strip()})"
        return f"{target}.{method.name}({raw_args.strip()})"

    def _bind_template_method_parameters(
        self,
        method: _MetalStructMethod,
        concrete_argument_types: List[Optional[str]],
        *,
        explicit_template_arguments: Optional[List[str]] = None,
        owner_struct: Optional[_MetalStructDefinition] = None,
        structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
        argument_type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]] = None,
        argument_type_position: Optional[int] = None,
        argument_type_fallback_position: Optional[int] = None,
    ) -> Optional[Dict[str, str]]:
        # Match each declared method-parameter type (which may contain the
        # template parameters) against the inferred concrete argument type and
        # collect consistent bindings. Returns None if any template parameter is
        # left unbound or a parameter binds inconsistently.
        template_parameter_set = set(method.template_parameters)
        source_type_aliases = self._source_type_alias_bindings

        def canonicalize_declared_aliases(
            type_text: str,
            *,
            excluded_aliases: Optional[Set[str]] = None,
        ) -> Optional[str]:
            if not source_type_aliases:
                return self._normalize_template_argument_text(type_text)
            return self._canonicalize_type_aliases_at(
                type_text,
                source_type_aliases,
                method.span[0],
                excluded_aliases=excluded_aliases,
            )

        def canonicalize_argument_aliases(type_text: str) -> Optional[str]:
            canonical = self._normalize_template_argument_text(type_text)
            primary_aliases = (
                source_type_aliases
                if argument_type_aliases is None
                else argument_type_aliases
            )
            if primary_aliases and argument_type_position is not None:
                canonical = self._canonicalize_type_aliases_at(
                    canonical,
                    primary_aliases,
                    argument_type_position,
                )
                if canonical is None:
                    return None
            if (
                primary_aliases is not source_type_aliases
                and source_type_aliases
                and argument_type_fallback_position is not None
            ):
                canonical = self._canonicalize_type_aliases_at(
                    canonical,
                    source_type_aliases,
                    argument_type_fallback_position,
                )
            return canonical

        # `method.parameters` is the already-extracted parameter list (the
        # `operator()` declarator is handled at parse time), so split it directly
        # rather than reparsing a synthesized header — a synthesized
        # `operator()` header would confuse the first-paren parameter finder.
        parameters = [
            parameter
            for parameter in self._split_top_level_commas(method.parameters)
            if parameter.strip()
        ]
        if len(concrete_argument_types) > len(parameters):
            return None
        for parameter in parameters[len(concrete_argument_types) :]:
            _declaration, default = self._split_top_level_assignment(parameter)
            if default is None or not default.strip():
                return None
        declared_parameter_types: List[str] = []
        declared_parameter_is_indirect: List[bool] = []
        declared_parameter_is_array: List[bool] = []
        declared_pointer_types: List[Optional[str]] = []
        for parameter in parameters[: len(concrete_argument_types)]:
            subscriptable = self._pointer_or_array_parameter_type(parameter)
            declared = (
                subscriptable.element_type
                if subscriptable is not None
                else self._function_parameter_value_type(parameter)
            )
            if owner_struct is not None:
                declared = self._canonicalize_struct_scoped_type(
                    declared, owner_struct, structs_by_name
                )
            normalized = self._normalize_inferred_type(declared)
            if normalized and normalized != "void":
                declared_parameter_types.append(normalized)
                declared_parameter_is_indirect.append(subscriptable is not None)
                declarator = self._strip_top_level_default_value(
                    self._strip_metal_attributes(parameter)
                ).strip()
                declared_parameter_is_array.append(declarator.endswith("]"))
                declared_pointer_types.append(
                    subscriptable.pointer_type if subscriptable is not None else None
                )
        if len(declared_parameter_types) != len(concrete_argument_types):
            return None
        # Bind each parameter into its OWN dict and merge with explicit conflict
        # detection. `_infer_template_parameter_bindings_from_type` silently
        # SKIPS a parameter whose new value conflicts with an existing binding;
        # that is the wrong behavior here — a template parameter that the call
        # site forces to two different concrete types (`pick(float, int)` for
        # `T pick(T, T)`) must clean-fail, not silently keep the first guess.
        explicit_template_arguments = explicit_template_arguments or []
        if len(explicit_template_arguments) > len(method.template_parameters):
            return None
        bindings: Dict[str, str] = {
            name: self._normalize_template_binding_type(argument)
            for name, argument in zip(
                method.template_parameters, explicit_template_arguments
            )
        }
        if any(not value for value in bindings.values()):
            return None
        indirect_pointee_types: List[Tuple[str, str]] = []
        for (
            declared_type,
            is_indirect,
            is_array,
            declared_pointer_type,
            concrete_type,
        ) in zip(
            declared_parameter_types,
            declared_parameter_is_indirect,
            declared_parameter_is_array,
            declared_pointer_types,
            concrete_argument_types,
        ):
            if concrete_type is None:
                continue
            local_bindings: Dict[str, str] = {}
            concrete_value_type = (
                self._normalize_template_binding_type(concrete_type).rstrip("&").strip()
            )
            if is_indirect and concrete_value_type.endswith("*"):
                canonical_declared_pointer = (
                    canonicalize_declared_aliases(
                        declared_pointer_type,
                        excluded_aliases=template_parameter_set,
                    )
                    if declared_pointer_type is not None
                    else None
                )
                canonical_concrete_pointer = canonicalize_argument_aliases(
                    concrete_value_type
                )
                if (
                    canonical_declared_pointer is None
                    or canonical_concrete_pointer is None
                    or not (
                        self._pointer_argument_matches_declared_type(
                            canonical_declared_pointer,
                            canonical_concrete_pointer,
                        )
                    )
                ):
                    return None
                concrete_value_type = self._pointer_pointee_value_type(
                    concrete_value_type
                )
                if concrete_value_type is None:
                    return None
                indirect_pointee_types.append((declared_type, concrete_value_type))
            elif is_array:
                # Internal method-body inference retains the element type for
                # local arrays. It is still a concrete pointee for validation.
                indirect_pointee_types.append((declared_type, concrete_value_type))
            self._infer_template_parameter_bindings_from_type(
                declared_type,
                concrete_value_type,
                template_parameter_set,
                local_bindings,
            )
            for name, value in local_bindings.items():
                existing = bindings.get(name)
                if existing is not None and existing != value:
                    return None
                bindings[name] = value
        # Apply declared defaults only after explicit and inferred bindings so a
        # dependent default can reference an earlier template parameter.
        for name in method.template_parameters:
            if name not in bindings:
                default = method.template_parameter_defaults.get(name)
                if default is None:
                    return None
                resolved_default = self._replace_identifiers(default, bindings)
                normalized_default = self._normalize_template_binding_type(
                    resolved_default
                )
                if not normalized_default:
                    return None
                bindings[name] = normalized_default
        # Pointer deduction above validates address-space/cv compatibility and
        # binds template names. Once explicit/default bindings are also known,
        # validate the complete pointee pattern so concrete portions cannot be
        # ignored merely because another parameter supplied every binding.
        for declared_type, concrete_value_type in indirect_pointee_types:
            resolved_declared_type = self._replace_identifiers(declared_type, bindings)
            if owner_struct is not None:
                resolved_declared_type = self._canonicalize_struct_scoped_type(
                    resolved_declared_type, owner_struct, structs_by_name
                )
            resolved_declared_type = canonicalize_declared_aliases(
                resolved_declared_type
            )
            concrete_value_type = canonicalize_argument_aliases(concrete_value_type)
            if resolved_declared_type is None or concrete_value_type is None:
                return None
            resolved_declared_type = self._canonical_template_binding_pointee_type(
                resolved_declared_type
            )
            concrete_value_type = self._canonical_template_binding_pointee_type(
                concrete_value_type
            )
            if resolved_declared_type != concrete_value_type:
                return None
        return bindings

    def _template_member_free_name(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        ordered_arguments: List[str],
    ) -> str:
        # Concrete free-function name for an instantiated template member method:
        # `S__m__<binding-suffix>` (or `S__operator_call__<suffix>`).
        base = self._struct_member_free_name(
            struct.name, method.name, method.is_operator_call
        )
        suffix = "_".join(
            self._template_member_name_component(value) for value in ordered_arguments
        ).strip("_")
        if not suffix:
            return base
        return f"{base}__{suffix}"

    @staticmethod
    def _template_member_name_component(value: str) -> str:
        text = str(value)
        text = re.sub(r"(?<![A-Za-z0-9_])-(?=\s*[0-9])", " negative ", text)
        text = text.replace("*", " ptr ").replace("&", " ref ")
        return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")

    def _emit_template_member_free_function(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        bindings: Dict[str, str],
        free_name: str,
        instantiated_template_functions: Optional[Dict[str, str]] = None,
        template_methods_by_struct: Optional[
            Dict[str, Dict[str, List[_MetalStructMethod]]]
        ] = None,
        methods_by_struct: Optional[Dict[str, Dict[str, _MetalStructMethod]]] = None,
        operator_call_structs: Optional[Set[str]] = None,
        rewrite_structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
    ) -> str:
        # Instantiate the template member method by substituting the bound
        # template parameters into copies of its return type, parameters and
        # body, then emit a concrete free function reusing the non-template
        # lowering machinery (member references resolved to `self.x`).
        instantiated_return = self._replace_identifiers(method.return_type, bindings)
        instantiated_return = re.sub(r"\s+", " ", instantiated_return).strip()
        instantiated_parameters = self._replace_identifiers(method.parameters, bindings)
        instantiated_body = self._replace_identifiers(method.body, bindings)
        instantiated_body = self._specialize_concrete_method_body(
            struct, method, instantiated_body, rewrite_structs_by_name
        )
        instantiated_body = self._substitute_integral_constant_parameter_values(
            instantiated_parameters, instantiated_body
        )
        # Lower any call to a SIBLING template member method made from this body
        # (the second SFINAE layer: `simd_reduce` calls `simd_reduce_impl`). With
        # the outer bindings applied the body's parameter/local types are concrete,
        # so each internal call selects+instantiates its own overload and is
        # rewritten to the concrete free function — leaving no dangling call.
        if instantiated_template_functions is not None and template_methods_by_struct:
            instantiated_body = self._lower_internal_template_member_calls(
                struct,
                method,
                instantiated_parameters,
                instantiated_body,
                instantiated_template_functions,
                template_methods_by_struct,
                methods_by_struct=methods_by_struct,
                operator_call_structs=operator_call_structs,
                rewrite_structs_by_name=rewrite_structs_by_name,
            )
        concrete_parameters = self._strip_function_parameter_defaults(
            instantiated_parameters
        )
        concrete_method = _MetalStructMethod(
            name=method.name,
            free_name=free_name,
            is_static=method.is_static,
            is_operator_call=method.is_operator_call,
            return_type=instantiated_return,
            parameters=concrete_parameters,
            parameter_names=self._parameter_identifier_names(concrete_parameters),
            body=instantiated_body,
            span=method.span,
            is_const=method.is_const,
        )
        return self._emit_free_function(
            struct, concrete_method, structs_by_name=rewrite_structs_by_name
        )

    def _substitute_integral_constant_parameter_values(
        self, parameters: str, body: str
    ) -> str:
        replacements_by_name: Dict[str, int] = {}
        shadowed_names = self._local_variable_names(body)
        for parameter in self._split_top_level_commas(parameters):
            name = self._declared_data_member_name(parameter)
            if (
                not name
                or name in shadowed_names
                or self._lambda_binds_identifier(body, name)
            ):
                continue
            parameter_type = self._function_parameter_value_type(parameter)
            parts = self._integral_constant_type_parts(parameter_type)
            if parts is None:
                continue
            _base_type, value = parts
            replacements_by_name[name] = value
        if not replacements_by_name:
            return body

        ignored_spans = self._find_comment_and_literal_spans(body)
        replacements: List[Tuple[int, int, str]] = []
        for name, value in replacements_by_name.items():
            pattern = re.compile(rf"\b{re.escape(name)}\s*\.\s*value\b")
            matches = [
                match
                for match in pattern.finditer(body)
                if self._containing_span(match.start(), ignored_spans) is None
                and not self._is_member_identifier_context(body, match.start())
            ]
            if any(
                not self._integral_constant_value_use_is_safe(
                    body, match.start(), match.end()
                )
                for match in matches
            ):
                continue
            for match in matches:
                replacements.append(
                    (
                        match.start(),
                        match.end(),
                        self._static_integral_literal_text(value),
                    )
                )
        return self._apply_text_replacements(body, replacements)

    def _lambda_binds_identifier(self, source: str, identifier: str) -> bool:
        visible = self._remove_spans(
            source, self._find_comment_and_literal_spans(source)
        )
        for match in re.finditer(r"\[[^\]]*\]\s*\((?P<parameters>[^()]*)\)", visible):
            for parameter in self._split_top_level_commas(match.group("parameters")):
                if self._declared_data_member_name(parameter) == identifier:
                    return True
        return False

    @staticmethod
    def _static_integral_literal_text(value: int) -> str:
        return f"({value})" if value < 0 else str(value)

    def _integral_constant_value_use_is_safe(
        self, source: str, identifier_start: int, member_end: int
    ) -> bool:
        prefix = source[:identifier_start].rstrip()
        while prefix.endswith("("):
            prefix = prefix[:-1].rstrip()
        if prefix.endswith(("&", "++", "--")):
            return False
        if re.search(r"\b(?:alignof|decltype|sizeof|typeid)$", prefix):
            return False

        suffix = source[member_end:].lstrip()
        if suffix.startswith(("++", "--", ".", "->", "(", "[")):
            return False
        return re.match(r"^(?:(?:<<|>>|[+\-*/%&|^])?=)(?!=)", suffix) is None

    def _specialize_concrete_method_body(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        body: str,
        structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
    ) -> str:
        specialized_method = replace(method, body=body)
        body = self._substitute_static_data_member_initializers(
            struct, specialized_method, body
        )
        if structs_by_name:
            body = self._resolve_qualified_static_constant_expression(
                body, structs_by_name
            )
        return self._lower_concrete_const_for_loop_callbacks(body)

    def _configure_integral_constant_contracts(self, source: str) -> None:
        self._integral_constant_contract_verified = (
            self._has_integral_constant_contract(source)
        )
        self._integral_constant_conversion_contract_verified = (
            self._integral_constant_contract_verified
            and self._has_integral_constant_conversion_contract(source)
        )
        self._int_alias_contract_verified = (
            self._integral_constant_contract_verified
            and self._has_int_alias_contract(source)
        )
        self._const_for_loop_contract_verified = (
            self._int_alias_contract_verified
            and self._has_const_for_loop_contract(source)
        )
        self._integral_constant_binary_operators = (
            self._find_integral_constant_binary_operators(source)
            if self._integral_constant_contract_verified
            else set()
        )

    def _has_integral_constant_contract(self, source: str) -> bool:
        pattern = re.compile(
            r"template\s*<\s*typename\s+(?P<type>[A-Za-z_]\w*)\s*,\s*"
            r"(?P=type)\s+(?P<value>[A-Za-z_]\w*)\s*>\s*"
            r"struct\s+integral_constant\s*\{"
        )
        ignored_spans = self._find_comment_and_literal_spans(source)
        declarations = [
            match
            for match in re.finditer(r"\bstruct\s+integral_constant\b", source)
            if self._containing_span(match.start(), ignored_spans) is None
        ]
        if len(declarations) != 1:
            return False
        for match in pattern.finditer(source):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            body_open = match.end() - 1
            body_close = self._find_matching_delimiter(source, body_open, "{", "}")
            if body_close is None:
                continue
            body = self._remove_spans(
                source[body_open + 1 : body_close],
                self._find_comment_and_literal_spans(
                    source[body_open + 1 : body_close]
                ),
            )
            value_declaration = re.compile(
                r"\bstatic(?:\s+(?:constexpr|constant|const))*\s+"
                rf"{re.escape(match.group('type'))}\s+value\s*=\s*"
                rf"{re.escape(match.group('value'))}\s*;"
            )
            if value_declaration.search(body):
                return True
        return False

    def _has_int_alias_contract(self, source: str) -> bool:
        pattern = re.compile(
            r"template\s*<\s*int\s+(?P<value>[A-Za-z_]\w*)\s*>\s*"
            r"using\s+Int\s*=\s*integral_constant\s*<\s*int\s*,\s*"
            r"(?P=value)\s*>\s*;"
        )
        ignored_spans = self._find_comment_and_literal_spans(source)
        declarations = [
            match
            for match in re.finditer(r"\b(?:class|struct|using)\s+Int\b", source)
            if self._containing_span(match.start(), ignored_spans) is None
        ]
        matches = [
            match
            for match in pattern.finditer(source)
            if self._containing_span(match.start(), ignored_spans) is None
        ]
        return len(declarations) == len(matches) == 1

    def _has_integral_constant_conversion_contract(self, source: str) -> bool:
        pattern = re.compile(
            r"template\s*<\s*typename\s+(?P<type>[A-Za-z_]\w*)\s*,\s*"
            r"(?P=type)\s+(?P<value>[A-Za-z_]\w*)\s*>\s*"
            r"struct\s+integral_constant\s*\{"
        )
        ignored_spans = self._find_comment_and_literal_spans(source)
        matched_contracts = 0
        for match in pattern.finditer(source):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            body_open = match.end() - 1
            body_close = self._find_matching_delimiter(source, body_open, "{", "}")
            if body_close is None:
                continue
            body = source[body_open + 1 : body_close]
            body = self._remove_spans(body, self._find_comment_and_literal_spans(body))
            compact = re.sub(r"\s+", "", body)
            type_name = re.escape(match.group("type"))
            alias = re.compile(rf"usingvalue_type={type_name};")
            conversion = re.compile(
                r"(?:METAL_FUNC)?constexproperatorvalue_type\(\)const"
                r"(?:noexcept)?\{returnvalue;\}"
            )
            if alias.search(compact) and conversion.search(compact):
                matched_contracts += 1
        return matched_contracts == 1

    def _has_const_for_loop_contract(self, source: str) -> bool:
        pattern = re.compile(
            r"template\s*<\s*int\s+(?P<start>[A-Za-z_]\w*)\s*,\s*"
            r"int\s+(?P<stop>[A-Za-z_]\w*)\s*,\s*"
            r"int\s+(?P<step>[A-Za-z_]\w*)\s*,\s*"
            r"typename\s+(?P<function_type>[A-Za-z_]\w*)\s*>\s*"
            r"(?:constexpr\s+)?void\s+const_for_loop\s*\(\s*"
            r"(?P=function_type)\s+(?P<function>[A-Za-z_]\w*)\s*\)\s*\{"
        )
        ignored_spans = self._find_comment_and_literal_spans(source)
        definitions = [
            match
            for match in re.finditer(r"\bconst_for_loop\s*\([^(){};]*\)\s*\{", source)
            if self._containing_span(match.start(), ignored_spans) is None
        ]
        if len(definitions) != 1:
            return False
        matched_contracts = 0
        for match in pattern.finditer(source):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            body_open = match.end() - 1
            body_close = self._find_matching_delimiter(source, body_open, "{", "}")
            if body_close is None:
                continue
            body = source[body_open + 1 : body_close]
            body = self._remove_spans(body, self._find_comment_and_literal_spans(body))
            compact = re.sub(r"\s+", "", body)
            start = re.escape(match.group("start"))
            stop = re.escape(match.group("stop"))
            step = re.escape(match.group("step"))
            function_type = re.escape(match.group("function_type"))
            function = re.escape(match.group("function"))
            contract = re.compile(
                rf"^ifconstexpr\({start}<{stop}\)\{{"
                rf"constexprauto(?P<index>[A-Za-z_]\w*)=Int<{start}>\{{\}};"
                rf"{function}\((?P=index)\);"
                rf"const_for_loop<{start}\+{step},{stop},{step},{function_type}>"
                rf"\({function}\);\}}$"
            )
            if contract.fullmatch(compact):
                matched_contracts += 1
        return matched_contracts == 1

    def _find_integral_constant_binary_operators(self, source: str) -> Set[str]:
        operators: Set[str] = set()
        rejected_operators: Set[str] = set()
        ignored_spans = self._find_comment_and_literal_spans(source)
        pattern = re.compile(r"\boperator\s*(?P<operator>[+*/-])\s*\(")
        integral_constant_parameter = re.compile(
            r"(?<!:)\bintegral_constant\s*<" r"[^,>]+,\s*(?P<value>[A-Za-z_]\w*)\s*>"
        )
        for match in pattern.finditer(source):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            paren_open = source.find("(", match.start(), match.end())
            if paren_open == -1:
                continue
            paren_close = self._find_matching_delimiter(source, paren_open, "(", ")")
            if paren_close is None:
                continue
            parameters = [
                parameter.strip()
                for parameter in self._split_top_level_commas(
                    source[paren_open + 1 : paren_close]
                )
                if parameter.strip()
            ]
            parameter_matches = [
                integral_constant_parameter.search(parameter)
                for parameter in parameters
            ]
            if len(parameters) != 2 or not all(parameter_matches):
                continue
            body_open = paren_close + 1
            while body_open < len(source) and source[body_open].isspace():
                body_open += 1
            while body_open < len(source) and source[body_open] != "{":
                if source[body_open] == ";":
                    break
                body_open += 1
            if body_open >= len(source) or source[body_open] != "{":
                continue
            body_close = self._find_matching_delimiter(source, body_open, "{", "}")
            if body_close is None:
                continue
            body = source[body_open + 1 : body_close]
            body = self._remove_spans(body, self._find_comment_and_literal_spans(body))
            compact = re.sub(r"\s+", "", body)
            left = re.escape(parameter_matches[0].group("value"))
            right = re.escape(parameter_matches[1].group("value"))
            arithmetic_operator = re.escape(match.group("operator"))
            direct_return = re.compile(
                r"^returnintegral_constant<"
                rf"decltype\({left}{arithmetic_operator}{right}\),"
                rf"{left}{arithmetic_operator}{right}>\{{\}};$"
            )
            named_result = re.compile(
                rf"^constexprauto(?P<result>[A-Za-z_]\w*)="
                rf"{left}{arithmetic_operator}{right};"
                r"returnintegral_constant<"
                r"decltype\((?P=result)\),(?P=result)>\{\};$"
            )
            operator_name = match.group("operator")
            if direct_return.fullmatch(compact) or named_result.fullmatch(compact):
                operators.add(operator_name)
            else:
                rejected_operators.add(operator_name)
        return operators - rejected_operators

    def _resolve_qualified_static_constant_expression(
        self,
        expression: str,
        structs_by_name: Dict[str, _MetalStructDefinition],
        resolving: Optional[Set[Tuple[str, str]]] = None,
    ) -> str:
        resolving = set(resolving or ())
        pattern = re.compile(
            r"(?<![A-Za-z0-9_])(?P<owner>[A-Za-z_][A-Za-z0-9_]*"
            r"(?:\s*::\s*[A-Za-z_][A-Za-z0-9_]*)*)\s*::\s*"
            r"(?P<member>[A-Za-z_][A-Za-z0-9_]*)\b"
        )
        ignored_spans = self._find_comment_and_literal_spans(expression)
        replacements: List[Tuple[int, int, str]] = []
        for match in pattern.finditer(expression):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            owner_name = re.sub(r"\s*::\s*", "::", match.group("owner"))
            member_name = match.group("member")
            key = (owner_name, member_name)
            if key in resolving:
                continue
            owner = structs_by_name.get(owner_name) or structs_by_name.get(
                owner_name.rsplit("::", 1)[-1]
            )
            if owner is None:
                continue
            initializer = self._resolved_static_data_member_initializers(owner).get(
                member_name
            )
            if initializer is None:
                continue
            replacement = self._resolve_qualified_static_constant_expression(
                initializer,
                structs_by_name,
                {*resolving, key},
            )
            replacements.append(
                (
                    match.start(),
                    match.end(),
                    self._static_initializer_reference(replacement),
                )
            )
        if not replacements:
            return expression
        return self._apply_text_replacements(expression, replacements)

    def _lower_concrete_const_for_loop_callbacks(
        self, source: str, *, depth: int = 0
    ) -> str:
        """Expand concrete ``const_for_loop`` callbacks in source order.

        Residual template declarations can still contain symbolic bounds. Those
        calls are deliberately left intact; only fully integral bounds and the
        narrow reference-capturing ``auto`` callback contract are expanded.
        """
        if not self._const_for_loop_contract_verified:
            return source
        result: List[str] = []
        i = 0
        while i < len(source):
            if source[i] in "\"'":
                literal, consumed = self._read_string(source, i)
                result.append(literal)
                i += consumed
                continue
            if source.startswith("//", i):
                end = source.find("\n", i)
                if end == -1:
                    result.append(source[i:])
                    break
                result.append(source[i:end])
                i = end
                continue
            if source.startswith("/*", i):
                end = source.find("*/", i + 2)
                if end == -1:
                    result.append(source[i:])
                    break
                result.append(source[i : end + 2])
                i = end + 2
                continue
            if source[i].isalpha() or source[i] == "_":
                identifier, consumed = self._read_identifier(source, i)
                identifier_end = i + consumed
                if identifier == "const_for_loop":
                    lowered = None
                    if not self._const_for_loop_is_qualified_call(source, i):
                        lowered = self._concrete_const_for_loop_replacement(
                            source, i, identifier_end, depth=depth
                        )
                    if lowered is not None:
                        end, replacement = lowered
                        result.append(replacement)
                        i = end
                        continue
                    call_end = self._const_for_loop_call_end(source, identifier_end)
                    if call_end is not None:
                        result.append(source[i:call_end])
                        i = call_end
                        continue
                result.append(identifier)
                i = identifier_end
                continue
            result.append(source[i])
            i += 1
        return "".join(result)

    def _const_for_loop_is_qualified_call(self, source: str, name_start: int) -> bool:
        ignored_spans = self._find_comment_and_literal_spans(source)
        previous = name_start - 1
        while previous >= 0:
            while previous >= 0 and source[previous].isspace():
                previous -= 1
            if previous < 0:
                return False
            containing_span = self._containing_span(previous, ignored_spans)
            if containing_span is not None and source.startswith(
                ("//", "/*"), containing_span[0]
            ):
                previous = containing_span[0] - 1
                continue
            line_start = source.rfind("\n", 0, previous + 1) + 1
            if source[line_start : previous + 1].lstrip().startswith("#"):
                previous = line_start - 2
                continue
            break
        if previous < 0:
            return False
        if source[previous] == ".":
            return True
        return previous >= 1 and source[previous - 1 : previous + 1] in {"::", "->"}

    def _const_for_loop_call_end(self, source: str, name_end: int) -> Optional[int]:
        angle_start = name_end
        while angle_start < len(source) and source[angle_start].isspace():
            angle_start += 1
        if angle_start >= len(source) or source[angle_start] != "<":
            return None
        angle_end = self._const_for_loop_template_argument_end(source, angle_start)
        if angle_end is None:
            return None
        call_open = angle_end + 1
        while call_open < len(source) and source[call_open].isspace():
            call_open += 1
        if call_open >= len(source) or source[call_open] != "(":
            return None
        call_close = self._find_matching_delimiter(source, call_open, "(", ")")
        return None if call_close is None else call_close + 1

    def _const_for_loop_template_argument_end(
        self, source: str, angle_start: int
    ) -> Optional[int]:
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        i = angle_start + 1
        while i < len(source):
            if source[i] in "\"'":
                _literal, consumed = self._read_string(source, i)
                i += consumed
                continue
            if source.startswith("//", i):
                end = source.find("\n", i)
                i = len(source) if end == -1 else end + 1
                continue
            if source.startswith("/*", i):
                end = source.find("*/", i + 2)
                i = len(source) if end == -1 else end + 2
                continue
            token = source[i]
            if token == "(":
                paren_depth += 1
            elif token == ")" and paren_depth:
                paren_depth -= 1
            elif token == "[":
                bracket_depth += 1
            elif token == "]" and bracket_depth:
                bracket_depth -= 1
            elif token == "{":
                brace_depth += 1
            elif token == "}" and brace_depth:
                brace_depth -= 1
            elif token == ">" and not (paren_depth or bracket_depth or brace_depth):
                call_open = i + 1
                while call_open < len(source) and source[call_open].isspace():
                    call_open += 1
                if call_open < len(source) and source[call_open] == "(":
                    callback_start = call_open + 1
                    while callback_start < len(source):
                        if source[callback_start].isspace():
                            callback_start += 1
                            continue
                        if source.startswith("//", callback_start):
                            end = source.find("\n", callback_start)
                            callback_start = len(source) if end == -1 else end + 1
                            continue
                        if source.startswith("/*", callback_start):
                            end = source.find("*/", callback_start + 2)
                            callback_start = len(source) if end == -1 else end + 2
                            continue
                        break
                    if callback_start < len(source) and source[callback_start] == "[":
                        return i
            i += 1
        return None

    def _concrete_const_for_loop_replacement(
        self, source: str, name_start: int, name_end: int, *, depth: int
    ) -> Optional[Tuple[int, str]]:
        angle_start = name_end
        while angle_start < len(source) and source[angle_start].isspace():
            angle_start += 1
        if angle_start >= len(source) or source[angle_start] != "<":
            return None
        angle_end = self._const_for_loop_template_argument_end(source, angle_start)
        if angle_end is None:
            return None
        bounds = [
            value.strip()
            for value in self._split_top_level_commas(
                source[angle_start + 1 : angle_end]
            )
        ]
        if len(bounds) != 3 or any(not value for value in bounds):
            return None

        values: List[int] = []
        for bound in bounds:
            folded, value = self._evaluate_static_integral_expression(bound)
            if not folded or value is None:
                return None
            values.append(value)
        start, stop, step = values
        # The supported helper contract recurses while start < stop and advances
        # by a positive step. Zero or descending steps are left intact rather than
        # assigning Python range semantics that the source helper may not have.
        if step <= 0:
            return None

        call_open = angle_end + 1
        while call_open < len(source) and source[call_open].isspace():
            call_open += 1
        if call_open >= len(source) or source[call_open] != "(":
            return None
        call_close = self._find_matching_delimiter(source, call_open, "(", ")")
        if call_close is None:
            return None
        if not self._const_for_loop_is_standalone_statement(
            source, name_start, call_close + 1
        ):
            return None
        callback = self._const_for_loop_callback_parts(
            source[call_open + 1 : call_close]
        )
        if callback is None:
            return None
        parameter, callback_body = callback

        iterations = range(start, stop, step)
        iteration_count = len(iterations)
        requested_signature = f"const_for_loop<{start}, {stop}, {step}>"
        if depth >= self.max_expansion_depth:
            self._raise_const_for_loop_expansion_limit(
                requested_signature,
                required_work_items=self._const_for_loop_expansion_work,
                detail=f"nesting depth {depth + 1} exceeds {self.max_expansion_depth}",
            )
        required_work_items = self._const_for_loop_expansion_work + iteration_count
        if required_work_items > self.max_template_specializations:
            self._raise_const_for_loop_expansion_limit(
                requested_signature,
                required_work_items=required_work_items,
                detail=(
                    f"{required_work_items} cumulative callback expansions requested"
                ),
            )
        self._const_for_loop_expansion_work = required_work_items

        expanded: List[str] = []
        for value in iterations:
            iteration_body = self._substitute_const_for_loop_parameter(
                callback_body, parameter, value
            )
            iteration_body = self._lower_const_for_loop_callback_returns(iteration_body)
            if iteration_body is None:
                return None
            iteration_body = self._lower_concrete_const_for_loop_callbacks(
                iteration_body, depth=depth + 1
            )
            expanded.append(f"{{{iteration_body}}}")
        replacement = "{" + "\n".join(expanded) + "}"
        return call_close + 1, replacement

    def _raise_const_for_loop_expansion_limit(
        self, requested_signature: str, *, required_work_items: int, detail: str
    ) -> None:
        suggested_action = (
            "raise max_template_specializations for this source pattern or reduce "
            "the compile-time loop expansion"
        )
        raise MetalTemplateSpecializationError(
            "Metal compile-time loop expansion limit exceeded for "
            f"'{requested_signature}'; {detail}, limit "
            f"{self.max_template_specializations} from "
            f"{self.template_specialization_limit_source}. Suggested action: "
            f"{suggested_action}.",
            limit=self.max_template_specializations,
            limit_source=self.template_specialization_limit_source,
            unique_specialization_count=required_work_items,
            required_work_items=required_work_items,
            requested_signature=requested_signature,
            suggested_action=suggested_action,
        )

    def _const_for_loop_is_standalone_statement(
        self, source: str, name_start: int, call_end: int
    ) -> bool:
        ignored_spans = self._find_comment_and_literal_spans(source)
        previous = name_start - 1
        while previous >= 0:
            while previous >= 0 and source[previous].isspace():
                previous -= 1
            if previous < 0:
                break
            containing_span = self._containing_span(previous, ignored_spans)
            if containing_span is not None and source.startswith(
                ("//", "/*"), containing_span[0]
            ):
                previous = containing_span[0] - 1
                continue
            line_start = source.rfind("\n", 0, previous + 1) + 1
            if source[line_start : previous + 1].lstrip().startswith("#"):
                previous = line_start - 2
                continue
            break
        if previous >= 0 and source[previous] not in "{;}":
            return False
        following = call_end
        while following < len(source) and source[following].isspace():
            following += 1
        return following < len(source) and source[following] == ";"

    def _const_for_loop_callback_parts(
        self, callback: str
    ) -> Optional[Tuple[str, str]]:
        match = re.match(
            r"\s*\[\s*&\s*\]\s*\(\s*auto\s+"
            r"(?P<parameter>[A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\{",
            callback,
        )
        if match is None:
            return None
        body_open = match.end() - 1
        body_close = self._find_matching_delimiter(callback, body_open, "{", "}")
        if body_close is None or callback[body_close + 1 :].strip():
            return None
        body = callback[body_open + 1 : body_close]
        if self._lower_const_for_loop_callback_returns(body) is None:
            return None
        parameter = match.group("parameter")
        if parameter in self._local_variable_names(body):
            return None
        if not self._const_for_loop_parameter_usage_is_safe(body, parameter):
            return None
        return parameter, body

    def _lower_const_for_loop_callback_returns(self, body: str) -> Optional[str]:
        nested_lambdas = self._const_for_loop_nested_lambda_spans(body)
        replacements: List[Tuple[int, int, str]] = []
        has_iteration_control = False
        i = 0
        while i < len(body):
            nested = self._containing_span(i, nested_lambdas)
            if nested is not None:
                i = nested[1]
                continue
            if body[i] in "\"'":
                _literal, consumed = self._read_string(body, i)
                i += consumed
                continue
            if body.startswith("//", i):
                end = body.find("\n", i + 2)
                i = len(body) if end == -1 else end + 1
                continue
            if body.startswith("/*", i):
                end = body.find("*/", i + 2)
                if end == -1:
                    return None
                i = end + 2
                continue
            if not (body[i].isalpha() or body[i] == "_"):
                i += 1
                continue
            identifier, consumed = self._read_identifier(body, i)
            identifier_end = i + consumed
            if identifier in {"break", "continue", "goto"}:
                return None
            if identifier in {"for", "while", "do", "switch"}:
                has_iteration_control = True
            if identifier != "return":
                i = identifier_end
                continue
            statement_end = self._const_for_loop_trivia_end(body, identifier_end)
            if statement_end >= len(body) or body[statement_end] != ";":
                return None
            replacements.append((i, statement_end + 1, "break;"))
            i = statement_end + 1

        if not replacements:
            return body
        if has_iteration_control:
            return None
        rewritten = self._apply_text_replacements(body, replacements)
        return f"do {{{rewritten}}} while (false);"

    def _const_for_loop_nested_lambda_spans(self, source: str) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        i = 0
        while i < len(source):
            if source[i] in "\"'":
                _literal, consumed = self._read_string(source, i)
                i += consumed
                continue
            if source.startswith("//", i):
                end = source.find("\n", i + 2)
                i = len(source) if end == -1 else end + 1
                continue
            if source.startswith("/*", i):
                end = source.find("*/", i + 2)
                i = len(source) if end == -1 else end + 2
                continue
            if source[i] != "[":
                i += 1
                continue
            capture_end = self._find_matching_delimiter(source, i, "[", "]")
            if capture_end is None:
                i += 1
                continue
            cursor = capture_end + 1
            while cursor < len(source) and source[cursor].isspace():
                cursor += 1
            if cursor < len(source) and source[cursor] == "(":
                parameter_end = self._find_matching_delimiter(source, cursor, "(", ")")
                if parameter_end is None:
                    i += 1
                    continue
                cursor = parameter_end + 1
            while cursor < len(source) and source[cursor].isspace():
                cursor += 1
            while True:
                qualifier = re.match(
                    r"(?:mutable|constexpr|consteval)\b",
                    source[cursor:],
                )
                if qualifier is None:
                    break
                cursor += qualifier.end()
                while cursor < len(source) and source[cursor].isspace():
                    cursor += 1
            if source.startswith("noexcept", cursor):
                cursor += len("noexcept")
                while cursor < len(source) and source[cursor].isspace():
                    cursor += 1
                if cursor < len(source) and source[cursor] == "(":
                    noexcept_end = self._find_matching_delimiter(
                        source, cursor, "(", ")"
                    )
                    if noexcept_end is None:
                        i += 1
                        continue
                    cursor = noexcept_end + 1
                    while cursor < len(source) and source[cursor].isspace():
                        cursor += 1
            if source.startswith("->", cursor):
                body_open = source.find("{", cursor + 2)
            else:
                body_open = (
                    cursor if cursor < len(source) and source[cursor] == "{" else -1
                )
            if body_open == -1 or ";" in source[cursor:body_open]:
                i = capture_end + 1
                continue
            body_close = self._find_matching_delimiter(source, body_open, "{", "}")
            if body_close is None:
                i = capture_end + 1
                continue
            spans.append((i, body_close + 1))
            i = body_close + 1
        return spans

    @staticmethod
    def _const_for_loop_trivia_end(source: str, start: int) -> int:
        cursor = start
        while cursor < len(source):
            if source[cursor].isspace():
                cursor += 1
                continue
            if source.startswith("//", cursor):
                end = source.find("\n", cursor + 2)
                if end == -1:
                    return len(source)
                cursor = end + 1
                continue
            if source.startswith("/*", cursor):
                end = source.find("*/", cursor + 2)
                if end == -1:
                    return len(source)
                cursor = end + 2
                continue
            break
        return cursor

    def _const_for_loop_parameter_usage_is_safe(
        self, source: str, parameter: str
    ) -> bool:
        i = 0
        while i < len(source):
            if source[i] in "\"'":
                _literal, consumed = self._read_string(source, i)
                i += consumed
                continue
            if source.startswith("//", i):
                end = source.find("\n", i)
                i = len(source) if end == -1 else end + 1
                continue
            if source.startswith("/*", i):
                end = source.find("*/", i + 2)
                i = len(source) if end == -1 else end + 2
                continue
            if not (source[i].isalpha() or source[i] == "_"):
                i += 1
                continue
            identifier, consumed = self._read_identifier(source, i)
            identifier_end = i + consumed
            if identifier != parameter or self._is_member_identifier_context(source, i):
                i = identifier_end
                continue

            member_start = identifier_end
            while member_start < len(source) and source[member_start].isspace():
                member_start += 1
            if member_start < len(source) and source[member_start] == ".":
                member_name_start = member_start + 1
                while (
                    member_name_start < len(source)
                    and source[member_name_start].isspace()
                ):
                    member_name_start += 1
                member_name, member_consumed = self._read_identifier(
                    source, member_name_start
                )
                if member_name != "value":
                    return False
                member_end = member_name_start + member_consumed
                if not self._integral_constant_value_use_is_safe(source, i, member_end):
                    return False
                i = member_end
                continue
            if not self._const_for_loop_integral_constant_binary_use(
                source, identifier_end
            ) and not self._const_for_loop_integral_conversion_use_is_safe(
                source, i, identifier_end
            ):
                return False
            i = identifier_end
        return True

    def _const_for_loop_integral_constant_binary_use(
        self, source: str, parameter_end: int
    ) -> bool:
        operator_position = parameter_end
        while operator_position < len(source) and source[operator_position].isspace():
            operator_position += 1
        if operator_position >= len(source):
            return False
        binary_operator = source[operator_position]
        if binary_operator not in self._integral_constant_binary_operators:
            return False
        operand_start = operator_position + 1
        while operand_start < len(source) and source[operand_start].isspace():
            operand_start += 1
        match = re.match(
            r"(?:(?:[A-Za-z_][A-Za-z0-9_]*\s*::\s*)*)" r"(?:Int|integral_constant)\s*<",
            source[operand_start:],
        )
        if match is None:
            return False
        angle_start = source.find("<", operand_start, operand_start + match.end())
        if angle_start == -1:
            return False
        angle_end = self._find_matching_angle(source, angle_start)
        if angle_end is None:
            return False
        construction_type = source[operand_start:angle_end]
        construction_type += ">"
        if self._integral_constant_type_parts(construction_type) is None:
            return False
        brace_start = angle_end + 1
        while brace_start < len(source) and source[brace_start].isspace():
            brace_start += 1
        if brace_start >= len(source) or source[brace_start] != "{":
            return False
        brace_end = self._find_matching_delimiter(source, brace_start, "{", "}")
        return brace_end == brace_start + 1

    def _const_for_loop_integral_conversion_use_is_safe(
        self, source: str, parameter_start: int, parameter_end: int
    ) -> bool:
        if not self._integral_constant_conversion_contract_verified:
            return False
        previous = parameter_start - 1
        while previous >= 0 and source[previous].isspace():
            previous -= 1
        following = parameter_end
        while following < len(source) and source[following].isspace():
            following += 1
        value_operators = "+-*/%<>=!&|^"
        return (previous >= 0 and source[previous] in value_operators) or (
            following < len(source) and source[following] in value_operators
        )

    def _substitute_const_for_loop_parameter(
        self, source: str, parameter: str, value: int
    ) -> str:
        result: List[str] = []
        i = 0
        while i < len(source):
            if source[i] in "\"'":
                literal, consumed = self._read_string(source, i)
                result.append(literal)
                i += consumed
                continue
            if source.startswith("//", i):
                end = source.find("\n", i)
                if end == -1:
                    result.append(source[i:])
                    break
                result.append(source[i:end])
                i = end
                continue
            if source.startswith("/*", i):
                end = source.find("*/", i + 2)
                if end == -1:
                    result.append(source[i:])
                    break
                result.append(source[i : end + 2])
                i = end + 2
                continue
            if source[i].isalpha() or source[i] == "_":
                identifier, consumed = self._read_identifier(source, i)
                identifier_end = i + consumed
                if identifier != parameter or self._is_member_identifier_context(
                    source, i
                ):
                    result.append(identifier)
                    i = identifier_end
                    continue
                member_start = identifier_end
                while member_start < len(source) and source[member_start].isspace():
                    member_start += 1
                if member_start < len(source) and source[member_start] == ".":
                    member_name_start = member_start + 1
                    while (
                        member_name_start < len(source)
                        and source[member_name_start].isspace()
                    ):
                        member_name_start += 1
                    member_name, member_consumed = self._read_identifier(
                        source, member_name_start
                    )
                    if member_name == "value":
                        result.append(self._static_integral_literal_text(value))
                        i = member_name_start + member_consumed
                        continue
                if self._const_for_loop_integral_conversion_use_is_safe(
                    source,
                    i,
                    identifier_end,
                ):
                    result.append(self._static_integral_literal_text(value))
                    i = identifier_end
                    continue
                result.append(f"integral_constant<int, {value}>{{}}")
                i = identifier_end
                continue
            result.append(source[i])
            i += 1
        return "".join(result)

    def _lower_internal_template_member_calls(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        instantiated_parameters: str,
        instantiated_body: str,
        instantiated_template_functions: Dict[str, str],
        template_methods_by_struct: Dict[str, Dict[str, List[_MetalStructMethod]]],
        methods_by_struct: Optional[Dict[str, Dict[str, _MetalStructMethod]]] = None,
        operator_call_structs: Optional[Set[str]] = None,
        rewrite_structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
    ) -> str:
        # Rewrite receiver-less calls to OTHER template member methods of the same
        # struct inside an already-substituted method body. Argument types are
        # inferred from the (now concrete) parameter types and the body's local
        # declarations; the existing argument-inference + overload selection is
        # reused, so an un-inferable / ambiguous internal call clean-fails exactly
        # like a top-level call. The instance receiver is `self` (the free
        # function's first parameter); a static sibling takes no receiver.
        sibling_overloads = template_methods_by_struct.get(struct.name, {})
        methods_by_struct = methods_by_struct or {}
        operator_call_structs = operator_call_structs or set()
        rewrite_structs_by_name = rewrite_structs_by_name or {}
        instantiated_body = self._lower_runtime_value_template_member_calls(
            struct,
            method,
            instantiated_body,
            sibling_overloads,
            instantiated_template_functions,
            template_methods_by_struct,
            methods_by_struct,
            operator_call_structs,
            rewrite_structs_by_name,
        )
        rewrite_struct_names = set(rewrite_structs_by_name)
        struct_type_aliases = self._collect_struct_type_aliases(
            instantiated_body,
            rewrite_struct_names,
            [],
            rewrite_structs_by_name,
        )
        for alias, target in struct.type_aliases.items():
            resolved_target = self._resolve_struct_type_alias_at(
                struct_type_aliases, target, 0
            )
            if resolved_target is None and target in rewrite_struct_names:
                resolved_target = target
            if resolved_target is None:
                continue
            struct_type_aliases.setdefault(alias, []).insert(
                0,
                _MetalTypeAliasBinding(
                    declaration_position=-1,
                    scope_start=0,
                    scope_end=len(instantiated_body),
                    target=resolved_target,
                ),
            )
        qualified_identifiers = set(
            re.findall(r"\b([A-Za-z_]\w*)\s*::", instantiated_body)
        )
        has_external_qualified_call = bool(
            qualified_identifiers & (rewrite_struct_names | set(struct_type_aliases))
        )
        has_external_member_call = bool(
            re.search(r"\bself\s*(?:\.|->)", instantiated_body)
            and (methods_by_struct or template_methods_by_struct)
        )
        has_concrete_sibling_call = any(
            re.search(
                rf"(?<![\w.:>]){re.escape(sibling.name)}\s*\(",
                instantiated_body,
            )
            for sibling in struct.methods
        )
        # A method whose body has no sibling-template or external struct call
        # candidate is returned unchanged.
        if (
            not sibling_overloads
            and not operator_call_structs
            and not has_external_qualified_call
            and not has_external_member_call
            and not has_concrete_sibling_call
        ):
            return instantiated_body
        # Build flat name->type views for the body scope from the concrete
        # parameters plus body-local declarations.
        local_view: Dict[str, str] = {}
        buffer_view: _MetalBufferTypeView = {}
        for parameter in self._split_top_level_commas(instantiated_parameters):
            if not parameter.strip():
                continue
            name = self._declared_data_member_name(parameter)
            if not name:
                continue
            subscriptable = self._pointer_or_array_parameter_type(parameter)
            if subscriptable is not None:
                buffer_view[name] = subscriptable
                continue
            scalar = self._function_parameter_value_type(parameter)
            if scalar:
                local_view[name] = scalar
        for statement in self._iter_simple_declarations(instantiated_body):
            name = self._declared_local_name(statement)
            if not name:
                continue
            element_type = self._declared_local_type(statement, name)
            if element_type:
                element_type = self._canonicalize_struct_scoped_type(
                    element_type, struct, rewrite_structs_by_name
                )
                local_view.setdefault(name, element_type)
        local_view["self"] = struct.name
        positioned_buffer_types = {
            name: [(0, value)] for name, value in buffer_view.items()
        }
        positioned_local_types = {
            name: [(0, value)] for name, value in local_view.items()
        }
        local_constant_owner_spans = [(0, len(instantiated_body))]
        local_constant_type_aliases = self._collect_local_type_alias_bindings(
            instantiated_body,
            local_constant_owner_spans,
        )
        local_integral_constants = self._collect_local_integral_constant_bindings(
            instantiated_body,
            local_constant_owner_spans,
            local_constant_type_aliases,
        )
        instantiated_body = self._rewrite_const_reference_alias_bindings(
            instantiated_body,
            positioned_local_types,
            rewrite_structs_by_name,
            const_receivers={"self"} if method.is_const else set(),
        )
        result: List[str] = []
        i = 0
        n = len(instantiated_body)
        while i < n:
            ch = instantiated_body[i]
            if ch in "\"'":
                literal, consumed = self._read_string(instantiated_body, i)
                result.append(literal)
                i += consumed
                continue
            if instantiated_body.startswith("//", i):
                end = instantiated_body.find("\n", i)
                if end == -1:
                    result.append(instantiated_body[i:])
                    break
                result.append(instantiated_body[i:end])
                i = end
                continue
            if instantiated_body.startswith("/*", i):
                end = instantiated_body.find("*/", i + 2)
                if end == -1:
                    result.append(instantiated_body[i:])
                    break
                result.append(instantiated_body[i : end + 2])
                i = end + 2
                continue
            if ch.isalpha() or ch == "_":
                ident, consumed = self._read_identifier(instantiated_body, i)
                ident_end = i + consumed
                rewrite = self._try_rewrite_internal_template_call(
                    struct,
                    method,
                    ident,
                    ident_end,
                    instantiated_body,
                    sibling_overloads,
                    local_view,
                    buffer_view,
                    instantiated_template_functions,
                    template_methods_by_struct,
                    local_integral_constants,
                    local_constant_type_aliases,
                )
                if rewrite is not None:
                    end, replacement = rewrite
                    result.append(replacement)
                    i = end
                    continue
                rewrite = self._try_rewrite_call_at(
                    instantiated_body,
                    i,
                    ident,
                    ident_end,
                    rewrite_struct_names,
                    methods_by_struct,
                    template_methods_by_struct,
                    operator_call_structs,
                    positioned_local_types,
                    rewrite_structs_by_name,
                    positioned_buffer_types,
                    positioned_local_types,
                    instantiated_template_functions,
                    field_variable_types=positioned_local_types,
                    field_structs_by_name=rewrite_structs_by_name,
                    struct_type_aliases=struct_type_aliases,
                    local_integral_constants=local_integral_constants,
                    type_aliases=local_constant_type_aliases,
                    type_alias_fallback_position=method.span[0],
                )
                if rewrite is not None:
                    end, replacement = rewrite
                    result.append(replacement)
                    i = end
                    continue
                result.append(ident)
                i = ident_end
                continue
            result.append(ch)
            i += 1
        return "".join(result)

    def _lower_runtime_value_template_member_calls(
        self,
        struct: _MetalStructDefinition,
        enclosing_method: _MetalStructMethod,
        body: str,
        sibling_overloads: Dict[str, List[_MetalStructMethod]],
        instantiated_template_functions: Dict[str, str],
        template_methods_by_struct: Dict[str, Dict[str, List[_MetalStructMethod]]],
        methods_by_struct: Dict[str, Dict[str, _MetalStructMethod]],
        operator_call_structs: Set[str],
        structs_by_name: Dict[str, _MetalStructDefinition],
    ) -> str:
        result: List[str] = []
        i = 0
        while i < len(body):
            if body[i] in "\"'":
                literal, consumed = self._read_string(body, i)
                result.append(literal)
                i += consumed
                continue
            if body.startswith("//", i):
                end = body.find("\n", i)
                if end == -1:
                    result.append(body[i:])
                    break
                result.append(body[i:end])
                i = end
                continue
            if body.startswith("/*", i):
                end = body.find("*/", i + 2)
                if end == -1:
                    result.append(body[i:])
                    break
                result.append(body[i : end + 2])
                i = end + 2
                continue
            if body[i].isalpha() or body[i] == "_":
                ident, consumed = self._read_identifier(body, i)
                ident_end = i + consumed
                rewrite = self._try_lower_runtime_value_template_member_call(
                    struct,
                    enclosing_method,
                    ident,
                    ident_end,
                    body,
                    sibling_overloads,
                    instantiated_template_functions,
                    template_methods_by_struct,
                    methods_by_struct,
                    operator_call_structs,
                    structs_by_name,
                )
                if rewrite is not None:
                    end, replacement = rewrite
                    result.append(replacement)
                    i = end
                    continue
                result.append(ident)
                i = ident_end
                continue
            result.append(body[i])
            i += 1
        return "".join(result)

    def _try_lower_runtime_value_template_member_call(
        self,
        struct: _MetalStructDefinition,
        enclosing_method: _MetalStructMethod,
        ident: str,
        ident_end: int,
        body: str,
        sibling_overloads: Dict[str, List[_MetalStructMethod]],
        instantiated_template_functions: Dict[str, str],
        template_methods_by_struct: Dict[str, Dict[str, List[_MetalStructMethod]]],
        methods_by_struct: Dict[str, Dict[str, _MetalStructMethod]],
        operator_call_structs: Set[str],
        structs_by_name: Dict[str, _MetalStructDefinition],
    ) -> Optional[Tuple[int, str]]:
        overloads = sibling_overloads.get(ident)
        if not overloads:
            return None
        previous = ident_end - len(ident) - 1
        while previous >= 0 and body[previous].isspace():
            previous -= 1
        if previous >= 0 and (
            body[previous] == "."
            or (previous >= 1 and body[previous - 1 : previous + 1] in {"->", "::"})
        ):
            return None
        call_suffix = self._member_template_call_suffix(body, ident_end)
        if call_suffix is None:
            return None
        arg_open, explicit_arguments = call_suffix
        if explicit_arguments is None:
            return None
        arg_close = self._find_matching_delimiter(body, arg_open, "(", ")")
        if arg_close is None or body[arg_open + 1 : arg_close].strip():
            return None

        candidates = [
            method
            for method in overloads
            if self._runtime_value_template_method_is_safe(method, explicit_arguments)
            and (method.is_const if enclosing_method.is_const else True)
        ]
        if not enclosing_method.is_const:
            non_const = [method for method in candidates if not method.is_const]
            if non_const:
                candidates = non_const
        if len(candidates) != 1:
            return None
        selected = candidates[0]
        self._reject_selected_reference_return(
            struct,
            selected,
            self._canonicalize_struct_scoped_type(
                selected.return_type,
                struct,
                structs_by_name,
            ),
            f"{struct.name}::{selected.name}<" f"{', '.join(explicit_arguments)}>()",
            source_location=self._source_location_for_offsets(
                body,
                ident_end - len(ident),
                arg_close + 1,
            ),
        )
        parameter_types = [
            selected.template_parameter_types[name]
            for name in selected.template_parameters
        ]
        type_suffix = "_".join(
            re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
            for value in parameter_types
        )
        free_name = (
            f"{self._struct_member_free_name(struct.name, ident, False)}"
            f"__runtime_values__{type_suffix}"
        )
        if selected.is_const:
            free_name += "__const"
        if free_name not in instantiated_template_functions:
            parameters = ", ".join(
                f"{type_name} {name}"
                for name, type_name in zip(
                    selected.template_parameters, parameter_types
                )
            )
            runtime_method = _MetalStructMethod(
                name=selected.name,
                free_name=free_name,
                is_static=selected.is_static,
                is_operator_call=selected.is_operator_call,
                return_type=selected.return_type,
                parameters=parameters,
                parameter_names=list(selected.template_parameters),
                body=selected.body,
                span=selected.span,
                is_const=selected.is_const,
            )
            instantiated_template_functions[free_name] = self._emit_free_function(
                struct,
                runtime_method,
                instantiated_template_functions=instantiated_template_functions,
                template_methods_by_struct=template_methods_by_struct,
                methods_by_struct=methods_by_struct,
                operator_call_structs=operator_call_structs,
                structs_by_name=structs_by_name,
            )
        call_arguments = ", ".join(explicit_arguments)
        if selected.is_static:
            replacement = f"{free_name}({call_arguments})"
        else:
            replacement = f"{free_name}(self, {call_arguments})"
        return arg_close + 1, replacement

    def _runtime_value_template_method_is_safe(
        self,
        method: _MetalStructMethod,
        explicit_arguments: List[str],
    ) -> bool:
        if method.parameters.strip() not in {"", "void"}:
            return False
        if len(method.template_parameters) != len(explicit_arguments):
            return False
        if set(method.template_parameter_types) != set(method.template_parameters):
            return False
        if (
            method.template_constraints
            or method.template_parameter_defaults
            or method.variadic_template_parameters
        ):
            return False
        for name in method.template_parameters:
            type_name = method.template_parameter_types[name]
            if self._is_integral_concrete_type(type_name) is not True:
                return False
            if re.search(rf"\b{re.escape(name)}\b", method.return_type):
                return False
        match = re.fullmatch(r"\s*return\s+(?P<expr>[^;]+);\s*", method.body)
        if match is None:
            return False
        expression = match.group("expr")
        if re.search(r"[<>{}?:,=]", expression):
            return False
        if re.search(r"\b(?:sizeof|alignof|decltype)\s*\(", expression):
            return False
        if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", expression):
            return False
        return all(
            re.search(rf"\b{re.escape(name)}\b", expression) is not None
            for name in method.template_parameters
        )

    def _try_rewrite_internal_template_call(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        ident: str,
        ident_end: int,
        body: str,
        sibling_overloads: Dict[str, List[_MetalStructMethod]],
        local_view: Dict[str, str],
        buffer_view: _MetalBufferTypeView,
        instantiated_template_functions: Dict[str, str],
        template_methods_by_struct: Dict[str, Dict[str, List[_MetalStructMethod]]],
        local_integral_constants: Dict[str, List[_MetalIntegralConstantBinding]],
        local_type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
    ) -> Optional[Tuple[int, str]]:
        # A receiver-less `name(args)` where `name` is a sibling member method
        # (template OR concrete) is lowered to its concrete free function with an
        # implicit `self` receiver. A member access (`x.name(...)`) is excluded.
        # Skip a `.name(`/`->name(` member access — handled by the normal call
        # rewriter at the top level, not as an implicit-this call here.
        previous = ident_end - len(ident) - 1
        while previous >= 0 and body[previous].isspace():
            previous -= 1
        if previous >= 0 and (
            body[previous] == "."
            or (previous >= 1 and body[previous - 1 : previous + 1] in {"->", "::"})
        ):
            return None

        # An implicit-this `operator()(args)` call (e.g. the sizeof==8 reduction
        # combining elements). The declarator is `operator` then an empty `()`
        # then the real argument list.
        if ident == "operator":
            after_operator = ident_end
            while after_operator < len(body) and body[after_operator].isspace():
                after_operator += 1
            if (
                after_operator < len(body)
                and body[after_operator] == "("
                and body[after_operator + 1 : after_operator + 2] == ")"
            ):
                return self._rewrite_internal_operator_call(
                    struct,
                    method,
                    after_operator,
                    body,
                    sibling_overloads,
                    local_view,
                    buffer_view,
                    instantiated_template_functions,
                    template_methods_by_struct,
                    local_integral_constants,
                    local_type_aliases,
                )
            return None

        template_overloads = sibling_overloads.get(ident)
        concrete_siblings = [m for m in struct.methods if m.name == ident]
        if not template_overloads and not concrete_siblings:
            return None
        call_suffix = self._member_template_call_suffix(body, ident_end)
        if call_suffix is None:
            return None
        arg_open, explicit_template_arguments = call_suffix
        arg_close = self._find_matching_delimiter(body, arg_open, "(", ")")
        if arg_close is None:
            return None
        raw_args = body[arg_open + 1 : arg_close]
        args = raw_args.strip()

        # A CONCRETE sibling has a fixed signature; lower it directly (no overload
        # selection / argument typing needed). When both a concrete and a template
        # sibling share the name, prefer the concrete one only if there is exactly
        # one — otherwise fall through to the template path / clean-fail.
        if (
            concrete_siblings
            and not template_overloads
            and explicit_template_arguments is None
        ):
            reference_methods = [
                candidate
                for candidate in concrete_siblings
                if self._method_returns_lvalue_reference(candidate)
            ]
            if reference_methods:
                concrete = self._select_direct_reference_method(
                    body,
                    struct,
                    reference_methods,
                    arg_open,
                    receiver_is_const=method.is_const,
                )
                return self._build_direct_reference_accessor_rewrite(
                    body,
                    struct,
                    concrete,
                    receiver="self",
                    arg_open=arg_open,
                    call_start=ident_end - len(ident),
                    enclosing_return_type=method.return_type,
                )
            if len(concrete_siblings) != 1:
                return None
            concrete = concrete_siblings[0]
            self._reject_reference_returning_method_call(
                body,
                struct,
                concrete,
                arg_open,
                arg_close,
            )
            if concrete.is_static:
                replacement = (
                    f"{concrete.free_name}({args})"
                    if args
                    else f"{concrete.free_name}()"
                )
            elif args:
                replacement = f"{concrete.free_name}(self, {args})"
            else:
                replacement = f"{concrete.free_name}(self)"
            return arg_close + 1, replacement

        overloads = template_overloads
        call_arguments = [
            argument
            for argument in self._split_top_level_commas(raw_args)
            if argument.strip()
        ]
        representative = overloads[0]
        signature = f"{struct.name}::{method.name} -> {ident}({raw_args.strip()})"
        local_constant_values = self._local_integral_constants_at(
            local_integral_constants,
            arg_open,
        )
        concrete_argument_types: List[Optional[str]] = []
        for argument_index, argument in enumerate(call_arguments):
            inference_argument = self._substitute_template_argument_static_constants(
                argument,
                local_constant_values,
            )
            inferred = self._infer_argument_type(
                inference_argument,
                buffer_view,
                local_view,
                {},
            )
            if inferred is None:
                if not self._template_member_argument_requires_inference(
                    representative,
                    argument_index,
                    explicit_template_arguments,
                ):
                    concrete_argument_types.append(None)
                    continue
                raise MetalStructMethodError(
                    "Cannot lower template member method "
                    f"'{struct.name}::{method.name}': internal call to "
                    f"'{ident}' has argument '{argument.strip()}' whose type "
                    f"could not be inferred. Requested call: {signature}.",
                    struct_name=struct.name,
                    method_name=method.name,
                    requested_signature=signature,
                )
            concrete_argument_types.append(inferred)

        resolved_explicit_template_arguments = (
            [
                self._substitute_template_argument_static_constants(
                    argument,
                    local_constant_values,
                )
                for argument in explicit_template_arguments
            ]
            if explicit_template_arguments is not None
            else None
        )

        free_name = self._instantiate_template_member_overload(
            struct,
            overloads,
            concrete_argument_types,
            signature,
            instantiated_template_functions,
            template_methods_by_struct,
            explicit_template_arguments=resolved_explicit_template_arguments,
            argument_type_aliases=local_type_aliases,
            argument_type_position=arg_open,
            argument_type_fallback_position=method.span[0],
        )
        args = self._expanded_template_member_call_arguments(free_name, raw_args)
        if representative.is_static:
            replacement = f"{free_name}({args})" if args else f"{free_name}()"
        elif args:
            replacement = f"{free_name}(self, {args})"
        else:
            replacement = f"{free_name}(self)"
        return arg_close + 1, replacement

    def _template_member_argument_requires_inference(
        self,
        method: _MetalStructMethod,
        argument_index: int,
        explicit_template_arguments: Optional[List[str]] = None,
    ) -> bool:
        parameters = [
            parameter
            for parameter in self._split_top_level_commas(method.parameters)
            if parameter.strip()
        ]
        if argument_index >= len(parameters):
            return True

        parameter_type = self._normalize_function_parameter_type_text(
            parameters[argument_index]
        )
        referenced = {
            name
            for name in method.template_parameters
            if re.search(rf"\b{re.escape(name)}\b", parameter_type)
        }
        if not referenced:
            return False

        explicit_count = len(explicit_template_arguments or [])
        available_without_inference = set(method.template_parameters[:explicit_count])
        return not referenced.issubset(available_without_inference)

    def _member_template_call_suffix(
        self, code: str, name_end: int
    ) -> Optional[Tuple[int, Optional[List[str]]]]:
        """Return a member call's argument opener and explicit template args."""
        index = name_end
        while index < len(code) and code[index].isspace():
            index += 1

        explicit_template_arguments: Optional[List[str]] = None
        if index < len(code) and code[index] == "<":
            angle_end = self._find_matching_angle(code, index)
            if angle_end is None:
                return None
            explicit_template_arguments = [
                argument.strip()
                for argument in self._split_top_level_commas(
                    code[index + 1 : angle_end]
                )
                if argument.strip()
            ]
            index = angle_end + 1
            while index < len(code) and code[index].isspace():
                index += 1

        if index >= len(code) or code[index] != "(":
            return None
        return index, explicit_template_arguments

    def _rewrite_internal_operator_call(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        empty_paren_open: int,
        body: str,
        sibling_overloads: Dict[str, List[_MetalStructMethod]],
        local_view: Dict[str, str],
        buffer_view: _MetalBufferTypeView,
        instantiated_template_functions: Dict[str, str],
        template_methods_by_struct: Dict[str, Dict[str, List[_MetalStructMethod]]],
        local_integral_constants: Dict[str, List[_MetalIntegralConstantBinding]],
        local_type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
    ) -> Optional[Tuple[int, str]]:
        # Lower an implicit-this `operator()(args)` call. `empty_paren_open` is the
        # `(` of the empty `()` after `operator`; the real argument list follows.
        empty_close = self._find_matching_delimiter(body, empty_paren_open, "(", ")")
        if empty_close is None:
            return None
        arg_open = empty_close + 1
        while arg_open < len(body) and body[arg_open].isspace():
            arg_open += 1
        if arg_open >= len(body) or body[arg_open] != "(":
            return None
        arg_close = self._find_matching_delimiter(body, arg_open, "(", ")")
        if arg_close is None:
            return None
        raw_args = body[arg_open + 1 : arg_close]
        args = raw_args.strip()
        # The caller has not yet emitted the `operator` token, so the returned
        # replacement fully replaces `operator()(args)` from that token onward;
        # only the end offset is needed.
        concrete = next((m for m in struct.methods if m.name == "operator()"), None)
        if concrete is not None:
            self._reject_reference_returning_method_call(
                body,
                struct,
                concrete,
                arg_open,
                arg_close,
            )
            replacement = (
                f"{concrete.free_name}(self, {args})"
                if args
                else f"{concrete.free_name}(self)"
            )
            return arg_close + 1, replacement
        template_overloads = sibling_overloads.get("operator()")
        if not template_overloads:
            return None
        call_arguments = [
            argument
            for argument in self._split_top_level_commas(raw_args)
            if argument.strip()
        ]
        signature = f"{struct.name}::{method.name} -> operator()({args})"
        local_constant_values = self._local_integral_constants_at(
            local_integral_constants,
            arg_open,
        )
        concrete_argument_types: List[str] = []
        for argument in call_arguments:
            inference_argument = self._substitute_template_argument_static_constants(
                argument,
                local_constant_values,
            )
            inferred = self._infer_argument_type(
                inference_argument,
                buffer_view,
                local_view,
                {},
            )
            if inferred is None:
                raise MetalStructMethodError(
                    "Cannot lower template member method "
                    f"'{struct.name}::{method.name}': internal call to "
                    f"'operator()' has argument '{argument.strip()}' whose type "
                    f"could not be inferred. Requested call: {signature}.",
                    struct_name=struct.name,
                    method_name=method.name,
                    requested_signature=signature,
                )
            concrete_argument_types.append(inferred)
        free_name = self._instantiate_template_member_overload(
            struct,
            template_overloads,
            concrete_argument_types,
            signature,
            instantiated_template_functions,
            template_methods_by_struct,
            argument_type_aliases=local_type_aliases,
            argument_type_position=arg_open,
            argument_type_fallback_position=method.span[0],
        )
        args = self._expanded_template_member_call_arguments(free_name, raw_args)
        replacement = f"{free_name}(self, {args})" if args else f"{free_name}(self)"
        return arg_close + 1, replacement

    def _build_template_call_rewrite(
        self,
        code: str,
        receiver: Optional[str],
        free_name: str,
        arg_open: int,
        arg_close: int,
        *,
        raw_arguments: Optional[str] = None,
    ) -> Tuple[int, str]:
        args = self._expanded_template_member_call_arguments(
            free_name,
            (
                code[arg_open + 1 : arg_close]
                if raw_arguments is None
                else raw_arguments
            ),
        )
        if receiver is None:
            # Static template member call: no `self` receiver.
            replacement = f"{free_name}({args})" if args else f"{free_name}()"
        elif args:
            replacement = f"{free_name}({receiver}, {args})"
        else:
            replacement = f"{free_name}({receiver})"
        return arg_close + 1, replacement

    def _expanded_template_member_call_arguments(
        self,
        free_name: str,
        raw_arguments: str,
    ) -> str:
        arguments = [
            argument.strip()
            for argument in self._split_top_level_commas(raw_arguments)
            if argument.strip()
        ]
        selection = self._instantiated_template_member_calls.get(free_name)
        if selection is None:
            return raw_arguments.strip()
        struct, method, bindings, structs_by_name = selection
        parameters = [
            parameter
            for parameter in self._split_top_level_commas(method.parameters)
            if parameter.strip()
        ]
        if len(arguments) >= len(parameters):
            return raw_arguments.strip()
        for parameter in parameters[len(arguments) :]:
            declaration, default = self._split_top_level_assignment(parameter)
            if default is None or not default.strip():
                return raw_arguments.strip()
            resolved_default = self._replace_identifiers(default, bindings).strip()
            resolved_default = self._substitute_static_data_member_initializers(
                struct, method, resolved_default
            )
            resolved_default = self._replace_identifiers(
                resolved_default, bindings
            ).strip()
            if resolved_default == "{}":
                type_name = self._function_parameter_value_type(declaration)
                type_name = self._replace_identifiers(type_name, bindings)
                type_name = self._canonicalize_struct_scoped_type(
                    type_name, struct, structs_by_name
                )
                resolved_default = f"{type_name}{{}}"
            arguments.append(resolved_default)
        return ", ".join(arguments)

    def _method_returns_lvalue_reference(self, method: _MetalStructMethod) -> bool:
        return bool(
            re.search(
                r"(?<!&)\&\s*$",
                self._normalize_template_argument_text(method.return_type),
            )
        )

    def _concrete_reference_methods(
        self,
        struct: _MetalStructDefinition,
        method_name: str,
    ) -> List[_MetalStructMethod]:
        return [
            method
            for method in struct.methods
            if method.name == method_name
            and self._method_returns_lvalue_reference(method)
        ]

    def _select_direct_reference_method(
        self,
        code: str,
        struct: _MetalStructDefinition,
        methods: List[_MetalStructMethod],
        arg_open: int,
        *,
        receiver_is_const: Optional[bool],
    ) -> _MetalStructMethod:
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        representative = methods[0]
        if arg_close is None:
            self._reject_reference_returning_method_call(
                code, struct, representative, arg_open, arg_open
            )
            raise AssertionError("reference-return rejection must raise")

        arguments = [
            argument
            for argument in self._split_top_level_commas(code[arg_open + 1 : arg_close])
            if argument.strip()
        ]
        if receiver_is_const is True:
            candidates = [
                method
                for method in methods
                if not method.is_static
                and method.is_const
                and re.search(r"\bconst\b", method.return_type) is not None
                and len(method.parameter_names) == len(arguments)
            ]
        else:
            candidates = [
                method
                for method in methods
                if not method.is_static
                and not method.is_const
                and re.search(r"\bconst\b", method.return_type) is None
                and len(method.parameter_names) == len(arguments)
            ]
        if receiver_is_const is not None and len(candidates) == 1:
            candidate = candidates[0]
            normalized_parameters = self._normalize_template_argument_text(
                candidate.parameters
            )
            competing_overloads = [
                method
                for method in [*struct.methods, *struct.template_methods]
                if method is not candidate
                and method.name == candidate.name
                and len(method.parameter_names) == len(arguments)
            ]
            matching_counterparts = [
                method
                for method in competing_overloads
                if method.is_const != candidate.is_const
                and not method.is_static
                and any(method is concrete for concrete in struct.methods)
                and self._normalize_template_argument_text(method.parameters)
                == normalized_parameters
            ]
            if len(matching_counterparts) <= 1 and len(matching_counterparts) == len(
                competing_overloads
            ):
                return candidate

        self._reject_reference_returning_method_call(
            code,
            struct,
            candidates[0] if candidates else representative,
            arg_open,
            arg_close,
        )
        raise AssertionError("reference-return rejection must raise")

    def _simple_receiver_constness(
        self,
        code: str,
        struct_name: str,
        receiver: str,
        call_start: int,
        *,
        type_aliases: Optional[Dict[str, List[_MetalTypeAliasBinding]]] = None,
    ) -> Optional[bool]:
        if re.fullmatch(r"[A-Za-z_]\w*", receiver) is None:
            return None
        ignored = self._find_comment_and_literal_spans(code)
        lexical_scopes = self._find_lexical_brace_scopes(code)
        pattern = re.compile(
            rf"(?P<leading>(?:(?:const|constant|device|thread|threadgroup|volatile)\s+)*)"
            rf"\b(?P<type>[A-Za-z_]\w*)\b"
            rf"(?P<trailing>\s+const\b)?\s*(?:[&*]\s*)*"
            rf"\b{re.escape(receiver)}\b\s*(?=[,;)=\[{{(])"
        )
        matches = []
        for match in pattern.finditer(code, 0, call_start):
            if self._containing_span(match.start(), ignored) is not None:
                continue
            scope = self._innermost_lexical_scope(
                lexical_scopes,
                match.start(),
                len(code),
            )
            if scope[0] <= call_start < scope[1]:
                matches.append(match)
        if not matches:
            return None
        declaration = matches[-1]
        declared_type = declaration.group("type")
        if declared_type != struct_name:
            declared_type = (
                self._resolve_struct_type_alias_at(
                    type_aliases,
                    declared_type,
                    declaration.start("type"),
                )
                if type_aliases is not None
                else None
            )
        if declared_type != struct_name:
            return None
        leading = declaration.group("leading") or ""
        if re.search(r"\bconstant\b", leading):
            return None
        return bool(re.search(r"\bconst\b", leading) or declaration.group("trailing"))

    def _rewrite_const_reference_alias_bindings(
        self,
        code: str,
        variable_types: Dict[str, List[Tuple[int, str]]],
        structs_by_name: Dict[str, _MetalStructDefinition],
        *,
        const_receivers: Set[str],
    ) -> str:
        """Inline proven local const-reference aliases without making a copy."""
        if "auto" not in code or "&" not in code:
            return code
        pattern = re.compile(
            r"\b(?P<qualifiers>(?:(?:thread|const)\s+)+)"
            r"auto\s*&\s*(?P<alias>[A-Za-z_]\w*)\s*=\s*"
        )
        working = code
        while True:
            masked = self._mask_comments_and_literals(working)
            bindings = list(pattern.finditer(masked))
            if not bindings:
                return working
            scopes = self._find_lexical_brace_scopes(working)
            call_argument_spans = self._potential_call_argument_spans(
                working,
                masked,
                0,
                len(working),
            )
            rewrites: List[Tuple[int, int, str]] = []
            for binding in bindings:
                qualifiers = binding.group("qualifiers").split()
                if qualifiers.count("const") != 1 or qualifiers.count("thread") > 1:
                    continue
                boundary = max(
                    masked.rfind(token, 0, binding.start()) for token in (";", "{", "}")
                )
                if masked[boundary + 1 : binding.start()].strip():
                    continue
                initializer = self._parse_dotted_reference_alias_initializer(
                    working,
                    masked,
                    binding.end(),
                )
                if initializer is None:
                    continue
                receiver_parts, method_name, arg_open, arg_close, statement_end = (
                    initializer
                )
                if len(receiver_parts) < 2:
                    continue

                receiver_type = self._resolve_declared_type_at(
                    variable_types,
                    receiver_parts[0],
                    binding.start(),
                )
                receiver_info = self._nested_member_receiver_info(
                    receiver_type,
                    structs_by_name,
                )
                if receiver_info is None or receiver_info[1] != ".":
                    continue
                current_struct = structs_by_name.get(receiver_info[0])
                if current_struct is None:
                    continue
                if receiver_parts[0] not in const_receivers:
                    receiver_is_const = self._simple_receiver_constness(
                        working,
                        current_struct.name,
                        receiver_parts[0],
                        binding.start(),
                    )
                    if receiver_is_const is not True:
                        continue

                value_chain_is_valid = True
                for member_name in receiver_parts[1:]:
                    member = next(
                        (
                            item
                            for item in current_struct.data_members
                            if item.name == member_name
                        ),
                        None,
                    )
                    if (
                        member is None
                        or member.array_suffix
                        or "*" in member.type_text
                        or "&" in member.type_text
                    ):
                        value_chain_is_valid = False
                        break
                    member_type = self._normalize_inferred_type(member.type_text)
                    current_struct = structs_by_name.get(member_type)
                    if current_struct is None:
                        value_chain_is_valid = False
                        break
                if not value_chain_is_valid:
                    continue

                reference_methods = self._concrete_reference_methods(
                    current_struct,
                    method_name,
                )
                if not reference_methods:
                    continue
                method = self._select_direct_reference_method(
                    working,
                    current_struct,
                    reference_methods,
                    arg_open,
                    receiver_is_const=True,
                )
                _call_end, storage = self._direct_reference_accessor_rewrite(
                    working,
                    current_struct,
                    method,
                    receiver=".".join(receiver_parts),
                    arg_open=arg_open,
                )

                scope_start, scope_end = self._innermost_lexical_scope(
                    scopes,
                    binding.start(),
                    len(working),
                )
                if not (scope_start <= binding.start() < statement_end <= scope_end):
                    continue
                alias_replacements = self._const_reference_alias_use_replacements(
                    working,
                    masked,
                    binding.group("alias"),
                    statement_end,
                    scope_end,
                    storage,
                )
                if (
                    alias_replacements is None
                    or not self._reference_alias_capture_is_stable(
                        working,
                        masked,
                        call_argument_spans,
                        scope_start,
                        statement_end,
                        scope_end,
                        storage,
                        {
                            match.group(0)
                            for argument in self._split_top_level_commas(
                                working[arg_open + 1 : arg_close]
                            )
                            for match in re.finditer(r"\b[A-Za-z_]\w*\b", argument)
                            if not self._is_member_identifier_context(
                                argument, match.start()
                            )
                        },
                    )
                ):
                    continue
                erased_binding = "".join(
                    "\n" if char == "\n" else " "
                    for char in working[binding.start() : statement_end]
                )
                rewrites.extend(
                    [
                        (binding.start(), statement_end, erased_binding),
                        *alias_replacements,
                    ]
                )
            if not rewrites:
                return working
            working = self._apply_text_replacements(working, rewrites)

    def _parse_dotted_reference_alias_initializer(
        self,
        code: str,
        masked: str,
        start: int,
    ) -> Optional[Tuple[List[str], str, int, int, int]]:
        parts: List[str] = []
        cursor = start
        arg_open = -1
        while True:
            while cursor < len(masked) and masked[cursor].isspace():
                cursor += 1
            part, consumed = self._read_identifier(masked, cursor)
            if not part:
                return None
            parts.append(part)
            cursor += consumed
            while cursor < len(masked) and masked[cursor].isspace():
                cursor += 1
            if cursor < len(masked) and masked[cursor] == ".":
                cursor += 1
                continue
            if cursor < len(masked) and masked[cursor] == "(":
                arg_open = cursor
                break
            return None
        if len(parts) < 3:
            return None
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            return None
        statement_end = arg_close + 1
        while statement_end < len(masked) and masked[statement_end].isspace():
            statement_end += 1
        if statement_end >= len(masked) or masked[statement_end] != ";":
            return None
        return parts[:-1], parts[-1], arg_open, arg_close, statement_end + 1

    def _const_reference_alias_use_replacements(
        self,
        code: str,
        masked: str,
        alias: str,
        lifetime_start: int,
        lifetime_end: int,
        storage: str,
    ) -> Optional[List[Tuple[int, int, str]]]:
        lifetime = code[lifetime_start:lifetime_end]
        if any(
            self._declared_local_name(statement) == alias
            for statement in self._iter_simple_declarations(lifetime)
        ):
            return None
        replacements: List[Tuple[int, int, str]] = []
        alias_pattern = re.compile(rf"\b{re.escape(alias)}\b")
        for match in alias_pattern.finditer(masked, lifetime_start, lifetime_end):
            if self._is_member_identifier_context(masked, match.start()):
                continue
            cursor = match.end()
            while cursor < lifetime_end and masked[cursor].isspace():
                cursor += 1
            if cursor >= lifetime_end or masked[cursor] != "[":
                return None
            while cursor < lifetime_end and masked[cursor] == "[":
                close = self._find_matching_delimiter(code, cursor, "[", "]")
                if close is None or close >= lifetime_end:
                    return None
                cursor = close + 1
                while cursor < lifetime_end and masked[cursor].isspace():
                    cursor += 1
            previous = match.start() - 1
            while previous >= lifetime_start and masked[previous].isspace():
                previous -= 1
            if previous >= lifetime_start and masked[previous] == "&":
                return None
            statement_boundary = max(
                masked.rfind(token, lifetime_start, match.start())
                for token in (";", "{", "}")
            )
            prefix = masked[statement_boundary + 1 : match.start()]
            if re.search(r"&\s*[A-Za-z_]\w*\s*=\s*$", prefix):
                return None
            if re.match(
                r"(?:\+\+|--|(?:(?:<<|>>|[+\-*/%&|^])?=(?!=)))",
                masked[cursor:],
            ):
                return None
            replacements.append((match.start(), match.end(), storage))
        return replacements or None

    def _reference_alias_capture_is_stable(
        self,
        code: str,
        masked: str,
        call_argument_spans: List[Tuple[int, int]],
        capture_scope_start: int,
        lifetime_start: int,
        lifetime_end: int,
        storage: str,
        argument_captures: Set[str],
    ) -> bool:
        captured = {
            match.group(0)
            for match in re.finditer(r"\b[A-Za-z_]\w*\b", storage)
            if not self._is_member_identifier_context(storage, match.start())
        }
        lifetime = code[lifetime_start:lifetime_end]
        if any(
            self._declared_local_name(statement) in captured
            for statement in self._iter_simple_declarations(lifetime)
        ):
            return False
        for name in captured:
            name_pattern = re.compile(rf"\b{re.escape(name)}\b")
            for match in name_pattern.finditer(
                masked,
                capture_scope_start,
                lifetime_end,
            ):
                if self._is_member_identifier_context(masked, match.start()):
                    continue
                previous = match.start() - 1
                while previous >= capture_scope_start and masked[previous].isspace():
                    previous -= 1
                statement_boundary = max(
                    masked.rfind(token, capture_scope_start, match.start())
                    for token in (";", "{", "}")
                )
                prefix = masked[statement_boundary + 1 : match.start()]
                reference_escape = (
                    previous >= capture_scope_start and masked[previous] == "&"
                ) or re.search(r"&\s*[A-Za-z_]\w*\s*=\s*$", prefix)
                if reference_escape and (
                    match.start() >= lifetime_start or name in argument_captures
                ):
                    return False
                if match.start() < lifetime_start:
                    continue
                if previous >= lifetime_start - 1 and masked[
                    max(lifetime_start, previous - 1) : previous + 1
                ] in {"++", "--"}:
                    return False
                if name in argument_captures and any(
                    lifetime_start <= start < match.start() < end <= lifetime_end
                    for start, end in call_argument_spans
                ):
                    return False
                cursor = self._reference_capture_access_end(
                    code,
                    masked,
                    match.end(),
                    lifetime_end,
                )
                if re.match(
                    r"(?:\+\+|--|(?:(?:<<|>>|[+\-*/%&|^])?=(?!=)))",
                    masked[cursor:lifetime_end],
                ):
                    return False
        return True

    def _reference_capture_access_end(
        self,
        code: str,
        masked: str,
        start: int,
        limit: int,
    ) -> int:
        cursor = start
        while cursor < limit:
            while cursor < limit and masked[cursor].isspace():
                cursor += 1
            if masked.startswith("->", cursor):
                cursor += 2
            elif masked[cursor : cursor + 1] == ".":
                cursor += 1
            elif masked[cursor : cursor + 1] == "[":
                close = self._find_matching_delimiter(code, cursor, "[", "]")
                if close is None or close >= limit:
                    return cursor
                cursor = close + 1
                continue
            else:
                break
            while cursor < limit and masked[cursor].isspace():
                cursor += 1
            _member, consumed = self._read_identifier(masked, cursor)
            if not consumed:
                break
            cursor += consumed
        while cursor < limit and masked[cursor].isspace():
            cursor += 1
        return cursor

    def _potential_call_argument_spans(
        self,
        code: str,
        masked: str,
        start: int,
        end: int,
    ) -> List[Tuple[int, int]]:
        pattern = re.compile(
            r"\b(?P<callee>[A-Za-z_]\w*" r"(?:\s*(?:\.|->|::)\s*[A-Za-z_]\w*)*)\s*\("
        )
        spans: List[Tuple[int, int]] = []
        control_names = {"if", "for", "while", "switch", "sizeof", "decltype"}
        for match in pattern.finditer(masked, start, end):
            callee_names = IDENTIFIER_RE.findall(match.group("callee"))
            if not callee_names or callee_names[-1] in control_names:
                continue
            open_paren = match.end() - 1
            close_paren = self._find_matching_delimiter(code, open_paren, "(", ")")
            if close_paren is not None and close_paren < end:
                spans.append((open_paren, close_paren))
        return spans

    def _mask_comments_and_literals(self, code: str) -> str:
        masked = list(code)
        for start, end in self._find_comment_and_literal_spans(code):
            for index in range(start, end):
                if masked[index] != "\n":
                    masked[index] = " "
        return "".join(masked)

    def _build_direct_reference_accessor_rewrite(
        self,
        code: str,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        *,
        receiver: str,
        arg_open: int,
        call_start: int,
        enclosing_return_type: Optional[str] = None,
    ) -> Tuple[int, str]:
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            self._reject_reference_returning_method_call(
                code, struct, method, arg_open, arg_open
            )
            raise AssertionError("reference-return rejection must raise")

        if self._reference_call_binds_or_escapes(
            code,
            call_start,
            enclosing_return_type=enclosing_return_type,
        ):
            self._reject_reference_returning_method_call(
                code, struct, method, arg_open, arg_close
            )

        return self._direct_reference_accessor_rewrite(
            code,
            struct,
            method,
            receiver=receiver,
            arg_open=arg_open,
        )

    def _direct_reference_accessor_rewrite(
        self,
        code: str,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        *,
        receiver: str,
        arg_open: int,
    ) -> Tuple[int, str]:
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            self._reject_reference_returning_method_call(
                code, struct, method, arg_open, arg_open
            )
            raise AssertionError("reference-return rejection must raise")

        expression = self._direct_reference_return_expression(struct, method)
        if expression is None:
            self._reject_reference_returning_method_call(
                code, struct, method, arg_open, arg_close
            )

        match = re.fullmatch(
            r"self\s*\.\s*(?P<member>[A-Za-z_]\w*)"
            r"(?P<indices>(?:\s*\[[^\[\]]+\])*)",
            expression,
        )
        if match is None:
            self._reject_reference_returning_method_call(
                code, struct, method, arg_open, arg_close
            )

        member = next(
            (
                item
                for item in struct.data_members
                if item.name == match.group("member")
            ),
            None,
        )
        member_qualifiers = (
            set(IDENTIFIER_RE.findall(member.type_text))
            if member is not None
            else set()
        )
        indices = re.findall(r"\[\s*([^\[\]]+)\s*\]", match.group("indices"))
        if (
            member is None
            or "*" in member.type_text
            or "&" in member.type_text
            or "static" in member_qualifiers
            or (member.array_suffix and not indices)
            or (not member.array_suffix and indices)
            or len(indices) != member.array_suffix.count("[")
            or any(
                not self._direct_reference_expression_is_safe(index)
                for index in indices
            )
        ):
            self._reject_reference_returning_method_call(
                code, struct, method, arg_open, arg_close
            )

        raw_arguments = [
            argument.strip()
            for argument in self._split_top_level_commas(code[arg_open + 1 : arg_close])
            if argument.strip()
        ]
        if len(raw_arguments) != len(method.parameter_names):
            self._reject_reference_returning_method_call(
                code, struct, method, arg_open, arg_close
            )

        replacements = {"self": receiver}
        for name, argument in zip(method.parameter_names, raw_arguments):
            if self._bare_identifier_use_count(
                expression, name
            ) != 1 or not self._direct_reference_expression_is_safe(argument):
                self._reject_reference_returning_method_call(
                    code, struct, method, arg_open, arg_close
                )
            replacements[name] = f"({argument})"

        return arg_close + 1, self._replace_identifiers(expression, replacements)

    def _direct_reference_return_expression(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
    ) -> Optional[str]:
        body = self._rewrite_method_body(struct, method)
        match = re.fullmatch(
            r"\s*return\s+(?P<expression>.+?)\s*;\s*",
            body,
            re.DOTALL,
        )
        if match is None:
            return None
        return self._strip_enclosing_parens(match.group("expression").strip())

    @staticmethod
    def _direct_reference_expression_is_safe(expression: str) -> bool:
        text = expression.strip()
        if not text or re.search(r"\+\+|--|&&|\|\||[=?:,;{}\[\]]", text):
            return False
        if re.search(r"\b[A-Za-z_]\w*\s*\(", text):
            return False
        return re.fullmatch(r"[A-Za-z0-9_'\s.()+\-*/%&|^<>]+", text) is not None

    def _bare_identifier_use_count(self, text: str, name: str) -> int:
        return sum(
            1
            for match in re.finditer(rf"\b{re.escape(name)}\b", text)
            if not self._is_member_identifier_context(text, match.start())
        )

    def _reference_call_binds_or_escapes(
        self,
        code: str,
        call_start: int,
        *,
        enclosing_return_type: Optional[str],
    ) -> bool:
        boundary = max(code.rfind(token, 0, call_start) for token in (";", "{", "}"))
        prefix = code[boundary + 1 : call_start]
        if re.search(r"&\s*[A-Za-z_]\w*\s*=\s*$", prefix):
            return True
        if re.search(r"\breturn\s*$", prefix) is None:
            return False
        if enclosing_return_type is None:
            return True
        return bool(
            re.search(
                r"(?<!&)\&\s*$",
                self._normalize_template_argument_text(enclosing_return_type),
            )
        )

    def _build_instance_call_rewrite(
        self,
        code: str,
        receiver: str,
        method: _MetalStructMethod,
        arg_open: int,
        *,
        struct_name: Optional[object] = None,
    ) -> Optional[Tuple[int, str]]:
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            return None
        self._reject_reference_returning_method_call(
            code,
            struct_name,
            method,
            arg_open,
            arg_close,
        )
        args = code[arg_open + 1 : arg_close].strip()
        if args:
            replacement = f"{method.free_name}({receiver}, {args})"
        else:
            replacement = f"{method.free_name}({receiver})"
        return arg_close + 1, replacement

    def _reject_reference_returning_method_call(
        self,
        code: str,
        struct_name: Optional[object],
        method: _MetalStructMethod,
        arg_open: int,
        arg_close: int,
    ) -> None:
        struct = struct_name or method.free_name.split("__", 1)[0]
        owner = getattr(struct, "name", None) or str(struct)
        return_type = method.return_type
        if isinstance(struct, _MetalStructDefinition):
            return_type = self._canonicalize_struct_scoped_type(
                return_type,
                struct,
                None,
            )
        arguments = code[arg_open + 1 : arg_close].strip()
        signature = f"{owner}::{method.name}({arguments})"
        call_start = code.rfind(method.name, 0, arg_open)
        if call_start < 0:
            call_start = arg_open
        self._reject_selected_reference_return(
            struct,
            method,
            return_type,
            signature,
            source_location=self._source_location_for_offsets(
                code,
                call_start,
                arg_close + 1,
            ),
        )

    def _reject_selected_reference_return(
        self,
        struct: object,
        method: _MetalStructMethod,
        return_type: str,
        signature: str,
        *,
        source_location: Optional[object] = None,
    ) -> None:
        normalized_return_type = self._normalize_template_argument_text(return_type)
        if (
            re.search(r"&&?\s*$", normalized_return_type) is None
            and normalized_return_type != "decltype(auto)"
        ):
            return

        owner = getattr(struct, "name", None) or str(struct)
        raise MetalStructMethodError(
            "Cannot lower reference-returning struct method "
            f"'{owner}::{method.name}' without preserving the returned lvalue "
            f"identity. Requested call: {signature}.",
            struct_name=owner,
            method_name=method.name,
            requested_signature=signature,
            suggested_action=(
                "rewrite the accessor as a direct operation on the underlying "
                "storage, or avoid binding or assigning through its returned "
                "reference until reference identity lowering is available"
            ),
            source_location=source_location,
            missing_capabilities=("struct.reference-return",),
            reason="reference-return-identity-unsupported",
            return_type=normalized_return_type,
        )

    # ------------------------------------------------------------------ #
    # Pointer-member promotion: construction and call-site rewriting.     #
    # ------------------------------------------------------------------ #

    def _collect_struct_type_aliases(
        self,
        code: str,
        struct_names: Set[str],
        skip_spans: List[Tuple[int, int]],
        structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
    ) -> Dict[str, List[_MetalTypeAliasBinding]]:
        # Map aliases to concrete structs with their lexical lifetime. Bindings
        # are position-ordered because kernels commonly reuse the same alias in
        # separate functions or nested blocks. Both `using ALIAS = Struct;` and
        # `typedef Struct ALIAS;` are recognized, including chains through an
        # already visible alias.
        raw_aliases: List[Tuple[int, str, str]] = []
        ignored_spans = sorted(
            [*skip_spans, *self._find_comment_and_literal_spans(code)]
        )
        for match in re.finditer(
            r"\busing\s+([A-Za-z_]\w*)\s*=\s*"
            r"((?:typename\s+)?[A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)?)\s*;",
            code,
        ):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            alias, target = match.group(1), match.group(2)
            raw_aliases.append(
                (match.start(), alias, self._normalize_template_argument_text(target))
            )
        for match in re.finditer(
            r"\btypedef\s+([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*;", code
        ):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            target, alias = match.group(1), match.group(2)
            raw_aliases.append((match.start(), alias, target))

        aliases: Dict[str, List[_MetalTypeAliasBinding]] = {}
        lexical_scopes = self._find_lexical_brace_scopes(code)
        for declaration_position, alias, target in sorted(raw_aliases):
            resolved_target = self._resolve_struct_type_alias_at(
                aliases, target, declaration_position
            )
            if resolved_target is None and target in struct_names:
                resolved_target = target
            if resolved_target is None and structs_by_name:
                qualified = re.fullmatch(
                    r"(?:typename\s+)?(?P<owner>[A-Za-z_]\w*)\s*::\s*"
                    r"(?P<member>[A-Za-z_]\w*)",
                    target,
                )
                if qualified is not None:
                    owner = qualified.group("owner")
                    resolved_owner = self._resolve_struct_type_alias_at(
                        aliases, owner, declaration_position
                    )
                    if resolved_owner is None and owner in struct_names:
                        resolved_owner = owner
                    owner_struct = structs_by_name.get(resolved_owner or "")
                    if owner_struct is not None:
                        canonical = self._canonicalize_struct_scoped_type(
                            qualified.group("member"),
                            owner_struct,
                            structs_by_name,
                        )
                        if IDENTIFIER_RE.fullmatch(canonical) and (
                            canonical in struct_names
                            or canonical in self._materialized_struct_specializations
                        ):
                            resolved_target = canonical
            if resolved_target is None:
                continue
            scope_start, scope_end = self._innermost_lexical_scope(
                lexical_scopes, declaration_position, len(code)
            )
            aliases.setdefault(alias, []).append(
                _MetalTypeAliasBinding(
                    declaration_position=declaration_position,
                    scope_start=scope_start,
                    scope_end=scope_end,
                    target=resolved_target,
                )
            )
        return aliases

    def _find_comment_and_literal_spans(self, code: str) -> List[Tuple[int, int]]:
        analysis = self._source_analysis(code)
        if analysis.comment_and_literal_spans is None:
            analysis.comment_and_literal_spans = self._scan_comment_and_literal_spans(
                code
            )
        return analysis.comment_and_literal_spans

    def _scan_comment_and_literal_spans(self, code: str) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        i = 0
        while i < len(code):
            if code[i] in "\"'":
                _literal, consumed = self._read_string(code, i)
                spans.append((i, i + consumed))
                i += consumed
                continue
            if code.startswith("//", i):
                newline = code.find("\n", i + 2)
                end = len(code) if newline == -1 else newline
                spans.append((i, end))
                i = end
                continue
            if code.startswith("/*", i):
                comment_end = code.find("*/", i + 2)
                end = len(code) if comment_end == -1 else comment_end + 2
                spans.append((i, end))
                i = end
                continue
            i += 1
        return spans

    def _find_lexical_brace_scopes(self, code: str) -> List[Tuple[int, int]]:
        analysis = self._source_analysis(code)
        if analysis.lexical_brace_scopes is None:
            analysis.lexical_brace_scopes = self._scan_lexical_brace_scopes(code)
        return analysis.lexical_brace_scopes

    def _scan_lexical_brace_scopes(self, code: str) -> List[Tuple[int, int]]:
        scopes: List[Tuple[int, int]] = [(0, len(code))]
        brace_stack: List[int] = []
        i = 0
        while i < len(code):
            if code[i] in "\"'":
                _literal, consumed = self._read_string(code, i)
                i += consumed
                continue
            if code.startswith("//", i):
                newline = code.find("\n", i + 2)
                if newline == -1:
                    break
                i = newline + 1
                continue
            if code.startswith("/*", i):
                comment_end = code.find("*/", i + 2)
                if comment_end == -1:
                    break
                i = comment_end + 2
                continue
            if code[i] == "{":
                brace_stack.append(i)
            elif code[i] == "}" and brace_stack:
                scopes.append((brace_stack.pop() + 1, i))
            i += 1
        return scopes

    @staticmethod
    def _innermost_lexical_scope(
        scopes: List[Tuple[int, int]], position: int, source_length: int
    ) -> Tuple[int, int]:
        containing = [scope for scope in scopes if scope[0] <= position < scope[1]]
        if not containing:
            return 0, source_length
        return max(containing, key=lambda scope: scope[0])

    @staticmethod
    def _resolve_type_alias_binding_at(
        type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
        name: str,
        position: int,
    ) -> Optional[_MetalTypeAliasBinding]:
        best: Optional[_MetalTypeAliasBinding] = None
        for binding in type_aliases.get(name, []):
            if binding.declaration_position > position:
                break
            if not (binding.scope_start <= position < binding.scope_end):
                continue
            if best is None or binding.declaration_position > best.declaration_position:
                best = binding
        return best

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

    def _resolve_promotable_type_token(
        self,
        token: str,
        position: int,
        struct_names: Set[str],
        type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
    ) -> Optional[str]:
        # Resolve an identifier used as a type to the promotable struct it names:
        # directly when it is a struct name, otherwise through the nearest
        # preceding alias at or before `position`. Returns None when the token is
        # neither.
        resolved = self._resolve_struct_type_alias_at(type_aliases, token, position)
        if resolved is None and token in struct_names:
            resolved = token
        return resolved if resolved in struct_names else None

    def _scan_pointer_struct_constructions(
        self,
        code: str,
        promotion_plans: Dict[str, _PointerPromotionPlan],
        skip_spans: List[Tuple[int, int]],
    ) -> Tuple[
        Dict[str, List[Tuple[int, str, List[str]]]],
        List[Tuple[int, int, str]],
        Set[str],
    ]:
        # Find every construction of a promotion-planned struct and, for each,
        # record the receiver's pointer expressions and a replacement that drops
        # the pointer arguments from the constructor call. A struct is promoted
        # only when it is constructed at least once AND every one of its
        # constructions is a clean `S v = S(args)` / `S v(args)` we can rewrite;
        # any other construction (default `S v;`, unknown initializer, argument
        # count mismatch) drops the struct from promotion so it falls back to the
        # ordinary lowering unchanged.
        #
        # Returns (pointer_args, construction_replacements, promoted_names) where
        # pointer_args maps a receiver name to a position-ordered list of
        # (construction_offset, struct_name, [pointer_expr, ...]).
        struct_names = set(promotion_plans)
        # A promotable struct is often constructed through a local `using` alias
        # (`using read_writer_t = ReadWriter_...;` then
        # `read_writer_t rw = read_writer_t(...)`), so resolve such aliases too.
        type_aliases = self._collect_struct_type_aliases(code, struct_names, skip_spans)
        pointer_args: Dict[str, List[Tuple[int, str, List[str]]]] = {}
        # Each pending replacement is tagged with its struct so replacements for a
        # struct that ends up NOT promoted can be discarded (its constructor keeps
        # its pointer arguments).
        pending_replacements: List[Tuple[str, Tuple[int, int, str]]] = []
        constructed: Set[str] = set()
        failed: Set[str] = set()
        i = 0
        n = len(code)
        while i < n:
            ch = code[i]
            if ch in "\"'":
                _literal, consumed = self._read_string(code, i)
                i += consumed
                continue
            if code.startswith("//", i):
                end = code.find("\n", i)
                i = n if end == -1 else end + 1
                continue
            if code.startswith("/*", i):
                end = code.find("*/", i + 2)
                i = n if end == -1 else end + 2
                continue
            span = self._containing_span(i, skip_spans)
            if span is not None:
                i = span[1]
                continue
            if ch.isalpha() or ch == "_":
                ident, consumed = self._read_identifier(code, i)
                ident_end = i + consumed
                target = self._resolve_promotable_type_token(
                    ident, i, struct_names, type_aliases
                )
                if target is not None and not self._is_member_identifier_context(
                    code, i
                ):
                    outcome = self._match_pointer_struct_construction(
                        code, ident, ident_end, promotion_plans[target]
                    )
                    if outcome is not None:
                        kind = outcome[0]
                        if kind == "construction":
                            _, next_index, receiver, ptr_exprs, replacement = outcome
                            constructed.add(target)
                            pointer_args.setdefault(receiver, []).append(
                                (i, target, ptr_exprs)
                            )
                            if replacement is not None:
                                pending_replacements.append((target, replacement))
                            i = next_index
                            continue
                        if kind == "fail":
                            failed.add(target)
                            i = outcome[1]
                            continue
                i = ident_end
                continue
            i += 1

        promoted_names = constructed - failed
        # Keep only what belongs to a struct we will actually promote.
        pointer_args = {
            receiver: [entry for entry in entries if entry[1] in promoted_names]
            for receiver, entries in pointer_args.items()
        }
        pointer_args = {
            receiver: entries for receiver, entries in pointer_args.items() if entries
        }
        construction_replacements = [
            replacement
            for struct_name, replacement in pending_replacements
            if struct_name in promoted_names
        ]
        for entries in pointer_args.values():
            entries.sort(key=lambda item: item[0])
        return pointer_args, construction_replacements, promoted_names

    def _match_pointer_struct_construction(
        self,
        code: str,
        struct_name: str,
        after_name: int,
        plan: _PointerPromotionPlan,
    ):
        # Classify a `StructName ...` occurrence that begins at a declaration
        # boundary. Returns one of:
        #   ("construction", next_index, receiver, [ptr_expr,...], replacement)
        #   ("fail", next_index)      -- a construction we cannot rewrite
        #   None                      -- not a construction (plain type use)
        # where `replacement` is a (start, end, text) tuple rewriting the
        # constructor call to drop its pointer arguments.
        n = len(code)
        j = after_name
        while j < n and code[j].isspace():
            j += 1
        if j >= n:
            return None
        # A declaration names a receiver identifier after the type.
        if not (code[j].isalpha() or code[j] == "_"):
            return None
        receiver, consumed = self._read_identifier(code, j)
        if not receiver:
            return None
        after_receiver = j + consumed
        k = after_receiver
        while k < n and code[k].isspace():
            k += 1
        if k >= n:
            return None
        if code[k] == "(":
            # Direct initialization: `StructName receiver(args)`.
            return self._build_pointer_construction(
                code, struct_name, receiver, k, plan
            )
        if code[k] == "=":
            # Copy initialization: `StructName receiver = StructName(args)`.
            r = k + 1
            while r < n and code[r].isspace():
                r += 1
            rhs_ident, rhs_consumed = self._read_identifier(code, r)
            if rhs_ident != struct_name:
                # Initialized from something other than the constructor -> we
                # cannot source the pointer expressions.
                return ("fail", after_receiver)
            after_rhs = r + rhs_consumed
            while after_rhs < n and code[after_rhs].isspace():
                after_rhs += 1
            if after_rhs >= n or code[after_rhs] != "(":
                return ("fail", after_receiver)
            return self._build_pointer_construction(
                code, struct_name, receiver, after_rhs, plan
            )
        if code[k] == ";":
            # Default construction cannot supply the pointer members.
            return ("fail", k + 1)
        return None

    def _build_pointer_construction(
        self,
        code: str,
        struct_name: str,
        receiver: str,
        arg_open: int,
        plan: _PointerPromotionPlan,
    ):
        # Parse the constructor-call argument list at `arg_open` (the `(`), map
        # each pointer member to its argument expression, and build a replacement
        # that keeps only the non-pointer arguments.
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            return ("fail", arg_open + 1)
        raw_args = code[arg_open + 1 : arg_close]
        arguments = [
            argument.strip()
            for argument in self._split_top_level_commas(raw_args)
            if argument.strip()
        ]
        constructor = plan.constructor
        if len(arguments) != len(constructor.param_names):
            # A default argument or variadic call breaks the positional mapping.
            return ("fail", arg_close + 1)
        pointer_exprs: List[str] = []
        for index in plan.pointer_ctor_arg_indices:
            if index >= len(arguments):
                return ("fail", arg_close + 1)
            pointer_exprs.append(arguments[index])
        drop = set(plan.pointer_ctor_arg_indices)
        kept = [
            argument for index, argument in enumerate(arguments) if index not in drop
        ]
        # Only the parenthesized argument group is rewritten (dropping the pointer
        # arguments); this is correct for BOTH `S v(args)` (direct init) and
        # `S v = S(args)` (copy init), where the struct name / receiver preceding
        # the arguments is left in place.
        replacement_text = f"({', '.join(kept)})"
        replacement = (arg_open, arg_close + 1, replacement_text)
        return ("construction", arg_close + 1, receiver, pointer_exprs, replacement)

    def _rewrite_promoted_call_sites(
        self,
        code: str,
        promoted_structs: List[_MetalStructDefinition],
        promotion_plans: Dict[str, _PointerPromotionPlan],
        pointer_args: Dict[str, List[Tuple[int, str, List[str]]]],
        skip_spans: List[Tuple[int, int]],
    ) -> List[Tuple[int, int, str]]:
        # Rewrite `receiver.method(args)` for a promoted-struct receiver to
        # `S__method(receiver, <pointer exprs>, args)`, forwarding the pointer
        # expressions captured at the receiver's construction.
        methods_by_struct: Dict[str, Dict[str, _MetalStructMethod]] = {}
        structs_by_name = {struct.name: struct for struct in promoted_structs}
        for struct in promoted_structs:
            methods_by_struct[struct.name] = {
                method.name: method for method in struct.methods
            }
        replacements: List[Tuple[int, int, str]] = []
        i = 0
        n = len(code)
        while i < n:
            ch = code[i]
            if ch in "\"'":
                _literal, consumed = self._read_string(code, i)
                i += consumed
                continue
            if code.startswith("//", i):
                end = code.find("\n", i)
                i = n if end == -1 else end + 1
                continue
            if code.startswith("/*", i):
                end = code.find("*/", i + 2)
                i = n if end == -1 else end + 2
                continue
            span = self._containing_span(i, skip_spans)
            if span is not None:
                i = span[1]
                continue
            if ch.isalpha() or ch == "_":
                ident, consumed = self._read_identifier(code, i)
                ident_end = i + consumed
                if not self._is_member_identifier_context(code, i):
                    resolved = self._resolve_promoted_receiver(pointer_args, ident, i)
                    if resolved is not None:
                        struct_name, pointer_exprs = resolved
                        rewrite = self._try_rewrite_promoted_instance_call(
                            code,
                            ident,
                            ident_end,
                            structs_by_name[struct_name],
                            pointer_exprs,
                            methods_by_struct.get(struct_name, {}),
                        )
                        if rewrite is not None:
                            end, replacement = rewrite
                            replacements.append((i, end, replacement))
                            i = end
                            continue
                i = ident_end
                continue
            i += 1
        return replacements

    def _try_rewrite_promoted_instance_call(
        self,
        code: str,
        receiver: str,
        ident_end: int,
        struct: _MetalStructDefinition,
        pointer_exprs: List[str],
        methods_by_name: Dict[str, _MetalStructMethod],
    ) -> Optional[Tuple[int, str]]:
        n = len(code)
        j = ident_end
        while j < n and code[j].isspace():
            j += 1
        if j >= n or code[j] != ".":
            return None
        k = j + 1
        while k < n and code[k].isspace():
            k += 1
        member, consumed = self._read_identifier(code, k)
        if not member:
            return None
        after = k + consumed
        while after < n and code[after].isspace():
            after += 1
        if after >= n or code[after] != "(":
            return None
        method = methods_by_name.get(member)
        if method is None or method.is_static:
            return None
        arg_open = after
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            return None
        self._reject_reference_returning_method_call(
            code,
            struct,
            method,
            arg_open,
            arg_close,
        )
        args = code[arg_open + 1 : arg_close].strip()
        pieces = [receiver]
        pieces.extend(pointer_exprs)
        if args:
            pieces.append(args)
        replacement = f"{method.free_name}({', '.join(pieces)})"
        return arg_close + 1, replacement

    def _resolve_promoted_receiver(
        self,
        pointer_args: Dict[str, List[Tuple[int, str, List[str]]]],
        name: str,
        position: int,
    ) -> Optional[Tuple[str, List[str]]]:
        # Resolve a receiver name to the (struct_name, pointer_exprs) of its
        # NEAREST construction at or before `position` (falling back to the first
        # construction for a forward reference), mirroring
        # `_resolve_declared_type_at`.
        entries = pointer_args.get(name)
        if not entries:
            return None
        best: Optional[Tuple[str, List[str]]] = None
        for construction_position, struct_name, pointer_exprs in entries:
            if construction_position <= position:
                best = (struct_name, pointer_exprs)
            else:
                break
        if best is not None:
            return best
        return entries[0][1], entries[0][2]

    def _collect_struct_variable_types(
        self,
        code: str,
        struct_names: Set[str],
        struct_spans: List[Tuple[int, int]],
        *,
        include_indirect: bool = False,
    ) -> Dict[str, List[Tuple[int, str]]]:
        # Map each variable name to the POSITION-ORDERED list of struct
        # declarations that introduce it outside struct definitions. Direct
        # declarations are always recorded; pointer and reference declarations
        # are included only for field-carrier inference. Returning every
        # declaration with its source offset makes call-site resolution
        # deterministic and selects the nearest preceding declaration.
        declarations: Dict[str, List[Tuple[int, str]]] = {}
        # Iterate struct names in a STABLE (sorted) order so the per-name lists,
        # before sorting, do not depend on set iteration order; the final sort by
        # position is what callers rely on, but a stable scan keeps ties stable.
        for struct_name in sorted(struct_names):
            pattern = re.compile(
                rf"\b{re.escape(struct_name)}\b"
                r"(?:\s+(?:const|volatile))*\s*"
                r"(?P<indirection>[*&]+)?\s*"
                r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*"
                r"(?=[;,)=\[{(])"
            )
            for match in pattern.finditer(code):
                if self._containing_span(match.start(), struct_spans) is not None:
                    continue
                # Skip a match that is itself a member access (`x.S var` cannot
                # happen, but guard scoped `Ns::S` by checking the preceding char).
                preceding = match.start() - 1
                while preceding >= 0 and code[preceding].isspace():
                    preceding -= 1
                if preceding >= 0 and code[preceding] in ".>":
                    continue
                indirection = match.group("indirection") or ""
                if indirection and not include_indirect:
                    continue
                declarations.setdefault(match.group("name"), []).append(
                    (match.start(), f"{struct_name}{indirection}")
                )
        for entries in declarations.values():
            entries.sort(key=lambda item: item[0])
        return declarations

    def _collect_aliased_struct_variable_types(
        self,
        code: str,
        type_aliases: Dict[str, List[_MetalTypeAliasBinding]],
        struct_spans: List[Tuple[int, int]],
    ) -> Dict[str, List[Tuple[int, str]]]:
        declarations: Dict[str, List[Tuple[int, str]]] = {}
        for alias in sorted(type_aliases):
            pattern = re.compile(
                rf"\b{re.escape(alias)}\b"
                r"(?:\s+(?:const|volatile))*\s*"
                r"(?P<indirection>[*&]+)?\s*"
                r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*"
                r"(?=[;,)=\[{(])"
            )
            for match in pattern.finditer(code):
                if self._containing_span(match.start(), struct_spans) is not None:
                    continue
                resolved = self._resolve_struct_type_alias_at(
                    type_aliases, alias, match.start()
                )
                if resolved is None:
                    continue
                indirection = match.group("indirection") or ""
                declarations.setdefault(match.group("name"), []).append(
                    (match.start(), f"{resolved}{indirection}")
                )
        for entries in declarations.values():
            entries.sort(key=lambda item: item[0])
        return declarations

    def _resolve_declared_type_at(
        self,
        declarations: Dict[str, List[Tuple[int, _DeclaredTypeT]]],
        name: str,
        position: int,
    ) -> Optional[_DeclaredTypeT]:
        # Resolve a name to the type of its NEAREST declaration appearing at or
        # before `position`. A later declaration cannot type an earlier use;
        # accepting it can leak a same-named parameter from another function
        # into template deduction. Deterministic regardless of hash seed.
        entries = declarations.get(name)
        if not entries:
            return None
        best: Optional[_DeclaredTypeT] = None
        for decl_position, declared_type in entries:
            if decl_position <= position:
                best = declared_type
            else:
                break
        if best is not None:
            return best
        return None

    def _flatten_types_at(
        self,
        declarations: Dict[str, List[Tuple[int, _DeclaredTypeT]]],
        position: int,
    ) -> Dict[str, _DeclaredTypeT]:
        # Flatten a position-ordered declaration map to a name -> type view valid
        # at `position` (nearest preceding declaration wins per name).
        flattened: Dict[str, _DeclaredTypeT] = {}
        for name in declarations:
            resolved = self._resolve_declared_type_at(declarations, name, position)
            if resolved is not None:
                flattened[name] = resolved
        return flattened

    def _struct_field_types_at(
        self,
        variable_types: Optional[Dict[str, List[Tuple[int, str]]]],
        structs_by_name: Optional[Dict[str, _MetalStructDefinition]],
        position: int,
    ) -> Dict[str, Dict[str, str]]:
        # Build a call-site view of each struct carrier's field types. Pointer
        # carriers use an `obj->` key; objects and references use `obj`, allowing
        # member inference to require the matching access operator. Multi-level
        # indirection is intentionally excluded.
        if not variable_types or not structs_by_name:
            return {}
        field_types: Dict[str, Dict[str, str]] = {}
        for name in variable_types:
            struct_type = self._resolve_declared_type_at(variable_types, name, position)
            if struct_type is None:
                continue
            indirection_match = re.search(r"(?P<indirection>[*&]+)\s*$", struct_type)
            indirection = (
                indirection_match.group("indirection")
                if indirection_match is not None
                else ""
            )
            if indirection not in {"", "*", "&"}:
                continue
            base_type = (
                struct_type[: indirection_match.start()].strip()
                if indirection_match is not None
                else struct_type
            )
            struct = structs_by_name.get(self._normalize_inferred_type(base_type))
            if struct is None or not struct.data_member_types:
                continue
            access_key = f"{name}->" if indirection == "*" else name
            field_types[access_key] = struct.data_member_types
        return field_types

    # ------------------------------------------------------------------ #
    # Conservative call-argument type inference (template member methods). #
    # ------------------------------------------------------------------ #
    # The rules below NEVER guess: each returns None for any expression shape it
    # does not recognize, and the caller turns a single un-inferable argument
    # into a clean translation failure rather than emitting a dangling call.

    def _collect_buffer_element_types(
        self, code: str, struct_spans: List[Tuple[int, int]]
    ) -> _MetalPositionedBufferTypes:
        # Map each subscriptable name to the POSITION-ORDERED list of declarations
        # that give it an ELEMENT type, so `name[expr]` can be element-typed. Three
        # declaration shapes contribute:
        #   * pointer/buffer parameters and locals: `device [const] T* buf`,
        #     `constant T* p`, `thread U* q` -> element T (function parameters AND
        #     local pointer variables).
        #   * array locals/parameters: `U totals[4]`, `threadgroup U sh[32]`,
        #     `const float vals[N]` -> element U/float (the dominant shape in MLX
        #     reduce, where reduction accumulators are stack arrays).
        # Struct bodies are excluded so member declarations do not pollute the map
        # (struct fields are typed separately via `data_member_types`).
        element_types: _MetalPositionedBufferTypes = {}

        def record(
            name: str,
            element_type: str,
            position: int,
            pointer_type: Optional[str] = None,
        ) -> None:
            normalized = self._normalize_inferred_type(element_type)
            if not normalized:
                return
            normalized_pointer = self._normalize_known_address_space_pointer_type(
                pointer_type or ""
            )
            element_types.setdefault(name, []).append(
                (
                    position,
                    _MetalSubscriptableType(normalized, normalized_pointer),
                )
            )

        pointer_pattern = re.compile(
            r"\b(?P<type>(?:(?:const|constexpr|volatile)\s+)*"
            r"(?:device|constant|threadgroup|thread)\b"
            r"[^;,()\[\]{}]*?)\*\s*"
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
        )
        for match in pointer_pattern.finditer(code):
            if self._containing_span(match.start(), struct_spans) is not None:
                continue
            record(
                match.group("name"),
                match.group("type"),
                match.start(),
                f"{match.group('type')}*",
            )

        # Array declarations: `<type> name[extent]` for locals and parameters.
        # The leading anchor keeps consecutive declarations from cannibalizing
        # each other; the type group allows address-space/cv qualifiers. Only a
        # RECOGNIZED element type (scalar/vector or a known struct/union) is
        # accepted so a non-declaration that happens to match the shape — e.g.
        # `return totals[i]` (type token `return`) or `else block[i]` — never
        # records a bogus element type (the never-guess contract).
        recognized_aggregates = self._aggregate_type_names(code, struct_spans)
        function_body_spans = [
            function.body_span
            for function in self._find_non_template_function_definitions(
                code, struct_spans
            )
        ]
        array_pattern = re.compile(
            r"(?:(?<=[;{}(,])|^)\s*"
            r"(?P<type>(?:const\s+|constexpr\s+|volatile\s+|thread\s+|"
            r"threadgroup\s+|device\s+|constant\s+)*"
            r"[A-Za-z_][A-Za-z0-9_]*)\s+"
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\[(?!\[)",
            re.MULTILINE,
        )
        for match in array_pattern.finditer(code):
            if self._containing_span(match.start(), struct_spans) is not None:
                continue
            normalized = self._normalize_inferred_type(match.group("type"))
            if (
                normalized not in self._METAL_SCALAR_VECTOR_TYPES
                and normalized not in recognized_aggregates
            ):
                continue
            pointer_type = f"{match.group('type')}*"
            if (
                self._normalize_known_address_space_pointer_type(pointer_type) is None
                and self._containing_span(match.start(), function_body_spans)
                is not None
            ):
                pointer_type = f"thread {match.group('type')}*"
            record(
                match.group("name"),
                normalized,
                match.start(),
                pointer_type,
            )

        for entries in element_types.values():
            entries.sort(key=lambda item: item[0])
        return element_types

    def _collect_local_variable_types(
        self, code: str, struct_spans: List[Tuple[int, int]]
    ) -> Dict[str, List[Tuple[int, str]]]:
        # Map each bare local/parameter name to the POSITION-ORDERED list of its
        # declared types. Two declaration shapes contribute:
        #   * scalar/vector locals: `T name;` / `T name = ...;` where T is a
        #     recognized Metal scalar/vector type.
        #   * union/struct locals: `bool4_or_uint update;` and any `Name var;`
        #     whose type names a struct/union declared in this source — recorded
        #     so a bare such local is inferable (and its members are resolvable).
        # Position ordering makes resolution deterministic and per-scope (nearest
        # preceding declaration wins), so same-named locals in sibling kernels do
        # not collapse to one PYTHONHASHSEED-dependent type.
        local_types: Dict[str, List[Tuple[int, str]]] = {}
        recognized_aggregates = self._aggregate_type_names(code, struct_spans)
        # A zero-width leading anchor (start-of-string or a statement/scope
        # boundary) keeps consecutive declarations like `float acc=...; float
        # x=...;` from cannibalizing each other's anchor.
        pattern = re.compile(
            r"(?:(?<=[;{}()])|^)\s*"
            r"(?P<type>(?:const\s+|constexpr\s+|thread\s+|threadgroup\s+)*"
            r"(?:::)?[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*"
            r"(?:\s*<[^;\n={}()]+>)?)\s+"
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?=[=;])",
            re.MULTILINE,
        )
        for match in pattern.finditer(code):
            if self._containing_span(match.start(), struct_spans) is not None:
                continue
            raw_type = match.group("type")
            normalized = self._normalize_inferred_type(raw_type)
            if (
                not self._is_metal_scalar_or_vector_type(normalized)
                and normalized not in recognized_aggregates
            ):
                continue
            local_types.setdefault(match.group("name"), []).append(
                (match.start(), normalized)
            )
        for entries in local_types.values():
            entries.sort(key=lambda item: item[0])
        return local_types

    def _aggregate_type_names(
        self, code: str, struct_spans: List[Tuple[int, int]]
    ) -> Set[str]:
        # Names of `struct`/`class`/`union` types DEFINED in this source. A bare
        # local of such a type is inferable, and (for structs) its members are
        # resolvable via `data_member_types`. Unions are included so reduce's
        # `bool4_or_uint update;` typed local is recognized.
        names: Set[str] = set()
        for match in re.finditer(
            r"\b(?:struct|class|union)\s+([A-Za-z_][A-Za-z0-9_]*)", code
        ):
            names.add(match.group(1))
        return names

    def _collect_function_parameter_types(
        self,
        code: str,
        struct_spans: List[Tuple[int, int]],
        buffer_element_types: _MetalPositionedBufferTypes,
        local_variable_types: Dict[str, List[Tuple[int, str]]],
    ) -> None:
        # Augment the (position-ordered) type maps with each FUNCTION PARAMETER,
        # so a call argument that is a parameter (or a subscript of one) resolves
        # via the parameter's declared type. Pointer/array parameters are already
        # captured by `_collect_buffer_element_types` (they match its patterns in
        # the header text); this adds SCALAR/VECTOR and struct/union parameters,
        # which the local-declaration scanner misses because a parameter is
        # comma-terminated rather than `=`/`;`-terminated. Each parameter is
        # recorded at the function BODY-START offset so nearest-preceding
        # resolution scopes it to that body. Struct method bodies are excluded
        # (their parameters are handled by the member-method lowering itself).
        recognized_aggregates = self._aggregate_type_names(code, struct_spans)
        for function in self._find_non_template_function_definitions(
            code, struct_spans
        ):
            header = code[function.span[0] : function.body_span[0] - 1]
            parameter_span = self._function_parameter_list_span(header)
            if parameter_span is None:
                continue
            paren_start, paren_end = parameter_span
            parameter_text = header[paren_start + 1 : paren_end]
            body_start = function.body_span[0]
            for parameter in self._split_top_level_commas(parameter_text):
                if not parameter.strip():
                    continue
                parameter_without_attributes = self._strip_metal_attributes(parameter)
                name = self._declared_data_member_name(parameter_without_attributes)
                if not name:
                    continue
                subscriptable = self._pointer_or_array_parameter_type(parameter)
                if subscriptable is not None:
                    buffer_element_types.setdefault(name, []).append(
                        (body_start, subscriptable)
                    )
                    continue
                scalar = self._function_parameter_value_type(
                    parameter_without_attributes
                )
                if (
                    self._is_metal_scalar_or_vector_type(scalar)
                    or scalar in recognized_aggregates
                ):
                    local_variable_types.setdefault(name, []).append(
                        (body_start, scalar)
                    )
        for entries in buffer_element_types.values():
            entries.sort(key=lambda item: item[0])
        for entries in local_variable_types.values():
            entries.sort(key=lambda item: item[0])

    def _collect_auto_local_variable_types(
        self,
        code: str,
        struct_spans: List[Tuple[int, int]],
        buffer_element_types: _MetalPositionedBufferTypes,
        local_variable_types: Dict[str, List[Tuple[int, str]]],
        structs_by_name: Optional[Dict[str, "_MetalStructDefinition"]],
    ) -> None:
        # Type `auto name = <initializer>;` locals from the inferred type of their
        # initializer, so a later call argument that uses such a local resolves.
        # MLX's complex unary ops build their template-member-call arguments from
        # `auto` temporaries (e.g. `auto i = complex64_t{0.0, 1.0};`), which the
        # scalar/aggregate scanner cannot type because the declarator names no
        # type. Declarations are processed in source order and each inferred
        # binding is recorded immediately, so one `auto` local may depend on an
        # earlier one. Conservative: only records a binding when the initializer
        # infers to a concrete type; `auto&` / `auto*` declarators are skipped.
        pattern = re.compile(
            r"(?:(?<=[;{}()])|^)\s*auto\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)",
            re.MULTILINE,
        )
        for match in pattern.finditer(code):
            if self._containing_span(match.start(), struct_spans) is not None:
                continue
            initializer_start = match.end()
            statement_end = self._statement_end(code, initializer_start)
            if statement_end is None:
                continue
            initializer = code[initializer_start : statement_end - 1].strip()
            if not initializer:
                continue
            position = match.start()
            buffer_view = self._flatten_types_at(buffer_element_types, position)
            local_view = self._flatten_types_at(local_variable_types, position)
            struct_field_types = self._struct_field_types_at(
                local_variable_types, structs_by_name, position
            )
            inferred = self._infer_argument_type(
                initializer,
                buffer_view,
                local_view,
                struct_field_types,
                structs_by_name,
            )
            if inferred is None:
                continue
            normalized = self._normalize_inferred_expression_type(inferred)
            if normalized is None:
                continue
            entries = local_variable_types.setdefault(match.group("name"), [])
            entries.append((position, normalized))
            entries.sort(key=lambda item: item[0])

    def _pointer_or_array_parameter_element_type(self, parameter: str) -> Optional[str]:
        # Element type of a pointer (`device T* p`) or array (`T p[N]`) parameter,
        # or None when the parameter is neither (so the caller treats it as a
        # plain value parameter). Conservative: returns None unless a `*` or `[`
        # declarator is present.
        subscriptable = self._pointer_or_array_parameter_type(parameter)
        return subscriptable.element_type if subscriptable is not None else None

    def _pointer_or_array_parameter_type(
        self, parameter: str
    ) -> Optional[_MetalSubscriptableType]:
        # Preserve the full pointer spelling only when one Metal address space is
        # explicit. Value inference still consumes the normalized element type;
        # address-of inference separately requires the proven pointer spelling.
        text = self._strip_top_level_default_value(parameter)
        text = self._strip_metal_attributes(text).strip()
        if not text:
            return None
        if "[" in text:
            type_text = text[: text.find("[")]
            type_text = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\s*$", "", type_text).strip()
            element_type = self._normalize_inferred_type(type_text)
            if not element_type:
                return None
            return _MetalSubscriptableType(
                element_type,
                self._normalize_known_address_space_pointer_type(f"{type_text}*"),
            )
        star = text.rfind("*")
        if star != -1:
            type_text = text[:star]
            element_type = self._normalize_inferred_type(type_text)
            if not element_type:
                return None
            return _MetalSubscriptableType(
                element_type,
                self._normalize_known_address_space_pointer_type(text[: star + 1]),
            )
        return None

    # Recognized Metal scalar / vector / matrix element types whose declarations
    # are reliable enough to type a bare local variable.
    _METAL_SCALAR_VECTOR_TYPES: Set[str] = {
        base + suffix
        for base in (
            "float",
            "half",
            "double",
            "int",
            "uint",
            "short",
            "ushort",
            "char",
            "uchar",
            "long",
            "ulong",
            "bool",
        )
        for suffix in ("", "2", "3", "4")
    } | {
        # C stdint / sized scalar aliases used pervasively across MLX kernels
        # (e.g. `int32_t values[N]`, `CumSum<int32_t>`). They have no Metal vector
        # spelling (vectors use `int2`/`uint4`/... instead) and already carry
        # known sizes in `_METAL_SCALAR_TYPE_SIZES`, so recognizing the scalar
        # forms lets array/subscript/local/cast inference type them — which in
        # turn lets a SFINAE member method like `simd_scan<T>` bind and resolve
        # its `sizeof(T)` overload from the concrete element type.
        "int8_t",
        "uint8_t",
        "int16_t",
        "uint16_t",
        "int32_t",
        "uint32_t",
        "int64_t",
        "uint64_t",
        "float16_t",
        "bfloat16_t",
        "size_t",
    }

    def _normalize_inferred_type(self, type_text: str) -> str:
        # Collapse whitespace and strip leading cv/address-space qualifiers that
        # do not change the value type used for template binding.
        text = self._normalize_template_argument_text(type_text or "")
        if not text:
            return ""
        tokens = text.split(" ")
        dropped = {
            "const",
            "constexpr",
            "device",
            "constant",
            "thread",
            "threadgroup",
            "volatile",
        }
        kept = [token for token in tokens if token not in dropped]
        return " ".join(kept).strip()

    def _normalize_known_address_space_pointer_type(
        self, type_text: str
    ) -> Optional[str]:
        # Address-of inference may expose a pointer only when its complete Metal
        # address space is known. Multi-level pointers and conflicting address
        # spaces stay unsupported because reconstructing either would be a guess.
        normalized = self._normalize_template_argument_text(type_text)
        if (
            not normalized.endswith("*")
            or normalized.count("*") != 1
            or "&" in normalized
        ):
            return None
        address_spaces = [
            token
            for token in IDENTIFIER_RE.findall(normalized[:-1])
            if token in {"device", "constant", "thread", "threadgroup"}
        ]
        if len(address_spaces) != 1:
            return None
        pointee_type = self._normalize_inferred_type(normalized[:-1])
        if not pointee_type or pointee_type.endswith("*"):
            return None
        return normalized

    def _normalize_inferred_expression_type(self, type_text: str) -> Optional[str]:
        # Value expressions discard top-level cv/address-space qualifiers during
        # template binding. Pointer expressions cannot: Metal requires a proven
        # address space, and dropping it can produce a validating but unusable
        # specialization. Preserve one fully qualified pointer; reject every
        # other pointer-bearing spelling rather than normalizing it to `T*`.
        normalized = self._normalize_template_argument_text(type_text or "")
        if not normalized:
            return None
        if "*" in normalized:
            return self._normalize_known_address_space_pointer_type(normalized)
        value_type = self._normalize_inferred_type(normalized)
        return value_type or None

    def _record_known_member_function_return_type(
        self, function_name: str, return_type: str
    ) -> None:
        normalized = self._normalize_inferred_expression_type(return_type)
        if normalized is None:
            self._known_member_function_return_types.pop(function_name, None)
            return
        self._known_member_function_return_types[function_name] = normalized

    def _normalize_template_binding_type(self, type_text: str) -> str:
        return self._normalize_inferred_expression_type(type_text) or ""

    def _pointer_pointee_value_type(self, pointer_type: str) -> Optional[str]:
        normalized = self._normalize_known_address_space_pointer_type(pointer_type)
        if normalized is None:
            return None
        pointee = self._normalize_inferred_type(normalized[:-1])
        return pointee or None

    def _canonical_template_binding_pointee_type(self, type_text: str) -> str:
        normalized = self._normalize_inferred_type(type_text)
        specialization = self._materialized_struct_specializations.get(normalized)
        if specialization is None:
            return normalized
        source_name, source_arguments = specialization
        return self._normalize_template_argument_text(
            f"{source_name}<{', '.join(source_arguments)}>"
        )

    @staticmethod
    def _pointer_argument_matches_declared_type(
        declared_pointer_type: str, concrete_pointer_type: str
    ) -> bool:
        address_space_tokens = {"device", "constant", "thread", "threadgroup"}
        declared_tokens = set(IDENTIFIER_RE.findall(declared_pointer_type))
        concrete_tokens = set(IDENTIFIER_RE.findall(concrete_pointer_type))
        declared_spaces = declared_tokens & address_space_tokens
        concrete_spaces = concrete_tokens & address_space_tokens
        if len(declared_spaces) != 1 or declared_spaces != concrete_spaces:
            return False
        # Pointee qualification may be added by the declared parameter, but it
        # may not be discarded from the concrete argument.
        for qualifier in ("const", "volatile"):
            if qualifier in concrete_tokens and qualifier not in declared_tokens:
                return False
        return True

    @staticmethod
    def _buffer_element_type(
        buffer_element_types: _MetalBufferTypeView, name: str
    ) -> Optional[str]:
        entry = buffer_element_types.get(name)
        if isinstance(entry, _MetalSubscriptableType):
            return entry.element_type
        return entry if isinstance(entry, str) and entry else None

    def _buffer_pointer_type(
        self, buffer_element_types: _MetalBufferTypeView, name: str
    ) -> Optional[str]:
        entry = buffer_element_types.get(name)
        if isinstance(entry, _MetalSubscriptableType):
            return self._normalize_known_address_space_pointer_type(
                entry.pointer_type or ""
            )
        return None

    def _buffer_pointer_expression_type(
        self, buffer_element_types: _MetalBufferTypeView, name: str
    ) -> Optional[str]:
        # Structured entries distinguish a proven pointer from an element type
        # whose address space is unknown. Plain strings remain supported for the
        # legacy unit-level element maps used by callers of this private helper.
        entry = buffer_element_types.get(name)
        if isinstance(entry, _MetalSubscriptableType):
            return self._normalize_known_address_space_pointer_type(
                entry.pointer_type or ""
            )
        if not isinstance(entry, str) or not entry or "*" in entry:
            return None
        return entry

    def _is_metal_scalar_or_vector_type(self, type_text: str) -> bool:
        return self._scalar_and_width(type_text) is not None

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

        # Address-of a proven buffer/array element preserves the source pointer's
        # address space, cv qualifiers, and pointee type. Every other unary `&`
        # form is deliberately unsupported and therefore fails closed.
        if expr.startswith("&"):
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

        # Literal types.
        literal_type = self._infer_literal_type(expr)
        if literal_type is not None:
            return literal_type

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

        # Subscript access `base[expr]` -> element type of `base`. `base` may be a
        # bare buffer/array name (`buf[i]`, `totals[i]`) OR a member-access into a
        # struct local (`obj.member[i]`); both balance to a single trailing
        # subscript.
        bracket = expr.find("[")
        if bracket != -1 and expr.endswith("]"):
            close = self._find_matching_delimiter(expr, bracket, "[", "]")
            if close == len(expr) - 1:
                base = expr[:bracket].strip()
                element = self._infer_subscript_base_element_type(
                    base, buffer_element_types, struct_field_types
                )
                if element is not None:
                    return element

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

        # Binary arithmetic `a + b`, `a - b`, `a * b`, `a / b` (checked last, as a
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

        return None

    def _infer_addressed_buffer_element_pointer_type(
        self,
        expr: str,
        buffer_element_types: _MetalBufferTypeView,
        local_variable_types: Dict[str, str],
        struct_field_types: Optional[Dict[str, Dict[str, str]]],
        structs_by_name: Optional[Dict[str, "_MetalStructDefinition"]],
    ) -> Optional[str]:
        # Accepted shape: `&base[index]` with optional enclosing parentheses
        # around the subscript. `base` must be a bare, tracked subscriptable whose
        # declaration retained one explicit Metal address space. The index must
        # be a side-effect-free integral expression; calls, assignment,
        # increment/decrement, comma/conditional expressions, nested subscripts,
        # and precedence-ambiguous forms are rejected.
        if not expr.startswith("&") or expr.startswith("&&"):
            return None
        operand = self._strip_enclosing_parens(expr[1:].strip())
        bracket = operand.find("[")
        if bracket <= 0 or not operand.endswith("]"):
            return None
        close = self._find_matching_delimiter(operand, bracket, "[", "]")
        if close != len(operand) - 1:
            return None
        base = operand[:bracket].strip()
        if IDENTIFIER_RE.fullmatch(base) is None:
            return None
        pointer_type = self._buffer_pointer_type(buffer_element_types, base)
        if pointer_type is None:
            return None
        index = operand[bracket + 1 : close].strip()
        if not self._direct_reference_expression_is_safe(
            index
        ) or self._static_integral_expression_has_ambiguous_precedence(index):
            return None
        index_type = self._infer_argument_type(
            index,
            buffer_element_types,
            local_variable_types,
            struct_field_types,
            structs_by_name,
        )
        if index_type is None or not self._is_integral_concrete_type(index_type):
            return None
        return pointer_type

    def _infer_known_member_function_call_type(self, expr: str) -> Optional[str]:
        match = re.match(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(", expr)
        if match is None:
            return None
        return_type = self._known_member_function_return_types.get(match.group("name"))
        if return_type is None:
            return None
        paren_start = expr.find("(", match.start("name") + len(match.group("name")))
        paren_end = self._find_matching_delimiter(expr, paren_start, "(", ")")
        if paren_end != len(expr) - 1:
            return None
        return self._normalize_inferred_expression_type(return_type)

    def _infer_pointer_arithmetic_type(
        self,
        expr: str,
        buffer_element_types: _MetalBufferTypeView,
        local_variable_types: Dict[str, str],
        struct_field_types: Optional[Dict[str, Dict[str, str]]],
        structs_by_name: Optional[Dict[str, "_MetalStructDefinition"]],
    ) -> Optional[str]:
        expression = self._strip_enclosing_parens(expr)
        if IDENTIFIER_RE.fullmatch(expression):
            if expression in buffer_element_types:
                return self._buffer_pointer_expression_type(
                    buffer_element_types, expression
                )
            if expression in local_variable_types:
                return self._normalize_known_address_space_pointer_type(
                    local_variable_types[expression]
                )
            return None
        split = self._split_top_level_binary_arithmetic(expression)
        if split is None:
            return None
        left_expr, operator, right_expr = split
        if operator not in {"+", "-"}:
            return None
        left_expr = left_expr.strip()
        right_expr = right_expr.strip()
        left_pointer = self._infer_pointer_arithmetic_type(
            left_expr,
            buffer_element_types,
            local_variable_types,
            struct_field_types,
            structs_by_name,
        )
        right_pointer = self._infer_pointer_arithmetic_type(
            right_expr,
            buffer_element_types,
            local_variable_types,
            struct_field_types,
            structs_by_name,
        )
        if left_pointer is not None and right_pointer is None:
            right_type = self._infer_argument_type(
                right_expr,
                buffer_element_types,
                local_variable_types,
                struct_field_types,
                structs_by_name,
            )
            if right_type is not None and self._is_integral_concrete_type(right_type):
                return left_pointer
        if operator == "+" and right_pointer is not None and left_pointer is None:
            left_type = self._infer_argument_type(
                left_expr,
                buffer_element_types,
                local_variable_types,
                struct_field_types,
                structs_by_name,
            )
            if left_type is not None and self._is_integral_concrete_type(left_type):
                return right_pointer
        return None

    def _contains_bare_pointer_operand(
        self,
        expr: str,
        buffer_element_types: _MetalBufferTypeView,
    ) -> bool:
        expression = self._strip_enclosing_parens(expr)
        if IDENTIFIER_RE.fullmatch(expression):
            return expression in buffer_element_types
        split = self._split_top_level_binary_arithmetic(expression)
        if split is None:
            return False
        left_expr, _operator, right_expr = split
        return self._contains_bare_pointer_operand(
            left_expr, buffer_element_types
        ) or self._contains_bare_pointer_operand(right_expr, buffer_element_types)

    def _infer_braced_construction_type(
        self,
        expr: str,
        structs_by_name: Optional[Dict[str, "_MetalStructDefinition"]],
    ) -> Optional[str]:
        # `T{...}` (brace-init construction, no trailing call) has type T when T
        # is a recognized scalar/vector type or a tracked struct/union. The brace
        # group must span the whole expression, so a functor construction-and-call
        # `T{...}(args)` (handled earlier) is not misread as a bare construction.
        match = re.match(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)", expr)
        if match is None:
            return None
        index = match.end()
        while index < len(expr) and expr[index].isspace():
            index += 1
        type_name = match.group("name")
        is_template_id = False
        if index < len(expr) and expr[index] == "<":
            angle_close = self._find_matching_angle(expr, index)
            if angle_close is None:
                return None
            type_name = self._normalize_template_argument_text(expr[: angle_close + 1])
            index = angle_close + 1
            while index < len(expr) and expr[index].isspace():
                index += 1
            is_template_id = True
        if index >= len(expr) or expr[index] != "{":
            return None
        brace_open = index
        brace_close = self._find_matching_delimiter(expr, brace_open, "{", "}")
        if brace_close != len(expr) - 1:
            return None
        if is_template_id:
            return type_name
        if type_name in self._METAL_SCALAR_VECTOR_TYPES or (
            structs_by_name is not None and type_name in structs_by_name
        ):
            return self._normalize_inferred_type(type_name)
        return None

    def _infer_functor_construction_call_type(
        self,
        expr: str,
        buffer_element_types: _MetalBufferTypeView,
        local_variable_types: Dict[str, str],
        struct_field_types: Optional[Dict[str, Dict[str, str]]],
        structs_by_name: Optional[Dict[str, "_MetalStructDefinition"]],
    ) -> Optional[str]:
        # A default-constructed functor temporary that is immediately CALLED:
        # `F{}(args)`, `F{...}(args)` or `F()(args)`. Its type is the result type
        # of `F::operator()` for the (recursively inferred) argument types. The
        # shape is a leading identifier, a construction group (`{...}` or `()`),
        # then a call group `(args)` that closes the whole expression.
        if not structs_by_name:
            return None
        match = re.match(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<open>[{(])", expr)
        if match is None:
            return None
        struct = structs_by_name.get(match.group("name"))
        if struct is None:
            return None
        open_char = match.group("open")
        open_pos = match.start("open")
        close_char = "}" if open_char == "{" else ")"
        constructor_close = self._find_matching_delimiter(
            expr, open_pos, open_char, close_char
        )
        if constructor_close is None:
            return None
        call_open = constructor_close + 1
        while call_open < len(expr) and expr[call_open].isspace():
            call_open += 1
        if call_open >= len(expr) or expr[call_open] != "(":
            return None
        call_close = self._find_matching_delimiter(expr, call_open, "(", ")")
        if call_close != len(expr) - 1:
            return None
        raw_args = expr[call_open + 1 : call_close]
        argument_exprs = [
            argument
            for argument in self._split_top_level_commas(raw_args)
            if argument.strip()
        ]
        return self._infer_functor_operator_call_result_type(
            struct,
            argument_exprs,
            buffer_element_types,
            local_variable_types,
            struct_field_types,
            structs_by_name,
        )

    def _infer_functor_operator_call_result_type(
        self,
        struct: "_MetalStructDefinition",
        argument_exprs: List[str],
        buffer_element_types: _MetalBufferTypeView,
        local_variable_types: Dict[str, str],
        struct_field_types: Optional[Dict[str, Dict[str, str]]],
        structs_by_name: Optional[Dict[str, "_MetalStructDefinition"]],
    ) -> Optional[str]:
        # Conservatively resolve the result type of `struct::operator()(args)`.
        concrete_ops = [method for method in struct.methods if method.is_operator_call]
        template_ops = [
            method for method in struct.template_methods if method.is_operator_call
        ]
        if not concrete_ops and not template_ops:
            return None

        argument_types = [
            self._infer_argument_type(
                argument,
                buffer_element_types,
                local_variable_types,
                struct_field_types,
                structs_by_name,
            )
            for argument in argument_exprs
        ]
        normalized_args = [
            self._normalize_inferred_type(argument) if argument is not None else None
            for argument in argument_types
        ]

        # 1. A concrete operator() overload whose declared parameter types match
        #    the inferred argument types -> its concrete return type (e.g. the
        #    `complex64_t operator()(complex64_t)` overload of Sqrt/Log/Abs).
        if all(argument is not None for argument in normalized_args):
            for method in concrete_ops:
                declared = self._operator_call_parameter_types(method)
                if declared == normalized_args:
                    result = self._normalize_inferred_expression_type(
                        method.return_type
                    )
                    if result is not None:
                        return result

        # 2. A template operator() whose parameters bind from the inferred
        #    argument types -> its return type with those bindings applied. Covers
        #    the identity `T operator()(T x)` (result == argument type) and any
        #    return whose template parameters are all deducible from the call.
        if argument_types and all(argument is not None for argument in argument_types):
            for method in template_ops:
                bindings = self._bind_template_method_parameters(
                    method,
                    list(argument_types),
                    owner_struct=struct,
                    structs_by_name=structs_by_name,
                )
                if bindings is None:
                    continue
                substituted = self._normalize_inferred_expression_type(
                    self._replace_identifiers(method.return_type, bindings)
                )
                if (
                    substituted is not None
                    and not self._type_references_template_parameter(
                        substituted, method.template_parameters
                    )
                ):
                    return substituted

        # 3. Argument-INDEPENDENT result: when EVERY operator() overload returns
        #    the same concrete (non-template-dependent) type, that type is the
        #    result regardless of which overload applies (e.g. Real/Imag always
        #    return float). Bails for an identity-like functor whose operator()
        #    return type is one of its own template parameters.
        fixed_returns: Set[str] = set()
        for method in [*concrete_ops, *template_ops]:
            result = self._normalize_inferred_expression_type(method.return_type)
            if result is None or self._type_references_template_parameter(
                result, method.template_parameters
            ):
                return None
            fixed_returns.add(result)
        if len(fixed_returns) == 1:
            return next(iter(fixed_returns))
        return None

    def _operator_call_parameter_types(self, method: "_MetalStructMethod") -> List[str]:
        # The normalized declared parameter types of an operator() overload, in
        # the same shape produced for inferred call-argument types so the two can
        # be compared directly.
        return [
            normalized
            for normalized in (
                self._normalize_inferred_type(
                    self._normalize_function_parameter_type_text(parameter)
                )
                for parameter in self._split_top_level_commas(method.parameters)
            )
            if normalized and normalized != "void"
        ]

    def _type_references_template_parameter(
        self, type_text: str, template_parameters: List[str]
    ) -> bool:
        # True when `type_text` still mentions one of `template_parameters` (i.e.
        # the type stayed template-dependent after binding substitution).
        if not template_parameters:
            return False
        parameters = set(template_parameters)
        return any(
            match.group(0) in parameters for match in IDENTIFIER_RE.finditer(type_text)
        )

    def _infer_binary_arithmetic_type(
        self,
        expr: str,
        buffer_element_types: _MetalBufferTypeView,
        local_variable_types: Dict[str, str],
        struct_field_types: Optional[Dict[str, Dict[str, str]]],
        structs_by_name: Optional[Dict[str, "_MetalStructDefinition"]],
    ) -> Optional[str]:
        # Split a compound expression at its lowest-precedence top-level binary
        # arithmetic operator (`+ - * /`), infer both operands, and combine them
        # under a conservative model of the usual arithmetic conversion.
        split = self._split_top_level_binary_arithmetic(expr)
        if split is None:
            return None
        left_expr, arithmetic_operator, right_expr = split
        left_expr = left_expr.strip()
        right_expr = right_expr.strip()
        if not left_expr or not right_expr:
            return None
        left_type = self._infer_argument_type(
            left_expr,
            buffer_element_types,
            local_variable_types,
            struct_field_types,
            structs_by_name,
        )
        right_type = self._infer_argument_type(
            right_expr,
            buffer_element_types,
            local_variable_types,
            struct_field_types,
            structs_by_name,
        )
        if left_type is None or right_type is None:
            return None
        integral_constant_type = self._integral_constant_binary_result_type(
            left_type, right_type, arithmetic_operator
        )
        if integral_constant_type is not None:
            return integral_constant_type
        return self._combine_binary_arithmetic_operand_types(
            left_type, right_type, left_expr, right_expr
        )

    def _integral_constant_binary_result_type(
        self, left_type: str, right_type: str, arithmetic_operator: str
    ) -> Optional[str]:
        if arithmetic_operator not in self._integral_constant_binary_operators:
            return None
        left = self._integral_constant_type_parts(left_type)
        right = self._integral_constant_type_parts(right_type)
        if left is None or right is None:
            return None
        left_base, left_value = left
        right_base, right_value = right
        # The MLX-style Int alias and the underlying operator overload both use
        # int operands. Wider and mixed signedness cases require the complete C++
        # usual-arithmetic-conversion contract and remain deliberately unresolved.
        if left_base != "int" or right_base != "int":
            return None
        folded, result = self._evaluate_static_integral_expression(
            f"({left_value}) {arithmetic_operator} ({right_value})"
        )
        if not folded or result is None:
            return None
        return self._normalize_template_argument_text(
            f"integral_constant<int, {result}>"
        )

    def _integral_constant_type_parts(
        self, type_text: str
    ) -> Optional[Tuple[str, int]]:
        normalized = self._normalize_template_argument_text(type_text)
        match = re.fullmatch(
            r"Int\s*<(?P<value>.+)>",
            normalized,
        )
        if match is not None and self._int_alias_contract_verified:
            folded, value = self._evaluate_static_integral_expression(
                match.group("value")
            )
            return ("int", value) if folded and value is not None else None

        match = re.fullmatch(
            r"integral_constant\s*<(?P<arguments>.+)>",
            normalized,
        )
        if match is None or not self._integral_constant_contract_verified:
            return None
        arguments = [
            argument.strip()
            for argument in self._split_top_level_commas(match.group("arguments"))
        ]
        if len(arguments) != 2:
            return None
        base_type = self._normalize_inferred_type(arguments[0])
        if self._is_integral_concrete_type(base_type) is not True:
            return None
        folded, value = self._evaluate_static_integral_expression(arguments[1])
        if not folded or value is None:
            return None
        return base_type, value

    def _combine_binary_arithmetic_operand_types(
        self, left_type: str, right_type: str, left_expr: str, right_expr: str
    ) -> Optional[str]:
        # Conservative usual-arithmetic-conversion result of two operand types:
        #   * identical types -> that type;
        #   * a concrete operand combined with a numeric literal:
        #       - an AGGREGATE concrete (struct/complex/etc.) absorbs the literal
        #         (its arithmetic operators are defined to return the aggregate);
        #       - a SCALAR/VECTOR concrete keeps its type UNLESS a floating-point
        #         literal meets an integer scalar (C++ then widens to floating and
        #         the target width is unknown), in which case we bail;
        #   * anything else (two different concrete non-literal types, or two
        #     differing literals) -> ambiguous, so bail rather than guess.
        # Pointer arithmetic has a dedicated, address-space-aware path. Reaching
        # this scalar fallback with any pointer means that path rejected the
        # expression, so normalizing qualifiers here would reopen it as `T*`.
        if "*" in left_type or "*" in right_type:
            return None
        left = self._normalize_inferred_type(left_type)
        right = self._normalize_inferred_type(right_type)
        if not left or not right:
            return None
        if left == right:
            return left
        promoted_left = self._promote_small_integral_scalar(left)
        promoted_right = self._promote_small_integral_scalar(right)
        if (
            promoted_left is not None
            and promoted_right is not None
            and promoted_left == promoted_right
        ):
            return promoted_left
        left_literal = self._infer_literal_type(self._strip_enclosing_parens(left_expr))
        right_literal = self._infer_literal_type(
            self._strip_enclosing_parens(right_expr)
        )
        left_is_literal = left_literal is not None
        right_is_literal = right_literal is not None
        if left_is_literal == right_is_literal:
            # Both literals (differing types) or both concrete (differing types):
            # ambiguous under this conservative model.
            return None
        if left_is_literal:
            concrete, literal_type = right, self._normalize_inferred_type(left_literal)
        else:
            concrete, literal_type = left, self._normalize_inferred_type(right_literal)
        # A struct/aggregate value's operators define the result type as the
        # aggregate (e.g. `complex64_t (+|-|*|/) float` -> complex64_t).
        if concrete not in self._METAL_SCALAR_VECTOR_TYPES:
            return concrete
        # Scalar/vector concrete: only trust the concrete type when the literal
        # does not out-rank it (a floating literal meeting an integer scalar
        # promotes to a floating type of unknown width -> bail).
        if self._numeric_literal_outranks_scalar(literal_type, concrete):
            return None
        return concrete

    def _promote_small_integral_scalar(self, type_text: str) -> Optional[str]:
        decomposed = self._scalar_and_width(type_text)
        if decomposed is None:
            return None
        base_type, width = decomposed
        if width != 1 or base_type not in self._METAL_INTEGRAL_SCALAR_TYPES:
            return None
        if (
            self._METAL_SCALAR_TYPE_SIZES[base_type]
            < self._METAL_SCALAR_TYPE_SIZES["int"]
        ):
            return "int"
        return base_type

    _INTEGER_SCALAR_BASES: Set[str] = {
        "bool",
        "char",
        "uchar",
        "short",
        "ushort",
        "int",
        "uint",
        "long",
        "ulong",
    }

    def _numeric_literal_outranks_scalar(
        self, literal_type: Optional[str], scalar_type: str
    ) -> bool:
        # True when a numeric literal of `literal_type` would promote an INTEGER
        # `scalar_type` to a wider (floating) type under the usual arithmetic
        # conversions -- i.e. a floating literal (`1.0`, `2.5h`) meeting an integer
        # scalar/vector. Integer literals never out-rank (they promote to the
        # concrete's own family), so those keep the concrete type.
        if literal_type not in {"float", "half"}:
            return False
        base = re.sub(r"[0-9]+$", "", scalar_type or "")
        return base in self._INTEGER_SCALAR_BASES

    def _strip_enclosing_parens(self, expr: str) -> str:
        expr = expr.strip()
        while (
            expr.startswith("(")
            and self._find_matching_delimiter(expr, 0, "(", ")") == len(expr) - 1
        ):
            expr = expr[1:-1].strip()
        return expr

    def _split_top_level_binary_arithmetic(
        self, expr: str
    ) -> Optional[Tuple[str, str, str]]:
        # Locate the operator at which to split `expr` for arithmetic inference:
        # the LOWEST-precedence top-level `+ - * /` (additive below
        # multiplicative), rightmost among equal precedence so the split mirrors
        # C++'s left-associative grouping. Operators inside (), [], {} or string
        # literals are skipped, as are unary signs, `->`, `++`/`--`, compound
        # assignments and floating-point exponent signs. Returns (left, op, right)
        # or None when no top-level binary arithmetic operator is present.
        precedence = {"+": 0, "-": 0, "*": 1, "/": 1}
        depth = 0
        best_index: Optional[int] = None
        best_precedence: Optional[int] = None
        i = 0
        n = len(expr)
        while i < n:
            ch = expr[i]
            if ch in "\"'":
                _literal, consumed = self._read_string(expr, i)
                i += consumed
                continue
            if ch in "([{":
                depth += 1
                i += 1
                continue
            if ch in ")]}":
                depth = max(0, depth - 1)
                i += 1
                continue
            if (
                depth == 0
                and ch in precedence
                and self._is_binary_operator_position(expr, i)
            ):
                prec = precedence[ch]
                if best_precedence is None or prec <= best_precedence:
                    best_precedence = prec
                    best_index = i
            i += 1
        if best_index is None:
            return None
        return expr[:best_index], expr[best_index], expr[best_index + 1 :]

    def _is_binary_operator_position(self, expr: str, index: int) -> bool:
        # True when the `+ - * /` at `index` is a BINARY operator rather than a
        # unary sign, a `->` arrow, an increment/decrement, a compound assignment,
        # or a floating-point exponent sign.
        ch = expr[index]
        nxt = expr[index + 1] if index + 1 < len(expr) else ""
        if ch == "-" and nxt == ">":
            return False
        if nxt == ch and ch in "+-":
            return False
        if nxt == "=":
            return False
        j = index - 1
        while j >= 0 and expr[j].isspace():
            j -= 1
        if j < 0:
            return False
        prev = expr[j]
        if not (prev.isalnum() or prev == "_" or prev in ")]}."):
            return False
        # `1e+5` / `2.0E-3`: a +/- right after an exponent marker that follows a
        # digit or dot belongs to the number, not a binary operator.
        if ch in "+-" and prev in "eE":
            k = j - 1
            if k >= 0 and (expr[k].isdigit() or expr[k] == "."):
                return False
        return True

    def _infer_subscript_base_element_type(
        self,
        base: str,
        buffer_element_types: _MetalBufferTypeView,
        struct_field_types: Optional[Dict[str, Dict[str, str]]],
    ) -> Optional[str]:
        # Element type of the subscript base `base[...]`:
        #   * bare name: a buffer/array element type (`buf[i]`, `totals[i]`).
        #   * member access `obj.member` / `obj->member`: the element type of the
        #     struct field `member`.
        if IDENTIFIER_RE.fullmatch(base):
            return self._buffer_element_type(buffer_element_types, base)
        field_type = self._struct_member_field_type(base, struct_field_types)
        if field_type is None:
            return None
        # A subscript yields the field's element type; strip a single pointer
        # marker if the field type recorded one. Array fields already record the
        # element type, so the value is returned as-is.
        return field_type.rstrip("*").strip() or None

    def _infer_group_builtin_call_type(
        self,
        expr: str,
        buffer_element_types: _MetalBufferTypeView,
        local_variable_types: Dict[str, str],
        struct_field_types: Optional[Dict[str, Dict[str, str]]],
        structs_by_name: Optional[Dict[str, "_MetalStructDefinition"]],
    ) -> Optional[str]:
        # `builtin(arg0, ...)` for a SIMD/quad group built-in that returns arg0's
        # type -> the inferred type of arg0. Conservative: only a recognized
        # built-in with a fully balanced trailing call and an inferable first
        # argument yields a type.
        match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", expr)
        if match is None:
            return None
        if match.group(1) not in self._METAL_FIRST_ARG_TYPED_GROUP_BUILTINS:
            return None
        paren_start = expr.find("(")
        paren_end = self._find_matching_delimiter(expr, paren_start, "(", ")")
        if paren_end != len(expr) - 1:
            return None
        arguments = [
            argument
            for argument in self._split_top_level_commas(
                expr[paren_start + 1 : paren_end]
            )
            if argument.strip()
        ]
        if not arguments:
            return None
        return self._infer_argument_type(
            arguments[0],
            buffer_element_types,
            local_variable_types,
            struct_field_types,
            structs_by_name,
        )

    def _infer_member_access_type(
        self,
        expr: str,
        struct_field_types: Optional[Dict[str, Dict[str, str]]],
    ) -> Optional[str]:
        field_type = self._struct_member_field_type(expr, struct_field_types)
        if field_type is None:
            return None
        # A bare member access onto a pointer field is not a value; reject it.
        if field_type.endswith("*"):
            return None
        return field_type or None

    def _infer_vector_member_access_type(
        self, expr: str, local_variable_types: Dict[str, str]
    ) -> Optional[str]:
        match = re.fullmatch(
            r"(?P<obj>[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*"
            r"(?P<swizzle>[A-Za-z_][A-Za-z0-9_]*)",
            expr,
        )
        if match is None:
            return None
        vector_type = self._normalize_inferred_type(
            local_variable_types.get(match.group("obj"), "")
        )
        vector_match = re.fullmatch(
            r"(?P<base>[A-Za-z_][A-Za-z0-9_]*)(?P<width>[234])", vector_type
        )
        if vector_match is None or vector_type not in self._METAL_SCALAR_VECTOR_TYPES:
            return None

        swizzle = match.group("swizzle")
        if not 1 <= len(swizzle) <= 4:
            return None
        component_sets = ("xyzw", "rgba")
        components = next(
            (
                component_set
                for component_set in component_sets
                if all(ch in component_set for ch in swizzle)
            ),
            None,
        )
        if components is None:
            return None
        width = int(vector_match.group("width"))
        if any(components.index(component) >= width for component in swizzle):
            return None
        base_type = vector_match.group("base")
        return base_type if len(swizzle) == 1 else f"{base_type}{len(swizzle)}"

    def _struct_member_field_type(
        self,
        access: str,
        struct_field_types: Optional[Dict[str, Dict[str, str]]],
    ) -> Optional[str]:
        # Resolve a single-level `obj.member` or `obj->member` access to the
        # member's declared field type. Pointer bindings are stored under an
        # `obj->` key so the access operator must match the declaration shape.
        if not struct_field_types:
            return None
        match = re.fullmatch(
            r"(?P<obj>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<access>\.|->)\s*"
            r"(?P<member>[A-Za-z_][A-Za-z0-9_]*)",
            access,
        )
        if match is None:
            return None
        lookup = match.group("obj")
        if match.group("access") == "->":
            lookup += "->"
        fields = struct_field_types.get(lookup)
        if not fields:
            return None
        return fields.get(match.group("member"))

    def _infer_literal_type(self, expr: str) -> Optional[str]:
        # Recognize the simple numeric/boolean literal forms required by the
        # conservative inference contract.
        if expr in {"true", "false"}:
            return "bool"
        # half literal: `1h`, `2.5h`.
        if re.fullmatch(r"[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?h", expr):
            return "half"
        # float literal: `1.0`, `2.5f`, `1.0f`, `3.f`, `1e3f`.
        if re.fullmatch(r"[0-9]*\.[0-9]*(?:[eE][+-]?[0-9]+)?f?", expr) and any(
            c.isdigit() for c in expr
        ):
            return "float"
        if re.fullmatch(r"[0-9]+(?:[eE][+-]?[0-9]+)?f", expr):
            return "float"
        if re.fullmatch(r"[0-9]+[eE][+-]?[0-9]+", expr):
            return "float"
        # unsigned integer literal: `1u`, `42U`.
        if re.fullmatch(r"[0-9]+[uU]", expr):
            return "uint"
        # plain integer literal: `1`, `42` (and hex).
        if re.fullmatch(r"[0-9]+", expr) or re.fullmatch(r"0[xX][0-9A-Fa-f]+", expr):
            return "int"
        return None

    def _remove_spans(self, text: str, spans: List[Tuple[int, int]]) -> str:
        if not spans:
            return text
        result: List[str] = []
        pos = 0
        for start, end in sorted(spans):
            if start < pos:
                continue
            result.append(text[pos:start])
            pos = end
        result.append(text[pos:])
        return "".join(result)

    def _collapse_blank_lines(self, text: str) -> str:
        # Collapse runs of blank lines left behind by removed method bodies so
        # the data-only struct stays tidy without altering meaningful content.
        return re.sub(r"\n[ \t]*\n[ \t]*(\n[ \t]*)+", "\n\n", text)

    def _find_mlx_kernel_instantiations(
        self, code: str
    ) -> List[_MLXKernelInstantiation]:
        return self._find_project_template_instantiations(code)

    def _find_project_template_instantiations(
        self, code: str
    ) -> List[_MLXKernelInstantiation]:
        instantiations: List[_MLXKernelInstantiation] = []
        instantiations.extend(self._find_raw_mlx_kernel_instantiations(code))
        instantiations.extend(self._find_declared_mlx_kernel_instantiations(code))
        instantiations.extend(self._find_declared_template_instantiations(code))
        instantiations = self._dedupe_template_instantiations(instantiations)
        return sorted(instantiations, key=lambda item: item.span[0])

    def _dedupe_template_instantiations(
        self, instantiations: List[_MLXKernelInstantiation]
    ) -> List[_MLXKernelInstantiation]:
        # The same expanded `template [[host_name(...)]] ... Name<args>;`
        # declaration is matched by more than one detector (the declared-host-name
        # scanner and the generic declared-template scanner), so an instantiation
        # can be reported twice with slightly different (overlapping) spans. Those
        # duplicates double the instantiation count, which both inflates the
        # `max_template_specializations` budget check (causing a spurious bail-out
        # that leaves generic templates un-materialized and later flagged as
        # unresolved) and adds redundant removal spans. Collapse entries that share
        # the same target specialization — identical function name, normalized
        # template arguments, and host name — keeping the widest covering span so
        # the declaration is fully removed during materialization.
        unique: dict[tuple[str, tuple[str, ...], str], _MLXKernelInstantiation] = {}
        for instantiation in instantiations:
            key = (
                instantiation.function_name,
                tuple(
                    self._normalize_template_argument_text(argument)
                    for argument in instantiation.template_arguments
                ),
                instantiation.host_name,
            )
            existing = unique.get(key)
            if existing is None:
                unique[key] = instantiation
                continue
            # Keep the widest span so the full declaration text is removed.
            start = min(existing.span[0], instantiation.span[0])
            end = max(existing.span[1], instantiation.span[1])
            if (start, end) != existing.span:
                unique[key] = replace(existing, span=(start, end))
        return list(unique.values())

    def _find_raw_mlx_kernel_instantiations(
        self, code: str
    ) -> List[_MLXKernelInstantiation]:
        instantiations: List[_MLXKernelInstantiation] = []
        for match in MLX_INSTANTIATE_KERNEL_RE.finditer(code):
            open_paren = code.find("(", match.start())
            args, consumed = self._parse_macro_args(code, open_paren)
            if not consumed or len(args) < 3:
                continue
            end = open_paren + consumed
            while end < len(code) and code[end].isspace():
                end += 1
            if end < len(code) and code[end] == ";":
                end += 1

            host_name = self._evaluate_metal_string_expression(args[0])
            function_name = args[1].strip()
            if not host_name or not IDENTIFIER_RE.fullmatch(function_name):
                continue
            instantiations.append(
                _MLXKernelInstantiation(
                    host_name=host_name,
                    function_name=function_name,
                    template_arguments=[arg.strip() for arg in args[2:]],
                    span=(match.start(), end),
                )
            )
        return instantiations

    def _find_declared_template_instantiations(
        self, code: str
    ) -> List[_MLXKernelInstantiation]:
        instantiations: List[_MLXKernelInstantiation] = []
        templates_by_name = {
            template.name: template for template in self._find_template_functions(code)
        }
        for match in re.finditer(
            r"(?:^|[;\n])\s*(?:(?:template\s+)?\[\[|template\s+(?!<))",
            code,
        ):
            start = match.start()
            if code[start : start + 1] == ";":
                start += 1
            declaration_end = self._statement_end(code, start)
            if declaration_end is None:
                continue
            declaration = code[start:declaration_end]
            if self._find_next_top_level_char(declaration, 0, "{") is not None:
                continue
            declared_function_name = self._declared_function_name(declaration)
            if (
                "decltype" not in declaration
                and not re.search(r"\b[A-Za-z_][A-Za-z0-9_:]*\s*<", declaration)
                and (
                    declared_function_name is None
                    or declared_function_name not in templates_by_name
                )
            ):
                continue
            host_name = self._host_name_from_attributes(declaration)
            template_id_candidate = self._declared_template_id_candidate(
                declaration,
                declared_function_name,
            )
            if template_id_candidate is not None:
                function_name, arguments = template_id_candidate
                instantiations.append(
                    _MLXKernelInstantiation(
                        host_name=host_name or function_name.split("::")[-1],
                        function_name=function_name.split("::")[-1],
                        template_arguments=arguments,
                        span=(start, declaration_end),
                    )
                )
                continue

            if declared_function_name is None:
                continue
            template = templates_by_name.get(declared_function_name)
            if template is None:
                continue
            arguments = self._infer_declared_template_arguments(
                template,
                declaration,
            )
            if not arguments:
                continue
            instantiations.append(
                _MLXKernelInstantiation(
                    host_name=host_name or declared_function_name,
                    function_name=declared_function_name,
                    template_arguments=arguments,
                    span=(start, declaration_end),
                )
            )
        return instantiations

    def _find_declared_mlx_kernel_instantiations(
        self, code: str
    ) -> List[_MLXKernelInstantiation]:
        instantiations: List[_MLXKernelInstantiation] = []
        for match in MLX_HOST_NAME_DECL_RE.finditer(code):
            args_start = match.end()
            args_end = self._find_matching_angle(code, args_start - 1)
            if args_end is None:
                continue
            declaration_end = code.find(";", args_end)
            if declaration_end == -1:
                continue
            host_name = self._evaluate_metal_string_expression(match.group("host"))
            function_name = match.group("function").split("::")[-1]
            template_arguments = self._split_top_level_commas(code[args_start:args_end])
            if not host_name or not template_arguments:
                continue
            instantiations.append(
                _MLXKernelInstantiation(
                    host_name=host_name,
                    function_name=function_name,
                    template_arguments=template_arguments,
                    span=(match.start(), declaration_end + 1),
                )
            )
        return instantiations

    def _find_template_functions(self, code: str) -> List[_MetalTemplateFunction]:
        templates: List[_MetalTemplateFunction] = []
        namespace_spans = self._find_namespace_spans(code)
        template_type_traits = self._find_template_type_traits(code, namespace_spans)
        constructor_template_spans = sorted(
            constructor.span
            for struct in self._find_concrete_struct_definitions(code)
            for constructor in struct.constructors
            if constructor.template_parameters and constructor.span is not None
        )
        pos = 0
        while True:
            match = re.search(r"\btemplate\s*<", code[pos:])
            if match is None:
                break
            start = pos + match.start()
            angle_start = code.find("<", start)
            angle_end = self._find_matching_template_param_angle(code, angle_start)
            if angle_end is None:
                pos = start + len("template")
                continue

            declaration_start = angle_end + 1
            body_start = self._find_next_top_level_char(code, declaration_start, "{")
            semicolon = self._find_next_top_level_char(code, declaration_start, ";")
            if body_start is None or (semicolon is not None and semicolon < body_start):
                pos = declaration_start
                continue
            body_end = self._find_matching_brace(code, body_start)
            if body_end is None:
                pos = body_start + 1
                continue

            # A member-template constructor has no free-function identity. Treating
            # it as a template function materializes ownerless declarations such as
            # `fp8_e4m3_float16_t(float16_t)` at global scope. Keep it inside its
            # concrete owner for the Metal parser/code generator constructor path.
            if self._containing_span(start, constructor_template_spans) is not None:
                pos = body_end
                continue

            header = code[declaration_start:body_start]
            function_name = self._function_name_from_header(header)
            if function_name is None:
                pos = body_end
                continue

            parameter_text = code[angle_start + 1 : angle_end]
            parameters = self._template_parameter_names(parameter_text)
            if not parameters:
                pos = body_end
                continue
            templates.append(
                _MetalTemplateFunction(
                    name=function_name,
                    template_parameters=parameters,
                    span=(start, body_end),
                    body_start=body_start,
                    source=code[declaration_start:body_end],
                    variadic_template_parameters=(
                        self._variadic_template_parameter_names(parameter_text)
                    ),
                    template_parameter_defaults=(
                        self._template_parameter_defaults(parameter_text)
                    ),
                    template_parameter_types={
                        parameter.name: parameter.declared_type
                        for parameter in self._parse_template_parameter_list(
                            parameter_text
                        )
                        if parameter.name is not None
                        and not parameter.is_type_parameter
                        and parameter.declared_type is not None
                    },
                    template_type_traits=template_type_traits,
                    namespace=self._namespace_at(namespace_spans, start),
                )
            )
            pos = body_end
        return templates

    def _find_template_structs(self, code: str) -> List[_MetalTemplateStruct]:
        # Detect `template <...> struct/class Name { ... }` declarations, the
        # struct counterpart of _find_template_functions. Foundation for the
        # struct-template materializer (issue #1354): explicit specializations
        # (empty `template <>`) yield no parameters and are skipped here.
        structs: List[_MetalTemplateStruct] = []
        namespace_spans = self._find_namespace_spans(code)
        template_type_traits = self._find_template_type_traits(code, namespace_spans)
        pos = 0
        while True:
            match = re.search(r"\btemplate\s*<", code[pos:])
            if match is None:
                break
            start = pos + match.start()
            angle_start = code.find("<", start)
            angle_end = self._find_matching_template_param_angle(code, angle_start)
            if angle_end is None:
                pos = start + len("template")
                continue

            declaration_start = angle_end + 1
            header = code[declaration_start : declaration_start + 512]
            header_match = re.match(
                r"\s*(?:\[\[[^\]]*\]\]\s*)*(?:struct|class)\s+"
                r"(?P<name>[A-Za-z_][A-Za-z0-9_:]*)\b",
                header,
                re.DOTALL,
            )
            parameter_text = code[angle_start + 1 : angle_end]
            parameters = self._template_parameter_names(parameter_text)
            if header_match is None or not parameters:
                pos = declaration_start
                continue

            body_start = self._find_next_top_level_char(code, declaration_start, "{")
            semicolon = self._find_next_top_level_char(code, declaration_start, ";")
            if body_start is None or (semicolon is not None and semicolon < body_start):
                pos = declaration_start
                continue
            body_end = self._find_matching_brace(code, body_start)
            if body_end is None:
                pos = body_start + 1
                continue

            structs.append(
                _MetalTemplateStruct(
                    name=header_match.group("name").split("::")[-1],
                    template_parameters=parameters,
                    span=(start, body_end),
                    source=code[declaration_start:body_end],
                    variadic_template_parameters=(
                        self._variadic_template_parameter_names(parameter_text)
                    ),
                    template_parameter_defaults=(
                        self._template_parameter_defaults(parameter_text)
                    ),
                    template_type_traits=template_type_traits,
                    namespace=self._namespace_at(namespace_spans, start),
                )
            )
            pos = body_end
        return structs

    def _materialize_template_function(
        self,
        code: str,
        template: _MetalTemplateFunction,
        instantiation: _MLXKernelInstantiation,
    ) -> str:
        function_identifier = self._materialized_function_identifier(
            instantiation.host_name, template.name
        )
        template_arguments = self._template_arguments_with_defaults(
            code,
            template,
            instantiation.template_arguments,
        )
        return self._materialize_template_function_with_name(
            template,
            template_arguments,
            function_identifier,
            host_name=instantiation.host_name,
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
        materialized = self._rename_function_definition(
            materialized,
            template.name,
            function_identifier,
        )

        if host_name is not None:
            insertion = f'[[host_name("{host_name}")]]\n'
            materialized = insertion + materialized.lstrip()
        if not materialized.endswith("\n"):
            materialized += "\n"
        return materialized

    def _materialize_template_struct_with_name(
        self,
        template: _MetalTemplateStruct,
        template_arguments: List[str],
        struct_identifier: str,
    ) -> str:
        # Struct counterpart of _materialize_template_function_with_name: bind the
        # template parameters (type and non-type) to the concrete arguments,
        # substitute them through the struct body, and rename the declaration.
        # Foundation for the struct-template materializer (issue #1354).
        if not self._template_arguments_satisfy_parameters(
            template,
            template_arguments,
        ):
            return ""
        substitutions, _variadic_bindings = self._template_argument_bindings(
            template,
            template_arguments,
        )
        if not substitutions:
            return ""
        materialized = self._replace_identifiers(template.source, substitutions)
        materialized = self._rename_struct_definition(
            materialized,
            template.name,
            struct_identifier,
        )
        materialized = self._rename_materialized_struct_constructors(
            materialized,
            template.name,
            struct_identifier,
        )
        # A struct/class definition must be terminated with a semicolon; the
        # captured template source ends at the closing brace, so restore it.
        materialized = materialized.rstrip()
        if not materialized.endswith(";"):
            materialized += ";"
        materialized = self._substitute_materialized_struct_static_constants(
            materialized,
            struct_identifier,
        )
        materialized += "\n"
        return materialized

    def _materialize_partial_template_struct_with_name(
        self,
        template: _MetalTemplateStruct,
        bindings: Dict[str, str],
        struct_identifier: str,
    ) -> str:
        if not bindings:
            return ""
        materialized = self._replace_identifiers(template.source, bindings)
        materialized = self._rename_struct_definition(
            materialized,
            template.name,
            struct_identifier,
            strip_specialization_arguments=True,
        )
        materialized = self._rename_materialized_struct_constructors(
            materialized,
            template.name,
            struct_identifier,
        )
        materialized = materialized.rstrip()
        if not materialized.endswith(";"):
            materialized += ";"
        materialized = self._substitute_materialized_struct_static_constants(
            materialized,
            struct_identifier,
        )
        return materialized + "\n"

    def _substitute_materialized_struct_static_constants(
        self, materialized: str, struct_identifier: str
    ) -> str:
        structs = self._find_concrete_struct_definitions(materialized)
        outer = next(
            (struct for struct in structs if struct.name == struct_identifier),
            None,
        )
        if outer is None:
            return materialized
        constants = self._resolved_static_data_member_initializers(outer)
        if not constants:
            return materialized

        protected_spans: List[Tuple[int, int]] = []
        for struct in structs:
            if not (
                outer.span[0] <= struct.span[0] and struct.span[1] <= outer.span[1]
            ):
                continue
            if struct is not outer and constants.keys() & struct.data_member_names:
                protected_spans.append(struct.span)
                continue
            protected_spans.extend(
                method.span for method in [*struct.methods, *struct.template_methods]
            )
            protected_spans.extend(
                constructor.span
                for constructor in struct.constructors
                if constructor.span is not None
            )

        body_start, body_end = outer.body_span
        clipped_spans = sorted(
            (
                max(body_start, start),
                min(body_end, end),
            )
            for start, end in protected_spans
            if start < body_end and body_start < end
        )
        merged_spans: List[Tuple[int, int]] = []
        for start, end in clipped_spans:
            if merged_spans and start <= merged_spans[-1][1]:
                merged_spans[-1] = (merged_spans[-1][0], max(merged_spans[-1][1], end))
            else:
                merged_spans.append((start, end))

        mapping = {
            name: self._static_initializer_reference(initializer)
            for name, initializer in constants.items()
        }
        replacements: List[Tuple[int, int, str]] = []
        cursor = body_start
        for start, end in [*merged_spans, (body_end, body_end)]:
            if cursor < start:
                segment = materialized[cursor:start]
                segment = self._substitute_struct_scoped_alias_owners(
                    segment,
                    outer,
                )
                replacements.append(
                    (
                        cursor,
                        start,
                        self._substitute_static_constants_in_declarations(
                            segment,
                            mapping,
                        ),
                    )
                )
            cursor = max(cursor, end)
        return self._apply_text_replacements(materialized, replacements)

    @staticmethod
    def _static_initializer_reference(initializer: str) -> str:
        normalized = initializer.strip()
        if re.fullmatch(r"(?:true|false|[-+]?\d+[uUlL]*)", normalized):
            return normalized
        return f"({normalized})"

    def _substitute_static_constants_in_declarations(
        self, source: str, mapping: Dict[str, str]
    ) -> str:
        # Preserve each constant's declaration name while resolving references in
        # its initializer and in subsequent aliases, array extents, alignments, and
        # nested type declarations.
        sentinels: Dict[str, str] = {}
        masked = source
        for index, name in enumerate(mapping):
            sentinel = f"__CROSSTL_STATIC_DECL_{index}__"
            while sentinel in masked:
                sentinel += "_"
            pattern = re.compile(rf"\b{re.escape(name)}\b(?=\s*=(?!=))")
            masked, count = pattern.subn(sentinel, masked)
            if count:
                sentinels[sentinel] = name
        substituted = self._substitute_bare_member_references(masked, mapping)
        for sentinel, name in sentinels.items():
            substituted = substituted.replace(sentinel, name)
        return substituted

    def _find_explicit_template_function_calls(
        self,
        code: str,
        templates_by_name: Dict[str, _MetalTemplateFunction],
        excluded_spans: List[Tuple[int, int]],
        explicit_specialization_keys: Set[Tuple[str, Tuple[str, ...]]],
        included_spans: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Tuple[str, List[str], Tuple[int, int]]]:
        calls: List[Tuple[str, List[str], Tuple[int, int]]] = []
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
                i += 1
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
                calls.append((ident, template_arguments, (span_start, angle_end + 1)))
                i = angle_end + 1
                continue
            i += 1
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
        for struct in self._find_concrete_struct_definitions(code):
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
        local_integral_constants = self._collect_local_integral_constant_bindings(
            code,
            local_binding_owner_spans,
            local_type_aliases,
        )
        resolved_calls: List[Tuple[str, List[str], Tuple[int, int]]] = []
        for function_name, template_arguments, span in calls:
            resolved_arguments = [
                self._resolve_type_aliases_at(
                    argument,
                    local_type_aliases,
                    span[0],
                )
                for argument in template_arguments
            ]
            local_constants = self._local_integral_constants_at(
                local_integral_constants,
                span[0],
            )
            if local_constants:
                resolved_arguments = [
                    self._substitute_template_argument_static_constants(
                        argument,
                        local_constants,
                    )
                    for argument in resolved_arguments
                ]
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
        local_integral_constants = self._collect_local_integral_constant_bindings(
            code,
            local_binding_owner_spans,
            local_type_aliases,
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
                template_arguments = self._split_top_level_commas(
                    code[j + 1 : angle_end]
                )
                template_arguments = [
                    self._resolve_type_aliases_at(
                        argument,
                        local_type_aliases,
                        i,
                    )
                    for argument in template_arguments
                ]
                visible_local_constants = self._local_integral_constant_bindings_at(
                    local_integral_constants,
                    i,
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
                unresolved_local_constants = sorted(
                    name
                    for name, binding in visible_local_constants.items()
                    if binding.value is None
                    and any(
                        self._substitute_template_argument_static_constants(
                            argument,
                            {name: "0"},
                        )
                        != argument
                        for argument in template_arguments
                    )
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
                primary_template = struct_templates_by_name[ident]
                resolved_arguments = (
                    self._template_arguments_with_resolved_defaults(
                        primary_template,
                        template_arguments,
                    )
                    or template_arguments
                )
                key = self._struct_specialization_comparison_key(
                    ident,
                    resolved_arguments,
                )
                if key in explicit_specialization_keys:
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
                    if not self._template_arguments_may_match_partial_struct_specialization(
                        partial_template,
                        specialization_arguments,
                        resolved_arguments,
                    ):
                        continue
                    bindings = self._partial_struct_specialization_bindings(
                        partial_template,
                        specialization_arguments,
                        resolved_arguments,
                    )
                    if bindings is None:
                        has_unmaterializable_partial = True
                    else:
                        matching_partials.append((partial_template, bindings))
                if matching_partials or has_unmaterializable_partial:
                    qualified_start = angle_end + 1
                    while (
                        qualified_start < len(code) and code[qualified_start].isspace()
                    ):
                        qualified_start += 1
                    if (
                        not code.startswith("::", qualified_start)
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
                    discovered_materialized_structs.setdefault(ident, {})[
                        specialization_key[1]
                    ] = self._template_specialization_identifier(
                        ident, list(specialization_key[1])
                    )
                i = angle_end + 1
                continue
            i += 1
        return instantiations, selected_partial_specializations

    def _replace_discovered_struct_template_references(
        self,
        argument: str,
        materialized_structs: Dict[str, Dict[Tuple[str, ...], str]],
    ) -> str:
        replacements: List[Tuple[int, int, str]] = []
        i = 0
        while i < len(argument):
            if argument[i] in "\"'":
                _literal, consumed = self._read_string(argument, i)
                i += consumed
                continue
            if argument.startswith("//", i):
                break
            if argument.startswith("/*", i):
                end = argument.find("*/", i + 2)
                if end == -1:
                    break
                i = end + 2
                continue
            if not (argument[i].isalpha() or argument[i] == "_"):
                i += 1
                continue
            ident, consumed = self._read_identifier(argument, i)
            known_specializations = materialized_structs.get(ident)
            if known_specializations is None:
                i += consumed
                continue
            angle_start = i + consumed
            while angle_start < len(argument) and argument[angle_start].isspace():
                angle_start += 1
            if angle_start >= len(argument) or argument[angle_start] != "<":
                i += consumed
                continue
            angle_end = self._find_matching_angle(argument, angle_start)
            if angle_end is None:
                i += consumed
                continue
            template_arguments = self._split_top_level_commas(
                argument[angle_start + 1 : angle_end]
            )
            key = self._template_specialization_key(ident, template_arguments)
            specialized_name = known_specializations.get(key[1])
            if specialized_name is None:
                i += consumed
                continue
            replacements.append(
                (
                    self._scoped_identifier_start(argument, i),
                    angle_end + 1,
                    specialized_name,
                )
            )
            i = angle_end + 1
        if not replacements:
            return argument
        return self._apply_text_replacements(argument, replacements)

    def _substitute_template_argument_static_constants(
        self, argument: str, constants: Dict[str, str]
    ) -> str:
        result: List[str] = []
        i = 0
        while i < len(argument):
            if argument[i] in "\"'":
                literal, consumed = self._read_string(argument, i)
                result.append(literal)
                i += consumed
                continue
            if argument.startswith("//", i):
                result.append(argument[i:])
                break
            if argument.startswith("/*", i):
                end = argument.find("*/", i + 2)
                if end == -1:
                    result.append(argument[i:])
                    break
                result.append(argument[i : end + 2])
                i = end + 2
                continue
            if argument[i].isalpha() or argument[i] == "_":
                identifier, consumed = self._read_identifier(argument, i)
                replacement = constants.get(identifier)
                previous = i - 1
                while previous >= 0 and argument[previous].isspace():
                    previous -= 1
                scoped = previous >= 0 and argument[previous] == "."
                if previous >= 0 and argument[previous] == ":":
                    qualifier_colon = previous - 1
                    while qualifier_colon >= 0 and argument[qualifier_colon].isspace():
                        qualifier_colon -= 1
                    scoped = qualifier_colon >= 0 and argument[qualifier_colon] == ":"
                following = i + consumed
                while following < len(argument) and argument[following].isspace():
                    following += 1
                if (
                    replacement is not None
                    and not scoped
                    and (following >= len(argument) or argument[following] != "(")
                ):
                    result.append(self._static_initializer_reference(replacement))
                else:
                    result.append(identifier)
                i += consumed
                continue
            result.append(argument[i])
            i += 1
        return "".join(result).strip()

    def _template_struct_specialization_arguments(
        self, template: _MetalTemplateStruct
    ) -> Optional[List[str]]:
        match = re.match(
            r"\s*(?:\[\[[^\]]*\]\]\s*)*(?:struct|class)\s+"
            + re.escape(template.name)
            + r"\s*<",
            template.source,
            re.DOTALL,
        )
        if match is None:
            return None
        angle_start = match.end() - 1
        angle_end = self._find_matching_angle(template.source, angle_start)
        if angle_end is None:
            return []
        return self._split_top_level_commas(
            template.source[angle_start + 1 : angle_end]
        )

    def _template_arguments_may_match_partial_struct_specialization(
        self,
        template: _MetalTemplateStruct,
        specialization_arguments: List[str],
        concrete_arguments: List[str],
    ) -> bool:
        if len(specialization_arguments) != len(concrete_arguments):
            return False
        parameter_names = set(template.template_parameters)
        if template.variadic_template_parameters:
            return self._template_trait_variadic_specialization_may_match(
                tuple(specialization_arguments),
                concrete_arguments,
                parameter_names,
                template.variadic_template_parameters,
            )
        return (
            self._partial_struct_specialization_bindings(
                template,
                specialization_arguments,
                concrete_arguments,
            )
            is not None
        )

    @staticmethod
    def _equivalent_template_boolean_literals(left: str, right: str) -> bool:
        def value(text: str) -> Optional[bool]:
            normalized = text.strip()
            if normalized == "true":
                return True
            if normalized == "false":
                return False
            if re.fullmatch(r"\+?1(?:[uUlL]+)?", normalized):
                return True
            if re.fullmatch(r"[+-]?0(?:[uUlL]+)?", normalized):
                return False
            return None

        left_value = value(left)
        right_value = value(right)
        return (
            left_value is not None
            and right_value is not None
            and left_value == right_value
        )

    def _partial_struct_specialization_bindings(
        self,
        template: _MetalTemplateStruct,
        specialization_arguments: List[str],
        concrete_arguments: List[str],
    ) -> Optional[Dict[str, str]]:
        if (
            len(specialization_arguments) != len(concrete_arguments)
            or template.variadic_template_parameters
        ):
            return None
        bindings: Dict[str, str] = {}
        parameter_names = set(template.template_parameters)
        for specialization_argument, concrete_argument in zip(
            specialization_arguments,
            concrete_arguments,
        ):
            normalized_pattern = self._normalize_template_argument_text(
                specialization_argument
            )
            normalized_argument = self._normalize_template_argument_text(
                concrete_argument
            )
            normalized_pattern = self._normalize_struct_specialization_argument(
                normalized_pattern
            )
            normalized_argument = self._normalize_struct_specialization_argument(
                normalized_argument
            )
            if normalized_pattern != normalized_argument and not (
                self._equivalent_template_boolean_literals(
                    normalized_pattern,
                    normalized_argument,
                )
            ):
                self._infer_template_parameter_bindings_from_type(
                    normalized_pattern,
                    normalized_argument,
                    parameter_names,
                    bindings,
                )
            resolved_pattern = self._normalize_template_argument_text(
                self._replace_identifiers(normalized_pattern, bindings)
            )
            if resolved_pattern != normalized_argument and not (
                self._equivalent_template_boolean_literals(
                    resolved_pattern,
                    normalized_argument,
                )
            ):
                return None
        if not parameter_names <= set(bindings):
            return None
        return bindings

    def _qualified_template_struct_alias_at(
        self,
        code: str,
        struct_span: Tuple[int, int],
    ) -> Optional[Tuple[str, Tuple[int, int]]]:
        cursor = struct_span[1]
        while cursor < len(code) and code[cursor].isspace():
            cursor += 1
        if not code.startswith("::", cursor):
            return None
        cursor += 2
        while cursor < len(code) and code[cursor].isspace():
            cursor += 1
        alias_match = IDENTIFIER_RE.match(code, cursor)
        if alias_match is None:
            return None

        replacement_start = struct_span[0]
        typename_match = re.search(r"\btypename\s*$", code[:replacement_start])
        if typename_match is not None:
            replacement_start = typename_match.start()
        return alias_match.group(0), (replacement_start, alias_match.end())

    def _concrete_template_struct_scalar_alias_target(
        self,
        template: _MetalTemplateStruct,
        bindings: Dict[str, str],
        alias_name: str,
    ) -> Optional[str]:
        body_start = template.source.find("{")
        if body_start == -1:
            return None
        body_end = self._find_matching_brace(template.source, body_start)
        if body_end is None:
            return None
        aliases = self._collect_struct_scope_type_aliases(
            template.source[body_start + 1 : body_end - 1]
        )
        if alias_name not in aliases:
            return None

        resolved = dict(bindings)
        for _ in range(len(aliases) + 1):
            previous = dict(resolved)
            for name, target in aliases.items():
                resolved[name] = self._normalize_template_argument_text(
                    self._replace_identifiers(target, resolved)
                )
            if resolved == previous:
                break
        target = re.sub(r"^typename\s+", "", resolved[alias_name]).strip()
        target = re.sub(r"^metal::", "", target)
        if target not in self._METAL_SCALAR_VECTOR_TYPES:
            return None
        return target

    def _find_explicit_struct_specialization_keys(
        self,
        code: str,
        primary_templates_by_name: Dict[str, _MetalTemplateStruct],
    ) -> Set[Tuple[str, Tuple[str, ...]]]:
        # Full explicit struct specializations suppress the primary template only
        # for their exact argument list. A name-wide suppression incorrectly blocks
        # unrelated instances such as `BlockMMA<float, ...>` when the source also
        # defines a `BlockMMA<complex64_t, ...>` specialization.
        keys: Set[Tuple[str, Tuple[str, ...]]] = set()
        for match in re.finditer(
            r"\btemplate\s*<\s*>\s*(?:struct|class)\s+"
            r"(?P<name>[A-Za-z_][A-Za-z0-9_:]*)\s*<",
            code,
        ):
            angle_start = match.end() - 1
            angle_end = self._find_matching_angle(code, angle_start)
            if angle_end is None:
                continue
            name = match.group("name").split("::")[-1]
            arguments = self._split_top_level_commas(code[angle_start + 1 : angle_end])
            primary_template = primary_templates_by_name.get(name)
            if primary_template is not None:
                arguments = (
                    self._template_arguments_with_resolved_defaults(
                        primary_template,
                        arguments,
                    )
                    or arguments
                )
            keys.add(self._struct_specialization_comparison_key(name, arguments))
        return keys

    def _struct_specialization_comparison_key(
        self, struct_name: str, template_arguments: List[str]
    ) -> Tuple[str, Tuple[str, ...]]:
        return (
            struct_name,
            tuple(
                self._normalize_struct_specialization_argument(argument)
                for argument in template_arguments
            ),
        )

    def _normalize_struct_specialization_argument(self, argument: str) -> str:
        text = self._normalize_template_argument_text(argument)
        if not text:
            return ""

        pointer_start = len(text)
        angle_depth = paren_depth = bracket_depth = 0
        for index, char in enumerate(text):
            if char == "<":
                angle_depth += 1
            elif char == ">":
                angle_depth = max(0, angle_depth - 1)
            elif char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth = max(0, paren_depth - 1)
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif char in "*&" and angle_depth == paren_depth == bracket_depth == 0:
                pointer_start = index
                break

        prefix = text[:pointer_start]
        suffix = text[pointer_start:]
        qualifier_spans: List[Tuple[int, int]] = []
        angle_depth = paren_depth = bracket_depth = 0
        index = 0
        while index < len(prefix):
            char = prefix[index]
            if char == "<":
                angle_depth += 1
            elif char == ">":
                angle_depth = max(0, angle_depth - 1)
            elif char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth = max(0, paren_depth - 1)
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif (
                char.isalpha() or char == "_"
            ) and angle_depth == paren_depth == bracket_depth == 0:
                identifier, consumed = self._read_identifier(prefix, index)
                if identifier in {"const", "volatile"}:
                    qualifier_spans.append((index, index + consumed))
                index += consumed
                continue
            index += 1
        if not qualifier_spans:
            return text

        qualifiers = {prefix[start:end] for start, end in qualifier_spans}
        base = self._apply_text_replacements(
            prefix,
            [(start, end, "") for start, end in qualifier_spans],
        )
        base = re.sub(r"\s+", " ", base).strip()
        ordered_qualifiers = [
            qualifier for qualifier in ("const", "volatile") if qualifier in qualifiers
        ]
        return self._normalize_template_argument_text(
            " ".join([*ordered_qualifiers, base]) + suffix
        )

    def _template_arguments_with_resolved_defaults(
        self,
        template: _MetalTemplateStruct,
        template_arguments: List[str],
    ) -> Optional[List[str]]:
        if not self._template_arguments_satisfy_parameters(
            template,
            template_arguments,
        ):
            return None
        substitutions, variadic_bindings = self._template_argument_bindings(
            template,
            template_arguments,
        )
        resolved: List[str] = []
        for parameter in template.template_parameters:
            if parameter in template.variadic_template_parameters:
                resolved.extend(variadic_bindings.get(parameter, []))
                continue
            argument = substitutions.get(parameter)
            if argument is None:
                return None
            resolved.append(argument)
        return resolved

    def _dedupe_explicit_template_function_calls(
        self,
        calls: List[Tuple[str, List[str], Tuple[int, int]]],
    ) -> List[Tuple[str, List[str], List[Tuple[int, int]]]]:
        grouped: Dict[
            Tuple[str, Tuple[str, ...]],
            Tuple[str, List[str], List[Tuple[int, int]]],
        ] = {}
        for function_name, template_arguments, span in calls:
            key = self._template_specialization_key(function_name, template_arguments)
            grouped_call = grouped.get(key)
            if grouped_call is None:
                grouped[key] = (function_name, template_arguments, [span])
            else:
                grouped_call[2].append(span)
        return list(grouped.values())

    def _host_name_from_attributes(self, declaration: str) -> str:
        match = re.search(
            r"\[\[\s*host_name\s*\(\s*(?P<host>"
            + METAL_STRING_EXPRESSION_PATTERN
            + r")\s*\)\s*\]\]",
            declaration,
            re.DOTALL,
        )
        if match is None:
            return ""
        return self._evaluate_metal_string_expression(match.group("host"))

    def _template_id_candidates(self, declaration: str) -> List[Tuple[str, List[str]]]:
        candidates: List[Tuple[str, List[str]]] = []
        i = 0
        while i < len(declaration):
            if declaration[i] in "\"'":
                _literal, consumed = self._read_string(declaration, i)
                i += consumed
                continue
            if declaration.startswith("//", i):
                end = declaration.find("\n", i)
                if end == -1:
                    break
                i = end + 1
                continue
            if declaration.startswith("/*", i):
                end = declaration.find("*/", i + 2)
                if end == -1:
                    break
                i = end + 2
                continue
            if declaration[i].isalpha() or declaration[i] == "_":
                ident, consumed = self._read_identifier(declaration, i)
                scoped_start = self._scoped_identifier_start(declaration, i)
                name = declaration[scoped_start : i + consumed]
                j = i + consumed
                while j < len(declaration) and declaration[j].isspace():
                    j += 1
                if j < len(declaration) and declaration[j] == "<":
                    angle_end = self._find_matching_angle(declaration, j)
                    if angle_end is not None:
                        arguments = self._split_top_level_commas(
                            declaration[j + 1 : angle_end]
                        )
                        if ident not in {"decltype", "static_cast", "as_type"}:
                            candidates.append((name, arguments))
                        i = angle_end + 1
                        continue
                i += consumed
                continue
            i += 1
        return candidates

    def _declared_function_name(self, declaration: str) -> Optional[str]:
        function_name = self._function_name_from_header(declaration)
        if function_name in {"decltype", "static_cast", "as_type"}:
            return None
        return function_name

    def _declared_template_id_candidate(
        self,
        declaration: str,
        declared_function_name: Optional[str],
    ) -> Optional[Tuple[str, List[str]]]:
        candidates = [
            (name, arguments)
            for name, arguments in self._template_id_candidates(declaration)
            if arguments
        ]
        if not candidates:
            return None
        if declared_function_name is None:
            return candidates[0]
        for name, arguments in candidates:
            if name.split("::")[-1] == declared_function_name:
                return name, arguments
        return None

    def _infer_declared_template_arguments(
        self,
        template: _MetalTemplateFunction,
        declaration: str,
    ) -> List[str]:
        template_parameters = list(template.template_parameters)
        if not template_parameters:
            return []
        template_parameter_set = set(template_parameters)
        template_param_types = self._function_parameter_type_texts(template.source)
        declaration_param_types = self._function_parameter_type_texts(declaration)
        if not template_param_types or not declaration_param_types:
            return []

        bindings: Dict[str, str] = {}
        for template_type, concrete_type in zip(
            template_param_types,
            declaration_param_types,
        ):
            self._infer_template_parameter_bindings_from_type(
                template_type,
                concrete_type,
                template_parameter_set,
                bindings,
            )
        if not bindings:
            return []

        arguments: List[str] = []
        defaults = getattr(template, "template_parameter_defaults", {}) or {}
        substitutions: Dict[str, str] = {}
        last_inferred_index = -1
        for index, name in enumerate(template_parameters):
            if name in bindings:
                argument = bindings[name]
                last_inferred_index = index
            else:
                default_argument = defaults.get(name)
                if default_argument is None:
                    return []
                argument = self._resolve_template_default_argument(
                    default_argument,
                    substitutions,
                    template,
                )
            substitutions[name] = argument
            arguments.append(argument)

        if last_inferred_index == -1:
            return []
        return arguments[: last_inferred_index + 1]

    def _function_parameter_type_texts(self, declaration: str) -> List[str]:
        open_paren = self._function_parameter_start(declaration)
        if open_paren is None:
            return []
        close_paren = self._find_matching_delimiter(declaration, open_paren, "(", ")")
        if close_paren is None:
            return []
        parameters = self._split_top_level_commas(
            declaration[open_paren + 1 : close_paren]
        )
        return [
            normalized
            for normalized in (
                self._normalize_function_parameter_type_text(parameter)
                for parameter in parameters
            )
            if normalized and normalized != "void"
        ]

    def _normalize_function_parameter_type_text(self, parameter: str) -> str:
        parameter = self._strip_top_level_default_value(parameter)
        attributes = " ".join(self._metal_attributes(parameter))
        type_text = self._strip_metal_attributes(parameter)
        type_text = re.sub(
            r"\s+\b[A-Za-z_][A-Za-z0-9_]*\s*$",
            "",
            type_text.strip(),
        )
        normalized = self._normalize_template_argument_text(type_text)
        if attributes:
            normalized_attributes = self._normalize_template_argument_text(attributes)
            if normalized:
                return f"{normalized} {normalized_attributes}"
            return normalized_attributes
        return normalized

    def _function_parameter_value_type(self, parameter: str) -> str:
        normalized = self._normalize_inferred_type(
            self._normalize_function_parameter_type_text(parameter)
        )
        return normalized.rstrip("&").strip()

    def _metal_attributes(self, text: str) -> List[str]:
        attributes: List[str] = []
        i = 0
        while i < len(text):
            if text.startswith("[[", i):
                end = text.find("]]", i + 2)
                if end == -1:
                    break
                attributes.append(text[i : end + 2])
                i = end + 2
                continue
            i += 1
        return attributes

    def _strip_metal_attributes(self, text: str) -> str:
        result = ""
        i = 0
        while i < len(text):
            if text.startswith("[[", i):
                end = text.find("]]", i + 2)
                if end == -1:
                    break
                i = end + 2
                continue
            result += text[i]
            i += 1
        return result

    def _strip_top_level_default_value(self, text: str) -> str:
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        angle_depth = 0
        i = 0
        while i < len(text):
            if text[i] in "\"'":
                _literal, consumed = self._read_string(text, i)
                i += consumed
                continue
            if text.startswith("//", i):
                break
            if text.startswith("/*", i):
                end = text.find("*/", i + 2)
                if end == -1:
                    break
                i = end + 2
                continue

            ch = text[i]
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)
            elif ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth = max(0, brace_depth - 1)
            elif ch == "<":
                angle_depth += 1
            elif ch == ">":
                angle_depth = max(0, angle_depth - 1)
            elif (
                ch == "="
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
                and angle_depth == 0
            ):
                return text[:i].strip()
            i += 1
        return text.strip()

    def _infer_template_parameter_bindings_from_type(
        self,
        template_type: str,
        concrete_type: str,
        template_parameters: Set[str],
        bindings: Dict[str, str],
    ) -> None:
        specialization = self._materialized_struct_specializations.get(
            self._normalize_inferred_type(concrete_type)
        )
        if specialization is not None:
            source_name, source_arguments = specialization
            if re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(source_name)}\s*<", template_type
            ):
                concrete_type = f"{source_name}<{', '.join(source_arguments)}>"
        captures: List[str] = []
        pattern_parts: List[str] = []
        position = 0
        for match in IDENTIFIER_RE.finditer(template_type):
            pattern_parts.append(re.escape(template_type[position : match.start()]))
            identifier = match.group(0)
            if identifier in template_parameters:
                pattern_parts.append("(.+?)")
                captures.append(identifier)
            else:
                pattern_parts.append(re.escape(identifier))
            position = match.end()
        pattern_parts.append(re.escape(template_type[position:]))
        if not captures:
            return

        match = re.fullmatch("".join(pattern_parts), concrete_type)
        if match is None:
            return
        inferred: Dict[str, str] = {}
        for index, parameter in enumerate(captures, start=1):
            value = self._normalize_template_argument_text(match.group(index))
            if not value:
                return
            existing = inferred.get(parameter, bindings.get(parameter))
            if existing is not None and existing != value:
                return
            inferred[parameter] = value
        bindings.update(inferred)

    def _reachable_function_spans(
        self,
        code: str,
        excluded_spans: List[Tuple[int, int]],
    ) -> Optional[List[Tuple[int, int]]]:
        functions = self._find_non_template_function_definitions(code, excluded_spans)
        if not functions:
            return None

        roots = {function.name for function in functions if function.is_entry}
        if not roots:
            return None

        by_name: Dict[str, List[_MetalFunctionDefinition]] = {}
        for function in functions:
            by_name.setdefault(function.name, []).append(function)

        known_names = set(by_name)
        reachable = set(roots)
        pending = list(roots)
        while pending:
            name = pending.pop()
            for function in by_name.get(name, ()):
                body_start, body_end = function.body_span
                for referenced in self._find_function_references(
                    code[body_start:body_end],
                    known_names,
                ):
                    if referenced not in reachable:
                        reachable.add(referenced)
                        pending.append(referenced)

        return [function.span for function in functions if function.name in reachable]

    def _find_non_template_function_definitions(
        self,
        code: str,
        excluded_spans: List[Tuple[int, int]],
    ) -> List[_MetalFunctionDefinition]:
        functions: List[_MetalFunctionDefinition] = []
        pos = 0
        while True:
            body_start = self._find_next_top_level_char(code, pos, "{")
            if body_start is None:
                break
            excluded = self._containing_span(body_start, excluded_spans)
            if excluded is not None:
                pos = excluded[1]
                continue
            body_end = self._find_matching_brace(code, body_start)
            if body_end is None:
                break

            declaration_start = self._function_declaration_start(code, body_start)
            header = code[declaration_start:body_start]
            function_name = self._function_name_from_header(header)
            if function_name is not None:
                functions.append(
                    _MetalFunctionDefinition(
                        name=function_name,
                        span=(declaration_start, body_end),
                        body_span=(body_start + 1, body_end - 1),
                        is_entry=METAL_ENTRY_FUNCTION_RE.search(header) is not None,
                    )
                )
            pos = body_end
        return functions

    def _function_declaration_start(self, code: str, body_start: int) -> int:
        previous_semicolon = code.rfind(";", 0, body_start)
        previous_block = code.rfind("}", 0, body_start)
        return max(previous_semicolon, previous_block) + 1

    def _find_function_references(
        self, code: str, function_names: Set[str]
    ) -> Set[str]:
        references: Set[str] = set()
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
            if code[i].isalpha() or code[i] == "_":
                ident, consumed = self._read_identifier(code, i)
                j = i + consumed
                while j < len(code) and code[j].isspace():
                    j += 1
                if ident in function_names and j < len(code) and code[j] == "(":
                    references.add(ident)
                i += consumed
                continue
            i += 1
        return references

    def _find_template_declaration_spans(self, code: str) -> List[Tuple[int, int]]:
        analysis = self._source_analysis(code)
        if analysis.template_declaration_spans is None:
            analysis.template_declaration_spans = self._scan_template_declaration_spans(
                code
            )
        return analysis.template_declaration_spans

    def _scan_template_declaration_spans(self, code: str) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        pos = 0
        while True:
            match = re.search(r"\btemplate\s*<", code[pos:])
            if match is None:
                break
            start = pos + match.start()
            angle_start = code.find("<", start)
            angle_end = self._find_matching_template_param_angle(code, angle_start)
            if angle_end is None:
                pos = start + len("template")
                continue

            declaration_start = angle_end + 1
            body_start = self._find_next_top_level_char(code, declaration_start, "{")
            semicolon = self._find_next_top_level_char(code, declaration_start, ";")
            if body_start is not None and (semicolon is None or body_start < semicolon):
                body_end = self._find_matching_brace(code, body_start)
                if body_end is None:
                    pos = body_start + 1
                    continue
                spans.append((start, body_end))
                pos = body_end
                continue
            if semicolon is not None:
                spans.append((start, semicolon + 1))
                pos = semicolon + 1
                continue
            pos = declaration_start
        return spans

    def _find_explicit_template_specialization_keys(
        self, code: str
    ) -> Set[Tuple[str, Tuple[str, ...]]]:
        keys: Set[Tuple[str, Tuple[str, ...]]] = set()
        for match in re.finditer(r"\btemplate\s*<\s*>\s*", code):
            declaration_start = match.end()
            body_start = self._find_next_top_level_char(code, declaration_start, "{")
            semicolon = self._find_next_top_level_char(code, declaration_start, ";")
            declaration_end = body_start
            if declaration_end is None or (
                semicolon is not None and semicolon < declaration_end
            ):
                declaration_end = semicolon
            if declaration_end is None:
                continue
            header = code[declaration_start:declaration_end]
            paren_index = header.find("(")
            if paren_index == -1:
                continue
            before_params = header[:paren_index].rstrip()
            angle_end = before_params.rfind(">")
            if angle_end == -1:
                continue
            angle_start = before_params.rfind("<", 0, angle_end)
            if angle_start == -1:
                continue
            name_match = re.search(
                r"([A-Za-z_][A-Za-z0-9_:]*)\s*$",
                before_params[:angle_start],
            )
            if name_match is None:
                continue
            function_name = name_match.group(1).split("::")[-1]
            args = self._split_top_level_commas(
                before_params[angle_start + 1 : angle_end]
            )
            keys.add(self._template_specialization_key(function_name, args))
        return keys

    def _containing_span(
        self, position: int, spans: List[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        if not spans:
            return None

        first = spans[0]
        last = spans[-1]
        cache = self._containing_span_cache
        key = id(spans)
        cached = cache.get(key)
        if (
            cached is None
            or cached[0] is not spans
            or cached[1] != len(spans)
            or cached[2] != first
            or cached[3] != last
        ):
            is_sorted_non_overlapping = all(
                spans[index - 1][0] <= spans[index][0]
                and spans[index - 1][1] <= spans[index][0]
                for index in range(1, len(spans))
            )
            starts = (
                [start for start, _end in spans] if is_sorted_non_overlapping else None
            )
            cached = (spans, len(spans), first, last, starts)
            cache[key] = cached
            if len(cache) > CONTAINING_SPAN_CACHE_LIMIT:
                del cache[next(iter(cache))]

        starts = cached[4]
        if starts is not None:
            index = bisect_right(starts, position) - 1
            if index >= 0:
                start, end = spans[index]
                if position < end:
                    return start, end
            return None

        for start, end in spans:
            if start <= position < end:
                return start, end
        return None

    def _scoped_identifier_start(self, code: str, identifier_start: int) -> int:
        start = identifier_start
        while start >= 2 and code[start - 2 : start] == "::":
            name_end = start - 2
            name_start = name_end
            while name_start > 0 and (
                code[name_start - 1].isalnum() or code[name_start - 1] == "_"
            ):
                name_start -= 1
            if name_start == name_end:
                start -= 2
                break
            start = name_start
        return start

    def _template_specialization_identifier(
        self, function_name: str, template_arguments: List[str]
    ) -> str:
        parts = [function_name, *template_arguments]
        identifier = "_".join(
            part
            for part in (
                re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_") for value in parts
            )
            if part
        )
        if not identifier:
            identifier = function_name
        if identifier[0].isdigit():
            identifier = f"{function_name}_{identifier}"
        return identifier

    def _template_specialization_key(
        self, function_name: str, template_arguments: List[str]
    ) -> Tuple[str, Tuple[str, ...]]:
        return (
            function_name,
            tuple(
                self._normalize_template_argument_text(argument)
                for argument in template_arguments
            ),
        )

    def _template_specialization_signature(
        self, function_name: str, template_arguments: List[str]
    ) -> str:
        return (
            f"{function_name}<"
            + ", ".join(
                self._normalize_template_argument_text(argument)
                for argument in template_arguments
            )
            + ">"
        )

    def _normalize_template_argument_text(self, value: str) -> str:
        value = self._strip_template_argument_comments(value).strip()
        if not value:
            return ""
        collapsed = re.sub(r"\s+", " ", value)
        return re.sub(r"\s*([<>,:*&\[\](){}])\s*", r"\1", collapsed).strip()

    def _strip_template_argument_comments(self, value: str) -> str:
        result = ""
        i = 0
        while i < len(value):
            if value[i] in "\"'":
                literal, consumed = self._read_string(value, i)
                result += literal
                i += consumed
                continue
            if value.startswith("//", i):
                end = value.find("\n", i)
                if end == -1:
                    break
                i = end + 1
                continue
            if value.startswith("/*", i):
                end = value.find("*/", i + 2)
                if end == -1:
                    break
                i = end + 2
                continue
            result += value[i]
            i += 1
        return result

    # ------------------------------------------------------------------ #
    # `<` disambiguation for template-PARAMETER lists.                    #
    # ------------------------------------------------------------------ #
    # A template-parameter list may contain a SFINAE non-type parameter whose
    # constraint embeds comparison operators, e.g.
    #   template <typename T, metal::enable_if_t<sizeof(T) < 8, bool> = true>
    # The generic `_find_matching_angle` / `_split_top_level_commas` treat the
    # comparison `<` in `sizeof(T) < 8` as a template-open angle, so the angle
    # nesting never balances and the parameter list (and the method it heads) is
    # dropped. The helpers below scan a template-parameter list treating a `<`
    # as a template-argument opener ONLY when it immediately follows a
    # template-name identifier character (e.g. `enable_if_t<`, `is_integral_v<`);
    # a `<`/`>` used as a comparison (`sizeof(T) < 8`, `<=`, `>=`) does NOT change
    # angle depth. These are deliberately SEPARATE from the template-ARGUMENT
    # handling so other `<` parsing is left untouched.

    def _find_matching_template_param_angle(
        self, code: str, start: int
    ) -> Optional[int]:
        # Index of the `>` that closes the template-parameter-list `<` at `start`,
        # ignoring comparison `<`/`>` inside SFINAE constraints. Returns None when
        # unbalanced.
        if start < 0 or start >= len(code) or code[start] != "<":
            return None
        # The `<` at `start` is the parameter-list opener by construction (it is
        # the `<` of `template <`), so it always counts as depth 1 regardless of
        # what precedes it; the open-disambiguation only applies to later `<`.
        depth = 1
        i = start + 1
        n = len(code)
        while i < n:
            ch = code[i]
            if ch in "\"'":
                _literal, consumed = self._read_string(code, i)
                i += consumed
                continue
            if code.startswith("//", i):
                end = code.find("\n", i)
                if end == -1:
                    return None
                i = end + 1
                continue
            if code.startswith("/*", i):
                end = code.find("*/", i + 2)
                if end == -1:
                    return None
                i = end + 2
                continue
            if ch == "<":
                if code[i + 1 : i + 2] == "=":
                    # `<=` comparison — never an angle; skip both chars.
                    i += 2
                    continue
                if self._template_param_angle_is_open(code, i):
                    depth += 1
                i += 1
                continue
            if ch == ">":
                if code[i + 1 : i + 2] == "=":
                    # `>=` comparison — never an angle; skip both chars.
                    i += 2
                    continue
                # `->` is a member access, not an angle close.
                if i > 0 and code[i - 1] == "-":
                    i += 1
                    continue
                if depth > 0:
                    depth -= 1
                    if depth == 0:
                        return i
                i += 1
                continue
            i += 1
        return None

    def _template_param_angle_is_open(self, code: str, index: int) -> bool:
        # A `<` opens a template-argument list (rather than being a comparison)
        # when the preceding significant token ends in an identifier. Comments
        # and whitespace do not change C++ token adjacency, so skip both here.
        # Comparisons such as `sizeof(T) < 8` remain distinguishable because the
        # preceding token ends in `)`.
        previous = index - 1
        skipped_trivia = False
        while previous >= 0:
            while previous >= 0 and code[previous].isspace():
                previous -= 1
                skipped_trivia = True
            if previous < 0:
                return False
            if previous >= 1 and code[previous - 1 : previous + 1] == "*/":
                comment_start = code.rfind("/*", 0, previous - 1)
                if comment_start == -1:
                    return False
                previous = comment_start - 1
                skipped_trivia = True
                continue
            line_start = code.rfind("\n", 0, previous + 1) + 1
            comment_start = code.find("//", line_start, previous + 1)
            if comment_start != -1:
                previous = comment_start - 1
                skipped_trivia = True
                continue
            break
        prev = code[previous]
        if not (prev.isalnum() or prev == "_"):
            return False
        if not skipped_trivia:
            return True

        parameter_start = max(
            code.rfind(",", 0, previous + 1),
            code.rfind("<", 0, previous + 1),
        )
        parameter_prefix = code[parameter_start + 1 : previous + 1]
        parameter_prefix = re.sub(
            r"/\*.*?\*/|//[^\n]*",
            " ",
            parameter_prefix,
            flags=re.DOTALL,
        )
        return (
            re.match(
                r"\s*(?:typename|class)\s+" r"[A-Za-z_][A-Za-z0-9_]*(?:\s*\.\.\.)?\s*=",
                parameter_prefix,
            )
            is not None
        )

    def _split_template_parameter_list(self, text: str) -> List[str]:
        # Top-level comma split of a template-parameter list that respects the
        # SFINAE `<` disambiguation above (so `enable_if_t<sizeof(T) < 8, bool>`
        # stays one parameter while the comma separating two parameters splits).
        parts: List[str] = []
        current = ""
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        angle_depth = 0
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch in "\"'":
                literal, consumed = self._read_string(text, i)
                current += literal
                i += consumed
                continue
            if text.startswith("//", i):
                end = text.find("\n", i + 2)
                if end == -1:
                    current += text[i:]
                    break
                current += text[i : end + 1]
                i = end + 1
                continue
            if text.startswith("/*", i):
                end = text.find("*/", i + 2)
                if end == -1:
                    current += text[i:]
                    break
                current += text[i : end + 2]
                i = end + 2
                continue
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)
            elif ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth = max(0, brace_depth - 1)
            elif ch == "<":
                if text[i + 1 : i + 2] == "=":
                    current += "<="
                    i += 2
                    continue
                if self._template_param_angle_is_open(text, i):
                    angle_depth += 1
            elif ch == ">":
                if text[i + 1 : i + 2] == "=":
                    current += ">="
                    i += 2
                    continue
                if not (i > 0 and text[i - 1] == "-") and angle_depth > 0:
                    angle_depth -= 1
            elif (
                ch == ","
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
                and angle_depth == 0
            ):
                parts.append(current.strip())
                current = ""
                i += 1
                continue
            current += ch
            i += 1
        if current.strip():
            parts.append(current.strip())
        return parts

    def _parse_template_parameter_list(self, text: str) -> List[_TemplateParameter]:
        # Parse a template-parameter list into structured records, classifying
        # each entry as a bindable type parameter or an anonymous SFINAE
        # constraint. A non-type parameter that DOES name a bindable value (e.g.
        # `int N`) is recorded as a bindable parameter too (existing behavior for
        # `<typename T, int N>`); only an ANONYMOUS non-type parameter (no
        # trailing declarator name) becomes a constraint.
        records: List[_TemplateParameter] = []
        for raw in self._split_template_parameter_list(text):
            parameter = raw.strip()
            if not parameter:
                continue
            default: Optional[str] = None
            declarator = parameter
            eq = self._template_parameter_default_split(parameter)
            if eq is not None:
                declarator, default = eq
            declarator_without_comments = re.sub(
                r"/\*.*?\*/",
                " ",
                declarator,
                flags=re.DOTALL,
            )
            declarator_without_comments = re.sub(
                r"//[^\n]*", " ", declarator_without_comments
            )
            is_variadic = "..." in declarator_without_comments
            declarator_no_pack = declarator_without_comments.replace("...", " ")
            tokens = IDENTIFIER_RE.findall(declarator_no_pack)
            if not tokens:
                continue
            if tokens[0] in {"typename", "class"}:
                # `typename T` / `class T` (optionally `typename...`). A trailing
                # name is the bindable parameter; a bare `typename` with no name
                # is anonymous and unbindable (skip — nothing to bind).
                if len(tokens) >= 2:
                    records.append(
                        _TemplateParameter(
                            name=tokens[-1],
                            is_type_parameter=True,
                            is_variadic=is_variadic,
                            default=default,
                        )
                    )
                continue
            # A non-type parameter. If its declarator ends in a bindable name
            # (`int N`) keep it as a (non-type) bindable parameter; otherwise it
            # is an anonymous SFINAE constraint (`metal::enable_if_t<...> [= v]`).
            if self._non_type_parameter_has_name(declarator_no_pack):
                name = tokens[-1]
                name_match = re.search(rf"\b{re.escape(name)}\s*$", declarator_no_pack)
                declared_type = ""
                if name_match is not None:
                    declared_type = self._normalize_inferred_type(
                        declarator_no_pack[: name_match.start()]
                    )
                records.append(
                    _TemplateParameter(
                        name=name,
                        is_type_parameter=False,
                        is_variadic=is_variadic,
                        default=default,
                        declared_type=declared_type or None,
                    )
                )
                continue
            records.append(
                _TemplateParameter(
                    name=None,
                    is_type_parameter=False,
                    is_variadic=is_variadic,
                    default=default,
                    constraint_text=declarator.strip(),
                )
            )
        return records

    def _template_parameter_default_split(
        self, parameter: str
    ) -> Optional[Tuple[str, str]]:
        # Split a template parameter on its TOP-LEVEL `=` default separator,
        # ignoring comments, `=` inside angles/parens/brackets, and the
        # `==`/`<=`/`>=`/`!=` comparison operators that appear in SFINAE
        # constraints. Returns
        # (declarator, default) or None when there is no default.
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        angle_depth = 0
        i = 0
        n = len(parameter)
        while i < n:
            if parameter.startswith("//", i):
                newline = parameter.find("\n", i + 2)
                if newline == -1:
                    break
                i = newline + 1
                continue
            if parameter.startswith("/*", i):
                comment_end = parameter.find("*/", i + 2)
                if comment_end == -1:
                    break
                i = comment_end + 2
                continue
            ch = parameter[i]
            if ch in "\"'":
                _literal, consumed = self._read_string(parameter, i)
                i += consumed
                continue
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)
            elif ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth = max(0, brace_depth - 1)
            elif ch == "<":
                if parameter[i + 1 : i + 2] == "=":
                    i += 2
                    continue
                if self._template_param_angle_is_open(parameter, i):
                    angle_depth += 1
            elif ch == ">":
                if parameter[i + 1 : i + 2] == "=":
                    i += 2
                    continue
                if not (i > 0 and parameter[i - 1] == "-") and angle_depth > 0:
                    angle_depth -= 1
            elif ch == "=":
                # Skip comparison operators `==`, `!=`, `<=`, `>=`.
                nxt = parameter[i + 1 : i + 2]
                prv = parameter[i - 1] if i > 0 else ""
                if nxt == "=" or prv in "=!<>":
                    i += 2 if nxt == "=" else 1
                    continue
                if (
                    paren_depth == 0
                    and bracket_depth == 0
                    and brace_depth == 0
                    and angle_depth == 0
                ):
                    return parameter[:i].strip(), parameter[i + 1 :].strip()
            i += 1
        return None

    def _non_type_parameter_has_name(self, declarator: str) -> bool:
        # A non-type template parameter `Type name` ends in a bindable identifier
        # that is NOT part of a trailing template-id. An anonymous SFINAE
        # parameter ends in `>` (the closing angle of `enable_if_t<...>`), so it
        # has no trailing name. Conservative: only a declarator whose final
        # top-level token is a bare identifier (not immediately following a `>`
        # close of its own type) is treated as named.
        text = declarator.strip()
        if not text:
            return False
        if text.endswith(">"):
            return False
        # The last token must be an identifier preceded by type text (so a single
        # token such as `bool` — a lone type with no name — is NOT a named
        # parameter; that is the `enable_if_t<...>` value-type leftover case).
        match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", text)
        if match is None:
            return False
        before = text[: match.start()].strip()
        return bool(before)

    def _template_parameter_names(self, template_parameters: str) -> List[str]:
        # Bindable parameter names only: type parameters (`typename T`) and named
        # non-type parameters (`int N`). Anonymous SFINAE constraint parameters
        # are deliberately EXCLUDED — they are not bound from call arguments.
        return [
            parameter.name
            for parameter in self._parse_template_parameter_list(template_parameters)
            if parameter.name is not None
        ]

    def _template_parameter_constraints(self, template_parameters: str) -> List[str]:
        # SFINAE constraint texts from the anonymous non-type parameters, in
        # source order, for overload selection.
        return [
            parameter.constraint_text
            for parameter in self._parse_template_parameter_list(template_parameters)
            if parameter.is_constraint and parameter.constraint_text
        ]

    def _template_parameter_defaults(self, template_parameters: str) -> Dict[str, str]:
        defaults: Dict[str, str] = {}
        for parameter in self._parse_template_parameter_list(template_parameters):
            if parameter.name is None or parameter.default is None:
                continue
            defaults[parameter.name] = parameter.default
        return defaults

    def _variadic_template_parameter_names(self, template_parameters: str) -> Set[str]:
        return {
            parameter.name
            for parameter in self._parse_template_parameter_list(template_parameters)
            if parameter.is_variadic
            and parameter.is_type_parameter
            and parameter.name is not None
        }

    def _template_arguments_with_defaults(
        self,
        code: str,
        template: _MetalTemplateFunction,
        template_arguments: List[str],
    ) -> List[str]:
        del code
        arguments = list(template_arguments)
        if self._template_arguments_satisfy_parameters(template, arguments):
            return arguments

        defaults = getattr(template, "template_parameter_defaults", {}) or {}
        for parameter in template.template_parameters[len(arguments) :]:
            if parameter in template.variadic_template_parameters:
                break
            default = defaults.get(parameter)
            if default is None:
                break
            arguments.append(default)
        return arguments

    def _template_argument_bindings(
        self,
        template: _MetalTemplateFunction,
        template_arguments: List[str],
    ) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        substitutions: Dict[str, str] = {}
        variadic_bindings: Dict[str, List[str]] = {}
        defaults = getattr(template, "template_parameter_defaults", {}) or {}
        argument_index = 0
        for parameter_index, name in enumerate(template.template_parameters):
            if name in template.variadic_template_parameters:
                remaining_fixed = (
                    len(template.template_parameters) - parameter_index - 1
                )
                variadic_count = max(
                    0,
                    len(template_arguments) - argument_index - remaining_fixed,
                )
                values = template_arguments[
                    argument_index : argument_index + variadic_count
                ]
                variadic_bindings[name] = values
                substitutions[name] = values[0] if values else "void"
                argument_index += variadic_count
                continue
            if argument_index < len(template_arguments):
                substitutions[name] = template_arguments[argument_index]
                argument_index += 1
                continue
            default_argument = defaults.get(name)
            if default_argument is None:
                continue
            substitutions[name] = self._resolve_template_default_argument(
                default_argument,
                substitutions,
                template,
            )
        return substitutions, variadic_bindings

    def _template_arguments_satisfy_parameters(
        self,
        template: _MetalTemplateFunction,
        template_arguments: List[str],
    ) -> bool:
        defaults = getattr(template, "template_parameter_defaults", {}) or {}
        argument_index = 0
        parameter_count = len(template.template_parameters)
        for parameter_index, name in enumerate(template.template_parameters):
            if name in template.variadic_template_parameters:
                remaining_fixed = parameter_count - parameter_index - 1
                remaining_arguments = len(template_arguments) - argument_index
                if remaining_arguments < remaining_fixed:
                    return False
                variadic_count = max(0, remaining_arguments - remaining_fixed)
                argument_index += variadic_count
                continue
            if argument_index < len(template_arguments):
                argument_index += 1
                continue
            if name not in defaults:
                return False
        return True

    def _resolve_template_default_argument(
        self,
        default_argument: str,
        substitutions: Dict[str, str],
        template: _MetalTemplateFunction,
    ) -> str:
        resolved = self._replace_identifiers(str(default_argument), substitutions)
        resolved = self._normalize_template_argument_text(resolved)
        trait_resolved = self._resolve_template_type_trait(
            resolved,
            getattr(template, "template_type_traits", {}) or {},
            getattr(template, "namespace", ""),
        )
        return trait_resolved or resolved

    def _resolve_template_type_trait(
        self,
        type_text: str,
        traits: Dict[str, Dict[str, object]],
        namespace: str = "",
    ) -> Optional[str]:
        text = str(type_text or "").strip()
        match = re.fullmatch(
            r"(?:typename\s+)?(?P<name>(?:::)?[A-Za-z_][A-Za-z0-9_:]*)\s*"
            r"<(?P<args>.*)>\s*::\s*type",
            text,
            re.DOTALL,
        )
        if match is None:
            return None
        raw_trait_name = match.group("name")
        globally_qualified = raw_trait_name.startswith("::")
        trait_name = raw_trait_name.lstrip(":")
        lookup_names = self._namespace_lookup_names(
            trait_name,
            namespace,
            globally_qualified,
        )
        trait = next(
            (traits[name] for name in lookup_names if name in traits),
            None,
        )
        if not trait:
            return None
        arguments = [
            self._normalize_template_argument_text(argument)
            for argument in self._split_top_level_commas(match.group("args"))
        ]
        specializations = trait.get("specializations", {})
        specialized = specializations.get(tuple(arguments))
        if specialized is not None:
            specialized_type = (
                specialized if isinstance(specialized, str) else specialized.get("type")
            )
            if isinstance(specialized_type, str) and specialized_type:
                return specialized_type

        partial_matches: List[Tuple[Tuple[int, int, int], str]] = []
        unresolved_variadic_match = False
        for patterns, specialization in specializations.items():
            if len(patterns) != len(arguments) or tuple(patterns) == tuple(arguments):
                continue
            if not isinstance(specialization, dict):
                continue
            parameters = set(specialization.get("parameters", ()))
            variadic_parameters = set(specialization.get("variadic_parameters", ()))
            specialization_type = specialization.get("type")
            if not parameters or not isinstance(specialization_type, str):
                continue
            if variadic_parameters:
                if self._template_trait_variadic_specialization_may_match(
                    patterns,
                    arguments,
                    parameters,
                    variadic_parameters,
                ):
                    unresolved_variadic_match = True
                continue
            bindings: Dict[str, str] = {}
            matched = True
            for pattern, argument in zip(patterns, arguments):
                normalized_pattern = self._normalize_template_argument_text(pattern)
                if normalized_pattern != argument and not (
                    self._equivalent_template_boolean_literals(
                        normalized_pattern,
                        argument,
                    )
                ):
                    self._infer_template_parameter_bindings_from_type(
                        normalized_pattern,
                        argument,
                        parameters,
                        bindings,
                    )
                resolved_pattern = self._normalize_template_argument_text(
                    self._replace_identifiers(normalized_pattern, bindings)
                )
                if resolved_pattern != argument and not (
                    self._equivalent_template_boolean_literals(
                        resolved_pattern,
                        argument,
                    )
                ):
                    matched = False
                    break
            if matched:
                partial_matches.append(
                    (
                        self._template_trait_specialization_specificity(
                            patterns,
                            parameters,
                        ),
                        self._normalize_template_argument_text(
                            self._replace_identifiers(specialization_type, bindings)
                        ),
                    )
                )
        if unresolved_variadic_match:
            return None
        if partial_matches:
            highest_specificity = max(score for score, _result in partial_matches)
            best_results = {
                result
                for score, result in partial_matches
                if score == highest_specificity
            }
            if len(best_results) == 1:
                return next(iter(best_results))
            return None
        parameters = trait.get("parameters", [])
        default_type = trait.get("default")
        if not isinstance(default_type, str) or len(arguments) < len(parameters):
            return None
        substitutions = dict(zip(parameters, arguments))
        resolved = self._replace_identifiers(default_type, substitutions)
        return self._normalize_template_argument_text(resolved)

    @staticmethod
    def _namespace_lookup_names(
        name: str,
        namespace: str,
        globally_qualified: bool = False,
    ) -> List[str]:
        if globally_qualified or not namespace:
            return [name]
        namespace_parts = [part for part in namespace.split("::") if part]
        candidates = [
            "::".join([*namespace_parts[:depth], name])
            for depth in range(len(namespace_parts), 0, -1)
        ]
        candidates.append(name)
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _template_trait_variadic_specialization_may_match(
        patterns: Tuple[str, ...],
        arguments: List[str],
        parameters: Set[str],
        variadic_parameters: Set[str],
    ) -> bool:
        if len(patterns) != len(arguments):
            minimum_arguments = len(patterns) - sum(
                1
                for pattern in patterns
                if any(
                    re.fullmatch(
                        rf"\s*{re.escape(parameter)}\s*\.\.\.\s*",
                        pattern,
                    )
                    for parameter in variadic_parameters
                )
            )
            return len(arguments) >= minimum_arguments and minimum_arguments < len(
                patterns
            )
        for pattern, argument in zip(patterns, arguments):
            normalized_pattern = re.sub(r"\s+", "", pattern)
            normalized_argument = re.sub(r"\s+", "", argument)
            pattern_parts: List[str] = []
            position = 0
            for match in IDENTIFIER_RE.finditer(normalized_pattern):
                pattern_parts.append(
                    re.escape(normalized_pattern[position : match.start()])
                )
                identifier = match.group(0)
                position = match.end()
                if identifier in variadic_parameters:
                    pattern_parts.append(".*?")
                    if normalized_pattern[position : position + 3] == "...":
                        position += 3
                elif identifier in parameters:
                    pattern_parts.append(".+?")
                else:
                    pattern_parts.append(re.escape(identifier))
            pattern_parts.append(re.escape(normalized_pattern[position:]))
            if re.fullmatch("".join(pattern_parts), normalized_argument) is None:
                return False
        return True

    @staticmethod
    def _template_trait_specialization_specificity(
        patterns: Tuple[str, ...],
        parameters: Set[str],
    ) -> Tuple[int, int, int]:
        parameter_occurrences: List[str] = []
        literal_characters = 0
        for pattern in patterns:
            position = 0
            for match in IDENTIFIER_RE.finditer(pattern):
                literal_characters += len(
                    re.sub(r"\s+", "", pattern[position : match.start()])
                )
                identifier = match.group(0)
                if identifier in parameters:
                    parameter_occurrences.append(identifier)
                else:
                    literal_characters += len(identifier)
                position = match.end()
            literal_characters += len(re.sub(r"\s+", "", pattern[position:]))
        repeated_constraints = len(parameter_occurrences) - len(
            set(parameter_occurrences)
        )
        return (
            literal_characters,
            repeated_constraints,
            -len(set(parameter_occurrences)),
        )

    @staticmethod
    def _skip_cpp_trivia(code: str, position: int) -> int:
        cursor = position
        while cursor < len(code):
            if code[cursor].isspace():
                cursor += 1
                continue
            if code.startswith("//", cursor):
                newline = code.find("\n", cursor + 2)
                if newline == -1:
                    return len(code)
                cursor = newline + 1
                continue
            if code.startswith("/*", cursor):
                comment_end = code.find("*/", cursor + 2)
                if comment_end == -1:
                    return len(code)
                cursor = comment_end + 2
                continue
            break
        return cursor

    def _read_namespace_components(
        self,
        code: str,
        position: int,
    ) -> Tuple[Tuple[str, ...], int]:
        components: List[str] = []
        cursor = self._skip_cpp_trivia(code, position)
        while cursor < len(code):
            name_match = IDENTIFIER_RE.match(code, cursor)
            if name_match is None:
                break
            components.append(name_match.group(0))
            cursor = self._skip_cpp_trivia(code, name_match.end())
            if not code.startswith("::", cursor):
                break
            cursor = self._skip_cpp_trivia(code, cursor + 2)
        return tuple(components), cursor

    def _find_namespace_spans(self, code: str) -> List[Tuple[int, int, str]]:
        analysis = self._source_analysis(code)
        if analysis.namespace_spans is None:
            analysis.namespace_spans = self._scan_namespace_spans(code)
        return analysis.namespace_spans

    def _scan_namespace_spans(self, code: str) -> List[Tuple[int, int, str]]:
        spans: List[Tuple[int, int, str]] = []
        brace_stack: List[Tuple[int, Optional[Tuple[str, ...]], str]] = []
        active_namespace: List[str] = []
        i = 0
        while i < len(code):
            if code[i] in "\"'":
                _literal, consumed = self._read_string(code, i)
                i += consumed
                continue
            if code.startswith("//", i):
                newline = code.find("\n", i + 2)
                if newline == -1:
                    break
                i = newline + 1
                continue
            if code.startswith("/*", i):
                comment_end = code.find("*/", i + 2)
                if comment_end == -1:
                    break
                i = comment_end + 2
                continue
            if (
                code.startswith("namespace", i)
                and (i == 0 or not (code[i - 1].isalnum() or code[i - 1] == "_"))
                and (
                    i + len("namespace") == len(code)
                    or not (
                        code[i + len("namespace")].isalnum()
                        or code[i + len("namespace")] == "_"
                    )
                )
            ):
                components, cursor = self._read_namespace_components(
                    code,
                    i + len("namespace"),
                )
                cursor = self._skip_cpp_trivia(code, cursor)
                if cursor < len(code) and code[cursor] == "{":
                    full_namespace = "::".join([*active_namespace, *components])
                    brace_stack.append((cursor, components, full_namespace))
                    active_namespace.extend(components)
                    i = cursor + 1
                    continue
            if code[i] == "{":
                brace_stack.append((i, None, ""))
            elif code[i] == "}" and brace_stack:
                open_brace, components, full_namespace = brace_stack.pop()
                if components is not None:
                    spans.append((open_brace + 1, i, full_namespace))
                    if components:
                        del active_namespace[-len(components) :]
            i += 1
        return spans

    @staticmethod
    def _namespace_at(
        namespace_spans: List[Tuple[int, int, str]],
        position: int,
    ) -> str:
        containing = [span for span in namespace_spans if span[0] <= position < span[1]]
        if not containing:
            return ""
        return max(containing, key=lambda span: span[0])[2]

    def _find_template_type_traits(
        self,
        code: str,
        namespace_spans: Optional[List[Tuple[int, int, str]]] = None,
    ) -> Dict[str, Dict[str, object]]:
        traits: Dict[str, Dict[str, object]] = {}
        if namespace_spans is None:
            namespace_spans = self._find_namespace_spans(code)
        pos = 0
        while True:
            match = re.search(r"\btemplate\s*<", code[pos:])
            if match is None:
                break
            start = pos + match.start()
            angle_start = code.find("<", start)
            angle_end = self._find_matching_template_param_angle(code, angle_start)
            if angle_end is None:
                pos = start + len("template")
                continue
            declaration_start = angle_end + 1
            body_start = self._find_next_top_level_char(code, declaration_start, "{")
            semicolon = self._find_next_top_level_char(code, declaration_start, ";")
            if body_start is None or (semicolon is not None and semicolon < body_start):
                pos = declaration_start
                continue
            body_end = self._find_matching_brace(code, body_start)
            if body_end is None:
                pos = body_start + 1
                continue
            header = code[declaration_start:body_start]
            body = code[body_start + 1 : body_end - 1]
            self._record_template_type_trait(
                traits,
                code[angle_start + 1 : angle_end],
                header,
                body,
                self._namespace_at(namespace_spans, start),
            )
            pos = body_end
        return traits

    def _record_template_type_trait(
        self,
        traits: Dict[str, Dict[str, object]],
        parameter_text: str,
        header: str,
        body: str,
        namespace: str = "",
    ) -> None:
        alias_type = self._template_type_trait_alias_type(body)
        if not alias_type:
            return
        header_match = re.search(
            r"\b(?:struct|class)\s+([A-Za-z_][A-Za-z0-9_:]*)\s*"
            r"(?:<(?P<args>[^{};]*)>)?",
            header,
            re.DOTALL,
        )
        if header_match is None:
            return
        name = header_match.group(1)
        if "::" not in name and namespace:
            name = f"{namespace}::{name}"
        trait = traits.setdefault(
            name,
            {
                "parameters": [],
                "default": None,
                "specializations": {},
            },
        )
        specialization_args = header_match.group("args")
        if specialization_args is not None:
            arguments = tuple(
                self._normalize_template_argument_text(argument)
                for argument in self._split_top_level_commas(specialization_args)
            )
            trait.setdefault("specializations", {})[arguments] = {
                "type": alias_type,
                "parameters": self._template_parameter_names(parameter_text),
                "variadic_parameters": sorted(
                    self._variadic_template_parameter_names(parameter_text)
                ),
            }
            return
        trait["parameters"] = self._template_parameter_names(parameter_text)
        trait["default"] = alias_type

    def _template_type_trait_alias_type(self, body: str) -> Optional[str]:
        match = re.search(
            r"\busing\s+type\s*=\s*(?P<type>[^;{}]+)\s*;",
            body,
            re.DOTALL,
        )
        if match is None:
            return None
        return self._normalize_template_argument_text(match.group("type"))

    def _expand_variadic_function_parameters(
        self, source: str, variadic_bindings: Dict[str, List[str]]
    ) -> str:
        header_end = source.find("{")
        if header_end == -1:
            return source
        header = source[:header_end]
        open_paren = header.find("(")
        if open_paren == -1:
            return source
        close_paren = self._find_matching_delimiter(header, open_paren, "(", ")")
        if close_paren is None:
            return source

        parameters = self._split_top_level_commas(header[open_paren + 1 : close_paren])
        expanded_parameters: List[str] = []
        renames: Dict[str, str] = {}
        for parameter in parameters:
            pack = self._variadic_function_parameter_name(parameter)
            if pack is None:
                expanded_parameters.append(parameter)
                continue
            type_name, value_name = pack
            bound_types = variadic_bindings.get(type_name, [])
            if not bound_types:
                continue
            for index, bound_type in enumerate(bound_types):
                generated_name = f"{value_name}_{index}"
                expanded = parameter.replace("...", "")
                expanded = re.sub(
                    rf"\b{re.escape(type_name)}\b",
                    bound_type,
                    expanded,
                    count=1,
                )
                expanded = re.sub(
                    rf"\b{re.escape(value_name)}\b",
                    generated_name,
                    expanded,
                    count=1,
                )
                expanded_parameters.append(expanded.strip())
            renames[value_name] = f"{value_name}_0"

        rebuilt_header = (
            header[: open_paren + 1]
            + ", ".join(expanded_parameters)
            + header[close_paren:]
        )
        body = source[header_end:]
        if renames:
            body = self._replace_identifiers(body, renames)
        return rebuilt_header + body

    def _variadic_function_parameter_name(
        self, parameter: str
    ) -> Optional[Tuple[str, str]]:
        if "..." not in parameter:
            return None
        cleaned = parameter.replace("...", " ")
        tokens = IDENTIFIER_RE.findall(cleaned)
        if len(tokens) < 2:
            return None
        return tokens[-2], tokens[-1]

    def _function_parameter_start(self, header: str) -> Optional[int]:
        paren_depth = 0
        bracket_depth = 0
        angle_depth = 0
        i = 0
        while i < len(header):
            if header[i] in "\"'":
                _literal, consumed = self._read_string(header, i)
                i += consumed
                continue
            if header.startswith("//", i):
                end = header.find("\n", i)
                if end == -1:
                    return None
                i = end + 1
                continue
            if header.startswith("/*", i):
                end = header.find("*/", i + 2)
                if end == -1:
                    return None
                i = end + 2
                continue

            ch = header[i]
            if (
                ch == "("
                and paren_depth == 0
                and bracket_depth == 0
                and angle_depth == 0
            ):
                return i
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)
            elif ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif ch == "<":
                angle_depth += 1
            elif ch == ">":
                angle_depth = max(0, angle_depth - 1)
            i += 1
        return None

    def _function_parameter_list_span(self, header: str) -> Optional[Tuple[int, int]]:
        # Locate the (open, close) offsets of a function header's REAL parameter
        # list. `operator()` is special: its declarator itself contains an empty
        # `()`, so the parameter list is the paren group that FOLLOWS it rather
        # than the first top-level `(` (which the plain scanner returns). Mirrors
        # the operator-aware handling in `_parse_struct_method` so an out-of-line
        # `Ret S::operator()(params)` definition exposes its parameters for
        # call-argument inference just like an in-struct declaration does.
        operator_match = re.search(r"\boperator\s*\(\s*\)", header)
        if operator_match is not None:
            relative_start = self._function_parameter_start(
                header[operator_match.end() :]
            )
            if relative_start is None:
                return None
            paren_start = relative_start + operator_match.end()
        else:
            paren_start = self._function_parameter_start(header)
            if paren_start is None:
                return None
        paren_end = self._find_matching_delimiter(header, paren_start, "(", ")")
        if paren_end is None:
            return None
        return paren_start, paren_end

    def _function_name_from_header(self, header: str) -> Optional[str]:
        paren_index = self._function_parameter_start(header)
        if paren_index is None:
            return None
        before_params = header[:paren_index].rstrip()
        match = re.search(r"([A-Za-z_][A-Za-z0-9_:]*)\s*$", before_params)
        if match is None:
            return None
        return match.group(1).split("::")[-1]

    def _rename_function_definition(
        self, source: str, old_name: str, new_name: str
    ) -> str:
        body_start = source.find("{")
        header = source if body_start == -1 else source[:body_start]
        pattern = re.compile(rf"\b{re.escape(old_name)}\s*(?=\()")
        matches = list(pattern.finditer(header))
        if not matches:
            return source
        match = matches[-1]
        return source[: match.start()] + new_name + source[match.end() :]

    def _rename_struct_definition(
        self,
        source: str,
        old_name: str,
        new_name: str,
        *,
        strip_specialization_arguments: bool = False,
    ) -> str:
        pattern = re.compile(rf"\b(struct|class)\s+{re.escape(old_name)}\b")
        match = pattern.search(source)
        if match is None:
            return source
        declaration_end = match.end()
        if strip_specialization_arguments:
            angle_start = declaration_end
            while angle_start < len(source) and source[angle_start].isspace():
                angle_start += 1
            if angle_start < len(source) and source[angle_start] == "<":
                angle_end = self._find_matching_angle(source, angle_start)
                if angle_end is None:
                    return source
                declaration_end = angle_end + 1
        return (
            source[: match.start()]
            + f"{match.group(1)} {new_name}"
            + source[declaration_end:]
        )

    def _rename_materialized_struct_constructors(
        self, source: str, old_name: str, new_name: str
    ) -> str:
        """Rename constructor declarators with a materialized template owner."""

        definitions = self._find_concrete_struct_definitions(source)
        owner = next((item for item in definitions if item.name == new_name), None)
        if owner is None:
            return source

        replacements: List[Tuple[int, int, str]] = []
        pattern = re.compile(rf"\b{re.escape(old_name)}\s*(?=\()")
        for constructor in owner.constructors:
            if constructor.span is None:
                continue
            start, end = constructor.span
            match = pattern.search(source, start, end)
            if match is not None:
                replacements.append((match.start(), match.end(), new_name))
        return self._apply_text_replacements(source, replacements)

    def _materialized_function_identifier(self, host_name: str, fallback: str) -> str:
        identifier = re.sub(r"\W", "_", host_name)
        if not identifier or identifier[0].isdigit():
            identifier = f"{fallback}_{identifier}" if identifier else fallback
        return identifier

    def _evaluate_metal_string_expression(self, expression: str) -> str:
        strings = re.findall(r'"((?:\\.|[^"\\])*)"|\'((?:\\.|[^\'\\])*)\'', expression)
        if strings:
            return "".join(
                self._unescape_metal_string(double or single)
                for double, single in strings
            )
        return expression.strip()

    def _unescape_metal_string(self, value: str) -> str:
        return bytes(value, "utf-8").decode("unicode_escape")

    def _split_top_level_commas(self, text: str) -> List[str]:
        parts: List[str] = []
        current = ""
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        angle_depth = 0
        i = 0
        while i < len(text):
            if text[i] in "\"'":
                literal, consumed = self._read_string(text, i)
                current += literal
                i += consumed
                continue
            if text.startswith("//", i):
                current += text[i:]
                break
            if text.startswith("/*", i):
                end = text.find("*/", i + 2)
                if end == -1:
                    current += text[i:]
                    break
                current += text[i : end + 2]
                i = end + 2
                continue

            ch = text[i]
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)
            elif ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth = max(0, brace_depth - 1)
            elif ch == "<":
                angle_depth += 1
            elif ch == ">":
                angle_depth = max(0, angle_depth - 1)
            elif (
                ch == ","
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
                and angle_depth == 0
            ):
                parts.append(current.strip())
                current = ""
                i += 1
                continue
            current += ch
            i += 1
        if current.strip():
            parts.append(current.strip())
        return parts

    def _replace_identifiers(self, text: str, replacements: Dict[str, str]) -> str:
        result = ""
        i = 0
        while i < len(text):
            if text[i] in "\"'":
                literal, consumed = self._read_string(text, i)
                result += literal
                i += consumed
                continue
            if text.startswith("//", i):
                end = text.find("\n", i)
                if end == -1:
                    result += text[i:]
                    break
                result += text[i:end]
                i = end
                continue
            if text.startswith("/*", i):
                end = text.find("*/", i + 2)
                if end == -1:
                    result += text[i:]
                    break
                result += text[i : end + 2]
                i = end + 2
                continue
            if text[i].isalpha() or text[i] == "_":
                ident, consumed = self._read_identifier(text, i)
                if self._is_member_identifier_context(text, i):
                    result += ident
                else:
                    result += replacements.get(ident, ident)
                i += consumed
                continue
            result += text[i]
            i += 1
        return result

    def _is_member_identifier_context(self, text: str, index: int) -> bool:
        previous = index - 1
        while previous >= 0 and text[previous].isspace():
            previous -= 1
        if previous < 0:
            return False
        if text[previous] == ".":
            return True
        if previous >= 1 and text[previous - 1 : previous + 1] == "::":
            return True
        return previous >= 1 and text[previous - 1 : previous + 1] == "->"

    def _find_matching_angle(self, code: str, start: int) -> Optional[int]:
        return self._find_matching_delimiter(code, start, "<", ">")

    def _find_matching_brace(self, code: str, start: int) -> Optional[int]:
        end = self._find_matching_delimiter(code, start, "{", "}")
        return None if end is None else end + 1

    def _find_matching_delimiter(
        self, code: str, start: int, opener: str, closer: str
    ) -> Optional[int]:
        if start < 0 or start >= len(code) or code[start] != opener:
            return None
        depth = 0
        i = start
        while i < len(code):
            if code[i] in "\"'":
                _literal, consumed = self._read_string(code, i)
                i += consumed
                continue
            if code.startswith("//", i):
                end = code.find("\n", i)
                if end == -1:
                    return None
                i = end + 1
                continue
            if code.startswith("/*", i):
                end = code.find("*/", i + 2)
                if end == -1:
                    return None
                i = end + 2
                continue
            if code[i] == opener:
                depth += 1
            elif code[i] == closer:
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        return None

    def _find_next_top_level_char(
        self, code: str, start: int, target: str
    ) -> Optional[int]:
        paren_depth = 0
        bracket_depth = 0
        angle_depth = 0
        i = start
        while i < len(code):
            ch = code[i]
            if ch == '"' or ch == "'":
                _literal, consumed = self._read_string(code, i)
                i += consumed
                continue
            if ch == "/":
                if code.startswith("//", i):
                    end = code.find("\n", i)
                    if end == -1:
                        return None
                    i = end + 1
                    continue
                if code.startswith("/*", i):
                    end = code.find("*/", i + 2)
                    if end == -1:
                        return None
                    i = end + 2
                    continue
            if (
                ch == target
                and paren_depth == 0
                and bracket_depth == 0
                and angle_depth == 0
            ):
                return i
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)
            elif ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif ch == "<":
                angle_depth += 1
            elif ch == ">":
                angle_depth = max(0, angle_depth - 1)
            i += 1
        return None

    def _statement_end(self, code: str, start: int) -> Optional[int]:
        paren_depth = 0
        bracket_depth = 0
        angle_depth = 0
        brace_depth = 0
        i = start
        while i < len(code):
            ch = code[i]
            if ch == '"' or ch == "'":
                _literal, consumed = self._read_string(code, i)
                i += consumed
                continue
            if ch == "/":
                if code.startswith("//", i):
                    end = code.find("\n", i)
                    if end == -1:
                        return len(code)
                    i = end + 1
                    continue
                if code.startswith("/*", i):
                    end = code.find("*/", i + 2)
                    if end == -1:
                        return None
                    i = end + 2
                    continue
            if (
                ch == ";"
                and paren_depth == 0
                and bracket_depth == 0
                and angle_depth == 0
                and brace_depth == 0
            ):
                return i + 1
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)
            elif ch == "[":
                bracket_depth += 1
            elif ch == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif ch == "<":
                angle_depth += 1
            elif ch == ">":
                angle_depth = max(0, angle_depth - 1)
            elif ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth = max(0, brace_depth - 1)
            i += 1
        return None

    def _apply_text_replacements(
        self, code: str, replacements: List[Tuple[int, int, str]]
    ) -> str:
        replacements = sorted(replacements, key=lambda item: item[0])
        result = []
        pos = 0
        for start, end, replacement in replacements:
            if start < pos:
                continue
            result.append(code[pos:start])
            result.append(replacement)
            pos = end
        result.append(code[pos:])
        return "".join(result)

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

    def _expand_clang_feature_test_macros(
        self, text: str, file_path: Optional[str]
    ) -> str:
        result = ""
        i = 0
        while i < len(text):
            if text[i] in "\"'":
                literal, consumed = self._read_string(text, i)
                result += literal
                i += consumed
                continue
            if text.startswith("//", i):
                result += text[i:]
                break
            if text.startswith("/*", i):
                end = text.find("*/", i + 2)
                if end == -1:
                    result += text[i:]
                    break
                result += text[i : end + 2]
                i = end + 2
                continue
            if text[i].isalpha() or text[i] == "_":
                ident, consumed = self._read_identifier(text, i)
                if ident in CLANG_FEATURE_TEST_MACROS:
                    replacement, consumed_call = self._expand_feature_test_call(
                        ident, text, i + consumed, file_path
                    )
                    if replacement is not None:
                        result += replacement
                        i += consumed + consumed_call
                        continue
                result += ident
                i += consumed
                continue
            result += text[i]
            i += 1
        return result

    def _expand_feature_test_call(
        self, name: str, text: str, start: int, file_path: Optional[str]
    ):
        i = start
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] != "(":
            return None, 0
        args, consumed = self._parse_macro_args(text, i)
        if not consumed or i + consumed > len(text):
            return None, 0
        if name in {"__has_include", "__has_include_next"}:
            value = self._has_include(args[0] if args else "", file_path)
        else:
            value = False
        return ("1" if value else "0"), i + consumed - start

    def _has_include(self, include_arg: str, file_path: Optional[str]) -> bool:
        include_arg = self._strip_macro_comments(include_arg).strip()
        match = re.match(r'([<"])([^>"]+)[>"]$', include_arg)
        if not match:
            return False

        delimiter, target = match.groups()
        search_paths: List[str] = []
        if delimiter == '"' and file_path:
            search_paths.append(os.path.dirname(file_path))
        search_paths.extend(self.include_paths)

        return any(os.path.isfile(os.path.join(base, target)) for base in search_paths)

    def _join_multiline_function_macro_call(
        self, lines: List[str], start: int
    ) -> Tuple[str, int]:
        line = lines[start]
        consumed = 1
        while self._find_incomplete_function_macro_call(
            line
        ) is not None and start + consumed < len(lines):
            next_line = lines[start + consumed]
            if next_line.lstrip().startswith("#"):
                break
            # Preserve a physical line boundary so a // comment in one argument
            # cannot consume the remaining invocation.
            line += "\n" + next_line.lstrip()
            consumed += 1
        return line, consumed

    def _find_incomplete_function_macro_call(
        self, text: str
    ) -> Optional[Tuple[str, int]]:
        i = 0
        while i < len(text):
            if text[i] in "\"'":
                _literal, consumed = self._read_string(text, i)
                i += consumed
                continue
            if text.startswith("//", i):
                newline = text.find("\n", i + 2)
                if newline == -1:
                    return None
                i = newline + 1
                continue
            if text.startswith("/*", i):
                end = text.find("*/", i + 2)
                if end == -1:
                    return None
                i = end + 2
                continue
            if text[i].isalpha() or text[i] == "_":
                ident_start = i
                ident, consumed = self._read_identifier(text, i)
                i += consumed
                macro = self.macros.get(ident)
                if macro is None:
                    continue
                if not macro.is_function_like():
                    continue
                j = i
                while j < len(text) and text[j].isspace():
                    j += 1
                if j < len(text) and text[j] == "(":
                    call_end = self._function_macro_call_end(text, j)
                    if call_end is None:
                        return ident, ident_start
                    i = call_end
                continue
            i += 1
        return None

    def _function_macro_call_end(self, text: str, start: int) -> Optional[int]:
        depth = 0
        i = start
        while i < len(text):
            if text[i] in "\"'":
                _literal, consumed = self._read_string(text, i)
                i += consumed
                continue
            if text.startswith("//", i):
                newline = text.find("\n", i + 2)
                if newline == -1:
                    return None
                i = newline + 1
                continue
            if text.startswith("/*", i):
                end = text.find("*/", i + 2)
                if end == -1:
                    return None
                i = end + 2
                continue
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return None

    def _parse_macro_args(self, text: str, start: int) -> Tuple[List[str], int]:
        assert text[start] == "("
        args: List[str] = []
        current: List[str] = []
        depth = 0
        i = start
        while i < len(text):
            if text[i] in "\"'":
                literal, consumed = self._read_string(text, i)
                current.append(literal)
                i += consumed
                continue
            if text.startswith("//", i):
                newline = text.find("\n", i + 2)
                if newline == -1:
                    return args, 0
                current.append(" ")
                i = newline + 1
                continue
            if text.startswith("/*", i):
                end = text.find("*/", i + 2)
                if end == -1:
                    return args, 0
                current.append(" ")
                i = end + 2
                continue

            ch = text[i]
            if ch == "(":
                depth += 1
                if depth > 1:
                    current.append(ch)
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    args.append("".join(current).strip())
                    return args, i - start + 1
                current.append(ch)
            elif ch == "," and depth == 1:
                args.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
            i += 1
        return args, 0

    @staticmethod
    def _raise_incomplete_function_macro_call(
        text: str,
        macro_name: str,
        start: int,
        line_num: int,
        file_path: Optional[str],
    ) -> None:
        line_offset = text.count("\n", 0, start)
        previous_newline = text.rfind("\n", 0, start)
        line = line_num + line_offset
        column = start - previous_newline
        location = {
            "line": line,
            "column": column,
            "length": len(macro_name),
            "endLine": line,
            "endColumn": column + len(macro_name),
        }
        if file_path is not None:
            location["file"] = file_path
        raise MetalMacroExpansionError(
            f"Cannot expand function-like Metal macro '{macro_name}': "
            "the invocation is not terminated before the end of its source region.",
            macro_name=macro_name,
            reason="unterminated-function-macro-invocation",
            source_location=location,
            suggested_action=(
                "Close the invocation with balanced parentheses before the next "
                "preprocessor directive or end of file."
            ),
        )

    def _handle_include(self, rest: str, file_path: Optional[str]) -> Optional[str]:
        included = super()._handle_include(rest, file_path)
        if included is not None or self.strict:
            return included
        include_target = rest.strip()
        if include_target.startswith("<") and include_target.endswith(">"):
            return f"{PRESERVED_INCLUDE_SENTINEL}{include_target}"
        return None
