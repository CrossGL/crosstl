"""Exceptions, records, patterns and helpers shared by the Metal preprocessor members."""

import operator
import re
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from ..receiver_visibility import ReceiverDeclarationSweep

PRESERVED_INCLUDE_SENTINEL = "__CROSSGL_METAL_PRESERVED_INCLUDE__ "
# Macros the Metal compiler predefines for every source (`metal -dM -E`): the language version as
# major * 100 + minor * 10, and __METAL__. MSL 3.1 is the importer's default language version.
METAL_PREDEFINED_MACROS = {"__METAL_VERSION__": "310", "__METAL__": "1"}
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
METAL_ANGLE_OPERATOR_RE = re.compile(
    r"\boperator"
    r"(?>\s+|/\*[^*]*\*+(?:[^/*][^*]*\*+)*/|//[^\n]*(?:\n|$))*+"
    r"(?:<<=|>>=|<=>|<<|>>|<=|>=|<|>)"
)
METAL_STATEMENT_DELIMITER_RE = re.compile(r"[()\[\]<>{};\"'/]")
METAL_TEMPLATE_DELIMITER_RE = re.compile(r"[()\[\]<>{};\"'/]")
METAL_COMMENT_LITERAL_RE = re.compile(r"[\"'/]")
METAL_SCOPE_DELIMITER_RE = re.compile(r"[{}\"'/]")
METAL_NAMESPACE_DELIMITER_RE = re.compile(r"namespace|[{}\"'/]")
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
LOCAL_DATA_MEMBER_MEMO_LIMIT = 65536
_UNPARSED = object()
SIMPLE_DECLARATION_EVENT_RE = re.compile(r"[\"'/(){};]")
# Characters at which a template-reference scanner does more than step past.
SCAN_STOP_RE = re.compile(r"[\"'/A-Za-z_]|[^\x00-\x7f]")

CONTAINING_SPAN_CACHE_LIMIT = 64
LEXICAL_SCOPE_CACHE_LIMIT = 64
METAL_RECEIVER_DECLARATION_RE = re.compile(
    r"(?<![A-Za-z0-9_:])"
    r"(?P<leading>(?:(?:const|volatile|thread|threadgroup|"
    r"threadgroup_imageblock|device|constant)\s+)*)"
    r"(?P<type>(?:::\s*)?(?:[A-Za-z_]\w*\s*::\s*)*[A-Za-z_]\w*)"
    r"(?P<trailing>(?:\s+(?:const|volatile|thread|threadgroup|"
    r"threadgroup_imageblock|device|constant))*)\s*"
    r"(?P<indirection>\*+|&&|&)?"
    r"(?P<indirection_qualifiers>"
    r"(?:\s+(?:const|volatile|restrict|__restrict|__restrict__))*)"
    r"\s*\b(?P<name>[A-Za-z_]\w*)\b\s*(?=[,;)=\[\{(])"
)
# Metal library templates whose member functions the frontend lowers itself,
# such as ``simdgroup_matrix::thread_elements()`` and ``texture2d::read()``.
# Struct-method lowering leaves calls on receivers declared with them alone.
METAL_BUILTIN_MEMBER_RECEIVER_TEMPLATES = frozenset(
    {
        "simdgroup_matrix",
        "texture1d",
        "texture1d_array",
        "texture2d",
        "texture2d_array",
        "texture2d_ms",
        "texture2d_ms_array",
        "texture3d",
        "texturecube",
        "texturecube_array",
        "texture_buffer",
        "depth2d",
        "depth2d_array",
        "depth2d_ms",
        "depth2d_ms_array",
        "depthcube",
        "depthcube_array",
    }
)
METAL_BUILTIN_MEMBER_RECEIVER_TEMPLATE_RE = re.compile(
    r"(?<![A-Za-z0-9_:])(?:metal\s*::\s*)?(?P<base>"
    + "|".join(sorted(METAL_BUILTIN_MEMBER_RECEIVER_TEMPLATES, key=len, reverse=True))
    + r")\s*<"
)
# The <metal_simdgroup_matrix> library typedefs of the 8x8 SIMD-group matrix
# specializations. Expanding them right after macro expansion lets every later
# pass see the template spelling it already lowers.
METAL_LIBRARY_MATRIX_TYPEDEFS = {
    "simdgroup_half8x8": "simdgroup_matrix<half, 8, 8>",
    "simdgroup_float8x8": "simdgroup_matrix<float, 8, 8>",
    "simdgroup_bfloat8x8": "simdgroup_matrix<bfloat, 8, 8>",
}
METAL_LIBRARY_MATRIX_TYPEDEF_RE = re.compile(
    r"(?<![A-Za-z0-9_:])(?P<qualifier>metal\s*::\s*)?(?P<name>"
    + "|".join(sorted(METAL_LIBRARY_MATRIX_TYPEDEFS, key=len, reverse=True))
    + r")\b"
)
METAL_RECEIVER_DECLARATOR_PREFIX_RE = re.compile(
    r"(?:\s+(?:const|volatile|thread|threadgroup|threadgroup_imageblock|device"
    r"|constant))*\s*(?:\*+|&&|&)?"
    r"(?:\s*(?:const|volatile|restrict|__restrict|__restrict__)\b)*\s*"
)
# A `::` that opens a name (not `A::metal`), before `metal::`.
_GLOBAL_METAL_QUALIFIER = re.compile(r"(?<![\w:])::\s*(?=metal\s*::)")


def _strip_global_metal_qualifiers(text: str) -> str:
    """Spell `::metal::name` as `metal::name`; both name the same declaration."""
    return _GLOBAL_METAL_QUALIFIER.sub("", text)


# Sort key of one `(source offset, declared type)` entry of a declaration map.
_declaration_position = operator.itemgetter(0)


def _keyword_before(text: str, start: int, end: int, keyword: str) -> Optional[int]:
    """Start of the match of a trailing keyword in ``text[start:end]``.

    Equivalent to ``re.search(rf"\\b{keyword}\\s*$", text[start:end])`` but without
    copying or rescanning the prefix.
    """
    while end > start and text[end - 1].isspace():
        end -= 1
    keyword_start = end - len(keyword)
    if keyword_start < start or not text.startswith(keyword, keyword_start):
        return None
    if keyword_start > start:
        previous = text[keyword_start - 1]
        if previous.isalnum() or previous == "_":
            return None
    return keyword_start


# The multi-word C++ spellings of integer types, to the Metal scalar each names (MSL 3.2 section 2.1).
_C_INTEGER_SPELLINGS = {
    "signed": "int",
    "signed int": "int",
    "unsigned": "uint",
    "unsigned int": "uint",
    "signed short": "short",
    "signed short int": "short",
    "unsigned short": "ushort",
    "unsigned short int": "ushort",
    "signed long": "long",
    "signed long int": "long",
    "unsigned long": "ulong",
    "unsigned long int": "ulong",
}


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
        receiver_type: Optional[str] = None,
        candidate_signatures: Optional[Tuple[str, ...]] = None,
        candidate_mismatches: Optional[Tuple[Dict[str, object], ...]] = None,
    ):
        super().__init__(message)
        self.struct_name = struct_name
        self.method_name = method_name
        self.requested_signature = requested_signature
        self.suggested_action = suggested_action
        self.source_location = source_location
        self.reason = reason
        self.return_type = return_type
        self.receiver_type = receiver_type
        self.candidate_signatures = candidate_signatures or ()
        self.candidate_mismatches = candidate_mismatches or ()
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


class MetalTypeTraitError(ValueError):
    """A concrete Metal type-trait value that cannot be evaluated safely."""

    project_diagnostic_code = "project.translate.metal-type-trait-unresolved"
    missing_capabilities = ("compile-time.type-trait-evaluation",)

    def __init__(
        self,
        message: str,
        *,
        trait: str,
        expression: str,
        operands: Tuple[str, ...],
        resolved_operands: Tuple[str, ...],
        reason: str,
        owner: Optional[str],
        source_location: object,
        suggested_action: str,
    ):
        super().__init__(message)
        self.trait = trait
        self.expression = expression
        self.operands = operands
        self.resolved_operands = resolved_operands
        self.reason = reason
        self.owner = owner
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
    template_constraints: List[str] = field(default_factory=list)
    namespace: str = ""
    materializations: List[str] = field(default_factory=list)
    internal_linkage: bool = False


@dataclass
class _MetalTemplateStruct:
    name: str
    template_parameters: List[str]
    span: Tuple[int, int]
    source: str
    variadic_template_parameters: Set[str] = field(default_factory=set)
    template_parameter_defaults: Dict[str, str] = field(default_factory=dict)
    template_parameter_types: Dict[str, str] = field(default_factory=dict)
    template_type_traits: Dict[str, Dict[str, object]] = field(default_factory=dict)
    namespace: str = ""
    is_forward_declaration: bool = False


@dataclass(frozen=True)
class _MetalFreeOperatorDefinition:
    operator: str
    template_text: str
    template_parameters: Tuple[str, ...]
    template_constraints: Tuple[str, ...]
    return_type: str
    parameters: str
    parameter_types: Tuple[str, ...]
    body: str
    span: Tuple[int, int]
    internal_linkage: bool = False


@dataclass(frozen=True)
class _MetalBooleanVariableTemplate:
    name: str
    template_parameters: Tuple[str, ...]
    argument_patterns: Tuple[str, ...]
    expression: str
    is_partial_specialization: bool
    span: Tuple[int, int]
    namespace: str
    # ``value`` for a struct template read as a trait (``Name<T>::value``).
    member: str = ""
    # Every declared template parameter as ``(name or "", default or None)``.
    parameter_slots: Tuple[Tuple[str, Optional[str]], ...] = ()


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
class _MetalOutOfLineMemberDefinition:
    function: _MetalFunctionDefinition
    owner_name: str
    method: "_MetalStructMethod"


@dataclass(frozen=True)
class _MetalOutOfLineMemberCallSite:
    method_name: str
    argument_open: int
    receiver_contract: Optional["_MetalReceiverContract"]


@dataclass
class _MetalReachabilityTypeContext:
    buffer_element_types: "_MetalPositionedBufferTypes"
    local_variable_types: Dict[str, List[Tuple[int, str]]]
    receiver_variable_types: Dict[str, List[Tuple[int, str]]]
    structs_by_name: Dict[str, "_MetalStructDefinition"]
    struct_type_aliases: Dict[str, List["_MetalTypeAliasBinding"]]
    source_type_aliases: Dict[str, List["_MetalTypeAliasBinding"]]


@dataclass(frozen=True)
class _MetalTypeAliasBinding:
    declaration_position: int
    scope_start: int
    scope_end: int
    target: Optional[str]


@dataclass(frozen=True)
class _MetalIntegralConstantBinding:
    declaration_position: int
    scope_start: int
    scope_end: int
    value: Optional[str]
    # The complete type the local was declared with, when ``decltype`` of it can be spelled.
    declared_type: Optional[str] = None
    # ``using``/``typedef`` names a type: it shadows outer constants but is no
    # unresolved value.
    type_alias: bool = False


@dataclass(frozen=True)
class _MetalReceiverDeclaration:
    name: str
    declaration_position: int
    declaration_end: int
    type_position: int
    scope_start: int
    scope_end: int
    declared_type: str
    raw_type_text: str


@dataclass
class _MetalSourceAnalysis:
    comment_and_literal_spans: Optional[List[Tuple[int, int]]] = None
    lexical_brace_scopes: Optional[List[Tuple[int, int]]] = None
    template_declaration_spans: Optional[List[Tuple[int, int]]] = None
    namespace_spans: Optional[List[Tuple[int, int, str]]] = None
    anonymous_namespace_spans: List[Tuple[int, int]] = field(default_factory=list)
    receiver_declarations: Optional[Dict[str, List[_MetalReceiverDeclaration]]] = None
    # Per receiver name, the sweep that answers visibility queries over ``receiver_declarations``.
    receiver_sweeps: Dict[str, ReceiverDeclarationSweep] = field(default_factory=dict)
    template_functions: Optional[List[_MetalTemplateFunction]] = None
    function_definitions: Dict[
        Tuple[Tuple[int, int], ...], Tuple[_MetalFunctionDefinition, ...]
    ] = field(default_factory=dict)
    template_type_traits: Dict[
        Tuple[Tuple[int, int, str], ...], Dict[str, Dict[str, object]]
    ] = field(default_factory=dict)
    namespace_visibility: Optional["_MetalNamespaceVisibility"] = None
    builtin_member_receiver_template_references: Optional[
        Tuple[List[Tuple[int, int]], List[int]]
    ] = None
    local_data_member_declarations: Dict[
        Tuple[int, int], Tuple[Tuple[int, str, "_MetalDataMember"], ...]
    ] = field(default_factory=dict)
    static_constexpr_functions: Optional[Dict[str, List["_MetalConstexprFunction"]]] = (
        None
    )
    # Whole-source query results memoized by name; see source_analysis_cache.memoized.
    values: Dict[str, object] = field(default_factory=dict)


@dataclass
class _MetalLexicalScopeLookup:
    scopes: List[Tuple[int, int]]
    source_length: int
    span_count: int
    first_span: Tuple[int, int]
    last_span: Tuple[int, int]
    boundaries: List[int]
    innermost_scopes: List[Optional[Tuple[int, int]]]


@dataclass
class _MetalContainingSpanLookup:
    spans: List[Tuple[int, int]]
    span_count: int
    first_span: Tuple[int, int]
    last_span: Tuple[int, int]
    starts: Optional[List[int]]
    boundaries: Optional[List[int]]
    containing_spans: Optional[List[Optional[Tuple[int, int]]]]
    # Every start and end, sorted: ``_containing_span`` is constant between them.
    all_boundaries: Optional[List[int]] = None


@dataclass(frozen=True)
class _MetalStaticAssertion:
    span: Tuple[int, int]
    condition_span: Tuple[int, int]
    condition: str
    message: Optional[str]


@dataclass(frozen=True)
class _MetalConstexprCall:
    name: str
    qualified_name: str
    template_arguments: Optional[List[str]]
    arguments: List[str]
    span: Tuple[int, int]
    argument_open: int


@dataclass(frozen=True)
class _MetalUsingNamespaceDirective:
    target: str
    globally_qualified: bool
    declaration_namespace: str
    declaration_position: int
    scope_start: int
    scope_end: int


@dataclass(frozen=True)
class _MetalNamespaceVisibility:
    source_length: int
    namespace_spans: Tuple[Tuple[int, int, str], ...]
    using_namespace_directives: Tuple[_MetalUsingNamespaceDirective, ...]


@dataclass
class _MetalConstexprFunction:
    name: str
    span: Tuple[int, int]
    source: str
    namespace: str = ""
    is_template: bool = False
    template_parameters: List[str] = field(default_factory=list)
    variadic_template_parameters: Set[str] = field(default_factory=set)
    template_parameter_defaults: Dict[str, str] = field(default_factory=dict)
    template_parameter_types: Dict[str, str] = field(default_factory=dict)
    namespace_visibility: Optional[_MetalNamespaceVisibility] = None
    # Plain `typename T` / `class T` parameters, bound to concrete types.
    type_template_parameters: FrozenSet[str] = frozenset()


@dataclass(frozen=True)
class _MetalConstexprScalarValue:
    type_name: str
    value: Union[bool, int, float]


@dataclass(frozen=True)
class _MetalConstexprAggregateValue:
    type_name: str
    members: Tuple[Tuple[str, _MetalConstexprScalarValue], ...]

    def member(self, name: str) -> Optional[_MetalConstexprScalarValue]:
        return dict(self.members).get(name)


@dataclass(frozen=True)
class _MetalConstexprBinaryOperator:
    operator: str
    span: Tuple[int, int]
    template_parameters: Tuple[str, ...]
    return_type: str
    parameter_names: Tuple[str, str]
    parameter_types: Tuple[str, str]
    return_expression: str


@dataclass(frozen=True)
class _MetalSubscriptableType:
    """Element type plus a proven address-space-qualified pointer type."""

    element_type: str
    pointer_type: Optional[str] = None


class _MetalAddressableValueType(str):
    """Keep scalar value inference compatible while retaining storage provenance."""

    def __new__(cls, value_type, pointer_type=None, scope=None):
        value = super().__new__(cls, value_type)
        value.pointer_type = pointer_type
        value.scope = scope
        return value


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
    is_conversion_operator: bool = False
    is_const: bool = False
    is_volatile: bool = False
    ref_qualifier: str = ""
    receiver_address_spaces: Tuple[str, ...] = ()
    trailing_qualifiers: Tuple[str, ...] = ()
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


@dataclass(frozen=True)
class _MetalExplicitMemberSpecialization:
    """One exact out-of-line specialization of a template-owner member."""

    owner_name: str
    owner_qualified_name: str
    owner_arguments: Tuple[str, ...]
    method: _MetalStructMethod
    span: Tuple[int, int]


@dataclass(frozen=True)
class _MetalReceiverContract:
    """Source implicit-object state retained until member overload binding."""

    struct_name: str
    is_const: bool = False
    is_volatile: bool = False
    address_spaces: Tuple[str, ...] = ("thread",)
    indirection: str = ""
    value_category: str = "lvalue"
    source_type: str = ""

    @property
    def is_readonly(self) -> bool:
        return self.is_const or "constant" in self.address_spaces


@dataclass
class _MetalDataMember:
    """One data member of a struct, kept in DECLARATION order.

    In addition to the full types in ``data_member_types``, this separates the
    declared type text (e.g. ``const device float2*``) from its trailing array
    suffix and retains any default initializer and source declaration span.
    It is what the pointer-member scalar-replacement
    and declaration-order constant-substitution paths need.
    """

    name: str
    type_text: str
    default: Optional[str] = None
    initializer_style: str = "="
    array_suffix: str = ""
    span: Optional[Tuple[int, int]] = None

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
    # Full member types retain array layers and pointer qualifiers until a
    # call-argument expression selects an element. Missing entries remain
    # un-inferable.
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
    member_function_body_spans: List[Tuple[int, int]] = field(default_factory=list)
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
    # When present, each promoted pointer keeps a scalar relative offset in the
    # residual struct. This is used only by the explicit derived-pointer opt-in;
    # the default direct-pointer promotion remains byte-for-byte unchanged.
    pointer_offset_initializers: Optional[List[str]] = None
    # Pointer parameters that a remaining member initializer reads. The
    # rebuilt constructor and every construction keep them.
    retained_pointer_ctor_arg_indices: Tuple[int, ...] = ()
    # Promoted members that a remaining initializer reads, mapped to the
    # parameter that initializes each one directly.
    initializer_member_parameters: Dict[str, str] = field(default_factory=dict)

    @property
    def dropped_pointer_ctor_arg_indices(self) -> Set[int]:
        return set(self.pointer_ctor_arg_indices) - set(
            self.retained_pointer_ctor_arg_indices
        )

    @property
    def pointer_member_names(self) -> List[str]:
        return [member.name for member in self.pointer_members]

    @property
    def pointer_parameter_decls(self) -> List[str]:
        # ``<type> <name>`` for each promoted pointer member, in order.
        return [f"{member.type_text} {member.name}" for member in self.pointer_members]


@dataclass(frozen=True)
class _PromotedPointerBinding:
    """One lexical binding of a promoted struct receiver to its pointer roots."""

    declaration_position: int
    scope_start: int
    scope_end: int
    struct_name: str
    pointer_exprs: Tuple[str, ...]


@dataclass(frozen=True)
class _PromotedFunctionParameter:
    """A promoted-struct function parameter and its generated companions."""

    index: int
    name: str
    struct_name: str
    companion_names: Tuple[str, ...]
    companion_declarations: Tuple[str, ...]


@dataclass(frozen=True)
class _PromotedFunctionDefinition:
    """A free function whose ABI forwards promoted pointer members."""

    function: _MetalFunctionDefinition
    parameter_count: int
    promoted_parameters: Tuple[_PromotedFunctionParameter, ...]


# Characters `_find_static_constexpr_calls` acts on: quotes, a slash that may
# open a comment, and every word character that is not a decimal digit (a
# superset of the identifier starts it reads).
_CONSTEXPR_CALL_SCAN_STOPS = re.compile(r"[\"'/]|[^\W\d]")
