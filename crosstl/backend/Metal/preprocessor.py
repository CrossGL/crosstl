"""Preprocessor support for Metal source imports."""

import operator
import os
import re
from bisect import bisect_right
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Set, Tuple

from crosstl.backend.DirectX.preprocessor import HLSLPreprocessor, Macro

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
TEMPLATE_MATERIALIZATION_SCAN_CHARS_PER_WORK_ITEM = 8


class MetalStructMethodError(ValueError):
    """A struct member-function call that cannot be lowered safely.

    Raised when a CALLED template member method of a struct cannot be
    instantiated — either because a call argument type cannot be inferred with
    the conservative rules, or because the method's template parameters do not
    bind consistently. It deliberately PROPAGATES out of ``preprocess`` (it is
    NOT swallowed by the regression-safety ``try/except`` in
    ``_lower_struct_member_functions``) so the pipeline reports the kernel as a
    clean translation FAILURE rather than emitting a dangling ``obj.method(...)``
    call / broken output.
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
    ):
        super().__init__(message)
        self.struct_name = struct_name
        self.method_name = method_name
        self.requested_signature = requested_signature
        self.suggested_action = suggested_action
        self.source_location = source_location


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
    ):
        super().__init__(message)
        self.limit = limit
        self.limit_source = limit_source
        self.unique_specialization_count = unique_specialization_count
        self.required_work_items = required_work_items
        self.requested_signature = requested_signature
        self.suggested_action = suggested_action
        self.source_location = source_location


@dataclass
class _MetalTemplateFunction:
    name: str
    template_parameters: List[str]
    span: Tuple[int, int]
    body_start: int
    source: str
    variadic_template_parameters: Set[str] = field(default_factory=set)
    template_parameter_defaults: Dict[str, str] = field(default_factory=dict)
    template_type_traits: Dict[str, Dict[str, object]] = field(default_factory=dict)
    materializations: List[str] = field(default_factory=list)


@dataclass
class _MetalTemplateStruct:
    name: str
    template_parameters: List[str]
    span: Tuple[int, int]
    source: str
    variadic_template_parameters: Set[str] = field(default_factory=set)
    template_parameter_defaults: Dict[str, str] = field(default_factory=dict)


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

    @property
    def is_constraint(self) -> bool:
        # A non-type parameter whose declarator is anonymous (no bindable name)
        # is a SFINAE enabler we keep only for overload selection.
        return not self.is_type_parameter and self.constraint_text is not None


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
    # Template member methods carry their template parameter names; an empty list
    # marks an ordinary (non-template) member function. The raw `RetType` /
    # `parameters` text still contains the template parameter identifiers; they
    # are substituted with concrete types at instantiation time.
    template_parameters: List[str] = field(default_factory=list)
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
class _MetalStructDefinition:
    """A concrete (non-template) struct/class with its members split out."""

    name: str
    span: Tuple[int, int]
    body_span: Tuple[int, int]
    data_member_names: Set[str]
    methods: List[_MetalStructMethod]
    has_operator_call: bool
    # Data-member name -> its declared element type (the value type a `self.x`
    # access yields, with array extents stripped). Populated best-effort for
    # members whose type is recognizable; missing entries are simply
    # un-inferable. Used to type a `obj.member` / `obj.member[i]` call argument.
    data_member_types: Dict[str, str] = field(default_factory=dict)
    # Template member methods are kept separate from `methods`: they have no
    # single concrete signature to emit up front, so they are instantiated on
    # demand from their call sites' inferred argument types.
    template_methods: List[_MetalStructMethod] = field(default_factory=list)


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
        self._containing_span_cache: Optional[
            Tuple[
                List[Tuple[int, int]],
                int,
                Optional[Tuple[int, int]],
                Optional[Tuple[int, int]],
                Optional[List[int]],
            ]
        ] = None
        self.macros.setdefault(
            "TARGET_OS_SIMULATOR",
            Macro(name="TARGET_OS_SIMULATOR", replacement="0"),
        )
        for name in CLANG_FEATURE_TEST_MACROS:
            self.macros.setdefault(name, Macro(name=name, replacement="0"))

    def preprocess(self, code: str, file_path: Optional[str] = None) -> str:
        code = self._strip_leading_compiler_diagnostics(code)
        processed = super().preprocess(code, file_path=file_path)
        processed = self._materialize_project_template_instantiations(processed)
        processed = self._materialize_explicit_template_function_calls(processed)
        processed = self._materialize_explicit_template_struct_instantiations(processed)
        processed = self._lower_struct_member_functions(processed)
        return processed.replace(PRESERVED_INCLUDE_SENTINEL, "#include ")

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

    def _materialize_project_template_instantiations(self, code: str) -> str:
        instantiations = self._find_project_template_instantiations(code)
        if not instantiations:
            return code
        if len(instantiations) > self.max_template_specializations:
            return code

        templates = self._find_template_functions(code)
        if not templates:
            return self._apply_text_replacements(
                code, [(inst.span[0], inst.span[1], "") for inst in instantiations]
            )

        templates_by_name = {template.name: template for template in templates}
        replacements: List[Tuple[int, int, str]] = [
            (inst.span[0], inst.span[1], "") for inst in instantiations
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
        self, code: str, *, work_budget: Optional[object] = None
    ) -> str:
        materialized_names: Dict[Tuple[str, Tuple[str, ...]], str] = {}
        working = code

        while True:
            templates = self._find_template_functions(working)
            if not templates:
                return working

            templates_by_name = {template.name: template for template in templates}
            template_spans = self._find_template_declaration_spans(working)
            if work_budget is not None:
                work_budget.consume(
                    self._template_materialization_scan_work_items(working),
                    offset=templates[0].span[0],
                    length=max(templates[0].span[1] - templates[0].span[0], 0),
                    context="explicit template source scan",
                )
                work_budget.consume(
                    len(templates) * max(1, len(template_spans)),
                    offset=templates[0].span[0],
                    length=max(templates[0].span[1] - templates[0].span[0], 0),
                    context="explicit template reachability scan",
                )
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
            if work_budget is not None:
                first_call_offset = calls[0][2][0] if calls else templates[0].span[0]
                work_budget.consume(
                    len(calls) * max(1, len(templates_by_name)),
                    offset=first_call_offset,
                    context="explicit template call matching",
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

    @staticmethod
    def _template_materialization_scan_work_items(code: str) -> int:
        return max(
            1,
            (len(code) + TEMPLATE_MATERIALIZATION_SCAN_CHARS_PER_WORK_ITEM - 1)
            // TEMPLATE_MATERIALIZATION_SCAN_CHARS_PER_WORK_ITEM,
        )

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
            if work_budget is not None:
                work_budget.consume(
                    self._template_materialization_scan_work_items(working),
                    offset=templates[0].span[0],
                    length=max(templates[0].span[1] - templates[0].span[0], 0),
                    context="explicit template struct source scan",
                )
            specialized = self._find_explicit_struct_specialization_names(working)
            templates_by_name = {
                template.name: template
                for template in templates
                if template.name not in specialized
            }
            if not templates_by_name:
                return working
            excluded_spans = self._find_template_declaration_spans(working)
            if work_budget is not None:
                work_budget.consume(
                    len(templates_by_name) * max(1, len(excluded_spans)),
                    offset=templates[0].span[0],
                    context="explicit template struct declaration matching",
                )
            instantiations = self._find_explicit_template_struct_instantiations(
                working,
                templates_by_name,
                excluded_spans,
            )
            if work_budget is not None:
                first_offset = (
                    instantiations[0][2][0] if instantiations else templates[0].span[0]
                )
                work_budget.consume(
                    len(instantiations) * max(1, len(templates_by_name)),
                    offset=first_offset,
                    context="explicit template struct instantiation matching",
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
                template = templates_by_name[struct_name]
                if not self._template_arguments_satisfy_parameters(
                    template, template_arguments
                ):
                    continue
                specialized_name = materialized_names.get(key)
                if specialized_name is not None:
                    replacements.extend(
                        (span[0], span[1], specialized_name) for span in spans
                    )
                    continue
                if len(materialized_names) >= self.max_template_specializations:
                    # Stay regression-safe instead of raising: leave the residual
                    # `Name<...>` so the existing template-materialization
                    # diagnostic fires exactly as it does today.
                    return code
                specialized_name = self._template_specialization_identifier(
                    struct_name, list(key[1])
                )
                materialized = self._materialize_template_struct_with_name(
                    template, template_arguments, specialized_name
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
        return working

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
        try:
            return self._lower_struct_member_functions_impl(code)
        except MetalStructMethodError:
            raise
        except Exception:
            return code

    def _lower_struct_member_functions_impl(self, code: str) -> str:
        structs = self._find_concrete_struct_definitions(code)
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
            return code

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

        replacements: List[Tuple[int, int, str]] = []
        free_functions: List[str] = []
        for struct in structs_with_methods:
            data_only, lowered = self._render_lowered_struct(code, struct)
            replacements.append((struct.span[0], struct.span[1], data_only))
            free_functions.extend(lowered)

        # Rewrite call sites across the rest of the source (outside the structs
        # we are replacing, so receiver-less internal references are handled when
        # the method body is emitted, not here). Template-method call sites are
        # instantiated here; each unique (struct, method, bindings) instance adds
        # one concrete free function, deduplicated across call sites.
        struct_spans = [struct.span for struct in structs_with_methods]
        # Member-access field resolution must see ALL struct/union spans (so a
        # field declaration is never mistaken for a local variable), and the
        # names of every struct/union type (so `Type var;` locals of a method-less
        # carrier are tracked).
        all_struct_spans = [struct.span for struct in structs]
        all_struct_names = {struct.name for struct in structs}
        instantiated_template_functions: Dict[str, str] = {}
        call_replacements = self._rewrite_struct_member_call_sites(
            code,
            struct_names,
            methods_by_struct,
            template_methods_by_struct,
            operator_call_structs,
            struct_spans,
            structs_by_name,
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
        return rewritten

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
        definitions: List[_MetalStructDefinition] = []
        for match in re.finditer(r"\b(?:struct|class|union)\s+", code):
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
            (
                data_member_names,
                data_member_types,
                methods,
                template_methods,
            ) = self._split_struct_body(name, body, body_start + 1)
            definitions.append(
                _MetalStructDefinition(
                    name=name,
                    span=(start, span_end),
                    body_span=(body_start + 1, body_end_after - 1),
                    data_member_names=data_member_names,
                    methods=methods,
                    has_operator_call=any(m.is_operator_call for m in methods)
                    or any(m.is_operator_call for m in template_methods),
                    template_methods=template_methods,
                    data_member_types=data_member_types,
                )
            )
        return definitions

    def _split_struct_body(
        self, struct_name: str, body: str, body_offset: int
    ) -> Tuple[
        Set[str], Dict[str, str], List[_MetalStructMethod], List[_MetalStructMethod]
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
                            template_methods.append(template_method)
                    i = method_body_end if method_body_end is not None else n
                elif semicolon is not None:
                    i = semicolon + 1
                else:
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
                    i = brace_end
                    continue
                declaration = body[i:decl_semicolon]
                name = self._declared_data_member_name(declaration)
                if name:
                    data_member_names.add(name)
                    self._record_data_member_type(data_member_types, name, declaration)
                i = decl_semicolon + 1
                continue
            if semicolon is None:
                break
            # A declaration terminated by `;`. It may still be a method
            # PROTOTYPE (declarator + params + ;) with no body — those have no
            # definition to lower, so record any data member name otherwise.
            declaration = body[i:semicolon]
            if not self._declaration_is_method_prototype(declaration):
                name = self._declared_data_member_name(declaration)
                if name:
                    data_member_names.add(name)
                    self._record_data_member_type(data_member_types, name, declaration)
            i = semicolon + 1
        return data_member_names, data_member_types, methods, template_methods

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
            return None
        method = self._parse_struct_method(
            struct_name, body, angle_end + 1, method_body_start
        )
        if method is None:
            return None
        method.template_parameters = template_parameters
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
        "short": 2,
        "ushort": 2,
        "half": 2,
        "int": 4,
        "uint": 4,
        "float": 4,
        "long": 8,
        "ulong": 8,
        "double": 8,
        "size_t": 8,
    }
    # Integral scalar element types for `is_integral_v<T>`.
    _METAL_INTEGRAL_SCALAR_TYPES: Set[str] = {
        "bool",
        "char",
        "uchar",
        "short",
        "ushort",
        "int",
        "uint",
        "long",
        "ulong",
        "size_t",
    }

    class _UnrecognizedConstraint(Exception):
        """A SFINAE constraint outside the small recognized set — clean-fail."""

    def _scalar_and_width(self, type_text: str) -> Optional[Tuple[str, int]]:
        # Decompose a scalar/vector type into (scalar base, component width). A
        # trailing 2/3/4 on a known scalar denotes a vector; otherwise width 1.
        # Returns None when the base is not a recognized scalar.
        text = self._normalize_inferred_type(type_text)
        if not text or " " in text:
            return None
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
        decomposed = self._scalar_and_width(type_text)
        if decomposed is None:
            return None
        base, width = decomposed
        return self._METAL_SCALAR_TYPE_SIZES[base] * width

    def _is_integral_concrete_type(self, type_text: str) -> Optional[bool]:
        decomposed = self._scalar_and_width(type_text)
        if decomposed is None:
            return None
        base, _width = decomposed
        return base in self._METAL_INTEGRAL_SCALAR_TYPES

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
        overloads: List[_MetalStructMethod],
        bindings: Dict[str, str],
    ) -> Optional[_MetalStructMethod]:
        # Pick the unique overload whose SFINAE constraints ALL evaluate true for
        # `bindings`. Returns None when zero or more than one overload is enabled,
        # or when any constraint is unrecognized — every such case is a clean
        # failure (never guess / never mis-select). An overload with NO
        # constraints is considered always-enabled (a plain template method),
        # which keeps the single-overload, no-SFINAE path unchanged.
        enabled: List[_MetalStructMethod] = []
        for overload in overloads:
            try:
                if all(
                    self._evaluate_template_constraint(constraint, bindings)
                    for constraint in overload.template_constraints
                ):
                    enabled.append(overload)
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

        # Detect and strip a leading `static` qualifier from the return type.
        is_static = bool(re.search(r"(^|\s)static(\s|$)", signature_prefix))
        return_type = re.sub(r"\bstatic\b", " ", signature_prefix)
        # Strip storage/qualifier keywords that are meaningless on a free
        # function return type while preserving the actual type tokens.
        return_type = re.sub(r"\binline\b", " ", return_type)
        return_type = re.sub(r"\bconstexpr\b", " ", return_type)
        return_type = re.sub(r"\s+", " ", return_type).strip()
        if not return_type:
            return None

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
        )

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
        # before it is a function prototype rather than a data member.
        paren_start = self._function_parameter_start(declaration)
        if paren_start is None:
            return False
        paren_end = self._find_matching_delimiter(declaration, paren_start, "(", ")")
        if paren_end is None:
            return False
        before = declaration[:paren_start].rstrip()
        if re.search(r"\boperator\s*\(\s*\)\s*$", before):
            return True
        return re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*$", before) is not None

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
        self, code: str, struct: _MetalStructDefinition
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
            self._emit_free_function(struct, method) for method in struct.methods
        ]
        return data_only, free_functions

    def _emit_free_function(
        self, struct: _MetalStructDefinition, method: _MetalStructMethod
    ) -> str:
        # Emit `RetType S__m(S self, <params>) { body' }` for an instance method,
        # or `RetType S__m(<params>) { body' }` for a static method. References to
        # the struct's data members inside the body are rewritten to `self.x`.
        rewritten_body = self._rewrite_method_body(struct, method)
        params = method.parameters.strip()
        if method.is_static:
            new_params = params if params and params != "void" else ""
        else:
            self_param = f"{struct.name} self"
            if params and params != "void":
                new_params = f"{self_param}, {params}"
            else:
                new_params = self_param
        return (
            f"{method.return_type} {method.free_name}({new_params}) "
            f"{{{rewritten_body}}}"
        )

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
        variable_types = self._collect_struct_variable_types(
            code, struct_names, struct_spans
        )
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
        # Struct-typed locals for EVERY struct/union type (used for `obj.member`
        # inference), resolved over the full struct span set so member-access
        # works for data carriers without methods.
        field_variable_types = self._collect_struct_variable_types(
            code, all_struct_names, all_struct_spans
        )
        buffer_element_types = self._collect_buffer_element_types(
            code, all_struct_spans
        )
        local_variable_types = self._collect_local_variable_types(
            code, all_struct_spans
        )
        # Fold the enclosing functions' parameters into the same position-ordered
        # maps so a call argument that is a parameter (or a subscript of one) is
        # inferable.
        self._collect_function_parameter_types(
            code,
            all_struct_spans,
            buffer_element_types,
            local_variable_types,
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
            span = self._containing_span(i, struct_spans)
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
        buffer_element_types: Dict[str, List[Tuple[int, str]]],
        local_variable_types: Dict[str, List[Tuple[int, str]]],
        instantiated_template_functions: Dict[str, str],
        field_variable_types: Optional[Dict[str, List[Tuple[int, str]]]] = None,
        field_structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
    ) -> Optional[Tuple[int, str]]:
        # `field_variable_types` / `field_structs_by_name` cover EVERY struct/union
        # type (including method-less data carriers) and are used only to resolve
        # `obj.member` / `obj.member[i]` call arguments; call REWRITING still uses
        # the method-driven `variable_types` / `structs_by_name`.
        if field_variable_types is None:
            field_variable_types = variable_types
        if field_structs_by_name is None:
            field_structs_by_name = structs_by_name
        # Member access on a previous object (`a.var.m(...)`) is not a struct
        # variable we tracked; skip when the identifier is a member access.
        if self._is_member_identifier_context(code, ident_start):
            return None

        j = ident_end
        while j < len(code) and code[j].isspace():
            j += 1
        if j >= len(code):
            return None

        # Qualified static call: `S::m(args)` -> `S__m(args)`.
        if ident in struct_names and code[j : j + 2] == "::":
            k = j + 2
            while k < len(code) and code[k].isspace():
                k += 1
            member, consumed = self._read_identifier(code, k)
            if not member:
                return None
            after = k + consumed
            while after < len(code) and code[after].isspace():
                after += 1
            if after >= len(code) or code[after] != "(":
                return None
            method = methods_by_struct.get(ident, {}).get(member)
            if method is not None:
                return after, method.free_name
            # A static TEMPLATE member call `S::m(args)`: instantiate from args.
            template_overloads = template_methods_by_struct.get(ident, {}).get(member)
            if template_overloads:
                rewrite = self._instantiate_template_member_call(
                    code,
                    structs_by_name[ident],
                    template_overloads,
                    receiver=None,
                    arg_open=after,
                    buffer_element_types=buffer_element_types,
                    local_variable_types=local_variable_types,
                    instantiated_template_functions=instantiated_template_functions,
                    structs_by_name=field_structs_by_name,
                    variable_types=field_variable_types,
                    template_methods_by_struct=template_methods_by_struct,
                )
                if rewrite is not None:
                    return rewrite
            return None

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
            after = k + consumed
            while after < len(code) and code[after].isspace():
                after += 1
            if after >= len(code) or code[after] != "(":
                # A data-member access (`var.field`) — leave untouched.
                return None
            arg_open = after
            method = methods_by_struct.get(struct_type, {}).get(member)
            if method is not None:
                return self._build_instance_call_rewrite(code, ident, method, arg_open)
            # An instance TEMPLATE member call `var.m(args)`.
            template_overloads = template_methods_by_struct.get(struct_type, {}).get(
                member
            )
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
                )
            return None

        # Functor call: `var(args)` -> `S__operator_call(var, ...)`.
        if code[j] == "(" and struct_type in operator_call_structs:
            method = methods_by_struct.get(struct_type, {}).get("operator()")
            if method is not None:
                return self._build_instance_call_rewrite(code, ident, method, j)
            # A template `operator()` functor call `var(args)`.
            template_overloads = template_methods_by_struct.get(struct_type, {}).get(
                "operator()"
            )
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
                )
            return None

        return None

    def _instantiate_template_member_call(
        self,
        code: str,
        struct: _MetalStructDefinition,
        overloads: List[_MetalStructMethod],
        receiver: Optional[str],
        arg_open: int,
        buffer_element_types: Dict[str, List[Tuple[int, str]]],
        local_variable_types: Dict[str, List[Tuple[int, str]]],
        instantiated_template_functions: Dict[str, str],
        structs_by_name: Optional[Dict[str, _MetalStructDefinition]] = None,
        variable_types: Optional[Dict[str, List[Tuple[int, str]]]] = None,
        template_methods_by_struct: Optional[
            Dict[str, Dict[str, List[_MetalStructMethod]]]
        ] = None,
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

        # Infer the concrete type of every call argument; one un-inferable
        # argument is a clean failure.
        concrete_argument_types: List[str] = []
        for argument in call_arguments:
            inferred = self._infer_argument_type(
                argument,
                buffer_view,
                local_view,
                struct_field_types,
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
        concrete_argument_types: List[str],
        signature: str,
        instantiated_template_functions: Dict[str, str],
        template_methods_by_struct: Optional[
            Dict[str, Dict[str, List[_MetalStructMethod]]]
        ],
    ) -> str:
        # Bind, select the enabled overload, materialize it (recursively lowering
        # any internal template-member calls in its body), and return the
        # concrete free-function name. Raises MetalStructMethodError on any
        # unresolved/ambiguous case.
        representative = overloads[0]
        bindings = self._bind_template_method_parameters(
            representative, concrete_argument_types
        )
        if bindings is None:
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

        selected = self._select_constrained_overload(overloads, bindings)
        if selected is None:
            raise MetalStructMethodError(
                "Cannot lower template member method "
                f"'{struct.name}::{representative.name}': no unique overload is "
                f"enabled by its SFINAE constraints for bindings {bindings} "
                f"(zero or several matched, or a constraint is unsupported). "
                f"Requested call: {signature}.",
                struct_name=struct.name,
                method_name=representative.name,
                requested_signature=signature,
                suggested_action=(
                    "specialize the method manually, or restrict the call to a "
                    "type whose constraint is recognized (sizeof / is_integral_v)"
                ),
            )

        ordered_arguments = [bindings[name] for name in selected.template_parameters]
        free_name = self._template_member_free_name(struct, selected, ordered_arguments)
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
        concrete_argument_types: List[str],
    ) -> Optional[Dict[str, str]]:
        # Match each declared method-parameter type (which may contain the
        # template parameters) against the inferred concrete argument type and
        # collect consistent bindings. Returns None if any template parameter is
        # left unbound or a parameter binds inconsistently.
        template_parameter_set = set(method.template_parameters)
        # `method.parameters` is the already-extracted parameter list (the
        # `operator()` declarator is handled at parse time), so split it directly
        # rather than reparsing a synthesized header — a synthesized
        # `operator()` header would confuse the first-paren parameter finder.
        declared_parameter_types = [
            normalized
            for normalized in (
                self._normalize_function_parameter_type_text(parameter)
                for parameter in self._split_top_level_commas(method.parameters)
            )
            if normalized and normalized != "void"
        ]
        if len(declared_parameter_types) != len(concrete_argument_types):
            return None
        # Bind each parameter into its OWN dict and merge with explicit conflict
        # detection. `_infer_template_parameter_bindings_from_type` silently
        # SKIPS a parameter whose new value conflicts with an existing binding;
        # that is the wrong behavior here — a template parameter that the call
        # site forces to two different concrete types (`pick(float, int)` for
        # `T pick(T, T)`) must clean-fail, not silently keep the first guess.
        bindings: Dict[str, str] = {}
        for declared_type, concrete_type in zip(
            declared_parameter_types, concrete_argument_types
        ):
            local_bindings: Dict[str, str] = {}
            self._infer_template_parameter_bindings_from_type(
                declared_type,
                self._normalize_inferred_type(concrete_type),
                template_parameter_set,
                local_bindings,
            )
            for name, value in local_bindings.items():
                existing = bindings.get(name)
                if existing is not None and existing != value:
                    return None
                bindings[name] = value
        # Every template parameter must be bound; a partially-bound method is a
        # clean failure (we never guess a default for an inferred call).
        for name in method.template_parameters:
            if name not in bindings:
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
            re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
            for value in ordered_arguments
        ).strip("_")
        if not suffix:
            return base
        return f"{base}__{suffix}"

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
    ) -> str:
        # Instantiate the template member method by substituting the bound
        # template parameters into copies of its return type, parameters and
        # body, then emit a concrete free function reusing the non-template
        # lowering machinery (member references resolved to `self.x`).
        instantiated_return = self._replace_identifiers(method.return_type, bindings)
        instantiated_return = re.sub(r"\s+", " ", instantiated_return).strip()
        instantiated_parameters = self._replace_identifiers(method.parameters, bindings)
        instantiated_body = self._replace_identifiers(method.body, bindings)
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
            )
        concrete_method = _MetalStructMethod(
            name=method.name,
            free_name=free_name,
            is_static=method.is_static,
            is_operator_call=method.is_operator_call,
            return_type=instantiated_return,
            parameters=instantiated_parameters.strip(),
            parameter_names=self._parameter_identifier_names(instantiated_parameters),
            body=instantiated_body,
            span=method.span,
        )
        return self._emit_free_function(struct, concrete_method)

    def _lower_internal_template_member_calls(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        instantiated_parameters: str,
        instantiated_body: str,
        instantiated_template_functions: Dict[str, str],
        template_methods_by_struct: Dict[str, Dict[str, List[_MetalStructMethod]]],
    ) -> str:
        # Rewrite receiver-less calls to OTHER template member methods of the same
        # struct inside an already-substituted method body. Argument types are
        # inferred from the (now concrete) parameter types and the body's local
        # declarations; the existing argument-inference + overload selection is
        # reused, so an un-inferable / ambiguous internal call clean-fails exactly
        # like a top-level call. The instance receiver is `self` (the free
        # function's first parameter); a static sibling takes no receiver.
        sibling_overloads = template_methods_by_struct.get(struct.name, {})
        # A method whose body has no sibling-template call is returned unchanged.
        if not sibling_overloads:
            return instantiated_body
        # Build flat name->type views for the body scope from the concrete
        # parameters plus body-local declarations.
        local_view: Dict[str, str] = {}
        buffer_view: Dict[str, str] = {}
        for parameter in self._split_top_level_commas(instantiated_parameters):
            if not parameter.strip():
                continue
            name = self._declared_data_member_name(parameter)
            if not name:
                continue
            element = self._pointer_or_array_parameter_element_type(parameter)
            if element is not None:
                buffer_view[name] = element
                continue
            scalar = self._normalize_inferred_type(
                self._normalize_function_parameter_type_text(parameter)
            )
            if scalar:
                local_view[name] = scalar
        for statement in self._iter_simple_declarations(instantiated_body):
            name = self._declared_local_name(statement)
            if not name:
                continue
            element_type = self._data_member_element_type(statement)
            if element_type:
                local_view.setdefault(name, element_type)

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

    def _try_rewrite_internal_template_call(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        ident: str,
        ident_end: int,
        body: str,
        sibling_overloads: Dict[str, List[_MetalStructMethod]],
        local_view: Dict[str, str],
        buffer_view: Dict[str, str],
        instantiated_template_functions: Dict[str, str],
        template_methods_by_struct: Dict[str, Dict[str, List[_MetalStructMethod]]],
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
            or (previous >= 1 and body[previous - 1 : previous + 1] == "->")
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
                )
            return None

        template_overloads = sibling_overloads.get(ident)
        concrete_siblings = [m for m in struct.methods if m.name == ident]
        if not template_overloads and not concrete_siblings:
            return None
        j = ident_end
        while j < len(body) and body[j].isspace():
            j += 1
        if j >= len(body) or body[j] != "(":
            return None
        arg_open = j
        arg_close = self._find_matching_delimiter(body, arg_open, "(", ")")
        if arg_close is None:
            return None
        raw_args = body[arg_open + 1 : arg_close]
        args = raw_args.strip()

        # A CONCRETE sibling has a fixed signature; lower it directly (no overload
        # selection / argument typing needed). When both a concrete and a template
        # sibling share the name, prefer the concrete one only if there is exactly
        # one — otherwise fall through to the template path / clean-fail.
        if concrete_siblings and not template_overloads:
            if len(concrete_siblings) != 1:
                return None
            concrete = concrete_siblings[0]
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
        concrete_argument_types: List[str] = []
        for argument in call_arguments:
            inferred = self._infer_argument_type(argument, buffer_view, local_view, {})
            if inferred is None:
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

        free_name = self._instantiate_template_member_overload(
            struct,
            overloads,
            concrete_argument_types,
            signature,
            instantiated_template_functions,
            template_methods_by_struct,
        )
        if representative.is_static:
            replacement = f"{free_name}({args})" if args else f"{free_name}()"
        elif args:
            replacement = f"{free_name}(self, {args})"
        else:
            replacement = f"{free_name}(self)"
        return arg_close + 1, replacement

    def _rewrite_internal_operator_call(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        empty_paren_open: int,
        body: str,
        sibling_overloads: Dict[str, List[_MetalStructMethod]],
        local_view: Dict[str, str],
        buffer_view: Dict[str, str],
        instantiated_template_functions: Dict[str, str],
        template_methods_by_struct: Dict[str, Dict[str, List[_MetalStructMethod]]],
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
        concrete_argument_types: List[str] = []
        for argument in call_arguments:
            inferred = self._infer_argument_type(argument, buffer_view, local_view, {})
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
        )
        replacement = f"{free_name}(self, {args})" if args else f"{free_name}(self)"
        return arg_close + 1, replacement

    def _build_template_call_rewrite(
        self,
        code: str,
        receiver: Optional[str],
        free_name: str,
        arg_open: int,
        arg_close: int,
    ) -> Tuple[int, str]:
        args = code[arg_open + 1 : arg_close].strip()
        if receiver is None:
            # Static template member call: no `self` receiver.
            replacement = f"{free_name}({args})" if args else f"{free_name}()"
        elif args:
            replacement = f"{free_name}({receiver}, {args})"
        else:
            replacement = f"{free_name}({receiver})"
        return arg_close + 1, replacement

    def _build_instance_call_rewrite(
        self,
        code: str,
        receiver: str,
        method: _MetalStructMethod,
        arg_open: int,
    ) -> Optional[Tuple[int, str]]:
        arg_close = self._find_matching_delimiter(code, arg_open, "(", ")")
        if arg_close is None:
            return None
        args = code[arg_open + 1 : arg_close].strip()
        if args:
            replacement = f"{method.free_name}({receiver}, {args})"
        else:
            replacement = f"{method.free_name}({receiver})"
        return arg_close + 1, replacement

    def _collect_struct_variable_types(
        self,
        code: str,
        struct_names: Set[str],
        struct_spans: List[Tuple[int, int]],
    ) -> Dict[str, List[Tuple[int, str]]]:
        # Map each local variable name to the POSITION-ORDERED list of struct
        # declarations that introduce it (`S var;`, `S var = ...;`, `S var(...)`)
        # outside the struct definitions. Returning every declaration with its
        # source offset (instead of a single "last wins" entry) makes resolution
        # deterministic and lets a call site bind to the NEAREST PRECEDING
        # declaration — so two kernels in one file that each declare `op` of a
        # different concrete functor type resolve `op` correctly per kernel
        # rather than collapsing to one PYTHONHASHSEED-dependent winner.
        declarations: Dict[str, List[Tuple[int, str]]] = {}
        # Iterate struct names in a STABLE (sorted) order so the per-name lists,
        # before sorting, do not depend on set iteration order; the final sort by
        # position is what callers rely on, but a stable scan keeps ties stable.
        for struct_name in sorted(struct_names):
            pattern = re.compile(
                rf"\b{re.escape(struct_name)}\s+([A-Za-z_][A-Za-z0-9_]*)\s*"
                rf"(?=[;={{(])"
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
                declarations.setdefault(match.group(1), []).append(
                    (match.start(), struct_name)
                )
        for entries in declarations.values():
            entries.sort(key=lambda item: item[0])
        return declarations

    def _resolve_declared_type_at(
        self,
        declarations: Dict[str, List[Tuple[int, str]]],
        name: str,
        position: int,
    ) -> Optional[str]:
        # Resolve a name to the type of its NEAREST declaration appearing at or
        # before `position`; falling back to the FIRST declaration when the name
        # is only declared later (a best-effort that suits forward references in
        # the flat kernels we translate). Deterministic regardless of hash seed.
        entries = declarations.get(name)
        if not entries:
            return None
        best: Optional[str] = None
        for decl_position, declared_type in entries:
            if decl_position <= position:
                best = declared_type
            else:
                break
        if best is not None:
            return best
        return entries[0][1]

    def _flatten_types_at(
        self,
        declarations: Dict[str, List[Tuple[int, str]]],
        position: int,
    ) -> Dict[str, str]:
        # Flatten a position-ordered declaration map to a name -> type view valid
        # at `position` (nearest preceding declaration wins per name).
        flattened: Dict[str, str] = {}
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
        # Build a `variable name -> {field name -> field type}` view at the call
        # site: for each struct-typed local in scope, expose its declaring
        # struct's field types so a `obj.member` / `obj.member[i]` argument can be
        # element-typed. Empty when struct metadata is unavailable.
        if not variable_types or not structs_by_name:
            return {}
        field_types: Dict[str, Dict[str, str]] = {}
        for name in variable_types:
            struct_type = self._resolve_declared_type_at(variable_types, name, position)
            if struct_type is None:
                continue
            struct = structs_by_name.get(struct_type)
            if struct is None or not struct.data_member_types:
                continue
            field_types[name] = struct.data_member_types
        return field_types

    # ------------------------------------------------------------------ #
    # Conservative call-argument type inference (template member methods). #
    # ------------------------------------------------------------------ #
    # The rules below NEVER guess: each returns None for any expression shape it
    # does not recognize, and the caller turns a single un-inferable argument
    # into a clean translation failure rather than emitting a dangling call.

    def _collect_buffer_element_types(
        self, code: str, struct_spans: List[Tuple[int, int]]
    ) -> Dict[str, List[Tuple[int, str]]]:
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
        element_types: Dict[str, List[Tuple[int, str]]] = {}

        def record(name: str, element_type: str, position: int) -> None:
            normalized = self._normalize_inferred_type(element_type)
            if not normalized:
                return
            element_types.setdefault(name, []).append((position, normalized))

        pointer_pattern = re.compile(
            r"\b(?:device|constant|threadgroup|thread)\b"
            r"(?P<type>[^;,()\[\]{}]*?)\*\s*"
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
        )
        for match in pointer_pattern.finditer(code):
            if self._containing_span(match.start(), struct_spans) is not None:
                continue
            record(match.group("name"), match.group("type"), match.start())

        # Array declarations: `<type> name[extent]` for locals and parameters.
        # The leading anchor keeps consecutive declarations from cannibalizing
        # each other; the type group allows address-space/cv qualifiers. Only a
        # RECOGNIZED element type (scalar/vector or a known struct/union) is
        # accepted so a non-declaration that happens to match the shape — e.g.
        # `return totals[i]` (type token `return`) or `else block[i]` — never
        # records a bogus element type (the never-guess contract).
        recognized_aggregates = self._aggregate_type_names(code, struct_spans)
        array_pattern = re.compile(
            r"(?:(?<=[;{}(,])|^)\s*"
            r"(?P<type>(?:const\s+|constexpr\s+|volatile\s+|thread\s+|"
            r"threadgroup\s+|device\s+|constant\s+)*"
            r"[A-Za-z_][A-Za-z0-9_]*)\s+"
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\[",
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
            record(match.group("name"), normalized, match.start())

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
            r"[A-Za-z_][A-Za-z0-9_]*)\s+"
            r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?=[=;])",
            re.MULTILINE,
        )
        for match in pattern.finditer(code):
            if self._containing_span(match.start(), struct_spans) is not None:
                continue
            raw_type = match.group("type")
            normalized = self._normalize_inferred_type(raw_type)
            if (
                normalized not in self._METAL_SCALAR_VECTOR_TYPES
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
        buffer_element_types: Dict[str, List[Tuple[int, str]]],
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
            paren_start = self._function_parameter_start(header)
            if paren_start is None:
                continue
            paren_end = self._find_matching_delimiter(header, paren_start, "(", ")")
            if paren_end is None:
                continue
            parameter_text = header[paren_start + 1 : paren_end]
            body_start = function.body_span[0]
            for parameter in self._split_top_level_commas(parameter_text):
                if not parameter.strip():
                    continue
                name = self._declared_data_member_name(parameter)
                if not name:
                    continue
                element = self._pointer_or_array_parameter_element_type(parameter)
                if element is not None:
                    buffer_element_types.setdefault(name, []).append(
                        (body_start, element)
                    )
                    continue
                scalar = self._normalize_inferred_type(
                    self._normalize_function_parameter_type_text(parameter)
                )
                if (
                    scalar in self._METAL_SCALAR_VECTOR_TYPES
                    or scalar in recognized_aggregates
                ):
                    local_variable_types.setdefault(name, []).append(
                        (body_start, scalar)
                    )
        for entries in buffer_element_types.values():
            entries.sort(key=lambda item: item[0])
        for entries in local_variable_types.values():
            entries.sort(key=lambda item: item[0])

    def _pointer_or_array_parameter_element_type(self, parameter: str) -> Optional[str]:
        # Element type of a pointer (`device T* p`) or array (`T p[N]`) parameter,
        # or None when the parameter is neither (so the caller treats it as a
        # plain value parameter). Conservative: returns None unless a `*` or `[`
        # declarator is present.
        text = self._strip_top_level_default_value(parameter)
        text = self._strip_metal_attributes(text).strip()
        if not text:
            return None
        if "[" in text:
            type_text = text[: text.find("[")]
            type_text = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\s*$", "", type_text).strip()
            return self._normalize_inferred_type(type_text) or None
        star = text.rfind("*")
        if star != -1:
            type_text = text[:star]
            return self._normalize_inferred_type(type_text) or None
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

    def _infer_argument_type(
        self,
        argument: str,
        buffer_element_types: Dict[str, str],
        local_variable_types: Dict[str, str],
        struct_field_types: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Optional[str]:
        # Conservatively infer the concrete type of a call-argument expression.
        # Returns None (un-inferable) for anything outside the recognized shapes.
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

        # Cast: `static_cast<T>(expr)` -> T.
        static_cast = re.match(r"static_cast\s*<(?P<type>.+)>\s*\(", expr, re.DOTALL)
        if static_cast is not None:
            angle_end = self._find_matching_angle(expr, expr.find("<"))
            if angle_end is not None:
                return self._normalize_inferred_type(
                    expr[expr.find("<") + 1 : angle_end]
                )

        # Literal types.
        literal_type = self._infer_literal_type(expr)
        if literal_type is not None:
            return literal_type

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

        # Bare local variable -> its declared type.
        if IDENTIFIER_RE.fullmatch(expr) and expr in local_variable_types:
            return local_variable_types[expr]

        # Member access `obj.member` -> the field type of `member` in `obj`'s
        # struct (non-subscript). Pointer fields are not value-typeable as a bare
        # access, so a trailing `*` marker is rejected here.
        member_field_type = self._infer_member_access_type(expr, struct_field_types)
        if member_field_type is not None:
            return member_field_type

        return None

    def _infer_subscript_base_element_type(
        self,
        base: str,
        buffer_element_types: Dict[str, str],
        struct_field_types: Optional[Dict[str, Dict[str, str]]],
    ) -> Optional[str]:
        # Element type of the subscript base `base[...]`:
        #   * bare name: a buffer/array element type (`buf[i]`, `totals[i]`).
        #   * member access `obj.member`: the element type of the struct field
        #     `member` (array or pointer field) of struct local `obj`.
        if IDENTIFIER_RE.fullmatch(base):
            return buffer_element_types.get(base)
        field_type = self._struct_member_field_type(base, struct_field_types)
        if field_type is None:
            return None
        # A subscript yields the field's element type; strip a single pointer
        # marker if the field type recorded one. Array fields already record the
        # element type, so the value is returned as-is.
        return field_type.rstrip("*").strip() or None

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

    def _struct_member_field_type(
        self,
        access: str,
        struct_field_types: Optional[Dict[str, Dict[str, str]]],
    ) -> Optional[str]:
        # Resolve a single-level `obj.member` access to `member`'s declared field
        # type, using the per-variable struct field-type view. Conservative:
        # only a bare `identifier.identifier` shape is recognized.
        if not struct_field_types:
            return None
        match = re.fullmatch(
            r"(?P<obj>[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*"
            r"(?P<member>[A-Za-z_][A-Za-z0-9_]*)",
            access,
        )
        if match is None:
            return None
        fields = struct_field_types.get(match.group("obj"))
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
        template_type_traits = self._find_template_type_traits(code)
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
                    template_type_traits=template_type_traits,
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
        # A struct/class definition must be terminated with a semicolon; the
        # captured template source ends at the closing brace, so restore it.
        materialized = materialized.rstrip()
        if not materialized.endswith(";"):
            materialized += ";"
        materialized += "\n"
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
        return calls

    def _find_explicit_template_struct_instantiations(
        self,
        code: str,
        struct_templates_by_name: Dict[str, _MetalTemplateStruct],
        excluded_spans: List[Tuple[int, int]],
    ) -> List[Tuple[str, List[str], Tuple[int, int]]]:
        # Struct counterpart of _find_explicit_template_function_calls: locate
        # concrete `StructName<args>` TYPE references (variable declarations, base
        # classes, casts, nested template arguments, ...). Unlike a function call
        # the reference need not be followed by "(", so that trailing guard is
        # dropped. References inside template declarations (excluded_spans) are
        # skipped: they are only materialized once their enclosing template is.
        instantiations: List[Tuple[str, List[str], Tuple[int, int]]] = []
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
                span_start = self._scoped_identifier_start(code, i)
                instantiations.append(
                    (ident, template_arguments, (span_start, angle_end + 1))
                )
                i = angle_end + 1
                continue
            i += 1
        return instantiations

    def _find_explicit_struct_specialization_names(self, code: str) -> Set[str]:
        # Names of struct/class templates that carry an explicit specialization
        # (`template <> struct Name<...>`). Materializing the primary template for
        # such a name would ignore the specialization, so these are left to the
        # future specialization-aware path and fall back to today's diagnostic.
        names: Set[str] = set()
        for match in re.finditer(
            r"\btemplate\s*<\s*>\s*(?:struct|class)\s+"
            r"(?P<name>[A-Za-z_][A-Za-z0-9_:]*)\s*<",
            code,
        ):
            names.add(match.group("name").split("::")[-1])
        return names

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
        cached = self._containing_span_cache
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
            self._containing_span_cache = cached

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
        # iff it immediately follows an identifier character — the formatting
        # `enable_if_t<` / `is_integral_v<` uses no separating space, whereas a
        # comparison is written `sizeof(T) < 8` (preceded by `)`/space). This is
        # conservative: an unexpected spaced template-open would fail to balance
        # and the method would simply be left unrecognized (clean-fail), never
        # mis-lowered.
        previous = index - 1
        if previous < 0:
            return False
        prev = code[previous]
        return prev.isalnum() or prev == "_"

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
            is_variadic = "..." in declarator
            declarator_no_pack = declarator.replace("...", " ")
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
                records.append(
                    _TemplateParameter(
                        name=tokens[-1],
                        is_type_parameter=False,
                        is_variadic=is_variadic,
                        default=default,
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
        # ignoring `=` inside angles/parens/brackets and the `==`/`<=`/`>=`/`!=`
        # comparison operators that appear in SFINAE constraints. Returns
        # (declarator, default) or None when there is no default.
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        angle_depth = 0
        i = 0
        n = len(parameter)
        while i < n:
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
        )
        return trait_resolved or resolved

    def _resolve_template_type_trait(
        self,
        type_text: str,
        traits: Dict[str, Dict[str, object]],
    ) -> Optional[str]:
        text = str(type_text or "").strip()
        match = re.fullmatch(
            r"(?:typename\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_:]*)\s*"
            r"<(?P<args>.*)>\s*::\s*type",
            text,
            re.DOTALL,
        )
        if match is None:
            return None
        trait = traits.get(match.group("name").split("::")[-1])
        if not trait:
            return None
        arguments = [
            self._normalize_template_argument_text(argument)
            for argument in self._split_top_level_commas(match.group("args"))
        ]
        specializations = trait.get("specializations", {})
        specialized = specializations.get(tuple(arguments))
        if isinstance(specialized, str) and specialized:
            return specialized
        parameters = trait.get("parameters", [])
        default_type = trait.get("default")
        if not isinstance(default_type, str) or len(arguments) < len(parameters):
            return None
        substitutions = dict(zip(parameters, arguments))
        resolved = self._replace_identifiers(default_type, substitutions)
        return self._normalize_template_argument_text(resolved)

    def _find_template_type_traits(self, code: str) -> Dict[str, Dict[str, object]]:
        traits: Dict[str, Dict[str, object]] = {}
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
            )
            pos = body_end
        return traits

    def _record_template_type_trait(
        self,
        traits: Dict[str, Dict[str, object]],
        parameter_text: str,
        header: str,
        body: str,
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
        name = header_match.group(1).split("::")[-1]
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
            trait.setdefault("specializations", {})[arguments] = alias_type
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
        self, source: str, old_name: str, new_name: str
    ) -> str:
        pattern = re.compile(rf"\b(struct|class)\s+{re.escape(old_name)}\b")
        match = pattern.search(source)
        if match is None:
            return source
        return (
            source[: match.start()]
            + f"{match.group(1)} {new_name}"
            + source[match.end() :]
        )

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
        if not in_expression and self._has_incomplete_function_macro_call(text):
            return text
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

    def _join_multiline_function_macro_call(self, lines: List[str], start: int):
        return lines[start], 1

    def _has_incomplete_function_macro_call(self, text: str) -> bool:
        i = 0
        while i < len(text):
            if text[i] in "\"'":
                _literal, consumed = self._read_string(text, i)
                i += consumed
                continue
            if text.startswith("//", i):
                return False
            if text.startswith("/*", i):
                end = text.find("*/", i + 2)
                if end == -1:
                    return False
                i = end + 2
                continue
            if text[i].isalpha() or text[i] == "_":
                ident, consumed = self._read_identifier(text, i)
                macro = self.macros.get(ident)
                i += consumed
                if macro is None or not macro.is_function_like():
                    continue
                j = i
                while j < len(text) and text[j].isspace():
                    j += 1
                if (
                    j < len(text)
                    and text[j] == "("
                    and not self._call_closes_on_line(text, j)
                ):
                    return True
                continue
            i += 1
        return False

    def _call_closes_on_line(self, text: str, start: int) -> bool:
        depth = 0
        i = start
        while i < len(text):
            if text[i] in "\"'":
                _literal, consumed = self._read_string(text, i)
                i += consumed
                continue
            if text.startswith("//", i):
                return False
            if text.startswith("/*", i):
                end = text.find("*/", i + 2)
                if end == -1:
                    return False
                i = end + 2
                continue
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    return True
            i += 1
        return False

    def _handle_include(self, rest: str, file_path: Optional[str]) -> Optional[str]:
        included = super()._handle_include(rest, file_path)
        if included is not None or self.strict:
            return included
        include_target = rest.strip()
        if include_target.startswith("<") and include_target.endswith(">"):
            return f"{PRESERVED_INCLUDE_SENTINEL}{include_target}"
        return None
