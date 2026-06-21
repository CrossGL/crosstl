"""Preprocessor support for Metal source imports."""

import operator
import os
import re
from dataclasses import dataclass, field
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


@dataclass
class _MetalStructDefinition:
    """A concrete (non-template) struct/class with its members split out."""

    name: str
    span: Tuple[int, int]
    body_span: Tuple[int, int]
    data_member_names: Set[str]
    methods: List[_MetalStructMethod]
    has_operator_call: bool


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

    def _materialize_explicit_template_function_calls(self, code: str) -> str:
        materialized_names: Dict[Tuple[str, Tuple[str, ...]], str] = {}
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

    def _materialize_explicit_template_struct_instantiations(self, code: str) -> str:
        # Struct counterpart of _materialize_explicit_template_function_calls
        # (issue #1354). Replaces concrete `StructName<args>` type references with
        # a materialized concrete struct, iterating so nested instantiations
        # (e.g. BlockLoader referencing BlockMMA<...>) resolve. Regression-safe:
        # any failure returns the unmodified source, so a kernel that previously
        # reached the template-materialization diagnostic is left unchanged rather
        # than emitting a half-rewritten translation.
        try:
            return self._materialize_explicit_template_struct_instantiations_impl(code)
        except Exception:
            return code

    def _materialize_explicit_template_struct_instantiations_impl(
        self, code: str
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
            specialized = self._find_explicit_struct_specialization_names(working)
            templates_by_name = {
                template.name: template
                for template in templates
                if template.name not in specialized
            }
            if not templates_by_name:
                return working
            excluded_spans = self._find_template_declaration_spans(working)
            instantiations = self._find_explicit_template_struct_instantiations(
                working,
                templates_by_name,
                excluded_spans,
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
        # rewrites the corresponding call sites BEFORE lexing/parsing. It runs
        # after the struct-template materializer so concrete structs produced from
        # templates (e.g. `Sum_float`) are lowered too. Regression-safe: any
        # failure returns the unmodified source, and the pass no-ops when there
        # are no struct methods (structs without methods stay byte-identical).
        try:
            return self._lower_struct_member_functions_impl(code)
        except Exception:
            return code

    def _lower_struct_member_functions_impl(self, code: str) -> str:
        structs = self._find_concrete_struct_definitions(code)
        if not structs:
            return code
        # Only structs that actually declare at least one (non-template,
        # non-skipped) method need rewriting; everything else stays untouched so
        # method-free structs and existing kernels are byte-identical.
        structs_with_methods = [struct for struct in structs if struct.methods]
        if not structs_with_methods:
            return code

        struct_names = {struct.name for struct in structs_with_methods}
        # Methods keyed by struct name for call-site rewriting.
        methods_by_struct: Dict[str, Dict[str, _MetalStructMethod]] = {}
        operator_call_structs: Set[str] = set()
        for struct in structs_with_methods:
            method_map: Dict[str, _MetalStructMethod] = {}
            for method in struct.methods:
                method_map[method.name] = method
                if method.is_operator_call:
                    operator_call_structs.add(struct.name)
            methods_by_struct[struct.name] = method_map

        replacements: List[Tuple[int, int, str]] = []
        free_functions: List[str] = []
        for struct in structs_with_methods:
            data_only, lowered = self._render_lowered_struct(code, struct)
            replacements.append((struct.span[0], struct.span[1], data_only))
            free_functions.extend(lowered)

        # Rewrite call sites across the rest of the source (outside the structs
        # we are replacing, so receiver-less internal references are handled when
        # the method body is emitted, not here).
        struct_spans = [struct.span for struct in structs_with_methods]
        replacements.extend(
            self._rewrite_struct_member_call_sites(
                code,
                struct_names,
                methods_by_struct,
                operator_call_structs,
                struct_spans,
            )
        )

        rewritten = self._apply_text_replacements(code, replacements)
        if free_functions:
            rewritten = rewritten.rstrip() + "\n\n" + "\n\n".join(free_functions)
            if not rewritten.endswith("\n"):
                rewritten += "\n"
        return rewritten

    def _find_concrete_struct_definitions(
        self, code: str
    ) -> List[_MetalStructDefinition]:
        # Locate every concrete (non-template) `struct/class Name { ... };` and
        # split its body into data members and method definitions. Template
        # structs/classes (`template <...> struct ...`) are skipped wholesale:
        # those are handled by the materializer, and their (possibly template)
        # methods are out of scope for lowering.
        template_spans = self._find_template_declaration_spans(code)
        definitions: List[_MetalStructDefinition] = []
        for match in re.finditer(r"\b(?:struct|class)\s+", code):
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
            data_member_names, methods = self._split_struct_body(
                name, body, body_start + 1
            )
            definitions.append(
                _MetalStructDefinition(
                    name=name,
                    span=(start, span_end),
                    body_span=(body_start + 1, body_end_after - 1),
                    data_member_names=data_member_names,
                    methods=methods,
                    has_operator_call=any(m.is_operator_call for m in methods),
                )
            )
        return definitions

    def _split_struct_body(
        self, struct_name: str, body: str, body_offset: int
    ) -> Tuple[Set[str], List[_MetalStructMethod]]:
        # Walk a struct body separating DATA members from METHOD definitions.
        # A method is a declarator followed by `(params)` then `{...}`; everything
        # else terminated by `;` (or an access-specifier label) is data. Template
        # member functions (`template <...> ...`) are skipped entirely.
        data_member_names: Set[str] = set()
        methods: List[_MetalStructMethod] = []
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

            # A `template <...>` member function is out of scope: skip the
            # declaration including its body/semicolon entirely.
            if re.match(r"template\s*<", body[i:]):
                angle_start = body.find("<", i)
                angle_end = self._find_matching_angle(body, angle_start)
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
                name = self._declared_data_member_name(body[i:decl_semicolon])
                if name:
                    data_member_names.add(name)
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
            i = semicolon + 1
        return data_member_names, methods

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
        removals: List[Tuple[int, int]] = []
        for method in struct.methods:
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
        operator_call_structs: Set[str],
        struct_spans: List[Tuple[int, int]],
    ) -> List[Tuple[int, int, str]]:
        # Rewrite call sites across the source (outside the struct definitions we
        # are replacing):
        #   var.m(args)  -> S__m(var, args)            (instance method)
        #   var(args)    -> S__operator_call(var, ...)  (var has operator())
        #   S::m(args)   -> S__m(args)                  (qualified static call)
        # Local variable struct types are tracked per the declarations that
        # introduce them (`S var;` / `S var = ...;` / `S var(...)`).
        variable_types = self._collect_struct_variable_types(
            code, struct_names, struct_spans
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
                    operator_call_structs,
                    variable_types,
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
        operator_call_structs: Set[str],
        variable_types: Dict[str, str],
    ) -> Optional[Tuple[int, str]]:
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
            method = methods_by_struct.get(ident, {}).get(member)
            after = k + consumed
            while after < len(code) and code[after].isspace():
                after += 1
            if method is None or after >= len(code) or code[after] != "(":
                return None
            return after, method.free_name

        struct_type = variable_types.get(ident)
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
            method = methods_by_struct.get(struct_type, {}).get(member)
            after = k + consumed
            while after < len(code) and code[after].isspace():
                after += 1
            if method is None or after >= len(code) or code[after] != "(":
                # A data-member access (`var.field`) — leave untouched.
                return None
            arg_open = after
            return self._build_instance_call_rewrite(code, ident, method, arg_open)

        # Functor call: `var(args)` -> `S__operator_call(var, ...)`.
        if code[j] == "(" and struct_type in operator_call_structs:
            method = methods_by_struct.get(struct_type, {}).get("operator()")
            if method is None:
                return None
            return self._build_instance_call_rewrite(code, ident, method, j)

        return None

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
    ) -> Dict[str, str]:
        # Map local variable names to the lowered struct type that declares them
        # by scanning `S var;`, `S var = ...;`, and `S var(...)` declarations
        # outside the struct definitions. Last declaration wins, which is a
        # best-effort approximation that suits the flat kernels we translate.
        variable_types: Dict[str, str] = {}
        for struct_name in struct_names:
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
                variable_types[match.group(1)] = struct_name
        return variable_types

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
        return sorted(instantiations, key=lambda item: item.span[0])

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
            angle_end = self._find_matching_angle(code, angle_start)
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
            angle_end = self._find_matching_angle(code, angle_start)
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
            angle_end = self._find_matching_angle(code, angle_start)
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

    def _template_parameter_names(self, template_parameters: str) -> List[str]:
        names: List[str] = []
        for parameter in self._split_top_level_commas(template_parameters):
            parameter = parameter.strip()
            if not parameter:
                continue
            parameter = parameter.split("=", 1)[0].strip()
            tokens = IDENTIFIER_RE.findall(parameter)
            if not tokens:
                continue
            if tokens[0] in {"typename", "class"} and len(tokens) >= 2:
                names.append(tokens[-1])
            elif len(tokens) >= 2:
                names.append(tokens[-1])
        return names

    def _template_parameter_defaults(self, template_parameters: str) -> Dict[str, str]:
        defaults: Dict[str, str] = {}
        for parameter in self._split_top_level_commas(template_parameters):
            if "=" not in parameter:
                continue
            names = self._template_parameter_names(parameter)
            if not names:
                continue
            defaults[names[-1]] = parameter.split("=", 1)[1].strip()
        return defaults

    def _variadic_template_parameter_names(self, template_parameters: str) -> Set[str]:
        names: Set[str] = set()
        for parameter in self._split_top_level_commas(template_parameters):
            parameter = parameter.split("=", 1)[0].strip()
            if "..." not in parameter:
                continue
            tokens = IDENTIFIER_RE.findall(parameter)
            if len(tokens) >= 2 and tokens[0] in {"typename", "class"}:
                names.add(tokens[-1])
        return names

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
            angle_end = self._find_matching_angle(code, angle_start)
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
