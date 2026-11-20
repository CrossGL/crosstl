"""Discovery of declared template instantiations and the materialization of template functions and structs."""

import re
from copy import deepcopy
from dataclasses import replace
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)

from ..namespaced_specializations import specialization_namespaces
from ..return_type_enablers import return_type_enabler_condition
from .support import (
    IDENTIFIER_RE,
    MLX_HOST_NAME_DECL_RE,
    MLX_INSTANTIATE_KERNEL_RE,
    _MetalTemplateFunction,
    _MetalTemplateStruct,
    _MLXKernelInstantiation,
)


class MetalTemplateInstantiationMixin:
    """Members of ``MetalPreprocessor``: discovery of declared template instantiations and the materialization of template functions and structs."""

    def _scan_project_template_instantiations(
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
            r"(?:^|[;\n])\s*(?P<declaration>(?:template\s+)?\[\[|template\s+(?!<))",
            code,
        ):
            # The declaration starts at its first token, after the separator and blank space before it.
            start = match.start("declaration")
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
        cached = self._source_analysis(code).template_functions
        if cached is not None:
            return deepcopy(cached)
        templates = self._scan_template_functions(code)
        # Materialization appends to these records; discovery owns a separate copy.
        self._source_analysis(code).template_functions = deepcopy(templates)
        return templates

    def _scan_template_functions(self, code: str) -> List[_MetalTemplateFunction]:
        templates: List[_MetalTemplateFunction] = []
        namespace_spans = self._find_namespace_spans(code)
        anonymous_namespace_spans = self._source_analysis(
            code
        ).anonymous_namespace_spans
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
            body_start = self._find_next_top_level_char(
                code, declaration_start, "{", stop_at=";"
            )
            semicolon = self._find_next_top_level_char(
                code, declaration_start, ";", stop_at="{"
            )
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
                    template_constraints=[
                        *self._template_parameter_constraints(parameter_text),
                        *filter(
                            None,
                            [
                                return_type_enabler_condition(
                                    self, header, function_name
                                )
                            ],
                        ),
                    ],
                    namespace=self._namespace_at(namespace_spans, start),
                    internal_linkage=any(
                        scope_start <= start < scope_end
                        for scope_start, scope_end in anonymous_namespace_spans
                    ),
                )
            )
            pos = body_end
        return templates

    def _find_template_structs(
        self, code: str, *, include_forward_declarations: bool = False
    ) -> List[_MetalTemplateStruct]:
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
                r"\s*(?:\[\[[^\]]*\]\]\s*)*(?:struct|class|union)\s+"
                r"(?P<name>[A-Za-z_][A-Za-z0-9_:]*)\b",
                header,
                re.DOTALL,
            )
            parameter_text = code[angle_start + 1 : angle_end]
            parameter_records = self._parse_template_parameter_list(parameter_text)
            if header_match is None or not parameter_records:
                pos = declaration_start
                continue

            body_start = self._find_next_top_level_char(
                code, declaration_start, "{", stop_at=";"
            )
            semicolon = self._find_next_top_level_char(
                code, declaration_start, ";", stop_at="{"
            )
            forward = semicolon is not None and (
                body_start is None or semicolon < body_start
            )
            if forward:
                after_name = declaration_start + header_match.end()
                if (
                    not include_forward_declarations
                    or code[after_name:semicolon].strip()
                ):
                    pos = semicolon + 1
                    continue
                body_end = semicolon + 1
            elif body_start is None:
                pos = declaration_start
                continue
            else:
                body_end = self._find_matching_brace(code, body_start)
            if body_end is None:
                pos = body_start + 1
                continue

            parameters = []
            defaults = {}
            declared_types = {}
            variadic_parameters = set()
            used_names = set(IDENTIFIER_RE.findall(code[angle_start + 1 : body_end]))
            for index, parameter in enumerate(parameter_records):
                name = parameter.name
                if name is None and parameter.is_type_parameter:
                    # Anonymous type parameters still occupy an argument slot.
                    # Give only the materializer an identity for their defaults.
                    name = f"crosstl_unnamed_type_{index}"
                    while name in used_names:
                        name += "_"
                    used_names.add(name)
                if name is None:
                    continue
                parameters.append(name)
                if parameter.default is not None:
                    defaults[name] = parameter.default
                if parameter.declared_type is not None:
                    declared_types[name] = parameter.declared_type
                if parameter.is_variadic:
                    variadic_parameters.add(name)
            if not parameters:
                pos = declaration_start
                continue

            structs.append(
                _MetalTemplateStruct(
                    name=header_match.group("name").split("::")[-1],
                    template_parameters=parameters,
                    span=(start, body_end),
                    source=code[declaration_start:body_end],
                    variadic_template_parameters=variadic_parameters,
                    template_parameter_defaults=defaults,
                    template_parameter_types=declared_types,
                    template_type_traits=template_type_traits,
                    namespace=self._namespace_at(namespace_spans, start),
                    is_forward_declaration=forward,
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

    def _template_specialization_placements(
        self, code: str, templates: Iterable[_MetalTemplateFunction]
    ) -> Mapping[Tuple[int, int], str]:
        templates = list(templates)
        if not any(template.namespace for template in templates):
            return {}
        struct_spans = [
            struct.span for struct in self._find_concrete_struct_definitions(code)
        ] + [struct.span for struct in self._find_template_structs(code)]
        return specialization_namespaces(templates, struct_spans)

    def _reconstruct_materialized_function_array_parameter_declarators(
        self,
        source: str,
        materialized: str,
        substitutions: Dict[str, str],
    ) -> str:
        """Move concrete array suffixes after exact template parameter names.

        Replacing ``W`` with a concrete argument such as ``float[8]`` in the
        source declarator ``W values`` produces the invalid prefix spelling
        ``float[8] values``.  Only reconstruct an exact, otherwise-unadorned
        type-template parameter and only for literal-size arrays; pointer,
        reference, dependent, and parenthesized declarators remain fail-closed.
        """

        literal = r"[-+]?(?:0[xX][0-9A-Fa-f]+|0[bB][01]+|\d+)[uUlL]*"
        array_bindings: Dict[str, Tuple[str, str, str]] = {}
        for parameter_name, concrete_type in substitutions.items():
            normalized = self._normalize_template_argument_text(concrete_type)
            array_start = normalized.find("[")
            if array_start <= 0:
                continue
            base = normalized[:array_start].strip()
            array_suffix = normalized[array_start:]
            if re.fullmatch(rf"(?:\[{literal}\])+", array_suffix) is None:
                continue
            if any(token in base for token in ("*", "&", "(", ")", "{", "}")):
                continue
            if (
                re.fullmatch(
                    r"(?:(?:const|volatile|thread|threadgroup|device|constant)\s+)*"
                    r"(?:::)?[A-Za-z_][A-Za-z0-9_]*"
                    r"(?:::[A-Za-z_][A-Za-z0-9_]*)*"
                    r"(?:<[^\[\](){}*&]+>)?",
                    base,
                )
                is None
            ):
                continue
            array_bindings[parameter_name] = (base, array_suffix, normalized)
        if not array_bindings:
            return materialized

        source_open = self._function_parameter_start(source)
        materialized_open = self._function_parameter_start(materialized)
        if source_open is None or materialized_open is None:
            return materialized
        source_close = self._find_matching_delimiter(source, source_open, "(", ")")
        materialized_close = self._find_matching_delimiter(
            materialized, materialized_open, "(", ")"
        )
        if source_close is None or materialized_close is None:
            return materialized

        source_parameters = self._split_top_level_commas(
            source[source_open + 1 : source_close]
        )
        materialized_parameters = self._split_top_level_commas(
            materialized[materialized_open + 1 : materialized_close]
        )
        if len(source_parameters) != len(materialized_parameters):
            return materialized

        changed = False
        for index, source_parameter in enumerate(source_parameters):
            for template_name, (
                base,
                array_suffix,
                concrete_type,
            ) in array_bindings.items():
                source_match = re.fullmatch(
                    rf"\s*{re.escape(template_name)}\s+"
                    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*",
                    source_parameter,
                )
                if source_match is None:
                    continue
                parameter_name = source_match.group("name")
                materialized_normalized = re.sub(
                    r"\s+", " ", materialized_parameters[index].strip()
                )
                if materialized_normalized != f"{concrete_type} {parameter_name}":
                    continue
                materialized_parameters[index] = (
                    f"{base} {parameter_name}{array_suffix}"
                )
                changed = True
                break
        if not changed:
            return materialized

        parameter_text = ", ".join(materialized_parameters)
        return (
            materialized[: materialized_open + 1]
            + parameter_text
            + materialized[materialized_close:]
        )

    def _materialized_template_function_return_type(
        self,
        template: _MetalTemplateFunction,
        substitutions: Dict[str, str],
    ) -> Optional[str]:
        """Return one exact concrete value type for a materialized helper.

        Struct-member template deduction may consume a nested call to a free
        template helper after that helper has already been materialized and
        renamed (for example ``op(cast_to<U>(value), total)`` in MLX reduce).
        Record only a source-declared return spelling whose complete template
        substitution is a plain concrete value type.  Dependent aliases,
        SFINAE wrappers, ``auto``/``decltype``, pointers, and references remain
        deliberately untyped so unsupported shapes still fail closed.
        """

        body_start = self._find_next_top_level_char(template.source, 0, "{")
        if body_start is None:
            return None
        header = template.source[:body_start]
        parameter_start = self._function_parameter_start(header)
        if parameter_start is None:
            return None
        name_match = re.search(
            rf"\b{re.escape(template.name)}\s*$",
            header[:parameter_start],
        )
        if name_match is None:
            return None
        return_type = self._replace_identifiers(
            header[: name_match.start()],
            substitutions,
        )
        return_type = re.sub(r"\[\[[^\]]*\]\]", " ", return_type)
        return_type = re.sub(
            r"\b(?:constexpr|consteval|inline|static|friend|kernel|vertex|fragment)\b",
            " ",
            return_type,
        )
        return_type = self._strip_function_qualifier_macros(return_type)
        return_type = self._normalize_template_argument_text(return_type)
        if (
            not return_type
            or return_type == "void"
            or any(token in return_type for token in ("auto", "decltype", "enable_if"))
            or any(token in return_type for token in ("*", "&", "<", ">", "[", "]"))
            or self._type_references_template_parameter(
                return_type,
                template.template_parameters,
            )
        ):
            return None
        if self._is_metal_scalar_or_vector_type(return_type):
            return return_type
        if re.fullmatch(
            r"(?:::)?[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*",
            return_type,
        ):
            return return_type
        return None

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
        if (
            template.is_forward_declaration
            or not self._template_arguments_satisfy_parameters(
                template,
                template_arguments,
            )
        ):
            return ""
        substitutions, _variadic_bindings = self._template_argument_bindings(
            template,
            template_arguments,
        )
        if not substitutions:
            return ""
        materialized = self._replace_identifiers(template.source, substitutions)
        materialized = self._rename_materialized_struct_constructors(
            materialized,
            template.name,
            struct_identifier,
        )
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
            template.name,
            strip_specialization_arguments=True,
        )
        materialized = self._rename_materialized_struct_constructors(
            materialized,
            template.name,
            struct_identifier,
        )
        materialized = self._rename_struct_definition(
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
        declaration_ends = {
            member.name: member.span[1]
            for member in outer.data_members
            if member.name in mapping and member.span is not None
        }
        replacements: List[Tuple[int, int, str]] = []
        cursor = body_start
        for start, end in [*merged_spans, (body_end, body_end)]:
            if cursor < start:
                segment = materialized[cursor:start]
                segment = self._substitute_static_constants_in_declarations(
                    segment,
                    mapping,
                    declaration_ends,
                    source_offset=cursor,
                )
                segment = self._substitute_struct_scoped_alias_owners(segment, outer)
                replacements.append(
                    (
                        cursor,
                        start,
                        segment,
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
        self,
        source: str,
        mapping: Dict[str, str],
        declaration_ends: Dict[str, int],
        *,
        source_offset: int,
    ) -> str:
        # A class member is not available to declarations that precede it. Apply
        # each resolved mapping only after its declaration terminator while still
        # allowing subsequent initializers, aliases, extents, and assertions to
        # use the value.
        source_end = source_offset + len(source)
        visible = {
            name: value
            for name, value in mapping.items()
            if declaration_ends.get(name, source_end + 1) <= source_offset
        }
        visibility_events: Dict[int, List[str]] = {}
        for name in mapping:
            declaration_end = declaration_ends.get(name)
            if declaration_end is None or not (
                source_offset < declaration_end <= source_end
            ):
                continue
            visibility_events.setdefault(declaration_end, []).append(name)

        substituted_parts: List[str] = []
        relative_cursor = 0
        for declaration_end in sorted(visibility_events):
            relative_end = declaration_end - source_offset
            substituted_parts.append(
                self._substitute_bare_member_references(
                    source[relative_cursor:relative_end],
                    visible,
                )
            )
            for name in visibility_events[declaration_end]:
                visible[name] = mapping[name]
            relative_cursor = relative_end
        substituted_parts.append(
            self._substitute_bare_member_references(
                source[relative_cursor:],
                visible,
            )
        )
        return "".join(substituted_parts)
