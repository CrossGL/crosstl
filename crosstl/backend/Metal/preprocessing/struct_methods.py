"""Struct method parsing, receiver contracts and the rendering of lowered structs."""

import re
from dataclasses import replace
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from ..static_member_constants import substitute_parameter_extent_static_members
from .support import (
    IDENTIFIER_RE,
    MetalStructMethodError,
    _MetalReceiverContract,
    _MetalStructDefinition,
    _MetalStructMethod,
)


class MetalStructMethodMixin:
    """Members of ``MetalPreprocessor``: struct method parsing, receiver contracts and the rendering of lowered structs."""

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
        conversion_match = re.search(
            r"\boperator\s+" r"(?P<type>[A-Za-z_][A-Za-z0-9_:]*(?:\s*<[^()]+>)?)\s*\(",
            header,
        )
        is_operator_call = False
        is_conversion_operator = False
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
        elif conversion_match is not None:
            paren_start = conversion_match.end() - 1
            paren_end = self._find_matching_delimiter(header, paren_start, "(", ")")
            if paren_end is None:
                return None
            parameters = header[paren_start + 1 : paren_end]
            if parameters.strip():
                return None
            conversion_type = self._normalize_template_argument_text(
                conversion_match.group("type")
            )
            if not conversion_type:
                return None
            is_conversion_operator = True
            method_name = f"operator {conversion_type}"
            signature_prefix = conversion_type
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
        receiver_qualifier_text = header[paren_end + 1 :]
        trailing_qualifiers = self._member_trailing_qualifiers(receiver_qualifier_text)
        is_const = "const" in trailing_qualifiers
        is_volatile = "volatile" in trailing_qualifiers
        ref_qualifiers = [
            qualifier for qualifier in trailing_qualifiers if qualifier in {"&", "&&"}
        ]
        ref_qualifier = ref_qualifiers[0] if len(ref_qualifiers) == 1 else ""
        receiver_address_spaces = self._receiver_address_spaces(receiver_qualifier_text)

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
            is_conversion_operator=is_conversion_operator,
            return_type=return_type,
            parameters=parameters.strip(),
            parameter_names=parameter_names,
            body=method_body,
            span=(decl_start, method_body_end),
            is_const=is_const,
            is_volatile=is_volatile,
            ref_qualifier=ref_qualifier,
            receiver_address_spaces=receiver_address_spaces,
            trailing_qualifiers=trailing_qualifiers,
        )

    def _member_trailing_qualifiers(self, text: str) -> Tuple[str, ...]:
        qualifiers: List[str] = []
        i = 0
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            if text.startswith("[[", i):
                end = text.find("]]", i + 2)
                if end == -1:
                    break
                i = end + 2
                continue
            if text[i].isalpha() or text[i] == "_":
                word, consumed = self._read_identifier(text, i)
                qualifiers.append(word)
                i += consumed
                if word == "noexcept":
                    while i < len(text) and text[i].isspace():
                        i += 1
                    if i < len(text) and text[i] == "(":
                        end = self._find_matching_delimiter(text, i, "(", ")")
                        if end is None:
                            break
                        i = end
                continue
            if text.startswith("&&", i):
                qualifiers.append("&&")
                i += 2
                continue
            if text[i] == "&":
                qualifiers.append("&")
            i += 1
        return tuple(qualifiers)

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
        if method_name.startswith("operator "):
            conversion_type = method_name[len("operator ") :].strip()
            suffix = re.sub(r"[^A-Za-z0-9_]+", "_", conversion_type).strip("_")
            return f"{struct_name}__operator_{suffix or 'conversion'}"
        return f"{struct_name}__{method_name}"

    def _assign_receiver_overload_free_names(
        self,
        struct_name: str,
        methods: List[_MetalStructMethod],
    ) -> None:
        groups: Dict[Tuple[str, bool, Tuple[str, ...]], List[_MetalStructMethod]] = {}
        for method in methods:
            parameter_contract = tuple(
                self._normalize_template_argument_text(
                    self._normalize_function_parameter_type_text(parameter)
                )
                for parameter in self._split_top_level_commas(method.parameters)
                if parameter.strip()
            )
            groups.setdefault(
                (method.name, method.is_static, parameter_contract), []
            ).append(method)

        for overloads in groups.values():
            if len(overloads) < 2 or overloads[0].is_static:
                continue
            receiver_contracts = {
                self._method_receiver_contract_key(method) for method in overloads
            }
            if len(receiver_contracts) < 2:
                continue
            for method in overloads:
                method.free_name = (
                    f"{self._struct_member_free_name(struct_name, method.name, method.is_operator_call)}"
                    f"__metal_receiver_{self._method_receiver_name_suffix(method)}"
                )

    @staticmethod
    def _method_receiver_contract_key(method: _MetalStructMethod) -> Tuple[object, ...]:
        return (
            method.is_const,
            method.is_volatile,
            method.ref_qualifier,
            method.receiver_address_spaces or ("thread",),
        )

    @staticmethod
    def _method_body_receiver_contract(
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
    ) -> Optional[_MetalReceiverContract]:
        if method.is_static:
            return None
        address_spaces = method.receiver_address_spaces or ("thread",)
        qualifiers = []
        if method.is_const:
            qualifiers.append("const")
        if method.is_volatile:
            qualifiers.append("volatile")
        qualifiers.extend(address_spaces)
        qualifiers.append(f"{struct.name}&")
        return _MetalReceiverContract(
            struct_name=struct.name,
            is_const=method.is_const,
            is_volatile=method.is_volatile,
            address_spaces=address_spaces,
            indirection="&",
            value_category="lvalue",
            source_type=" ".join(qualifiers),
        )

    @staticmethod
    def _method_receiver_name_suffix(method: _MetalStructMethod) -> str:
        if method.is_const and method.is_volatile:
            cv = "const_volatile"
        elif method.is_const:
            cv = "const"
        elif method.is_volatile:
            cv = "volatile"
        else:
            cv = "mutable"
        address = "_".join(method.receiver_address_spaces or ("thread",))
        reference = {
            "": "unqualified",
            "&": "lvalue",
            "&&": "rvalue",
        }.get(method.ref_qualifier, "unknown_ref")
        return f"{cv}_{address}_{reference}"

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

    @staticmethod
    def _strip_trailing_metal_attributes(declaration: str) -> str:
        text = declaration.rstrip()
        while True:
            match = re.search(r"\s*\[\[[^\[\]]*\]\]\s*$", text)
            if match is None:
                return text
            text = text[: match.start()].rstrip()

    def _declared_data_member_name(self, declaration: str) -> Optional[str]:
        # Extract the declared identifier from a data-member declaration such as
        # `float bias`, `T data[N]`, `device float* ptr`, or
        # `static constexpr constant U init = U(0)`.
        text = self._strip_top_level_default_value(declaration).strip()
        text = self._strip_trailing_metal_attributes(text)
        if not text:
            return None
        # Drop any trailing array extents so the bare name is left.
        while text.endswith("]"):
            open_bracket = text.rfind("[")
            if open_bracket == -1:
                break
            text = text[:open_bracket].rstrip()
        # A reference-to-array declarator, `thread T (&vals)[N]`, names the
        # parenthesized identifier; MLX's sort helpers take their per-thread
        # register arrays this way.
        reference = re.search(r"\(\s*&\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*$", text)
        if reference is not None:
            return reference.group(1)
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
        specialized_body = self._rewrite_explicit_conversion_operator_calls(
            struct,
            specialized_body,
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
        rewritten_body = self._canonicalize_struct_scoped_template_arguments(
            rewritten_body, struct, method, structs_by_name
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
        params = substitute_parameter_extent_static_members(
            self, struct, method, params
        )
        params = self._preserve_parameter_usage(params, method.body, rewritten_body)
        if method.is_static:
            new_params = params if params and params != "void" else ""
        else:
            self_param = self._instance_receiver_parameter(struct, method)
            if not self._bare_identifier_use_count(
                self._mask_comments_and_literals(rewritten_body), "self"
            ):
                self_param += " [[maybe_unused]]"
            if params and params != "void":
                new_params = f"{self_param}, {params}"
            else:
                new_params = self_param
        return f"static inline {return_type} {method.free_name}({new_params}) {{{rewritten_body}}}"

    def _preserve_parameter_usage(
        self, parameters: str, original_body: str, rewritten_body: str
    ) -> str:
        """Retain source usage intent without hiding originally unused parameters."""
        original = self._mask_comments_and_literals(original_body)
        rewritten = self._mask_comments_and_literals(rewritten_body)
        shadowed = self._local_variable_names(original)
        declarations = self._split_top_level_commas(parameters)
        changed = False
        for index, declaration in enumerate(declarations):
            name = self._declared_data_member_name(declaration)
            if (
                not name
                or name in shadowed
                or self._lambda_binds_identifier(original, name)
                or "[[maybe_unused]]" in declaration
                or not self._bare_identifier_use_count(original, name)
                or self._bare_identifier_use_count(rewritten, name)
            ):
                continue
            declaration = declaration.strip()
            prefix = self._strip_top_level_default_value(declaration)
            declarations[index] = (
                prefix + " [[maybe_unused]]" + declaration[len(prefix) :]
            )
            changed = True
        return ", ".join(declarations) if changed else parameters

    def _preserve_rewritten_function_parameter_usage(
        self,
        code: str,
        call_replacements: List[Tuple[int, int, str]],
        excluded_spans: List[Tuple[int, int]],
        existing_replacements: List[Tuple[int, int, str]],
    ) -> List[Tuple[int, int, str]]:
        replacements = []
        for function in self._find_non_template_function_definitions(
            code, excluded_spans
        ):
            start, end = function.body_span
            body_replacements = [
                (left - start, right - start, value)
                for left, right, value in call_replacements
                if start <= left < right <= end
            ]
            if not body_replacements:
                continue
            header_start = function.span[0]
            header = code[header_start : start - 1]
            parameter_span = self._function_parameter_list_span(header)
            if parameter_span is None:
                continue
            left, right = parameter_span
            left += header_start + 1
            right += header_start
            if any(
                old_left < right and left < old_right
                for old_left, old_right, _ in existing_replacements
            ):
                continue
            body = code[start:end]
            parameters = self._preserve_parameter_usage(
                code[left:right],
                body,
                self._apply_text_replacements(body, body_replacements),
            )
            if parameters != code[left:right]:
                replacements.append((left, right, parameters))
        return replacements

    def _rewrite_explicit_conversion_operator_calls(
        self,
        struct: _MetalStructDefinition,
        body: str,
    ) -> str:
        rewritten = body
        for candidate in struct.methods:
            if not candidate.is_conversion_operator:
                continue
            conversion_type = self._normalize_template_argument_text(
                candidate.return_type
            )
            type_pattern = re.escape(conversion_type).replace(r"\ ", r"\s+")
            pattern = (
                r"\b(?:this\s*(?:->|\.)|self\s*\.)\s*operator\s+"
                + type_pattern
                + r"\s*\(\s*\)"
            )
            rewritten = re.sub(pattern, f"{candidate.free_name}(self)", rewritten)
        return rewritten

    def _conversion_operator_receiver_is_readonly(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        active: Optional[Set[int]] = None,
    ) -> bool:
        if method.is_const:
            return True
        if not method.is_conversion_operator:
            return False
        active = set(active or ())
        if id(method) in active:
            return False
        active.add(id(method))

        body = self._mask_comments_and_literals(method.body)
        explicit_conversion_pattern = re.compile(
            r"\bthis\s*(?:->|\.)\s*operator\s+"
            r"(?P<type>[A-Za-z_][A-Za-z0-9_:]*(?:\s*<[^()]+>)?)"
            r"\s*\(\s*\)"
        )
        for conversion_call in explicit_conversion_pattern.finditer(body):
            conversion_type = self._normalize_inferred_type(
                conversion_call.group("type")
            )
            candidates = [
                candidate
                for candidate in struct.methods
                if candidate.is_conversion_operator
                and self._normalize_inferred_type(candidate.return_type)
                == conversion_type
            ]
            if len(
                candidates
            ) != 1 or not self._conversion_operator_receiver_is_readonly(
                struct, candidates[0], active
            ):
                return False

        member_names = sorted(struct.data_member_names, key=len, reverse=True)
        for member in member_names:
            access = rf"(?:\bthis\s*(?:->|\.)\s*)?\b{re.escape(member)}\b"
            for match in re.finditer(access, body):
                prefix = body[max(0, match.start() - 8) : match.start()]
                suffix = body[match.end() : match.end() + 8]
                if re.search(r"(?:\+\+|--|&)\s*$", prefix):
                    return False
                if re.match(
                    r"\s*(?:\+\+|--|(?:<<|>>|[+\-*/%&|^])=|=(?!=))",
                    suffix,
                ):
                    return False
                statement_start = body.rfind(";", 0, match.start()) + 1
                statement_prefix = body[statement_start : match.start()]
                if re.search(
                    r"&\s*[A-Za-z_]\w*\s*=\s*$",
                    statement_prefix,
                ):
                    return False

        call_scan = explicit_conversion_pattern.sub("conversion_result", body)
        safe_calls = set(self._METAL_SCALAR_VECTOR_TYPES)
        safe_calls.update({"alignof", "as_type", "sizeof", "static_cast"})
        call_pattern = re.compile(
            r"(?P<callee>(?:(?:[A-Za-z_]\w*)::)*[A-Za-z_]\w*"
            r"(?:\s*<[^(){};]+>)?)\s*\("
        )
        for call in call_pattern.finditer(call_scan):
            callee = re.sub(r"\s+", "", call.group("callee"))
            base = callee.split("<", 1)[0].rsplit("::", 1)[-1]
            if base in {"if", "switch", "while", "for", "return"}:
                continue
            open_paren = call_scan.find("(", call.start("callee"))
            close_paren = self._find_matching_delimiter(call_scan, open_paren, "(", ")")
            if close_paren is None:
                return False
            if base in safe_calls:
                continue
            arguments = call_scan[open_paren + 1 : close_paren]
            observes_receiver = bool(re.search(r"\bthis\b", arguments)) or any(
                re.search(
                    rf"(?:\bthis\s*(?:->|\.)\s*)?\b{re.escape(member)}\b",
                    arguments,
                )
                for member in member_names
            )
            if "::" not in callee or observes_receiver:
                return False

        allowed_this = explicit_conversion_pattern.sub("conversion_result", body)
        allowed_this = re.sub(
            r"\bthis\s*(?:->|\.)\s*[A-Za-z_]\w*",
            "member_read",
            allowed_this,
        )
        return re.search(r"\bthis\b", allowed_this) is None

    def _instance_receiver_parameter(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
    ) -> str:
        address_spaces = method.receiver_address_spaces or ("thread",)
        if len(address_spaces) != 1:
            raise MetalStructMethodError(
                "Cannot lower member function with an unrepresentable receiver "
                f"address-space contract: {struct.name}::{method.name}.",
                struct_name=struct.name,
                method_name=method.name,
                requested_signature=f"{struct.name}::{method.name}({method.parameters})",
                suggested_action=(
                    "use exactly one Metal receiver address-space qualifier"
                ),
                reason="receiver-address-space-unrepresentable",
                receiver_type=" ".join(address_spaces),
                missing_capabilities=("metal.member-receiver-qualifiers",),
            )
        cv_qualifiers = []
        if method.is_const or self._conversion_operator_receiver_is_readonly(
            struct, method
        ):
            cv_qualifiers.append("const")
        if method.is_volatile:
            cv_qualifiers.append("volatile")
        cv_text = f"{' '.join(cv_qualifiers)} " if cv_qualifiers else ""
        reference = "&&" if method.ref_qualifier == "&&" else "&"
        return f"{address_spaces[0]} {cv_text}{struct.name}{reference} self"
