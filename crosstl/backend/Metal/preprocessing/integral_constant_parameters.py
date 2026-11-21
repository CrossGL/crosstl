"""Integral-constant parameter substitution, concrete method specialization and const-for-loop lowering."""

import re
from dataclasses import replace
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from ..integral_constant_locals import (
    is_integral_constant_operand,
    resolve_const_for_loop_bound_constants,
)
from ..static_member_constants import canonicalize_materialized_template_id_owners
from .support import (
    MetalTemplateSpecializationError,
    _MetalStructDefinition,
    _MetalStructMethod,
)


class MetalIntegralConstantParameterMixin:
    """Members of ``MetalPreprocessor``: integral-constant parameter substitution, concrete method specialization and const-for-loop lowering."""

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
            body = canonicalize_materialized_template_id_owners(
                self, body, structs_by_name
            )
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
            # A `const thread` member qualifier only restricts the object's
            # address space (MSL section 4.2); the conversion is unchanged.
            conversion = re.compile(
                r"(?:METAL_FUNC)?constexproperatorvalue_type\(\)const(?:thread)?"
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
            # `remove_addrspace_t` drops only an address-space qualifier, so the
            # aliased result type names the same integral type (MSL 6.2).
            aliased_result = re.compile(
                rf"^constexprauto(?P<result>[A-Za-z_]\w*)="
                rf"{left}{arithmetic_operator}{right};"
                r"using(?P<alias>[A-Za-z_]\w*)=(?:metal::)?remove_addrspace_t<"
                r"decltype\((?P=result)\)>;"
                r"returnintegral_constant<(?P=alias),(?P=result)>\{\};$"
            )
            operator_name = match.group("operator")
            if (
                direct_return.fullmatch(compact)
                or named_result.fullmatch(compact)
                or aliased_result.fullmatch(compact)
            ):
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
        masked_expression = self._mask_comments_and_literals(expression)
        replacements: List[Tuple[int, int, str]] = []
        for match in pattern.finditer(expression):
            if self._containing_span(match.start(), ignored_spans) is not None:
                continue
            if self._static_member_use_requires_declared_storage(
                masked_expression,
                match.start(),
            ):
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
        if depth == 0:
            source = resolve_const_for_loop_bound_constants(self, source)
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
                if not is_integral_constant_operand(
                    self, source, i, identifier_end
                ) and self._const_for_loop_integral_conversion_use_is_safe(
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
