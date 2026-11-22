"""Template declarations in the Metal parser: template prefixes, parameter lists and defaults, template variables and functions, and template skipping."""

import sys

from crosstl.translator.refusals import MetalSyntaxError

from ..explicit_entry_instantiations import instantiated_entry_name
from ..MetalAst import (
    AssignmentNode,
    TypeAliasNode,
    VariableNode,
)
from .parser_tables import (
    STAGE_TOKENS,
    TYPE_TOKENS,
)


class MetalTemplateParsingMixin:
    """Template members of ``MetalParser``."""

    def is_template_declaration_start(self):
        return (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == "template"
        )

    def is_template_alias_declaration_start(self):
        if not self.is_template_declaration_start():
            return False
        angle_start = self.pos + 1
        if (
            angle_start >= len(self.tokens)
            or self.tokens[angle_start][0] != "LESS_THAN"
        ):
            return False
        declaration_start = self.skip_template_argument_list_at(angle_start)
        return (
            declaration_start < len(self.tokens)
            and self.tokens[declaration_start][0] == "USING"
        )

    def is_union_declaration_start(self):
        idx = self.skip_gnu_extension_prefix_tokens_at(self.pos)
        return (
            idx < len(self.tokens)
            and self.tokens[idx] == ("IDENTIFIER", "union")
            and idx + 1 < len(self.tokens)
            and self.tokens[idx + 1][0] in {"IDENTIFIER", "LBRACE"}
        )

    def is_qualified_aggregate_declaration_start(self):
        idx = self.pos
        saw_qualifier = False
        while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
            saw_qualifier = True
            idx += 1
        if not saw_qualifier or idx >= len(self.tokens):
            return False
        if self.tokens[idx][0] in {"STRUCT", "CLASS"}:
            idx += 1
            idx = self.skip_alignas_specifier_tokens_at(idx)
            if idx >= len(self.tokens) or not self.is_name_token_at(idx):
                return False
            idx += 1
            if idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
                idx = self.skip_template_argument_list_at(idx)
            idx = self.skip_alignas_specifier_tokens_at(idx)
            return idx < len(self.tokens) and self.tokens[idx][0] in {
                "LBRACE",
                "COLON",
            }
        return (
            self.tokens[idx] == ("IDENTIFIER", "union")
            and idx + 2 < len(self.tokens)
            and self.is_name_token_at(idx + 1)
            and self.tokens[idx + 2][0] == "LBRACE"
        )

    def parse_qualified_aggregate_declaration(self):
        qualifiers = []
        while self.is_type_qualifier_start():
            qualifiers.extend(self.parse_type_qualifier())

        if self.current_token[0] == "STRUCT":
            aggregate = self.parse_struct()
        elif self.current_token[0] == "CLASS":
            aggregate = self.parse_class()
        elif self.is_union_declaration_start():
            aggregate = self.parse_union()
        else:
            raise MetalSyntaxError(
                f"Expected aggregate declaration, got {self.current_token[0]}"
            )

        for declaration in getattr(aggregate, "trailing_declarations", []) or []:
            var_node = (
                declaration.left
                if isinstance(declaration, AssignmentNode)
                and isinstance(declaration.left, VariableNode)
                else declaration
            )
            if isinstance(var_node, VariableNode):
                var_node.qualifiers = list(qualifiers) + list(
                    getattr(var_node, "qualifiers", []) or []
                )
        return aggregate

    def parse_template_declaration(self):
        start_token = self.current_token
        if self.peek(1)[0] != "LESS_THAN":
            instantiation = instantiated_entry_name(self.tokens, self.pos + 1)
            if instantiation is not None:
                self.explicit_entry_instantiations.append(instantiation)
        template_parameters = self.parse_template_prefix()
        template_parameter_defaults = dict(
            getattr(self, "last_template_parameter_defaults", {}) or {}
        )
        template_type_names = {
            name
            for kind, name in template_parameters
            if kind in {"typename", "class", "typename...", "class..."} and name
        }
        template_value_names = {
            name for kind, name in template_parameters if kind == "value" and name
        }
        added_type_names = template_type_names - self.known_types
        self.known_types.update(added_type_names)
        added_value_names = template_value_names - self.known_value_template_parameters
        self.known_value_template_parameters.update(added_value_names)

        try:
            self.push_declaration_context("template declaration")
            if self.current_token[0] in {"STRUCT", "CLASS"}:
                struct = (
                    self.parse_struct()
                    if self.current_token[0] == "STRUCT"
                    else self.parse_class()
                )
                if struct is not None:
                    struct.generics = [
                        name for _kind, name in template_parameters if name
                    ]
                    struct.template_parameters = template_parameters
                    struct.template_parameter_defaults = template_parameter_defaults
                    template_location = self.source_span_from_tokens(
                        start_token, self.tokens[self.pos - 1]
                    )
                    struct.template_source_location = template_location
                    if not getattr(struct, "source_location", None):
                        struct.source_location = template_location
                return struct

            if self.current_token[0] == "USING":
                alias = self.parse_using_statement()
                if isinstance(alias, TypeAliasNode):
                    alias.generics = [
                        name for _kind, name in template_parameters if name
                    ]
                    alias.template_parameters = template_parameters
                    alias.template_parameter_defaults = template_parameter_defaults
                    alias.is_template_alias = True
                    template_location = self.source_span_from_tokens(
                        start_token, self.tokens[self.pos - 1]
                    )
                    alias.template_source_location = template_location
                    alias.source_location = template_location
                return alias

            if not self.is_function_definition():
                function_template_name = self.template_function_declaration_name_at(
                    self.pos
                )
                if function_template_name:
                    self.known_function_templates.add(function_template_name)
                variable_template_name = self.template_variable_declaration_name_at(
                    self.pos
                )
                if variable_template_name:
                    self.known_variable_templates.add(variable_template_name)
                self.skip_template_declaration()
                return None

            function = self.parse_function_template_definition_or_prototype()
            if function is not None:
                self.known_function_templates.add(function.name)
                function.generics = [
                    name for _kind, name in template_parameters if name
                ]
                function.template_parameters = template_parameters
                function.template_parameter_defaults = template_parameter_defaults
                template_location = self.source_span_from_tokens(
                    start_token, self.tokens[self.pos - 1]
                )
                function.template_source_location = template_location
                if not getattr(function, "source_location", None):
                    function.source_location = template_location
            return function
        finally:
            if sys.exc_info()[0] is None:
                self.pop_declaration_context()
            self.known_types.difference_update(added_type_names)
            self.known_value_template_parameters.difference_update(added_value_names)

    def parse_template_prefix(self):
        self.eat("IDENTIFIER")
        self.last_template_parameter_defaults = {}
        if self.current_token[0] != "LESS_THAN":
            return []

        parameters = []
        defaults = {}
        parameter_tokens = []
        depth = 1
        grouped_depth = 0
        self.eat("LESS_THAN")

        while depth > 0 and self.current_token[0] != "EOF":
            token = self.current_token
            if token[0] in {"LPAREN", "LBRACKET", "LBRACE"}:
                grouped_depth += 1
                parameter_tokens.append(token)
                self.eat(token[0])
            elif token[0] in {"RPAREN", "RBRACKET", "RBRACE"} and grouped_depth > 0:
                grouped_depth -= 1
                parameter_tokens.append(token)
                self.eat(token[0])
            elif grouped_depth == 0 and token[0] == "LESS_THAN":
                depth += 1
                parameter_tokens.append(token)
                self.eat("LESS_THAN")
            elif grouped_depth == 0 and token[0] == "GREATER_THAN":
                depth -= 1
                if depth == 0:
                    self.eat("GREATER_THAN")
                    break
                parameter_tokens.append(token)
                self.eat("GREATER_THAN")
            elif grouped_depth == 0 and token[0] == "SHIFT_RIGHT" and depth >= 2:
                closers_to_keep = min(2, depth - 1)
                parameter_tokens.extend(
                    ("GREATER_THAN", ">") for _ in range(closers_to_keep)
                )
                depth -= 2
                self.eat("SHIFT_RIGHT")
                if depth == 0:
                    break
            elif grouped_depth == 0 and token[0] == "COMMA" and depth == 1:
                parsed = self.parse_template_parameter_tokens(parameter_tokens)
                if parsed is not None:
                    parameters.append(parsed)
                    default_type = self.template_parameter_default(parameter_tokens)
                    if default_type:
                        defaults[parsed[1]] = default_type
                parameter_tokens = []
                self.eat("COMMA")
            else:
                parameter_tokens.append(token)
                self.eat(token[0])

        parsed = self.parse_template_parameter_tokens(parameter_tokens)
        if parsed is not None:
            parameters.append(parsed)
            default_type = self.template_parameter_default(parameter_tokens)
            if default_type:
                defaults[parsed[1]] = default_type
        self.last_template_parameter_defaults = defaults
        return parameters

    def parse_template_parameter_tokens(self, tokens):
        declarator_tokens = tokens
        for idx, (token_type, _value) in enumerate(tokens):
            if token_type == "EQUALS":
                declarator_tokens = tokens[:idx]
                break

        for idx, (token_type, value) in enumerate(declarator_tokens):
            if value == "typename" or token_type == "CLASS":
                name_idx = idx + 1
                is_variadic = False
                if (
                    name_idx < len(declarator_tokens)
                    and declarator_tokens[name_idx][0] == "ELLIPSIS"
                ):
                    is_variadic = True
                    name_idx += 1
                if (
                    name_idx < len(declarator_tokens)
                    and declarator_tokens[name_idx][0] == "IDENTIFIER"
                ):
                    kind = f"{value}..." if is_variadic else value
                    return (kind, declarator_tokens[name_idx][1])
        if len(declarator_tokens) >= 2 and declarator_tokens[-1][0] == "IDENTIFIER":
            return ("value", declarator_tokens[-1][1])
        return None

    def template_parameter_default(self, tokens):
        for idx, (token_type, _value) in enumerate(tokens):
            if token_type != "EQUALS":
                continue
            default_tokens = [token[1] for token in tokens[idx + 1 :]]
            default_text = self.format_generic_type_tokens(default_tokens).strip()
            return default_text or None
        return None

    def template_variable_declaration_name_at(self, idx):
        idx = self.skip_leading_attribute_tokens_at(idx)
        while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
            idx += 1
        if idx >= len(self.tokens) or self.tokens[idx][0] not in TYPE_TOKENS:
            return None

        idx += 1
        idx = self.skip_scoped_type_suffix_at(idx)
        if idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
            idx = self.skip_template_argument_list_at(idx)

        while idx < len(self.tokens):
            if self.is_qualifier_token_at(idx):
                idx += 1
                continue
            if self.tokens[idx][0] in {"MULTIPLY", "BITWISE_AND"}:
                idx += 1
                continue
            break

        idx = self.skip_leading_attribute_tokens_at(idx)
        if idx >= len(self.tokens) or not self.is_name_token_at(idx):
            return None

        name = self.tokens[idx][1]
        if idx + 1 < len(self.tokens) and self.tokens[idx + 1][0] == "LPAREN":
            return None
        return name

    def template_function_declaration_name_at(self, idx):
        idx = self.skip_leading_attribute_tokens_at(idx)
        while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
            idx += 1
            idx = self.skip_leading_attribute_tokens_at(idx)
        if idx >= len(self.tokens):
            return None

        if idx < len(self.tokens) and self.tokens[idx][0] in STAGE_TOKENS:
            idx += 1
            idx = self.skip_leading_attribute_tokens_at(idx)
            while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
                idx += 1
                idx = self.skip_leading_attribute_tokens_at(idx)

        if idx >= len(self.tokens) or self.tokens[idx][0] not in TYPE_TOKENS:
            return None

        idx += 1
        idx = self.skip_scoped_type_suffix_at(idx)
        if idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
            idx = self.skip_template_argument_list_at(idx)

        while idx < len(self.tokens):
            if self.is_qualifier_token_at(idx):
                idx += 1
                continue
            if self.tokens[idx][0] in {"MULTIPLY", "BITWISE_AND"}:
                idx += 1
                continue
            break

        idx = self.skip_leading_attribute_tokens_at(idx)
        if idx >= len(self.tokens) or not self.is_name_token_at(idx):
            return None
        name = self.tokens[idx][1]
        idx += 1
        if idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
            idx = self.skip_template_argument_list_at(idx)
        if idx >= len(self.tokens) or self.tokens[idx][0] != "LPAREN":
            return None
        end = self.skip_balanced_tokens_at(idx, "LPAREN", "RPAREN")
        return (
            name
            if end < len(self.tokens) and self.tokens[end][0] == "SEMICOLON"
            else None
        )

    def parse_function_template_definition_or_prototype(self):
        """Parse a function template, keeping ``R f(params);`` as a body-less node.

        The prototype is recorded like any other dropped prototype, so calls of
        a template nothing defines stay refused, and
        :mod:`.declared_template_primaries` keeps the node only when explicit
        specializations complete it.
        """
        function = self.parse_function(preserve_declaration=True)
        if function is None or function.body is not None:
            return function
        self.function_prototypes.setdefault(
            function.name, len(self.function_prototypes)
        )
        function.is_template_prototype = True
        function.declaration_kind = "declaration"
        return function

    def skip_template_declaration(self):
        paren_depth = 0
        brace_depth = 0
        while self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if token_type == "LPAREN":
                paren_depth += 1
            elif token_type == "RPAREN":
                paren_depth = max(0, paren_depth - 1)
            elif token_type == "LBRACE":
                brace_depth += 1
            elif token_type == "RBRACE":
                if brace_depth > 0:
                    brace_depth -= 1
                    self.eat("RBRACE")
                    if brace_depth == 0 and paren_depth == 0:
                        if self.current_token[0] == "SEMICOLON":
                            self.eat("SEMICOLON")
                        return
                    continue
                self.eat("RBRACE")
                if self.current_token[0] == "SEMICOLON":
                    self.eat("SEMICOLON")
                return
            elif token_type == "SEMICOLON" and paren_depth == 0 and brace_depth == 0:
                self.eat("SEMICOLON")
                return
            self.eat(token_type)
