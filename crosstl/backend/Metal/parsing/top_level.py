"""Top-level declarations in the Metal parser: out-of-line call-operator association, global declaration registration, top-level expression and parameter fragments, bare macro invocations, namespaces and constant-buffer detection."""

import re

from crosstl.translator.refusals import MetalSyntaxError

from ..MetalAst import (
    AssignmentNode,
    VariableNode,
)
from .parser_tables import (
    QUALIFIER_TOKENS,
    TYPE_TOKENS,
    MetalCallOperatorAssociationError,
)


class MetalTopLevelParsingMixin:
    """Top level members of ``MetalParser``."""

    def associate_out_of_line_call_operators(self, shader):
        declarations_by_owner = {}
        for struct_node in getattr(shader, "structs", []) or []:
            declarations = list(
                getattr(struct_node, "call_operator_declarations", []) or []
            )
            if not declarations:
                continue
            owner_names = {
                str(getattr(struct_node, "name", "")),
                str(getattr(struct_node, "qualified_name", "")),
            }
            for owner_name in owner_names - {""}:
                declarations_by_owner.setdefault(owner_name, []).extend(declarations)

        definitions = [
            function
            for function in getattr(shader, "functions", []) or []
            if getattr(function, "out_of_line_call_operator_owner", None)
        ]
        definitions_by_contract = {}
        for definition in definitions:
            owner = definition.out_of_line_call_operator_owner
            owner_declarations = list(declarations_by_owner.get(owner, []))
            if not owner_declarations and "::" in owner:
                owner_declarations = list(
                    declarations_by_owner.get(owner.rsplit("::", 1)[-1], [])
                )
            contract = self.call_operator_contract(definition)
            matches = [
                declaration
                for declaration in owner_declarations
                if self.call_operator_contract(declaration) == contract
            ]
            signature = self.call_operator_signature(definition)
            if len(matches) != 1:
                reason = (
                    "no declaration matches the definition contract"
                    if not matches
                    else "multiple declarations match the definition contract"
                )
                raise MetalCallOperatorAssociationError(
                    owner,
                    signature,
                    reason,
                    definition_location=getattr(
                        definition, "declaration_source_location", None
                    ),
                    declaration_locations=tuple(
                        getattr(candidate, "declaration_source_location", None)
                        for candidate in matches or owner_declarations
                    ),
                    candidates=tuple(
                        self.call_operator_signature(candidate)
                        for candidate in matches or owner_declarations
                    ),
                )

            declaration = matches[0]
            definition.out_of_line_call_operator_declaration = {
                "owner": getattr(declaration, "owner_qualified_name", owner),
                "signature": self.call_operator_signature(declaration),
                "source_location": getattr(
                    declaration, "declaration_source_location", None
                ),
            }
            key = (owner, contract)
            definitions_by_contract.setdefault(key, []).append(definition)

        for (owner, _contract), matching_definitions in definitions_by_contract.items():
            if len(matching_definitions) == 1:
                continue
            first = matching_definitions[0]
            raise MetalCallOperatorAssociationError(
                owner,
                self.call_operator_signature(first),
                "multiple definitions match one declaration contract",
                definition_location=getattr(
                    matching_definitions[-1], "declaration_source_location", None
                ),
                declaration_locations=(
                    getattr(first, "out_of_line_call_operator_declaration", {}).get(
                        "source_location"
                    ),
                ),
                candidates=tuple(
                    self.call_operator_signature(definition)
                    for definition in matching_definitions
                ),
            )

    @classmethod
    def call_operator_contract(cls, function):
        return (
            cls.normalize_contract_type(getattr(function, "return_type", "")),
            tuple(
                (
                    cls.normalize_contract_type(getattr(parameter, "vtype", "")),
                    tuple(getattr(parameter, "qualifiers", []) or []),
                    len(getattr(parameter, "array_sizes", []) or []),
                )
                for parameter in getattr(function, "params", []) or []
            ),
            tuple(getattr(function, "method_qualifiers", []) or []),
        )

    @classmethod
    def call_operator_signature(cls, function):
        parameter_types = ", ".join(
            cls.normalize_contract_type(getattr(parameter, "vtype", ""))
            for parameter in getattr(function, "params", []) or []
        )
        qualifiers = " ".join(getattr(function, "method_qualifiers", []) or [])
        suffix = f" {qualifiers}" if qualifiers else ""
        return f"operator()({parameter_types}){suffix}"

    @staticmethod
    def normalize_contract_type(type_name):
        text = re.sub(r"\s+", " ", str(type_name or "").strip())
        return re.sub(r"\s*([<>,*&])\s*", r"\1", text)

    def add_global_declaration(self, global_variables, declaration):
        declarations = declaration if isinstance(declaration, list) else [declaration]
        for item in declarations:
            scoped_item = (
                item.left
                if isinstance(item, AssignmentNode)
                and isinstance(item.left, VariableNode)
                else item
            )
            self.annotate_declaration_scope(scoped_item)
        if isinstance(declaration, list):
            global_variables.extend(declaration)
        else:
            global_variables.append(declaration)

    def is_bare_macro_invocation(self):
        return (
            self.current_token[0] == "IDENTIFIER"
            and self.peek(1)[0] == "LPAREN"
            and not self.is_gnu_attribute_start()
        )

    def is_top_level_expression_statement_start(self):
        return self.current_token[0] == "IDENTIFIER" and self.peek(1)[0] == "DOT"

    def is_top_level_parameter_fragment_start(self):
        if not (
            self.current_token[0] in TYPE_TOKENS
            or self.current_token[0] in QUALIFIER_TOKENS
            or self.is_type_qualifier_start()
        ):
            return False

        idx = self.pos
        depth = 0
        saw_comma = False
        while idx < len(self.tokens):
            token_type = self.tokens[idx][0]
            if token_type == "EOF":
                return saw_comma
            if depth == 0 and token_type in {"SEMICOLON", "LBRACE"}:
                return False
            if depth == 0 and token_type == "COMMA":
                saw_comma = True
            if token_type in {"LPAREN", "LBRACKET", "LESS_THAN"}:
                depth += 1
            elif token_type in {"RPAREN", "RBRACKET", "GREATER_THAN"} and depth > 0:
                depth -= 1
            idx += 1
        return False

    def parse_top_level_parameter_fragment(self):
        while self.current_token[0] != "EOF":
            self.parse_attributes()
            self.parse_type_specifier()
            self.parse_declarator()
            self.parse_attributes()
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                self.parse_expression()
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                break
            if self.current_token[0] == "EOF":
                break
            raise MetalSyntaxError(
                f"Expected comma or end of parameter fragment, got {self.current_token[0]}"
            )

    def skip_bare_macro_invocation(self):
        self.eat("IDENTIFIER")
        self.eat("LPAREN")
        depth = 1
        while depth > 0 and self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    self.eat("RPAREN")
                    break
            self.eat(token_type)
        if depth != 0:
            raise MetalSyntaxError("Unterminated macro invocation")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

    def parse_namespace_start(self):
        self.eat("NAMESPACE")
        namespace_name = None
        if self.current_token[0] in {"IDENTIFIER", "METAL"}:
            namespace_name = self.parse_scoped_identifier()
        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            target_name = self.parse_scoped_identifier()
            if namespace_name:
                self.namespace_aliases[namespace_name] = target_name
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
            return
        if self.current_token[0] == "LBRACE":
            self.eat("LBRACE")
            components = tuple(
                component
                for component in str(namespace_name or "").split("::")
                if component
            )
            self.namespace_scope_stack.append(components)
        elif self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

    def is_constant_buffer(self):
        if self.current_token[0] != "CONSTANT":
            return False
        next_tok = self.peek(1)
        next_next = self.peek(2)
        return next_tok[0] == "IDENTIFIER" and next_next[0] == "LBRACE"
