"""Struct, class and union declarations in the Metal parser: base clauses, members, constructors and initializers, nested aggregates, access specifiers and constant buffers."""

from crosstl.translator.refusals import MetalSyntaxError

from ..local_aggregate_declarators import unnamed_aggregate_name
from ..member_aggregates import (
    anonymous_member_declaration,
    parse_unnamed_member_aggregate,
    unnamed_member_aggregate_kind,
)
from ..member_enums import (
    is_anonymous_member_enum_start,
    parse_anonymous_member_enum,
)
from ..MetalAst import (
    AssignmentNode,
    ConstantBufferNode,
    ConstructorInitializerNode,
    ConstructorNode,
    StructNode,
    TypeAliasNode,
    VariableNode,
)
from .parser_tables import (
    GNU_EXTENSION_PREFIXES,
    MACRO_QUALIFIERS,
    STRUCT_METHOD_PREFIXES,
)


class MetalRecordParsingMixin:
    """Record members of ``MetalParser``."""

    def parse_struct(self, pre_alignas=None, allow_unnamed=False, anonymous=False):
        start_token = self.current_token
        alignas_specs = (
            list(pre_alignas)
            if pre_alignas is not None
            else self.parse_alignas_specifiers()
        )
        self.eat("STRUCT")
        struct_attributes = self.parse_attributes()
        alignas_specs.extend(self.parse_alignas_specifiers())
        unnamed = allow_unnamed and self.current_token[0] == "LBRACE"
        if unnamed:
            name = self.reserve_unnamed_aggregate_name("struct")
        elif not self.is_current_name_token():
            raise MetalSyntaxError(f"Expected identifier, got {self.current_token[0]}")
        else:
            name = self.current_token[1]
            self.eat(self.current_token[0])
            self.register_known_type(name)
        if self.current_token[0] == "LESS_THAN":
            template_args = self.parse_template_argument_suffix()
            name = f"{name}<{self.format_generic_type_tokens(template_args)}>"
            self.register_known_type(name)
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return None
        self.push_declaration_context(f"struct {name}")
        base_types = []
        if self.current_token[0] == "COLON":
            base_types = self.parse_struct_base_clause()
        self.eat("LBRACE")

        members = self.parse_struct_members(name)

        struct_node = self.annotate_declaration_scope(StructNode(name, members))
        struct_node.attributes = struct_attributes
        struct_node.alignas = alignas_specs
        struct_node.base_types = base_types
        struct_node.type_aliases = list(self.last_struct_type_aliases)
        struct_node.constructors = list(self.last_struct_constructors)
        struct_node.declares_assignment_operator = (
            self.last_struct_declares_assignment_operator
        )
        struct_node.declares_subscript_operator = (
            self.last_struct_declares_subscript_operator
        )
        struct_node.call_operator_declarations = list(
            self.last_struct_call_operator_declarations
        )
        struct_node.nested_structs = list(self.last_struct_nested_aggregates)
        for constructor in struct_node.constructors:
            constructor.owner_name = name
            constructor.name = name
            constructor.owner_qualified_name = struct_node.qualified_name
        for alias in struct_node.type_aliases:
            alias.owner_name = name
            alias.owner_qualified_name = struct_node.qualified_name
        self.eat("RBRACE")
        struct_node.trailing_declarations = self.parse_trailing_aggregate_declarations(
            name
        )
        if unnamed:
            self.require_unnamed_aggregate_declarators(struct_node, "struct", anonymous)
        struct_node.declaration_source_location = self.source_span_from_tokens(
            start_token, self.tokens[self.pos - 1]
        )
        self.pop_declaration_context()
        return struct_node

    def parse_class(self, pre_alignas=None):
        start_token = self.current_token
        alignas_specs = (
            list(pre_alignas)
            if pre_alignas is not None
            else self.parse_alignas_specifiers()
        )
        self.eat("CLASS")
        class_attributes = self.parse_attributes()
        alignas_specs.extend(self.parse_alignas_specifiers())
        if not self.is_current_name_token():
            raise MetalSyntaxError(f"Expected identifier, got {self.current_token[0]}")
        name = self.current_token[1]
        self.eat(self.current_token[0])
        self.register_known_type(name)
        if self.current_token[0] == "LESS_THAN":
            template_args = self.parse_template_argument_suffix()
            name = f"{name}<{self.format_generic_type_tokens(template_args)}>"
            self.register_known_type(name)
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return None
        self.push_declaration_context(f"class {name}")
        base_types = []
        if self.current_token[0] == "COLON":
            base_types = self.parse_struct_base_clause()
        self.eat("LBRACE")

        members = self.parse_struct_members(name)

        class_node = self.annotate_declaration_scope(StructNode(name, members))
        class_node.attributes = class_attributes
        class_node.alignas = alignas_specs
        class_node.base_types = base_types
        class_node.aggregate_kind = "class"
        class_node.type_aliases = list(self.last_struct_type_aliases)
        class_node.constructors = list(self.last_struct_constructors)
        class_node.declares_assignment_operator = (
            self.last_struct_declares_assignment_operator
        )
        class_node.declares_subscript_operator = (
            self.last_struct_declares_subscript_operator
        )
        class_node.call_operator_declarations = list(
            self.last_struct_call_operator_declarations
        )
        class_node.nested_structs = list(self.last_struct_nested_aggregates)
        for constructor in class_node.constructors:
            constructor.owner_name = name
            constructor.name = name
            constructor.owner_qualified_name = class_node.qualified_name
        for alias in class_node.type_aliases:
            alias.owner_name = name
            alias.owner_qualified_name = class_node.qualified_name
        self.eat("RBRACE")
        class_node.trailing_declarations = self.parse_trailing_aggregate_declarations(
            name
        )
        class_node.declaration_source_location = self.source_span_from_tokens(
            start_token, self.tokens[self.pos - 1]
        )
        self.pop_declaration_context()
        return class_node

    def parse_struct_base_clause(self):
        self.eat("COLON")
        bases = []
        parts = []
        depth = 0
        paired_closers = {
            "LESS_THAN": "GREATER_THAN",
            "LPAREN": "RPAREN",
            "LBRACKET": "RBRACKET",
        }
        open_tokens = set(paired_closers)
        close_tokens = set(paired_closers.values())

        while self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if token_type == "LBRACE" or (depth == 0 and token_type == "COMMA"):
                if parts:
                    bases.append(self.format_generic_type_tokens(parts))
                    parts = []
                if token_type == "COMMA":
                    self.eat("COMMA")
                    continue
                break
            if token_type in open_tokens:
                depth += 1
            elif token_type == "SHIFT_RIGHT" and depth > 0:
                depth = max(0, depth - 2)
            elif token_type in close_tokens and depth > 0:
                depth -= 1
            parts.append(token_value)
            self.eat(token_type)

        if parts:
            bases.append(self.format_generic_type_tokens(parts))
        return bases

    def parse_union(self, allow_unnamed=False, anonymous=False):
        start_token = self.current_token
        self.skip_gnu_extension_prefix_tokens()
        self.eat("IDENTIFIER")
        name = self.current_token[1] if self.is_current_name_token() else None
        if name:
            self.eat(self.current_token[0])
            self.register_known_type(name)
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return None
        unnamed = not name and allow_unnamed and self.current_token[0] == "LBRACE"
        if unnamed:
            name = self.reserve_unnamed_aggregate_name("union")
        if not name:
            raise MetalSyntaxError("Expected union name")
        self.push_declaration_context(f"union {name}")
        self.eat("LBRACE")

        members = self.parse_struct_members(name)

        union_node = self.annotate_declaration_scope(StructNode(name, members))
        union_node.aggregate_kind = "union"
        union_node.type_aliases = list(self.last_struct_type_aliases)
        union_node.constructors = list(self.last_struct_constructors)
        union_node.declares_assignment_operator = (
            self.last_struct_declares_assignment_operator
        )
        union_node.declares_subscript_operator = (
            self.last_struct_declares_subscript_operator
        )
        union_node.call_operator_declarations = list(
            self.last_struct_call_operator_declarations
        )
        union_node.nested_structs = list(self.last_struct_nested_aggregates)
        for constructor in union_node.constructors:
            constructor.owner_name = name
            constructor.name = name
            constructor.owner_qualified_name = union_node.qualified_name
        for alias in union_node.type_aliases:
            alias.owner_name = name
            alias.owner_qualified_name = union_node.qualified_name
        self.eat("RBRACE")
        union_node.trailing_declarations = self.parse_trailing_aggregate_declarations(
            name
        )
        if unnamed:
            self.require_unnamed_aggregate_declarators(union_node, "union", anonymous)
        union_node.declaration_source_location = self.source_span_from_tokens(
            start_token, self.tokens[self.pos - 1]
        )
        self.pop_declaration_context()
        return union_node

    def reserve_unnamed_aggregate_name(self, kind):
        self.unnamed_aggregate_count = getattr(self, "unnamed_aggregate_count", 0) + 1
        name = unnamed_aggregate_name(kind, self.unnamed_aggregate_count)
        self.register_known_type(name)
        return name

    def require_unnamed_aggregate_declarators(self, aggregate, kind, anonymous=False):
        aggregate.unnamed_aggregate_kind = kind
        if anonymous and not aggregate.trailing_declarations:
            # An anonymous member of a class: its members are reached through a reserved member name
            # (member_aggregates.py).
            aggregate.trailing_declarations = [
                anonymous_member_declaration(self, aggregate)
            ]
            return
        if not aggregate.trailing_declarations:
            # [class.union.anon]: the members of an anonymous union join the
            # enclosing scope, which hoisting cannot represent.
            raise MetalSyntaxError(
                f"An unnamed block-scope {kind} must declare an object; "
                f"anonymous {kind} members of a block are not supported"
            )

    def parse_trailing_aggregate_declarations(self, aggregate_type):
        declarations = []
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return declarations

        while True:
            name, array_sizes, type_suffix, grouped_suffix = self.parse_declarator()
            decl_type = self.apply_declarator_type_suffix(aggregate_type, type_suffix)
            attributes = self.parse_attributes()
            var_node = VariableNode(decl_type, name, attributes=attributes)
            var_node.array_sizes = array_sizes
            self.apply_declarator_metadata(var_node, type_suffix, grouped_suffix)
            self.register_local_variable_name(name)

            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                declarations.append(AssignmentNode(var_node, self.parse_expression()))
            elif self.current_token[0] == "LPAREN":
                declarations.append(
                    AssignmentNode(var_node, self.parse_vector_constructor(decl_type))
                )
            elif self.current_token[0] == "LBRACE":
                declarations.append(
                    AssignmentNode(var_node, self.parse_initializer_list())
                )
            else:
                declarations.append(var_node)

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            self.eat("SEMICOLON")
            return declarations

    def parse_struct_members(self, owner_name=None):
        members = []
        type_aliases = []
        constructors = []
        call_operator_declarations = []
        nested_aggregates = []
        declares_assignment_operator = False
        declares_subscript_operator = False

        def capture_nested_aggregate(nested):
            if nested is None:
                return
            nested.lexical_owner_name = owner_name
            nested_aggregates.append(nested)
            members.extend(getattr(nested, "trailing_declarations", ()) or ())

        while self.current_token[0] != "RBRACE":
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                continue
            if self.is_access_specifier_label():
                self.parse_access_specifier_label()
                continue
            if self.is_nested_aggregate_declaration_start():
                nested = self.parse_nested_aggregate_declaration()
                capture_nested_aggregate(nested)
                continue
            if self.current_token == ("IDENTIFIER", "friend"):
                self.skip_struct_method()
                continue
            if (
                self.current_token[0] == "IDENTIFIER"
                and self.current_token[1] in STRUCT_METHOD_PREFIXES
            ):
                self.skip_struct_method()
                continue
            if self.current_token[0] == "STATIC_ASSERT":
                members.append(self.parse_static_assert())
                continue
            if is_anonymous_member_enum_start(self.tokens, self.pos):
                members.extend(parse_anonymous_member_enum(self))
                continue
            if self.current_token[0] == "USING":
                alias = self.parse_using_statement()
                if isinstance(alias, TypeAliasNode):
                    type_aliases.append(alias)
                continue
            if self.current_token[0] == "TYPEDEF":
                alias = self.parse_typedef()
                if isinstance(alias, TypeAliasNode):
                    self.annotate_declaration_scope(alias)
                    type_aliases.append(alias)
                continue
            if self.is_template_declaration_start():
                if self.is_template_alias_declaration_start():
                    alias = self.parse_template_declaration()
                    if isinstance(alias, TypeAliasNode):
                        type_aliases.append(alias)
                    continue
                if owner_name and self.is_template_constructor_declaration_start(
                    owner_name
                ):
                    constructors.append(
                        self.parse_template_constructor_declaration(owner_name)
                    )
                    continue
                declares_assignment_operator |= (
                    self.skipped_member_declares_assignment_operator()
                )
                declares_subscript_operator |= (
                    self.skipped_member_declares_subscript_operator()
                )
                self.skip_template_declaration()
                continue
            member_alignas = self.parse_alignas_specifiers()
            if self.is_nested_aggregate_declaration_start():
                nested = self.parse_nested_aggregate_declaration(member_alignas)
                capture_nested_aggregate(nested)
                continue
            if owner_name and self.is_struct_call_operator_declaration_start():
                declaration = self.parse_function(preserve_declaration=True)
                declaration.owner_name = owner_name
                namespace = self.current_namespace()
                declaration.owner_qualified_name = (
                    f"{namespace}::{owner_name}" if namespace else owner_name
                )
                declaration.qualified_name = (
                    f"{declaration.owner_qualified_name}::operator()"
                )
                declaration.declaration_kind = (
                    "definition" if declaration.body is not None else "declaration"
                )
                declaration.alignas = member_alignas
                call_operator_declarations.append(declaration)
                continue
            if self.is_struct_conversion_operator_start():
                self.skip_struct_method()
                continue
            if self.is_struct_destructor_start():
                self.skip_struct_method()
                continue
            if owner_name and self.is_struct_constructor_start(owner_name):
                constructor = self.parse_struct_constructor(owner_name)
                constructor.alignas = member_alignas
                constructors.append(constructor)
                continue
            vtype, qualifiers = self.parse_type_specifier()
            if self.current_token[0] == "OPERATOR":
                declares_assignment_operator |= (
                    self.skipped_member_declares_assignment_operator()
                )
                declares_subscript_operator |= (
                    self.skipped_member_declares_subscript_operator()
                )
                self.skip_struct_method()
                continue
            qualifier_contract = dict(self.last_type_specifier_qualifier_contract)
            var_name, array_sizes, type_suffix, grouped_suffix = self.parse_declarator()
            member_type = self.apply_declarator_type_suffix(vtype, type_suffix)
            if var_name == "operator":
                declares_assignment_operator |= self.current_token[0] == "EQUALS"
                # The declarator parser reads ``operator[]`` as an array of unknown bound.
                declares_subscript_operator |= bool(array_sizes) or (
                    self.current_token[0] == "LBRACKET"
                )
                self.skip_struct_method()
                continue
            if self.current_token[0] == "LPAREN":
                self.skip_struct_method()
                continue
            attributes = self.parse_attributes()
            array_sizes.extend(self.parse_declarator_array_sizes())
            default_value = None
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                default_value = self.parse_expression()
            elif self.current_token[0] == "LBRACE":
                default_value = self.parse_initializer_list()
            var_node = VariableNode(
                member_type, var_name, qualifiers=qualifiers, attributes=attributes
            )
            var_node.pointee_qualifiers = list(qualifier_contract["pointee"])
            var_node.indirection_qualifiers = list(qualifier_contract["indirection"])
            var_node.array_sizes = array_sizes
            self.apply_declarator_metadata(var_node, type_suffix, grouped_suffix)
            var_node.alignas = member_alignas
            var_node.default_value = default_value
            members.append(var_node)
            if self.current_token[0] == "COMMA":
                members.extend(
                    self.parse_remaining_struct_member_declarations(
                        self.base_type_for_remaining_declarators(vtype),
                        qualifiers,
                        member_alignas,
                    )
                )
                continue
            self.eat("SEMICOLON")
        self.last_struct_type_aliases = type_aliases
        self.last_struct_constructors = constructors
        self.last_struct_call_operator_declarations = call_operator_declarations
        self.last_struct_nested_aggregates = nested_aggregates
        self.last_struct_declares_assignment_operator = declares_assignment_operator
        self.last_struct_declares_subscript_operator = declares_subscript_operator
        return members

    def skipped_member_declares_assignment_operator(self):
        """Whether the member declaration at the cursor declares ``operator=``.

        The struct parser skips member operators, but a user-declared copy assignment operator replaces the
        member-wise assignment an implicit conversion would otherwise feed ([class.copy.assign]).
        """
        index = self.pos
        while index + 1 < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type in {"LBRACE", "SEMICOLON", "EOF"}:
                return False
            names_operator = token_type == "OPERATOR" or self.tokens[index] == (
                "IDENTIFIER",
                "operator",
            )
            if names_operator and self.tokens[index + 1][0] == "EQUALS":
                return True
            index += 1
        return False

    def skipped_member_declares_subscript_operator(self):
        """Whether the member declaration at the cursor declares ``operator[]``.

        The struct parser skips member operators, but a subscript of a dependent value may select a user-declared
        subscript operator when its template is instantiated ([over.sub]), which the importer has to know.
        """
        index = self.pos
        while index + 2 < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type in {"LBRACE", "SEMICOLON", "EOF"}:
                return False
            if (
                self.tokens[index] == ("IDENTIFIER", "operator")
                and self.tokens[index + 1][0] == "LBRACKET"
                and self.tokens[index + 2][0] == "RBRACKET"
            ):
                return True
            index += 1
        return False

    def is_struct_call_operator_declaration_start(self):
        if not self.is_operator_function_definition():
            return False
        idx = self.pos
        while idx + 3 < len(self.tokens):
            token = self.tokens[idx]
            if token == ("IDENTIFIER", "operator"):
                return (
                    self.tokens[idx + 1][0] == "LPAREN"
                    and self.tokens[idx + 2][0] == "RPAREN"
                    and self.tokens[idx + 3][0] == "LPAREN"
                )
            if self.tokens[idx][0] in {"SEMICOLON", "LBRACE", "RBRACE", "EOF"}:
                return False
            idx += 1
        return False

    def is_template_constructor_declaration_start(self, owner_name):
        if not self.is_template_declaration_start():
            return False
        angle_start = self.pos + 1
        if (
            angle_start >= len(self.tokens)
            or self.tokens[angle_start][0] != "LESS_THAN"
        ):
            return False
        declaration_start = self.skip_template_argument_list_at(angle_start)
        return self.is_struct_constructor_start_at(declaration_start, owner_name)

    def is_struct_constructor_start(self, owner_name):
        return self.is_struct_constructor_start_at(self.pos, owner_name)

    def is_struct_constructor_start_at(self, idx, owner_name):
        owner = str(owner_name).split("<", 1)[0].rsplit("::", 1)[-1]
        while idx < len(self.tokens):
            token_type, token_value = self.tokens[idx]
            if token_type == "ATTRIBUTE":
                idx += 1
                continue
            if token_type == "ALIGNAS":
                idx = self.skip_alignas_specifier_tokens_at(idx)
                continue
            if self.is_qualifier_token_at(idx) or (
                token_type == "IDENTIFIER"
                and token_value in {
                    "explicit",
                    "inline",
                    "constexpr",
                    *MACRO_QUALIFIERS,
                }
            ):
                idx += 1
                continue
            break
        return (
            idx + 1 < len(self.tokens)
            and self.tokens[idx][1] == owner
            and self.tokens[idx + 1][0] == "LPAREN"
        )

    def parse_template_constructor_declaration(self, owner_name):
        start_token = self.current_token
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
        added_value_names = template_value_names - self.known_value_template_parameters
        self.known_types.update(added_type_names)
        self.known_value_template_parameters.update(added_value_names)
        try:
            constructor = self.parse_struct_constructor(owner_name)
            constructor.template_parameters = list(template_parameters)
            constructor.template_parameter_defaults = template_parameter_defaults
            constructor.generics = [name for _kind, name in template_parameters if name]
            constructor.template_source_location = self.source_span_from_tokens(
                start_token, self.tokens[self.pos - 1]
            )
            if constructor.source_location is None:
                constructor.source_location = constructor.template_source_location
            return constructor
        finally:
            self.known_types.difference_update(added_type_names)
            self.known_value_template_parameters.difference_update(added_value_names)

    def parse_struct_constructor(self, owner_name):
        start_token = self.current_token
        qualifiers = []
        while True:
            if self.current_token[0] == "ATTRIBUTE":
                qualifiers.append(self.current_token[1])
                self.eat("ATTRIBUTE")
                continue
            if self.is_type_qualifier_start():
                qualifiers.extend(self.parse_type_qualifier())
                continue
            if self.current_token[0] == "IDENTIFIER" and self.current_token[1] in {
                "explicit",
                "inline",
                "constexpr",
                *MACRO_QUALIFIERS,
            }:
                qualifiers.append(self.current_token[1])
                self.eat("IDENTIFIER")
                continue
            break

        owner = str(owner_name).split("<", 1)[0].rsplit("::", 1)[-1]
        if self.current_token[1] != owner:
            raise self.syntax_error(
                f"Expected constructor owner {owner}, got {self.current_token[1]}"
            )
        self.eat(self.current_token[0])
        self.push_declaration_context(f"constructor {owner_name}")
        self.eat("LPAREN")
        params = self.parse_parameters()
        self.eat("RPAREN")

        receiver_qualifiers = []
        while self.current_token[0] not in {
            "COLON",
            "LBRACE",
            "SEMICOLON",
            "EQUALS",
            "EOF",
        }:
            if self.current_token == ("IDENTIFIER", "noexcept"):
                receiver_qualifiers.append(self.current_token[1])
                self.eat("IDENTIFIER")
                if self.current_token[0] == "LPAREN":
                    receiver_qualifiers.append(
                        self.parse_balanced_token_text(
                            "LPAREN",
                            "RPAREN",
                            error_message="Unterminated constructor noexcept qualifier",
                        )
                    )
                continue
            receiver_qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])

        initializers = []
        if self.current_token[0] == "COLON":
            self.eat("COLON")
            while True:
                initializers.append(self.parse_constructor_initializer())
                if self.current_token[0] != "COMMA":
                    break
                self.eat("COMMA")

        declaration_kind = "definition"
        body = None
        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            declaration_kind = str(self.current_token[1])
            self.eat(self.current_token[0])
            self.eat("SEMICOLON")
        elif self.current_token[0] == "SEMICOLON":
            declaration_kind = "declaration"
            self.eat("SEMICOLON")
        else:
            self.pending_block_scope_names.append(
                {param.name for param in params if getattr(param, "name", None)}
            )
            body = self.parse_block()
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")

        constructor = ConstructorNode(
            owner_name,
            params,
            body,
            initializers=initializers,
            qualifiers=[*qualifiers, *receiver_qualifiers],
            declaration_kind=declaration_kind,
            source_location=self.source_span_from_tokens(
                start_token, self.tokens[self.pos - 1]
            ),
        )
        self.pop_declaration_context()
        return constructor

    def parse_constructor_initializer(self):
        start_token = self.current_token
        target_parts = []
        angle_depth = 0
        while self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if angle_depth == 0 and token_type in {"LPAREN", "LBRACE"}:
                break
            if token_type == "LESS_THAN":
                angle_depth += 1
            elif token_type == "GREATER_THAN" and angle_depth:
                angle_depth -= 1
            elif token_type == "SHIFT_RIGHT" and angle_depth >= 2:
                angle_depth -= 2
            target_parts.append(token_value)
            self.eat(token_type)
        target = self.format_generic_type_tokens(target_parts)
        if not target:
            raise self.syntax_error("Expected constructor initializer target")

        if self.current_token[0] == "LPAREN":
            style = "paren"
            arguments = self.parse_parenthesized_arguments()
        elif self.current_token[0] == "LBRACE":
            style = "brace"
            arguments = self.parse_initializer_list().elements
        else:
            raise self.syntax_error(
                "Expected parenthesized or braced constructor initializer"
            )
        return ConstructorInitializerNode(
            target,
            arguments,
            style=style,
            source_location=self.source_span_from_tokens(
                start_token, self.tokens[self.pos - 1]
            ),
        )

    def parse_remaining_struct_member_declarations(
        self, vtype, qualifiers, alignas_specs
    ):
        members = []
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            name, array_sizes, type_suffix, grouped_suffix = self.parse_declarator()
            member_type = self.apply_declarator_type_suffix(vtype, type_suffix)
            attributes = self.parse_attributes()
            array_sizes.extend(self.parse_declarator_array_sizes())
            default_value = None
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                default_value = self.parse_expression()
            elif self.current_token[0] == "LBRACE":
                default_value = self.parse_initializer_list()

            var_node = VariableNode(
                member_type,
                name,
                qualifiers=list(qualifiers),
                attributes=attributes,
            )
            var_node.array_sizes = array_sizes
            self.apply_declarator_metadata(var_node, type_suffix, grouped_suffix)
            var_node.alignas = list(alignas_specs)
            var_node.default_value = default_value
            members.append(var_node)

        self.eat("SEMICOLON")
        return members

    def is_nested_aggregate_declaration_start(self):
        idx = self.skip_gnu_extension_prefix_tokens_at(self.pos)
        if idx >= len(self.tokens):
            return False
        if self.tokens[idx][0] not in {"STRUCT", "CLASS"} and self.tokens[idx] != (
            "IDENTIFIER",
            "union",
        ):
            return False

        idx += 1
        idx = self.skip_alignas_specifier_tokens_at(idx)
        if idx < len(self.tokens) and self.is_name_token_at(idx):
            idx += 1
            if idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
                idx = self.skip_template_argument_list_at(idx)
            idx = self.skip_alignas_specifier_tokens_at(idx)

        if idx >= len(self.tokens):
            return False
        return self.tokens[idx][0] in {"LBRACE", "COLON", "SEMICOLON"}

    def skip_gnu_extension_prefix_tokens_at(self, idx):
        while (
            idx < len(self.tokens)
            and self.tokens[idx][0] == "IDENTIFIER"
            and self.tokens[idx][1] in GNU_EXTENSION_PREFIXES
        ):
            idx += 1
        return idx

    def skip_gnu_extension_prefix_tokens(self):
        while (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] in GNU_EXTENSION_PREFIXES
        ):
            self.eat("IDENTIFIER")

    def skip_alignas_specifier_tokens_at(self, idx):
        while idx < len(self.tokens) and self.tokens[idx][0] == "ALIGNAS":
            idx += 1
            if idx < len(self.tokens) and self.tokens[idx][0] == "LPAREN":
                idx = self.skip_balanced_tokens_at(idx, "LPAREN", "RPAREN")
        return idx

    def parse_nested_aggregate_declaration(self, pre_alignas=None):
        self.skip_gnu_extension_prefix_tokens()
        declaration_start = self.pos
        if self.current_token[0] in {"STRUCT", "CLASS"} or self.current_token == (
            "IDENTIFIER",
            "union",
        ):
            name_index = self.skip_leading_attribute_tokens_at(declaration_start + 1)
            name_index = self.skip_alignas_specifier_tokens_at(name_index)
            if name_index >= len(self.tokens) or not self.is_name_token_at(name_index):
                if unnamed_member_aggregate_kind(self) is not None:
                    return parse_unnamed_member_aggregate(self, pre_alignas)
                self.skip_nested_aggregate_declaration()
                return None

        if self.current_token[0] == "STRUCT":
            return self.parse_struct(pre_alignas)
        if self.current_token[0] == "CLASS":
            return self.parse_class(pre_alignas)
        if self.current_token == ("IDENTIFIER", "union"):
            nested = self.parse_union()
            if nested is not None and pre_alignas:
                nested.alignas = list(pre_alignas)
            return nested
        raise MetalSyntaxError(
            f"Expected nested aggregate declaration, got {self.current_token[0]}"
        )

    def skip_nested_aggregate_declaration(self):
        paren_depth = 0
        brace_depth = 0
        bracket_depth = 0
        angle_depth = 0

        while self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if (
                token_type == "SEMICOLON"
                and paren_depth == 0
                and brace_depth == 0
                and bracket_depth == 0
                and angle_depth == 0
            ):
                self.eat("SEMICOLON")
                return

            if token_type == "LPAREN":
                paren_depth += 1
            elif token_type == "RPAREN" and paren_depth > 0:
                paren_depth -= 1
            elif token_type == "LBRACE":
                brace_depth += 1
            elif token_type == "RBRACE" and brace_depth > 0:
                brace_depth -= 1
            elif token_type == "LBRACKET":
                bracket_depth += 1
            elif token_type == "RBRACKET" and bracket_depth > 0:
                bracket_depth -= 1
            elif token_type == "LESS_THAN" and brace_depth == 0:
                angle_depth += 1
            elif token_type == "SHIFT_RIGHT" and brace_depth == 0 and angle_depth >= 2:
                angle_depth -= 2
            elif token_type == "GREATER_THAN" and brace_depth == 0 and angle_depth > 0:
                angle_depth -= 1

            self.eat(token_type)

        raise MetalSyntaxError("Unterminated nested aggregate declaration")

    def is_struct_conversion_operator_start(self):
        idx = self.pos
        while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
            idx += 1
        while idx < len(self.tokens) and self.tokens[idx] == ("IDENTIFIER", "explicit"):
            idx += 1
        return idx < len(self.tokens) and self.tokens[idx] == ("IDENTIFIER", "operator")

    def is_struct_destructor_start(self):
        return (
            self.current_token[0] == "BITWISE_NOT"
            and self.pos + 2 < len(self.tokens)
            and self.is_name_token_at(self.pos + 1)
            and self.tokens[self.pos + 2][0] == "LPAREN"
        )

    def is_access_specifier_label(self):
        return (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] in {"public", "private", "protected"}
            and self.peek(1)[0] == "COLON"
        )

    def parse_access_specifier_label(self):
        self.eat("IDENTIFIER")
        self.eat("COLON")

    def skip_struct_method(self):
        while self.current_token[0] not in {"LBRACE", "SEMICOLON", "EOF"}:
            self.eat(self.current_token[0])
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return
        if self.current_token[0] == "LBRACE":
            self.skip_balanced_block()
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")

    def skip_balanced_block(self):
        self.eat("LBRACE")
        depth = 1
        while depth > 0 and self.current_token[0] != "EOF":
            if self.current_token[0] == "LBRACE":
                depth += 1
            elif self.current_token[0] == "RBRACE":
                depth -= 1
                if depth == 0:
                    self.eat("RBRACE")
                    break
            self.eat(self.current_token[0])

    def parse_constant_buffer(self):
        self.eat("CONSTANT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LBRACE")

        members = []
        while self.current_token[0] != "RBRACE":
            member_alignas = self.parse_alignas_specifiers()
            vtype, qualifiers = self.parse_type_specifier()
            var_name, array_sizes, type_suffix, grouped_suffix = self.parse_declarator()
            member_type = self.apply_declarator_type_suffix(vtype, type_suffix)
            self.eat("SEMICOLON")
            var_node = VariableNode(member_type, var_name, qualifiers=qualifiers)
            var_node.array_sizes = array_sizes
            self.apply_declarator_metadata(var_node, type_suffix, grouped_suffix)
            var_node.alignas = member_alignas
            members.append(var_node)

        self.eat("RBRACE")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

        return ConstantBufferNode(name, members)
