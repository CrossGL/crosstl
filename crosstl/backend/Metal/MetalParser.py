"""Parser for Metal source AST construction."""

import re  # noqa: F401
import sys

from crosstl.translator.refusals import MetalSyntaxError

from ..switch_clauses import record_default_position  # noqa: F401
from .declared_template_primaries import retain_declared_template_primaries
from .explicit_entry_instantiations import (  # noqa: F401
    instantiated_entry_name,
    refuse_unmaterialized_entry_instantiations,
)
from .floating_point import contraction_requirement
from .local_aggregate_declarators import (  # noqa: F401
    split_block_aggregate_declaration,
    unnamed_aggregate_name,
)
from .member_enums import (  # noqa: F401
    is_anonymous_member_enum_start,
    parse_anonymous_member_enum,
)
from .MetalAst import *
from .MetalLexer import *
from .parsing.aliases import MetalAliasParsingMixin
from .parsing.declaration_lookahead import MetalDeclarationLookaheadParsingMixin
from .parsing.declarators import MetalDeclaratorParsingMixin
from .parsing.expressions import MetalExpressionParsingMixin
from .parsing.functions import MetalFunctionParsingMixin
from .parsing.parser_tables import (  # noqa: F401
    CAST_TYPE_QUALIFIER_NAMES,
    CONSTRUCTOR_TYPE_TOKENS,
    GNU_EXTENSION_PREFIXES,
    IDENTIFIER_PREFIX_TYPE_QUALIFIERS,
    IDENTIFIER_TYPE_QUALIFIERS,
    KEYWORD_IDENTIFIER_TOKENS,
    MACRO_QUALIFIERS,
    OPERATOR_OVERLOAD_TOKENS,
    QUALIFIER_TOKENS,
    RAYTRACING_TYPE_QUALIFIERS,
    SCOPED_IDENTIFIER_PART_TOKENS,
    SIGNED_PREFIX_TYPE_TOKENS,
    SIGNED_TYPE_PREFIXES,
    STAGE_ATTRIBUTE_NAMES,
    STAGE_TOKENS,
    STRUCT_METHOD_PREFIXES,
    TEMPLATE_IDENTIFIER_SUFFIX_FOLLOW_TOKENS,
    TEMPLATE_VARIABLE_SUFFIX_FOLLOW_TOKENS,
    TYPE_IDENTIFIER_TOKENS,
    TYPE_QUALIFIER_FUNCTIONS,
    TYPE_TOKENS,
    UNARY_KEYWORDS,
    UNSIGNED_INTEGER_TYPES,
    MetalCallOperatorAssociationError,
)
from .parsing.postfix import MetalPostfixParsingMixin
from .parsing.primary import MetalPrimaryParsingMixin
from .parsing.records import MetalRecordParsingMixin
from .parsing.statements import MetalStatementParsingMixin
from .parsing.templates import MetalTemplateParsingMixin
from .parsing.top_level import MetalTopLevelParsingMixin
from .parsing.types import MetalTypeParsingMixin
from .preprocessor import DEFAULT_EXPLICIT_TEMPLATE_SPECIALIZATION_LIMIT
from .signed_type_spellings import fold_signed_type_spellings  # noqa: F401
from .simd_aliases import SIMD_TYPE_ALIASES
from .string_literals import concatenate_string_literals  # noqa: F401

MIN_METAL_PARSE_RECURSION_LIMIT = 10000


def ensure_metal_parse_recursion_limit():
    # Dawn/Tint generated MSL can contain deeply nested braced constructor chains.
    if sys.getrecursionlimit() < MIN_METAL_PARSE_RECURSION_LIMIT:
        sys.setrecursionlimit(MIN_METAL_PARSE_RECURSION_LIMIT)


class MetalParserError(SyntaxError):
    def __init__(
        self,
        message,
        *,
        token=None,
        declaration_context=None,
        file_path=None,
    ):
        line = getattr(token, "line", None)
        column = getattr(token, "column", None)
        rendered = message
        if declaration_context:
            rendered = f"{rendered} in {declaration_context}"
        if line is not None and column is not None:
            rendered = f"{rendered} at line {line}, column {column}"
        super().__init__(rendered)
        self.raw_message = message
        self.token = token
        self.declaration_context = declaration_context
        self.project_diagnostic_code = "project.translate.metal-parse-failed"
        self.missing_capabilities = ["metal.parser"]
        if line is not None and column is not None:
            self.source_location = {
                "file": file_path,
                "line": line,
                "column": column,
                "end_line": line,
                "end_column": column + max(1, len(str(token[1]))),
            }


class MetalParser(
    MetalTopLevelParsingMixin,
    MetalTemplateParsingMixin,
    MetalDeclarationLookaheadParsingMixin,
    MetalAliasParsingMixin,
    MetalTypeParsingMixin,
    MetalDeclaratorParsingMixin,
    MetalRecordParsingMixin,
    MetalFunctionParsingMixin,
    MetalStatementParsingMixin,
    MetalExpressionParsingMixin,
    MetalPostfixParsingMixin,
    MetalPrimaryParsingMixin,
):
    """Parse Metal tokens into the Metal backend shader AST."""

    def __init__(
        self,
        tokens,
        file_path=None,
        max_template_specializations=None,
        template_specialization_limit_source=None,
    ):
        self.tokens = tokens
        self.pos = 0
        self.file_path = file_path
        self.explicit_entry_instantiations = []
        self.max_template_specializations = (
            DEFAULT_EXPLICIT_TEMPLATE_SPECIALIZATION_LIMIT
            if max_template_specializations is None
            else max_template_specializations
        )
        self.template_specialization_limit_source = (
            template_specialization_limit_source or "max_template_specializations"
        )
        self.current_token = (
            self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)
        )
        self.skip_comments()
        self.known_types = {
            "void",
            "bool",
            "char",
            "uchar",
            "short",
            "ushort",
            "int",
            "uint",
            "long",
            "ulong",
            "float",
            "half",
            "xhalf",
            "double",
            "size_t",
            "ptrdiff_t",
            "int64_t",
            "uint64_t",
            "int8_t",
            "uint8_t",
            "int16_t",
            "uint16_t",
            "int32_t",
            "uint32_t",
            "sampler",
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
            "acceleration_structure",
            "intersection_function_table",
            "visible_function_table",
            "indirect_command_buffer",
            "atomic_int",
            "atomic_uint",
            "atomic_bool",
            "enum",
            "ray",
            "ray_data",
            "intersection_result",
            "intersection_params",
            "triangle_intersection_params",
            "intersector",
            "packed_float2",
            "packed_float3",
            "packed_float4",
            "xhalf2",
            "xhalf3",
            "xhalf4",
            "xhalf2x2",
            "xhalf2x3",
            "xhalf2x4",
            "xhalf3x2",
            "xhalf3x3",
            "xhalf3x4",
            "xhalf4x2",
            "xhalf4x3",
            "xhalf4x4",
            "packed_half2",
            "packed_half3",
            "packed_half4",
            "packed_int2",
            "packed_int3",
            "packed_int4",
            "packed_uint2",
            "packed_uint3",
            "packed_uint4",
            *SIMD_TYPE_ALIASES,
        }
        self.local_variable_scopes = []
        self.local_type_scopes = []
        self.pending_block_scope_names = []
        self.known_variable_templates = set()
        self.known_function_templates = set()
        # Names of the function prototypes the parser drops (see external_functions).
        self.function_prototypes = {}
        self.known_value_template_parameters = set()
        self.declaration_context_stack = []
        self.namespace_aliases = {}
        self.namespace_scope_stack = []
        self.declared_namespaces = set()
        self.using_namespace_directives = []
        self.using_declarations = []
        self.last_struct_type_aliases = []
        self.last_struct_constructors = []
        self.last_struct_declares_assignment_operator = False
        self.last_struct_declares_subscript_operator = False
        self.last_struct_call_operator_declarations = []
        self.last_struct_nested_aggregates = []
        self.last_type_specifier_qualifier_contract = {
            "pointee": (),
            "indirection": (),
        }

    def skip_comments(self):
        while self.pos < len(self.tokens) and self.current_token[0] in [
            "COMMENT_SINGLE",
            "COMMENT_MULTI",
        ]:
            self.pos += 1
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)
            )

    def expect_end_of_input(self):
        if self.current_token[0] != "EOF":
            raise self.syntax_error(f"Expected EOF, got {self.current_token[0]}")

    def eat(self, token_type):
        # The end of input is never consumed: a construct that reads past it is unterminated.
        if token_type == "EOF" and self.current_token[0] == "EOF":
            raise self.syntax_error("Unexpected end of input")
        if self.current_token[0] == token_type:
            self.pos += 1
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)
            )
            self.skip_comments()
        else:
            raise self.syntax_error(
                f"Expected {token_type}, got {self.current_token[0]}"
            )

    def syntax_error(self, message, token=None):
        return MetalParserError(
            message,
            token=token if token is not None else self.current_token,
            declaration_context=self.current_declaration_context(),
            file_path=self.file_path,
        )

    def source_span_from_tokens(self, start_token, end_token):
        line = getattr(start_token, "line", None)
        column = getattr(start_token, "column", None)
        end_line = getattr(end_token, "line", None)
        end_column = getattr(end_token, "column", None)
        if None in {line, column, end_line, end_column}:
            return None

        end_text = str(end_token[1] or "")
        if "\n" in end_text:
            end_line += end_text.count("\n")
            end_column = len(end_text.rsplit("\n", 1)[-1]) + 1
        else:
            end_column += len(end_text)
        location = {
            "file": self.file_path,
            "line": line,
            "column": column,
            "end_line": end_line,
            "end_column": end_column,
        }
        start_offset = getattr(start_token, "offset", None)
        end_offset = getattr(end_token, "offset", None)
        if start_offset is not None and end_offset is not None:
            location["offset"] = start_offset
            location["length"] = end_offset + len(end_text) - start_offset
            location["end_offset"] = end_offset + len(end_text)
        return location

    def expression_contains_scoped_reference(self, node):
        if node is None or isinstance(node, (str, int, float, bool)):
            return False
        if isinstance(node, VariableNode) and "::" in str(node.name):
            return True
        if isinstance(node, dict):
            return any(
                self.expression_contains_scoped_reference(value)
                for value in node.values()
            )
        if isinstance(node, (list, tuple, set)):
            return any(
                self.expression_contains_scoped_reference(value) for value in node
            )
        return any(
            self.expression_contains_scoped_reference(value)
            for value in getattr(node, "__dict__", {}).values()
        )

    def current_declaration_context(self):
        if not self.declaration_context_stack:
            return None
        return self.declaration_context_stack[-1]

    def annotate_syntax_error(self, exc):
        return MetalParserError(
            str(exc),
            token=self.current_token,
            declaration_context=self.current_declaration_context(),
            file_path=self.file_path,
        )

    def push_declaration_context(self, context):
        self.declaration_context_stack.append(context)

    def pop_declaration_context(self):
        if self.declaration_context_stack:
            self.declaration_context_stack.pop()

    def current_namespace(self):
        return "::".join(
            component
            for scope in self.namespace_scope_stack
            for component in scope
            if component
        )

    def annotate_declaration_scope(self, node):
        if node is None:
            return node
        namespace = self.current_namespace()
        node.namespace = namespace
        node.internal_linkage = any(not scope for scope in self.namespace_scope_stack)
        name = getattr(node, "name", None)
        if name:
            node.qualified_name = f"{namespace}::{name}" if namespace else name
        return node

    def close_namespace_scope(self):
        self.eat("RBRACE")
        if self.namespace_scope_stack:
            self.namespace_scope_stack.pop()

    def peek(self, offset=1):
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return ("EOF", None)

    def parse(self):
        ensure_metal_parse_recursion_limit()
        no_contraction = contraction_requirement(self.tokens)
        try:
            shader = self.parse_shader()
            # A contraction or reassociation directive makes every function @precise (floating_point.py).
            shader.no_contraction = no_contraction
            self.expect_end_of_input()
            refuse_unmaterialized_entry_instantiations(
                self.explicit_entry_instantiations,
                getattr(shader, "functions", None),
                self.max_template_specializations,
            )
            return shader
        except (MetalParserError, MetalCallOperatorAssociationError):
            raise
        except SyntaxError as exc:
            raise self.annotate_syntax_error(exc) from exc

    def parse_shader(self):
        functions = []
        preprocessors = []
        structs = []
        enums = []
        typedefs = []
        constants = []
        global_variables = []

        while self.current_token[0] != "EOF":
            if self.current_token[0] == "PREPROCESSOR":
                directive = self.parse_preprocessor_directive()
                if directive is not None:
                    preprocessors.append(directive)
            elif self.current_token[0] == "NAMESPACE":
                self.parse_namespace_start()
            elif self.current_token[0] == "USING":
                alias = self.parse_using_statement()
                if isinstance(alias, StructNode):
                    structs.append(alias)
                elif alias is not None:
                    typedefs.append(alias)
            elif self.is_union_declaration_start():
                union = self.parse_union()
                if union is not None:
                    structs.append(union)
                    global_variables.extend(getattr(union, "trailing_declarations", []))
            elif self.is_bare_macro_invocation():
                self.skip_bare_macro_invocation()
            elif self.is_top_level_expression_statement_start():
                self.parse_expression_statement()
            elif self.is_template_declaration_start():
                declaration = self.parse_template_declaration()
                if isinstance(declaration, FunctionNode):
                    functions.append(declaration)
                elif isinstance(declaration, StructNode):
                    structs.append(declaration)
                elif isinstance(declaration, TypeAliasNode):
                    typedefs.append(declaration)
            elif self.current_token[0] == "STRUCT":
                if self.is_function_definition():
                    function = self.parse_function()
                    if function is not None:
                        functions.append(function)
                else:
                    struct = self.parse_struct()
                    if struct is not None:
                        structs.append(struct)
                        global_variables.extend(
                            getattr(struct, "trailing_declarations", [])
                        )
            elif self.current_token[0] == "CLASS":
                class_node = self.parse_class()
                if class_node is not None:
                    structs.append(class_node)
                    global_variables.extend(
                        getattr(class_node, "trailing_declarations", [])
                    )
            elif self.current_token[0] == "ALIGNAS":
                alignas_specs = self.parse_alignas_specifiers()
                if self.current_token[0] in {"STRUCT", "CLASS"}:
                    struct = (
                        self.parse_struct(alignas_specs)
                        if self.current_token[0] == "STRUCT"
                        else self.parse_class(alignas_specs)
                    )
                    if struct is not None:
                        structs.append(struct)
                        global_variables.extend(
                            getattr(struct, "trailing_declarations", [])
                        )
                else:
                    self.add_global_declaration(
                        global_variables,
                        self.parse_global_variable(pre_alignas=alignas_specs),
                    )
            elif self.current_token[0] == "ENUM":
                if self.is_function_definition():
                    function = self.parse_function()
                    if function is not None:
                        functions.append(function)
                else:
                    enums.append(self.parse_enum())
            elif self.current_token[0] == "TYPEDEF":
                typedef = self.parse_typedef()
                if isinstance(typedef, StructNode):
                    structs.append(typedef)
                elif isinstance(typedef, EnumNode):
                    enums.append(typedef)
                elif typedef is not None:
                    self.annotate_declaration_scope(typedef)
                    typedefs.append(typedef)
            elif self.current_token[0] == "STATIC_ASSERT":
                global_variables.append(
                    self.annotate_declaration_scope(self.parse_static_assert())
                )
            elif self.is_conversion_operator_fragment():
                self.skip_struct_method()
            elif self.is_operator_function_definition():
                self.skip_operator_function_definition()
            elif self.is_decltype_template_instantiation_declaration():
                self.skip_decltype_template_instantiation_declaration()
            elif self.is_top_level_parameter_fragment_start():
                self.parse_top_level_parameter_fragment()
            elif self.is_qualified_aggregate_declaration_start():
                aggregate = self.parse_qualified_aggregate_declaration()
                if aggregate is not None:
                    structs.append(aggregate)
                    global_variables.extend(
                        getattr(aggregate, "trailing_declarations", [])
                    )
            elif self.current_token[0] == "CONSTANT":
                if self.is_constant_buffer():
                    constants.append(self.parse_constant_buffer())
                else:
                    self.add_global_declaration(
                        global_variables, self.parse_global_variable()
                    )
            elif self.current_token[0] == "ATTRIBUTE" or self.is_gnu_attribute_start():
                if self.is_function_definition():
                    function = self.parse_function()
                    if function is not None:
                        functions.append(function)
                else:
                    self.add_global_declaration(
                        global_variables, self.parse_global_variable()
                    )
            elif self.current_token[0] in STAGE_TOKENS or (
                self.current_token[0] in TYPE_TOKENS
                or self.current_token[0] in QUALIFIER_TOKENS
            ):
                if self.is_function_definition():
                    function = self.parse_function()
                    if function is not None:
                        functions.append(function)
                else:
                    self.add_global_declaration(
                        global_variables, self.parse_global_variable()
                    )
            elif self.current_token[0] == "RBRACE" and self.namespace_scope_stack:
                self.close_namespace_scope()
            else:
                self.eat(self.current_token[0])

        shader = ShaderNode(
            includes=preprocessors,
            functions=retain_declared_template_primaries(functions),
            structs=structs,
            global_variables=global_variables,
            constant=constants,
            enums=enums,
            typedefs=typedefs,
        )
        shader.max_template_specializations = self.max_template_specializations
        shader.template_specialization_limit_source = (
            self.template_specialization_limit_source
        )
        shader.using_namespace_directives = list(self.using_namespace_directives)
        shader.using_declarations = list(self.using_declarations)
        shader.namespace_aliases = dict(self.namespace_aliases)
        shader.declared_namespaces = sorted(self.declared_namespaces)
        shader.function_prototypes = dict(self.function_prototypes)
        self.associate_out_of_line_call_operators(shader)
        return shader

    def parse_function(self, preserve_declaration=False):
        start_token = self.current_token
        attributes = self.parse_attributes()

        qualifier = None
        if self.current_token[0] in STAGE_TOKENS:
            qualifier = self.current_token[1]
            self.eat(self.current_token[0])

        return_type, return_qualifiers = self.parse_type_specifier(
            attributes=attributes
        )

        if self.current_token[0] in STAGE_TOKENS and self.peek(1)[0] != "LPAREN":
            if qualifier is None:
                qualifier = self.current_token[1]
            self.eat(self.current_token[0])

        attributes.extend(self.parse_attributes())

        name = self.current_token[1]
        if not self.is_current_name_token():
            raise MetalSyntaxError(
                f"Expected function name, got {self.current_token[0]}"
            )
        self.eat(self.current_token[0])
        while self.current_token[0] == "SCOPE":
            self.eat("SCOPE")
            if not self.is_current_name_token():
                raise MetalSyntaxError(
                    f"Expected function name after '::', got {self.current_token[0]}"
                )
            name += f"::{self.current_token[1]}"
            self.eat(self.current_token[0])
        name = self.parse_function_operator_suffix(name)
        if self.template_argument_list_followed_by_call(follow_token_types={"LPAREN"}):
            template_args = self.parse_template_argument_suffix()
            name += f"<{self.format_generic_type_tokens(template_args)}>"

        self.push_declaration_context(f"function {name}")
        self.eat("LPAREN")
        params = self.parse_parameters()
        self.eat("RPAREN")

        post_attributes = self.parse_attributes()
        attributes.extend(post_attributes)
        method_qualifiers = self.parse_function_method_qualifiers()
        trailing_return_type = self.parse_optional_trailing_return_type(attributes)
        if trailing_return_type is not None:
            return_type = trailing_return_type
            attributes.extend(self.parse_attributes())
            method_qualifiers.extend(self.parse_function_method_qualifiers())
        attribute_qualifier, attributes = self.extract_stage_attributes(attributes)
        if qualifier is None:
            qualifier = attribute_qualifier

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            # Always record the prototype name so external-function detection can
            # identify which names were only declared and not defined here.
            self.function_prototypes.setdefault(name, len(self.function_prototypes))
            if not preserve_declaration:
                # For ordinary (non-template) free-function declarations we
                # preserve the node so the cross-file callable can be emitted as
                # a forward declaration in the generated output.  Stage-entry
                # functions (vertex/fragment/kernel/…) have no meaningful
                # declaration form outside the translation unit they define, so
                # they are still dropped to avoid emitting an unusable prototype.
                if qualifier in {"vertex", "fragment", "kernel", "mesh", "object",
                                  "amplification", "intersection", "anyhit",
                                  "closesthit", "miss", "callable"}:
                    self.pop_declaration_context()
                    return None
            body = None
        elif self.is_defaulted_or_deleted_function_declaration():
            self.skip_defaulted_or_deleted_function_declaration()
            self.pop_declaration_context()
            return None
        else:
            self.pending_block_scope_names.append(
                {param.name for param in params if getattr(param, "name", None)}
            )
            body = self.parse_block()

        function = FunctionNode(
            return_type=return_type,
            name=name,
            params=params,
            body=body,
            qualifiers=[qualifier] if qualifier else [],
            attributes=attributes,
            qualifier=qualifier,  # Also store as single qualifier for backward compatibility
            return_qualifiers=return_qualifiers,
            method_qualifiers=method_qualifiers,
        )
        self.annotate_declaration_scope(function)
        # Mark plain (non-template) declaration-only nodes so the code
        # generator knows to emit a forward declaration rather than a full body.
        if body is None and not getattr(function, "is_template_prototype", False):
            function.declaration_kind = "declaration"
            function.is_free_function_prototype = True
        function.declaration_qualifiers = list(return_qualifiers)
        function.method_qualifiers = list(method_qualifiers)
        function.declaration_source_location = self.source_span_from_tokens(
            start_token, self.tokens[self.pos - 1]
        )
        if name.endswith("::operator()"):
            owner = name[: -len("::operator()")]
            namespace = self.current_namespace()
            if namespace and not owner.startswith(f"{namespace}::"):
                owner = f"{namespace}::{owner}"
            function.out_of_line_call_operator_owner = owner
        self.pop_declaration_context()
        return function
