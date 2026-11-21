"""Functions in the CrossGL parser: return types, parameters, parameter and resource qualifiers, pointer resource contracts and resource parameters."""

from copy import deepcopy

from ..ast import (
    ArrayType,
    IdentifierNode,
    NamedType,
    ParameterNode,
    PointerType,
    ReferenceType,
    ResourceMemoryQualifierNode,
    TypeNode,
)
from ..codegen.array_utils import evaluate_literal_int_expression
from ..refusals import CrossGLSyntaxError
from .vocabulary import (
    PARAMETER_PRIMITIVE_QUALIFIER_NAMES,
    RESOURCE_ACCESS_MODE_NAMES,
    RESOURCE_ADDRESS_SPACE_NAMES,
    RESOURCE_MEMORY_QUALIFIER_NAMES,
)


def array_extent_value(extent):
    """Return the value an ``array<T, N>`` extent names.

    An extent is always a value. A bare name parses as a type argument (``NamedType``) because the generic argument
    list cannot tell ``N`` from a type, but in the extent position it names a constant or a generic value parameter,
    exactly as the ``N`` of ``T name[N]`` does.
    """
    if type(extent) is NamedType and not extent.generic_args:
        return IdentifierNode(extent.name, source_location=extent.source_location)
    return extent


class ParserFunctionMixin:
    """Function members of ``Parser``."""

    def parse_return_type_attributes(self):
        """Parse WGSL-style metadata between ``->`` and the return type."""
        return self.parse_attribute_annotations(allow_single_square=False)

    def parse_function_return_type(self):
        """Keep pointer return address spaces and pointee access in the type."""
        qualifiers, resource_qualifiers = self.partition_resource_qualifiers(
            self.parse_parameter_qualifiers()
        )
        return_type = self.parse_type()
        self.apply_pointer_resource_contract(
            return_type, qualifiers, resource_qualifiers
        )
        if isinstance(return_type, PointerType):
            return_type.is_mutable = not set(qualifiers).intersection(
                {"const", "constant", "readonly", "in"}
            )
        return return_type

    def parse_parameter_list(self):
        """Parse a comma-separated function parameter list."""
        parameters = []

        if self.current_token[0] == "VOID" and self.peek()[0] == "RPAREN":
            self.eat("VOID")
            return parameters

        while self.current_token[0] != "RPAREN":
            param = self.parse_parameter()
            parameters.append(param)

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
            elif self.current_token[0] != "RPAREN":
                break

        return parameters

    def parse_parameter(self):
        """Parse one function parameter declaration."""
        attributes = []
        attributes.extend(self.parse_attribute_annotations())

        if self.current_token[0] == "VAR":
            return self.parse_resource_parameter(attributes)

        parsed_qualifiers = self.parse_parameter_qualifiers()
        qualifiers, resource_qualifiers = self.partition_resource_qualifiers(
            parsed_qualifiers
        )

        is_mutable = False
        if self.current_token[0] == "MUT":
            is_mutable = True
            self.eat("MUT")

        if self.current_token_is_binding_identifier() and self.peek()[0] == "COLON":
            name = self.parse_binding_identifier()
            self.eat("COLON")
            param_type = self.parse_type()
        elif (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == "self"
            and self.peek()[0] in ["COMMA", "RPAREN"]
        ):
            name = self.parse_binding_identifier()
            param_type = NamedType("Self")
        else:
            param_type = self.parse_type()
            name = self.parse_binding_identifier()

        sizes = []
        while self.current_token[
            0
        ] == "LBRACKET" and not self.current_token_starts_square_attribute(
            allow_single_square=False
        ):
            self.eat("LBRACKET")
            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
            self.eat("RBRACKET")
            sizes.append(size)
        param_type = self.wrap_array_dimensions(param_type, sizes)

        if isinstance(param_type, ReferenceType):
            qualifier_names = {str(qualifier).lower() for qualifier in qualifiers}
            writable_address_space = qualifier_names.intersection(
                {"thread", "threadgroup", "device", "ray_data", "object_data"}
            )
            readonly_reference = qualifier_names.intersection(
                {"const", "constant", "readonly", "in"}
            )
            if (is_mutable or writable_address_space) and not readonly_reference:
                param_type.is_mutable = True

        attributes.extend(self.parse_post_declaration_attributes())

        default_value = None
        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            default_value = self.parse_expression()

        declaration_resource_qualifiers = self.apply_pointer_resource_contract(
            param_type, qualifiers, resource_qualifiers
        )

        return ParameterNode(
            name=name,
            param_type=param_type,
            default_value=default_value,
            attributes=attributes,
            qualifiers=qualifiers,
            resource_qualifiers=declaration_resource_qualifiers,
            is_mutable=is_mutable,
        )

    def parse_parameter_qualifiers(self):
        """Parse backend-neutral parameter qualifiers when present."""
        qualifiers = []

        while True:
            token_type, token_value = self.current_token
            normalized = str(token_value).lower() if token_value else ""

            if self.current_token_is_binding_identifier() and self.peek()[0] == "COLON":
                return qualifiers

            if normalized in RESOURCE_MEMORY_QUALIFIER_NAMES:
                qualifiers.append(self.parse_resource_memory_qualifier())
                continue

            if self.current_token_is_parameter_qualifier():
                qualifiers.append(normalized)
                self.eat(token_type)
                continue

            if (
                token_type == "IDENTIFIER"
                and str(token_value) in PARAMETER_PRIMITIVE_QUALIFIER_NAMES
                and self.current_token_starts_qualified_parameter_type()
            ):
                qualifiers.append(normalized)
                self.eat("IDENTIFIER")
                continue

            return qualifiers

    def parse_resource_memory_qualifier(self):
        """Parse volatility or scoped coherence as typed resource metadata."""
        token_type, token_value = self.current_token
        kind = str(token_value).lower()
        if kind not in RESOURCE_MEMORY_QUALIFIER_NAMES:
            raise CrossGLSyntaxError(
                f"Expected resource memory qualifier, got {token_type}"
            )
        self.eat(token_type)

        scope = None
        if kind == "coherent" and self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            scope_token_type, scope_value = self.current_token
            if not isinstance(scope_value, str) or not scope_value.isidentifier():
                raise CrossGLSyntaxError(
                    f"Expected coherence scope, got {scope_token_type}"
                )
            scope = scope_value.lower()
            self.eat(scope_token_type)
            self.eat("RPAREN")

        return ResourceMemoryQualifierNode(kind=kind, scope=scope)

    @staticmethod
    def partition_resource_qualifiers(qualifiers):
        resource_qualifiers = [
            qualifier
            for qualifier in qualifiers
            if isinstance(qualifier, ResourceMemoryQualifierNode)
        ]
        declaration_qualifiers = [
            (
                str(qualifier)
                if isinstance(qualifier, ResourceMemoryQualifierNode)
                else qualifier
            )
            for qualifier in qualifiers
        ]
        return declaration_qualifiers, resource_qualifiers

    def apply_pointer_resource_contract(
        self, type_node, declaration_qualifiers, resource_qualifiers
    ):
        """Attach declaration metadata to its pointer/reference type when possible."""
        resource_qualifiers = deepcopy(list(resource_qualifiers))
        owner = type_node
        while isinstance(owner, ArrayType):
            owner = owner.element_type

        if not isinstance(owner, (PointerType, ReferenceType)):
            return list(resource_qualifiers)

        if resource_qualifiers:
            existing = list(getattr(owner, "resource_qualifiers", []) or [])
            existing_text = {str(qualifier) for qualifier in existing}
            owner.resource_qualifiers = existing + [
                qualifier
                for qualifier in resource_qualifiers
                if str(qualifier) not in existing_text
            ]
        address_spaces = [
            str(qualifier).lower()
            for qualifier in declaration_qualifiers
            if str(qualifier).lower() in RESOURCE_ADDRESS_SPACE_NAMES
        ]
        if len(set(address_spaces)) == 1:
            owner.address_space = address_spaces[0]

        access_modes = [
            str(qualifier).lower()
            for qualifier in declaration_qualifiers
            if str(qualifier).lower() in RESOURCE_ACCESS_MODE_NAMES
        ]
        if len(set(access_modes)) == 1:
            owner.access_mode = access_modes[0]
        return []

    def current_token_starts_qualified_parameter_type(self):
        """Return whether a primitive qualifier is followed by a real parameter."""
        next_token = self.peek()
        token_after_type_start = self.peek(2)
        if next_token[0] not in self.type_start_tokens():
            return False
        return token_after_type_start[0] not in {
            "COMMA",
            "RPAREN",
            "EQUALS",
            "AT",
            "ATTRIBUTE",
        }

    def parse_resource_parameter(self, attributes):
        """Parse WGSL-style resource parameters emitted by reverse backends."""
        self.eat("VAR")
        resource_qualifiers = []

        if self.current_token[0] == "LESS_THAN":
            self.eat("LESS_THAN")
            while not self.current_token_is_type_greater_than():
                if self.current_token[0] != "COMMA":
                    resource_qualifiers.append(self.current_token[1])
                    self.eat(self.current_token[0])
                    continue
                self.eat("COMMA")
            self.eat_type_greater_than()

        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("COLON")
        param_type = self.parse_resource_parameter_type()

        attributes.extend(self.parse_attribute_annotations())

        parameter = ParameterNode(
            name=name,
            param_type=param_type,
            attributes=attributes,
        )
        parameter.resource_qualifiers = resource_qualifiers
        return parameter

    def parse_resource_parameter_type(self):
        if self.current_token[0] == "IDENTIFIER" and self.current_token[1] == "array":
            self.eat("IDENTIFIER")
            self.eat("LESS_THAN")
            element_type = self.parse_resource_parameter_type()
            size = None
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                if self.peek()[0] in {"GREATER_THAN", "BITWISE_SHIFT_RIGHT"}:
                    size = self.current_token[1]
                    self.eat(self.current_token[0])
                else:
                    # ``var<workgroup> s: array<f32, (A + B)>``: an extent of several tokens is a value expression.
                    size = array_extent_value(self.parse_generic_argument())
            self.eat_type_greater_than()
            return ArrayType(element_type, size)

        return self.parse_type()

    def parse_array_template_type(self):
        """Parse ``array<T, N>`` as the ``ArrayType`` that ``T name[N]`` declares, wherever a type is written.

        Parameters, locals, struct members, globals and return types all take this spelling, and a plain
        ``NamedType`` left every generator without array semantics for it. Two forms keep the ``array`` template
        type: an extent of zero, the zero-extent standard array that C-style ``T name[0]`` cannot declare
        (MSL 2.3.4 ``array<T, 0>``), and ``array<T>`` with no extent, whose meaning depends on where it is declared.
        """
        self.eat("IDENTIFIER")
        generic_args = self.parse_generic_arguments()
        if len(generic_args) == 2:
            element_type, extent = generic_args
            extent = array_extent_value(extent)
            generic_args[1] = extent
            if isinstance(element_type, IdentifierNode):
                element_type = NamedType(element_type.name, [])
            if isinstance(element_type, TypeNode) and (
                evaluate_literal_int_expression(extent) != 0
            ):
                return ArrayType(element_type, extent)
        return NamedType("array", generic_args)
