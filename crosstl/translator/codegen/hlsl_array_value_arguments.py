"""Array values passed to by-value HLSL array parameters.

HLSL has no array-typed expression other than an lvalue: a function cannot return an array, and an initializer list
``{...}`` is only valid in a declaration. CrossGL passes arrays by value (GLSL 4.60 section 6.1.1), so a by-value
``T v[N]`` parameter can receive an array constructor, a call result, or a local that the DirectX emitter holds in
a lowered form. Each is turned into an HLSL expression of the array type here:

* An array literal becomes a member of a helper-constructed struct. One struct and one constructor are emitted per
  array type, ``struct CrossGLArrayValue_float_5 { float v[5]; };`` and
  ``CrossGLArrayValue_float_5 crossgl_array_value_float_5(float e0, ..., float e4)``, and the argument is
  ``crossgl_array_value_float_5(a, b, c, d, e).v``. A member of a struct rvalue is an array rvalue, which binds to
  an array parameter by copy. Nested array literals are built the same way, innermost first.
* A ``T[N]`` call result is lowered to ``TN`` (scalar ``T``, ``N`` in 2..4) or to a ``CrossGLFixedArray`` struct with
  one field per element, and a local initialized from such a call takes that lowered type. Neither converts to
  ``T[N]`` implicitly, so the argument is flat-cast through a typedef, ``(CrossGLArray_float_5)value``. An explicit
  cast between aggregates whose flattened components match converts them element by element (HLSL flat
  conversion), and here the components are identical.

Elements of an array literal are evaluated as helper arguments. When more than one element has observable side
effects, the order is not proven and the argument is refused. Writable (``out``/``inout``) array parameters need an
lvalue of the array type, so a lowered value passed to one is refused as well.
"""

from __future__ import annotations

import re
from hashlib import sha256

from ..ast import ArrayLiteralNode, FunctionCallNode
from .array_utils import (
    evaluate_literal_int_expression,
    format_c_style_array_declaration,
)

__all__ = ["HlslArrayValueArguments"]


class HlslArrayValueArguments:
    """Render array literals and lowered array values as arguments of by-value HLSL array parameters."""

    def __init__(self, codegen):
        self.codegen = codegen
        # Mapped array type text -> (struct name, constructor name, element type, extent).
        self.value_structs = {}
        # Mapped array type text -> typedef name.
        self.typedefs = {}

    # Shapes -----------------------------------------------------------------------------------------------------

    def array_shape(self, expected_type):
        """Return ``(mapped_type, element_type, extent)`` for a fixed-size array type, or ``None``."""
        outer = self.codegen.hlsl_outer_array_type(expected_type)
        if outer is None:
            return None
        mapped_type, extent_text, element_type = outer
        if not extent_text:
            return None
        extent = evaluate_literal_int_expression(
            extent_text, self.codegen.hlsl_current_visible_int_constants()
        )
        if extent is None or extent <= 0:
            return None
        return mapped_type, element_type, extent

    def lowered_value_type(self, expected_type):
        """The vector or struct type a ``T[N]`` call result has in HLSL, without registering a new struct."""
        codegen = self.codegen
        components = codegen.hlsl_fixed_array_return_components(expected_type)
        if components is None:
            return None
        type_text = components[0]
        vector_type = codegen.hlsl_fixed_array_return_vector_type(type_text)
        if vector_type is not None:
            return vector_type
        return codegen.hlsl_fixed_array_return_structs.get(type_text)

    def unique_name(self, prefix, mapped_type):
        codegen = self.codegen
        base = prefix + re.sub(r"\W+", "_", mapped_type).strip("_")
        taken = set(getattr(codegen, "structs_by_name", {}) or {})
        taken.update(getattr(codegen, "function_return_types", {}) or {})
        if base in taken:
            base = f"{base}_{sha256(mapped_type.encode('utf-8')).hexdigest()[:8]}"
        return base

    # Arguments --------------------------------------------------------------------------------------------------

    def argument(self, arg, expected_type):
        """Render ``arg`` for a by-value parameter of array type ``expected_type``; ``None`` leaves it as it is."""
        shape = self.array_shape(expected_type)
        if shape is None:
            return None
        if isinstance(arg, ArrayLiteralNode):
            return self.literal(arg, shape)
        return self.lowered_value(arg, expected_type, shape)

    def lowered_value(self, arg, expected_type, shape):
        codegen = self.codegen
        lowered_type = self.lowered_value_type(expected_type)
        if lowered_type is None:
            return None
        value_type = codegen.expression_result_type(arg)
        if value_type is None or codegen.map_type(value_type) != codegen.map_type(
            lowered_type
        ):
            return None
        source_type = codegen.hlsl_source_expression_type(arg)
        if source_type is not None and codegen.hlsl_source_type_signature(
            source_type
        ) != codegen.hlsl_source_type_signature(expected_type):
            return None
        rendered = codegen.generate_expression_with_expected(arg, None)
        return f"({self.typedef(shape)})({rendered})"

    def writable_lowered_value(self, arg, expected_type):
        """Refuse a lowered array value bound to an ``out``/``inout`` array parameter; ``True`` when refused."""
        shape = self.array_shape(expected_type)
        if shape is None or isinstance(arg, ArrayLiteralNode):
            return False
        codegen = self.codegen
        lowered_type = self.lowered_value_type(expected_type)
        value_type = codegen.expression_result_type(arg)
        if (
            lowered_type is None
            or value_type is None
            or codegen.map_type(value_type) != codegen.map_type(lowered_type)
        ):
            return False
        codegen.refuse_lowering(
            f"array argument: a writable '{shape[0]}' parameter needs an array lvalue, but the "
            f"argument is held as '{codegen.map_type(lowered_type)}' in HLSL"
        )
        return True

    def literal(self, arg, shape):
        codegen = self.codegen
        mapped_type, element_type, extent = shape
        elements = list(getattr(arg, "elements", []) or [])
        if codegen.hlsl_is_zero_aggregate_initializer(arg):
            elements = []
        if len(elements) > extent:
            codegen.hlsl_aggregate_error(
                arg,
                expected_type=mapped_type,
                reason="element-count-mismatch",
                expected_element_count=extent,
            )
        effectful = [element for element in elements if self.has_effects(element)]
        if effectful and len(elements) > 1:
            codegen.hlsl_aggregate_error(
                arg,
                expected_type=mapped_type,
                reason="field-evaluation-order-unproven",
                detail=(
                    "array literal elements with side effects would be evaluated as helper "
                    "arguments, whose order is not proven"
                ),
            )
        rendered = [self.element(element, element_type, arg) for element in elements]
        rendered.extend(
            codegen.hlsl_value_initialized_expression(arg, element_type)
            for _index in range(len(elements), extent)
        )
        _struct_name, constructor = self.value_struct(shape)
        return f"{constructor}({', '.join(rendered)}).v"

    def has_effects(self, expr):
        """Whether evaluating ``expr`` writes anything; struct and array constructors write nothing themselves."""
        codegen = self.codegen
        if isinstance(expr, ArrayLiteralNode):
            return any(self.has_effects(element) for element in expr.elements or [])
        if isinstance(expr, FunctionCallNode):
            function = getattr(expr, "function", getattr(expr, "name", None))
            name = getattr(function, "name", function)
            if (
                isinstance(name, str)
                and name not in codegen.current_hlsl_available_functions
                and not codegen.hlsl_function_name_is_shadowed(name)
                and codegen.hlsl_struct_constructor_fields(name) is not None
            ):
                arguments = getattr(expr, "arguments", getattr(expr, "args", [])) or []
                return any(self.has_effects(argument) for argument in arguments)
        return codegen.hlsl_expression_has_observable_side_effects(
            expr, allow_numeric_constructors=True, allow_buffer_loads=True
        )

    def element(self, element, element_type, aggregate):
        codegen = self.codegen
        if self.array_shape(element_type) is not None:
            if isinstance(element, ArrayLiteralNode):
                return self.literal(element, self.array_shape(element_type))
            lowered = self.lowered_value(
                element, element_type, self.array_shape(element_type)
            )
            if lowered is not None:
                return lowered
        elif isinstance(element, ArrayLiteralNode):
            codegen.hlsl_aggregate_error(
                aggregate,
                expected_type=codegen.map_type(element_type),
                reason="array-argument-unsupported",
                detail=(
                    "a braced element initializer has no HLSL expression form outside a "
                    "declaration initializer"
                ),
            )
        return codegen.hlsl_contextual_initializer_element_expression(
            element, element_type, aggregate
        )

    # Declarations -----------------------------------------------------------------------------------------------

    def value_struct(self, shape):
        mapped_type, element_type, extent = shape
        entry = self.value_structs.get(mapped_type)
        if entry is None:
            struct_name = self.unique_name("CrossGLArrayValue_", mapped_type)
            constructor = (
                "crossgl_array_value_" + struct_name[len("CrossGLArrayValue_") :]
            )
            entry = (struct_name, constructor, element_type, extent)
            self.value_structs[mapped_type] = entry
        return entry[0], entry[1]

    def typedef(self, shape):
        mapped_type = shape[0]
        name = self.typedefs.get(mapped_type)
        if name is None:
            name = self.unique_name("CrossGLArray_", mapped_type)
            self.typedefs[mapped_type] = name
        return name

    def declarations(self):
        """Typedefs, value structs and their constructors, after the program's struct declarations."""
        codegen = self.codegen
        code = ""
        for mapped_type, name in sorted(
            self.typedefs.items(), key=lambda item: item[1]
        ):
            code += f"typedef {format_c_style_array_declaration(mapped_type, name)};\n"
        if self.typedefs:
            code += "\n"
        # A nested value struct's constructor takes the inner struct's array, so emit inner types first.
        entries = sorted(
            self.value_structs.items(),
            key=lambda item: (item[0].count("["), item[1][0]),
        )
        for _mapped_type, (struct_name, constructor, element_type, extent) in entries:
            mapped_element = codegen.map_type(element_type)
            member = format_c_style_array_declaration(mapped_element, f"v[{extent}]")
            parameters = ", ".join(
                format_c_style_array_declaration(mapped_element, f"e{index}")
                for index in range(extent)
            )
            code += f"struct {struct_name}\n{{\n    {member};\n}};\n\n"
            code += f"{struct_name} {constructor}({parameters})\n{{\n"
            code += f"    {struct_name} value;\n"
            for index in range(extent):
                code += f"    value.v[{index}] = e{index};\n"
            code += "    return value;\n}\n\n"
        return code
