"""Array arguments that Metal cannot pass to a C array parameter as they are.

CrossGL passes an array parameter without ``out`` or ``inout`` by value, as GLSL, HLSL and SPIR-V do (GLSL 4.60
section 6.1.1, "Function Calling Conventions"). The Metal emitter declares such a parameter with C syntax,
``float4 a[2]``, which C++ adjusts to a pointer into the ``thread`` address space ([dcl.fct]/5; MSL specification
section 4.0, "Address Spaces", gives a parameter without an address space the ``thread`` one). Several arguments
cannot bind to that pointer:

* an array in another address space, such as a constant-buffer member (``constant``), a program-scope constant
  (``constant``), a threadgroup array (``threadgroup``) or a buffer array (``device``), because Metal has no
  conversion between pointers of different address spaces ("no matching function for call");
* a constant-buffer array whose elements the layout planner stores in another shape, such as ``float4`` storage
  for a std140 ``float[3]`` or register words for a row-major matrix;
* an array literal, which is no expression with an address.

:class:`MetalArrayArguments` passes such an argument as a ``thread`` copy:
``crosstl_array_value<float4, 2>{{u.c[0], u.c[1]}}.v``. The temporary structure lives until the end of the full
expression that contains the call ([class.temporary]/4), and its array member converts to the pointer the
parameter expects. Each element goes back through the emitter, so layout conversions apply to it as to any other
element read. That evaluates the argument once per element, so a non-literal argument must be free of effects;
another one is refused. The copy is exactly the value semantics of the source call; arguments already in
``thread`` storage keep pointer passing, which reads cannot tell apart from a copy, and callees that write an
``in`` array already work on a copy of their own (:mod:`..semantics.array_parameters`).
"""

from __future__ import annotations

import re

from ..ast import ArrayAccessNode, ArrayLiteralNode, LiteralNode, PrimitiveType
from ..semantics.rewrite import is_pure
from .array_utils import fold_array_type_counts, split_array_type_suffix

__all__ = ["METAL_ARRAY_VALUE_TYPE", "MetalArrayArguments"]

METAL_ARRAY_VALUE_TYPE = "crosstl_array_value"
_DEFINITION = (
    f"template <typename T, int N> struct {METAL_ARRAY_VALUE_TYPE} {{\n"
    "  T v[N];\n"
    "};\n\n"
)
_ARRAY_TYPE = re.compile(r"^(?P<element>.*?)\s*\[\s*(?P<extent>\d+)\s*\](?P<rest>.*)$")
_WRITING_QUALIFIERS = frozenset({"out", "inout"})
# Address spaces of storage that a ``thread`` pointer cannot point into.
_OTHER_ADDRESS_SPACES = frozenset(
    {
        "constant",
        "device",
        "threadgroup",
        "threadgroup_imageblock",
        "object_data",
        "ray_data",
    }
)


def _array_parts(type_name):
    """``(element type, extent)`` of a sized C-array spelling, ``("float[3]", 2)`` for ``float[2][3]``."""
    match = _ARRAY_TYPE.match(str(type_name or "").strip())
    if match is None:
        return None
    return match.group("element").strip() + match.group("rest").strip(), int(
        match.group("extent")
    )


def _element(expr, index):
    return ArrayAccessNode(expr, LiteralNode(index, PrimitiveType("int")))


class MetalArrayArguments:
    """Thread copies of array arguments for one translation, and the template they need."""

    def __init__(self, codegen):
        self.codegen = codegen
        self.required = False

    def code(self):
        """The definition of :data:`METAL_ARRAY_VALUE_TYPE` when a copy uses it, else ``""``."""
        return _DEFINITION if self.required else ""

    def argument(self, func_name, index, arg, call_args):
        """Rendered thread copy of argument ``index`` of a call to ``func_name``, or ``None`` to pass it as is."""
        codegen = self.codegen
        parameter = self._parameter(func_name, index, call_args)
        if parameter is None:
            return None
        parameter_type = self._value_array_type(parameter)
        if parameter_type is None:
            return None
        if (
            not isinstance(arg, ArrayLiteralNode)
            and self._storage_address_space(arg) is None
        ):
            return None
        mapped = codegen.map_type(parameter_type)
        base, suffix = split_array_type_suffix(mapped)
        parts = _array_parts(f"{base}{suffix}")
        if parts is None:
            return None
        element_spelling, extent = parts
        elements = self._elements(func_name, arg, parameter_type)
        if elements is None:
            return None
        self.required = True
        return f"{METAL_ARRAY_VALUE_TYPE}<{element_spelling}, {extent}>{{{elements}}}.v"

    def _parameter(self, func_name, index, call_args):
        codegen = self.codegen
        if func_name not in codegen.user_function_names:
            return None
        selected = codegen.metal_user_function_overload_for_call(func_name, call_args)
        if selected is None:
            return None
        parameters = list(
            getattr(selected, "parameters", getattr(selected, "params", [])) or []
        )
        return parameters[index] if index < len(parameters) else None

    def _value_array_type(self, parameter):
        """Folded CrossGL type of a by-value ``thread`` array parameter with literal extents, or ``None``."""
        codegen = self.codegen
        raw_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        if raw_type is None or not codegen.is_array_type_node(raw_type):
            return None
        if codegen.parameter_qualifier_names(parameter) & _WRITING_QUALIFIERS:
            return None
        if codegen.parameter_address_space(parameter) not in (None, "thread"):
            return None
        if (
            codegen.resource_array_parameter(raw_type, parameter) is not None
            or codegen.is_structured_buffer_type(raw_type)
            or codegen.array_indirect_element_type_node(raw_type) is not None
        ):
            return None
        folded = fold_array_type_counts(
            codegen.type_name_string(raw_type),
            getattr(codegen, "literal_int_constants", None) or {},
        )
        type_name = folded
        while "[" in type_name:
            parts = _array_parts(type_name)
            if parts is None:
                return None
            type_name = parts[0]
        return folded

    def _storage_address_space(self, arg):
        """Address space of an array argument's storage when it is not ``thread``, else ``None``."""
        codegen = self.codegen
        address_space = codegen.argument_address_space(arg)
        root = codegen.assignment_target_root_name(arg)
        if address_space == "constant" and not self._is_parameter(root):
            # A function-scope ``constant`` array is declared ``const`` in thread storage.
            if isinstance(root, str) and root in codegen.local_variable_types:
                return None
        if address_space in _OTHER_ADDRESS_SPACES:
            return address_space
        if address_space is not None:
            return None
        if not isinstance(root, str) or root in codegen.local_variable_types:
            return None
        if root in (getattr(codegen, "cbuffer_member_references", None) or {}):
            return "constant"
        if root in (
            getattr(codegen, "metal_program_scope_constant_types", None) or {}
        ) and root not in (
            getattr(codegen, "metal_runtime_initialized_constants", None) or set()
        ):
            return "constant"
        return None

    def _is_parameter(self, name):
        codegen = self.codegen
        parameters = (getattr(codegen, "function_parameter_nodes", None) or {}).get(
            codegen.current_function_name, []
        )
        return any(getattr(parameter, "name", None) == name for parameter in parameters)

    def _elements(self, func_name, expr, type_name):
        """Brace list of the elements of array value ``expr`` of type ``type_name``, or ``None``."""
        codegen = self.codegen
        parts = _array_parts(type_name)
        if parts is None:
            return None
        element_type, extent = parts
        if isinstance(expr, ArrayLiteralNode):
            values = list(expr.elements or [])
            if len(values) != extent:
                return codegen.refuse_lowering(
                    f"array literal argument to '{func_name}': the parameter holds {extent} elements and the "
                    f"literal {len(values)}"
                )
        elif is_pure(expr):
            values = [_element(expr, index) for index in range(extent)]
        else:
            return codegen.refuse_lowering(
                f"array argument to '{func_name}': Metal passes this array as a thread copy built element by "
                "element, which evaluates the argument once per element, and this argument has effects"
            )
        rendered = []
        for value in values:
            if "[" in element_type:
                nested = self._elements(func_name, value, element_type)
                if nested is None:
                    return None
                rendered.append(nested)
            else:
                rendered.append(
                    codegen.generate_expression_with_expected(value, element_type)
                )
        return "{" + ", ".join(rendered) + "}"
