"""Buffer attributes, stage entry buffer bindings and structured buffer access."""

from ..atomic_result_conversions import (
    convert_atomic_result,
    is_atomic_result_call,
)
from ..implicit_constructor_conversions import converting_constructor_value
from ..MetalAst import (
    ArrayAccessNode,
    AttributeNode,
    VariableNode,
)
from .support import MetalStageEntryArrayResourceError


class MetalStructuredBufferMixin:
    """Members of ``MetalToCrossGLConverter``: buffer attributes, stage entry buffer bindings and structured buffer access."""

    def is_samplerless_sample_expression(self, expr):
        metal_type = self.resolve_type_alias(self.expression_metal_type(expr))
        return self.normalized_metal_type(metal_type) in {"SwiftUI::Layer"}

    def has_attribute(self, node, name):
        name = str(name).lower()
        return any(
            str(getattr(attr, "name", "")).lower() == name
            for attr in getattr(node, "attributes", []) or []
        )

    def buffer_attribute_binding(self, node):
        for attr in getattr(node, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() != "buffer":
                continue
            args = getattr(attr, "args", getattr(attr, "arguments", [])) or []
            if not args:
                return None
            text = self.format_metadata_argument(args[0]).strip()
            return int(text) if text.isdigit() else None
        return None

    def reference_element_type(self, metal_type):
        if not metal_type:
            return None
        base = str(metal_type).strip()
        if base.startswith("metal::"):
            base = base.split("metal::", 1)[1]

        reference_depth = 0
        while base.endswith("*") or base.endswith("&"):
            if base.endswith("&"):
                reference_depth += 1
            base = base[:-1].strip()
        return base if reference_depth else None

    def pointer_element_type(self, metal_type):
        if not metal_type:
            return None
        base = str(metal_type).strip()
        if base.startswith("metal::"):
            base = base.split("metal::", 1)[1]

        pointer_depth = 0
        while base.endswith("*") or base.endswith("&"):
            if base.endswith("*"):
                pointer_depth += 1
            base = base[:-1].strip()
        return base if pointer_depth else None

    def is_stage_entry_buffer_resource_parameter(self, var):
        if not getattr(var, "name", None):
            return False
        qualifiers = set(self.effective_declaration_qualifiers(var))
        if not qualifiers.intersection({"device", "constant"}):
            return False
        if self.stage_entry_array_resource_element_type(var) is not None:
            return True
        raw_type = self.resolve_type_alias(getattr(var, "vtype", None))
        if self.pointer_element_type(raw_type):
            return True
        return bool(self.reference_element_type(raw_type))

    def stage_entry_array_resource_element_type(self, var):
        array_dimensions = list(getattr(var, "array_sizes", None) or [])
        if not array_dimensions:
            return None
        qualifiers = set(self.effective_declaration_qualifiers(var))
        if not qualifiers.intersection({"device", "constant"}):
            return None
        element_type = str(getattr(var, "vtype", "") or "").strip()
        if not element_type or element_type.endswith(("*", "&")):
            return None
        if len(array_dimensions) != 1:
            raise MetalStageEntryArrayResourceError(
                getattr(var, "name", None),
                array_dimensions,
                "multidimensional-parameter-array",
                getattr(var, "source_location", None),
            )
        return element_type

    def apply_implicit_stage_entry_buffer_bindings(self, func):
        params = list(getattr(func, "params", []) or [])
        used_bindings = {
            binding
            for param in params
            for binding in [self.buffer_attribute_binding(param)]
            if binding is not None
        }
        snapshots = []
        next_binding = 0
        for param in params:
            if not self.is_stage_entry_buffer_resource_parameter(param):
                continue
            if self.buffer_attribute_binding(param) is not None:
                continue
            previous_attributes = list(getattr(param, "attributes", []) or [])
            if previous_attributes:
                continue
            while next_binding in used_bindings:
                next_binding += 1
            param.attributes = previous_attributes + [
                AttributeNode("buffer", [str(next_binding)])
            ]
            snapshots.append((param, previous_attributes))
            used_bindings.add(next_binding)
            next_binding += 1
        return snapshots

    def structured_buffer_element_source_type(self, var):
        """The Metal element type of a buffer resource declaration that becomes a structured buffer."""
        if not (
            self.has_attribute(var, "buffer")
            or id(var) in self.current_stage_entry_resource_parameter_ids
        ):
            return None

        qualifiers = set(self.effective_declaration_qualifiers(var))
        if not qualifiers.intersection({"device", "constant"}):
            return None

        element_type = self.pointer_element_type(
            self.resolve_type_alias(getattr(var, "vtype", None))
        )
        array_element_type = self.stage_entry_array_resource_element_type(var)
        if element_type is None:
            element_type = array_element_type
        if not element_type:
            return None
        return self.resolve_type_alias(element_type)

    def structured_buffer_pointer_type(self, var):
        element_type = self.structured_buffer_element_source_type(var)
        if not element_type:
            return None
        qualifiers = set(self.effective_declaration_qualifiers(var))
        # A ``device const`` buffer that the function stores through after a
        # const cast is still bound writable: keep the store.
        read_only = qualifiers.intersection({"constant", "const"}) and not getattr(
            var, "metal_storage_written", False
        )
        buffer_type = "StructuredBuffer" if read_only else "RWStructuredBuffer"
        mapped_element_type = self.map_resource_pointer_element_type(var, element_type)
        return f"{buffer_type}<{mapped_element_type}>"

    def is_structured_buffer_expression(self, expr):
        name = self.expression_base_name(expr)
        return bool(
            name
            and (
                name in self.current_structured_buffer_names
                or name in self.global_structured_buffer_names
            )
        )

    def is_structured_buffer_element_access(self, expr):
        return self.is_direct_structured_buffer_element_access(expr)

    def is_direct_structured_buffer_element_access(self, expr):
        if not isinstance(expr, ArrayAccessNode):
            return False
        if not isinstance(expr.array, VariableNode):
            return False
        if not self.is_structured_buffer_expression(expr.array):
            return False
        base_type = self.expression_metal_type(expr.array)
        return bool(
            self.metal_pointer_pointee_type_once(base_type) is not None
            or self.split_outer_metal_declarator_array_type(base_type) is not None
        )

    def generate_without_structured_buffer_index_lowering(self, expr, is_main=False):
        previous = self.suppress_structured_buffer_index_lowering
        self.suppress_structured_buffer_index_lowering = True
        try:
            return self.generate_expression(expr, is_main)
        finally:
            self.suppress_structured_buffer_index_lowering = previous

    def binds_reference_to_structured_buffer_element(self, node):
        declaration = getattr(node, "left", None)
        return (
            getattr(node, "operator", None) == "="
            and isinstance(declaration, VariableNode)
            and bool(getattr(declaration, "vtype", None))
            and self.reference_parameter(declaration)
            and self.is_structured_buffer_element_access(node.right)
        )

    def generate_structured_buffer_element_lvalue(self, access, is_main=False):
        buffer = self.generate_without_structured_buffer_index_lowering(
            access.array, is_main
        )
        index = self.generate_expression(access.index, is_main)
        return f"{buffer}[{index}]"

    def generate_structured_buffer_load(self, access, is_main=False):
        buffer = self.generate_without_structured_buffer_index_lowering(
            access.array, is_main
        )
        index = self.generate_expression(access.index, is_main)
        return f"buffer_load({buffer}, {index})"

    def generate_structured_buffer_store(self, access, value, operator, is_main=False):
        buffer = self.generate_without_structured_buffer_index_lowering(
            access.array, is_main
        )
        index = self.generate_expression(access.index, is_main)
        converted = (
            converting_constructor_value(
                self,
                self.expression_metal_type(access),
                value,
                is_main,
                assignment=True,
            )
            if operator == "="
            else None
        )
        if converted is not None:
            return f"buffer_store({buffer}, {index}, {converted})"
        rendered_value = self.generate_expression(value, is_main)
        if operator == "=" and is_atomic_result_call(self, value):
            rendered_value = convert_atomic_result(
                self, value, rendered_value, self.expression_metal_type(access)
            )
        if self.binary16_widening_profile is not None:
            rendered_value = self.render_metal_half_widening(
                rendered_value,
                self.expression_metal_type(value),
                self.expression_metal_type(access),
                getattr(value, "source_location", None),
            )
        if operator != "=":
            compound_ops = {
                "+=": "+",
                "-=": "-",
                "*=": "*",
                "/=": "/",
                "%=": "%",
                "&=": "&",
                "|=": "|",
                "^=": "^",
                "<<=": "<<",
                ">>=": ">>",
            }
            binary_op = compound_ops.get(operator)
            if binary_op is None:
                return None
            bfloat_plan = self.metal_bfloat_arithmetic_plan(
                binary_op,
                self.expression_metal_type(access),
                self.expression_metal_type(value),
                getattr(access, "source_location", None),
            )
            if bfloat_plan is not None and bfloat_plan[1][1] is not None:
                rendered_value = f"{self.map_type(bfloat_plan[1][1])}({rendered_value})"
            conversion = self.metal_vector_scalar_conversion(
                binary_op,
                self.expression_metal_type(access),
                self.expression_metal_type(value),
            )
            if conversion is not None and conversion[0] == 1:
                rendered_value = self.metal_vector_scalar_operand(
                    rendered_value, self.expression_metal_type(value), conversion[1]
                )
            # Keep the lvalue intact so target lowering evaluates its index once.
            return f"{buffer}[{index}] {operator} {rendered_value}"
        return f"buffer_store({buffer}, {index}, {rendered_value})"
