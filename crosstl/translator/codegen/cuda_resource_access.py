"""Resource access of the CUDA generator: GLSL buffer blocks, buffer access diagnostics, qualifiers, the variable registry and resource queries."""

import re

from ..ast import (
    ArrayAccessNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    PointerAccessNode,
    VariableNode,
)


class CudaResourceAccessMixin:
    """Resource access members of ``CudaCodeGen``."""

    def is_glsl_buffer_block_node(self, node):
        attributes = {
            str(getattr(attr, "name", "")).lower()
            for attr in getattr(node, "attributes", []) or []
        }
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        if "glsl_buffer_block" in attributes:
            return True
        return "buffer" in qualifiers and bool(
            attributes & {"std140", "std430", "scalar"}
        )

    def glsl_buffer_block_layout(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name == "glsl_buffer_block":
                arguments = getattr(attr, "arguments", []) or []
                if arguments:
                    return self.attribute_value_to_string(arguments[0])
            if attr_name in {"std140", "std430", "scalar"}:
                return attr_name
        return None

    def glsl_buffer_block_declaration_type(self, type_name, node):
        type_name = self.type_name_string(type_name)
        if not self.is_glsl_buffer_block_node(node):
            return type_name
        if not type_name or self.explicit_resource_access(node) != "readonly":
            return type_name
        if type_name.startswith("const "):
            return type_name
        # Only a runtime-sized array becomes a pointer whose pointee can be
        # const. A block object, sized array or scalar would be an
        # uninitialized `__device__ const` object, which CUDA rejects, so it
        # stays writable and the readonly diagnostics guard its writes.
        if "[" not in type_name or "]" not in type_name:
            return type_name
        open_bracket = type_name.find("[")
        base_type = type_name[:open_bracket].strip()
        array_suffix = type_name[open_bracket:]
        if array_suffix != "[]":
            return type_name
        return f"const {base_type}{array_suffix}"

    def glsl_buffer_block_metadata_comment(self, node, type_name):
        name = getattr(node, "name", None)
        if not name or not self.is_glsl_buffer_block_node(node):
            return ""
        parts = [
            "// CrossGL resource metadata:",
            f"name={name}",
            "kind=glsl_buffer_block",
        ]
        layout = self.glsl_buffer_block_layout(node)
        if layout:
            parts.append(f"layout={layout}")
        access = self.explicit_resource_access(node)
        if access:
            parts.append(f"access={access}")
        return " ".join(parts)

    def glsl_buffer_block_root_name(self, expr):
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
            return self.glsl_buffer_block_root_name(array_expr)
        if isinstance(expr, (MemberAccessNode, PointerAccessNode)):
            object_expr = getattr(
                expr,
                "object_expr",
                getattr(expr, "object", getattr(expr, "pointer_expr", None)),
            )
            return self.glsl_buffer_block_root_name(object_expr)
        if isinstance(expr, IdentifierNode):
            return expr.name
        if isinstance(expr, VariableNode):
            return expr.name
        if isinstance(expr, str):
            return expr
        return getattr(expr, "name", None)

    def glsl_buffer_block_access(self, expr):
        root_name = self.glsl_buffer_block_root_name(expr)
        if root_name is None:
            return None, None
        return self.glsl_buffer_block_accesses.get(root_name), root_name

    def glsl_buffer_block_diagnostic(self, operation, expr, reason, fallback):
        _, resource_name = self.glsl_buffer_block_access(expr)
        resource_name = resource_name or "unknown"
        return self.refuse_lowering(
            f"{self.resource_backend_name()} GLSL buffer block {operation}: resource '{resource_name}' {reason}",
            fallback,
        )

    def glsl_buffer_block_read_diagnostic(self, expr, operation):
        access, _ = self.glsl_buffer_block_access(expr)
        if access != "writeonly":
            return None
        fallback = self.diagnostic_zero_value_for_type(
            self.expression_result_type(expr)
        )
        return self.glsl_buffer_block_diagnostic(
            operation, expr, "is writeonly", fallback
        )

    def glsl_buffer_block_write_diagnostic(self, expr, operation):
        access, _ = self.glsl_buffer_block_access(expr)
        if access != "readonly":
            return None
        return self.glsl_buffer_block_diagnostic(
            operation, expr, "is readonly", "((void)0)"
        )

    def glsl_buffer_block_read_write_diagnostic(self, expr, operation, fallback):
        access, _ = self.glsl_buffer_block_access(expr)
        if access == "readonly":
            return self.glsl_buffer_block_diagnostic(
                operation, expr, "is readonly", fallback
            )
        if access == "writeonly":
            return self.glsl_buffer_block_diagnostic(
                operation, expr, "is writeonly", fallback
            )
        return None

    def buffer_resource_access(self, buffer_arg):
        """Return explicit readonly/writeonly access tracked for buffer resources."""
        if isinstance(buffer_arg, FunctionCallNode):
            callee_name = self.raw_function_call_name(buffer_arg)
            return_source = self.query_return_sources.get(callee_name)
            if return_source is None:
                return None
            raw_args = getattr(
                buffer_arg,
                "arguments",
                getattr(buffer_arg, "args", []),
            )
            return self.query_return_source_buffer_access(return_source, raw_args)

        if isinstance(buffer_arg, ArrayAccessNode):
            array_node = getattr(
                buffer_arg, "array", getattr(buffer_arg, "array_expr", None)
            )
            return self.buffer_resource_access(array_node)

        if isinstance(buffer_arg, MemberAccessNode):
            object_node = getattr(
                buffer_arg,
                "object_expr",
                getattr(buffer_arg, "object", None),
            )
            object_type = self.struct_member_lookup_type(
                self.expression_result_type(object_node)
            )
            member = getattr(buffer_arg, "member", None)
            access = self.struct_member_image_accesses.get(object_type, {}).get(member)
            if access is not None:
                return access
            return self.buffer_resource_access(object_node)

        if isinstance(buffer_arg, PointerAccessNode):
            pointer_expr = getattr(buffer_arg, "pointer_expr", None)
            pointer_type = self.expression_result_type(pointer_expr)
            pointer_info = self.cuda_indirect_type_info(pointer_type)
            if pointer_info is not None:
                object_type = self.struct_member_lookup_type(
                    pointer_info["pointee_type"]
                )
                member = getattr(buffer_arg, "member", None)
                access = self.struct_member_image_accesses.get(object_type, {}).get(
                    member
                )
                if access is not None:
                    return access
            return self.buffer_resource_access(pointer_expr)

        buffer_name = self.get_expression_name(buffer_arg)
        if not buffer_name:
            return None
        return self.buffer_resource_accesses.get(buffer_name)

    def query_return_source_buffer_access(self, return_source, raw_args):
        """Return buffer access for a traceable returned resource."""
        kind = return_source.get("kind")
        if kind == "ternary":
            true_access = self.query_return_source_buffer_access(
                return_source["true_source"],
                raw_args,
            )
            false_access = self.query_return_source_buffer_access(
                return_source["false_source"],
                raw_args,
            )
            if true_access is None or true_access != false_access:
                return None
            return true_access

        if kind == "global":
            return self.buffer_resource_access(return_source.get("name"))

        if kind == "parameter":
            index = return_source.get("index")
            if index is None or index >= len(raw_args):
                return None
            return self.buffer_resource_access(raw_args[index])

        if kind == "member":
            object_type = return_source.get("object_type")
            member = return_source.get("member")
            return self.struct_member_image_accesses.get(object_type, {}).get(member)

        return None

    def unsupported_buffer_access_call(
        self, resource_label, operation, buffer_type, reason, fallback
    ):
        buffer_type = self.type_name_string(buffer_type) or "unknown buffer"
        return self.refuse_lowering(
            f"{self.resource_backend_name()} {resource_label} access: {operation} {reason} on {buffer_type}",
            fallback,
        )

    def structured_buffer_access_diagnostic(
        self, operation, buffer_type, buffer_expr, requirement, fallback
    ):
        access = self.buffer_resource_access(buffer_expr)
        if requirement == "read" and access == "writeonly":
            return self.unsupported_buffer_access_call(
                "structured buffer",
                operation,
                buffer_type,
                "requires readable buffer resource",
                fallback,
            )
        if requirement == "write" and access == "readonly":
            return self.unsupported_buffer_access_call(
                "structured buffer",
                operation,
                buffer_type,
                "requires writable buffer resource",
                fallback,
            )
        if requirement == "readwrite" and access in {"readonly", "writeonly"}:
            return self.unsupported_buffer_access_call(
                "structured buffer",
                operation,
                buffer_type,
                "requires readwrite buffer resource",
                fallback,
            )
        return None

    def byte_address_buffer_access_diagnostic(
        self, operation, buffer_type, buffer_expr, requirement, fallback
    ):
        access = self.buffer_resource_access(buffer_expr)
        if requirement == "read" and access == "writeonly":
            return self.unsupported_buffer_access_call(
                "byte-address buffer",
                operation,
                buffer_type,
                "requires readable buffer resource",
                fallback,
            )
        if requirement == "write" and access == "readonly":
            return self.unsupported_buffer_access_call(
                "byte-address buffer",
                operation,
                buffer_type,
                "requires writable buffer resource",
                fallback,
            )
        if requirement == "readwrite" and access in {"readonly", "writeonly"}:
            return self.unsupported_buffer_access_call(
                "byte-address buffer",
                operation,
                buffer_type,
                "requires readwrite buffer resource",
                fallback,
            )
        return None

    def structured_buffer_element_resource_type(self, expr):
        """Return the owning structured-buffer type for an element expression."""
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
            array_type = self.expression_result_type(array_expr)
            array_type_string = self.type_name_string(array_type) or ""
            if (
                self.structured_buffer_type_parts(array_type) is not None
                and "[" not in array_type_string
            ):
                return array_type
            return None

        if isinstance(expr, (MemberAccessNode, PointerAccessNode)):
            object_expr = getattr(
                expr,
                "object_expr",
                getattr(expr, "object", getattr(expr, "pointer_expr", None)),
            )
            return self.structured_buffer_element_resource_type(object_expr)

        return None

    def structured_buffer_element_read_diagnostic(self, expr, operation):
        buffer_type = self.structured_buffer_element_resource_type(expr)
        if buffer_type is None:
            return None
        fallback = self.diagnostic_zero_value_for_type(
            self.expression_result_type(expr)
        )
        return self.structured_buffer_access_diagnostic(
            operation,
            buffer_type,
            expr,
            "read",
            fallback,
        )

    def structured_buffer_element_write_diagnostic(self, expr, operation, operator):
        buffer_type = self.structured_buffer_element_resource_type(expr)
        if buffer_type is None:
            return None
        requirement = "readwrite" if operator != "=" else "write"
        return self.structured_buffer_access_diagnostic(
            operation,
            buffer_type,
            expr,
            requirement,
            "((void)0)",
        )

    def declaration_qualifier_names(self, node):
        """Return normalized qualifier and attribute names on declarations."""
        names = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name:
                names.add(attr_name)
        return names

    def apply_restrict_qualifier_to_declaration(self, declaration, name, node):
        """Write a CrossGL ``restrict`` pointer declaration with ``__restrict__``, as HIP output does."""
        if (
            "*" not in declaration
            or not name
            or "__restrict__" in declaration
            or "restrict" not in self.declaration_qualifier_names(node)
        ):
            return declaration
        position = declaration.rfind(f" {name}")
        if position < 0:
            return declaration
        return f"{declaration[:position]} __restrict__{declaration[position:]}"

    def apply_volatile_qualifier_to_parameter(self, declarator, node):
        """Write a CrossGL ``volatile`` buffer parameter as a pointer to ``volatile`` memory, as HIP output does."""
        if (
            "*" not in declarator
            or "volatile" not in self.declaration_qualifier_names(node)
            or re.search(r"\bvolatile\b", declarator)
        ):
            return declarator
        if declarator.startswith("const "):
            return f"const volatile {declarator[len('const '):]}"
        return f"volatile {declarator}"

    def apply_readonly_qualifier_to_type(self, type_name, node):
        """Apply readonly declaration qualifiers to pointer/reference types."""
        type_string = self.type_name_string(type_name)
        if not type_string:
            return type_name

        if self.cuda_pointer_or_reference_type_info(type_string) is None:
            return type_string

        if not (
            self.declaration_qualifier_names(node) & {"const", "readonly", "constant"}
        ):
            return type_string

        if type_string.startswith("const "):
            return type_string
        return f"const {type_string}"

    def resource_type_with_access(self, type_name, node):
        """Apply resource access metadata to CUDA declaration type spelling."""
        type_string = self.apply_readonly_qualifier_to_type(type_name, node)
        type_string = self.type_name_string(type_string)
        if not type_string:
            return type_string

        access = self.explicit_resource_access(node)
        if access is None:
            return type_string

        base_type = type_string
        array_suffix = ""
        if "[" in type_string and "]" in type_string:
            open_bracket = type_string.find("[")
            base_type = type_string[:open_bracket]
            array_suffix = type_string[open_bracket:]

        parts = self.structured_buffer_type_parts(base_type)
        if parts is not None:
            base_name, element_type = parts
            if base_name in {"StructuredBuffer", "RWStructuredBuffer"}:
                mapped_base = (
                    "StructuredBuffer" if access == "readonly" else "RWStructuredBuffer"
                )
                return f"{mapped_base}<{element_type}>{array_suffix}"

        byte_base = self.byte_address_buffer_base_type(base_type)
        if byte_base in {"ByteAddressBuffer", "RWByteAddressBuffer"}:
            mapped_base = (
                "ByteAddressBuffer" if access == "readonly" else "RWByteAddressBuffer"
            )
            return f"{mapped_base}{array_suffix}"

        return type_string

    def is_storage_image_type(self, type_name):
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        base_type = self.resource_base_type(type_name)
        if not isinstance(base_type, str):
            return False
        image_shapes = {
            "image1D",
            "image1DArray",
            "image2D",
            "image2DArray",
            "image3D",
            "imageCube",
            "imageCubeArray",
            "image2DMS",
            "image2DMSArray",
        }
        return base_type in image_shapes or any(
            base_type == f"{prefix}{shape}"
            for prefix in ("i", "u")
            for shape in image_shapes
        )

    def is_buffer_resource_type(self, type_name):
        """Return whether a type names a CUDA-lowered buffer resource."""
        return (
            self.structured_buffer_type_parts(type_name) is not None
            or self.byte_address_buffer_base_type(type_name) is not None
        )

    def register_variable_type(self, name, type_name, node=None, source_node=None):
        if not name or type_name is None:
            return
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        type_name = self.apply_readonly_qualifier_to_type(type_name, node)
        self.variable_types[name] = type_name
        if self.is_glsl_buffer_block_node(node):
            access = self.explicit_resource_access(node) or "readwrite"
            self.glsl_buffer_block_accesses[name] = access
            self.glsl_buffer_block_layouts[name] = self.glsl_buffer_block_layout(node)
        else:
            self.glsl_buffer_block_accesses.pop(name, None)
            self.glsl_buffer_block_layouts.pop(name, None)

        if self.is_buffer_resource_type(type_name):
            access = self.explicit_resource_access(node)
            if access is None and source_node is not None:
                access = self.buffer_resource_access(source_node)
            if access is None:
                self.buffer_resource_accesses.pop(name, None)
            else:
                self.buffer_resource_accesses[name] = access
        else:
            self.buffer_resource_accesses.pop(name, None)

        self.register_storage_image_texel(name, type_name, node)
        if not self.is_storage_image_type(type_name):
            self.image_resource_accesses.pop(name, None)
            return

        access = self.explicit_resource_access(node)
        if access is None and source_node is not None:
            access = self.image_resource_access(source_node)
        if access is None:
            self.image_resource_accesses.pop(name, None)
        else:
            self.image_resource_accesses[name] = access

    def next_cuda_temp_variable(self, prefix):
        index = self.match_temp_variable_index
        self.match_temp_variable_index += 1
        return f"__crossgl_{prefix}_{index}"

    def get_expression_name(self, node):
        if isinstance(node, IdentifierNode):
            return node.name
        if isinstance(node, VariableNode):
            return node.name
        if isinstance(node, str):
            return node
        if isinstance(node, ArrayAccessNode):
            array_node = getattr(node, "array", getattr(node, "array_expr", None))
            return self.get_expression_name(array_node)
        return None

    def get_expression_type(self, node):
        name = self.get_expression_name(node)
        if name is None:
            return None
        return self.variable_types.get(name)

    def resource_expression_type(self, node):
        type_name = self.get_expression_type(node)
        if type_name is None:
            type_name = self.expression_result_type(node)
        if type_name is not None and not isinstance(type_name, str):
            return self.convert_type_node_to_string(type_name)
        return type_name

    def unsupported_dimension_metadata_query_call(self, func_name, resource_type):
        spec = self.dimension_query_spec(resource_type)
        if spec is None:
            return None
        return_type = self.query_return_type(spec["dimensions"])
        fallback = self.query_constructor(
            return_type,
            ["0"] * len(spec["dimensions"]),
        )
        return self.resource_query_metadata_unavailable_call(
            func_name, resource_type, fallback
        )

    def unsupported_scalar_metadata_query_call(self, func_name, resource_type):
        return self.resource_query_metadata_unavailable_call(
            func_name, resource_type, "0"
        )

    def unsupported_dimension_resource_query_call(self, func_name, resource_type):
        spec = self.dimension_query_spec(resource_type)
        if spec is None:
            return None
        return_type = self.query_return_type(spec["dimensions"])
        fallback = self.query_constructor(
            return_type,
            ["0"] * len(spec["dimensions"]),
        )
        return self.refuse_lowering(
            f"{self.resource_backend_name()} resource query: {func_name} on {resource_type}",
            fallback,
        )

    def generate_dimension_query(self, func_name, raw_args, args):
        if not raw_args:
            return None

        resource_type = self.resource_base_type(
            self.resource_expression_type(raw_args[0])
        )
        if resource_type is None:
            return None
        if func_name == "textureSize" and not self.is_sampled_resource_type(
            resource_type
        ):
            return self.unsupported_dimension_resource_query_call(
                func_name, resource_type
            )
        if func_name == "imageSize" and not self.is_storage_image_type(resource_type):
            return self.unsupported_dimension_resource_query_call(
                func_name, resource_type
            )
        spec = self.dimension_query_spec(resource_type)
        if spec is None:
            return None

        metadata_expr = self.query_metadata_expression(
            raw_args[0], allow_function_call_object=True
        )
        if metadata_expr is None:
            return self.unsupported_dimension_metadata_query_call(
                func_name, resource_type
            )

        self.resource_query_info_required = True
        helper_name = f"cgl_{func_name}_{resource_type}"
        if spec["mip"]:
            self.ensure_query_prefix_helper()
        self.require_helper_function(
            helper_name, self.build_dimension_query_helper(helper_name, spec)
        )

        if spec["mip"]:
            lod = args[1] if len(args) > 1 else "0"
            return f"{helper_name}({metadata_expr}, {lod})"
        return f"{helper_name}({metadata_expr})"

    def generate_sample_count_query(self, func_name, raw_args, args):
        if not raw_args:
            return None

        resource_type = self.resource_base_type(
            self.resource_expression_type(raw_args[0])
        )
        expected_resource = (
            self.is_sampled_resource_type(resource_type)
            if func_name == "textureSamples"
            else self.is_storage_image_type(resource_type)
        )
        if not expected_resource:
            return None

        spec = self.dimension_query_spec(resource_type)
        if spec is None or not spec["samples"]:
            return None

        metadata_expr = self.query_metadata_expression(
            raw_args[0], allow_function_call_object=True
        )
        if metadata_expr is None:
            return self.unsupported_scalar_metadata_query_call(func_name, resource_type)

        self.resource_query_info_required = True
        helper_name = f"cgl_{func_name}_{resource_type}"
        self.require_helper_function(
            helper_name, self.build_sample_count_query_helper(helper_name)
        )
        return f"{helper_name}({metadata_expr})"

    def generate_texture_query_levels(self, raw_args):
        if not raw_args:
            return None

        resource_type = self.resource_base_type(
            self.resource_expression_type(raw_args[0])
        )
        spec = self.dimension_query_spec(resource_type)
        if (
            not self.is_sampled_resource_type(resource_type)
            or spec is None
            or not spec["mip"]
        ):
            return self.unsupported_scalar_resource_query_call(
                "textureQueryLevels", resource_type
            )

        metadata_expr = self.query_metadata_expression(
            raw_args[0], allow_function_call_object=True
        )
        if metadata_expr is None:
            return self.unsupported_scalar_metadata_query_call(
                "textureQueryLevels", resource_type
            )

        self.resource_query_info_required = True
        helper_name = f"cgl_textureQueryLevels_{resource_type}"
        self.require_helper_function(
            helper_name, self.build_texture_query_levels_helper(helper_name, spec)
        )
        return f"{helper_name}({metadata_expr})"

    def map_vector_arithmetic_type(self, type_name):
        return self.convert_crossgl_type_to_cuda(type_name)
