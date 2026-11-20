"""Type mapping of the HIP generator: CrossGL type spellings, stream and patch types, resource access types, declarators and the kernel wrapper."""

from ..ast import FunctionNode
from ..matrix_shape import matrix_type_spelling
from .array_utils import prepend_array_dimension
from .c_narrow_integer_vectors import generic_narrow_vector_spelling
from .c_wide_integer_vectors import wide_integer_vector_type
from .enum_utils import generic_enum_specialized_type_name
from .generic_struct_utils import generic_struct_specialized_type_name
from .resource_arrays import format_array_declarator
from .written_type_names import (
    HIP_TYPE_NAMES,
    written_type_name,
)


class HipTypeMixin:
    """Type members of ``HipCodeGen``."""

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if hasattr(type_node, "pointee_type"):
            pointee_type = self.convert_type_node_to_string(type_node.pointee_type)
            prefix = "" if getattr(type_node, "is_mutable", True) else "const "
            return f"{prefix}{pointee_type}*"
        if hasattr(type_node, "referenced_type"):
            referenced_type = self.convert_type_node_to_string(
                type_node.referenced_type
            )
            prefix = "" if getattr(type_node, "is_mutable", False) else "const "
            return f"{prefix}{referenced_type}&"
        if hasattr(type_node, "name"):
            generic_args = getattr(type_node, "generic_args", [])
            if generic_args:
                converted_args = []
                for arg in generic_args:
                    if (
                        hasattr(arg, "name")
                        or hasattr(arg, "element_type")
                        or hasattr(arg, "pointee_type")
                        or hasattr(arg, "referenced_type")
                    ):
                        converted = self.convert_type_node_to_string(arg)
                    elif hasattr(arg, "value"):
                        converted = str(arg.value)
                    else:
                        converted = str(arg)
                    if converted is None:
                        if hasattr(arg, "value"):
                            converted = str(arg.value)
                        else:
                            converted = str(arg)
                    converted_args.append(converted)
                args = ", ".join(converted_args)
                return f"{type_node.name}<{args}>"
            return type_node.name
        elif hasattr(type_node, "element_type"):
            if hasattr(type_node, "rows"):
                element_type = self.convert_type_node_to_string(type_node.element_type)
                return matrix_type_spelling(type_node, element_type, short_square=False)
            elif not hasattr(type_node, "size"):
                return str(type_node)
            elif str(type(type_node)).find("ArrayType") != -1:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                if type_node.size is not None:
                    return prepend_array_dimension(
                        element_type, self.format_array_size(type_node.size)
                    )
                else:
                    return prepend_array_dimension(element_type)
            else:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                size = type_node.size
                if element_type == "float":
                    return f"float{size}"
                elif element_type in {"f16", "float16", "half"}:
                    return f"vec{size}<f16>"
                elif element_type == "int":
                    return f"int{size}"
                elif element_type in {"i8", "u8", "i16", "u16"}:
                    return f"vec{size}<{element_type}>"
                elif wide_integer_vector_type(f"vec{size}<{element_type}>"):
                    return f"vec{size}<{element_type}>"
                else:
                    return f"{element_type}{size}"
        else:
            return str(type_node)

    def map_type(self, type_name) -> str:
        """Map a CrossGL type name or type node to a HIP type string."""
        if (
            hasattr(type_name, "name")
            or hasattr(type_name, "element_type")
            or hasattr(type_name, "pointee_type")
            or hasattr(type_name, "referenced_type")
        ):
            type_str = self.convert_type_node_to_string(type_name)
        else:
            type_str = str(type_name)
        type_str = generic_narrow_vector_spelling(type_str) or type_str

        generic_enum_type = generic_enum_specialized_type_name(self, type_str)
        if generic_enum_type is not None:
            return generic_enum_type

        generic_struct_type = generic_struct_specialized_type_name(self, type_str)
        if generic_struct_type is not None:
            return generic_struct_type

        if type_str in getattr(self, "enum_type_names", set()):
            return "int"

        if type_str in getattr(self, "enum_struct_type_names", set()):
            return type_str

        geometry_stream_type = self.hip_geometry_stream_mapped_type(type_str)
        if geometry_stream_type is not None:
            return geometry_stream_type

        tessellation_patch_type = self.hip_tessellation_patch_mapped_type(type_str)
        if tessellation_patch_type is not None:
            return tessellation_patch_type

        if "[" in type_str and "]" in type_str:
            base_type = type_str.split("[")[0]
            array_part = type_str[type_str.find("[") :]
            canonical_base = self.canonical_resource_type(base_type)
            mapped_base = self.map_type(canonical_base or base_type)
            return f"{mapped_base}{array_part}"

        structured_buffer_type = self.hip_structured_buffer_type(type_str)
        if structured_buffer_type is not None:
            return structured_buffer_type

        byte_address_buffer_type = self.hip_byte_address_buffer_type(type_str)
        if byte_address_buffer_type is not None:
            return byte_address_buffer_type

        wide_vector_type = wide_integer_vector_type(type_str)
        if wide_vector_type is not None:
            return wide_vector_type
        indirect_info = self.hip_pointer_or_reference_type_info(type_str)
        if indirect_info is not None:
            pointee_type = wide_integer_vector_type(indirect_info["pointee_type"])
            if pointee_type is not None:
                prefix = "const " if indirect_info["readonly"] else ""
                suffix = "*" if indirect_info["kind"] == "pointer" else "&"
                return f"{prefix}{pointee_type}{suffix}"
        if type_str.startswith("ptr<") and type_str.endswith(">"):
            return f"{self.map_type(type_str[4:-1])}*"

        canonical_type = self.canonical_resource_type(type_str)
        if canonical_type is not None:
            # Every canonical resource spelling is a type_map key.
            return self.hip_mapped_type_result(self.type_map[canonical_type])
        if type_str in self.type_map:
            return self.hip_mapped_type_result(self.type_map[type_str])
        # A program type or a HIP type is written as it is; a built-in type of another language is recorded and
        # refused by render_checked_program if the output still names it.
        return self.hip_mapped_type_result(
            written_type_name(
                type_str,
                HIP_TYPE_NAMES,
                self.program_type_names,
                self.written_foreign_type_names,
            )
        )

    def hip_geometry_stream_info(self, type_name):
        if type_name is None:
            return None

        stream_names = {"PointStream", "LineStream", "TriangleStream"}
        if hasattr(type_name, "pointee_type") or hasattr(type_name, "referenced_type"):
            return None

        name = getattr(type_name, "name", None)
        generic_args = getattr(type_name, "generic_args", []) or []
        if name in stream_names and generic_args:
            return name, self.map_type(generic_args[0])

        type_text = self.type_name_string(type_name)
        if not type_text or "<" not in type_text or not type_text.endswith(">"):
            return None

        base_name, generic_arg = type_text.split("<", 1)
        base_name = base_name.strip()
        if base_name not in stream_names:
            return None
        generic_arg = generic_arg[:-1].strip()
        if not generic_arg:
            return None
        return base_name, self.map_type(generic_arg)

    def hip_geometry_stream_mapped_type(self, type_name):
        stream_info = self.hip_geometry_stream_info(type_name)
        if stream_info is None:
            return None
        stream_name, output_type = stream_info
        return f"CglHip{stream_name}<{output_type}>"

    def format_hip_geometry_stream_parameter(self, raw_param_type, name):
        stream_type = self.hip_geometry_stream_mapped_type(raw_param_type)
        if stream_type is None:
            return None
        return f"{stream_type}& {name}"

    def hip_tessellation_patch_type_info(self, type_name):
        parts = self.generic_type_parts(type_name)
        if parts is None:
            return None
        base_name, args = parts
        if base_name not in {"InputPatch", "OutputPatch"} or len(args) != 2:
            return None
        element_type, point_count = args
        if not element_type or not point_count:
            return None
        mapped_element_type = self.map_type(element_type)
        return base_name, mapped_element_type, point_count

    def hip_tessellation_patch_mapped_type(self, type_name):
        patch_info = self.hip_tessellation_patch_type_info(type_name)
        if patch_info is None:
            return None
        base_name, element_type, point_count = patch_info
        helper_name = {
            "InputPatch": "CglHipInputPatch",
            "OutputPatch": "CglHipOutputPatch",
        }[base_name]
        return f"{helper_name}<{element_type}, {point_count}>"

    def hip_mapped_type_result(self, mapped_type):
        base_type = str(mapped_type).split("[", 1)[0].strip()
        if base_type == "cgl_half3":
            self.require_hip_half3_helper()
        if base_type == "cgl_half4":
            self.require_hip_half4_helper()
        if base_type in {
            "CglRayTracingAccelerationStructure",
            "CglRayDesc",
            "CglRayQuery",
            "CglBuiltInTriangleIntersectionAttributes",
        }:
            self.require_hip_ray_runtime_helpers()
        return mapped_type

    def apply_readonly_qualifier_to_type(self, type_name, node):
        """Apply readonly declaration qualifiers to pointer/reference types."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return type_name

        if self.hip_pointer_or_reference_type_info(type_name) is None:
            return type_name

        if not (
            self.declaration_qualifier_names(node) & {"const", "readonly", "constant"}
        ):
            return type_name

        if type_name.startswith("const "):
            return type_name
        return f"const {type_name}"

    def resource_type_with_access(self, type_name, node):
        type_name = self.apply_readonly_qualifier_to_type(type_name, node)
        if not type_name:
            return type_name

        access = self.explicit_resource_access(node)
        if access is None:
            return type_name

        base_type = type_name
        array_suffix = ""
        if "[" in type_name and "]" in type_name:
            open_bracket = type_name.find("[")
            base_type = type_name[:open_bracket]
            array_suffix = type_name[open_bracket:]

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

        return type_name

    def generic_type_parts(self, type_name):
        """Split a generic type name into base name and top-level arguments."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None

        base_type = type_name.split("[", 1)[0].strip()
        generic_start = base_type.find("<")
        generic_end = base_type.rfind(">")
        if generic_start == -1 or generic_end < generic_start:
            return None

        base_name = base_type[:generic_start].strip()
        args_text = base_type[generic_start + 1 : generic_end].strip()
        args = []
        depth = 0
        current = []
        for char in args_text:
            if char == "<":
                depth += 1
                current.append(char)
            elif char == ">":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                args.append("".join(current).strip())
                current = []
            else:
                current.append(char)

        trailing_arg = "".join(current).strip()
        if trailing_arg:
            args.append(trailing_arg)
        return base_name, args

    def structured_buffer_type_parts(self, type_name):
        """Return structured-buffer base and element type, if applicable."""
        parts = self.generic_type_parts(type_name)
        if parts is None:
            return None

        base_name, args = parts
        if (
            base_name not in {
                "StructuredBuffer",
                "RWStructuredBuffer",
                "AppendStructuredBuffer",
                "ConsumeStructuredBuffer",
            }
            or not args
        ):
            return None
        return base_name, args[0]

    def structured_buffer_is_writable(self, type_name):
        """Return whether a structured-buffer type permits writes."""
        parts = self.structured_buffer_type_parts(type_name)
        return parts is not None and parts[0] == "RWStructuredBuffer"

    def structured_buffer_requires_counter(self, type_name):
        """Return whether a structured-buffer type needs an explicit counter."""
        parts = self.structured_buffer_type_parts(type_name)
        return parts is not None and parts[0] in {
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
        }

    def hip_structured_buffer_type(self, type_name):
        """Map structured-buffer resources to HIP pointer types."""
        parts = self.structured_buffer_type_parts(type_name)
        if parts is None:
            return None

        base_name, element_type = parts
        hip_element_type = self.map_type(element_type)
        if base_name in {"StructuredBuffer", "ConsumeStructuredBuffer"}:
            return f"const {hip_element_type}*"
        return f"{hip_element_type}*"

    def byte_address_buffer_base_type(self, type_name):
        """Return the byte-address buffer base type, if applicable."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None
        base_type = type_name.split("[", 1)[0].strip()
        if base_type in {"ByteAddressBuffer", "RWByteAddressBuffer"}:
            return base_type
        return None

    def byte_address_buffer_is_writable(self, type_name):
        """Return whether a byte-address buffer type permits writes."""
        return self.byte_address_buffer_base_type(type_name) == "RWByteAddressBuffer"

    def hip_byte_address_buffer_type(self, type_name):
        """Map ByteAddressBuffer and RWByteAddressBuffer to HIP byte pointers."""
        base_type = self.byte_address_buffer_base_type(type_name)
        if base_type == "ByteAddressBuffer":
            return "const unsigned char*"
        if base_type == "RWByteAddressBuffer":
            return "unsigned char*"
        return None

    def format_typed_declarator(self, type_name, name, dynamic_array_as_pointer=True):
        if (
            hasattr(type_name, "name")
            or hasattr(type_name, "element_type")
            or hasattr(type_name, "pointee_type")
            or hasattr(type_name, "referenced_type")
        ):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)

        if "[" not in type_name or "]" not in type_name:
            return f"{self.map_type(type_name)} {name}"

        open_bracket = type_name.find("[")
        base_type = type_name[:open_bracket]
        array_suffix = type_name[open_bracket:]
        mapped_base = self.map_type(base_type)

        return format_array_declarator(
            mapped_base,
            name,
            array_suffix,
            dynamic_array_as_pointer=dynamic_array_as_pointer,
        )

    def format_array_size(self, size):
        if size is None:
            return ""
        if isinstance(size, int):
            return str(size)
        return self.visit(size)

    def generate_kernel_wrapper(self, kernel_node: FunctionNode) -> str:
        """Generate a host-side HIP launch wrapper for a kernel node."""
        wrapper_lines = []

        wrapper_name = f"launch_{kernel_node.name}"
        params = []
        args = []

        for param in kernel_node.parameters:
            param_type = self.map_type(param.param_type)
            params.append(f"{param_type} {param.name}")
            args.append(param.name)

        params.extend(["dim3 gridSize", "dim3 blockSize", "hipStream_t stream = 0"])

        wrapper_lines.extend(
            [
                f"void {wrapper_name}({', '.join(params)})",
                "{",
                f"    hipLaunchKernelGGL({kernel_node.name}, gridSize, blockSize, 0, stream, {', '.join(args)});",
                "}",
            ]
        )

        return "\n".join(wrapper_lines)
