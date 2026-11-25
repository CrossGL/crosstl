"""Surface reads and writes in the HIP to CrossGL converter.

HIP declares the CUDA surface functions in amd_surface_functions.h, all in the pointer form for reads
(``surf2Dread(&v, s, x, y)``) and with a pointer value for ``surfCubemapLayeredwrite``. The element types, image
formats, texel conversions and byte-offset division they share with the CUDA importer are in
:mod:`crosstl.backend.CUDA.surface_formats`.
"""

from ...CUDA.surface_formats import SurfaceAccessMixin, surface_value_type
from ..HipAst import UnaryOpNode, VariableNode


class HipSurfaceMixin(SurfaceAccessMixin):
    """Surface members of ``HipToCrossGLConverter``."""

    SURFACE_LANGUAGE = "HIP"

    def surface_write_value_is_pointer(self, base_name):
        return base_name == "surfCubemapLayeredwrite"

    def surface_member_type(self, object_type, member):
        """The declared type of member ``member`` of the program's record ``object_type``, or ``None``."""
        definition = self.record_definition_from_type(str(object_type))
        if definition is None:
            return None
        for field in getattr(definition, "members", []) or []:
            if isinstance(field, VariableNode) and field.name == member:
                return self.strip_type_qualifiers(field.vtype)
        return None

    def format_hip_surface_read(
        self, function_name, args, dimensions, value_type, raw_args=None
    ):
        if not args:
            return None

        if self.is_surface_output_target(args[0]):
            if len(args) < 2:
                return None
            output_target = self.strip_surface_output_target(args[0])
            coord_start = 2
            expected_arg_count = dimensions + 2
        else:
            output_target = None
            coord_start = 1
            expected_arg_count = dimensions + 1
        surface_name = args[coord_start - 1]

        def assigned(expression):
            return (
                expression
                if output_target is None
                else f"{output_target} = {expression}"
            )

        if len(args) < expected_arg_count:
            if len(args) > coord_start:
                return assigned(
                    self.format_hip_surface_coordinate_shape_diagnostic(
                        function_name, value_type
                    )
                )
            return None

        raw_args = list(raw_args or [])
        if output_target is not None and value_type is None and raw_args:
            dest_node = raw_args[0]
            if isinstance(dest_node, UnaryOpNode) and dest_node.op == "&":
                value_type = self.hip_surface_value_type(dest_node.operand)

        if self.has_hip_surface_x_shape_mismatch(args, coord_start, raw_args):
            return assigned(
                self.format_hip_surface_coordinate_shape_diagnostic(
                    function_name, value_type
                )
            )
        raw_surface = (
            raw_args[coord_start - 1] if len(raw_args) >= coord_start else surface_name
        )
        access = self.surface_access_element(function_name, raw_surface, value_type)
        if access is None:
            return assigned("0")
        element, size = access
        coord_args = self.format_hip_surface_coordinate_args(
            args, coord_start, dimensions, size, raw_args, function_name
        )
        if self.has_hip_surface_coordinate_shape_mismatch(coord_args):
            return assigned(
                self.format_hip_surface_coordinate_shape_diagnostic(
                    function_name, value_type
                )
            )

        coord = self.format_vector_constructor(f"vec{dimensions}", coord_args, "i32")
        image_load = f"imageLoad({surface_name}, {coord})"
        if element is not None:
            image_load = element.load(image_load)
        return assigned(image_load)

    def format_hip_surface_write(
        self,
        function_name,
        args,
        dimensions,
        value_type,
        value_is_pointer=False,
        raw_args=None,
    ):
        if len(args) < dimensions + 2:
            if len(args) > 2:
                return self.format_hip_surface_coordinate_shape_diagnostic(
                    function_name, None
                )
            return None
        raw_args = list(raw_args or [])
        value = args[0]
        if value_is_pointer:
            value = self.strip_surface_output_target(value)
        raw_value = raw_args[0] if raw_args else None
        if value_is_pointer and isinstance(raw_value, UnaryOpNode):
            if raw_value.op == "&":
                raw_value = raw_value.operand
        if value_type is None and raw_value is not None:
            value_type = self.hip_surface_value_type(raw_value)
        surface_name = args[1]
        if self.has_hip_surface_x_shape_mismatch(args, 2, raw_args):
            return self.format_hip_surface_coordinate_shape_diagnostic(
                function_name, None
            )
        raw_surface = raw_args[1] if len(raw_args) > 1 else surface_name
        access = self.surface_access_element(function_name, raw_surface, value_type)
        if access is None:
            return "0"
        element, size = access
        coord_args = self.format_hip_surface_coordinate_args(
            args, 2, dimensions, size, raw_args, function_name
        )
        if self.has_hip_surface_coordinate_shape_mismatch(coord_args):
            return self.format_hip_surface_coordinate_shape_diagnostic(
                function_name, None
            )
        coord = self.format_vector_constructor(f"vec{dimensions}", coord_args, "i32")
        if element is not None:
            value = element.store(value)
        return f"imageStore({surface_name}, {coord}, {value})"

    def hip_surface_value_type(self, expression):
        """The HIP type of a surface write's value or a pointer-form read's destination, or ``None``."""
        return surface_value_type(
            expression, self.lookup_variable_type, self.surface_member_type
        )

    def has_hip_surface_x_shape_mismatch(self, args, coord_start, raw_args):
        """Whether the x coordinate is a vector: a shape mismatch, not a byte offset to scale."""
        raw_offset = raw_args[coord_start] if len(raw_args) > coord_start else None
        x_rank = self.infer_texture_coordinate_rank(args[coord_start], raw_offset)
        return x_rank is not None and x_rank != 1

    def format_hip_surface_coordinate_args(
        self,
        args,
        coord_start,
        dimensions,
        element_size,
        raw_args=None,
        function_name=None,
    ):
        raw_offset = (
            raw_args[coord_start]
            if raw_args is not None and len(raw_args) > coord_start
            else None
        )
        coord_args = [
            self.scale_hip_surface_byte_offset(
                raw_offset, element_size, function_name or "surf"
            )
        ]
        coord_args.extend(args[coord_start + 1 : coord_start + dimensions])
        return coord_args

    def has_hip_surface_coordinate_shape_mismatch(self, coord_args):
        for coordinate in coord_args:
            rank = self.infer_texture_coordinate_rank(coordinate)
            if rank is not None and rank != 1:
                return True
        return False

    def format_hip_surface_coordinate_shape_diagnostic(
        self, function_name, value_type=None
    ):
        fallback = self.format_hip_zero_value(value_type) if value_type else "0"
        return self.format_unsupported_hip_resource_expression(
            "surface",
            f"{function_name} coordinate shape mismatch",
            fallback,
        )

    def format_hip_zero_value(self, type_name):
        if not type_name:
            return "0"

        crossgl_type = self.convert_hip_type_to_crossgl(type_name)
        if (
            crossgl_type.startswith("vec")
            and len(crossgl_type) >= 4
            and crossgl_type[3].isdigit()
            and "<" in crossgl_type
            and crossgl_type.endswith(">")
        ):
            rank = int(crossgl_type[3])
            element_type = crossgl_type.split("<", 1)[1][:-1]
            zero = self.format_hip_zero_scalar(element_type)
            return f"{crossgl_type}({', '.join([zero] * rank)})"
        return self.format_hip_zero_scalar(crossgl_type)

    def format_hip_zero_scalar(self, type_name):
        if type_name == "bool":
            return "false"
        if type_name in {"f32", "f64"}:
            return "0.0"
        return "0"

    def is_surface_output_target(self, expression):
        text = str(expression).strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()
        return text.startswith("&")

    def strip_surface_output_target(self, expression):
        text = str(expression).strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()
        if text.startswith("&"):
            return text[1:].strip()
        return text
