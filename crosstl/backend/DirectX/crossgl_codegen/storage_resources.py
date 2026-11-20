"""Storage resources in the HLSL to CrossGL converter: resource type checks, storage image and typed buffer access, texture operator fetches and interlocked atomics."""

import re

from ...common_ast import ArrayAccessNode
from ..DirectxAst import MemberAccessNode


class StorageResourceMixin:
    """Storage resource members of ``HLSLToCrossGLConverter``."""

    def is_rasterizer_ordered_resource_type(self, type_name):
        return self.raw_type_base(type_name).startswith("RasterizerOrdered")

    def is_rw_texture_type(self, type_name):
        """Whether a resource is a storage image, including storage texel buffers."""
        return self.raw_type_base(type_name).startswith(
            ("RWTexture", "RasterizerOrderedTexture")
        ) or self.is_storage_texel_buffer_type(type_name)

    def is_multisample_texture_type(self, type_name):
        return self.raw_type_base(type_name) in {
            "Texture2DMS",
            "Texture2DMSArray",
            "RWTexture2DMS",
            "RWTexture2DMSArray",
            "RasterizerOrderedTexture2DMS",
            "RasterizerOrderedTexture2DMSArray",
        }

    def is_buffer_resource_type(self, type_name):
        return self.raw_type_base(type_name) in {
            "Buffer",
            "RWBuffer",
            "StructuredBuffer",
            "RWStructuredBuffer",
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
            "ByteAddressBuffer",
            "RWByteAddressBuffer",
            "RasterizerOrderedBuffer",
            "RasterizerOrderedStructuredBuffer",
            "RasterizerOrderedByteAddressBuffer",
        }

    def is_byte_address_buffer_type(self, type_name):
        return self.raw_type_base(type_name) in {
            "ByteAddressBuffer",
            "RWByteAddressBuffer",
            "RasterizerOrderedByteAddressBuffer",
        }

    def is_rw_typed_buffer_type(self, type_name):
        return self.raw_type_base(type_name) in {
            "RWBuffer",
            "RWStructuredBuffer",
            "RasterizerOrderedBuffer",
            "RasterizerOrderedStructuredBuffer",
        }

    def array_access_depth(self, expr):
        depth = 0
        while isinstance(expr, ArrayAccessNode):
            depth += 1
            expr = expr.array
        return depth

    def is_storage_image_texel_access(self, expr):
        if not isinstance(expr, ArrayAccessNode):
            return False
        if not self.is_rw_texture_type(self.expression_raw_type(expr)):
            return False
        # One index after the resource-array indices selects the texel; a
        # further index subscripts a component of the loaded texel.
        return (
            self.array_access_depth(expr)
            == self.expression_resource_array_dims(expr) + 1
        )

    def generate_without_storage_index_lowering(self, expr, is_main=False):
        previous = self.suppress_storage_image_index_lowering
        self.suppress_storage_image_index_lowering = True
        try:
            return self.generate_expression(expr, is_main)
        finally:
            self.suppress_storage_image_index_lowering = previous

    def generate_storage_image_access_parts(self, expr, is_main=False):
        image = self.generate_without_storage_index_lowering(expr.array, is_main)
        coord = self.generate_expression(expr.index, is_main)
        return image, coord

    def generate_storage_image_load(self, expr, is_main=False, narrow=True):
        image, coord = self.generate_storage_image_access_parts(expr, is_main)
        load = f"imageLoad({image}, {coord})"
        if not narrow:
            return load
        return load + self.texel_narrowing_suffix(self.expression_raw_type(expr.array))

    def generate_storage_image_store(self, access, value, operator, is_main=False):
        image, coord = self.generate_storage_image_access_parts(access, is_main)
        rendered_value = self.generate_expression(value, is_main)
        resource_type = self.expression_raw_type(access.array)
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
            current_value = self.generate_storage_image_load(access, is_main)
            rendered_value = (
                f"{current_value} {binary_op} "
                f"{self.maybe_parenthesize(value, rendered_value)}"
            )
            value = access
        rendered_value = self.widen_storage_image_value(
            resource_type, value, rendered_value
        )
        rendered_value = self.unorm_store_value(
            access.array, resource_type, rendered_value
        )
        return f"imageStore({image}, {coord}, {rendered_value})"

    def storage_image_component_assignment_access(self, expr):
        if isinstance(expr, ArrayAccessNode):
            # ``image[p][i]`` writes component ``i`` of a texel.
            index = expr.index
            index = getattr(index, "value", index)
            text = str(index).rstrip("uU") if isinstance(index, (int, str)) else ""
            if (
                text.isdigit()
                and int(text) < 4
                and self.is_storage_image_texel_access(expr.array)
            ):
                return expr.array, "xyzw"[int(text)]
            return None
        if not isinstance(expr, MemberAccessNode):
            return None
        if not isinstance(expr.member, str):
            return None
        if not re.fullmatch(r"[xyzwrgba]{1,4}", expr.member):
            return None
        if not self.is_storage_image_texel_access(expr.object):
            return None
        return expr.object, expr.member

    def storage_image_component_type(self, access):
        raw_type = self.expression_raw_type(access)
        if raw_type is None:
            return None
        type_name = str(raw_type)
        if "<" not in type_name or ">" not in type_name:
            return None
        return type_name.split("<", 1)[1].rsplit(">", 1)[0].strip()

    def normalize_resource_component_type(self, component_type):
        if component_type is None:
            return None
        component_type = str(component_type).strip()
        component_type = re.sub(r"^(snorm|unorm)\s*,?\s+", "", component_type)
        return self.canonical_composite_type(component_type)

    def storage_image_component_count(self, component_type):
        component_type = self.normalize_resource_component_type(component_type)
        if not component_type:
            return None
        if re.fullmatch(
            r"(?:min16float|min10float|min16uint|min16int|min12int|float16_t|"
            r"uint16_t|int16_t|float|half|fixed|double|uint|int|bool)[2-4]",
            component_type,
        ):
            return int(component_type[-1])
        if component_type in {
            "min16float",
            "min10float",
            "min16uint",
            "min16int",
            "min12int",
            "float16_t",
            "uint16_t",
            "int16_t",
            "float",
            "half",
            "fixed",
            "double",
            "uint",
            "int",
            "bool",
        }:
            return 1
        return None

    def generate_storage_image_component_store(
        self, access, component, value, operator, is_main=False
    ):
        component_type = self.storage_image_component_type(access)
        component_count = self.storage_image_component_count(component_type)
        if component_count == 1 and len(component) == 1 and component in {"x", "r"}:
            return self.generate_storage_image_store(access, value, operator, is_main)
        return None

    def generate_storage_image_atomic_value(self, arg, component_type, is_main):
        if (
            component_type == "uint"
            and isinstance(arg, int)
            and not isinstance(arg, bool)
            and arg >= 0
        ):
            return f"{arg}u"
        return self.generate_expression(arg, is_main)

    def typed_buffer_component_type(self, access):
        raw_type = self.expression_raw_type(access)
        if raw_type is None:
            return None
        type_name = str(raw_type)
        if "<" not in type_name or ">" not in type_name:
            return None
        return type_name.split("<", 1)[1].rsplit(">", 1)[0].strip()

    def is_typed_buffer_element_access(self, expr):
        if not isinstance(expr, ArrayAccessNode):
            return False
        if not self.is_rw_typed_buffer_type(self.expression_raw_type(expr)):
            return False
        return self.array_access_depth(expr) > self.expression_resource_array_dims(expr)

    def is_sampled_multisample_texture_type(self, type_name):
        return self.raw_type_base(type_name) in {"Texture2DMS", "Texture2DMSArray"}

    def is_sampled_texture_operator_type(self, type_name):
        return self.raw_type_base(type_name) in {
            "Texture1D",
            "Texture1DArray",
            "Texture2D",
            "Texture2DArray",
            "Texture3D",
        }

    def generate_multisample_texture_operator_fetch(
        self, expr, is_main=False, narrow=True
    ):
        if not isinstance(expr, ArrayAccessNode):
            return None

        if (
            isinstance(expr.array, ArrayAccessNode)
            and isinstance(expr.array.array, MemberAccessNode)
            and str(expr.array.array.member).lower() == "sample"
            and self.is_sampled_multisample_texture_type(
                self.expression_raw_type(expr.array.array.object)
            )
        ):
            texture = self.generate_expression(expr.array.array.object, is_main)
            sample_index = self.integer_texel_operand(expr.array.index, is_main=is_main)
            coord = self.integer_texel_operand(expr.index, is_main=is_main)
            suffix = (
                self.texel_narrowing_suffix(
                    self.expression_raw_type(expr.array.array.object)
                )
                if narrow
                else ""
            )
            return f"texelFetch({texture}, {coord}, {sample_index}){suffix}"

        if (
            isinstance(expr.array, MemberAccessNode)
            and str(expr.array.member).lower() == "sample"
        ):
            return None

        if not self.is_sampled_multisample_texture_type(
            self.expression_raw_type(expr.array)
        ):
            return None

        if (
            self.array_access_depth(expr)
            != self.expression_resource_array_dims(expr) + 1
        ):
            return None

        texture = self.generate_expression(expr.array, is_main)
        coord = self.integer_texel_operand(expr.index, is_main=is_main)
        suffix = (
            self.texel_narrowing_suffix(self.expression_raw_type(expr.array))
            if narrow
            else ""
        )
        return f"texelFetch({texture}, {coord}, 0){suffix}"

    def generate_sampled_texture_operator_fetch(self, expr, is_main=False, narrow=True):
        if not isinstance(expr, ArrayAccessNode):
            return None
        if not self.is_sampled_texture_operator_type(
            self.expression_raw_type(expr.array)
        ):
            return None
        if (
            self.array_access_depth(expr)
            != self.expression_resource_array_dims(expr) + 1
        ):
            return None

        texture = self.generate_expression(expr.array, is_main)
        coord = self.integer_texel_operand(expr.index, is_main=is_main)
        suffix = (
            self.texel_narrowing_suffix(self.expression_raw_type(expr.array))
            if narrow
            else ""
        )
        return f"texelFetch({texture}, {coord}, 0){suffix}"

    def generate_typed_buffer_atomic_value(self, arg, component_type, is_main):
        if (
            component_type == "uint"
            and isinstance(arg, int)
            and not isinstance(arg, bool)
            and arg >= 0
        ):
            return f"{arg}u"
        return self.generate_expression(arg, is_main)

    def interlocked_storage_image_atomic_expression(self, func_name, args, is_main):
        operation_map = {
            "InterlockedAdd": "imageAtomicAdd",
            "InterlockedAnd": "imageAtomicAnd",
            "InterlockedOr": "imageAtomicOr",
            "InterlockedXor": "imageAtomicXor",
            "InterlockedMin": "imageAtomicMin",
            "InterlockedMax": "imageAtomicMax",
            "InterlockedExchange": "imageAtomicExchange",
            "InterlockedCompareExchange": "imageAtomicCompSwap",
        }
        operation = operation_map.get(func_name)
        if (
            operation is None
            or not args
            or not self.is_storage_image_texel_access(args[0])
        ):
            return None

        expected_min_args = 3 if func_name == "InterlockedCompareExchange" else 2
        if len(args) < expected_min_args:
            return None

        image, coord = self.generate_storage_image_access_parts(args[0], is_main)
        component_type = self.storage_image_component_type(args[0])
        if func_name == "InterlockedCompareExchange":
            value_args = [
                self.generate_storage_image_atomic_value(
                    args[1], component_type, is_main
                ),
                self.generate_storage_image_atomic_value(
                    args[2], component_type, is_main
                ),
            ]
            original_index = 3
        else:
            value_args = [
                self.generate_storage_image_atomic_value(
                    args[1], component_type, is_main
                )
            ]
            original_index = 2

        atomic_call = f"{operation}({image}, {coord}, {', '.join(value_args)})"
        if len(args) > original_index:
            original = self.generate_without_storage_index_lowering(
                args[original_index], is_main
            )
            return f"{original} = {atomic_call}"
        return atomic_call

    def interlocked_typed_buffer_atomic_expression(self, func_name, args, is_main):
        operation_map = {
            "InterlockedAdd": "atomicAdd",
            "InterlockedAnd": "atomicAnd",
            "InterlockedOr": "atomicOr",
            "InterlockedXor": "atomicXor",
            "InterlockedMin": "atomicMin",
            "InterlockedMax": "atomicMax",
            "InterlockedExchange": "atomicExchange",
            "InterlockedCompareExchange": "atomicCompareExchange",
        }
        operation = operation_map.get(func_name)
        if (
            operation is None
            or not args
            or not self.is_typed_buffer_element_access(args[0])
        ):
            return None

        expected_min_args = 3 if func_name == "InterlockedCompareExchange" else 2
        if len(args) < expected_min_args:
            return None

        target = self.generate_without_storage_index_lowering(args[0], is_main)
        component_type = self.typed_buffer_component_type(args[0])
        if func_name == "InterlockedCompareExchange":
            value_args = [
                self.generate_typed_buffer_atomic_value(
                    args[1], component_type, is_main
                ),
                self.generate_typed_buffer_atomic_value(
                    args[2], component_type, is_main
                ),
            ]
            original_index = 3
        else:
            value_args = [
                self.generate_typed_buffer_atomic_value(
                    args[1], component_type, is_main
                )
            ]
            original_index = 2

        atomic_call = f"{operation}({target}, {', '.join(value_args)})"
        if len(args) > original_index:
            original = self.generate_without_storage_index_lowering(
                args[original_index], is_main
            )
            return f"{original} = {atomic_call}"
        return atomic_call
