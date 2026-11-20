"""Resource methods in the HLSL to CrossGL converter: method descriptors and arguments, byte-address atomics, texture gathers and loads, legacy texture functions and their bound samplers, and swizzles."""

import re

from ..DirectxAst import VectorConstructorNode
from ..subpass_inputs import SUBPASS_INPUT_TYPES


class ResourceMethodMixin:
    """Resource method members of ``HLSLToCrossGLConverter``."""

    def resource_method_descriptor(self, member, arg_count=None, resource_type=None):
        member = self.templated_method_base(member)
        if self.raw_type_base(resource_type) in {
            "PointStream",
            "LineStream",
            "TriangleStream",
        }:
            # Geometry stream Append/RestartStrip stay stream methods in CrossGL.
            return None
        if (
            member == "SubpassLoad"
            and self.raw_type_base(resource_type) in SUBPASS_INPUT_TYPES
        ):
            return {
                "member": member,
                "function": "subpassLoad",
                "texture_function": "subpassLoad",
                "buffer_function": None,
                "component": None,
                "usage": "regular",
                "buffer_when_max_args": None,
                "resource": "subpass_input",
                "resource_type": resource_type,
                "operation": "subpass_load",
            }

        texture_descriptor = self.texture_method_descriptor(
            member, arg_count, resource_type
        )
        if texture_descriptor:
            descriptor = dict(texture_descriptor)
            uses_buffer = (
                descriptor["buffer_function"] is not None
                and descriptor["function"] == descriptor["buffer_function"]
            )
            if descriptor["function"] == "imageLoad":
                descriptor["resource"] = "image"
            else:
                descriptor["resource"] = "buffer" if uses_buffer else "texture"
            descriptor["operation"] = {
                "Load": "load",
                "GetDimensions": "dimensions",
                "Sample": "sample",
                "SampleLevel": "sample_lod",
                "SampleGrad": "sample_grad",
                "SampleBias": "sample_bias",
                "SampleCmp": "sample_compare",
                "SampleCmpLevel": "sample_compare_lod",
                "SampleCmpGrad": "sample_compare_grad",
                "SampleCmpBias": "sample_compare_bias",
                "SampleCmpLevelZero": "sample_compare",
                "CalculateLevelOfDetail": "query_lod",
                "CalculateLevelOfDetailUnclamped": "query_lod",
                "GetSamplePosition": "query_sample_position",
                "Gather": "gather",
                "GatherRed": "gather",
                "GatherGreen": "gather",
                "GatherBlue": "gather",
                "GatherAlpha": "gather",
                "GatherCmp": "gather_compare",
                "GatherCmpRed": "gather_compare",
                "GatherCmpGreen": "gather_compare",
                "GatherCmpBlue": "gather_compare",
                "GatherCmpAlpha": "gather_compare",
            }.get(member)
            return descriptor

        if member in {"Load2", "Load3", "Load4"} and self.is_byte_address_buffer_type(
            resource_type
        ):
            descriptor = {
                "member": member,
                "function": self.buffer_method_map[member],
                "texture_function": None,
                "buffer_function": self.buffer_method_map[member],
                "component": None,
                "usage": None,
                "buffer_when_max_args": None,
                "resource_type": resource_type,
                "resource": "buffer",
                "operation": "load",
                "diagnostic_kind": "tiled_resource_status",
            }
            if arg_count == 2:
                descriptor["drop_trailing_args"] = 1
                descriptor["dropped_parameters"] = ["status output"]
            return descriptor

        byte_address_interlocked_args = {
            "InterlockedCompareExchange": {4},
            "InterlockedCompareStore": {3},
        }.get(member, {2, 3})
        if (
            member in self.byte_address_interlocked_method_map
            and self.is_byte_address_buffer_type(resource_type)
            and arg_count in byte_address_interlocked_args
        ):
            if member == "InterlockedCompareExchange":
                byte_address_operation = "atomic_compare_exchange"
            elif member == "InterlockedCompareStore":
                byte_address_operation = "atomic_compare_store"
            elif member == "InterlockedAdd":
                byte_address_operation = "atomic_add"
            else:
                byte_address_operation = "atomic"
            return {
                "member": member,
                "function": self.byte_address_interlocked_method_map[member],
                "texture_function": None,
                "buffer_function": self.byte_address_interlocked_method_map[member],
                "component": None,
                "usage": None,
                "buffer_when_max_args": None,
                "resource": "buffer",
                "operation": byte_address_operation,
                "byte_address_atomic": True,
            }

        if member in self.buffer_method_map:
            return {
                "member": member,
                "function": self.buffer_method_map[member],
                "texture_function": None,
                "buffer_function": self.buffer_method_map[member],
                "component": None,
                "usage": None,
                "buffer_when_max_args": None,
                "resource": "buffer",
                "operation": (
                    {
                        "Store": "store",
                        "Append": "append",
                        "Consume": "consume",
                        "IncrementCounter": "increment_counter",
                        "DecrementCounter": "decrement_counter",
                        "Load": "load",
                        "GetDimensions": "dimensions",
                    }.get(member)
                ),
            }
        return None

    @staticmethod
    def templated_method_base(member):
        if isinstance(member, str) and "<" in member:
            return member.split("<", 1)[0]
        return member

    def resource_method_arguments(
        self, obj, member, rendered_args, descriptor, raw_args=None, is_main=False
    ):
        member = self.templated_method_base(member)
        drop_trailing_args = descriptor.get("drop_trailing_args", 0)
        if drop_trailing_args and member != "SampleCmpBias":
            rendered_args = rendered_args[:-drop_trailing_args]
            if raw_args is not None:
                raw_args = raw_args[:-drop_trailing_args]

        if member == "Load" and descriptor["function"] in {
            "texelFetch",
            "texelFetchOffset",
        }:
            return self.texture_load_method_arguments(
                obj, rendered_args, descriptor, raw_args or [], is_main
            )

        if (
            member == "SampleBias"
            and descriptor["function"] == "textureOffset"
            and len(rendered_args) >= 4
        ):
            sampler, coords, bias, offset = rendered_args[:4]
            method_args = [obj, sampler, coords, offset, bias]
        elif (
            member == "SampleBias"
            and descriptor["function"] == "textureOffsetClamp"
            and len(rendered_args) >= 5
        ):
            sampler, coords, bias, offset, clamp = rendered_args[:5]
            method_args = [obj, sampler, coords, offset, clamp, bias]
        elif (
            member == "SampleBias"
            and descriptor["function"] == "textureClamp"
            and len(rendered_args) >= 4
        ):
            sampler, coords, bias, clamp = rendered_args[:4]
            method_args = [obj, sampler, coords, clamp, bias]
        elif member == "SampleCmpLevelZero" and len(rendered_args) >= 3:
            # The explicit level 0 goes between the reference and the offset (textureCompareLod[Offset]).
            sampler, coords, compare = rendered_args[:3]
            method_args = [obj, sampler, coords, compare, "0.0", *rendered_args[3:]]
        elif member == "SampleCmpBias" and descriptor["function"] == "textureCompare":
            sampler, coords, compare = rendered_args[:3]
            method_args = [obj, sampler, coords, compare]
        elif (
            member == "SampleCmpBias"
            and descriptor["function"] == "textureCompareOffset"
            and len(rendered_args) >= 5
        ):
            sampler, coords, compare, _bias, offset = rendered_args[:5]
            method_args = [obj, sampler, coords, compare, offset]
        else:
            method_args = [obj, *rendered_args]

        if descriptor["component"] is not None:
            method_args.append(descriptor["component"])
        return method_args

    def byte_address_atomic_method_expression(
        self, obj, rendered_args, descriptor, raw_args=None, is_main=False
    ):
        if not descriptor.get("byte_address_atomic") or len(rendered_args) < 2:
            return None

        if descriptor.get("operation") == "atomic_compare_store":
            # InterlockedCompareStore is a compare-exchange whose previous value is not returned.
            offset, compare, value = rendered_args[:3]
            return f"{descriptor['function']}({obj}, {offset}, {compare}, {value})"

        if descriptor.get("operation") == "atomic_compare_exchange":
            if len(rendered_args) < 4:
                return None
            offset, compare, value = rendered_args[:3]
            atomic_call = (
                f"{descriptor['function']}({obj}, {offset}, {compare}, {value})"
            )
            original = self.generate_without_storage_index_lowering(
                (
                    (raw_args or [])[3]
                    if raw_args and len(raw_args) >= 4
                    else rendered_args[3]
                ),
                is_main,
            )
            return f"{original} = {atomic_call}"

        offset, value = rendered_args[:2]
        atomic_call = f"{descriptor['function']}({obj}, {offset}, {value})"
        if len(rendered_args) >= 3:
            original = self.generate_without_storage_index_lowering(
                (
                    (raw_args or [])[2]
                    if raw_args and len(raw_args) >= 3
                    else rendered_args[2]
                ),
                is_main,
            )
            return f"{original} = {atomic_call}"
        return atomic_call

    def texture_gather_compare_red_offsets_expression(self, obj, rendered_args):
        sampler, coord, compare, *offset_args = rendered_args[:7]
        components = ("x", "y", "z", "w")
        values = []
        for offset, component in zip(offset_args, components):
            gather = (
                f"textureGatherCompareOffset({obj}, {sampler}, {coord}, "
                f"{compare}, {offset})"
            )
            values.append(f"{gather}.{component}")
        return f"vec4({', '.join(values)})"

    def refine_texture_load_status_descriptor(self, member, args, descriptor):
        if (
            member != "Load"
            or descriptor.get("diagnostic_kind") != "tiled_resource_status"
        ):
            return descriptor

        resource_type = descriptor.get("resource_type")
        if self.is_rw_texture_type(resource_type):
            return descriptor

        resource_base = self.raw_type_base(resource_type)
        if not resource_base.startswith(("Texture", "FeedbackTexture")):
            return descriptor

        status_arg_count = 3 if self.is_multisample_texture_type(resource_type) else 2
        if len(args) != status_arg_count:
            return descriptor

        status_type = self.raw_type_base(self.expression_raw_type(args[-1]))
        if status_type != "uint":
            return descriptor

        descriptor = dict(descriptor)
        descriptor["function"] = descriptor["texture_function"]
        descriptor["drop_trailing_args"] = 1
        descriptor["dropped_parameters"] = ["status output"]
        return descriptor

    def texture_load_method_arguments(
        self, obj, rendered_args, descriptor, raw_args, is_main=False
    ):
        texel_buffer_args = self.texel_buffer_load_arguments(
            obj, rendered_args, descriptor
        )
        if texel_buffer_args is not None:
            return texel_buffer_args
        resource_base = self.raw_type_base(descriptor.get("resource_type"))
        is_multisample = resource_base in {"Texture2DMS", "Texture2DMSArray"}
        if is_multisample:
            if raw_args and rendered_args:
                rendered_args = [
                    self.integer_texel_operand(raw_args[0], rendered_args[0]),
                    *rendered_args[1:],
                ]
            return [obj, *rendered_args]

        split_location = None
        if raw_args:
            split_location = self.split_texture_load_location(
                raw_args[0], resource_base, is_main
            )

        if split_location:
            coord, lod = split_location
            if descriptor["function"] == "texelFetchOffset" and len(rendered_args) >= 2:
                return [obj, coord, lod, rendered_args[1]]
            return [obj, coord, lod]

        if descriptor["function"] == "texelFetchOffset":
            descriptor["function"] = "texelFetch"
        if descriptor["function"] == "texelFetch" and len(rendered_args) == 1:
            return [obj, rendered_args[0], "0"]
        return [obj, *rendered_args]

    def split_texture_load_location(self, location_arg, resource_base, is_main=False):
        coord_components = {
            "Texture1D": 1,
            "Texture1DArray": 2,
            "Texture2D": 2,
            "Texture2DArray": 3,
            "Texture3D": 3,
        }.get(resource_base)
        if coord_components is None:
            return None

        if not isinstance(location_arg, VectorConstructorNode):
            rendered_location = self.generate_expression(location_arg, is_main)
            component_count = self.numeric_vector_component_count(
                self.expression_value_raw_type(location_arg)
            )
            if component_count != coord_components + 1:
                return None

            swizzles = ("x", "y", "z", "w")
            lod = self.swizzle_rendered_expression(
                rendered_location, swizzles[coord_components]
            )
            if coord_components == 1:
                coord = self.swizzle_rendered_expression(rendered_location, "x")
            else:
                coord = self.swizzle_rendered_expression(
                    rendered_location, "".join(swizzles[:coord_components])
                )
            return (
                self.integer_texel_operand(location_arg, coord, coord_components),
                self.integer_texel_operand(location_arg, lod, 1),
            )

        ctor_args = getattr(location_arg, "args", None)
        if not ctor_args:
            return None

        if len(ctor_args) == 2 and coord_components > 1:
            first_components = self.numeric_vector_component_count(
                self.expression_value_raw_type(ctor_args[0])
            )
            second_components = self.numeric_vector_component_count(
                self.expression_value_raw_type(ctor_args[1])
            )
            if first_components != coord_components or second_components:
                return None
            return (
                self.integer_texel_operand(ctor_args[0], is_main=is_main),
                self.integer_texel_operand(ctor_args[1], is_main=is_main),
            )

        split_location = self.split_texture_load_constructor_arguments(
            ctor_args, coord_components, is_main
        )
        if split_location is not None:
            return split_location

        if len(ctor_args) != coord_components + 1:
            return None

        lod = self.integer_texel_operand(ctor_args[-1], is_main=is_main)
        coord_args = [self.generate_expression(arg, is_main) for arg in ctor_args[:-1]]
        if coord_components == 1:
            coord = self.integer_texel_operand(ctor_args[0], coord_args[0], 1)
        else:
            coord = f"ivec{coord_components}({', '.join(coord_args)})"
        return coord, lod

    def split_texture_load_constructor_arguments(
        self, ctor_args, coord_components, is_main=False
    ):
        coord_args = []
        consumed_components = 0
        lod_arg = None

        for arg in ctor_args:
            arg_components = (
                self.numeric_vector_component_count(self.expression_value_raw_type(arg))
                or 1
            )
            if consumed_components + arg_components <= coord_components:
                coord_args.append(self.generate_expression(arg, is_main))
                consumed_components += arg_components
                continue

            if consumed_components == coord_components and arg_components == 1:
                lod_arg = arg
                continue

            return None

        if consumed_components != coord_components or lod_arg is None:
            return None

        lod = self.integer_texel_operand(lod_arg, is_main=is_main)
        if coord_components == 1:
            coord = self.integer_texel_operand(ctor_args[0], coord_args[0], 1)
        else:
            coord = f"ivec{coord_components}({', '.join(coord_args)})"
        return coord, lod

    def resource_method_diagnostic(self, member, descriptor):
        """Refuse a resource method CrossGL cannot carry faithfully; return whether one was refused.

        A descriptor with a ``diagnostic_reason`` has no CrossGL helper at all, and one with ``dropped_parameters``
        would lose operands (an LOD clamp, a tiled-resource status output) that change the result.
        """
        diagnostic_reason = descriptor.get("diagnostic_reason")
        if diagnostic_reason:
            resource = self.raw_type_base(descriptor.get("resource_type")) or "resource"
            label = descriptor.get("diagnostic_label", "texture LOD query")
            self.refuse_import(f"DirectX {label} for {resource}: {diagnostic_reason}")
            return True
        dropped_parameters = descriptor.get("dropped_parameters")
        if not dropped_parameters:
            return False
        parameters = ", ".join(dropped_parameters)
        if descriptor.get("diagnostic_kind") == "tiled_resource_status":
            kind = "tiled-resource status"
        else:
            kind = "texture overload extras"
        self.refuse_import(
            f"DirectX {kind} for {member}: CrossGL has no operand for {parameters}"
        )
        return True

    def legacy_texture_function_call(
        self, func_name, raw_args, rendered_args, is_main=False
    ):
        expected_sampler_types = self.legacy_texture_function_sampler_types.get(
            func_name
        )
        if expected_sampler_types is None or not raw_args:
            return None

        rendered_args = list(rendered_args)
        sampler_type = self.raw_type_base(self.expression_raw_type(raw_args[0]))
        bound_texture = self.legacy_sampler_bound_texture(raw_args[0])
        if bound_texture is not None:
            sampler_type = self.legacy_texture_function_bound_sampler_type(func_name)
            rendered_args[0] = self.generate_expression(bound_texture, is_main)

        if sampler_type not in expected_sampler_types:
            return None

        if func_name in {"tex1D", "tex2D", "tex3D", "texCUBE"} and len(raw_args) == 2:
            return f"texture({', '.join(rendered_args)})"

        if func_name in {"tex2Dlod", "tex2Dbias"} and len(raw_args) == 2:
            coord, explicit_lod = self.legacy_texture_packed_2d_coord_and_w(
                raw_args[1], rendered_args[1], is_main
            )
            helper = "textureLod" if func_name == "tex2Dlod" else "texture"
            return f"{helper}({rendered_args[0]}, {coord}, {explicit_lod})"

        if func_name == "tex2Dgrad" and len(raw_args) == 4:
            return f"textureGrad({', '.join(rendered_args)})"

        return None

    def legacy_texture_function_bound_sampler_type(self, func_name):
        return {
            "tex1D": "sampler1D",
            "tex2D": "sampler2D",
            "tex2Dlod": "sampler2D",
            "tex2Dbias": "sampler2D",
            "tex2Dgrad": "sampler2D",
            "tex3D": "sampler3D",
            "texCUBE": "samplerCube",
        }.get(func_name)

    def legacy_sampler_bound_texture(self, expr):
        sampler_name = self.expression_base_name(expr)
        if not sampler_name:
            return None
        return self.legacy_sampler_bindings.get(sampler_name)

    def legacy_texture_packed_2d_coord_and_w(
        self, coord_arg, rendered_coord, is_main=False
    ):
        if isinstance(coord_arg, VectorConstructorNode):
            ctor_args = getattr(coord_arg, "args", []) or []
            if (
                len(ctor_args) >= 3
                and self.numeric_vector_component_count(
                    self.expression_value_raw_type(ctor_args[0])
                )
                == 2
            ):
                coord = self.generate_expression(ctor_args[0], is_main)
                explicit_lod = self.generate_expression(ctor_args[2], is_main)
                return coord, explicit_lod
            if len(ctor_args) >= 4:
                x = self.generate_expression(ctor_args[0], is_main)
                y = self.generate_expression(ctor_args[1], is_main)
                explicit_lod = self.generate_expression(ctor_args[3], is_main)
                return f"vec2({x}, {y})", explicit_lod

        return (
            self.swizzle_rendered_expression(rendered_coord, "xy"),
            self.swizzle_rendered_expression(rendered_coord, "w"),
        )

    def numeric_vector_component_count(self, type_name):
        base = self.canonical_composite_type(str(type_name or "").strip())
        match = re.fullmatch(
            r"(min16float|min10float|min16uint|min16int|min12int|float16_t|"
            r"uint16_t|int16_t|double|float|half|fixed|uint|int|bool)([1-4])",
            base,
        )
        return int(match.group(2)) if match else None

    def swizzle_rendered_expression(self, rendered, swizzle):
        if re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*",
            rendered,
        ):
            return f"{rendered}.{swizzle}"
        return f"({rendered}).{swizzle}"
