"""Precise and hyperbolic helper functions of the HLSL generator, wave shuffle helpers, union and fixed array declarations, generic struct declarations and relative-difference wrappers."""

import re

from ..ast import (
    ArrayNode,
    IdentifierNode,
    StructNode,
)
from ..math_namespaces import (
    METAL_FREE_ARITHMETIC_OPERATOR_PREFIXES,
    METAL_MATH_NAMESPACE_ANNOTATION,
)
from ..refusals import DirectXRefusal
from ..semantics.hlsl_geometry_streams import EMPTY_STREAM_VERTEX_ATTRIBUTE
from ..semantics.hlsl_ray_payloads import (
    PAYLOAD_ACCESS_ATTRIBUTES,
    PAYLOAD_STAGES,
    RAY_PAYLOAD_ATTRIBUTE,
    HlslRayPayloadUnsupportedError,
)
from ..semantics.unsized_arrays import UnsizedArrayUnsupportedError
from .array_utils import (
    format_c_style_array_declaration,
    split_array_type_suffix,
)
from .directx_values import DirectXUnionLayoutError
from .enum_utils import (
    build_generic_enum_specialization,
    enum_struct_fields,
    generic_enum_specialized_fields,
    generic_type_parts,
)
from .generic_struct_utils import (
    build_generic_struct_specialization,
    generic_struct_specialized_fields,
)
from .hlsl_bfloat16_vectors import hlsl_bfloat16_vector_helper_sources
from .hlsl_vk_locations import hlsl_vk_location_prefix
from .union_lanes import hlsl_lane_helper_definitions
from .union_layout import UnionLayoutCollector


class DirectXPreciseHelperMixin:
    """Precise helper members of ``HLSLCodeGen``."""

    def hlsl_inverse_hyperbolic_wrapper_helper(self, operation, value_type):
        helper_name = self.hlsl_inverse_hyperbolic_helper_name(operation, value_type)
        scalar_helper = self.hlsl_inverse_hyperbolic_helper_name(operation, "float")
        match = re.fullmatch(r"(float16_t|float|half|min16float)([234]?)", value_type)
        if match is None or value_type == "float":
            raise DirectXRefusal(
                f"Unsupported inverse-hyperbolic helper type: {value_type}"
            )
        scalar_type, lane_text = match.groups()
        lanes = int(lane_text) if lane_text else 1
        if lanes == 1:
            return (
                f"{value_type} {helper_name}({value_type} value) {{\n"
                f"    return ({value_type}){scalar_helper}((float)value);\n"
                "}\n"
            )

        components = "xyzw"[:lanes]
        arguments = ",\n        ".join(
            f"({scalar_type}){scalar_helper}((float)value.{component})"
            for component in components
        )
        return (
            f"{value_type} {helper_name}({value_type} value) {{\n"
            f"    return {value_type}(\n        {arguments}\n    );\n"
            "}\n"
        )

    def generate_hlsl_inverse_hyperbolic_helpers(self):
        required_helpers = getattr(
            self, "required_hlsl_inverse_hyperbolic_helpers", set()
        )
        if not required_helpers:
            return ""

        operations = sorted(operation for operation, _value_type in required_helpers)
        helpers = [self.hlsl_inverse_hyperbolic_log1p_helper()]
        helpers.extend(
            self.hlsl_inverse_hyperbolic_scalar_helper(operation)
            for operation in sorted(set(operations))
        )
        helpers.extend(
            self.hlsl_inverse_hyperbolic_wrapper_helper(operation, value_type)
            for operation, value_type in sorted(required_helpers)
            if value_type != "float"
        )
        return "".join(helpers) + "\n"

    # Metal's precise:: sinh and tanh are odd, IEEE-exact at zeros, NaN and
    # saturation.  DXIL Hsin and Htan are not on every driver: WARP returns +0
    # for sinh(-0) and tanh(-0), NaN for tanh of operands whose exp(2|x|)
    # overflows, and 1 - 2^-24 for tanh(20).
    HLSL_PRECISE_HYPERBOLIC_NAMES = frozenset({"sinh", "tanh"})
    HLSL_PRECISE_HYPERBOLIC_TYPE = re.compile(
        r"(float16_t|float|half|min16float)([234]?)"
    )

    def hlsl_precise_helper_reserved_names(self):
        used_names = set(getattr(self, "function_return_types", {}))
        used_names.update(getattr(self, "global_variable_types", {}))
        used_names.update(getattr(self, "local_variable_types", {}))
        used_names.update(getattr(self, "structs_by_name", {}))
        used_names.update(self.hlsl_trailing_zero_helper_names.values())
        used_names.update(self.hlsl_inverse_hyperbolic_helper_names.values())
        used_names.update(self.hlsl_precise_hyperbolic_helper_names.values())
        used_names.update(self.hlsl_precise_difference_helper_names.values())
        return used_names

    def hlsl_precise_hyperbolic_helper_name(self, operation, value_type):
        key = (operation, value_type)
        helper_name = self.hlsl_precise_hyperbolic_helper_names.get(key)
        if helper_name is None:
            used_names = self.hlsl_precise_helper_reserved_names()
            helper_name = f"__crossgl_precise_{operation}_{value_type}"
            while helper_name in used_names:
                helper_name += "_"
            self.hlsl_precise_hyperbolic_helper_names[key] = helper_name
        return helper_name

    def generate_hlsl_precise_hyperbolic_call(self, func_name, args, *, call_node=None):
        """Lower Metal ``precise::sinh``/``tanh`` to sign- and NaN-exact helpers."""
        if func_name not in self.HLSL_PRECISE_HYPERBOLIC_NAMES or len(args) != 1:
            return None
        get_annotation = getattr(call_node, "get_annotation", None)
        if not callable(get_annotation) or (
            get_annotation(METAL_MATH_NAMESPACE_ANNOTATION) != "precise"
        ):
            return None
        if func_name in self.global_variable_types or (
            func_name in self.function_return_types
            and func_name not in self.hlsl_function_name_aliases
        ):
            return None
        if (
            self.resolve_hlsl_function_overload(func_name, args, call_node=call_node)
            is not None
        ):
            return None
        source_type = self.hlsl_source_expression_type(args[0])
        if source_type is None:
            return None
        if self.is_hlsl_bfloat16_type(source_type):
            self.required_hlsl_precise_hyperbolic_helpers.add((func_name, "float"))
            helper_name = self.hlsl_precise_hyperbolic_helper_name(func_name, "float")
            decoded = self.hlsl_bfloat16_to_float_expression(
                self.generate_expression(args[0])
            )
            return self.hlsl_float_to_bfloat16_expression(f"{helper_name}({decoded})")
        mapped_type = self.map_type(source_type)
        if self.HLSL_PRECISE_HYPERBOLIC_TYPE.fullmatch(mapped_type or "") is None:
            return None
        self.required_hlsl_precise_hyperbolic_helpers.add((func_name, mapped_type))
        helper_name = self.hlsl_precise_hyperbolic_helper_name(func_name, mapped_type)
        argument = self.generate_expression_with_expected(args[0], mapped_type)
        return f"{helper_name}({argument})"

    def hlsl_precise_hyperbolic_scalar_helper(self, operation):
        helper_name = self.hlsl_precise_hyperbolic_helper_name(operation, "float")
        if operation == "sinh":
            magnitude = "asuint(sinh(asfloat(magnitudeBits))) & 0x7fffffffu"
            comment = ""
        else:
            magnitude = (
                "magnitudeBits >= 0x41102cb4u\n"
                "        ? 0x3f800000u\n"
                "        : asuint(tanh(asfloat(magnitudeBits))) & 0x7fffffffu"
            )
            comment = (
                "    // Binary32 tanh rounds to one from |x| = 9.0109138; Htan's\n"
                "    // exp(2|x|) can overflow first.\n"
            )
        # Integer tests are immune to DXC's fast-math folding. NaN keeps its
        # payload; below 2^-12 both functions round to the operand itself.
        return (
            f"float {helper_name}(float value) {{\n"
            "    uint bits = asuint(value);\n"
            "    uint magnitudeBits = bits & 0x7fffffffu;\n"
            "    if (magnitudeBits > 0x7f800000u || magnitudeBits < 0x39800000u) {\n"
            "        return value;\n"
            "    }\n"
            f"{comment}"
            f"    uint resultBits = {magnitude};\n"
            "    return asfloat(resultBits | (bits & 0x80000000u));\n"
            "}\n"
        )

    def hlsl_precise_hyperbolic_wrapper_helper(self, operation, value_type):
        helper_name = self.hlsl_precise_hyperbolic_helper_name(operation, value_type)
        scalar_helper = self.hlsl_precise_hyperbolic_helper_name(operation, "float")
        scalar_type, lane_text = self.HLSL_PRECISE_HYPERBOLIC_TYPE.fullmatch(
            value_type
        ).groups()

        def narrowed(operand):
            result = f"{scalar_helper}((float){operand})"
            # Metal rounds a binary32 sinh beyond 65520 to infinity, but a
            # native float16_t conversion clamps it to 65504 on WARP.  tanh's
            # magnitude never exceeds one, so its native narrowing is exact.
            if operation == "sinh" and scalar_type == "float16_t":
                return self.hlsl_half_conversion_expression(result, 1, ("float",))
            return f"({scalar_type}){result}"

        # Metal's precise:: overloads are binary32: narrow operands widen.
        if not lane_text:
            return (
                f"{value_type} {helper_name}({value_type} value) {{\n"
                f"    return {narrowed('value')};\n"
                "}\n"
            )
        arguments = ",\n        ".join(
            narrowed(f"value.{component}") for component in "xyzw"[: int(lane_text)]
        )
        return (
            f"{value_type} {helper_name}({value_type} value) {{\n"
            f"    return {value_type}(\n        {arguments}\n    );\n"
            "}\n"
        )

    def generate_hlsl_precise_hyperbolic_helpers(self):
        required_helpers = getattr(
            self, "required_hlsl_precise_hyperbolic_helpers", set()
        )
        if not required_helpers:
            return ""
        operations = sorted({operation for operation, _value_type in required_helpers})
        helpers = [
            self.hlsl_precise_hyperbolic_scalar_helper(operation)
            for operation in operations
        ]
        helpers.extend(
            self.hlsl_precise_hyperbolic_wrapper_helper(operation, value_type)
            for operation, value_type in sorted(required_helpers)
            if value_type != "float"
        )
        return "".join(helpers) + "\n"

    def hlsl_precise_difference_call(self, expression, left, right):
        """Keep ``x - x`` exact: optimized DXC folds it to 0, hiding inf and NaN."""
        if not (
            isinstance(expression.left, IdentifierNode)
            and isinstance(expression.right, IdentifierNode)
            and expression.left.name == expression.right.name
            and left == right
        ):
            return None
        source_type = self.hlsl_source_expression_type(expression.left)
        if source_type is None or any(
            marker in (self.type_name_string(source_type) or "")
            for marker in ("*", "&")
        ):
            return None
        info = self.hlsl_floating_arithmetic_type_info(source_type)
        if info is None or info["base_type"] not in {"float", "float16_t"}:
            return None
        value_type = info["mapped_type"]
        helper_name = self.hlsl_precise_difference_helper_names.get(value_type)
        if helper_name is None:
            used_names = self.hlsl_precise_helper_reserved_names()
            helper_name = f"__crossgl_precise_difference_{value_type}"
            while helper_name in used_names:
                helper_name += "_"
            self.hlsl_precise_difference_helper_names[value_type] = helper_name
        self.required_hlsl_precise_difference_helpers.add(value_type)
        return f"{helper_name}({left}, {right})"

    def hlsl_is_metal_free_arithmetic_operator(self, func, return_type):
        """Return whether ``func`` is a Metal free arithmetic operator on floats.

        DXC compiles HLSL with fast-math.  Once such a helper is inlined it
        folds the zero components of complex constants (ArcTan's ``i`` and
        ``1 / 2i``) out of the products, so signed zeros differ from Metal's
        ``-fno-fast-math`` build.  A precise result makes DXC compute it, and
        every operation it depends on, with IEEE semantics.
        """
        name = str(getattr(func, "name", "") or "")
        if not name.startswith(METAL_FREE_ARITHMETIC_OPERATOR_PREFIXES):
            return False
        fields = self.hlsl_struct_constructor_fields(return_type)
        member_types = (
            [field_type for _field_name, field_type in fields]
            if fields is not None
            else [return_type]
        )
        return bool(member_types) and all(
            (info := self.hlsl_floating_arithmetic_type_info(member_type)) is not None
            and info["base_type"] in {"float", "float16_t", "half"}
            for member_type in member_types
        )

    def generate_hlsl_precise_difference_helpers(self):
        required_types = getattr(
            self, "required_hlsl_precise_difference_helpers", set()
        )
        helpers = []
        for value_type in sorted(required_types):
            helper_name = self.hlsl_precise_difference_helper_names[value_type]
            helpers.append(
                f"{value_type} {helper_name}({value_type} left, {value_type} right) {{\n"
                f"    precise {value_type} result = left - right;\n"
                "    return result;\n"
                "}\n"
            )
        return "".join(helpers) + ("\n" if helpers else "")

    def hlsl_defined_relative_wave_shuffle_helper_name(self, operation, value_type):
        mapped_value_type = self.map_type(value_type)
        key = (operation, mapped_value_type)
        existing = self.hlsl_defined_relative_wave_shuffle_helper_names.get(key)
        if existing is not None:
            return existing

        operation_suffix = {
            "WaveShuffleDown": "down",
            "WaveShuffleUp": "up",
            "WaveShuffleXor": "xor",
        }[operation]
        type_suffix = re.sub(r"[^A-Za-z0-9_]", "_", mapped_value_type)
        base_name = "__crossgl_wave_shuffle_" f"{operation_suffix}_self_{type_suffix}"
        reserved_names = set(self.function_return_types)
        reserved_names.update(self.global_variable_types)
        reserved_names.update(self.structs_by_name)
        reserved_names.update(
            self.hlsl_defined_relative_wave_shuffle_helper_names.values()
        )
        helper_name = base_name
        suffix = 1
        while helper_name in reserved_names:
            helper_name = f"{base_name}_{suffix}"
            suffix += 1
        self.hlsl_defined_relative_wave_shuffle_helper_names[key] = helper_name
        return helper_name

    def hlsl_defined_relative_wave_shuffle_call(
        self, operation, value_type, value_expression, delta_expression
    ):
        mapped_value_type = self.map_type(value_type)
        key = (operation, mapped_value_type)
        self.required_hlsl_defined_relative_wave_shuffle_helpers.add(key)
        if self.is_hlsl_complex64_type(mapped_value_type):
            self.required_hlsl_complex64_helpers.add(
                "__crossgl_complex64_wave_read_lane_at"
            )
        helper_name = self.hlsl_defined_relative_wave_shuffle_helper_name(
            operation, mapped_value_type
        )
        return f"{helper_name}({value_expression}, " f"uint({delta_expression}))"

    def generate_hlsl_defined_relative_wave_shuffle_helpers(self):
        code = ""
        for operation, value_type in sorted(
            getattr(
                self,
                "required_hlsl_defined_relative_wave_shuffle_helpers",
                set(),
            )
        ):
            helper_name = self.hlsl_defined_relative_wave_shuffle_helper_name(
                operation, value_type
            )
            if operation == "WaveShuffleDown":
                valid_expression = "delta < (laneCount - lane)"
                source_expression = "lane + delta"
            elif operation == "WaveShuffleUp":
                valid_expression = "delta <= lane"
                source_expression = "lane - delta"
            elif operation == "WaveShuffleXor":
                valid_expression = "(lane ^ delta) < laneCount"
                source_expression = "lane ^ delta"
            else:  # pragma: no cover - registration is closed over known operators.
                raise DirectXRefusal(
                    f"Unsupported relative wave shuffle operation: {operation}"
                )

            code += (
                f"{value_type} {helper_name}({value_type} value, uint delta) {{\n"
                "    uint lane = WaveGetLaneIndex();\n"
                "    uint laneCount = WaveGetLaneCount();\n"
                f"    bool valid = {valid_expression};\n"
                "    uint sourceLane = valid\n"
                f"        ? ({source_expression})\n"
                "        : lane;\n"
            )

            if self.hlsl_result_component_type(value_type) == "bool":
                component_count = self.value_component_count(value_type) or 1
                payload_type = (
                    "uint" if component_count == 1 else f"uint{component_count}"
                )
                if component_count == 1:
                    payload = "value ? 1u : 0u"
                    zero = "0u"
                else:
                    fields = "xyzw"[:component_count]
                    payload = (
                        f"{payload_type}("
                        + ", ".join(f"value.{field} ? 1u : 0u" for field in fields)
                        + ")"
                    )
                    zero = f"{payload_type}({', '.join(['0u'] * component_count)})"
                code += (
                    f"    {payload_type} payload = {payload};\n"
                    f"    {payload_type} shuffledBits = "
                    "WaveReadLaneAt(payload, sourceLane);\n"
                    f"    {value_type} shuffled = (shuffledBits != {zero});\n"
                )
            elif self.is_hlsl_complex64_type(value_type):
                code += (
                    f"    {value_type} shuffled = "
                    "__crossgl_complex64_wave_read_lane_at(value, sourceLane);\n"
                )
            else:
                code += (
                    f"    {value_type} shuffled = "
                    "WaveReadLaneAt(value, sourceLane);\n"
                )

            # The selected source is always in range, and selecting the calling
            # lane makes the wave read itself produce the requested fallback.
            # Return it directly so DXC keeps one wave operation with every lane
            # active instead of splitting the intrinsic across divergent branches.
            code += "    return shuffled;\n}\n\n"

        return code

    def hlsl_wave_shuffle_and_fill_up_helper_name(self, value_type):
        type_suffix = re.sub(r"[^A-Za-z0-9_]", "_", self.map_type(value_type))
        return f"__crossgl_wave_shuffle_and_fill_up_{type_suffix}"

    def generate_hlsl_wave_shuffle_and_fill_up_helpers(self):
        code = ""
        for value_type in sorted(
            getattr(self, "required_hlsl_wave_shuffle_and_fill_up_types", set())
        ):
            helper_name = self.hlsl_wave_shuffle_and_fill_up_helper_name(value_type)
            code += (
                f"{value_type} {helper_name}("
                f"{value_type} value, {value_type} fill, uint delta, "
                "uint modulo) {\n"
                "    uint lane = WaveGetLaneIndex();\n"
                "    uint segmentLane = lane % modulo;\n"
                "    uint segmentBase = lane - segmentLane;\n"
                "    bool useValue = segmentLane >= delta;\n"
                "    uint valueSourceLane = useValue ? (lane - delta) : lane;\n"
                "    uint fillSourceLane = useValue\n"
                "        ? lane\n"
                "        : (segmentBase + modulo + segmentLane - delta);\n"
            )

            if self.hlsl_result_component_type(value_type) == "bool":
                component_count = self.value_component_count(value_type) or 1
                payload_type = (
                    "uint" if component_count == 1 else f"uint{component_count}"
                )
                if component_count == 1:
                    payload = "value ? 1u : 0u"
                    fill_payload = "fill ? 1u : 0u"
                    zero = "0u"
                else:
                    fields = "xyzw"[:component_count]
                    payload = (
                        f"{payload_type}("
                        + ", ".join(f"value.{field} ? 1u : 0u" for field in fields)
                        + ")"
                    )
                    fill_payload = (
                        f"{payload_type}("
                        + ", ".join(f"fill.{field} ? 1u : 0u" for field in fields)
                        + ")"
                    )
                    zero = f"{payload_type}({', '.join(['0u'] * component_count)})"
                code += (
                    f"    {payload_type} payload = {payload};\n"
                    f"    {payload_type} fillPayload = {fill_payload};\n"
                    f"    {payload_type} shuffledBits = "
                    "WaveReadLaneAt(payload, valueSourceLane);\n"
                    f"    {payload_type} shuffledFillBits = "
                    "WaveReadLaneAt(fillPayload, fillSourceLane);\n"
                    f"    {value_type} shuffled = (shuffledBits != {zero});\n"
                    f"    {value_type} shuffledFill = "
                    f"(shuffledFillBits != {zero});\n"
                )
            else:
                code += (
                    f"    {value_type} shuffled = "
                    "WaveReadLaneAt(value, valueSourceLane);\n"
                    f"    {value_type} shuffledFill = "
                    "WaveReadLaneAt(fill, fillSourceLane);\n"
                )

            code += "    return useValue ? shuffled : shuffledFill;\n}\n\n"

        return code

    def generate_hlsl_bfloat16_helpers(self):
        required = getattr(self, "required_hlsl_bfloat16_helpers", set())
        requires_storage = getattr(self, "requires_hlsl_bfloat16_storage", False)
        requires_native_16_bit_types = getattr(
            self, "requires_hlsl_native_16_bit_types", False
        )
        if not required and not requires_storage:
            return ""
        native_16_bit_mode = requires_storage or requires_native_16_bit_types
        uint16_type = "uint16_t" if native_16_bit_mode else "min16uint"

        code = [
            "// CrossGL exact bfloat16 lowering: register payloads use uint low bits.\n"
        ]
        if native_16_bit_mode:
            code.append(
                "// Native 16-bit types require Shader Model 6.2 and "
                "dxc -enable-16bit-types.\n"
            )

        helper_sources = {
            "from_uint": (
                """
uint __crossgl_bfloat16_from_uint(uint value) {
    if (value < 256u) { return asuint(float(value)) >> 16u; }
    uint leading = uint(firstbithigh(value));
    uint shift = leading - 7u;
    uint retained = value >> shift;
    uint remainder = value & ((1u << shift) - 1u);
    uint midpoint = 1u << (shift - 1u);
    if (remainder > midpoint || (remainder == midpoint && (retained & 1u) != 0u)) {
        retained += 1u;
    }
    return ((leading + 126u) << 7u) + retained;
}
"""
            ),
            "from_int": (
                """
uint __crossgl_bfloat16_from_uint(uint value);
uint __crossgl_bfloat16_from_int(int value) {
    uint magnitude = value < 0 ? 0u - uint(value) : uint(value);
    return __crossgl_bfloat16_from_uint(magnitude) | (value < 0 ? 0x8000u : 0u);
}
"""
            ),
            "from_uint64": (
                """
uint __crossgl_bfloat16_from_uint64(uint64_t value) {
    if (value < uint64_t(256)) { return asuint(float(uint(value))) >> 16u; }
    uint high = uint(value >> 32u);
    uint leading = high != 0u ? 32u + uint(firstbithigh(high)) : uint(firstbithigh(uint(value)));
    uint shift = leading - 7u;
    uint retained = uint(value >> shift);
    uint64_t remainder = value & ((uint64_t(1) << shift) - uint64_t(1));
    uint64_t midpoint = uint64_t(1) << (shift - 1u);
    if (remainder > midpoint || (remainder == midpoint && (retained & 1u) != 0u)) {
        retained += 1u;
    }
    return ((leading + 126u) << 7u) + retained;
}
"""
            ),
            "from_int64": (
                """
uint __crossgl_bfloat16_from_uint64(uint64_t value);
uint __crossgl_bfloat16_from_int64(int64_t value) {
    uint64_t magnitude = value < int64_t(0) ? uint64_t(0) - uint64_t(value) : uint64_t(value);
    return __crossgl_bfloat16_from_uint64(magnitude) | (value < int64_t(0) ? 0x8000u : 0u);
}
"""
            ),
            "from_float": (
                """
uint __crossgl_bfloat16_from_float(float value) {
    uint bits = asuint(value);
    uint upperBits = bits >> 16u;
    uint absoluteBits = bits & 0x7fffffffu;
    if (absoluteBits > 0x7f800000u) {
        return (upperBits | 0x40u) & 0xffffu;
    }
    uint roundingBias = 0x7fffu + (upperBits & 1u);
    return ((bits + roundingBias) >> 16u) & 0xffffu;
}
"""
            ),
            "to_float": (
                """
float __crossgl_bfloat16_to_float(uint value) {
    return asfloat((value & 0xffffu) << 16u);
}
"""
            ),
            "to_float_vectors": (
                """
float2 __crossgl_bfloat16_to_float(uint2 value) {
    return asfloat((value & 0xffffu) << 16u);
}
float3 __crossgl_bfloat16_to_float(uint3 value) {
    return asfloat((value & 0xffffu) << 16u);
}
float4 __crossgl_bfloat16_to_float(uint4 value) {
    return asfloat((value & 0xffffu) << 16u);
}
"""
            ),
            "from_uint16": (
                f"""
uint __crossgl_bfloat16_from_uint16({uint16_type} value) {{
    return uint(value) & 0xffffu;
}}
"""
            ),
            "to_uint16": (
                f"""
{uint16_type} __crossgl_bfloat16_to_uint16(uint value) {{
    return {uint16_type}(value & 0xffffu);
}}
"""
            ),
        }
        helper_sources.update(hlsl_bfloat16_vector_helper_sources())
        code.extend(helper_sources[name] for name in sorted(required))
        return "".join(code) + "\n"

    def generate_hlsl_explicit_bitcast_helpers(self):
        helper_sources = {
            "uint_to_uint16x2": (
                "uint16_t2 __crossgl_bitcast_uint_to_uint16x2(uint value) {\n"
                "    return uint16_t2(uint16_t(value & 65535u), uint16_t(value >> 16u));\n"
                "}\n"
            ),
            "uint16x2_to_uint": (
                "uint __crossgl_bitcast_uint16x2_to_uint(uint16_t2 value) {\n"
                "    return uint(value.x) | (uint(value.y) << 16u);\n"
                "}\n"
            ),
            "binary16_to_float": (
                """
float __crossgl_binary16_to_float(uint value) {
    uint bits = value & 0xffffu;
    uint sign = (bits & 0x8000u) << 16u;
    uint exponent = bits & 0x7c00u;
    uint mantissa = bits & 0x03ffu;
    if (exponent == 0u) {
        if (mantissa == 0u) {
            return asfloat(sign);
        }
        float magnitude = float(mantissa) * 5.9604644775390625e-8f;
        return asfloat(asuint(magnitude) | sign);
    }
    if (exponent == 0x7c00u) {
        return asfloat(sign | 0x7f800000u | (mantissa << 13u));
    }
    uint floatExponent = ((exponent >> 10u) + 112u) << 23u;
    return asfloat(sign | floatExponent | (mantissa << 13u));
}

float2 __crossgl_binary16_to_float(uint2 value) {
    return float2(
        __crossgl_binary16_to_float(value.x),
        __crossgl_binary16_to_float(value.y)
    );
}

float3 __crossgl_binary16_to_float(uint3 value) {
    return float3(
        __crossgl_binary16_to_float(value.x),
        __crossgl_binary16_to_float(value.y),
        __crossgl_binary16_to_float(value.z)
    );
}

float4 __crossgl_binary16_to_float(uint4 value) {
    return float4(
        __crossgl_binary16_to_float(value.x),
        __crossgl_binary16_to_float(value.y),
        __crossgl_binary16_to_float(value.z),
        __crossgl_binary16_to_float(value.w)
    );
}
"""
            ),
            "double_to_uint2": (
                """
uint2 __crossgl_bitcast_double_to_uint2(double value) {
    uint lowBits;
    uint highBits;
    asuint(value, lowBits, highBits);
    return uint2(lowBits, highBits);
}
"""
            ),
            "uint2_to_double": (
                """
double __crossgl_bitcast_uint2_to_double(uint2 value) {
    return asdouble(value.x, value.y);
}
"""
            ),
            "uint64_to_uint2": (
                """
uint2 __crossgl_bitcast_uint64_to_uint2(uint64_t value) {
    return uint2(uint(value), uint(value >> 32));
}
"""
            ),
            "uint2_to_uint64": (
                """
uint64_t __crossgl_bitcast_uint2_to_uint64(uint2 value) {
    return (uint64_t(value.y) << 32) | uint64_t(value.x);
}
"""
            ),
        }
        helpers = "".join(
            helper_sources[helper]
            for helper in sorted(self.required_hlsl_explicit_bitcast_helpers)
        )
        return f"{helpers}\n" if helpers else ""

    def generate_hlsl_union_storage_helpers(self):
        member_kinds = {
            member["kind"]
            for layout in self.hlsl_union_layouts.values()
            for member in layout["members"].values()
        }
        code = ""
        if member_kinds & {"u8x4", "u8x4_array"}:
            code += """
uint4 __crossgl_union_unpack_u8x4(uint word) {
    return uint4(
        word & 0xffu,
        (word >> 8u) & 0xffu,
        (word >> 16u) & 0xffu,
        (word >> 24u) & 0xffu
    );
}

uint __crossgl_union_pack_u8x4(uint4 value) {
    return (value.x & 0xffu)
        | ((value.y & 0xffu) << 8u)
        | ((value.z & 0xffu) << 16u)
        | ((value.w & 0xffu) << 24u);
}

uint __crossgl_union_set_u8x4_lane(
    inout uint word,
    uint lane,
    uint selected
) {
    uint shift = (lane & 3u) * 8u;
    uint mask = 0xffu << shift;
    word = (word & ~mask) | ((selected & 0xffu) << shift);
    return selected & 0xffu;
}

"""
        if member_kinds & {"bool8x4", "bool8x4_array"}:
            code += """
bool4 __crossgl_union_unpack_bool4(uint word) {
    return bool4(
        (word & 0xffu) != 0u,
        ((word >> 8u) & 0xffu) != 0u,
        ((word >> 16u) & 0xffu) != 0u,
        ((word >> 24u) & 0xffu) != 0u
    );
}

uint __crossgl_union_pack_bool4(bool4 value) {
    return (value.x ? 1u : 0u)
        | ((value.y ? 1u : 0u) << 8u)
        | ((value.z ? 1u : 0u) << 16u)
        | ((value.w ? 1u : 0u) << 24u);
}

bool __crossgl_union_set_bool4_lane(
    inout uint word,
    uint lane,
    bool selected
) {
    uint shift = (lane & 3u) * 8u;
    uint mask = 0xffu << shift;
    uint encoded = selected ? 1u : 0u;
    word = (word & ~mask) | (encoded << shift);
    return selected;
}

"""
        return code + hlsl_lane_helper_definitions(self)

    def generate_hlsl_complex64_helpers(self):
        if not getattr(self, "required_hlsl_complex64_helpers", set()):
            return ""
        code = """
complex64_t __crossgl_complex64_make(float real, float imag) {
    complex64_t result;
    result.real = real;
    result.imag = imag;
    return result;
}

complex64_t __crossgl_complex64_negate(complex64_t value) {
    return __crossgl_complex64_make(-value.real, -value.imag);
}

complex64_t __crossgl_complex64_add(complex64_t left, complex64_t right) {
    return __crossgl_complex64_make(left.real + right.real, left.imag + right.imag);
}

complex64_t __crossgl_complex64_sub(complex64_t left, complex64_t right) {
    return __crossgl_complex64_make(left.real - right.real, left.imag - right.imag);
}

complex64_t __crossgl_complex64_mul(complex64_t left, complex64_t right) {
    return __crossgl_complex64_make(
        (left.real * right.real) - (left.imag * right.imag),
        (left.real * right.imag) + (left.imag * right.real)
    );
}

complex64_t __crossgl_complex64_div(complex64_t left, complex64_t right) {
    float denom = (right.real * right.real) + (right.imag * right.imag);
    return __crossgl_complex64_make(
        ((left.real * right.real) + (left.imag * right.imag)) / denom,
        ((left.imag * right.real) - (left.real * right.imag)) / denom
    );
}

float __crossgl_complex64_mod_component(float left, float right) {
    float value = left - (right * trunc(left / right));
    if ((value != 0.0) && ((value < 0.0) != (right < 0.0))) {
        value += right;
    }
    return value;
}

complex64_t __crossgl_complex64_mod(complex64_t left, complex64_t right) {
    return __crossgl_complex64_make(
        __crossgl_complex64_mod_component(left.real, right.real),
        __crossgl_complex64_mod_component(left.imag, right.imag)
    );
}

bool __crossgl_complex64_equal(complex64_t left, complex64_t right) {
    return (left.real == right.real) && (left.imag == right.imag);
}

bool __crossgl_complex64_not_equal(complex64_t left, complex64_t right) {
    return !__crossgl_complex64_equal(left, right);
}

bool __crossgl_complex64_greater(complex64_t left, complex64_t right) {
    return (left.real > right.real)
        || ((left.real == right.real) && (left.imag > right.imag));
}

bool __crossgl_complex64_greater_equal(complex64_t left, complex64_t right) {
    return (left.real > right.real)
        || ((left.real == right.real) && (left.imag >= right.imag));
}

bool __crossgl_complex64_less(complex64_t left, complex64_t right) {
    return __crossgl_complex64_greater(right, left);
}

bool __crossgl_complex64_less_equal(complex64_t left, complex64_t right) {
    return __crossgl_complex64_greater_equal(right, left);
}

"""
        if "__crossgl_complex64_wave_read_lane_at" in (
            self.required_hlsl_complex64_helpers
        ):
            code += """
complex64_t __crossgl_complex64_wave_read_lane_at(
    complex64_t value,
    uint sourceLane
) {
    return __crossgl_complex64_make(
        WaveReadLaneAt(value.real, sourceLane),
        WaveReadLaneAt(value.imag, sourceLane)
    );
}

"""
        if "__crossgl_complex64_wave_shuffle_and_fill_up" in (
            self.required_hlsl_complex64_helpers
        ):
            code += """
complex64_t __crossgl_complex64_wave_shuffle_and_fill_up(
    complex64_t value,
    complex64_t fill,
    uint delta
) {
    uint lane = WaveGetLaneIndex();
    bool useValue = lane >= delta;
    uint valueLane = useValue ? lane - delta : lane;
    uint fillLane = useValue ? lane : WaveGetLaneCount() + lane - delta;
    complex64_t valueResult = __crossgl_complex64_wave_read_lane_at(value, valueLane);
    complex64_t fillResult = __crossgl_complex64_wave_read_lane_at(fill, fillLane);
    if (useValue) {
        return valueResult;
    }
    return fillResult;
}

"""
        return code

    def generate_hlsl_fixed_array_return_helpers(self):
        code = ""
        for struct_name in sorted(self.required_hlsl_fixed_array_read_helpers):
            metadata = self.hlsl_fixed_array_return_struct_metadata.get(struct_name)
            if metadata is None:
                continue
            base_type = self.map_type(metadata["base_type"])
            helper_name = self.hlsl_fixed_array_read_helper_name(struct_name)
            code += f"{base_type} {helper_name}({struct_name} value, int index) {{\n"
            for index in range(metadata["size"] - 1):
                field_name = self.hlsl_fixed_array_return_field_name(index)
                code += f"    if (index == {index}) {{\n"
                code += f"        return value.{field_name};\n"
                code += "    }\n"
            last_field = self.hlsl_fixed_array_return_field_name(metadata["size"] - 1)
            code += f"    return value.{last_field};\n"
            code += "}\n\n"
        return code

    def generate_hlsl_fragment_shading_rate_helper(self):
        if not self.required_hlsl_fragment_shading_rate_helper:
            return ""
        return (
            "uint2 CrossGLFragmentSizeFromShadingRate(uint shadingRate) {\n"
            "    uint width = 1u << ((shadingRate >> 2u) & 3u);\n"
            "    uint height = 1u << (shadingRate & 3u);\n"
            "    return uint2(width, height);\n"
            "}\n\n"
        )

    def generate_hlsl_ordered_struct_declarations(self, structs):
        ordered_structs = list(structs or [])
        known_names = {
            getattr(struct, "name", None)
            for struct in ordered_structs
            if getattr(struct, "name", None)
        }
        for name, struct in self.hlsl_fixed_array_return_struct_nodes.items():
            if name not in known_names:
                ordered_structs.append(struct)
                known_names.add(name)
        records = self.hlsl_struct_declaration_records(ordered_structs)
        ordered_records = self.hlsl_order_struct_declaration_records(records)
        return "".join(record["code"] for record in ordered_records)

    def collect_hlsl_generic_specializations_from_function_types(self, functions):
        visited = set()
        for func in functions or []:
            self.register_hlsl_generic_specialization_type(
                getattr(func, "return_type", None),
                visited,
            )
            for parameter in getattr(func, "parameters", []) or []:
                self.register_hlsl_generic_specialization_type(
                    getattr(
                        parameter,
                        "param_type",
                        getattr(
                            parameter, "vtype", getattr(parameter, "var_type", None)
                        ),
                    ),
                    visited,
                )

    def register_hlsl_generic_specialization_type(self, type_value, visited=None):
        type_text = self.type_name_string(type_value)
        if not type_text:
            return
        base_text, _array_suffix = split_array_type_suffix(str(type_text).strip())
        if not base_text or base_text in visited:
            return
        visited.add(base_text)

        base_name, generic_args = generic_type_parts(base_text)
        for generic_arg in generic_args:
            self.register_hlsl_generic_specialization_type(generic_arg, visited)
        if not generic_args:
            return

        enum_definition = self.generic_enum_struct_definitions.get(base_name)
        if enum_definition is not None and len(generic_args) == len(
            enum_definition["generic_params"]
        ):
            specialization = self.generic_enum_specializations.get(base_text)
            if specialization is None:
                specialization = build_generic_enum_specialization(
                    base_text,
                    enum_definition,
                )
                self.generic_enum_specializations[base_text] = specialization
            for _field_name, field_type in generic_enum_specialized_fields(
                self,
                specialization,
            ):
                self.register_hlsl_generic_specialization_type(field_type, visited)
            return

        struct_definition = self.generic_struct_definitions.get(base_name)
        if struct_definition is not None and len(generic_args) == len(
            struct_definition["generic_params"]
        ):
            specialization = self.generic_struct_specializations.get(base_text)
            if specialization is None:
                specialization = build_generic_struct_specialization(
                    base_text,
                    struct_definition,
                )
                self.generic_struct_specializations[base_text] = specialization
            for _field_name, field_type in generic_struct_specialized_fields(
                self.type_name_string,
                specialization,
            ):
                self.register_hlsl_generic_specialization_type(field_type, visited)

    def hlsl_struct_declaration_records(self, structs):
        records = []

        def add_record(name, code, field_types):
            records.append(
                {
                    "name": name,
                    "code": code,
                    "field_types": list(field_types or []),
                    "index": len(records),
                }
            )

        for specialization in (self.generic_struct_specializations or {}).values():
            fields = generic_struct_specialized_fields(
                self.type_name_string,
                specialization,
            )
            add_record(
                specialization["struct_name"],
                self.generate_hlsl_generic_struct_declaration(
                    specialization,
                    fields,
                ),
                [field_type for _field_name, field_type in fields],
            )

        for enum in self.struct_payload_enums or []:
            fields = enum_struct_fields(enum) or []
            add_record(
                enum.name,
                self.generate_hlsl_enum_struct_declaration(enum, fields),
                [field_type for _field_name, field_type in fields],
            )

        for specialization in (self.generic_enum_specializations or {}).values():
            fields = generic_enum_specialized_fields(self, specialization)
            add_record(
                specialization["struct_name"],
                self.generate_hlsl_generic_enum_struct_declaration(
                    specialization,
                    fields,
                ),
                [field_type for _field_name, field_type in fields],
            )

        for node in structs:
            if not isinstance(node, StructNode):
                continue
            if node.name in self.generic_enum_struct_definitions:
                continue
            if node.name in self.generic_struct_definitions:
                continue
            if node.name in self.lowered_glsl_buffer_block_struct_names:
                continue
            if node.name in self.glsl_buffer_block_struct_names:
                code = self.glsl_buffer_block_diagnostic("HLSL", node.name, None, None)
                code += self.unsupported_glsl_buffer_block_struct_placeholder(
                    "HLSL",
                    node.name,
                )
                add_record(node.name, code, [])
                continue
            add_record(
                node.name,
                self.generate_hlsl_regular_struct_declaration(node),
                (
                    [self.hlsl_union_layouts[node.name]["storage_type"]]
                    if node.name in self.hlsl_union_layouts
                    else self.struct_member_types.get(node.name, {}).values()
                ),
            )

        all_names = {record["name"] for record in records}
        for record in records:
            record["dependencies"] = self.hlsl_struct_field_dependencies(
                record["name"],
                record["field_types"],
                all_names,
            )
        return records

    def generate_hlsl_generic_struct_declaration(self, specialization, fields):
        code = f"struct {specialization['struct_name']} {{\n"
        for field_name, field_type in fields:
            self.hlsl_reject_struct_pointer_member(
                specialization["struct_name"], field_name, field_type
            )
            declaration = format_c_style_array_declaration(
                self.map_type(field_type),
                field_name,
            )
            code += f"    {declaration};\n"
        code += "};\n\n"
        return code

    def generate_hlsl_enum_struct_declaration(self, enum, fields):
        code = f"struct {enum.name} {{\n"
        code += "    int variant;\n"
        for field_name, field_type in fields:
            self.hlsl_reject_struct_pointer_member(enum.name, field_name, field_type)
            declaration = format_c_style_array_declaration(
                self.map_type(field_type),
                field_name,
            )
            code += f"    {declaration};\n"
        code += "};\n\n"
        return code

    def generate_hlsl_generic_enum_struct_declaration(self, specialization, fields):
        code = f"struct {specialization['struct_name']} {{\n"
        code += "    int variant;\n"
        for field_name, field_type in fields:
            self.hlsl_reject_struct_pointer_member(
                specialization["struct_name"], field_name, field_type
            )
            declaration = format_c_style_array_declaration(
                self.map_type(field_type),
                field_name,
            )
            code += f"    {declaration};\n"
        code += "};\n\n"
        return code

    def hlsl_crossgl_fixed_array_components(self, type_value):
        rendered_type = self.type_name_string(type_value)
        dimensions = []
        element_type = str(rendered_type or "").strip()
        while element_type:
            base_name, generic_args = generic_type_parts(element_type)
            if base_name.rsplit("::", 1)[-1] != "array" or len(generic_args) != 2:
                break
            extent = str(generic_args[1]).strip()
            if not extent:
                return None
            dimensions.append(extent)
            element_type = str(generic_args[0]).strip()
        if not dimensions:
            return None
        return element_type, dimensions

    def hlsl_crossgl_fixed_array_member_type(
        self,
        struct_name,
        member_name,
        type_value,
    ):
        components = self.hlsl_crossgl_fixed_array_components(type_value)
        if components is None:
            return None
        element_type, dimensions = components
        mapped_element_type = self.map_struct_member_type(
            struct_name,
            member_name,
            element_type,
        )
        return mapped_element_type + "".join(
            f"[{dimension}]" for dimension in dimensions
        )

    def hlsl_crossgl_fixed_array_value_type(self, type_value):
        components = self.hlsl_crossgl_fixed_array_components(type_value)
        if components is None:
            return None
        element_type, dimensions = components
        return self.map_type(element_type) + "".join(
            f"[{dimension}]" for dimension in dimensions
        )

    def generate_hlsl_regular_struct_declaration(self, node):
        union_layout = self.hlsl_union_layouts.get(node.name)
        if union_layout is not None:
            return self.generate_hlsl_union_storage_declaration(node, union_layout)

        payload = (
            "[raypayload] "
            if self.hlsl_union_attribute(node, RAY_PAYLOAD_ATTRIBUTE)
            else ""
        )
        code = f"struct {payload}{node.name} {{\n"
        members = [
            member
            for member in getattr(node, "members", [])
            if not self.hlsl_static_struct_member(member)
        ]
        if not members and not self.hlsl_union_attribute(
            node, EMPTY_STREAM_VERTEX_ATTRIBUTE
        ):
            code += "    int _crossgl_empty;\n"
        default_member_semantics = self.hlsl_default_struct_member_semantics(node)
        for member in members:
            self.hlsl_reject_struct_pointer_member(
                node.name,
                getattr(member, "name", None),
                self.hlsl_struct_member_raw_type(member),
                array_node=isinstance(member, ArrayNode),
                source_node=member,
            )
            if isinstance(member, ArrayNode):
                element_type = getattr(
                    member,
                    "element_type",
                    getattr(member, "vtype", "float"),
                )
                member_type = self.map_struct_member_type(
                    node.name,
                    member.name,
                    element_type,
                )
                if self.hlsl_should_lower_struct_resource_member(
                    node.name, member.name
                ):
                    continue
                if member.size:
                    declaration_type = f"{member_type}[{member.size}]"
                elif self.hlsl_struct_member_is_resource(element_type):
                    declaration_type = f"{member_type}[]"
                else:
                    raise UnsizedArrayUnsupportedError(
                        "directx", f"{node.name}.{member.name}", "is a struct member"
                    )
                semantic = default_member_semantics.get(member.name)
                if semantic is None:
                    semantic = self.hlsl_struct_member_declared_semantic(member)
                self.validate_hlsl_struct_member_semantic_type(
                    node.name,
                    member.name,
                    declaration_type,
                    semantic,
                )
                semantic_attr = self.map_semantic(
                    semantic
                ) + self.hlsl_payload_access_qualifiers(member)
                interpolation_prefix = self.hlsl_interpolation_modifier_prefix(member)
                vk_location = hlsl_vk_location_prefix(self, member, "    ")
                precise_prefix = self.hlsl_precise_modifier_prefix(member)
                precise_prefix += self.hlsl_matrix_orientation_prefix(member)
                tess_factor_declaration = self.hlsl_tess_factor_member_declaration(
                    declaration_type,
                    member.name,
                    semantic,
                )
                if tess_factor_declaration is not None:
                    code += f"    {tess_factor_declaration} {semantic_attr};\n"
                else:
                    declaration = format_c_style_array_declaration(
                        f"{interpolation_prefix}{precise_prefix}{declaration_type}",
                        member.name,
                    )
                    code += f"{vk_location}    {declaration}{semantic_attr};\n"
                continue

            generic_fixed_array_type = None
            if hasattr(member, "member_type"):
                generic_fixed_array_type = self.hlsl_crossgl_fixed_array_member_type(
                    node.name,
                    member.name,
                    member.member_type,
                )
                if generic_fixed_array_type is not None:
                    member_type = generic_fixed_array_type
                    array_syntax = None
                elif str(type(member.member_type)).find("ArrayType") != -1:
                    raw_element_type = self.convert_type_node_to_string(
                        member.member_type.element_type
                    )
                    member_type_str = self.convert_type_node_to_string(
                        member.member_type
                    )
                    member_type = self.map_struct_member_type(
                        node.name,
                        member.name,
                        member_type_str,
                    )
                    array_syntax = None
                else:
                    member_type = self.map_struct_member_type(
                        node.name,
                        member.name,
                        member.member_type,
                    )
                    array_syntax = ""
            elif hasattr(member, "vtype"):
                member_type = self.map_struct_member_type(
                    node.name,
                    member.name,
                    member.vtype,
                )
                array_syntax = ""
            else:
                member_type = "float"
                array_syntax = ""

            if self.hlsl_should_lower_struct_resource_member(node.name, member.name):
                continue

            if (
                array_syntax is None
                and generic_fixed_array_type is None
                and not self.hlsl_struct_member_is_resource(raw_element_type)
            ):
                array_size = getattr(member.member_type, "size", None)
                if array_size is None:
                    raise UnsizedArrayUnsupportedError(
                        "directx", f"{node.name}.{member.name}", "is a struct member"
                    )

            semantic = default_member_semantics.get(member.name)
            if semantic is None:
                semantic = self.hlsl_struct_member_declared_semantic(member)
            member_type = self.hlsl_unsigned_system_value_type(member_type, semantic)

            self.validate_hlsl_struct_member_semantic_type(
                node.name,
                member.name,
                member_type,
                semantic,
            )
            semantic_attr = self.map_semantic(
                semantic
            ) + self.hlsl_payload_access_qualifiers(member)
            interpolation_prefix = self.hlsl_interpolation_modifier_prefix(member)
            vk_location = hlsl_vk_location_prefix(self, member, "    ")
            precise_prefix = self.hlsl_precise_modifier_prefix(member)
            precise_prefix += self.hlsl_matrix_orientation_prefix(member)
            tess_factor_declaration = self.hlsl_tess_factor_member_declaration(
                member_type,
                member.name,
                semantic,
            )
            if tess_factor_declaration is not None:
                code += f"    {tess_factor_declaration} {semantic_attr};\n"
                continue

            if array_syntax is None:
                declaration = format_c_style_array_declaration(
                    f"{interpolation_prefix}{precise_prefix}{member_type}",
                    member.name,
                )
                code += f"{vk_location}    {declaration}{semantic_attr};\n"
            else:
                code += (
                    f"{vk_location}    {interpolation_prefix}{precise_prefix}{member_type} "
                    f"{member.name}{array_syntax}{semantic_attr};\n"
                )
        code += "};\n"
        return code

    def generate_hlsl_union_storage_declaration(self, node, layout):
        return (
            f"struct {node.name} {{\n"
            f"    {layout['storage_type']} {layout['storage_field']};\n"
            "};\n"
        )

    def hlsl_payload_access_qualifiers(self, member):
        """``member``'s ray payload access qualifiers, `` : read(caller) : write(closesthit)``, or ``""``."""
        qualifiers = ""
        for attribute in getattr(member, "attributes", None) or []:
            keyword = PAYLOAD_ACCESS_ATTRIBUTES.get(
                str(getattr(attribute, "name", "")).lower()
            )
            if keyword is None:
                continue
            stages = []
            for argument in getattr(attribute, "arguments", None) or []:
                stage = str(getattr(argument, "name", argument)).strip().lower()
                if stage not in PAYLOAD_STAGES:
                    raise HlslRayPayloadUnsupportedError(
                        f"'{stage}' in the {keyword} qualifier of payload field "
                        f"'{getattr(member, 'name', '')}' is not a payload stage; "
                        f"the stages are {', '.join(PAYLOAD_STAGES)}"
                    )
                stages.append(stage)
            qualifiers += f" : {keyword}({', '.join(stages)})"
        return qualifiers

    def hlsl_union_attribute(self, node, name):
        for attribute in getattr(node, "attributes", []) or []:
            if str(getattr(attribute, "name", "")).lower() == name:
                return attribute
        return None

    def hlsl_union_layout_error(
        self,
        node,
        *,
        reason,
        detail,
        member=None,
        member_type=None,
        size=None,
        alignment=None,
    ):
        union_name = getattr(node, "name", None) or "<anonymous>"
        member_name = getattr(member, "name", None)
        member_detail = f" member '{member_name}'" if member_name else ""
        return DirectXUnionLayoutError(
            f"DirectX cannot lower source union '{union_name}'{member_detail} "
            f"to canonical HLSL storage: {detail}",
            union_name=union_name,
            member_name=member_name,
            member_type=member_type,
            size=size,
            alignment=alignment,
            reason=reason,
            source_location=(
                getattr(member, "source_location", None)
                or getattr(node, "source_location", None)
            ),
        )

    def collect_hlsl_union_layouts(self, structs):
        return UnionLayoutCollector(
            self, self.hlsl_union_layout_error, self.hlsl_static_struct_member
        ).collect(structs)
