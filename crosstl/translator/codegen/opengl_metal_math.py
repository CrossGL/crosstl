"""Metal math in the GLSL generator: floating remainder helpers, bitcasts, signbit, copysign, clip, and software subgroup support."""

import re

from ..ast import (
    FunctionCallNode,
    VariableNode,
)
from ..refusals import OpenGLRefusal
from .opengl_values import (
    OpenGLCopySignError,
    OpenGLSignBitError,
)
from .software_quad_reductions import (
    SOFTWARE_QUAD_REDUCTIONS,
    glsl_quad_sum_helper,
)


class OpenGLMetalMathMixin:
    """Metal math members of ``GLSLCodeGen``."""

    def generate_glsl_floating_remainder_helpers(self):
        helpers = []
        # Integer significand division avoids overflowing the floating quotient
        # and retains exact zero signs and representable subnormal remainders.
        for scalar, bits, fraction_bits, exponent_bits in (
            ("float", 32, 23, 8),
            ("double", 64, 52, 11),
        ):
            name = self.glsl_floating_remainder_helper_names.get(scalar)
            if name is None:
                continue
            word = "uint" if bits == 32 else "uint64_t"
            zero, one = f"{word}(0)", f"{word}(1)"

            def unpack(value):
                if bits == 32:
                    return f"floatBitsToUint({value})"
                return f"(uint64_t(unpackDouble2x32({value}).x) | (uint64_t(unpackDouble2x32({value}).y) << 32u))"

            def pack(value):
                if bits == 32:
                    return f"uintBitsToFloat({value})"
                return f"packDouble2x32(uvec2(uint({value}), uint(({value}) >> 32u)))"

            helpers.append(
                f"{scalar} {name}({scalar} left, {scalar} right) {{\n"
                f"    {word} left_bits = {unpack('left')};\n"
                f"    {word} right_bits = {unpack('right')};\n"
                f"    {word} sign_mask = {one} << {bits - 1}u;\n"
                f"    {word} sign = left_bits & sign_mask;\n"
                f"    {word} x = left_bits & ~sign_mask;\n"
                f"    {word} y = right_bits & ~sign_mask;\n"
                f"    {word} hidden_bit = {one} << {fraction_bits}u;\n"
                f"    {word} fraction_mask = hidden_bit - {one};\n"
                f"    {word} infinity = (({one} << {exponent_bits}u) - {one}) << {fraction_bits}u;\n"
                f"    if (x >= infinity || y > infinity || y == {zero}) {{\n"
                f"        {word} invalid = infinity | (hidden_bit >> 1u);\n"
                f"        return {pack('invalid')};\n"
                "    }\n"
                "    if (x < y) { return left; }\n"
                f"    if (x == y) {{ return {pack('sign')}; }}\n"
                f"    int x_exponent = int(x >> {fraction_bits}u);\n"
                f"    int y_exponent = int(y >> {fraction_bits}u);\n"
                f"    {word} remainder = x & fraction_mask;\n"
                f"    {word} divisor = y & fraction_mask;\n"
                "    if (x_exponent == 0) {\n"
                "        x_exponent = 1;\n"
                "        while (remainder < hidden_bit) { remainder <<= 1u; x_exponent -= 1; }\n"
                "    } else { remainder |= hidden_bit; }\n"
                "    if (y_exponent == 0) {\n"
                "        y_exponent = 1;\n"
                "        while (divisor < hidden_bit) { divisor <<= 1u; y_exponent -= 1; }\n"
                "    } else { divisor |= hidden_bit; }\n"
                "    for (int shift = x_exponent - y_exponent; shift >= 0; shift -= 1) {\n"
                "        if (remainder >= divisor) { remainder -= divisor; }\n"
                f"        if (remainder == {zero}) {{ return {pack('sign')}; }}\n"
                "        if (shift > 0) { remainder <<= 1u; }\n"
                "    }\n"
                "    while (remainder < hidden_bit) { remainder <<= 1u; y_exponent -= 1; }\n"
                "    if (y_exponent > 0) {\n"
                f"        remainder = ({word}(y_exponent) << {fraction_bits}u) | (remainder & fraction_mask);\n"
                "    } else { remainder >>= uint(1 - y_exponent); }\n"
                "    remainder |= sign;\n"
                f"    return {pack('remainder')};\n"
                "}\n\n"
            )
        for value_type, name in sorted(
            self.glsl_floating_remainder_helper_names.items()
        ):
            match = re.fullmatch(r"(d?)vec([234])", value_type)
            if match is None:
                continue
            scalar = "double" if match.group(1) else "float"
            scalar_name = self.glsl_floating_remainder_helper_names[scalar]
            components = [
                f"{scalar_name}(left.{lane}, right.{lane})"
                for lane in "xyzw"[: int(match.group(2))]
            ]
            helpers.append(
                f"{value_type} {name}({value_type} left, {value_type} right) {{\n    return {value_type}({', '.join(components)});\n}}\n\n"
            )
        return "".join(helpers)

    def generate_glsl_log10_call(self, func_name, args):
        if (
            func_name not in {"log10", "metal::log10", "metal_u3a_u3alog10"}
            or func_name in self.function_return_types
        ):
            return None
        if len(args) != 1:
            raise OpenGLRefusal("OpenGL log10 requires 1 argument")

        value = self.generate_expression(args[0])
        return f"(log2({value}) * 0.3010299956639812)"

    def glsl_signbit_type_info(self, type_name):
        if type_name is None:
            return None
        mapped_type = self.map_type(type_name)
        if mapped_type == "float":
            return {"type": mapped_type, "lanes": 1, "result_type": "bool"}
        match = re.fullmatch(r"vec([234])", mapped_type or "")
        if match is None:
            return None
        lanes = int(match.group(1))
        return {
            "type": mapped_type,
            "lanes": lanes,
            "result_type": f"bvec{lanes}",
        }

    def glsl_signbit_result_type(self, func_name, args, *, source_type=False):
        if func_name not in self.GLSL_SIGNBIT_NAMES or len(args) != 1:
            return None
        argument_type = (
            self.glsl_source_expression_type(args[0])
            if source_type
            else self.expression_result_type(args[0])
        )
        info = self.glsl_signbit_type_info(argument_type)
        return info["result_type"] if info is not None else None

    def glsl_signbit_error(
        self,
        message,
        *,
        operand_type=None,
        result_type=None,
        reason,
        source_location=None,
    ):
        return OpenGLSignBitError(
            message,
            operand_type=operand_type,
            result_type=result_type,
            reason=reason,
            source_location=source_location,
        )

    def generate_glsl_signbit_call(self, func_name, args, *, call_node=None):
        if func_name not in self.GLSL_SIGNBIT_NAMES:
            return None

        resolved_overload = self.resolve_glsl_function_overload(
            func_name,
            args,
            call_node=call_node,
        )
        if resolved_overload is not None:
            return None

        source_location = getattr(call_node, "source_location", None)
        if len(args) != 1:
            raise self.glsl_signbit_error(
                "OpenGL signbit requires exactly one operand",
                reason="invalid-arity",
                source_location=source_location,
            )

        source_type = self.glsl_source_expression_type(args[0])
        display_type = (
            self.type_name_string(source_type) if source_type is not None else None
        )
        if source_type is None:
            raise self.glsl_signbit_error(
                "OpenGL signbit requires a statically known operand type",
                operand_type=display_type,
                reason="unresolved-operand-type",
                source_location=source_location,
            )

        info = self.glsl_signbit_type_info(source_type)
        if info is None:
            raise self.glsl_signbit_error(
                "OpenGL signbit supports only 32-bit float scalar and vector "
                f"operands, got {display_type}",
                operand_type=display_type,
                reason="unsupported-operand-type",
                source_location=source_location,
            )

        value = self.generate_expression_with_expected(args[0], info["type"])
        bits = f"(floatBitsToUint({value}) & 0x80000000u)"
        if info["lanes"] == 1:
            return f"({bits} != 0u)"
        mask_type = f"uvec{info['lanes']}"
        bits = f"(floatBitsToUint({value}) & {mask_type}(0x80000000u))"
        return f"notEqual({bits}, {mask_type}(0u))"

    def glsl_copysign_type_info(self, type_name):
        if type_name is None:
            return None
        mapped_type = self.map_type(type_name)
        if mapped_type == "float":
            return {"type": mapped_type, "lanes": 1}
        match = re.fullmatch(r"vec([234])", mapped_type or "")
        if match is None:
            return None
        return {"type": mapped_type, "lanes": int(match.group(1))}

    def glsl_copysign_error(
        self,
        message,
        *,
        operand_types,
        result_type=None,
        reason,
        source_location=None,
    ):
        return OpenGLCopySignError(
            message,
            operand_types=operand_types,
            result_type=result_type,
            reason=reason,
            source_location=source_location,
        )

    def generate_glsl_copysign_call(
        self,
        func_name,
        args,
        *,
        call_node=None,
    ):
        if func_name not in self.GLSL_COPYSIGN_NAMES:
            return None

        resolved_overload = self.resolve_glsl_function_overload(
            func_name,
            args,
            call_node=call_node,
        )
        if resolved_overload is not None:
            return None

        source_location = getattr(call_node, "source_location", None)
        if len(args) != 2:
            raise self.glsl_copysign_error(
                "OpenGL copysign requires exactly two operands",
                operand_types=(),
                reason="invalid-arity",
                source_location=source_location,
            )

        source_types = tuple(self.glsl_source_expression_type(arg) for arg in args)
        display_types = tuple(
            self.type_name_string(source_type) if source_type is not None else None
            for source_type in source_types
        )
        if any(source_type is None for source_type in source_types):
            raise self.glsl_copysign_error(
                "OpenGL copysign requires statically known operand types",
                operand_types=display_types,
                reason="unresolved-operand-type",
                source_location=source_location,
            )

        operand_info = tuple(
            self.glsl_copysign_type_info(source_type) for source_type in source_types
        )
        if any(info is None for info in operand_info):
            raise self.glsl_copysign_error(
                "OpenGL copysign supports only 32-bit float scalar and vector "
                f"operands, got {display_types[0]} and {display_types[1]}",
                operand_types=display_types,
                reason="unsupported-operand-type",
                source_location=source_location,
            )

        magnitude_info, sign_info = operand_info
        if magnitude_info["type"] != sign_info["type"]:
            raise self.glsl_copysign_error(
                "OpenGL copysign operands must have identical scalar or vector "
                f"shapes, got {display_types[0]} and {display_types[1]}",
                operand_types=display_types,
                result_type=magnitude_info["type"],
                reason="operand-shape-mismatch",
                source_location=source_location,
            )

        value_type = magnitude_info["type"]
        magnitude = self.generate_expression_with_expected(args[0], value_type)
        sign_source = self.generate_expression_with_expected(args[1], value_type)
        lanes = magnitude_info["lanes"]
        mask_type = "uint" if lanes == 1 else f"uvec{lanes}"
        magnitude_mask = "0x7fffffffu"
        sign_mask = "0x80000000u"
        if lanes != 1:
            magnitude_mask = f"{mask_type}({magnitude_mask})"
            sign_mask = f"{mask_type}({sign_mask})"
        return (
            "uintBitsToFloat(((floatBitsToUint({magnitude}) & {magnitude_mask}) | "
            "(floatBitsToUint({sign_source}) & {sign_mask})))"
        ).format(
            magnitude=magnitude,
            magnitude_mask=magnitude_mask,
            sign_source=sign_source,
            sign_mask=sign_mask,
        )

    def generate_bitcast_call(self, func_name, args):
        if (
            func_name not in self.GLSL_BITCAST_FUNCTIONS
            or func_name in self.function_return_types
        ):
            return None
        if len(args) != 1:
            raise OpenGLRefusal(f"OpenGL {func_name} alias requires 1 argument")

        target = self.glsl_bitcast_target(func_name, args[0])
        if target is None:
            source_type = self.map_type(self.expression_result_type(args[0]))
            raise OpenGLRefusal(
                f"OpenGL {func_name} alias cannot bitcast from {source_type}"
            )

        value = self.generate_expression(args[0])
        if target == "__crossgl_identity":
            return value
        if target.startswith("__crossgl_cast:"):
            return f"{target.split(':', 1)[1]}({value})"
        if target == "__crossgl_bfloat16_to_uint":
            return f"(floatBitsToUint({value}) >> 16u)"
        return f"{target}({value})"

    def glsl_bitcast_target(self, func_name, value_expr):
        source_type = self.expression_result_type(value_expr)
        source_type_name = self.glsl_normalized_source_type(source_type)
        source_component_type = (
            self.vector_component_type(source_type_name) or source_type_name
        )
        if (
            func_name == "asuint"
            and source_component_type in self.GLSL_BFLOAT16_ALIASES
        ):
            return "__crossgl_bfloat16_to_uint"

        value_type = self.map_type(source_type)
        component_type = self.vector_component_type(value_type) or value_type
        if (
            (func_name == "asfloat" and component_type == "float")
            or (func_name == "asint" and component_type == "int")
            or (func_name == "asuint" and component_type == "uint")
        ):
            return "__crossgl_identity"
        if func_name == "asint" and component_type == "uint":
            if value_type == "uint":
                return "__crossgl_cast:int"
            if value_type.startswith("uvec"):
                return f"__crossgl_cast:ivec{value_type[-1]}"
        if func_name == "asuint" and component_type == "int":
            if value_type == "int":
                return "__crossgl_cast:uint"
            if value_type.startswith("ivec"):
                return f"__crossgl_cast:uvec{value_type[-1]}"
        if func_name == "asfloat":
            if component_type == "int":
                return "intBitsToFloat"
            if component_type == "uint":
                return "uintBitsToFloat"
        if func_name == "asint" and component_type == "float":
            return "floatBitsToInt"
        if func_name == "asuint" and component_type == "float":
            return "floatBitsToUint"
        return None

    def glsl_bitcast_result_type(self, func_name, value_expr):
        value_type = self.map_type(self.expression_result_type(value_expr))
        if self.glsl_bitcast_target(func_name, value_expr) is None:
            return None

        if func_name == "asfloat":
            if value_type in {"int", "uint"}:
                return "float"
            if value_type.endswith(("2", "3", "4")):
                return f"vec{value_type[-1]}"
        if func_name == "asint":
            if value_type == "int":
                return "int"
            if value_type == "float":
                return "int"
            if value_type == "uint":
                return "int"
            if value_type.startswith("ivec"):
                return value_type
            if value_type.startswith("uvec"):
                return f"ivec{value_type[-1]}"
            if value_type.endswith(("2", "3", "4")):
                return f"ivec{value_type[-1]}"
        if func_name == "asuint":
            if value_type == "uint":
                return "uint"
            if value_type == "float":
                return "uint"
            if value_type == "int":
                return "uint"
            if value_type.startswith("uvec"):
                return value_type
            if value_type.startswith("ivec"):
                return f"uvec{value_type[-1]}"
            if value_type.endswith(("2", "3", "4")):
                return f"uvec{value_type[-1]}"
        return None

    def metal_as_type_target(self, func_name):
        text = str(func_name or "").strip()
        for prefix in ("as_type<", "metal::as_type<"):
            if text.startswith(prefix) and text.endswith(">"):
                return text[len(prefix) : -1].strip().split("::")[-1]
        return None

    def glsl_numeric_bitcast_shape(self, type_value):
        mapped_type = self.map_type(type_value)
        scalar_types = {
            "float": ("float", 32),
            "double": ("float", 64),
            "int": ("int", 32),
            "uint": ("uint", 32),
            "int64_t": ("int", 64),
            "uint64_t": ("uint", 64),
        }
        if mapped_type in scalar_types:
            component_kind, component_width = scalar_types[mapped_type]
            return {
                "type": mapped_type,
                "component_kind": component_kind,
                "component_width": component_width,
                "components": 1,
                "total_width": component_width,
            }

        vector_match = re.fullmatch(
            r"(dvec|i64vec|u64vec|vec|ivec|uvec)([2-4])", mapped_type
        )
        if vector_match is None:
            return None
        prefix, component_count = vector_match.groups()
        component_kind, component_width = {
            "dvec": ("float", 64),
            "i64vec": ("int", 64),
            "u64vec": ("uint", 64),
            "vec": ("float", 32),
            "ivec": ("int", 32),
            "uvec": ("uint", 32),
        }[prefix]
        components = int(component_count)
        return {
            "type": mapped_type,
            "component_kind": component_kind,
            "component_width": component_width,
            "components": components,
            "total_width": component_width * components,
        }

    def generate_glsl_explicit_bitcast(self, source_type, target_type, value):
        source = self.glsl_numeric_bitcast_shape(source_type)
        target = self.glsl_numeric_bitcast_shape(target_type)
        if source is None or target is None:
            raise OpenGLRefusal(
                f"OpenGL as_type<{target_type}> requires numeric scalar or vector "
                f"types; inferred source type is {source_type}"
            )
        if source["total_width"] != target["total_width"]:
            raise OpenGLRefusal(
                f"OpenGL as_type<{target['type']}> cannot bitcast from "
                f"{source['type']}: total widths are {target['total_width']} and "
                f"{source['total_width']} bits"
            )

        source_name = source["type"]
        target_name = target["type"]
        if source_name == target_name:
            return value

        integer_pair_types = {"ivec2", "uvec2"}
        if (
            source_name in self.GLSL_INT64_SCALAR_TYPES
            and target_name in integer_pair_types
        ):
            if source_name == "uint64_t":
                pair_value = f"unpackUint2x32({value})"
                return pair_value if target_name == "uvec2" else f"ivec2({pair_value})"
            pair_value = f"unpackInt2x32({value})"
            return pair_value if target_name == "ivec2" else f"uvec2({pair_value})"

        if (
            source_name in integer_pair_types
            and target_name in self.GLSL_INT64_SCALAR_TYPES
        ):
            if target_name == "uint64_t":
                pair_value = value if source_name == "uvec2" else f"uvec2({value})"
                return f"packUint2x32({pair_value})"
            pair_value = value if source_name == "ivec2" else f"ivec2({value})"
            return f"packInt2x32({pair_value})"

        if source_name == "double" and target_name in integer_pair_types:
            pair_value = f"unpackDouble2x32({value})"
            return pair_value if target_name == "uvec2" else f"ivec2({pair_value})"

        if source_name in integer_pair_types and target_name == "double":
            pair_value = value if source_name == "uvec2" else f"uvec2({value})"
            return f"packDouble2x32({pair_value})"

        if (
            source["components"] == target["components"]
            and source["component_width"] == target["component_width"]
        ):
            source_kind = source["component_kind"]
            target_kind = target["component_kind"]
            if source_kind == target_kind:
                return value
            if source["component_width"] == 32 and target_kind == "float":
                return {
                    "int": f"intBitsToFloat({value})",
                    "uint": f"uintBitsToFloat({value})",
                }[source_kind]
            if source["component_width"] == 32 and source_kind == "float":
                return {
                    "int": f"floatBitsToInt({value})",
                    "uint": f"floatBitsToUint({value})",
                }[target_kind]
            if source["component_width"] == 64 and target_kind == "float":
                return {
                    "int": f"int64BitsToDouble({value})",
                    "uint": f"uint64BitsToDouble({value})",
                }[source_kind]
            if source["component_width"] == 64 and source_kind == "float":
                return {
                    "int": f"doubleBitsToInt64({value})",
                    "uint": f"doubleBitsToUint64({value})",
                }[target_kind]
            return f"{target_name}({value})"

        raise OpenGLRefusal(
            f"OpenGL as_type<{target_name}> cannot exactly bitcast from "
            f"{source_name}; the equal-width shape is unsupported"
        )

    def generate_glsl_widened_16bit_as_type(
        self,
        source_type,
        target_type,
        value,
    ):
        source_name = self.glsl_normalized_source_type(source_type)
        target_name = self.glsl_normalized_source_type(target_type)
        half_scalar_types = {"f16", "float16", "float16_t", "half"}
        half_vector_types = {
            "f16vec2",
            "float16_t2",
            "half2",
            "f16vec3",
            "float16_t3",
            "half3",
            "f16vec4",
            "float16_t4",
            "half4",
        }
        source_narrow = self.glsl_narrow_integer_contract(source_name)
        target_narrow = self.glsl_narrow_integer_contract(target_name)

        if (
            source_narrow is not None
            and target_narrow is not None
            and source_narrow["bits"] == target_narrow["bits"] == 16
            and source_narrow["width"] == target_narrow["width"]
        ):
            converted = f"{self.map_type(target_type)}({value})"
            return self.glsl_apply_narrow_integer_contract(converted, target_type)

        if target_name in half_scalar_types:
            if source_name in half_scalar_types:
                return value
            if (
                source_narrow is not None
                and source_narrow["bits"] == 16
                and source_narrow["width"] == 1
            ):
                payload = value if not source_narrow["signed"] else f"uint({value})"
                return f"{self.glsl_half_bitcast_helper('decode')}({payload})"
            source_display = source_name or self.map_type(source_type)
            raise OpenGLRefusal(
                f"OpenGL as_type<{target_name}> cannot bitcast from "
                f"{source_display}: binary16 requires one exact 16-bit integer "
                "payload"
            )

        if source_name in half_scalar_types:
            if (
                target_narrow is not None
                and target_narrow["bits"] == 16
                and target_narrow["width"] == 1
            ):
                payload = f"{self.glsl_half_bitcast_helper('encode')}({value})"
                if target_narrow["signed"]:
                    return f"bitfieldExtract(int({payload}), 0, 16)"
                return f"bitfieldExtract({payload}, 0, 16)"
            target_display = target_name or self.map_type(target_type)
            raise OpenGLRefusal(
                f"OpenGL as_type<{target_display}> cannot bitcast from "
                f"{source_name}: binary16 requires one exact 16-bit integer "
                "result"
            )

        if target_name in {"f16vec2", "float16_t2", "half2"}:
            source_mapped = self.map_type(source_type)
            if source_mapped in {"int", "uint"} and source_narrow is None:
                payload = value if source_mapped == "uint" else f"uint({value})"
                return f"{self.glsl_half_bitcast_helper('unpack')}({payload})"
            raise OpenGLRefusal(
                f"OpenGL as_type<{target_name}> requires one exact 32-bit "
                "integer payload"
            )

        if source_name in {"f16vec2", "float16_t2", "half2"}:
            target_mapped = self.map_type(target_type)
            if target_mapped in {"int", "uint"} and target_narrow is None:
                payload = f"{self.glsl_half_bitcast_helper('pack')}({value})"
                return payload if target_mapped == "uint" else f"int({payload})"
            raise OpenGLRefusal(
                f"OpenGL as_type<{target_name}> cannot bitcast from "
                f"{source_name}: binary16x2 requires one exact 32-bit integer "
                "result"
            )

        if source_name in half_vector_types or target_name in half_vector_types:
            raise OpenGLRefusal(
                f"OpenGL as_type<{target_name}> cannot exactly bitcast from "
                f"{source_name}: widened binary16 vector shape is unsupported"
            )

        if source_name in self.GLSL_BFLOAT16_ALIASES and target_narrow is not None:
            if target_narrow["bits"] == 16 and target_narrow["width"] == 1:
                payload = f"(floatBitsToUint({value}) >> 16u)"
                if target_narrow["signed"]:
                    return f"bitfieldExtract(int({payload}), 0, 16)"
                return f"bitfieldExtract({payload}, 0, 16)"
        if (
            any(
                narrow is not None and narrow["bits"] == 16
                for narrow in (source_narrow, target_narrow)
            )
            and target_name not in self.GLSL_BFLOAT16_ALIASES
        ):
            raise OpenGLRefusal(
                f"OpenGL as_type<{target_name}> cannot exactly bitcast from "
                f"{source_name}: 16-bit integer payloads require a matching "
                "logical width and supported shape"
            )
        return None

    def generate_metal_as_type_call(self, func_name, args):
        if func_name in self.function_return_types:
            return None

        target_type = self.metal_as_type_target(func_name)
        if target_type is None:
            return None
        if len(args) != 1:
            raise OpenGLRefusal(
                f"OpenGL as_type<{target_type}> alias requires 1 argument"
            )

        value_expr = args[0]
        source_type = self.glsl_source_expression_type(value_expr)
        if source_type is None:
            raise OpenGLRefusal(
                f"OpenGL as_type<{target_type}> cannot infer its source type"
            )

        value = self.generate_expression(value_expr)
        widened_16bit = self.generate_glsl_widened_16bit_as_type(
            source_type,
            target_type,
            value,
        )
        if widened_16bit is not None:
            return widened_16bit

        if self.glsl_bfloat_width(target_type) == 1:
            value_type = self.map_type(source_type)
            component_type = self.vector_component_type(value_type) or value_type
            if component_type not in {"int", "uint"}:
                raise OpenGLRefusal(
                    "OpenGL as_type<bfloat16_t> requires an integer payload"
                )

            payload = value if component_type == "uint" else f"uint({value})"
            return f"uintBitsToFloat(({payload} << 16u))"

        return self.generate_glsl_explicit_bitcast(
            self.map_type(source_type), self.map_type(target_type), value
        )

    def collect_glsl_type_aliases(self, ast):
        return {
            node.name: getattr(node, "var_type", None)
            for node in getattr(ast, "global_variables", []) or []
            if self.is_elided_glsl_type_alias_global(node)
        }

    def is_glsl_type_alias_declaration(self, node):
        return isinstance(node, VariableNode) and bool(
            getattr(node, "is_type_alias", False)
        )

    def is_elided_glsl_type_alias_global(self, node):
        if self.is_glsl_type_alias_declaration(node):
            return True
        if not isinstance(node, VariableNode):
            return False
        if getattr(node, "name", None) not in self.GLSL_BFLOAT16_ALIASES:
            return False
        if getattr(node, "initial_value", None) is not None:
            return False
        if getattr(node, "attributes", None):
            return False
        if getattr(node, "qualifiers", None):
            return False
        return self.type_name_string(getattr(node, "var_type", None)) in {
            "f16",
            "float16",
            "bfloat",
            "bfloat16",
        }

    def generate_mul_call(self, func_name, args):
        if func_name != "mul" or func_name in self.function_return_types:
            return None
        if len(args) != 2:
            raise OpenGLRefusal("OpenGL mul alias requires 2 arguments")

        left = self.generate_expression(args[0])
        right = self.generate_expression(args[1])
        return f"({left} * {right})"

    def generate_packed_byte_call(self, func_name, args):
        template = self.PACKED_BYTE_INTRINSICS.get(func_name)
        if template is None or func_name in self.function_return_types:
            return None
        if len(args) != 1:
            raise OpenGLRefusal(
                f"OpenGL {func_name} alias requires 1 argument; got {len(args)}"
            )
        return template.format(value=self.generate_expression(args[0]))

    def generate_mad_call(self, func_name, args):
        if func_name != "mad" or func_name in self.function_return_types:
            return None
        if len(args) != 3:
            raise OpenGLRefusal("OpenGL mad alias requires 3 arguments")

        left = self.generate_expression(args[0])
        right = self.generate_expression(args[1])
        addend = self.generate_expression(args[2])
        return f"(({left} * {right}) + {addend})"

    def generate_packed_dot4add_call(self, func_name, args):
        lowering = self.PACKED_DOT4ADD_LANES.get(func_name)
        if lowering is None or func_name in self.function_return_types:
            return None
        if len(args) != 3:
            raise OpenGLRefusal(
                f"OpenGL {func_name} alias requires 3 arguments; got {len(args)}"
            )

        scalar_type, lanes = lowering
        left = lanes.format(value=self.generate_expression(args[0]))
        right = lanes.format(value=self.generate_expression(args[1]))
        accumulator = self.generate_expression(args[2])
        return (
            f"({scalar_type}({accumulator}) + "
            f"{scalar_type}(dot(vec4({left}), vec4({right}))))"
        )

    def generate_clip_statement(self, expression, indent):
        if not isinstance(expression, FunctionCallNode):
            return None

        func_name = self.function_call_name(expression)
        if func_name != "clip" or func_name in self.function_return_types:
            return None

        args = getattr(expression, "arguments", getattr(expression, "args", [])) or []
        if len(args) != 1:
            raise OpenGLRefusal("OpenGL clip alias requires 1 argument")
        if self.current_stage_entry_type not in (None, "fragment"):
            raise OpenGLRefusal(
                "OpenGL clip alias lowers to discard and is only valid in "
                "fragment stages"
            )

        value_expr = args[0]
        value = self.generate_expression(value_expr)
        condition = self.clip_discard_condition(value_expr, value)
        indent_str = "    " * indent
        return (
            f"{indent_str}if ({condition}) {{\n"
            f"{indent_str}    discard;\n"
            f"{indent_str}}}\n"
        )

    def clip_discard_condition(self, value_expr, value):
        value_type = self.map_type(self.expression_result_type(value_expr))
        zero = self.clip_zero_literal(value_type)
        if self.is_vector_value_type(value_type):
            return f"any(lessThan({value}, {value_type}({zero})))"
        return f"{value} < {zero}"

    def clip_zero_literal(self, value_type):
        component_type = self.vector_component_type(value_type) or value_type
        if component_type == "uint":
            return "0u"
        if component_type == "int":
            return "0"
        return "0.0"

    def saturate_bound_literals(self, value_expr):
        value_type = self.expression_result_type(value_expr)
        mapped_type = self.map_type(value_type)
        component_type = self.vector_component_type(mapped_type) or mapped_type
        if component_type == "uint":
            return "0u", "1u"
        if component_type == "int":
            return "0", "1"
        return "0.0", "1.0"

    def glsl_software_subgroup_payload_type(self, operation, source_type):
        if (
            operation in {"WaveActiveSum", "WaveActiveProduct"}
            or operation in self.GLSL_SOFTWARE_SUBGROUP_SCANS
        ) and self.glsl_half_width(source_type) == 1:
            self.glsl_apply_half_contract("value", "half", None)
            return "half"
        return self.map_type(source_type)

    def glsl_software_subgroup_scratch_name(self, value_type):
        suffix = self.GLSL_SOFTWARE_SUBGROUP_TYPE_SUFFIXES[value_type]
        return self.glsl_generated_module_identifier(
            ("software-subgroup-scratch", value_type),
            f"crossglSoftwareSubgroupScratch{suffix}",
        )

    def glsl_software_subgroup_helper_name(self, operation, value_type):
        operation_suffix = {
            "WaveActiveSum": "Sum",
            "WaveActiveProduct": "Product",
            "WaveActiveMin": "Min",
            "WaveActiveMax": "Max",
            "WaveActiveAllTrue": "All",
            "WaveActiveAnyTrue": "Any",
            "WaveShuffleDown": "ShuffleDown",
            "WaveShuffleUp": "ShuffleUp",
            "WaveShuffleXor": "ShuffleXor",
            "WaveReadLaneAt": "ReadLaneAt",
            "WaveShuffleAndFillUp": "ShuffleAndFillUp",
            "WavePrefixSum": "PrefixSum",
            "WavePrefixProduct": "PrefixProduct",
            "WavePrefixInclusiveSum": "PrefixInclusiveSum",
            "WavePrefixInclusiveProduct": "PrefixInclusiveProduct",
            "QuadActiveSum": "QuadSum",
        }[operation]
        type_suffix = self.GLSL_SOFTWARE_SUBGROUP_TYPE_SUFFIXES[value_type]
        return self.glsl_generated_module_identifier(
            ("software-subgroup-helper", operation, value_type),
            f"crossglSoftwareSubgroup{operation_suffix}{type_suffix}",
        )

    def glsl_software_subgroup_scan_helper(
        self, operation, value_type, helper, scratch, *, binary16=False
    ):
        operator, inclusive = self.GLSL_SOFTWARE_SUBGROUP_SCANS[operation]
        qualifier = "precise " if value_type == "float" else ""
        left = f"{scratch}[subgroupBase + (lane & ~(2u * stride - 1u)) + stride - 1u]"
        combine = f"{left} {operator} result"
        if value_type == "int":
            combine = f"int(uint({left}) {operator} uint(result))"
        if binary16:
            # A binary32 add or multiply of two binary16 values, rounded once to
            # binary16, is the correctly rounded binary16 result at every node.
            combine = f"{self.glsl_half_helper_names[1]}({combine})"
        # Separate each read phase from shared writes, including partial subgroups.
        code = (
            f"{value_type} {helper}({value_type} value) {{\n"
            "    uint invocation = gl_LocalInvocationIndex;\n"
            f"    uint lane = invocation % {self.software_subgroup_width}u;\n"
            "    uint subgroupBase = invocation - lane;\n"
            f"    {qualifier}{value_type} result = value;\n"
            f"    {scratch}[invocation] = value;\n"
            "    memoryBarrierShared();\n"
            "    barrier();\n"
            f"    for (uint stride = 1u; stride < {self.software_subgroup_width}u; stride <<= 1u) {{\n"
            "        if ((lane & stride) != 0u) {\n"
            f"            result = {combine};\n"
            "        }\n"
            "        memoryBarrierShared();\n"
            "        barrier();\n"
            f"        {scratch}[invocation] = result;\n"
            "        memoryBarrierShared();\n"
            "        barrier();\n"
            "    }\n"
        )
        identity = f"{value_type}({'0' if operator == '+' else '1'})"
        if value_type == "float" and operator == "+":
            identity = "uintBitsToFloat(0x80000000u)"
        code += (
            f"    result = {identity};\n"
            "    if (lane > 0u) {\n"
            f"        result = {scratch}[invocation - 1u];\n"
            "    }\n"
        )
        if inclusive:
            combine = f"result {operator} value"
            if value_type == "int":
                combine = f"int(uint(result) {operator} uint(value))"
            if binary16:
                combine = f"{self.glsl_half_helper_names[1]}({combine})"
            code += f"    result = {combine};\n"
        return code + (
            "    memoryBarrierShared();\n"
            "    barrier();\n"
            "    return result;\n"
            "}\n\n"
        )

    def generate_glsl_software_subgroup_support(self):
        if not self.required_glsl_software_subgroup_helpers:
            return ""

        invocation_count = (
            self.glsl_software_subgroup_workgroup_invocation_count
            or self.software_subgroup_width
        )
        partial = invocation_count % self.software_subgroup_width != 0
        value_types = sorted(
            {
                value_type
                for _operation, value_type in (
                    self.required_glsl_software_subgroup_helpers
                )
                if value_type in self.GLSL_SOFTWARE_SUBGROUP_VALUE_TYPES
                or value_type in {"bool", "half"}
                or (_operation == "WaveShuffleAndFillUp" and value_type == "uvec2")
            },
            key=lambda value_type: self.GLSL_SOFTWARE_SUBGROUP_TYPE_SUFFIXES[
                value_type
            ],
        )
        code = ""
        if "half" in value_types:
            code += f"float {self.glsl_half_helper_names[1]}(float value);\n"
        for value_type in value_types:
            scratch = self.glsl_software_subgroup_scratch_name(value_type)
            storage_type = "float" if value_type == "half" else value_type
            code += f"shared {storage_type} {scratch}[{invocation_count}];\n"
        code += "\n"

        operation_order = {
            "WaveActiveSum": 0,
            "WaveActiveMin": 1,
            "WaveActiveMax": 2,
            "WaveShuffleDown": 3,
            "WaveActiveAllTrue": 4,
            "WaveActiveAnyTrue": 5,
            "WaveActiveProduct": 6,
            "WavePrefixSum": 7,
            "WavePrefixProduct": 8,
            "WavePrefixInclusiveSum": 9,
            "WavePrefixInclusiveProduct": 10,
            "WaveReadLaneAt": 11,
            "WaveShuffleAndFillUp": 12,
            "WaveShuffleUp": 13,
            "WaveShuffleXor": 14,
            "QuadActiveSum": 15,
        }
        for operation, value_type in sorted(
            self.required_glsl_software_subgroup_helpers,
            key=lambda item: (
                operation_order[item[0]],
                self.GLSL_SOFTWARE_SUBGROUP_TYPE_SUFFIXES[item[1]],
            ),
        ):
            helper = self.glsl_software_subgroup_helper_name(operation, value_type)
            if (
                operation in self.GLSL_SOFTWARE_SUBGROUP_LANE_SHUFFLES
                and value_type == "uvec2"
            ):
                component_helper = self.glsl_software_subgroup_helper_name(
                    operation, "uint"
                )
                if operation == "WaveReadLaneAt":
                    code += (
                        f"uvec2 {helper}(uvec2 value, uint index) {{\n"
                        f"    uint x = {component_helper}(value.x, index);\n"
                        f"    uint y = {component_helper}(value.y, index);\n"
                        "    return uvec2(x, y);\n"
                        "}\n\n"
                    )
                    continue
                code += (
                    f"uvec2 {helper}(uvec2 value, uint delta) {{\n"
                    f"    return uvec2({component_helper}(value.x, delta), "
                    f"{component_helper}(value.y, delta));\n"
                    "}\n\n"
                )
                continue
            scratch = self.glsl_software_subgroup_scratch_name(value_type)
            half_sum = value_type == "half"
            if half_sum:
                value_type = "float"
            if operation == "WaveShuffleAndFillUp":
                width = self.software_subgroup_width
                code += (
                    f"{value_type} {helper}({value_type} value, {value_type} fill, uint delta, uint modulo) {{\n"
                    "    uint invocation = gl_LocalInvocationIndex;\n"
                    f"    uint lane = invocation % {width}u;\n"
                    "    uint subgroupBase = invocation - lane;\n"
                    f"    uint activeCount = min({width}u, {invocation_count}u - subgroupBase);\n"
                    "    bool validModulo = modulo == 2u || modulo == 4u || modulo == 8u || modulo == 16u || modulo == 32u;\n"
                    "    uint segmentWidth = validModulo ? modulo : 32u;\n"
                    "    uint segmentLane = lane % segmentWidth;\n"
                    "    bool useValue = segmentLane >= delta;\n"
                    "    uint sourceLane = lane;\n"
                    "    if (validModulo && delta <= segmentWidth) {\n"
                    "        sourceLane = useValue ? lane - delta : lane + segmentWidth - delta;\n"
                    "    }\n"
                    "    bool sourceValid = validModulo && delta <= segmentWidth && sourceLane < activeCount;\n"
                    "    sourceLane = sourceValid ? sourceLane : lane;\n"
                    f"    {scratch}[invocation] = value;\n"
                    "    memoryBarrierShared();\n"
                    "    barrier();\n"
                    f"    {value_type} result = {scratch}[subgroupBase + sourceLane];\n"
                    "    memoryBarrierShared();\n"
                    "    barrier();\n"
                    f"    {scratch}[invocation] = fill;\n"
                    "    memoryBarrierShared();\n"
                    "    barrier();\n"
                    "    if (sourceValid && !useValue) {\n"
                    f"        result = {scratch}[subgroupBase + sourceLane];\n"
                    "    }\n"
                    "    memoryBarrierShared();\n"
                    "    barrier();\n"
                    "    return result;\n"
                    "}\n\n"
                )
                continue
            if operation in self.GLSL_SOFTWARE_SUBGROUP_SCANS:
                code += self.glsl_software_subgroup_scan_helper(
                    operation, value_type, helper, scratch, binary16=half_sum
                )
                continue
            if operation in SOFTWARE_QUAD_REDUCTIONS:
                code += glsl_quad_sum_helper(
                    value_type, helper, scratch, invocation_count
                )
                continue
            if invocation_count == self.software_subgroup_width:
                lane_setup = "    uint lane = gl_LocalInvocationIndex;\n"
                left = f"{scratch}[lane]"
                right = f"{scratch}[lane + stride]"
                result_index = "0u"
            else:
                lane_setup = (
                    "    uint invocation = gl_LocalInvocationIndex;\n"
                    "    uint lane = invocation % "
                    f"{self.GLSL_SOFTWARE_SUBGROUP_WIDTH_MACRO};\n"
                    "    uint subgroupBase = invocation - lane;\n"
                )
                left = f"{scratch}[subgroupBase + lane]"
                right = f"{scratch}[subgroupBase + lane + stride]"
                result_index = "subgroupBase"
                if partial:
                    lane_setup += f"    uint activeCount = min({self.software_subgroup_width}u, {invocation_count}u - subgroupBase);\n"
            if operation in {
                "WaveActiveSum",
                "WaveActiveProduct",
                "WaveActiveMin",
                "WaveActiveMax",
                "WaveActiveAllTrue",
                "WaveActiveAnyTrue",
            }:
                reduction = {
                    "WaveActiveSum": lambda left, right: f"({left} + {right})",
                    "WaveActiveProduct": lambda left, right: f"({left} * {right})",
                    "WaveActiveMin": lambda left, right: f"min({left}, {right})",
                    "WaveActiveMax": lambda left, right: f"max({left}, {right})",
                    "WaveActiveAllTrue": lambda left, right: f"({left} && {right})",
                    "WaveActiveAnyTrue": lambda left, right: f"({left} || {right})",
                }[operation]
                write_index = (
                    "lane"
                    if invocation_count == self.software_subgroup_width
                    else "invocation"
                )
                # Explicit shared-memory ordering preserves scratch reuse through
                # helper calls in dynamically uniform branches on Mesa drivers.
                product = operation == "WaveActiveProduct"
                adjacent_pairs = product or half_sum
                loop = (
                    f"for (uint stride = 1u; stride < {self.software_subgroup_width}u; stride <<= 1u)"
                    if adjacent_pairs
                    else "for (uint stride = 16u; stride > 0u; stride >>= 1u)"
                )
                active = (
                    "(lane & (2u * stride - 1u)) == 0u"
                    if adjacent_pairs
                    else "lane < stride"
                )
                if partial:
                    active += " && lane + stride < activeCount"
                combine = f"            {left} = {reduction(left, right)};\n"
                if half_sum:
                    node = "product" if product else "sum"
                    combine = (
                        f"            precise float {node} = {reduction(left, right)};\n"
                        f"            {left} = {self.glsl_half_helper_names[1]}({node});\n"
                    )
                code += (
                    f"{value_type} {helper}({value_type} value) {{\n"
                    f"{lane_setup}"
                    f"    {scratch}[{write_index}] = value;\n"
                    "    memoryBarrierShared();\n"
                    "    barrier();\n"
                    f"    {loop} {{\n"
                    f"        if ({active}) {{\n"
                    f"{combine}"
                    "        }\n"
                    "        memoryBarrierShared();\n"
                    "        barrier();\n"
                    "    }\n"
                    f"    {value_type} result = {scratch}[{result_index}];\n"
                    "    memoryBarrierShared();\n"
                    "    barrier();\n"
                    "    return result;\n"
                    "}\n\n"
                )
            else:
                write_index = (
                    "lane"
                    if invocation_count == self.software_subgroup_width
                    else "invocation"
                )
                active_count = (
                    "activeCount" if partial else f"{self.software_subgroup_width}u"
                )
                source_index = (
                    "sourceLane"
                    if invocation_count == self.software_subgroup_width
                    else "subgroupBase + sourceLane"
                )
                absolute = operation == "WaveReadLaneAt"
                parameter = "index" if absolute else "delta"
                if absolute:
                    valid, source_lane = f"index < {active_count}", "index"
                else:
                    valid, source_lane = self.GLSL_SOFTWARE_SUBGROUP_RELATIVE_SHUFFLES[
                        operation
                    ]
                    valid = valid.format(active=active_count)
                code += (
                    f"{value_type} {helper}({value_type} value, uint {parameter}) {{\n"
                    f"{lane_setup}"
                    f"    {scratch}[{write_index}] = value;\n"
                    "    memoryBarrierShared();\n"
                    "    barrier();\n"
                    f"    bool sourceValid = {valid};\n"
                    f"    uint sourceLane = sourceValid ? {source_lane} : lane;\n"
                    f"    {value_type} result = sourceValid ? {scratch}[{source_index}] : value;\n"
                    "    memoryBarrierShared();\n"
                    "    barrier();\n"
                    "    return result;\n"
                    "}\n\n"
                )
        return code
