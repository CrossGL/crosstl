"""Half, complex, type-cast, integer, float, byte-permute and vote intrinsics in the HIP to CrossGL converter."""


class HipMathIntrinsicMixin:
    """Math intrinsic members of ``HipToCrossGLConverter``."""

    def format_hip_fp16_intrinsic_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        if function_name == "__half2float" and len(args) == 1:
            return f"f32({args[0]})"
        if function_name == "__float2half" and len(args) == 1:
            return f"f16({args[0]})"
        if function_name == "__float2half2_rn" and len(args) == 1:
            return self.format_vector_constructor("vec2", [args[0], args[0]], "f16")
        if function_name == "__floats2half2_rn" and len(args) == 2:
            return self.format_vector_constructor("vec2", args, "f16")
        if function_name == "__halves2half2" and len(args) == 2:
            return self.format_vector_constructor("vec2", args, "f16")
        if function_name == "__low2float" and len(args) == 1:
            return f"f32({self.format_vector_component_access(args[0], 'x')})"
        if function_name == "__high2float" and len(args) == 1:
            return f"f32({self.format_vector_component_access(args[0], 'y')})"
        if function_name == "__low2half" and len(args) == 1:
            return self.format_vector_component_access(args[0], "x")
        if function_name == "__high2half" and len(args) == 1:
            return self.format_vector_component_access(args[0], "y")
        if function_name == "__low2half2" and len(args) == 1:
            low = self.format_vector_component_access(args[0], "x")
            return self.format_vector_constructor("vec2", [low, low], "f16")
        if function_name == "__high2half2" and len(args) == 1:
            high = self.format_vector_component_access(args[0], "y")
            return self.format_vector_constructor("vec2", [high, high], "f16")
        if function_name in {"__lows2half2", "__low2half2"} and len(args) == 2:
            return self.format_vector_constructor(
                "vec2",
                [
                    self.format_vector_component_access(args[0], "x"),
                    self.format_vector_component_access(args[1], "x"),
                ],
                "f16",
            )
        if function_name == "__highs2half2" and len(args) == 2:
            return self.format_vector_constructor(
                "vec2",
                [
                    self.format_vector_component_access(args[0], "y"),
                    self.format_vector_component_access(args[1], "y"),
                ],
                "f16",
            )
        if function_name == "__lowhigh2highlow" and len(args) == 1:
            return self.format_vector_constructor(
                "vec2",
                [
                    self.format_vector_component_access(args[0], "y"),
                    self.format_vector_component_access(args[0], "x"),
                ],
                "f16",
            )
        if function_name == "__hadd2" and len(args) == 2:
            return f"({args[0]} + {args[1]})"
        if function_name == "__hmul2" and len(args) == 2:
            return f"({args[0]} * {args[1]})"
        if function_name == "__hfma2" and len(args) == 3:
            return f"fma({args[0]}, {args[1]}, {args[2]})"
        return None

    def format_hip_complex_intrinsic_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        if function_name in {"make_hipComplex", "make_hipFloatComplex"}:
            if len(args) == 2:
                return self.format_vector_constructor("vec2", args, "f32")
            return None
        if function_name == "make_hipDoubleComplex":
            if len(args) == 2:
                return self.format_vector_constructor("vec2", args, "f64")
            return None

        if function_name in {"hipCrealf", "hipCreal"} and len(args) == 1:
            return self.format_vector_component_access(args[0], "x")
        if function_name in {"hipCimagf", "hipCimag"} and len(args) == 1:
            return self.format_vector_component_access(args[0], "y")

        if function_name in {"hipCaddf", "hipCadd"} and len(args) == 2:
            return f"({args[0]} + {args[1]})"
        if function_name in {"hipCsubf", "hipCsub"} and len(args) == 2:
            return f"({args[0]} - {args[1]})"
        if function_name == "hipCmulf" and len(args) == 2:
            return self.format_hip_complex_multiply(args[0], args[1], "f32")
        if function_name == "hipCmul" and len(args) == 2:
            return self.format_hip_complex_multiply(args[0], args[1], "f64")

        return None

    def format_hip_complex_multiply(self, left, right, scalar_type):
        left_real = self.format_vector_component_access(left, "x")
        left_imag = self.format_vector_component_access(left, "y")
        right_real = self.format_vector_component_access(right, "x")
        right_imag = self.format_vector_component_access(right, "y")
        return self.format_vector_constructor(
            "vec2",
            [
                f"(({left_real} * {right_real}) - ({left_imag} * {right_imag}))",
                f"(({left_real} * {right_imag}) + ({left_imag} * {right_real}))",
            ],
            scalar_type,
        )

    def format_hip_type_cast_intrinsic_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        bit_reinterpret_intrinsics = {
            "__double_as_longlong": "doubleBitsToLong",
            "__float_as_int": "floatBitsToInt",
            "__float_as_uint": "floatBitsToUint",
            "__int_as_float": "intBitsToFloat",
            "__longlong_as_double": "longBitsToDouble",
            "__uint_as_float": "uintBitsToFloat",
        }
        mapped_name = bit_reinterpret_intrinsics.get(function_name)
        if mapped_name is not None and len(args) == 1:
            return f"{mapped_name}({args[0]})"

        return None

    def format_hip_integer_intrinsic_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        if function_name == "__mul24" and len(args) == 2:
            left = self.format_hip_signed_24_bit_operand(args[0])
            right = self.format_hip_signed_24_bit_operand(args[1])
            return f"({left} * {right})"
        if function_name == "__umul24" and len(args) == 2:
            return f"(({args[0]} & 0x00ffffffu) * ({args[1]} & 0x00ffffffu))"
        if function_name == "__mulhi" and len(args) == 2:
            return self.format_hip_signed_multiply_high(args[0], args[1])
        if function_name == "__umulhi" and len(args) == 2:
            return self.format_hip_unsigned_multiply_high(args[0], args[1])
        if function_name == "__byte_perm" and len(args) == 3:
            return self.format_hip_byte_perm(args[0], args[1], args[2])
        if (
            function_name in {
                "__funnelshift_l",
                "__funnelshift_lc",
                "__funnelshift_r",
                "__funnelshift_rc",
            }
            and len(args) == 3
        ):
            return self.format_hip_funnelshift(function_name, args[0], args[1], args[2])
        if function_name == "__bitextract_u32" and len(args) == 3:
            return self.format_hip_unsigned_bit_extract(args[0], args[1], args[2])
        if function_name in {"__ffs", "__ffsll"} and len(args) == 1:
            return f"(findLSB({args[0]}) + 1)"
        if function_name in {"__clz", "__clzll"} and len(args) == 1:
            return f"countLeadingZeros({args[0]})"
        if function_name in {"__brev", "__brevll"} and len(args) == 1:
            return f"bitfieldReverse({args[0]})"
        if function_name in {"__hadd", "__uhadd"} and len(args) == 2:
            return self.format_integer_average_floor(args[0], args[1])
        if function_name in {"__rhadd", "__urhadd"} and len(args) == 2:
            return self.format_integer_average_rounded(args[0], args[1])
        if function_name == "__sad" and len(args) == 3:
            return f"(abs({args[0]} - {args[1]}) + {args[2]})"
        if function_name == "__usad" and len(args) == 3:
            return (
                f"((({args[0]} > {args[1]}) ? ({args[0]} - {args[1]}) : "
                f"({args[1]} - {args[0]})) + {args[2]})"
            )
        if function_name in {"__popc", "__popcll"} and len(args) == 1:
            return f"bitCount({args[0]})"
        return None

    def format_integer_average_floor(self, left, right):
        return f"(({left} & {right}) + (({left} ^ {right}) >> 1))"

    def format_integer_average_rounded(self, left, right):
        return f"(({left} | {right}) - (({left} ^ {right}) >> 1))"

    def format_hip_signed_24_bit_operand(self, arg):
        return f"(({arg} << 8) >> 8)"

    def format_hip_signed_multiply_high(self, left, right):
        return f"i32((i64({left}) * i64({right})) >> 32)"

    def format_hip_unsigned_multiply_high(self, left, right):
        return f"u32((u64({left}) * u64({right})) >> 32u)"

    def format_hip_unsigned_bit_extract(self, value, offset, width):
        width_value = self.parse_hip_integer_literal(width)
        if width_value is not None:
            if width_value <= 0:
                mask = "0u"
            elif width_value >= 32:
                mask = "0xffffffffu"
            else:
                mask = f"0x{(1 << width_value) - 1:x}u"
        else:
            mask = f"((1u << {width}) - 1u)"
        return f"(({value} >> {offset}) & {mask})"

    def format_hip_funnelshift(self, function_name, low, high, shift):
        shift_amount = f"({shift} & 31u)"
        if function_name.endswith("lc") or function_name.endswith("rc"):
            shift_amount = f"min({shift}, 32u)"

        source = f"((u64({high}) << 32u) | u64({low}))"
        if function_name.endswith("l") or function_name.endswith("lc"):
            return f"u32(({source} << {shift_amount}) >> 32u)"
        return f"u32({source} >> {shift_amount})"

    def format_hip_float_intrinsic_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        if function_name == "__saturatef" and len(args) == 1:
            return f"clamp({args[0]}, 0.0f, 1.0f)"

        if (
            function_name in {
                "__fdividef",
                "__fdiv_rd",
                "__fdiv_rn",
                "__fdiv_ru",
                "__fdiv_rz",
            }
            and len(args) == 2
        ):
            return f"({args[0]} / {args[1]})"

        if (
            function_name in {"__fadd_rd", "__fadd_rn", "__fadd_ru", "__fadd_rz"}
            and len(args) == 2
        ):
            return f"({args[0]} + {args[1]})"

        if (
            function_name in {"__fsub_rd", "__fsub_rn", "__fsub_ru", "__fsub_rz"}
            and len(args) == 2
        ):
            return f"({args[0]} - {args[1]})"

        if (
            function_name in {"__fmul_rd", "__fmul_rn", "__fmul_ru", "__fmul_rz"}
            and len(args) == 2
        ):
            return f"({args[0]} * {args[1]})"

        if (
            function_name in {"__frcp_rd", "__frcp_rn", "__frcp_ru", "__frcp_rz"}
            and len(args) == 1
        ):
            return f"(1.0f / {args[0]})"

        return None

    def format_hip_byte_perm(self, left, right, selector):
        selector_value = self.parse_hip_integer_literal(selector)
        if selector_value is not None:
            return self.format_hip_static_byte_perm(left, right, selector_value)

        terms = []
        for result_index in range(4):
            selected = self.format_hip_dynamic_byte_perm_byte(
                left, right, selector, result_index
            )
            if result_index:
                selected = f"({selected} << {result_index * 8})"
            terms.append(selected)
        return f"({' | '.join(terms)})"

    def format_hip_static_byte_perm(self, left, right, selector_value):
        terms = []
        for result_index in range(4):
            source_index = (selector_value >> (result_index * 4)) & 0xF
            byte = self.format_hip_byte_source(left, right, source_index)
            if byte == "0":
                continue
            if result_index:
                byte = f"({byte} << {result_index * 8})"
            terms.append(byte)
        return f"({' | '.join(terms)})" if terms else "0"

    def format_hip_dynamic_byte_perm_byte(self, left, right, selector, result_index):
        selector_nibble = (
            f"(({selector} >> {result_index * 4}) & 0xf)"
            if result_index
            else f"({selector} & 0xf)"
        )
        fallback = "0"
        for source_index in reversed(range(8)):
            byte = self.format_hip_byte_source(left, right, source_index)
            fallback = f"(({selector_nibble} == {source_index}) ? {byte} : {fallback})"
        return fallback

    def format_hip_byte_source(self, left, right, source_index):
        if source_index < 0 or source_index > 7:
            return "0"
        source = left if source_index < 4 else right
        shift = (source_index % 4) * 8
        if shift:
            return f"(({source} >> {shift}) & 0xffu)"
        return f"({source} & 0xffu)"

    def parse_hip_integer_literal(self, value):
        text = self.strip_wrapping_parentheses(str(value).strip()).replace("'", "")
        if not text:
            return None
        while text and text[-1].lower() in {"u", "l"}:
            text = text[:-1]
        try:
            return int(text, 0)
        except ValueError:
            return None

    def format_vector_component_access(self, expression, component):
        text = str(expression).strip()
        if text and all(char.isalnum() or char in "_." for char in text):
            return f"{text}.{component}"
        return f"({text}).{component}"
