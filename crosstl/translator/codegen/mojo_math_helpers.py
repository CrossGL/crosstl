"""Math, select, swizzle and constructor helper functions the Mojo generator emits."""

from ..ast import (
    MatchNode,
    MemberAccessNode,
    TernaryOpNode,
)
from .mojo_tables import (
    MOJO_DTYPE_INFO,
    MOJO_DTYPE_SUFFIX,
    MOJO_VECTOR_ARITHMETIC_OPS,
    ROUND_EVEN_FLOAT_DTYPES,
)


class MojoMathHelperMixin:
    """Math-helper members of ``MojoCodeGen``."""

    def generate_math_helper(self, helper_name):
        if helper_name == "clamp":
            return (
                "def clamp(value: Float32, min_value: Float32, max_value: Float32) -> Float32:\n"
                "    return min(max(value, min_value), max_value)\n\n"
                "def clamp(value: SIMD[DType.float32, 2], min_value: Float32, max_value: Float32) -> SIMD[DType.float32, 2]:\n"
                "    return SIMD[DType.float32, 2](clamp(value[0], min_value, max_value), clamp(value[1], min_value, max_value))\n\n"
                "def clamp(value: SIMD[DType.float32, 2], min_value: SIMD[DType.float32, 2], max_value: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:\n"
                "    return SIMD[DType.float32, 2](clamp(value[0], min_value[0], max_value[0]), clamp(value[1], min_value[1], max_value[1]))\n\n"
                "def clamp(value: SIMD[DType.float32, 4], min_value: Float32, max_value: Float32) -> SIMD[DType.float32, 4]:\n"
                "    return SIMD[DType.float32, 4](clamp(value[0], min_value, max_value), clamp(value[1], min_value, max_value), clamp(value[2], min_value, max_value), clamp(value[3], min_value, max_value))\n\n"
                "def clamp(value: SIMD[DType.float32, 4], min_value: SIMD[DType.float32, 4], max_value: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:\n"
                "    return SIMD[DType.float32, 4](clamp(value[0], min_value[0], max_value[0]), clamp(value[1], min_value[1], max_value[1]), clamp(value[2], min_value[2], max_value[2]), clamp(value[3], min_value[3], max_value[3]))\n\n"
            )
        if helper_name == "max":
            return self.generate_min_max_helper("max", ">=")
        if helper_name == "min":
            return self.generate_min_max_helper("min", "<=")
        if helper_name == "dot_product":
            return (
                "def dot_product(a: SIMD[DType.float32, 2], b: SIMD[DType.float32, 2]) -> Float32:\n"
                "    return a[0] * b[0] + a[1] * b[1]\n\n"
                "def dot_product(a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]) -> Float32:\n"
                "    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]\n\n"
            )
        if helper_name == "magnitude":
            return (
                "def magnitude(v: SIMD[DType.float32, 2]) -> Float32:\n"
                "    return sqrt(dot_product(v, v))\n\n"
                "def magnitude(v: SIMD[DType.float32, 4]) -> Float32:\n"
                "    return sqrt(dot_product(v, v))\n\n"
            )
        if helper_name == "normalize":
            return (
                "def normalize(v: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:\n"
                "    var len = magnitude(v)\n"
                "    if len == 0.0:\n"
                "        return v\n"
                "    return v / len\n\n"
                "def normalize(v: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:\n"
                "    var len = magnitude(v)\n"
                "    if len == 0.0:\n"
                "        return v\n"
                "    return v / len\n\n"
            )
        if helper_name == "lerp":
            return (
                "def lerp(a: Float32, b: Float32, t: Float32) -> Float32:\n"
                "    return a + (b - a) * t\n\n"
                "def lerp(a: SIMD[DType.float32, 2], b: SIMD[DType.float32, 2], t: Float32) -> SIMD[DType.float32, 2]:\n"
                "    return a + (b - a) * t\n\n"
                "def lerp(a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4], t: Float32) -> SIMD[DType.float32, 4]:\n"
                "    return a + (b - a) * t\n\n"
            )
        if helper_name == "power":
            return (
                "def power(a: Float32, b: Float32) -> Float32:\n"
                "    return pow(a, b)\n\n"
                "def power(a: SIMD[DType.float32, 2], b: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:\n"
                "    return SIMD[DType.float32, 2](pow(a[0], b[0]), pow(a[1], b[1]))\n\n"
                "def power(a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:\n"
                "    return SIMD[DType.float32, 4](pow(a[0], b[0]), pow(a[1], b[1]), pow(a[2], b[2]), pow(a[3], b[3]))\n\n"
            )
        if helper_name == "fmod":
            return (
                "def fmod(a: Float32, b: Float32) -> Float32:\n"
                "    return a - floor(a / b) * b\n\n"
                "def fmod(a: SIMD[DType.float32, 2], b: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:\n"
                "    return SIMD[DType.float32, 2](fmod(a[0], b[0]), fmod(a[1], b[1]))\n\n"
                "def fmod(a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:\n"
                "    return SIMD[DType.float32, 4](fmod(a[0], b[0]), fmod(a[1], b[1]), fmod(a[2], b[2]), fmod(a[3], b[3]))\n\n"
            )
        if helper_name == "round_even":
            return self.generate_round_even_helper()
        if helper_name == "cross_product":
            return (
                "def cross_product(a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:\n"
                "    return SIMD[DType.float32, 4](a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0], 0.0)\n\n"
            )
        if helper_name == "reflect":
            return (
                "def reflect(i: SIMD[DType.float32, 4], n: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:\n"
                "    return i - n * (2.0 * dot_product(n, i))\n\n"
            )
        if helper_name == "refract":
            return (
                "def refract(i: SIMD[DType.float32, 4], n: SIMD[DType.float32, 4], eta: Float32) -> SIMD[DType.float32, 4]:\n"
                "    var ndoti = dot_product(n, i)\n"
                "    var k = 1.0 - eta * eta * (1.0 - ndoti * ndoti)\n"
                "    if k < 0.0:\n"
                "        return SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)\n"
                "    return i * eta - n * (eta * ndoti + sqrt(k))\n\n"
            )
        if helper_name == "step":
            return (
                "def step(edge: Float32, x: Float32) -> Float32:\n"
                "    return 0.0 if x < edge else 1.0\n\n"
                "def step(edge: Float32, x: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:\n"
                "    return SIMD[DType.float32, 2](step(edge, x[0]), step(edge, x[1]))\n\n"
                "def step(edge: SIMD[DType.float32, 2], x: Float32) -> SIMD[DType.float32, 2]:\n"
                "    return SIMD[DType.float32, 2](step(edge[0], x), step(edge[1], x))\n\n"
                "def step(edge: SIMD[DType.float32, 2], x: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:\n"
                "    return SIMD[DType.float32, 2](step(edge[0], x[0]), step(edge[1], x[1]))\n\n"
                "def step(edge: Float32, x: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:\n"
                "    return SIMD[DType.float32, 4](step(edge, x[0]), step(edge, x[1]), step(edge, x[2]), step(edge, x[3]))\n\n"
                "def step(edge: SIMD[DType.float32, 4], x: Float32) -> SIMD[DType.float32, 4]:\n"
                "    return SIMD[DType.float32, 4](step(edge[0], x), step(edge[1], x), step(edge[2], x), step(edge[3], x))\n\n"
                "def step(edge: SIMD[DType.float32, 4], x: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:\n"
                "    return SIMD[DType.float32, 4](step(edge[0], x[0]), step(edge[1], x[1]), step(edge[2], x[2]), step(edge[3], x[3]))\n\n"
            )
        if helper_name == "smoothstep":
            return (
                "def smoothstep(edge0: Float32, edge1: Float32, x: Float32) -> Float32:\n"
                "    var t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)\n"
                "    return t * t * (3.0 - 2.0 * t)\n\n"
                "def smoothstep(edge0: Float32, edge1: Float32, x: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:\n"
                "    return SIMD[DType.float32, 2](smoothstep(edge0, edge1, x[0]), smoothstep(edge0, edge1, x[1]))\n\n"
                "def smoothstep(edge0: SIMD[DType.float32, 2], edge1: SIMD[DType.float32, 2], x: Float32) -> SIMD[DType.float32, 2]:\n"
                "    return SIMD[DType.float32, 2](smoothstep(edge0[0], edge1[0], x), smoothstep(edge0[1], edge1[1], x))\n\n"
                "def smoothstep(edge0: SIMD[DType.float32, 2], edge1: SIMD[DType.float32, 2], x: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:\n"
                "    return SIMD[DType.float32, 2](smoothstep(edge0[0], edge1[0], x[0]), smoothstep(edge0[1], edge1[1], x[1]))\n\n"
                "def smoothstep(edge0: Float32, edge1: Float32, x: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:\n"
                "    return SIMD[DType.float32, 4](smoothstep(edge0, edge1, x[0]), smoothstep(edge0, edge1, x[1]), smoothstep(edge0, edge1, x[2]), smoothstep(edge0, edge1, x[3]))\n\n"
                "def smoothstep(edge0: SIMD[DType.float32, 4], edge1: SIMD[DType.float32, 4], x: Float32) -> SIMD[DType.float32, 4]:\n"
                "    return SIMD[DType.float32, 4](smoothstep(edge0[0], edge1[0], x), smoothstep(edge0[1], edge1[1], x), smoothstep(edge0[2], edge1[2], x), smoothstep(edge0[3], edge1[3], x))\n\n"
                "def smoothstep(edge0: SIMD[DType.float32, 4], edge1: SIMD[DType.float32, 4], x: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:\n"
                "    return SIMD[DType.float32, 4](smoothstep(edge0[0], edge1[0], x[0]), smoothstep(edge0[1], edge1[1], x[1]), smoothstep(edge0[2], edge1[2], x[2]), smoothstep(edge0[3], edge1[3], x[3]))\n\n"
            )
        return ""

    def generate_round_even_helper(self):
        """Round halfway cases to the nearest even integer, per element type."""
        code = ""
        for dtype, scalar_type in ROUND_EVEN_FLOAT_DTYPES.items():
            code += (
                f"def round_even(x: {scalar_type}) -> {scalar_type}:\n"
                "    var lower = floor(x)\n"
                "    var fraction = x - lower\n"
                "    if fraction < 0.5:\n"
                "        return lower\n"
                "    if fraction > 0.5:\n"
                "        return lower + 1.0\n"
                "    var parity = lower - floor(lower / 2.0) * 2.0\n"
                "    if parity == 0.0:\n"
                "        return lower\n"
                "    return lower + 1.0\n\n"
            )
            for width in (2, 4):
                vector_type = f"SIMD[{dtype}, {width}]"
                components = ", ".join(
                    f"round_even(v[{index}])" for index in range(width)
                )
                code += (
                    f"def round_even(v: {vector_type}) -> {vector_type}:\n"
                    f"    return {vector_type}({components})\n\n"
                )
        return code

    def generate_min_max_helper(self, helper_name, comparator):
        selector = f"a if a {comparator} b else b"

        def lane_selector(left, right):
            return f"{left} if {left} {comparator} {right} else {right}"

        code = (
            f"def {helper_name}(a: Float32, b: Float32) -> Float32:\n"
            f"    return {selector}\n\n"
        )
        for scalar_type in (
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "UInt8",
            "UInt16",
            "UInt32",
            "UInt64",
        ):
            code += (
                f"def {helper_name}(a: {scalar_type}, b: {scalar_type}) -> "
                f"{scalar_type}:\n"
                f"    return {selector}\n\n"
            )
        for width in (2, 4):
            vector_type = f"SIMD[DType.float32, {width}]"
            components = ", ".join(
                lane_selector(f"a[{index}]", f"b[{index}]") for index in range(width)
            )
            code += (
                f"def {helper_name}(a: {vector_type}, b: {vector_type}) -> "
                f"{vector_type}:\n"
                f"    return {vector_type}({components})\n\n"
            )
            components = ", ".join(
                lane_selector(f"a[{index}]", "b") for index in range(width)
            )
            code += (
                f"def {helper_name}(a: {vector_type}, b: Float32) -> "
                f"{vector_type}:\n"
                f"    return {vector_type}({components})\n\n"
            )
            components = ", ".join(
                lane_selector("a", f"b[{index}]") for index in range(width)
            )
            code += (
                f"def {helper_name}(a: Float32, b: {vector_type}) -> "
                f"{vector_type}:\n"
                f"    return {vector_type}({components})\n\n"
            )
        return code

    def generate_fract_helper(self, key):
        kind, dtype, source_width, storage_width = key
        scalar_type, _, pad_literal = MOJO_DTYPE_INFO[dtype]
        mojo_scalar_type = self.map_type(scalar_type)

        if kind == "scalar":
            helper_name = self.fract_scalar_helper_name(dtype)
            code = f"def {helper_name}(x: {mojo_scalar_type}) -> {mojo_scalar_type}:\n"
            code += "    return x - floor(x)\n\n"
            return code

        helper_name = self.fract_vector_helper_name(dtype, source_width, storage_width)
        vector_type = f"SIMD[{dtype}, {storage_width}]"
        components = [
            f"v[{index}] - floor(v[{index}])" for index in range(source_width)
        ]
        if storage_width > source_width:
            components.append(pad_literal)

        code = f"def {helper_name}(v: {vector_type}) -> {vector_type}:\n"
        code += f"    return {vector_type}({', '.join(components)})\n\n"
        return code

    def fract_scalar_helper_name(self, dtype):
        return f"_crossgl_fract_{MOJO_DTYPE_SUFFIX[dtype]}"

    def fract_vector_helper_name(self, dtype, source_width, storage_width):
        dtype_suffix = MOJO_DTYPE_SUFFIX[dtype]
        return f"_crossgl_fract_{dtype_suffix}_{source_width}_{storage_width}"

    def generate_mod_helper(self, key):
        dtype, kind, source_width, storage_width = key
        scalar_type, _, pad_literal = MOJO_DTYPE_INFO[dtype]
        mojo_scalar_type = self.map_type(scalar_type)
        helper_name = self.mod_helper_name(dtype, kind, source_width, storage_width)

        if kind == "ss":
            return (
                f"def {helper_name}(a: {mojo_scalar_type}, "
                f"b: {mojo_scalar_type}) -> {mojo_scalar_type}:\n"
                "    return a - floor(a / b) * b\n\n"
            )

        vector_type = f"SIMD[{dtype}, {storage_width}]"
        if kind == "vv":
            params = f"a: {vector_type}, b: {vector_type}"
            components = [
                f"{self.mod_helper_name(dtype, 'ss', 1, 1)}(a[{index}], b[{index}])"
                for index in range(source_width)
            ]
        elif kind == "vs":
            params = f"a: {vector_type}, b: {mojo_scalar_type}"
            components = [
                f"{self.mod_helper_name(dtype, 'ss', 1, 1)}(a[{index}], b)"
                for index in range(source_width)
            ]
        else:
            params = f"a: {mojo_scalar_type}, b: {vector_type}"
            components = [
                f"{self.mod_helper_name(dtype, 'ss', 1, 1)}(a, b[{index}])"
                for index in range(source_width)
            ]
        if storage_width > source_width:
            components.append(pad_literal)

        return (
            f"def {helper_name}({params}) -> {vector_type}:\n"
            f"    return {vector_type}({', '.join(components)})\n\n"
        )

    def mod_helper_name(self, dtype, kind, source_width, storage_width):
        dtype_suffix = MOJO_DTYPE_SUFFIX[dtype]
        if kind == "ss":
            return f"_crossgl_mod_{dtype_suffix}"
        return f"_crossgl_mod_{dtype_suffix}_{source_width}_{storage_width}_{kind}"

    def generate_saturate_helper(self, key):
        kind, dtype, width_or_columns, storage_width_or_rows = key
        scalar_type, _, pad_literal = MOJO_DTYPE_INFO[dtype]

        if kind == "scalar":
            mojo_scalar_type = self.map_type(scalar_type)
            helper_name = self.saturate_scalar_helper_name(dtype)
            value = self.saturate_value_expression("x")
            return (
                f"def {helper_name}(x: {mojo_scalar_type}) -> {mojo_scalar_type}:\n"
                f"    return {value}\n\n"
            )

        if kind == "matrix":
            columns = width_or_columns
            rows = storage_width_or_rows
            matrix_type = self.matrix_type_name(dtype, columns, rows)
            storage_rows = self.matrix_storage_rows(rows)
            column_type = f"SIMD[{dtype}, {storage_rows}]"
            pad_literal = self.matrix_zero_literal(dtype)
            column_args = []
            for column in range(columns):
                components = [
                    self.saturate_value_expression(f"m.c{column}[{row}]")
                    for row in range(rows)
                ]
                if rows == 3:
                    components.append(pad_literal)
                column_args.append(f"{column_type}({', '.join(components)})")

            helper_name = self.saturate_matrix_helper_name(dtype, columns, rows)
            code = f"def {helper_name}(m: {matrix_type}) -> {matrix_type}:\n"
            code += f"    return {matrix_type}({', '.join(column_args)})\n\n"
            return code

        source_width = width_or_columns
        storage_width = storage_width_or_rows
        vector_type = f"SIMD[{dtype}, {storage_width}]"
        components = [
            self.saturate_value_expression(f"v[{index}]")
            for index in range(source_width)
        ]
        if storage_width > source_width:
            components.append(pad_literal)

        helper_name = self.saturate_vector_helper_name(
            dtype, source_width, storage_width
        )
        code = f"def {helper_name}(v: {vector_type}) -> {vector_type}:\n"
        code += f"    return {vector_type}({', '.join(components)})\n\n"
        return code

    def saturate_value_expression(self, value):
        return f"0.0 if {value} < 0.0 else (1.0 if {value} > 1.0 else {value})"

    def saturate_scalar_helper_name(self, dtype):
        return f"_crossgl_saturate_{MOJO_DTYPE_SUFFIX[dtype]}"

    def saturate_vector_helper_name(self, dtype, source_width, storage_width):
        dtype_suffix = MOJO_DTYPE_SUFFIX[dtype]
        return f"_crossgl_saturate_{dtype_suffix}_{source_width}_{storage_width}"

    def saturate_matrix_helper_name(self, dtype, columns, rows):
        dtype_suffix = MOJO_DTYPE_SUFFIX[dtype]
        return f"_crossgl_saturate_{dtype_suffix}_c{columns}_r{rows}"

    def generate_vector_binary_helper(self, dtype, op, helper_kind):
        scalar_type, _, pad_literal = MOJO_DTYPE_INFO[dtype]
        mojo_scalar_type = self.map_type(scalar_type)
        vector_type = f"SIMD[{dtype}, 4]"
        helper_name = self.vector_binary_helper_name(dtype, op, helper_kind)

        if helper_kind == "vv":
            params = f"a: {vector_type}, b: {vector_type}"
            components = [f"a[{index}] {op} b[{index}]" for index in range(3)]
        elif helper_kind == "vs":
            params = f"v: {vector_type}, s: {mojo_scalar_type}"
            components = [f"v[{index}] {op} s" for index in range(3)]
        else:
            params = f"s: {mojo_scalar_type}, v: {vector_type}"
            components = [f"s {op} v[{index}]" for index in range(3)]

        components.append(pad_literal)
        args = ", ".join(components)
        code = f"def {helper_name}({params}) -> {vector_type}:\n"
        code += f"    return {vector_type}({args})\n\n"
        return code

    def vector_binary_helper_name(self, dtype, op, helper_kind):
        op_name = MOJO_VECTOR_ARITHMETIC_OPS[op]
        dtype_suffix = MOJO_DTYPE_SUFFIX[dtype]
        return f"_crossgl_vec3_{op_name}_{dtype_suffix}_{helper_kind}"

    def generate_select_helper(self, key):
        dtype, source_width, storage_width = key
        _, _, pad_literal = MOJO_DTYPE_INFO[dtype]
        vector_type = f"SIMD[{dtype}, {storage_width}]"
        mask_type = f"SIMD[DType.bool, {storage_width}]"
        components = [
            f"true_value[{index}] if mask[{index}] else false_value[{index}]"
            for index in range(source_width)
        ]
        if storage_width > source_width:
            components.append(pad_literal)

        helper_name = self.select_helper_name(dtype, source_width, storage_width)
        code = (
            f"def {helper_name}(mask: {mask_type}, true_value: {vector_type}, "
            f"false_value: {vector_type}) -> {vector_type}:\n"
        )
        code += f"    return {vector_type}({', '.join(components)})\n\n"
        return code

    def select_helper_name(self, dtype, source_width, storage_width):
        dtype_suffix = MOJO_DTYPE_SUFFIX[dtype]
        return f"_crossgl_select_{dtype_suffix}_{source_width}_{storage_width}"

    def generate_vec3_splat_helper(self, dtype):
        scalar_type, _, pad_literal = MOJO_DTYPE_INFO[dtype]
        mojo_scalar_type = self.map_type(scalar_type)
        vector_type = f"SIMD[{dtype}, 4]"
        helper_name = self.vec3_splat_helper_name(dtype)
        code = f"def {helper_name}(s: {mojo_scalar_type}) -> {vector_type}:\n"
        code += f"    return {vector_type}(s, s, s, {pad_literal})\n\n"
        return code

    def vec3_splat_helper_name(self, dtype):
        return f"_crossgl_vec3_splat_{MOJO_DTYPE_SUFFIX[dtype]}"

    def generate_swizzle_helper(self, dtype, source_width, member):
        _, _, pad_literal = MOJO_DTYPE_INFO[dtype]
        swizzle_indices = self.get_swizzle_indices(member)
        result_width = 2 if len(swizzle_indices) == 2 else 4
        source_type = f"SIMD[{dtype}, {source_width}]"
        result_type = f"SIMD[{dtype}, {result_width}]"
        helper_name = self.swizzle_helper_name(dtype, source_width, member)
        components = [f"v[{index}]" for index in swizzle_indices]
        if len(swizzle_indices) == 3:
            components.append(pad_literal)

        code = f"def {helper_name}(v: {source_type}) -> {result_type}:\n"
        code += f"    return {result_type}({', '.join(components)})\n\n"
        return code

    def swizzle_helper_name(self, dtype, source_width, member):
        dtype_suffix = MOJO_DTYPE_SUFFIX[dtype]
        return f"_crossgl_swizzle_{dtype_suffix}_{source_width}_{member}"

    def generate_constructor_helper_call(self, func_name, args):
        dtype, source_width, storage_width, pad_literal = self.vector_constructor_info[
            func_name
        ]
        pieces = []

        for arg in args:
            piece = self.constructor_piece_for_expression(arg, dtype)
            if piece is None:
                return None
            pieces.append(piece)

        pieces = self.select_constructor_pieces(pieces, source_width, dtype)
        if pieces is None:
            return None

        has_duplicate_sensitive_vector = any(
            piece["kind"] == "vector" and piece["duplicate_sensitive"]
            for piece in pieces
        )

        if not has_duplicate_sensitive_vector:
            return None

        key = self.constructor_helper_key(dtype, source_width, storage_width, pieces)
        helper_name = self.constructor_helper_name(key)
        self.required_constructor_helpers[key] = {
            "key": key,
            "dtype": dtype,
            "storage_width": storage_width,
            "pad_literal": pad_literal,
            "pieces": pieces,
        }

        call_args = [
            self.generate_constructor_helper_argument(
                piece, dtype, f"constructor helper argument {index}"
            )
            for index, piece in enumerate(pieces, start=1)
        ]
        return f"{helper_name}({', '.join(call_args)})"

    def select_constructor_pieces(self, pieces, source_width, target_dtype=None):
        selected = []
        remaining = source_width

        for piece in pieces:
            if remaining == 0:
                break
            if piece["kind"] == "vector":
                indices = piece["indices"][:remaining]
                if indices:
                    selected.append({**piece, "indices": tuple(indices)})
                    remaining -= len(indices)
            else:
                selected.append(piece)
                remaining -= 1

        if remaining != 0:
            if target_dtype is None:
                return None
            zero_literal = MOJO_DTYPE_INFO[target_dtype][2]
            selected.extend(
                {
                    "kind": "scalar",
                    "expr": zero_literal,
                    "dtype": target_dtype,
                }
                for _ in range(remaining)
            )
        return selected

    def constructor_piece_for_expression(self, expr, target_dtype):
        if isinstance(expr, MemberAccessNode):
            swizzle_indices = self.get_swizzle_indices(expr.member)
            if swizzle_indices is not None:
                source_type = self.expression_result_type(expr.object)
                source_info = self.vector_type_info(source_type)
                if source_info is None:
                    return None
                return {
                    "kind": "vector",
                    "dtype": source_info[0],
                    "storage_width": source_info[2],
                    "indices": tuple(swizzle_indices),
                    "expr": expr.object,
                    "duplicate_sensitive": self.is_duplicate_sensitive_expression(
                        expr.object
                    ),
                }

        expr_type = self.expression_result_type(expr)
        info = self.vector_type_info(expr_type)
        if info is not None:
            _, source_width, storage_width, _ = info
            return {
                "kind": "vector",
                "dtype": info[0],
                "storage_width": storage_width,
                "indices": tuple(range(source_width)),
                "expr": expr,
                "duplicate_sensitive": self.is_duplicate_sensitive_expression(expr),
            }

        return {
            "kind": "scalar",
            "expr": expr,
            "dtype": self.expression_mojo_dtype(expr),
        }

    def generate_constructor_helper_argument(self, piece, target_dtype, context):
        target_type = self.constructor_helper_argument_target_type(piece, target_dtype)
        self.validate_expression_target_shape(piece["expr"], target_type, context)
        return self.generate_expression(piece["expr"], target_type, context)

    def generate_scalar_constructor_argument(self, expr, target_dtype, context):
        if target_dtype in {"DType.float16", "DType.float32"} and (
            self.is_scalar_bool_expression(expr)
        ):
            # Mojo's Float16/Float32 constructors do not accept Bool directly.
            # Converting through Int preserves GLSL's 0/1 numeric cast.
            return f"Int({self.generate_expression(expr)})"

        target_type = self.constructor_scalar_expression_target_type(
            expr, target_dtype, force_scalar_target=True
        )
        self.validate_expression_target_shape(expr, target_type, context)
        return self.generate_expression(expr, target_type, context)

    def constructor_helper_argument_target_type(self, piece, target_dtype):
        if piece["kind"] == "scalar" and self.image_result_target_types_in_expression(
            piece["expr"]
        ):
            return MOJO_DTYPE_INFO[target_dtype][0]

        expr_type = self.expression_result_type(piece["expr"])
        if expr_type is not None:
            return expr_type

        if piece["kind"] == "scalar":
            piece_dtype = piece.get("dtype")
            return self.constructor_scalar_expression_target_type(
                piece["expr"], target_dtype, piece_dtype
            )

        return None

    def constructor_scalar_expression_target_type(
        self, expr, target_dtype, fallback_dtype=None, force_scalar_target=False
    ):
        if self.image_result_target_types_in_expression(expr):
            return MOJO_DTYPE_INFO[target_dtype][0]

        expr_type = self.expression_result_type(expr)
        if expr_type is not None:
            if force_scalar_target and self.vector_type_info(expr_type) is not None:
                resource_targets = self.expression_target_types_in_expression(expr)
                if any(
                    self.vector_type_info(resource_target) is not None
                    for resource_target in resource_targets
                ):
                    return MOJO_DTYPE_INFO[target_dtype][0]
            return expr_type

        resource_targets = self.expression_target_types_in_expression(expr)
        if any(
            self.vector_type_info(resource_target) is not None
            for resource_target in resource_targets
        ):
            return MOJO_DTYPE_INFO[target_dtype][0]
        if "int" in resource_targets:
            return "int"

        dtype = fallback_dtype if fallback_dtype is not None else target_dtype
        return MOJO_DTYPE_INFO[dtype][0]

    def resource_size_target_types_in_expression(self, expr):
        size_info = self.resource_size_expression_info(expr)
        if size_info is not None:
            _, _, expected_type = size_info
            return [expected_type] if expected_type is not None else []

        if isinstance(expr, TernaryOpNode):
            return [
                *self.resource_size_target_types_in_expression(expr.true_expr),
                *self.resource_size_target_types_in_expression(expr.false_expr),
            ]

        if isinstance(expr, MatchNode):
            target_types = []
            for arm in getattr(expr, "arms", []) or []:
                arm_expr, _ = self.match_arm_value_expression(arm)
                target_types.extend(
                    self.resource_size_target_types_in_expression(arm_expr)
                )
            return target_types

        return []

    def image_result_target_types_in_expression(self, expr):
        image_info = self.image_result_expression_info(expr)
        if image_info is not None:
            _, _, expected_type = image_info
            return [expected_type] if expected_type is not None else []

        if isinstance(expr, TernaryOpNode):
            return [
                *self.image_result_target_types_in_expression(expr.true_expr),
                *self.image_result_target_types_in_expression(expr.false_expr),
            ]

        if isinstance(expr, MatchNode):
            target_types = []
            for arm in getattr(expr, "arms", []) or []:
                arm_expr, _ = self.match_arm_value_expression(arm)
                target_types.extend(
                    self.image_result_target_types_in_expression(arm_expr)
                )
            return target_types

        return []

    def buffer_result_target_types_in_expression(self, expr):
        buffer_info = self.buffer_result_expression_info(expr)
        if buffer_info is not None:
            _, _, expected_type = buffer_info
            return [expected_type] if expected_type is not None else []

        if isinstance(expr, TernaryOpNode):
            return [
                *self.buffer_result_target_types_in_expression(expr.true_expr),
                *self.buffer_result_target_types_in_expression(expr.false_expr),
            ]

        if isinstance(expr, MatchNode):
            target_types = []
            for arm in getattr(expr, "arms", []) or []:
                arm_expr, _ = self.match_arm_value_expression(arm)
                target_types.extend(
                    self.buffer_result_target_types_in_expression(arm_expr)
                )
            return target_types

        return []

    def reinterpret_result_target_types_in_expression(self, expr):
        reinterpret_info = self.reinterpret_result_expression_info(expr)
        if reinterpret_info is not None:
            _, expected_type = reinterpret_info
            return [expected_type] if expected_type is not None else []

        if isinstance(expr, TernaryOpNode):
            return [
                *self.reinterpret_result_target_types_in_expression(expr.true_expr),
                *self.reinterpret_result_target_types_in_expression(expr.false_expr),
            ]

        if isinstance(expr, MatchNode):
            target_types = []
            for arm in getattr(expr, "arms", []) or []:
                arm_expr, _ = self.match_arm_value_expression(arm)
                target_types.extend(
                    self.reinterpret_result_target_types_in_expression(arm_expr)
                )
            return target_types

        return []

    def expression_target_types_in_expression(self, expr):
        return [
            *self.resource_size_target_types_in_expression(expr),
            *self.image_result_target_types_in_expression(expr),
            *self.buffer_result_target_types_in_expression(expr),
            *self.reinterpret_result_target_types_in_expression(expr),
        ]

    def constructor_helper_key(self, dtype, source_width, storage_width, pieces):
        signature = []
        for piece in pieces:
            if piece["kind"] == "vector":
                signature.append(
                    (
                        "v",
                        piece["dtype"],
                        piece["storage_width"],
                        piece["indices"],
                    )
                )
            else:
                piece_dtype = piece.get("dtype")
                if piece_dtype is not None and piece_dtype != dtype:
                    signature.append(("s", piece_dtype))
                else:
                    signature.append(("s",))
        return (dtype, source_width, storage_width, tuple(signature))

    def constructor_helper_name(self, key):
        dtype, _, storage_width, signature = key
        parts = []
        for piece in signature:
            if piece[0] == "v":
                _, piece_dtype, piece_storage_width, indices = piece
                index_text = "".join(str(index) for index in indices)
                parts.append(
                    f"v{MOJO_DTYPE_SUFFIX[piece_dtype]}{piece_storage_width}_{index_text}"
                )
            elif len(piece) > 1:
                parts.append(f"s{MOJO_DTYPE_SUFFIX[piece[1]]}")
            else:
                parts.append("s")
        suffix = "_".join(parts)
        return f"_crossgl_construct_{MOJO_DTYPE_SUFFIX[dtype]}_{storage_width}_{suffix}"

    def generate_constructor_helper(self, helper):
        dtype = helper["dtype"]
        scalar_type, _, _ = MOJO_DTYPE_INFO[dtype]
        mojo_scalar_type = self.map_type(scalar_type)
        result_type = f"SIMD[{dtype}, {helper['storage_width']}]"
        params = []
        components = []
        prelude = []

        for index, piece in enumerate(helper["pieces"]):
            if piece["kind"] == "vector":
                param_name = f"v{index}"
                vector_type = f"SIMD[{piece['dtype']}, {piece['storage_width']}]"
                params.append(f"{param_name}: {vector_type}")
                vector_expr = param_name
                if piece["dtype"] != dtype:
                    vector_expr = f"{param_name}_cast"
                    prelude.append(
                        f"    var {vector_expr} = {param_name}.cast[{dtype}]()\n"
                    )
                components.extend(
                    f"{vector_expr}[{component_index}]"
                    for component_index in piece["indices"]
                )
            else:
                param_name = f"s{index}"
                piece_dtype = piece.get("dtype")
                param_scalar_type = mojo_scalar_type
                if piece_dtype is not None and piece_dtype != dtype:
                    scalar_type = MOJO_DTYPE_INFO[piece_dtype][0]
                    param_scalar_type = self.map_type(scalar_type)
                params.append(f"{param_name}: {param_scalar_type}")
                components.append(
                    self.cast_scalar_text(param_name, piece_dtype, dtype)
                    if piece_dtype is not None
                    else param_name
                )

        if helper["pad_literal"] is not None and len(components) == 3:
            components.append(helper["pad_literal"])

        helper_name = self.constructor_helper_name(helper["key"])
        code = f"def {helper_name}({', '.join(params)}) -> {result_type}:\n"
        code += "".join(prelude)
        code += f"    return {result_type}({', '.join(components)})\n\n"
        return code
