"""Numeric helpers in the GLSL generator: cooperative matrix transfers, boolean ordering and IEEE helpers, complex64 values, half helpers and registered structures."""

from ..ast import MemberAccessNode
from ..refusals import OpenGLRefusal
from ..semantics.half_helpers import glsl_source as glsl_half_helper_source
from ..structure_conversions import (
    ScalarKind,
    StructureConversionKind,
    StructureFieldValue,
    registered_structure_conversion_for_identity,
)
from .cooperative_matrix_diagonal import (
    generate_glsl_cooperative_matrix_diagonal_helper,
)
from .glsl_precise_odd import glsl_precise_odd_helper


class OpenGLNumericHelperMixin:
    """Numeric helper members of ``GLSLCodeGen``."""

    def generate_glsl_cooperative_matrix_transfer(self, node):
        """Lower simdgroup_load/simdgroup_store to lane-local buffer accesses.

        Lane ``l`` holds the elements the fragment mapping assigns it, so a
        transfer needs no cross-lane exchange: each element reads or writes
        ``pointer[row * stride + column]``, or ``pointer[column * stride +
        row]`` for a transposed matrix.
        """
        operation = node.operation
        arguments = list(node.arguments)
        if not 2 <= len(arguments) <= 5:
            self.glsl_cooperative_matrix_operation_error(
                node,
                f"OpenGL cooperative-matrix {operation} expects 2 to 5 arguments, "
                f"got {len(arguments)}",
                reason="invalid-argument-count",
            )
        if operation == "load":
            matrix, pointer = arguments[:2]
        else:
            pointer, matrix = arguments[:2]
        stride = arguments[2] if len(arguments) > 2 else None
        origin = arguments[3] if len(arguments) > 3 else None
        transpose = self.glsl_cooperative_matrix_literal_bool(
            arguments[4] if len(arguments) > 4 else None
        )
        contract = self.glsl_cooperative_matrix_expression_contract(
            matrix, operation=operation
        )
        if contract is None:
            self.glsl_cooperative_matrix_operation_error(
                node,
                f"OpenGL cooperative-matrix {operation} requires a matrix with a "
                "complete lane-fragment contract",
                reason="missing-operand-contract",
            )
        mapping = contract["fragment_mapping_contract"]
        closed_form = self.GLSL_COOPERATIVE_MATRIX_TRANSFER_MAPPINGS.get(
            contract["fragment_mapping"]
        )
        if (
            closed_form is None
            or (contract["rows"], contract["columns"]) != (8, 8)
            or (contract["subgroup_size"], contract["elements_per_lane"]) != (32, 2)
            or any(
                mapping.coordinate(lane, element)
                != self.glsl_cooperative_matrix_transfer_coordinate(lane, element)
                for lane in range(32)
                for element in range(2)
            )
        ):
            self.glsl_cooperative_matrix_operation_error(
                node,
                f"OpenGL cooperative-matrix {operation} only supports the "
                "tile_4x4_row_pair 8x8/32x2 fragment contract",
                reason="incompatible-transfer-contract",
                matrix_type=contract["matrix_type"],
            )
        if self.software_subgroup_width != 32:
            self.glsl_cooperative_matrix_operation_error(
                node,
                f"OpenGL cooperative-matrix {operation} requires 32-lane software "
                "subgroups",
                reason="exact-subgroup-width-required",
                matrix_type=contract["matrix_type"],
            )
        if transpose is None:
            self.glsl_cooperative_matrix_operation_error(
                node,
                f"OpenGL cooperative-matrix {operation} requires a literal "
                "transpose flag",
                reason="dynamic-transpose",
                matrix_type=contract["matrix_type"],
            )
        if not self.glsl_cooperative_matrix_zero_origin(origin):
            self.glsl_cooperative_matrix_operation_error(
                node,
                f"OpenGL cooperative-matrix {operation} only supports a zero "
                "matrix origin",
                reason="nonzero-matrix-origin",
                matrix_type=contract["matrix_type"],
            )
        for label, value in (("matrix", matrix), ("pointer", pointer)):
            if not self.glsl_cooperative_matrix_pure_expression(value):
                self.glsl_cooperative_matrix_operation_error(
                    node,
                    f"OpenGL cooperative-matrix {operation} {label} must be free "
                    "of side effects",
                    reason="impure-transfer-operand",
                    matrix_type=contract["matrix_type"],
                )
        if stride is not None and not self.glsl_cooperative_matrix_pure_expression(
            stride
        ):
            self.glsl_cooperative_matrix_operation_error(
                node,
                f"OpenGL cooperative-matrix {operation} stride must be free of "
                "side effects",
                reason="impure-transfer-operand",
                matrix_type=contract["matrix_type"],
            )
        if operation == "load" and not self.glsl_reference_argument_is_lvalue(matrix):
            self.glsl_cooperative_matrix_operation_error(
                node,
                "OpenGL cooperative-matrix load destination must be assignable",
                reason="destination-requires-lvalue",
                matrix_type=contract["matrix_type"],
            )
        binding = self.glsl_storage_pointer_binding(
            pointer, self.glsl_storage_pointer_aliases()
        )
        if binding is None or binding.get("pointer_reinterpretation") is not None:
            self.glsl_cooperative_matrix_operation_error(
                node,
                f"OpenGL cooperative-matrix {operation} requires a storage-buffer "
                "pointer with the matrix component type",
                reason="unsupported-transfer-pointer",
                matrix_type=contract["matrix_type"],
            )
        element_type = binding.get("element_type") if binding is not None else None
        if (
            element_type is None
            or self.map_type(element_type) != contract["mapped_component_type"]
        ):
            self.glsl_cooperative_matrix_operation_error(
                node,
                f"OpenGL cooperative-matrix {operation} requires a storage-buffer "
                "pointer with the matrix component type",
                reason="unsupported-transfer-pointer",
                matrix_type=contract["matrix_type"],
            )
        if operation == "store" and binding.get("access") == "read":
            self.glsl_cooperative_matrix_operation_error(
                node,
                "OpenGL cooperative-matrix store requires a writable storage buffer",
                reason="read-only-transfer-pointer",
                matrix_type=contract["matrix_type"],
            )
        helper = self.glsl_cooperative_matrix_helper_name(contract, "fragment_offset")
        stride_text = (
            f"uint({self.generate_expression(stride)})"
            if stride is not None
            else f"{contract['columns']}u"
        )
        transpose_text = "true" if transpose else "false"
        matrix_text = self.generate_expression(matrix)
        root = binding["root"]
        base = self.glsl_workgroup_pointer_offset_expression(binding)
        parts = []
        for element in range(contract["elements_per_lane"]):
            offset = f"int({helper}({element}u, {stride_text}, {transpose_text}))"
            index = offset if base in {"0", "0u"} else f"({base} + {offset})"
            fragment = f"{matrix_text}.elements[{element}]"
            if operation == "load":
                parts.append(f"{fragment} = {root}[{index}]")
            else:
                parts.append(f"{root}[{index}] = {fragment}")
        return f"({', '.join(parts)})"

    @staticmethod
    def glsl_cooperative_matrix_uint_array(name, values):
        rendered = [f"{value}u" for value in values]
        chunks = [
            ", ".join(rendered[index : index + 8])
            for index in range(0, len(rendered), 8)
        ]
        initializer = ",\n        ".join(chunks)
        return (
            f"    const uint {name}[{len(values)}] = uint[{len(values)}](\n"
            f"        {initializer}\n"
            "    );"
        )

    def generate_glsl_cooperative_matrix_multiply_accumulate_helper(
        self,
        helper_name,
        result_contract,
        left_contract,
        right_contract,
        accumulator_contract,
    ):
        if self.software_subgroup_width == 32:
            return self.generate_glsl_software_matrix_multiply_accumulate_helper(
                helper_name,
                result_contract,
                left_contract,
                right_contract,
                accumulator_contract,
            )
        mapping = result_contract["fragment_mapping_contract"]
        owner_lanes = []
        owner_elements = []
        for row in range(mapping.rows):
            for column in range(mapping.columns):
                lane, element = mapping.owner(row, column)
                owner_lanes.append(lane)
                owner_elements.append(element)

        result_coordinates = []
        for element in range(mapping.elements_per_lane):
            result_coordinates.append(
                [
                    row * mapping.columns + column
                    for lane in range(mapping.subgroup_size)
                    for row, column in [mapping.coordinate(lane, element)]
                ]
            )

        result_type = result_contract["type_name"]
        left_type = left_contract["type_name"]
        right_type = right_contract["type_name"]
        accumulator_type = accumulator_contract["type_name"]
        component_type = result_contract["mapped_component_type"]
        lines = [
            f"{result_type} {helper_name}(",
            f"    {left_type} left, {right_type} right,",
            f"    {accumulator_type} accumulator) {{",
            self.glsl_cooperative_matrix_uint_array("crossglOwnerLanes", owner_lanes),
            self.glsl_cooperative_matrix_uint_array(
                "crossglOwnerElements", owner_elements
            ),
        ]
        for element, coordinates in enumerate(result_coordinates):
            lines.append(
                self.glsl_cooperative_matrix_uint_array(
                    f"crossglResultCoordinates{element}", coordinates
                )
            )
        lines.append(f"    {result_type} result = accumulator;")
        rounded = self.cooperative_matrix_mma_profile is not None

        for element in range(mapping.elements_per_lane):
            if rounded:
                lines.append(
                    f"    uint sum{element} = floatBitsToUint(accumulator.elements[{element}]);"
                )
            lines.extend(
                [
                    f"    uint crossglOutputCoordinate{element} = "
                    f"crossglResultCoordinates{element}[gl_SubgroupInvocationID];",
                    f"    uint crossglOutputRow{element} = "
                    f"crossglOutputCoordinate{element} / {mapping.columns}u;",
                    f"    uint crossglOutputColumn{element} = "
                    f"crossglOutputCoordinate{element} % {mapping.columns}u;",
                ]
            )
            for inner in range(left_contract["columns"]):
                prefix = f"crossgl{element}_{inner}"
                lines.extend(
                    [
                        f"    uint {prefix}LeftIndex = "
                        f"crossglOutputRow{element} * {left_contract['columns']}u + "
                        f"{inner}u;",
                        f"    uint {prefix}RightIndex = {inner}u * "
                        f"{right_contract['columns']}u + "
                        f"crossglOutputColumn{element};",
                    ]
                )
                for local_element in range(left_contract["elements_per_lane"]):
                    lines.append(
                        f"    {component_type} {prefix}Left{local_element} = "
                        f"subgroupShuffle(left.elements[{local_element}], "
                        f"crossglOwnerLanes[{prefix}LeftIndex]);"
                    )
                for local_element in range(right_contract["elements_per_lane"]):
                    lines.append(
                        f"    {component_type} {prefix}Right{local_element} = "
                        f"subgroupShuffle(right.elements[{local_element}], "
                        f"crossglOwnerLanes[{prefix}RightIndex]);"
                    )
                left_values = " : ".join(
                    f"{prefix}Left{local_element}"
                    for local_element in range(left_contract["elements_per_lane"])
                )
                right_values = " : ".join(
                    f"{prefix}Right{local_element}"
                    for local_element in range(right_contract["elements_per_lane"])
                )
                lines.extend(
                    [
                        f"    {component_type} {prefix}LeftValue = "
                        f"crossglOwnerElements[{prefix}LeftIndex] == 0u ? "
                        f"{left_values};",
                        f"    {component_type} {prefix}RightValue = "
                        f"crossglOwnerElements[{prefix}RightIndex] == 0u ? "
                        f"{right_values};",
                    ]
                )
                if rounded:
                    lines.extend(
                        self.glsl_cooperative_matrix_rounded_term(
                            f"{prefix}LeftValue",
                            f"{prefix}RightValue",
                            f"sum{element}",
                            prefix,
                        )
                    )
                else:
                    lines.append(
                        f"    result.elements[{element}] += "
                        f"{prefix}LeftValue * {prefix}RightValue;"
                    )
            if rounded:
                lines.append(
                    f"    result.elements[{element}] = uintBitsToFloat(sum{element});"
                )
        lines.extend(["    return result;", "}"])
        return "\n".join(lines) + "\n"

    def glsl_cooperative_matrix_rounding_names(self):
        from ..fused_math import FMA_HELPER_KEYS

        return {
            key: self.glsl_generated_module_identifier(
                ("cooperative-matrix-rounding", key), f"crossgl_matrix_round_{key}"
            )
            for key in FMA_HELPER_KEYS
        }

    def glsl_cooperative_matrix_rounded_term(self, left, right, accumulator, prefix):
        helper = self.glsl_cooperative_matrix_rounding_names()["bits"]
        flush = (
            "true"
            if self.cooperative_matrix_mma_profile.endswith("-flush")
            else "false"
        )
        lines = [
            f"    uint {prefix}LeftBits = floatBitsToUint({left});",
            f"    uint {prefix}RightBits = floatBitsToUint({right});",
        ]
        if self.cooperative_matrix_mma_profile.startswith("fma-"):
            lines.append(
                f"    {accumulator} = {helper}({prefix}LeftBits, {prefix}RightBits, "
                f"{accumulator}, {flush});"
            )
        else:
            # Preserve the product's signed zero across separate rounding boundaries.
            lines.extend(
                [
                    f"    uint {prefix}Product = {helper}({prefix}LeftBits, {prefix}RightBits, "
                    f"({prefix}LeftBits ^ {prefix}RightBits) & 0x80000000u, {flush});",
                    f"    {accumulator} = {helper}({accumulator}, 0x3f800000u, {prefix}Product, {flush});",
                ]
            )
        return lines

    def generate_glsl_software_matrix_multiply_accumulate_helper(
        self,
        helper_name,
        result_contract,
        left_contract,
        right_contract,
        accumulator_contract,
    ):
        mapping = result_contract["fragment_mapping_contract"]
        scratch = [
            self.glsl_generated_module_identifier(
                ("cooperative-matrix-scratch", helper_name, operand),
                f"{helper_name}_{operand}",
            )
            for operand in ("left", "right")
        ]
        count = self.glsl_software_subgroup_workgroup_invocation_count
        rounded = self.cooperative_matrix_mma_profile is not None
        lines = [f"shared vec2 {name}[{count}];" for name in scratch]
        lines.extend(
            [
                f"{result_contract['type_name']} {helper_name}(",
                f"    {left_contract['type_name']} left, {right_contract['type_name']} right,",
                f"    {accumulator_contract['type_name']} accumulator) {{",
                "    uint invocation = gl_LocalInvocationIndex;",
                "    uint lane = invocation % 32u;",
                "    uint subgroupBase = invocation - lane;",
                f"    {scratch[0]}[invocation] = vec2(left.elements[0], left.elements[1]);",
                f"    {scratch[1]}[invocation] = vec2(right.elements[0], right.elements[1]);",
                "    barrier();",
                f"    {result_contract['type_name']} result = accumulator;",
            ]
        )
        # Registered source ownership is independent of the driver's subgroup layout.
        owners = [mapping.owner(row, column) for row in range(8) for column in range(8)]
        tables = {
            "ownerLanes": [owner[0] for owner in owners],
            "ownerElements": [owner[1] for owner in owners],
        }
        for element in range(2):
            tables[f"coordinates{element}"] = [
                row * 8 + column
                for lane in range(32)
                for row, column in [mapping.coordinate(lane, element)]
            ]
        for name, values in tables.items():
            lines.append(self.glsl_cooperative_matrix_uint_array(name, values))
        for element in range(2):
            lines.extend(
                [
                    f"    uint row{element} = coordinates{element}[lane] / 8u;",
                    f"    uint column{element} = coordinates{element}[lane] % 8u;",
                    (
                        f"    uint sum{element} = floatBitsToUint(accumulator.elements[{element}]);"
                        if rounded
                        else f"    precise float sum{element} = accumulator.elements[{element}];"
                    ),
                ]
            )
            for inner in range(8):
                prefix = f"term{element}_{inner}"
                lines.extend(
                    [
                        f"    uint {prefix}A = row{element} * 8u + {inner}u;",
                        f"    uint {prefix}B = {inner}u * 8u + column{element};",
                    ]
                )
                left = f"{scratch[0]}[subgroupBase + ownerLanes[{prefix}A]][ownerElements[{prefix}A]]"
                right = f"{scratch[1]}[subgroupBase + ownerLanes[{prefix}B]][ownerElements[{prefix}B]]"
                if rounded:
                    lines.extend(
                        self.glsl_cooperative_matrix_rounded_term(
                            left, right, f"sum{element}", prefix
                        )
                    )
                else:
                    lines.extend(
                        [
                            f"    precise float {prefix} = {left} * {right};",
                            f"    sum{element} = sum{element} + {prefix};",
                        ]
                    )
            value = f"uintBitsToFloat(sum{element})" if rounded else f"sum{element}"
            lines.append(f"    result.elements[{element}] = {value};")
        # Every invocation finishes reading before a subsequent call reuses storage.
        lines.extend(["    barrier();", "    return result;", "}"])
        return "\n".join(lines) + "\n"

    def generate_glsl_cooperative_matrix_software_support(self):
        if not self.cooperative_matrix_software_lowering:
            return ""
        blocks = []
        if self.cooperative_matrix_mma_profile is not None and any(
            record[1] == "multiply_accumulate"
            for record in self.required_glsl_cooperative_matrix_helpers
        ):
            blocks.append(self.generate_glsl_cooperative_matrix_rounding_support())
        for type_name, contract in sorted(
            self.glsl_cooperative_matrix_fragment_contracts.items()
        ):
            blocks.append(
                f"struct {type_name} {{\n"
                f"    {contract['mapped_component_type']} "
                f"elements[{contract['elements_per_lane']}];\n"
                "};\n"
            )

        operators = {
            "elementwise_add": "+",
            "elementwise_subtract": "-",
            "elementwise_multiply": "*",
        }
        for helper_record in sorted(self.required_glsl_cooperative_matrix_helpers):
            type_name, operation, *operand_type_names = helper_record
            contract = self.glsl_cooperative_matrix_fragment_contracts[type_name]
            if operation == "multiply_accumulate":
                left_contract, right_contract, accumulator_contract = (
                    self.glsl_cooperative_matrix_fragment_contracts[name]
                    for name in operand_type_names
                )
                helper_key = (
                    operation,
                    left_contract["contract_key"],
                    right_contract["contract_key"],
                    accumulator_contract["contract_key"],
                )
                blocks.append(
                    self.generate_glsl_cooperative_matrix_multiply_accumulate_helper(
                        contract["helpers"][helper_key],
                        contract,
                        left_contract,
                        right_contract,
                        accumulator_contract,
                    )
                )
                continue

            helper_name = contract["helpers"][operation]
            if operation == "diagonal":
                blocks.append(
                    generate_glsl_cooperative_matrix_diagonal_helper(
                        type_name, helper_name, contract
                    )
                )
                continue
            if operation == "fill":
                component = contract["mapped_component_type"]
                lines = [
                    f"{type_name} {helper_name}({component} value) {{",
                    f"    {type_name} result;",
                    *(
                        f"    result.elements[{index}] = value;"
                        for index in range(contract["elements_per_lane"])
                    ),
                    "    return result;",
                    "}",
                ]
                blocks.append("\n".join(lines) + "\n")
                continue
            if operation == "fragment_offset":
                row, column = self.GLSL_COOPERATIVE_MATRIX_TRANSFER_MAPPINGS[
                    contract["fragment_mapping"]
                ]
                lines = [
                    f"uint {helper_name}(uint element, uint stride, bool transpose) {{",
                    "    // The registered lane coordinates of this fragment element.",
                    "    uint lane = gl_LocalInvocationIndex % 32u;",
                    f"    uint row = {row};",
                    f"    uint column = {column};",
                    "    return transpose ? column * stride + row : row * stride + column;",
                    "}",
                ]
                blocks.append("\n".join(lines) + "\n")
                continue
            lines = [f"{type_name} {helper_name}("]
            if operation == "negate":
                lines[-1] += f"{type_name} value) {{"
            else:
                lines[-1] += f"{type_name} left, {type_name} right) {{"
            lines.append(f"    {type_name} result;")
            for index in range(contract["elements_per_lane"]):
                if operation == "negate":
                    expression = f"-value.elements[{index}]"
                else:
                    operator = operators[operation]
                    expression = (
                        f"left.elements[{index}] {operator} right.elements[{index}]"
                    )
                lines.append(f"    result.elements[{index}] = {expression};")
            lines.extend(["    return result;", "}"])
            blocks.append("\n".join(lines) + "\n")
        return ("\n".join(blocks) + "\n") if blocks else ""

    def generate_glsl_trailing_zero_helpers(self):
        helpers = []
        for operand_type in sorted(self.required_glsl_trailing_zero_helpers):
            helper_name = self.glsl_trailing_zero_helper_name(operand_type)
            if operand_type == "uint":
                helpers.append(f"""int {helper_name}(uint value) {{
    // Use the canonical least-bit sentinel for Clang's undefined zero case.
    return value == 0u ? -1 : findLSB(value);
}}
""")
            elif operand_type == "uint64_t":
                helpers.append(f"""int {helper_name}(uint64_t value) {{
    uint low_bits = uint(value);
    if (low_bits != 0u) {{
        return findLSB(low_bits);
    }}
    uint high_bits = uint(value >> 32);
    // Use the canonical least-bit sentinel for Clang's undefined zero case.
    return high_bits == 0u ? -1 : (32 + findLSB(high_bits));
}}
""")
        return "\n".join(helpers) + ("\n" if helpers else "")

    def generate_glsl_boolean_order_helpers(self):
        helpers = []
        components = "xyzw"
        for helper_name, operation, width in sorted(
            getattr(self, "required_glsl_boolean_order_helpers", set())
        ):
            operator = "&&" if operation == "min" else "||"
            values = ", ".join(
                f"left.{component} {operator} right.{component}"
                for component in components[:width]
            )
            helpers.append(
                f"bvec{width} {helper_name}(bvec{width} left, bvec{width} right) "
                f"{{\n    return bvec{width}({values});\n}}\n"
            )
        return ("\n".join(helpers) + "\n") if helpers else ""

    def glsl_half_bitcast_helper(self, operation):
        for builtin in ("floatBitsToUint", "uintBitsToFloat", "findMSB"):
            if (
                builtin in self.function_return_types
                or builtin in self.global_variable_types
            ):
                raise OpenGLRefusal(
                    f"OpenGL half bitcasts require unshadowed {builtin}"
                )
        operations = {
            "decode": ("decode",),
            "encode": ("encode",),
            "unpack": ("decode", "unpack"),
            "pack": ("encode", "pack"),
        }[operation]
        for required in operations:
            if required not in self.glsl_half_bitcast_helper_names:
                used_names = set(self.glsl_module_used_identifier_names)
                used_names.update(
                    getattr(self, "glsl_numeric_helper_reserved_names", set())
                )
                name = self.glsl_unique_identifier(
                    f"crossgl_half_bits_{required}", used_names
                )
                self.glsl_half_bitcast_helper_names[required] = name
                self.glsl_module_used_identifier_names.add(name)
        return self.glsl_half_bitcast_helper_names[operation]

    def generate_glsl_half_bitcast_helpers(self):
        names = self.glsl_half_bitcast_helper_names
        code = ""
        # The physical float32 carrier retains each binary16 sign and payload.
        # Floating pack/unpack builtins may quiet or canonicalize these NaNs.
        if "decode" in names:
            code += f"""float {names["decode"]}(uint value) {{
    uint sign = (value & 0x8000u) << 16u;
    uint exponent = (value >> 10u) & 31u;
    uint fraction = value & 1023u;
    if (exponent == 0u) {{
        if (fraction == 0u) {{ return uintBitsToFloat(sign); }}
        uint leading = uint(findMSB(fraction));
        uint bits = ((leading + 103u) << 23u) | ((fraction << (23u - leading)) & 0x7fffffu);
        return uintBitsToFloat(sign | bits);
    }}
    uint wideExponent = exponent == 31u ? 255u : exponent + 112u;
    return uintBitsToFloat(sign | (wideExponent << 23u) | (fraction << 13u));
}}
"""
        if "encode" in names:
            code += f"""uint {names["encode"]}(float value) {{
    uint bits = floatBitsToUint(value);
    uint sign = (bits >> 16u) & 0x8000u;
    uint exponent = (bits >> 23u) & 255u;
    uint fraction = bits & 0x7fffffu;
    if (exponent == 255u) {{ return sign | 0x7c00u | (fraction >> 13u); }}
    if (exponent < 103u) {{ return sign; }}
    if (exponent < 113u) {{ return sign | ((fraction | 0x800000u) >> (126u - exponent)); }}
    return sign | ((exponent - 112u) << 10u) | (fraction >> 13u);
}}
"""
        if "unpack" in names:
            code += (
                f'vec2 {names["unpack"]}(uint value) {{\n'
                f'    return vec2({names["decode"]}(value), {names["decode"]}(value >> 16u));\n'
                "}\n"
            )
        if "pack" in names:
            code += (
                f'uint {names["pack"]}(vec2 value) {{\n'
                f'    return {names["encode"]}(value.x) | ({names["encode"]}(value.y) << 16u);\n'
                "}\n"
            )
        return code

    def generate_glsl_half_helpers(self):
        if not self.glsl_half_helper_names:
            return ""
        scalar = self.glsl_half_helper_names[1]
        helpers = [f"""float {scalar}(float value) {{
    uint bits = floatBitsToUint(value);
    uint sign = bits & 0x80000000u;
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) {{
        return uintBitsToFloat(sign | (magnitude == 0x7f800000u ? 0x7f800000u : 0x7fc00000u));
    }}
    if (magnitude >= 0x477ff000u) {{
        return uintBitsToFloat(sign | 0x7f800000u);
    }}
    if (magnitude >= 0x38800000u) {{
        uint rounded = (magnitude + 0xfffu + ((magnitude >> 13u) & 1u)) & 0xffffe000u;
        return uintBitsToFloat(sign | rounded);
    }}
    if (magnitude < 0x33000000u) {{
        return uintBitsToFloat(sign);
    }}
    // Round the subnormal significand before rebuilding its exact float32 value.
    uint shift = 126u - (magnitude >> 23u);
    uint significand = (magnitude & 0x7fffffu) | 0x800000u;
    uint rounded = significand >> shift;
    uint remainder = significand & ((1u << shift) - 1u);
    uint midpoint = 1u << (shift - 1u);
    if (remainder > midpoint || (remainder == midpoint && (rounded & 1u) != 0u)) {{
        rounded += 1u;
    }}
    if (rounded == 0u) {{
        return uintBitsToFloat(sign);
    }}
    uint leading = uint(findMSB(rounded));
    uint result = ((leading + 103u) << 23u) | ((rounded << (23u - leading)) & 0x7fffffu);
    return uintBitsToFloat(sign | result);
}}
"""]
        for width, name in sorted(self.glsl_half_helper_names.items()):
            if width == 1:
                continue
            values = ", ".join(f"{scalar}(value.{c})" for c in "xyzw"[:width])
            helpers.append(
                f"vec{width} {name}(vec{width} value) {{\n"
                f"    return vec{width}({values});\n}}\n"
            )
        return "\n".join(helpers) + "\n"

    def generate_glsl_bfloat_integer_helpers(self):
        name = self.glsl_bfloat_integer_helper_name
        if name is None:
            return ""
        code = f"""float {name}(uint value) {{
    if (value < 256u) {{ return float(value); }}
    uint leading = uint(findMSB(value));
    uint shift = leading - 7u;
    uint retained = value >> shift;
    uint remainder = value & ((1u << shift) - 1u);
    uint midpoint = 1u << (shift - 1u);
    if (remainder > midpoint || (remainder == midpoint && (retained & 1u) != 0u)) {{
        retained += 1u;
    }}
    return uintBitsToFloat(((leading + 126u) << 23u) + (retained << 16u));
}}
float {name}(int value) {{
    uint magnitude = value < 0 ? 0u - uint(value) : uint(value);
    float rounded = {name}(magnitude);
    return value < 0 ? -rounded : rounded;
}}

"""
        if self.glsl_bfloat_wide_integer_helper_required:
            code += f"""float {name}(uint64_t value) {{
    if (value < uint64_t(256)) {{ return float(uint(value)); }}
    uint high = uint(value >> 32u);
    uint leading = high != 0u ? 32u + uint(findMSB(high)) : uint(findMSB(uint(value)));
    uint shift = leading - 7u;
    uint retained = uint(value >> shift);
    uint64_t remainder = value & ((uint64_t(1) << shift) - uint64_t(1));
    uint64_t midpoint = uint64_t(1) << (shift - 1u);
    if (remainder > midpoint || (remainder == midpoint && (retained & 1u) != 0u)) {{
        retained += 1u;
    }}
    return uintBitsToFloat(((leading + 126u) << 23u) + (retained << 16u));
}}
float {name}(int64_t value) {{
    uint64_t magnitude = value < int64_t(0) ? uint64_t(0) - uint64_t(value) : uint64_t(value);
    float rounded = {name}(magnitude);
    return value < int64_t(0) ? -rounded : rounded;
}}

"""
        return code

    def generate_glsl_bfloat_helpers(self):
        if not self.glsl_bfloat_helper_names:
            return ""
        scalar = self.glsl_bfloat_helper_names[1]
        helpers = [f"""float {scalar}(float value) {{
    uint bits = floatBitsToUint(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) {{
        return uintBitsToFloat((bits & 0xffff0000u) | 0x00400000u);
    }}
    uint bias = 0x7fffu + ((bits >> 16u) & 1u);
    return uintBitsToFloat((bits + bias) & 0xffff0000u);
}}
"""]
        for width, name in sorted(self.glsl_bfloat_helper_names.items()):
            if width == 1:
                continue
            values = ", ".join(f"{scalar}(value.{c})" for c in "xyzw"[:width])
            helpers.append(
                f"vec{width} {name}(vec{width} value) {{\n"
                f"    return vec{width}({values});\n}}\n"
            )
        return "\n".join(helpers) + "\n"

    def generate_glsl_ieee_math_helper(self, name, operation, value_type, width):
        if operation == "half-fma":
            if width == 1:
                return glsl_half_helper_source("cgl_half_fma", name)
            scalar = self.glsl_ieee_helper_name("half-fma", "float")
            lanes = ", ".join(
                f"{scalar}(left.{c}, right.{c}, addend.{c})" for c in "xyzw"[:width]
            )
            return (
                f"{value_type} {name}({value_type} left, {value_type} right, "
                f"{value_type} addend) {{\n    return {value_type}({lanes});\n}}\n"
            )
        if operation == "fma":
            # A precise result makes fma() one operation (GLSL 4.60, 8.3):
            # Mesa otherwise splits it and folds (a * b + c) - c into a * b.
            return (
                f"{value_type} {name}({value_type} left, {value_type} right, "
                f"{value_type} addend) {{\n"
                f"    precise {value_type} result = fma(left, right, addend);\n"
                "    return result;\n}\n"
            )
        if operation == "difference":
            return (
                f"{value_type} {name}({value_type} left, {value_type} right) {{\n"
                f"    precise {value_type} result = left - right;\n"
                "    return result;\n}\n"
            )
        if operation.startswith("ieee-"):
            return self.generate_glsl_ieee_operator_helper(name, operation)
        if operation == "atan2":
            return self.generate_glsl_atan2_helper(name, value_type, width)
        return glsl_precise_odd_helper(
            name, operation.removeprefix("precise-"), value_type, width
        )

    def generate_glsl_ieee_operator_helper(self, name, operation):
        """Binary32 arithmetic whose zero operands survive Mesa's folding."""
        operator = {
            "ieee-sum": "+",
            "ieee-difference": "-",
            "ieee-product": "*",
            "ieee-quotient": "/",
        }[operation]
        if operation in {"ieee-sum", "ieee-difference"}:
            # Two zeros sum to -0 only when both are -0 (round to nearest).
            right_sign = (
                "rightBits" if operation == "ieee-sum" else "(rightBits ^ 0x80000000u)"
            )
            zero = (
                "    if (((leftBits | rightBits) & 0x7fffffffu) == 0u) {\n"
                f"        return uintBitsToFloat(leftBits & {right_sign});\n"
                "    }\n"
            )
        elif operation == "ieee-product":
            zero = (
                "    uint leftMagnitude = leftBits & 0x7fffffffu;\n"
                "    uint rightMagnitude = rightBits & 0x7fffffffu;\n"
                "    if (leftMagnitude == 0u || rightMagnitude == 0u) {\n"
                "        uint other = leftMagnitude == 0u ? rightMagnitude : leftMagnitude;\n"
                "        return uintBitsToFloat(other >= 0x7f800000u\n"
                "            ? 0x7fc00000u : (leftBits ^ rightBits) & 0x80000000u);\n"
                "    }\n"
            )
        else:
            zero = (
                "    uint rightMagnitude = rightBits & 0x7fffffffu;\n"
                "    if ((leftBits & 0x7fffffffu) == 0u) {\n"
                "        return uintBitsToFloat(\n"
                "            rightMagnitude == 0u || rightMagnitude > 0x7f800000u\n"
                "            ? 0x7fc00000u : (leftBits ^ rightBits) & 0x80000000u);\n"
                "    }\n"
            )
        return (
            f"float {name}(float left, float right) {{\n"
            "    uint leftBits = floatBitsToUint(left);\n"
            "    uint rightBits = floatBitsToUint(right);\n"
            f"{zero}"
            f"    precise float result = left {operator} right;\n"
            "    return result;\n}\n"
        )

    def generate_glsl_atan2_helper(self, name, value_type, width):
        """``atan2`` with C99 Annex F zeros, infinities and NaN on binary32."""
        if width != 1:
            scalar = self.glsl_ieee_helper_name("atan2", "float")
            lanes = ", ".join(f"{scalar}(y.{c}, x.{c})" for c in "xyzw"[:width])
            return (
                f"{value_type} {name}({value_type} y, {value_type} x) {{\n"
                f"    return {value_type}({lanes});\n}}\n"
            )
        # GLSL leaves atan(y, x) undefined at y = x = 0, and Mesa drops the
        # sign of a zero y; integer tests settle every exactly defined case
        # and leave the others to the builtin.
        return f"""float {name}(float y, float x) {{
    uint yBits = floatBitsToUint(y);
    uint xBits = floatBitsToUint(x);
    uint yMagnitude = yBits & 0x7fffffffu;
    uint xMagnitude = xBits & 0x7fffffffu;
    uint sign = yBits & 0x80000000u;
    bool negativeX = (xBits & 0x80000000u) != 0u;
    if (yMagnitude > 0x7f800000u || xMagnitude > 0x7f800000u) {{
        return uintBitsToFloat(0x7fc00000u);
    }}
    if (yMagnitude == 0u) {{
        return uintBitsToFloat(sign | (negativeX ? 0x40490fdbu : 0u));
    }}
    if (xMagnitude == 0u) {{
        return uintBitsToFloat(sign | 0x3fc90fdbu);
    }}
    if (yMagnitude == 0x7f800000u) {{
        uint angle = xMagnitude == 0x7f800000u
            ? (negativeX ? 0x4016cbe4u : 0x3f490fdbu) : 0x3fc90fdbu;
        return uintBitsToFloat(sign | angle);
    }}
    if (xMagnitude == 0x7f800000u) {{
        return uintBitsToFloat(sign | (negativeX ? 0x40490fdbu : 0u));
    }}
    return atan(y, x);
}}
"""

    def generate_glsl_metal_math_helpers(self):
        helpers = []
        for name, operation, value_type, condition_type, width in sorted(
            self.required_glsl_metal_math_helpers
        ):
            if operation in {
                "fma",
                "half-fma",
                "difference",
                "atan2",
            } or operation.startswith(("precise-", "ieee-")):
                helpers.append(
                    self.generate_glsl_ieee_math_helper(
                        name, operation, value_type, width
                    )
                )
                continue
            parameters = f"{value_type} left, {value_type} right"
            if operation == "select":
                parameters += f", {condition_type} condition"
            values = []
            for component in ("",) if width == 1 else "xyzw"[:width]:
                suffix = f".{component}" if component else ""
                left, right = f"left{suffix}", f"right{suffix}"
                if operation == "select":
                    condition = "condition"
                    if condition_type != "bool":
                        condition += suffix
                    value = f"({condition} ? {right} : {left})"
                else:
                    operator = "<" if operation == "fmin" else ">"
                    value = (
                        f"(isnan({left}) ? {right} : (isnan({right}) ? {left} : "
                        f"({left} {operator} {right} ? {left} : {right})))"
                    )
                values.append(value)
            result = values[0] if width == 1 else f"{value_type}({', '.join(values)})"
            helpers.append(
                f"{value_type} {name}({parameters}) {{\n    return {result};\n}}\n"
            )
        return "\n".join(helpers) + ("\n" if helpers else "")

    def generate_glsl_complex64_helpers(self):
        if not getattr(self, "required_glsl_complex64_helpers", set()):
            return ""
        return """
complex64_t crossgl_complex64_negate(complex64_t value) {
    return complex64_t(-value.real, -value.imag);
}

complex64_t crossgl_complex64_add(complex64_t left, complex64_t right) {
    return complex64_t(left.real + right.real, left.imag + right.imag);
}

complex64_t crossgl_complex64_sub(complex64_t left, complex64_t right) {
    return complex64_t(left.real - right.real, left.imag - right.imag);
}

complex64_t crossgl_complex64_mul(complex64_t left, complex64_t right) {
    return complex64_t(
        (left.real * right.real) - (left.imag * right.imag),
        (left.real * right.imag) + (left.imag * right.real)
    );
}

complex64_t crossgl_complex64_div(complex64_t left, complex64_t right) {
    float denominator = (right.real * right.real) + (right.imag * right.imag);
    return complex64_t(
        ((left.real * right.real) + (left.imag * right.imag)) / denominator,
        ((left.imag * right.real) - (left.real * right.imag)) / denominator
    );
}

float crossgl_complex64_mod_component(float left, float right) {
    float value = left - (right * trunc(left / right));
    if ((value != 0.0) && ((value < 0.0) != (right < 0.0))) {
        value += right;
    }
    return value;
}

complex64_t crossgl_complex64_mod(complex64_t left, complex64_t right) {
    return complex64_t(
        crossgl_complex64_mod_component(left.real, right.real),
        crossgl_complex64_mod_component(left.imag, right.imag)
    );
}

bool crossgl_complex64_equal(complex64_t left, complex64_t right) {
    return (left.real == right.real) && (left.imag == right.imag);
}

bool crossgl_complex64_not_equal(complex64_t left, complex64_t right) {
    return !crossgl_complex64_equal(left, right);
}

bool crossgl_complex64_greater(complex64_t left, complex64_t right) {
    return (left.real > right.real)
        || ((left.real == right.real) && (left.imag > right.imag));
}

bool crossgl_complex64_greater_equal(complex64_t left, complex64_t right) {
    return (left.real > right.real)
        || ((left.real == right.real) && (left.imag >= right.imag));
}

bool crossgl_complex64_less(complex64_t left, complex64_t right) {
    return crossgl_complex64_greater(right, left);
}

bool crossgl_complex64_less_equal(complex64_t left, complex64_t right) {
    return crossgl_complex64_greater_equal(right, left);
}

complex64_t crossgl_complex64_add_assign(
    inout complex64_t left,
    complex64_t right
) {
    left = crossgl_complex64_add(left, right);
    return left;
}

complex64_t crossgl_complex64_sub_assign(
    inout complex64_t left,
    complex64_t right
) {
    left = crossgl_complex64_sub(left, right);
    return left;
}

complex64_t crossgl_complex64_mul_assign(
    inout complex64_t left,
    complex64_t right
) {
    left = crossgl_complex64_mul(left, right);
    return left;
}

complex64_t crossgl_complex64_div_assign(
    inout complex64_t left,
    complex64_t right
) {
    left = crossgl_complex64_div(left, right);
    return left;
}

complex64_t crossgl_complex64_mod_assign(
    inout complex64_t left,
    complex64_t right
) {
    left = crossgl_complex64_mod(left, right);
    return left;
}

"""

    def is_glsl_registered_complex64_type(self, type_text):
        binding = self.glsl_registered_scalar_structure_contract(type_text)
        return binding is not None and binding[0].destination_type == "complex64_t"

    def is_glsl_complex64_type(self, type_text):
        return (
            self.is_glsl_registered_complex64_type(type_text)
            and self.map_type(type_text) == "complex64_t"
        )

    def glsl_complex64_operand(self, rendered, type_text, expr=None):
        if self.is_glsl_complex64_type(type_text):
            return rendered
        return self.glsl_registered_scalar_to_struct_conversion(
            expr,
            rendered,
            type_text,
            "complex64_t",
            StructureConversionKind.SCALAR_PROMOTION.value,
        )

    def glsl_explicit_complex64_scalar_constructor(self, expr, destination_type):
        arguments = list(getattr(expr, "args", []) or [])
        destination = self.glsl_value_type_info(destination_type)
        if len(arguments) != 1 or destination is None or destination["width"] != 1:
            return None

        argument = arguments[0]
        source_type = self.glsl_source_expression_type(argument)
        if source_type is None or not self.is_glsl_registered_complex64_type(
            source_type
        ):
            return None

        binding = self.glsl_registered_scalar_structure_contract(source_type)
        if binding is None:
            return None
        self.glsl_validate_registered_structure_destination(
            expr,
            binding,
            source_type,
            StructureConversionKind.VALUE_CONVERSION.value,
        )

        real_member = MemberAccessNode(argument, "real")
        rendered = f"({self.generate_expression(argument)}).real"
        converted = self.glsl_expected_type_conversion(
            real_member,
            rendered,
            destination_type,
        )
        # Half and bfloat16 values live in float32 storage, so the projected
        # real part must still be rounded: unrounded, 65520.0 stays finite
        # instead of becoming +inf and pi keeps its float32 low bits.
        return self.glsl_apply_narrow_float_contract(
            converted, destination_type, real_member
        )

    @staticmethod
    def glsl_structure_conversion_scalar_kind(value_type):
        if value_type is None or value_type["width"] != 1:
            return None
        return {
            "bool": ScalarKind.BOOLEAN,
            "int": ScalarKind.SIGNED_INTEGER,
            "uint": ScalarKind.UNSIGNED_INTEGER,
            "float": ScalarKind.FLOATING,
        }.get(value_type["family"])

    def glsl_registered_scalar_structure_contract(self, destination_type):
        destination_type_name = self.glsl_normalized_source_type(destination_type)
        if not destination_type_name:
            return None
        source_destination_type = self.glsl_source_type_identifier_name(
            destination_type_name
        )
        candidates = (
            source_destination_type,
            destination_type_name,
            source_destination_type.rsplit("::", 1)[-1],
            destination_type_name.rsplit("::", 1)[-1],
        )
        for candidate in candidates:
            contract = registered_structure_conversion_for_identity(candidate)
            if contract is not None:
                mapped_destination_type = self.map_type(destination_type_name)
                source_destination_type = self.glsl_source_type_identifier_name(
                    mapped_destination_type
                )
                return contract, source_destination_type, mapped_destination_type
        return None

    def glsl_registered_structure_destination_shape(self, binding):
        contract, source_destination_type, _mapped_destination_type = binding
        member_types = self.struct_member_types.get(source_destination_type)
        if member_types is None:
            member_types = self.struct_member_types.get(contract.destination_type)
        if member_types is None:
            return None
        return tuple(
            (
                name,
                self.glsl_normalized_source_type(member_type)
                or self.type_name_string(member_type),
            )
            for name, member_type in member_types.items()
        )

    def glsl_validate_registered_structure_destination(
        self,
        expr,
        binding,
        source_type,
        conversion_kind,
    ):
        contract, _source_destination_type, mapped_destination_type = binding
        destination_shape = self.glsl_registered_structure_destination_shape(binding)
        if destination_shape is None:
            self.glsl_struct_construction_error(
                expr,
                mapped_destination_type,
                source_type,
                conversion_kind,
                "destination-shape-unknown",
            )
        if not contract.matches_destination_shape(destination_shape):
            self.glsl_struct_construction_error(
                expr,
                mapped_destination_type,
                source_type,
                conversion_kind,
                "destination-shape-mismatch",
            )

    def glsl_render_scalar_structure_field(self, rendered, source_info, target_type):
        target_info = self.glsl_value_type_info(target_type)
        if target_info is None or target_info["width"] != 1:
            return None
        if target_info["family"] == "bool":
            if source_info["family"] == "bool":
                return rendered
            source_zero = self.zero_value_expression(source_info["mapped"])
            return f"({rendered} != {source_zero})"
        if source_info["family"] == "bool":
            target_zero = self.zero_value_expression(target_info["mapped"])
            target_one = {
                "int": "1",
                "uint": "1u",
                "float": "1.0",
                "double": "1.0",
            }.get(target_info["mapped"], f"{target_info['mapped']}(1)")
            return f"(({rendered}) ? {target_one} : {target_zero})"
        return f"{target_info['mapped']}({rendered})"

    def glsl_registered_scalar_to_struct_conversion(
        self,
        expr,
        rendered,
        source_type,
        destination_type,
        conversion_kind,
    ):
        binding = self.glsl_registered_scalar_structure_contract(destination_type)
        if binding is None:
            return None
        contract, _source_destination_type, mapped_destination_type = binding
        source_info = self.glsl_value_type_info(source_type)
        scalar_kind = self.glsl_structure_conversion_scalar_kind(source_info)
        if scalar_kind not in contract.source_kinds:
            return None
        self.glsl_validate_registered_structure_destination(
            expr,
            binding,
            source_type,
            conversion_kind,
        )
        if contract.scalar_source_use_count != 1:
            self.glsl_struct_construction_error(
                expr,
                mapped_destination_type,
                source_type,
                conversion_kind,
                "source-single-evaluation-unsupported",
            )

        field_values = []
        for field in contract.fields:
            if field.scalar_value is StructureFieldValue.ZERO:
                field_values.append(self.zero_value_expression(field.type_name))
                continue
            converted = self.glsl_render_scalar_structure_field(
                rendered,
                source_info,
                field.type_name,
            )
            if converted is None:
                self.glsl_struct_construction_error(
                    expr,
                    mapped_destination_type,
                    source_type,
                    conversion_kind,
                    "field-conversion-unsupported",
                )
            field_values.append(converted)
        return f"{mapped_destination_type}({', '.join(field_values)})"

    def glsl_registered_default_struct_construction(self, expr, destination_type):
        binding = self.glsl_registered_scalar_structure_contract(destination_type)
        if binding is None:
            return None
        contract, _source_destination_type, mapped_destination_type = binding
        if not contract.supports_default_construction:
            return None
        conversion_kind = StructureConversionKind.DEFAULT_CONSTRUCTION.value
        self.glsl_validate_registered_structure_destination(
            expr,
            binding,
            None,
            conversion_kind,
        )
        field_values = []
        for field in contract.fields:
            if field.default_value is not StructureFieldValue.ZERO:
                self.glsl_struct_construction_error(
                    expr,
                    mapped_destination_type,
                    None,
                    conversion_kind,
                    "field-default-unsupported",
                )
            field_values.append(self.zero_value_expression(field.type_name))
        return f"{mapped_destination_type}({', '.join(field_values)})"
