"""Software subgroup helpers in the HLSL generator: helper preparation, masked bodies, physical lane ids, selection, half and float conversions, inverse hyperbolic and atan2 helpers, and trailing zero counts."""

import re

from ..ast import (
    BinaryOpNode,
    ExpressionStatementNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    ReturnNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from ..refusals import DirectXRefusal
from .cooperative_matrix_diagonal import (
    generate_hlsl_cooperative_matrix_diagonal_helpers,
)
from .directx_values import DirectXContextualConversionError
from .pointer_reinterpret import scalar_storage_layout
from .software_quad_reductions import (
    SOFTWARE_QUAD_REDUCTIONS,
    hlsl_quad_sum_body,
)


class DirectXSubgroupHelperMixin:
    """Subgroup helper members of ``HLSLCodeGen``."""

    def generate_hlsl_software_subgroup_masked_if(
        self, node, indent, *, additional_condition=None
    ):
        plan = self.hlsl_software_subgroup_masked_if_plans[id(node)]
        statements = self.hlsl_software_subgroup_statement_list(
            getattr(node, "then_branch", getattr(node, "if_body", None))
        )
        code = ""
        hoisted_index = plan.get("hoistedTargetDeclarationIndex")
        if hoisted_index is not None:
            # Declared before the guard, the target stays in scope for the
            # assignment after the reduction.
            code += self.generate_statement(statements[hoisted_index], indent)
        reductions = []
        for index in plan.get("statementIndices", (plan["statementIndex"],)):
            assignment = statements[index]
            if isinstance(assignment, ExpressionStatementNode):
                assignment = assignment.expression
            wrappers, inner = self.hlsl_software_subgroup_conversion_chain(
                getattr(assignment, "value", getattr(assignment, "right", None))
            )
            operation, arguments, operation_node = (
                self.hlsl_software_subgroup_direct_operation(inner)
            )
            target = getattr(assignment, "target", getattr(assignment, "left", None))
            value_type = self.expression_result_type(arguments[0])
            mapped_value_type = self.map_type(value_type)
            identity = self.HLSL_SOFTWARE_SUBGROUP_MASKED_IDENTITIES[operation].get(
                mapped_value_type
            )
            if (
                identity is None
                or not wrappers
                and self.map_type(self.expression_result_type(target))
                != mapped_value_type
            ):
                raise self.hlsl_software_subgroup_error(
                    "DirectX masked software subgroup reductions require a matching "
                    "32-bit float, int, or uint or binary16 payload and assignment "
                    "target",
                    workgroup_size=self.hlsl_software_subgroup_workgroup_size,
                    operation=operation,
                    reason="value-type-unsupported",
                    source_location=getattr(operation_node, "source_location", None),
                )
            if (
                operation == "WaveActiveSum"
                and mapped_value_type == "float16_t"
                and not plan.get("subgroupUniform")
            ):
                # Pairwise binary16 partials round at every step, so a guard that
                # splits a subgroup has no established source participation.
                raise self.hlsl_software_subgroup_error(
                    "DirectX masked binary16 sums require a guard that every "
                    "invocation of a subgroup evaluates alike",
                    workgroup_size=self.hlsl_software_subgroup_workgroup_size,
                    operation=operation,
                    reason="half-masked-participation-unproven",
                    source_location=getattr(operation_node, "source_location", None),
                )
            reductions.append(
                {
                    "index": index,
                    "wrappers": wrappers,
                    "operation": operation,
                    "arguments": arguments,
                    "operation_node": operation_node,
                    "target": target,
                    "value_type": value_type,
                    "mapped_value_type": mapped_value_type,
                    "identity": identity,
                }
            )

        used_names = set(self.local_variable_types)
        used_names.update(self.global_variable_types)
        used_names.update(self.current_identifier_aliases.values())

        def unique_name(value_name):
            name = self.hlsl_unique_local_identifier(value_name, used_names)
            used_names.add(name)
            return name

        active_name = unique_name("crossglSoftwareSubgroupActive")
        self.local_variable_types[active_name] = "bool"
        for reduction in reductions:
            reduction["input"] = unique_name("crossglSoftwareSubgroupInput")
            reduction["result"] = unique_name("crossglSoftwareSubgroupResult")
            self.local_variable_types[reduction["input"]] = reduction[
                "mapped_value_type"
            ]
            self.local_variable_types[reduction["result"]] = reduction[
                "mapped_value_type"
            ]

        indent_str = "    " * indent
        inner_indent = "    " * (indent + 1)
        condition = self.hlsl_condition_expression(
            getattr(node, "condition", getattr(node, "if_condition", None))
        )
        if additional_condition is not None:
            condition = f"({additional_condition} && ({condition}))"
        code += f"{indent_str}bool {active_name} = {condition};\n"
        previous = None
        for reduction in reductions:
            mapped_value_type = reduction["mapped_value_type"]
            code += (
                f"{indent_str}{mapped_value_type} {reduction['input']} = "
                f"{reduction['identity']};\n"
            )
            code += f"{indent_str}if ({active_name}) {{\n"
            if previous is None:
                code += self.generate_scoped_statement_body(
                    [
                        statement
                        for index, statement in enumerate(
                            statements[: reduction["index"]]
                        )
                        if index != hoisted_index
                    ],
                    indent + 1,
                )
            else:
                # The previous assignment and the statements up to this
                # reduction keep their source order under the guard.
                code += self.generate_hlsl_software_subgroup_masked_assignment(
                    previous,
                    statements[previous["index"] : reduction["index"]],
                    indent + 1,
                )
            payload = self.generate_expression_with_expected(
                reduction["arguments"][0], reduction["value_type"]
            )
            code += f"{inner_indent}{reduction['input']} = {payload};\n"
            code += f"{indent_str}}}\n"
            call = self.hlsl_software_subgroup_call(
                reduction["operation"], reduction["value_type"], reduction["input"]
            )
            code += f"{indent_str}{mapped_value_type} {reduction['result']} = {call};\n"
            previous = reduction
        code += f"{indent_str}if ({active_name}) {{\n"
        code += self.generate_hlsl_software_subgroup_masked_assignment(
            previous, statements[previous["index"] :], indent + 1
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_hlsl_software_subgroup_masked_assignment(
        self, reduction, statements, indent
    ):
        """Emit a predicated reduction's assignment and the statements after it.

        ``statements`` starts with the reduction's source assignment.
        """
        if reduction["wrappers"]:
            # Emit the source assignment, conversions included, with its
            # reduction replaced by the result every invocation computed.
            key = id(reduction["operation_node"])
            self.hlsl_software_subgroup_result_overrides[key] = reduction["result"]
            try:
                return self.generate_scoped_statement_body(statements, indent)
            finally:
                self.hlsl_software_subgroup_result_overrides.pop(key, None)
        target = self.generate_expression(reduction["target"])
        code = f"{'    ' * indent}{target} = {reduction['result']};\n"
        return code + self.generate_scoped_statement_body(statements[1:], indent)

    def hlsl_software_subgroup_identifier(self, key, base_name):
        existing = self.hlsl_software_subgroup_helper_names.get(key)
        if existing is not None:
            return existing
        reserved_names = set(self.function_return_types)
        reserved_names.update(self.global_variable_types)
        reserved_names.update(self.structs_by_name)
        reserved_names.update(self.hlsl_software_subgroup_reserved_names)
        reserved_names.update(self.hlsl_software_subgroup_helper_names.values())
        name = base_name
        suffix = 1
        while name in reserved_names:
            name = f"{base_name}_{suffix}"
            suffix += 1
        self.hlsl_software_subgroup_helper_names[key] = name
        return name

    def hlsl_software_subgroup_scratch_name(self, value_type):
        suffix = re.sub(r"[^A-Za-z0-9_]", "_", value_type)
        return self.hlsl_software_subgroup_identifier(
            ("scratch", value_type),
            f"__crossgl_software_subgroup_scratch_{suffix}",
        )

    def hlsl_software_subgroup_helper_name(self, operation, value_type):
        suffix = re.sub(r"[^A-Za-z0-9_]", "_", value_type)
        operation_name = self.HLSL_SOFTWARE_SUBGROUP_REDUCTIONS.get(
            operation, "shuffle_down"
        )
        if operation in self.HLSL_SOFTWARE_SUBGROUP_SCANS:
            operation_name = operation[len("Wave") :]
        elif operation == "WaveReadLaneAt":
            operation_name = "read_lane_at"
        elif operation in self.HLSL_SOFTWARE_SUBGROUP_RELATIVE_SHUFFLES:
            operation_name = self.HLSL_SOFTWARE_SUBGROUP_RELATIVE_SHUFFLES[operation][0]
        elif operation == "WaveShuffleAndFillUp":
            operation_name = "shuffle_and_fill_up"
        elif operation == "QuadActiveSum":
            operation_name = "quad_sum"
        return self.hlsl_software_subgroup_identifier(
            ("helper", operation, value_type),
            f"__crossgl_software_subgroup_{operation_name}_{suffix}",
        )

    def hlsl_software_subgroup_call(
        self,
        operation,
        value_type,
        value_expression,
        delta_expression=None,
        *,
        fill_expression=None,
        modulo_expression=None,
    ):
        mapped_value_type = self.map_type(value_type)
        is_vote = operation in self.HLSL_SOFTWARE_SUBGROUP_VOTES
        # Shuffles move registers unchanged, so every native scalar register
        # type is lossless, including binary16 and widened 8-bit values.
        shuffle_types = self.HLSL_SOFTWARE_SUBGROUP_SHUFFLE_TYPES | {"float16_t"}
        valid_type = (
            mapped_value_type == "bool"
            if is_vote
            else mapped_value_type in (
                shuffle_types
                if operation == "WaveReadLaneAt"
                or operation in self.HLSL_SOFTWARE_SUBGROUP_RELATIVE_SHUFFLES
                else self.HLSL_SOFTWARE_SUBGROUP_VALUE_TYPES
            )
        )
        if (
            operation in {"WaveActiveSum", "WaveActiveMin", "WaveActiveMax"}
            and mapped_value_type == "float16_t"
        ):
            valid_type = True
        if operation in {"WaveActiveMin", "WaveActiveMax"} and mapped_value_type in {
            "int16_t",
            "uint16_t",
        }:
            # A comparison selects one operand, so 16-bit integer extrema are
            # exact in their native registers.
            valid_type = True
        binary16_product = (
            operation == "WaveActiveProduct" and mapped_value_type == "float16_t"
        )
        if binary16_product:
            # Native binary16 multiplies round every node of the product tree.
            valid_type = True
        narrow_wrap = None
        if (
            (operation == "WaveActiveProduct" and not binary16_product)
            or operation in self.HLSL_SOFTWARE_SUBGROUP_SCANS
            or operation in SOFTWARE_QUAD_REDUCTIONS
        ):
            layout = scalar_storage_layout(self.type_name_string(value_type))
            scan = operation in self.HLSL_SOFTWARE_SUBGROUP_SCANS
            if (
                scan
                and layout is not None
                and layout.bit_width < 32
                and (
                    layout.kind == "integer"
                    and mapped_value_type in {"int", "uint", "int16_t", "uint16_t"}
                    or mapped_value_type == "float16_t"
                )
            ):
                # 16-bit types scan natively; 8-bit values held in 32-bit
                # registers scan modulo 2^32 and are rewrapped afterwards.
                # bfloat16 carriers are raw bits and must never scan as integers.
                valid_type = True
                if mapped_value_type in {"int", "uint"}:
                    narrow_wrap = layout
            else:
                valid_type = (
                    valid_type and layout is not None and layout.bit_width == 32
                )
        if operation == "WaveShuffleAndFillUp":
            valid_type = mapped_value_type in {"bool", "uint2"} | shuffle_types
        if not valid_type:
            raise self.hlsl_software_subgroup_error(
                "DirectX software subgroup votes require scalar bool payloads; "
                "arithmetic supports 32-bit float, int, and uint, plus binary16 sums, "
                "products, minima and maxima and 16-bit integer minima and maxima; "
                "shuffles also support signed and unsigned 16-bit "
                "and 64-bit integers and two-word uint2 payloads",
                workgroup_size=self.hlsl_software_subgroup_workgroup_size,
                operation=operation,
                reason="value-type-unsupported",
            )
        invocation = self.hlsl_software_subgroup_invocation_expressions.get(
            self.current_function_name
        )
        if invocation is None:
            raise self.hlsl_software_subgroup_error(
                "DirectX software subgroup operation has no proven logical "
                "invocation index in the current function",
                workgroup_size=self.hlsl_software_subgroup_workgroup_size,
                operation=operation,
                reason="invocation-index-unavailable",
            )
        key = (operation, mapped_value_type)
        self.required_hlsl_software_subgroup_helpers.add(key)
        helper = self.hlsl_software_subgroup_helper_name(operation, mapped_value_type)
        arguments = [value_expression]
        if fill_expression is not None:
            arguments.append(fill_expression)
        if delta_expression is not None:
            arguments.append(f"uint({delta_expression})")
        if modulo_expression is not None:
            arguments.append(f"uint({modulo_expression})")
        arguments.append(f"uint({invocation})")
        call = f"{helper}({', '.join(arguments)})"
        if narrow_wrap is not None:
            shift = 32 - narrow_wrap.bit_width
            if mapped_value_type == "int":
                # Arithmetic right shift restores the source sign extension.
                return f"(({call} << {shift}) >> {shift})"
            return f"({call} & {(1 << narrow_wrap.bit_width) - 1}u)"
        return call

    def hlsl_software_subgroup_reduction_body(self, operation, value_type, scratch):
        reducer = self.HLSL_SOFTWARE_SUBGROUP_REDUCTIONS[operation]
        qualifier = "precise " if value_type in {"float", "float16_t"} else ""
        partial = (
            self.hlsl_software_subgroup_invocation_count % self.software_subgroup_width
            != 0
        )
        active = "(lane & (2u * stride - 1u)) == 0u"
        if partial:
            active += " && lane + stride < activeCount"
        if reducer == "product" or (value_type == "float16_t" and reducer == "sum"):
            operator = "*" if reducer == "product" else "+"
            return (
                "    [unroll]\n"
                f"    for (uint stride = 1u; stride < {self.software_subgroup_width}u; stride <<= 1u) {{\n"
                f"        if ({active}) {{\n"
                f"            {qualifier}{value_type} {reducer} = {scratch}[subgroupBase + lane] {operator} {scratch}[subgroupBase + lane + stride];\n"
                f"            {scratch}[subgroupBase + lane] = {reducer};\n"
                "        }\n"
                "        GroupMemoryBarrierWithGroupSync();\n"
                "    }\n"
                f"    {value_type} result = {scratch}[subgroupBase];\n"
            )
        code = (
            f"    {qualifier}{value_type} result = {scratch}[subgroupBase];\n"
            "    [unroll]\n"
            f"    for (uint offset = 1u; offset < {self.software_subgroup_width}u; ++offset) {{\n"
            + ("        if (offset >= activeCount) { continue; }\n" if partial else "")
            + f"        {value_type} operand = {scratch}[subgroupBase + offset];\n"
        )
        if reducer in {"all", "any"}:
            operator = "&&" if reducer == "all" else "||"
            code += f"        result = result {operator} operand;\n"
        elif reducer == "sum":
            code += "        result = result + operand;\n"
        elif value_type == "float":
            # Classify bits before comparisons so NaNs cannot erase numeric lanes.
            zero_operator = "|" if reducer == "min" else "&"
            code += (
                "        uint leftBits = asuint(result);\n"
                "        uint rightBits = asuint(operand);\n"
                "        if ((leftBits & 0x7fffffffu) > 0x7f800000u) {\n"
                "            result = operand;\n"
                "        } else if ((rightBits & 0x7fffffffu) <= 0x7f800000u) {\n"
                "            if (((leftBits | rightBits) & 0x7fffffffu) == 0u) {\n"
                f"                result = asfloat(leftBits {zero_operator} rightBits);\n"
                "            } else {\n"
                f"                result = {reducer}(result, operand);\n"
                "            }\n"
                "        }\n"
            )
        elif value_type == "float16_t":
            # The binary16 form of the classification above; a comparison
            # selects one operand, so no rounding is involved.
            zero_operator = "|" if reducer == "min" else "&"
            code += (
                "        uint leftBits = uint(asuint16(result));\n"
                "        uint rightBits = uint(asuint16(operand));\n"
                "        if ((leftBits & 0x7fffu) > 0x7c00u) {\n"
                "            result = operand;\n"
                "        } else if ((rightBits & 0x7fffu) <= 0x7c00u) {\n"
                "            if (((leftBits | rightBits) & 0x7fffu) == 0u) {\n"
                f"                result = asfloat16(uint16_t(leftBits {zero_operator} rightBits));\n"
                "            } else {\n"
                f"                result = {reducer}(result, operand);\n"
                "            }\n"
                "        }\n"
            )
        else:
            code += f"        result = {reducer}(result, operand);\n"
        return code + "    }\n"

    def hlsl_software_subgroup_scan_body(self, operation, value_type, scratch):
        operator, inclusive = self.HLSL_SOFTWARE_SUBGROUP_SCANS[operation]
        qualifier = "precise " if value_type in {"float", "float16_t"} else ""
        left = f"{scratch}[subgroupBase + (lane & ~(2u * stride - 1u)) + stride - 1u]"
        combine = f"{left} {operator} result"
        if value_type == "int":
            combine = f"asint(asuint({left}) {operator} asuint(result))"
        # Read every previous-stage value before any lane overwrites shared scratch.
        code = (
            f"    {qualifier}{value_type} result = value;\n"
            "    [unroll]\n"
            f"    for (uint stride = 1u; stride < {self.software_subgroup_width}u; stride <<= 1u) {{\n"
            "        if ((lane & stride) != 0u) {\n"
            f"            result = {combine};\n"
            "        }\n"
            "        GroupMemoryBarrierWithGroupSync();\n"
            f"        {scratch}[invocation] = result;\n"
            "        GroupMemoryBarrierWithGroupSync();\n"
            "    }\n"
        )
        identity = f"{value_type}({'0' if operator == '+' else '1'})"
        if value_type == "float" and operator == "+":
            identity = "asfloat(0x80000000u)"
        if value_type == "float16_t" and operator == "+":
            identity = "asfloat16((uint16_t)0x8000u)"
        code += (
            f"    result = {identity};\n"
            "    if (lane > 0u) {\n"
            f"        result = {scratch}[invocation - 1u];\n"
            "    }\n"
        )
        if inclusive:
            combine = f"result {operator} value"
            if value_type == "int":
                combine = f"asint(asuint(result) {operator} asuint(value))"
            code += f"    result = {combine};\n"
        return code

    def generate_hlsl_software_subgroup_helpers(self):
        if not (
            self.required_hlsl_software_subgroup_helpers
            or self.required_hlsl_cooperative_matrix_mma
            or self.required_hlsl_cooperative_matrix_fragment_offset
            or self.required_hlsl_cooperative_matrix_diagonals
            or self.hlsl_software_subgroup_invocation_variable is not None
        ):
            return ""
        invocation_count = self.hlsl_software_subgroup_invocation_count
        width = self.software_subgroup_width
        partial = invocation_count % width != 0
        value_types = sorted(
            {
                value_type
                for _operation, value_type in (
                    self.required_hlsl_software_subgroup_helpers
                )
            }
        )
        code = ""
        if self.hlsl_software_subgroup_invocation_variable is not None:
            variable = self.hlsl_software_subgroup_invocation_variable
            code += f"static uint {variable};\n"
        for value_type in value_types:
            scratch = self.hlsl_software_subgroup_scratch_name(value_type)
            code += f"groupshared {value_type} {scratch}[{invocation_count}];\n"
        code += "\n"
        for operation, value_type in sorted(
            self.required_hlsl_software_subgroup_helpers
        ):
            helper = self.hlsl_software_subgroup_helper_name(operation, value_type)
            scratch = self.hlsl_software_subgroup_scratch_name(value_type)
            reduction = operation in self.HLSL_SOFTWARE_SUBGROUP_REDUCTIONS
            scan = operation in self.HLSL_SOFTWARE_SUBGROUP_SCANS
            relative = self.HLSL_SOFTWARE_SUBGROUP_RELATIVE_SHUFFLES.get(operation)
            delta_parameter = "uint delta, " if relative is not None else ""
            if operation == "WaveReadLaneAt":
                delta_parameter = "uint index, "
            if operation == "WaveShuffleAndFillUp":
                delta_parameter = f"{value_type} fill, uint delta, uint modulo, "
            code += (
                f"{value_type} {helper}({value_type} value, {delta_parameter}"
                "uint invocation) {\n"
                f"    uint lane = invocation % {width}u;\n"
                "    uint subgroupBase = invocation - lane;\n"
                + (
                    f"    uint activeCount = min({width}u, {invocation_count}u - subgroupBase);\n"
                    if partial
                    else ""
                )
                + f"    {scratch}[invocation] = value;\n"
                "    GroupMemoryBarrierWithGroupSync();\n"
            )
            if operation == "WaveShuffleAndFillUp":
                active_count = "activeCount" if partial else f"{width}u"
                # Reuse scratch only after every lane has read the value phase.
                # Invalid/inactive source lanes have no portable result, but must
                # never address another subgroup or escape the shared allocation.
                code += (
                    "    bool validModulo = modulo == 2u || modulo == 4u || modulo == 8u || modulo == 16u || modulo == 32u;\n"
                    "    uint segmentWidth = validModulo ? modulo : 32u;\n"
                    "    uint segmentLane = lane % segmentWidth;\n"
                    "    bool useValue = segmentLane >= delta;\n"
                    "    uint sourceLane = lane;\n"
                    "    if (validModulo && delta <= segmentWidth) {\n"
                    "        sourceLane = useValue ? lane - delta : lane + segmentWidth - delta;\n"
                    "    }\n"
                    f"    bool sourceValid = validModulo && delta <= segmentWidth && sourceLane < {active_count};\n"
                    "    sourceLane = sourceValid ? sourceLane : lane;\n"
                    f"    {value_type} result = {scratch}[subgroupBase + sourceLane];\n"
                    "    GroupMemoryBarrierWithGroupSync();\n"
                    f"    {scratch}[invocation] = fill;\n"
                    "    GroupMemoryBarrierWithGroupSync();\n"
                    "    if (sourceValid && !useValue) {\n"
                    f"        result = {scratch}[subgroupBase + sourceLane];\n"
                    "    }\n"
                )
            elif scan:
                code += self.hlsl_software_subgroup_scan_body(
                    operation, value_type, scratch
                )
            elif reduction:
                code += self.hlsl_software_subgroup_reduction_body(
                    operation, value_type, scratch
                )
            elif operation in SOFTWARE_QUAD_REDUCTIONS:
                code += hlsl_quad_sum_body(value_type, scratch, invocation_count)
            else:
                active_count = "activeCount" if partial else f"{width}u"
                if operation == "WaveReadLaneAt":
                    valid, source_lane = f"index < {active_count}", "index"
                else:
                    # A source lane outside the active subgroup yields the
                    # calling lane's value, the explicit out-of-range policy.
                    _name, valid, source_lane = relative
                    valid = valid.format(active=active_count)
                code += (
                    f"    bool sourceValid = {valid};\n"
                    f"    uint sourceLane = sourceValid ? {source_lane} : lane;\n"
                    f"    {value_type} result = sourceValid\n"
                    f"        ? {scratch}[subgroupBase + sourceLane]\n"
                    "        : value;\n"
                )
            code += (
                "    GroupMemoryBarrierWithGroupSync();\n"
                "    return result;\n"
                "}\n\n"
            )
        if self.required_hlsl_cooperative_matrix_mma:
            code += self.generate_hlsl_cooperative_matrix_mma_helper()
        if self.required_hlsl_cooperative_matrix_fragment_offset:
            code += self.generate_hlsl_cooperative_matrix_fragment_offset_helper()
        code += generate_hlsl_cooperative_matrix_diagonal_helpers(self)
        return code

    def prepare_hlsl_physical_subgroup_id_helper_names(self, functions):
        used_names = self.hlsl_helper_reserved_names(functions)
        used_names.update(self.hlsl_trailing_zero_helper_names.values())
        used_names.update(self.hlsl_inverse_hyperbolic_helper_names.values())
        names = {}
        for key, base_name in (
            ("leaders", "__crossgl_physical_subgroup_leaders"),
            ("helper", "__crossgl_physical_subgroup_id"),
        ):
            name = base_name
            while name in used_names:
                name += "_"
            names[key] = name
            used_names.add(name)
        self.hlsl_physical_subgroup_id_helper_names = names

    def hlsl_physical_subgroup_id_helper_name(self):
        helper_name = self.hlsl_physical_subgroup_id_helper_names.get("helper")
        if helper_name is not None:
            return helper_name
        self.prepare_hlsl_physical_subgroup_id_helper_names(())
        return self.hlsl_physical_subgroup_id_helper_names["helper"]

    def generate_hlsl_physical_subgroup_id_helper(self):
        if not self.required_hlsl_physical_subgroup_id_helper:
            return ""
        leaders = self.hlsl_physical_subgroup_id_helper_names["leaders"]
        helper_name = self.hlsl_physical_subgroup_id_helper_name()
        # Source languages number subgroups by thread position: subgroup 0
        # holds invocation 0. Direct3D numbers no waves, so each wave sets
        # the bit of the smallest SV_GroupIndex it holds, and its ID is the
        # number of waves whose smallest index is smaller. One bit per
        # invocation covers D3D12's 1024-invocation limit in 32 words.
        return f"""groupshared uint {leaders}[32];
uint {helper_name}(uint groupIndex) {{
    if (groupIndex < 32u) {{
        {leaders}[groupIndex] = 0u;
    }}
    GroupMemoryBarrierWithGroupSync();
    uint leader = WaveActiveMin(groupIndex);
    uint leaderWord = leader >> 5u;
    uint leaderBit = 1u << (leader & 31u);
    if (WaveIsFirstLane()) {{
        InterlockedOr({leaders}[leaderWord], leaderBit);
    }}
    GroupMemoryBarrierWithGroupSync();
    uint subgroupIndex = countbits({leaders}[leaderWord] & (leaderBit - 1u));
    for (uint word = 0u; word < leaderWord; ++word) {{
        subgroupIndex += countbits({leaders}[word]);
    }}
    return subgroupIndex;
}}
"""

    def hlsl_trailing_zero_helper_name(self, operand_type):
        helper_name = self.hlsl_trailing_zero_helper_names.get(operand_type)
        if helper_name is not None:
            return helper_name

        used_names = set(getattr(self, "function_return_types", {}))
        used_names.update(getattr(self, "global_variable_types", {}))
        used_names.update(getattr(self, "local_variable_types", {}))
        used_names.update(self.hlsl_trailing_zero_helper_names.values())
        helper_name = f"__crossgl_ctz_{operand_type}"
        while helper_name in used_names:
            helper_name += "_"
        self.hlsl_trailing_zero_helper_names[operand_type] = helper_name
        return helper_name

    def generate_hlsl_trailing_zero_helpers(self):
        helpers = []
        for operand_type in sorted(self.required_hlsl_trailing_zero_helpers):
            helper_name = self.hlsl_trailing_zero_helper_name(operand_type)
            if operand_type == "uint":
                helpers.append(f"""int {helper_name}(uint value) {{
    // Clang leaves ctz(0) undefined; use the canonical least-bit -1 sentinel.
    return value == 0u ? -1 : int(firstbitlow(value));
}}
""")
            elif operand_type == "uint64_t":
                helpers.append(f"""int {helper_name}(uint64_t value) {{
    uint lowBits = uint(value);
    if (lowBits != 0u) {{
        return int(firstbitlow(lowBits));
    }}
    uint highBits = uint(value >> 32);
    // Clang leaves ctz(0) undefined; use the canonical least-bit -1 sentinel.
    return highBits == 0u ? -1 : (32 + int(firstbitlow(highBits)));
}}
""")
        return "\n".join(helpers) + ("\n" if helpers else "")

    def prepare_hlsl_half_selection_helper_names(self, functions):
        used_names = self.hlsl_helper_reserved_names(functions)
        self.hlsl_half_selection_helper_names = {}
        for width in range(1, 5):
            name = f"__crossgl_select_half_bits{width}"
            while name in used_names:
                name += "_"
            self.hlsl_half_selection_helper_names[width] = name
            used_names.add(name)

    def hlsl_half_selection_condition_is_eager_safe(self, node):
        if isinstance(node, (VariableNode, IdentifierNode, LiteralNode)):
            return True
        if isinstance(node, UnaryOpNode) and node.op in {"!", "~", "+", "-"}:
            return self.hlsl_half_selection_condition_is_eager_safe(node.operand)
        if isinstance(node, BinaryOpNode) and node.op in {
            "==",
            "!=",
            "<",
            "<=",
            ">",
            ">=",
            "&&",
            "||",
            "&",
            "|",
            "^",
        }:
            return all(
                self.hlsl_half_selection_condition_is_eager_safe(value)
                for value in (node.left, node.right)
            )
        if isinstance(node, FunctionCallNode):
            name = self.function_call_name(node)
            return (
                name in {"isnan", "isinf", "isfinite"}
                and not self.hlsl_function_name_is_shadowed(name)
                and all(
                    isinstance(arg, (VariableNode, IdentifierNode))
                    for arg in node.arguments
                )
            )
        return False

    def hlsl_half_selection_info(self, node, *, allow_condition_reads=True):
        if self.current_function_name is None:
            return None
        if not self.hlsl_half_selection_condition_is_eager_safe(
            node.condition
        ) and not (
            allow_condition_reads
            and not self.hlsl_expression_has_observable_side_effects(node.condition)
        ):
            return None
        # Only private values and recursively safe selections may be eager arms.
        for arm in (node.true_expr, node.false_expr):
            if isinstance(arm, (VariableNode, IdentifierNode)):
                continue
            if isinstance(arm, TernaryOpNode) and self.hlsl_half_selection_info(
                arm, allow_condition_reads=False
            ):
                continue
            return None
        if self.map_type(self.expression_result_type(node.condition)) != "bool":
            return None
        infos = [
            self.hlsl_floating_arithmetic_type_info(self.expression_result_type(arm))
            for arm in (node.true_expr, node.false_expr)
        ]
        if not all(info and info["native_16_bit"] for info in infos):
            return None
        if infos[0]["width"] != infos[1]["width"]:
            return None
        return infos[0]

    def generate_hlsl_half_selection(self, node):
        info = self.hlsl_half_selection_info(node)
        if info is None:
            return None
        for intrinsic in ("asuint16", "asfloat16"):
            if intrinsic in self.global_variable_types or (
                intrinsic in self.function_return_types
                and intrinsic not in self.hlsl_function_name_aliases
            ):
                raise DirectXContextualConversionError(
                    f"DirectX cannot preserve binary16 selection: '{intrinsic}' is shadowed",
                    source_type=info["mapped_type"],
                    target_type=info["mapped_type"],
                    reason="half-selection-intrinsic-shadowed",
                    source_location=getattr(node, "source_location", None),
                )
        width = info["width"]
        self.required_hlsl_half_selection_helpers.add(width)
        condition = self.generate_expression_with_expected(node.condition, "bool")
        yes = self.generate_expression_with_expected(node.true_expr, None)
        no = self.generate_expression_with_expected(node.false_expr, None)
        return (
            f"{self.hlsl_half_selection_helper_names[width]}({condition}, {yes}, {no})"
        )

    def hlsl_half_selection_tail_return(self, branch, fallback):
        if not isinstance(branch, IfNode) or not isinstance(fallback, ReturnNode):
            return None
        if (
            branch.else_branch is not None
            or branch.else_body is not None
            or branch.else_if_conditions
        ):
            return None
        statements = self.hlsl_statement_body_items(branch.then_branch)
        if len(statements) != 1 or not isinstance(statements[0], ReturnNode):
            return None
        selection = TernaryOpNode(
            branch.condition,
            statements[0].value,
            fallback.value,
            source_location=getattr(branch, "source_location", None),
        )
        if self.hlsl_half_selection_info(selection) is None:
            return None
        # A separate early return can reintroduce a half-valued select after CSE.
        return ReturnNode(
            selection, source_location=getattr(branch, "source_location", None)
        )

    def generate_hlsl_half_selection_helpers(self):
        code = ""
        for width in sorted(self.required_hlsl_half_selection_helpers):
            suffix = str(width) if width > 1 else ""
            value_type = f"float16_t{suffix}"
            bits_type = f"uint16_t{suffix}"
            mask_type = f"uint{suffix}"
            name = self.hlsl_half_selection_helper_names[width]
            # Direct integer ternaries are optimized back to half-valued selects.
            code += (
                f"{value_type} {name}(bool condition, {value_type} yes, {value_type} no) {{\n"
                f"    {mask_type} mask = ({mask_type})(0u - uint(condition));\n"
                f"    return asfloat16({bits_type}(({mask_type}(asuint16(yes)) & mask) | "
                f"({mask_type}(asuint16(no)) & ~mask)));\n"
                "}\n\n"
            )
        return code

    def prepare_hlsl_unsigned_float_helper_names(self, functions):
        used_names = self.hlsl_helper_reserved_names(functions)
        self.hlsl_unsigned_float_helper_names = {}
        for width in range(1, 5):
            name = f"__crossgl_uint_to_float{width}"
            while name in used_names:
                name += "_"
            self.hlsl_unsigned_float_helper_names[width] = name
            used_names.add(name)
        self.hlsl_wide_integer_float_helper_names = {}
        for base_type, stem in self.HLSL_WIDE_INTEGER_FLOAT_HELPER_STEMS.items():
            for width in range(1, 5):
                name = f"{stem}{width}"
                while name in used_names:
                    name += "_"
                self.hlsl_wide_integer_float_helper_names[(base_type, width)] = name
                used_names.add(name)

    def generate_hlsl_wide_integer_float_helpers(self, scalar):
        """Emit exact int64_t/uint64_t-to-binary32 helpers built on ``scalar``.

        ``scalar`` is the exact uint-to-float helper; the 64-bit helpers keep
        32 leading bits, fold the discarded bits into a sticky low bit below
        the binary32 rounding position, round once and rescale the exponent.
        """
        required = self.required_hlsl_wide_integer_float_helpers
        if not required:
            return ""
        names = self.hlsl_wide_integer_float_helper_names
        unsigned = names[("uint64_t", 1)]
        code = f"""float {unsigned}(uint64_t value) {{
    uint high = uint(value >> 32u);
    if (high == 0u) {{ return {scalar}(uint(value)); }}
    // Round once: a sticky bit stands in for every discarded low bit.
    uint shift = uint(firstbithigh(high)) + 1u;
    uint64_t discarded = value & ((uint64_t(1) << shift) - uint64_t(1));
    uint kept = uint(value >> shift) | (discarded != uint64_t(0) ? 1u : 0u);
    return asfloat(asuint({scalar}(kept)) + (shift << 23u));
}}
"""
        if any(base_type == "int64_t" for base_type, _width in required):
            signed = names[("int64_t", 1)]
            code += f"""float {signed}(int64_t value) {{
    uint64_t magnitude = value < int64_t(0) ? uint64_t(0) - uint64_t(value) : uint64_t(value);
    float rounded = {unsigned}(magnitude);
    return value < int64_t(0) ? -rounded : rounded;
}}
"""
        for base_type, width in sorted(required):
            if width == 1:
                continue
            name = names[(base_type, width)]
            lane_helper = names[(base_type, 1)]
            lanes = ", ".join(f"{lane_helper}(value.{lane})" for lane in "xyzw"[:width])
            code += (
                f"float{width} {name}({base_type}{width} value) {{\n"
                f"    return float{width}({lanes});\n}}\n"
            )
        return code

    def generate_hlsl_unsigned_float_helpers(self):
        if not self.required_hlsl_unsigned_float_helpers:
            return ""
        scalar = self.hlsl_unsigned_float_helper_names[1]
        code = f"""float {scalar}(uint value) {{
    if (value == 0u) {{ return 0.0f; }}
    uint exponent = uint(firstbithigh(value));
    uint significand;
    if (exponent <= 23u) {{
        significand = value << (23u - exponent);
    }} else {{
        // Round the integer once, including a carry into the next exponent.
        uint shift = exponent - 23u;
        significand = value >> shift;
        uint remainder = value & ((1u << shift) - 1u);
        uint midpoint = 1u << (shift - 1u);
        if (remainder > midpoint || (remainder == midpoint && (significand & 1u) != 0u)) {{
            significand += 1u;
        }}
        if (significand == 0x1000000u) {{
            significand >>= 1u;
            exponent += 1u;
        }}
    }}
    return asfloat(((exponent + 127u) << 23u) | (significand & 0x7fffffu));
}}
"""
        for width in sorted(self.required_hlsl_unsigned_float_helpers - {1}):
            name = self.hlsl_unsigned_float_helper_names[width]
            lanes = ", ".join(f"{scalar}(value.{lane})" for lane in "xyzw"[:width])
            code += (
                f"float{width} {name}(uint{width} value) {{\n"
                f"    return float{width}({lanes});\n}}\n"
            )
        code += self.generate_hlsl_wide_integer_float_helpers(scalar)
        return code + "\n"

    def hlsl_wide_integer_float_conversion_expression(
        self, rendered, expected, source, *, source_location=None
    ):
        """Convert int64_t/uint64_t lanes to binary32 with one rounding.

        A native float(uint64_t) is not round-to-nearest-even on every D3D
        device: WARP rounds through an intermediate format, turning
        2^63 + 2^39 + 1 into 2^63 instead of 2^63 + 2^40.
        """
        for intrinsic in ("asfloat", "asuint", "firstbithigh"):
            if intrinsic in self.global_variable_types or (
                intrinsic in self.function_return_types
                and intrinsic not in self.hlsl_function_name_aliases
            ):
                raise DirectXContextualConversionError(
                    "DirectX cannot preserve 64-bit-integer-to-float rounding: "
                    f"'{intrinsic}' is shadowed",
                    source_type=source["mapped_type"],
                    target_type=expected["mapped_type"],
                    reason="wide-integer-float-intrinsic-shadowed",
                    source_location=source_location,
                )
        width = source["width"]
        key = (source["base_type"], width)
        self.required_hlsl_wide_integer_float_helpers.update(
            {key, (source["base_type"], 1), ("uint64_t", 1)}
        )
        self.required_hlsl_unsigned_float_helpers.add(1)
        if not self.hlsl_wide_integer_float_helper_names:
            self.prepare_hlsl_unsigned_float_helper_names([])
        converted = f"{self.hlsl_wide_integer_float_helper_names[key]}({rendered})"
        if width != expected["width"]:
            return self.hlsl_scalar_splat_cast(expected["mapped_type"], converted)
        return converted

    def hlsl_unsigned_float_conversion_expression(
        self, rendered, expected_type, source_type, *, source_location=None
    ):
        expected = self.hlsl_floating_arithmetic_type_info(expected_type)
        source = self.hlsl_integer_arithmetic_type_info(source_type)
        if (
            expected is not None
            and expected["base_type"] == "float"
            and source is not None
            and source["base_type"] in self.HLSL_WIDE_INTEGER_FLOAT_HELPER_STEMS
            and source["width"] in {1, expected["width"]}
        ):
            return self.hlsl_wide_integer_float_conversion_expression(
                rendered, expected, source, source_location=source_location
            )
        if (
            expected is None
            or expected["base_type"] != "float"
            or source is None
            or source["base_type"] != "uint"
            or source["width"] not in {1, expected["width"]}
        ):
            return None
        for intrinsic in ("asfloat", "firstbithigh"):
            if intrinsic in self.global_variable_types or (
                intrinsic in self.function_return_types
                and intrinsic not in self.hlsl_function_name_aliases
            ):
                raise DirectXContextualConversionError(
                    f"DirectX cannot preserve unsigned-to-float rounding: '{intrinsic}' is shadowed",
                    source_type=source["mapped_type"],
                    target_type=expected["mapped_type"],
                    reason="unsigned-float-intrinsic-shadowed",
                    source_location=source_location,
                )
        width = source["width"]
        self.required_hlsl_unsigned_float_helpers.add(width)
        if not self.hlsl_unsigned_float_helper_names:
            self.prepare_hlsl_unsigned_float_helper_names([])
        converted = f"{self.hlsl_unsigned_float_helper_names[width]}({rendered})"
        if width != expected["width"]:
            return self.hlsl_scalar_splat_cast(expected["mapped_type"], converted)
        return converted

    def prepare_hlsl_half_helper_names(self, functions):
        used_names = self.hlsl_helper_reserved_names(functions)
        self.hlsl_half_helper_names = {}
        for width in range(1, 5):
            name = f"__crossgl_round_half{width}"
            while name in used_names:
                name += "_"
            self.hlsl_half_helper_names[width] = name
            used_names.add(name)

    def generate_hlsl_half_helpers(self):
        if not self.required_hlsl_half_helpers:
            return ""
        scalar = self.hlsl_half_helper_names[1]
        code = f"""float16_t {scalar}(float value) {{
    uint bits = asuint(value);
    uint sign = (bits >> 16u) & 0x8000u;
    uint magnitude = bits & 0x7fffffffu;
    uint result;
    if (magnitude >= 0x7f800000u) {{
        result = magnitude == 0x7f800000u ? 0x7c00u : 0x7e00u;
    }} else if (magnitude >= 0x477ff000u) {{
        result = 0x7c00u;
    }} else if (magnitude >= 0x38800000u) {{
        result = (magnitude - 0x38000000u + 0xfffu + ((magnitude >> 13u) & 1u)) >> 13u;
    }} else if (magnitude < 0x33000000u) {{
        result = 0u;
    }} else {{
        // Round the subnormal significand, including the smallest normal carry.
        uint shift = 126u - (magnitude >> 23u);
        uint significand = (magnitude & 0x7fffffu) | 0x800000u;
        result = significand >> shift;
        uint remainder = significand & ((1u << shift) - 1u);
        uint midpoint = 1u << (shift - 1u);
        if (remainder > midpoint || (remainder == midpoint && (result & 1u) != 0u)) {{
            result += 1u;
        }}
    }}
    return asfloat16(uint16_t(sign | result));
}}
"""
        for width in sorted(self.required_hlsl_half_helpers - {1}):
            name = self.hlsl_half_helper_names[width]
            values = ", ".join(f"{scalar}(value.{c})" for c in "xyzw"[:width])
            code += (
                f"float16_t{width} {name}(float{width} value) {{\n"
                f"    return float16_t{width}({values});\n}}\n"
            )
        return code + "\n"

    def hlsl_half_conversion_expression(
        self, rendered, width, source_types, *, source_location=None
    ):
        target = "float16_t" + (str(width) if width > 1 else "")

        def unsupported(reason, detail):
            return DirectXContextualConversionError(
                f"DirectX cannot preserve binary16 rounding: {detail}",
                source_type=", ".join(str(value) for value in source_types),
                target_type=target,
                reason=reason,
                source_location=source_location,
            )

        for source in source_types:
            info = self.hlsl_floating_arithmetic_type_info(source)
            if info is not None and info["base_type"] == "double":
                raise unsupported(
                    "half-double-rounding",
                    "double-to-half conversion cannot pass through float32",
                )
        for intrinsic in ("asuint", "asfloat16"):
            if intrinsic in self.global_variable_types or (
                intrinsic in self.function_return_types
                and intrinsic not in self.hlsl_function_name_aliases
            ):
                raise unsupported(
                    "half-target-intrinsic-shadowed", f"'{intrinsic}' is shadowed"
                )
        self.required_hlsl_half_helpers.add(width)
        if not self.hlsl_half_helper_names:
            self.prepare_hlsl_half_helper_names([])
        return f"{self.hlsl_half_helper_names[width]}({rendered})"

    def prepare_hlsl_atan2_helper_names(self, functions):
        used_names = self.hlsl_helper_reserved_names(functions)
        self.hlsl_atan2_helper_names = {}
        for width in range(1, 5):
            value_type = "float" if width == 1 else f"float{width}"
            name = f"__crossgl_atan2_{value_type}"
            while name in used_names:
                name += "_"
            self.hlsl_atan2_helper_names[value_type] = name
            used_names.add(name)

    def generate_hlsl_atan2_helpers(self):
        if not self.required_hlsl_atan2_helpers:
            return ""
        scalar = self.hlsl_atan2_helper_names["float"]
        code = f"""float {scalar}(float y, float x) {{
    uint yBits = asuint(y);
    uint xBits = asuint(x);
    uint yMagnitude = yBits & 0x7fffffffu;
    uint xMagnitude = xBits & 0x7fffffffu;
    uint sign = yBits & 0x80000000u;
    bool negativeX = (xBits & 0x80000000u) != 0u;
    if (yMagnitude > 0x7f800000u || xMagnitude > 0x7f800000u) {{
        return asfloat(0x7fc00000u);
    }}
    // HLSL atan2 uses numeric comparisons, which do not distinguish signed zero.
    if (yMagnitude == 0u) {{
        return asfloat(sign | (negativeX ? 0x40490fdbu : 0u));
    }}
    if (xMagnitude == 0u) {{ return asfloat(sign | 0x3fc90fdbu); }}
    if (yMagnitude == 0x7f800000u) {{
        uint angle = xMagnitude == 0x7f800000u
            ? (negativeX ? 0x4016cbe4u : 0x3f490fdbu) : 0x3fc90fdbu;
        return asfloat(sign | angle);
    }}
    if (xMagnitude == 0x7f800000u) {{
        return asfloat(sign | (negativeX ? 0x40490fdbu : 0u));
    }}
    bool invert = yMagnitude > xMagnitude;
    uint smaller = invert ? xMagnitude : yMagnitude;
    uint larger = invert ? yMagnitude : xMagnitude;
    uint smallExponent = smaller >> 23;
    uint largeExponent = larger >> 23;
    uint smallSignificand = (smaller & 0x007fffffu)
        | (smallExponent == 0u ? 0u : 0x00800000u);
    uint largeSignificand = (larger & 0x007fffffu)
        | (largeExponent == 0u ? 0u : 0x00800000u);
    uint difference = (largeExponent == 0u ? 1u : largeExponent)
        - (smallExponent == 0u ? 1u : smallExponent);
    precise float fraction = float(smallSignificand) / float(largeSignificand);
    // Scale significands instead of dividing subnormal or overflowing operands.
    if (difference > 24u) {{
        if (invert) {{ return asfloat(sign | 0x3fc90fdbu); }}
        if (negativeX) {{ return asfloat(sign | 0x40490fdbu); }}
        uint fractionBits = asuint(fraction);
        int exponent = int(fractionBits >> 23) - int(difference);
        uint result = 0u;
        if (exponent > 0) {{
            result = fractionBits - (difference << 23);
        }} else if (exponent >= -23) {{
            // Round a subnormal result to nearest-even without float flushing.
            uint shift = uint(1 - exponent);
            uint significand = (fractionBits & 0x007fffffu) | 0x00800000u;
            result = significand >> shift;
            uint remainder = significand & ((1u << shift) - 1u);
            uint halfway = 1u << (shift - 1u);
            if (remainder > halfway || (remainder == halfway && (result & 1u) != 0u)) {{
                result += 1u;
            }}
        }}
        return asfloat(result | sign);
    }}
    precise float ratio = fraction * asfloat((127u - difference) << 23);
    bool reduce = ratio > 0.4142135623730950488f;
    precise float reduced = reduce ? (ratio - 1.0f) / (ratio + 1.0f) : ratio;
    precise float squared = reduced * reduced;
    // atan's alternating series through degree 17 on |reduced| <= tan(pi/8).
    precise float series = 1.0f / 17.0f;
    series = -1.0f / 15.0f + squared * series;
    series = 1.0f / 13.0f + squared * series;
    series = -1.0f / 11.0f + squared * series;
    series = 1.0f / 9.0f + squared * series;
    series = -1.0f / 7.0f + squared * series;
    series = 1.0f / 5.0f + squared * series;
    series = -1.0f / 3.0f + squared * series;
    precise float angle = reduced + (reduced * squared) * series;
    if (reduce) {{ angle = 0.7853981633974483096f + angle; }}
    if (invert) {{ angle = 1.5707963267948966192f - angle; }}
    if (negativeX) {{ angle = 3.1415926535897932385f - angle; }}
    return asfloat(asuint(angle) | sign);
}}
"""
        for value_type in sorted(self.required_hlsl_atan2_helpers - {"float"}):
            name = self.hlsl_atan2_helper_names[value_type]
            width = int(value_type[-1])
            lanes = ", ".join(
                f"{scalar}(y.{lane}, x.{lane})" for lane in "xyzw"[:width]
            )
            code += (
                f"{value_type} {name}({value_type} y, {value_type} x) {{\n"
                f"    return {value_type}({lanes});\n}}\n"
            )
        return code + "\n"

    def generate_hlsl_atan2_call(self, func_name, args, *, call_node=None):
        if func_name != "atan2":
            return None
        if (
            self.resolve_hlsl_function_overload(func_name, args, call_node=call_node)
            is not None
        ):
            return None
        source_types = [self.hlsl_source_expression_type(argument) for argument in args]
        mapped_types = [
            self.map_type(value) if value is not None else None
            for value in source_types
        ]
        source_location = getattr(call_node, "source_location", None)

        if any(self.is_hlsl_bfloat16_type(value) for value in source_types) and (
            self.hlsl_bfloat16_builtin_result_type(func_name, source_types, len(args))
            is None
        ):
            return self.generate_hlsl_bfloat16_builtin_call(
                func_name, args, source_types, source_location=source_location
            )

        def unsupported(reason):
            return DirectXContextualConversionError(
                "DirectX canonical atan2 requires two matching floating scalar or vector operands",
                source_type=", ".join(str(value) for value in source_types),
                target_type="float or floating vector",
                reason=reason,
                source_location=source_location,
            )

        if len(args) != 2:
            raise unsupported("atan2-invalid-arity")
        if any(value is None for value in source_types):
            raise unsupported("atan2-operand-unresolved")
        bfloat = all(self.is_hlsl_bfloat16_type(value) for value in source_types)
        if mapped_types[0] != mapped_types[1]:
            raise unsupported("atan2-operand-shape-mismatch")
        shape = re.fullmatch(
            r"(float|float16_t|half|min16float)([234]?)", mapped_types[0]
        )
        if not bfloat and shape is None:
            raise unsupported("atan2-operand-type-unsupported")
        for intrinsic in ("atan2", "asuint", "asfloat"):
            if intrinsic in self.global_variable_types or (
                intrinsic in self.function_return_types
                and intrinsic not in self.hlsl_function_name_aliases
            ):
                raise unsupported("atan2-target-intrinsic-shadowed")
        value_type = "float" + (shape.group(2) if shape else "")
        self.required_hlsl_atan2_helpers.add(value_type)
        name = self.hlsl_atan2_helper_names[value_type]
        arguments = [self.generate_expression(argument) for argument in args]
        if bfloat:
            arguments = [
                self.hlsl_bfloat16_to_float_expression(argument)
                for argument in arguments
            ]
        elif mapped_types[0] != value_type:
            arguments = [f"{value_type}({argument})" for argument in arguments]
        call = f"{name}({', '.join(arguments)})"
        if bfloat:
            return self.hlsl_float_to_bfloat16_expression(call)
        if mapped_types[0] != value_type:
            return f"{mapped_types[0]}({call})"
        return call

    def prepare_hlsl_inverse_hyperbolic_helper_names(self, functions):
        used_names = self.hlsl_helper_reserved_names(functions)
        used_names.update(self.hlsl_trailing_zero_helper_names.values())
        self.hlsl_inverse_hyperbolic_helper_names = {}

        helper_keys = [("log1p", "float")]
        helper_types = sorted(
            self.HLSL_INVERSE_HYPERBOLIC_FLOAT_TYPES
            | self.HLSL_INVERSE_HYPERBOLIC_NARROW_TYPES
        )
        helper_keys.extend(
            (operation, value_type)
            for operation in sorted(self.HLSL_INVERSE_HYPERBOLIC_NAMES)
            for value_type in helper_types
        )
        for operation, value_type in helper_keys:
            base_name = f"__crossgl_{operation}_{value_type}"
            helper_name = base_name
            while helper_name in used_names:
                helper_name += "_"
            self.hlsl_inverse_hyperbolic_helper_names[(operation, value_type)] = (
                helper_name
            )
            used_names.add(helper_name)

    def hlsl_inverse_hyperbolic_helper_name(self, operation, value_type):
        key = (operation, value_type)
        helper_name = self.hlsl_inverse_hyperbolic_helper_names.get(key)
        if helper_name is not None:
            return helper_name

        used_names = set(getattr(self, "function_return_types", {}))
        used_names.update(getattr(self, "global_variable_types", {}))
        used_names.update(getattr(self, "local_variable_types", {}))
        used_names.update(self.hlsl_trailing_zero_helper_names.values())
        used_names.update(self.hlsl_inverse_hyperbolic_helper_names.values())
        helper_name = f"__crossgl_{operation}_{value_type}"
        while helper_name in used_names:
            helper_name += "_"
        self.hlsl_inverse_hyperbolic_helper_names[key] = helper_name
        return helper_name

    def hlsl_inverse_hyperbolic_log1p_helper(self):
        helper_name = self.hlsl_inverse_hyperbolic_helper_name("log1p", "float")
        return f"""float {helper_name}(float value) {{
    if (value > 0.5f) {{
        return log(1.0f + value);
    }}
    float reduced = value / (2.0f + value);
    float reduced_squared = reduced * reduced;
    float series = 1.0f + reduced_squared * (
        0.33333333333333333333f + reduced_squared * (
            0.2f + reduced_squared * (
                0.14285714285714285714f + reduced_squared * (
                    0.11111111111111111111f
                    + reduced_squared * 0.09090909090909090909f
                )
            )
        )
    );
    return (2.0f * reduced) * series;
}}
"""

    def hlsl_inverse_hyperbolic_scalar_helper(self, operation):
        helper_name = self.hlsl_inverse_hyperbolic_helper_name(operation, "float")
        log1p_name = self.hlsl_inverse_hyperbolic_helper_name("log1p", "float")
        if operation == "acosh":
            return f"""float {helper_name}(float value) {{
    if (isnan(value)) {{
        return value;
    }}
    if (value < 1.0f) {{
        return asfloat(0x7fc00000u);
    }}
    if (value == 1.0f) {{
        return 0.0f;
    }}
    if (value > 4096.0f) {{
        return log(value) + 0.69314718055994530942f;
    }}
    float offset = value - 1.0f;
    if (offset <= 0.0625f) {{
        float series = 0.00189887152777777778f;
        series = -0.00558035714285714286f + offset * series;
        series = 0.01875f + offset * series;
        series = -0.08333333333333333333f + offset * series;
        series = 1.0f + offset * series;
        return sqrt(2.0f * offset) * series;
    }}
    return {log1p_name}(offset + sqrt(offset * (value + 1.0f)));
}}
"""
        if operation == "asinh":
            return f"""float {helper_name}(float value) {{
    if (isnan(value)) {{
        return value;
    }}
    float magnitude = abs(value);
    if (magnitude < 0.000244140625f) {{
        return value;
    }}
    float result;
    if (magnitude > 4096.0f) {{
        result = log(magnitude) + 0.69314718055994530942f;
    }} else {{
        float squared = magnitude * magnitude;
        result = {log1p_name}(
            magnitude + squared / (1.0f + sqrt(1.0f + squared))
        );
    }}
    return asfloat(
        (asuint(result) & 0x7fffffffu) | (asuint(value) & 0x80000000u)
    );
}}
"""
        if operation == "atanh":
            return f"""float {helper_name}(float value) {{
    if (isnan(value)) {{
        return value;
    }}
    float magnitude = abs(value);
    if (magnitude > 1.0f) {{
        return asfloat(0x7fc00000u);
    }}
    if (magnitude == 1.0f) {{
        return asfloat(0x7f800000u | (asuint(value) & 0x80000000u));
    }}
    if (magnitude < 0.000244140625f) {{
        return value;
    }}
    float result = 0.5f * {log1p_name}(
        (2.0f * magnitude) / (1.0f - magnitude)
    );
    return asfloat(
        (asuint(result) & 0x7fffffffu) | (asuint(value) & 0x80000000u)
    );
}}
"""
        raise DirectXRefusal(f"Unsupported inverse-hyperbolic operation: {operation}")
