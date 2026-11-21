"""Program structure in the GLSL generator: subgroup control-flow refusals, struct and enum dependencies, wave matches, ray query receivers, generic helpers and deferred top-level helpers."""

import re
from copy import deepcopy

from ..ast import (
    ArrayAccessNode,
    ArrayNode,
    AssignmentNode,
    BinaryOpNode,
    CooperativeMatrixOpNode,
    FunctionCallNode,
    MemberAccessNode,
    RayQueryOpNode,
    StructNode,
    UnaryOpNode,
    VariableNode,
)
from ..dispatch_region_contract import (
    codegen_dispatch_region,
    with_dispatch_region_header,
)
from ..immutable_storage import (
    checked_immutable_storage,
    codegen_immutable_storage_names,
    with_immutable_storage_header,
)
from ..metadata_preconditions import (
    checked_metadata_preconditions,
    validate_codegen_metadata_preconditions,
    with_metadata_precondition_header,
)
from ..semantics.byte_address_buffers import lower_byte_address_buffers
from ..semantics.declared_types import declared_type_names
from ..semantics.structured_buffer_counters import (
    materialize_structured_buffer_counters,
)
from .array_utils import (
    _UnsignedLiteralInt,
    evaluate_literal_int_expression,
    format_c_style_array_declaration,
    split_array_type_suffix,
)
from .counter_loops import (
    any_start_counter_loop_terminates,
    any_start_counter_loops,
    counter_update,
    wrapping_counter_loop_uniform,
)
from .enum_utils import (
    enum_struct_fields,
    generic_enum_specialized_fields,
    generic_enum_specialized_variant_fields,
)
from .fixed_sampler_contracts import (
    fixed_sampler_records,
    with_glsl_fixed_samplers,
)
from .generic_function_utils import (
    generic_function_definition_name,
    generic_function_parameters,
)
from .generic_pointer_spaces import (
    assign_generic_pointer_spaces,
    refuse_written_generic_pointers,
)
from .generic_struct_utils import (
    generic_struct_specialized_fields,
    generic_struct_specialized_type_name,
)
from .glsl_exact_lookups import glsl_undeclared_call_reasons
from .glsl_float_atomic_storage import FloatAtomicStorage
from .glsl_refusals import stageless_entry_refusal_reason
from .glsl_string_literals import string_literal_reasons
from .glsl_target_builtins import unavailable_builtin_reasons
from .glsl_tessellation_modes import program_control_sources
from .loop_induction import (
    checked_induction_value,
    init_less_counter_loops,
    metadata_bound_key,
    offset_counter_loop_terminates,
    subgroup_strided_trips,
)
from .opengl_values import (
    OpenGLSoftwareSubgroupError,
    OpenGLSubgroupWidthError,
    OpenGLTemplateTypeError,
)
from .sampler_pairs import with_combined_sampler_header
from .stage_utils import (
    collect_stage_entry_records,
    function_stage_name,
    normalize_stage_name,
    should_emit_qualified_function,
    stage_matches,
)
from .subgroup_control_flow import has_subgroup_early_exits
from .written_type_names import refuse_written_foreign_types


class OpenGLProgramStructureMixin:
    """Program structure members of ``GLSLCodeGen``."""

    def reject_glsl_software_subgroup_control_flow(self, root):
        if self.software_subgroup_width is None:
            return
        record = self.glsl_software_subgroup_first_operation(root)
        if record is None:
            return
        operation, node = record
        raise self.glsl_software_subgroup_error(
            "OpenGL software subgroup barriers require statically uniform control flow",
            workgroup_size=(
                self.glsl_software_subgroup_workgroup_size
                or (self.software_subgroup_width, 1, 1)
            ),
            operation=operation,
            reason="potentially-divergent-control-flow",
            source_location=getattr(node, "source_location", None),
        )

    def glsl_software_subgroup_entry_counter_loops(self, entry_function):
        """Initializer-free counter loops whose counter no call may write.

        Before the helper pass resolves call effects, every name a call or a
        pointer or reference declaration reads counts as written. Loops that
        ``init_less_counter_loops`` proves start from the counter's literal
        initializer are recorded as fresh; ``any_start_counter_loops`` adds
        loops the counter may reach with any value, which the flow analysis
        must prove uniform.
        """
        mutations = getattr(self, "glsl_software_subgroup_call_mutations", {})

        def writes(node):
            if id(node) in mutations:
                return mutations[id(node)]
            if isinstance(node, VariableNode):
                return self.glsl_software_subgroup_expression_identifier_names(
                    node.initial_value
                )
            if self.glsl_software_subgroup_exit_constructor(node):
                return set()
            return self.glsl_software_subgroup_expression_identifier_names(
                list(node.arguments or [])
            )

        fresh = init_less_counter_loops(entry_function, self.walk_ast, writes)
        self.glsl_software_subgroup_fresh_counter_loops = set(fresh)
        loops = any_start_counter_loops(entry_function, self.walk_ast, writes)
        loops.update(fresh)
        return loops

    def glsl_software_subgroup_uniform_counter(self, node, uniform_names, _components):
        """The counter of ``node`` when every invocation runs its trips alike."""
        record = getattr(self, "glsl_software_subgroup_counter_loops", {}).get(id(node))
        if record is None or not self.glsl_software_subgroup_counter_loop_uniform(
            node, uniform_names
        ):
            return None
        return record[0]

    def glsl_software_subgroup_counter_loop_uniform(self, node, uniform_names):
        """Accept MLX's ``int t = 0; for (; t + C <= T; t += C)`` chunk loop.

        ``init_less_counter_loops`` proves the counter is written only by this
        loop's update and reached with its initializer's value. With a uniform
        bound, a literal offset and stride, and a body that leaves them alone,
        every invocation runs the same trips, so the counter stays uniform
        after the loop too. The checked termination proof uses the bound's
        literal value or metadata range; a 32-bit unsigned counter, which
        wraps alike on every target, needs none.
        """
        record = getattr(self, "glsl_software_subgroup_counter_loops", {}).get(id(node))
        if record is None or uniform_names is None:
            return False
        name, declaration, _loop = record
        if name not in uniform_names or any(
            self.glsl_software_subgroup_uniform_returns._qualified(value, {"volatile"})
            for value in (declaration, declaration.var_type)
        ):
            return False
        constants = {
            key: value
            for key, value in (
                getattr(self, "glsl_software_subgroup_loop_constants", {})
                .get(id(node), self.glsl_active_literal_int_constants())
                .items()
            )
            if key in uniform_names and key != name
        }
        # The initializer runs at the declaration, where a later local may
        # not yet shadow a constant it names; only literals are proven. A loop
        # the counter may reach with another value proves every start.
        start = (
            checked_induction_value(declaration.initial_value, {})
            if id(node)
            in getattr(self, "glsl_software_subgroup_fresh_counter_loops", ())
            else None
        )
        condition = getattr(node, "condition", None)
        if not isinstance(condition, BinaryOpNode):
            return False
        operator = self.map_operator(condition.operator)
        left = condition.left
        offset = 0
        offset_names = set()
        if isinstance(left, BinaryOpNode) and self.map_operator(left.operator) == "+":
            if self.expression_name(left.left) == name:
                offset_node = left.right
            elif self.expression_name(left.right) == name:
                offset_node = left.left
            else:
                return False
            offset = checked_induction_value(offset_node, constants)
            offset_names = self.glsl_software_subgroup_expression_identifier_names(
                offset_node
            )
            if offset is None or name in offset_names:
                return False
        elif self.expression_name(left) != name:
            return False
        bound = condition.right
        bound_names = self.glsl_software_subgroup_expression_identifier_names(bound)
        if name in bound_names:
            return False
        literal_bound = checked_induction_value(bound, constants)
        if literal_bound is not None:
            stops = [literal_bound]
        else:
            if not self.glsl_software_subgroup_uniform_expression(bound, uniform_names):
                return False
            record = (
                getattr(self, "glsl_software_subgroup_loop_metadata", {})
                .get(id(node), {})
                .get(metadata_bound_key(bound))
            )
            stops = []
            if record is not None:
                convert = (
                    _UnsignedLiteralInt if record["dtype"].startswith("u") else int
                )
                stops = [convert(record["minimum"]), convert(record["maximum"])]
        update = counter_update(node.update)
        if update is None or update[0] != name:
            return False
        _counter, sign, stride = update
        stride_names = set()
        step = sign
        if stride is not None:
            magnitude = checked_induction_value(stride, constants)
            step = None if magnitude is None else sign * magnitude
            stride_names = self.glsl_software_subgroup_expression_identifier_names(
                stride
            )
        type_name = self.type_name_string(declaration.var_type)
        if step is None or not (
            stops
            and (
                start is not None
                and offset_counter_loop_terminates(
                    type_name, start, offset, stops, step, operator
                )
                or any_start_counter_loop_terminates(
                    type_name, offset, stops, step, operator
                )
            )
            or wrapping_counter_loop_uniform(type_name, offset, step, operator)
        ):
            return False
        return self.glsl_software_subgroup_loop_controls_preserved(
            node, {name} | bound_names | offset_names | stride_names
        )

    def glsl_software_subgroup_trip_uniform_for(self, node, uniform_names):
        """Accept ``for (i = simd_gid; i < B; i += S)`` whose trips agree.

        The counter starts at the entry's never-written subgroup index, so it
        differs between subgroups and callers keep it out of the body's
        uniform names. With a literal bound and stride that give every
        subgroup the same trip count, as MLX's Winograd transforms'
        ``for (c = simd_group_id; c < 32; c += 4)`` over four SIMD-groups, every
        invocation reaches the body's collectives equally often.
        """
        initializer = getattr(node, "init", None)
        if (
            self.software_subgroup_width is None
            or not self.glsl_software_subgroup_count
            or not isinstance(initializer, VariableNode)
            or not initializer.name
            or any(
                self.glsl_software_subgroup_uniform_returns._qualified(
                    value, {"volatile"}
                )
                for value in (initializer, initializer.var_type)
            )
        ):
            return False
        start = getattr(initializer, "initial_value", None)
        if not self.glsl_software_subgroup_subgroup_id_expression(
            start
        ) or not self.glsl_software_subgroup_expression_identifier_names(start) <= (
            getattr(self, "glsl_software_subgroup_guard_names", set())
        ):
            return False
        loop_name = initializer.name
        condition = getattr(node, "condition", None)
        if (
            not isinstance(condition, BinaryOpNode)
            or self.expression_name(condition.left) != loop_name
            or loop_name in self.glsl_software_subgroup_expression_identifier_names(
                condition.right
            )
        ):
            return False
        constants = {
            name: value
            for name, value in (
                getattr(self, "glsl_software_subgroup_loop_constants", {})
                .get(id(node), self.glsl_active_literal_int_constants())
                .items()
            )
            if name in uniform_names and name != loop_name
        }
        update = getattr(node, "update", None)
        stride_names = set()
        if isinstance(update, UnaryOpNode):
            if (
                self.map_operator(
                    getattr(update, "op", getattr(update, "operator", None))
                )
                != "++"
                or self.expression_name(getattr(update, "operand", None)) != loop_name
            ):
                return False
            step = 1
        elif (
            isinstance(update, AssignmentNode)
            and self.map_operator(
                getattr(update, "op", getattr(update, "operator", None))
            )
            == "+="
            and self.glsl_software_subgroup_assignment_target_name(update) == loop_name
        ):
            value = getattr(update, "value", getattr(update, "right", None))
            step = evaluate_literal_int_expression(value, constants)
            stride_names = self.glsl_software_subgroup_expression_identifier_names(
                value
            )
        else:
            return False
        trips = subgroup_strided_trips(
            self.type_name_string(initializer.var_type),
            evaluate_literal_int_expression(condition.right, constants),
            step,
            self.map_operator(
                getattr(condition, "op", getattr(condition, "operator", None))
            ),
            self.glsl_software_subgroup_count,
        )
        return (
            trips is not None
            and self.glsl_software_subgroup_loop_controls_preserved(
                node,
                {loop_name}
                | stride_names
                | self.glsl_software_subgroup_expression_identifier_names(
                    condition.right
                ),
            )
        )

    def glsl_software_cooperative_matrix_uses_multiply_accumulate(
        self, ast, target_stage=None
    ):
        if not self.cooperative_matrix_software_lowering:
            return False
        return any(
            isinstance(node, CooperativeMatrixOpNode)
            and node.operation == "multiply_accumulate"
            for root in self.ray_query_search_roots(ast, target_stage)
            for node in self.walk_ast(root)
        )

    def glsl_exact_subgroup_width(self, function, shader_type=None):
        attributes = [
            attribute
            for attribute in getattr(function, "attributes", []) or []
            if self.glsl_wave_size_attribute(attribute)
        ]
        if not attributes:
            return None
        if len(attributes) != 1:
            raise OpenGLSubgroupWidthError(
                "OpenGL compute entry points require exactly one WaveSize attribute.",
                reason="source-metadata-ambiguous",
            )
        if shader_type is not None and shader_type != "compute":
            raise OpenGLSubgroupWidthError(
                "OpenGL WaveSize is only valid on compute shader entry points.",
                reason="stage-unsupported",
            )
        arguments = list(getattr(attributes[0], "arguments", []) or [])
        if len(arguments) != 1:
            raise OpenGLSubgroupWidthError(
                "OpenGL WaveSize requires exactly one integer argument.",
                reason="source-metadata-not-exact",
            )
        width = self.literal_int_value(arguments[0])
        if width not in self.GLSL_EXACT_SUBGROUP_WIDTHS:
            raise OpenGLSubgroupWidthError(
                "OpenGL WaveSize must be a power of two from 1 through 128.",
                subgroup_width=width,
                reason="target-width-unsupported",
            )
        if self.software_subgroup_width is not None:
            # Software subgroups have exactly software_subgroup_width lanes and never read
            # gl_SubgroupSize, so an equal WaveSize holds without a hardware guard.
            if width != self.software_subgroup_width:
                raise self.glsl_software_subgroup_error(
                    "OpenGL software subgroup width conflicts with the source "
                    "WaveSize contract",
                    reason="hardware-contract-conflict",
                    source_location=getattr(attributes[0], "source_location", None),
                )
            return None
        return width

    def glsl_stage_exact_subgroup_width(self, ast, target_stage=None):
        widths = set()
        for stage_name, stage in getattr(ast, "stages", {}).items():
            normalized_stage = normalize_stage_name(stage_name)
            if target_stage is not None and normalized_stage != target_stage:
                continue
            function = getattr(stage, "entry_point", None)
            if function is None:
                continue
            width = self.glsl_exact_subgroup_width(function, normalized_stage)
            if width is not None:
                widths.add(width)
        if len(widths) > 1:
            raise OpenGLSubgroupWidthError(
                "One OpenGL artifact cannot require multiple exact subgroup widths.",
                reason="entry-width-conflict",
            )
        return next(iter(widths), None)

    def glsl_stage_requires_exact_subgroup_width(self, ast, target_stage=None):
        return self.glsl_stage_exact_subgroup_width(ast, target_stage) is not None

    def generate_glsl_exact_subgroup_width_guard(self, function, shader_type):
        width = self.glsl_exact_subgroup_width(function, shader_type)
        if width is None:
            return ""
        macro = self.GLSL_REQUIRED_SUBGROUP_WIDTH_MACRO
        return f"    if (gl_SubgroupSize != {macro}) {{\n        return;\n    }}\n"

    def generate_glsl_wave_helpers(self, ast, target_stage=None):
        if self.software_subgroup_width is not None:
            return ""
        operations = self.glsl_wave_operations(ast, target_stage)
        needs_shuffle_and_fill_up = "WaveShuffleAndFillUp" in operations
        needs_match = "WaveMatch" in operations
        multi_prefix_operations = [
            operation
            for operation in self.GLSL_WAVE_MULTI_PREFIX_HELPERS
            if operation in operations
        ]
        if (
            not needs_shuffle_and_fill_up
            and not needs_match
            and not multi_prefix_operations
        ):
            return ""

        code = ""
        if needs_shuffle_and_fill_up:
            code += self.glsl_wave_shuffle_and_fill_up_helpers()
        if needs_match or multi_prefix_operations:
            code += self.glsl_wave_mask_contains_helper()
        if needs_match:
            code += self.glsl_wave_mask_set_helper()
            code += self.glsl_wave_match_helpers()
        for operation in multi_prefix_operations:
            code += self.glsl_wave_multi_prefix_helper(operation)
        return code

    def glsl_wave_shuffle_and_fill_up_helpers(self):
        code = ""
        for value_type in self.GLSL_WAVE_HELPER_VALUE_TYPES:
            code += (
                f"{value_type} crossglWaveShuffleAndFillUp("
                f"{value_type} value, {value_type} fill, uint delta, "
                "uint modulo) {\n"
                "    uint lane = gl_SubgroupInvocationID;\n"
                "    uint segmentLane = lane % modulo;\n"
                "    uint segmentBase = lane - segmentLane;\n"
                "    bool useValue = segmentLane >= delta;\n"
                "    uint valueSourceLane = useValue ? (lane - delta) : lane;\n"
                "    uint fillSourceLane = useValue\n"
                "        ? lane\n"
                "        : (segmentBase + modulo + segmentLane - delta);\n"
                f"    {value_type} shuffledValue = "
                "subgroupShuffle(value, valueSourceLane);\n"
                f"    {value_type} shuffledFill = "
                "subgroupShuffle(fill, fillSourceLane);\n"
                "    return useValue ? shuffledValue : shuffledFill;\n"
                "}\n\n"
            )
        return code

    def glsl_wave_mask_contains_helper(self):
        return (
            "bool crossglWaveMaskContains(uvec4 mask, uint lane) {\n"
            "    if (lane < 32u) {\n"
            "        return (mask.x & (1u << lane)) != 0u;\n"
            "    }\n"
            "    if (lane < 64u) {\n"
            "        return (mask.y & (1u << (lane - 32u))) != 0u;\n"
            "    }\n"
            "    if (lane < 96u) {\n"
            "        return (mask.z & (1u << (lane - 64u))) != 0u;\n"
            "    }\n"
            "    return (mask.w & (1u << (lane - 96u))) != 0u;\n"
            "}\n\n"
        )

    def glsl_wave_mask_set_helper(self):
        return (
            "void crossglWaveMaskSet(inout uvec4 mask, uint lane) {\n"
            "    if (lane < 32u) {\n"
            "        mask.x |= (1u << lane);\n"
            "    } else if (lane < 64u) {\n"
            "        mask.y |= (1u << (lane - 32u));\n"
            "    } else if (lane < 96u) {\n"
            "        mask.z |= (1u << (lane - 64u));\n"
            "    } else {\n"
            "        mask.w |= (1u << (lane - 96u));\n"
            "    }\n"
            "}\n\n"
        )

    def glsl_wave_match_helpers(self):
        code = ""
        for value_type in self.GLSL_WAVE_HELPER_VALUE_TYPES:
            condition = self.glsl_wave_match_condition(value_type)
            code += (
                f"uvec4 crossglWaveMatch({value_type} value) {{\n"
                "    uvec4 activeMask = subgroupBallot(true);\n"
                "    uvec4 result = uvec4(0u);\n"
                "    for (uint lane = 0u; lane < gl_SubgroupSize; ++lane) {\n"
                "        if (crossglWaveMaskContains(activeMask, lane) && "
                f"{condition}) {{\n"
                "            crossglWaveMaskSet(result, lane);\n"
                "        }\n"
                "    }\n"
                "    return result;\n"
                "}\n\n"
            )
        return code

    def glsl_wave_match_condition(self, value_type):
        if value_type.startswith(("bvec", "ivec", "uvec", "vec", "dvec")):
            return "all(equal(subgroupShuffle(value, lane), value))"
        return "subgroupShuffle(value, lane) == value"

    def glsl_wave_multi_prefix_helper(self, operation):
        helper_name = self.GLSL_WAVE_MULTI_PREFIX_HELPERS[operation]
        if operation == "WaveMultiPrefixCountBits":
            return (
                f"uint {helper_name}(bool value, uvec4 mask) {{\n"
                "    uint laneValue = value ? 1u : 0u;\n"
                "    uint result = 0u;\n"
                "    uvec4 activeMask = subgroupBallot(true);\n"
                "    uint limit = min(gl_SubgroupInvocationID, gl_SubgroupSize);\n"
                "    for (uint lane = 0u; lane < limit; ++lane) {\n"
                "        if (crossglWaveMaskContains(mask, lane) && "
                "crossglWaveMaskContains(activeMask, lane)) {\n"
                "            result += subgroupShuffle(laneValue, lane);\n"
                "        }\n"
                "    }\n"
                "    return result;\n"
                "}\n\n"
            )

        identity, assignment = self.glsl_wave_multi_prefix_identity_and_assignment(
            operation
        )
        code = ""
        for value_type in self.glsl_wave_multi_prefix_value_types(operation):
            code += (
                f"{value_type} {helper_name}({value_type} value, uvec4 mask) {{\n"
                f"    {value_type} result = {value_type}({identity});\n"
                "    uvec4 activeMask = subgroupBallot(true);\n"
                "    uint limit = min(gl_SubgroupInvocationID, gl_SubgroupSize);\n"
                "    for (uint lane = 0u; lane < limit; ++lane) {\n"
                "        if (crossglWaveMaskContains(mask, lane) && "
                "crossglWaveMaskContains(activeMask, lane)) {\n"
                f"            result {assignment} subgroupShuffle(value, lane);\n"
                "        }\n"
                "    }\n"
                "    return result;\n"
                "}\n\n"
            )
        return code

    def glsl_wave_multi_prefix_identity_and_assignment(self, operation):
        if operation == "WaveMultiPrefixProduct":
            return "1", "*="
        if operation == "WaveMultiPrefixBitAnd":
            return "~0", "&="
        if operation == "WaveMultiPrefixBitOr":
            return "0", "|="
        if operation == "WaveMultiPrefixBitXor":
            return "0", "^="
        return "0", "+="

    def glsl_wave_multi_prefix_value_types(self, operation):
        if operation in self.GLSL_WAVE_MULTI_PREFIX_INTEGER_OPERATIONS:
            return (
                "int",
                "ivec2",
                "ivec3",
                "ivec4",
                "uint",
                "uvec2",
                "uvec3",
                "uvec4",
            )
        return (
            "int",
            "ivec2",
            "ivec3",
            "ivec4",
            "uint",
            "uvec2",
            "uvec3",
            "uvec4",
            "float",
            "vec2",
            "vec3",
            "vec4",
            "double",
            "dvec2",
            "dvec3",
            "dvec4",
        )

    def uses_ray_extension_type(self, ast, target_stage=None):
        return (
            self.uses_ray_query(ast, target_stage)
            or self.uses_acceleration_structure(ast, target_stage)
            or self.uses_ray_tracing_position_fetch(ast, target_stage)
        )

    def uses_ray_query(self, ast, target_stage=None):
        for root in self.ray_query_search_roots(ast, target_stage):
            for node in self.walk_ast(root):
                if any(
                    self.is_ray_query_type(type_node)
                    for type_node in self.ray_extension_type_nodes(node)
                ):
                    return True
        return False

    def uses_ray_tracing_position_fetch(self, ast, target_stage=None):
        global_query_receivers = self.ray_query_declared_receiver_names(
            getattr(ast, "global_variables", []) or []
        )
        for root in self.ray_query_search_roots(ast, target_stage):
            query_receivers = (
                global_query_receivers | self.ray_query_declared_receiver_names(root)
            )
            for node in self.walk_ast(root):
                if isinstance(node, RayQueryOpNode) and (
                    node.operation in self.RAY_QUERY_POSITION_FETCH_METHODS
                ):
                    if self.ray_query_receiver_name(node.query_expr) in query_receivers:
                        return True
                if isinstance(node, FunctionCallNode):
                    func_expr = getattr(node, "function", getattr(node, "name", None))
                    if (
                        isinstance(func_expr, MemberAccessNode)
                        and func_expr.member in self.RAY_QUERY_POSITION_FETCH_METHODS
                    ):
                        if (
                            self.ray_query_receiver_name(func_expr.object)
                            in query_receivers
                        ):
                            return True
                    if (
                        self.function_call_name(node)
                        == "rayQueryGetIntersectionTriangleVertexPositionsEXT"
                    ):
                        return True
                if getattr(node, "name", None) == "gl_HitTriangleVertexPositionsEXT":
                    return True
        return False

    def uses_acceleration_structure(self, ast, target_stage=None):
        for root in self.ray_query_search_roots(ast, target_stage):
            for node in self.walk_ast(root):
                if any(
                    self.is_acceleration_structure_type(type_node)
                    for type_node in self.ray_extension_type_nodes(node)
                ):
                    return True
        return False

    def ray_extension_type_nodes(self, node):
        return (
            getattr(node, "var_type", None),
            getattr(node, "vtype", None),
            getattr(node, "param_type", None),
            getattr(node, "return_type", None),
        )

    def ray_query_declared_receiver_names(self, root):
        roots = root if isinstance(root, (list, tuple)) else [root]
        names = set()
        for search_root in roots:
            for node in self.walk_ast(search_root):
                if any(
                    self.is_ray_query_type(type_node)
                    for type_node in self.ray_extension_type_nodes(node)
                ):
                    name = getattr(node, "name", None)
                    if isinstance(name, str):
                        names.add(name)
        return names

    def ray_query_receiver_name(self, expr):
        if isinstance(expr, ArrayAccessNode):
            return self.ray_query_receiver_name(expr.array)
        name = getattr(expr, "name", None)
        return name if isinstance(name, str) else None

    def ray_query_search_roots(self, ast, target_stage=None):
        target_stage = normalize_stage_name(target_stage)
        if target_stage is None:
            return [ast]

        roots = []
        roots.extend(getattr(ast, "global_variables", []) or [])
        roots.extend(getattr(ast, "constants", []) or [])

        for func in getattr(ast, "functions", []) or []:
            stage_name = function_stage_name(func)
            if should_emit_qualified_function(target_stage, stage_name):
                roots.append(func)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if stage_matches(target_stage, stage_name):
                roots.append(stage)

        return roots

    def glsl_stage_guard_macro(self, stage_name):
        return self.GLSL_STAGE_GUARD_MACROS.get(normalize_stage_name(stage_name))

    def should_emit_stage_guards(self, ast, target_stage=None):
        if target_stage is not None:
            return False

        stage_entry_types = self.combined_stage_entry_types()
        stage_names = {
            stage_name
            for _, stage_name, _ in collect_stage_entry_records(
                ast, None, stage_entry_types
            )
        }
        return len(stage_names) > 1

    def wrap_stage_guard(self, code, stage_name):
        if (
            not code
            or self.current_target_stage is not None
            or not self.emit_stage_guards
        ):
            return code

        macro = self.glsl_stage_guard_macro(stage_name)
        if macro is None:
            return code
        if not code.endswith("\n"):
            code += "\n"
        return f"#ifdef {macro}\n{code}#endif\n"

    def ordered_generic_struct_specializations(self):
        specializations = self.generic_struct_specializations or {}
        if not specializations:
            return {}

        type_text_by_struct_name = {
            specialization["struct_name"]: type_text
            for type_text, specialization in specializations.items()
        }
        dependencies = {type_text: set() for type_text in specializations}
        for type_text, specialization in specializations.items():
            for _field_name, field_type in generic_struct_specialized_fields(
                self.type_name_string,
                specialization,
            ):
                dependency_name = generic_struct_specialized_type_name(
                    self,
                    field_type,
                )
                dependency_type = type_text_by_struct_name.get(dependency_name)
                if dependency_type and dependency_type != type_text:
                    dependencies[type_text].add(dependency_type)

        ordered = {}
        visiting = set()
        visited = set()
        type_order = {
            type_text: index for index, type_text in enumerate(specializations)
        }

        def visit(type_text):
            if type_text in visited:
                return
            if type_text in visiting:
                return
            visiting.add(type_text)
            for dependency_type in sorted(
                dependencies[type_text],
                key=lambda item: type_order[item],
            ):
                visit(dependency_type)
            visiting.remove(type_text)
            visited.add(type_text)
            ordered[type_text] = specializations[type_text]

        for type_text in specializations:
            visit(type_text)
        return ordered

    def is_struct_declaration_node(self, node):
        return isinstance(node, StructNode) or (
            getattr(node, "name", None) and hasattr(node, "members")
        )

    def mapped_type_dependency_name(self, type_value):
        mapped_type = self.map_type(type_value)
        base_type, _array_suffix = split_array_type_suffix(str(mapped_type or ""))
        return base_type.strip() or None

    def struct_node_dependency_names(self, node):
        dependencies = []
        for member in getattr(node, "members", []) or []:
            if isinstance(member, ArrayNode):
                member_type = getattr(
                    member,
                    "element_type",
                    getattr(member, "vtype", "float"),
                )
            elif hasattr(member, "member_type"):
                member_type = self.convert_type_node_to_string(member.member_type)
            else:
                member_type = getattr(member, "vtype", "float")

            dependency = self.mapped_type_dependency_name(member_type)
            if dependency is not None:
                dependencies.append(dependency)
        return dependencies

    def order_struct_declaration_entries(self, entries):
        by_name = {}
        for entry in entries:
            by_name.setdefault(entry["name"], entry)

        order_index = {entry["name"]: index for index, entry in enumerate(entries)}
        dependencies = {}
        for entry in entries:
            dependencies[entry["name"]] = [
                dependency
                for dependency in entry["dependencies"]
                if dependency in by_name and dependency != entry["name"]
            ]

        ordered = []
        visiting = set()
        visited = set()

        def visit(name):
            if name in visited:
                return
            if name in visiting:
                return
            visiting.add(name)
            for dependency in sorted(
                dependencies.get(name, []),
                key=lambda item: order_index[item],
            ):
                visit(dependency)
            visiting.remove(name)
            visited.add(name)
            ordered.append(by_name[name])

        for entry in entries:
            visit(entry["name"])
        return ordered

    def generate_ordered_data_struct_declarations(self, data_struct_nodes):
        entries = []
        for node in data_struct_nodes:
            entries.append(
                {
                    "name": node.name,
                    "dependencies": self.struct_node_dependency_names(node),
                    "code": self.generate_struct(node) + "\n",
                }
            )

        for specialization in self.ordered_generic_struct_specializations().values():
            entries.append(
                {
                    "name": specialization["struct_name"],
                    "dependencies": [
                        self.mapped_type_dependency_name(field_type)
                        for _field_name, field_type in (
                            generic_struct_specialized_fields(
                                self.type_name_string,
                                specialization,
                            )
                        )
                    ],
                    "code": self.generate_glsl_generic_struct(specialization),
                }
            )

        for enum in self.struct_payload_enums:
            fields = enum_struct_fields(enum) or []
            entries.append(
                {
                    "name": enum.name,
                    "dependencies": [
                        self.mapped_type_dependency_name(field_type)
                        for _field_name, field_type in fields
                    ],
                    "code": self.generate_glsl_payload_enum_struct(enum),
                }
            )

        for type_text, specialization in self.generic_enum_specializations.items():
            entries.append(
                {
                    "name": specialization["struct_name"],
                    "dependencies": [
                        self.mapped_type_dependency_name(field_type)
                        for _field_name, field_type in generic_enum_specialized_fields(
                            self,
                            specialization,
                        )
                    ],
                    "code": self.generate_glsl_generic_enum_struct(specialization),
                }
            )

        code = ""
        for entry in self.order_struct_declaration_entries(entries):
            code += entry["code"]
        return code

    def generate_glsl_enum_constants(self, enums, qualifier="const"):
        lines = []
        for enum in enums or ():
            next_value = 0
            for variant in getattr(enum, "variants", ()) or ():
                if getattr(variant, "value", None) is None:
                    value = str(next_value)
                    next_value += 1
                else:
                    value = self.generate_expression(variant.value)
                    literal_value = self.literal_int_value(variant.value)
                    next_value = literal_value + 1 if literal_value is not None else 0
                path = f"{enum.name}::{variant.name}"
                name = self.enum_variant_constants[path]
                lines.append(f"{qualifier} int {name} = {value};")
        return "\n".join(lines) + ("\n\n" if lines else "")

    def generate_glsl_generic_enum_constants(self, definitions, qualifier="const"):
        lines = []
        for name, definition in sorted((definitions or {}).items()):
            next_value = 0
            for variant in getattr(definition["enum"], "variants", ()) or ():
                if getattr(variant, "value", None) is None:
                    value = str(next_value)
                    next_value += 1
                else:
                    value = self.generate_expression(variant.value)
                    literal_value = self.literal_int_value(variant.value)
                    next_value = literal_value + 1 if literal_value is not None else 0
                path = f"{name}::{variant.name}"
                constant_name = self.enum_variant_constants[path]
                lines.append(f"{qualifier} int {constant_name} = {value};")
        return "\n".join(lines) + ("\n\n" if lines else "")

    def generate_glsl_struct_like_declaration(self, source_name, fields):
        type_name = self.glsl_type_identifier_name(source_name)
        code = f"struct {type_name} {{\n"
        for field_name, field_type in fields:
            member_name = self.glsl_struct_member_name(source_name, field_name)
            declaration = format_c_style_array_declaration(
                self.map_type(field_type), member_name
            )
            code += f"    {declaration};\n"
        return f"{code}}};\n\n"

    def generate_glsl_generic_struct(self, specialization):
        fields = generic_struct_specialized_fields(
            self.type_name_string,
            specialization,
        )
        return self.generate_glsl_struct_like_declaration(
            specialization["struct_name"], fields
        )

    def generate_glsl_payload_enum_struct(self, enum):
        fields = [("variant", "int"), *(enum_struct_fields(enum) or [])]
        return self.generate_glsl_struct_like_declaration(enum.name, fields)

    def generate_glsl_generic_enum_struct(self, specialization):
        fields = [
            ("variant", "int"),
            *generic_enum_specialized_fields(self, specialization),
        ]
        return self.generate_glsl_struct_like_declaration(
            specialization["struct_name"], fields
        )

    def glsl_enum_default_value(self, field_type):
        return self.default_value_expression_for_type(
            field_type
        ) or self.primitive_default_value_expression(field_type)

    def generate_glsl_enum_constructor_function(
        self,
        source_type_name,
        constructor_name,
        constant_name,
        all_fields,
        variant_fields,
    ):
        target_type = self.glsl_type_identifier_name(source_type_name)
        params = ", ".join(
            f"{self.map_type(field_type)} payload{index}"
            for index, (_field_name, field_type) in enumerate(variant_fields)
        )
        variant_member = self.glsl_struct_member_name(source_type_name, "variant")
        code = f"{target_type} {constructor_name}({params}) {{\n"
        code += f"    {target_type} result;\n"
        code += f"    result.{variant_member} = {constant_name};\n"
        active_fields = {field_name for field_name, _field_type in variant_fields}
        for field_name, field_type in all_fields:
            if field_name in active_fields:
                continue
            member_name = self.glsl_struct_member_name(source_type_name, field_name)
            code += (
                f"    result.{member_name} = "
                f"{self.glsl_enum_default_value(field_type)};\n"
            )
        for index, (field_name, _field_type) in enumerate(variant_fields):
            member_name = self.glsl_struct_member_name(source_type_name, field_name)
            code += f"    result.{member_name} = payload{index};\n"
        code += "    return result;\n"
        return f"{code}}}\n\n"

    def generate_glsl_enum_constructor_functions(self, enums):
        code = ""
        for enum in enums or ():
            all_fields = enum_struct_fields(enum) or []
            for variant in getattr(enum, "variants", ()) or ():
                path = f"{enum.name}::{variant.name}"
                variant_fields = list(
                    self.enum_struct_variant_fields.get(enum.name, {})
                    .get(variant.name, {})
                    .items()
                )
                code += self.generate_glsl_enum_constructor_function(
                    enum.name,
                    self.enum_variant_constructors[path],
                    self.enum_variant_constants[path],
                    all_fields,
                    variant_fields,
                )
        return code

    def generate_glsl_generic_enum_constructor_functions(self, specializations):
        code = ""
        for type_text, specialization in sorted((specializations or {}).items()):
            all_fields = generic_enum_specialized_fields(self, specialization)
            definition = specialization["definition"]
            for variant in getattr(definition["enum"], "variants", ()) or ():
                variant_fields = generic_enum_specialized_variant_fields(
                    self,
                    specialization,
                    variant.name,
                )
                constructor_name = self.glsl_generic_enum_constructor_names[
                    (type_text, variant.name)
                ]
                constant_name = self.enum_variant_constants[
                    f"{definition['name']}::{variant.name}"
                ]
                code += self.generate_glsl_enum_constructor_function(
                    specialization["struct_name"],
                    constructor_name,
                    constant_name,
                    all_fields,
                    variant_fields,
                )
        return code

    def default_value_expression_for_type(self, type_value):
        mapped_type = self.map_type(type_value)
        base_type, array_suffix = split_array_type_suffix(str(mapped_type or ""))
        if array_suffix:
            return None

        fields = self.default_value_struct_field_types(base_type)
        if fields is None:
            return None

        field_defaults = [
            self.default_value_expression_for_type(field_type)
            or self.primitive_default_value_expression(field_type)
            for field_type in fields
        ]
        return f"{base_type}({', '.join(field_defaults)})"

    def default_value_struct_field_types(self, type_name):
        source_type_name = self.glsl_source_type_identifier_name(type_name)
        union = self.union_storage.layout(source_type_name)
        if union is not None:
            return [union["storage_type"]]
        member_types = self.struct_member_types.get(source_type_name)
        if member_types is not None:
            return list(member_types.values())

        for enum in self.struct_payload_enums:
            if enum.name != source_type_name:
                continue
            fields = enum_struct_fields(enum) or []
            return ["int"] + [field_type for _field_name, field_type in fields]

        return None

    def primitive_default_value_expression(self, type_value):
        mapped_type = self.map_type(type_value)
        base_type, array_suffix = split_array_type_suffix(str(mapped_type or ""))
        if array_suffix:
            return f"{base_type}(0)"
        if base_type == "bool":
            return "false"
        if base_type.startswith("bool") or base_type.startswith("bvec"):
            return f"{base_type}(false)"
        return f"{base_type}(0)"

    def set_metadata_preconditions(self, records):
        self.metadata_preconditions = checked_metadata_preconditions(records)

    def set_immutable_storage(self, records):
        """Name read-only storage parameters that no writable binding aliases."""
        self.immutable_storage = checked_immutable_storage(records)

    def generate_program(self, ast, target_stage=None):
        """Render an AST to GLSL, optionally filtering stage entry points.

        Raises the emitter's ``LOWERING_REFUSAL_TYPE`` listing every construct that has no faithful lowering.
        """
        self.glsl_tessellation_control_sources = program_control_sources(self, ast)
        with self.opengl_descriptor_set_plan_for(ast):
            return self.refusals.run(
                self.render_program_with_profile_builtins, ast, target_stage
            )

    def render_program_with_profile_builtins(self, ast, target_stage=None):
        """Render the program and refuse every built-in the target profile does not have."""
        ast = lower_byte_address_buffers(ast, self.refuse_lowering)
        ast = materialize_structured_buffer_counters(ast, self.refuse_lowering)
        ast = assign_generic_pointer_spaces(ast, self.refuse_lowering)
        self.program_type_names = declared_type_names(ast)
        self.written_foreign_type_names = set()
        try:
            code = self.render_program_at_required_version(ast, target_stage)
            refuse_written_foreign_types(
                code,
                self.written_foreign_type_names,
                self.LOWERING_REFUSAL_TYPE.label,
            )
            refuse_written_generic_pointers(
                code, self.refuse_lowering, self.LOWERING_REFUSAL_TYPE.label
            )
        finally:
            self.written_foreign_type_names = None
        for reason in unavailable_builtin_reasons(
            code,
            self.GLSL_UNAVAILABLE_BUILTINS,
            self.GLSL_UNAVAILABLE_FUNCTION_PATTERNS,
        ):
            self.refuse_lowering(reason)
        for reason in string_literal_reasons(code):
            self.refuse_lowering(reason)
        for reason in glsl_undeclared_call_reasons(self, code):
            self.refuse_lowering(reason)
        return code

    def render_program_at_required_version(self, ast, target_stage=None):
        """Render the program at the lowest ``#version`` its constructs need.

        Emission decides some forms from the version (``layout(binding)``, variable gather offsets), so a program
        whose first rendering raised the version line is rendered again starting from the raised line, with only
        the refusals of that final rendering.
        """
        recorded = list(self.refusals.reasons)
        self.glsl_required_version_line = None
        try:
            code = self.render_program(ast, target_stage)
            for _ in range(self.GLSL_VERSION_RESOLUTION_PASSES - 1):
                raised = getattr(self, "glsl_raised_version_line", None)
                if raised is None:
                    break
                self.refusals.reasons = list(recorded)
                self.glsl_required_version_line = raised
                code = self.render_program(ast, target_stage)
            return code
        finally:
            self.glsl_required_version_line = None

    def render_program(self, ast, target_stage=None):
        region = codegen_dispatch_region(
            ast, self.glsl_software_subgroup_stage_entries(ast, target_stage)
        )
        records = validate_codegen_metadata_preconditions(
            self.glsl_software_subgroup_stage_entries(ast, target_stage),
            self.metadata_preconditions,
            getattr(ast, "structs", ()),
        )
        immutable, self.glsl_immutable_storage_names = codegen_immutable_storage_names(
            self.glsl_software_subgroup_stage_entries(ast, target_stage),
            self.immutable_storage,
            ast,
        )
        try:
            self.float_atomic_storage = FloatAtomicStorage(self, discover=True)
            code = self._generate_program(ast, target_stage)
            roots = self.float_atomic_storage.roots
            if roots:
                self.float_atomic_storage = FloatAtomicStorage(self, roots=roots)
                code = self._generate_program(ast, target_stage)
                self.float_atomic_storage.verify()
        finally:
            self.glsl_immutable_storage_names = frozenset()
        if records:
            code = with_metadata_precondition_header(code, records)
        code = with_combined_sampler_header(
            code, getattr(self, "combined_sampler_records", ())
        )
        if immutable:
            code = with_immutable_storage_header(code, immutable)
        code = with_glsl_fixed_samplers(
            code,
            fixed_sampler_records(ast),
            getattr(self, "combined_sampler_pairs", ()),
            ast,
        )
        return with_dispatch_region_header(code, region)

    def _generate_program(self, ast, target_stage=None):
        """Render an AST to GLSL, optionally filtering stage entry points.

        Software subgroups need every invocation at their barriers. A program
        whose early returns cannot be proven uniform is rendered once more
        with ``converge_subgroup_early_exits``; programs that validate as
        written keep their output, and a failed retry reports the original
        error.
        """
        retry = self.software_subgroup_width is not None and has_subgroup_early_exits(
            ast, self.walk_ast
        )
        pristine = deepcopy(ast) if retry else None
        try:
            return self._generate_program_with_vulkan_forms(ast, target_stage, False)
        except OpenGLSoftwareSubgroupError as error:
            if not retry or error.reason not in self.GLSL_EARLY_EXIT_RETRY_REASONS:
                raise
            try:
                return self._generate_program_with_vulkan_forms(
                    pristine, target_stage, True
                )
            except OpenGLSoftwareSubgroupError:
                raise error from None

    def _generate_program_with_vulkan_forms(self, ast, target_stage, early_exits):
        """Render the program, again without each Vulkan resource form that a use outside it would break.

        :mod:`.glsl_vulkan_resources` plans its forms from the uses it can see in the AST; a texture that the
        rendering still names outside them, or a runtime array it names without an index, keeps the combined
        OpenGL form in both compilations.
        """
        self.glsl_vulkan_resource_exclusions = frozenset()
        while True:
            code = self._generate_program_attempt(ast, target_stage, early_exits)
            unwrapped = self.glsl_vulkan_resources.unwrapped(code)
            if not unwrapped:
                return code
            self.glsl_vulkan_resource_exclusions = (
                self.glsl_vulkan_resource_exclusions | unwrapped
            )

    def collect_called_function_names(self, root):
        """Names of the functions ``root`` calls.

        A call spelled ``helper<float>(x)`` also names its generic definition ``helper``, so a call that comes
        before the definition gets the specialization's prototype like the implicit call ``helper(x)``.
        """
        names = set()
        for node in self.walk_ast(root):
            if not isinstance(node, FunctionCallNode):
                continue
            name = self.function_call_name(node)
            if name:
                names.add(name)
                names.add(generic_function_definition_name(self, name))
        return names

    def should_skip_unreachable_lowered_member_helper(self, func):
        reachable_names = self.glsl_stage_reachable_function_names
        if reachable_names is None:
            return False

        name = getattr(func, "name", None)
        if not name or name in reachable_names or "__" not in name:
            return False
        if function_stage_name(func):
            return False

        parameters = list(
            getattr(func, "parameters", getattr(func, "params", [])) or []
        )
        return bool(parameters and getattr(parameters[0], "name", None) == "self")

    def refuse_stageless_glsl_entry(self, func):
        reason = stageless_entry_refusal_reason(
            func, self.type_name_string(getattr(func, "return_type", None))
        )
        if reason is not None:
            self.refuse_lowering(reason)

    def should_skip_unresolved_top_level_helper(self, func, called_function_names):
        if generic_function_parameters(func):
            return False
        if getattr(func, "name", None) in called_function_names:
            return False

        qualifier = (
            func.qualifiers[0]
            if getattr(func, "qualifiers", None)
            else getattr(func, "qualifier", None)
        )
        if normalize_stage_name(qualifier):
            return False

        return self.function_signature_has_unresolved_type(func)

    def function_signature_has_unresolved_type(self, func):
        if self.type_name_is_unresolved_glsl_placeholder(
            getattr(func, "return_type", None)
        ):
            return True
        for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
            param_type = getattr(param, "param_type", getattr(param, "vtype", None))
            if self.type_name_is_unresolved_glsl_placeholder(param_type):
                return True
        return False

    def type_name_is_unresolved_glsl_placeholder(self, vtype):
        type_name = self.type_name_string(vtype)
        if not type_name or type_name == "void":
            return False

        try:
            mapped_type = self.map_type(type_name)
        except OpenGLTemplateTypeError:
            return True
        base_type, _array_suffix = split_array_type_suffix(str(mapped_type or ""))
        base_type = base_type.rstrip("*&").strip()
        if not base_type:
            return False
        if "<" in base_type or ">" in base_type:
            return True
        if base_type in self.type_mapping or self.is_opaque_resource_type(base_type):
            return False
        if (
            self.is_scalar_value_type(base_type)
            or self.is_vector_value_type(base_type)
            or self.is_matrix_value_type(base_type)
        ):
            return False
        if self.canonical_resource_type(base_type) is not None:
            return False
        if base_type in getattr(self, "structs_by_name", {}):
            return False
        if base_type in getattr(self, "struct_member_types", {}):
            return False
        if base_type in getattr(self, "enum_type_names", set()):
            return False
        if base_type in getattr(self, "enum_struct_type_names", set()):
            return False
        return (
            mapped_type == type_name
            and self.type_name_looks_like_generic_placeholder(base_type)
        )

    def type_name_looks_like_generic_placeholder(self, type_name):
        return bool(re.fullmatch(r"[A-Z][A-Za-z0-9_]*", str(type_name)))

    def collect_stage_deferred_top_level_helpers(self, ast, functions, target_stage):
        if target_stage is None or not getattr(ast, "stages", None):
            return {}

        deferred_by_stage = {}
        for stage_type, stage in ast.stages.items():
            stage_name = normalize_stage_name(stage_type)
            if not stage_matches(target_stage, stage_name):
                continue

            local_functions = list(getattr(stage, "local_functions", []) or [])
            deferred = self.top_level_helpers_depending_on_stage_locals(
                functions,
                local_functions,
            )
            if deferred:
                deferred_by_stage[stage_name] = deferred

        return deferred_by_stage
