"""CrossGL-to-Slang code generator."""

from ..ast import (
    AtomicOpNode,
    ArrayNode,
    ArrayAccessNode,
    ArrayLiteralNode,
    AssignmentNode,
    BinaryOpNode,
    BreakNode,
    CaseNode,
    CbufferNode,
    ContinueNode,
    ConstructorPatternNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierPatternNode,
    IdentifierNode,
    FunctionNode,
    IfNode,
    LiteralNode,
    LiteralPatternNode,
    MatchNode,
    MemberAccessNode,
    MeshOpNode,
    ParameterNode,
    PointerAccessNode,
    PointerType,
    RangeNode,
    ReferenceType,
    ReturnNode,
    RayQueryOpNode,
    RayTracingOpNode,
    ShaderNode,
    StructPatternNode,
    StructNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    DoWhileNode,
    WaveOpNode,
    WhileNode,
    WildcardPatternNode,
)
from .array_utils import (
    collect_struct_member_types,
    evaluate_literal_int_expression,
    format_c_style_array_declaration,
    get_array_size_from_node,
    split_array_type_suffix,
)
from .stage_utils import compute_local_size


class SlangCodeGen:
    """Emit Slang shader source from the shared CrossGL AST."""

    BINARY_PRECEDENCE = {
        "||": 1,
        "&&": 2,
        "|": 3,
        "^": 4,
        "&": 5,
        "==": 6,
        "!=": 6,
        "<": 7,
        ">": 7,
        "<=": 7,
        ">=": 7,
        "<<": 8,
        ">>": 8,
        "+": 9,
        "-": 9,
        "*": 10,
        "/": 10,
        "%": 10,
    }
    ASSOCIATIVE_BINARY_OPS = {"+", "*", "&&", "||", "&", "|", "^"}
    SLANG_WAVE_INTRINSIC_ARITIES = {
        "WaveGetLaneCount": 0,
        "WaveGetLaneIndex": 0,
        "WaveIsFirstLane": 0,
        "WaveActiveSum": 1,
        "WaveActiveProduct": 1,
        "WaveActiveBitAnd": 1,
        "WaveActiveBitOr": 1,
        "WaveActiveBitXor": 1,
        "WaveActiveMin": 1,
        "WaveActiveMax": 1,
        "WaveActiveAllTrue": 1,
        "WaveActiveAnyTrue": 1,
        "WaveActiveBallot": 1,
        "WaveReadLaneAt": 2,
        "WaveReadLaneFirst": 1,
        "WavePrefixSum": 1,
        "WavePrefixProduct": 1,
        "QuadReadAcrossX": 1,
        "QuadReadAcrossY": 1,
        "QuadReadAcrossDiagonal": 1,
        "QuadReadLaneAt": 2,
        "WaveMatch": 1,
        "WaveMultiPrefixSum": 2,
        "WaveMultiPrefixProduct": 2,
        "WaveMultiPrefixBitAnd": 2,
        "WaveMultiPrefixBitOr": 2,
        "WaveMultiPrefixBitXor": 2,
    }
    SLANG_MESH_INTRINSIC_ARITIES = {
        "SetMeshOutputCounts": {2},
        "DispatchMesh": {3, 4},
    }

    def __init__(self):
        """Initialize Slang generation state and helper caches."""
        self.indent_level = 0
        self.indent_str = "    "
        self.variable_types = {}
        self.image_resource_types = {}
        self.image_resource_accesses = {}
        self.buffer_resource_types = {}
        self.buffer_resource_accesses = {}
        self.helper_functions = {}
        self.helper_name_aliases = {}
        self.user_symbol_names = set()
        self.current_function_return_type = None
        self.current_shader_type = None
        self.current_expression_expected_type = None
        self.user_function_names = set()
        self.user_functions_by_name = {}
        self.user_function_return_types = {}
        self.user_function_parameter_types = {}
        self.user_struct_names = set()
        self.user_structs_by_name = {}
        self.struct_member_types = {}
        self.slang_mesh_payload_parameter_types = set()
        self.slang_ray_payload_parameter_types = set()
        self.slang_callable_data_parameter_types = set()
        self.slang_hit_attribute_parameter_types = set()
        self.stage_entry_name_overrides = {}
        self.identifier_aliases = {}
        self.slang_resource_register_cursors = {}
        self.slang_used_resource_registers = {}
        self.expression_prelude_stack = []
        self.expression_prelude_result_stack = []
        self.expression_temp_names = set()
        self.atomic_value_context_stack = []
        self.current_hull_output_rewrite = None
        self._generating = False
        self.semantic_map = {
            "gl_Position": "SV_Position",
            "gl_PointSize": "PSIZE",
            "gl_ClipDistance": "SV_ClipDistance",
            "gl_CullDistance": "SV_CullDistance",
            "gl_VertexID": "SV_VertexID",
            "gl_InstanceID": "SV_InstanceID",
            "gl_BaseVertex": "SV_StartVertexLocation",
            "gl_BaseInstance": "SV_StartInstanceLocation",
            "gl_DrawID": "SV_DrawID",
            "gl_PrimitiveID": "SV_PrimitiveID",
            "gl_PrimitiveIDIn": "SV_PrimitiveID",
            "gl_TessCoord": "SV_DomainLocation",
            "gl_TessLevelOuter": "SV_TessFactor",
            "gl_TessLevelInner": "SV_InsideTessFactor",
            "gl_Layer": "SV_RenderTargetArrayIndex",
            "gl_ViewportIndex": "SV_ViewportArrayIndex",
            "gl_FragCoord": "SV_Position",
            "gl_FrontFacing": "SV_IsFrontFace",
            "gl_FragDepth": "SV_Depth",
            "gl_FragColor": "SV_Target",
            "gl_SampleID": "SV_SampleIndex",
            "gl_FragColor0": "SV_Target0",
            "gl_FragColor1": "SV_Target1",
            "gl_FragColor2": "SV_Target2",
            "gl_FragColor3": "SV_Target3",
            "gl_FragColor4": "SV_Target4",
            "gl_FragColor5": "SV_Target5",
            "gl_FragColor6": "SV_Target6",
            "gl_FragColor7": "SV_Target7",
            "gl_WorkGroupID": "SV_GroupID",
            "gl_LocalInvocationID": "SV_GroupThreadID",
            "gl_GlobalInvocationID": "SV_DispatchThreadID",
            "gl_LocalInvocationIndex": "SV_GroupIndex",
        }
        self.function_map = {
            "mix": "lerp",
            "mod": "fmod",
            "fract": "frac",
            "inversesqrt": "rsqrt",
            "workgroupBarrier": "GroupMemoryBarrierWithGroupSync",
        }

    def indent(self):
        """Return whitespace for the current indentation level."""
        return self.indent_str * self.indent_level

    def generate(self, ast):
        """Generate Slang source for a CrossGL AST or AST fragment."""
        outermost = not self._generating
        if outermost:
            self._generating = True
            self.variable_types = {}
            self.image_resource_types = {}
            self.image_resource_accesses = {}
            self.buffer_resource_types = {}
            self.buffer_resource_accesses = {}
            self.helper_functions = {}
            self.helper_name_aliases = {}
            self.user_symbol_names = self.collect_user_symbol_names(ast)
            self.current_function_return_type = None
            self.current_shader_type = None
            self.current_expression_expected_type = None
            self.user_function_names = self.collect_user_function_names(ast)
            self.user_functions_by_name = self.collect_user_functions_by_name(ast)
            self.user_function_return_types = self.collect_user_function_return_types(
                ast
            )
            self.user_function_parameter_types = (
                self.collect_user_function_parameter_types(ast)
            )
            user_structs = self.collect_user_structs(ast)
            self.user_struct_names = {
                struct.name for struct in user_structs if getattr(struct, "name", None)
            }
            self.user_structs_by_name = {
                struct.name: struct
                for struct in user_structs
                if getattr(struct, "name", None)
            }
            self.struct_member_types = collect_struct_member_types(
                user_structs, self.type_name_string
            )
            self.slang_mesh_payload_parameter_types = set()
            self.slang_ray_payload_parameter_types = set()
            self.slang_callable_data_parameter_types = set()
            self.slang_hit_attribute_parameter_types = set()
            self.stage_entry_name_overrides = {}
            self.identifier_aliases = {}
            self.slang_resource_register_cursors = {}
            self.slang_used_resource_registers = {}
            self.expression_prelude_stack = []
            self.expression_prelude_result_stack = []
            self.expression_temp_names = set()
            self.atomic_value_context_stack = []
            self.reserve_explicit_slang_resource_declarations(ast)

        if isinstance(ast, list):
            result = ""
            for node in ast:
                result += self.generate(node) + "\n"
            return self.finish_generation(result, outermost)
        elif isinstance(ast, ShaderNode):
            return self.finish_generation(self.generate_shader(ast), outermost)
        elif isinstance(ast, StructNode):
            return self.finish_generation(self.generate_struct(ast), outermost)
        else:
            # Handle new AST structure
            result = ""

            structs = getattr(ast, "structs", [])
            for struct in structs:
                result += self.generate_struct(struct) + "\n\n"

            global_vars = getattr(ast, "global_variables", [])
            for node in global_vars:
                result += self.generate_global_variable(node)

            cbuffers = getattr(ast, "cbuffers", [])
            for node in cbuffers:
                result += self.generate_cbuffer(node) + "\n\n"

            functions = getattr(ast, "functions", [])
            for function in functions:
                # Handle both old and new AST function structures
                if hasattr(function, "qualifiers") and function.qualifiers:
                    qualifier = function.qualifiers[0] if function.qualifiers else None
                else:
                    qualifier = getattr(function, "qualifier", None)

                if qualifier == "vertex":
                    result += "// Vertex Shader\n"
                    result += self.generate_function(function) + "\n\n"
                elif qualifier == "fragment":
                    result += "// Fragment Shader\n"
                    result += self.generate_function(function) + "\n\n"
                else:
                    result += self.generate_function(function) + "\n\n"

            # Handle shader stages (new AST structure)
            if hasattr(ast, "stages") and ast.stages:
                self.validate_slang_tessellation_stage_shapes(ast.stages)
                self.slang_mesh_payload_parameter_types = (
                    self.collect_slang_mesh_payload_parameter_types(ast.stages)
                )
                self.slang_ray_payload_parameter_types = (
                    self.collect_slang_ray_payload_parameter_types(ast.stages)
                )
                self.slang_callable_data_parameter_types = (
                    self.collect_slang_callable_data_parameter_types(ast.stages)
                )
                self.slang_hit_attribute_parameter_types = (
                    self.collect_slang_hit_attribute_parameter_types(ast.stages)
                )
                self.stage_entry_name_overrides = (
                    self.collect_stage_entry_name_overrides(ast.stages)
                )
                for stage_type, stage in ast.stages.items():
                    result += self.generate_stage(stage_type, stage)

            return self.finish_generation(result, outermost)

    def finish_generation(self, result, outermost):
        if not outermost:
            return result

        helpers = self.emit_helper_functions()
        self._generating = False
        if helpers:
            return helpers + result
        return result

    def emit_helper_functions(self):
        if not self.helper_functions:
            return ""
        return "\n\n".join(self.helper_functions.values()) + "\n\n"

    def collect_user_function_names(self, node):
        names = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, FunctionNode):
                names.add(current.name)
            for function in getattr(current, "functions", []):
                collect(function)
            for function in getattr(current, "local_functions", []):
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        names.discard(None)
        return names

    def collect_user_functions_by_name(self, node):
        functions = {}
        ambiguous_names = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, FunctionNode):
                name = getattr(current, "name", None)
                if name:
                    if name in functions and functions[name] is not current:
                        ambiguous_names.add(name)
                    else:
                        functions[name] = current
            for function in getattr(current, "functions", []):
                collect(function)
            for function in getattr(current, "local_functions", []):
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        for name in ambiguous_names:
            functions.pop(name, None)
        functions.pop(None, None)
        return functions

    def collect_user_symbol_names(self, node):
        names = set()

        def add_name(current):
            name = getattr(current, "name", None)
            if name:
                names.add(name)

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, (FunctionNode, StructNode)):
                add_name(current)
            for attr in ("global_variables", "cbuffers"):
                for declaration in getattr(current, attr, []) or []:
                    add_name(declaration)
            for function in getattr(current, "functions", []) or []:
                collect(function)
            for function in getattr(current, "local_functions", []) or []:
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    for declaration in getattr(stage, "local_variables", []) or []:
                        add_name(declaration)
                    collect(stage)

        collect(node)
        names.discard(None)
        return names

    def collect_user_function_return_types(self, node):
        return_types = {}

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, FunctionNode):
                return_type = getattr(current, "return_type", None)
                return_types[current.name] = (
                    self.convert_type_node_to_string(return_type)
                    if return_type is not None
                    else "void"
                )
            for function in getattr(current, "functions", []):
                collect(function)
            for function in getattr(current, "local_functions", []):
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return_types.pop(None, None)
        return return_types

    def collect_user_function_parameter_types(self, node):
        parameter_types = {}

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, FunctionNode):
                type_names = self.function_parameter_type_names(current)
                if (
                    current.name in parameter_types
                    and parameter_types[current.name] != type_names
                ):
                    parameter_types[current.name] = []
                else:
                    parameter_types[current.name] = type_names
            for function in getattr(current, "functions", []):
                collect(function)
            for function in getattr(current, "local_functions", []):
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        parameter_types.pop(None, None)
        return parameter_types

    def function_parameter_type_names(self, function):
        names = []
        for parameter in getattr(
            function, "parameters", getattr(function, "params", [])
        ):
            if hasattr(parameter, "param_type"):
                names.append(self.convert_type_node_to_string(parameter.param_type))
            elif hasattr(parameter, "vtype"):
                names.append(str(parameter.vtype))
            elif isinstance(parameter, (list, tuple)) and parameter:
                names.append(self.type_name_string(parameter[0]))
            else:
                names.append(None)
        return names

    def collect_user_structs(self, node):
        structs = []
        seen = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, StructNode):
                node_id = id(current)
                if node_id not in seen:
                    seen.add(node_id)
                    structs.append(current)
            for struct in getattr(current, "structs", []) or []:
                collect(struct)
            for function in getattr(current, "functions", []) or []:
                collect(function)
            for function in getattr(current, "local_functions", []) or []:
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return structs

    def collect_user_struct_names(self, node):
        names = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, StructNode):
                names.add(current.name)
            for struct in getattr(current, "structs", []) or []:
                collect(struct)
            for function in getattr(current, "functions", []) or []:
                collect(function)
            for function in getattr(current, "local_functions", []) or []:
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        names.discard(None)
        return names

    def generate_shader(self, node):
        """Render a full CrossGL shader AST as a Slang translation unit."""
        result = ""

        structs = getattr(node, "structs", [])
        for struct in structs:
            result += self.generate_struct(struct) + "\n\n"

        global_vars = getattr(node, "global_variables", [])
        for global_var in global_vars:
            result += self.generate_global_variable(global_var)

        cbuffers = getattr(node, "cbuffers", [])
        for cbuffer in cbuffers:
            result += self.generate_cbuffer(cbuffer) + "\n\n"

        functions = getattr(node, "functions", [])
        for function in functions:
            stage_name = self.get_function_stage(function)
            if stage_name:
                result += f"// {stage_name.title()} Shader\n"
                result += self.generate_function(function, shader_type=stage_name)
                result += "\n\n"
            else:
                result += self.generate_function(function) + "\n\n"

        stages = getattr(node, "stages", {})
        self.validate_slang_tessellation_stage_shapes(stages)
        self.slang_mesh_payload_parameter_types = (
            self.collect_slang_mesh_payload_parameter_types(stages)
        )
        self.slang_ray_payload_parameter_types = (
            self.collect_slang_ray_payload_parameter_types(stages)
        )
        self.slang_callable_data_parameter_types = (
            self.collect_slang_callable_data_parameter_types(stages)
        )
        self.slang_hit_attribute_parameter_types = (
            self.collect_slang_hit_attribute_parameter_types(stages)
        )
        self.stage_entry_name_overrides = self.collect_stage_entry_name_overrides(
            stages
        )
        for stage_type, stage in stages.items():
            result += self.generate_stage(stage_type, stage)

        return result

    def get_stage_name(self, stage_type):
        if hasattr(stage_type, "value"):
            return stage_type.value
        return str(stage_type).split(".")[-1].lower()

    def get_function_stage(self, function):
        if hasattr(function, "qualifiers") and function.qualifiers:
            qualifier = function.qualifiers[0]
        else:
            qualifier = getattr(function, "qualifier", None)

        if qualifier in {"vertex", "fragment", "compute"}:
            return qualifier
        return None

    def slang_shader_stage_name(self, stage_name):
        """Return the Slang [shader(...)] spelling for a CrossGL stage name."""
        stage_map = {
            "tessellation_control": "hull",
            "tessellation_evaluation": "domain",
            "tesscontrol": "hull",
            "tesseval": "domain",
            "task": "amplification",
            "object": "amplification",
            "ray_generation": "raygeneration",
            "raygen": "raygeneration",
            "ray_intersection": "intersection",
            "ray_closest_hit": "closesthit",
            "ray_any_hit": "anyhit",
            "ray_miss": "miss",
            "ray_callable": "callable",
        }
        return stage_map.get(stage_name, stage_name)

    def collect_stage_entry_name_overrides(self, stages):
        """Return replacement entry names for stage blocks with duplicate names."""
        entries_by_name = {}
        for stage_type, stage in stages.items():
            entry_point = getattr(stage, "entry_point", None)
            entry_name = getattr(entry_point, "name", None)
            if entry_name is None:
                continue
            stage_name = self.get_stage_name(stage_type)
            entries_by_name.setdefault(entry_name, []).append((stage_name, entry_point))

        used_names = set(self.user_function_names)
        overrides = {}
        for entries in entries_by_name.values():
            if len(entries) < 2:
                continue
            for stage_name, entry_point in entries:
                candidate = self.slang_stage_entry_function_name(stage_name)
                unique_name = self.unique_stage_entry_name(candidate, used_names)
                used_names.add(unique_name)
                overrides[id(entry_point)] = unique_name
        return overrides

    def slang_stage_entry_function_name(self, stage_name):
        stage_name = self.slang_shader_stage_name(stage_name)
        stage_map = {
            "vertex": "VSMain",
            "fragment": "PSMain",
            "compute": "CSMain",
            "geometry": "GSMain",
            "hull": "HSMain",
            "domain": "DSMain",
            "mesh": "MSMain",
            "amplification": "ASMain",
            "raygeneration": "RayGenMain",
            "intersection": "IntersectionMain",
            "closesthit": "ClosestHitMain",
            "anyhit": "AnyHitMain",
            "miss": "MissMain",
            "callable": "CallableMain",
        }
        return stage_map.get(stage_name, f"{stage_name}_main")

    def unique_stage_entry_name(self, candidate, used_names):
        if candidate not in used_names:
            return candidate

        suffix = 1
        while f"{candidate}_{suffix}" in used_names:
            suffix += 1
        return f"{candidate}_{suffix}"

    def slang_stage_by_shader_name(self, stages, expected_stage_name):
        for stage_type, stage in (stages or {}).items():
            stage_name = self.get_stage_name(stage_type)
            if self.slang_shader_stage_name(stage_name) == expected_stage_name:
                return stage
        return None

    def validate_slang_tessellation_stage_shapes(self, stages):
        if not isinstance(stages, dict):
            return

        hull_stage = self.slang_stage_by_shader_name(stages, "hull")
        domain_stage = self.slang_stage_by_shader_name(stages, "domain")
        if hull_stage is None:
            return

        hull_entry = getattr(hull_stage, "entry_point", None)
        if hull_entry is None:
            return

        hull_shape = self.slang_hull_output_patch_shape(hull_entry)
        self.validate_slang_hull_output_patch_shape(hull_entry, hull_shape)

        if domain_stage is None:
            return

        domain_entry = getattr(domain_stage, "entry_point", None)
        if domain_entry is None:
            return

        self.validate_slang_tessellation_domain_attributes_match(
            hull_entry, domain_entry
        )
        self.validate_slang_domain_output_patch_shape(domain_entry, hull_shape)

    def validate_slang_tessellation_domain_attributes_match(
        self, hull_entry, domain_entry
    ):
        hull_domain = self.normalized_slang_stage_attribute_argument(
            hull_entry, "domain"
        )
        domain_domain = self.normalized_slang_stage_attribute_argument(
            domain_entry, "domain"
        )
        if hull_domain and not domain_domain:
            raise ValueError(
                "Slang tessellation_evaluation stage requires a domain "
                "attribute matching tessellation_control"
            )
        if domain_domain and not hull_domain:
            raise ValueError(
                "Slang tessellation_control stage requires a domain "
                "attribute matching tessellation_evaluation"
            )
        if not hull_domain or not domain_domain:
            return

        hull_domain = self.canonical_slang_tessellation_domain(hull_domain)
        domain_domain = self.canonical_slang_tessellation_domain(domain_domain)
        if hull_domain != domain_domain:
            raise ValueError(
                "Slang tessellation_evaluation stage domain "
                f"'{domain_domain}' must match tessellation_control domain "
                f"'{hull_domain}'"
            )

    def validate_slang_hull_output_patch_shape(self, hull_entry, hull_shape):
        output_patches = self.slang_patch_parameters(
            getattr(hull_entry, "parameters", getattr(hull_entry, "params", [])),
            "OutputPatch",
        )
        if len(output_patches) > 1:
            raise ValueError(
                "Slang tessellation_control stage requires at most one "
                "OutputPatch<..., N> parameter"
            )

        output_control_points = self.slang_stage_attribute_int_argument(
            hull_entry, "outputcontrolpoints"
        )
        if output_control_points is None or hull_shape is None:
            return

        _element_type, patch_size = hull_shape
        if patch_size is None or str(output_control_points) == str(patch_size):
            return

        output_patch = self.format_slang_patch_shape("OutputPatch", hull_shape)
        raise ValueError(
            "Slang tessellation_control stage "
            f"{output_patch} must match outputcontrolpoints({output_control_points})"
        )

    def validate_slang_domain_output_patch_shape(self, domain_entry, hull_shape):
        domain_output_patches = self.slang_patch_parameters(
            getattr(domain_entry, "parameters", getattr(domain_entry, "params", [])),
            "OutputPatch",
        )
        if len(domain_output_patches) > 1:
            raise ValueError(
                "Slang tessellation_evaluation stage requires at most one "
                "OutputPatch<..., N> parameter"
            )
        if not domain_output_patches or hull_shape is None:
            return

        _domain_param, domain_shape = domain_output_patches[0]
        hull_type, hull_size = hull_shape
        domain_type, domain_size = domain_shape
        if hull_type and domain_type and hull_type != domain_type:
            raise ValueError(
                "Slang tessellation_evaluation stage "
                f"{self.format_slang_patch_shape('OutputPatch', domain_shape)} "
                "must match tessellation_control output "
                f"{self.format_slang_patch_shape('OutputPatch', hull_shape)}"
            )
        if (
            hull_size is not None
            and domain_size is not None
            and hull_size != domain_size
        ):
            raise ValueError(
                "Slang tessellation_evaluation stage "
                f"{self.format_slang_patch_shape('OutputPatch', domain_shape)} "
                "must match tessellation_control output "
                f"{self.format_slang_patch_shape('OutputPatch', hull_shape)}"
            )

    def slang_hull_output_patch_shape(self, hull_entry):
        param_list = getattr(
            hull_entry, "parameters", getattr(hull_entry, "params", [])
        )
        output_patches = self.slang_patch_parameters(param_list, "OutputPatch")
        output_control_points = self.slang_stage_attribute_int_argument(
            hull_entry, "outputcontrolpoints"
        )

        return_type_name = self.convert_type_node_to_string(
            getattr(hull_entry, "return_type", "void")
        )
        if self.convert_type(return_type_name) != "void":
            return return_type_name, (
                str(output_control_points)
                if output_control_points is not None
                else None
            )

        if output_patches:
            _param, patch_shape = output_patches[0]
            element_type, patch_size = patch_shape
            return element_type, patch_size

        if output_control_points is not None:
            return None, str(output_control_points)
        return None

    def slang_stage_uses_numthreads(self, stage_name):
        shader_stage = self.slang_shader_stage_name(stage_name)
        return shader_stage in {"compute", "mesh", "amplification"}

    def slang_stage_attribute_name(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return None

        normalized = str(attr_name).lower()
        if normalized.startswith("slang_"):
            normalized = normalized[len("slang_") :]
        elif normalized.startswith("hlsl_"):
            normalized = normalized[len("hlsl_") :]

        valid_names = {
            "domain",
            "maxvertexcount",
            "maxtessfactor",
            "numthreads",
            "outputcontrolpoints",
            "outputtopology",
            "partitioning",
            "patchconstantfunc",
        }
        if normalized in valid_names:
            return normalized
        return None

    def slang_stage_attribute_arguments(self, func, expected_name):
        for attr in getattr(func, "attributes", []) or []:
            attr_name = self.slang_stage_attribute_name(attr)
            if attr_name == expected_name:
                return getattr(attr, "arguments", []) or []
        return []

    def slang_stage_attribute_names(self, func):
        names = set()
        for attr in getattr(func, "attributes", []) or []:
            attr_name = self.slang_stage_attribute_name(attr)
            if attr_name:
                names.add(attr_name)
        return names

    def slang_stage_attribute_value_to_string(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if hasattr(value, "value"):
            return str(value.value).strip('"')
        if hasattr(value, "name"):
            return str(value.name)
        return str(value)

    def canonical_slang_tessellation_domain(self, domain):
        if domain is None:
            return None
        normalized = str(domain).strip('"').lower()
        if normalized == "triangle":
            return "tri"
        return normalized

    def normalized_slang_stage_attribute_argument(self, func, expected_name):
        arguments = self.slang_stage_attribute_arguments(func, expected_name)
        if not arguments:
            return None
        value = self.slang_stage_attribute_value_to_string(arguments[0])
        if value is None:
            return None
        return str(value).strip('"').lower()

    def slang_int_literal_value(self, value):
        if value is None:
            return None
        if hasattr(value, "value"):
            value = value.value
        elif hasattr(value, "name"):
            value = value.name
        if isinstance(value, int) and not isinstance(value, bool):
            return value

        text = str(value).strip().strip('"').replace("_", "")
        if not text:
            return None
        while text and text[-1] in {"u", "U", "l", "L"}:
            text = text[:-1]
        try:
            return int(text, 0)
        except ValueError:
            return None

    def slang_stage_attribute_int_argument(self, func, expected_name):
        arguments = self.slang_stage_attribute_arguments(func, expected_name)
        if not arguments:
            return None
        return self.slang_int_literal_value(arguments[0])

    def validate_positive_slang_stage_attribute(self, func, stage_name, attr_name):
        value = self.slang_stage_attribute_int_argument(func, attr_name)
        if value is None:
            return
        if value <= 0:
            raise ValueError(
                f"Slang {stage_name} stage {attr_name} ({value}) must be positive"
            )

    def validate_slang_stage_attribute_applicability(self, func, stage_name):
        shader_stage = self.slang_shader_stage_name(stage_name)
        allowed_by_stage = {
            "compute": {"numthreads"},
            "geometry": {"maxvertexcount"},
            "hull": {
                "domain",
                "maxtessfactor",
                "outputcontrolpoints",
                "outputtopology",
                "partitioning",
                "patchconstantfunc",
            },
            "domain": {"domain"},
            "mesh": {"numthreads", "outputtopology"},
            "amplification": {"numthreads"},
        }
        allowed_attributes = allowed_by_stage.get(shader_stage, set())
        for attr_name in self.slang_stage_attribute_names(func):
            if attr_name not in allowed_attributes:
                raise ValueError(
                    f"Slang {stage_name} stage does not support "
                    f"{attr_name} attribute"
                )

    def validate_slang_tessellation_domain(self, func, stage_name):
        shader_stage = self.slang_shader_stage_name(stage_name)
        if shader_stage not in {"hull", "domain"}:
            return

        domain = self.normalized_slang_stage_attribute_argument(func, "domain")
        if not domain:
            return

        canonical_domain = self.canonical_slang_tessellation_domain(domain)
        valid_domains = {"tri", "quad", "isoline"}
        if canonical_domain not in valid_domains:
            valid_values = ", ".join(sorted(valid_domains))
            raise ValueError(
                f"Slang {stage_name} stage domain '{domain}' must be one of: "
                f"{valid_values}"
            )

    def validate_slang_tessellation_output_topology(self, func, stage_name):
        if self.slang_shader_stage_name(stage_name) != "hull":
            return

        topology = self.normalized_slang_stage_attribute_argument(
            func, "outputtopology"
        )
        if not topology:
            return

        valid_topologies = {
            "point",
            "line",
            "triangle_cw",
            "triangle_ccw",
        }
        if topology not in valid_topologies:
            valid_values = ", ".join(sorted(valid_topologies))
            raise ValueError(
                "Slang tessellation_control stage outputtopology "
                f"'{topology}' must be one of: {valid_values}"
            )

    def validate_slang_tessellation_domain_topology(self, func, stage_name):
        if self.slang_shader_stage_name(stage_name) != "hull":
            return

        domain = self.normalized_slang_stage_attribute_argument(func, "domain")
        topology = self.normalized_slang_stage_attribute_argument(
            func, "outputtopology"
        )
        if not domain or not topology:
            return

        domain = self.canonical_slang_tessellation_domain(domain)
        if domain in {"tri", "quad"} and topology == "line":
            raise ValueError(
                "Slang tessellation_control stage domain "
                f"'{domain}' requires outputtopology triangle_cw or triangle_ccw"
            )
        if domain == "isoline" and topology in {"triangle_cw", "triangle_ccw"}:
            raise ValueError(
                "Slang tessellation_control stage domain 'isoline' requires "
                "outputtopology line"
            )

    def validate_slang_mesh_output_topology(self, func, stage_name):
        if self.slang_shader_stage_name(stage_name) != "mesh":
            return

        topology = self.normalized_slang_stage_attribute_argument(
            func, "outputtopology"
        )
        if not topology:
            return

        valid_topologies = {"point", "line", "triangle"}
        if topology not in valid_topologies:
            valid_values = ", ".join(sorted(valid_topologies))
            raise ValueError(
                f"Slang mesh stage outputtopology '{topology}' must be one of: "
                f"{valid_values}"
            )

    def validate_slang_tessellation_partitioning(self, func, stage_name):
        if self.slang_shader_stage_name(stage_name) != "hull":
            return

        partitioning = self.normalized_slang_stage_attribute_argument(
            func, "partitioning"
        )
        if not partitioning:
            return

        valid_partitioning = {
            "integer",
            "fractional_even",
            "fractional_odd",
            "pow2",
        }
        if partitioning not in valid_partitioning:
            valid_values = ", ".join(sorted(valid_partitioning))
            raise ValueError(
                "Slang tessellation_control stage partitioning "
                f"'{partitioning}' must be one of: {valid_values}"
            )

    def validate_slang_patch_constant_function(self, func, stage_name):
        if self.slang_shader_stage_name(stage_name) != "hull":
            return

        arguments = self.slang_stage_attribute_arguments(func, "patchconstantfunc")
        if not arguments:
            return
        if len(arguments) != 1:
            raise ValueError(
                "Slang tessellation_control stage patchconstantfunc requires "
                "exactly one function name"
            )

        function_name = self.slang_stage_attribute_value_to_string(arguments[0])
        if not function_name:
            raise ValueError(
                "Slang tessellation_control stage patchconstantfunc requires "
                "a function name"
            )
        if function_name not in self.user_function_names:
            raise ValueError(
                "Slang tessellation_control stage patchconstantfunc "
                f"'{function_name}' does not reference a generated function"
            )

    def validate_slang_numthreads(self, func, stage_name):
        if not self.slang_stage_uses_numthreads(stage_name):
            return

        arguments = self.slang_stage_attribute_arguments(func, "numthreads")
        if not arguments:
            return
        if len(arguments) > 3:
            raise ValueError(
                f"Slang {stage_name} stage numthreads requires at most "
                "three arguments"
            )

        for argument in arguments:
            value = self.slang_int_literal_value(argument)
            if value is not None and value <= 0:
                raise ValueError(
                    f"Slang {stage_name} stage numthreads values must be positive"
                )

    def validate_slang_stage_attributes(self, func, stage_name):
        if stage_name is None:
            return

        self.validate_slang_stage_attribute_applicability(func, stage_name)
        self.validate_positive_slang_stage_attribute(func, stage_name, "maxvertexcount")
        self.validate_positive_slang_stage_attribute(
            func, stage_name, "outputcontrolpoints"
        )
        self.validate_slang_tessellation_domain(func, stage_name)
        self.validate_slang_tessellation_output_topology(func, stage_name)
        self.validate_slang_tessellation_domain_topology(func, stage_name)
        self.validate_slang_tessellation_partitioning(func, stage_name)
        self.validate_slang_patch_constant_function(func, stage_name)
        self.validate_slang_mesh_output_topology(func, stage_name)
        self.validate_slang_numthreads(func, stage_name)

    def validate_slang_mesh_payload_parameter(self, shader_type, parameters):
        payload_parameters = self.slang_mesh_payload_parameters(parameters)
        if not payload_parameters:
            return

        shader_stage = self.slang_shader_stage_name(shader_type)
        if shader_stage != "mesh":
            raise ValueError(
                f"Slang {shader_type} stage cannot declare a mesh payload parameter"
            )
        if len(payload_parameters) > 1:
            raise ValueError(
                "Slang mesh stage must declare at most one mesh payload parameter"
            )

        parameter = payload_parameters[0]
        directions = self.slang_parameter_direction_qualifiers(parameter)
        if "in" not in directions or directions & {"out", "inout"}:
            raise ValueError(
                f"Slang mesh payload parameter '{parameter.name}' "
                "must use the in qualifier"
            )
        if self.slang_parameter_user_struct_type(parameter) is None:
            raise ValueError(
                f"Slang mesh payload parameter '{parameter.name}' "
                "must use a user-defined struct type"
            )

    def validate_slang_stage_body_builtins(
        self, body_statements, stage_name, params, stage_role=None
    ):
        shader_stage = self.slang_shader_stage_name(stage_name)
        if shader_stage not in {"hull", "domain"}:
            return

        unsupported_tess_factor_builtins = {
            "gl_TessLevelOuter",
            "gl_TessLevelInner",
        }
        declared_names = {
            getattr(param, "name", None)
            for param in params or []
            if getattr(param, "name", None)
        }
        for node in self.walk_ast(body_statements):
            if isinstance(node, VariableNode):
                declared_names.add(node.name)

        if shader_stage == "hull" and stage_role == "patch_constant":
            invalid_semantics = {"SV_OutputControlPointID"}
            for param in params or []:
                semantic = self.semantic_from_node(param)
                if (
                    semantic
                    and self.map_semantic(semantic, stage_name) in invalid_semantics
                ):
                    raise ValueError(
                        "Slang patch constant functions cannot use "
                        "gl_InvocationID or SV_OutputControlPointID"
                    )

            for node in self.walk_ast(body_statements):
                if not isinstance(node, IdentifierNode):
                    continue
                if node.name == "gl_InvocationID" and node.name not in declared_names:
                    raise ValueError(
                        "Slang patch constant functions cannot use "
                        "gl_InvocationID or SV_OutputControlPointID"
                    )

        for node in self.walk_ast(body_statements):
            if not isinstance(node, IdentifierNode):
                continue
            if node.name not in unsupported_tess_factor_builtins:
                continue
            if node.name in declared_names:
                continue
            raise ValueError(
                "Slang tessellation factor built-ins gl_TessLevelOuter and "
                "gl_TessLevelInner require an explicit patch constant function "
                "return value using SV_TessFactor and SV_InsideTessFactor"
            )

    def generate_slang_stage_numthreads(self, func, stage_name, execution_config=None):
        if not self.slang_stage_uses_numthreads(stage_name):
            return ""

        arguments = self.slang_stage_attribute_arguments(func, "numthreads")
        if arguments:
            values = [
                self.slang_stage_attribute_value_to_string(argument)
                for argument in arguments
            ]
            values.extend(["1"] * (3 - len(values)))
            return f"[numthreads({', '.join(values[:3])})]\n"

        return self.generate_compute_numthreads(execution_config)

    def generate_slang_stage_attributes(self, func, stage_name):
        if stage_name not in {
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
            "mesh",
        }:
            return ""

        quoted_argument_attributes = {
            "domain",
            "outputtopology",
            "partitioning",
            "patchconstantfunc",
        }
        result = ""
        for attr in getattr(func, "attributes", []) or []:
            attr_name = self.slang_stage_attribute_name(attr)
            if attr_name is None or attr_name == "numthreads":
                continue

            arguments = getattr(attr, "arguments", []) or []
            if not arguments:
                continue

            values = [
                self.slang_stage_attribute_value_to_string(argument)
                for argument in arguments
            ]
            if attr_name == "domain":
                values = [
                    self.canonical_slang_tessellation_domain(value) or value
                    for value in values
                ]
            if attr_name in quoted_argument_attributes:
                values = [f'"{value}"' for value in values]
            result += f"[{attr_name}({', '.join(values)})]\n"

        return result

    def function_return_semantic(self, node):
        return self.semantic_from_node(node, skip_stage_attributes=True)

    def semantic_from_node(self, node, skip_stage_attributes=False):
        semantic = getattr(node, "semantic", None)
        if semantic:
            if skip_stage_attributes and self.slang_shader_stage_marker_name(semantic):
                return None
            return semantic

        for attr in getattr(node, "attributes", []) or []:
            if skip_stage_attributes and (
                self.slang_stage_attribute_name(attr)
                or self.slang_shader_stage_marker_name(getattr(attr, "name", None))
            ):
                continue
            if (
                self.is_resource_format_attribute(attr)
                or self.is_resource_binding_attribute(attr)
                or self.is_resource_memory_attribute(attr)
                or self.slang_mesh_payload_parameter_attribute_name(attr)
                or self.slang_mesh_output_parameter_role(attr)
            ):
                continue
            if hasattr(attr, "name"):
                return attr.name
        return None

    def slang_mesh_output_parameter_role(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return None

        normalized = str(attr_name).lower()
        if normalized.startswith("slang_"):
            normalized = normalized[len("slang_") :]
        elif normalized.startswith("hlsl_"):
            normalized = normalized[len("hlsl_") :]

        if normalized in {"vertices", "indices", "primitives"}:
            return normalized
        return None

    def slang_mesh_output_role_from_parameter(self, parameter):
        roles = self.slang_mesh_output_roles_from_parameter(parameter)
        if len(roles) > 1:
            raise ValueError(
                f"Slang mesh stage parameter '{parameter.name}' "
                "can use only one mesh role qualifier"
            )
        if roles:
            return roles[0]
        return None

    def slang_mesh_output_roles_from_parameter(self, parameter):
        roles = []
        for attr in getattr(parameter, "attributes", []) or []:
            role = self.slang_mesh_output_parameter_role(attr)
            if role is not None and role not in roles:
                roles.append(role)
        return roles

    def slang_parameter_mapped_base_and_array_suffix(self, parameter):
        type_name = self.slang_parameter_type_name(parameter)
        if not type_name:
            return None, None

        mapped_type = self.map_resource_type_with_format(type_name, parameter)
        return split_array_type_suffix(str(mapped_type))

    def slang_parameter_array_size_expression(self, parameter):
        type_name = self.slang_parameter_type_name(parameter)
        if not type_name:
            return None

        _base_type, array_suffix = split_array_type_suffix(str(type_name))
        if not array_suffix.startswith("["):
            return None

        closing_bracket = array_suffix.find("]")
        if closing_bracket < 0:
            return None
        size_expr = array_suffix[1:closing_bracket]
        if not size_expr:
            return None
        return size_expr

    def slang_parameter_array_count(self, parameter):
        size_expr = self.slang_parameter_array_size_expression(parameter)
        if size_expr is None:
            return None
        return self.slang_int_literal_value(size_expr)

    def validate_slang_mesh_output_array_parameter(self, parameter, role):
        directions = self.slang_parameter_direction_qualifiers(parameter)
        if "out" not in directions or directions & {"in", "inout", "const"}:
            raise ValueError(
                f"Slang mesh stage {role} parameter '{parameter.name}' "
                "must use the out qualifier"
            )

        if self.slang_parameter_array_size_expression(parameter) is None:
            raise ValueError(
                f"Slang mesh stage {role} parameter '{parameter.name}' "
                "must declare a static array size"
            )

        array_count = self.slang_parameter_array_count(parameter)
        if array_count is not None and array_count <= 0:
            raise ValueError(
                f"Slang mesh stage {role} parameter '{parameter.name}' "
                "array size must be positive"
            )

    def slang_set_mesh_output_count_calls(self, func):
        calls = []
        for node in self.walk_ast(getattr(func, "body", [])):
            if isinstance(node, MeshOpNode):
                if getattr(node, "operation", None) == "SetMeshOutputCounts":
                    calls.append(getattr(node, "arguments", []))
                continue
            if not isinstance(node, FunctionCallNode):
                continue

            func_expr = getattr(node, "function", None) or getattr(node, "name", None)
            if hasattr(func_expr, "name"):
                call_name = func_expr.name
            elif isinstance(func_expr, str):
                call_name = func_expr
            else:
                call_name = None
            if call_name == "SetMeshOutputCounts":
                calls.append(getattr(node, "arguments", getattr(node, "args", [])))
        return calls

    def validate_slang_mesh_output_parameters(self, func, shader_type, parameters):
        if self.slang_shader_stage_name(shader_type) != "mesh":
            return

        role_parameters = {}
        for parameter in parameters or []:
            roles = self.slang_mesh_output_roles_from_parameter(parameter)
            if len(roles) > 1:
                raise ValueError(
                    f"Slang mesh stage parameter '{parameter.name}' "
                    "can use only one mesh role qualifier"
                )
            if roles:
                role_parameters.setdefault(roles[0], []).append(parameter)

        if not role_parameters:
            return

        for role, role_params in role_parameters.items():
            if len(role_params) > 1:
                raise ValueError(
                    f"Slang mesh stage must declare at most one {role} "
                    "output parameter"
                )
            self.validate_slang_mesh_output_array_parameter(role_params[0], role)

        if "vertices" not in role_parameters:
            raise ValueError(
                "Slang mesh stage output signature must declare an out vertices "
                "array"
            )
        if "indices" not in role_parameters:
            raise ValueError(
                "Slang mesh stage output signature must declare an out indices array"
            )

        topology = self.normalized_slang_stage_attribute_argument(
            func, "outputtopology"
        )
        expected_index_types = {
            "point": "uint",
            "line": "uint2",
            "triangle": "uint3",
        }
        expected_index_type = expected_index_types.get(topology)
        if expected_index_type is not None:
            index_param = role_parameters["indices"][0]
            index_base_type, _array_suffix = (
                self.slang_parameter_mapped_base_and_array_suffix(index_param)
            )
            if index_base_type != expected_index_type:
                raise ValueError(
                    f"Slang mesh stage outputtopology '{topology}' requires "
                    f"indices parameter '{index_param.name}' to use "
                    f"{expected_index_type}, got {index_base_type}"
                )

        if "primitives" in role_parameters:
            index_count = self.slang_parameter_array_count(
                role_parameters["indices"][0]
            )
            primitive_count = self.slang_parameter_array_count(
                role_parameters["primitives"][0]
            )
            if (
                index_count is not None
                and primitive_count is not None
                and index_count != primitive_count
            ):
                raise ValueError(
                    "Slang mesh stage primitives output array size must match "
                    "the indices output array size"
                )

        output_count_calls = self.slang_set_mesh_output_count_calls(func)
        if len(output_count_calls) != 1:
            raise ValueError(
                "Slang mesh stage output signature must call "
                "SetMeshOutputCounts exactly once"
            )
        self.validate_slang_set_mesh_output_counts(
            output_count_calls[0], role_parameters
        )
        self.validate_slang_mesh_output_accesses(func, role_parameters)

    def validate_slang_set_mesh_output_counts(self, args, role_parameters):
        if len(args) != 2:
            raise ValueError(
                "Slang mesh SetMeshOutputCounts requires exactly two arguments"
            )

        vertex_count = self.slang_int_literal_value(args[0])
        primitive_count = self.slang_int_literal_value(args[1])
        declared_counts = {
            role: self.slang_parameter_array_count(parameters[0])
            for role, parameters in role_parameters.items()
            if role in {"vertices", "indices", "primitives"}
        }

        if (
            vertex_count is not None
            and declared_counts.get("vertices") is not None
            and vertex_count > declared_counts["vertices"]
        ):
            raise ValueError(
                "Slang mesh SetMeshOutputCounts vertex count exceeds the "
                "vertices output array size"
            )

        for role in ("indices", "primitives"):
            if (
                primitive_count is not None
                and declared_counts.get(role) is not None
                and primitive_count > declared_counts[role]
            ):
                raise ValueError(
                    "Slang mesh SetMeshOutputCounts primitive count exceeds the "
                    f"{role} output array size"
                )

    def slang_call_name(self, node):
        if isinstance(node, MeshOpNode):
            return getattr(node, "operation", None)
        if not isinstance(node, FunctionCallNode):
            return None

        func_expr = getattr(node, "function", None) or getattr(node, "name", None)
        if hasattr(func_expr, "name"):
            return func_expr.name
        if isinstance(func_expr, str):
            return func_expr
        return None

    def slang_statement_expression(self, statement):
        if isinstance(statement, ExpressionStatementNode):
            return getattr(statement, "expression", None)
        return statement

    def slang_statement_is_set_mesh_output_counts(self, statement):
        expr = self.slang_statement_expression(statement)
        return self.slang_call_name(expr) == "SetMeshOutputCounts"

    def slang_mesh_output_role_by_parameter(self, role_parameters):
        role_by_name = {}
        for role, parameters in role_parameters.items():
            if not parameters:
                continue
            param_name = getattr(parameters[0], "name", None)
            if param_name:
                role_by_name[param_name] = role
        return role_by_name

    def slang_mesh_output_access(self, target, role_by_name):
        if isinstance(target, MemberAccessNode):
            obj = getattr(target, "object", getattr(target, "object_expr", None))
            return self.slang_mesh_output_access(obj, role_by_name)
        if isinstance(target, ArrayAccessNode):
            array = getattr(target, "array", getattr(target, "array_expr", None))
            array_name = self.identifier_name(array)
            if array_name in role_by_name:
                return (
                    role_by_name[array_name],
                    array_name,
                    getattr(target, "index", getattr(target, "index_expr", None)),
                )
            return self.slang_mesh_output_access(array, role_by_name)

        target_name = self.identifier_name(target)
        if target_name in role_by_name:
            return role_by_name[target_name], target_name, None
        return None

    def slang_expression_mentions_mesh_output(self, expr, role_by_name):
        for node in self.walk_ast(expr):
            if isinstance(node, IdentifierNode) and node.name in role_by_name:
                return True
            if self.slang_mesh_output_access(node, role_by_name) is not None:
                return True
        return False

    def validate_slang_mesh_output_accesses(self, func, role_parameters):
        role_by_name = self.slang_mesh_output_role_by_parameter(role_parameters)
        if not role_by_name:
            return

        declared_counts = {
            getattr(parameters[0], "name", None): self.slang_parameter_array_count(
                parameters[0]
            )
            for parameters in role_parameters.values()
            if parameters and getattr(parameters[0], "name", None)
        }

        output_counts_seen = False
        for statement in self.get_statements(getattr(func, "body", [])):
            if self.slang_statement_is_set_mesh_output_counts(statement):
                output_counts_seen = True
                continue

            for node in self.walk_ast(statement):
                if isinstance(node, ArrayAccessNode):
                    access = self.slang_mesh_output_access(node, role_by_name)
                    if access is not None:
                        self.validate_slang_mesh_output_literal_index(
                            access, declared_counts
                        )

                if isinstance(node, AssignmentNode):
                    access = self.slang_mesh_output_access(node.left, role_by_name)
                    if access is None:
                        continue
                    role, _name, index_expr = access
                    if index_expr is None:
                        raise ValueError(
                            f"Slang mesh {role} output writes must target an "
                            "indexed output element"
                        )
                    if not output_counts_seen:
                        raise ValueError(
                            "Slang mesh output writes must occur after "
                            "SetMeshOutputCounts"
                        )

                if output_counts_seen or not isinstance(
                    node, (FunctionCallNode, MeshOpNode)
                ):
                    continue
                if self.slang_call_name(node) == "SetMeshOutputCounts":
                    continue
                for arg in getattr(node, "arguments", getattr(node, "args", [])):
                    if self.slang_expression_mentions_mesh_output(arg, role_by_name):
                        raise ValueError(
                            "Slang mesh output arrays cannot be passed to helper "
                            "calls before SetMeshOutputCounts"
                        )

    def validate_slang_mesh_output_literal_index(self, access, declared_counts):
        role, param_name, index_expr = access
        index_value = self.slang_int_literal_value(index_expr)
        if index_value is None:
            return

        declared_count = declared_counts.get(param_name)
        if declared_count is None:
            return
        if index_value < 0 or index_value >= declared_count:
            raise ValueError(
                f"Slang mesh {role} output literal index {index_value} exceeds "
                f"the {role} output array size"
            )

    def slang_mesh_output_parameter_declaration(
        self, declaration, parameter, shader_type
    ):
        role = self.slang_mesh_output_role_from_parameter(parameter)
        if role is None:
            return None

        shader_stage = self.slang_shader_stage_name(shader_type)
        if shader_stage == "mesh":
            directions = self.slang_parameter_direction_qualifiers(parameter)
            if "out" not in directions or directions & {"in", "inout", "const"}:
                raise ValueError(
                    f"Slang mesh output parameter '{parameter.name}' "
                    "must use the out qualifier"
                )
            return f"out {role} {declaration}"

        if shader_type:
            raise ValueError(
                f"Slang {shader_type} stage cannot declare mesh {role} "
                "output parameter"
            )

        return f"{self.slang_parameter_qualifier_prefix(parameter)}{declaration}"

    def slang_mesh_payload_parameter_attribute_name(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return None

        normalized = str(attr_name).lower()
        if normalized.startswith("slang_"):
            normalized = normalized[len("slang_") :]
        elif normalized.startswith("hlsl_"):
            normalized = normalized[len("hlsl_") :]

        if normalized == "mesh_payload":
            return "payload"
        return None

    def is_slang_mesh_payload_parameter(self, parameter):
        return any(
            self.slang_mesh_payload_parameter_attribute_name(attr)
            for attr in getattr(parameter, "attributes", []) or []
        )

    def slang_mesh_payload_parameters(self, parameters):
        return [
            parameter
            for parameter in parameters or []
            if self.is_slang_mesh_payload_parameter(parameter)
        ]

    def collect_slang_mesh_payload_parameter_types(self, stages):
        payload_types = set()
        if not isinstance(stages, dict):
            return payload_types

        for stage_type, stage in stages.items():
            stage_name = self.get_stage_name(stage_type)
            if self.slang_shader_stage_name(stage_name) != "mesh":
                continue

            entry_point = getattr(stage, "entry_point", None)
            if entry_point is None:
                continue

            parameters = getattr(
                entry_point, "parameters", getattr(entry_point, "params", [])
            )
            for parameter in self.slang_mesh_payload_parameters(parameters):
                payload_type = self.slang_parameter_user_struct_type(parameter)
                if payload_type is not None:
                    payload_types.add(payload_type)

        return payload_types

    def collect_slang_callable_data_parameter_types(self, stages):
        callable_data_types = set()
        if not isinstance(stages, dict):
            return callable_data_types

        for stage_type, stage in stages.items():
            stage_name = self.get_stage_name(stage_type)
            if self.slang_shader_stage_name(stage_name) != "callable":
                continue

            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                parameters = getattr(
                    entry_point, "parameters", getattr(entry_point, "params", [])
                )
                for parameter in parameters or []:
                    if (
                        self.slang_ray_semantic_role(parameter, stage_name)
                        != "callable_data"
                    ):
                        continue
                    callable_type = self.slang_parameter_user_struct_type(parameter)
                    if callable_type is not None:
                        callable_data_types.add(callable_type)

            for local_var in self.slang_stage_interface_local_variables(
                stage_name, getattr(stage, "local_variables", [])
            ):
                if (
                    self.slang_ray_semantic_role(local_var, stage_name)
                    != "callable_data"
                ):
                    continue
                callable_type = self.slang_parameter_user_struct_type(local_var)
                if callable_type is not None:
                    callable_data_types.add(callable_type)

        return callable_data_types

    def collect_slang_ray_payload_parameter_types(self, stages):
        payload_types = set()
        if not isinstance(stages, dict):
            return payload_types

        for stage_type, stage in stages.items():
            stage_name = self.get_stage_name(stage_type)
            if (
                self.slang_shader_stage_name(stage_name)
                not in self.slang_ray_stage_types()
            ):
                continue

            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                parameters = getattr(
                    entry_point, "parameters", getattr(entry_point, "params", [])
                )
                for parameter in parameters or []:
                    if self.slang_ray_semantic_role(parameter, stage_name) != "payload":
                        continue
                    payload_type = self.slang_parameter_user_struct_type(parameter)
                    if payload_type is not None:
                        payload_types.add(payload_type)

            for local_var in self.slang_stage_interface_local_variables(
                stage_name, getattr(stage, "local_variables", [])
            ):
                if self.slang_ray_semantic_role(local_var, stage_name) != "payload":
                    continue
                payload_type = self.slang_parameter_user_struct_type(local_var)
                if payload_type is not None:
                    payload_types.add(payload_type)

        return payload_types

    def collect_slang_hit_attribute_parameter_types(self, stages):
        hit_attribute_types = set()
        if not isinstance(stages, dict):
            return hit_attribute_types

        for stage_type, stage in stages.items():
            stage_name = self.get_stage_name(stage_type)
            if (
                self.slang_shader_stage_name(stage_name)
                not in self.slang_ray_stage_types()
            ):
                continue

            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                parameters = getattr(
                    entry_point, "parameters", getattr(entry_point, "params", [])
                )
                for parameter in parameters or []:
                    if (
                        self.slang_ray_semantic_role(parameter, stage_name)
                        != "hit_attribute"
                    ):
                        continue
                    hit_attribute_type = self.slang_parameter_user_struct_type(
                        parameter
                    )
                    if hit_attribute_type is not None:
                        hit_attribute_types.add(hit_attribute_type)

            for local_var in self.slang_stage_interface_local_variables(
                stage_name, getattr(stage, "local_variables", [])
            ):
                if (
                    self.slang_ray_semantic_role(local_var, stage_name)
                    != "hit_attribute"
                ):
                    continue
                hit_attribute_type = self.slang_parameter_user_struct_type(local_var)
                if hit_attribute_type is not None:
                    hit_attribute_types.add(hit_attribute_type)

        return hit_attribute_types

    def slang_parameter_direction_qualifiers(self, parameter):
        return {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "qualifiers", []) or []
            if str(qualifier).lower() in {"const", "in", "out", "inout"}
        }

    def slang_parameter_user_struct_type(self, parameter):
        type_name = self.type_name_string(
            getattr(
                parameter,
                "param_type",
                getattr(parameter, "var_type", getattr(parameter, "vtype", None)),
            )
        )
        if not type_name:
            return None

        base_type, array_suffix = split_array_type_suffix(type_name)
        if array_suffix:
            return None

        mapped_type = self.convert_type(base_type)
        if mapped_type in self.user_struct_names:
            return mapped_type
        return None

    def slang_shader_stage_marker_name(self, name):
        if name is None:
            return None

        normalized = str(name).lower()
        if normalized.startswith("slang_"):
            normalized = normalized[len("slang_") :]
        elif normalized.startswith("hlsl_"):
            normalized = normalized[len("hlsl_") :]

        valid_names = {
            "amplification",
            "anyhit",
            "callable",
            "closesthit",
            "compute",
            "domain",
            "fragment",
            "geometry",
            "hull",
            "intersection",
            "local_size",
            "local_size_x",
            "local_size_y",
            "local_size_z",
            "mesh",
            "miss",
            "numthreads",
            "object",
            "pixel",
            "ray_any_hit",
            "ray_callable",
            "ray_closest_hit",
            "ray_generation",
            "ray_intersection",
            "ray_miss",
            "raygen",
            "raygeneration",
            "task",
            "tesscontrol",
            "tesseval",
            "tessellation_control",
            "tessellation_evaluation",
            "vertex",
            "workgroup_size",
        }
        if normalized in valid_names:
            return normalized
        return None

    def is_resource_format_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        attr_name = str(attr_name).lower()
        return attr_name == "format" or attr_name in self.supported_image_formats()

    def is_resource_binding_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        return str(attr_name).lower() in {
            "binding",
            "buffer",
            "group",
            "register",
            "sampler",
            "set",
            "space",
            "texture",
        }

    def is_resource_memory_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        return str(attr_name).lower() in {
            "coherent",
            "globallycoherent",
            "readonly",
            "readwrite",
            "restrict",
            "volatile",
            "writeonly",
        }

    def stage_semantic_map(self, shader_type):
        shader_stage = self.slang_shader_stage_name(shader_type)
        if shader_stage == "geometry":
            return {
                "gl_PrimitiveIDIn": "SV_PrimitiveID",
                "gl_InvocationID": "SV_GSInstanceID",
            }
        if shader_stage == "hull":
            return {
                "gl_InvocationID": "SV_OutputControlPointID",
                "gl_PrimitiveID": "SV_PrimitiveID",
            }
        if shader_stage == "domain":
            return {
                "gl_TessCoord": "SV_DomainLocation",
                "gl_PrimitiveID": "SV_PrimitiveID",
            }
        return {}

    def map_semantic(self, semantic, shader_type=None):
        if semantic is None:
            return None
        semantic_name = str(semantic)
        stage_semantic = self.stage_semantic_map(shader_type).get(semantic_name)
        if stage_semantic:
            return stage_semantic
        if semantic_name.startswith("gl_FragColor"):
            target_index = semantic_name[len("gl_FragColor") :]
            if target_index.isdigit():
                return f"SV_Target{target_index}"
        return self.semantic_map.get(semantic_name, semantic_name)

    def semantic_suffix(self, semantic, shader_type=None):
        mapped_semantic = self.map_semantic(semantic, shader_type)
        return f" : {mapped_semantic}" if mapped_semantic else ""

    def generate_stage(self, stage_type, stage):
        """Render one staged entry point and its local functions."""
        stage_name = self.get_stage_name(stage_type)
        result = f"// {stage_name.title()} Shader\n"

        local_variables = getattr(stage, "local_variables", [])
        for local_var in self.slang_stage_global_local_variables(
            stage_name, local_variables
        ):
            result += self.generate_global_variable(local_var)

        entry_point = getattr(stage, "entry_point", None)
        local_functions = getattr(stage, "local_functions", [])
        self.validate_slang_stage_patch_constant_function_shapes(
            entry_point, local_functions, stage_name
        )
        patch_constant_function_names = self.slang_stage_patch_constant_function_names(
            entry_point, stage_name
        )
        for func in local_functions:
            if getattr(func, "name", None) in patch_constant_function_names:
                result += (
                    self.generate_function(
                        func,
                        shader_type=stage_name,
                        emit_stage_decorations=False,
                        stage_role="patch_constant",
                    )
                    + "\n\n"
                )
            else:
                result += self.generate_function(func) + "\n\n"

        if entry_point is not None:
            result += self.generate_function(
                entry_point,
                shader_type=stage_name,
                execution_config=getattr(stage, "execution_config", None),
                entry_name=self.stage_entry_name_overrides.get(id(entry_point)),
                extra_parameters=self.slang_stage_interface_parameters(
                    stage_name, local_variables
                ),
            )
            result += "\n\n"

        return result

    def slang_stage_interface_local_variables(self, stage_name, local_variables):
        """Return stage-local declarations that lower to Slang entry parameters."""
        shader_stage = self.slang_shader_stage_name(stage_name)
        if shader_stage not in self.slang_ray_stage_types():
            return []

        return [
            local_var
            for local_var in local_variables or []
            if self.slang_ray_semantic_role(local_var, stage_name) is not None
        ]

    def slang_stage_global_local_variables(self, stage_name, local_variables):
        interface_ids = {
            id(local_var)
            for local_var in self.slang_stage_interface_local_variables(
                stage_name, local_variables
            )
        }
        return [
            local_var
            for local_var in local_variables or []
            if id(local_var) not in interface_ids
        ]

    def slang_stage_interface_parameters(self, stage_name, local_variables):
        parameters = []
        for local_var in self.slang_stage_interface_local_variables(
            stage_name, local_variables
        ):
            if getattr(local_var, "initial_value", None) is not None:
                raise ValueError(
                    "Slang ray stage interface variable "
                    f"'{local_var.name}' cannot have an initializer"
                )

            parameter = ParameterNode(
                name=local_var.name,
                param_type=getattr(local_var, "var_type", getattr(local_var, "vtype")),
                attributes=list(getattr(local_var, "attributes", []) or []),
                qualifiers=list(getattr(local_var, "qualifiers", []) or []),
                source_location=getattr(local_var, "source_location", None),
            )
            parameter.semantic = getattr(local_var, "semantic", None)
            parameter.add_annotation("slang_stage_local_interface", True)
            parameters.append(parameter)
        return parameters

    def slang_merge_function_parameters(
        self, param_list, extra_parameters, shader_type=None
    ):
        if not extra_parameters:
            return param_list

        merged_parameters = list(param_list or [])
        existing_names = {
            getattr(param, "name", None)
            for param in merged_parameters
            if getattr(param, "name", None)
        }
        for parameter in extra_parameters:
            if parameter.name in existing_names:
                stage_name = shader_type or "function"
                raise ValueError(
                    f"Slang {stage_name} stage interface variable "
                    f"'{parameter.name}' duplicates an entry parameter"
                )
            existing_names.add(parameter.name)
            merged_parameters.append(parameter)
        return merged_parameters

    def slang_stage_patch_constant_function_names(self, entry_point, stage_name):
        if entry_point is None or self.slang_shader_stage_name(stage_name) != "hull":
            return set()

        arguments = self.slang_stage_attribute_arguments(
            entry_point, "patchconstantfunc"
        )
        if len(arguments) != 1:
            return set()

        function_name = self.slang_stage_attribute_value_to_string(arguments[0])
        return {function_name} if function_name else set()

    def validate_slang_stage_patch_constant_function_shapes(
        self, entry_point, local_functions, stage_name
    ):
        if entry_point is None or self.slang_shader_stage_name(stage_name) != "hull":
            return

        patch_constant_function_names = self.slang_stage_patch_constant_function_names(
            entry_point, stage_name
        )
        if not patch_constant_function_names:
            return

        local_functions_by_name = {
            getattr(func, "name", None): func for func in local_functions or []
        }
        hull_input_patches = self.slang_patch_parameters(
            getattr(entry_point, "parameters", getattr(entry_point, "params", [])),
            "InputPatch",
        )

        for function_name in patch_constant_function_names:
            patch_constant_function = local_functions_by_name.get(function_name)
            if patch_constant_function is None:
                continue

            return_type_name = self.convert_type_node_to_string(
                getattr(patch_constant_function, "return_type", "void")
            )
            if self.convert_type(return_type_name) == "void":
                raise ValueError(
                    "Slang tessellation_control stage patchconstantfunc "
                    f"'{function_name}' requires a non-void return type"
                )

            patch_constant_input_patches = self.slang_patch_parameters(
                getattr(
                    patch_constant_function,
                    "parameters",
                    getattr(patch_constant_function, "params", []),
                ),
                "InputPatch",
            )
            if len(patch_constant_input_patches) > 1:
                raise ValueError(
                    "Slang tessellation_control stage patchconstantfunc "
                    f"'{function_name}' requires at most one "
                    "InputPatch<..., N> parameter"
                )

            if patch_constant_input_patches and hull_input_patches:
                _patch_param, patch_shape = patch_constant_input_patches[0]
                _hull_param, hull_shape = hull_input_patches[0]
                if patch_shape != hull_shape:
                    patch_type = self.format_slang_patch_shape(
                        "InputPatch", patch_shape
                    )
                    hull_type = self.format_slang_patch_shape("InputPatch", hull_shape)
                    raise ValueError(
                        "Slang tessellation_control stage patchconstantfunc "
                        f"'{function_name}' {patch_type} must match hull entry "
                        f"{hull_type}"
                    )

            self.validate_slang_patch_constant_tess_factor_semantics(
                entry_point, patch_constant_function, function_name
            )

    def validate_slang_patch_constant_tess_factor_semantics(
        self, hull_entry, patch_function, function_name
    ):
        return_struct = self.slang_return_struct(patch_function)
        if return_struct is None:
            return

        semantics = self.slang_struct_semantics(return_struct)
        if "sv_tessfactor" not in semantics:
            raise ValueError(
                "Slang tessellation_control stage patchconstantfunc "
                f"'{function_name}' must return a struct containing "
                "SV_TessFactor"
            )

        domain = self.normalized_slang_stage_attribute_argument(hull_entry, "domain")
        domain = self.canonical_slang_tessellation_domain(domain)
        if domain in {"tri", "quad"} and "sv_insidetessfactor" not in semantics:
            raise ValueError(
                "Slang tessellation_control stage patchconstantfunc "
                f"'{function_name}' must return a struct containing "
                f"SV_InsideTessFactor for {domain} domains"
            )

        self.validate_slang_tess_factor_member_types(return_struct, function_name)

        factor_counts = self.slang_struct_semantic_member_counts(return_struct)
        expected_outer_counts = {
            "tri": 3,
            "quad": 4,
            "isoline": 2,
        }
        expected_inner_counts = {
            "tri": 1,
            "quad": 2,
        }
        expected_outer_count = expected_outer_counts.get(domain)
        expected_inner_count = expected_inner_counts.get(domain)
        outer_count = factor_counts.get("sv_tessfactor")
        inner_count = factor_counts.get("sv_insidetessfactor")

        if (
            expected_outer_count is not None
            and outer_count is not None
            and outer_count != expected_outer_count
        ):
            raise ValueError(
                "Slang tessellation_control stage patchconstantfunc "
                f"'{function_name}' must return {expected_outer_count} "
                f"SV_TessFactor value(s) for {domain} domains"
            )
        if (
            expected_inner_count is not None
            and inner_count is not None
            and inner_count != expected_inner_count
        ):
            raise ValueError(
                "Slang tessellation_control stage patchconstantfunc "
                f"'{function_name}' must return {expected_inner_count} "
                f"SV_InsideTessFactor value(s) for {domain} domains"
            )
        if expected_inner_count is None and inner_count is not None:
            raise ValueError(
                "Slang tessellation_control stage patchconstantfunc "
                f"'{function_name}' must not return SV_InsideTessFactor "
                f"for {domain} domains"
            )

    def validate_slang_tess_factor_member_types(self, struct_node, function_name):
        tess_factor_semantics = {
            "sv_tessfactor": "SV_TessFactor",
            "sv_insidetessfactor": "SV_InsideTessFactor",
        }
        for member in getattr(struct_node, "members", []) or []:
            semantic = self.semantic_from_node(member)
            if semantic is None:
                continue

            semantic_key = str(self.map_semantic(semantic)).lower()
            semantic_name = tess_factor_semantics.get(semantic_key)
            if semantic_name is None:
                continue

            if self.slang_tess_factor_member_count(member) is not None:
                continue

            member_name = getattr(member, "name", "<anonymous>")
            member_type = self.slang_tess_factor_member_type_name(member)
            mapped_type = self.convert_type(member_type) or member_type
            raise ValueError(
                "Slang tessellation_control stage patchconstantfunc "
                f"'{function_name}' {semantic_name} member '{member_name}' "
                f"uses invalid type {mapped_type}; tessellation factors "
                "must use a floating scalar, vector, or scalar array type"
            )

    def slang_return_struct(self, func):
        return_type_name = self.type_name_string(
            getattr(func, "return_type", getattr(func, "vtype", None))
        )
        if not return_type_name:
            return None
        base_type = return_type_name.split("<", 1)[0].split("[", 1)[0].strip()
        return self.user_structs_by_name.get(base_type)

    def slang_struct_semantics(self, struct_node):
        semantics = set()
        for member in getattr(struct_node, "members", []) or []:
            semantic = self.semantic_from_node(member)
            if semantic is None:
                continue
            semantics.add(str(self.map_semantic(semantic)).lower())
        return semantics

    def slang_struct_semantic_member_counts(self, struct_node):
        counts = {}
        for member in getattr(struct_node, "members", []) or []:
            semantic = self.semantic_from_node(member)
            if semantic is None:
                continue
            semantic_key = str(self.map_semantic(semantic)).lower()
            count = self.slang_tess_factor_member_count(member)
            if count is not None:
                counts[semantic_key] = counts.get(semantic_key, 0) + count
        return counts

    def slang_tess_factor_member_count(self, member):
        type_name = self.slang_tess_factor_member_type_name(member)
        mapped_type = self.convert_type(type_name)
        if mapped_type is None:
            return None

        base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        if array_suffix:
            if self.slang_tess_factor_scalar_count(base_type) != 1:
                return None

            first_dimension = array_suffix[1:].split("]", 1)[0]
            remaining_dimensions = array_suffix.split("]", 1)[1]
            if remaining_dimensions:
                return None

            try:
                return int(first_dimension)
            except ValueError:
                return None

        return self.slang_tess_factor_scalar_count(mapped_type)

    def slang_tess_factor_member_type_name(self, member):
        member_type = getattr(member, "member_type", getattr(member, "vtype", None))
        if member_type is None and hasattr(member, "element_type"):
            member_type = member.element_type
        return self.type_name_string(member_type)

    def slang_tess_factor_scalar_count(self, mapped_type):
        if mapped_type is None:
            return None

        for scalar_base in ("min16float", "float", "half", "double"):
            if mapped_type.startswith(scalar_base):
                suffix = mapped_type[len(scalar_base) :]
                if suffix in {"2", "3", "4"}:
                    return int(suffix)
                if suffix == "":
                    return 1
        return None

    def format_slang_patch_shape(self, patch_type, shape):
        element_type, patch_size = shape
        if patch_size is None:
            return f"{patch_type}<{element_type}>"
        return f"{patch_type}<{element_type}, {patch_size}>"

    def convert_type_node_to_string(self, type_node) -> str:
        if isinstance(type_node, LiteralNode):
            return self.generate_literal(type_node)
        if isinstance(type_node, PointerType):
            return f"{self.convert_type_node_to_string(type_node.pointee_type)}*"
        if isinstance(type_node, ReferenceType):
            return f"{self.convert_type_node_to_string(type_node.referenced_type)}&"
        if hasattr(type_node, "name"):
            generic_args = getattr(type_node, "generic_args", [])
            if generic_args:
                args = ", ".join(
                    self.convert_type_node_to_string(arg) for arg in generic_args
                )
                return f"{type_node.name}<{args}>"
            return type_node.name
        if hasattr(type_node, "rows") and hasattr(type_node, "cols"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            if element_type == "float":
                if type_node.rows == type_node.cols:
                    return f"mat{type_node.rows}"
                return f"mat{type_node.rows}x{type_node.cols}"
            return f"{element_type}{type_node.rows}x{type_node.cols}"
        if hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            if type_node.__class__.__name__ == "ArrayType":
                if type_node.size is None:
                    return f"{element_type}[]"
                size = self.format_array_size_expression(type_node.size)
                return f"{element_type}[{size}]"
            if element_type == "float":
                return f"vec{type_node.size}"
            if element_type == "int":
                return f"ivec{type_node.size}"
            if element_type == "uint":
                return f"uvec{type_node.size}"
            if element_type == "bool":
                return f"bvec{type_node.size}"
            return f"{element_type}{type_node.size}"
        return str(type_node)

    def format_array_size_expression(self, expr):
        if isinstance(expr, int):
            return str(expr)
        if isinstance(expr, BinaryOpNode):
            left = self.format_array_size_expression(expr.left)
            right = self.format_array_size_expression(expr.right)
            return f"({left} {expr.op} {right})"
        if isinstance(expr, UnaryOpNode):
            return f"{expr.op}{self.format_array_size_expression(expr.operand)}"
        return self.generate_expression(expr)

    def format_declaration(self, type_name, name, node=None):
        mapped_type = self.map_resource_type_with_format(type_name, node)
        return format_c_style_array_declaration(mapped_type, name)

    def slang_declaration_qualifier_prefix(self, node):
        qualifiers = []
        seen = set()
        for qualifier in getattr(node, "qualifiers", []) or []:
            normalized = str(qualifier).lower()
            if normalized in {"groupshared", "shared", "threadgroup", "workgroup"}:
                mapped = "groupshared"
            else:
                mapped = None
            if mapped and mapped not in seen:
                seen.add(mapped)
                qualifiers.append(mapped)

        if not qualifiers:
            return ""
        return " ".join(qualifiers) + " "

    def slang_parameter_qualifier_prefix(self, node):
        qualifiers = []
        seen = set()
        for qualifier in getattr(node, "qualifiers", []) or []:
            normalized = str(qualifier).lower()
            if normalized in {"const", "in", "out", "inout"}:
                mapped = normalized
            else:
                mapped = None
            if mapped and mapped not in seen:
                seen.add(mapped)
                qualifiers.append(mapped)

        if not qualifiers:
            return ""
        return " ".join(qualifiers) + " "

    def get_variable_type(self, node):
        var_type = getattr(node, "var_type", None)
        if var_type is not None:
            return self.convert_type_node_to_string(var_type)

        vtype = getattr(node, "vtype", None)
        if vtype is not None and vtype != "":
            return vtype

        return None

    def variable_declaration_type(self, node, initial_value=None):
        var_type = self.get_variable_type(node)
        if var_type is not None:
            return var_type
        if initial_value is not None:
            return self.expression_result_type(initial_value) or "auto"
        return "float"

    def initializer_expected_type(self, var_type):
        return None if var_type == "auto" else var_type

    def array_literal_element_expected_type(self, expected_type):
        expected_type = self.type_name_string(expected_type)
        if not expected_type or "[" not in expected_type:
            return None

        return self.array_element_type(expected_type)

    def array_element_type(self, array_type):
        array_type = self.type_name_string(array_type)
        if not array_type or "[" not in array_type:
            return None

        base_type, array_suffix = split_array_type_suffix(array_type)
        if not array_suffix.startswith("["):
            return None

        closing_bracket = array_suffix.find("]")
        if closing_bracket < 0:
            return None

        remaining_suffix = array_suffix[closing_bracket + 1 :]
        return f"{base_type}{remaining_suffix}" if remaining_suffix else base_type

    def register_variable_type(self, name, type_name, node=None):
        if not name or type_name is None:
            return
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        self.variable_types[name] = type_name
        mapped_type = self.map_resource_type_with_format(type_name, node)
        if self.is_storage_image_type(type_name):
            self.image_resource_types[name] = mapped_type
            access = self.explicit_resource_access(node)
            if access is not None:
                self.image_resource_accesses[name] = access
        if self.is_buffer_resource_type(mapped_type):
            self.buffer_resource_types[name] = self.resource_base_type(mapped_type)
            access = self.explicit_resource_access(node)
            if access is not None:
                self.buffer_resource_accesses[name] = access

    def binding_index_value(self, value, prefixes=()):
        if hasattr(value, "value") and value.value is not None:
            raw_value = value.value
        elif hasattr(value, "name") and value.name is not None:
            raw_value = value.name
        else:
            raw_value = self.attribute_value_to_string(value)
        if raw_value is None:
            return None
        raw_value = str(raw_value).strip().lower()
        if raw_value.isdigit():
            return int(raw_value)
        for prefix in prefixes:
            if raw_value.startswith(prefix) and raw_value[len(prefix) :].isdigit():
                return int(raw_value[len(prefix) :])
        return None

    def binding_expr_value(self, value, prefixes=()):
        text = self.attribute_value_to_string(value)
        if text is None:
            return None
        text = str(text).strip()
        lower_text = text.lower()
        for prefix in prefixes:
            if lower_text.startswith(prefix) and text[len(prefix) :]:
                return text[len(prefix) :]
        return text

    def register_space_index(self, value):
        if hasattr(value, "value") and value.value is not None:
            raw_value = value.value
        elif hasattr(value, "name") and value.name is not None:
            raw_value = value.name
        else:
            raw_value = self.attribute_value_to_string(value)
        if raw_value is None:
            return None
        raw_value = str(raw_value).strip().lower()
        if raw_value.isdigit():
            return int(raw_value)
        if raw_value.startswith("space") and raw_value[5:].isdigit():
            return int(raw_value[5:])
        return None

    def register_space_expr(self, value):
        text = self.attribute_value_to_string(value)
        if text is None:
            return None
        text = str(text).strip()
        lower_text = text.lower()
        if lower_text.startswith("space") and text[5:]:
            return text[5:]
        return text

    def explicit_slang_resource_binding(self, node):
        binding = None
        binding_expr = None
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            arguments = getattr(attr, "arguments", []) or []
            if attr_name != "binding" or not arguments:
                continue
            binding = self.binding_index_value(arguments[0])
            binding_expr = self.binding_expr_value(arguments[0])
            break
        return binding, binding_expr

    def explicit_slang_resource_set(self, node):
        descriptor_set = None
        descriptor_set_expr = None
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            arguments = getattr(attr, "arguments", []) or []
            if attr_name not in {"set", "group", "space"} or not arguments:
                continue
            descriptor_set = self.binding_index_value(arguments[0])
            descriptor_set_expr = self.binding_expr_value(arguments[0])
            break
        return descriptor_set, descriptor_set_expr

    def explicit_slang_register(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            arguments = getattr(attr, "arguments", []) or []
            if attr_name != "register" or not arguments:
                continue
            register_arg = self.attribute_value_to_string(arguments[0])
            if register_arg is None:
                return None, None, None, None, None
            register_arg = str(register_arg).strip()
            register_prefix = ""
            for char in register_arg:
                if not char.isalpha():
                    break
                register_prefix += char
            register_prefix = register_prefix.lower() or None
            binding = self.binding_index_value(arguments[0], (register_prefix or "",))
            binding_expr = self.binding_expr_value(
                arguments[0], (register_prefix or "",)
            )
            space = None
            space_expr = None
            for argument in arguments[1:]:
                space = self.register_space_index(argument)
                space_expr = self.register_space_expr(argument)
                if space_expr is not None:
                    break
            return register_prefix, binding, binding_expr, space, space_expr
        return None, None, None, None, None

    def slang_register_prefix_for_type(self, type_name, node=None, forced_prefix=None):
        if forced_prefix:
            return forced_prefix
        mapped_type = self.map_resource_type_with_format(type_name, node)
        base_type = self.resource_base_type(mapped_type)
        if not isinstance(base_type, str):
            return None
        if base_type == "SamplerState":
            return "s"
        if base_type.startswith(
            (
                "RWTexture",
                "RWBuffer",
                "RWStructuredBuffer",
                "AppendStructuredBuffer",
                "ConsumeStructuredBuffer",
                "RWByteAddressBuffer",
            )
        ):
            return "u"
        if base_type.startswith(
            (
                "Sampler",
                "Texture",
                "StructuredBuffer",
                "ByteAddressBuffer",
                "RaytracingAccelerationStructure",
            )
        ):
            return "t"
        if base_type.startswith("ConstantBuffer"):
            return "b"
        return None

    def slang_resource_array_count(self, node, type_name):
        count = self.slang_resource_array_count_from_type_node(
            getattr(node, "var_type", None)
        )
        if count is None:
            count = self.slang_resource_array_count_from_type_node(
                getattr(node, "param_type", None)
            )
        if count is None:
            count = self.slang_resource_array_count_from_type_name(type_name)
        return max(count or 1, 1)

    def slang_resource_array_count_from_type_node(self, type_node):
        if type_node is None:
            return None
        if type_node.__class__.__name__ != "ArrayType":
            return None

        total = 1
        current = type_node
        saw_array = False
        while current is not None and current.__class__.__name__ == "ArrayType":
            saw_array = True
            size = evaluate_literal_int_expression(getattr(current, "size", None))
            if size is None or size <= 0:
                return None
            total *= size
            current = getattr(current, "element_type", None)
        return total if saw_array else None

    def slang_resource_array_count_from_type_name(self, type_name):
        if not isinstance(type_name, str) or "[" not in type_name:
            return None

        total = 1
        index = 0
        saw_array = False
        while True:
            start = type_name.find("[", index)
            if start < 0:
                break
            end = type_name.find("]", start + 1)
            if end < 0:
                return None
            saw_array = True
            size_text = type_name[start + 1 : end].strip()
            if not size_text or not size_text.isdigit():
                return None
            size = int(size_text)
            if size <= 0:
                return None
            total *= size
            index = end + 1
        return total if saw_array else None

    def slang_resource_register_space_key(self, descriptor_set, register_space):
        if descriptor_set is not None:
            return descriptor_set
        if register_space is not None:
            return register_space
        return 0

    def reserve_explicit_slang_resource_declarations(self, node):
        for declaration, type_name, forced_prefix in self.slang_resource_declarations(
            node
        ):
            self.reserve_explicit_slang_resource_declaration(
                declaration, type_name, forced_prefix
            )

    def slang_resource_declarations(self, node):
        declarations = []

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return

            for global_var in getattr(current, "global_variables", []) or []:
                type_name = self.variable_declaration_type(global_var)
                declarations.append((global_var, type_name, None))

            for cbuffer in getattr(current, "cbuffers", []) or []:
                declarations.append((cbuffer, "ConstantBuffer", "b"))

            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage_type, stage in stages.items():
                    stage_name = self.get_stage_name(stage_type)
                    for local_var in self.slang_stage_global_local_variables(
                        stage_name, getattr(stage, "local_variables", [])
                    ):
                        type_name = self.variable_declaration_type(local_var)
                        declarations.append((local_var, type_name, None))

        collect(node)
        return declarations

    def reserve_explicit_slang_resource_declaration(
        self, node, type_name, forced_register_prefix=None
    ):
        explicit_binding, _explicit_binding_expr = self.explicit_slang_resource_binding(
            node
        )
        explicit_set, _explicit_set_expr = self.explicit_slang_resource_set(node)
        (
            register_prefix,
            register_binding,
            _register_binding_expr,
            register_space,
            _register_space_expr,
        ) = self.explicit_slang_register(node)
        self.validate_slang_resource_binding_consistency(
            node, explicit_binding, register_binding, explicit_set, register_space
        )
        binding = explicit_binding if explicit_binding is not None else register_binding
        if binding is None:
            return

        prefix = self.slang_register_prefix_for_type(
            type_name, node, register_prefix or forced_register_prefix
        )
        if prefix is None:
            return

        space_key = self.slang_resource_register_space_key(explicit_set, register_space)
        self.reserve_slang_resource_register_range(
            prefix,
            binding,
            self.slang_resource_array_count(node, type_name),
            self.slang_resource_name(node),
            space_key,
        )

    def slang_resource_name(self, node):
        return getattr(node, "name", getattr(node, "variable_name", "<anonymous>"))

    def next_available_slang_resource_register(self, register_prefix, space, count):
        count = max(count or 1, 1)
        binding = self.slang_resource_register_cursors.get((register_prefix, space), 0)
        ranges = self.slang_used_resource_registers.get((register_prefix, space), [])
        while True:
            end = binding + count - 1
            conflict_end = None
            for used_start, used_end, _used_name in ranges:
                if binding <= used_end and used_start <= end:
                    conflict_end = (
                        used_end
                        if conflict_end is None
                        else max(conflict_end, used_end)
                    )
            if conflict_end is None:
                return binding
            binding = conflict_end + 1

    def advance_slang_resource_register(self, register_prefix, space, start, count):
        count = max(count or 1, 1)
        key = (register_prefix, space)
        self.slang_resource_register_cursors[key] = max(
            self.slang_resource_register_cursors.get(key, 0), start + count
        )

    def reserve_slang_resource_register_range(
        self, register_prefix, start, count, name, space=0
    ):
        count = max(count or 1, 1)
        end = start + count - 1
        namespace = (register_prefix, space)
        ranges = self.slang_used_resource_registers.setdefault(namespace, [])
        for used_start, used_end, used_name in ranges:
            if start <= used_end and used_start <= end:
                if used_start == start and used_end == end and used_name == name:
                    return
                raise ValueError(
                    f"Conflicting Slang resource binding for '{name}': "
                    f"{self.slang_resource_range_label(register_prefix, start, end, space)} "
                    f"overlaps '{used_name}' "
                    f"{self.slang_resource_range_label(register_prefix, used_start, used_end, space)}"
                )
        ranges.append((start, end, name))

    def slang_resource_range_label(self, register_prefix, start, end, space=0):
        if start == end:
            label = f"{register_prefix}{start}"
        else:
            label = f"{register_prefix}{start}-{register_prefix}{end}"
        if space:
            return f"{label}, space{space}"
        return label

    def validate_slang_resource_binding_consistency(
        self, node, explicit_binding, register_binding, explicit_set, register_space
    ):
        name = getattr(node, "name", getattr(node, "variable_name", "<anonymous>"))
        if (
            explicit_binding is not None
            and register_binding is not None
            and explicit_binding != register_binding
        ):
            raise ValueError(
                "Conflicting Slang resource binding metadata for "
                f"'{name}': binding {explicit_binding} does not match "
                f"register binding {register_binding}"
            )
        if (
            explicit_set is not None
            and register_space is not None
            and explicit_set != register_space
        ):
            raise ValueError(
                "Conflicting Slang resource set metadata for "
                f"'{name}': set {explicit_set} does not match "
                f"register space{register_space}"
            )

    def slang_resource_binding_decorations(
        self, node, type_name, forced_register_prefix=None, auto_assign=False
    ):
        explicit_binding, explicit_binding_expr = self.explicit_slang_resource_binding(
            node
        )
        explicit_set, explicit_set_expr = self.explicit_slang_resource_set(node)
        (
            register_prefix,
            register_binding,
            register_binding_expr,
            register_space,
            register_space_expr,
        ) = self.explicit_slang_register(node)
        self.validate_slang_resource_binding_consistency(
            node, explicit_binding, register_binding, explicit_set, register_space
        )

        if register_prefix is None:
            register_prefix = self.slang_register_prefix_for_type(
                type_name, node, forced_register_prefix
            )
        if register_prefix is None:
            return "", ""

        binding = explicit_binding if explicit_binding is not None else register_binding
        binding_expr = explicit_binding_expr or register_binding_expr
        descriptor_set = explicit_set if explicit_set is not None else register_space
        descriptor_set_expr = explicit_set_expr or register_space_expr
        space_key = self.slang_resource_register_space_key(
            descriptor_set, register_space
        )
        resource_count = self.slang_resource_array_count(node, type_name)

        if binding is None and binding_expr is None and auto_assign:
            binding = self.next_available_slang_resource_register(
                register_prefix, space_key, resource_count
            )
            binding_expr = str(binding)

        if binding_expr is None and binding is not None:
            binding_expr = str(binding)
        if descriptor_set_expr is None and descriptor_set is not None:
            descriptor_set_expr = str(descriptor_set)

        prefix = ""
        if binding_expr is not None:
            vk_set_expr = descriptor_set_expr or "0"
            prefix = f"[[vk::binding({binding_expr}, {vk_set_expr})]] "

        if register_binding is None and binding is not None:
            register_binding = binding
        register_space_suffix = ""
        if descriptor_set is not None:
            register_space_suffix = f", space{descriptor_set}"
        elif register_space_expr is not None and register_space_expr.isdigit():
            register_space_suffix = f", space{register_space_expr}"

        suffix = ""
        if register_prefix and register_binding is not None:
            suffix = (
                f" : register({register_prefix}{register_binding}"
                f"{register_space_suffix})"
            )
        if binding is not None:
            self.reserve_slang_resource_register_range(
                register_prefix,
                binding,
                resource_count,
                self.slang_resource_name(node),
                space_key,
            )
            self.advance_slang_resource_register(
                register_prefix, space_key, binding, resource_count
            )
        return prefix, suffix

    def apply_slang_resource_binding_decorations(
        self,
        declaration,
        node,
        type_name,
        forced_register_prefix=None,
        auto_assign=False,
    ):
        prefix, suffix = self.slang_resource_binding_decorations(
            node, type_name, forced_register_prefix, auto_assign
        )
        return f"{prefix}{declaration}{suffix}"

    def generate_cbuffer(self, node):
        name = getattr(node, "name", None)
        if not name or not hasattr(node, "members"):
            return ""
        declaration = self.apply_slang_resource_binding_decorations(
            f"cbuffer {name}",
            node,
            "ConstantBuffer",
            forced_register_prefix="b",
            auto_assign=True,
        )
        result = f"{declaration} {{\n"
        for member in node.members:
            if hasattr(member, "member_type"):
                member_type = self.convert_type(
                    self.convert_type_node_to_string(member.member_type)
                )
            else:
                member_type = self.convert_type(getattr(member, "vtype", "float"))
            result += f"    {self.format_declaration(member_type, member.name)};\n"
        result += "};"
        return result

    def generate_global_variable(self, node):
        if isinstance(node, ArrayNode):
            self.register_variable_type(node.name, node.element_type)
            element_type = self.convert_type(node.element_type)
            size = get_array_size_from_node(node)
            prefix = self.slang_declaration_qualifier_prefix(node)
            if size is None:
                return f"{prefix}{element_type} {node.name}[];\n"
            return f"{prefix}{element_type} {node.name}[{size}];\n"

        initial_value = getattr(node, "initial_value", getattr(node, "value", None))
        vtype = self.variable_declaration_type(node, initial_value)
        self.register_variable_type(node.name, vtype, node)
        declaration = self.format_declaration(vtype, node.name, node)
        declaration = self.slang_declaration_qualifier_prefix(node) + declaration
        declaration = self.apply_slang_resource_binding_decorations(
            declaration, node, vtype, auto_assign=True
        )
        if initial_value is not None:
            initial_expr = self.generate_expression_with_expected(
                initial_value,
                self.initializer_expected_type(vtype),
            )
            return f"{declaration} = {initial_expr};\n"
        return f"{declaration};\n"

    def generate_struct(self, node):
        result = f"struct {node.name}\n{{\n"
        self.indent_level += 1

        members = getattr(node, "members", [])
        for member in members:
            if hasattr(member, "member_type"):
                member_type = self.convert_type(
                    self.convert_type_node_to_string(member.member_type)
                )
            elif hasattr(member, "vtype"):
                member_type = self.convert_type(member.vtype)
            else:
                member_type = "float"

            semantic_str = self.semantic_suffix(self.semantic_from_node(member))
            declaration = self.format_declaration(member_type, member.name)
            result += f"{self.indent()}{declaration}{semantic_str};\n"

        self.indent_level -= 1
        result += "};"
        return result

    def generate_struct_definition(self, node):
        result = f"{node.name}\n{{\n"

        members = getattr(node, "members", [])
        for member in members:
            if hasattr(member, "member_type"):
                member_type = self.convert_type_node_to_string(member.member_type)
            else:
                member_type = getattr(member, "vtype", "float")
            result += f"    {self.format_declaration(member_type, member.name)};\n"

        result += "};"
        return result

    def generate_function(
        self,
        node,
        shader_type=None,
        execution_config=None,
        entry_name=None,
        emit_stage_decorations=True,
        stage_role=None,
        extra_parameters=None,
    ):
        """Render one CrossGL function or shader entry point as Slang code."""
        saved_variable_types = self.variable_types.copy()
        saved_image_resource_types = self.image_resource_types.copy()
        saved_image_resource_accesses = self.image_resource_accesses.copy()
        saved_buffer_resource_types = self.buffer_resource_types.copy()
        saved_buffer_resource_accesses = self.buffer_resource_accesses.copy()
        saved_function_return_type = self.current_function_return_type
        saved_shader_type = self.current_shader_type
        saved_identifier_aliases = self.identifier_aliases.copy()
        saved_hull_output_rewrite = self.current_hull_output_rewrite
        saved_expression_temp_names = self.expression_temp_names
        self.expression_temp_names = set()
        if hasattr(node, "return_type"):
            ret_type_name = self.convert_type_node_to_string(node.return_type)
            ret_type = self.convert_type(ret_type_name)
        else:
            ret_type_name = "void"
            ret_type = "void"

        semantic = self.function_return_semantic(node)
        body = getattr(node, "body", [])
        body_statements = self.get_statements(body)
        param_list = getattr(node, "parameters", getattr(node, "params", []))
        param_list = self.slang_merge_function_parameters(
            param_list, extra_parameters, shader_type
        )
        self.validate_slang_return_semantic(
            shader_type, semantic, stage_role, ret_type_name
        )
        hull_output_rewrite = None
        if stage_role != "patch_constant":
            hull_output_rewrite = self.slang_stage_hull_output_rewrite(
                body_statements, shader_type, ret_type_name, semantic, param_list
            )
        if hull_output_rewrite is not None:
            ret_type_name = hull_output_rewrite["return_type_name"]
            ret_type = self.convert_type(ret_type_name)
            semantic = None

        builtin_return_rewrite = self.slang_stage_builtin_return_rewrite(
            body_statements, shader_type, ret_type_name, semantic
        )
        if builtin_return_rewrite is not None:
            ret_type_name = builtin_return_rewrite["return_type_name"]
            ret_type = self.convert_type(ret_type_name)
            semantic = builtin_return_rewrite["semantic"]

        self.current_function_return_type = ret_type_name
        self.current_shader_type = shader_type
        semantic_str = self.semantic_suffix(semantic, shader_type)

        effective_param_list = self.slang_filtered_stage_parameters(
            param_list, hull_output_rewrite
        )
        if shader_type:
            self.validate_slang_ray_stage_parameters(
                node, shader_type, effective_param_list
            )
            self.validate_slang_ray_tracing_calls(
                node, shader_type, effective_param_list
            )
            self.validate_slang_ray_query_calls(node, shader_type, effective_param_list)
            self.validate_slang_mesh_intrinsic_calls(node, shader_type)
            self.validate_slang_mesh_payload_parameter(
                shader_type, effective_param_list
            )
            self.validate_slang_mesh_output_parameters(
                node, shader_type, effective_param_list
            )
            self.validate_slang_stage_body_builtins(
                body_statements,
                shader_type,
                effective_param_list,
                stage_role=stage_role,
            )
        else:
            self.validate_slang_ray_query_calls(node, "function", effective_param_list)
        params = []
        if effective_param_list:
            if effective_param_list and hasattr(effective_param_list[0], "name"):
                for param in effective_param_list:
                    if hasattr(param, "param_type"):
                        param_type_name = self.convert_type_node_to_string(
                            param.param_type
                        )
                        self.register_variable_type(param.name, param_type_name, param)
                        param_type = self.map_resource_type_with_format(
                            param_type_name, param
                        )
                    elif hasattr(param, "vtype"):
                        param_type_name = param.vtype
                        self.register_variable_type(param.name, param_type_name, param)
                        param_type = self.map_resource_type_with_format(
                            param_type_name, param
                        )
                    else:
                        param_type_name = "float"
                        param_type = "float"
                    declaration = format_c_style_array_declaration(
                        param_type, param.name
                    )
                    ray_declaration = self.slang_ray_stage_parameter_declaration(
                        declaration, param, shader_type
                    )
                    if ray_declaration is not None:
                        params.append(ray_declaration)
                        continue
                    mesh_payload_declaration = (
                        self.slang_mesh_payload_parameter_declaration(
                            declaration, param, shader_type
                        )
                    )
                    if mesh_payload_declaration is not None:
                        params.append(mesh_payload_declaration)
                        continue
                    mesh_output_declaration = (
                        self.slang_mesh_output_parameter_declaration(
                            declaration, param, shader_type
                        )
                    )
                    if mesh_output_declaration is not None:
                        params.append(mesh_output_declaration)
                        continue
                    declaration = (
                        self.slang_parameter_qualifier_prefix(param) + declaration
                    )
                    declaration = self.apply_slang_resource_binding_decorations(
                        declaration, param, param_type_name
                    )
                    params.append(
                        declaration
                        + self.semantic_suffix(
                            self.semantic_from_node(param), shader_type
                        )
                    )
            else:
                for param_type, param_name in effective_param_list:
                    self.register_variable_type(param_name, param_type)
                    params.append(
                        f"{self.map_resource_type_with_format(param_type)} {param_name}"
                    )

        for (
            param_type,
            param_name,
            semantic,
        ) in self.slang_implicit_stage_parameters(
            body_statements, shader_type, effective_param_list, stage_role=stage_role
        ):
            self.register_variable_type(param_name, param_type)
            params.append(f"{self.convert_type(param_type)} {param_name} : {semantic}")

        params_str = ", ".join(params)
        identifier_aliases = self.slang_stage_system_value_aliases(
            body_statements,
            shader_type,
            effective_param_list,
            stage_role=stage_role,
        )
        identifier_aliases.update(
            self.slang_stage_intrinsic_builtin_aliases(
                body_statements, shader_type, effective_param_list
            )
        )
        identifier_aliases.update(
            self.slang_stage_patch_input_aliases(
                body_statements, shader_type, effective_param_list
            )
        )
        identifier_aliases.update(
            self.slang_stage_builtin_return_aliases(builtin_return_rewrite)
        )
        self.identifier_aliases = identifier_aliases
        self.current_hull_output_rewrite = hull_output_rewrite

        result = ""
        if (
            builtin_return_rewrite is not None
            and builtin_return_rewrite["mode"] == "struct"
        ):
            result += self.generate_slang_builtin_output_struct(builtin_return_rewrite)
            result += "\n\n"
        if shader_type and emit_stage_decorations:
            self.validate_slang_stage_attributes(node, shader_type)
            result += self.generate_slang_stage_numthreads(
                node, shader_type, execution_config
            )
            result += self.generate_slang_stage_attributes(node, shader_type)
            shader_stage = self.slang_shader_stage_name(shader_type)
            result += f'[shader("{shader_stage}")]\n'
        function_name = entry_name or node.name
        result += f"{ret_type} {function_name}({params_str}){semantic_str}\n{{\n"
        self.indent_level += 1

        if builtin_return_rewrite is not None and builtin_return_rewrite["mode"] in {
            "local",
            "struct",
        }:
            local_type = self.convert_type(builtin_return_rewrite["return_type_name"])
            local_name = builtin_return_rewrite["local_name"]
            result += f"{self.indent()}{local_type} {local_name};\n"
        if hull_output_rewrite is not None:
            local_type = self.convert_type(hull_output_rewrite["return_type_name"])
            local_name = hull_output_rewrite["local_name"]
            result += f"{self.indent()}{local_type} {local_name};\n"

        for statement_index, stmt in enumerate(body_statements):
            result += (
                self.emit_function_body_statement(
                    stmt, statement_index, builtin_return_rewrite
                )
                + "\n"
            )

        if builtin_return_rewrite is not None and builtin_return_rewrite["mode"] in {
            "local",
            "struct",
        }:
            result += f"{self.indent()}return {builtin_return_rewrite['local_name']};\n"
        if hull_output_rewrite is not None:
            result += f"{self.indent()}return {hull_output_rewrite['local_name']};\n"

        self.indent_level -= 1
        result += "}"
        self.variable_types = saved_variable_types
        self.image_resource_types = saved_image_resource_types
        self.image_resource_accesses = saved_image_resource_accesses
        self.buffer_resource_types = saved_buffer_resource_types
        self.buffer_resource_accesses = saved_buffer_resource_accesses
        self.current_function_return_type = saved_function_return_type
        self.current_shader_type = saved_shader_type
        self.identifier_aliases = saved_identifier_aliases
        self.current_hull_output_rewrite = saved_hull_output_rewrite
        self.expression_temp_names = saved_expression_temp_names
        return result

    def generate_compute_numthreads(self, execution_config=None):
        x, y, z = compute_local_size(execution_config)
        return f"[numthreads({x}, {y}, {z})]\n"

    def validate_slang_return_semantic(
        self, shader_type, semantic, stage_role, ret_type_name
    ):
        if semantic is None:
            return

        shader_stage = self.slang_shader_stage_name(shader_type)
        if shader_stage == "hull":
            if stage_role == "patch_constant":
                raise ValueError(
                    "Slang patch constant function returns must put semantics on "
                    "the returned struct members"
                )

            raise ValueError(
                "Slang tessellation_control stage returns must put semantics on "
                "the output control-point struct members"
            )

        if self.convert_type(ret_type_name) == "void":
            if shader_type:
                raise ValueError(
                    f"Slang {shader_type} stage void return cannot use return "
                    f"semantic {semantic}"
                )
            raise ValueError(
                f"Slang void function return cannot use return semantic {semantic}"
            )

    def slang_stage_hull_output_rewrite(
        self, body_statements, shader_type, ret_type_name, semantic, param_list
    ):
        if self.slang_shader_stage_name(shader_type) != "hull":
            return None

        gl_out_assignments = []
        for node in self.walk_ast(body_statements):
            if not isinstance(node, AssignmentNode):
                continue
            index = self.slang_hull_output_access_index(node.left)
            if index is not None:
                gl_out_assignments.append((node, index))

        if not gl_out_assignments:
            if self.slang_stage_uses_gl_out(body_statements):
                raise ValueError(
                    "Slang hull stage gl_out requires assignments to "
                    "gl_out[gl_InvocationID] or gl_out[gl_InvocationID].field"
                )
            return None

        if semantic is not None:
            raise ValueError(
                "Slang hull stage gl_out outputs cannot use a return semantic"
            )
        if self.contains_return_statement(body_statements):
            raise ValueError(
                "Slang hull stage gl_out outputs cannot be mixed with explicit returns"
            )

        for assignment, index in gl_out_assignments:
            if getattr(assignment, "operator", None) != "=":
                raise ValueError(
                    "Slang hull stage gl_out outputs require simple assignment"
                )
            if not self.is_slang_hull_output_index(index, param_list):
                raise ValueError(
                    "Slang hull stage gl_out writes must target "
                    "gl_out[gl_InvocationID] or the SV_OutputControlPointID parameter"
                )

        for index in self.slang_hull_output_access_indices(body_statements):
            if not self.is_slang_hull_output_index(index, param_list):
                raise ValueError(
                    "Slang hull stage gl_out accesses must target "
                    "gl_out[gl_InvocationID] or the SV_OutputControlPointID parameter"
                )

        indexed_identifier_ids = self.slang_hull_output_indexed_identifier_ids(
            body_statements
        )
        for node in self.walk_ast(body_statements):
            if (
                isinstance(node, IdentifierNode)
                and node.name == "gl_out"
                and id(node) not in indexed_identifier_ids
            ):
                raise ValueError(
                    "Slang hull stage gl_out must be indexed as "
                    "gl_out[gl_InvocationID]"
                )

        output_type_name = None
        removed_param_names = set()
        if self.convert_type(ret_type_name) != "void":
            output_type_name = ret_type_name
        else:
            output_patch_params = []
            for param in param_list or []:
                element_type = self.slang_patch_parameter_element_type_name(
                    param, "OutputPatch"
                )
                if element_type:
                    output_patch_params.append((param, element_type))

            if len(output_patch_params) == 1:
                output_patch_param, output_type_name = output_patch_params[0]
                removed_param_names.add(output_patch_param.name)
            else:
                raise ValueError(
                    "Slang hull stage gl_out requires a non-void return type or "
                    "exactly one explicit OutputPatch<..., N> parameter"
                )

        return {
            "return_type_name": output_type_name,
            "local_name": self.unique_slang_builtin_output_local_name(
                "gl_OutputControlPoint", body_statements
            ),
            "removed_param_names": removed_param_names,
        }

    def slang_filtered_stage_parameters(self, param_list, hull_output_rewrite):
        if hull_output_rewrite is None:
            return param_list

        removed_param_names = hull_output_rewrite["removed_param_names"]
        if not removed_param_names:
            return param_list
        if not param_list:
            return param_list

        if hasattr(param_list[0], "name"):
            return [
                param
                for param in param_list
                if getattr(param, "name", None) not in removed_param_names
            ]
        return [
            param
            for param in param_list
            if len(param) < 2 or param[1] not in removed_param_names
        ]

    def slang_stage_uses_gl_out(self, body_statements):
        return any(
            isinstance(node, IdentifierNode) and node.name == "gl_out"
            for node in self.walk_ast(body_statements)
        )

    def slang_hull_output_access_index(self, target):
        if isinstance(target, MemberAccessNode):
            return self.slang_hull_output_access_index(
                getattr(target, "object", getattr(target, "object_expr", None))
            )
        if isinstance(target, ArrayAccessNode):
            array = getattr(target, "array", getattr(target, "array_expr", None))
            if self.identifier_name(array) == "gl_out":
                return getattr(target, "index", getattr(target, "index_expr", None))
            return self.slang_hull_output_access_index(array)
        return None

    def slang_hull_output_access_indices(self, body_statements):
        indices = []
        for node in self.walk_ast(body_statements):
            if not isinstance(node, ArrayAccessNode):
                continue
            array = getattr(node, "array", getattr(node, "array_expr", None))
            if self.identifier_name(array) == "gl_out":
                indices.append(
                    getattr(node, "index", getattr(node, "index_expr", None))
                )
        return indices

    def slang_hull_output_indexed_identifier_ids(self, body_statements):
        identifier_ids = set()
        for node in self.walk_ast(body_statements):
            if not isinstance(node, ArrayAccessNode):
                continue
            array = getattr(node, "array", getattr(node, "array_expr", None))
            if isinstance(array, IdentifierNode) and array.name == "gl_out":
                identifier_ids.add(id(array))
        return identifier_ids

    def slang_hull_output_member_target(self, target):
        if not isinstance(target, MemberAccessNode):
            return None

        obj = getattr(target, "object", getattr(target, "object_expr", None))
        if not isinstance(obj, ArrayAccessNode):
            return None

        array = getattr(obj, "array", getattr(obj, "array_expr", None))
        if self.identifier_name(array) != "gl_out":
            return None

        index = getattr(obj, "index", getattr(obj, "index_expr", None))
        return target.member, index

    def is_slang_hull_output_index(self, index, param_list):
        index_name = self.identifier_name(index)
        if index_name == "gl_InvocationID":
            return True

        for param in param_list or []:
            param_name = getattr(param, "name", None)
            if param_name != index_name:
                continue
            semantic = self.semantic_from_node(param)
            if self.map_semantic(semantic, "tessellation_control") == (
                "SV_OutputControlPointID"
            ):
                return True
        return False

    def slang_patch_parameter_element_type_name(self, param, patch_type):
        shape = self.slang_patch_parameter_shape(param, patch_type)
        if shape is not None:
            return shape[0]
        return None

    def slang_patch_parameters(self, param_list, patch_type):
        patch_params = []
        for param in param_list or []:
            shape = self.slang_patch_parameter_shape(param, patch_type)
            if shape is not None:
                patch_params.append((param, shape))
        return patch_params

    def slang_patch_parameter_shape(self, param, patch_type):
        type_node = getattr(param, "param_type", None)
        if getattr(type_node, "name", None) == patch_type:
            generic_args = getattr(type_node, "generic_args", []) or []
            if generic_args:
                element_type = self.convert_type_node_to_string(generic_args[0])
                patch_size = None
                if len(generic_args) > 1:
                    patch_size = self.convert_type_node_to_string(generic_args[1])
                return element_type, patch_size

        type_name = self.slang_parameter_type_name(param)
        prefix = f"{patch_type}<"
        if not type_name.startswith(prefix) or not type_name.endswith(">"):
            return None

        args = type_name[len(prefix) : -1]
        generic_args = self.slang_generic_arguments(args)
        if not generic_args:
            return None
        element_type = generic_args[0]
        patch_size = generic_args[1] if len(generic_args) > 1 else None
        return element_type, patch_size

    def first_slang_generic_argument(self, args):
        generic_args = self.slang_generic_arguments(args)
        if generic_args:
            return generic_args[0]
        return args.strip()

    def slang_generic_arguments(self, args):
        arguments = []
        start = 0
        depth = 0
        for index, char in enumerate(args):
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            elif char == "," and depth == 0:
                arguments.append(args[start:index].strip())
                start = index + 1
        trailing_arg = args[start:].strip()
        if trailing_arg:
            arguments.append(trailing_arg)
        return arguments

    def slang_implicit_stage_parameters(
        self, body_statements, shader_type, param_list, stage_role=None
    ):
        candidates = self.slang_implicit_stage_parameter_candidates(
            shader_type, stage_role=stage_role
        )
        if not candidates:
            return []

        declared_names = {
            getattr(param, "name", None)
            for param in param_list or []
            if getattr(param, "name", None)
        }
        existing_semantics = {
            self.map_semantic(self.semantic_from_node(param), shader_type)
            for param in param_list or []
            if self.semantic_from_node(param)
        }
        for node in self.walk_ast(body_statements):
            if isinstance(node, VariableNode):
                declared_names.add(node.name)

        used_names = set()
        for node in self.walk_ast(body_statements):
            if isinstance(node, IdentifierNode) and node.name in candidates:
                used_names.add(node.name)

        implicit_params = []
        for name, (param_type, semantic) in candidates.items():
            mapped_semantic = self.map_semantic(semantic)
            if (
                name in used_names
                and name not in declared_names
                and mapped_semantic not in existing_semantics
            ):
                implicit_params.append((param_type, name, mapped_semantic))
        return implicit_params

    def slang_stage_system_value_aliases(
        self, body_statements, shader_type, param_list, stage_role=None
    ):
        candidates = self.slang_implicit_stage_parameter_candidates(
            shader_type, stage_role=stage_role
        )
        if not candidates:
            return {}

        declared_names = {
            getattr(param, "name", None)
            for param in param_list or []
            if getattr(param, "name", None)
        }
        semantic_parameters = {}
        for param in param_list or []:
            param_name = getattr(param, "name", None)
            semantic = self.semantic_from_node(param)
            if param_name and semantic:
                semantic_parameters[self.map_semantic(semantic, shader_type)] = (
                    param_name
                )

        for node in self.walk_ast(body_statements):
            if isinstance(node, VariableNode):
                declared_names.add(node.name)

        aliases = {}
        for node in self.walk_ast(body_statements):
            if not isinstance(node, IdentifierNode) or node.name not in candidates:
                continue
            if node.name in declared_names:
                continue

            _param_type, semantic = candidates[node.name]
            existing_param_name = semantic_parameters.get(
                self.map_semantic(semantic, shader_type)
            )
            if existing_param_name and existing_param_name != node.name:
                aliases[node.name] = existing_param_name
        return aliases

    def slang_stage_intrinsic_builtin_aliases(
        self, body_statements, shader_type, param_list
    ):
        candidates = self.slang_intrinsic_builtin_candidates(shader_type)
        if not candidates:
            return {}

        declared_names = {
            getattr(param, "name", None)
            for param in param_list or []
            if getattr(param, "name", None)
        }
        for node in self.walk_ast(body_statements):
            if isinstance(node, VariableNode):
                declared_names.add(node.name)

        aliases = {}
        for node in self.walk_ast(body_statements):
            if not isinstance(node, IdentifierNode):
                continue
            if node.name not in candidates or node.name in declared_names:
                continue

            return_type, intrinsic = candidates[node.name]
            aliases[node.name] = intrinsic
            self.register_variable_type(node.name, return_type)
        return aliases

    def slang_stage_patch_input_aliases(self, body_statements, shader_type, param_list):
        shader_stage = self.slang_shader_stage_name(shader_type)
        patch_type = {
            "hull": "InputPatch",
            "domain": "OutputPatch",
        }.get(shader_stage)
        if patch_type is None:
            return {}

        declared_names = {
            getattr(param, "name", None)
            for param in param_list or []
            if getattr(param, "name", None)
        }
        for node in self.walk_ast(body_statements):
            if isinstance(node, VariableNode):
                declared_names.add(node.name)

        uses_gl_in = any(
            isinstance(node, IdentifierNode)
            and node.name == "gl_in"
            and node.name not in declared_names
            for node in self.walk_ast(body_statements)
        )
        if not uses_gl_in:
            return {}

        patch_params = [
            param
            for param in param_list or []
            if self.slang_parameter_type_name(param).startswith(f"{patch_type}<")
        ]
        if len(patch_params) == 1:
            patch_param = patch_params[0]
            self.register_variable_type(
                "gl_in", self.slang_parameter_type_name(patch_param), patch_param
            )
            return {"gl_in": patch_param.name}

        raise ValueError(
            f"Slang {shader_stage} stage gl_in requires exactly one explicit "
            f"{patch_type}<..., N> parameter"
        )

    def slang_parameter_type_name(self, param):
        if hasattr(param, "param_type"):
            return self.convert_type_node_to_string(param.param_type)
        if hasattr(param, "vtype"):
            return str(param.vtype)
        return ""

    def slang_ray_stage_types(self):
        return {
            "raygeneration",
            "intersection",
            "closesthit",
            "anyhit",
            "miss",
            "callable",
        }

    def slang_ray_role_from_name(self, name, shader_stage=None):
        if not name:
            return None

        normalized = str(name).lower()
        compact = normalized.replace("_", "")
        if compact in {"payload", "raypayloadext", "raypayloadinext"}:
            if shader_stage == "callable":
                return "callable_data"
            return "payload"
        if compact in {"hitattribute", "hitattributeext"}:
            return "hit_attribute"
        if compact in {"callabledata", "callabledataext", "callabledatainext"}:
            return "callable_data"
        return None

    def slang_ray_attribute_role_name(self, attr, shader_stage=None):
        return self.slang_ray_role_from_name(getattr(attr, "name", None), shader_stage)

    def slang_ray_semantic_role(self, parameter, shader_type=None):
        shader_stage = self.slang_shader_stage_name(shader_type)
        semantic = getattr(parameter, "semantic", None)
        if semantic:
            role = self.slang_ray_role_from_name(semantic, shader_stage)
            if role:
                return role

        for attr in getattr(parameter, "attributes", []) or []:
            role = self.slang_ray_attribute_role_name(attr, shader_stage)
            if role:
                return role
        return None

    def slang_ray_role_parameters(self, parameters, shader_type):
        role_parameters = {}
        for parameter in parameters or []:
            role = self.slang_ray_semantic_role(parameter, shader_type)
            if role:
                role_parameters.setdefault(role, []).append(parameter)
        return role_parameters

    def is_slang_stage_local_interface_parameter(self, parameter):
        if hasattr(parameter, "get_annotation"):
            return bool(parameter.get_annotation("slang_stage_local_interface"))
        return bool(
            getattr(parameter, "annotations", {}).get("slang_stage_local_interface")
        )

    def validate_slang_ray_parameter_type(self, parameter, role):
        type_name = self.slang_parameter_type_name(parameter)
        if not type_name:
            return

        mapped_type = self.convert_type(type_name)
        base_type, array_suffix = split_array_type_suffix(mapped_type)
        allowed_builtin_types = {
            "hit_attribute": {"BuiltInTriangleIntersectionAttributes"},
        }.get(role, set())
        if role == "hit_attribute" and self.is_slang_stage_local_interface_parameter(
            parameter
        ):
            allowed_builtin_types.update({"float2", "float3", "float4"})
        if array_suffix or (
            base_type not in self.user_struct_names
            and base_type not in allowed_builtin_types
        ):
            raise ValueError(
                f"Slang ray {role} parameter '{parameter.name}' must use a "
                "user-defined struct type"
            )

    def validate_slang_ray_stage_parameters(self, func, shader_type, parameters):
        shader_stage = self.slang_shader_stage_name(shader_type)
        if shader_stage not in self.slang_ray_stage_types():
            return

        role_parameters = self.slang_ray_role_parameters(parameters, shader_type)
        allowed_stages = {
            "payload": {"closesthit", "anyhit", "miss"},
            "hit_attribute": {"closesthit", "anyhit"},
            "callable_data": {"callable"},
        }
        for role, role_params in role_parameters.items():
            if len(role_params) > 1:
                raise ValueError(
                    f"Slang {shader_type} stage must declare at most one "
                    f"{role} parameter"
                )
            if shader_stage not in allowed_stages.get(role, set()):
                if not (
                    shader_stage == "raygeneration"
                    and role in {"payload", "callable_data"}
                    and self.is_slang_stage_local_interface_parameter(role_params[0])
                ):
                    raise ValueError(
                        f"Slang {shader_type} stage cannot use {role} parameter "
                        f"'{role_params[0].name}'"
                    )
            self.validate_slang_ray_parameter_type(role_params[0], role)

    def slang_function_scope_variable_types(self, func):
        variables = {}
        for node in self.walk_ast(getattr(func, "body", [])):
            if not isinstance(node, VariableNode):
                continue
            type_name = self.get_variable_type(node)
            if type_name is not None:
                variables[node.name] = type_name
        return variables

    def slang_ray_tracing_calls(self, func):
        calls = []
        ray_intrinsics = {
            "TraceRay",
            "CallShader",
            "ReportHit",
            "AcceptHitAndEndSearch",
            "IgnoreHit",
        }
        for node in self.walk_ast(getattr(func, "body", [])):
            if isinstance(node, RayTracingOpNode):
                calls.append(
                    (getattr(node, "operation", None), getattr(node, "arguments", []))
                )
                continue

            if not isinstance(node, FunctionCallNode):
                continue
            func_expr = getattr(node, "function", None) or getattr(node, "name", None)
            func_name = getattr(func_expr, "name", func_expr)
            if func_name in ray_intrinsics:
                calls.append(
                    (func_name, getattr(node, "arguments", getattr(node, "args", [])))
                )
        return calls

    def slang_function_call_name(self, node):
        if not isinstance(node, FunctionCallNode):
            return None
        func_expr = getattr(node, "function", None) or getattr(node, "name", None)
        if hasattr(func_expr, "name"):
            return func_expr.name
        if isinstance(func_expr, str):
            return func_expr
        return None

    def slang_user_function_calls(self, func):
        calls = []
        for node in self.walk_ast(getattr(func, "body", [])):
            call_name = self.slang_function_call_name(node)
            if call_name in self.user_functions_by_name:
                calls.append(call_name)
        return calls

    def slang_user_function_call_nodes(self, func):
        calls = []
        for node in self.walk_ast(getattr(func, "body", [])):
            call_name = self.slang_function_call_name(node)
            if call_name in self.user_functions_by_name:
                calls.append(node)
        return calls

    def slang_expression_root_identifier(self, expr):
        if isinstance(expr, IdentifierNode):
            return expr.name
        if isinstance(expr, VariableNode):
            return expr.name
        if isinstance(expr, ArrayAccessNode):
            array = getattr(expr, "array", getattr(expr, "array_expr", None))
            return self.slang_expression_root_identifier(array)
        if isinstance(expr, MemberAccessNode):
            obj = getattr(expr, "object", getattr(expr, "object_expr", None))
            return self.slang_expression_root_identifier(obj)
        if isinstance(expr, PointerAccessNode):
            pointer = getattr(expr, "pointer_expr", None)
            return self.slang_expression_root_identifier(pointer)
        return None

    def slang_resolve_alias_root(self, root_name, aliases):
        seen = set()
        while root_name in aliases and root_name not in seen:
            seen.add(root_name)
            root_name = aliases[root_name]
        return root_name

    def slang_parameter_alias_roots(self, func, parameter_names):
        aliases = {}

        def alias_root(expr):
            root_name = self.slang_expression_root_identifier(expr)
            return self.slang_resolve_alias_root(root_name, aliases)

        for node in self.walk_ast(getattr(func, "body", [])):
            if isinstance(node, VariableNode):
                initial_value = getattr(
                    node, "initial_value", getattr(node, "value", None)
                )
                source_root = alias_root(initial_value)
                if source_root in parameter_names:
                    aliases[node.name] = source_root
                continue

            if isinstance(node, AssignmentNode):
                target = getattr(node, "left", getattr(node, "target", None))
                target_root = self.slang_expression_root_identifier(target)
                if target_root in parameter_names:
                    continue
                source_root = alias_root(
                    getattr(node, "right", getattr(node, "value", None))
                )
                if target_root and source_root in parameter_names:
                    aliases[target_root] = source_root

        return aliases

    def slang_ray_interface_argument_role(self, operation, args):
        if operation == "TraceRay" and args:
            return args[-1], "payload"
        if operation == "CallShader" and len(args) >= 2:
            return args[1], "callable data"
        if operation == "ReportHit" and len(args) >= 3:
            return args[2], "hit attribute"
        return None, None

    def slang_ray_interface_parameter_roles(self, func, visited_helpers=None):
        if visited_helpers is None:
            visited_helpers = set()

        func_name = getattr(func, "name", None)
        if func_name in visited_helpers:
            return {}
        if func_name:
            visited_helpers = visited_helpers | {func_name}

        parameters = getattr(func, "parameters", getattr(func, "params", []))
        parameter_names = {
            getattr(parameter, "name", None) for parameter in parameters or []
        }
        parameter_names.discard(None)
        roles = {}
        alias_roots = self.slang_parameter_alias_roots(func, parameter_names)

        for operation, args in self.slang_ray_tracing_calls(func):
            argument, role = self.slang_ray_interface_argument_role(operation, args)
            if argument is None:
                continue
            root_name = self.slang_expression_root_identifier(argument)
            root_name = self.slang_resolve_alias_root(root_name, alias_roots)
            if root_name in parameter_names and root_name not in roles:
                roles[root_name] = role

        for call_node in self.slang_user_function_call_nodes(func):
            helper_name = self.slang_function_call_name(call_node)
            if helper_name in visited_helpers:
                continue
            helper_func = self.user_functions_by_name.get(helper_name)
            if helper_func is None:
                continue
            helper_roles = self.slang_ray_interface_parameter_roles(
                helper_func, visited_helpers
            )
            if not helper_roles:
                continue
            helper_params = getattr(
                helper_func, "parameters", getattr(helper_func, "params", [])
            )
            args = getattr(call_node, "arguments", getattr(call_node, "args", []))
            for index, helper_param in enumerate(helper_params or []):
                helper_param_name = getattr(helper_param, "name", None)
                if helper_param_name not in helper_roles or index >= len(args):
                    continue
                root_name = self.slang_expression_root_identifier(args[index])
                root_name = self.slang_resolve_alias_root(root_name, alias_roots)
                if root_name in parameter_names and root_name not in roles:
                    roles[root_name] = helper_roles[helper_param_name]

        return roles

    def slang_expression_mapped_base_and_array_suffix(self, expr):
        expr_type = self.expression_result_type(expr)
        if expr_type is None:
            return None, ""
        expr_type = self.reference_referent_type_name(expr_type) or expr_type
        mapped_type = self.convert_type(expr_type)
        return split_array_type_suffix(mapped_type)

    def slang_expression_is_lvalue(self, expr):
        if isinstance(expr, (VariableNode, IdentifierNode)):
            return True
        if isinstance(expr, UnaryOpNode):
            if getattr(expr, "op", None) == "*":
                return (
                    self.pointer_pointee_type_name(
                        self.expression_result_type(getattr(expr, "operand", None))
                    )
                    is not None
                )
            return False
        if isinstance(expr, ArrayAccessNode):
            return self.slang_expression_is_lvalue(
                getattr(expr, "array", getattr(expr, "array_expr", None))
            )
        if isinstance(expr, MemberAccessNode):
            return self.slang_expression_is_lvalue(
                getattr(expr, "object", getattr(expr, "object_expr", None))
            )
        if isinstance(expr, PointerAccessNode):
            return self.slang_expression_is_lvalue(getattr(expr, "pointer_expr", None))
        return False

    def validate_slang_ray_lvalue_argument(
        self, argument, shader_type, operation, role
    ):
        base_type, array_suffix = self.slang_expression_mapped_base_and_array_suffix(
            argument
        )
        if base_type is None:
            return
        if self.slang_expression_is_lvalue(argument):
            return
        actual_type = f"{base_type}{array_suffix}"
        raise ValueError(
            f"Slang {shader_type} {operation} {role} argument must be "
            f"an lvalue, got {actual_type}"
        )

    def validate_slang_ray_address_of_lvalue_argument(
        self, argument, shader_type, operation, role
    ):
        if not (
            isinstance(argument, UnaryOpNode)
            and getattr(argument, "op", None) == "&"
            and not getattr(argument, "is_postfix", False)
        ):
            return

        operand = getattr(argument, "operand", None)
        base_type, array_suffix = self.slang_expression_mapped_base_and_array_suffix(
            operand
        )
        if base_type is None:
            return
        if self.slang_expression_is_lvalue(operand):
            return
        actual_type = f"{base_type}{array_suffix}"
        raise ValueError(
            f"Slang {shader_type} {operation} {role} argument must take the "
            f"address of an lvalue, got {actual_type}"
        )

    def slang_parameter_requires_lvalue_argument(self, parameter):
        directions = self.slang_parameter_direction_qualifiers(parameter)
        if directions & {"out", "inout"}:
            return True

        param_type = self.slang_parameter_type_name(parameter)
        return self.reference_referent_type_name(param_type) is not None

    def validate_slang_ray_exact_type_argument(
        self, argument, shader_type, operation, role, expected_type
    ):
        base_type, array_suffix = self.slang_expression_mapped_base_and_array_suffix(
            argument
        )
        if base_type is None:
            return
        if array_suffix or base_type != expected_type:
            actual_type = f"{base_type}{array_suffix}"
            raise ValueError(
                f"Slang {shader_type} {operation} {role} argument must be "
                f"{expected_type}, got {actual_type}"
            )

    def validate_slang_ray_scalar_int_uint_argument(
        self, argument, shader_type, operation, role
    ):
        base_type, array_suffix = self.slang_expression_mapped_base_and_array_suffix(
            argument
        )
        if base_type is None:
            return
        if array_suffix or base_type not in {"int", "uint"}:
            actual_type = f"{base_type}{array_suffix}"
            raise ValueError(
                f"Slang {shader_type} {operation} {role} argument must be "
                f"scalar int or uint, got {actual_type}"
            )

    def validate_slang_ray_scalar_float_argument(
        self, argument, shader_type, operation, role
    ):
        base_type, array_suffix = self.slang_expression_mapped_base_and_array_suffix(
            argument
        )
        if base_type is None:
            return
        if array_suffix or base_type not in {"float", "double"}:
            actual_type = f"{base_type}{array_suffix}"
            raise ValueError(
                f"Slang {shader_type} {operation} {role} argument must be "
                f"scalar floating, got {actual_type}"
            )

    def validate_slang_ray_struct_argument(
        self, argument, shader_type, operation, role
    ):
        base_type, array_suffix = self.slang_expression_mapped_base_and_array_suffix(
            argument
        )
        if base_type is None:
            return
        allowed_builtin_types = {"BuiltInTriangleIntersectionAttributes"}
        if array_suffix or (
            base_type not in self.user_struct_names
            and base_type not in allowed_builtin_types
        ):
            actual_type = f"{base_type}{array_suffix}"
            raise ValueError(
                f"Slang {shader_type} {operation} {role} argument must use a "
                f"user-defined struct type, got {actual_type}"
            )

    def validate_slang_call_shader_callable_data_type(self, argument, shader_type):
        if not self.slang_callable_data_parameter_types:
            return

        base_type, array_suffix = self.slang_expression_mapped_base_and_array_suffix(
            argument
        )
        if base_type is None or array_suffix:
            return
        if base_type in self.slang_callable_data_parameter_types:
            return

        expected_label = " or ".join(sorted(self.slang_callable_data_parameter_types))
        raise ValueError(
            f"Slang {shader_type} CallShader callable data argument type "
            f"{base_type} must match callable data parameter type {expected_label}"
        )

    def validate_slang_trace_ray_payload_type(self, argument, shader_type):
        if not self.slang_ray_payload_parameter_types:
            return

        base_type, array_suffix = self.slang_expression_mapped_base_and_array_suffix(
            argument
        )
        if base_type is None or array_suffix:
            return
        if base_type in self.slang_ray_payload_parameter_types:
            return

        expected_label = " or ".join(sorted(self.slang_ray_payload_parameter_types))
        raise ValueError(
            f"Slang {shader_type} TraceRay payload argument type {base_type} "
            f"must match ray payload parameter type {expected_label}"
        )

    def validate_slang_report_hit_attribute_type(self, argument, shader_type):
        if not self.slang_hit_attribute_parameter_types:
            return

        base_type, array_suffix = self.slang_expression_mapped_base_and_array_suffix(
            argument
        )
        if base_type is None or array_suffix:
            return
        if base_type in self.slang_hit_attribute_parameter_types:
            return

        expected_label = " or ".join(sorted(self.slang_hit_attribute_parameter_types))
        raise ValueError(
            f"Slang {shader_type} ReportHit hit attribute argument type {base_type} "
            f"must match hit attribute parameter type {expected_label}"
        )

    def validate_slang_trace_ray_arguments(self, args, shader_type):
        self.validate_slang_ray_exact_type_argument(
            args[0],
            shader_type,
            "TraceRay",
            "acceleration structure",
            "RaytracingAccelerationStructure",
        )
        for index, role in (
            (1, "ray flags"),
            (2, "instance inclusion mask"),
            (3, "ray contribution to hit group index"),
            (4, "geometry contribution multiplier"),
            (5, "miss shader index"),
        ):
            self.validate_slang_ray_scalar_int_uint_argument(
                args[index], shader_type, "TraceRay", role
            )
        if len(args) == 8:
            self.validate_slang_ray_exact_type_argument(
                args[6], shader_type, "TraceRay", "ray descriptor", "RayDesc"
            )
        else:
            self.validate_slang_ray_exact_type_argument(
                args[6], shader_type, "TraceRay", "origin", "float3"
            )
            self.validate_slang_ray_scalar_float_argument(
                args[7], shader_type, "TraceRay", "minimum distance"
            )
            self.validate_slang_ray_exact_type_argument(
                args[8], shader_type, "TraceRay", "direction", "float3"
            )
            self.validate_slang_ray_scalar_float_argument(
                args[9], shader_type, "TraceRay", "maximum distance"
            )
        self.validate_slang_ray_struct_argument(
            args[-1], shader_type, "TraceRay", "payload"
        )
        self.validate_slang_trace_ray_payload_type(args[-1], shader_type)
        self.validate_slang_ray_lvalue_argument(
            args[-1], shader_type, "TraceRay", "payload"
        )

    def validate_slang_ray_tracing_call_arguments(self, operation, args, shader_type):
        if operation == "TraceRay":
            self.validate_slang_trace_ray_arguments(args, shader_type)
        elif operation == "CallShader":
            self.validate_slang_ray_scalar_int_uint_argument(
                args[0], shader_type, "CallShader", "shader index"
            )
            self.validate_slang_ray_struct_argument(
                args[1], shader_type, "CallShader", "callable data"
            )
            self.validate_slang_call_shader_callable_data_type(args[1], shader_type)
            self.validate_slang_ray_lvalue_argument(
                args[1], shader_type, "CallShader", "callable data"
            )
        elif operation == "ReportHit":
            self.validate_slang_ray_scalar_float_argument(
                args[0], shader_type, "ReportHit", "hit distance"
            )
            self.validate_slang_ray_scalar_int_uint_argument(
                args[1], shader_type, "ReportHit", "hit kind"
            )
            if len(args) == 3:
                self.validate_slang_ray_struct_argument(
                    args[2], shader_type, "ReportHit", "hit attribute"
                )
                self.validate_slang_report_hit_attribute_type(args[2], shader_type)

    def validate_slang_ray_helper_call_arguments(self, call_node, shader_type):
        helper_name = self.slang_function_call_name(call_node)
        helper_func = self.user_functions_by_name.get(helper_name)
        if helper_func is None:
            return

        roles = self.slang_ray_interface_parameter_roles(helper_func)
        if not roles:
            return

        parameters = getattr(
            helper_func, "parameters", getattr(helper_func, "params", [])
        )
        args = getattr(call_node, "arguments", getattr(call_node, "args", []))
        for index, parameter in enumerate(parameters or []):
            param_name = getattr(parameter, "name", None)
            if param_name not in roles or index >= len(args):
                continue
            role = roles[param_name]
            expected_type = self.slang_parameter_type_name(parameter)
            if not expected_type:
                continue
            expected_compare_type = (
                self.reference_referent_type_name(expected_type) or expected_type
            )
            expected_base, expected_suffix = split_array_type_suffix(
                self.convert_type(expected_compare_type)
            )
            actual_base, actual_suffix = (
                self.slang_expression_mapped_base_and_array_suffix(args[index])
            )
            if actual_base is None:
                continue
            if actual_base == expected_base and actual_suffix == expected_suffix:
                self.validate_slang_ray_address_of_lvalue_argument(
                    args[index], shader_type, helper_name, role
                )
                if role in {
                    "payload",
                    "callable data",
                } and self.slang_parameter_requires_lvalue_argument(parameter):
                    self.validate_slang_ray_lvalue_argument(
                        args[index], shader_type, helper_name, role
                    )
                continue
            actual_type = f"{actual_base}{actual_suffix}"
            expected_label = self.convert_type(expected_type)
            raise ValueError(
                f"Slang {shader_type} {helper_name} {role} "
                f"argument type {actual_type} must match parameter type "
                f"{expected_label}"
            )

    def validate_slang_ray_tracing_calls(
        self, func, shader_type, parameters, visited_helpers=None
    ):
        calls = self.slang_ray_tracing_calls(func)
        helper_call_nodes = self.slang_user_function_call_nodes(func)
        if not calls and not helper_call_nodes:
            return
        if visited_helpers is None:
            visited_helpers = set()

        shader_stage = self.slang_shader_stage_name(shader_type)
        allowed_stages = {
            "TraceRay": {"raygeneration", "closesthit", "miss"},
            "CallShader": {"raygeneration", "closesthit", "miss", "callable"},
            "ReportHit": {"intersection"},
            "AcceptHitAndEndSearch": {"anyhit"},
            "IgnoreHit": {"anyhit"},
        }
        expected_arg_counts = {
            "TraceRay": {8, 11},
            "CallShader": {2},
            "ReportHit": {2, 3},
            "AcceptHitAndEndSearch": {0},
            "IgnoreHit": {0},
        }
        saved_variable_types = self.variable_types.copy()
        saved_buffer_resource_types = self.buffer_resource_types.copy()
        saved_buffer_resource_accesses = self.buffer_resource_accesses.copy()
        try:
            for parameter in parameters or []:
                type_name = self.slang_parameter_type_name(parameter)
                if type_name:
                    self.register_variable_type(parameter.name, type_name, parameter)
            for name, type_name in self.slang_function_scope_variable_types(
                func
            ).items():
                self.register_variable_type(name, type_name)

            for operation, args in calls:
                if operation not in allowed_stages:
                    continue
                if shader_stage not in allowed_stages[operation]:
                    valid_stages = ", ".join(sorted(allowed_stages[operation]))
                    raise ValueError(
                        f"Slang {shader_type} stage cannot call {operation}; "
                        f"{operation} is only valid in: {valid_stages}"
                    )
                expected_counts = expected_arg_counts[operation]
                if len(args) not in expected_counts:
                    expected = " or ".join(
                        str(count) for count in sorted(expected_counts)
                    )
                    raise ValueError(
                        f"Slang {shader_type} {operation} requires {expected} "
                        f"argument(s), got {len(args)}"
                    )
                self.validate_slang_ray_tracing_call_arguments(
                    operation, args, shader_type
                )
            for helper_call in helper_call_nodes:
                self.validate_slang_ray_helper_call_arguments(helper_call, shader_type)

            for helper_call in helper_call_nodes:
                helper_name = self.slang_function_call_name(helper_call)
                if helper_name in visited_helpers:
                    continue
                helper_func = self.user_functions_by_name.get(helper_name)
                if helper_func is None or helper_func is func:
                    continue
                helper_params = getattr(
                    helper_func, "parameters", getattr(helper_func, "params", [])
                )
                self.validate_slang_ray_tracing_calls(
                    helper_func,
                    shader_type,
                    helper_params,
                    visited_helpers | {helper_name},
                )
        finally:
            self.variable_types = saved_variable_types
            self.buffer_resource_types = saved_buffer_resource_types
            self.buffer_resource_accesses = saved_buffer_resource_accesses

    def slang_ray_query_method_return_type(self, operation):
        return {
            "Proceed": "bool",
            "Abort": "void",
            "TraceRayInline": "void",
            "CommitNonOpaqueTriangleHit": "void",
            "CommitProceduralPrimitiveHit": "void",
            "CandidateType": "uint",
            "CommittedType": "uint",
            "CommittedStatus": "uint",
            "CandidatePrimitiveIndex": "uint",
            "CommittedPrimitiveIndex": "uint",
            "CandidateInstanceID": "uint",
            "CommittedInstanceID": "uint",
            "CandidateInstanceIndex": "uint",
            "CommittedInstanceIndex": "uint",
            "CandidateGeometryIndex": "uint",
            "CommittedGeometryIndex": "uint",
            "CandidateObjectRayOrigin": "float3",
            "CandidateObjectRayDirection": "float3",
            "CommittedObjectRayOrigin": "float3",
            "CommittedObjectRayDirection": "float3",
            "CandidateRayT": "float",
            "CommittedRayT": "float",
            "CandidateObjectRayTMin": "float",
            "CandidateTriangleBarycentrics": "float2",
            "CommittedTriangleBarycentrics": "float2",
            "CandidateTriangleFrontFace": "bool",
            "CommittedTriangleFrontFace": "bool",
            "CandidateObjectToWorld3x4": "float3x4",
            "CandidateWorldToObject3x4": "float3x4",
            "CommittedObjectToWorld3x4": "float3x4",
            "CommittedWorldToObject3x4": "float3x4",
        }.get(operation)

    def slang_ray_query_method_names(self):
        return {
            "Proceed",
            "Abort",
            "TraceRayInline",
            "CommitNonOpaqueTriangleHit",
            "CommitProceduralPrimitiveHit",
            "CandidateType",
            "CommittedType",
            "CommittedStatus",
            "CandidatePrimitiveIndex",
            "CommittedPrimitiveIndex",
            "CandidateInstanceID",
            "CommittedInstanceID",
            "CandidateInstanceIndex",
            "CommittedInstanceIndex",
            "CandidateGeometryIndex",
            "CommittedGeometryIndex",
            "CandidateObjectRayOrigin",
            "CandidateObjectRayDirection",
            "CommittedObjectRayOrigin",
            "CommittedObjectRayDirection",
            "CandidateRayT",
            "CommittedRayT",
            "CandidateObjectRayTMin",
            "CandidateTriangleBarycentrics",
            "CommittedTriangleBarycentrics",
            "CandidateTriangleFrontFace",
            "CommittedTriangleFrontFace",
            "CandidateObjectToWorld3x4",
            "CandidateWorldToObject3x4",
            "CommittedObjectToWorld3x4",
            "CommittedWorldToObject3x4",
        }

    def slang_ray_query_call_parts(self, node):
        if isinstance(node, RayQueryOpNode):
            return (
                getattr(node, "operation", None),
                getattr(node, "query_expr", None),
                getattr(node, "arguments", []),
            )

        if not isinstance(node, FunctionCallNode):
            return None

        func_expr = getattr(node, "function", None) or getattr(node, "name", None)
        if isinstance(func_expr, PointerAccessNode):
            operation = str(getattr(func_expr, "member", ""))
            if operation not in self.slang_ray_query_method_names():
                return None
            return (
                operation,
                getattr(func_expr, "pointer_expr", None),
                getattr(node, "arguments", getattr(node, "args", [])),
            )

        if not isinstance(func_expr, MemberAccessNode):
            return None

        operation = str(getattr(func_expr, "member", ""))
        if operation not in self.slang_ray_query_method_names():
            return None
        return (
            operation,
            getattr(func_expr, "object", getattr(func_expr, "object_expr", None)),
            getattr(node, "arguments", getattr(node, "args", [])),
        )

    def slang_ray_query_calls(self, func):
        calls = []
        for node in self.walk_ast(getattr(func, "body", [])):
            call = self.slang_ray_query_call_parts(node)
            if call is not None:
                calls.append(call)
        return calls

    def validate_slang_ray_query_receiver(self, query_expr, shader_type, operation):
        base_type, array_suffix = self.slang_expression_mapped_base_and_array_suffix(
            query_expr
        )
        if base_type is None:
            return
        is_ray_query_type = base_type == "RayQuery" or (
            base_type.startswith("RayQuery<") and base_type.endswith(">")
        )
        if not (not array_suffix and is_ray_query_type):
            actual_type = f"{base_type}{array_suffix}"
            raise ValueError(
                f"Slang {shader_type} RayQuery.{operation} receiver must be "
                f"RayQuery, got {actual_type}"
            )
        if not self.slang_expression_is_lvalue(query_expr):
            actual_type = f"{base_type}{array_suffix}"
            raise ValueError(
                f"Slang {shader_type} RayQuery.{operation} receiver must be "
                f"RayQuery lvalue, got {actual_type}"
            )

    def validate_slang_ray_query_call_arguments(
        self, operation, query_expr, args, shader_type
    ):
        self.validate_slang_ray_query_receiver(query_expr, shader_type, operation)

        expected_arg_counts = {
            "TraceRayInline": {4},
            "CommitProceduralPrimitiveHit": {1},
        }
        expected_counts = expected_arg_counts.get(operation, {0})
        if len(args) not in expected_counts:
            expected = " or ".join(str(count) for count in sorted(expected_counts))
            raise ValueError(
                f"Slang {shader_type} RayQuery.{operation} requires {expected} "
                f"argument(s), got {len(args)}"
            )

        if operation == "TraceRayInline":
            self.validate_slang_ray_exact_type_argument(
                args[0],
                shader_type,
                "RayQuery.TraceRayInline",
                "acceleration structure",
                "RaytracingAccelerationStructure",
            )
            self.validate_slang_ray_scalar_int_uint_argument(
                args[1], shader_type, "RayQuery.TraceRayInline", "ray flags"
            )
            self.validate_slang_ray_scalar_int_uint_argument(
                args[2],
                shader_type,
                "RayQuery.TraceRayInline",
                "instance inclusion mask",
            )
            self.validate_slang_ray_exact_type_argument(
                args[3],
                shader_type,
                "RayQuery.TraceRayInline",
                "ray descriptor",
                "RayDesc",
            )
        elif operation == "CommitProceduralPrimitiveHit":
            self.validate_slang_ray_scalar_float_argument(
                args[0],
                shader_type,
                "RayQuery.CommitProceduralPrimitiveHit",
                "hit distance",
            )

    def validate_slang_ray_query_calls(self, func, shader_type, parameters):
        calls = self.slang_ray_query_calls(func)
        if not calls:
            return

        saved_variable_types = self.variable_types.copy()
        saved_buffer_resource_types = self.buffer_resource_types.copy()
        saved_buffer_resource_accesses = self.buffer_resource_accesses.copy()
        try:
            for parameter in parameters or []:
                type_name = self.slang_parameter_type_name(parameter)
                if type_name:
                    self.register_variable_type(parameter.name, type_name, parameter)
            for name, type_name in self.slang_function_scope_variable_types(
                func
            ).items():
                self.register_variable_type(name, type_name)

            for operation, query_expr, args in calls:
                if operation not in self.slang_ray_query_method_names():
                    continue
                self.validate_slang_ray_query_call_arguments(
                    operation, query_expr, args, shader_type
                )
        finally:
            self.variable_types = saved_variable_types
            self.buffer_resource_types = saved_buffer_resource_types
            self.buffer_resource_accesses = saved_buffer_resource_accesses

    def slang_ray_stage_parameter_declaration(
        self, declaration, parameter, shader_type
    ):
        shader_stage = self.slang_shader_stage_name(shader_type)
        if shader_stage not in self.slang_ray_stage_types():
            return None

        role = self.slang_ray_semantic_role(parameter, shader_type)
        if role is None:
            return None

        qualifier = "in" if role == "hit_attribute" else "inout"
        return f"{qualifier} {declaration}"

    def slang_mesh_payload_parameter_declaration(
        self, declaration, parameter, shader_type
    ):
        if self.slang_shader_stage_name(shader_type) != "mesh":
            return None
        if not self.is_slang_mesh_payload_parameter(parameter):
            return None

        return (
            f"{self.slang_parameter_qualifier_prefix(parameter)}payload {declaration}"
        )

    def slang_intrinsic_builtin_candidates(self, shader_type):
        shader_stage = self.slang_shader_stage_name(shader_type)
        if shader_stage not in self.slang_ray_stage_types():
            return {}

        return {
            "gl_HitKindEXT": ("uint", "HitKind()"),
            "gl_HitTEXT": ("float", "RayTCurrent()"),
            "gl_InstanceCustomIndexEXT": ("uint", "InstanceIndex()"),
            "gl_InstanceID": ("uint", "InstanceID()"),
            "gl_LaunchIDEXT": ("uvec3", "DispatchRaysIndex()"),
            "gl_LaunchSizeEXT": ("uvec3", "DispatchRaysDimensions()"),
            "gl_GeometryIndexEXT": ("uint", "GeometryIndex()"),
            "gl_ObjectRayDirectionEXT": ("vec3", "ObjectRayDirection()"),
            "gl_ObjectRayOriginEXT": ("vec3", "ObjectRayOrigin()"),
            "gl_PrimitiveID": ("uint", "PrimitiveIndex()"),
            "gl_RayTmaxEXT": ("float", "RayTCurrent()"),
            "gl_RayTminEXT": ("float", "RayTMin()"),
            "gl_WorldRayDirectionEXT": ("vec3", "WorldRayDirection()"),
            "gl_WorldRayOriginEXT": ("vec3", "WorldRayOrigin()"),
        }

    def slang_implicit_stage_parameter_candidates(self, shader_type, stage_role=None):
        shader_stage = self.slang_shader_stage_name(shader_type)
        thread_parameters = {
            "gl_WorkGroupID": ("uvec3", "SV_GroupID"),
            "gl_LocalInvocationID": ("uvec3", "SV_GroupThreadID"),
            "gl_GlobalInvocationID": ("uvec3", "SV_DispatchThreadID"),
            "gl_LocalInvocationIndex": ("uint", "SV_GroupIndex"),
        }
        if shader_stage in {"compute", "mesh", "amplification"}:
            return thread_parameters
        if shader_stage == "vertex":
            return {
                "gl_VertexID": ("uint", "SV_VertexID"),
                "gl_InstanceID": ("uint", "SV_InstanceID"),
                "gl_BaseVertex": ("int", "SV_StartVertexLocation"),
                "gl_BaseInstance": ("uint", "SV_StartInstanceLocation"),
                "gl_DrawID": ("uint", "SV_DrawID"),
            }
        if shader_stage == "fragment":
            return {
                "gl_FragCoord": ("vec4", "SV_Position"),
                "gl_FrontFacing": ("bool", "SV_IsFrontFace"),
                "gl_PrimitiveID": ("uint", "SV_PrimitiveID"),
                "gl_SampleID": ("uint", "SV_SampleIndex"),
                "gl_Layer": ("uint", "SV_RenderTargetArrayIndex"),
                "gl_ViewportIndex": ("uint", "SV_ViewportArrayIndex"),
            }
        if shader_stage == "geometry":
            return {
                "gl_PrimitiveIDIn": ("uint", "SV_PrimitiveID"),
                "gl_InvocationID": ("uint", "SV_GSInstanceID"),
            }
        if shader_stage == "hull":
            if stage_role == "patch_constant":
                return {
                    "gl_PrimitiveID": ("uint", "SV_PrimitiveID"),
                }
            return {
                "gl_InvocationID": ("uint", "SV_OutputControlPointID"),
                "gl_PrimitiveID": ("uint", "SV_PrimitiveID"),
            }
        if shader_stage == "domain":
            return {
                "gl_TessCoord": ("vec3", "SV_DomainLocation"),
                "gl_PrimitiveID": ("uint", "SV_PrimitiveID"),
            }
        return {}

    def walk_ast(self, root):
        visited = set()

        def walk(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, dict):
                for item in value.values():
                    yield from walk(item)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    yield from walk(item)
                return

            value_id = id(value)
            if value_id in visited:
                return
            visited.add(value_id)
            yield value

            if hasattr(value, "__dict__"):
                for child in vars(value).values():
                    yield from walk(child)

        yield from walk(root)

    def slang_stage_builtin_return_rewrite(
        self, body_statements, shader_type, ret_type_name, semantic
    ):
        if semantic is not None:
            return None
        if self.convert_type(ret_type_name) != "void" or not body_statements:
            return None
        if self.contains_return_statement(body_statements):
            return None

        output_assignments = self.slang_stage_builtin_output_assignments(
            body_statements, shader_type
        )
        output_targets = self.unique_slang_builtin_output_targets(output_assignments)
        if not output_targets:
            return None

        if len(output_targets) > 1:
            return self.slang_stage_builtin_struct_return_rewrite(
                shader_type, body_statements, output_targets
            )

        target_name = output_targets[0]
        return_type_name = self.slang_stage_builtin_return_type(
            shader_type, target_name
        )
        if return_type_name is None:
            return None

        final_statement = body_statements[-1]
        assignment = self.assignment_from_statement(final_statement)
        if (
            len(output_assignments) == 1
            and assignment is output_assignments[0][0]
            and getattr(assignment, "operator", None) == "="
        ):
            return {
                "mode": "direct",
                "statement_index": len(body_statements) - 1,
                "return_type_name": return_type_name,
                "semantic": target_name,
                "value": getattr(
                    assignment, "right", getattr(assignment, "value", None)
                ),
            }

        return {
            "mode": "local",
            "return_type_name": return_type_name,
            "semantic": target_name,
            "target_name": target_name,
            "local_name": self.unique_slang_builtin_output_local_name(
                target_name, body_statements
            ),
        }

    def contains_return_statement(self, body_statements):
        return any(
            isinstance(node, ReturnNode) for node in self.walk_ast(body_statements)
        )

    def unique_slang_builtin_output_targets(self, output_assignments):
        targets = []
        seen = set()
        for _assignment, target_name in output_assignments:
            if target_name not in seen:
                seen.add(target_name)
                targets.append(target_name)
        return targets

    def slang_stage_builtin_struct_return_rewrite(
        self, shader_type, body_statements, output_targets
    ):
        struct_name = self.unique_slang_builtin_output_struct_name(shader_type)
        local_name = self.unique_slang_builtin_output_local_name(
            "gl_Output", body_statements
        )
        fields = []
        for target_name in output_targets:
            return_type_name = self.slang_stage_builtin_return_type(
                shader_type, target_name
            )
            if return_type_name is None:
                return None
            fields.append(
                {
                    "target_name": target_name,
                    "field_name": self.slang_builtin_output_field_name(target_name),
                    "return_type_name": return_type_name,
                    "semantic": target_name,
                }
            )

        return {
            "mode": "struct",
            "return_type_name": struct_name,
            "semantic": None,
            "local_name": local_name,
            "fields": fields,
        }

    def unique_slang_builtin_output_struct_name(self, shader_type):
        base_name = f"CGL{self.slang_stage_entry_function_name(shader_type)}Output"
        used_names = set(self.user_symbol_names)
        if base_name not in used_names:
            return base_name

        suffix = 1
        while f"{base_name}_{suffix}" in used_names:
            suffix += 1
        return f"{base_name}_{suffix}"

    def slang_builtin_output_field_name(self, target_name):
        target_suffix = (
            target_name[3:] if target_name.startswith("gl_") else target_name
        )
        return f"cgl_{target_suffix}"

    def slang_stage_builtin_return_aliases(self, rewrite):
        if rewrite is None:
            return {}
        if rewrite["mode"] == "local":
            return {rewrite["target_name"]: rewrite["local_name"]}
        if rewrite["mode"] == "struct":
            return {
                field["target_name"]: f"{rewrite['local_name']}.{field['field_name']}"
                for field in rewrite["fields"]
            }
        return {}

    def generate_slang_builtin_output_struct(self, rewrite):
        result = f"struct {rewrite['return_type_name']}\n{{\n"
        for field in rewrite["fields"]:
            field_type = self.convert_type(field["return_type_name"])
            semantic = self.semantic_suffix(field["semantic"])
            result += f"    {field_type} {field['field_name']}{semantic};\n"
        result += "};"
        return result

    def slang_stage_builtin_output_assignments(self, body_statements, shader_type):
        assignments = []
        for node in self.walk_ast(body_statements):
            if not isinstance(node, AssignmentNode):
                continue
            if getattr(node, "operator", None) != "=":
                continue

            target = getattr(node, "left", getattr(node, "target", None))
            target_name = self.slang_stage_builtin_output_target_name(target)
            if self.slang_stage_builtin_return_type(shader_type, target_name):
                assignments.append((node, target_name))
        return assignments

    def slang_stage_builtin_output_target_name(self, target):
        target_name = self.identifier_name(target)
        if target_name is not None:
            return target_name

        if isinstance(target, ArrayAccessNode):
            array_name = self.identifier_name(getattr(target, "array", None))
            index = self.literal_int_value(getattr(target, "index", None))
            if array_name == "gl_FragData":
                if index is None or index < 0:
                    raise ValueError(
                        "Slang fragment output gl_FragData requires a "
                        "non-negative literal render-target index"
                    )
                return f"gl_FragColor{index}"

        return None

    def unique_slang_builtin_output_local_name(self, target_name, body_statements):
        target_suffix = (
            target_name[3:] if target_name.startswith("gl_") else target_name
        )
        base_name = f"cgl_{target_suffix}"
        used_names = set()
        for node in self.walk_ast(body_statements):
            if hasattr(node, "name") and isinstance(getattr(node, "name"), str):
                used_names.add(node.name)
        if base_name not in used_names:
            return base_name

        suffix = 1
        while f"{base_name}_{suffix}" in used_names:
            suffix += 1
        return f"{base_name}_{suffix}"

    def slang_stage_builtin_return_type(self, shader_type, target_name):
        if target_name is None:
            return None
        if shader_type == "vertex":
            if target_name == "gl_Position":
                return "vec4"
            if target_name == "gl_PointSize":
                return "float"
            if target_name in {"gl_Layer", "gl_ViewportIndex"}:
                return "uint"
        if shader_type == "fragment":
            if target_name == "gl_FragDepth":
                return "float"
            if target_name == "gl_FragColor" or (
                target_name.startswith("gl_FragColor")
                and target_name[len("gl_FragColor") :].isdigit()
            ):
                return "vec4"
        return None

    def assignment_from_statement(self, statement):
        if isinstance(statement, AssignmentNode):
            return statement
        if isinstance(statement, ExpressionStatementNode) and isinstance(
            getattr(statement, "expression", None), AssignmentNode
        ):
            return statement.expression
        return None

    def identifier_name(self, node):
        if isinstance(node, IdentifierNode):
            return node.name
        return None

    def emit_function_body_statement(self, statement, statement_index, rewrite):
        if (
            rewrite is not None
            and rewrite["mode"] == "direct"
            and statement_index == rewrite["statement_index"]
        ):
            value = self.generate_expression_with_expected(
                rewrite["value"], rewrite["return_type_name"]
            )
            return f"{self.indent()}return {value};"
        return self.emit_statement(statement)

    def emit_statement(self, node):
        statement = self.generate_statement(node)
        return self.indent_statement_text(statement)

    def indent_statement_text(self, statement):
        lines = statement.splitlines()
        return "\n".join(
            self.indent() + line if line and not line[0].isspace() else line
            for line in lines
        )

    def expression_prelude_active(self):
        return bool(self.expression_prelude_stack)

    def add_expression_prelude(self, lines, result_name=None):
        if not self.expression_prelude_stack:
            return
        self.expression_prelude_stack[-1].extend(lines)
        if result_name:
            self.expression_prelude_result_stack[-1].add(result_name)

    def generate_expression_with_prelude(self, expr, expected_type=None):
        self.expression_prelude_stack.append([])
        self.expression_prelude_result_stack.append(set())
        try:
            if expected_type is None:
                value = self.generate_expression(expr)
            else:
                value = self.generate_expression_with_expected(expr, expected_type)
            prelude = self.expression_prelude_stack[-1]
            result_names = self.expression_prelude_result_stack[-1]
            return prelude, result_names, value
        finally:
            self.expression_prelude_stack.pop()
            self.expression_prelude_result_stack.pop()

    def statement_with_prelude(self, prelude, statement):
        if not prelude:
            return statement
        statements = list(prelude)
        if statement:
            statements.append(statement)
        return "\n".join(statements)

    def generate_statement(self, node):
        """Render a single CrossGL statement as Slang code."""
        if isinstance(node, ReturnNode):
            if node.value is None:
                return "return;"
            prelude, _results, value = self.generate_expression_with_prelude(
                node.value, self.current_function_return_type
            )
            return self.statement_with_prelude(prelude, f"return {value};")
        elif isinstance(node, AssignmentNode):
            return self.generate_assignment_statement(node)
        elif isinstance(node, ExpressionStatementNode):
            tail_return = self.generate_tail_expression_statement(node)
            if tail_return is not None:
                return tail_return
            synchronization_statement = self.generate_slang_synchronization_statement(
                node.expression
            )
            if synchronization_statement is not None:
                return synchronization_statement
            prelude, result_names, expr = self.generate_expression_with_prelude(
                node.expression
            )
            statement = "" if prelude and expr in result_names else f"{expr};"
            return self.statement_with_prelude(prelude, statement)
        elif isinstance(node, VariableNode):
            return self.generate_variable_statement(node)
        elif isinstance(node, IfNode):
            return self.generate_if(node)
        elif isinstance(node, ForNode):
            return self.generate_for(node)
        elif isinstance(node, ForInNode):
            return self.generate_for_in(node)
        elif isinstance(node, WhileNode):
            return self.generate_while(node)
        elif isinstance(node, DoWhileNode):
            return self.generate_do_while(node)
        elif isinstance(node, MatchNode):
            return self.generate_match(node)
        elif isinstance(node, SwitchNode):
            return self.generate_switch(node)
        elif isinstance(node, BreakNode):
            return "break;"
        elif isinstance(node, ContinueNode):
            return "continue;"
        else:
            synchronization_statement = self.generate_slang_synchronization_statement(
                node
            )
            if synchronization_statement is not None:
                return synchronization_statement
            prelude, result_names, expr = self.generate_expression_with_prelude(node)
            statement = "" if prelude and expr in result_names else f"{expr};"
            return self.statement_with_prelude(prelude, statement)

    def generate_variable_statement(self, node):
        initial_value = getattr(node, "initial_value", getattr(node, "value", None))
        var_type = self.variable_declaration_type(node, initial_value)
        self.register_variable_type(node.name, var_type, node)
        declaration = self.format_declaration(var_type, node.name, node)
        declaration = self.slang_declaration_qualifier_prefix(node) + declaration
        if initial_value is not None:
            prelude, _results, initial_expr = self.generate_expression_with_prelude(
                initial_value,
                self.initializer_expected_type(var_type),
            )
            return self.statement_with_prelude(
                prelude, f"{declaration} = {initial_expr};"
            )
        return f"{declaration};"

    def generate_assignment(self, node):
        left = self.slang_assignment_target_alias(
            node.left
        ) or self.generate_expression(node.left)
        right = self.generate_expression_with_expected(
            node.right, self.expression_result_type(node.left)
        )
        if node.operator == "%=" and self.modulo_requires_fmod(node.left, node.right):
            return f"{left} = fmod({left}, {right})"
        return f"{left} {node.operator} {right}"

    def generate_assignment_statement(self, node):
        left = self.slang_assignment_target_alias(
            node.left
        ) or self.generate_expression(node.left)
        prelude, _results, right = self.generate_expression_with_prelude(
            node.right, self.expression_result_type(node.left)
        )
        if node.operator == "%=" and self.modulo_requires_fmod(node.left, node.right):
            statement = f"{left} = fmod({left}, {right});"
        else:
            statement = f"{left} {node.operator} {right};"
        return self.statement_with_prelude(prelude, statement)

    def tail_expression_returns(self, node):
        if not isinstance(node, ExpressionStatementNode):
            return False
        if not getattr(node, "is_tail_expression", False):
            return False
        return_type = self.type_name_string(self.current_function_return_type)
        return bool(return_type and return_type != "void")

    def generate_tail_expression_statement(self, node):
        if not self.tail_expression_returns(node):
            return None
        return_type = self.type_name_string(self.current_function_return_type)
        prelude, _results, value = self.generate_expression_with_prelude(
            node.expression,
            return_type,
        )
        return self.statement_with_prelude(prelude, f"return {value};")

    def generate_for_header_statement(self, node, atomic_value_context=None):
        if atomic_value_context is not None:
            self.atomic_value_context_stack.append(atomic_value_context)
            try:
                return self.generate_for_header_statement(node)
            finally:
                self.atomic_value_context_stack.pop()

        if isinstance(node, AssignmentNode):
            return self.generate_assignment(node)
        if isinstance(node, VariableNode):
            initial_value = getattr(node, "initial_value", getattr(node, "value", None))
            var_type = self.variable_declaration_type(node, initial_value)
            self.register_variable_type(node.name, var_type, node)
            declaration = self.format_declaration(var_type, node.name, node)
            declaration = self.slang_declaration_qualifier_prefix(node) + declaration
            if initial_value is not None:
                initial_expr = self.generate_expression_with_expected(
                    initial_value,
                    self.initializer_expected_type(var_type),
                )
                return f"{declaration} = {initial_expr}"
            return declaration
        return self.generate_expression(node)

    def generate_loop_condition_expression(self, node, atomic_value_context):
        self.atomic_value_context_stack.append(atomic_value_context)
        try:
            return self.generate_expression(node)
        finally:
            self.atomic_value_context_stack.pop()

    def slang_assignment_target_alias(self, target):
        hull_output_alias = self.slang_hull_output_array_alias(target)
        if hull_output_alias is not None:
            return hull_output_alias

        hull_output_alias = self.slang_hull_output_member_alias(target)
        if hull_output_alias is not None:
            return hull_output_alias

        target_name = self.slang_stage_builtin_output_target_name(target)
        if target_name is None:
            return None
        return self.identifier_aliases.get(target_name)

    def slang_hull_output_array_alias(self, target):
        if self.current_hull_output_rewrite is None:
            return None
        if not isinstance(target, ArrayAccessNode):
            return None

        array = getattr(target, "array", getattr(target, "array_expr", None))
        if self.identifier_name(array) != "gl_out":
            return None

        return self.current_hull_output_rewrite["local_name"]

    def slang_hull_output_member_alias(self, target):
        if self.current_hull_output_rewrite is None:
            return None

        member_target = self.slang_hull_output_member_target(target)
        if member_target is None:
            return None

        member, _index = member_target
        return f"{self.current_hull_output_rewrite['local_name']}.{member}"

    def generate_expression_with_expected(self, expr, expected_type):
        previous_expected_type = self.current_expression_expected_type
        self.current_expression_expected_type = self.type_name_string(expected_type)
        try:
            return self.generate_expression(expr)
        finally:
            self.current_expression_expected_type = previous_expected_type

    def type_name_string(self, type_name):
        if type_name is None:
            return None
        if not isinstance(type_name, str):
            return self.convert_type_node_to_string(type_name)
        return type_name

    def is_scalar_value_type(self, type_name):
        type_name = self.type_name_string(type_name)
        if not type_name:
            return False
        return self.convert_type(type_name) in {
            "float",
            "double",
            "int",
            "uint",
            "bool",
        }

    def is_vector_value_type(self, type_name):
        type_name = self.type_name_string(type_name)
        if not type_name:
            return False
        return self.convert_type(type_name) in {
            "float2",
            "float3",
            "float4",
            "double2",
            "double3",
            "double4",
            "int2",
            "int3",
            "int4",
            "uint2",
            "uint3",
            "uint4",
            "bool2",
            "bool3",
            "bool4",
        }

    def vector_component_type(self, type_name):
        mapped_type = self.convert_type(type_name)
        if mapped_type.startswith("double"):
            return "double"
        if mapped_type.startswith("float"):
            return "float"
        if mapped_type.startswith("uint"):
            return "uint"
        if mapped_type.startswith("int"):
            return "int"
        if mapped_type.startswith("bool"):
            return "bool"
        return None

    def vector_value_info(self, type_name):
        if type_name is None:
            return None
        mapped_type = self.convert_type(type_name)
        for component_type in ("double", "float", "uint", "int", "bool"):
            if not mapped_type.startswith(component_type):
                continue
            suffix = mapped_type[len(component_type) :]
            if suffix in {"2", "3", "4"}:
                size = int(suffix)
                return {
                    "type": mapped_type,
                    "component_type": component_type,
                    "size": size,
                    "components": ("x", "y", "z", "w")[:size],
                }
        return None

    def generate_bool_mix_call(self, args):
        if len(args) != 3:
            return None

        condition_type = self.expression_result_type(args[2])
        condition_info = self.vector_value_info(condition_type)
        if condition_info is not None:
            if condition_info["component_type"] != "bool":
                return None
            return self.generate_bool_vector_select_expression(
                args[2], args[1], args[0]
            )

        if self.convert_type(condition_type) != "bool":
            return None

        condition = self.generate_expression(args[2])
        true_value = self.generate_expression(args[1])
        false_value = self.generate_expression(args[0])
        return f"({condition} ? {true_value} : {false_value})"

    def generate_bool_vector_select_expression(
        self, condition_expr, true_expr, false_expr
    ):
        condition_info = self.vector_value_info(
            self.expression_result_type(condition_expr)
        )
        if condition_info is None or condition_info["component_type"] != "bool":
            return None

        true_info = self.vector_value_info(self.expression_result_type(true_expr))
        false_info = self.vector_value_info(self.expression_result_type(false_expr))
        if (
            true_info is None
            or false_info is None
            or true_info["type"] != false_info["type"]
            or condition_info["size"] != true_info["size"]
        ):
            return None

        helper_name = self.require_vector_select_helper(true_info, condition_info)
        condition = self.generate_expression(condition_expr)
        true_value = self.generate_expression(true_expr)
        false_value = self.generate_expression(false_expr)
        return f"{helper_name}({condition}, {true_value}, {false_value})"

    def require_vector_select_helper(self, result_info, condition_info):
        result_type = result_info["type"]
        condition_type = condition_info["type"]
        base_name = f"_crossgl_select_{condition_type}_{result_type}"
        helper_name = self.helper_function_name(base_name)
        if helper_name in self.helper_functions:
            return helper_name

        components = [
            f"(mask.{component} ? trueValue.{component} : falseValue.{component})"
            for component in result_info["components"]
        ]
        helper = (
            f"{result_type} {helper_name}("
            f"{condition_type} mask, {result_type} trueValue, "
            f"{result_type} falseValue)\n"
            "{\n"
            f"    return {result_type}({', '.join(components)});\n"
            "}"
        )
        self.register_helper_function(helper_name, helper)
        return helper_name

    def modulo_requires_fmod(self, left_expr, right_expr):
        """Return whether scalar/vector modulo needs Slang fmod lowering."""
        left_component = self.vector_component_type(
            self.expression_result_type(left_expr)
        )
        right_component = self.vector_component_type(
            self.expression_result_type(right_expr)
        )
        return left_component in {"float", "double"} or right_component in {
            "float",
            "double",
        }

    def binary_expression_result_type(self, expr):
        left_type = self.expression_result_type(expr.left)
        right_type = self.expression_result_type(expr.right)
        if expr.op in {"==", "!=", "<", ">", "<=", ">="}:
            return self.comparison_expression_result_type(left_type, right_type)
        if expr.op in {"&&", "||"}:
            return self.logical_expression_result_type(left_type, right_type)
        if self.is_vector_value_type(left_type):
            return left_type
        if self.is_vector_value_type(right_type):
            return right_type
        if left_type == "float" or right_type == "float":
            return "float"
        return left_type or right_type

    def comparison_expression_result_type(self, left_type, right_type):
        left_info = self.vector_value_info(left_type)
        right_info = self.vector_value_info(right_type)
        if left_info is not None and right_info is not None:
            if left_info["size"] == right_info["size"]:
                return f"bool{left_info['size']}"
            return None
        if left_info is not None and self.is_scalar_value_type(right_type):
            return f"bool{left_info['size']}"
        if right_info is not None and self.is_scalar_value_type(left_type):
            return f"bool{right_info['size']}"
        if left_type is not None or right_type is not None:
            return "bool"
        return None

    def logical_expression_result_type(self, left_type, right_type):
        left_info = self.vector_value_info(left_type)
        right_info = self.vector_value_info(right_type)
        if (
            left_info is not None
            and left_info["component_type"] == "bool"
            and right_type == "bool"
        ):
            return f"bool{left_info['size']}"
        if (
            right_info is not None
            and right_info["component_type"] == "bool"
            and left_type == "bool"
        ):
            return f"bool{right_info['size']}"
        if (
            left_info is not None
            and right_info is not None
            and left_info["component_type"] == "bool"
            and right_info["component_type"] == "bool"
            and left_info["size"] == right_info["size"]
        ):
            return f"bool{left_info['size']}"
        if left_type == "bool" and right_type == "bool":
            return "bool"
        return None

    def expression_result_type(self, expr):
        if expr is None:
            return None
        if isinstance(expr, VariableNode):
            return self.variable_types.get(getattr(expr, "name", None))
        if isinstance(expr, IdentifierNode):
            return self.variable_types.get(getattr(expr, "name", None))
        if isinstance(expr, LiteralNode):
            literal_type = getattr(getattr(expr, "literal_type", None), "name", None)
            if literal_type:
                return literal_type
            if isinstance(expr.value, float):
                return "float"
            if isinstance(expr.value, int) and not isinstance(expr.value, bool):
                return "int"
            if isinstance(expr.value, bool):
                return "bool"
        if isinstance(expr, BinaryOpNode):
            return self.binary_expression_result_type(expr)
        if isinstance(expr, UnaryOpNode):
            operand_type = self.expression_result_type(expr.operand)
            if expr.op == "&" and operand_type is not None:
                operand_type = (
                    self.reference_referent_type_name(operand_type) or operand_type
                )
                return f"{operand_type}*"
            if expr.op == "*" and operand_type is not None:
                return self.pointer_pointee_type_name(operand_type) or operand_type
            return operand_type
        if isinstance(expr, AssignmentNode):
            return self.expression_result_type(getattr(expr, "left", None))
        if isinstance(expr, MatchNode):
            result_types = []
            for arm in getattr(expr, "arms", []) or []:
                _prefix, tail = self.match_expression_tail(getattr(arm, "body", []))
                result_type = self.expression_result_type(tail)
                if result_type is not None:
                    result_types.append(result_type)
            if result_types and all(
                result_type == result_types[0] for result_type in result_types
            ):
                return result_types[0]
            return None
        if isinstance(expr, ArrayAccessNode):
            array_type = self.type_name_string(self.expression_result_type(expr.array))
            if array_type and "[" in array_type and "]" in array_type:
                return self.array_element_type(array_type)
            return array_type
        if isinstance(expr, MemberAccessNode):
            object_type = self.expression_result_type(expr.object)
            member = str(expr.member)
            if object_type and all(ch in "xyzwrgba" for ch in member):
                component_type = self.vector_component_type(object_type)
                if component_type and len(member) == 1:
                    return component_type
                if component_type:
                    return f"{component_type}{len(member)}"
            member_type = self.struct_member_type(object_type, member)
            if member_type is not None:
                return member_type
            return None
        if isinstance(expr, PointerAccessNode):
            pointer_type = self.expression_result_type(expr.pointer_expr)
            member = str(expr.member)
            member_type = self.struct_member_type(pointer_type, member)
            if member_type is not None:
                return member_type
            return None
        if isinstance(expr, RayQueryOpNode):
            return self.slang_ray_query_method_return_type(expr.operation)
        if isinstance(expr, WaveOpNode):
            return self.slang_wave_result_type(expr)
        if isinstance(expr, AtomicOpNode):
            operation = self.image_atomic_operation_from_atomic_op(expr.operation)
            if operation is None:
                return None
            image_type = self.resource_base_type(self.image_resource_type(expr.target))
            return self.image_atomic_return_type(image_type)
        if isinstance(expr, FunctionCallNode):
            ray_query_call = self.slang_ray_query_call_parts(expr)
            if ray_query_call is not None:
                operation, _query_expr, _args = ray_query_call
                return self.slang_ray_query_method_return_type(operation)
            func_expr = getattr(expr, "function", None) or getattr(expr, "name", None)
            func_name = getattr(func_expr, "name", func_expr)
            if (
                isinstance(func_name, str)
                and func_name in self.structured_buffer_atomic_operations()
                and getattr(expr, "args", None)
            ):
                buffer_type = self.structured_buffer_atomic_target_resource_type(
                    expr.args[0]
                )
                if self.is_structured_buffer_resource_type(buffer_type):
                    element_type = self.structured_buffer_element_type(buffer_type)
                    return element_type if element_type in {"int", "uint"} else "uint"
            if func_name == "imageLoad" and getattr(expr, "args", None):
                return self.image_resource_element_type(
                    self.image_resource_type(expr.args[0])
                )
            if func_name == "buffer_load" and getattr(expr, "args", None):
                return self.structured_buffer_element_type(
                    self.structured_buffer_resource_type(expr.args[0])
                )
            if func_name == "buffer_consume" and getattr(expr, "args", None):
                return self.structured_buffer_element_type(
                    self.structured_buffer_resource_type(expr.args[0])
                )
            if func_name == "buffer_dimensions":
                return "uint"
            byte_address_result_type = self.byte_address_member_call_result_type(
                func_expr
            )
            if byte_address_result_type is not None:
                return byte_address_result_type
            structured_result_type = self.structured_buffer_member_call_result_type(
                func_expr
            )
            if structured_result_type is not None:
                return structured_result_type
            if isinstance(func_name, str) and func_name in self.user_struct_names:
                return func_name
            if isinstance(func_name, str) and func_name in {
                "float",
                "double",
                "int",
                "uint",
                "bool",
                "vec2",
                "vec3",
                "vec4",
                "ivec2",
                "ivec3",
                "ivec4",
                "uvec2",
                "uvec3",
                "uvec4",
                "bvec2",
                "bvec3",
                "bvec4",
                "float2",
                "float3",
                "float4",
                "int2",
                "int3",
                "int4",
                "uint2",
                "uint3",
                "uint4",
                "bool2",
                "bool3",
                "bool4",
            }:
                return str(func_name)
            return self.user_function_return_types.get(func_name)
        return None

    def struct_member_type(self, object_type, member_name):
        if object_type is None or member_name is None:
            return None
        object_type = self.type_name_string(self.member_lookup_type_name(object_type))
        if not object_type:
            return None
        object_type, _array_suffix = split_array_type_suffix(object_type)
        return self.struct_member_types.get(object_type, {}).get(member_name)

    def struct_constructor_argument_types(self, struct_name):
        member_types = self.struct_member_types.get(struct_name)
        if not member_types:
            return []
        return list(member_types.values())

    def generate_struct_constructor_arguments(self, struct_name, args):
        expected_types = self.struct_constructor_argument_types(struct_name)
        return self.generate_call_arguments(args, expected_types)

    def user_function_argument_types(self, function_name):
        return self.user_function_parameter_types.get(function_name) or []

    def generate_user_function_arguments(self, function_name, args):
        expected_types = self.user_function_argument_types(function_name)
        return self.generate_call_arguments(args, expected_types)

    def generate_call_arguments(self, args, expected_types):
        generated_args = []
        for index, arg in enumerate(args or []):
            expected_type = (
                expected_types[index] if index < len(expected_types) else None
            )
            if expected_type is None:
                generated_args.append(self.generate_expression(arg))
            else:
                generated_args.append(
                    self.generate_expression_with_expected(arg, expected_type)
                )
        return ", ".join(generated_args)

    def generate_literal(self, node):
        value = node.value
        literal_type = getattr(getattr(node, "literal_type", None), "name", None)

        if isinstance(value, bool):
            return "true" if value else "false"
        if literal_type == "bool" and isinstance(value, str):
            lower_value = value.lower()
            if lower_value in {"true", "false"}:
                return lower_value
        if literal_type == "char":
            escaped = self.escape_literal(value, quote="'")
            return f"'{escaped}'"
        if (
            literal_type == "uint"
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            return f"{value}u"
        if isinstance(value, str):
            escaped = self.escape_literal(value, quote='"')
            return f'"{escaped}"'
        return str(value)

    def escape_literal(self, value, quote):
        text = str(value)
        escaped = []
        for index, char in enumerate(text):
            if char == "\n":
                escaped.append("\\n")
            elif char == "\r":
                escaped.append("\\r")
            elif char == "\t":
                escaped.append("\\t")
            elif char == quote and (index == 0 or text[index - 1] != "\\"):
                escaped.append("\\" + char)
            else:
                escaped.append(char)
        return "".join(escaped)

    def binary_precedence(self, op):
        return self.BINARY_PRECEDENCE.get(op, 0)

    def binary_child_needs_parentheses(self, parent_op, child, is_right_child=False):
        if not isinstance(child, BinaryOpNode):
            return False

        parent_precedence = self.binary_precedence(parent_op)
        child_op = getattr(child, "op", getattr(child, "operator", ""))
        child_precedence = self.binary_precedence(child_op)
        if child_precedence < parent_precedence:
            return True
        if child_precedence > parent_precedence:
            return False
        return is_right_child and (
            parent_op not in self.ASSOCIATIVE_BINARY_OPS or child_op != parent_op
        )

    def generate_binary_expression(self, node):
        left = self.generate_expression(node.left)
        right = self.generate_expression(node.right)
        if self.binary_child_needs_parentheses(node.op, node.left):
            left = f"({left})"
        if self.binary_child_needs_parentheses(node.op, node.right, True):
            right = f"({right})"
        if node.op == "%" and self.modulo_requires_fmod(node.left, node.right):
            return f"fmod({left}, {right})"
        return f"{left} {node.op} {right}"

    def generate_expression(self, node):
        """Render a CrossGL expression as Slang expression syntax."""
        if isinstance(node, VariableNode):
            return node.name
        elif isinstance(node, IdentifierNode):
            return self.identifier_aliases.get(node.name, node.name)
        elif isinstance(node, LiteralNode):
            return self.generate_literal(node)
        elif isinstance(node, ExpressionStatementNode):
            return self.generate_expression(node.expression)
        elif isinstance(node, AssignmentNode):
            return self.generate_assignment(node)
        elif isinstance(node, ArrayAccessNode):
            hull_output_alias = self.slang_hull_output_array_alias(node)
            if hull_output_alias is not None:
                return hull_output_alias
            array = self.generate_expression(
                getattr(node, "array", getattr(node, "array_expr", None))
            )
            index = self.format_array_access_index(
                getattr(node, "index", getattr(node, "index_expr", None))
            )
            return f"{array}[{index}]"
        elif isinstance(node, ArrayLiteralNode):
            expected_element_type = self.array_literal_element_expected_type(
                self.current_expression_expected_type
            )
            elements = ", ".join(
                (
                    self.generate_expression_with_expected(
                        element, expected_element_type
                    )
                    if expected_element_type is not None
                    else self.generate_expression(element)
                )
                for element in node.elements
            )
            return f"{{{elements}}}"
        elif isinstance(node, MemberAccessNode):
            hull_output_alias = self.slang_hull_output_member_alias(node)
            if hull_output_alias is not None:
                return hull_output_alias
            obj = self.generate_expression(node.object)
            return f"{obj}.{node.member}"
        elif isinstance(node, PointerAccessNode):
            pointer = self.generate_expression(node.pointer_expr)
            return f"{pointer}->{node.member}"
        elif isinstance(node, BinaryOpNode):
            return self.generate_binary_expression(node)
        elif isinstance(node, WaveOpNode):
            return self.generate_slang_wave_op_expression(node)
        elif isinstance(node, MeshOpNode):
            return self.generate_slang_mesh_op_expression(node)
        elif isinstance(node, AtomicOpNode):
            return self.generate_slang_atomic_op_expression(node)
        elif isinstance(node, RayQueryOpNode):
            query = self.generate_expression(node.query_expr)
            args = ", ".join(self.generate_expression(arg) for arg in node.arguments)
            return f"{query}.{node.operation}({args})"
        elif isinstance(node, FunctionCallNode):
            func_expr = getattr(node, "function", None)
            if func_expr is None:
                func_expr = node.name
            if hasattr(func_expr, "name") and getattr(func_expr, "name", None):
                callee = func_expr.name
            elif isinstance(func_expr, str):
                callee = func_expr
            else:
                callee = self.generate_expression(func_expr)
            synchronization_call = self.generate_slang_synchronization_call(
                callee, node.args, statement_context=False
            )
            if synchronization_call is not None:
                return synchronization_call
            if callee not in self.user_function_names:
                resource_call = self.generate_resource_call(callee, node.args)
                if resource_call is not None:
                    return resource_call
            if isinstance(func_expr, MemberAccessNode):
                resource_member_call = self.generate_resource_member_call(
                    func_expr, node.args
                )
                if resource_member_call is not None:
                    return resource_member_call
            if callee == "mix" and callee not in self.user_function_names:
                bool_mix = self.generate_bool_mix_call(node.args)
                if bool_mix is not None:
                    return bool_mix
            if callee == "lambda":
                lambda_expr = self.generate_lambda_expression(node.args)
                if lambda_expr is not None:
                    return lambda_expr
            if (
                isinstance(callee, str)
                and callee in self.user_struct_names
                and callee not in self.user_function_names
            ):
                args = self.generate_struct_constructor_arguments(callee, node.args)
                return f"{self.convert_type(callee)}({args})"
            if isinstance(callee, str) and callee in self.user_function_names:
                args = self.generate_user_function_arguments(callee, node.args)
            else:
                args = ", ".join([self.generate_expression(arg) for arg in node.args])
            callee = self.convert_type(callee)
            if (
                callee == "saturate"
                and len(node.args) == 1
                and callee not in self.user_function_names
            ):
                return f"clamp({args}, 0.0, 1.0)"
            if callee not in self.user_function_names:
                callee = self.function_map.get(callee, callee)
            return f"{callee}({args})"
        elif isinstance(node, RayTracingOpNode):
            return self.generate_slang_ray_tracing_op_expression(node)
        elif isinstance(node, UnaryOpNode):
            operand = self.generate_expression(node.operand)
            if isinstance(node.operand, BinaryOpNode):
                operand = f"({operand})"
            if getattr(node, "is_postfix", False):
                return f"{operand}{node.op}"
            return f"{node.op}{operand}"
        elif isinstance(node, TernaryOpNode):
            bool_vector_select = self.generate_bool_vector_select_expression(
                node.condition, node.true_expr, node.false_expr
            )
            if bool_vector_select is not None:
                return bool_vector_select
            condition = self.generate_expression(node.condition)
            if self.current_expression_expected_type is not None:
                true_expr = self.generate_expression_with_expected(
                    node.true_expr, self.current_expression_expected_type
                )
                false_expr = self.generate_expression_with_expected(
                    node.false_expr, self.current_expression_expected_type
                )
            else:
                true_expr = self.generate_expression(node.true_expr)
                false_expr = self.generate_expression(node.false_expr)
            return f"({condition} ? {true_expr} : {false_expr})"
        elif isinstance(node, MatchNode):
            return self.generate_match_expression(node)
        elif isinstance(node, str):
            return node
        else:
            return str(node)

    def function_call_simple_callee_name(self, node):
        if not isinstance(node, FunctionCallNode):
            return None

        func_expr = getattr(node, "function", None)
        if func_expr is None:
            func_expr = getattr(node, "name", None)
        if hasattr(func_expr, "name") and getattr(func_expr, "name", None):
            return func_expr.name
        if isinstance(func_expr, str):
            return func_expr
        return None

    def generate_slang_synchronization_statement(self, node):
        callee = self.function_call_simple_callee_name(node)
        call = self.generate_slang_synchronization_call(
            callee, getattr(node, "args", []), statement_context=True
        )
        if call is None:
            return None
        return f"{call};"

    def generate_slang_synchronization_call(
        self, callee, args, statement_context=False
    ):
        if not isinstance(callee, str) or callee in self.user_function_names:
            return None

        intrinsic = self.slang_synchronization_intrinsic_name(callee)
        if intrinsic is None:
            return None
        if args:
            raise ValueError(
                f"Slang synchronization builtin '{callee}' requires 0 "
                f"argument(s), got {len(args)}"
            )
        if not statement_context:
            raise ValueError(
                f"Slang synchronization builtin '{callee}' is statement-only "
                "and cannot be used as a value"
            )
        return f"{intrinsic}()"

    def slang_synchronization_intrinsic_name(self, callee):
        return {
            "barrier": "GroupMemoryBarrierWithGroupSync",
            "workgroupBarrier": "GroupMemoryBarrierWithGroupSync",
            "groupMemoryBarrier": "GroupMemoryBarrier",
            "memoryBarrierShared": "GroupMemoryBarrier",
            "memoryBarrierBuffer": "DeviceMemoryBarrier",
            "deviceMemoryBarrier": "DeviceMemoryBarrier",
            "memoryBarrierImage": "DeviceMemoryBarrier",
            "memoryBarrier": "AllMemoryBarrier",
            "allMemoryBarrier": "AllMemoryBarrier",
        }.get(callee)

    def generate_slang_ray_tracing_op_expression(self, node):
        args = [self.generate_expression(arg) for arg in node.arguments]
        if node.operation == "TraceRay" and len(args) == 11:
            ray_desc = f"RayDesc({args[6]}, {args[7]}, {args[8]}, {args[9]})"
            args = args[:6] + [ray_desc, args[10]]
        return f"{node.operation}({', '.join(args)})"

    def generate_slang_wave_op_expression(self, node):
        expected_arity = self.SLANG_WAVE_INTRINSIC_ARITIES.get(node.operation)
        if expected_arity is None:
            return self.unsupported_slang_wave_op_expression(
                node.operation, "is not recognized by the Slang backend"
            )

        actual_arity = len(node.arguments)
        rejection_reason = None
        if actual_arity != expected_arity:
            rejection_reason = f"expects {expected_arity} arguments, got {actual_arity}"
        else:
            rejection_reason = self.slang_wave_argument_rejection_reason(
                node.operation, node.arguments
            )
        if rejection_reason is not None:
            return self.unsupported_slang_wave_op_expression(
                node.operation, rejection_reason
            )
        result_type = self.slang_wave_result_type(node)
        target_reason = self.slang_wave_target_type_rejection_reason(result_type)
        if target_reason is not None:
            return self.unsupported_slang_wave_op_expression(
                node.operation, target_reason
            )

        args = ", ".join(self.generate_expression(arg) for arg in node.arguments)
        return f"{node.operation}({args})"

    def slang_wave_result_type(self, node):
        operation = getattr(node, "operation", None)
        args = getattr(node, "arguments", []) or []
        if operation in {"WaveGetLaneCount", "WaveGetLaneIndex"}:
            return "uint"
        if operation in {"WaveIsFirstLane", "WaveActiveAllTrue", "WaveActiveAnyTrue"}:
            return "bool"
        if operation in {"WaveActiveBallot", "WaveMatch"}:
            return "uint4"
        if operation in {
            "WaveActiveSum",
            "WaveActiveProduct",
            "WaveActiveBitAnd",
            "WaveActiveBitOr",
            "WaveActiveBitXor",
            "WaveActiveMin",
            "WaveActiveMax",
            "WaveReadLaneAt",
            "WaveReadLaneFirst",
            "WavePrefixSum",
            "WavePrefixProduct",
            "QuadReadAcrossX",
            "QuadReadAcrossY",
            "QuadReadAcrossDiagonal",
            "QuadReadLaneAt",
            "WaveMultiPrefixSum",
            "WaveMultiPrefixProduct",
            "WaveMultiPrefixBitAnd",
            "WaveMultiPrefixBitOr",
            "WaveMultiPrefixBitXor",
        }:
            return self.expression_result_type(args[0]) if args else None
        return None

    def slang_wave_argument_rejection_reason(self, operation, args):
        integer_bit_ops = {
            "WaveActiveBitAnd",
            "WaveActiveBitOr",
            "WaveActiveBitXor",
            "WaveMultiPrefixBitAnd",
            "WaveMultiPrefixBitOr",
            "WaveMultiPrefixBitXor",
        }
        if operation in integer_bit_ops:
            value_type = self.slang_wave_argument_mapped_type(args[0])
            if value_type is not None and not self.is_slang_wave_integer_value_type(
                value_type
            ):
                return (
                    "value must be scalar or vector int or uint, " f"got {value_type}"
                )

        if operation in {
            "WaveActiveAllTrue",
            "WaveActiveAnyTrue",
            "WaveActiveBallot",
        }:
            predicate_type = self.slang_wave_argument_mapped_type(args[0])
            if predicate_type is not None and predicate_type != "bool":
                return f"predicate must be scalar bool, got {predicate_type}"

        if operation in {"WaveReadLaneAt", "QuadReadLaneAt"}:
            lane_type = self.slang_wave_argument_mapped_type(args[1])
            if lane_type is not None and lane_type not in {"int", "uint"}:
                return f"lane index must be scalar int or uint, got {lane_type}"

        if operation.startswith("WaveMultiPrefix"):
            mask_type = self.slang_wave_argument_mapped_type(args[1])
            if mask_type is not None and mask_type != "uint4":
                return f"partition mask must be uint4, got {mask_type}"

        return None

    def slang_wave_argument_mapped_type(self, arg):
        arg_type = self.expression_result_type(arg)
        if arg_type is None:
            return None
        return self.convert_type(arg_type)

    def slang_wave_target_type_rejection_reason(self, result_type):
        expected_type = self.slang_wave_expected_target_type()
        result_type = self.type_name_string(result_type)
        if result_type:
            result_type = self.convert_type(result_type)
        if not expected_type or not result_type:
            return None
        if not self.is_slang_wave_value_type(result_type):
            return None
        if expected_type == result_type:
            return None
        return f"returns {result_type} but target expects {expected_type}"

    def slang_wave_expected_target_type(self):
        expected_type = self.type_name_string(self.current_expression_expected_type)
        if not expected_type:
            return None
        expected_type = self.convert_type(expected_type)
        if expected_type == "auto":
            return None
        if self.is_slang_wave_value_type(expected_type):
            return expected_type
        return None

    def is_slang_wave_value_type(self, type_name):
        return self.is_scalar_value_type(type_name) or self.is_vector_value_type(
            type_name
        )

    def is_slang_wave_integer_value_type(self, type_name):
        return type_name in {
            "int",
            "uint",
            "int2",
            "int3",
            "int4",
            "uint2",
            "uint3",
            "uint4",
        }

    def unsupported_slang_wave_op_expression(self, operation, reason):
        return (
            f"/* unsupported Slang wave intrinsic: {operation} {reason} */ "
            f"{self.slang_wave_fallback_value(operation)}"
        )

    def slang_wave_fallback_value(self, operation):
        default_value = self.slang_wave_default_value(operation)
        expected_type = self.slang_wave_expected_target_type()
        if expected_type and not self.slang_wave_default_matches_expected_type(
            default_value, expected_type
        ):
            return self.zero_value_for_type(expected_type)
        return default_value

    def slang_wave_default_matches_expected_type(self, default_value, expected_type):
        if self.is_vector_value_type(expected_type):
            return default_value.startswith(f"{expected_type}(")
        if expected_type == "bool":
            return default_value == "false"
        if expected_type in {"int", "uint", "float", "double"}:
            return default_value in {"0", "0u"}
        return False

    def slang_wave_default_value(self, operation):
        if operation in {"WaveIsFirstLane", "WaveActiveAllTrue", "WaveActiveAnyTrue"}:
            return "false"
        if operation in {"WaveActiveBallot", "WaveMatch"}:
            return "uint4(0)"
        return "0"

    def generate_slang_mesh_op_expression(self, node):
        expected_arities = self.SLANG_MESH_INTRINSIC_ARITIES.get(node.operation)
        if expected_arities is None:
            return self.unsupported_slang_mesh_op_expression(
                node.operation, "is not recognized by the Slang backend"
            )

        actual_arity = len(node.arguments)
        rejection_reason = None
        if actual_arity not in expected_arities:
            expected = " or ".join(str(arity) for arity in sorted(expected_arities))
            rejection_reason = f"expects {expected} arguments, got {actual_arity}"
        else:
            self.validate_slang_mesh_op_stage(node.operation)
            rejection_reason = self.slang_mesh_argument_rejection_reason(
                node.operation, node.arguments
            )
            if rejection_reason is None:
                self.validate_slang_dispatch_mesh_payload_argument(node)
                args = ", ".join(
                    self.generate_expression(arg) for arg in node.arguments
                )
                return f"{node.operation}({args})"

        if rejection_reason is not None:
            return self.unsupported_slang_mesh_op_expression(
                node.operation, rejection_reason
            )

    def validate_slang_mesh_op_stage_for_shader(self, operation, shader_type):
        if not shader_type:
            return

        shader_stage = self.slang_shader_stage_name(shader_type)
        if operation == "SetMeshOutputCounts" and shader_stage != "mesh":
            raise ValueError(
                f"Slang {shader_type} stage cannot call "
                "SetMeshOutputCounts; SetMeshOutputCounts is only valid in mesh stages"
            )
        if operation == "DispatchMesh" and shader_stage != "amplification":
            raise ValueError(
                f"Slang {shader_type} stage cannot call DispatchMesh; "
                "DispatchMesh is only valid in amplification/task/object stages"
            )

    def validate_slang_mesh_op_stage(self, operation):
        self.validate_slang_mesh_op_stage_for_shader(
            operation, self.current_shader_type
        )

    def slang_mesh_intrinsic_calls(self, func):
        calls = []
        for node in self.walk_ast(getattr(func, "body", [])):
            if isinstance(node, MeshOpNode):
                calls.append(
                    (
                        getattr(node, "operation", None),
                        getattr(node, "arguments", []),
                    )
                )
                continue
            call_name = self.slang_function_call_name(node)
            if call_name in self.SLANG_MESH_INTRINSIC_ARITIES:
                calls.append(
                    (call_name, getattr(node, "arguments", getattr(node, "args", [])))
                )
        return calls

    def validate_slang_mesh_intrinsic_calls(
        self, func, shader_type, visited_helpers=None
    ):
        if visited_helpers is None:
            visited_helpers = set()

        for operation, args in self.slang_mesh_intrinsic_calls(func):
            if operation in self.SLANG_MESH_INTRINSIC_ARITIES:
                expected_arities = self.SLANG_MESH_INTRINSIC_ARITIES[operation]
                if len(args) not in expected_arities:
                    continue
                self.validate_slang_mesh_op_stage_for_shader(operation, shader_type)

        for helper_call in self.slang_user_function_call_nodes(func):
            helper_name = self.slang_function_call_name(helper_call)
            if helper_name in visited_helpers:
                continue
            helper_func = self.user_functions_by_name.get(helper_name)
            if helper_func is None or helper_func is func:
                continue
            self.validate_slang_mesh_intrinsic_calls(
                helper_func, shader_type, visited_helpers | {helper_name}
            )

    def validate_slang_dispatch_mesh_payload_argument(self, node):
        if node.operation != "DispatchMesh" or len(node.arguments) != 4:
            return

        payload_types = sorted(self.slang_mesh_payload_parameter_types)
        if not payload_types:
            raise ValueError(
                "Slang DispatchMesh payload argument requires a mesh stage "
                "@mesh_payload parameter"
            )

        payload_argument = node.arguments[3]
        payload_type = self.slang_dispatch_mesh_payload_argument_type(payload_argument)
        if payload_type is None:
            return

        if payload_type in payload_types:
            if self.slang_expression_is_lvalue(payload_argument):
                return
            raise ValueError(
                "Slang DispatchMesh payload argument must be an lvalue, "
                f"got {payload_type}"
            )

        expected_label = self.slang_dispatch_mesh_payload_type_label(payload_types)
        raise ValueError(
            f"Slang DispatchMesh payload argument type {payload_type} must match "
            f"mesh payload type {expected_label}"
        )

    def slang_dispatch_mesh_payload_argument_type(self, expr):
        type_name = self.type_name_string(self.expression_result_type(expr))
        if not type_name:
            return None

        base_type, array_suffix = split_array_type_suffix(type_name)
        mapped_type = self.convert_type(base_type)
        if array_suffix:
            return f"{mapped_type}{array_suffix}"
        return mapped_type

    def slang_dispatch_mesh_payload_type_label(self, payload_types):
        if len(payload_types) == 1:
            return payload_types[0]
        return " or ".join(payload_types)

    def slang_mesh_argument_rejection_reason(self, operation, args):
        if operation == "SetMeshOutputCounts":
            for index, label in enumerate(("vertex count", "primitive count")):
                count_type = self.slang_mesh_argument_mapped_type(args[index])
                if count_type is not None and count_type not in {"int", "uint"}:
                    return f"{label} must be scalar int or uint, got {count_type}"

        if operation == "DispatchMesh":
            for index, label in enumerate(("x count", "y count", "z count")):
                count_type = self.slang_mesh_argument_mapped_type(args[index])
                if count_type is not None and count_type not in {"int", "uint"}:
                    return f"{label} must be scalar int or uint, got {count_type}"
        return None

    def slang_mesh_argument_mapped_type(self, arg):
        arg_type = self.expression_result_type(arg)
        if arg_type is None:
            return None
        return self.convert_type(arg_type)

    def unsupported_slang_mesh_op_expression(self, operation, reason):
        return f"/* unsupported Slang mesh intrinsic: {operation} {reason} */ 0"

    def generate_lambda_expression(self, args):
        """Render supported CrossGL pseudo-lambdas as Slang lambda expressions."""
        if not args:
            return None

        params = []
        for arg in args[:-1]:
            param = self.generate_lambda_parameter(arg)
            if param is None:
                return None
            params.append(param)

        body = self.generate_lambda_body(args[-1])
        if body is None:
            return None
        return f"({', '.join(params)}) => {body}"

    def generate_lambda_parameter(self, arg):
        raw = self.lambda_raw_argument_text(arg).strip()
        typed_param = self.split_lambda_typed_parameter(raw)
        if typed_param is None:
            return None

        type_name, param_name = typed_param
        mapped_type = self.lambda_parameter_type(type_name)
        if mapped_type is None:
            return None
        return f"{mapped_type} {param_name}"

    def generate_lambda_body(self, arg):
        raw = self.lambda_raw_argument_text(arg).strip()
        if not raw:
            return None
        return raw

    def lambda_raw_argument_text(self, arg):
        if isinstance(arg, IdentifierNode):
            return arg.name
        if isinstance(arg, str):
            return arg
        return self.generate_expression(arg)

    def split_lambda_typed_parameter(self, raw):
        if not raw or ":" in raw:
            return None
        if any(char in raw for char in "{}()"):
            return None

        parts = raw.rsplit(None, 1)
        if len(parts) != 2:
            return None

        type_name, param_name = parts
        if not param_name.isidentifier():
            return None
        return type_name, param_name

    def lambda_parameter_type(self, type_name):
        canonical_type = (
            "".join(type_name.split())
            if "<" in type_name or ">" in type_name
            else type_name
        )
        if any(char.isspace() for char in canonical_type):
            return None
        if any(char in canonical_type for char in "{},;[]()"):
            return None

        mapped_type = self.convert_type(canonical_type)
        if "<" in canonical_type or ">" in canonical_type:
            if mapped_type == canonical_type:
                return None

        if any(char in mapped_type for char in "<>{},;[]()"):
            return None
        if not mapped_type:
            return None
        if not (mapped_type[0].isalpha() or mapped_type[0] == "_"):
            return None
        if not all(char.isalnum() or char == "_" for char in mapped_type):
            return None
        return mapped_type

    def format_array_access_index(self, index):
        if isinstance(index, BinaryOpNode):
            return self.format_array_size_expression(index)
        return self.generate_expression(index)

    def generate_if(self, node):
        prelude, _results, condition = self.generate_expression_with_prelude(
            getattr(node, "condition", getattr(node, "if_condition", None))
        )
        result = f"if ({condition})\n{{\n"

        self.indent_level += 1
        for stmt in self.get_statements(getattr(node, "if_body", [])):
            result += self.emit_statement(stmt) + "\n"
        self.indent_level -= 1

        result += self.indent() + "}"

        else_body = getattr(node, "else_body", None)
        if else_body:
            result += "\nelse\n{\n"
            self.indent_level += 1
            for stmt in self.get_statements(else_body):
                result += self.emit_statement(stmt) + "\n"
            self.indent_level -= 1
            result += self.indent() + "}"

        return self.statement_with_prelude(prelude, result)

    def generate_for(self, node):
        init = (
            self.generate_for_header_statement(
                node.init, "for-loop initializer context"
            )
            if node.init
            else ""
        )
        condition = (
            self.generate_loop_condition_expression(
                node.condition, "for-loop condition context"
            )
            if node.condition
            else ""
        )
        update = (
            self.generate_for_header_statement(node.update, "for-loop update context")
            if node.update
            else ""
        )

        result = f"for ({init}; {condition}; {update})\n{{\n"

        self.indent_level += 1
        for stmt in self.get_statements(node.body):
            result += self.emit_statement(stmt) + "\n"
        self.indent_level -= 1

        result += self.indent() + "}"
        return result

    def generate_for_in(self, node):
        pattern = getattr(node, "pattern", "item")
        iterable = getattr(node, "iterable", None)

        if isinstance(iterable, RangeNode):
            start = self.generate_expression(iterable.start)
            end = self.generate_expression(iterable.end)
            comparator = "<=" if iterable.inclusive else "<"
        else:
            start = "0"
            end = self.generate_expression(iterable)
            comparator = "<"

        result = (
            f"for (int {pattern} = {start}; "
            f"{pattern} {comparator} {end}; ++{pattern})\n{{\n"
        )

        self.indent_level += 1
        for stmt in self.get_statements(node.body):
            result += self.emit_statement(stmt) + "\n"
        self.indent_level -= 1

        result += self.indent() + "}"
        return result

    def generate_while(self, node):
        condition = self.generate_loop_condition_expression(
            node.condition, "while-loop condition context"
        )
        result = f"while ({condition})\n{{\n"

        self.indent_level += 1
        for stmt in self.get_statements(node.body):
            result += self.emit_statement(stmt) + "\n"
        self.indent_level -= 1

        result += self.indent() + "}"
        return result

    def generate_do_while(self, node):
        condition = self.generate_loop_condition_expression(
            node.condition, "do-while-loop condition context"
        )
        result = "do\n{\n"

        self.indent_level += 1
        for stmt in self.get_statements(node.body):
            result += self.emit_statement(stmt) + "\n"
        self.indent_level -= 1

        result += self.indent() + f"}} while ({condition});"
        return result

    def generate_switch(self, node):
        prelude, _results, expression = self.generate_expression_with_prelude(
            node.expression
        )
        result = f"switch ({expression})\n{{\n"

        self.indent_level += 1
        for case in getattr(node, "cases", []):
            if not isinstance(case, CaseNode):
                continue

            if case.value is None:
                result += self.indent() + "default:\n"
            else:
                case_value = self.generate_expression(case.value)
                result += self.indent() + f"case {case_value}:\n"

            self.indent_level += 1
            for stmt in self.get_statements(case.statements):
                result += self.emit_statement(stmt) + "\n"
            self.indent_level -= 1
        self.indent_level -= 1

        result += self.indent() + "}"
        return self.statement_with_prelude(prelude, result)

    def generate_match(self, node):
        prelude, _results, expression = self.generate_expression_with_prelude(
            getattr(node, "expression", None)
        )
        result = f"switch ({expression})\n{{\n"

        arms = getattr(node, "arms", []) or []
        rejection_reason = self.match_arm_rejection_reason(arms)
        if rejection_reason is not None:
            raise ValueError(
                f"Unsupported match arm for Slang codegen; {rejection_reason}"
            )

        wildcard_body = None
        self.indent_level += 1
        for arm in arms:
            pattern = getattr(arm, "pattern", None)
            if isinstance(pattern, WildcardPatternNode):
                wildcard_body = getattr(arm, "body", [])
                continue

            result += (
                self.indent() + f"case {self.generate_expression(pattern.literal)}:\n"
            )
            self.indent_level += 1
            body = getattr(arm, "body", [])
            for stmt in self.get_statements(body):
                result += self.emit_statement(stmt) + "\n"
            if not self.statement_body_terminates(body):
                result += self.indent() + "break;\n"
            self.indent_level -= 1

        if wildcard_body is not None:
            result += self.indent() + "default:\n"
            self.indent_level += 1
            for stmt in self.get_statements(wildcard_body):
                result += self.emit_statement(stmt) + "\n"
            if not self.statement_body_terminates(wildcard_body):
                result += self.indent() + "break;\n"
            self.indent_level -= 1

        self.indent_level -= 1

        result += self.indent() + "}"
        return self.statement_with_prelude(prelude, result)

    def generate_match_expression(self, node):
        expected_type = self.type_name_string(self.current_expression_expected_type)
        if not expected_type:
            expected_type = self.expression_result_type(node)
        if not expected_type or expected_type in {"auto", "void"}:
            return "/* unsupported Slang match expression: requires typed result context */ 0"

        if not self.expression_prelude_active():
            return (
                "/* unsupported Slang match expression: requires statement "
                f"prelude context */ {self.zero_value_for_type(expected_type)}"
            )

        arms = getattr(node, "arms", []) or []
        rejection_reason = self.match_arm_rejection_reason(arms)
        if rejection_reason is not None:
            return (
                f"/* unsupported Slang match expression: {rejection_reason} */ "
                f"{self.zero_value_for_type(expected_type)}"
            )

        temp_name = self.unique_expression_temp_name("cgl_match_value")
        self.register_variable_type(temp_name, expected_type)
        expression = self.generate_expression(getattr(node, "expression", None))
        lines = [
            f"{self.format_declaration(expected_type, temp_name)};",
            f"switch ({expression})",
            "{",
        ]

        wildcard_body = None
        self.indent_level += 1
        try:
            for arm in arms:
                pattern = getattr(arm, "pattern", None)
                if isinstance(pattern, WildcardPatternNode):
                    wildcard_body = getattr(arm, "body", [])
                    continue

                lines.append(
                    self.indent() + f"case {self.generate_expression(pattern.literal)}:"
                )
                self.indent_level += 1
                try:
                    body = getattr(arm, "body", [])
                    lines.extend(
                        self.generate_match_expression_arm_lines(
                            body, temp_name, expected_type
                        )
                    )
                    if not self.match_expression_arm_terminates(body):
                        lines.append(self.indent() + "break;")
                finally:
                    self.indent_level -= 1

            if wildcard_body is not None:
                lines.append(self.indent() + "default:")
                self.indent_level += 1
                try:
                    lines.extend(
                        self.generate_match_expression_arm_lines(
                            wildcard_body, temp_name, expected_type
                        )
                    )
                    if not self.match_expression_arm_terminates(wildcard_body):
                        lines.append(self.indent() + "break;")
                finally:
                    self.indent_level -= 1
            else:
                lines.append(self.indent() + "default:")
                self.indent_level += 1
                try:
                    lines.extend(
                        self.generate_match_expression_diagnostic_assignment_lines(
                            temp_name,
                            expected_type,
                            "no wildcard arm handles remaining cases",
                        )
                    )
                    lines.append(self.indent() + "break;")
                finally:
                    self.indent_level -= 1
        finally:
            self.indent_level -= 1

        lines.append("}")
        self.add_expression_prelude(lines, result_name=temp_name)
        return temp_name

    def generate_match_expression_arm_lines(self, body, temp_name, expected_type):
        prefix, tail = self.match_expression_tail(body)
        lines = []
        for stmt in prefix:
            lines.extend(self.emit_statement(stmt).splitlines())
        if tail is None:
            lines.extend(
                self.generate_match_expression_diagnostic_assignment_lines(
                    temp_name, expected_type, "arm does not produce a value"
                )
            )
            return lines

        prelude, _results, value = self.generate_expression_with_prelude(
            tail, expected_type
        )
        statement = self.statement_with_prelude(prelude, f"{temp_name} = {value};")
        lines.extend(self.indent_statement_text(statement).splitlines())
        return lines

    def generate_match_expression_diagnostic_assignment_lines(
        self, temp_name, expected_type, reason
    ):
        return [
            self.indent()
            + f"{temp_name} = /* unsupported Slang match expression: {reason} */ "
            f"{self.zero_value_for_type(expected_type)};"
        ]

    def match_expression_tail(self, body):
        if body is None:
            return [], None

        if isinstance(body, ExpressionStatementNode):
            if getattr(body, "is_tail_expression", False):
                return [], body.expression
            return [body], None

        if hasattr(body, "statements"):
            statements = list(body.statements)
            if not statements:
                return [], None

            tail = statements[-1]
            if isinstance(tail, ExpressionStatementNode) and getattr(
                tail, "is_tail_expression", False
            ):
                return statements[:-1], tail.expression
            return statements, None

        if isinstance(body, list):
            statements = body
        else:
            statements = [body]

        if not statements:
            return [], None

        tail = statements[-1]
        if isinstance(tail, ExpressionStatementNode) and getattr(
            tail, "is_tail_expression", False
        ):
            return statements[:-1], tail.expression
        if self.is_match_expression_value_node(tail):
            return statements[:-1], tail
        return statements, None

    def is_match_expression_value_node(self, node):
        return isinstance(
            node,
            (
                ArrayAccessNode,
                ArrayLiteralNode,
                AssignmentNode,
                AtomicOpNode,
                BinaryOpNode,
                FunctionCallNode,
                IdentifierNode,
                LiteralNode,
                MatchNode,
                MemberAccessNode,
                MeshOpNode,
                RayQueryOpNode,
                RayTracingOpNode,
                TernaryOpNode,
                UnaryOpNode,
                WaveOpNode,
            ),
        ) or isinstance(node, str)

    def match_expression_arm_terminates(self, body):
        statements = self.get_statements(body)
        if not statements:
            return False
        return isinstance(statements[-1], (BreakNode, ContinueNode, ReturnNode))

    def is_supported_match_arm(self, arm):
        if getattr(arm, "guard", None) is not None:
            return False
        pattern = getattr(arm, "pattern", None)
        return isinstance(pattern, (LiteralPatternNode, WildcardPatternNode))

    def match_arm_rejection_reason(self, arms):
        wildcard_index = None
        for index, arm in enumerate(arms):
            if getattr(arm, "guard", None) is not None:
                return "guarded arms cannot be lowered to switch"

            pattern = getattr(arm, "pattern", None)
            if self.is_range_style_match_pattern(pattern):
                return "range-style patterns cannot be lowered to switch"

            if isinstance(pattern, WildcardPatternNode):
                if wildcard_index is not None:
                    return "multiple wildcard arms cannot be lowered to switch"
                wildcard_index = index
                continue

            if isinstance(pattern, LiteralPatternNode):
                continue

            if isinstance(pattern, IdentifierPatternNode):
                return "identifier binding patterns cannot be lowered to switch"

            if isinstance(pattern, ConstructorPatternNode):
                return "constructor patterns cannot be lowered to switch"

            if isinstance(pattern, StructPatternNode):
                return "struct destructuring patterns cannot be lowered to switch"

            return (
                "only unguarded literal patterns and a final wildcard can be "
                "lowered to switch"
            )

        if wildcard_index is not None and wildcard_index != len(arms) - 1:
            return "wildcard arm must be final"
        return None

    def pointer_pointee_type_name(self, vtype):
        if isinstance(vtype, PointerType):
            return self.type_name_string(vtype.pointee_type)
        type_name = self.type_name_string(vtype)
        if not type_name:
            return None
        type_name = str(type_name).strip()
        return type_name[:-1].strip() if type_name.endswith("*") else None

    def reference_referent_type_name(self, vtype):
        if isinstance(vtype, ReferenceType):
            return self.type_name_string(vtype.referenced_type)
        type_name = self.type_name_string(vtype)
        if not type_name:
            return None
        type_name = str(type_name).strip()
        return type_name[:-1].strip() if type_name.endswith("&") else None

    def member_lookup_type_name(self, vtype):
        return (
            self.pointer_pointee_type_name(vtype)
            or self.reference_referent_type_name(vtype)
            or vtype
        )

    def is_range_style_match_pattern(self, pattern):
        if isinstance(pattern, RangeNode):
            return True
        return isinstance(pattern, IdentifierPatternNode) and pattern.name == ".."

    def validate_match_arms(self, arms):
        return self.match_arm_rejection_reason(arms) is None

    def statement_body_terminates(self, body):
        statements = self.get_statements(body)
        if not statements:
            return False
        tail = statements[-1]
        return isinstance(tail, (BreakNode, ContinueNode, ReturnNode)) or (
            self.tail_expression_returns(tail)
        )

    def get_statements(self, body):
        if body is None:
            return []
        if hasattr(body, "statements"):
            return body.statements
        if isinstance(body, list):
            return body
        return [body]

    def convert_type(self, type_name):
        """Map a CrossGL type name or type node to a Slang type string."""
        type_name = self.type_name_string(type_name)
        if type_name is None:
            return None

        for suffix in ("*", "&"):
            if type_name.endswith(suffix):
                base_type = type_name[: -len(suffix)].strip()
                return f"{self.convert_type(base_type)}{suffix}"

        base_type, array_suffix = split_array_type_suffix(type_name)
        if array_suffix:
            mapped_base_type = self.convert_type(base_type)
            return f"{mapped_base_type}{array_suffix}"

        # Map CrossGL types to Slang types
        type_map = {
            "vec2<f32>": "float2",
            "vec3<f32>": "float3",
            "vec4<f32>": "float4",
            "vec2<f64>": "double2",
            "vec3<f64>": "double3",
            "vec4<f64>": "double4",
            "vec2<i32>": "int2",
            "vec3<i32>": "int3",
            "vec4<i32>": "int4",
            "vec2<u32>": "uint2",
            "vec3<u32>": "uint3",
            "vec4<u32>": "uint4",
            "vec2<bool>": "bool2",
            "vec3<bool>": "bool3",
            "vec4<bool>": "bool4",
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "dvec2": "double2",
            "dvec3": "double3",
            "dvec4": "double4",
            "bvec2": "bool2",
            "bvec3": "bool3",
            "bvec4": "bool4",
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "mat2x2": "float2x2",
            "mat2x3": "float2x3",
            "mat2x4": "float2x4",
            "mat3x2": "float3x2",
            "mat3x3": "float3x3",
            "mat3x4": "float3x4",
            "mat4x2": "float4x2",
            "mat4x3": "float4x3",
            "mat4x4": "float4x4",
            "dmat2": "double2x2",
            "dmat3": "double3x3",
            "dmat4": "double4x4",
            "dmat2x2": "double2x2",
            "dmat2x3": "double2x3",
            "dmat2x4": "double2x4",
            "dmat3x2": "double3x2",
            "dmat3x3": "double3x3",
            "dmat3x4": "double3x4",
            "dmat4x2": "double4x2",
            "dmat4x3": "double4x3",
            "dmat4x4": "double4x4",
            "float": "float",
            "int": "int",
            "uint": "uint",
            "bool": "bool",
            "void": "void",
            "sampler": "SamplerState",
            "sampler1D": "Sampler1D<float4>",
            "sampler1DArray": "Sampler1DArray<float4>",
            "sampler2D": "Sampler2D<float4>",
            "sampler3D": "Sampler3D<float4>",
            "samplerCube": "SamplerCube<float4>",
            "sampler2DArray": "Sampler2DArray<float4>",
            "samplerCubeArray": "SamplerCubeArray<float4>",
            "sampler2DMS": "Sampler2DMS<float4>",
            "sampler2DMSArray": "Sampler2DMSArray<float4>",
            "sampler2DShadow": "Sampler2DShadow",
            "sampler2DArrayShadow": "Sampler2DArrayShadow",
            "samplerCubeShadow": "SamplerCubeShadow",
            "samplerCubeArrayShadow": "SamplerCubeArrayShadow",
            "iimage1D": "RWTexture1D<int>",
            "iimage1DArray": "RWTexture1DArray<int>",
            "iimage2D": "RWTexture2D<int>",
            "iimage3D": "RWTexture3D<int>",
            "iimage2DArray": "RWTexture2DArray<int>",
            "iimage2DMS": "RWTexture2DMS<int>",
            "iimage2DMSArray": "RWTexture2DMSArray<int>",
            "uimage1D": "RWTexture1D<uint>",
            "uimage1DArray": "RWTexture1DArray<uint>",
            "uimage2D": "RWTexture2D<uint>",
            "uimage3D": "RWTexture3D<uint>",
            "uimage2DArray": "RWTexture2DArray<uint>",
            "uimage2DMS": "RWTexture2DMS<uint>",
            "uimage2DMSArray": "RWTexture2DMSArray<uint>",
            "image1D": "RWTexture1D<float4>",
            "image1DArray": "RWTexture1DArray<float4>",
            "image2D": "RWTexture2D<float4>",
            "image3D": "RWTexture3D<float4>",
            "image2DArray": "RWTexture2DArray<float4>",
            "image2DMS": "RWTexture2DMS<float4>",
            "image2DMSArray": "RWTexture2DMSArray<float4>",
            "accelerationStructureEXT": "RaytracingAccelerationStructure",
            "AccelerationStructure": "RaytracingAccelerationStructure",
            "acceleration_structure": "RaytracingAccelerationStructure",
            "RaytracingAccelerationStructure": "RaytracingAccelerationStructure",
            "RayDesc": "RayDesc",
            "RayQuery": "RayQuery",
        }

        mapped_type = type_map.get(type_name)
        if mapped_type is not None:
            return mapped_type

        generic_resource_type = self.map_slang_generic_resource_type(type_name)
        if generic_resource_type is not None:
            return generic_resource_type

        return type_name

    def map_slang_generic_resource_type(self, type_name):
        """Map generic resource element aliases while preserving resource spelling."""
        if not isinstance(type_name, str):
            return None

        base_type = self.resource_base_type(type_name)
        generic_resource_types = {
            "StructuredBuffer",
            "RWStructuredBuffer",
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
        }
        for resource_type in generic_resource_types:
            prefix = f"{resource_type}<"
            if not base_type.startswith(prefix) or not base_type.endswith(">"):
                continue
            element_type = base_type[len(prefix) : -1].strip()
            if not element_type:
                return None
            mapped_element_type = self.convert_type(element_type)
            return type_name.replace(
                base_type, f"{resource_type}<{mapped_element_type}>", 1
            )
        return None

    def supported_image_formats(self):
        return {
            "r8",
            "r8_snorm",
            "r8i",
            "r8ui",
            "r16",
            "r16_snorm",
            "r16f",
            "r16i",
            "r16ui",
            "r32f",
            "r32i",
            "r32ui",
            "rg8",
            "rg8_snorm",
            "rg8i",
            "rg8ui",
            "rg16",
            "rg16_snorm",
            "rg16f",
            "rg16i",
            "rg16ui",
            "rg32f",
            "rg32i",
            "rg32ui",
            "rgba8",
            "rgba8_snorm",
            "rgba8i",
            "rgba8ui",
            "rgba16",
            "rgba16_snorm",
            "rgba16f",
            "rgba16i",
            "rgba16ui",
            "rgba32f",
            "rgba32i",
            "rgba32ui",
        }

    def scalar_image_format_components(self):
        return {
            "r8": "float",
            "r8_snorm": "float",
            "r16": "float",
            "r16_snorm": "float",
            "r16f": "float",
            "r32f": "float",
            "r8i": "int",
            "r16i": "int",
            "r32i": "int",
            "r8ui": "uint",
            "r16ui": "uint",
            "r32ui": "uint",
        }

    def vector_image_format_components(self):
        return {
            "rg8": "float2",
            "rg8_snorm": "float2",
            "rg16": "float2",
            "rg16_snorm": "float2",
            "rg16f": "float2",
            "rg8i": "int2",
            "rg16i": "int2",
            "rg8ui": "uint2",
            "rg16ui": "uint2",
            "rg32f": "float2",
            "rg32i": "int2",
            "rg32ui": "uint2",
            "rgba8": "float4",
            "rgba8_snorm": "float4",
            "rgba16": "float4",
            "rgba16_snorm": "float4",
            "rgba16f": "float4",
            "rgba32f": "float4",
            "rgba8i": "int4",
            "rgba16i": "int4",
            "rgba32i": "int4",
            "rgba8ui": "uint4",
            "rgba16ui": "uint4",
            "rgba32ui": "uint4",
        }

    def attribute_value_to_string(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if hasattr(value, "name") and value.name is not None:
            return str(value.name)
        if hasattr(value, "value") and value.value is not None:
            return str(value.value).strip('"')
        return str(value)

    def explicit_image_format(self, node):
        if not hasattr(node, "attributes"):
            return None
        supported_formats = self.supported_image_formats()
        for attr in node.attributes:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            attr_name = str(attr_name).lower()
            if attr_name in supported_formats:
                return attr_name
            if attr_name == "format":
                arguments = getattr(attr, "arguments", []) or []
                if not arguments:
                    continue
                format_name = self.attribute_value_to_string(arguments[0])
                if format_name is None:
                    continue
                format_name = str(format_name).lower()
                if format_name in supported_formats:
                    return format_name
        return None

    def explicit_resource_access(self, node):
        if node is None:
            return None

        access_names = {
            "read": "readonly",
            "readonly": "readonly",
            "write": "writeonly",
            "writeonly": "writeonly",
            "read_write": "readwrite",
            "readwrite": "readwrite",
            "access::read": "readonly",
            "access::write": "writeonly",
            "access::read_write": "readwrite",
        }
        for qualifier in getattr(node, "qualifiers", []) or []:
            access = access_names.get(str(qualifier).lower())
            if access is not None:
                return access

        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name == "access":
                arguments = getattr(attr, "arguments", []) or []
                if not arguments:
                    continue
                raw_access = self.attribute_value_to_string(arguments[0])
                if raw_access is None:
                    continue
                access = access_names.get(str(raw_access).lower())
            else:
                access = access_names.get(attr_name)
            if access is not None:
                return access
        return None

    def map_resource_type_with_format(self, type_name, node=None):
        type_name = self.type_name_string(type_name)
        if type_name is None:
            return self.convert_type(type_name)

        if "[" in type_name and "]" in type_name:
            base_type, array_suffix = split_array_type_suffix(type_name)
            mapped_base = self.map_resource_base_type_with_format(base_type, node)
            return f"{mapped_base}{array_suffix}"
        return self.map_resource_base_type_with_format(type_name, node)

    def map_resource_base_type_with_format(self, type_name, node=None):
        mapped_type = self.map_image_base_type_with_format(type_name, node)
        return self.map_buffer_base_type_with_access(mapped_type, node)

    def map_image_base_type_with_format(self, type_name, node=None):
        base_type = self.resource_base_type(type_name)
        explicit_format = self.explicit_image_format(node) if node is not None else None
        component_type = self.scalar_image_format_components().get(
            explicit_format
        ) or self.vector_image_format_components().get(explicit_format)
        texture_types = {
            "image1D": "RWTexture1D",
            "iimage1D": "RWTexture1D",
            "uimage1D": "RWTexture1D",
            "image2D": "RWTexture2D",
            "iimage2D": "RWTexture2D",
            "uimage2D": "RWTexture2D",
            "image3D": "RWTexture3D",
            "iimage3D": "RWTexture3D",
            "uimage3D": "RWTexture3D",
            "image1DArray": "RWTexture1DArray",
            "iimage1DArray": "RWTexture1DArray",
            "uimage1DArray": "RWTexture1DArray",
            "image2DArray": "RWTexture2DArray",
            "iimage2DArray": "RWTexture2DArray",
            "uimage2DArray": "RWTexture2DArray",
            "image2DMS": "RWTexture2DMS",
            "iimage2DMS": "RWTexture2DMS",
            "uimage2DMS": "RWTexture2DMS",
            "image2DMSArray": "RWTexture2DMSArray",
            "iimage2DMSArray": "RWTexture2DMSArray",
            "uimage2DMSArray": "RWTexture2DMSArray",
        }
        texture_type = texture_types.get(base_type)
        if component_type and texture_type:
            return f"{texture_type}<{component_type}>"
        return self.convert_type(type_name)

    def map_buffer_base_type_with_access(self, type_name, node=None):
        access = self.explicit_resource_access(node)
        if access is None:
            return type_name

        base_type = self.resource_base_type(type_name)
        if not isinstance(base_type, str):
            return type_name

        if base_type in {"ByteAddressBuffer", "RWByteAddressBuffer"}:
            mapped_base = (
                "ByteAddressBuffer" if access == "readonly" else "RWByteAddressBuffer"
            )
            return type_name.replace(base_type, mapped_base, 1)

        structured_prefixes = ("StructuredBuffer<", "RWStructuredBuffer<")
        if not base_type.startswith(structured_prefixes) or not base_type.endswith(">"):
            return type_name

        element_type = base_type[base_type.find("<") + 1 : -1].strip()
        if not element_type:
            return type_name

        resource_type = (
            "StructuredBuffer" if access == "readonly" else "RWStructuredBuffer"
        )
        return type_name.replace(base_type, f"{resource_type}<{element_type}>", 1)

    def is_storage_image_type(self, type_name):
        base_type = self.resource_base_type(type_name)
        return isinstance(base_type, str) and base_type in {
            "image1D",
            "iimage1D",
            "uimage1D",
            "image2D",
            "iimage2D",
            "uimage2D",
            "image3D",
            "iimage3D",
            "uimage3D",
            "image1DArray",
            "iimage1DArray",
            "uimage1DArray",
            "image2DArray",
            "iimage2DArray",
            "uimage2DArray",
            "image2DMS",
            "iimage2DMS",
            "uimage2DMS",
            "image2DMSArray",
            "iimage2DMSArray",
            "uimage2DMSArray",
        }

    def image_resource_type(self, image_arg):
        image_name = self.get_expression_name(image_arg)
        if not image_name:
            return None
        return self.image_resource_types.get(image_name)

    def image_resource_access(self, image_arg):
        image_name = self.get_expression_name(image_arg)
        if not image_name:
            return None
        return self.image_resource_accesses.get(image_name)

    def image_resource_element_type(self, image_type):
        image_type = self.resource_base_type(image_type)
        if not image_type or "<" not in image_type or not image_type.endswith(">"):
            return None
        return image_type[image_type.find("<") + 1 : -1]

    def vector_size(self, type_name):
        if not isinstance(type_name, str) or not type_name[-1:].isdigit():
            return None
        size = int(type_name[-1])
        return size if size in {2, 3, 4} else None

    def vector_zero_value(self, type_name):
        if isinstance(type_name, str) and type_name.startswith("bool"):
            return "false"
        if isinstance(type_name, str) and type_name.startswith("uint"):
            return "0u"
        if isinstance(type_name, str) and type_name.startswith("int"):
            return "0"
        return "0.0"

    def unsupported_image_access_call(self, operation, reason, result_type=None):
        comment = f"/* unsupported Slang image access: {operation} {reason} */"
        if result_type is None:
            return comment
        return f"{comment} {self.zero_value_for_type(result_type)}"

    def image_load_expression(self, args):
        image_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[1])
        image_type = self.image_resource_type(args[0])
        element_type = self.image_resource_element_type(image_type)
        if self.image_resource_access(args[0]) == "writeonly":
            return self.unsupported_image_access_call(
                "imageLoad",
                "requires readable image resource",
                element_type or self.current_expression_expected_type or "uint",
            )

        if len(args) >= 3:
            sample = self.generate_expression(args[2])
            load_expr = f"{image_name}[{coord}, {sample}]"
        else:
            load_expr = f"{image_name}[{coord}]"

        if self.vector_size(element_type) and self.is_scalar_value_type(
            self.current_expression_expected_type
        ):
            return f"{load_expr}.x"
        return load_expr

    def image_store_value_expression(self, image_arg, value_arg):
        value = self.generate_expression(value_arg)
        image_type = self.image_resource_type(image_arg)
        element_type = self.image_resource_element_type(image_type)
        if not self.vector_size(element_type):
            return value
        if not self.is_scalar_value_type(self.expression_result_type(value_arg)):
            return value

        if self.vector_size(element_type) == 2:
            return f"{element_type}({value}, {self.vector_zero_value(element_type)})"
        return f"{element_type}({value})"

    def image_store_expression(self, args):
        image_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[1])
        if self.image_resource_access(args[0]) == "readonly":
            return self.unsupported_image_access_call(
                "imageStore", "requires writable image resource"
            )

        if len(args) >= 4:
            sample = self.generate_expression(args[2])
            value = self.image_store_value_expression(args[0], args[3])
            return f"{image_name}[{coord}, {sample}] = {value}"

        value = self.image_store_value_expression(args[0], args[2])
        return f"{image_name}[{coord}] = {value}"

    def structured_buffer_resource_type(self, buffer_arg):
        buffer_name = self.get_expression_name(buffer_arg)
        if buffer_name:
            tracked_type = self.buffer_resource_types.get(buffer_name)
            if tracked_type is not None:
                return tracked_type

        buffer_type = self.get_expression_type(buffer_arg)
        if buffer_type is None:
            buffer_type = self.expression_result_type(buffer_arg)
        mapped_type = self.map_resource_type_with_format(buffer_type)
        return self.resource_base_type(mapped_type)

    def buffer_resource_access(self, buffer_arg):
        buffer_name = self.get_expression_name(buffer_arg)
        if not buffer_name:
            return None
        return self.buffer_resource_accesses.get(buffer_name)

    def is_buffer_resource_type(self, buffer_type):
        return (
            self.is_byte_address_buffer_resource_type(buffer_type)
            or self.is_structured_buffer_resource_type(buffer_type)
            or self.is_append_structured_buffer_resource_type(buffer_type)
            or self.is_consume_structured_buffer_resource_type(buffer_type)
        )

    def is_byte_address_buffer_resource_type(self, buffer_type):
        buffer_type = self.resource_base_type(buffer_type)
        return buffer_type in {"ByteAddressBuffer", "RWByteAddressBuffer"}

    def is_writable_byte_address_buffer_resource_type(self, buffer_type):
        buffer_type = self.resource_base_type(buffer_type)
        return buffer_type == "RWByteAddressBuffer"

    def is_structured_buffer_resource_type(self, buffer_type):
        buffer_type = self.resource_base_type(buffer_type)
        return isinstance(buffer_type, str) and buffer_type.startswith(
            ("StructuredBuffer<", "RWStructuredBuffer<")
        )

    def is_writable_structured_buffer_resource_type(self, buffer_type):
        buffer_type = self.resource_base_type(buffer_type)
        return isinstance(buffer_type, str) and buffer_type.startswith(
            "RWStructuredBuffer<"
        )

    def is_append_structured_buffer_resource_type(self, buffer_type):
        buffer_type = self.resource_base_type(buffer_type)
        return isinstance(buffer_type, str) and buffer_type.startswith(
            "AppendStructuredBuffer<"
        )

    def is_consume_structured_buffer_resource_type(self, buffer_type):
        buffer_type = self.resource_base_type(buffer_type)
        return isinstance(buffer_type, str) and buffer_type.startswith(
            "ConsumeStructuredBuffer<"
        )

    def structured_buffer_element_type(self, buffer_type):
        buffer_type = self.resource_base_type(buffer_type)
        if (
            not isinstance(buffer_type, str)
            or "<" not in buffer_type
            or not buffer_type.endswith(">")
        ):
            return None
        element_type = buffer_type[buffer_type.find("<") + 1 : -1].strip()
        if not element_type:
            return None
        return self.convert_type(element_type)

    def structured_buffer_atomic_operations(self):
        return {
            "atomicAdd": ("InterlockedAdd", 1),
            "atomicMin": ("InterlockedMin", 1),
            "atomicMax": ("InterlockedMax", 1),
            "atomicAnd": ("InterlockedAnd", 1),
            "atomicOr": ("InterlockedOr", 1),
            "atomicXor": ("InterlockedXor", 1),
            "atomicExchange": ("InterlockedExchange", 1),
            "atomicCompSwap": ("InterlockedCompareExchange", 2),
            "atomicCompareExchange": ("InterlockedCompareExchange", 2),
        }

    def structured_buffer_atomic_target_resource_type(self, target):
        if isinstance(target, ArrayAccessNode):
            array_expr = getattr(target, "array", getattr(target, "array_expr", None))
            array_type = self.structured_buffer_resource_type(array_expr)
            if self.is_structured_buffer_resource_type(array_type):
                return array_type
            return self.structured_buffer_atomic_target_resource_type(array_expr)
        if isinstance(target, MemberAccessNode):
            object_expr = getattr(
                target, "object", getattr(target, "object_expr", None)
            )
            return self.structured_buffer_atomic_target_resource_type(object_expr)
        return self.structured_buffer_resource_type(target)

    def unique_expression_temp_name(self, base_name):
        candidate = base_name
        suffix = 1
        used_names = set(self.user_symbol_names) | set(self.expression_temp_names)
        used_names.update(name for name in self.variable_types if name)
        while candidate in used_names:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        self.expression_temp_names.add(candidate)
        return candidate

    def structured_buffer_atomic_value_expression(
        self, func_name, intrinsic, target, value_args, element_type
    ):
        if not self.expression_prelude_active():
            if self.atomic_value_context_stack:
                context = self.atomic_value_context_stack[-1]
                return self.unsupported_structured_buffer_call(
                    func_name,
                    "expression-valued atomic cannot be used in "
                    f"{context}; use explicit original output argument",
                    element_type,
                )
            return self.unsupported_structured_buffer_call(
                func_name,
                "requires target, value, and original output outside "
                "statement expression context",
                element_type,
            )

        original_name = self.unique_expression_temp_name(f"cgl_{func_name}_original")
        target_expr = self.generate_expression(target)
        value_exprs = [
            self.generate_expression_with_expected(value_arg, element_type)
            for value_arg in value_args
        ]
        self.add_expression_prelude(
            [
                f"{element_type} {original_name};",
                f"{intrinsic}({', '.join([target_expr, *value_exprs, original_name])});",
            ],
            result_name=original_name,
        )
        return original_name

    def structured_buffer_atomic_expression(self, func_name, args):
        operation = self.structured_buffer_atomic_operations().get(func_name)
        if operation is None or not args:
            return None

        diagnostic = self.unsupported_structured_buffer_call
        target = args[0]
        buffer_type = self.structured_buffer_atomic_target_resource_type(target)
        if not self.is_structured_buffer_resource_type(buffer_type):
            return None

        element_type = self.structured_buffer_element_type(buffer_type) or "uint"
        result_type = element_type if element_type in {"int", "uint"} else "uint"
        if self.buffer_resource_access(target) == "readonly":
            return diagnostic(
                func_name,
                "requires writable structured buffer resource",
                result_type,
            )
        if not self.is_writable_structured_buffer_resource_type(buffer_type):
            return diagnostic(
                func_name,
                "requires RWStructuredBuffer resource",
                result_type,
            )
        if element_type not in {"int", "uint"}:
            return diagnostic(
                func_name,
                "requires scalar int or uint RWStructuredBuffer element",
                result_type,
            )

        intrinsic, value_arg_count = operation
        expression_args = 1 + value_arg_count
        explicit_result_args = expression_args + 1
        if len(args) == expression_args:
            return self.structured_buffer_atomic_value_expression(
                func_name, intrinsic, target, args[1:], element_type
            )
        if len(args) != explicit_result_args:
            required_shape = (
                "target, compare, value, and original output"
                if value_arg_count == 2
                else "target, value, and original output"
            )
            return diagnostic(
                func_name,
                f"requires {required_shape}",
                result_type,
            )

        target_expr = self.generate_expression(target)
        value_exprs = [
            self.generate_expression_with_expected(value_arg, element_type)
            for value_arg in args[1 : 1 + value_arg_count]
        ]
        original_expr = self.generate_expression(args[-1])
        return f"{intrinsic}({', '.join([target_expr, *value_exprs, original_expr])})"

    def unsupported_structured_buffer_call(self, operation, reason, result_type=None):
        if result_type is None:
            return f"/* unsupported Slang structured buffer: {operation} {reason} */"
        return (
            f"/* unsupported Slang structured buffer: {operation} {reason} */ "
            f"{self.zero_value_for_type(result_type)}"
        )

    def unsupported_byte_address_buffer_call(self, operation, reason, result_type=None):
        if result_type is None:
            return f"/* unsupported Slang byte-address buffer: {operation} {reason} */"
        return (
            f"/* unsupported Slang byte-address buffer: {operation} {reason} */ "
            f"{self.zero_value_for_type(result_type)}"
        )

    def zero_value_for_type(self, type_name):
        type_name = self.convert_type(type_name)
        base_type, array_suffix = split_array_type_suffix(type_name)
        if array_suffix:
            return self.zero_array_value(base_type, array_suffix)
        if type_name == "bool":
            return "false"
        if type_name == "uint":
            return "0u"
        if type_name in {"int", "float", "double"}:
            return "0"
        if self.is_vector_value_type(type_name):
            component_zero = self.vector_zero_value(type_name)
            return f"{type_name}({component_zero})"
        if type_name in self.user_struct_names:
            return f"{type_name}()"
        return "0"

    def zero_array_value(self, base_type, array_suffix):
        dimensions = self.array_suffix_literal_dimensions(array_suffix)
        if not dimensions:
            return "0"
        return self.zero_array_literal(base_type, dimensions)

    def array_suffix_literal_dimensions(self, array_suffix):
        dimensions = []
        remaining = array_suffix
        while remaining.startswith("["):
            closing_bracket = remaining.find("]")
            if closing_bracket < 0:
                return []
            size = remaining[1:closing_bracket].strip()
            if not size.isdigit():
                return []
            dimensions.append(int(size))
            remaining = remaining[closing_bracket + 1 :]
        return dimensions if not remaining else []

    def zero_array_literal(self, base_type, dimensions):
        if not dimensions:
            return self.zero_value_for_type(base_type)
        element = self.zero_array_literal(base_type, dimensions[1:])
        return "{" + ", ".join(element for _ in range(dimensions[0])) + "}"

    def buffer_load_expression(self, args):
        if len(args) < 2:
            return self.unsupported_structured_buffer_call(
                "buffer_load", "requires buffer and index arguments", "uint"
            )

        buffer_type = self.structured_buffer_resource_type(args[0])
        element_type = self.structured_buffer_element_type(buffer_type) or "uint"
        if self.buffer_resource_access(args[0]) == "writeonly":
            if self.is_byte_address_buffer_resource_type(buffer_type):
                return self.unsupported_byte_address_buffer_call(
                    "buffer_load",
                    "requires readable byte-address buffer resource",
                    "uint",
                )
            return self.unsupported_structured_buffer_call(
                "buffer_load",
                "requires readable structured buffer resource",
                element_type,
            )
        if self.is_byte_address_buffer_resource_type(buffer_type):
            buffer = self.generate_expression(args[0])
            index = self.generate_expression(args[1])
            return f"{buffer}.Load({index})"

        if not self.is_structured_buffer_resource_type(buffer_type):
            return self.unsupported_structured_buffer_call(
                "buffer_load",
                "requires StructuredBuffer or RWStructuredBuffer resource",
                element_type,
            )

        buffer = self.generate_expression(args[0])
        index = self.generate_expression(args[1])
        return f"{buffer}.Load({index})"

    def buffer_store_expression(self, args):
        if len(args) < 3:
            return self.unsupported_structured_buffer_call(
                "buffer_store", "requires buffer, index, and value arguments"
            )

        buffer_type = self.structured_buffer_resource_type(args[0])
        if self.is_byte_address_buffer_resource_type(buffer_type):
            if self.buffer_resource_access(args[0]) == "readonly":
                return self.unsupported_byte_address_buffer_call(
                    "buffer_store", "requires writable byte-address buffer resource"
                )
            if not self.is_writable_byte_address_buffer_resource_type(buffer_type):
                return self.unsupported_byte_address_buffer_call(
                    "buffer_store", "requires RWByteAddressBuffer resource"
                )
            buffer = self.generate_expression(args[0])
            index = self.generate_expression(args[1])
            value = self.generate_expression(args[2])
            return f"{buffer}.Store({index}, {value})"

        if self.buffer_resource_access(args[0]) == "readonly":
            return self.unsupported_structured_buffer_call(
                "buffer_store", "requires writable structured buffer resource"
            )

        if not self.is_writable_structured_buffer_resource_type(buffer_type):
            return self.unsupported_structured_buffer_call(
                "buffer_store", "requires RWStructuredBuffer resource"
            )

        buffer = self.generate_expression(args[0])
        index = self.generate_expression(args[1])
        value = self.generate_expression(args[2])
        return f"{buffer}.Store({index}, {value})"

    def buffer_append_expression(self, args):
        if len(args) < 2:
            return self.unsupported_structured_buffer_call(
                "buffer_append", "requires buffer and value arguments"
            )

        buffer_type = self.structured_buffer_resource_type(args[0])
        if self.buffer_resource_access(args[0]) == "readonly":
            return self.unsupported_structured_buffer_call(
                "buffer_append", "requires writable structured buffer resource"
            )
        if not self.is_append_structured_buffer_resource_type(buffer_type):
            return self.unsupported_structured_buffer_call(
                "buffer_append", "requires AppendStructuredBuffer resource"
            )

        buffer = self.generate_expression(args[0])
        value = self.generate_expression(args[1])
        return f"{buffer}.Append({value})"

    def buffer_consume_expression(self, args):
        if not args:
            return self.unsupported_structured_buffer_call(
                "buffer_consume", "requires a buffer argument", "uint"
            )

        buffer_type = self.structured_buffer_resource_type(args[0])
        element_type = self.structured_buffer_element_type(buffer_type) or "uint"
        if self.buffer_resource_access(args[0]) == "writeonly":
            return self.unsupported_structured_buffer_call(
                "buffer_consume",
                "requires readable structured buffer resource",
                element_type,
            )
        if not self.is_consume_structured_buffer_resource_type(buffer_type):
            return self.unsupported_structured_buffer_call(
                "buffer_consume",
                "requires ConsumeStructuredBuffer resource",
                element_type,
            )

        buffer = self.generate_expression(args[0])
        return f"{buffer}.Consume()"

    def buffer_dimensions_expression(self, args):
        if not args:
            return self.unsupported_structured_buffer_call(
                "buffer_dimensions", "requires a buffer argument", "uint"
            )

        buffer_type = self.structured_buffer_resource_type(args[0])
        if self.is_byte_address_buffer_resource_type(buffer_type):
            buffer = self.generate_expression(args[0])
            if len(args) >= 2:
                dimensions = ", ".join(
                    self.generate_expression(arg) for arg in args[1:]
                )
                return f"{buffer}.GetDimensions({dimensions})"

            helper_name = self.buffer_dimensions_helper_name(buffer_type)
            return f"{helper_name}({buffer})"

        if not self.is_structured_buffer_resource_type(buffer_type):
            return self.unsupported_structured_buffer_call(
                "buffer_dimensions",
                "requires StructuredBuffer or RWStructuredBuffer resource",
                "uint",
            )

        buffer = self.generate_expression(args[0])
        if len(args) >= 2:
            dimensions = ", ".join(self.generate_expression(arg) for arg in args[1:])
            return f"{buffer}.GetDimensions({dimensions})"

        helper_name = self.buffer_dimensions_helper_name(buffer_type)
        return f"{helper_name}({buffer})"

    def buffer_dimensions_helper_name(self, buffer_type):
        helper_name = self.helper_function_name(
            "cgl_bufferDimensions_" f"{self.resource_helper_type_suffix(buffer_type)}"
        )
        self.register_helper_function(
            helper_name,
            self.build_buffer_dimensions_helper(helper_name, buffer_type),
        )
        return helper_name

    def build_buffer_dimensions_helper(self, helper_name, buffer_type):
        return (
            f"uint {helper_name}({buffer_type} buffer)\n"
            "{\n"
            "    uint count;\n"
            "    buffer.GetDimensions(count);\n"
            "    return count;\n"
            "}"
        )

    def byte_address_member_call_result_type(self, func_expr):
        if not isinstance(func_expr, MemberAccessNode):
            return None
        receiver = getattr(func_expr, "object", getattr(func_expr, "object_expr", None))
        if not self.is_byte_address_buffer_resource_type(
            self.structured_buffer_resource_type(receiver)
        ):
            return None
        member = str(getattr(func_expr, "member", ""))
        return {
            "Load": "uint",
            "Load2": "uint2",
            "Load3": "uint3",
            "Load4": "uint4",
        }.get(member)

    def byte_address_interlocked_members(self):
        return {
            "InterlockedAdd",
            "InterlockedAnd",
            "InterlockedCompareExchange",
            "InterlockedCompareStore",
            "InterlockedExchange",
            "InterlockedMax",
            "InterlockedMin",
            "InterlockedOr",
            "InterlockedXor",
        }

    def structured_buffer_member_call_result_type(self, func_expr):
        if not isinstance(func_expr, MemberAccessNode):
            return None
        receiver = getattr(func_expr, "object", getattr(func_expr, "object_expr", None))
        member = str(getattr(func_expr, "member", ""))
        if member != "Consume":
            return None
        receiver_type = self.structured_buffer_resource_type(receiver)
        if not (
            self.is_consume_structured_buffer_resource_type(receiver_type)
            or self.is_structured_buffer_resource_type(receiver_type)
            or self.is_append_structured_buffer_resource_type(receiver_type)
        ):
            return None
        return self.structured_buffer_element_type(receiver_type) or "uint"

    def generate_resource_member_call(self, func_expr, args):
        receiver = getattr(func_expr, "object", getattr(func_expr, "object_expr", None))
        receiver_type = self.structured_buffer_resource_type(receiver)

        member = str(getattr(func_expr, "member", ""))
        if self.is_byte_address_buffer_resource_type(receiver_type) and member in {
            "Load",
            "Load2",
            "Load3",
            "Load4",
        }:
            if self.buffer_resource_access(receiver) == "writeonly":
                result_type = self.byte_address_member_call_result_type(func_expr)
                return self.unsupported_byte_address_buffer_call(
                    member,
                    "requires readable byte-address buffer receiver",
                    result_type or "uint",
                )
            receiver_expr = self.generate_expression(receiver)
            args_expr = ", ".join(self.generate_expression(arg) for arg in args)
            return f"{receiver_expr}.{member}({args_expr})"

        if self.is_byte_address_buffer_resource_type(receiver_type) and member in {
            "Store",
            "Store2",
            "Store3",
            "Store4",
        }:
            if self.buffer_resource_access(receiver) == "readonly":
                return self.unsupported_byte_address_buffer_call(
                    member, "requires writable byte-address buffer receiver"
                )
            if not self.is_writable_byte_address_buffer_resource_type(receiver_type):
                return self.unsupported_byte_address_buffer_call(
                    member, "requires RWByteAddressBuffer receiver"
                )
            receiver_expr = self.generate_expression(receiver)
            args_expr = ", ".join(self.generate_expression(arg) for arg in args)
            return f"{receiver_expr}.{member}({args_expr})"

        if (
            self.is_byte_address_buffer_resource_type(receiver_type)
            and member in self.byte_address_interlocked_members()
        ):
            if self.buffer_resource_access(receiver) == "readonly":
                return self.unsupported_byte_address_buffer_call(
                    member, "requires writable byte-address buffer receiver"
                )
            if not self.is_writable_byte_address_buffer_resource_type(receiver_type):
                return self.unsupported_byte_address_buffer_call(
                    member, "requires RWByteAddressBuffer receiver"
                )
            receiver_expr = self.generate_expression(receiver)
            args_expr = ", ".join(self.generate_expression(arg) for arg in args)
            return f"{receiver_expr}.{member}({args_expr})"

        if member in {"Append", "Consume"} and not (
            self.is_structured_buffer_resource_type(receiver_type)
            or self.is_append_structured_buffer_resource_type(receiver_type)
            or self.is_consume_structured_buffer_resource_type(receiver_type)
        ):
            return None

        if member == "Append":
            if len(args) < 1:
                return self.unsupported_structured_buffer_call(
                    "Append", "requires a value argument"
                )
            if self.buffer_resource_access(receiver) == "readonly":
                return self.unsupported_structured_buffer_call(
                    "Append", "requires writable structured buffer receiver"
                )
            if not self.is_append_structured_buffer_resource_type(receiver_type):
                return self.unsupported_structured_buffer_call(
                    "Append", "requires AppendStructuredBuffer receiver"
                )
            receiver_expr = self.generate_expression(receiver)
            args_expr = ", ".join(self.generate_expression(arg) for arg in args)
            return f"{receiver_expr}.{member}({args_expr})"

        if member == "Consume":
            result_type = self.structured_buffer_member_call_result_type(func_expr)
            if args:
                return self.unsupported_structured_buffer_call(
                    "Consume", "requires no arguments", result_type or "uint"
                )
            if self.buffer_resource_access(receiver) == "writeonly":
                return self.unsupported_structured_buffer_call(
                    "Consume",
                    "requires readable structured buffer receiver",
                    result_type or "uint",
                )
            if not self.is_consume_structured_buffer_resource_type(receiver_type):
                return self.unsupported_structured_buffer_call(
                    "Consume",
                    "requires ConsumeStructuredBuffer receiver",
                    result_type or "uint",
                )
            receiver_expr = self.generate_expression(receiver)
            return f"{receiver_expr}.{member}()"

        return None

    def image_atomic_intrinsic(self, operation):
        return {
            "imageAtomicAdd": "InterlockedAdd",
            "imageAtomicMin": "InterlockedMin",
            "imageAtomicMax": "InterlockedMax",
            "imageAtomicAnd": "InterlockedAnd",
            "imageAtomicOr": "InterlockedOr",
            "imageAtomicXor": "InterlockedXor",
            "imageAtomicExchange": "InterlockedExchange",
            "imageAtomicCompSwap": "InterlockedCompareExchange",
        }.get(operation)

    def image_atomic_operation_from_atomic_op(self, operation):
        return {
            "imageAtomicAdd": "imageAtomicAdd",
            "imageAtomicMin": "imageAtomicMin",
            "imageAtomicMax": "imageAtomicMax",
            "imageAtomicAnd": "imageAtomicAnd",
            "imageAtomicOr": "imageAtomicOr",
            "imageAtomicXor": "imageAtomicXor",
            "imageAtomicExchange": "imageAtomicExchange",
            "imageAtomicCompSwap": "imageAtomicCompSwap",
            "imageAtomicCompareExchange": "imageAtomicCompSwap",
            "atomicAdd": "imageAtomicAdd",
            "atomicMin": "imageAtomicMin",
            "atomicMax": "imageAtomicMax",
            "atomicAnd": "imageAtomicAnd",
            "atomicOr": "imageAtomicOr",
            "atomicXor": "imageAtomicXor",
            "atomicExchange": "imageAtomicExchange",
            "atomicCompSwap": "imageAtomicCompSwap",
            "atomicCompareExchange": "imageAtomicCompSwap",
            "Add": "imageAtomicAdd",
            "Min": "imageAtomicMin",
            "Max": "imageAtomicMax",
            "And": "imageAtomicAnd",
            "Or": "imageAtomicOr",
            "Xor": "imageAtomicXor",
            "Exchange": "imageAtomicExchange",
            "CompSwap": "imageAtomicCompSwap",
            "CompareExchange": "imageAtomicCompSwap",
        }.get(operation)

    def generate_slang_atomic_op_expression(self, node):
        operation = self.image_atomic_operation_from_atomic_op(node.operation)
        if operation is None:
            return self.unsupported_image_atomic_call(
                node.operation, "is not recognized by the Slang backend"
            )

        result = self.image_atomic_expression(operation, [node.target, *node.arguments])
        if result is not None:
            return result

        return self.unsupported_image_atomic_call(
            node.operation, "requires image resource target"
        )

    def image_atomic_helper_suffix(self, image_type):
        return {
            "RWTexture1D<int>": "iimage1D",
            "RWTexture1D<uint>": "uimage1D",
            "RWTexture2D<int>": "iimage2D",
            "RWTexture2D<uint>": "uimage2D",
            "RWTexture3D<int>": "iimage3D",
            "RWTexture3D<uint>": "uimage3D",
            "RWTexture1DArray<int>": "iimage1DArray",
            "RWTexture1DArray<uint>": "uimage1DArray",
            "RWTexture2DArray<int>": "iimage2DArray",
            "RWTexture2DArray<uint>": "uimage2DArray",
        }.get(image_type)

    def image_atomic_return_type(self, image_type):
        element_type = self.image_resource_element_type(image_type)
        if element_type in {"int", "uint"}:
            return element_type
        return None

    def image_atomic_coord_type(self, image_type):
        if image_type in {"RWTexture1D<int>", "RWTexture1D<uint>"}:
            return "int"
        if image_type in {"RWTexture1DArray<int>", "RWTexture1DArray<uint>"}:
            return "int2"
        if image_type in {"RWTexture2D<int>", "RWTexture2D<uint>"}:
            return "int2"
        if image_type in {
            "RWTexture3D<int>",
            "RWTexture3D<uint>",
            "RWTexture2DArray<int>",
            "RWTexture2DArray<uint>",
        }:
            return "int3"
        return None

    def image_atomic_helper_name(self, operation, image_type):
        suffix = self.image_atomic_helper_suffix(image_type)
        if not suffix:
            return None
        return f"cgl_{operation}_{suffix}"

    def image_atomic_zero_value(self, image_type=None):
        element_type = self.image_resource_element_type(image_type)
        if isinstance(element_type, str) and element_type.startswith("uint"):
            return "0u"

        expected_type = self.convert_type(self.current_expression_expected_type)
        if expected_type == "uint":
            return "0u"
        return "0"

    def unsupported_image_atomic_call(self, operation, reason, image_type=None):
        return (
            f"/* unsupported Slang image atomic: {operation} {reason} */ "
            f"{self.image_atomic_zero_value(image_type)}"
        )

    def image_atomic_required_args_reason(self, operation):
        if operation == "imageAtomicCompSwap":
            return "requires image, coordinate, compare, and value arguments"
        return "requires image, coordinate, and value arguments"

    def image_atomic_value_expression(
        self, operation, args, image_type, intrinsic, return_type
    ):
        if not self.expression_prelude_active():
            if self.atomic_value_context_stack:
                context = self.atomic_value_context_stack[-1]
                return self.unsupported_image_atomic_call(
                    operation,
                    "expression-valued atomic cannot be used in "
                    f"{context}; assign the atomic result before the loop",
                    image_type,
                )
            return None

        original_name = self.unique_expression_temp_name(f"cgl_{operation}_original")
        image_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[1])
        target_expr = f"{image_name}[{coord}]"
        if operation == "imageAtomicCompSwap":
            compare = self.generate_expression_with_expected(args[2], return_type)
            value = self.generate_expression_with_expected(args[3], return_type)
            atomic_statement = (
                f"InterlockedCompareExchange({target_expr}, "
                f"{compare}, {value}, {original_name});"
            )
        else:
            value = self.generate_expression_with_expected(args[2], return_type)
            atomic_statement = f"{intrinsic}({target_expr}, {value}, {original_name});"

        self.add_expression_prelude(
            [
                f"{return_type} {original_name};",
                atomic_statement,
            ],
            result_name=original_name,
        )
        return original_name

    def image_atomic_expression(self, operation, args):
        if not self.image_atomic_intrinsic(operation):
            return None

        required_args = 4 if operation == "imageAtomicCompSwap" else 3
        if len(args) < required_args:
            return self.unsupported_image_atomic_call(
                operation, self.image_atomic_required_args_reason(operation)
            )

        image_type = self.resource_base_type(self.image_resource_type(args[0]))
        base_helper_name = self.image_atomic_helper_name(operation, image_type)
        if self.image_resource_access(args[0]) in {"readonly", "writeonly"}:
            return self.unsupported_image_atomic_call(
                operation,
                "requires readwrite image resource",
                image_type,
            )

        if not base_helper_name:
            reason = (
                "requires scalar int or uint "
                "image1D/image1DArray/image2D/image3D/image2DArray resource"
            )
            return self.unsupported_image_atomic_call(
                operation,
                reason,
                image_type,
            )

        return_type = self.image_atomic_return_type(image_type)
        intrinsic = self.image_atomic_intrinsic(operation)
        value_expr = self.image_atomic_value_expression(
            operation,
            args,
            image_type,
            intrinsic,
            return_type,
        )
        if value_expr is not None:
            return value_expr

        image_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[1])
        helper_name = self.helper_function_name(base_helper_name)
        self.register_helper_function(
            helper_name,
            self.build_image_atomic_helper(helper_name, operation, image_type),
        )
        if operation == "imageAtomicCompSwap":
            compare = self.generate_expression(args[2])
            value = self.generate_expression(args[3])
            return f"{helper_name}({image_name}, {coord}, {compare}, {value})"

        value = self.generate_expression(args[2])
        return f"{helper_name}({image_name}, {coord}, {value})"

    def build_image_atomic_helper(self, helper_name, operation, image_type):
        return_type = self.image_atomic_return_type(image_type)
        coord_type = self.image_atomic_coord_type(image_type)
        intrinsic = self.image_atomic_intrinsic(operation)
        if not return_type or not coord_type or not intrinsic:
            return ""

        if operation == "imageAtomicCompSwap":
            return (
                f"{return_type} {helper_name}({image_type} image, "
                f"{coord_type} coord, {return_type} compareValue, "
                f"{return_type} value)\n"
                "{\n"
                f"    {return_type} original;\n"
                "    InterlockedCompareExchange(image[coord], compareValue, value, original);\n"
                "    return original;\n"
                "}"
            )

        return (
            f"{return_type} {helper_name}({image_type} image, "
            f"{coord_type} coord, {return_type} value)\n"
            "{\n"
            f"    {return_type} original;\n"
            f"    {intrinsic}(image[coord], value, original);\n"
            "    return original;\n"
            "}"
        )

    def resource_query_slang_type(self, resource_arg, resource_type):
        if self.is_storage_image_type(resource_type):
            image_type = self.resource_base_type(self.image_resource_type(resource_arg))
            if image_type:
                return image_type
        return self.convert_type(resource_type)

    def resource_query_helper_name(self, func_name, resource_type, resource_slang_type):
        base_name = f"cgl_{func_name}_{resource_type}"
        if resource_slang_type == self.convert_type(resource_type):
            return base_name
        return f"{base_name}_{self.resource_helper_type_suffix(resource_slang_type)}"

    def resource_helper_type_suffix(self, resource_slang_type):
        return "".join(
            char if char.isalnum() else "_"
            for char in str(resource_slang_type).strip("_")
        ).strip("_")

    def generate_resource_call(self, func_name, args):
        if func_name == "imageLoad" and len(args) >= 2:
            return self.image_load_expression(args)

        if func_name == "imageStore" and len(args) >= 3:
            return self.image_store_expression(args)

        if func_name == "buffer_load":
            return self.buffer_load_expression(args)

        if func_name == "buffer_store":
            return self.buffer_store_expression(args)

        if func_name == "buffer_append":
            return self.buffer_append_expression(args)

        if func_name == "buffer_consume":
            return self.buffer_consume_expression(args)

        if func_name == "buffer_dimensions":
            return self.buffer_dimensions_expression(args)

        if func_name in {
            "imageAtomicAdd",
            "imageAtomicMin",
            "imageAtomicMax",
            "imageAtomicAnd",
            "imageAtomicOr",
            "imageAtomicXor",
            "imageAtomicExchange",
            "imageAtomicCompSwap",
        }:
            return self.image_atomic_expression(func_name, args)

        structured_buffer_atomic = self.structured_buffer_atomic_expression(
            func_name, args
        )
        if structured_buffer_atomic is not None:
            return structured_buffer_atomic

        if func_name in {"texture", "textureLod", "textureGrad"}:
            sample_args = self.sampled_texture_operation_args(func_name, args)
            if sample_args is None:
                return self.unsupported_sampled_texture_call(
                    func_name,
                    self.sampled_texture_operation_unsupported_reason(func_name, args),
                )

            texture_name, coord, extra_args = sample_args
            coord_node = args[self.sampled_texture_coord_index(args)]
            coord_reason = self.sampled_texture_coordinate_rank_unsupported_reason(
                args[0], coord_node
            )
            if coord_reason:
                return self.unsupported_sampled_texture_call(func_name, coord_reason)

            if func_name == "texture":
                if extra_args:
                    bias_reason = self.scalar_texture_bias_unsupported_reason(
                        extra_args[0]
                    )
                    if bias_reason:
                        return self.unsupported_sampled_texture_call(
                            func_name, bias_reason
                        )
                    bias = self.generate_expression(extra_args[0])
                    return f"{texture_name}.SampleBias({coord}, {bias})"
                return f"{texture_name}.Sample({coord})"

            if func_name == "textureLod" and extra_args:
                lod_reason = self.scalar_texture_lod_unsupported_reason(extra_args[0])
                if lod_reason:
                    return self.unsupported_sampled_texture_call(func_name, lod_reason)
                lod = self.generate_expression(extra_args[0])
                return f"{texture_name}.SampleLevel({coord}, {lod})"

            if func_name == "textureGrad" and len(extra_args) >= 2:
                grad_reason = self.texture_gradient_rank_unsupported_reason(
                    args[0], extra_args[0]
                ) or self.texture_gradient_rank_unsupported_reason(
                    args[0], extra_args[1]
                )
                if grad_reason:
                    return self.unsupported_sampled_texture_call(func_name, grad_reason)
                ddx = self.generate_expression(extra_args[0])
                ddy = self.generate_expression(extra_args[1])
                return f"{texture_name}.SampleGrad({coord}, {ddx}, {ddy})"

            return self.unsupported_sampled_texture_call(
                func_name,
                self.sampled_texture_operation_arity_requirement(func_name),
            )

        if func_name in {"textureOffset", "textureLodOffset", "textureGradOffset"}:
            return self.generate_texture_offset(func_name, args)

        if func_name in {
            "textureProj",
            "textureProjOffset",
            "textureProjLod",
            "textureProjLodOffset",
            "textureProjGrad",
            "textureProjGradOffset",
        }:
            return self.generate_texture_projected(func_name, args)

        if func_name in {
            "textureGather",
            "textureGatherOffset",
            "textureGatherOffsets",
        }:
            return self.generate_texture_gather(func_name, args)

        if func_name in {
            "textureCompare",
            "textureCompareLod",
            "textureCompareGrad",
            "textureCompareOffset",
        }:
            return self.generate_texture_compare(func_name, args)

        if func_name in {"textureGatherCompare", "textureGatherCompareOffset"}:
            return self.generate_texture_gather_compare(func_name, args)

        if func_name == "texelFetch":
            fetch_args = self.sampled_texture_operation_args(func_name, args)
            if fetch_args is None:
                return self.unsupported_sampled_texture_call(
                    func_name,
                    self.sampled_texture_operation_unsupported_reason(func_name, args),
                )
            texture_name, coord, extra_args = fetch_args

            coord_node = args[self.sampled_texture_coord_index(args)]
            coord_reason = self.texel_fetch_coordinate_rank_unsupported_reason(
                args[0], coord_node
            )
            if coord_reason:
                return self.unsupported_sampled_texture_call(func_name, coord_reason)
            fetch_index_reason = (
                self.scalar_integer_texture_argument_unsupported_reason(
                    extra_args[0], "fetch index argument"
                )
            )
            if fetch_index_reason:
                return self.unsupported_sampled_texture_call(
                    func_name, fetch_index_reason
                )
            lod_or_sample = self.generate_expression(extra_args[0])
            texture_type = self.get_expression_type(args[0])
            if self.is_multisample_sampler_type(texture_type):
                return f"{texture_name}[{coord}, {lod_or_sample}]"
            coord_constructor = self.texel_fetch_coord_constructor(texture_type)
            return f"{texture_name}.Load({coord_constructor}({coord}, {lod_or_sample}))"

        if func_name in {"textureSize", "imageSize"}:
            return self.generate_dimension_query(func_name, args)

        if func_name in {"textureSamples", "imageSamples"}:
            return self.generate_sample_count_query(func_name, args)

        if func_name == "textureQueryLevels":
            return self.generate_texture_query_levels(args)

        if func_name == "textureQueryLod":
            return self.generate_texture_query_lod(args)

        return None

    def generate_dimension_query(self, func_name, args):
        if not args:
            return self.unsupported_resource_query_call(
                func_name, "requires a resource argument"
            )

        resource_type = self.resource_base_type(self.get_expression_type(args[0]))
        if not self.dimension_query_accepts_resource(func_name, resource_type):
            return self.unsupported_resource_query_call(
                func_name, self.dimension_query_requirement(func_name)
            )

        spec = self.dimension_query_spec(resource_type)
        if spec is None:
            return self.unsupported_resource_query_call(
                func_name, self.dimension_query_requirement(func_name)
            )

        arity_reason = self.dimension_query_arity_unsupported_reason(
            func_name, args, spec
        )
        if arity_reason:
            return self.unsupported_resource_query_call(func_name, arity_reason)

        result_type = self.query_return_type(spec["dimensions"])
        expected_reason = self.resource_query_expected_type_unsupported_reason(
            func_name, result_type
        )
        if expected_reason:
            return self.unsupported_resource_query_call(
                func_name,
                expected_reason,
                self.zero_value_for_type(self.current_expression_expected_type),
            )

        resource_name = self.generate_expression(args[0])
        resource_slang_type = self.resource_query_slang_type(args[0], resource_type)
        base_helper_name = self.resource_query_helper_name(
            func_name, resource_type, resource_slang_type
        )
        helper_name = self.helper_function_name(base_helper_name)
        self.register_helper_function(
            helper_name,
            self.build_dimension_query_helper(
                helper_name, resource_type, spec, resource_slang_type
            ),
        )

        if spec["mip"]:
            lod = self.generate_expression(args[1]) if len(args) > 1 else "0"
            return f"{helper_name}({resource_name}, {lod})"
        return f"{helper_name}({resource_name})"

    def dimension_query_accepts_resource(self, func_name, resource_type):
        if func_name == "textureSize":
            return self.is_sampled_texture_resource_type(resource_type)
        if func_name == "imageSize":
            return self.is_storage_image_type(resource_type)
        return False

    def dimension_query_requirement(self, func_name):
        if func_name == "textureSize":
            return "requires a sampled texture resource"
        if func_name == "imageSize":
            return "requires an image resource"
        return "requires a resource"

    def dimension_query_arity_unsupported_reason(self, func_name, args, spec):
        if func_name == "imageSize":
            if len(args) != 1:
                return "accepts only a resource argument"
            return None

        if len(args) > 2:
            return "accepts resource and optional mip argument"
        if len(args) == 2:
            return self.scalar_texture_mip_unsupported_reason(args[1])
        return None

    def resource_query_expected_type_unsupported_reason(self, func_name, result_type):
        expected_type = self.convert_type(self.current_expression_expected_type)
        if not expected_type or expected_type == "auto":
            return None
        if not (
            self.is_scalar_value_type(expected_type)
            or self.is_vector_value_type(expected_type)
        ):
            return None
        if expected_type == result_type:
            return None
        return f"returns {result_type} but target expects {expected_type}"

    def generate_sample_count_query(self, func_name, args):
        if not args:
            return self.unsupported_resource_query_call(
                func_name, "requires a resource argument"
            )
        if len(args) != 1:
            return self.unsupported_resource_query_call(
                func_name, "accepts only a resource argument"
            )

        resource_type = self.resource_base_type(self.get_expression_type(args[0]))
        spec = self.dimension_query_spec(resource_type)
        if not self.sample_count_query_accepts_resource(func_name, resource_type):
            return self.unsupported_resource_query_call(
                func_name, self.sample_count_query_requirement(func_name)
            )
        if spec is None or not spec["samples"]:
            return self.unsupported_resource_query_call(
                func_name, self.sample_count_query_requirement(func_name)
            )

        expected_reason = self.resource_query_expected_type_unsupported_reason(
            func_name, "int"
        )
        if expected_reason:
            return self.unsupported_resource_query_call(
                func_name,
                expected_reason,
                self.zero_value_for_type(self.current_expression_expected_type),
            )

        resource_name = self.generate_expression(args[0])
        resource_slang_type = self.resource_query_slang_type(args[0], resource_type)
        base_helper_name = self.resource_query_helper_name(
            func_name, resource_type, resource_slang_type
        )
        helper_name = self.helper_function_name(base_helper_name)
        self.register_helper_function(
            helper_name,
            self.build_sample_count_query_helper(
                helper_name, resource_type, spec, resource_slang_type
            ),
        )
        return f"{helper_name}({resource_name})"

    def sampled_texture_args(self, args):
        coord_index = self.sampled_texture_coord_index(args)
        if len(args) <= coord_index:
            return None

        texture_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[coord_index])
        return texture_name, coord, args[coord_index + 1 :]

    def sampled_texture_coord_index(self, args):
        return 2 if self.is_explicit_sampler_argument(args) else 1

    def sampled_texture_operation_args(self, func_name, args):
        sample_args = self.sampled_texture_args(args)
        if sample_args is None:
            return None

        if not self.sampled_texture_operation_accepts_resource(func_name, args[0]):
            return None

        _texture_name, _coord, extra_args = sample_args
        if not self.sampled_texture_operation_accepts_extra_args(func_name, extra_args):
            return None

        return sample_args

    def sampled_texture_operation_accepts_extra_args(self, func_name, extra_args):
        if func_name == "texture":
            return len(extra_args) <= 1
        if func_name in {"textureLod", "texelFetch"}:
            return len(extra_args) == 1
        if func_name == "textureGrad":
            return len(extra_args) == 2
        return False

    def sampled_texture_operation_accepts_resource(self, func_name, texture_arg):
        resource_type = self.resource_base_type(self.get_expression_type(texture_arg))
        if resource_type is None:
            return True
        if func_name == "texelFetch":
            return self.is_texel_fetch_sampler_type(resource_type)
        return self.sampled_texture_sampling_accepts_resource(texture_arg)

    def sampled_texture_sampling_accepts_resource(self, texture_arg):
        resource_type = self.resource_base_type(self.get_expression_type(texture_arg))
        if resource_type is None:
            return True
        return self.is_lod_query_sampler_type(resource_type)

    def sampled_texture_operation_unsupported_reason(self, func_name, args):
        if not args:
            return "requires texture and coordinate arguments"

        coord_index = self.sampled_texture_coord_index(args)
        if len(args) <= coord_index:
            return "requires texture and coordinate arguments"

        if not self.sampled_texture_operation_accepts_resource(func_name, args[0]):
            return self.sampled_texture_operation_resource_requirement(func_name)

        return self.sampled_texture_operation_arity_requirement(func_name)

    def sampled_texture_operation_resource_requirement(self, func_name):
        if func_name == "texelFetch":
            return "requires a non-shadow texel-fetchable sampled texture resource"
        return "requires a non-shadow non-multisampled sampled texture resource"

    def sampled_texture_operation_arity_requirement(self, func_name):
        if func_name == "texture":
            return "accepts coordinate and optional bias arguments"
        if func_name == "textureLod":
            return "requires one lod argument"
        if func_name == "textureGrad":
            return "requires gradient x and gradient y arguments"
        if func_name == "texelFetch":
            return "requires one lod/sample argument"
        return "has unsupported arguments"

    def unsupported_sampled_texture_call(self, func_name, reason):
        return (
            f"/* unsupported Slang sampled texture: "
            f"{func_name} {reason} */ float4(0.0)"
        )

    def generate_texture_offset(self, func_name, args):
        sample_args = self.sampled_texture_args(args)
        if sample_args is None:
            return self.unsupported_texture_offset_call(
                func_name, "requires texture and coordinate arguments"
            )

        if not self.sampled_texture_sampling_accepts_resource(args[0]):
            return self.unsupported_texture_offset_call(
                func_name,
                "requires a non-shadow non-multisampled sampled texture resource",
            )

        texture_name, coord, extra_args = sample_args
        coord_node = args[self.sampled_texture_coord_index(args)]
        coord_reason = self.sampled_texture_coordinate_rank_unsupported_reason(
            args[0], coord_node
        )
        if coord_reason:
            return self.unsupported_texture_offset_call(func_name, coord_reason)

        if func_name == "textureOffset":
            if len(extra_args) not in {1, 2}:
                return self.unsupported_texture_offset_call(
                    func_name, "requires offset and optional bias arguments"
                )
            offset_reason = self.texture_offset_rank_unsupported_reason(
                args[0], extra_args[0]
            )
            if offset_reason:
                return self.unsupported_texture_offset_call(func_name, offset_reason)
            offset = self.generate_expression(extra_args[0])
            if len(extra_args) == 2:
                bias_reason = self.scalar_texture_bias_unsupported_reason(extra_args[1])
                if bias_reason:
                    return self.unsupported_texture_offset_call(func_name, bias_reason)
                bias = self.generate_expression(extra_args[1])
                return f"{texture_name}.SampleBias({coord}, {bias}, {offset})"
            return f"{texture_name}.Sample({coord}, {offset})"

        if func_name == "textureLodOffset":
            if len(extra_args) != 2:
                return self.unsupported_texture_offset_call(
                    func_name, "requires lod and offset arguments"
                )
            lod_reason = self.scalar_texture_lod_unsupported_reason(extra_args[0])
            if lod_reason:
                return self.unsupported_texture_offset_call(func_name, lod_reason)
            offset_reason = self.texture_offset_rank_unsupported_reason(
                args[0], extra_args[1]
            )
            if offset_reason:
                return self.unsupported_texture_offset_call(func_name, offset_reason)
            lod = self.generate_expression(extra_args[0])
            offset = self.generate_expression(extra_args[1])
            return f"{texture_name}.SampleLevel({coord}, {lod}, {offset})"

        if len(extra_args) != 3:
            return self.unsupported_texture_offset_call(
                func_name, "requires gradient x, gradient y, and offset arguments"
            )
        grad_reason = self.texture_gradient_rank_unsupported_reason(
            args[0], extra_args[0]
        ) or self.texture_gradient_rank_unsupported_reason(args[0], extra_args[1])
        if grad_reason:
            return self.unsupported_texture_offset_call(func_name, grad_reason)
        offset_reason = self.texture_offset_rank_unsupported_reason(
            args[0], extra_args[2]
        )
        if offset_reason:
            return self.unsupported_texture_offset_call(func_name, offset_reason)
        ddx = self.generate_expression(extra_args[0])
        ddy = self.generate_expression(extra_args[1])
        offset = self.generate_expression(extra_args[2])
        return f"{texture_name}.SampleGrad({coord}, {ddx}, {ddy}, {offset})"

    def unsupported_texture_offset_call(self, func_name, reason):
        return (
            f"/* unsupported Slang texture offset: {func_name} {reason} */ float4(0.0)"
        )

    def generate_texture_projected(self, func_name, args):
        sample_args = self.sampled_texture_args(args)
        if sample_args is None:
            return self.unsupported_texture_projected_call(
                func_name, "requires texture and projected coordinate arguments"
            )

        if not self.projected_texture_accepts_resource(args[0]):
            return self.unsupported_texture_projected_call(
                func_name, "requires sampler1D/2D/3D texture resource"
            )

        texture_name, coord, extra_args = sample_args
        coord_node = args[self.sampled_texture_coord_index(args)]
        projected_coord = self.projected_texture_coord(args[0], coord_node, coord)
        if projected_coord is None:
            return self.unsupported_texture_projected_call(
                func_name, "requires sampler1D/2D/3D projection coordinates"
            )

        if func_name == "textureProj":
            if not extra_args:
                return f"{texture_name}.Sample({projected_coord})"
            if len(extra_args) == 1:
                bias_reason = self.scalar_texture_bias_unsupported_reason(extra_args[0])
                if bias_reason:
                    return self.unsupported_texture_projected_call(
                        func_name, bias_reason
                    )
                bias = self.generate_expression(extra_args[0])
                return f"{texture_name}.SampleBias({projected_coord}, {bias})"
            return self.unsupported_texture_projected_call(
                func_name, "accepts at most one bias argument"
            )

        if func_name == "textureProjOffset":
            if len(extra_args) == 1:
                offset_reason = self.texture_offset_rank_unsupported_reason(
                    args[0], extra_args[0]
                )
                if offset_reason:
                    return self.unsupported_texture_projected_call(
                        func_name, offset_reason
                    )
                offset = self.generate_expression(extra_args[0])
                return f"{texture_name}.Sample({projected_coord}, {offset})"
            if len(extra_args) == 2:
                offset_reason = self.texture_offset_rank_unsupported_reason(
                    args[0], extra_args[0]
                )
                if offset_reason:
                    return self.unsupported_texture_projected_call(
                        func_name, offset_reason
                    )
                bias_reason = self.scalar_texture_bias_unsupported_reason(extra_args[1])
                if bias_reason:
                    return self.unsupported_texture_projected_call(
                        func_name, bias_reason
                    )
                offset = self.generate_expression(extra_args[0])
                bias = self.generate_expression(extra_args[1])
                return f"{texture_name}.SampleBias({projected_coord}, {bias}, {offset})"
            return self.unsupported_texture_projected_call(
                func_name, "requires offset and optional bias arguments"
            )

        if func_name == "textureProjLod":
            if len(extra_args) != 1:
                return self.unsupported_texture_projected_call(
                    func_name, "requires one lod argument"
                )
            lod_reason = self.scalar_texture_lod_unsupported_reason(extra_args[0])
            if lod_reason:
                return self.unsupported_texture_projected_call(func_name, lod_reason)
            lod = self.generate_expression(extra_args[0])
            return f"{texture_name}.SampleLevel({projected_coord}, {lod})"

        if func_name == "textureProjLodOffset":
            if len(extra_args) != 2:
                return self.unsupported_texture_projected_call(
                    func_name, "requires lod and offset arguments"
                )
            lod_reason = self.scalar_texture_lod_unsupported_reason(extra_args[0])
            if lod_reason:
                return self.unsupported_texture_projected_call(func_name, lod_reason)
            offset_reason = self.texture_offset_rank_unsupported_reason(
                args[0], extra_args[1]
            )
            if offset_reason:
                return self.unsupported_texture_projected_call(func_name, offset_reason)
            lod = self.generate_expression(extra_args[0])
            offset = self.generate_expression(extra_args[1])
            return f"{texture_name}.SampleLevel({projected_coord}, {lod}, {offset})"

        if func_name == "textureProjGrad":
            if len(extra_args) != 2:
                return self.unsupported_texture_projected_call(
                    func_name, "requires gradient x and gradient y arguments"
                )
            grad_reason = self.texture_gradient_rank_unsupported_reason(
                args[0], extra_args[0]
            ) or self.texture_gradient_rank_unsupported_reason(args[0], extra_args[1])
            if grad_reason:
                return self.unsupported_texture_projected_call(func_name, grad_reason)
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            return f"{texture_name}.SampleGrad({projected_coord}, {ddx}, {ddy})"

        if len(extra_args) != 3:
            return self.unsupported_texture_projected_call(
                func_name, "requires gradient x, gradient y, and offset arguments"
            )
        grad_reason = self.texture_gradient_rank_unsupported_reason(
            args[0], extra_args[0]
        ) or self.texture_gradient_rank_unsupported_reason(args[0], extra_args[1])
        if grad_reason:
            return self.unsupported_texture_projected_call(func_name, grad_reason)
        offset_reason = self.texture_offset_rank_unsupported_reason(
            args[0], extra_args[2]
        )
        if offset_reason:
            return self.unsupported_texture_projected_call(func_name, offset_reason)
        ddx = self.generate_expression(extra_args[0])
        ddy = self.generate_expression(extra_args[1])
        offset = self.generate_expression(extra_args[2])
        return f"{texture_name}.SampleGrad({projected_coord}, {ddx}, {ddy}, {offset})"

    def projected_texture_accepts_resource(self, texture_node):
        resource_type = self.resource_base_type(self.get_expression_type(texture_node))
        if resource_type is None:
            return True
        return resource_type in {"sampler1D", "sampler2D", "sampler3D"}

    def projected_texture_coord(self, texture_node, coord_node, coord):
        resource_type = self.resource_base_type(self.get_expression_type(texture_node))
        coord_type = self.resource_base_type(self.get_expression_type(coord_node))
        specs = {
            "sampler1D": {
                "vec2": ("x", "y"),
                "float2": ("x", "y"),
                "vec4": ("x", "w"),
                "float4": ("x", "w"),
            },
            "sampler2D": {
                "vec3": ("xy", "z"),
                "float3": ("xy", "z"),
                "vec4": ("xy", "w"),
                "float4": ("xy", "w"),
            },
            "sampler3D": {
                "vec4": ("xyz", "w"),
                "float4": ("xyz", "w"),
            },
        }
        resource_specs = specs.get(resource_type)
        if resource_specs is None:
            return None
        coord_spec = resource_specs.get(coord_type)
        if coord_spec is None:
            return None
        numerator, divisor = coord_spec
        return f"{coord}.{numerator} / {coord}.{divisor}"

    def unsupported_texture_projected_call(self, func_name, reason):
        return (
            f"/* unsupported Slang projected texture: "
            f"{func_name} {reason} */ float4(0.0)"
        )

    def generate_texture_gather(self, func_name, args):
        gather_args = self.sampled_texture_args(args)
        if gather_args is None:
            return self.unsupported_texture_gather_call(
                func_name, "requires texture and coordinate arguments"
            )

        if not self.sampled_texture_sampling_accepts_resource(args[0]):
            return self.unsupported_texture_gather_call(
                func_name,
                "requires a non-shadow non-multisampled sampled texture resource",
            )

        texture_name, coord, extra_args = gather_args
        coord_node = args[self.sampled_texture_coord_index(args)]
        coord_reason = self.sampled_texture_coordinate_rank_unsupported_reason(
            args[0], coord_node
        )
        if coord_reason:
            return self.unsupported_texture_gather_call(func_name, coord_reason)
        offset_args = []
        component_arg = None

        if func_name == "textureGather":
            if len(extra_args) > 1:
                return self.unsupported_texture_gather_call(
                    func_name, "accepts at most one component argument"
                )
            if extra_args:
                component_arg = extra_args[0]
        elif func_name == "textureGatherOffset":
            if len(extra_args) not in {1, 2}:
                return self.unsupported_texture_gather_call(
                    func_name, "requires offset and optional component arguments"
                )
            offset_reason = self.gather_offset_rank_unsupported_reason(
                args[0], extra_args[0]
            )
            if offset_reason:
                return self.unsupported_texture_gather_call(func_name, offset_reason)
            offset_args = [extra_args[0]]
            if len(extra_args) == 2:
                component_arg = extra_args[1]
        else:
            offsets_reason = self.gather_offsets_rank_unsupported_reason(
                args[0], extra_args
            )
            if offsets_reason:
                return self.unsupported_texture_gather_call(func_name, offsets_reason)
            offset_args, component_arg = self.texture_gather_offsets_args(extra_args)
            if offset_args is None:
                return self.unsupported_texture_gather_call(
                    func_name,
                    "requires a typed offsets array or four offset arguments",
                )

        component_reason = self.texture_gather_component_unsupported_reason(
            component_arg
        )
        if component_reason:
            return self.unsupported_texture_gather_call(func_name, component_reason)

        method_args = [coord] + [
            self.generate_expression(offset_arg) for offset_arg in offset_args
        ]
        method = self.texture_gather_method(component_arg)
        if method is not None:
            return f"{texture_name}.{method}({', '.join(method_args)})"
        if isinstance(component_arg, LiteralNode):
            return self.unsupported_texture_gather_call(
                func_name, "component literal must be 0, 1, 2, or 3"
            )

        component = self.generate_expression(component_arg)
        return self.texture_gather_component_expression(
            texture_name, method_args, component
        )

    def texture_gather_offsets_args(self, extra_args):
        if len(extra_args) in {1, 2} and self.is_array_expression(extra_args[0]):
            offsets_name = self.generate_expression(extra_args[0])
            offset_args = [f"{offsets_name}[{index}]" for index in range(4)]
            component_arg = extra_args[1] if len(extra_args) == 2 else None
            return offset_args, component_arg

        if len(extra_args) in {4, 5}:
            component_arg = extra_args[4] if len(extra_args) == 5 else None
            return extra_args[:4], component_arg

        return None, None

    def texture_gather_method(self, component_arg):
        if component_arg is None:
            return "Gather"

        methods = {
            0: "GatherRed",
            1: "GatherGreen",
            2: "GatherBlue",
            3: "GatherAlpha",
        }
        return methods.get(self.literal_int_value(component_arg))

    def texture_gather_component_unsupported_reason(self, component_arg):
        if component_arg is None:
            return None

        component_type = self.expression_result_type(component_arg)
        if component_type is None:
            return None

        mapped_type = self.convert_type(component_type)
        if mapped_type in {"int", "uint"}:
            return None

        return f"component argument must be scalar int or uint, got {mapped_type}"

    def texture_gather_component_expression(self, texture_name, method_args, component):
        arg_list = ", ".join(method_args)
        component_calls = [
            f"{texture_name}.{method}({arg_list})"
            for method in (
                "GatherRed",
                "GatherGreen",
                "GatherBlue",
                "GatherAlpha",
            )
        ]
        return (
            f"({component} == 0 ? {component_calls[0]} : "
            f"{component} == 1 ? {component_calls[1]} : "
            f"{component} == 2 ? {component_calls[2]} : "
            f"{component_calls[3]})"
        )

    def unsupported_texture_gather_call(self, func_name, reason):
        return (
            f"/* unsupported Slang texture gather: {func_name} {reason} */ float4(0.0)"
        )

    def generate_texture_compare(self, func_name, args):
        compare_args = self.texture_compare_args(func_name, args)
        if compare_args is None:
            return self.unsupported_texture_compare_call(
                func_name, "requires texture, coordinate, and compare arguments"
            )

        texture_name, coord, compare, extra_args = compare_args
        if not self.is_shadow_compare_resource(args[0]):
            return self.unsupported_texture_compare_call(
                func_name, "requires a shadow sampler resource"
            )
        coord_index = self.sampled_texture_coord_index(args)
        coord_reason = self.shadow_compare_coordinate_rank_unsupported_reason(
            args[0], args[coord_index]
        )
        if coord_reason:
            return self.unsupported_texture_compare_call(func_name, coord_reason)
        compare_reason = self.compare_reference_unsupported_reason(
            args[coord_index + 1]
        )
        if compare_reason:
            return self.unsupported_texture_compare_call(func_name, compare_reason)
        expected_reason = self.resource_query_expected_type_unsupported_reason(
            func_name, "float"
        )
        if expected_reason:
            return self.unsupported_texture_compare_call(
                func_name,
                expected_reason,
                self.zero_value_for_type(self.current_expression_expected_type),
            )

        if func_name == "textureCompare":
            if extra_args:
                return self.unsupported_texture_compare_call(
                    func_name, "accepts no extra arguments"
                )
            return f"{texture_name}.SampleCmp({coord}, {compare})"

        if func_name == "textureCompareOffset":
            if len(extra_args) != 1:
                return self.unsupported_texture_compare_call(
                    func_name, "requires one offset argument"
                )
            offset_reason = self.shadow_compare_offset_rank_unsupported_reason(
                args[0], extra_args[0]
            )
            if offset_reason:
                return self.unsupported_texture_compare_call(func_name, offset_reason)
            offset = self.generate_expression(extra_args[0])
            return f"{texture_name}.SampleCmp({coord}, {compare}, {offset})"

        if func_name == "textureCompareLod":
            if len(extra_args) != 1:
                return self.unsupported_texture_compare_call(
                    func_name, "requires one lod argument"
                )
            lod_reason = self.scalar_numeric_texture_argument_unsupported_reason(
                extra_args[0], "lod argument"
            )
            if lod_reason:
                return self.unsupported_texture_compare_call(func_name, lod_reason)
            lod = self.generate_expression(extra_args[0])
            return f"{texture_name}.SampleCmpLevel({coord}, {compare}, {lod})"

        if len(extra_args) != 2:
            return self.unsupported_texture_compare_call(
                func_name, "requires gradient x and gradient y arguments"
            )
        grad_reason = self.shadow_compare_gradient_rank_unsupported_reason(
            args[0], extra_args[0]
        ) or self.shadow_compare_gradient_rank_unsupported_reason(
            args[0], extra_args[1]
        )
        if grad_reason:
            return self.unsupported_texture_compare_call(func_name, grad_reason)
        ddx = self.generate_expression(extra_args[0])
        ddy = self.generate_expression(extra_args[1])
        return f"{texture_name}.SampleCmpGrad({coord}, {compare}, {ddx}, {ddy})"

    def generate_texture_gather_compare(self, func_name, args):
        compare_args = self.texture_compare_args(func_name, args)
        if compare_args is None:
            return self.unsupported_texture_gather_compare_call(
                func_name, "requires texture, coordinate, and compare arguments"
            )

        texture_name, coord, compare, extra_args = compare_args
        if not self.is_shadow_compare_resource(args[0]):
            return self.unsupported_texture_gather_compare_call(
                func_name, "requires a shadow sampler resource"
            )
        coord_index = self.sampled_texture_coord_index(args)
        coord_reason = self.shadow_compare_coordinate_rank_unsupported_reason(
            args[0], args[coord_index]
        )
        if coord_reason:
            return self.unsupported_texture_gather_compare_call(func_name, coord_reason)
        compare_reason = self.compare_reference_unsupported_reason(
            args[coord_index + 1]
        )
        if compare_reason:
            return self.unsupported_texture_gather_compare_call(
                func_name, compare_reason
            )
        expected_reason = self.resource_query_expected_type_unsupported_reason(
            func_name, "float4"
        )
        if expected_reason:
            return self.unsupported_texture_gather_compare_call(
                func_name,
                expected_reason,
                self.zero_value_for_type(self.current_expression_expected_type),
            )

        if func_name == "textureGatherCompare":
            if extra_args:
                return self.unsupported_texture_gather_compare_call(
                    func_name, "accepts no extra arguments"
                )
            return f"{texture_name}.GatherCmp({coord}, {compare})"

        if len(extra_args) != 1:
            return self.unsupported_texture_gather_compare_call(
                func_name, "requires one offset argument"
            )
        offset_reason = self.shadow_compare_offset_rank_unsupported_reason(
            args[0], extra_args[0]
        )
        if offset_reason:
            return self.unsupported_texture_gather_compare_call(
                func_name, offset_reason
            )
        offset = self.generate_expression(extra_args[0])
        return f"{texture_name}.GatherCmp({coord}, {compare}, {offset})"

    def texture_compare_args(self, func_name, args):
        coord_index = 2 if self.is_explicit_sampler_argument(args) else 1
        if len(args) <= coord_index + 1:
            return None

        texture_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[coord_index])
        compare = self.generate_expression(args[coord_index + 1])
        return texture_name, coord, compare, args[coord_index + 2 :]

    def sampled_texture_coordinate_rank_unsupported_reason(self, texture_node, coord):
        resource_type = self.resource_base_type(self.get_expression_type(texture_node))
        expected_rank = self.sampled_texture_coordinate_rank(resource_type)
        return self.texture_rank_unsupported_reason(
            coord, expected_rank, resource_type, "coordinate"
        )

    def texel_fetch_coordinate_rank_unsupported_reason(self, texture_node, coord):
        resource_type = self.resource_base_type(self.get_expression_type(texture_node))
        expected_rank = self.texel_fetch_coordinate_rank(resource_type)
        return self.texture_rank_unsupported_reason(
            coord, expected_rank, resource_type, "coordinate"
        )

    def texture_offset_rank_unsupported_reason(self, texture_node, offset):
        resource_type = self.resource_base_type(self.get_expression_type(texture_node))
        if resource_type is None:
            return None
        expected_rank = self.texture_offset_rank(resource_type)
        if expected_rank is None:
            return "requires an offset-capable sampler1D/2D/3D texture resource"
        rank_reason = self.texture_rank_unsupported_reason(
            offset, expected_rank, resource_type, "offset"
        )
        if rank_reason:
            return rank_reason
        return self.texture_offset_type_unsupported_reason(offset)

    def texture_gradient_rank_unsupported_reason(self, texture_node, gradient):
        resource_type = self.resource_base_type(self.get_expression_type(texture_node))
        expected_rank = self.texture_gradient_rank(resource_type)
        rank_reason = self.texture_rank_unsupported_reason(
            gradient, expected_rank, resource_type, "gradient"
        )
        if rank_reason:
            return rank_reason
        return self.texture_gradient_type_unsupported_reason(gradient)

    def gather_offset_rank_unsupported_reason(self, texture_node, offset):
        resource_type = self.resource_base_type(self.get_expression_type(texture_node))
        if resource_type is None:
            return None
        expected_rank = self.gather_offset_rank(resource_type)
        if expected_rank is None:
            return "requires a gather-offset-capable sampler2D/2DArray texture resource"
        rank_reason = self.texture_rank_unsupported_reason(
            offset, expected_rank, resource_type, "offset"
        )
        if rank_reason:
            return rank_reason
        return self.texture_offset_type_unsupported_reason(offset)

    def gather_offsets_rank_unsupported_reason(self, texture_node, extra_args):
        resource_type = self.resource_base_type(self.get_expression_type(texture_node))
        if resource_type is None:
            return None
        expected_rank = self.gather_offset_rank(resource_type)
        if expected_rank is None:
            return "requires a gather-offset-capable sampler2D/2DArray texture resource"

        if len(extra_args) in {1, 2} and self.is_array_expression(extra_args[0]):
            rank_reason = self.texture_rank_unsupported_reason(
                extra_args[0],
                expected_rank,
                resource_type,
                "offset",
                array_element=True,
            )
            if rank_reason:
                return rank_reason
            return self.texture_offset_type_unsupported_reason(
                extra_args[0], array_element=True
            )
        if len(extra_args) in {4, 5}:
            for offset in extra_args[:4]:
                reason = self.texture_rank_unsupported_reason(
                    offset, expected_rank, resource_type, "offset"
                )
                if reason:
                    return reason
                reason = self.texture_offset_type_unsupported_reason(offset)
                if reason:
                    return reason
        return None

    def shadow_compare_coordinate_rank_unsupported_reason(self, texture_node, coord):
        resource_type = self.resource_base_type(self.get_expression_type(texture_node))
        expected_rank = self.shadow_compare_coordinate_rank(resource_type)
        return self.texture_rank_unsupported_reason(
            coord, expected_rank, resource_type, "coordinate"
        )

    def shadow_compare_offset_rank_unsupported_reason(self, texture_node, offset):
        resource_type = self.resource_base_type(self.get_expression_type(texture_node))
        if resource_type is None:
            return None
        expected_rank = self.shadow_compare_offset_rank(resource_type)
        if expected_rank is None:
            return "requires an offset-capable sampler2DShadow/2DArrayShadow resource"
        rank_reason = self.texture_rank_unsupported_reason(
            offset, expected_rank, resource_type, "offset"
        )
        if rank_reason:
            return rank_reason
        return self.texture_offset_type_unsupported_reason(offset)

    def shadow_compare_gradient_rank_unsupported_reason(self, texture_node, gradient):
        resource_type = self.resource_base_type(self.get_expression_type(texture_node))
        expected_rank = self.shadow_compare_gradient_rank(resource_type)
        rank_reason = self.texture_rank_unsupported_reason(
            gradient, expected_rank, resource_type, "gradient"
        )
        if rank_reason:
            return rank_reason
        return self.texture_gradient_type_unsupported_reason(gradient)

    def compare_reference_unsupported_reason(self, compare):
        type_name = self.type_name_string(self.expression_result_type(compare))
        if type_name is None:
            return None

        mapped_type = self.convert_type(type_name)
        if mapped_type in {"float", "double"}:
            return None

        return f"compare reference must be scalar float or double, got {mapped_type}"

    def texture_offset_type_unsupported_reason(self, offset, array_element=False):
        type_name = self.type_name_string(self.expression_result_type(offset))
        if type_name is None:
            return None
        if array_element and "[" in type_name and "]" in type_name:
            type_name, _suffix = split_array_type_suffix(type_name)

        mapped_type = self.convert_type(type_name)
        if mapped_type == "int":
            return None

        info = self.vector_value_info(type_name)
        if info is not None and info["component_type"] == "int":
            return None

        return f"offset must be scalar or vector int, got {mapped_type}"

    def texture_gradient_type_unsupported_reason(self, gradient):
        type_name = self.type_name_string(self.expression_result_type(gradient))
        if type_name is None:
            return None

        mapped_type = self.convert_type(type_name)
        if mapped_type in {"float", "double"}:
            return None

        info = self.vector_value_info(type_name)
        if info is not None and info["component_type"] in {"float", "double"}:
            return None

        return f"gradient must be scalar or vector float or double, got {mapped_type}"

    def scalar_texture_argument_rank_unsupported_reason(self, node, role):
        actual_rank = self.expression_value_rank(node)
        if actual_rank is None or actual_rank == 1:
            return None
        return f"requires {self.texture_rank_phrase(1, role)}"

    def scalar_texture_lod_unsupported_reason(self, node):
        return self.scalar_texture_argument_rank_unsupported_reason(
            node, "lod argument"
        ) or self.scalar_numeric_texture_argument_unsupported_reason(
            node, "lod argument"
        )

    def scalar_texture_bias_unsupported_reason(self, node):
        return self.scalar_texture_argument_rank_unsupported_reason(
            node, "bias argument"
        ) or self.scalar_numeric_texture_argument_unsupported_reason(
            node, "bias argument"
        )

    def scalar_texture_mip_unsupported_reason(self, node):
        return self.scalar_texture_argument_rank_unsupported_reason(
            node, "mip argument"
        ) or self.scalar_integer_texture_argument_unsupported_reason(
            node, "mip argument"
        )

    def scalar_integer_texture_argument_unsupported_reason(self, node, role):
        type_name = self.type_name_string(self.expression_result_type(node))
        if type_name is None:
            return None

        mapped_type = self.convert_type(type_name)
        if mapped_type in {"int", "uint"}:
            return None

        return f"{role} must be scalar int or uint, got {mapped_type}"

    def scalar_numeric_texture_argument_unsupported_reason(self, node, role):
        type_name = self.type_name_string(self.expression_result_type(node))
        if type_name is None:
            return None

        mapped_type = self.convert_type(type_name)
        if mapped_type in {"int", "uint", "float", "double"}:
            return None

        return f"{role} must be scalar int, uint, float, or double, got {mapped_type}"

    def texture_rank_unsupported_reason(
        self, node, expected_rank, resource_type, role, array_element=False
    ):
        if expected_rank is None or resource_type is None:
            return None
        actual_rank = self.expression_value_rank(node, array_element=array_element)
        if actual_rank is None or actual_rank == expected_rank:
            return None
        return (
            f"requires {self.texture_rank_phrase(expected_rank, role)} "
            f"for {resource_type}"
        )

    def texture_rank_phrase(self, rank, role):
        if rank == 1:
            return f"a scalar {role}"
        return f"a {rank}-component {role}"

    def expression_value_rank(self, node, array_element=False):
        type_name = self.type_name_string(self.expression_result_type(node))
        if not type_name:
            return None
        if array_element and "[" in type_name and "]" in type_name:
            type_name, _suffix = split_array_type_suffix(type_name)
        if self.is_scalar_value_type(type_name):
            return 1
        info = self.vector_value_info(type_name)
        if info is None:
            return None
        return info["size"]

    def sampled_texture_coordinate_rank(self, resource_type):
        return {
            "sampler1D": 1,
            "sampler1DArray": 2,
            "sampler2D": 2,
            "sampler2DArray": 3,
            "sampler3D": 3,
            "samplerCube": 3,
            "samplerCubeArray": 4,
        }.get(resource_type)

    def texel_fetch_coordinate_rank(self, resource_type):
        return {
            "sampler1D": 1,
            "sampler1DArray": 2,
            "sampler2D": 2,
            "sampler2DArray": 3,
            "sampler3D": 3,
            "sampler2DMS": 2,
            "sampler2DMSArray": 3,
        }.get(resource_type)

    def texture_offset_rank(self, resource_type):
        return {
            "sampler1D": 1,
            "sampler1DArray": 1,
            "sampler2D": 2,
            "sampler2DArray": 2,
            "sampler3D": 3,
        }.get(resource_type)

    def texture_gradient_rank(self, resource_type):
        return {
            "sampler1D": 1,
            "sampler1DArray": 1,
            "sampler2D": 2,
            "sampler2DArray": 2,
            "sampler3D": 3,
            "samplerCube": 3,
            "samplerCubeArray": 3,
        }.get(resource_type)

    def gather_offset_rank(self, resource_type):
        return {
            "sampler2D": 2,
            "sampler2DArray": 2,
        }.get(resource_type)

    def shadow_compare_coordinate_rank(self, resource_type):
        return {
            "sampler2DShadow": 2,
            "sampler2DArrayShadow": 3,
            "samplerCubeShadow": 3,
            "samplerCubeArrayShadow": 4,
        }.get(resource_type)

    def shadow_compare_offset_rank(self, resource_type):
        return {
            "sampler2DShadow": 2,
            "sampler2DArrayShadow": 2,
        }.get(resource_type)

    def shadow_compare_gradient_rank(self, resource_type):
        return {
            "sampler2DShadow": 2,
            "sampler2DArrayShadow": 2,
            "samplerCubeShadow": 3,
            "samplerCubeArrayShadow": 3,
        }.get(resource_type)

    def is_shadow_compare_resource(self, node):
        resource_type = self.resource_base_type(self.get_expression_type(node))
        return resource_type is None or resource_type in {
            "sampler2DShadow",
            "sampler2DArrayShadow",
            "samplerCubeShadow",
            "samplerCubeArrayShadow",
        }

    def unsupported_texture_compare_call(self, func_name, reason, fallback="0.0"):
        return (
            f"/* unsupported Slang shadow compare: {func_name} {reason} */ "
            f"{fallback}"
        )

    def unsupported_texture_gather_compare_call(
        self, func_name, reason, fallback="float4(0.0)"
    ):
        return (
            f"/* unsupported Slang shadow gather compare: "
            f"{func_name} {reason} */ {fallback}"
        )

    def literal_int_value(self, node):
        if not isinstance(node, LiteralNode):
            return None
        value = node.value
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value, 0)
            except ValueError:
                return None
        return None

    def is_array_expression(self, node):
        type_name = self.type_name_string(self.expression_result_type(node))
        return isinstance(type_name, str) and "[" in type_name and "]" in type_name

    def generate_texture_query_levels(self, args):
        if not args:
            return self.unsupported_resource_query_call(
                "textureQueryLevels", "requires a resource argument"
            )
        if len(args) != 1:
            return self.unsupported_resource_query_call(
                "textureQueryLevels", "accepts only a resource argument"
            )

        resource_type = self.resource_base_type(self.get_expression_type(args[0]))
        spec = self.dimension_query_spec(resource_type)
        if (
            spec is None
            or not spec["mip"]
            or not self.is_sampled_texture_resource_type(resource_type)
        ):
            return self.unsupported_resource_query_call(
                "textureQueryLevels",
                "requires a mipmapped sampled texture resource",
            )

        expected_reason = self.resource_query_expected_type_unsupported_reason(
            "textureQueryLevels", "int"
        )
        if expected_reason:
            return self.unsupported_resource_query_call(
                "textureQueryLevels",
                expected_reason,
                self.zero_value_for_type(self.current_expression_expected_type),
            )

        resource_name = self.generate_expression(args[0])
        base_helper_name = f"cgl_textureQueryLevels_{resource_type}"
        helper_name = self.helper_function_name(base_helper_name)
        self.register_helper_function(
            helper_name,
            self.build_texture_query_levels_helper(helper_name, resource_type, spec),
        )
        return f"{helper_name}({resource_name})"

    def sample_count_query_accepts_resource(self, func_name, resource_type):
        if func_name == "textureSamples":
            return self.is_sampled_texture_resource_type(resource_type)
        if func_name == "imageSamples":
            return self.is_storage_image_type(resource_type)
        return False

    def sample_count_query_requirement(self, func_name):
        if func_name == "textureSamples":
            return "requires a multisampled texture resource"
        if func_name == "imageSamples":
            return "requires a multisampled image resource"
        return "requires a multisampled resource"

    def unsupported_resource_query_call(self, func_name, reason, fallback="0"):
        return (
            f"/* unsupported Slang resource query: {func_name} {reason} */ "
            f"{fallback}"
        )

    def generate_texture_query_lod(self, args):
        query_args = self.texture_query_lod_args(args)
        if query_args is None:
            return self.unsupported_texture_query_lod_call(
                self.texture_query_lod_unsupported_reason(args)
            )

        texture_name, coord = query_args
        coord_node = args[self.texture_query_lod_coord_index(args)]
        coord_reason = self.sampled_texture_coordinate_rank_unsupported_reason(
            args[0], coord_node
        )
        if coord_reason:
            return self.unsupported_texture_query_lod_call(coord_reason)
        expected_reason = self.resource_query_expected_type_unsupported_reason(
            "textureQueryLod", "float2"
        )
        if expected_reason:
            return self.unsupported_texture_query_lod_call(
                expected_reason,
                self.zero_value_for_type(self.current_expression_expected_type),
            )
        unclamped = f"{texture_name}.CalculateLevelOfDetailUnclamped({coord})"
        clamped = f"{texture_name}.CalculateLevelOfDetail({coord})"
        return f"float2({unclamped}, {clamped})"

    def texture_query_lod_args(self, args):
        coord_index = self.texture_query_lod_coord_index(args)
        if len(args) <= coord_index or len(args) != coord_index + 1:
            return None

        resource_type = self.resource_base_type(self.get_expression_type(args[0]))
        if not self.is_lod_query_sampler_type(resource_type):
            return None

        texture_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[coord_index])
        return texture_name, coord

    def texture_query_lod_coord_index(self, args):
        if len(args) >= 2 and self.is_sampler_state_type(
            self.get_expression_type(args[1])
        ):
            return 2
        return 1

    def texture_query_lod_unsupported_reason(self, args):
        if not args:
            return "requires texture and coordinate arguments"

        coord_index = self.texture_query_lod_coord_index(args)
        if len(args) <= coord_index:
            return "requires texture and coordinate arguments"
        if len(args) != coord_index + 1:
            return "accepts only texture, optional sampler, and coordinate arguments"

        resource_type = self.resource_base_type(self.get_expression_type(args[0]))
        if not self.is_lod_query_sampler_type(resource_type):
            return "requires a non-shadow non-multisampled sampled texture resource"

        return "requires texture and coordinate arguments"

    def unsupported_texture_query_lod_call(self, reason, fallback="float2(0.0, 0.0)"):
        return self.unsupported_resource_query_call("textureQueryLod", reason, fallback)

    def register_helper_function(self, name, source):
        if name not in self.helper_functions:
            self.helper_functions[name] = source

    def helper_function_name(self, base_name):
        if base_name in self.helper_name_aliases:
            return self.helper_name_aliases[base_name]

        candidate = base_name
        suffix = 1
        used_helper_names = set(self.helper_functions) | set(
            self.helper_name_aliases.values()
        )
        while candidate in self.user_symbol_names or candidate in used_helper_names:
            candidate = f"{base_name}_{suffix}"
            suffix += 1

        self.helper_name_aliases[base_name] = candidate
        return candidate

    def build_dimension_query_helper(
        self, helper_name, resource_type, spec, resource_slang_type=None
    ):
        resource_slang_type = resource_slang_type or self.convert_type(resource_type)
        return_type = self.query_return_type(spec["dimensions"])
        params = f"{resource_slang_type} tex"
        if spec["mip"]:
            params += ", uint mipLevel"

        declarations = self.query_local_declarations(spec)
        get_dimensions_args = self.get_dimensions_args(spec)
        dimensions = ", ".join(spec["dimensions"])
        if len(spec["dimensions"]) == 1:
            return_value = spec["dimensions"][0]
        else:
            return_value = f"{return_type}({dimensions})"

        return (
            f"{return_type} {helper_name}({params})\n"
            "{\n"
            f"{declarations}"
            f"    tex.GetDimensions({get_dimensions_args});\n"
            f"    return {return_value};\n"
            "}"
        )

    def build_sample_count_query_helper(
        self, helper_name, resource_type, spec, resource_slang_type=None
    ):
        resource_slang_type = resource_slang_type or self.convert_type(resource_type)
        declarations = self.query_local_declarations(spec)
        get_dimensions_args = self.get_dimensions_args(spec)
        return (
            f"int {helper_name}({resource_slang_type} tex)\n"
            "{\n"
            f"{declarations}"
            f"    tex.GetDimensions({get_dimensions_args});\n"
            "    return samples;\n"
            "}"
        )

    def build_texture_query_levels_helper(self, helper_name, resource_type, spec):
        resource_slang_type = self.convert_type(resource_type)
        declarations = self.query_local_declarations(spec)
        get_dimensions_args = self.texture_query_levels_args(spec)
        return (
            f"int {helper_name}({resource_slang_type} tex)\n"
            "{\n"
            f"{declarations}"
            f"    tex.GetDimensions({get_dimensions_args});\n"
            "    return levels;\n"
            "}"
        )

    def query_return_type(self, dimensions):
        if len(dimensions) == 1:
            return "int"
        return f"int{len(dimensions)}"

    def query_local_declarations(self, spec):
        names = list(spec["dimensions"])
        if spec["samples"]:
            names.append("samples")
        if spec["mip"]:
            names.append("levels")
        return "".join(f"    int {name};\n" for name in names)

    def get_dimensions_args(self, spec):
        args = []
        if spec["mip"]:
            args.append("mipLevel")
        args.extend(spec["dimensions"])
        if spec["samples"]:
            args.append("samples")
        if spec["mip"]:
            args.append("levels")
        return ", ".join(args)

    def texture_query_levels_args(self, spec):
        args = list(spec["dimensions"])
        args.append("levels")
        return ", ".join(args)

    def dimension_query_spec(self, type_name):
        specs = {
            "sampler1D": (("width",), True, False),
            "sampler1DArray": (("width", "elements"), True, False),
            "sampler2D": (("width", "height"), True, False),
            "sampler2DShadow": (("width", "height"), True, False),
            "sampler2DArray": (("width", "height", "elements"), True, False),
            "sampler2DArrayShadow": (
                ("width", "height", "elements"),
                True,
                False,
            ),
            "sampler3D": (("width", "height", "depth"), True, False),
            "samplerCube": (("width", "height"), True, False),
            "samplerCubeShadow": (("width", "height"), True, False),
            "samplerCubeArray": (("width", "height", "elements"), True, False),
            "samplerCubeArrayShadow": (
                ("width", "height", "elements"),
                True,
                False,
            ),
            "sampler2DMS": (("width", "height"), False, True),
            "sampler2DMSArray": (("width", "height", "elements"), False, True),
            "image1D": (("width",), False, False),
            "iimage1D": (("width",), False, False),
            "uimage1D": (("width",), False, False),
            "image1DArray": (("width", "elements"), False, False),
            "iimage1DArray": (("width", "elements"), False, False),
            "uimage1DArray": (("width", "elements"), False, False),
            "image2D": (("width", "height"), False, False),
            "iimage2D": (("width", "height"), False, False),
            "uimage2D": (("width", "height"), False, False),
            "image2DArray": (("width", "height", "elements"), False, False),
            "iimage2DArray": (("width", "height", "elements"), False, False),
            "uimage2DArray": (("width", "height", "elements"), False, False),
            "image3D": (("width", "height", "depth"), False, False),
            "iimage3D": (("width", "height", "depth"), False, False),
            "uimage3D": (("width", "height", "depth"), False, False),
            "image2DMS": (("width", "height"), False, True),
            "iimage2DMS": (("width", "height"), False, True),
            "uimage2DMS": (("width", "height"), False, True),
            "image2DMSArray": (("width", "height", "elements"), False, True),
            "iimage2DMSArray": (("width", "height", "elements"), False, True),
            "uimage2DMSArray": (("width", "height", "elements"), False, True),
        }
        spec = specs.get(type_name)
        if spec is None:
            return None
        dimensions, mip, samples = spec
        return {
            "dimensions": dimensions,
            "mip": mip,
            "samples": samples,
        }

    def is_explicit_sampler_argument(self, args):
        if len(args) < 2:
            return False
        return self.is_sampler_state_type(self.get_expression_type(args[1]))

    def is_sampler_state_type(self, type_name):
        return self.resource_base_type(type_name) in {
            "sampler",
            "SamplerState",
            "SamplerComparisonState",
        }

    def is_lod_query_sampler_type(self, type_name):
        resource_type = self.resource_base_type(type_name)
        return (
            isinstance(resource_type, str)
            and resource_type.startswith("sampler")
            and resource_type != "sampler"
            and "MS" not in resource_type
            and "Shadow" not in resource_type
        )

    def is_sampled_texture_resource_type(self, type_name):
        resource_type = self.resource_base_type(type_name)
        return (
            isinstance(resource_type, str)
            and resource_type.startswith("sampler")
            and resource_type != "sampler"
        )

    def get_expression_type(self, node):
        name = self.get_expression_name(node)
        if name is None:
            return None
        return self.variable_types.get(name)

    def get_expression_name(self, node):
        if isinstance(node, IdentifierNode):
            return node.name
        if isinstance(node, VariableNode):
            return node.name
        if isinstance(node, str):
            return node
        if isinstance(node, ArrayAccessNode):
            return self.get_expression_name(
                getattr(node, "array", getattr(node, "array_expr", None))
            )
        return None

    def resource_base_type(self, type_name):
        if not isinstance(type_name, str):
            return None
        return type_name.split("[", 1)[0]

    def is_multisample_sampler_type(self, type_name):
        return self.resource_base_type(type_name) in {
            "sampler2DMS",
            "sampler2DMSArray",
        }

    def is_texel_fetch_sampler_type(self, type_name):
        return self.resource_base_type(type_name) in {
            "sampler1D",
            "sampler1DArray",
            "sampler2D",
            "sampler2DArray",
            "sampler3D",
            "sampler2DMS",
            "sampler2DMSArray",
        }

    def texel_fetch_coord_constructor(self, type_name):
        base_type = self.resource_base_type(type_name)
        if base_type in {"sampler1D", "sampler1DArray"}:
            return "int2" if base_type == "sampler1D" else "int3"
        if base_type in {"sampler3D", "sampler2DArray"}:
            return "int4"
        return "int3"
