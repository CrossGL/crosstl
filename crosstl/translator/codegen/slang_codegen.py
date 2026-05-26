"""CrossGL-to-Slang code generator."""

from ..ast import (
    ArrayNode,
    ArrayAccessNode,
    ArrayLiteralNode,
    AssignmentNode,
    BinaryOpNode,
    BreakNode,
    CaseNode,
    CbufferNode,
    ContinueNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    FunctionNode,
    IfNode,
    LiteralNode,
    LiteralPatternNode,
    MatchNode,
    MemberAccessNode,
    RangeNode,
    ReturnNode,
    ShaderNode,
    StructNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    DoWhileNode,
    WhileNode,
    WildcardPatternNode,
)
from .array_utils import (
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

    def __init__(self):
        """Initialize Slang generation state and helper caches."""
        self.indent_level = 0
        self.indent_str = "    "
        self.variable_types = {}
        self.image_resource_types = {}
        self.helper_functions = {}
        self.helper_name_aliases = {}
        self.user_symbol_names = set()
        self.current_function_return_type = None
        self.current_expression_expected_type = None
        self.user_function_names = set()
        self.user_function_return_types = {}
        self.stage_entry_name_overrides = {}
        self.identifier_aliases = {}
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
            self.helper_functions = {}
            self.helper_name_aliases = {}
            self.user_symbol_names = self.collect_user_symbol_names(ast)
            self.current_function_return_type = None
            self.current_expression_expected_type = None
            self.user_function_names = self.collect_user_function_names(ast)
            self.user_function_return_types = self.collect_user_function_return_types(
                ast
            )
            self.stage_entry_name_overrides = {}
            self.identifier_aliases = {}

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
                if isinstance(node, StructNode):
                    result += (
                        "cbuffer " + self.generate_struct_definition(node) + "\n\n"
                    )
                elif hasattr(node, "name") and hasattr(node, "members"):
                    result += f"cbuffer {node.name} {{\n"
                    for member in node.members:
                        if hasattr(member, "member_type"):
                            member_type = str(member.member_type)
                        else:
                            member_type = getattr(member, "vtype", "float")
                        result += (
                            f"    {self.convert_type(member_type)} {member.name};\n"
                        )
                    result += "};\n\n"

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

    def generate_shader(self, node):
        """Render a full CrossGL shader AST as a Slang translation unit."""
        result = ""

        structs = getattr(node, "structs", [])
        for struct in structs:
            result += self.generate_struct(struct) + "\n\n"

        global_vars = getattr(node, "global_variables", [])
        for global_var in global_vars:
            result += self.generate_global_variable(global_var)

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

    def validate_slang_stage_body_builtins(self, body_statements, stage_name, params):
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
            return semantic

        for attr in getattr(node, "attributes", []) or []:
            if skip_stage_attributes and self.slang_stage_attribute_name(attr):
                continue
            if self.is_resource_format_attribute(attr):
                continue
            if hasattr(attr, "name"):
                return attr.name
        return None

    def is_resource_format_attribute(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return False
        attr_name = str(attr_name).lower()
        return attr_name == "format" or attr_name in self.supported_image_formats()

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
        for local_var in local_variables:
            result += self.generate_global_variable(local_var)

        for func in getattr(stage, "local_functions", []):
            result += self.generate_function(func) + "\n\n"

        entry_point = getattr(stage, "entry_point", None)
        if entry_point is not None:
            result += self.generate_function(
                entry_point,
                shader_type=stage_name,
                execution_config=getattr(stage, "execution_config", None),
                entry_name=self.stage_entry_name_overrides.get(id(entry_point)),
            )
            result += "\n\n"

        return result

    def convert_type_node_to_string(self, type_node) -> str:
        if isinstance(type_node, LiteralNode):
            return self.generate_literal(type_node)
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

    def register_variable_type(self, name, type_name, node=None):
        if not name or type_name is None:
            return
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        self.variable_types[name] = type_name
        if self.is_storage_image_type(type_name):
            self.image_resource_types[name] = self.map_resource_type_with_format(
                type_name, node
            )

    def generate_global_variable(self, node):
        if isinstance(node, ArrayNode):
            self.register_variable_type(node.name, node.element_type)
            element_type = self.convert_type(node.element_type)
            size = get_array_size_from_node(node)
            if size is None:
                return f"{element_type} {node.name}[];\n"
            return f"{element_type} {node.name}[{size}];\n"

        initial_value = getattr(node, "initial_value", getattr(node, "value", None))
        vtype = self.variable_declaration_type(node, initial_value)
        self.register_variable_type(node.name, vtype, node)
        declaration = self.format_declaration(vtype, node.name, node)
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
        self, node, shader_type=None, execution_config=None, entry_name=None
    ):
        """Render one CrossGL function or shader entry point as Slang code."""
        saved_variable_types = self.variable_types.copy()
        saved_image_resource_types = self.image_resource_types.copy()
        saved_function_return_type = self.current_function_return_type
        saved_identifier_aliases = self.identifier_aliases.copy()
        saved_hull_output_rewrite = self.current_hull_output_rewrite
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
        semantic_str = self.semantic_suffix(semantic, shader_type)

        effective_param_list = self.slang_filtered_stage_parameters(
            param_list, hull_output_rewrite
        )
        if shader_type:
            self.validate_slang_stage_body_builtins(
                body_statements, shader_type, effective_param_list
            )
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
                        self.register_variable_type(param.name, param.vtype, param)
                        param_type = self.map_resource_type_with_format(
                            param.vtype, param
                        )
                    else:
                        param_type = "float"
                    declaration = format_c_style_array_declaration(
                        param_type, param.name
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

        for param_type, param_name, semantic in self.slang_implicit_stage_parameters(
            body_statements, shader_type, effective_param_list
        ):
            self.register_variable_type(param_name, param_type)
            params.append(f"{self.convert_type(param_type)} {param_name} : {semantic}")

        params_str = ", ".join(params)
        identifier_aliases = self.slang_stage_system_value_aliases(
            body_statements, shader_type, effective_param_list
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
        if shader_type:
            self.validate_slang_stage_attributes(node, shader_type)
        result += self.generate_slang_stage_numthreads(
            node, shader_type, execution_config
        )
        if shader_type:
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
        self.current_function_return_type = saved_function_return_type
        self.identifier_aliases = saved_identifier_aliases
        self.current_hull_output_rewrite = saved_hull_output_rewrite
        return result

    def generate_compute_numthreads(self, execution_config=None):
        x, y, z = compute_local_size(execution_config)
        return f"[numthreads({x}, {y}, {z})]\n"

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
        type_node = getattr(param, "param_type", None)
        if getattr(type_node, "name", None) == patch_type:
            generic_args = getattr(type_node, "generic_args", []) or []
            if generic_args:
                return self.convert_type_node_to_string(generic_args[0])

        type_name = self.slang_parameter_type_name(param)
        prefix = f"{patch_type}<"
        if not type_name.startswith(prefix) or not type_name.endswith(">"):
            return None

        args = type_name[len(prefix) : -1]
        return self.first_slang_generic_argument(args)

    def first_slang_generic_argument(self, args):
        depth = 0
        for index, char in enumerate(args):
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            elif char == "," and depth == 0:
                return args[:index].strip()
        return args.strip()

    def slang_implicit_stage_parameters(self, body_statements, shader_type, param_list):
        candidates = self.slang_implicit_stage_parameter_candidates(shader_type)
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
        self, body_statements, shader_type, param_list
    ):
        candidates = self.slang_implicit_stage_parameter_candidates(shader_type)
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

    def slang_implicit_stage_parameter_candidates(self, shader_type):
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
        lines = statement.splitlines()
        return "\n".join(
            self.indent() + line if line and not line[0].isspace() else line
            for line in lines
        )

    def generate_statement(self, node):
        """Render a single CrossGL statement as Slang code."""
        if isinstance(node, ReturnNode):
            if node.value is None:
                return "return;"
            return (
                "return "
                f"{self.generate_expression_with_expected(node.value, self.current_function_return_type)};"
            )
        elif isinstance(node, AssignmentNode):
            return self.generate_assignment(node) + ";"
        elif isinstance(node, ExpressionStatementNode):
            return self.generate_expression(node.expression) + ";"
        elif isinstance(node, VariableNode):
            initial_value = getattr(node, "initial_value", getattr(node, "value", None))
            var_type = self.variable_declaration_type(node, initial_value)
            self.register_variable_type(node.name, var_type, node)
            declaration = self.format_declaration(var_type, node.name, node)
            if initial_value is not None:
                initial_expr = self.generate_expression_with_expected(
                    initial_value,
                    self.initializer_expected_type(var_type),
                )
                return f"{declaration} = {initial_expr};"
            return f"{declaration};"
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
            return self.generate_expression(node) + ";"

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
            left_type = self.expression_result_type(expr.left)
            right_type = self.expression_result_type(expr.right)
            if self.is_vector_value_type(left_type):
                return left_type
            if self.is_vector_value_type(right_type):
                return right_type
            if left_type == "float" or right_type == "float":
                return "float"
            return left_type or right_type
        if isinstance(expr, UnaryOpNode):
            return self.expression_result_type(expr.operand)
        if isinstance(expr, AssignmentNode):
            return self.expression_result_type(getattr(expr, "left", None))
        if isinstance(expr, ArrayAccessNode):
            array_type = self.type_name_string(self.expression_result_type(expr.array))
            if array_type and "[" in array_type and "]" in array_type:
                base_type, _ = split_array_type_suffix(array_type)
                return base_type
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
            return None
        if isinstance(expr, FunctionCallNode):
            func_expr = getattr(expr, "function", None) or getattr(expr, "name", None)
            func_name = getattr(func_expr, "name", func_expr)
            if func_name == "imageLoad" and getattr(expr, "args", None):
                return self.image_resource_element_type(
                    self.image_resource_type(expr.args[0])
                )
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
            elements = ", ".join(
                self.generate_expression(element) for element in node.elements
            )
            return f"{{{elements}}}"
        elif isinstance(node, MemberAccessNode):
            hull_output_alias = self.slang_hull_output_member_alias(node)
            if hull_output_alias is not None:
                return hull_output_alias
            obj = self.generate_expression(node.object)
            return f"{obj}.{node.member}"
        elif isinstance(node, BinaryOpNode):
            return self.generate_binary_expression(node)
        elif isinstance(node, FunctionCallNode):
            func_expr = getattr(node, "function", None)
            if func_expr is None:
                func_expr = node.name
            if hasattr(func_expr, "name"):
                callee = func_expr.name
            elif isinstance(func_expr, str):
                callee = func_expr
            else:
                callee = self.generate_expression(func_expr)
            if callee not in self.user_function_names:
                resource_call = self.generate_resource_call(callee, node.args)
                if resource_call is not None:
                    return resource_call
            if callee == "mix" and callee not in self.user_function_names:
                bool_mix = self.generate_bool_mix_call(node.args)
                if bool_mix is not None:
                    return bool_mix
            if callee == "lambda":
                lambda_expr = self.generate_lambda_expression(node.args)
                if lambda_expr is not None:
                    return lambda_expr
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
            true_expr = self.generate_expression(node.true_expr)
            false_expr = self.generate_expression(node.false_expr)
            return f"({condition} ? {true_expr} : {false_expr})"
        elif isinstance(node, str):
            return node
        else:
            return str(node)

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
        condition = self.generate_expression(
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

        return result

    def generate_for(self, node):
        init = self.generate_statement(node.init).rstrip(";") if node.init else ""
        condition = self.generate_expression(node.condition) if node.condition else ""
        update = self.generate_statement(node.update).rstrip(";") if node.update else ""

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
        condition = self.generate_expression(node.condition)
        result = f"while ({condition})\n{{\n"

        self.indent_level += 1
        for stmt in self.get_statements(node.body):
            result += self.emit_statement(stmt) + "\n"
        self.indent_level -= 1

        result += self.indent() + "}"
        return result

    def generate_do_while(self, node):
        condition = self.generate_expression(node.condition)
        result = "do\n{\n"

        self.indent_level += 1
        for stmt in self.get_statements(node.body):
            result += self.emit_statement(stmt) + "\n"
        self.indent_level -= 1

        result += self.indent() + f"}} while ({condition});"
        return result

    def generate_switch(self, node):
        expression = self.generate_expression(node.expression)
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
        return result

    def generate_match(self, node):
        expression = self.generate_expression(getattr(node, "expression", None))
        result = f"switch ({expression})\n{{\n"

        arms = getattr(node, "arms", []) or []
        if not self.validate_match_arms(arms):
            raise ValueError(
                "Unsupported match arm for Slang codegen; only unguarded "
                "literal patterns and a final wildcard can be lowered to switch"
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
        return result

    def is_supported_match_arm(self, arm):
        if getattr(arm, "guard", None) is not None:
            return False
        pattern = getattr(arm, "pattern", None)
        return isinstance(pattern, (LiteralPatternNode, WildcardPatternNode))

    def validate_match_arms(self, arms):
        wildcard_index = None
        for index, arm in enumerate(arms):
            if not self.is_supported_match_arm(arm):
                return False
            if isinstance(getattr(arm, "pattern", None), WildcardPatternNode):
                if wildcard_index is not None:
                    return False
                wildcard_index = index
        return wildcard_index is None or wildcard_index == len(arms) - 1

    def statement_body_terminates(self, body):
        statements = self.get_statements(body)
        if not statements:
            return False
        return isinstance(statements[-1], (BreakNode, ContinueNode, ReturnNode))

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
        }

        return type_map.get(type_name, type_name)

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
        if hasattr(value, "name"):
            return str(value.name)
        if hasattr(value, "value"):
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

    def map_resource_type_with_format(self, type_name, node=None):
        type_name = self.type_name_string(type_name)
        if type_name is None:
            return self.convert_type(type_name)

        if "[" in type_name and "]" in type_name:
            base_type, array_suffix = split_array_type_suffix(type_name)
            mapped_base = self.map_image_base_type_with_format(base_type, node)
            return f"{mapped_base}{array_suffix}"
        return self.map_image_base_type_with_format(type_name, node)

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
        if isinstance(type_name, str) and type_name.startswith("uint"):
            return "0u"
        if isinstance(type_name, str) and type_name.startswith("int"):
            return "0"
        return "0.0"

    def image_load_expression(self, args):
        image_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[1])
        if len(args) >= 3:
            sample = self.generate_expression(args[2])
            load_expr = f"{image_name}[{coord}, {sample}]"
        else:
            load_expr = f"{image_name}[{coord}]"

        image_type = self.image_resource_type(args[0])
        element_type = self.image_resource_element_type(image_type)
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
        if len(args) >= 4:
            sample = self.generate_expression(args[2])
            value = self.image_store_value_expression(args[0], args[3])
            return f"{image_name}[{coord}, {sample}] = {value}"

        value = self.image_store_value_expression(args[0], args[2])
        return f"{image_name}[{coord}] = {value}"

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
        helper_name = self.helper_function_name(base_helper_name)

        self.register_helper_function(
            helper_name,
            self.build_image_atomic_helper(helper_name, operation, image_type),
        )

        image_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[1])
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

        if func_name in {"texture", "textureLod", "textureGrad"}:
            sample_args = self.sampled_texture_args(args)
            if sample_args is None:
                return None

            texture_name, coord, extra_args = sample_args
            if func_name == "texture":
                if extra_args:
                    bias = self.generate_expression(extra_args[0])
                    return f"{texture_name}.SampleBias({coord}, {bias})"
                return f"{texture_name}.Sample({coord})"

            if func_name == "textureLod" and extra_args:
                lod = self.generate_expression(extra_args[0])
                return f"{texture_name}.SampleLevel({coord}, {lod})"

            if func_name == "textureGrad" and len(extra_args) >= 2:
                ddx = self.generate_expression(extra_args[0])
                ddy = self.generate_expression(extra_args[1])
                return f"{texture_name}.SampleGrad({coord}, {ddx}, {ddy})"

            return None

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
            fetch_args = self.sampled_texture_args(args)
            if fetch_args is None:
                return None
            texture_name, coord, extra_args = fetch_args
            if not extra_args:
                return None

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
            return None

        resource_name = self.generate_expression(args[0])
        resource_type = self.resource_base_type(self.get_expression_type(args[0]))
        spec = self.dimension_query_spec(resource_type)
        if spec is None:
            return None

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

    def generate_sample_count_query(self, func_name, args):
        if not args:
            return None

        resource_name = self.generate_expression(args[0])
        resource_type = self.resource_base_type(self.get_expression_type(args[0]))
        spec = self.dimension_query_spec(resource_type)
        if spec is None or not spec["samples"]:
            return None

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

    def generate_texture_offset(self, func_name, args):
        sample_args = self.sampled_texture_args(args)
        if sample_args is None:
            return self.unsupported_texture_offset_call(
                func_name, "requires texture and coordinate arguments"
            )

        texture_name, coord, extra_args = sample_args

        if func_name == "textureOffset":
            if len(extra_args) != 1:
                return self.unsupported_texture_offset_call(
                    func_name, "requires one offset argument"
                )
            offset = self.generate_expression(extra_args[0])
            return f"{texture_name}.Sample({coord}, {offset})"

        if func_name == "textureLodOffset":
            if len(extra_args) != 2:
                return self.unsupported_texture_offset_call(
                    func_name, "requires lod and offset arguments"
                )
            lod = self.generate_expression(extra_args[0])
            offset = self.generate_expression(extra_args[1])
            return f"{texture_name}.SampleLevel({coord}, {lod}, {offset})"

        if len(extra_args) != 3:
            return self.unsupported_texture_offset_call(
                func_name, "requires gradient x, gradient y, and offset arguments"
            )
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
                bias = self.generate_expression(extra_args[0])
                return f"{texture_name}.SampleBias({projected_coord}, {bias})"
            return self.unsupported_texture_projected_call(
                func_name, "accepts at most one bias argument"
            )

        if func_name == "textureProjOffset":
            if len(extra_args) == 1:
                offset = self.generate_expression(extra_args[0])
                return f"{texture_name}.Sample({projected_coord}, {offset})"
            if len(extra_args) == 2:
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
            lod = self.generate_expression(extra_args[0])
            return f"{texture_name}.SampleLevel({projected_coord}, {lod})"

        if func_name == "textureProjLodOffset":
            if len(extra_args) != 2:
                return self.unsupported_texture_projected_call(
                    func_name, "requires lod and offset arguments"
                )
            lod = self.generate_expression(extra_args[0])
            offset = self.generate_expression(extra_args[1])
            return f"{texture_name}.SampleLevel({projected_coord}, {lod}, {offset})"

        if func_name == "textureProjGrad":
            if len(extra_args) != 2:
                return self.unsupported_texture_projected_call(
                    func_name, "requires gradient x and gradient y arguments"
                )
            ddx = self.generate_expression(extra_args[0])
            ddy = self.generate_expression(extra_args[1])
            return f"{texture_name}.SampleGrad({projected_coord}, {ddx}, {ddy})"

        if len(extra_args) != 3:
            return self.unsupported_texture_projected_call(
                func_name, "requires gradient x, gradient y, and offset arguments"
            )
        ddx = self.generate_expression(extra_args[0])
        ddy = self.generate_expression(extra_args[1])
        offset = self.generate_expression(extra_args[2])
        return f"{texture_name}.SampleGrad({projected_coord}, {ddx}, {ddy}, {offset})"

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

        texture_name, coord, extra_args = gather_args
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
            offset_args = [extra_args[0]]
            if len(extra_args) == 2:
                component_arg = extra_args[1]
        else:
            offset_args, component_arg = self.texture_gather_offsets_args(extra_args)
            if offset_args is None:
                return self.unsupported_texture_gather_call(
                    func_name,
                    "requires a typed offsets array or four offset arguments",
                )

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
            offset = self.generate_expression(extra_args[0])
            return f"{texture_name}.SampleCmp({coord}, {compare}, {offset})"

        if func_name == "textureCompareLod":
            if len(extra_args) != 1:
                return self.unsupported_texture_compare_call(
                    func_name, "requires one lod argument"
                )
            lod = self.generate_expression(extra_args[0])
            return f"{texture_name}.SampleCmpLevel({coord}, {compare}, {lod})"

        if len(extra_args) != 2:
            return self.unsupported_texture_compare_call(
                func_name, "requires gradient x and gradient y arguments"
            )
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

    def is_shadow_compare_resource(self, node):
        resource_type = self.resource_base_type(self.get_expression_type(node))
        return resource_type is None or resource_type in {
            "sampler2DShadow",
            "sampler2DArrayShadow",
            "samplerCubeShadow",
            "samplerCubeArrayShadow",
        }

    def unsupported_texture_compare_call(self, func_name, reason):
        return f"/* unsupported Slang shadow compare: {func_name} {reason} */ 0.0"

    def unsupported_texture_gather_compare_call(self, func_name, reason):
        return (
            f"/* unsupported Slang shadow gather compare: "
            f"{func_name} {reason} */ float4(0.0)"
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
        type_name = self.get_expression_type(node)
        return isinstance(type_name, str) and "[" in type_name and "]" in type_name

    def generate_texture_query_levels(self, args):
        if not args:
            return None

        resource_name = self.generate_expression(args[0])
        resource_type = self.resource_base_type(self.get_expression_type(args[0]))
        spec = self.dimension_query_spec(resource_type)
        if spec is None or not spec["mip"]:
            return None

        base_helper_name = f"cgl_textureQueryLevels_{resource_type}"
        helper_name = self.helper_function_name(base_helper_name)
        self.register_helper_function(
            helper_name,
            self.build_texture_query_levels_helper(helper_name, resource_type, spec),
        )
        return f"{helper_name}({resource_name})"

    def generate_texture_query_lod(self, args):
        query_args = self.texture_query_lod_args(args)
        if query_args is None:
            return None

        texture_name, coord = query_args
        unclamped = f"{texture_name}.CalculateLevelOfDetailUnclamped({coord})"
        clamped = f"{texture_name}.CalculateLevelOfDetail({coord})"
        return f"float2({unclamped}, {clamped})"

    def texture_query_lod_args(self, args):
        coord_index = 2 if self.is_explicit_sampler_argument(args) else 1
        if len(args) <= coord_index:
            return None

        resource_type = self.resource_base_type(self.get_expression_type(args[0]))
        if not self.is_lod_query_sampler_type(resource_type):
            return None

        texture_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[coord_index])
        return texture_name, coord

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
        if len(args) < 3:
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

    def texel_fetch_coord_constructor(self, type_name):
        base_type = self.resource_base_type(type_name)
        if base_type in {"sampler1D", "sampler1DArray"}:
            return "int2" if base_type == "sampler1D" else "int3"
        if base_type in {"sampler3D", "sampler2DArray"}:
            return "int4"
        return "int3"
