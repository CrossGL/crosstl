"""Reverse code generator that emits CrossGL from Vulkan SPIR-V AST nodes."""

from ..common_ast import InitializerListNode
from .VulkanAst import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    BreakNode,
    CaseNode,
    ContinueNode,
    DefaultNode,
    DiscardNode,
    DoWhileNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    LayoutNode,
    MemberAccessNode,
    MethodCallNode,
    ReturnNode,
    StructNode,
    SwitchNode,
    TernaryOpNode,
    UnaryOpNode,
    UniformNode,
    VariableNode,
    WhileNode,
)


class VulkanToCrossGLConverter:
    """Serialize Vulkan backend AST nodes back into CrossGL source."""

    RAY_STORAGE_QUALIFIER_ATTRIBUTES = {
        "callabledataext": "callableDataEXT",
        "callabledatainext": "callableDataInEXT",
        "hitattributeext": "hitAttributeEXT",
        "raypayloadext": "rayPayloadEXT",
        "raypayloadinext": "rayPayloadInEXT",
    }
    MESH_STORAGE_QUALIFIER_ATTRIBUTES = {
        "taskpayloadsharedext": "taskPayloadSharedEXT",
        "taskpayloadsharednv": "taskPayloadSharedNV",
    }

    def __init__(self):
        self.type_map = {
            "void": "void",
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
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
            "int": "int",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "uint": "uint",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "bool": "bool",
            "bvec2": "bool2",
            "bvec3": "bool3",
            "bvec4": "bool4",
            "float": "float",
            "double": "double",
            "subpassInput": "subpassInput",
            "subpassInputMS": "subpassInputMS",
            "sampler": "sampler",
            "sampler2D": "Texture2D",
            "isampler1D": "isampler1D",
            "isampler1DArray": "isampler1DArray",
            "isampler2D": "isampler2D",
            "isampler2DArray": "isampler2DArray",
            "isampler2DMS": "isampler2DMS",
            "isampler2DMSArray": "isampler2DMSArray",
            "isampler3D": "isampler3D",
            "isamplerBuffer": "isamplerBuffer",
            "isamplerCube": "isamplerCube",
            "isamplerCubeArray": "isamplerCubeArray",
            "usampler1D": "usampler1D",
            "usampler1DArray": "usampler1DArray",
            "usampler2D": "usampler2D",
            "usampler2DArray": "usampler2DArray",
            "usampler2DMS": "usampler2DMS",
            "usampler2DMSArray": "usampler2DMSArray",
            "usampler3D": "usampler3D",
            "usamplerBuffer": "usamplerBuffer",
            "usamplerCube": "usamplerCube",
            "usamplerCubeArray": "usamplerCubeArray",
            "samplerCube": "TextureCube",
            "sampler3D": "Texture3D",
            "sampler2DArray": "Texture2DArray",
            "samplerCubeArray": "TextureCubeArray",
            "sampler1D": "Texture1D",
            "sampler1DArray": "Texture1DArray",
            "sampler2DMS": "Texture2DMS",
            "sampler2DMSArray": "Texture2DMSArray",
            "samplerBuffer": "Buffer",
            "image1D": "RWTexture1D",
            "image1DArray": "RWTexture1DArray",
            "image2D": "RWTexture2D",
            "image2DArray": "RWTexture2DArray",
            "image2DMS": "RWTexture2DMS",
            "image2DMSArray": "RWTexture2DMSArray",
            "image3D": "RWTexture3D",
            "imageBuffer": "RWBuffer",
            "imageCube": "RWTextureCube",
            "imageCubeArray": "RWTextureCubeArray",
            "iimage1D": "RWTexture1D<int>",
            "iimage1DArray": "RWTexture1DArray<int>",
            "iimage2D": "RWTexture2D<int>",
            "iimage2DArray": "RWTexture2DArray<int>",
            "iimage2DMS": "RWTexture2DMS<int>",
            "iimage2DMSArray": "RWTexture2DMSArray<int>",
            "iimage3D": "RWTexture3D<int>",
            "iimageBuffer": "RWBuffer<int>",
            "iimageCube": "RWTextureCube<int>",
            "iimageCubeArray": "RWTextureCubeArray<int>",
            "uimage1D": "RWTexture1D<uint>",
            "uimage1DArray": "RWTexture1DArray<uint>",
            "uimage2D": "RWTexture2D<uint>",
            "uimage2DArray": "RWTexture2DArray<uint>",
            "uimage2DMS": "RWTexture2DMS<uint>",
            "uimage2DMSArray": "RWTexture2DMSArray<uint>",
            "uimage3D": "RWTexture3D<uint>",
            "uimageBuffer": "RWBuffer<uint>",
            "uimageCube": "RWTextureCube<uint>",
            "uimageCubeArray": "RWTextureCubeArray<uint>",
            "atomic_uint": "RWStructuredBuffer<uint>",
        }
        self.semantic_map = {
            # Vertex inputs
            "gl_VertexID": "SV_VertexID",
            "gl_InstanceID": "SV_InstanceID",
            # Vertex outputs
            "gl_Position": "SV_Position",
            # Fragment inputs
            "gl_FragCoord": "SV_Position",
            "gl_FrontFacing": "SV_IsFrontFace",
            "gl_PointCoord": "SV_PointCoord",
            "gl_PrimitiveID": "SV_PrimitiveID",
            # Fragment outputs
            "gl_FragColor": "SV_Target",
            "gl_FragData[0]": "SV_Target0",
            "gl_FragData[1]": "SV_Target1",
            "gl_FragData[2]": "SV_Target2",
            "gl_FragData[3]": "SV_Target3",
            "gl_FragDepth": "SV_Depth",
        }
        self.bitwise_op_map = {
            "&": "&",
            "|": "|",
            "^": "^",
            "~": "~",
            "<<": "<<",
            ">>": ">>",
        }
        self.indentation = 0
        self.code = []
        self.flattened_uniform_block_instances = {}
        self.storage_buffer_type_names = {}
        self.storage_buffer_name_signatures = {}
        self.emitted_storage_buffer_type_names = set()
        self.used_global_declaration_names = set()
        self.non_renamable_global_declaration_names = set()
        self.renamed_spirv_global_names = {}
        self.spirv_module_shader_stages = set()

    def get_indent(self):
        return "    " * self.indentation

    def generate(self, ast):
        self.flattened_uniform_block_instances = {}
        self.storage_buffer_type_names = {}
        self.storage_buffer_name_signatures = self.initial_type_name_signatures(ast)
        self.emitted_storage_buffer_type_names = set(
            self.storage_buffer_name_signatures
        )
        self.used_global_declaration_names = set(self.storage_buffer_name_signatures)
        self.non_renamable_global_declaration_names = (
            self.collect_non_renamable_global_declaration_names(ast)
        )
        self.renamed_spirv_global_names = {}
        self.spirv_module_shader_stages = self.collect_spirv_module_shader_stages(ast)
        code = "shader main {\n"
        stage_interface_layouts = self.spirv_stage_interface_layouts(ast)
        compute_layouts = [
            node
            for node in getattr(ast, "global_variables", [])
            if isinstance(node, LayoutNode) and self.is_compute_layout(node)
        ]
        fragment_execution_layouts = [
            node
            for node in getattr(ast, "global_variables", [])
            if isinstance(node, LayoutNode) and self.is_fragment_execution_layout(node)
        ]
        top_level_nodes = []
        top_level_nodes.extend(getattr(ast, "structs", []))
        top_level_nodes.extend(getattr(ast, "global_variables", []))
        top_level_nodes.extend(getattr(ast, "functions", []))

        for node in top_level_nodes:
            if isinstance(node, LayoutNode):
                if self.is_spirv_stage_interface_layout(node):
                    continue
                if self.is_compute_layout(node) or self.is_fragment_execution_layout(
                    node
                ):
                    continue
                code += self.generate_layout(node)
            elif isinstance(node, StructNode):
                code += self.generate_struct(node)
            elif isinstance(node, UniformNode):
                code += self.generate_uniform(node)
            elif isinstance(node, VariableNode):
                code += f"    {self.generate_global_variable_declaration(node)};\n"
            elif isinstance(node, AssignmentNode):
                code += f"    {self.generate_global_assignment(node)};\n"
            elif isinstance(node, FunctionNode):
                shader_stage = self.function_shader_stage(node)
                if shader_stage is not None or node.name == "main":
                    if shader_stage is None:
                        shader_stage = self.infer_main_shader_stage(
                            node, compute_layouts
                        )

                    if shader_stage == "compute":
                        code += "    // Compute Shader\n"
                        code += "    compute {\n"
                        stage_compute_layouts = (
                            compute_layouts + self.spirv_compute_execution_layouts(node)
                        )
                        for layout in stage_compute_layouts:
                            code += self.generate_compute_layout(layout)
                        code += self.generate_stage_interface_layouts(
                            node, stage_interface_layouts
                        )
                        code += self.generate_function(node)
                        code += "    }\n\n"
                    elif shader_stage == "vertex":
                        code += "    // Vertex Shader\n"
                        code += "    vertex {\n"
                        code += self.generate_stage_interface_layouts(
                            node, stage_interface_layouts
                        )
                        code += self.generate_function(node)
                        code += "    }\n\n"
                    elif shader_stage == "fragment":
                        code += "    // Fragment Shader\n"
                        code += "    fragment {\n"
                        for layout in (
                            fragment_execution_layouts
                            + self.spirv_fragment_execution_layouts(node)
                        ):
                            code += self.generate_fragment_execution_layout(layout)
                        code += self.generate_stage_interface_layouts(
                            node, stage_interface_layouts
                        )
                        code += self.generate_function(node)
                        code += "    }\n\n"
                    elif shader_stage == "geometry":
                        code += "    // Geometry Shader\n"
                        code += "    geometry {\n"
                        for layout in self.spirv_geometry_execution_layouts(node):
                            code += self.generate_stage_layout(layout)
                        code += self.generate_stage_interface_layouts(
                            node, stage_interface_layouts
                        )
                        code += self.generate_function(node)
                        code += "    }\n\n"
                    elif shader_stage == "tessellation_control":
                        code += "    // Tessellation Control Shader\n"
                        code += "    tessellation_control {\n"
                        for layout in self.spirv_tessellation_control_execution_layouts(
                            node
                        ):
                            code += self.generate_stage_layout(layout)
                        code += self.generate_stage_interface_layouts(
                            node, stage_interface_layouts
                        )
                        code += self.generate_function(node)
                        code += "    }\n\n"
                    elif shader_stage == "tessellation_evaluation":
                        code += "    // Tessellation Evaluation Shader\n"
                        code += "    tessellation_evaluation {\n"
                        for (
                            layout
                        ) in self.spirv_tessellation_evaluation_execution_layouts(node):
                            code += self.generate_stage_layout(layout)
                        code += self.generate_stage_interface_layouts(
                            node, stage_interface_layouts
                        )
                        code += self.generate_function(node)
                        code += "    }\n\n"
                    else:
                        code += self.generate_function(node)
                else:
                    code += self.generate_function(node)

        code += "}\n"
        return code

    def spirv_stage_interface_layouts(self, ast):
        layouts = [
            node
            for node in getattr(ast, "global_variables", []) or []
            if self.is_spirv_stage_interface_layout(node)
        ]
        if not layouts:
            return {}

        by_interface_id = {}
        for layout in layouts:
            by_interface_id.setdefault(
                self.spirv_interface_layout_id(layout), []
            ).append(layout)

        functions_by_spirv_id = {
            getattr(function, "spirv_id", None): function
            for function in getattr(ast, "functions", []) or []
            if getattr(function, "spirv_id", None)
        }
        duplicated_interface_ids = self.spirv_duplicated_entry_point_interface_ids(ast)
        stage_layouts = {}
        for function in getattr(ast, "functions", []) or []:
            interface_ids = []
            for entry_point in getattr(function, "spirv_entry_points", []) or []:
                interface_ids.extend(entry_point.get("interface_ids", []) or [])
            if not interface_ids:
                entry_point = getattr(function, "spirv_entry_point", None)
                interface_ids.extend((entry_point or {}).get("interface_ids", []) or [])
            if not interface_ids:
                continue

            referenced_ids = self.spirv_function_reachable_referenced_ids(
                function, functions_by_spirv_id
            )
            referenced_duplicate_interface_ids = {
                interface_id
                for interface_id in interface_ids
                if interface_id in duplicated_interface_ids
                if self.spirv_interface_id_is_referenced(interface_id, referenced_ids)
            }
            if referenced_duplicate_interface_ids:
                interface_ids = [
                    interface_id
                    for interface_id in interface_ids
                    if interface_id not in duplicated_interface_ids
                    or interface_id in referenced_duplicate_interface_ids
                ]

            seen = set()
            for interface_id in interface_ids:
                for layout in by_interface_id.get(interface_id, []):
                    layout_key = id(layout)
                    if layout_key in seen:
                        continue
                    seen.add(layout_key)
                    stage_layouts.setdefault(id(function), []).append(layout)

        return stage_layouts

    def spirv_duplicated_entry_point_interface_ids(self, ast):
        interface_id_counts = {}
        for function in getattr(ast, "functions", []) or []:
            for entry_point in getattr(function, "spirv_entry_points", []) or []:
                for interface_id in set(entry_point.get("interface_ids", []) or []):
                    interface_id_counts[interface_id] = (
                        interface_id_counts.get(interface_id, 0) + 1
                    )
        return {
            interface_id
            for interface_id, count in interface_id_counts.items()
            if count > 1
        }

    def spirv_function_reachable_referenced_ids(
        self, function, functions_by_spirv_id, visited=None
    ):
        visited = visited or set()
        function_id = getattr(function, "spirv_id", None) or id(function)
        if function_id in visited:
            return set()
        visited.add(function_id)

        referenced_ids = set()
        called_function_ids = set()
        for instruction in getattr(function, "spirv_raw_instructions", []) or []:
            operands = instruction.get("operands", []) or []
            for operand in operands:
                if isinstance(operand, str) and operand.startswith("%"):
                    referenced_ids.add(operand)
            if instruction.get("opcode") == "OpFunctionCall" and len(operands) >= 2:
                called_function_ids.add(operands[1])

        for called_function_id in called_function_ids:
            called_function = functions_by_spirv_id.get(called_function_id)
            if called_function is None:
                continue
            referenced_ids.update(
                self.spirv_function_reachable_referenced_ids(
                    called_function, functions_by_spirv_id, visited
                )
            )
        return referenced_ids

    def spirv_interface_id_is_referenced(self, interface_id, referenced_ids):
        interface_base_id = str(interface_id).split(".", 1)[0]
        return interface_base_id in referenced_ids

    def is_spirv_stage_interface_layout(self, node):
        if not isinstance(node, LayoutNode):
            return False
        if (node.layout_type or "").lower() not in {"in", "out"}:
            return False
        return getattr(node, "spirv_id", None) is not None

    def spirv_interface_layout_id(self, node):
        return str(getattr(node, "spirv_id", "")).split(".", 1)[0]

    def generate_stage_interface_layouts(self, function, stage_interface_layouts):
        code = ""
        layouts = stage_interface_layouts.get(id(function), [])
        clip_position_location = self.crossgl_vertex_clip_position_location(
            function, layouts
        )
        for layout in layouts:
            override_name = None
            forced_builtin = None
            qualifiers = None
            suppressed_qualifiers = self.suppressed_stage_interface_qualifiers(
                function, layout
            )
            if self.is_crossgl_vertex_clip_position_layout(
                function, layout, clip_position_location
            ):
                override_name = "gl_Position"
                forced_builtin = "Position"
                suppressed_qualifiers = {
                    *suppressed_qualifiers,
                    "location",
                    "component",
                    "index",
                }
                self.record_stage_interface_renamed_spirv_id(layout, override_name)
            elif clip_position_location is not None and self.is_stage_output_layout(
                layout
            ):
                qualifiers = self.shift_stage_interface_locations(
                    layout, clip_position_location
                )

            code += self.indent_generated_layout(
                self.generate_layout(
                    layout,
                    suppressed_interface_qualifiers=suppressed_qualifiers,
                    override_interface_qualifiers=qualifiers,
                    override_variable_name=override_name,
                    forced_builtin=forced_builtin,
                )
            )
        return code

    def crossgl_vertex_clip_position_location(self, function, layouts):
        if self.function_shader_stage(function) != "vertex":
            return None

        for layout in layouts:
            if not self.is_crossgl_clip_position_output_layout(layout):
                continue
            location = self.layout_location(layout)
            if location == 0:
                return location
        return None

    def is_crossgl_vertex_clip_position_layout(
        self, function, layout, clip_position_location
    ):
        if clip_position_location is None:
            return False
        if self.function_shader_stage(function) != "vertex":
            return False
        if not self.is_crossgl_clip_position_output_layout(layout):
            return False
        return self.layout_location(layout) == clip_position_location

    def is_crossgl_clip_position_output_layout(self, layout):
        if not self.is_stage_output_layout(layout):
            return False
        if str(getattr(layout, "data_type", "")).lower() not in {"vec4", "float4"}:
            return False
        return self.is_clip_position_name(getattr(layout, "variable_name", None))

    def is_stage_output_layout(self, layout):
        if not isinstance(layout, LayoutNode):
            return False
        return (getattr(layout, "layout_type", None) or "").lower() == "out"

    def is_clip_position_name(self, name):
        normalized = "".join(char for char in str(name or "").lower() if char.isalnum())
        return normalized == "clipposition" or (
            normalized.startswith("crossglvertexoutput")
            and normalized.endswith("clipposition")
        )

    def layout_location(self, layout):
        for name, value in getattr(layout, "qualifiers", []) or []:
            if str(name).lower() != "location":
                continue
            try:
                return int(str(value), 0)
            except (TypeError, ValueError):
                return None
        return None

    def shift_stage_interface_locations(self, layout, after_location):
        shifted = []
        changed = False
        for name, value in getattr(layout, "qualifiers", []) or []:
            if str(name).lower() != "location":
                shifted.append((name, value))
                continue
            try:
                location = int(str(value), 0)
            except (TypeError, ValueError):
                shifted.append((name, value))
                continue
            if location <= after_location:
                shifted.append((name, value))
                continue
            shifted.append((name, str(location - 1)))
            changed = True
        return shifted if changed else None

    def record_stage_interface_renamed_spirv_id(self, layout, generated_name):
        spirv_id = getattr(layout, "spirv_id", None)
        if not spirv_id or "." in str(spirv_id):
            return
        self.renamed_spirv_global_names[spirv_id] = self.declaration_base_name(
            generated_name
        )

    def suppressed_stage_interface_qualifiers(self, function, layout):
        layout_type = (getattr(layout, "layout_type", None) or "").lower()
        if layout_type != "in" or self.function_shader_stage(function) != "geometry":
            return set()
        if not self.spirv_module_shader_stages.intersection(
            {"vertex", "tessellation_control", "tessellation_evaluation"}
        ):
            return set()
        if "[" not in str(getattr(layout, "variable_name", "")):
            return set()
        return {"location", "component", "index"}

    def collect_spirv_module_shader_stages(self, ast):
        return {
            stage
            for function in getattr(ast, "functions", []) or []
            for stage in [self.function_shader_stage(function)]
            if stage is not None
        }

    def indent_generated_layout(self, layout_code):
        return "".join(
            f"    {line}" if line.strip() else line
            for line in layout_code.splitlines(keepends=True)
        )

    def is_compute_layout(self, node):
        if not isinstance(node, LayoutNode):
            return False
        if (node.layout_type or "").lower() != "in":
            return False
        return any(
            name in {"local_size_x", "local_size_y", "local_size_z"}
            for name, _ in node.qualifiers
        )

    def function_shader_stage(self, node):
        return self.spirv_execution_model_stage(
            getattr(node, "spirv_execution_model", None)
        )

    def spirv_execution_model_stage(self, execution_model):
        return {
            "Vertex": "vertex",
            "Fragment": "fragment",
            "Geometry": "geometry",
            "TessellationControl": "tessellation_control",
            "TessellationEvaluation": "tessellation_evaluation",
            "GLCompute": "compute",
        }.get(execution_model)

    def infer_main_shader_stage(self, node, compute_layouts):
        if compute_layouts:
            return "compute"
        for stmt in node.body:
            if self.is_position_assignment(stmt):
                return "vertex"
        return "fragment"

    def spirv_compute_execution_layouts(self, node):
        layouts = []
        for mode in getattr(node, "spirv_execution_modes", []) or []:
            if mode.get("mode") not in {"LocalSize", "LocalSizeId"}:
                continue
            operands = mode.get("operands", [])
            if len(operands) < 3:
                continue
            if mode.get("mode") == "LocalSizeId":
                operands = [
                    self.spirv_execution_mode_id_operand(node, operand)
                    for operand in operands
                ]
            layouts.append(
                LayoutNode(
                    [
                        ("local_size_x", operands[0]),
                        ("local_size_y", operands[1]),
                        ("local_size_z", operands[2]),
                    ],
                    layout_type="in",
                )
            )
        return layouts

    def spirv_execution_mode_id_operand(self, node, operand):
        names = getattr(node, "spirv_names", {}) or {}
        constants = getattr(node, "spirv_constants", {}) or {}
        if operand in names and names[operand]:
            return names[operand]
        if operand in constants:
            return self.generate_expression(constants[operand])
        if isinstance(operand, str) and operand.startswith("%"):
            return operand.lstrip("%")
        return operand

    def spirv_fragment_execution_layouts(self, node):
        layouts = []
        for mode in getattr(node, "spirv_execution_modes", []) or []:
            if mode.get("mode") == "EarlyFragmentTests":
                layouts.append(
                    LayoutNode([("early_fragment_tests", None)], layout_type="in")
                )
        return layouts

    def spirv_geometry_execution_layouts(self, node):
        input_qualifiers = []
        output_qualifiers = []
        input_modes = {
            "InputPoints": "points",
            "InputLines": "lines",
            "InputLinesAdjacency": "lines_adjacency",
            "Triangles": "triangles",
            "InputTrianglesAdjacency": "triangles_adjacency",
        }
        output_modes = {
            "OutputPoints": "points",
            "OutputLineStrip": "line_strip",
            "OutputTriangleStrip": "triangle_strip",
        }

        for mode in getattr(node, "spirv_execution_modes", []) or []:
            mode_name = mode.get("mode")
            operands = mode.get("operands", [])
            if mode_name in input_modes:
                input_qualifiers.append((input_modes[mode_name], None))
            elif mode_name == "Invocations" and operands:
                input_qualifiers.append(("invocations", operands[0]))
            elif mode_name in output_modes:
                output_qualifiers.append((output_modes[mode_name], None))
            elif mode_name == "OutputVertices" and operands:
                output_qualifiers.append(("max_vertices", operands[0]))

        layouts = []
        if input_qualifiers:
            layouts.append(LayoutNode(input_qualifiers, layout_type="in"))
        if output_qualifiers:
            layouts.append(LayoutNode(output_qualifiers, layout_type="out"))
        return layouts

    def spirv_tessellation_control_execution_layouts(self, node):
        qualifiers = []
        for mode in getattr(node, "spirv_execution_modes", []) or []:
            mode_name = mode.get("mode")
            operands = mode.get("operands", [])
            if mode_name == "OutputVertices" and operands:
                qualifiers.append(("vertices", operands[0]))

        if not qualifiers:
            return []
        return [LayoutNode(qualifiers, layout_type="out")]

    def spirv_tessellation_evaluation_execution_layouts(self, node):
        modes = {
            "Triangles": "triangles",
            "Quads": "quads",
            "Isolines": "isolines",
            "SpacingEqual": "equal_spacing",
            "SpacingFractionalEven": "fractional_even_spacing",
            "SpacingFractionalOdd": "fractional_odd_spacing",
            "VertexOrderCw": "cw",
            "VertexOrderCcw": "ccw",
            "PointMode": "point_mode",
        }
        qualifiers = []
        for mode in getattr(node, "spirv_execution_modes", []) or []:
            mapped_name = modes.get(mode.get("mode"))
            if mapped_name is not None:
                qualifiers.append((mapped_name, None))

        if not qualifiers:
            return []
        return [LayoutNode(qualifiers, layout_type="in")]

    def is_fragment_execution_layout(self, node):
        if not isinstance(node, LayoutNode):
            return False
        if (node.layout_type or "").lower() != "in":
            return False
        if getattr(node, "data_type", None) or getattr(node, "variable_name", None):
            return False
        qualifier_names = {str(name).lower() for name, _ in node.qualifiers}
        return bool(qualifier_names & self.fragment_execution_layout_qualifiers())

    def fragment_execution_layout_qualifiers(self):
        return {"early_fragment_tests"}

    def generate_compute_layout(self, node):
        qualifiers = ", ".join(
            f"{name} = {value}" if value is not None else name
            for name, value in node.qualifiers
        )
        return f"        layout({qualifiers}) in;\n"

    def generate_fragment_execution_layout(self, node):
        return self.generate_stage_layout(node)

    def generate_stage_layout(self, node):
        qualifiers = ", ".join(
            f"{name} = {value}" if value is not None else name
            for name, value in node.qualifiers
        )
        layout_type = (node.layout_type or "in").lower()
        return f"        layout({qualifiers}) {layout_type};\n"

    def is_position_assignment(self, stmt):
        if isinstance(stmt, AssignmentNode):
            lhs = self.assignment_left(stmt)
            if isinstance(lhs, str) and "gl_Position" in lhs:
                return True
            elif hasattr(lhs, "name") and "gl_Position" in lhs.name:
                return True
        return False

    def variable_type(self, node):
        return getattr(node, "vtype", getattr(node, "var_type", ""))

    def function_params(self, node):
        return getattr(node, "params", getattr(node, "parameters", []))

    def assignment_left(self, node):
        return getattr(node, "left", getattr(node, "name", None))

    def assignment_right(self, node):
        return getattr(node, "right", getattr(node, "value", None))

    def initial_type_name_signatures(self, ast):
        signatures = {}
        for struct in getattr(ast, "structs", []) or []:
            struct_name = getattr(struct, "name", None)
            if not struct_name:
                continue
            signatures.setdefault(struct_name, self.struct_signature(struct))
        return signatures

    def struct_signature(self, struct):
        signature = []
        for member in getattr(struct, "members", []) or []:
            if isinstance(member, VariableNode):
                signature.append((str(self.variable_type(member)), str(member.name)))
            elif isinstance(member, AssignmentNode):
                lhs = self.assignment_left(member)
                if isinstance(lhs, VariableNode):
                    signature.append((str(self.variable_type(lhs)), str(lhs.name)))
        return tuple(signature)

    def declaration_base_name(self, name):
        return str(name or "").split("[", 1)[0]

    def declaration_array_suffix(self, name):
        text = str(name or "")
        bracket_index = text.find("[")
        return "" if bracket_index < 0 else text[bracket_index:]

    def reserve_global_declaration_name(self, name):
        base_name = self.declaration_base_name(name)
        if base_name:
            self.used_global_declaration_names.add(base_name)

    def unique_global_declaration_name(self, preferred_name, suffix_source):
        preferred_name = self.declaration_base_name(preferred_name)
        if not preferred_name:
            preferred_name = "Declaration"
        if preferred_name not in self.used_global_declaration_names:
            self.used_global_declaration_names.add(preferred_name)
            return preferred_name

        suffix = self.storage_buffer_type_suffix(suffix_source)
        candidate = f"{preferred_name}_{suffix}"
        counter = 2
        while candidate in self.used_global_declaration_names:
            candidate = f"{preferred_name}_{suffix}_{counter}"
            counter += 1
        self.used_global_declaration_names.add(candidate)
        return candidate

    def unique_global_variable_name(self, preferred_name, suffix_source):
        unique_base = self.unique_global_declaration_name(
            self.declaration_base_name(preferred_name), suffix_source
        )
        return f"{unique_base}{self.declaration_array_suffix(preferred_name)}"

    def collect_non_renamable_global_declaration_names(self, ast):
        names = set()
        for node in getattr(ast, "global_variables", []) or []:
            if isinstance(node, LayoutNode):
                if self.is_flattened_uniform_block_layout(node):
                    continue
                names.add(self.layout_declaration_base_name(node))
            elif isinstance(node, UniformNode):
                names.add(self.declaration_base_name(node.name))
        return {name for name in names if name}

    def layout_declaration_base_name(self, node):
        if getattr(node, "variable_name", None):
            return self.declaration_base_name(node.variable_name)
        declaration = getattr(node, "declaration", None)
        if isinstance(declaration, AssignmentNode):
            declaration = self.assignment_left(declaration)
        if isinstance(declaration, VariableNode):
            return self.declaration_base_name(declaration.name)
        return None

    def is_flattened_uniform_block_layout(self, node):
        layout_type = (getattr(node, "layout_type", None) or "").lower()
        return layout_type == "uniform" and bool(getattr(node, "struct_fields", None))

    def generate_global_variable_declaration(self, node, extra_attributes=None):
        original_name = node.name
        generated_name = self.unique_global_variable_name(
            original_name, getattr(node, "spirv_id", original_name)
        )
        self.record_spirv_global_name(node, generated_name)
        return self.generate_variable_declaration(
            node, extra_attributes=extra_attributes, override_name=generated_name
        )

    def generate_global_assignment(self, node):
        lhs_node = self.assignment_left(node)
        if isinstance(lhs_node, VariableNode) and self.variable_type(lhs_node):
            rhs = self.generate_expression(self.assignment_right(node))
            operator = getattr(node, "operator", "=")
            lhs = self.generate_global_variable_declaration(lhs_node)
            return f"{lhs} {operator} {rhs}"
        return self.generate_assignment(node)

    def record_spirv_global_name(self, node, generated_name):
        spirv_id = getattr(node, "spirv_id", None)
        if spirv_id and generated_name != node.name:
            self.renamed_spirv_global_names[spirv_id] = self.declaration_base_name(
                generated_name
            )

    def generate_layout(
        self,
        node,
        suppressed_interface_qualifiers=None,
        override_interface_qualifiers=None,
        override_variable_name=None,
        forced_builtin=None,
    ):
        code = ""
        if (getattr(node, "layout_type", None) or "").lower() == "const":
            return self.generate_specialization_constant_layout(node)

        layout_type = node.layout_type.lower() if node.layout_type else ""

        if layout_type == "uniform":
            if node.struct_fields:
                block_name = self.unique_global_declaration_name(
                    node.block_name or node.variable_name or "UniformBuffer",
                    node.variable_name or node.spirv_id or "block",
                )
                field_aliases = self.flattened_uniform_block_field_aliases(node)
                self.record_flattened_uniform_block_instance(node, field_aliases)
                attributes = self.uniform_block_attribute_suffix(node)
                code += f"    cbuffer {block_name}{attributes} {{\n"
                for field_type, field_name in node.struct_fields:
                    generated_field_name = self.flattened_uniform_block_field_name(
                        field_name, field_aliases
                    )
                    code += (
                        f"        {self.map_type(field_type)} "
                        f"{generated_field_name};\n"
                    )
                code += "    }\n\n"
            else:
                code += (
                    f"    {self.map_type(node.data_type)} {node.variable_name}"
                    f"{self.uniform_resource_attribute_suffix(node)};\n"
                )
                self.reserve_global_declaration_name(node.variable_name)
        elif layout_type == "buffer":
            if node.struct_fields:
                block_name = self.storage_buffer_block_type_name(node)
                variable_name = (
                    node.variable_name or block_name[0].lower() + block_name[1:]
                )
                buffer_type = (
                    "StructuredBuffer"
                    if "readonly" in getattr(node, "declaration_qualifiers", [])
                    else "RWStructuredBuffer"
                )
                attributes = self.uniform_block_attribute_suffix(node)
                if block_name not in self.emitted_storage_buffer_type_names:
                    code += f"    struct {block_name} {{\n"
                    for field_type, field_name in node.struct_fields:
                        code += f"        {self.map_type(field_type)} {field_name};\n"
                    code += "    };\n\n"
                    self.emitted_storage_buffer_type_names.add(block_name)
                code += (
                    f"    {buffer_type}<{block_name}> {variable_name}"
                    f"{attributes};\n\n"
                )
                self.reserve_global_declaration_name(variable_name)
        elif layout_type == "in" or layout_type == "out":
            if node.data_type and node.variable_name:
                attributes = self.interface_layout_attribute_suffix(
                    node,
                    suppressed_interface_qualifiers,
                    override_interface_qualifiers,
                    forced_builtin,
                )
                code += (
                    f"    {self.map_type(node.data_type)} "
                    f"{override_variable_name or node.variable_name}"
                    f"{attributes};\n"
                )
                self.reserve_global_declaration_name(
                    override_variable_name or node.variable_name
                )
        elif self.is_ray_storage_layout(node):
            code += (
                f"    {self.map_type(node.data_type)} {node.variable_name}"
                f"{self.ray_storage_layout_attribute_suffix(node)};\n"
            )
            self.reserve_global_declaration_name(node.variable_name)

        return code

    def storage_buffer_block_type_name(self, node):
        base_name = node.block_name or node.variable_name or "StorageBuffer"
        signature = tuple(
            (str(field_type), str(field_name))
            for field_type, field_name in getattr(node, "struct_fields", []) or []
        )
        key = (base_name, signature)
        if key in self.storage_buffer_type_names:
            return self.storage_buffer_type_names[key]

        existing_signature = self.storage_buffer_name_signatures.get(base_name)
        if (
            existing_signature is None
            and base_name not in self.used_global_declaration_names
        ) or existing_signature == signature:
            type_name = base_name
        else:
            suffix_source = node.variable_name or str(
                len(self.storage_buffer_type_names)
            )
            suffix = self.storage_buffer_type_suffix(suffix_source)
            type_name = f"{base_name}_{suffix}"
            counter = 2
            while type_name in self.used_global_declaration_names or (
                type_name in self.storage_buffer_name_signatures
                and self.storage_buffer_name_signatures[type_name] != signature
            ):
                type_name = f"{base_name}_{suffix}_{counter}"
                counter += 1

        self.storage_buffer_type_names[key] = type_name
        self.storage_buffer_name_signatures[type_name] = signature
        self.used_global_declaration_names.add(type_name)
        return type_name

    def storage_buffer_type_suffix(self, raw_value):
        suffix = str(raw_value or "").split("[", 1)[0]
        suffix = "".join(
            char if char.isalnum() or char == "_" else "_" for char in suffix
        )
        suffix = suffix.strip("_")
        return suffix or "block"

    def is_specialization_constant_layout(self, node):
        if (getattr(node, "layout_type", None) or "").lower() != "const":
            return False
        return any(
            str(name).lower() == "constant_id"
            for name, _ in getattr(node, "qualifiers", []) or []
        )

    def generate_specialization_constant_layout(self, node):
        declaration = getattr(node, "declaration", None)
        if declaration is None:
            return ""

        metadata = self.const_layout_attribute_suffix(node)

        if isinstance(declaration, AssignmentNode):
            lhs = self.specialization_constant_declaration_lhs(
                self.assignment_left(declaration), metadata
            )
            rhs = self.generate_expression(self.assignment_right(declaration))
            return f"    {lhs} = {rhs};\n"

        lhs = self.specialization_constant_declaration_lhs(declaration, metadata)
        return f"    {lhs};\n"

    def const_layout_attribute_suffix(self, node):
        attributes = []
        for name, value in getattr(node, "qualifiers", []) or []:
            qualifier_name = str(name).lower()
            if qualifier_name == "constant_id" and value is not None:
                attributes.append(f"@constant_id({value})")
            elif qualifier_name == "builtin" and value is not None:
                attributes.append(self.crossgl_builtin_attribute(value))
        if not attributes:
            return ""
        return " " + " ".join(attributes)

    def specialization_constant_declaration_lhs(self, declaration, metadata):
        if isinstance(declaration, VariableNode) and self.variable_type(declaration):
            return (
                f"{self.map_type(self.variable_type(declaration))} "
                f"{declaration.name}{metadata}"
            )
        return f"{self.generate_expression(declaration)}{metadata}"

    def uniform_resource_attribute_suffix(self, node):
        storage_suffix = self.storage_image_layout_attribute_suffix(node)
        if storage_suffix:
            return storage_suffix
        if self.is_subpass_input_type(getattr(node, "data_type", None)):
            return self.uniform_block_attribute_suffix(node)
        if getattr(node, "spirv_storage_class", None) == "UniformConstant":
            return self.uniform_block_attribute_suffix(node)
        return ""

    def storage_image_layout_attribute_suffix(self, node):
        data_type = getattr(node, "data_type", None)
        if not self.is_storage_image_type(data_type):
            return ""

        attributes = []
        descriptor_set = None
        binding = None
        image_format = None
        for name, value in getattr(node, "qualifiers", []) or []:
            qualifier_name = str(name).lower()
            if qualifier_name == "set" and value is not None:
                descriptor_set = value
            elif qualifier_name == "binding" and value is not None:
                binding = value
            elif (
                qualifier_name in self.supported_image_formats()
                and not self.is_storage_image_buffer_type(data_type)
            ):
                image_format = qualifier_name

        declaration_attributes = []
        for qualifier in getattr(node, "declaration_qualifiers", []) or []:
            qualifier_name = str(qualifier).lower()
            if qualifier_name in {
                "coherent",
                "volatile",
                "restrict",
                "readonly",
                "writeonly",
            }:
                declaration_attributes.append(f"@{qualifier_name}")

        if image_format is None and not declaration_attributes:
            return ""

        if descriptor_set is not None:
            attributes.append(f"@set({descriptor_set})")
        if binding is not None:
            attributes.append(f"@binding({binding})")
        if image_format is not None:
            attributes.append(f"@{image_format}")
        attributes.extend(declaration_attributes)
        return f" {' '.join(attributes)}" if attributes else ""

    def uniform_block_attribute_suffix(self, node):
        attributes = []
        descriptor_set = None
        binding = None
        input_attachment_index = None
        for name, value in getattr(node, "qualifiers", []) or []:
            qualifier_name = str(name).lower()
            if qualifier_name == "set" and value is not None:
                descriptor_set = value
            elif qualifier_name == "binding" and value is not None:
                binding = value
            elif qualifier_name == "input_attachment_index" and value is not None:
                input_attachment_index = value
        spirv_input_attachment_index = self.spirv_input_attachment_index(node)
        if spirv_input_attachment_index is not None:
            input_attachment_index = spirv_input_attachment_index
        if descriptor_set is not None:
            attributes.append(f"@set({descriptor_set})")
        if binding is not None:
            attributes.append(f"@binding({binding})")
        if input_attachment_index is not None:
            attributes.append(f"@input_attachment_index({input_attachment_index})")
        if getattr(node, "push_constant", False):
            attributes.append("@push_constant")
        return f" {' '.join(attributes)}" if attributes else ""

    def spirv_input_attachment_index(self, node):
        for decoration, operands in getattr(node, "spirv_decorations", []) or []:
            if decoration == "InputAttachmentIndex" and operands:
                return operands[0]
        return None

    def is_subpass_input_type(self, type_name):
        return type_name in {"subpassInput", "subpassInputMS"}

    def is_storage_image_type(self, type_name):
        return isinstance(type_name, str) and type_name.startswith(
            ("image", "iimage", "uimage")
        )

    def is_storage_image_buffer_type(self, type_name):
        return isinstance(type_name, str) and type_name.endswith("imageBuffer")

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

    def interface_layout_attribute_suffix(
        self,
        node,
        suppressed_qualifiers=None,
        override_qualifiers=None,
        forced_builtin=None,
    ):
        layout_type = node.layout_type.lower() if node.layout_type else ""
        if layout_type not in {"in", "out"}:
            return ""

        suppressed_qualifiers = {
            str(qualifier).lower() for qualifier in suppressed_qualifiers or set()
        }
        attributes = ["@input" if layout_type == "in" else "@output"]
        for name, value in (
            override_qualifiers
            if override_qualifiers is not None
            else getattr(node, "qualifiers", []) or []
        ):
            qualifier_name = str(name).lower()
            if qualifier_name in suppressed_qualifiers:
                continue
            if (
                qualifier_name in {"location", "component", "index"}
                and value is not None
            ):
                attributes.append(f"@{qualifier_name}({value})")
            elif qualifier_name == "builtin" and value is not None:
                attributes.append(
                    self.crossgl_builtin_attribute(
                        value, getattr(node, "spirv_storage_class", None)
                    )
                )
        if forced_builtin is not None:
            attributes.append(
                self.crossgl_builtin_attribute(
                    forced_builtin, getattr(node, "spirv_storage_class", None)
                )
            )

        supported_qualifiers = {
            "centroid",
            "flat",
            "highp",
            "invariant",
            "lowp",
            "mediump",
            "noperspective",
            "patch",
            "pervertexext",
            "precise",
            "sample",
            "smooth",
        }
        for qualifier in getattr(node, "declaration_qualifiers", []) or []:
            qualifier_text = str(qualifier)
            if qualifier_text.lower() in supported_qualifiers:
                attributes.append(f"@{qualifier_text}")

        return f" {' '.join(attributes)}"

    def crossgl_builtin_attribute(self, builtin_name, storage_class=None):
        storage_mapped_builtins = {
            ("SampleMask", "Input"): "gl_SampleMaskIn",
            ("SampleMask", "Output"): "gl_SampleMask",
        }
        mapped_builtins = {
            "BaseInstance": "gl_BaseInstance",
            "BaseVertex": "gl_BaseVertex",
            "BaryCoordKHR": "gl_BaryCoordEXT",
            "BaryCoordNV": "gl_BaryCoordEXT",
            "BaryCoordNoPerspKHR": "gl_BaryCoordNoPerspEXT",
            "BaryCoordNoPerspNV": "gl_BaryCoordNoPerspEXT",
            "ClipDistance": "gl_ClipDistance",
            "CullDistance": "gl_CullDistance",
            "FragCoord": "gl_FragCoord",
            "FragDepth": "gl_FragDepth",
            "FragStencilRefEXT": "gl_FragStencilRefEXT",
            "FrontFacing": "gl_FrontFacing",
            "GlobalInvocationId": "gl_GlobalInvocationID",
            "HelperInvocation": "gl_HelperInvocation",
            "InstanceId": "gl_InstanceID",
            "InstanceIndex": "gl_InstanceID",
            "InvocationId": "gl_InvocationID",
            "Layer": "gl_Layer",
            "LocalInvocationId": "gl_LocalInvocationID",
            "LocalInvocationIndex": "gl_LocalInvocationIndex",
            "NumSubgroups": "gl_NumSubgroups",
            "NumWorkgroups": "gl_NumWorkGroups",
            "PatchVertices": "gl_PatchVerticesIn",
            "PointCoord": "gl_PointCoord",
            "PointSize": "gl_PointSize",
            "Position": "gl_Position",
            "PrimitiveId": "gl_PrimitiveID",
            "SampleId": "gl_SampleID",
            "SampleMask": "gl_SampleMask",
            "SamplePosition": "gl_SamplePosition",
            "SubgroupId": "gl_SubgroupID",
            "SubgroupLocalInvocationId": "gl_SubgroupInvocationID",
            "SubgroupSize": "gl_SubgroupSize",
            "TessCoord": "gl_TessCoord",
            "TessLevelInner": "gl_TessLevelInner",
            "TessLevelOuter": "gl_TessLevelOuter",
            "VertexId": "gl_VertexID",
            "VertexIndex": "gl_VertexID",
            "ViewportIndex": "gl_ViewportIndex",
            "ViewIndex": "gl_ViewIndex",
            "WorkgroupId": "gl_WorkGroupID",
            "WorkgroupSize": "gl_WorkGroupSize",
        }
        storage_mapped_name = storage_mapped_builtins.get(
            (str(builtin_name), storage_class)
        )
        if storage_mapped_name:
            return f"@{storage_mapped_name}"
        mapped_name = mapped_builtins.get(str(builtin_name))
        if mapped_name:
            return f"@{mapped_name}"
        return f"@builtin({str(builtin_name).lower()})"

    def flattened_uniform_block_field_aliases(self, node):
        aliases = {}
        for _field_type, field_name in node.struct_fields:
            field_base = self.declaration_base_name(field_name)
            generated_base = self.unique_flattened_uniform_block_field_name(
                node, field_base
            )
            aliases[field_base] = generated_base
        return aliases

    def unique_flattened_uniform_block_field_name(self, node, field_name):
        if not field_name:
            return field_name

        reserved_names = (
            self.used_global_declaration_names
            | self.non_renamable_global_declaration_names
        )
        if not getattr(node, "variable_name", None) or field_name not in reserved_names:
            self.reserve_global_declaration_name(field_name)
            return field_name

        instance_name = self.declaration_base_name(node.variable_name)
        alias_base = self.storage_buffer_type_suffix(f"{instance_name}_{field_name}")
        alias = alias_base
        counter = 2
        while alias in reserved_names:
            alias = f"{alias_base}_{counter}"
            counter += 1
        self.reserve_global_declaration_name(alias)
        return alias

    def flattened_uniform_block_field_name(self, field_name, field_aliases):
        field_base = self.declaration_base_name(field_name)
        generated_base = field_aliases.get(field_base, field_base)
        return f"{generated_base}{self.declaration_array_suffix(field_name)}"

    def record_flattened_uniform_block_instance(self, node, field_aliases=None):
        if not node.variable_name:
            return
        instance_name = str(node.variable_name).split("[", 1)[0]
        if not instance_name:
            return
        if field_aliases is None:
            field_aliases = {
                str(field_name).split("[", 1)[0]: str(field_name).split("[", 1)[0]
                for _, field_name in node.struct_fields
            }
        self.flattened_uniform_block_instances[instance_name] = field_aliases

    def generate_uniform(self, node):
        self.reserve_global_declaration_name(node.name)
        return f"    {self.map_type(node.vtype)} {node.name};\n"

    def generate_struct(self, node):
        self.reserve_global_declaration_name(node.name)
        code = f"    struct {node.name} {{\n"
        for member in node.members:
            if isinstance(member, VariableNode):
                code += f"        {self.map_type(self.variable_type(member))} {member.name};\n"
            elif isinstance(member, AssignmentNode):
                lhs = self.assignment_left(member)
                code += (
                    f"        {self.map_type(self.variable_type(lhs))} {lhs.name};\n"
                )
        code += "    }\n\n"
        return code

    def generate_function(self, node, indent=1):
        """Render one Vulkan backend function node as a CrossGL function."""
        code = "  " * indent
        return_type = self.map_type(node.return_type)
        params = ", ".join(
            self.generate_function_parameter(param)
            for param in self.function_params(node)
        )
        attributes = self.function_attribute_suffix(node)
        code += f"    {return_type} {node.name}({params}){attributes} {{\n"
        code += self.generate_function_body(node.body, indent=indent + 1)
        code += "    }\n\n"
        return code

    def function_attribute_suffix(self, node):
        """Return CrossGL attributes needed to preserve stage entry identity."""
        if self.function_shader_stage(node) is not None and node.name != "main":
            return " @stage_entry"
        return ""

    def generate_function_parameter(self, node):
        type_name, ray_attributes = self.ray_storage_qualified_type_parts(
            self.variable_type(node)
        )
        qualifier_attributes = [
            attribute
            for qualifier in getattr(node, "qualifiers", []) or []
            if (attribute := self.storage_qualifier_attribute(qualifier)) is not None
        ]
        attributes = self.ray_storage_attribute_suffix(
            [*qualifier_attributes, *ray_attributes]
        )
        return (
            f"{self.parameter_qualifier_prefix(node)}"
            f"{self.map_type(type_name)} {node.name}{attributes}"
        )

    def parameter_qualifier_prefix(self, node):
        qualifiers = []
        for qualifier in getattr(node, "qualifiers", []) or []:
            qualifier_name = str(qualifier).lower()
            if qualifier_name in {"in", "out", "inout"}:
                qualifiers.append(qualifier_name)
        return f"{' '.join(qualifiers)} " if qualifiers else ""

    def generate_function_body(self, body, indent=1):
        code = ""
        if not isinstance(body, list):
            body = [body]

        for stmt in body:
            code += "    " * indent
            if isinstance(stmt, VariableNode):
                code += f"{self.generate_variable_declaration(stmt)};\n"
            elif isinstance(stmt, AssignmentNode):
                code += self.generate_assignment(stmt) + ";\n"
            elif isinstance(stmt, BinaryOpNode):
                code += f"{self.generate_expression(stmt.left)} {stmt.op} {self.generate_expression(stmt.right)};\n"
            elif isinstance(stmt, ReturnNode):
                if stmt.value:
                    code += f"return {self.generate_expression(stmt.value)};\n"
                else:
                    code += "return;\n"
            elif isinstance(stmt, ForNode):
                code += self.generate_for_loop(stmt, indent)
            elif isinstance(stmt, WhileNode):
                code += self.generate_while_loop(stmt, indent)
            elif isinstance(stmt, DoWhileNode):
                code += self.generate_do_while_loop(stmt, indent)
            elif isinstance(stmt, IfNode):
                code += self.generate_if_statement(stmt, indent)
            elif isinstance(stmt, SwitchNode):
                code += self.generate_switch_statement(stmt, indent)
            elif isinstance(stmt, (FunctionCallNode, MethodCallNode)):
                code += f"{self.generate_expression(stmt)};\n"
            elif isinstance(stmt, UnaryOpNode):
                code += f"{self.generate_expression(stmt)};\n"
            elif isinstance(stmt, BreakNode):
                code += "break;\n"
            elif isinstance(stmt, ContinueNode):
                code += "continue;\n"
            elif isinstance(stmt, DiscardNode):
                code += "discard;\n"
            elif isinstance(stmt, str):
                code += f"{stmt};\n"
            else:
                code += f"// Unhandled statement type: {type(stmt).__name__}\n"
        return code

    def generate_assignment(self, node):
        lhs_node = self.assignment_left(node)
        rhs = self.generate_expression(self.assignment_right(node))
        operator = getattr(node, "operator", "=")

        if isinstance(lhs_node, VariableNode) and self.variable_type(lhs_node):
            lhs = self.generate_variable_declaration(lhs_node)
        else:
            lhs = self.generate_expression(lhs_node)

        return f"{lhs} {operator} {rhs}"

    def generate_variable_declaration(
        self, node, extra_attributes=None, override_name=None
    ):
        type_name, ray_attributes = self.ray_storage_qualified_type_parts(
            self.variable_type(node)
        )
        attributes = self.ray_storage_attribute_suffix(
            [*ray_attributes, *(extra_attributes or [])]
        )
        return f"{self.map_type(type_name)} {override_name or node.name}{attributes}"

    def ray_storage_qualified_type_parts(self, type_name):
        parts = str(type_name or "").split()
        if not parts:
            return type_name, []

        remaining_parts = []
        attributes = []
        for part in parts:
            attribute = self.storage_qualifier_attribute(part)
            if attribute is None:
                remaining_parts.append(part)
            else:
                attributes.append(attribute)

        if not attributes:
            return type_name, []
        return " ".join(remaining_parts), attributes

    def is_ray_storage_layout(self, node):
        if not isinstance(node, LayoutNode):
            return False
        if not getattr(node, "data_type", None) or not getattr(
            node, "variable_name", None
        ):
            return False
        return any(
            self.ray_storage_attribute(qualifier) is not None
            for qualifier in getattr(node, "declaration_qualifiers", []) or []
        )

    def ray_storage_layout_attribute_suffix(self, node):
        attributes = []
        for name, value in getattr(node, "qualifiers", []) or []:
            qualifier_name = str(name).lower()
            if (
                qualifier_name in {"location", "component", "index"}
                and value is not None
            ):
                attributes.append(f"{qualifier_name}({value})")
        attributes.extend(
            attribute
            for qualifier in getattr(node, "declaration_qualifiers", []) or []
            if (attribute := self.ray_storage_attribute(qualifier)) is not None
        )
        return self.ray_storage_attribute_suffix(attributes)

    def ray_storage_attribute_suffix(self, attributes):
        if not attributes:
            return ""
        return f" {' '.join(f'@{attribute}' for attribute in attributes)}"

    def ray_storage_attribute(self, qualifier):
        return self.RAY_STORAGE_QUALIFIER_ATTRIBUTES.get(str(qualifier).lower())

    def storage_qualifier_attribute(self, qualifier):
        qualifier_name = str(qualifier).lower()
        return self.RAY_STORAGE_QUALIFIER_ATTRIBUTES.get(
            qualifier_name
        ) or self.MESH_STORAGE_QUALIFIER_ATTRIBUTES.get(qualifier_name)

    def generate_expression(self, expr):
        """Render a Vulkan backend expression node as CrossGL syntax."""
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, int) or isinstance(expr, float):
            return str(expr)
        elif isinstance(expr, VariableNode):
            return self.generated_variable_expression_name(expr)
        elif isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr)
        elif isinstance(expr, BinaryOpNode):
            left = self.generate_nested_expression(expr.left)
            right = self.generate_nested_expression(expr.right)

            if expr.op in self.bitwise_op_map:
                op = self.bitwise_op_map[expr.op]
                return f"({left} {op} {right})"

            return f"({left} {expr.op} {right})"
        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_nested_expression(expr.operand)
            if expr.op == "~":
                return f"(~{operand})"
            if expr.op == "POST_INCREMENT":
                return f"{operand}++"
            if expr.op == "POST_DECREMENT":
                return f"{operand}--"
            if expr.op == "PRE_INCREMENT":
                return f"++{operand}"
            if expr.op == "PRE_DECREMENT":
                return f"--{operand}"
            return f"{expr.op}{operand}"
        elif isinstance(expr, TernaryOpNode):
            condition = self.generate_nested_expression(expr.condition)
            true_expr = self.generate_nested_expression(expr.true_expr)
            false_expr = self.generate_nested_expression(expr.false_expr)
            return f"({condition} ? {true_expr} : {false_expr})"
        elif isinstance(expr, FunctionCallNode):
            return self.generate_function_call(expr)
        elif isinstance(expr, InitializerListNode):
            args = ", ".join(self.generate_expression(arg) for arg in expr.elements)
            return f"{{{args}}}"
        elif isinstance(expr, MethodCallNode):
            args = ", ".join(self.generate_expression(arg) for arg in expr.args)
            return f"{self.generate_expression(expr.object)}.{expr.method}({args})"
        elif isinstance(expr, MemberAccessNode):
            flattened_member = self.flattened_uniform_block_member(expr)
            if flattened_member is not None:
                return flattened_member
            obj = self.generate_postfix_base_expression(expr.object)
            return f"{obj}.{expr.member}"
        elif isinstance(expr, ArrayAccessNode):
            array = self.generate_postfix_base_expression(expr.array)
            index = self.generate_expression(expr.index)
            return f"{array}[{index}]"
        else:
            return str(expr)

    def generate_nested_expression(self, expr):
        rendered = self.generate_expression(expr)
        if isinstance(expr, AssignmentNode):
            return f"({rendered})"
        return rendered

    def generate_postfix_base_expression(self, expr):
        rendered = self.generate_expression(expr)
        if isinstance(expr, (AssignmentNode, UnaryOpNode)):
            return f"({rendered})"
        return rendered

    def generated_variable_expression_name(self, expr):
        spirv_id = getattr(expr, "spirv_id", None)
        if spirv_id in self.renamed_spirv_global_names:
            return self.renamed_spirv_global_names[spirv_id]
        return expr.name

    def expression_base_name(self, expr):
        if isinstance(expr, str):
            return expr
        if isinstance(expr, VariableNode):
            return expr.name
        return None

    def flattened_uniform_block_member(self, expr):
        instance_name = self.expression_base_name(expr.object)
        if instance_name is None:
            return None
        fields = self.flattened_uniform_block_instances.get(instance_name)
        if fields is None:
            return None
        if isinstance(fields, dict):
            return fields.get(expr.member)
        if expr.member not in fields:
            return None
        return expr.member

    def generate_function_call(self, node):
        args = ", ".join(self.generate_expression(arg) for arg in node.args)
        return f"{self.map_type(node.name)}({args})"

    def generate_for_loop(self, node, indent):
        init = self.generate_for_clause(node.init)
        condition = self.generate_for_clause(node.condition)
        update = self.generate_for_clause(node.update)

        code = f"for ({init}; {condition}; {update}) {{\n"
        code += self.generate_function_body(node.body, indent=indent + 1)
        code += "    " * indent + "}\n"
        return code

    def generate_for_clause(self, clause):
        if clause is None:
            return ""
        if isinstance(clause, list):
            declaration_type = self.for_clause_declaration_type(clause[0])
            return ", ".join(
                self.generate_for_clause_item(item, declaration_type, index)
                for index, item in enumerate(clause)
            )
        if isinstance(clause, str):
            return clause
        return self.generate_expression(clause)

    def generate_for_clause_item(self, item, declaration_type, index):
        if index > 0 and declaration_type:
            if isinstance(item, AssignmentNode):
                lhs_node = self.assignment_left(item)
                if (
                    isinstance(lhs_node, VariableNode)
                    and self.variable_type(lhs_node) == declaration_type
                ):
                    rhs = self.generate_expression(self.assignment_right(item))
                    operator = getattr(item, "operator", "=")
                    return f"{lhs_node.name} {operator} {rhs}"
            elif (
                isinstance(item, VariableNode)
                and self.variable_type(item) == declaration_type
            ):
                return item.name
        return self.generate_for_clause(item)

    def for_clause_declaration_type(self, item):
        target = (
            self.assignment_left(item) if isinstance(item, AssignmentNode) else item
        )
        if isinstance(target, VariableNode):
            return self.variable_type(target)
        return ""

    def generate_while_loop(self, node, indent):
        condition = self.generate_expression(node.condition)

        code = f"while ({condition}) {{\n"
        code += self.generate_function_body(node.body, indent=indent + 1)
        code += "    " * indent + "}\n"
        return code

    def generate_do_while_loop(self, node, indent):
        condition = self.generate_expression(node.condition)

        code = "do {\n"
        code += self.generate_function_body(node.body, indent=indent + 1)
        code += "    " * indent + "} "
        code += f"while ({condition});\n"
        return code

    def generate_if_statement(self, node, indent):
        condition = self.generate_expression(
            getattr(node, "condition", getattr(node, "if_condition", None))
        )

        code = f"if ({condition}) {{\n"
        code += self.generate_function_body(node.if_body, indent=indent + 1)
        code += "    " * indent + "}"

        else_if_chain = getattr(node, "else_if_chain", [])
        if else_if_chain:
            for else_if_condition, else_if_body in else_if_chain:
                code += f" else if ({self.generate_expression(else_if_condition)}) {{\n"
                code += self.generate_function_body(else_if_body, indent=indent + 1)
                code += "    " * indent + "}"

        if hasattr(node, "else_if_conditions") and node.else_if_conditions:
            for i in range(len(node.else_if_conditions)):
                else_if_condition = self.generate_expression(node.else_if_conditions[i])
                code += f" else if ({else_if_condition}) {{\n"
                code += self.generate_function_body(
                    node.else_if_bodies[i], indent=indent + 1
                )
                code += "    " * indent + "}"

        if node.else_body:
            code += " else {\n"
            code += self.generate_function_body(node.else_body, indent=indent + 1)
            code += "    " * indent + "}"

        code += "\n"
        return code

    def generate_switch_statement(self, node, indent):
        expression = self.generate_expression(node.expression)

        code = f"switch ({expression}) {{\n"
        emitted_default = False

        for case in node.cases:
            if isinstance(case, CaseNode):
                if case.value is None:
                    code += "    " * (indent + 1) + "default:\n"
                    code += self.generate_function_body(
                        self.switch_case_body(case), indent=indent + 2
                    )
                    emitted_default = True
                    continue
                value = self.generate_expression(case.value)
                code += "    " * (indent + 1) + f"case {value}:\n"
                code += self.generate_function_body(
                    self.switch_case_body(case), indent=indent + 2
                )
            elif isinstance(case, DefaultNode):
                code += "    " * (indent + 1) + "default:\n"
                code += self.generate_function_body(
                    self.switch_case_body(case), indent=indent + 2
                )
                emitted_default = True

        default_case = getattr(node, "default_case", None)
        if default_case is not None and not emitted_default:
            code += "    " * (indent + 1) + "default:\n"
            code += self.generate_function_body(
                self.switch_case_body(default_case), indent=indent + 2
            )

        code += "    " * indent + "}\n"
        return code

    def switch_case_body(self, case):
        if case is None:
            return []
        if hasattr(case, "body"):
            return case.body or []
        if hasattr(case, "statements"):
            return case.statements or []
        if isinstance(case, list):
            return case
        return [case]

    def map_type(self, vulkan_type):
        """Map a Vulkan/SPIR-V type name to the closest CrossGL type name."""
        bracket_index = vulkan_type.find("[")
        if bracket_index > 0:
            base_type = vulkan_type[:bracket_index]
            array_suffix = vulkan_type[bracket_index:]
            return f"{self.map_type(base_type)}{array_suffix}"

        if vulkan_type in self.type_map:
            return self.type_map[vulkan_type]

        type_parts = vulkan_type.split()
        if len(type_parts) > 1:
            qualifiers = type_parts[:-1]
            base_type = type_parts[-1]
            mapped_base = self.type_map.get(base_type, base_type)
            return " ".join([*qualifiers, mapped_base])

        return vulkan_type
