"""Reverse code generator that emits CrossGL from Vulkan SPIR-V AST nodes."""

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
            "subpassInput": "Texture2D",
            "subpassInputMS": "Texture2DMS",
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

    def get_indent(self):
        return "    " * self.indentation

    def generate(self, ast):
        self.flattened_uniform_block_instances = {}
        code = "shader main {\n"
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
                code += f"    {self.map_type(self.variable_type(node))} {node.name};\n"
            elif isinstance(node, AssignmentNode):
                code += f"    {self.generate_assignment(node)};\n"
            elif isinstance(node, FunctionNode):
                # Determine if this is a vertex or fragment shader based on the function name
                if node.name == "main":
                    is_vertex_shader = False
                    for stmt in node.body:
                        if self.is_position_assignment(stmt):
                            is_vertex_shader = True
                            break

                    if compute_layouts:
                        code += "    // Compute Shader\n"
                        code += "    compute {\n"
                        for layout in compute_layouts:
                            code += self.generate_compute_layout(layout)
                        code += self.generate_function(node)
                        code += "    }\n\n"
                    elif is_vertex_shader:
                        code += "    // Vertex Shader\n"
                        code += "    vertex {\n"
                        code += self.generate_function(node)
                        code += "    }\n\n"
                    else:
                        code += "    // Fragment Shader\n"
                        code += "    fragment {\n"
                        for layout in fragment_execution_layouts:
                            code += self.generate_fragment_execution_layout(layout)
                        code += self.generate_function(node)
                        code += "    }\n\n"
                else:
                    code += self.generate_function(node)

        code += "}\n"
        return code

    def is_compute_layout(self, node):
        if not isinstance(node, LayoutNode):
            return False
        if (node.layout_type or "").lower() != "in":
            return False
        return any(
            name in {"local_size_x", "local_size_y", "local_size_z"}
            for name, _ in node.qualifiers
        )

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
        qualifiers = ", ".join(
            f"{name} = {value}" if value is not None else name
            for name, value in node.qualifiers
        )
        return f"        layout({qualifiers}) in;\n"

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

    def generate_layout(self, node):
        code = ""
        if self.is_specialization_constant_layout(node):
            return self.generate_specialization_constant_layout(node)

        layout_type = node.layout_type.lower() if node.layout_type else ""

        if layout_type == "uniform":
            if node.struct_fields:
                block_name = node.block_name or node.variable_name or "UniformBuffer"
                self.record_flattened_uniform_block_instance(node)
                attributes = self.uniform_block_attribute_suffix(node)
                code += f"    cbuffer {block_name}{attributes} {{\n"
                for field_type, field_name in node.struct_fields:
                    code += f"        {self.map_type(field_type)} {field_name};\n"
                code += "    }\n\n"
            else:
                code += (
                    f"    {self.map_type(node.data_type)} {node.variable_name}"
                    f"{self.storage_image_layout_attribute_suffix(node)};\n"
                )
        elif layout_type == "buffer":
            if node.struct_fields:
                block_name = node.block_name or node.variable_name or "StorageBuffer"
                variable_name = (
                    node.variable_name or block_name[0].lower() + block_name[1:]
                )
                buffer_type = (
                    "StructuredBuffer"
                    if "readonly" in getattr(node, "declaration_qualifiers", [])
                    else "RWStructuredBuffer"
                )
                code += f"    struct {block_name} {{\n"
                for field_type, field_name in node.struct_fields:
                    code += f"        {self.map_type(field_type)} {field_name};\n"
                code += "    };\n\n"
                code += f"    {buffer_type}<{block_name}> {variable_name};\n\n"
        elif layout_type == "in" or layout_type == "out":
            if node.data_type and node.variable_name:
                code += (
                    f"    {self.map_type(node.data_type)} {node.variable_name}"
                    f"{self.interface_layout_attribute_suffix(node)};\n"
                )

        return code

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

        constant_id = next(
            (
                value
                for name, value in getattr(node, "qualifiers", []) or []
                if str(name).lower() == "constant_id"
            ),
            None,
        )
        metadata = f" @constant_id({constant_id})" if constant_id is not None else ""

        if isinstance(declaration, AssignmentNode):
            lhs = self.specialization_constant_declaration_lhs(
                self.assignment_left(declaration), metadata
            )
            rhs = self.generate_expression(self.assignment_right(declaration))
            return f"    {lhs} = {rhs};\n"

        lhs = self.specialization_constant_declaration_lhs(declaration, metadata)
        return f"    {lhs};\n"

    def specialization_constant_declaration_lhs(self, declaration, metadata):
        if isinstance(declaration, VariableNode) and self.variable_type(declaration):
            return (
                f"{self.map_type(self.variable_type(declaration))} "
                f"{declaration.name}{metadata}"
            )
        return f"{self.generate_expression(declaration)}{metadata}"

    def storage_image_layout_attribute_suffix(self, node):
        if not self.is_storage_image_type(getattr(node, "data_type", None)):
            return ""

        attributes = []
        binding = None
        image_format = None
        for name, value in getattr(node, "qualifiers", []) or []:
            qualifier_name = str(name).lower()
            if qualifier_name == "binding" and value is not None:
                binding = value
            elif qualifier_name in self.supported_image_formats():
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

        if binding is not None:
            attributes.append(f"@binding({binding})")
        if image_format is not None:
            attributes.append(f"@{image_format}")
        attributes.extend(declaration_attributes)
        return f" {' '.join(attributes)}" if attributes else ""

    def uniform_block_attribute_suffix(self, node):
        attributes = []
        for name, value in getattr(node, "qualifiers", []) or []:
            qualifier_name = str(name).lower()
            if qualifier_name == "set" and value is not None:
                attributes.append(f"@set({value})")
            elif qualifier_name == "binding" and value is not None:
                attributes.append(f"@binding({value})")
        if getattr(node, "push_constant", False):
            attributes.append("@push_constant")
        return f" {' '.join(attributes)}" if attributes else ""

    def is_storage_image_type(self, type_name):
        return isinstance(type_name, str) and type_name.startswith(
            ("image", "iimage", "uimage")
        )

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

    def interface_layout_attribute_suffix(self, node):
        layout_type = node.layout_type.lower() if node.layout_type else ""
        if layout_type not in {"in", "out"}:
            return ""

        attributes = ["@input" if layout_type == "in" else "@output"]
        for name, value in getattr(node, "qualifiers", []) or []:
            qualifier_name = str(name).lower()
            if (
                qualifier_name in {"location", "component", "index"}
                and value is not None
            ):
                attributes.append(f"@{qualifier_name}({value})")
            elif qualifier_name == "builtin" and value is not None:
                attributes.append(self.crossgl_builtin_attribute(value))

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

    def crossgl_builtin_attribute(self, builtin_name):
        mapped_builtins = {
            "BaseInstance": "gl_BaseInstance",
            "BaseVertex": "gl_BaseVertex",
            "ClipDistance": "gl_ClipDistance",
            "CullDistance": "gl_CullDistance",
            "FragCoord": "gl_FragCoord",
            "FragDepth": "gl_FragDepth",
            "FrontFacing": "gl_FrontFacing",
            "GlobalInvocationId": "gl_GlobalInvocationID",
            "InstanceIndex": "gl_InstanceID",
            "LocalInvocationId": "gl_LocalInvocationID",
            "LocalInvocationIndex": "gl_LocalInvocationIndex",
            "NumWorkgroups": "gl_NumWorkGroups",
            "PointCoord": "gl_PointCoord",
            "PointSize": "gl_PointSize",
            "Position": "gl_Position",
            "PrimitiveId": "gl_PrimitiveID",
            "SubgroupLocalInvocationId": "gl_SubgroupInvocationID",
            "SubgroupSize": "gl_SubgroupSize",
            "VertexIndex": "gl_VertexID",
            "WorkgroupId": "gl_WorkGroupID",
        }
        mapped_name = mapped_builtins.get(str(builtin_name))
        if mapped_name:
            return f"@{mapped_name}"
        return f"@builtin({str(builtin_name).lower()})"

    def record_flattened_uniform_block_instance(self, node):
        if not node.variable_name:
            return
        instance_name = str(node.variable_name).split("[", 1)[0]
        if not instance_name:
            return
        self.flattened_uniform_block_instances[instance_name] = {
            str(field_name).split("[", 1)[0] for _, field_name in node.struct_fields
        }

    def generate_uniform(self, node):
        return f"    {self.map_type(node.vtype)} {node.name};\n"

    def generate_struct(self, node):
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
        code += f"    {return_type} {node.name}({params}) {{\n"
        code += self.generate_function_body(node.body, indent=indent + 1)
        code += "    }\n\n"
        return code

    def generate_function_parameter(self, node):
        return (
            f"{self.parameter_qualifier_prefix(node)}"
            f"{self.map_type(self.variable_type(node))} {node.name}"
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
                code += f"{self.map_type(self.variable_type(stmt))} {stmt.name};\n"
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
            lhs = f"{self.map_type(self.variable_type(lhs_node))} {lhs_node.name}"
        else:
            lhs = self.generate_expression(lhs_node)

        return f"{lhs} {operator} {rhs}"

    def generate_expression(self, expr):
        """Render a Vulkan backend expression node as CrossGL syntax."""
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, int) or isinstance(expr, float):
            return str(expr)
        elif isinstance(expr, VariableNode):
            return f"{expr.name}"
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
        elif isinstance(expr, MethodCallNode):
            args = ", ".join(self.generate_expression(arg) for arg in expr.args)
            return f"{self.generate_expression(expr.object)}.{expr.method}({args})"
        elif isinstance(expr, MemberAccessNode):
            flattened_member = self.flattened_uniform_block_member(expr)
            if flattened_member is not None:
                return flattened_member
            obj = self.generate_expression(expr.object)
            return f"{obj}.{expr.member}"
        elif isinstance(expr, ArrayAccessNode):
            array = self.generate_expression(expr.array)
            index = self.generate_expression(expr.index)
            return f"{array}[{index}]"
        else:
            return str(expr)

    def generate_nested_expression(self, expr):
        rendered = self.generate_expression(expr)
        if isinstance(expr, AssignmentNode):
            return f"({rendered})"
        return rendered

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
        if fields is None or expr.member not in fields:
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
        if vulkan_type in self.type_map:
            return self.type_map[vulkan_type]

        type_parts = vulkan_type.split()
        if len(type_parts) > 1:
            qualifiers = type_parts[:-1]
            base_type = type_parts[-1]
            mapped_base = self.type_map.get(base_type, base_type)
            return " ".join([*qualifiers, mapped_base])

        return vulkan_type
