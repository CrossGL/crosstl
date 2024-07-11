from ..ast import ShaderNode, AssignmentNode


class SPIRVCodeGen:
    def __init__(self):
        self.id_counter = 1
        self.type_ids = {}
        self.variable_ids = {}

    def generate_assignment(self, node):
        return f"{node.name} = {node.value};"

    def generate(self, ast):
        if isinstance(ast, ShaderNode):
            return self.generate_shader(ast)
        return ""

    def generate_shader(self, node):
        self.id_counter = 1
        code = "; SPIR-V\n"
        code += "; Version: 1.0\n"
        code += "; Generator: Custom SPIR-V CodeGen\n"
        code += f"; Bound: {self.id_counter + 30}\n"
        code += "; Schema: 0\n"
        code += "OpCapability Shader\n"
        code += '%1 = OpExtInstImport "GLSL.std.450"\n'
        code += "OpMemoryModel Logical GLSL450\n"

        # EntryPoint
        entry_point_args = " ".join(
            f"%{name}" for _, name in node.inputs + node.outputs
        )
        code += f'OpEntryPoint Vertex %main "main" {entry_point_args}\n'

        code += "OpSource GLSL 450\n"

        # Names and decorations
        code += 'OpName %main "main"\n'
        for _, name in node.inputs + node.outputs:
            code += f'OpName %{name} "{name}"\n'

        for i, (_, name) in enumerate(node.inputs + node.outputs):
            code += f"OpDecorate %{name} Location {i}\n"

        # Type declarations
        code += "%void = OpTypeVoid\n"
        code += f"%{self.get_id()} = OpTypeFunction %void\n"
        code += "%float = OpTypeFloat 32\n"

        for vtype in set(vtype for vtype, _ in node.inputs + node.outputs):
            components = self.map_type(vtype)
            self.type_ids[vtype] = self.get_id()
            code += f"%{vtype} = OpTypeVector %float {components}\n"

        # Pointer types and variable declarations
        for vtype, name in node.outputs:
            self.get_id()
            code += f"%_ptr_Output_{vtype} = OpTypePointer Output %{vtype}\n"
            self.variable_ids[name] = self.get_id()
            code += f"%{name} = OpVariable %_ptr_Output_{vtype} Output\n"

        for vtype, name in node.inputs:
            self.get_id()
            code += f"%_ptr_Input_{vtype} = OpTypePointer Input %{vtype}\n"
            self.variable_ids[name] = self.get_id()
            code += f"%{name} = OpVariable %_ptr_Input_{vtype} Input\n"

        # Constants
        code += "%float_1 = OpConstant %float 1\n"

        # Main function
        code += self.generate_function(node.main_function)

        return code

    def generate_function(self, node):
        code = "%main = OpFunction %void None %3\n"
        code += f"%{self.get_id()} = OpLabel\n"

        for stmt in node.body:
            if isinstance(stmt, AssignmentNode):
                code += self.generate_assignment(stmt)

        code += "OpReturn\n"
        code += "OpFunctionEnd\n"
        return code

    def generate_assignment(self, node):
        if node.name == "color" and node.value.startswith("vec4"):
            # Assuming the assignment is color = vec4(position, 1.0);
            code = f"%{self.get_id()} = OpLoad %v3float %position\n"
            for i in range(3):
                code += f"%{self.get_id()} = OpCompositeExtract %float %{self.id_counter - 3} {i}\n"
            code += f"%{self.get_id()} = OpCompositeConstruct %v4float %{self.id_counter - 3} %{self.id_counter - 2} %{self.id_counter - 1} %float_1\n"
            code += f"OpStore %color %{self.id_counter - 1}\n"
        else:
            # For other assignments, you'd need to implement more sophisticated parsing and generation
            code = f"; Unhandled assignment: {node.name} = {node.value}\n"
        return code

    def map_type(self, vtype):
        type_mapping = {"vec2": "2", "vec3": "3", "vec4": "4"}
        return type_mapping.get(vtype, "1")

    def get_id(self):
        id = self.id_counter
        self.id_counter += 1
        return id
