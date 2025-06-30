import pytest

from crosstl.translator.codegen.SPIRV_codegen import (
    SpirvType,
    SpirvId,
    VulkanSPIRVCodeGen,
)
from crosstl.translator.ast import (
    ShaderNode,
    StructNode,
    VariableNode,
    FunctionNode,
    AssignmentNode,
    ReturnNode,
    BinaryOpNode,
    ExecutionModel,
    PrimitiveType,
    BlockNode,
)


class TestSpirvType:
    def test_initialization(self):
        # Test without storage class
        type1 = SpirvType("float")
        assert type1.base_type == "float"
        assert type1.storage_class is None
        assert str(type1) == "float"

        # Test with storage class
        type2 = SpirvType("float", "Function")
        assert type2.base_type == "float"
        assert type2.storage_class == "Function"
        assert str(type2) == "float (Function)"


class TestSpirvId:
    def test_initialization(self):
        # Test without name
        type1 = SpirvType("float")
        id1 = SpirvId(42, type1)
        assert id1.id == 42
        assert id1.type == type1
        assert id1.name is None
        assert str(id1) == "%42 (float)"

        # Test with name
        id2 = SpirvId(43, type1, "my_var")
        assert id2.id == 43
        assert id2.type == type1
        assert id2.name == "my_var"
        assert str(id2) == "%43 (my_var: float)"


class TestVulkanSPIRVCodeGen:
    def test_initialization(self):
        gen = VulkanSPIRVCodeGen()
        assert gen.next_id == 1
        assert gen.code_lines == []
        assert gen.primitive_types == {}
        assert gen.is_vertex_shader is False

    def test_get_id(self):
        gen = VulkanSPIRVCodeGen()
        assert gen.get_id() == 1
        assert gen.get_id() == 2
        assert gen.get_id() == 3
        assert gen.next_id == 4
        assert gen.bound_id == 3

    def test_emit(self):
        gen = VulkanSPIRVCodeGen()
        gen.emit("OpCapability Shader")
        gen.emit("OpMemoryModel Logical GLSL450")
        assert gen.code_lines == [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
        ]

    def test_register_primitive_type(self):
        gen = VulkanSPIRVCodeGen()

        # Test void type
        void_id = gen.register_primitive_type("void")
        assert void_id.id == 1
        assert void_id.type.base_type == "void"
        assert gen.primitive_types["void"] == void_id
        assert gen.code_lines[0] == "%1 = OpTypeVoid"

        # Test float type
        float_id = gen.register_primitive_type("float")
        assert float_id.id == 2
        assert float_id.type.base_type == "float"
        assert gen.primitive_types["float"] == float_id
        assert gen.code_lines[1] == "%2 = OpTypeFloat 32"

        # Test reusing already registered type
        float_id2 = gen.register_primitive_type("float")
        assert float_id2.id == float_id.id

    def test_register_vector_type(self):
        gen = VulkanSPIRVCodeGen()
        float_id = gen.register_primitive_type("float")

        # Test vec2
        vec2_id = gen.register_vector_type(float_id, 2)
        assert vec2_id.id == 2
        assert vec2_id.type.base_type == "v2float"
        assert gen.code_lines[1] == "%2 = OpTypeVector %1 2"

        # Test vec3
        vec3_id = gen.register_vector_type(float_id, 3)
        assert vec3_id.id == 3
        assert vec3_id.type.base_type == "v3float"
        assert gen.code_lines[2] == "%3 = OpTypeVector %1 3"

        # Test reusing already registered type
        vec2_id2 = gen.register_vector_type(float_id, 2)
        assert vec2_id2.id == vec2_id.id

    def test_register_struct_type(self):
        gen = VulkanSPIRVCodeGen()
        float_id = gen.register_primitive_type("float")
        vec2_id = gen.register_vector_type(float_id, 2)

        # Test struct with multiple members
        members = [(float_id, "a"), (vec2_id, "b")]
        struct_id = gen.register_struct_type("MyStruct", members)

        assert struct_id.id == 3
        assert struct_id.type.base_type == "MyStruct"
        assert "%3 = OpTypeStruct %1 %2" in gen.code_lines
        assert 'OpName %3 "MyStruct"' in gen.code_lines
        assert 'OpMemberName %3 0 "a"' in gen.code_lines
        assert 'OpMemberName %3 1 "b"' in gen.code_lines

        # Check stored member info
        assert gen.current_struct_members["MyStruct"] == members

    def test_register_pointer_type(self):
        gen = VulkanSPIRVCodeGen()
        float_id = gen.register_primitive_type("float")

        # Test function pointer
        function_ptr = gen.register_pointer_type(float_id, "Function")
        assert function_ptr.id == 2
        assert function_ptr.type.base_type == "ptr_float"
        assert function_ptr.type.storage_class == "Function"
        assert gen.code_lines[1] == "%2 = OpTypePointer Function %1"

        # Test input pointer
        input_ptr = gen.register_pointer_type(float_id, "Input")
        assert input_ptr.id == 3
        assert input_ptr.type.base_type == "ptr_float"
        assert input_ptr.type.storage_class == "Input"
        assert gen.code_lines[2] == "%3 = OpTypePointer Input %1"

        # Test reusing already registered type
        function_ptr2 = gen.register_pointer_type(float_id, "Function")
        assert function_ptr2.id == function_ptr.id

    def test_register_constant(self):
        gen = VulkanSPIRVCodeGen()
        float_id = gen.register_primitive_type("float")

        # Test float constant
        const_id = gen.register_constant(3.14, float_id)
        assert const_id.id == 2
        assert const_id.type == float_id.type
        assert "%2 = OpConstant %1 3.14" in gen.code_lines

        # Test another constant
        another_const = gen.register_constant(2.71, float_id)
        assert another_const.id == 3
        assert "%3 = OpConstant %1 2.71" in gen.code_lines

    def test_create_variable(self):
        gen = VulkanSPIRVCodeGen()
        float_id = gen.register_primitive_type("float")

        # Test without name
        var1 = gen.create_variable(float_id, "Function")
        assert var1.id == 3
        assert var1.type.storage_class == "Function"
        assert gen.code_lines[2] == "%3 = OpVariable %2 Function"

        # Test with name
        var2 = gen.create_variable(float_id, "Input", "my_var")
        assert var2.id == 5
        assert var2.name == "my_var"
        assert gen.code_lines[4] == "%5 = OpVariable %4 Input"
        assert gen.code_lines[5] == 'OpName %5 "my_var"'

    def test_binary_operation(self):
        gen = VulkanSPIRVCodeGen()
        float_id = gen.register_primitive_type("float")
        left = gen.register_constant(1.0, float_id)
        right = gen.register_constant(2.0, float_id)

        # Test addition
        result = gen.binary_operation("+", float_id, left, right)
        assert result.id == 4
        assert result.type == float_id.type
        assert "%4 = OpFAdd %1 %2 %3" in gen.code_lines

        # Test multiplication
        result = gen.binary_operation("*", float_id, left, right)
        assert result.id == 5
        assert "%5 = OpFMul %1 %2 %3" in gen.code_lines

    def test_unary_operation(self):
        gen = VulkanSPIRVCodeGen()
        float_id = gen.register_primitive_type("float")
        operand = gen.register_constant(1.0, float_id)

        # Test negation
        result = gen.unary_operation("-", float_id, operand)
        assert result.id == 3
        assert result.type == float_id.type
        assert "%3 = OpFNegate %1 %2" in gen.code_lines

        # Test positive (no-op)
        result = gen.unary_operation("+", float_id, operand)
        assert result.id == operand.id  # Should just return the operand

    def test_process_expression(self):
        gen = VulkanSPIRVCodeGen()
        gen.register_primitive_type("float")

        # Test literal
        result = gen.process_expression(1.0)
        assert isinstance(result, SpirvId)

        # Test binary operation
        expr = BinaryOpNode(1.0, "+", 2.0)
        result = gen.process_expression(expr)
        assert isinstance(result, SpirvId)

    def test_simple_shader_generation(self):
        # Test a simple shader with struct and function
        gen = VulkanSPIRVCodeGen()

        # Create proper TypeNode objects
        float_type = PrimitiveType("float")
        vec2_type = PrimitiveType("vec2")  # Simplified for test

        struct_members = [
            VariableNode("x", float_type), 
            VariableNode("uv", vec2_type)
        ]

        struct_node = StructNode("TestStruct", struct_members)

        function_body = BlockNode([ReturnNode()])

        function_node = FunctionNode(
            name="testFunction",
            return_type=float_type,
            parameters=[VariableNode("param", float_type)],
            body=function_body
        )

        # Use the correct ShaderNode constructor with required parameters
        shader_node = ShaderNode(
            name="TestShader",
            execution_model=ExecutionModel.GRAPHICS_PIPELINE,
            structs=[struct_node],
            functions=[function_node],
            global_variables=[],
            constants=[],
        )

        # Generate SPIR-V
        spv_code = gen.generate(shader_node)

        # Basic validation
        assert "; SPIR-V" in spv_code
        assert "OpCapability Shader" in spv_code
        assert "OpMemoryModel Logical GLSL450" in spv_code
        assert "OpTypeStruct" in spv_code
        assert "OpFunction" in spv_code
        assert "OpReturn" in spv_code
        assert "OpFunctionEnd" in spv_code

    def test_register_input(self):
        gen = VulkanSPIRVCodeGen()
        float_id = gen.register_primitive_type("float")
        gen.register_pointer_type(float_id, "Input")

        # Register an input
        input_id = gen.register_input("in_var", float_id, 0, 0)
        assert input_id.id == 3
        assert "%3 = OpVariable %2 Input" in gen.code_lines
        assert 'OpName %3 "in_var"' in gen.code_lines
        assert "OpDecorate %3 Location 0" in gen.decorations

        # Another input with different location
        input2_id = gen.register_input("in_var2", float_id, 1, 0)
        assert input2_id.id == 4
        assert "%4 = OpVariable %2 Input" in gen.code_lines
        assert 'OpName %4 "in_var2"' in gen.code_lines
        assert "OpDecorate %4 Location 1" in gen.decorations

    def test_register_output(self):
        gen = VulkanSPIRVCodeGen()
        float_id = gen.register_primitive_type("float")
        gen.register_pointer_type(float_id, "Output")

        # Register an output
        output_id = gen.register_output("out_var", float_id, 0, 0)
        assert output_id.id == 3
        assert "%3 = OpVariable %2 Output" in gen.code_lines
        assert 'OpName %3 "out_var"' in gen.code_lines
        assert "OpDecorate %3 Location 0" in gen.decorations

        # Another output with different location
        output2_id = gen.register_output("out_var2", float_id, 1, 0)
        assert output2_id.id == 4
        assert "%4 = OpVariable %2 Output" in gen.code_lines
        assert 'OpName %4 "out_var2"' in gen.code_lines
        assert "OpDecorate %4 Location 1" in gen.decorations
