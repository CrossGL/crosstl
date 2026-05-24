import re

import pytest

from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
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
    ShaderStage,
)


def spirv_named_ids(spv_code, name):
    return re.findall(rf'OpName (%\d+) "{re.escape(name)}"', spv_code)


def spirv_named_id(spv_code, name):
    ids = spirv_named_ids(spv_code, name)
    assert ids, f"Missing OpName for {name}"
    return ids[0]


def spirv_named_variable(spv_code, name, pointer_type=None, storage_class=None):
    pointer_pattern = re.escape(pointer_type) if pointer_type else r"%\d+"
    storage_pattern = storage_class if storage_class else r"\w+"
    for variable_id in spirv_named_ids(spv_code, name):
        if re.search(
            rf"{re.escape(variable_id)} = OpVariable {pointer_pattern} "
            rf"{storage_pattern}\b",
            spv_code,
        ):
            return variable_id
    raise AssertionError(f"Missing variable {name}")


def spirv_result_ids_for_opcode(spv_code, opcode):
    return re.findall(rf"(%\d+) = {re.escape(opcode)}\b", spv_code)


def spirv_vector_type_widths(spv_code):
    return {
        type_id: int(width)
        for type_id, width in re.findall(
            r"(%\d+) = OpTypeVector %\d+ ([234])\b", spv_code
        )
    }


def assert_vector_constructs_have_exact_scalar_operands(spv_code):
    vector_widths = spirv_vector_type_widths(spv_code)
    vector_values = {
        result_id
        for result_id, result_type in re.findall(r"(%\d+) = Op\w+ (%\d+)\b", spv_code)
        if result_type in vector_widths
    }

    for _, result_type, operands in re.findall(
        r"(%\d+) = OpCompositeConstruct (%\d+) ([^\n]+)",
        spv_code,
    ):
        if result_type not in vector_widths:
            continue

        operand_ids = operands.split()
        assert len(operand_ids) == vector_widths[result_type]
        assert not set(operand_ids) & vector_values


def assert_matrix_constructs_have_exact_column_operands(spv_code):
    matrix_types = {
        type_id: (column_type, int(column_count))
        for type_id, column_type, column_count in re.findall(
            r"(%\d+) = OpTypeMatrix (%\d+) (\d+)\b", spv_code
        )
    }
    result_types = {
        result_id: result_type
        for result_id, result_type in re.findall(r"(%\d+) = Op\w+ (%\d+)\b", spv_code)
    }

    for _, result_type, operands in re.findall(
        r"(%\d+) = OpCompositeConstruct (%\d+) ([^\n]+)",
        spv_code,
    ):
        if result_type not in matrix_types:
            continue

        column_type, column_count = matrix_types[result_type]
        operand_ids = operands.split()
        assert len(operand_ids) == column_count
        assert all(
            result_types.get(operand_id) == column_type for operand_id in operand_ids
        )


def spirv_named_parameters(spv_code, name, pointer_type=None):
    pointer_pattern = re.escape(pointer_type) if pointer_type else r"%\d+"
    return [
        parameter_id
        for parameter_id in spirv_named_ids(spv_code, name)
        if re.search(
            rf"{re.escape(parameter_id)} = OpFunctionParameter {pointer_pattern}\b",
            spv_code,
        )
    ]


def spirv_named_parameter(spv_code, name, pointer_type=None):
    parameters = spirv_named_parameters(spv_code, name, pointer_type)
    assert parameters, f"Missing function parameter {name}"
    return parameters[0]


def assert_spirv_function_variables_are_first_block_declarations(spv_code):
    lines = spv_code.splitlines()
    function_start = None

    for index, line in enumerate(lines):
        if re.match(r"%\d+ = OpFunction\b", line):
            function_start = index
            continue

        if function_start is not None and line == "OpFunctionEnd":
            function_lines = lines[function_start : index + 1]
            first_label = next(
                i
                for i, function_line in enumerate(function_lines)
                if re.match(r"%\d+ = OpLabel\b", function_line)
            )
            first_non_variable = next(
                (
                    i
                    for i, function_line in enumerate(function_lines[first_label + 1 :])
                    if not re.match(r"%\d+ = OpVariable %\d+ Function\b", function_line)
                ),
                None,
            )
            if first_non_variable is None:
                first_non_variable = len(function_lines)
            else:
                first_non_variable += first_label + 1

            for variable_index, function_line in enumerate(function_lines):
                if re.match(r"%\d+ = OpVariable %\d+ Function\b", function_line):
                    assert first_label < variable_index < first_non_variable

            function_start = None


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
        bool_id = gen.register_primitive_type("bool")

        # Test float constant
        const_id = gen.register_constant(3.14, float_id)
        assert const_id.id == 3
        assert const_id.type == float_id.type
        assert "%3 = OpConstant %1 3.14" in gen.code_lines

        # Test another constant
        another_const = gen.register_constant(2.71, float_id)
        assert another_const.id == 4
        assert "%4 = OpConstant %1 2.71" in gen.code_lines

        true_const = gen.register_constant(True, bool_id)
        false_const = gen.register_constant(False, bool_id)
        assert true_const.type == bool_id.type
        assert false_const.type == bool_id.type
        assert "%5 = OpConstantTrue %2" in gen.code_lines
        assert "%6 = OpConstantFalse %2" in gen.code_lines

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

        int_id = gen.register_primitive_type("int")
        int_operand = gen.register_constant(1, int_id)
        result = gen.unary_operation("-", int_id, int_operand)
        assert result.type == int_id.type
        assert any("OpSNegate" in line for line in gen.code_lines)

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

        struct_members = [VariableNode("x", float_type), VariableNode("uv", vec2_type)]

        struct_node = StructNode("TestStruct", struct_members)

        function_body = BlockNode([ReturnNode()])

        function_node = FunctionNode(
            name="testFunction",
            return_type=float_type,
            parameters=[VariableNode("param", float_type)],
            body=function_body,
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

    def test_generic_vector_composite_types(self):
        source_code = """
        shader GenericComposite {
            struct Packed {
                vec2<f64> precise;
                vec3<i32> index;
                vec4<u32> mask;
                vec2<bool> flags;
            };

            vec2<f64> passthrough(vec2<f64> value, vec3<i32> index, vec4<u32> mask, vec2<bool> flags) {
                vec2<f64> localValues[2];
                localValues[0] = value;
                return localValues[0];
            }
        }
        """

        ast = Parser(Lexer(source_code).tokens).parse()
        spv_code = VulkanSPIRVCodeGen().generate(ast)

        assert "OpCapability Float64" in spv_code
        assert "OpTypeFloat 64" in spv_code
        assert "OpTypeInt 32 0" in spv_code
        assert "OpTypeStruct" in spv_code
        assert "OpMemberName" in spv_code
        assert '"precise"' in spv_code
        assert '"index"' in spv_code
        assert '"mask"' in spv_code
        assert '"flags"' in spv_code
        assert "OpTypeFunction" in spv_code
        assert spv_code.count("OpFunctionParameter") == 4
        assert "OpTypeArray" in spv_code
        assert '"localValues"' in spv_code
        assert "OpStore" in spv_code
        assert "OpReturnValue" in spv_code
        assert "WARNING" not in spv_code

    def test_vector_comparisons_return_bool_vector_type(self):
        source_code = """
        shader VecCompareBug {
            compute {
                vec2<bool> compareFloat(vec2 a, vec2 b) {
                    return a < b;
                }

                vec2<bool> compareInt(ivec2 a, ivec2 b) {
                    return a < b;
                }

                void main() {
                    vec2<bool> mask = compareFloat(vec2(0.0, 1.0), vec2(1.0, 0.0));
                    vec2<bool> imask = compareInt(ivec2(0, 1), ivec2(1, 0));
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        bool_type = re.search(r"(%\d+) = OpTypeBool", spv_code)
        assert bool_type is not None
        bool_type_id = bool_type.group(1)
        bool_vec2_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(bool_type_id)} 2", spv_code
        )
        assert bool_vec2_type is not None
        bool_vec2_id = bool_vec2_type.group(1)

        assert re.search(
            rf"OpFOrdLessThan {re.escape(bool_vec2_id)} %\d+ %\d+\b", spv_code
        )
        assert re.search(
            rf"OpSLessThan {re.escape(bool_vec2_id)} %\d+ %\d+\b", spv_code
        )
        assert not re.search(rf"OpFOrdLessThan {re.escape(bool_type_id)}\b", spv_code)
        assert not re.search(rf"OpSLessThan {re.escape(bool_type_id)}\b", spv_code)
        assert "WARNING" not in spv_code

    def test_bool_vector_logical_ops_return_bool_vector_type(self):
        source_code = """
        shader BoolVectorLogic {
            compute {
                void main() {
                    bvec2 a = bvec2(true, false);
                    bvec2 b = bvec2(false, true);
                    bvec2 c = a && b;
                    bvec2 d = a || b;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        bool_type = re.search(r"(%\d+) = OpTypeBool", spv_code)
        assert bool_type is not None
        bool_type_id = bool_type.group(1)
        bool_vec2_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(bool_type_id)} 2", spv_code
        )
        assert bool_vec2_type is not None
        bool_vec2_id = bool_vec2_type.group(1)

        assert re.search(
            rf"OpLogicalAnd {re.escape(bool_vec2_id)} %\d+ %\d+\b", spv_code
        )
        assert re.search(
            rf"OpLogicalOr {re.escape(bool_vec2_id)} %\d+ %\d+\b", spv_code
        )
        assert not re.search(
            rf"OpLogicalAnd {re.escape(bool_type_id)} %\d+ %\d+\b", spv_code
        )
        assert not re.search(
            rf"OpLogicalOr {re.escape(bool_type_id)} %\d+ %\d+\b", spv_code
        )
        assert "WARNING" not in spv_code

    def test_bool_vector_constructors_flatten_bool_vectors_and_comparisons(self):
        source_code = """
        shader BoolVectorConstructors {
            compute {
                void main() {
                    bvec4 splat = bvec4(true);
                    bvec2 lanes = bvec2(true, false);
                    bvec2 cmp = bvec2(vec2(1.0, 2.0) < vec2(2.0, 1.0));
                    bvec4 merged = bvec4(bvec2(true, false), bvec2(false, true));
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        bool_type = re.search(r"(%\d+) = OpTypeBool", spv_code)
        assert bool_type is not None
        bool_vec4_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(bool_type.group(1))} 4",
            spv_code,
        )
        assert bool_vec4_type is not None

        assert re.search(
            rf"OpCompositeConstruct {re.escape(bool_vec4_type.group(1))} "
            r"%\d+ %\d+ %\d+ %\d+",
            spv_code,
        )
        assert "OpFOrdLessThan" in spv_code
        assert_vector_constructs_have_exact_scalar_operands(spv_code)
        assert "WARNING" not in spv_code

    def test_bool_vector_constructors_reject_numeric_lanes_and_guard_arity(self):
        source_code = """
        shader InvalidBoolVectorConstructors {
            compute {
                void main() {
                    bvec2 numeric = bvec2(1, 0);
                    bvec3 padded = bvec3(bvec2(true, false));
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert "Constructor bvec2 cannot convert int component to bool" in spv_code
        assert "Constructor bvec3 expected 3 components but got 2" in spv_code
        assert_vector_constructs_have_exact_scalar_operands(spv_code)

    def test_scalar_logical_ops_still_return_bool_type(self):
        source_code = """
        shader ScalarLogic {
            compute {
                void main() {
                    bool a = true;
                    bool b = false;
                    bool c = a && b;
                    bool d = a || b;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        bool_type = re.search(r"(%\d+) = OpTypeBool", spv_code)
        assert bool_type is not None
        bool_type_id = bool_type.group(1)

        assert re.search(
            rf"OpLogicalAnd {re.escape(bool_type_id)} %\d+ %\d+\b", spv_code
        )
        assert re.search(
            rf"OpLogicalOr {re.escape(bool_type_id)} %\d+ %\d+\b", spv_code
        )
        assert "WARNING" not in spv_code

    def test_scalar_vector_constructors_splat_spirv_components(self):
        source_code = """
        shader VectorSplat {
            compute {
                void main() {
                    vec4 zero = vec4(0.0);
                    ivec2 pixel = ivec2(0);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        float_type = re.search(r"(%\d+) = OpTypeFloat 32", spv_code)
        int_type = re.search(r"(%\d+) = OpTypeInt 32 1", spv_code)

        assert float_type is not None
        assert int_type is not None
        vec4_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(float_type.group(1))} 4",
            spv_code,
        )
        ivec2_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(int_type.group(1))} 2",
            spv_code,
        )
        float_zero = re.search(
            rf"(%\d+) = OpConstant {re.escape(float_type.group(1))} 0.0",
            spv_code,
        )
        int_zero = re.search(
            rf"(%\d+) = OpConstant {re.escape(int_type.group(1))} 0\b",
            spv_code,
        )
        assert vec4_type is not None
        assert ivec2_type is not None
        assert float_zero is not None
        assert int_zero is not None
        assert re.search(
            rf"OpCompositeConstruct {re.escape(vec4_type.group(1))} "
            rf"{re.escape(float_zero.group(1))} {re.escape(float_zero.group(1))} "
            rf"{re.escape(float_zero.group(1))} {re.escape(float_zero.group(1))}",
            spv_code,
        )
        assert re.search(
            rf"OpCompositeConstruct {re.escape(ivec2_type.group(1))} "
            rf"{re.escape(int_zero.group(1))} {re.escape(int_zero.group(1))}",
            spv_code,
        )
        assert "WARNING" not in spv_code

    def test_vector_constructors_flatten_vector_and_swizzle_arguments(self):
        source_code = """
        shader VectorConstructorFlattening {
            compute {
                void main() {
                    vec4 color = vec4(1.0, 2.0, 3.0, 4.0);
                    vec4 fromSwizzle = vec4(color.xy, 0.0, 1.0);
                    vec3 fromParts = vec3(color.rg, color.b);
                    vec4 fromVectors = vec4(vec2(1.0, 2.0), vec2(3.0, 4.0));
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert re.search(r"%\d+ = OpVectorShuffle %\d+ %\d+ %\d+ 0 1\b", spv_code)
        assert_vector_constructs_have_exact_scalar_operands(spv_code)
        assert "WARNING" not in spv_code

    def test_vector_constructors_guard_component_count_mismatches(self):
        source_code = """
        shader VectorConstructorComponentMismatches {
            compute {
                void main() {
                    vec4 padded = vec4(vec2(1.0, 2.0), 3.0);
                    vec2 truncated = vec2(vec2(1.0, 2.0), 3.0);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert "Constructor vec4 expected 4 components but got 3" in spv_code
        assert "Constructor vec2 expected 2 components but got 3" in spv_code
        assert_vector_constructs_have_exact_scalar_operands(spv_code)

    def test_matrix_constructors_flatten_scalar_vector_and_matrix_arguments(self):
        source_code = """
        shader MatrixConstructorFlattening {
            compute {
                void main() {
                    mat2 scalarMat = mat2(1.0, 2.0, 3.0, 4.0);
                    mat2 vectorMat = mat2(vec2(1.0, 2.0), vec2(3.0, 4.0));
                    mat3 columnMat = mat3(
                        vec3(1.0, 2.0, 3.0),
                        vec3(4.0, 5.0, 6.0),
                        vec3(7.0, 8.0, 9.0)
                    );
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert_vector_constructs_have_exact_scalar_operands(spv_code)
        assert_matrix_constructs_have_exact_column_operands(spv_code)
        assert "WARNING" not in spv_code

    def test_matrix_constructors_guard_component_count_mismatches(self):
        source_code = """
        shader MatrixConstructorComponentMismatches {
            compute {
                void main() {
                    mat2 tooFew = mat2(vec2(1.0, 2.0), 3.0);
                    mat2 tooMany = mat2(vec2(1.0, 2.0), vec2(3.0, 4.0), 5.0);
                    mat2 source = mat2(1.0, 2.0, 3.0, 4.0);
                    mat3 expanded = mat3(source);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert "Constructor mat2 expected 4 components but got 3" in spv_code
        assert "Constructor mat2 expected 4 components but got 5" in spv_code
        assert "Constructor mat3 expected 9 components but got 4" in spv_code
        assert_vector_constructs_have_exact_scalar_operands(spv_code)
        assert_matrix_constructs_have_exact_column_operands(spv_code)

    def test_double_matrix_constructors_use_double_columns(self):
        source_code = """
        shader DoubleMatrixConstructors {
            compute {
                void main() {
                    dmat2 scalarMat = dmat2(1.0, 2.0, 3.0, 4.0);
                    dmat2 vectorMat = dmat2(dvec2(1.0, 2.0), dvec2(3.0, 4.0));
                    dmat2x3 rectangular = dmat2x3(
                        1.0, 2.0, 3.0,
                        4.0, 5.0, 6.0
                    );
                    dmat3 identity = dmat3();
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        double_type = re.search(r"(%\d+) = OpTypeFloat 64", spv_code)
        assert double_type is not None
        assert "OpCapability Float64" in spv_code
        assert re.search(
            rf"%\d+ = OpTypeVector {re.escape(double_type.group(1))} 2\b",
            spv_code,
        )
        assert re.search(
            rf"%\d+ = OpTypeVector {re.escape(double_type.group(1))} 3\b",
            spv_code,
        )
        assert "Dmat" not in spv_code
        assert_vector_constructs_have_exact_scalar_operands(spv_code)
        assert_matrix_constructs_have_exact_column_operands(spv_code)
        assert "WARNING" not in spv_code

    def test_matrix_constructors_convert_between_float_and_double_matrices(self):
        source_code = """
        shader MixedPrecisionMatrixConstructors {
            compute {
                void main() {
                    mat2 floatMat = mat2(1.0, 2.0, 3.0, 4.0);
                    dmat2 doubleMat = dmat2(floatMat);
                    mat2 roundTrip = mat2(doubleMat);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert spv_code.count("OpFConvert") >= 8
        assert "Dmat" not in spv_code
        assert_vector_constructs_have_exact_scalar_operands(spv_code)
        assert_matrix_constructs_have_exact_column_operands(spv_code)
        assert "WARNING" not in spv_code

    def test_scalar_condition_vector_ternary_uses_selection_flow(self):
        source_code = """
        shader VectorSelect {
            compute {
                void main() {
                    bool useFirst = true;
                    vec4 selected = useFirst ? vec4(1.0) : vec4(0.0);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert not re.search(r"=\s+OpSelect\b", spv_code)
        assert "OpSelectionMerge" in spv_code
        assert "OpBranchConditional" in spv_code
        assert_spirv_function_variables_are_first_block_declarations(spv_code)
        assert "WARNING" not in spv_code

    def test_vector_scalar_arithmetic_splats_scalar_operands(self):
        source_code = """
        shader VectorScalarArithmetic {
            compute {
                void main() {
                    float weight = 0.5;
                    vec4 scaled = vec4(1.0) * weight;
                    vec4 shifted = weight + vec4(0.0);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        vec4_type = re.search(r"(%\d+) = OpTypeVector %\d+ 4", spv_code)
        weight = spirv_named_variable(spv_code, "weight", storage_class="Function")
        assert vec4_type is not None

        weight_load = re.search(rf"(%\d+) = OpLoad %\d+ {re.escape(weight)}", spv_code)
        assert weight_load is not None
        splatted_weight = re.search(
            rf"(%\d+) = OpCompositeConstruct {re.escape(vec4_type.group(1))} "
            rf"{re.escape(weight_load.group(1))} {re.escape(weight_load.group(1))} "
            rf"{re.escape(weight_load.group(1))} {re.escape(weight_load.group(1))}",
            spv_code,
        )
        assert splatted_weight is not None
        assert re.search(
            rf"OpFMul {re.escape(vec4_type.group(1))} %\d+ "
            rf"{re.escape(splatted_weight.group(1))}",
            spv_code,
        )
        assert re.search(rf"OpFAdd {re.escape(vec4_type.group(1))}", spv_code)
        assert "WARNING" not in spv_code

    def test_precise_variable_decorates_float_arithmetic_results_no_contraction(self):
        source_code = """
        shader PreciseArithmetic {
            compute {
                void main() {
                    float a = 2.0;
                    float b = 4.0;
                    float acc @precise = a * b + 3.0;
                    acc = -acc / 2.0;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        acc_id = spirv_named_variable(spv_code, "acc", storage_class="Function")
        decorated_ids = set(re.findall(r"OpDecorate (%\d+) NoContraction", spv_code))

        assert acc_id not in decorated_ids
        for opcode in ("OpFMul", "OpFAdd", "OpFNegate", "OpFDiv"):
            result_ids = spirv_result_ids_for_opcode(spv_code, opcode)
            assert result_ids, f"Missing {opcode}"
            assert set(result_ids) <= decorated_ids
        assert "WARNING" not in spv_code

    def test_relaxed_float_arithmetic_does_not_emit_no_contraction(self):
        source_code = """
        shader RelaxedArithmetic {
            compute {
                void main() {
                    float a = 2.0;
                    float b = 4.0;
                    float loose = a * b + 3.0;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert "NoContraction" not in spv_code
        assert "WARNING" not in spv_code

    def test_precise_global_output_assignment_decorates_rhs_arithmetic(self):
        source_code = """
        shader PreciseOutput {
            float inputValue @input @location(0);
            float fragColor @output @location(0) @precise;

            fragment {
                void main() {
                    fragColor = inputValue * 2.0;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        frag_color_id = spirv_named_variable(
            spv_code, "fragColor", storage_class="Output"
        )
        multiply_ids = spirv_result_ids_for_opcode(spv_code, "OpFMul")
        decorated_ids = set(re.findall(r"OpDecorate (%\d+) NoContraction", spv_code))

        assert frag_color_id not in decorated_ids
        assert multiply_ids
        assert set(multiply_ids) <= decorated_ids
        assert "WARNING" not in spv_code

    def test_compound_assignments_load_apply_op_and_store_result(self):
        source_code = """
        shader CompoundAssignments {
            compute {
                void main() {
                    float value = 1.0;
                    value += 2.0;
                    int bits = 7;
                    bits &= 3;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        value_id = spirv_named_variable(spv_code, "value", storage_class="Function")
        bits_id = spirv_named_variable(spv_code, "bits", storage_class="Function")
        add_ids = spirv_result_ids_for_opcode(spv_code, "OpFAdd")
        and_ids = spirv_result_ids_for_opcode(spv_code, "OpBitwiseAnd")

        assert add_ids
        assert and_ids
        assert any(
            f"OpStore {value_id} {result_id}" in spv_code for result_id in add_ids
        )
        assert any(
            f"OpStore {bits_id} {result_id}" in spv_code for result_id in and_ids
        )
        assert "NoContraction" not in spv_code
        assert "WARNING" not in spv_code

    def test_precise_compound_assignments_decorate_generated_float_ops(self):
        source_code = """
        shader PreciseCompoundAssignments {
            compute {
                void main() {
                    float acc @precise = 1.0;
                    acc *= 2.0;
                    acc += 3.0;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        acc_id = spirv_named_variable(spv_code, "acc", storage_class="Function")
        decorated_ids = set(re.findall(r"OpDecorate (%\d+) NoContraction", spv_code))

        assert acc_id not in decorated_ids
        for opcode in ("OpFMul", "OpFAdd"):
            result_ids = spirv_result_ids_for_opcode(spv_code, opcode)
            assert result_ids, f"Missing {opcode}"
            assert set(result_ids) <= decorated_ids
        assert "WARNING" not in spv_code

    def test_compound_assignments_handle_member_and_precise_array_targets(self):
        source_code = """
        shader CompoundSubtargets {
            struct Accum {
                float total;
            };

            compute {
                void main() {
                    Accum acc;
                    acc.total = 1.0;
                    acc.total += 2.0;

                    float values[2] @precise;
                    values[0] = 1.0;
                    values[0] *= 3.0;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        acc_id = spirv_named_variable(spv_code, "acc", storage_class="Function")
        values_id = spirv_named_variable(spv_code, "values", storage_class="Function")
        decorated_ids = set(re.findall(r"OpDecorate (%\d+) NoContraction", spv_code))

        member_accesses = re.findall(
            rf"(%\d+) = OpAccessChain %\d+ {re.escape(acc_id)} %\d+",
            spv_code,
        )
        add_ids = spirv_result_ids_for_opcode(spv_code, "OpFAdd")
        assert member_accesses
        assert add_ids
        assert any(
            f"OpStore {access_id} {add_id}" in spv_code
            for access_id in member_accesses
            for add_id in add_ids
        )

        element_accesses = re.findall(
            rf"(%\d+) = OpAccessChain %\d+ {re.escape(values_id)} %\d+",
            spv_code,
        )
        multiply_ids = spirv_result_ids_for_opcode(spv_code, "OpFMul")
        assert element_accesses
        assert multiply_ids
        assert any(
            f"OpStore {access_id} {multiply_id}" in spv_code
            for access_id in element_accesses
            for multiply_id in multiply_ids
        )
        assert values_id not in decorated_ids
        assert set(multiply_ids) <= decorated_ids
        assert not set(add_ids) & decorated_ids
        assert "WARNING" not in spv_code

    def test_vector_component_assignments_insert_updated_vector_components(self):
        source_code = """
        shader VectorComponentAssignments {
            compute {
                void main() {
                    vec4 color = vec4(1.0, 2.0, 3.0, 4.0);
                    color.r += 0.1;
                    color.g = color.g + 0.2;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        color_id = spirv_named_variable(spv_code, "color", storage_class="Function")
        insert_ids = re.findall(
            rf"(%\d+) = OpCompositeInsert %\d+ %\d+ %\d+ [01]\b",
            spv_code,
        )

        assert len(insert_ids) >= 2
        assert spv_code.count("OpFAdd") >= 2
        assert all(
            f"OpStore {color_id} {insert_id}" in spv_code for insert_id in insert_ids
        )
        assert "NoContraction" not in spv_code
        assert "WARNING" not in spv_code

    def test_precise_vector_component_compound_assignment_decorates_generated_op(self):
        source_code = """
        shader PreciseVectorComponentAssignments {
            compute {
                void main() {
                    vec4 color @precise = vec4(1.0, 2.0, 3.0, 4.0);
                    color.r += 0.1;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        color_id = spirv_named_variable(spv_code, "color", storage_class="Function")
        add_ids = spirv_result_ids_for_opcode(spv_code, "OpFAdd")
        decorated_ids = set(re.findall(r"OpDecorate (%\d+) NoContraction", spv_code))

        assert color_id not in decorated_ids
        assert add_ids
        assert set(add_ids) <= decorated_ids
        assert "OpCompositeInsert" in spv_code
        assert "WARNING" not in spv_code

    def test_vector_component_assignment_rejects_composite_rhs(self):
        source_code = """
        shader VectorComponentCompositeRhs {
            compute {
                void main() {
                    vec4 color = vec4(1.0, 2.0, 3.0, 4.0);
                    color.r = vec2(0.1, 0.2);
                    color.g += vec2(0.3, 0.4);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert "Cannot assign composite value to vector component r" in spv_code
        assert "Cannot assign composite value to vector component g" in spv_code
        assert "OpCompositeInsert" not in spv_code
        assert "OpFAdd" not in spv_code

    def test_nested_vector_component_assignments_update_base_lvalue(self):
        source_code = """
        shader NestedVectorComponentLvalues {
            struct Payload { vec4 color; }
            compute {
                void main() {
                    Payload payload;
                    vec4 colors[2];
                    payload.color.b = 0.5;
                    payload.color.g += 0.25;
                    colors[0].a = 0.75;
                    colors[1].r += 0.125;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        access_ids = re.findall(r"(%\d+) = OpAccessChain %\d+ %\d+ %\d+", spv_code)
        insert_ids = re.findall(
            r"(%\d+) = OpCompositeInsert %\d+ %\d+ %\d+ ([0123])\b",
            spv_code,
        )

        assert len(access_ids) == 4
        assert [component for _, component in insert_ids] == ["2", "1", "3", "0"]
        assert spv_code.count("OpCompositeExtract") == 2
        assert spv_code.count("OpFAdd") == 2
        assert all(
            re.search(rf"OpStore {re.escape(access_id)} %\d+", spv_code)
            for access_id in access_ids
        )
        assert "WARNING" not in spv_code

    def test_vector_swizzle_reads_emit_vector_shuffle(self):
        source_code = """
        shader VectorSwizzleReads {
            compute {
                void main() {
                    vec4 color = vec4(1.0, 2.0, 3.0, 4.0);
                    vec2 rg = color.rg;
                    vec3 bgr = color.bgr;
                    vec2 xx = color.xx;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert re.search(r"%\d+ = OpVectorShuffle %\d+ %\d+ %\d+ 0 1\b", spv_code)
        assert re.search(r"%\d+ = OpVectorShuffle %\d+ %\d+ %\d+ 2 1 0\b", spv_code)
        assert re.search(r"%\d+ = OpVectorShuffle %\d+ %\d+ %\d+ 0 0\b", spv_code)
        assert "WARNING" not in spv_code

    def test_vector_swizzle_assignment_inserts_each_written_component(self):
        source_code = """
        shader VectorSwizzleAssignments {
            compute {
                void main() {
                    vec4 color = vec4(1.0, 2.0, 3.0, 4.0);
                    color.xy = vec2(0.1, 0.2);
                    color.ba = vec2(0.3, 0.4);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        color_id = spirv_named_variable(spv_code, "color", storage_class="Function")
        inserted_components = re.findall(
            rf"%\d+ = OpCompositeInsert %\d+ %\d+ %\d+ ([0123])\b",
            spv_code,
        )

        assert inserted_components == ["0", "1", "2", "3"]
        assert spv_code.count("OpCompositeExtract") == 4
        assert spv_code.count(f"OpStore {color_id}") == 3
        assert "WARNING" not in spv_code

    def test_vector_swizzle_assignment_rejects_invalid_masks_and_rhs_sizes(self):
        source_code = """
        shader InvalidVectorSwizzleAssignments {
            compute {
                void main() {
                    vec4 color = vec4(1.0, 2.0, 3.0, 4.0);
                    vec3 small = vec3(1.0, 2.0, 3.0);
                    color.xx = vec2(0.1, 0.2);
                    color.xyz = vec2(0.3, 0.4);
                    vec2 invalidRead = small.rq;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert (
            "Cannot assign to vector swizzle xx with duplicate components" in spv_code
        )
        assert "Cannot assign 2-component vector to 3-component swizzle xyz" in spv_code
        assert "Invalid vector swizzle rq for v3float" in spv_code

    def test_vector_swizzle_compound_assignments_update_masked_components(self):
        source_code = """
        shader VectorSwizzleCompoundAssignments {
            compute {
                void main() {
                    vec4 color = vec4(1.0, 2.0, 3.0, 4.0);
                    color.xy += vec2(0.1, 0.2);
                    color.ba *= vec2(0.3, 0.4);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        color_id = spirv_named_variable(spv_code, "color", storage_class="Function")
        inserted_components = re.findall(
            r"%\d+ = OpCompositeInsert %\d+ %\d+ %\d+ ([0123])\b",
            spv_code,
        )

        assert re.search(r"%\d+ = OpVectorShuffle %\d+ %\d+ %\d+ 0 1\b", spv_code)
        assert re.search(r"%\d+ = OpVectorShuffle %\d+ %\d+ %\d+ 2 3\b", spv_code)
        assert spirv_result_ids_for_opcode(spv_code, "OpFAdd")
        assert spirv_result_ids_for_opcode(spv_code, "OpFMul")
        assert inserted_components == ["0", "1", "2", "3"]
        assert spv_code.count(f"OpStore {color_id}") == 3
        assert "WARNING" not in spv_code

    def test_precise_vector_swizzle_compound_decorates_vector_arithmetic(self):
        source_code = """
        shader PreciseVectorSwizzleCompoundAssignments {
            compute {
                void main() {
                    vec4 color @precise = vec4(1.0, 2.0, 3.0, 4.0);
                    color.xy += vec2(0.1, 0.2);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        color_id = spirv_named_variable(spv_code, "color", storage_class="Function")
        add_ids = spirv_result_ids_for_opcode(spv_code, "OpFAdd")
        decorated_ids = set(re.findall(r"OpDecorate (%\d+) NoContraction", spv_code))

        assert color_id not in decorated_ids
        assert add_ids
        assert set(add_ids) <= decorated_ids
        assert "OpVectorShuffle" in spv_code
        assert "OpCompositeInsert" in spv_code
        assert "WARNING" not in spv_code

    def test_vector_swizzle_compound_assignment_rejects_invalid_targets_and_rhs(self):
        source_code = """
        shader InvalidVectorSwizzleCompoundAssignments {
            compute {
                void main() {
                    vec4 color = vec4(1.0, 2.0, 3.0, 4.0);
                    color.xx += vec2(0.1, 0.2);
                    color.xyz += vec2(0.3, 0.4);
                    color.xy += 0.5;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert (
            "Cannot assign to vector swizzle xx with duplicate components" in spv_code
        )
        assert "Cannot assign 2-component vector to 3-component swizzle xyz" in spv_code
        assert "Cannot assign scalar value to vector swizzle xy" in spv_code
        assert "OpCompositeInsert" not in spv_code
        assert "OpFAdd" not in spv_code

    def test_nested_vector_swizzle_lvalues_update_base_lvalue(self):
        source_code = """
        shader NestedVectorSwizzleLvalues {
            struct Payload { vec4 color; }
            compute {
                void main() {
                    Payload payload;
                    vec4 colors[2];
                    payload.color.xy = vec2(0.1, 0.2);
                    colors[1].ba = vec2(0.3, 0.4);
                    payload.color.zw += vec2(0.5, 0.6);
                    colors[0].rg *= vec2(0.7, 0.8);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        access_ids = re.findall(r"(%\d+) = OpAccessChain %\d+ %\d+ %\d+", spv_code)
        inserted_components = re.findall(
            r"%\d+ = OpCompositeInsert %\d+ %\d+ %\d+ ([0123])\b",
            spv_code,
        )

        assert len(access_ids) == 4
        assert inserted_components == ["0", "1", "2", "3", "2", "3", "0", "1"]
        assert re.search(r"%\d+ = OpVectorShuffle %\d+ %\d+ %\d+ 2 3\b", spv_code)
        assert re.search(r"%\d+ = OpVectorShuffle %\d+ %\d+ %\d+ 0 1\b", spv_code)
        assert spirv_result_ids_for_opcode(spv_code, "OpFAdd")
        assert spirv_result_ids_for_opcode(spv_code, "OpFMul")
        assert all(
            re.search(rf"OpStore {re.escape(access_id)} %\d+", spv_code)
            for access_id in access_ids
        )
        assert "WARNING" not in spv_code

    def test_invalid_mixed_family_swizzle_writes_use_swizzle_warning(self):
        source_code = """
        shader InvalidMixedFamilySwizzleWrites {
            compute {
                void main() {
                    vec4 color = vec4(1.0, 2.0, 3.0, 4.0);
                    color.xg = vec2(0.1, 0.2);
                    color.xg += vec2(0.3, 0.4);
                    vec2 bad = color.xg;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert spv_code.count("Invalid vector swizzle xg for v4float") == 3
        assert "Could not find member xg" not in spv_code
        assert "Unsupported LHS type in assignment: MemberAccessNode" not in spv_code
        assert "OpCompositeInsert" not in spv_code
        assert "OpFAdd" not in spv_code

    def test_fmod_intrinsics_use_core_fmod_and_argument_result_type(self):
        source_code = """
        shader FmodIntrinsics {
            compute {
                void main() {
                    float y = mod(5.0, 2.0);
                    vec2 x = fmod(vec2(5.0, 7.0), vec2(2.0, 3.0));
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )
        float_type = re.search(r"(%\d+) = OpTypeFloat 32", spv_code)
        vec2_type = re.search(r"(%\d+) = OpTypeVector %\d+ 2", spv_code)

        assert float_type is not None
        assert vec2_type is not None
        assert re.search(
            rf"%\d+ = OpFMod {re.escape(float_type.group(1))} %\d+ %\d+",
            spv_code,
        )
        assert re.search(
            rf"%\d+ = OpFMod {re.escape(vec2_type.group(1))} %\d+ %\d+",
            spv_code,
        )
        assert " Fmod " not in spv_code

    def test_atan2_intrinsics_use_argument_result_type(self):
        source_code = """
        shader Atan2Intrinsics {
            compute {
                void main() {
                    float scalar = atan2(1.0, 2.0);
                    vec2 y = vec2(1.0, 2.0);
                    vec2 x = vec2(3.0, 4.0);
                    vec2 vector = atan2(y, x);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )
        float_type = re.search(r"(%\d+) = OpTypeFloat 32", spv_code)
        vec2_type = re.search(r"(%\d+) = OpTypeVector %\d+ 2", spv_code)

        assert float_type is not None
        assert vec2_type is not None
        assert re.search(
            rf"%\d+ = OpExtInst {re.escape(float_type.group(1))} %\d+ "
            r"Atan2 %\d+ %\d+",
            spv_code,
        )
        assert re.search(
            rf"%\d+ = OpExtInst {re.escape(vec2_type.group(1))} %\d+ "
            r"Atan2 %\d+ %\d+",
            spv_code,
        )

    def test_frac_and_scalar_saturate_builtins_lower_to_spirv_extinst(self):
        source_code = """
        shader BuiltinAliases {
            compute {
                void main() {
                    float f = frac(1.25);
                    float x = 2.0;
                    float s = saturate(x);
                    double d;
                    double ds = saturate(d);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )
        float_type = re.search(r"(%\d+) = OpTypeFloat 32", spv_code)
        double_type = re.search(r"(%\d+) = OpTypeFloat 64", spv_code)

        assert float_type is not None
        assert double_type is not None
        assert re.search(
            rf"%\d+ = OpExtInst {re.escape(float_type.group(1))} %\d+ Fract %\d+",
            spv_code,
        )
        assert re.search(
            rf"%\d+ = OpExtInst {re.escape(float_type.group(1))} %\d+ "
            r"FClamp %\d+ %\d+ %\d+",
            spv_code,
        )
        assert re.search(
            rf"%\d+ = OpExtInst {re.escape(double_type.group(1))} %\d+ "
            r"FClamp %\d+ %\d+ %\d+",
            spv_code,
        )
        assert " Frac " not in spv_code
        assert " Saturate " not in spv_code

    def test_vector_saturate_builtins_lower_to_vector_fclamp(self):
        source_code = """
        shader BuiltinAliases {
            compute {
                void main() {
                    vec2 v = vec2(-1.0, 2.0);
                    vec2 s = saturate(v);
                    dvec2 d = dvec2(-1.0, 2.0);
                    dvec2 ds = saturate(d);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )
        float_type = re.search(r"(%\d+) = OpTypeFloat 32", spv_code)
        double_type = re.search(r"(%\d+) = OpTypeFloat 64", spv_code)

        assert float_type is not None
        assert double_type is not None

        vec2_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(float_type.group(1))} 2",
            spv_code,
        )
        dvec2_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(double_type.group(1))} 2",
            spv_code,
        )

        assert vec2_type is not None
        assert dvec2_type is not None
        assert re.search(
            rf"%\d+ = OpExtInst {re.escape(vec2_type.group(1))} %\d+ "
            r"FClamp %\d+ %\d+ %\d+",
            spv_code,
        )
        assert re.search(
            rf"%\d+ = OpExtInst {re.escape(dvec2_type.group(1))} %\d+ "
            r"FClamp %\d+ %\d+ %\d+",
            spv_code,
        )
        assert " Saturate " not in spv_code

    def test_float_min_max_clamp_builtins_use_typed_spirv_extinsts(self):
        source_code = """
        shader MinMaxClamp {
            compute {
                void main() {
                    float x = 0.5;
                    float c = clamp(x, 0.0, 1.0);
                    float mn = min(c, 1.0);
                    float mx = max(mn, 0.0);
                    vec2 v = vec2(-1.0, 2.0);
                    vec2 vc = clamp(v, 0.0, 1.0);
                    vec2 vmn = min(vc, vec2(1.0, 1.0));
                    vec2 vmx = max(vmn, vec2(0.0, 0.0));
                    dvec2 dv = dvec2(-1.0, 2.0);
                    dvec2 dvc = clamp(dv, 0.0, 1.0);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )
        float_type = re.search(r"(%\d+) = OpTypeFloat 32", spv_code)
        double_type = re.search(r"(%\d+) = OpTypeFloat 64", spv_code)

        assert float_type is not None
        assert double_type is not None
        vec2_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(float_type.group(1))} 2",
            spv_code,
        )
        dvec2_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(double_type.group(1))} 2",
            spv_code,
        )

        assert vec2_type is not None
        assert dvec2_type is not None
        for op_name in ("FClamp", "FMin", "FMax"):
            assert re.search(
                rf"%\d+ = OpExtInst {re.escape(float_type.group(1))} %\d+ "
                rf"{op_name} %\d+",
                spv_code,
            )
            assert re.search(
                rf"%\d+ = OpExtInst {re.escape(vec2_type.group(1))} %\d+ "
                rf"{op_name} %\d+",
                spv_code,
            )
        assert re.search(
            rf"%\d+ = OpExtInst {re.escape(dvec2_type.group(1))} %\d+ "
            r"FClamp %\d+ %\d+ %\d+",
            spv_code,
        )

        assert " Clamp " not in spv_code
        assert " Min " not in spv_code
        assert " Max " not in spv_code

    def test_integer_min_max_clamp_builtins_use_signedness_spirv_extinsts(self):
        source_code = """
        shader MinMaxClamp {
            compute {
                void main() {
                    int i;
                    int j;
                    int lo;
                    int hi;
                    int ic = clamp(i, lo, hi);
                    int imn = min(i, j);
                    int imx = max(i, j);
                    uint u;
                    uint v;
                    uint ulo;
                    uint uhi;
                    uint uc = clamp(u, ulo, uhi);
                    uint umn = min(u, v);
                    uint umx = max(u, v);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )
        int_type = re.search(r"(%\d+) = OpTypeInt 32 1", spv_code)
        uint_type = re.search(r"(%\d+) = OpTypeInt 32 0", spv_code)

        assert int_type is not None
        assert uint_type is not None
        for op_name in ("SClamp", "SMin", "SMax"):
            assert re.search(
                rf"%\d+ = OpExtInst {re.escape(int_type.group(1))} %\d+ "
                rf"{op_name} %\d+",
                spv_code,
            )
        for op_name in ("UClamp", "UMin", "UMax"):
            assert re.search(
                rf"%\d+ = OpExtInst {re.escape(uint_type.group(1))} %\d+ "
                rf"{op_name} %\d+",
                spv_code,
            )

        assert " Clamp " not in spv_code
        assert " Min " not in spv_code
        assert " Max " not in spv_code

    def test_sign_builtins_use_typed_spirv_operations(self):
        source_code = """
        shader SignBuiltins {
            compute {
                void main() {
                    float f;
                    float sf = sign(f);
                    double d;
                    double sd = sign(d);
                    int i;
                    int si = sign(i);
                    ivec3 vi = ivec3(-1, 0, 2);
                    ivec3 vsi = sign(vi);
                    uint u;
                    uint su = sign(u);
                    uvec4 vu = uvec4(0, 1, 2, 3);
                    uvec4 vsu = sign(vu);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )
        bool_type = re.search(r"(%\d+) = OpTypeBool", spv_code)
        float_type = re.search(r"(%\d+) = OpTypeFloat 32", spv_code)
        double_type = re.search(r"(%\d+) = OpTypeFloat 64", spv_code)
        int_type = re.search(r"(%\d+) = OpTypeInt 32 1", spv_code)
        uint_type = re.search(r"(%\d+) = OpTypeInt 32 0", spv_code)

        assert bool_type is not None
        assert float_type is not None
        assert double_type is not None
        assert int_type is not None
        assert uint_type is not None

        ivec3_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(int_type.group(1))} 3",
            spv_code,
        )
        uvec4_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(uint_type.group(1))} 4",
            spv_code,
        )
        bvec4_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(bool_type.group(1))} 4",
            spv_code,
        )

        assert ivec3_type is not None
        assert uvec4_type is not None
        assert bvec4_type is not None
        assert re.search(
            rf"%\d+ = OpExtInst {re.escape(float_type.group(1))} %\d+ " r"FSign %\d+",
            spv_code,
        )
        assert re.search(
            rf"%\d+ = OpExtInst {re.escape(double_type.group(1))} %\d+ " r"FSign %\d+",
            spv_code,
        )
        assert re.search(
            rf"%\d+ = OpExtInst {re.escape(int_type.group(1))} %\d+ " r"SSign %\d+",
            spv_code,
        )
        assert re.search(
            rf"%\d+ = OpExtInst {re.escape(ivec3_type.group(1))} %\d+ " r"SSign %\d+",
            spv_code,
        )
        assert re.search(
            rf"%\d+ = OpUGreaterThan {re.escape(bool_type.group(1))} %\d+ %\d+",
            spv_code,
        )
        assert re.search(
            rf"%\d+ = OpSelect {re.escape(uint_type.group(1))} %\d+ %\d+ %\d+",
            spv_code,
        )
        assert re.search(
            rf"%\d+ = OpUGreaterThan {re.escape(bvec4_type.group(1))} %\d+ %\d+",
            spv_code,
        )
        assert re.search(
            rf"%\d+ = OpSelect {re.escape(uvec4_type.group(1))} %\d+ %\d+ %\d+",
            spv_code,
        )
        assert not re.search(
            rf"OpExtInst {re.escape(int_type.group(1))} %\d+ FSign",
            spv_code,
        )
        assert not re.search(
            rf"OpExtInst {re.escape(uint_type.group(1))} %\d+ FSign",
            spv_code,
        )
        assert " USign " not in spv_code

    def test_dot_uses_vector_component_result_type(self):
        source_code = """
        shader DotBuiltins {
            compute {
                void main() {
                    vec2 fv0 = vec2(1.0, 2.0);
                    vec2 fv1 = vec2(3.0, 4.0);
                    float fd = dot(fv0, fv1);

                    dvec2 dv0 = dvec2(1.0, 2.0);
                    dvec2 dv1 = dvec2(3.0, 4.0);
                    double dd = dot(dv0, dv1);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )
        float_type = re.search(r"(%\d+) = OpTypeFloat 32", spv_code)
        double_type = re.search(r"(%\d+) = OpTypeFloat 64", spv_code)

        assert float_type is not None
        assert double_type is not None
        assert re.search(
            rf"%\d+ = OpDot {re.escape(float_type.group(1))} %\d+ %\d+",
            spv_code,
        )
        assert re.search(
            rf"%\d+ = OpDot {re.escape(double_type.group(1))} %\d+ %\d+",
            spv_code,
        )

    def test_compute_stage_entry_point_generates_defined_function(self):
        source_code = """
        shader main {
            compute {
                void main() {
                    int x = 1;
                }
            }
        }
        """

        ast = Parser(Lexer(source_code).tokens).parse()
        spv_code = VulkanSPIRVCodeGen().generate(ast)

        entry_match = re.search(r'OpEntryPoint GLCompute %(\d+) "main"', spv_code)
        assert entry_match is not None
        entry_id = entry_match.group(1)
        assert f"%{entry_id} = OpFunction" in spv_code
        assert f"OpExecutionMode %{entry_id} LocalSize 1 1 1" in spv_code
        assert "OpName" in spv_code
        assert '"x"' in spv_code
        assert "OpStore" in spv_code
        assert "OpExecutionMode %main" not in spv_code

    def test_compute_stage_builtin_ids_emit_spirv_builtins(self):
        source_code = """
        shader ComputeBuiltins {
            compute {
                void main() {
                    uint gx = gl_GlobalInvocationID.x;
                    uint lx = gl_LocalInvocationID.x;
                    uint group = gl_WorkGroupID.x;
                    uint index = gl_LocalInvocationIndex;
                    uint size = gl_WorkGroupSize.z;
                    uint groups = gl_NumWorkGroups.x;
                }
            }
        }
        """

        ast = Parser(Lexer(source_code).tokens).parse()
        ast.stages[ShaderStage.COMPUTE].execution_config = {"local_size": (8, 4, 2)}
        spv_code = VulkanSPIRVCodeGen().generate(ast)

        entry_match = re.search(r'OpEntryPoint GLCompute %(\d+) "main"(.+)', spv_code)
        assert entry_match is not None
        assert f"OpExecutionMode %{entry_match.group(1)} LocalSize 8 4 2" in spv_code

        builtin_variables = {
            "gl_GlobalInvocationID": "GlobalInvocationId",
            "gl_LocalInvocationID": "LocalInvocationId",
            "gl_WorkGroupID": "WorkgroupId",
            "gl_LocalInvocationIndex": "LocalInvocationIndex",
            "gl_NumWorkGroups": "NumWorkgroups",
        }
        for variable_name, builtin_name in builtin_variables.items():
            variable_id = spirv_named_variable(
                spv_code, variable_name, storage_class="Input"
            )
            assert f"OpDecorate {variable_id} BuiltIn {builtin_name}" in spv_code
            assert variable_id in entry_match.group(2)

        workgroup_size_id = spirv_named_id(spv_code, "gl_WorkGroupSize")
        assert f"OpDecorate {workgroup_size_id} BuiltIn WorkgroupSize" in spv_code
        assert re.search(
            rf"{re.escape(workgroup_size_id)} = OpConstantComposite %\d+ "
            r"%\d+ %\d+ %\d+",
            spv_code,
        )
        assert re.search(r"%\d+ = OpCompositeExtract %\d+ %\d+ 2\b", spv_code)
        assert "WARNING" not in spv_code

    def test_vector_member_access_extracts_spirv_component(self):
        source_code = """
        shader VectorMemberAccess {
            compute {
                void main() {
                    vec3 position = vec3(1.0, 2.0, 3.0);
                    float x = position.x;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        float_type = re.search(r"(%\d+) = OpTypeFloat 32", spv_code)
        assert float_type is not None
        assert re.search(
            rf"OpCompositeExtract {re.escape(float_type.group(1))} %\d+ 0\b",
            spv_code,
        )
        assert "WARNING" not in spv_code

    def test_fragment_stage_execution_mode_uses_entry_function_id(self):
        source_code = """
        shader main {
            fragment {
                void main() {
                    float x = 1.0;
                }
            }
        }
        """

        ast = Parser(Lexer(source_code).tokens).parse()
        spv_code = VulkanSPIRVCodeGen().generate(ast)

        entry_match = re.search(r'OpEntryPoint Fragment %(\d+) "main"', spv_code)
        assert entry_match is not None
        entry_id = entry_match.group(1)
        assert f"%{entry_id} = OpFunction" in spv_code
        assert f"OpExecutionMode %{entry_id} OriginUpperLeft" in spv_code
        assert "OpExecutionMode %main" not in spv_code

    def test_vertex_stage_entry_point_has_no_fragment_execution_mode(self):
        source_code = """
        shader main {
            vertex {
                void main() {
                    float x = 1.0;
                }
            }
        }
        """

        ast = Parser(Lexer(source_code).tokens).parse()
        spv_code = VulkanSPIRVCodeGen().generate(ast)

        entry_match = re.search(r'OpEntryPoint Vertex %(\d+) "main"', spv_code)
        assert entry_match is not None
        entry_id = entry_match.group(1)
        assert f"%{entry_id} = OpFunction" in spv_code
        assert "OriginUpperLeft" not in spv_code
        assert "OpExecutionMode %main" not in spv_code

    def test_generate_resets_cached_state_between_calls(self):
        first_source = """
        shader First {
            compute {
                void main() {
                    int x = 1;
                }
            }
        }
        """
        second_source = """
        shader Second {
            compute {
                void main() {
                    float y = 2.0;
                }
            }
        }
        """

        codegen = VulkanSPIRVCodeGen()
        first = codegen.generate(Parser(Lexer(first_source).tokens).parse())
        second = codegen.generate(Parser(Lexer(second_source).tokens).parse())

        assert '%1 = OpExtInstImport "GLSL.std.450"' in first
        assert '%1 = OpExtInstImport "GLSL.std.450"' in second
        assert "%2 = OpTypeVoid" in second
        assert "%5 = OpTypeFloat 32" in second
        assert "%9 = OpTypeFunction %2" in second

        entry_match = re.search(r'OpEntryPoint GLCompute %(\d+) "main"', second)
        assert entry_match is not None
        assert f"%{entry_match.group(1)} = OpFunction" in second
        assert "OpExecutionMode %main" not in second

    def test_generated_spirv_uses_logical_module_layout(self):
        source_code = """
        struct Payload {
            float value;
        };

        shader LayoutCheck {
            image2D outputImage @rgba16f;

            Payload makePayload(float value) {
                Payload payload;
                payload.value = value;
                return payload;
            }

            compute {
                void main() {
                    Payload payload = makePayload(1.0);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )
        lines = spv_code.splitlines()

        def first_index(predicate):
            for index, line in enumerate(lines):
                if predicate(line):
                    return index
            raise AssertionError("Expected SPIR-V section was not emitted")

        capability = first_index(lambda line: line.startswith("OpCapability "))
        import_index = first_index(lambda line: " = OpExtInstImport " in line)
        memory_model = first_index(lambda line: line.startswith("OpMemoryModel "))
        entry_point = first_index(lambda line: line.startswith("OpEntryPoint "))
        execution_mode = first_index(lambda line: line.startswith("OpExecutionMode "))
        debug_name = first_index(lambda line: line.startswith("OpName "))
        decoration = first_index(lambda line: line.startswith("OpDecorate "))
        declaration = first_index(
            lambda line: re.match(r"%\d+ = (OpType|OpConstant)", line)
        )
        function = first_index(lambda line: re.match(r"%\d+ = OpFunction\b", line))

        assert capability < import_index < memory_model
        assert memory_model < entry_point < execution_mode
        assert execution_mode < debug_name < decoration < declaration < function
        assert all(
            not line.startswith(("OpName ", "OpMemberName ", "OpDecorate "))
            for line in lines[declaration:]
        )
        assert all(
            not re.match(r"%\d+ = (OpType|OpConstant)", line)
            for line in lines[function:]
        )
        assert "WARNING" not in spv_code

    def test_global_and_stage_variables_are_emitted_and_loaded(self):
        source_code = """
        shader Variables {
            float globalValue;

            compute {
                float stageValue;

                void main() {
                    float fromGlobal = globalValue;
                    float fromStage = stageValue;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert '"globalValue"' in spv_code
        assert '"stageValue"' in spv_code
        assert "OpTypePointer Private" in spv_code
        assert "OpVariable" in spv_code
        assert spv_code.count("OpLoad %5") >= 2
        assert "Unknown variable globalValue" not in spv_code
        assert "Unknown variable stageValue" not in spv_code
        assert "WARNING" not in spv_code

    def test_global_input_output_variables_emit_storage_and_decorations(self):
        source_code = """
        shader Variables {
            float inputValue @input;
            float outputValue @output;

            compute {
                void main() {
                    float localValue = inputValue;
                    outputValue = localValue;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert '"inputValue"' in spv_code
        assert '"outputValue"' in spv_code
        assert "OpVariable" in spv_code
        assert " Input" in spv_code
        assert " Output" in spv_code
        assert "OpDecorate" in spv_code
        assert "Location 0" in spv_code
        input_id = spirv_named_variable(spv_code, "inputValue", storage_class="Input")
        output_id = spirv_named_variable(
            spv_code, "outputValue", storage_class="Output"
        )
        assert re.search(
            rf'OpEntryPoint GLCompute %\d+ "main" '
            rf".*{re.escape(input_id)}.*{re.escape(output_id)}",
            spv_code,
        )
        assert "Unknown variable inputValue" not in spv_code
        assert "Unknown variable outputValue" not in spv_code
        assert "WARNING" not in spv_code

    def test_global_input_output_variables_respect_explicit_locations(self):
        source_code = """
        shader Variables {
            float inputValue @input @location(3);
            float nextInput @input;
            float outputValue @output @location(2);
            float nextOutput @output;

            compute {
                void main() {
                    float localValue = inputValue + nextInput;
                    outputValue = localValue;
                    nextOutput = localValue;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        input_id = spirv_named_variable(spv_code, "inputValue", storage_class="Input")
        next_input_id = spirv_named_variable(
            spv_code, "nextInput", storage_class="Input"
        )
        output_id = spirv_named_variable(
            spv_code, "outputValue", storage_class="Output"
        )
        next_output_id = spirv_named_variable(
            spv_code, "nextOutput", storage_class="Output"
        )

        assert f"OpDecorate {input_id} Location 3" in spv_code
        assert f"OpDecorate {next_input_id} Location 0" in spv_code
        assert f"OpDecorate {output_id} Location 2" in spv_code
        assert f"OpDecorate {next_output_id} Location 0" in spv_code
        assert "WARNING" not in spv_code

    def test_global_interface_variables_emit_component_and_index_decorations(self):
        source_code = """
        shader Variables {
            float color @input @location(1) @component(2);
            float fragColor @output @location(0) @index(1);

            compute {
                void main() {
                    fragColor = color;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        color_id = spirv_named_variable(spv_code, "color", storage_class="Input")
        frag_color_id = spirv_named_variable(
            spv_code, "fragColor", storage_class="Output"
        )

        assert f"OpDecorate {color_id} Location 1" in spv_code
        assert f"OpDecorate {color_id} Component 2" in spv_code
        assert f"OpDecorate {frag_color_id} Location 0" in spv_code
        assert f"OpDecorate {frag_color_id} Index 1" in spv_code
        assert "WARNING" not in spv_code

    def test_global_interface_variables_emit_interpolation_decorations(self):
        source_code = """
        shader Variables {
            float faceId @input @location(0) @flat;
            float edgeDistance @input @location(1) @noperspective @centroid;
            float sampleValue @input @location(2) @sample;
            float fragColor @output @location(0);

            fragment {
                void main() {
                    fragColor = faceId + edgeDistance + sampleValue;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        face_id = spirv_named_variable(spv_code, "faceId", storage_class="Input")
        edge_distance_id = spirv_named_variable(
            spv_code, "edgeDistance", storage_class="Input"
        )
        sample_value_id = spirv_named_variable(
            spv_code, "sampleValue", storage_class="Input"
        )

        assert f"OpDecorate {face_id} Flat" in spv_code
        assert f"OpDecorate {edge_distance_id} NoPerspective" in spv_code
        assert f"OpDecorate {edge_distance_id} Centroid" in spv_code
        assert f"OpDecorate {sample_value_id} Sample" in spv_code
        assert "OpCapability SampleRateShading" in spv_code
        assert "WARNING" not in spv_code

    def test_global_interface_variables_emit_invariant_decoration(self):
        source_code = """
        shader Variables {
            float inputValue @input @location(0) @invariant;
            float fragColor @output @location(0) @invariant;

            fragment {
                void main() {
                    fragColor = inputValue;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        input_id = spirv_named_variable(spv_code, "inputValue", storage_class="Input")
        frag_color_id = spirv_named_variable(
            spv_code, "fragColor", storage_class="Output"
        )

        assert f"OpDecorate {input_id} Invariant" in spv_code
        assert f"OpDecorate {frag_color_id} Invariant" in spv_code
        assert "WARNING" not in spv_code

    @pytest.mark.parametrize(
        ("declaration", "message"),
        [
            (
                "float value @input @location(0) @flat @noperspective;",
                "@flat and @noperspective",
            ),
            (
                "float value @input @location(0) @centroid @sample;",
                "@centroid and @sample",
            ),
        ],
    )
    def test_global_interface_variables_reject_conflicting_interpolation_attributes(
        self, declaration, message
    ):
        invalid_source = f"""
        shader Variables {{
            {declaration}
        }}
        """

        with pytest.raises(ValueError, match=message):
            VulkanSPIRVCodeGen().generate(Parser(Lexer(invalid_source).tokens).parse())

    def test_global_input_variables_allow_disjoint_component_packing(self):
        source_code = """
        shader Variables {
            float lane0 @input @location(0) @component(0);
            float lane1 @input @location(0) @component(1);
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        lane0_id = spirv_named_variable(spv_code, "lane0", storage_class="Input")
        lane1_id = spirv_named_variable(spv_code, "lane1", storage_class="Input")

        assert f"OpDecorate {lane0_id} Location 0" in spv_code
        assert f"OpDecorate {lane0_id} Component 0" in spv_code
        assert f"OpDecorate {lane1_id} Location 0" in spv_code
        assert f"OpDecorate {lane1_id} Component 1" in spv_code
        assert "WARNING" not in spv_code

    def test_global_output_variables_allow_same_location_with_different_index(self):
        source_code = """
        shader Variables {
            float color0 @output @location(0) @index(0);
            float color1 @output @location(0) @index(1);
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        color0_id = spirv_named_variable(spv_code, "color0", storage_class="Output")
        color1_id = spirv_named_variable(spv_code, "color1", storage_class="Output")

        assert f"OpDecorate {color0_id} Location 0" in spv_code
        assert f"OpDecorate {color0_id} Index 0" in spv_code
        assert f"OpDecorate {color1_id} Location 0" in spv_code
        assert f"OpDecorate {color1_id} Index 1" in spv_code
        assert "WARNING" not in spv_code

    @pytest.mark.parametrize(
        ("declaration", "message"),
        [
            ("float value @input @location(0) @component(-1);", "non-negative"),
            ("float value @input @location(0) @component(4);", "0..3"),
            ("vec3 value @input @location(0) @component(2);", "overflows"),
        ],
    )
    def test_global_interface_variables_reject_invalid_components(
        self, declaration, message
    ):
        source_code = f"""
        shader Variables {{
            {declaration}
        }}
        """

        with pytest.raises(ValueError, match=message):
            VulkanSPIRVCodeGen().generate(Parser(Lexer(source_code).tokens).parse())

    def test_global_interface_variables_reject_overlapping_component_slots(self):
        invalid_source = """
        shader Variables {
            float lane0 @input @location(0) @component(1);
            vec2 lane1 @input @location(0) @component(1);
        }
        """

        with pytest.raises(ValueError, match="Duplicate SPIR-V input location 0"):
            VulkanSPIRVCodeGen().generate(Parser(Lexer(invalid_source).tokens).parse())

    def test_global_input_output_auto_locations_skip_explicit_zero(self):
        source_code = """
        shader Variables {
            float inputValue @input @location(0);
            float nextInput @input;
            float outputValue @output @location(0);
            float nextOutput @output;

            compute {
                void main() {
                    float localValue = inputValue + nextInput;
                    outputValue = localValue;
                    nextOutput = localValue;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        input_id = spirv_named_variable(spv_code, "inputValue", storage_class="Input")
        next_input_id = spirv_named_variable(
            spv_code, "nextInput", storage_class="Input"
        )
        output_id = spirv_named_variable(
            spv_code, "outputValue", storage_class="Output"
        )
        next_output_id = spirv_named_variable(
            spv_code, "nextOutput", storage_class="Output"
        )

        assert f"OpDecorate {input_id} Location 0" in spv_code
        assert f"OpDecorate {next_input_id} Location 1" in spv_code
        assert f"OpDecorate {output_id} Location 0" in spv_code
        assert f"OpDecorate {next_output_id} Location 1" in spv_code
        assert "WARNING" not in spv_code

    @pytest.mark.parametrize("attribute", ["input", "output"])
    def test_global_interface_variables_reject_duplicate_explicit_locations(
        self, attribute
    ):
        source_code = f"""
        shader Variables {{
            float first @{attribute} @location(1);
            float second @{attribute} @location(1);
        }}
        """

        with pytest.raises(
            ValueError, match=f"Duplicate SPIR-V {attribute} location 1"
        ):
            VulkanSPIRVCodeGen().generate(Parser(Lexer(source_code).tokens).parse())

    def test_sampler_globals_emit_spirv_resource_types(self):
        source_code = """
        shader Resources {
            sampler linearSampler;
            sampler2d colorMap;
            samplercube environmentMap;
            sampler2dshadow shadowMap;
            sampler2dms colorMs;
            sampler2d textures[4];

            compute {
                void main() {
                    float x = 1.0;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert "OpTypeSampler" in spv_code
        assert spv_code.count("OpTypeSampledImage") >= 4
        assert "OpTypeImage" in spv_code
        assert " Cube " in spv_code
        assert " 2D 1 0 0 1 Unknown" in spv_code
        assert " 2D 0 0 1 1 Unknown" in spv_code
        assert "OpTypeArray" in spv_code
        assert spv_code.count("UniformConstant") >= 6
        assert "DescriptorSet 0" in spv_code
        assert "Binding 5" in spv_code
        assert '"linearSampler"' in spv_code
        assert '"colorMap"' in spv_code
        assert '"environmentMap"' in spv_code
        assert '"shadowMap"' in spv_code
        assert '"colorMs"' in spv_code
        assert '"textures"' in spv_code
        assert "using float as default" not in spv_code
        assert "WARNING" not in spv_code

    def test_image_globals_emit_spirv_storage_image_types(self):
        source_code = """
        shader Resources {
            image2D outputImage;
            image3D volumeImage;
            image2DArray layerImage;
            iimage2d signedImage;
            uimage2d counters;

            compute {
                void main() {
                    float x = 1.0;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert "OpTypeImage %5 2D 0 0 0 2 Unknown" in spv_code
        assert " 3D 0 0 0 2 Unknown" in spv_code
        assert " 2D 0 1 0 2 Unknown" in spv_code
        assert "OpTypeImage %4 2D 0 0 0 2 Unknown" in spv_code
        assert "OpTypeInt 32 0" in spv_code
        assert "OpTypeSampledImage" not in spv_code
        assert spv_code.count("UniformConstant") >= 5
        assert "Binding 4" in spv_code
        assert '"outputImage"' in spv_code
        assert '"volumeImage"' in spv_code
        assert '"layerImage"' in spv_code
        assert '"signedImage"' in spv_code
        assert '"counters"' in spv_code
        assert "using float as default" not in spv_code
        assert "WARNING" not in spv_code

    def test_image_globals_preserve_explicit_spirv_formats(self):
        source_code = """
        shader Resources {
            image2D halfColor @rgba16f;
            image2D normalizedColor @format(rgba8);
            uimage2D counters @r32ui;
            image2DArray layers @format(rgba16f);

            compute {
                void main() {
                    float x = 1.0;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert " 2D 0 0 0 2 Rgba16f" in spv_code
        assert " 2D 0 0 0 2 Rgba8" in spv_code
        assert " 2D 0 0 0 2 R32ui" in spv_code
        assert " 2D 0 1 0 2 Rgba16f" in spv_code
        assert re.search(r"OpTypeImage %\d+ 2D 0 0 0 2 R32ui", spv_code)
        assert "OpTypeInt 32 0" in spv_code
        assert '"halfColor"' in spv_code
        assert '"normalizedColor"' in spv_code
        assert '"counters"' in spv_code
        assert '"layers"' in spv_code
        assert "using float as default" not in spv_code
        assert "WARNING" not in spv_code

    def test_image_parameters_preserve_explicit_spirv_formats(self):
        source_code = """
        shader Resources {
            float touchImages(
                image2D scalarImage @r32f,
                image2D signedImage @r32i,
                image2D unsignedImage @format(r32ui)
            ) {
                return 0.0;
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert " 2D 0 0 0 2 R32f" in spv_code
        assert " 2D 0 0 0 2 R32i" in spv_code
        assert " 2D 0 0 0 2 R32ui" in spv_code
        assert "OpTypeInt 32 1" in spv_code
        assert "OpTypeInt 32 0" in spv_code
        assert '"scalarImage"' in spv_code
        assert '"signedImage"' in spv_code
        assert '"unsignedImage"' in spv_code
        assert "using float as default" not in spv_code
        assert "WARNING" not in spv_code

    def test_formatted_image_load_uses_matching_spirv_type(self):
        source_code = """
        shader Resources {
            image2D halfColor @rgba16f;
            image2D normalizedColor @format(rgba8);

            float touchImage(image2D value @format(rgba8)) {
                return 0.0;
            }

            compute {
                void main() {
                    float x = touchImage(normalizedColor);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        rgba8_type = re.search(r"(%\d+) = OpTypeImage %\d+ 2D 0 0 0 2 Rgba8", spv_code)
        rgba16f_type = re.search(
            r"(%\d+) = OpTypeImage %\d+ 2D 0 0 0 2 Rgba16f", spv_code
        )
        normalized_var = spirv_named_variable(
            spv_code, "normalizedColor", storage_class="UniformConstant"
        )

        assert rgba8_type is not None
        assert rgba16f_type is not None
        assert f"= OpLoad {rgba8_type.group(1)} {normalized_var}" in spv_code
        assert f"= OpLoad {rgba16f_type.group(1)} {normalized_var}" not in spv_code
        assert "WARNING" not in spv_code

    def test_resource_operations_emit_spirv_image_instructions(self):
        source_code = """
        shader Resources {
            image2D colorImage @rgba16f;
            uimage2D counters @r32ui;
            sampler2d colorMap;

            compute {
                void main() {
                    ivec2 pixel = ivec2(0, 0);
                    vec4 color = imageLoad(colorImage, pixel);
                    imageStore(colorImage, pixel, color);
                    uint count = imageLoad(counters, pixel);
                    imageStore(counters, pixel, count);
                    vec2 uv = vec2(0.5, 0.25);
                    vec4 sampled = texture(colorMap, uv);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        uint_type = re.search(r"(%\d+) = OpTypeInt 32 0", spv_code)
        vec4_type = re.search(r"(%\d+) = OpTypeVector %\d+ 4", spv_code)

        assert uint_type is not None
        assert vec4_type is not None
        assert spv_code.count("OpImageRead") == 2
        assert spv_code.count("OpImageWrite") == 2
        assert "OpImageSampleImplicitLod" not in spv_code
        assert "OpImageSampleExplicitLod" in spv_code
        assert f"OpImageRead {uint_type.group(1)}" in spv_code
        assert f"OpImageSampleExplicitLod {vec4_type.group(1)}" in spv_code
        assert "OpFunctionCall" not in spv_code
        assert "imageLoad" not in spv_code
        assert "imageStore" not in spv_code
        assert "WARNING" not in spv_code

    def test_user_defined_texture_function_shadows_resource_builtin(self):
        source_code = """
        shader ShadowResourceName {
            compute {
                float texture(float x) {
                    return x + 1.0;
                }

                void main() {
                    float y = texture(2.0);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert re.search(r"OpFunctionCall %\d+ %\d+ %\d+", spv_code)
        assert "texture requires" not in spv_code
        assert "WARNING" not in spv_code

    def test_forward_user_defined_texture_function_shadows_resource_builtin(self):
        source_code = """
        shader ShadowForwardResourceName {
            compute {
                float caller(float x) {
                    return texture(x);
                }

                float texture(float x) {
                    return x + 1.0;
                }

                void main() {
                    float y = caller(2.0);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert spv_code.count("OpFunctionCall") >= 2
        assert "texture requires" not in spv_code
        assert "WARNING" not in spv_code
        assert "OpImageSample" not in spv_code

    def test_forward_user_defined_saturate_function_shadows_builtin(self):
        source_code = """
        shader ShadowForwardBuiltinName {
            compute {
                float caller(float x) {
                    return saturate(x);
                }

                float saturate(float x) {
                    return x + 1.0;
                }

                void main() {
                    float y = caller(2.0);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert spv_code.count("OpFunctionCall") >= 2
        assert "FClamp" not in spv_code
        assert "WARNING" not in spv_code

    def test_forward_top_level_user_defined_saturate_function_shadows_builtin(self):
        source_code = """
        shader ShadowForwardTopLevelBuiltinName {
            float caller(float x) {
                return saturate(x);
            }

            float saturate(float x) {
                return x + 1.0;
            }

            compute {
                void main() {
                    float y = caller(2.0);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert spv_code.count("OpFunctionCall") >= 2
        assert "FClamp" not in spv_code
        assert "WARNING" not in spv_code

    def test_multisample_image_load_store_use_sample_operand(self):
        source_code = """
        shader Resources {
            image2DMS msColor @rgba16f;
            image2DMSArray msLayers @format(rgba8);
            uimage2DMS counters @r32ui;
            iimage2DMSArray signedLayers @r32i;

            compute {
                void main() {
                    int sampleIndex = 2;
                    ivec2 pixel = ivec2(4, 8);
                    ivec3 pixelLayer = ivec3(4, 8, 1);
                    vec4 color = imageLoad(msColor, pixel, sampleIndex);
                    imageStore(msColor, pixel, sampleIndex, color);
                    vec4 layer = imageLoad(msLayers, pixelLayer, sampleIndex);
                    imageStore(msLayers, pixelLayer, sampleIndex, layer);
                    uint count = imageLoad(counters, pixel, sampleIndex);
                    imageStore(counters, pixel, sampleIndex, count);
                    int signedValue = imageLoad(
                        signedLayers,
                        pixelLayer,
                        sampleIndex
                    );
                    imageStore(
                        signedLayers,
                        pixelLayer,
                        sampleIndex,
                        signedValue
                    );
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        int_type = re.search(r"(%\d+) = OpTypeInt 32 1", spv_code)
        uint_type = re.search(r"(%\d+) = OpTypeInt 32 0", spv_code)
        vec4_type = re.search(r"(%\d+) = OpTypeVector %\d+ 4", spv_code)
        assert int_type is not None
        assert uint_type is not None
        assert vec4_type is not None
        assert spv_code.count("OpCapability StorageImageMultisample") == 1
        assert spv_code.count("OpCapability ImageMSArray") == 1
        assert " 2D 0 0 1 2 Rgba16f" in spv_code
        assert " 2D 0 1 1 2 Rgba8" in spv_code
        assert " 2D 0 0 1 2 R32ui" in spv_code
        assert " 2D 0 1 1 2 R32i" in spv_code
        assert spv_code.count("OpImageRead") == 4
        assert spv_code.count("OpImageWrite") == 4
        assert f"OpImageRead {vec4_type.group(1)}" in spv_code
        assert f"OpImageRead {uint_type.group(1)}" in spv_code
        assert f"OpImageRead {int_type.group(1)}" in spv_code

        image_access_lines = [
            line
            for line in spv_code.splitlines()
            if "OpImageRead" in line or "OpImageWrite" in line
        ]
        assert len(image_access_lines) == 8
        assert all(" Sample " in line for line in image_access_lines)
        assert all(" Lod " not in line for line in image_access_lines)
        assert "OpFunctionCall" not in spv_code
        assert "imageLoad" not in spv_code
        assert "imageStore" not in spv_code
        assert "WARNING" not in spv_code

    def test_texture_lod_and_grad_emit_explicit_lod_sampling(self):
        source_code = """
        shader Resources {
            sampler2d colorMap;

            compute {
                void main() {
                    vec2 uv = vec2(0.25, 0.75);
                    vec4 mip = textureLod(colorMap, uv, 2.0);
                    vec2 ddx = vec2(0.1, 0.0);
                    vec2 ddy = vec2(0.0, 0.1);
                    vec4 grad = textureGrad(colorMap, uv, ddx, ddy);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        vec4_type = re.search(r"(%\d+) = OpTypeVector %\d+ 4", spv_code)

        assert vec4_type is not None
        assert spv_code.count("OpImageSampleExplicitLod") == 2
        assert f"OpImageSampleExplicitLod {vec4_type.group(1)}" in spv_code
        assert re.search(
            r"OpImageSampleExplicitLod %\d+ %\d+ %\d+ Lod %\d+",
            spv_code,
        )
        assert re.search(
            r"OpImageSampleExplicitLod %\d+ %\d+ %\d+ Grad %\d+ %\d+",
            spv_code,
        )
        assert "OpImageSampleImplicitLod" not in spv_code
        assert "OpFunctionCall" not in spv_code
        assert "textureLod" not in spv_code
        assert "textureGrad" not in spv_code
        assert "WARNING" not in spv_code

    def test_texture_offset_gather_and_fetch_emit_image_operations(self):
        source_code = """
        shader Resources {
            sampler2d colorMap;
            sampler2d textures[4];

            compute {
                void main() {
                    int layer = 1;
                    vec2 uv = vec2(0.25, 0.75);
                    ivec2 pixel = ivec2(4, 8);
                    ivec2 offset = ivec2(1, 0);
                    vec4 offsetColor = textureOffset(colorMap, uv, offset);
                    vec4 gathered = textureGather(colorMap, uv);
                    vec4 fetched = texelFetch(colorMap, pixel, 2);
                    vec4 arrayOffset = textureOffset(textures[layer], uv, offset);
                    vec4 arrayGathered = textureGather(textures[layer], uv, 1);
                    vec4 arrayFetched = texelFetch(textures[layer], pixel, 0);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )
        assert_spirv_function_variables_are_first_block_declarations(spv_code)

        sampled_image_type = re.search(r"(%\d+) = OpTypeSampledImage %\d+", spv_code)
        vec4_type = re.search(r"(%\d+) = OpTypeVector %\d+ 4", spv_code)
        assert sampled_image_type is not None
        assert vec4_type is not None

        array_type = re.search(
            rf"(%\d+) = OpTypeArray " rf"{re.escape(sampled_image_type.group(1))} %\d+",
            spv_code,
        )

        assert array_type is not None
        assert spv_code.count("OpImageSampleImplicitLod") == 0
        assert spv_code.count("OpImageSampleExplicitLod") == 2
        assert "ConstOffset" not in spv_code
        assert (
            len(re.findall(r"\bOpImageSampleExplicitLod\b.*\bOffset\b", spv_code)) == 2
        )
        assert spv_code.count("OpCapability ImageGatherExtended") == 1
        assert spv_code.count("OpImageGather") == 2
        assert spv_code.count("OpImageFetch") == 2
        assert len(re.findall(r"%\d+ = OpImage %\d+ %\d+", spv_code)) == 2
        assert f"OpImageGather {vec4_type.group(1)}" in spv_code
        assert f"OpImageFetch {vec4_type.group(1)}" in spv_code
        assert f"OpLoad {array_type.group(1)}" not in spv_code
        assert "OpFunctionCall" not in spv_code
        assert "textureOffset" not in spv_code
        assert "textureGather" not in spv_code
        assert "texelFetch" not in spv_code
        assert "WARNING" not in spv_code

    def test_literal_texture_offsets_emit_spirv_const_offset_operands(self):
        source_code = """
        shader Resources {
            sampler2d colorMap;
            sampler2dshadow shadowMap;

            compute {
                void main() {
                    vec2 uv = vec2(0.25, 0.75);
                    vec4 color = textureOffset(colorMap, uv, ivec2(-1, 0));
                    float shadow = textureCompareOffset(
                        shadowMap,
                        uv,
                        0.5,
                        ivec2(1, -1)
                    );
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        int_type = re.search(r"(%\d+) = OpTypeInt 32 1", spv_code)
        assert int_type is not None
        ivec2_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(int_type.group(1))} 2",
            spv_code,
        )
        assert ivec2_type is not None

        assert re.search(
            rf"%\d+ = OpConstantComposite {re.escape(ivec2_type.group(1))} "
            r"%\d+ %\d+",
            spv_code,
        )
        assert re.search(
            rf"%\d+ = OpConstant {re.escape(int_type.group(1))} -1", spv_code
        )
        assert len(re.findall(r"\bConstOffset\b", spv_code)) == 2
        assert not re.search(r"\bOffset\b", spv_code)
        assert "OpSNegate" not in spv_code
        assert "OpCapability ImageGatherExtended" not in spv_code
        assert "textureOffset" not in spv_code
        assert "textureCompareOffset" not in spv_code
        assert "WARNING" not in spv_code

    def test_multisample_texel_fetch_uses_sample_operand(self):
        source_code = """
        shader Resources {
            sampler2dms colorMs;
            sampler2dmsarray arrayMs;
            sampler linearSampler;
            sampler samplers[2];

            compute {
                void main() {
                    int sampleIndex = 2;
                    ivec2 pixel = ivec2(4, 8);
                    ivec3 pixelLayer = ivec3(4, 8, 1);
                    vec4 direct = texelFetch(colorMs, pixel, sampleIndex);
                    vec4 arrayFetch = texelFetch(arrayMs, pixelLayer, sampleIndex);
                    vec4 explicitSampler = texelFetch(
                        colorMs,
                        linearSampler,
                        pixel,
                        sampleIndex
                    );
                    vec4 indexedSampler = texelFetch(
                        arrayMs,
                        samplers[1],
                        pixelLayer,
                        sampleIndex
                    );
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        sampler_type = re.search(r"(%\d+) = OpTypeSampler", spv_code)
        vec4_type = re.search(r"(%\d+) = OpTypeVector %\d+ 4", spv_code)
        assert sampler_type is not None
        assert vec4_type is not None
        sampler_array_type = re.search(
            rf"(%\d+) = OpTypeArray {re.escape(sampler_type.group(1))} %\d+",
            spv_code,
        )

        fetch_lines = [line for line in spv_code.splitlines() if "OpImageFetch" in line]
        assert len(fetch_lines) == 4
        assert sampler_array_type is not None
        assert " 2D 0 0 1 1 Unknown" in spv_code
        assert " 2D 0 1 1 1 Unknown" in spv_code
        assert all(f"OpImageFetch {vec4_type.group(1)}" in line for line in fetch_lines)
        assert all(" Sample " in line for line in fetch_lines)
        assert all(" Lod " not in line for line in fetch_lines)
        assert len(re.findall(r"%\d+ = OpImage %\d+ %\d+", spv_code)) == 4
        assert f"OpLoad {sampler_array_type.group(1)}" not in spv_code
        assert "OpFunctionCall" not in spv_code
        assert "texelFetch" not in spv_code
        assert "WARNING" not in spv_code

    def test_sampled_operations_ignore_explicit_sampler_operands(self):
        source_code = """
        shader Resources {
            sampler2d colorMap;
            sampler2d textures[4];
            sampler linearSampler;
            sampler samplers[4];

            compute {
                void main() {
                    int layer = 1;
                    vec2 uv = vec2(0.25, 0.75);
                    vec2 ddx = vec2(0.1, 0.0);
                    vec2 ddy = vec2(0.0, 0.1);
                    ivec2 pixel = ivec2(4, 8);
                    ivec2 offset = ivec2(1, 0);
                    vec4 sampled = texture(colorMap, linearSampler, uv);
                    vec4 sampledArray = texture(textures[layer], samplers[layer], uv);
                    vec4 mip = textureLod(colorMap, linearSampler, uv, 2.0);
                    vec4 grad = textureGrad(
                        textures[layer],
                        samplers[layer],
                        uv,
                        ddx,
                        ddy
                    );
                    vec4 offsetColor = textureOffset(
                        colorMap,
                        linearSampler,
                        uv,
                        offset
                    );
                    vec4 gathered = textureGather(colorMap, linearSampler, uv, 1);
                    vec4 gatherOffset = textureGatherOffset(
                        textures[layer],
                        samplers[layer],
                        uv,
                        offset,
                        2
                    );
                    vec4 gatherOffsets = textureGatherOffsets(
                        colorMap,
                        linearSampler,
                        uv,
                        offset
                    );
                    vec4 fetched = texelFetch(
                        textures[layer],
                        samplers[layer],
                        pixel,
                        0
                    );
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )
        assert_spirv_function_variables_are_first_block_declarations(spv_code)

        sampled_image_type = re.search(r"(%\d+) = OpTypeSampledImage %\d+", spv_code)
        sampler_type = re.search(r"(%\d+) = OpTypeSampler", spv_code)
        vec4_type = re.search(r"(%\d+) = OpTypeVector %\d+ 4", spv_code)
        assert sampled_image_type is not None
        assert sampler_type is not None
        assert vec4_type is not None

        texture_array_type = re.search(
            rf"(%\d+) = OpTypeArray " rf"{re.escape(sampled_image_type.group(1))} %\d+",
            spv_code,
        )
        sampler_array_type = re.search(
            rf"(%\d+) = OpTypeArray {re.escape(sampler_type.group(1))} %\d+",
            spv_code,
        )
        assert texture_array_type is not None
        assert sampler_array_type is not None
        assert spv_code.count("OpImageSampleImplicitLod") == 0
        assert spv_code.count("OpImageSampleExplicitLod") == 5
        assert spv_code.count("OpImageGather") == 6
        assert spv_code.count("OpImageFetch") == 1
        assert spv_code.count("OpCapability ImageGatherExtended") == 1
        assert "ConstOffset" not in spv_code
        assert len(re.findall(r"\bOffset\b", spv_code)) == 6
        assert "ConstOffsets" not in spv_code
        assert f"OpImageGather {vec4_type.group(1)}" in spv_code
        assert f"OpImageFetch {vec4_type.group(1)}" in spv_code
        assert f"OpLoad {texture_array_type.group(1)}" not in spv_code
        assert f"OpLoad {sampler_array_type.group(1)}" not in spv_code
        assert "OpFunctionCall" not in spv_code
        for function_name in [
            "textureLod",
            "textureGrad",
            "textureOffset",
            "textureGather",
            "textureGatherOffset",
            "textureGatherOffsets",
            "texelFetch",
        ]:
            assert function_name not in spv_code
        assert "WARNING" not in spv_code

    def test_texture_gather_offsets_emit_lane_specific_spirv_gathers(self):
        source_code = """
        shader Resources {
            sampler2darray layers;

            compute {
                void main() {
                    vec3 uvLayer = vec3(0.25, 0.75, 1.0);
                    ivec2 offset0 = ivec2(-1, 0);
                    ivec2 offset1 = ivec2(0, -1);
                    ivec2 offset2 = ivec2(1, 0);
                    ivec2 offset3 = ivec2(0, 1);
                    int component = 2;
                    vec4 gathered = textureGatherOffsets(
                        layers,
                        uvLayer,
                        offset0,
                        offset1,
                        offset2,
                        offset3,
                        component
                    );
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        float_type = re.search(r"(%\d+) = OpTypeFloat 32", spv_code)
        assert float_type is not None
        vec4_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(float_type.group(1))} 4",
            spv_code,
        )
        assert vec4_type is not None

        gather_lines = re.findall(
            rf"%\d+ = OpImageGather {re.escape(vec4_type.group(1))} "
            r"%\d+ %\d+ %\d+ Offset %\d+",
            spv_code,
        )
        assert spv_code.count("OpCapability ImageGatherExtended") == 1
        assert spv_code.count("OpSNegate") == 2
        assert len(gather_lines) == 4
        for member_index in range(4):
            assert re.search(
                rf"OpCompositeExtract {re.escape(float_type.group(1))} "
                rf"%\d+ {member_index}\b",
                spv_code,
            )
        assert re.search(
            rf"%\d+ = OpCompositeConstruct {re.escape(vec4_type.group(1))} "
            r"%\d+ %\d+ %\d+ %\d+",
            spv_code,
        )
        assert "ConstOffsets" not in spv_code
        assert "textureGatherOffsets" not in spv_code
        assert "WARNING" not in spv_code

    def test_texture_gather_offsets_mix_const_and_dynamic_spirv_offsets(self):
        source_code = """
        shader Resources {
            sampler2darray layers;

            compute {
                void main() {
                    vec3 uvLayer = vec3(0.25, 0.75, 1.0);
                    ivec2 dynamicOffset0 = ivec2(0, -1);
                    ivec2 dynamicOffset1 = ivec2(0, 1);
                    vec4 gathered = textureGatherOffsets(
                        layers,
                        uvLayer,
                        ivec2(-1, 0),
                        dynamicOffset0,
                        ivec2(1, -1),
                        dynamicOffset1,
                        2
                    );
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        float_type = re.search(r"(%\d+) = OpTypeFloat 32", spv_code)
        int_type = re.search(r"(%\d+) = OpTypeInt 32 1", spv_code)
        assert float_type is not None
        assert int_type is not None
        vec4_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(float_type.group(1))} 4",
            spv_code,
        )
        ivec2_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(int_type.group(1))} 2",
            spv_code,
        )
        assert vec4_type is not None
        assert ivec2_type is not None

        assert (
            len(
                re.findall(
                    rf"%\d+ = OpImageGather {re.escape(vec4_type.group(1))} "
                    r"%\d+ %\d+ %\d+ ConstOffset %\d+",
                    spv_code,
                )
            )
            == 2
        )
        assert (
            len(
                re.findall(
                    rf"%\d+ = OpImageGather {re.escape(vec4_type.group(1))} "
                    r"%\d+ %\d+ %\d+ Offset %\d+",
                    spv_code,
                )
            )
            == 2
        )
        assert (
            len(
                re.findall(
                    rf"%\d+ = OpConstantComposite {re.escape(ivec2_type.group(1))} "
                    r"%\d+ %\d+",
                    spv_code,
                )
            )
            == 2
        )
        assert spv_code.count("OpCapability ImageGatherExtended") == 1
        assert "ConstOffsets" not in spv_code
        assert "textureGatherOffsets" not in spv_code
        assert "WARNING" not in spv_code

    def test_texture_and_image_queries_emit_spirv_query_operations(self):
        source_code = """
        shader Resources {
            sampler2d colorMap;
            sampler2d textures[4];
            sampler querySampler;
            image2D colorImage @rgba16f;
            uimage3D volumeImage @r32ui;

            compute {
                void main() {
                    int layer = 1;
                    vec2 uv = vec2(0.25, 0.75);
                    ivec2 texExtent = textureSize(colorMap, 2);
                    ivec2 arrayExtent = textureSize(textures[layer], 0);
                    int levels = textureQueryLevels(colorMap);
                    vec2 lod = textureQueryLod(colorMap, querySampler, uv);
                    ivec2 imageExtent = imageSize(colorImage);
                    ivec3 volumeExtent = imageSize(volumeImage);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        int_type = re.search(r"(%\d+) = OpTypeInt 32 1", spv_code)
        float_type = re.search(r"(%\d+) = OpTypeFloat 32", spv_code)
        sampled_image_type = re.search(r"(%\d+) = OpTypeSampledImage %\d+", spv_code)
        assert int_type is not None
        assert float_type is not None
        assert sampled_image_type is not None

        ivec2_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(int_type.group(1))} 2",
            spv_code,
        )
        ivec3_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(int_type.group(1))} 3",
            spv_code,
        )
        vec2_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(float_type.group(1))} 2",
            spv_code,
        )
        array_type = re.search(
            rf"(%\d+) = OpTypeArray " rf"{re.escape(sampled_image_type.group(1))} %\d+",
            spv_code,
        )

        assert ivec2_type is not None
        assert ivec3_type is not None
        assert vec2_type is not None
        assert array_type is not None
        assert spv_code.count("OpImageQuerySizeLod") == 2
        assert spv_code.count("OpImageQuerySize") == 4
        assert spv_code.count("OpImageQueryLevels") == 1
        assert spv_code.count("OpImageQueryLod") == 1
        assert 'OpExtension "SPV_KHR_compute_shader_derivatives"' in spv_code
        assert "OpCapability ComputeDerivativeGroupQuadsKHR" in spv_code
        assert "DerivativeGroupQuadsKHR" in spv_code
        assert re.search(r"OpExecutionMode %\d+ LocalSize 2 2 1", spv_code)
        assert f"OpImageQuerySizeLod {ivec2_type.group(1)}" in spv_code
        assert f"OpImageQuerySize {ivec2_type.group(1)}" in spv_code
        assert f"OpImageQuerySize {ivec3_type.group(1)}" in spv_code
        assert f"OpImageQueryLevels {int_type.group(1)}" in spv_code
        assert f"OpImageQueryLod {vec2_type.group(1)}" in spv_code
        assert f"OpLoad {array_type.group(1)}" not in spv_code
        assert "OpFunctionCall" not in spv_code
        assert "textureSize(" not in spv_code
        assert "imageSize(" not in spv_code
        assert "textureQueryLevels" not in spv_code
        assert "textureQueryLod" not in spv_code
        assert "WARNING" not in spv_code

    def test_dynamic_and_nested_resource_array_queries_emit_spirv_element_access(self):
        source_code = """
        shader Resources {
            sampler2d textureGrid[2][3];
            image2D imageGrid @rgba16f[2][4];

            void queryArrays(
                sampler2d dynamicTextures[],
                sampler2d fixedTextures[3],
                sampler2dms dynamicMsTextures[],
                int layer,
                int slot
            ) {
                ivec2 dynamicSize = textureSize(dynamicTextures[slot], 0);
                int fixedLevels = textureQueryLevels(fixedTextures[slot]);
                int dynamicSamples = textureSamples(dynamicMsTextures[slot]);
                ivec2 gridSize = textureSize(textureGrid[layer][slot], 0);
                ivec2 imageGridSize = imageSize(imageGrid[layer][slot]);
            }

            compute {
                void main() {}
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        float_type = re.search(r"(%\d+) = OpTypeFloat 32", spv_code)
        int_type = re.search(r"(%\d+) = OpTypeInt 32 1", spv_code)
        assert float_type is not None
        assert int_type is not None

        sampled_image_type = re.search(
            rf"(%\d+) = OpTypeImage {re.escape(float_type.group(1))} "
            r"2D 0 0 0 1 Unknown\n(%\d+) = OpTypeSampledImage \1",
            spv_code,
        )
        ms_sampled_image_type = re.search(
            rf"(%\d+) = OpTypeImage {re.escape(float_type.group(1))} "
            r"2D 0 0 1 1 Unknown\n(%\d+) = OpTypeSampledImage \1",
            spv_code,
        )
        storage_image_type = re.search(
            rf"(%\d+) = OpTypeImage {re.escape(float_type.group(1))} "
            r"2D 0 0 0 2 Rgba16f",
            spv_code,
        )
        assert sampled_image_type is not None
        assert ms_sampled_image_type is not None
        assert storage_image_type is not None

        sampled_type = sampled_image_type.group(2)
        storage_type = storage_image_type.group(1)

        sampled_inner_array = re.search(
            rf"(%\d+) = OpTypeArray {re.escape(sampled_type)} %\d+",
            spv_code,
        )
        sampled_outer_array = re.search(
            rf"(%\d+) = OpTypeArray {re.escape(sampled_inner_array.group(1))} %\d+",
            spv_code,
        )
        storage_inner_array = re.search(
            rf"(%\d+) = OpTypeArray {re.escape(storage_type)} %\d+",
            spv_code,
        )
        storage_outer_array = re.search(
            rf"(%\d+) = OpTypeArray {re.escape(storage_inner_array.group(1))} %\d+",
            spv_code,
        )
        assert sampled_inner_array is not None
        assert sampled_outer_array is not None
        assert storage_inner_array is not None
        assert storage_outer_array is not None

        sampled_outer_pointer = re.search(
            rf"(%\d+) = OpTypePointer UniformConstant "
            rf"{re.escape(sampled_outer_array.group(1))}",
            spv_code,
        )
        sampled_inner_pointer = re.search(
            rf"(%\d+) = OpTypePointer UniformConstant "
            rf"{re.escape(sampled_inner_array.group(1))}",
            spv_code,
        )
        sampled_element_pointer = re.search(
            rf"(%\d+) = OpTypePointer UniformConstant {re.escape(sampled_type)}",
            spv_code,
        )
        storage_outer_pointer = re.search(
            rf"(%\d+) = OpTypePointer UniformConstant "
            rf"{re.escape(storage_outer_array.group(1))}",
            spv_code,
        )
        storage_inner_pointer = re.search(
            rf"(%\d+) = OpTypePointer UniformConstant "
            rf"{re.escape(storage_inner_array.group(1))}",
            spv_code,
        )
        storage_element_pointer = re.search(
            rf"(%\d+) = OpTypePointer UniformConstant {re.escape(storage_type)}",
            spv_code,
        )
        assert sampled_outer_pointer is not None
        assert sampled_inner_pointer is not None
        assert sampled_element_pointer is not None
        assert storage_outer_pointer is not None
        assert storage_inner_pointer is not None
        assert storage_element_pointer is not None

        texture_grid = spirv_named_variable(
            spv_code,
            "textureGrid",
            sampled_outer_pointer.group(1),
            "UniformConstant",
        )
        image_grid = spirv_named_variable(
            spv_code, "imageGrid", storage_outer_pointer.group(1), "UniformConstant"
        )
        dynamic_textures = spirv_named_parameter(spv_code, "dynamicTextures")
        fixed_textures = spirv_named_parameter(
            spv_code, "fixedTextures", sampled_inner_pointer.group(1)
        )
        spirv_named_parameter(spv_code, "dynamicMsTextures")

        texture_grid_row = re.search(
            rf"(%\d+) = OpAccessChain {sampled_inner_pointer.group(1)} "
            rf"{texture_grid} %\d+",
            spv_code,
        )
        image_grid_row = re.search(
            rf"(%\d+) = OpAccessChain {storage_inner_pointer.group(1)} "
            rf"{image_grid} %\d+",
            spv_code,
        )
        assert texture_grid_row is not None
        assert image_grid_row is not None
        assert re.search(
            rf"OpAccessChain {sampled_element_pointer.group(1)} "
            rf"{texture_grid_row.group(1)} %\d+",
            spv_code,
        )
        assert re.search(
            rf"OpAccessChain {storage_element_pointer.group(1)} "
            rf"{image_grid_row.group(1)} %\d+",
            spv_code,
        )
        assert re.search(
            rf"OpAccessChain {sampled_element_pointer.group(1)} "
            rf"{dynamic_textures} %\d+",
            spv_code,
        )
        assert re.search(
            rf"OpAccessChain {sampled_element_pointer.group(1)} "
            rf"{fixed_textures} %\d+",
            spv_code,
        )

        assert f"OpTypeRuntimeArray {sampled_outer_array.group(1)}" not in spv_code
        assert f"OpLoad {sampled_inner_array.group(1)}" not in spv_code
        assert f"OpLoad {sampled_outer_array.group(1)}" not in spv_code
        assert f"OpLoad {storage_inner_array.group(1)}" not in spv_code
        assert f"OpLoad {storage_outer_array.group(1)}" not in spv_code
        assert len(re.findall(r"\bOpImageQuerySizeLod\b", spv_code)) == 2
        assert len(re.findall(r"\bOpImageQueryLevels\b", spv_code)) == 1
        assert len(re.findall(r"\bOpImageQuerySamples\b", spv_code)) == 1
        assert len(re.findall(r"\bOpImageQuerySize\b", spv_code)) == 1
        assert f"OpImageQuerySamples {int_type.group(1)}" in spv_code
        assert "textureSize" not in spv_code
        assert "textureSamples" not in spv_code
        assert "textureQueryLevels" not in spv_code
        assert "imageSize" not in spv_code
        assert "WARNING" not in spv_code

    def test_dynamic_resource_array_calls_use_spirv_fixed_pointer_signatures(self):
        source_code = """
        shader Resources {
            sampler2d textureGrid[2][3];
            image2D imageGrid @rgba16f[2][3];
            uimage2D counterGrid @r32ui[2][2];

            vec4 sampleDynamic(
                sampler2d dynGrid[][3],
                int layer,
                int slot,
                vec2 uv
            ) {
                return texture(dynGrid[layer][slot], uv);
            }

            vec4 readDynamic(
                image2D dynImages[][3] @rgba16f,
                int layer,
                int slot,
                ivec2 pixel
            ) {
                return imageLoad(dynImages[layer][slot], pixel);
            }

            uint readCounter(
                uimage2D dynCounters[][2] @r32ui,
                int layer,
                int slot,
                ivec2 pixel
            ) {
                return imageLoad(dynCounters[layer][slot], pixel);
            }

            void queryDynamic(
                sampler2d dynGrid[][3],
                image2D dynImages[][3] @rgba16f,
                int layer,
                int slot
            ) {
                ivec2 texSize = textureSize(dynGrid[layer][slot], 0);
                ivec2 imageSizeValue = imageSize(dynImages[layer][slot]);
            }

            compute {
                void main() {
                    vec2 uv = vec2(0.5, 0.25);
                    ivec2 pixel = ivec2(0, 0);
                    vec4 sampled = sampleDynamic(textureGrid, 1, 2, uv);
                    vec4 color = readDynamic(imageGrid, 1, 2, pixel);
                    uint count = readCounter(counterGrid, 1, 1, pixel);
                    queryDynamic(textureGrid, imageGrid, 1, 2);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        sampled_image_type = re.search(r"(%\d+) = OpTypeSampledImage %\d+", spv_code)
        rgba_image_type = re.search(
            r"(%\d+) = OpTypeImage %\d+ 2D 0 0 0 2 Rgba16f",
            spv_code,
        )
        uint_type = re.search(r"(%\d+) = OpTypeInt 32 0", spv_code)
        r32ui_image_type = re.search(
            r"(%\d+) = OpTypeImage %\d+ 2D 0 0 0 2 R32ui",
            spv_code,
        )
        assert sampled_image_type is not None
        assert rgba_image_type is not None
        assert uint_type is not None
        assert r32ui_image_type is not None

        sampled_row = re.search(
            rf"(%\d+) = OpTypeArray {re.escape(sampled_image_type.group(1))} %\d+",
            spv_code,
        )
        rgba_row = re.search(
            rf"(%\d+) = OpTypeArray {re.escape(rgba_image_type.group(1))} %\d+",
            spv_code,
        )
        r32ui_row = re.search(
            rf"(%\d+) = OpTypeArray {re.escape(r32ui_image_type.group(1))} %\d+",
            spv_code,
        )
        assert sampled_row is not None
        assert rgba_row is not None
        assert r32ui_row is not None

        sampled_grid = re.search(
            rf"(%\d+) = OpTypeArray {re.escape(sampled_row.group(1))} %\d+",
            spv_code,
        )
        rgba_grid = re.search(
            rf"(%\d+) = OpTypeArray {re.escape(rgba_row.group(1))} %\d+",
            spv_code,
        )
        r32ui_grid = re.search(
            rf"(%\d+) = OpTypeArray {re.escape(r32ui_row.group(1))} %\d+",
            spv_code,
        )
        assert sampled_grid is not None
        assert rgba_grid is not None
        assert r32ui_grid is not None

        sampled_grid_pointer = re.search(
            rf"(%\d+) = OpTypePointer UniformConstant "
            rf"{re.escape(sampled_grid.group(1))}",
            spv_code,
        )
        rgba_grid_pointer = re.search(
            rf"(%\d+) = OpTypePointer UniformConstant "
            rf"{re.escape(rgba_grid.group(1))}",
            spv_code,
        )
        r32ui_grid_pointer = re.search(
            rf"(%\d+) = OpTypePointer UniformConstant "
            rf"{re.escape(r32ui_grid.group(1))}",
            spv_code,
        )
        assert sampled_grid_pointer is not None
        assert rgba_grid_pointer is not None
        assert r32ui_grid_pointer is not None

        texture_grid = spirv_named_variable(
            spv_code, "textureGrid", sampled_grid_pointer.group(1), "UniformConstant"
        )
        image_grid = spirv_named_variable(
            spv_code, "imageGrid", rgba_grid_pointer.group(1), "UniformConstant"
        )
        counter_grid = spirv_named_variable(
            spv_code, "counterGrid", r32ui_grid_pointer.group(1), "UniformConstant"
        )

        dyn_grid_params = spirv_named_parameters(
            spv_code, "dynGrid", sampled_grid_pointer.group(1)
        )
        dyn_image_params = spirv_named_parameters(
            spv_code, "dynImages", rgba_grid_pointer.group(1)
        )
        dyn_counter_params = spirv_named_parameters(
            spv_code, "dynCounters", r32ui_grid_pointer.group(1)
        )
        assert len(dyn_grid_params) == 2
        assert len(dyn_image_params) == 2
        assert len(dyn_counter_params) == 1

        assert re.search(
            rf"OpFunctionCall %\d+ %\d+ {re.escape(texture_grid)}",
            spv_code,
        )
        assert re.search(
            rf"OpFunctionCall %\d+ %\d+ {re.escape(image_grid)}",
            spv_code,
        )
        assert re.search(
            rf"OpFunctionCall %\d+ %\d+ {re.escape(counter_grid)}",
            spv_code,
        )
        assert f"OpTypeRuntimeArray {sampled_row.group(1)}" not in spv_code
        assert f"OpTypeRuntimeArray {rgba_row.group(1)}" not in spv_code
        assert f"OpTypeRuntimeArray {r32ui_row.group(1)}" not in spv_code
        assert f"OpLoad {sampled_grid.group(1)}" not in spv_code
        assert f"OpLoad {rgba_grid.group(1)}" not in spv_code
        assert f"OpLoad {r32ui_grid.group(1)}" not in spv_code
        assert "OpImageSampleExplicitLod" in spv_code
        assert spv_code.count("OpImageRead") >= 2
        assert f"OpImageRead {uint_type.group(1)}" in spv_code
        assert "OpImageQuerySizeLod" in spv_code
        assert "OpImageQuerySize" in spv_code
        assert "textureSize(" not in spv_code
        assert "imageSize(" not in spv_code
        assert "WARNING" not in spv_code

    def test_forwarded_dynamic_resource_arrays_use_spirv_fixed_pointer_signatures(self):
        source_code = """
        shader Resources {
            sampler2d textureGrid[2][3];
            image2D imageGrid @rgba16f[2][3];
            uimage2D counterGrid @r32ui[2][2];

            vec4 leafSample(sampler2d dynGrid[][3], int layer, int slot, vec2 uv) {
                return texture(dynGrid[layer][slot], uv);
            }

            vec4 forwardSample(sampler2d dynGrid[][3], int layer, int slot, vec2 uv) {
                return leafSample(dynGrid, layer, slot, uv);
            }

            vec4 leafImage(
                image2D dynImages[][3] @rgba16f,
                int layer,
                int slot,
                ivec2 pixel
            ) {
                vec4 color = imageLoad(dynImages[layer][slot], pixel);
                imageStore(dynImages[layer][slot], pixel, color);
                return color;
            }

            vec4 forwardImage(
                image2D dynImages[][3] @rgba16f,
                int layer,
                int slot,
                ivec2 pixel
            ) {
                return leafImage(dynImages, layer, slot, pixel);
            }

            uint leafCounter(
                uimage2D dynCounters[][2] @r32ui,
                int layer,
                int slot,
                ivec2 pixel
            ) {
                uint count = imageLoad(dynCounters[layer][slot], pixel);
                imageStore(dynCounters[layer][slot], pixel, count);
                return count;
            }

            uint forwardCounter(
                uimage2D dynCounters[][2] @r32ui,
                int layer,
                int slot,
                ivec2 pixel
            ) {
                return leafCounter(dynCounters, layer, slot, pixel);
            }

            void leafQuery(
                sampler2d dynGrid[][3],
                image2D dynImages[][3] @rgba16f,
                int layer,
                int slot
            ) {
                ivec2 texSize = textureSize(dynGrid[layer][slot], 0);
                ivec2 imageSizeValue = imageSize(dynImages[layer][slot]);
            }

            void forwardQuery(
                sampler2d dynGrid[][3],
                image2D dynImages[][3] @rgba16f,
                int layer,
                int slot
            ) {
                leafQuery(dynGrid, dynImages, layer, slot);
            }

            compute {
                void main() {
                    vec2 uv = vec2(0.5, 0.25);
                    ivec2 pixel = ivec2(0, 0);
                    vec4 sampled = forwardSample(textureGrid, 1, 2, uv);
                    vec4 color = forwardImage(imageGrid, 1, 2, pixel);
                    uint count = forwardCounter(counterGrid, 1, 1, pixel);
                    forwardQuery(textureGrid, imageGrid, 1, 2);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        texture_grid = spirv_named_variable(
            spv_code, "textureGrid", storage_class="UniformConstant"
        )
        image_grid = spirv_named_variable(
            spv_code, "imageGrid", storage_class="UniformConstant"
        )
        counter_grid = spirv_named_variable(
            spv_code, "counterGrid", storage_class="UniformConstant"
        )

        texture_pointer = re.search(
            rf"{re.escape(texture_grid)} = OpVariable (%\d+) UniformConstant",
            spv_code,
        ).group(1)
        image_pointer = re.search(
            rf"{re.escape(image_grid)} = OpVariable (%\d+) UniformConstant",
            spv_code,
        ).group(1)
        counter_pointer = re.search(
            rf"{re.escape(counter_grid)} = OpVariable (%\d+) UniformConstant",
            spv_code,
        ).group(1)
        dyn_grid_params = spirv_named_parameters(spv_code, "dynGrid", texture_pointer)
        dyn_image_params = spirv_named_parameters(spv_code, "dynImages", image_pointer)
        dyn_counter_params = spirv_named_parameters(
            spv_code, "dynCounters", counter_pointer
        )
        assert len(dyn_grid_params) == 4
        assert len(dyn_image_params) == 4
        assert len(dyn_counter_params) == 2

        assert any(
            re.search(rf"OpFunctionCall %\d+ %\d+ {param_id}\b", spv_code)
            for param_id in dyn_grid_params
        )
        assert any(
            re.search(rf"OpFunctionCall %\d+ %\d+ {param_id}\b", spv_code)
            for param_id in dyn_image_params
        )
        assert any(
            re.search(rf"OpFunctionCall %\d+ %\d+ {param_id}\b", spv_code)
            for param_id in dyn_counter_params
        )
        assert re.search(
            rf"OpFunctionCall %\d+ %\d+ {re.escape(texture_grid)}",
            spv_code,
        )
        assert re.search(
            rf"OpFunctionCall %\d+ %\d+ {re.escape(image_grid)}",
            spv_code,
        )
        assert re.search(
            rf"OpFunctionCall %\d+ %\d+ {re.escape(counter_grid)}",
            spv_code,
        )
        assert "OpTypeRuntimeArray" not in spv_code
        assert "OpImageSampleExplicitLod" in spv_code
        assert spv_code.count("OpImageRead") >= 2
        assert "OpImageWrite" in spv_code
        assert "OpImageQuerySizeLod" in spv_code
        assert "OpImageQuerySize" in spv_code
        assert "textureSize(" not in spv_code
        assert "imageSize(" not in spv_code
        assert "WARNING" not in spv_code

    def test_mixed_fixed_dynamic_resource_arrays_use_spirv_row_and_grid_pointers(self):
        source_code = """
        shader Resources {
            sampler2d textureGrid[2][3];
            image2D imageGrid @rgba16f[2][3];

            void leafMixed(
                sampler2d dynamicGrid[][3],
                sampler2d fixedRow[3],
                image2D dynamicImages[][3] @rgba16f,
                image2D fixedImages[3] @rgba16f,
                int layer,
                int slot,
                vec2 uv,
                ivec2 pixel
            ) {
                vec4 dynamicSample = texture(dynamicGrid[layer][slot], uv);
                vec4 fixedSample = texture(fixedRow[slot], uv);
                vec4 dynamicColor = imageLoad(dynamicImages[layer][slot], pixel);
                vec4 fixedColor = imageLoad(fixedImages[slot], pixel);
                imageStore(dynamicImages[layer][slot], pixel, dynamicColor);
                imageStore(fixedImages[slot], pixel, fixedColor);
                ivec2 dynamicSize = textureSize(dynamicGrid[layer][slot], 0);
                ivec2 fixedSize = textureSize(fixedRow[slot], 0);
                ivec2 dynamicImageSize = imageSize(dynamicImages[layer][slot]);
                ivec2 fixedImageSize = imageSize(fixedImages[slot]);
            }

            void forwardMixed(
                sampler2d dynamicGrid[][3],
                sampler2d fixedRow[3],
                image2D dynamicImages[][3] @rgba16f,
                image2D fixedImages[3] @rgba16f,
                int layer,
                int slot,
                vec2 uv,
                ivec2 pixel
            ) {
                leafMixed(
                    dynamicGrid,
                    fixedRow,
                    dynamicImages,
                    fixedImages,
                    layer,
                    slot,
                    uv,
                    pixel
                );
            }

            compute {
                void main() {
                    vec2 uv = vec2(0.5, 0.25);
                    ivec2 pixel = ivec2(0, 0);
                    forwardMixed(
                        textureGrid,
                        textureGrid[1],
                        imageGrid,
                        imageGrid[1],
                        1,
                        2,
                        uv,
                        pixel
                    );
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        sampled_image_type = re.search(r"(%\d+) = OpTypeSampledImage %\d+", spv_code)
        rgba_image_type = re.search(
            r"(%\d+) = OpTypeImage %\d+ 2D 0 0 0 2 Rgba16f",
            spv_code,
        )
        assert sampled_image_type is not None
        assert rgba_image_type is not None

        def array_type(element_type):
            match = re.search(
                rf"(%\d+) = OpTypeArray {re.escape(element_type)} %\d+",
                spv_code,
            )
            assert match is not None
            return match.group(1)

        def uniform_pointer(pointed_type):
            match = re.search(
                rf"(%\d+) = OpTypePointer UniformConstant "
                rf"{re.escape(pointed_type)}",
                spv_code,
            )
            assert match is not None
            return match.group(1)

        sampled_row = array_type(sampled_image_type.group(1))
        sampled_grid = array_type(sampled_row)
        rgba_row = array_type(rgba_image_type.group(1))
        rgba_grid = array_type(rgba_row)
        sampled_row_pointer = uniform_pointer(sampled_row)
        sampled_grid_pointer = uniform_pointer(sampled_grid)
        rgba_row_pointer = uniform_pointer(rgba_row)
        rgba_grid_pointer = uniform_pointer(rgba_grid)

        def named_variable(name, pointer_type):
            return spirv_named_variable(spv_code, name, pointer_type, "UniformConstant")

        texture_grid = named_variable("textureGrid", sampled_grid_pointer)
        image_grid = named_variable("imageGrid", rgba_grid_pointer)

        def named_params(name, pointer_type):
            return spirv_named_parameters(spv_code, name, pointer_type)

        dynamic_grid_params = named_params("dynamicGrid", sampled_grid_pointer)
        fixed_row_params = named_params("fixedRow", sampled_row_pointer)
        dynamic_image_params = named_params("dynamicImages", rgba_grid_pointer)
        fixed_image_params = named_params("fixedImages", rgba_row_pointer)
        assert len(dynamic_grid_params) == 2
        assert len(fixed_row_params) == 2
        assert len(dynamic_image_params) == 2
        assert len(fixed_image_params) == 2

        sampled_row_access = re.search(
            rf"(%\d+) = OpAccessChain {re.escape(sampled_row_pointer)} "
            rf"{re.escape(texture_grid)} %\d+",
            spv_code,
        )
        rgba_row_access = re.search(
            rf"(%\d+) = OpAccessChain {re.escape(rgba_row_pointer)} "
            rf"{re.escape(image_grid)} %\d+",
            spv_code,
        )
        assert sampled_row_access is not None
        assert rgba_row_access is not None

        assert any(
            re.search(
                rf"OpFunctionCall %\d+ %\d+ {re.escape(dynamic_grid)} "
                rf"{re.escape(fixed_row)} {re.escape(dynamic_images)} "
                rf"{re.escape(fixed_images)}\b",
                spv_code,
            )
            for dynamic_grid in dynamic_grid_params
            for fixed_row in fixed_row_params
            for dynamic_images in dynamic_image_params
            for fixed_images in fixed_image_params
        )
        assert re.search(
            rf"OpFunctionCall %\d+ %\d+ {re.escape(texture_grid)} "
            rf"{re.escape(sampled_row_access.group(1))} "
            rf"{re.escape(image_grid)} {re.escape(rgba_row_access.group(1))}",
            spv_code,
        )

        assert "OpTypeRuntimeArray" not in spv_code
        assert "OpImageSampleExplicitLod" in spv_code
        assert spv_code.count("OpImageRead") >= 2
        assert "OpImageWrite" in spv_code
        assert "OpImageQuerySizeLod" in spv_code
        assert "OpImageQuerySize" in spv_code
        assert "textureSize(" not in spv_code
        assert "imageSize(" not in spv_code
        assert "WARNING" not in spv_code

    def test_multihop_reordered_resource_arrays_use_spirv_row_and_grid_pointers(self):
        source_code = """
        shader Resources {
            sampler2d textureGrid[2][3];
            image2D imageGrid @rgba16f[2][3];

            void leafShuffle(
                sampler2d dynA[][3],
                image2D fixedImageRow[3] @rgba16f,
                sampler2d fixedTexRow[3],
                image2D dynImageGrid[][3] @rgba16f,
                sampler2d dynAlias[][3],
                int layer,
                int slot,
                vec2 uv,
                ivec2 pixel
            ) {
                vec4 sampledA = texture(dynA[layer][slot], uv);
                vec4 sampledFixed = texture(fixedTexRow[slot], uv);
                vec4 sampledAlias = texture(dynAlias[layer][slot], uv);
                vec4 dynamicColor = imageLoad(dynImageGrid[layer][slot], pixel);
                vec4 fixedColor = imageLoad(fixedImageRow[slot], pixel);
                imageStore(dynImageGrid[layer][slot], pixel, dynamicColor);
                imageStore(fixedImageRow[slot], pixel, fixedColor);
                ivec2 dynSize = textureSize(dynA[layer][slot], 0);
                ivec2 fixedTexSize = textureSize(fixedTexRow[slot], 0);
                ivec2 aliasSize = textureSize(dynAlias[layer][slot], 0);
                ivec2 dynamicImageSize = imageSize(dynImageGrid[layer][slot]);
                ivec2 fixedImageSize = imageSize(fixedImageRow[slot]);
            }

            void midForward(
                sampler2d firstDyn[][3],
                sampler2d firstFixed[3],
                image2D firstDynImages[][3] @rgba16f,
                image2D firstFixedImages[3] @rgba16f,
                int layer,
                int slot,
                vec2 uv,
                ivec2 pixel
            ) {
                leafShuffle(
                    firstDyn,
                    firstFixedImages,
                    firstFixed,
                    firstDynImages,
                    firstDyn,
                    layer,
                    slot,
                    uv,
                    pixel
                );
            }

            void topForward(
                sampler2d topFixedTex[3],
                image2D topDynImages[][3] @rgba16f,
                sampler2d topDynTex[][3],
                image2D topFixedImages[3] @rgba16f,
                int layer,
                int slot,
                vec2 uv,
                ivec2 pixel
            ) {
                midForward(
                    topDynTex,
                    topFixedTex,
                    topDynImages,
                    topFixedImages,
                    layer,
                    slot,
                    uv,
                    pixel
                );
            }

            compute {
                void main() {
                    vec2 uv = vec2(0.5, 0.25);
                    ivec2 pixel = ivec2(0, 0);
                    topForward(
                        textureGrid[1],
                        imageGrid,
                        textureGrid,
                        imageGrid[1],
                        1,
                        2,
                        uv,
                        pixel
                    );
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        sampled_image_type = re.search(r"(%\d+) = OpTypeSampledImage %\d+", spv_code)
        rgba_image_type = re.search(
            r"(%\d+) = OpTypeImage %\d+ 2D 0 0 0 2 Rgba16f",
            spv_code,
        )
        assert sampled_image_type is not None
        assert rgba_image_type is not None

        def array_type(element_type):
            match = re.search(
                rf"(%\d+) = OpTypeArray {re.escape(element_type)} %\d+",
                spv_code,
            )
            assert match is not None
            return match.group(1)

        def uniform_pointer(pointed_type):
            match = re.search(
                rf"(%\d+) = OpTypePointer UniformConstant "
                rf"{re.escape(pointed_type)}",
                spv_code,
            )
            assert match is not None
            return match.group(1)

        sampled_row = array_type(sampled_image_type.group(1))
        sampled_grid = array_type(sampled_row)
        rgba_row = array_type(rgba_image_type.group(1))
        rgba_grid = array_type(rgba_row)
        sampled_row_pointer = uniform_pointer(sampled_row)
        sampled_grid_pointer = uniform_pointer(sampled_grid)
        rgba_row_pointer = uniform_pointer(rgba_row)
        rgba_grid_pointer = uniform_pointer(rgba_grid)

        def named_variable(name, pointer_type):
            return spirv_named_variable(spv_code, name, pointer_type, "UniformConstant")

        def named_param(name, pointer_type):
            return spirv_named_parameter(spv_code, name, pointer_type)

        texture_grid = named_variable("textureGrid", sampled_grid_pointer)
        image_grid = named_variable("imageGrid", rgba_grid_pointer)

        dyn_a = named_param("dynA", sampled_grid_pointer)
        dyn_alias = named_param("dynAlias", sampled_grid_pointer)
        dyn_image_grid = named_param("dynImageGrid", rgba_grid_pointer)
        fixed_image_row = named_param("fixedImageRow", rgba_row_pointer)
        fixed_tex_row = named_param("fixedTexRow", sampled_row_pointer)
        first_dyn = named_param("firstDyn", sampled_grid_pointer)
        first_dyn_images = named_param("firstDynImages", rgba_grid_pointer)
        first_fixed = named_param("firstFixed", sampled_row_pointer)
        first_fixed_images = named_param("firstFixedImages", rgba_row_pointer)
        top_fixed_tex = named_param("topFixedTex", sampled_row_pointer)
        top_dyn_images = named_param("topDynImages", rgba_grid_pointer)
        top_dyn_tex = named_param("topDynTex", sampled_grid_pointer)
        top_fixed_images = named_param("topFixedImages", rgba_row_pointer)

        assert dyn_a != dyn_alias
        assert dyn_image_grid != fixed_image_row
        assert fixed_tex_row != top_fixed_tex
        assert first_dyn != top_dyn_tex
        assert first_dyn_images != top_dyn_images
        assert first_fixed != top_fixed_tex
        assert first_fixed_images != top_fixed_images

        sampled_row_access = re.search(
            rf"(%\d+) = OpAccessChain {re.escape(sampled_row_pointer)} "
            rf"{re.escape(texture_grid)} %\d+",
            spv_code,
        )
        rgba_row_access = re.search(
            rf"(%\d+) = OpAccessChain {re.escape(rgba_row_pointer)} "
            rf"{re.escape(image_grid)} %\d+",
            spv_code,
        )
        assert sampled_row_access is not None
        assert rgba_row_access is not None

        assert re.search(
            rf"OpFunctionCall %\d+ %\d+ {re.escape(first_dyn)} "
            rf"{re.escape(first_fixed_images)} {re.escape(first_fixed)} "
            rf"{re.escape(first_dyn_images)} {re.escape(first_dyn)}\b",
            spv_code,
        )
        assert re.search(
            rf"OpFunctionCall %\d+ %\d+ {re.escape(top_dyn_tex)} "
            rf"{re.escape(top_fixed_tex)} {re.escape(top_dyn_images)} "
            rf"{re.escape(top_fixed_images)}\b",
            spv_code,
        )
        assert re.search(
            rf"OpFunctionCall %\d+ %\d+ {re.escape(sampled_row_access.group(1))} "
            rf"{re.escape(image_grid)} {re.escape(texture_grid)} "
            rf"{re.escape(rgba_row_access.group(1))}",
            spv_code,
        )

        assert "OpTypeRuntimeArray" not in spv_code
        assert "OpImageSampleExplicitLod" in spv_code
        assert spv_code.count("OpImageRead") >= 2
        assert "OpImageWrite" in spv_code
        assert "OpImageQuerySizeLod" in spv_code
        assert "OpImageQuerySize" in spv_code
        assert "textureSize(" not in spv_code
        assert "imageSize(" not in spv_code
        assert "WARNING" not in spv_code

    def test_resource_calls_in_control_flow_use_spirv_selection_and_calls(self):
        source_code = """
        shader Resources {
            sampler2d textureGrid[2][3];
            image2D imageGrid @rgba16f[2][3];

            vec4 sampleDynamic(
                sampler2d dynTex[][3],
                image2D dynImages[][3] @rgba16f,
                int layer,
                int slot,
                vec2 uv,
                ivec2 pixel
            ) {
                vec4 sampled = texture(dynTex[layer][slot], uv);
                vec4 loaded = imageLoad(dynImages[layer][slot], pixel);
                ivec2 texSize = textureSize(dynTex[layer][slot], 0);
                ivec2 imageSizeValue = imageSize(dynImages[layer][slot]);
                return sampled + loaded;
            }

            vec4 sampleFixed(
                sampler2d fixedTex[3],
                image2D fixedImages[3] @rgba16f,
                int slot,
                vec2 uv,
                ivec2 pixel
            ) {
                vec4 sampled = texture(fixedTex[slot], uv);
                vec4 loaded = imageLoad(fixedImages[slot], pixel);
                ivec2 texSize = textureSize(fixedTex[slot], 0);
                ivec2 imageSizeValue = imageSize(fixedImages[slot]);
                return sampled + loaded;
            }

            vec4 chooseBranch(
                sampler2d dynTex[][3],
                sampler2d fixedTex[3],
                image2D dynImages[][3] @rgba16f,
                image2D fixedImages[3] @rgba16f,
                bool useFixed,
                int layer,
                int slot,
                vec2 uv,
                ivec2 pixel
            ) {
                vec4 result = sampleDynamic(dynTex, dynImages, layer, slot, uv, pixel);
                if (useFixed) {
                    result = sampleFixed(fixedTex, fixedImages, slot, uv, pixel);
                }
                return result;
            }

            vec4 chooseReturn(
                sampler2d dynTex[][3],
                sampler2d fixedTex[3],
                image2D dynImages[][3] @rgba16f,
                image2D fixedImages[3] @rgba16f,
                bool useFixed,
                int layer,
                int slot,
                vec2 uv,
                ivec2 pixel
            ) {
                if (useFixed) {
                    return sampleFixed(fixedTex, fixedImages, slot, uv, pixel);
                }
                return sampleDynamic(dynTex, dynImages, layer, slot, uv, pixel);
            }

            vec4 chooseTernary(
                sampler2d dynTex[][3],
                sampler2d fixedTex[3],
                image2D dynImages[][3] @rgba16f,
                image2D fixedImages[3] @rgba16f,
                bool useFixed,
                int layer,
                int slot,
                vec2 uv,
                ivec2 pixel
            ) {
                return useFixed
                    ? sampleFixed(fixedTex, fixedImages, slot, uv, pixel)
                    : sampleDynamic(dynTex, dynImages, layer, slot, uv, pixel);
            }

            compute {
                void main() {
                    vec2 uv = vec2(0.5, 0.25);
                    ivec2 pixel = ivec2(0, 0);
                    vec4 branchValue = chooseBranch(
                        textureGrid,
                        textureGrid[1],
                        imageGrid,
                        imageGrid[1],
                        true,
                        1,
                        2,
                        uv,
                        pixel
                    );
                    vec4 returnValue = chooseReturn(
                        textureGrid,
                        textureGrid[1],
                        imageGrid,
                        imageGrid[1],
                        false,
                        1,
                        2,
                        uv,
                        pixel
                    );
                    vec4 ternaryValue = chooseTernary(
                        textureGrid,
                        textureGrid[1],
                        imageGrid,
                        imageGrid[1],
                        true,
                        1,
                        2,
                        uv,
                        pixel
                    );
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert "OpBranchConditional" in spv_code
        assert re.search(
            r"(?P<true_value>%\d+) = OpFunctionCall[^\n]+\n"
            r"(?P<false_value>%\d+) = OpFunctionCall[^\n]+\n"
            r"OpSelectionMerge (?P<merge>%\d+) None\n"
            r"OpBranchConditional %\d+ (?P<true_label>%\d+) (?P<false_label>%\d+)\n"
            r"(?P=true_label) = OpLabel\n"
            r"OpStore (?P<result>%\d+) (?P=true_value)\n"
            r"OpBranch (?P=merge)\n"
            r"(?P=false_label) = OpLabel\n"
            r"OpStore (?P=result) (?P=false_value)\n"
            r"OpBranch (?P=merge)\n"
            r"(?P=merge) = OpLabel\n"
            r"(?P<loaded>%\d+) = OpLoad %\d+ (?P=result)\n"
            r"OpReturnValue (?P=loaded)",
            spv_code,
        )
        assert not re.search(r"=\s+OpSelect\b", spv_code)
        assert not re.search(r"OpReturnValue %\d+\nOpBranch %\d+", spv_code)
        assert spv_code.count("OpFunctionCall") >= 8
        assert "OpImageSampleExplicitLod" in spv_code
        assert "OpImageRead" in spv_code
        assert "OpImageQuerySizeLod" in spv_code
        assert "OpImageQuerySize" in spv_code
        assert "texture(" not in spv_code
        assert "textureSize(" not in spv_code
        assert "imageSize(" not in spv_code
        assert "imageLoad(" not in spv_code
        assert "WARNING" not in spv_code

    def test_loop_node_emits_spirv_loop_control_and_body(self):
        source_code = """
        shader LoopNodeSmoke {
            void helper() {
                int i = 0;
                loop {
                    i = i + 1;
                    if (i > 2) {
                        break;
                    }
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert "OpLoopMerge" in spv_code
        assert "OpIAdd" in spv_code
        assert "OpSGreaterThan" in spv_code
        assert "OpSelectionMerge" in spv_code
        loop_merges = re.findall(r"OpLoopMerge (%\d+) (%\d+) None", spv_code)
        assert loop_merges
        assert any(
            f"OpBranch {merge_label}" in spv_code for merge_label, _ in loop_merges
        )
        assert "LoopNode(" not in spv_code
        assert "WARNING" not in spv_code

    def test_do_while_node_emits_condition_after_body_loop(self):
        source_code = """
        shader DoWhileSmoke {
            void helper() {
                int i = 0;
                do {
                    i = i + 1;
                } while (i < 4);
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        loop_header = re.search(
            r"(%\d+) = OpLabel\nOpLoopMerge (%\d+) (%\d+) None\nOpBranch (%\d+)",
            spv_code,
        )
        assert loop_header
        header_label, merge_label, continue_label, body_label = loop_header.groups()
        assert f"{body_label} = OpLabel" in spv_code
        assert spv_code.index("OpIAdd") < spv_code.index("OpSLessThan")

        continue_block = re.search(
            rf"{re.escape(continue_label)} = OpLabel\n"
            rf"(?P<body>.*?OpBranchConditional %\d+ "
            rf"{re.escape(header_label)} {re.escape(merge_label)})",
            spv_code,
            re.S,
        )
        assert continue_block
        assert "OpSLessThan" in continue_block.group("body")
        assert "DoWhileNode(" not in spv_code
        assert "WARNING" not in spv_code

    def test_for_in_node_emits_spirv_counted_loops(self):
        source_code = """
        shader ForInSmoke {
            void helper() {
                int total = 0;
                for i in 4 {
                    total = total + i;
                }
                for j in 2..5 {
                    total = total + j;
                }
                for k in 1..=4 {
                    total = total + k;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        loop_merges = re.findall(r"OpLoopMerge (%\d+) (%\d+) None", spv_code)
        assert len(loop_merges) >= 3
        assert "OpSLessThan" in spv_code
        assert "OpSLessThanEqual" in spv_code
        assert spv_code.count("OpIAdd") >= 6
        assert spirv_named_ids(spv_code, "i")
        assert spirv_named_ids(spv_code, "j")
        assert spirv_named_ids(spv_code, "k")
        assert "ForInNode(" not in spv_code
        assert "RangeNode(" not in spv_code
        assert "WARNING" not in spv_code

    def test_match_node_emits_spirv_selection_chain(self):
        source_code = """
        shader MatchSmoke {
            int helper(int mode) {
                int value = 0;
                match mode {
                    0 => {
                        value = 1;
                    }
                    _ => {
                        value = 2;
                    }
                }
                return value;
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert "OpSelectionMerge" in spv_code
        assert "OpBranchConditional" in spv_code
        assert "OpIEqual" in spv_code
        assert "OpStore" in spv_code
        assert "MatchNode(" not in spv_code
        assert "MatchArmNode(" not in spv_code
        assert "WARNING" not in spv_code

    def test_switch_node_emits_spirv_selection_chain_and_default(self):
        source_code = """
        shader SwitchSmoke {
            int helper(int mode) {
                int value = 0;
                switch (mode) {
                    case 0:
                        value = 1;
                        break;
                    case 1:
                        value = 2;
                        break;
                    default:
                        value = 3;
                }
                return value;
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert "OpSelectionMerge" in spv_code
        assert "OpBranchConditional" in spv_code
        assert spv_code.count("OpIEqual") >= 2
        assert spv_code.count("OpStore") >= 3
        assert "SwitchNode(" not in spv_code
        assert "CaseNode(" not in spv_code
        assert "WARNING" not in spv_code

    def test_switch_break_uses_nearest_break_target_inside_nested_loop(self):
        source_code = """
        shader SwitchNestedLoopSmoke {
            void helper(int mode) {
                switch (mode) {
                    case 0:
                        loop {
                            break;
                        }
                        break;
                    default:
                        break;
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        loop_merges = re.findall(r"OpLoopMerge (%\d+) (%\d+) None", spv_code)
        assert len(loop_merges) == 1
        inner_merge, inner_continue = loop_merges[0]
        assert re.search(
            rf"OpLoopMerge {re.escape(inner_merge)} {re.escape(inner_continue)} None\n"
            rf"OpBranch %\d+\n"
            rf"%\d+ = OpLabel\n"
            rf"OpBranch {re.escape(inner_merge)}",
            spv_code,
        )
        assert "OpSelectionMerge" in spv_code
        assert "WARNING" not in spv_code

    def test_match_guarded_arm_rejected_for_spirv_codegen(self):
        invalid_code = """
        shader MatchSmoke {
            int helper(int mode) {
                int value = 0;
                match mode {
                    0 if mode > 0 => {
                        value = 1;
                    }
                    _ => {
                        value = 2;
                    }
                }
                return value;
            }
        }
        """

        ast = Parser(Lexer(invalid_code).tokens).parse()

        with pytest.raises(ValueError, match="Unsupported match arm for SPIR-V"):
            VulkanSPIRVCodeGen().generate(ast)

    def test_resource_calls_in_loops_use_spirv_loop_control_and_indices(self):
        source_code = """
        shader Resources {
            sampler2d textureGrid[2][3];
            image2D imageGrid @rgba16f[2][3];

            vec4 sampleLoop(
                sampler2d dynTex[][3],
                sampler2d fixedTex[3],
                image2D dynImages[][3] @rgba16f,
                image2D fixedImages[3] @rgba16f,
                int layer,
                vec2 uv,
                ivec2 pixel
            ) {
                vec4 accum = vec4(0.0);
                for (int slot = 0; slot < 3; slot++) {
                    if (slot == 1) {
                        continue;
                    }
                    vec4 sampled = texture(dynTex[layer][slot], uv);
                    vec4 loaded = imageLoad(dynImages[layer][slot], pixel);
                    ivec2 texSize = textureSize(dynTex[layer][slot], 0);
                    ivec2 imageSizeValue = imageSize(dynImages[layer][slot]);
                    accum = accum + sampled + loaded;
                    if (slot == 2) {
                        break;
                    }
                }
                int fixedSlot = 0;
                while (fixedSlot < 3) {
                    accum = accum + texture(fixedTex[fixedSlot], uv);
                    accum = accum + imageLoad(fixedImages[fixedSlot], pixel);
                    ivec2 fixedTexSize = textureSize(fixedTex[fixedSlot], 0);
                    ivec2 fixedImageSize = imageSize(fixedImages[fixedSlot]);
                    fixedSlot++;
                }
                return accum;
            }

            compute {
                void main() {
                    vec2 uv = vec2(0.5, 0.25);
                    ivec2 pixel = ivec2(0, 0);
                    vec4 value = sampleLoop(
                        textureGrid,
                        textureGrid[1],
                        imageGrid,
                        imageGrid[1],
                        1,
                        uv,
                        pixel
                    );
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        loop_merges = re.findall(r"OpLoopMerge (%\d+) (%\d+) None", spv_code)
        assert len(loop_merges) >= 2
        for_merge, for_continue = loop_merges[0]
        assert f"OpBranch {for_continue}" in spv_code
        assert f"OpBranch {for_merge}" in spv_code
        assert "OpSLessThan" in spv_code
        assert "OpIEqual" in spv_code
        assert re.search(r"(%\d+) = OpIAdd %\d+ %\d+ %\d+\nOpStore %\d+ \1", spv_code)
        assert spv_code.count("OpImageSampleExplicitLod") >= 2
        assert spv_code.count("OpImageRead") >= 2
        assert spv_code.count("OpImageQuerySizeLod") >= 2
        assert spv_code.count("OpImageQuerySize") >= 2
        assert "OpFunctionCall" in spv_code
        assert "texture(" not in spv_code
        assert "textureSize(" not in spv_code
        assert "imageSize(" not in spv_code
        assert "imageLoad(" not in spv_code
        assert "BreakNode" not in spv_code
        assert "ContinueNode" not in spv_code
        assert "WARNING" not in spv_code

    def test_nested_loop_resource_indices_use_spirv_loop_stack_and_updates(self):
        source_code = """
        shader Resources {
            sampler2d textureGrid[2][3];
            image2D imageGrid @rgba16f[2][3];

            vec4 sampleNestedLoops(
                sampler2d dynTex[][3],
                image2D dynImages[][3] @rgba16f,
                vec2 uv,
                ivec2 pixel
            ) {
                vec4 accum = vec4(0.0);
                for (int layer = 0; layer < 2; layer++) {
                    if (layer == 1) {
                        break;
                    }
                    for (int slot = 0; slot < 3; slot++) {
                        if (slot == 1) {
                            continue;
                        }
                        vec4 sampled = texture(dynTex[layer][slot], uv);
                        vec4 loaded = imageLoad(dynImages[layer][slot], pixel);
                        imageStore(dynImages[layer][slot], pixel, loaded);
                        ivec2 texSize = textureSize(dynTex[layer][slot], 0);
                        ivec2 imageSizeValue = imageSize(dynImages[layer][slot]);
                        accum = accum + sampled + loaded;
                    }
                }
                return accum;
            }

            compute {
                void main() {
                    vec2 uv = vec2(0.5, 0.25);
                    ivec2 pixel = ivec2(0, 0);
                    vec4 value = sampleNestedLoops(textureGrid, imageGrid, uv, pixel);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        loop_merges = re.findall(r"OpLoopMerge (%\d+) (%\d+) None", spv_code)
        assert len(loop_merges) >= 2
        outer_merge, outer_continue = loop_merges[0]
        inner_merge, inner_continue = loop_merges[1]
        assert f"OpBranch {outer_merge}" in spv_code
        assert f"OpBranch {inner_continue}" in spv_code
        assert f"OpBranch {inner_merge}" not in spv_code
        assert f"OpBranch {outer_continue}" in spv_code
        assert spv_code.count("OpSLessThan") >= 2
        assert spv_code.count("OpIEqual") >= 2
        assert (
            len(
                re.findall(r"(%\d+) = OpIAdd %\d+ %\d+ %\d+\nOpStore %\d+ \1", spv_code)
            )
            >= 2
        )
        assert "OpImageSampleExplicitLod" in spv_code
        assert "OpImageRead" in spv_code
        assert "OpImageWrite" in spv_code
        assert "OpImageQuerySizeLod" in spv_code
        assert "OpImageQuerySize" in spv_code
        assert "OpFunctionCall" in spv_code
        assert "texture(" not in spv_code
        assert "textureSize(" not in spv_code
        assert "imageSize(" not in spv_code
        assert "imageLoad(" not in spv_code
        assert "imageStore(" not in spv_code
        assert "BreakNode" not in spv_code
        assert "ContinueNode" not in spv_code
        assert "WARNING" not in spv_code

    def test_struct_and_scalar_params_use_spirv_value_extracts_and_resource_calls(self):
        source_code = """
        struct SampleParams {
            vec2 uv;
            ivec2 pixel;
            int layer;
            int slot;
        };

        shader Resources {
            sampler2d textureGrid[2][3];
            image2D imageGrid @rgba16f[2][3];

            vec4 samplePacked(
                sampler2d dynTex[][3],
                SampleParams params,
                image2D dynImages[][3] @rgba16f,
                float weight,
                sampler2d fixedTex[3],
                image2D fixedImages[3] @rgba16f
            ) {
                vec4 dynamicSample = texture(
                    dynTex[params.layer][params.slot],
                    params.uv
                );
                vec4 fixedSample = texture(fixedTex[params.slot], params.uv);
                vec4 dynamicColor = imageLoad(
                    dynImages[params.layer][params.slot],
                    params.pixel
                );
                vec4 fixedColor = imageLoad(fixedImages[params.slot], params.pixel);
                ivec2 dynSize = textureSize(dynTex[params.layer][params.slot], 0);
                ivec2 fixedSize = textureSize(fixedTex[params.slot], 0);
                ivec2 dynImageSize = imageSize(dynImages[params.layer][params.slot]);
                ivec2 fixedImageSize = imageSize(fixedImages[params.slot]);
                vec4 combined = dynamicSample + fixedSample + dynamicColor + fixedColor;
                return combined * weight;
            }

            vec4 passPacked(
                SampleParams params,
                sampler2d firstDyn[][3],
                float weight,
                image2D firstImages[][3] @rgba16f,
                sampler2d fixedTex[3],
                image2D fixedImages[3] @rgba16f
            ) {
                return samplePacked(
                    firstDyn,
                    params,
                    firstImages,
                    weight,
                    fixedTex,
                    fixedImages
                );
            }

            compute {
                void main() {
                    SampleParams params;
                    params.uv = vec2(0.5, 0.25);
                    params.pixel = ivec2(0, 0);
                    params.layer = 1;
                    params.slot = 2;
                    vec4 value = passPacked(
                        params,
                        textureGrid,
                        0.5,
                        imageGrid,
                        textureGrid[1],
                        imageGrid[1]
                    );
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        params = spirv_named_parameters(spv_code, "params")
        first_dyn = spirv_named_parameter(spv_code, "firstDyn")
        first_images = spirv_named_parameter(spv_code, "firstImages")
        weight = spirv_named_parameters(spv_code, "weight")
        fixed_tex = spirv_named_parameters(spv_code, "fixedTex")
        fixed_images = spirv_named_parameters(spv_code, "fixedImages")
        assert len(params) >= 2
        assert len(weight) >= 2
        assert len(fixed_tex) >= 2
        assert len(fixed_images) >= 2

        sample_params = params[0]
        pass_params = params[-1]
        assert re.search(
            rf"OpCompositeExtract %\d+ {re.escape(sample_params)} 0", spv_code
        )
        assert re.search(
            rf"OpCompositeExtract %\d+ {re.escape(sample_params)} 1", spv_code
        )
        assert re.search(
            rf"OpCompositeExtract %\d+ {re.escape(sample_params)} 2", spv_code
        )
        assert re.search(
            rf"OpCompositeExtract %\d+ {re.escape(sample_params)} 3", spv_code
        )
        assert not re.search(
            rf"OpAccessChain %\d+ {re.escape(sample_params)} ", spv_code
        )
        assert re.search(
            rf"OpFunctionCall %\d+ %\d+ {re.escape(first_dyn)} "
            rf"{re.escape(pass_params)} {re.escape(first_images)} "
            rf"{re.escape(weight[-1])} {re.escape(fixed_tex[-1])} "
            rf"{re.escape(fixed_images[-1])}",
            spv_code,
        )
        assert "OpImageSampleExplicitLod" in spv_code
        assert "OpImageRead" in spv_code
        assert "OpImageQuerySizeLod" in spv_code
        assert "OpImageQuerySize" in spv_code
        assert "texture(" not in spv_code
        assert "textureSize(" not in spv_code
        assert "imageSize(" not in spv_code
        assert "imageLoad(" not in spv_code
        assert "WARNING" not in spv_code

    def test_resource_values_in_struct_returns_use_spirv_composites(self):
        source_code = """
        struct SampleResult {
            vec4 sampled;
            vec4 loaded;
            ivec2 texSize;
            ivec2 imageSizeValue;
        };

        shader Resources {
            sampler2d textureGrid[2][3];
            image2D imageGrid @rgba16f[2][3];

            SampleResult buildAssigned(
                sampler2d dynTex[][3],
                image2D dynImages[][3] @rgba16f,
                int layer,
                int slot,
                vec2 uv,
                ivec2 pixel
            ) {
                SampleResult result;
                result.sampled = texture(dynTex[layer][slot], uv);
                result.loaded = imageLoad(dynImages[layer][slot], pixel);
                result.texSize = textureSize(dynTex[layer][slot], 0);
                result.imageSizeValue = imageSize(dynImages[layer][slot]);
                return result;
            }

            SampleResult buildConstructed(
                sampler2d dynTex[][3],
                image2D dynImages[][3] @rgba16f,
                int layer,
                int slot,
                vec2 uv,
                ivec2 pixel
            ) {
                return SampleResult(
                    texture(dynTex[layer][slot], uv),
                    imageLoad(dynImages[layer][slot], pixel),
                    textureSize(dynTex[layer][slot], 0),
                    imageSize(dynImages[layer][slot])
                );
            }

            vec4 consumeResults(
                sampler2d dynTex[][3],
                image2D dynImages[][3] @rgba16f,
                int layer,
                int slot,
                vec2 uv,
                ivec2 pixel
            ) {
                SampleResult assigned = buildAssigned(
                    dynTex,
                    dynImages,
                    layer,
                    slot,
                    uv,
                    pixel
                );
                SampleResult constructed = buildConstructed(
                    dynTex,
                    dynImages,
                    layer,
                    slot,
                    uv,
                    pixel
                );
                return assigned.sampled + assigned.loaded +
                    constructed.sampled + constructed.loaded;
            }

            compute {
                void main() {
                    vec2 uv = vec2(0.5, 0.25);
                    ivec2 pixel = ivec2(0, 0);
                    vec4 value = consumeResults(textureGrid, imageGrid, 1, 2, uv, pixel);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        struct_id = spirv_named_id(spv_code, "SampleResult")
        struct_type = re.search(
            rf"{re.escape(struct_id)} = OpTypeStruct %\d+ %\d+ %\d+ %\d+",
            spv_code,
        )
        assert struct_type is not None
        assert re.search(
            rf"(%\d+) = OpCompositeConstruct "
            rf"{re.escape(struct_id)} %\d+ %\d+ %\d+ %\d+",
            spv_code,
        )
        assert not re.search(r"OpExtInst %\d+ %\d+ SampleResult", spv_code)
        assert spv_code.count("OpStore") >= 4
        assert spv_code.count("OpAccessChain") >= 6
        assert spv_code.count("OpLoad") >= 2
        assert "OpReturnValue" in spv_code
        assert "OpImageSampleExplicitLod" in spv_code
        assert "OpImageRead" in spv_code
        assert "OpImageQuerySizeLod" in spv_code
        assert "OpImageQuerySize" in spv_code
        assert "OpFunctionCall" in spv_code
        assert "texture(" not in spv_code
        assert "textureSize(" not in spv_code
        assert "imageSize(" not in spv_code
        assert "imageLoad(" not in spv_code
        assert "WARNING" not in spv_code

    def test_nested_struct_resource_assignments_store_spirv_member_pointers(self):
        source_code = """
        struct SampleResult {
            vec4 sampled;
            vec4 loaded;
            ivec2 texSize;
            ivec2 imageSizeValue;
        };

        struct SampleEnvelope {
            SampleResult result;
            vec4 bias;
        };

        shader Resources {
            sampler2d textureGrid[2][3];
            image2D imageGrid @rgba16f[2][3];

            SampleEnvelope buildNestedAssigned(
                sampler2d dynTex[][3],
                image2D dynImages[][3] @rgba16f,
                int layer,
                int slot,
                vec2 uv,
                ivec2 pixel
            ) {
                SampleEnvelope envelope;
                envelope.result.sampled = texture(dynTex[layer][slot], uv);
                envelope.result.loaded = imageLoad(dynImages[layer][slot], pixel);
                envelope.result.texSize = textureSize(dynTex[layer][slot], 0);
                envelope.result.imageSizeValue = imageSize(dynImages[layer][slot]);
                envelope.bias = texture(dynTex[layer][slot], uv) +
                    imageLoad(dynImages[layer][slot], pixel);
                return envelope;
            }

            vec4 consumeNested(
                sampler2d dynTex[][3],
                image2D dynImages[][3] @rgba16f,
                int layer,
                int slot,
                vec2 uv,
                ivec2 pixel
            ) {
                SampleEnvelope envelope = buildNestedAssigned(
                    dynTex,
                    dynImages,
                    layer,
                    slot,
                    uv,
                    pixel
                );
                return envelope.result.sampled + envelope.result.loaded + envelope.bias;
            }

            compute {
                void main() {
                    vec2 uv = vec2(0.5, 0.25);
                    ivec2 pixel = ivec2(0, 0);
                    vec4 value = consumeNested(textureGrid, imageGrid, 1, 2, uv, pixel);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert "OpName" in spv_code and '"SampleEnvelope"' in spv_code

        def resource_result_is_stored_through_nested_access(opcode):
            lines = spv_code.splitlines()
            for index, line in enumerate(lines):
                match = re.match(rf"(%\d+) = {opcode}\b", line)
                if not match:
                    continue

                result_id = match.group(1)
                window = lines[index + 1 : index + 8]
                if sum("OpAccessChain" in item for item in window) >= 2 and any(
                    item.startswith("OpStore ") and item.endswith(f" {result_id}")
                    for item in window
                ):
                    return True
            return False

        for opcode in [
            "OpImageSampleExplicitLod",
            "OpImageRead",
            "OpImageQuerySizeLod",
            "OpImageQuerySize",
        ]:
            assert resource_result_is_stored_through_nested_access(opcode)

        assert re.search(
            r"(?:%\d+ = OpAccessChain [^\n]+\n){2}%\d+ = OpLoad %\d+ %\d+",
            spv_code,
        )
        assert "OpFunctionCall" in spv_code
        assert "OpReturnValue" in spv_code
        assert "texture(" not in spv_code
        assert "textureSize(" not in spv_code
        assert "imageSize(" not in spv_code
        assert "imageLoad(" not in spv_code
        assert "WARNING" not in spv_code

    def test_struct_member_array_resource_assignments_use_spirv_pointer_chain(self):
        source_code = """
        struct SampleResult {
            vec4 sampled;
            vec4 loaded;
            ivec2 texSize;
            ivec2 imageSizeValue;
        };

        struct SampleEnvelope {
            SampleResult results[3];
            vec4 bias;
        };

        shader Resources {
            sampler2d textureGrid[2][3];
            image2D imageGrid @rgba16f[2][3];

            SampleEnvelope buildArrayEnvelope(
                sampler2d dynTex[][3],
                image2D dynImages[][3] @rgba16f,
                int layer,
                int slot,
                vec2 uv,
                ivec2 pixel
            ) {
                SampleEnvelope envelope;
                envelope.results[slot].sampled = texture(dynTex[layer][slot], uv);
                envelope.results[slot].loaded = imageLoad(dynImages[layer][slot], pixel);
                envelope.results[slot].texSize = textureSize(dynTex[layer][slot], 0);
                envelope.results[slot].imageSizeValue = imageSize(dynImages[layer][slot]);
                envelope.bias = envelope.results[slot].sampled +
                    envelope.results[slot].loaded;
                return envelope;
            }

            vec4 consumeArrayEnvelope(
                sampler2d dynTex[][3],
                image2D dynImages[][3] @rgba16f,
                int layer,
                int slot,
                vec2 uv,
                ivec2 pixel
            ) {
                SampleEnvelope envelope = buildArrayEnvelope(
                    dynTex,
                    dynImages,
                    layer,
                    slot,
                    uv,
                    pixel
                );
                return envelope.results[slot].sampled +
                    envelope.results[slot].loaded + envelope.bias;
            }

            compute {
                void main() {
                    vec2 uv = vec2(0.5, 0.25);
                    ivec2 pixel = ivec2(0, 0);
                    vec4 value = consumeArrayEnvelope(
                        textureGrid,
                        imageGrid,
                        1,
                        2,
                        uv,
                        pixel
                    );
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert "OpTypeArray" in spv_code
        assert "OpName" in spv_code and '"SampleEnvelope"' in spv_code

        def resource_result_is_stored_through_member_array(opcode):
            lines = spv_code.splitlines()
            for index, line in enumerate(lines):
                match = re.match(rf"(%\d+) = {opcode}\b", line)
                if not match:
                    continue

                result_id = match.group(1)
                window = lines[index + 1 : index + 12]
                if sum("OpAccessChain" in item for item in window) >= 3 and any(
                    item.startswith("OpStore ") and item.endswith(f" {result_id}")
                    for item in window
                ):
                    return True
            return False

        for opcode in [
            "OpImageSampleExplicitLod",
            "OpImageRead",
            "OpImageQuerySizeLod",
            "OpImageQuerySize",
        ]:
            assert resource_result_is_stored_through_member_array(opcode)

        assert re.search(
            r"(?:%\d+ = OpAccessChain [^\n]+\n){3}%\d+ = OpLoad %\d+ %\d+",
            spv_code,
        )
        assert "OpFunctionCall" in spv_code
        assert "OpReturnValue" in spv_code
        assert "texture(" not in spv_code
        assert "textureSize(" not in spv_code
        assert "imageSize(" not in spv_code
        assert "imageLoad(" not in spv_code
        assert "WARNING" not in spv_code

    def test_control_flow_struct_resource_assignments_use_spirv_merges(self):
        source_code = """
        struct SampleResult {
            vec4 sampled;
            vec4 loaded;
            ivec2 texSize;
            ivec2 imageSizeValue;
        };

        struct SampleEnvelope {
            SampleResult result;
            vec4 accum;
            int chosen;
        };

        shader Resources {
            sampler2d textureGrid[2][3];
            image2D imageGrid @rgba16f[2][3];

            SampleEnvelope fillControlled(
                sampler2d dynTex[][3],
                image2D dynImages[][3] @rgba16f,
                int layer,
                int slot,
                bool useAlternate,
                vec2 uv,
                ivec2 pixel
            ) {
                SampleEnvelope envelope;
                envelope.accum = vec4(0.0);
                envelope.chosen = slot;

                if (useAlternate) {
                    envelope.result.sampled = texture(dynTex[layer][slot], uv);
                    envelope.result.loaded = imageLoad(dynImages[layer][slot], pixel);
                } else {
                    int fallback = 0;
                    envelope.result.sampled = texture(dynTex[fallback][slot], uv);
                    envelope.result.loaded = imageLoad(dynImages[fallback][slot], pixel);
                }

                for (int i = 0; i < 3; i++) {
                    if (i == 1) {
                        continue;
                    }
                    envelope.result.texSize = textureSize(dynTex[layer][i], 0);
                    envelope.result.imageSizeValue = imageSize(dynImages[layer][i]);
                    envelope.accum = envelope.accum + texture(dynTex[layer][i], uv);
                    if (i == slot) {
                        break;
                    }
                }

                return envelope;
            }

            vec4 consumeControlled(
                sampler2d dynTex[][3],
                image2D dynImages[][3] @rgba16f,
                int layer,
                int slot,
                vec2 uv,
                ivec2 pixel
            ) {
                SampleEnvelope envelope = fillControlled(
                    dynTex,
                    dynImages,
                    layer,
                    slot,
                    true,
                    uv,
                    pixel
                );
                return envelope.result.sampled + envelope.result.loaded +
                    envelope.accum;
            }

            compute {
                void main() {
                    vec2 uv = vec2(0.5, 0.25);
                    ivec2 pixel = ivec2(0, 0);
                    vec4 value = consumeControlled(
                        textureGrid,
                        imageGrid,
                        1,
                        2,
                        uv,
                        pixel
                    );
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert "OpSelectionMerge" in spv_code
        loop_merges = re.findall(r"OpLoopMerge (%\d+) (%\d+) None", spv_code)
        assert loop_merges
        loop_merge, loop_continue = loop_merges[0]
        assert f"OpBranch {loop_continue}" in spv_code
        assert f"OpBranch {loop_merge}" in spv_code

        def resource_result_is_stored_through_nested_access(opcode):
            lines = spv_code.splitlines()
            for index, line in enumerate(lines):
                match = re.match(rf"(%\d+) = {opcode}\b", line)
                if not match:
                    continue

                result_id = match.group(1)
                window = lines[index + 1 : index + 10]
                if sum("OpAccessChain" in item for item in window) >= 2 and any(
                    item.startswith("OpStore ") and item.endswith(f" {result_id}")
                    for item in window
                ):
                    return True
            return False

        for opcode in [
            "OpImageSampleExplicitLod",
            "OpImageRead",
            "OpImageQuerySizeLod",
            "OpImageQuerySize",
        ]:
            assert resource_result_is_stored_through_nested_access(opcode)

        assert spv_code.count("OpImageSampleExplicitLod") >= 3
        assert spv_code.count("OpImageRead") >= 2
        assert "OpFunctionCall" in spv_code
        assert "OpReturnValue" in spv_code
        assert "texture(" not in spv_code
        assert "textureSize(" not in spv_code
        assert "imageSize(" not in spv_code
        assert "imageLoad(" not in spv_code
        assert "WARNING" not in spv_code

    def test_struct_parameter_resource_values_mutate_through_spirv_local_copy(self):
        source_code = """
        struct SampleResult {
            vec4 sampled;
            vec4 loaded;
            ivec2 texSize;
            ivec2 imageSizeValue;
        };

        shader Resources {
            sampler2d textureGrid[2][3];
            image2D imageGrid @rgba16f[2][3];

            SampleResult buildResourceResult(
                sampler2d dynTex[][3],
                image2D dynImages[][3] @rgba16f,
                int layer,
                int slot,
                vec2 uv,
                ivec2 pixel
            ) {
                SampleResult result;
                result.sampled = texture(dynTex[layer][slot], uv);
                result.loaded = imageLoad(dynImages[layer][slot], pixel);
                result.texSize = textureSize(dynTex[layer][slot], 0);
                result.imageSizeValue = imageSize(dynImages[layer][slot]);
                return result;
            }

            SampleResult adjustResult(SampleResult payload, float weight) {
                payload.sampled = payload.sampled * weight;
                payload.loaded = payload.loaded + payload.sampled;
                return payload;
            }

            vec4 consumeAdjusted(
                sampler2d dynTex[][3],
                image2D dynImages[][3] @rgba16f,
                int layer,
                int slot,
                vec2 uv,
                ivec2 pixel
            ) {
                SampleResult raw = buildResourceResult(
                    dynTex,
                    dynImages,
                    layer,
                    slot,
                    uv,
                    pixel
                );
                SampleResult adjusted = adjustResult(raw, 0.5);
                return adjusted.sampled + adjusted.loaded;
            }

            compute {
                void main() {
                    vec2 uv = vec2(0.5, 0.25);
                    ivec2 pixel = ivec2(0, 0);
                    vec4 value = consumeAdjusted(textureGrid, imageGrid, 1, 2, uv, pixel);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        payload_param = spirv_named_parameter(spv_code, "payload")
        mutable_payload = spirv_named_variable(
            spv_code, "payload", storage_class="Function"
        )
        assert re.search(
            rf"OpStore {re.escape(mutable_payload)} {re.escape(payload_param)}",
            spv_code,
        )

        scaled_sample = re.search(r"(%\d+) = OpFMul %\d+ %\d+ %\d+", spv_code)
        assert scaled_sample is not None
        assert re.search(
            rf"OpStore %\d+ {re.escape(scaled_sample.group(1))}",
            spv_code,
        )
        adjusted_load = re.search(r"(%\d+) = OpFAdd %\d+ %\d+ %\d+", spv_code)
        assert adjusted_load is not None
        assert re.search(
            rf"OpStore %\d+ {re.escape(adjusted_load.group(1))}",
            spv_code,
        )
        assert re.search(
            rf"(%\d+) = OpLoad %\d+ {re.escape(mutable_payload)}\n"
            rf"OpReturnValue \1",
            spv_code,
        )
        assert "OpImageSampleExplicitLod" in spv_code
        assert "OpImageRead" in spv_code
        assert "OpImageQuerySizeLod" in spv_code
        assert "OpImageQuerySize" in spv_code
        assert spv_code.count("OpFunctionCall") >= 3
        assert "texture(" not in spv_code
        assert "textureSize(" not in spv_code
        assert "imageSize(" not in spv_code
        assert "imageLoad(" not in spv_code
        assert "WARNING" not in spv_code

    def test_mutable_scalar_and_array_params_use_spirv_local_copies(self):
        source_code = """
        shader MutableParams {
            float bumpScalar(float weight) {
                weight = weight + 1.0;
                return weight * 2.0;
            }

            float bumpArray(float values[3], int slot) {
                values[slot] = values[slot] + 1.0;
                values[0] = values[slot] * 0.5;
                return values[slot] + values[0];
            }

            compute {
                void main() {
                    float values[3];
                    values[0] = 1.0;
                    values[1] = 2.0;
                    values[2] = 3.0;
                    float a = bumpScalar(0.5);
                    float b = bumpArray(values, 1);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        weight_param = spirv_named_parameter(spv_code, "weight")
        mutable_weight = spirv_named_variable(
            spv_code, "weight", storage_class="Function"
        )
        assert re.search(
            rf"OpStore {re.escape(mutable_weight)} {re.escape(weight_param)}",
            spv_code,
        )
        assert re.search(
            rf"OpStore {re.escape(mutable_weight)} %\d+",
            spv_code,
        )
        assert re.search(
            rf"%\d+ = OpLoad %\d+ {re.escape(mutable_weight)}",
            spv_code,
        )

        values_param = spirv_named_parameter(spv_code, "values")
        mutable_values = spirv_named_variable(
            spv_code, "values", storage_class="Function"
        )
        assert re.search(
            rf"OpStore {re.escape(mutable_values)} {re.escape(values_param)}",
            spv_code,
        )
        assert not re.search(
            rf"OpAccessChain %\d+ {re.escape(values_param)} ",
            spv_code,
        )
        assert re.search(
            rf"OpAccessChain %\d+ {re.escape(mutable_values)} %\d+",
            spv_code,
        )
        assert "OpFAdd" in spv_code
        assert "OpFMul" in spv_code
        assert spv_code.count("OpFunctionCall") >= 2
        assert "WARNING" not in spv_code

    def test_array_literals_emit_spirv_composite_constructs(self):
        source_code = """
        float globalWeights[4] = {1.0, 2.0};

        float pickScalar(int index) {
            float values[4] = {1.0, 2.0, 3.0, 4.0};
            return values[index];
        }

        float[4] makeValues() {
            return {1.0, 2.0};
        }

        vec3 pickColor(int index) {
            vec3 colors[2] = {
                vec3(1.0, 2.0, 3.0),
                vec3(4.0, 5.0, 6.0)
            };
            return colors[index];
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        global_var = spirv_named_variable(
            spv_code, "globalWeights", storage_class="Private"
        )
        assert re.search(
            rf"{re.escape(global_var)} = OpVariable %\d+ Private %\d+",
            spv_code,
        )
        assert re.search(
            r"%\d+ = OpConstantComposite %\d+ %\d+ %\d+ %\d+ %\d+",
            spv_code,
        )
        assert re.search(
            r"%\d+ = OpCompositeConstruct %\d+ %\d+ %\d+ %\d+ %\d+",
            spv_code,
        )
        assert spv_code.count("OpCompositeConstruct") >= 4
        assert "ArrayLiteralNode" not in spv_code
        assert "WARNING" not in spv_code

    def test_nested_array_and_struct_member_params_use_spirv_local_copies(self):
        source_code = """
        struct Payload {
            float values[3];
            float bias;
        };

        shader NestedMutableParams {
            float bumpNested(float values[2][3], int row, int col) {
                values[row][col] = values[row][col] + 1.0;
                values[0][col] = values[row][col] * 0.5;
                return values[row][col] + values[0][col];
            }

            float bumpPayload(Payload payload, int slot) {
                payload.values[slot] = payload.values[slot] + payload.bias;
                payload.bias = payload.values[slot] * 0.5;
                return payload.values[slot] + payload.bias;
            }

            compute {
                void main() {
                    float grid[2][3];
                    grid[0][0] = 1.0;
                    grid[1][2] = 3.0;
                    Payload payload;
                    payload.values[0] = 2.0;
                    payload.values[1] = 4.0;
                    payload.bias = 0.25;
                    float a = bumpNested(grid, 1, 2);
                    float b = bumpPayload(payload, 1);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        values_param = spirv_named_parameter(spv_code, "values")
        mutable_values = spirv_named_variable(
            spv_code, "values", storage_class="Function"
        )
        assert re.search(
            rf"OpStore {re.escape(mutable_values)} {re.escape(values_param)}",
            spv_code,
        )
        assert not re.search(
            rf"OpAccessChain %\d+ {re.escape(values_param)} ",
            spv_code,
        )
        values_rows = re.findall(
            rf"(%\d+) = OpAccessChain %\d+ " rf"{re.escape(mutable_values)} %\d+",
            spv_code,
        )
        assert values_rows
        assert any(
            re.search(rf"OpAccessChain %\d+ {re.escape(row)} %\d+", spv_code)
            for row in values_rows
        )

        payload_param = spirv_named_parameter(spv_code, "payload")
        mutable_payload = spirv_named_variable(
            spv_code, "payload", storage_class="Function"
        )
        assert re.search(
            rf"OpStore {re.escape(mutable_payload)} {re.escape(payload_param)}",
            spv_code,
        )
        assert not re.search(
            rf"OpAccessChain %\d+ {re.escape(payload_param)} ",
            spv_code,
        )
        payload_members = re.findall(
            rf"(%\d+) = OpAccessChain %\d+ " rf"{re.escape(mutable_payload)} %\d+",
            spv_code,
        )
        assert payload_members
        assert any(
            re.search(rf"OpAccessChain %\d+ {re.escape(member)} %\d+", spv_code)
            for member in payload_members
        )
        assert "OpFAdd" in spv_code
        assert "OpFMul" in spv_code
        assert spv_code.count("OpFunctionCall") >= 2
        assert "WARNING" not in spv_code

    def test_returned_nested_struct_params_use_spirv_local_copies(self):
        source_code = """
        struct InnerPayload {
            float values[3];
            float bias;
        };

        struct OuterPayload {
            InnerPayload inner;
            float scale;
        };

        shader ReturnedParamPayloads {
            OuterPayload makeOuter(float first, float second, float bias, float scale) {
                OuterPayload outer;
                outer.inner.values[0] = first;
                outer.inner.values[1] = second;
                outer.inner.values[2] = first + second;
                outer.inner.bias = bias;
                outer.scale = scale;
                return outer;
            }

            OuterPayload adjustOuter(OuterPayload payload, int slot) {
                payload.inner.values[slot] = payload.inner.values[slot] + payload.inner.bias;
                payload.inner.values[0] = payload.inner.values[slot] * payload.scale;
                payload.scale = payload.inner.values[0] + payload.inner.bias;
                return payload;
            }

            float consumeAdjustedOuter(int slot) {
                OuterPayload adjusted = adjustOuter(makeOuter(1.0, 2.0, 0.25, 4.0), slot);
                return adjusted.inner.values[slot] + adjusted.inner.values[0] + adjusted.scale;
            }

            compute {
                void main() {
                    float value = consumeAdjustedOuter(1);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        payload_param = spirv_named_parameter(spv_code, "payload")
        mutable_payload = spirv_named_variable(
            spv_code, "payload", storage_class="Function"
        )
        assert re.search(
            rf"OpStore {re.escape(mutable_payload)} {re.escape(payload_param)}",
            spv_code,
        )
        assert not re.search(
            rf"OpAccessChain %\d+ {re.escape(payload_param)} ",
            spv_code,
        )

        payload_members = re.findall(
            rf"(%\d+) = OpAccessChain %\d+ " rf"{re.escape(mutable_payload)} %\d+",
            spv_code,
        )
        payload_nested_members = [
            nested_access
            for member_access in payload_members
            for nested_access in re.findall(
                rf"(%\d+) = OpAccessChain %\d+ {re.escape(member_access)} %\d+",
                spv_code,
            )
        ]
        assert payload_nested_members
        assert any(
            re.search(
                rf"%\d+ = OpAccessChain %\d+ {re.escape(nested_access)} %\d+",
                spv_code,
            )
            for nested_access in payload_nested_members
        )
        assert re.search(
            rf"(%\d+) = OpLoad %\d+ {re.escape(mutable_payload)}\n"
            rf"OpReturnValue \1",
            spv_code,
        )
        assert re.search(
            r"(%\d+) = OpFunctionCall %\d+ %\d+ %\d+ %\d+ %\d+ %\d+\n"
            r"%\d+ = OpFunctionCall %\d+ %\d+ \1 %\d+",
            spv_code,
        )

        adjusted = spirv_named_variable(spv_code, "adjusted", storage_class="Function")
        assert re.search(rf"OpStore {re.escape(adjusted)} %\d+", spv_code)
        assert re.search(
            rf"%\d+ = OpAccessChain %\d+ {re.escape(adjusted)} %\d+",
            spv_code,
        )
        assert "OpFAdd" in spv_code
        assert "OpFMul" in spv_code
        assert spv_code.count("OpFunctionCall") >= 3
        assert "WARNING" not in spv_code

    def test_conditional_returned_nested_structs_use_spirv_composite_selection(
        self,
    ):
        source_code = """
        struct InnerPayload {
            float values[3];
            float bias;
        };

        struct OuterPayload {
            InnerPayload inner;
            float scale;
        };

        shader ConditionalReturnedPayloads {
            OuterPayload makeOuter(float first, float second, float bias, float scale) {
                OuterPayload outer;
                outer.inner.values[0] = first;
                outer.inner.values[1] = second;
                outer.inner.values[2] = first + second;
                outer.inner.bias = bias;
                outer.scale = scale;
                return outer;
            }

            OuterPayload chooseBranch(bool useSecond) {
                if (useSecond) {
                    return makeOuter(3.0, 4.0, 0.5, 5.0);
                }
                return makeOuter(1.0, 2.0, 0.25, 4.0);
            }

            OuterPayload chooseTernary(bool useSecond) {
                return useSecond
                    ? makeOuter(3.0, 4.0, 0.5, 5.0)
                    : makeOuter(1.0, 2.0, 0.25, 4.0);
            }

            float consumeConditional(int slot, bool useSecond) {
                OuterPayload branchPayload = chooseBranch(useSecond);
                OuterPayload ternaryPayload = chooseTernary(useSecond);
                ternaryPayload.inner.values[slot] =
                    ternaryPayload.inner.values[slot] + ternaryPayload.inner.bias;
                branchPayload.scale =
                    branchPayload.inner.values[slot] + ternaryPayload.scale;
                return branchPayload.scale + ternaryPayload.inner.values[slot];
            }

            compute {
                void main() {
                    float value = consumeConditional(1, true);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        assert not re.search(r"=\s+OpSelect\b", spv_code)
        assert "OpConstantTrue" in spv_code
        assert not re.search(r"OpConstant %\d+ true", spv_code)

        def has_composite_selection_flow():
            lines = spv_code.splitlines()
            for index in range(len(lines) - 10):
                merge = re.match(r"OpSelectionMerge (%\d+) None", lines[index])
                branch = re.match(
                    r"OpBranchConditional %\d+ (%\d+) (%\d+)",
                    lines[index + 1],
                )
                then_store = re.match(r"OpStore (%\d+) (%\d+)", lines[index + 3])
                else_store = re.match(r"OpStore (%\d+) (%\d+)", lines[index + 6])
                load = re.match(
                    r"(%\d+) = OpLoad %\d+ (%\d+)",
                    lines[index + 9],
                )
                if not all([merge, branch, then_store, else_store, load]):
                    continue

                merge_label = merge.group(1)
                then_label = branch.group(1)
                else_label = branch.group(2)
                selected_value = load.group(1)
                return (
                    lines[index + 2] == f"{then_label} = OpLabel"
                    and lines[index + 4] == f"OpBranch {merge_label}"
                    and lines[index + 5] == f"{else_label} = OpLabel"
                    and lines[index + 7] == f"OpBranch {merge_label}"
                    and lines[index + 8] == f"{merge_label} = OpLabel"
                    and then_store.group(1) == else_store.group(1)
                    and load.group(2) == then_store.group(1)
                    and lines[index + 10] == f"OpReturnValue {selected_value}"
                )
            return False

        assert has_composite_selection_flow()

        branch_payload = spirv_named_variable(
            spv_code, "branchPayload", storage_class="Function"
        )
        ternary_payload = spirv_named_variable(
            spv_code, "ternaryPayload", storage_class="Function"
        )
        assert re.search(rf"OpStore {re.escape(branch_payload)} %\d+", spv_code)
        assert re.search(rf"OpStore {re.escape(ternary_payload)} %\d+", spv_code)
        assert re.search(
            rf"%\d+ = OpAccessChain %\d+ " rf"{re.escape(ternary_payload)} %\d+",
            spv_code,
        )
        assert re.search(
            rf"%\d+ = OpAccessChain %\d+ " rf"{re.escape(branch_payload)} %\d+",
            spv_code,
        )
        assert "OpFAdd" in spv_code
        assert spv_code.count("OpFunctionCall") >= 6
        assert "WARNING" not in spv_code

    def test_temporary_struct_array_member_reads_use_spirv_local_array_copies(self):
        source_code = """
        struct Payload {
            float values[3];
            float bias;
        };

        shader TemporaryPayloads {
            Payload makePayload(float first, float second, float bias) {
                Payload payload;
                payload.values[0] = first;
                payload.values[1] = second;
                payload.values[2] = first + second;
                payload.bias = bias;
                return payload;
            }

            float readTemporary(int slot) {
                float dynamicValue = makePayload(1.0, 2.0, 0.25).values[slot];
                float fixedValue = makePayload(3.0, 4.0, 0.5).values[0];
                float biasValue = makePayload(5.0, 6.0, 0.75).bias;
                return dynamicValue + fixedValue + biasValue;
            }

            compute {
                void main() {
                    float value = readTemporary(1);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        array_extracts = re.findall(
            r"(%\d+) = OpCompositeExtract %\d+ %\d+ 0",
            spv_code,
        )
        assert len(array_extracts) >= 2
        for extract_id in array_extracts:
            assert not re.search(
                rf"OpAccessChain %\d+ {re.escape(extract_id)} ", spv_code
            )
            copy_store = re.search(rf"OpStore (%\d+) {re.escape(extract_id)}", spv_code)
            assert copy_store is not None
            copy_var = copy_store.group(1)
            assert re.search(
                rf"{re.escape(copy_var)} = OpVariable %\d+ Function", spv_code
            )
            assert re.search(
                rf"%\d+ = OpAccessChain %\d+ {re.escape(copy_var)} %\d+",
                spv_code,
            )

        assert re.search(r"%\d+ = OpCompositeExtract %\d+ %\d+ 1", spv_code)
        assert "OpFunctionCall" in spv_code
        assert "OpReturnValue" in spv_code
        assert "OpFAdd" in spv_code
        assert "WARNING" not in spv_code

    def test_nested_temporary_struct_array_member_reads_use_spirv_local_array_copies(
        self,
    ):
        source_code = """
        struct InnerPayload {
            float values[3];
            float bias;
        };

        struct OuterPayload {
            InnerPayload inner;
            float scale;
        };

        shader NestedTemporaryPayloads {
            OuterPayload makeOuter(float first, float second, float bias, float scale) {
                OuterPayload outer;
                outer.inner.values[0] = first;
                outer.inner.values[1] = second;
                outer.inner.values[2] = first + second;
                outer.inner.bias = bias;
                outer.scale = scale;
                return outer;
            }

            float readNestedTemporary(int slot) {
                float dynamicValue = makeOuter(1.0, 2.0, 0.25, 4.0).inner.values[slot];
                float fixedValue = makeOuter(3.0, 4.0, 0.5, 5.0).inner.values[0];
                float biasValue = makeOuter(5.0, 6.0, 0.75, 6.0).inner.bias;
                float scaleValue = makeOuter(7.0, 8.0, 1.0, 9.0).scale;
                return dynamicValue + fixedValue + biasValue + scaleValue;
            }

            compute {
                void main() {
                    float value = readNestedTemporary(1);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        nested_array_extracts = re.findall(
            r"(%\d+) = OpCompositeExtract %\d+ %\d+ 0\n"
            r"(%\d+) = OpCompositeExtract %\d+ \1 0",
            spv_code,
        )
        assert len(nested_array_extracts) >= 2
        for _, array_extract in nested_array_extracts:
            assert not re.search(
                rf"OpAccessChain %\d+ {re.escape(array_extract)} ",
                spv_code,
            )
            copy_store = re.search(
                rf"OpStore (%\d+) {re.escape(array_extract)}", spv_code
            )
            assert copy_store is not None
            copy_var = copy_store.group(1)
            assert re.search(
                rf"{re.escape(copy_var)} = OpVariable %\d+ Function", spv_code
            )
            assert re.search(
                rf"%\d+ = OpAccessChain %\d+ {re.escape(copy_var)} %\d+",
                spv_code,
            )

        assert re.search(
            r"%\d+ = OpCompositeExtract %\d+ %\d+ 0\n"
            r"%\d+ = OpCompositeExtract %\d+ %\d+ 1",
            spv_code,
        )
        assert re.search(r"%\d+ = OpCompositeExtract %\d+ %\d+ 1", spv_code)
        assert "OpFunctionCall" in spv_code
        assert "OpReturnValue" in spv_code
        assert spv_code.count("OpFAdd") >= 3
        assert "WARNING" not in spv_code

    def test_returned_local_nested_struct_array_writes_use_spirv_local_access_chains(
        self,
    ):
        source_code = """
        struct InnerPayload {
            float values[3];
            float bias;
        };

        struct OuterPayload {
            InnerPayload inner;
            float scale;
        };

        shader LocalReturnedPayloads {
            OuterPayload makeOuter(float first, float second, float bias, float scale) {
                OuterPayload outer;
                outer.inner.values[0] = first;
                outer.inner.values[1] = second;
                outer.inner.values[2] = first + second;
                outer.inner.bias = bias;
                outer.scale = scale;
                return outer;
            }

            float mutateReturnedLocal(int slot) {
                OuterPayload outer = makeOuter(1.0, 2.0, 0.25, 4.0);
                outer.inner.values[slot] = outer.inner.values[slot] + outer.inner.bias;
                outer.inner.values[0] = outer.inner.values[slot] * outer.scale;
                outer.scale = outer.inner.values[0] + outer.inner.bias;
                return outer.inner.values[slot] + outer.inner.values[0] + outer.scale;
            }

            compute {
                void main() {
                    float value = mutateReturnedLocal(1);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        outer_ids = [
            outer_id
            for outer_id in spirv_named_ids(spv_code, "outer")
            if re.search(rf"{re.escape(outer_id)} = OpVariable %\d+ Function", spv_code)
        ]
        assert outer_ids
        outer_id = next(
            (
                candidate
                for candidate in outer_ids
                if re.search(
                    rf"(%\d+) = OpFunctionCall %\d+ %\d+ %\d+ %\d+ %\d+ %\d+\n"
                    rf"OpStore {re.escape(candidate)} \1",
                    spv_code,
                )
            ),
            None,
        )
        assert outer_id is not None

        outer_member_accesses = re.findall(
            rf"(%\d+) = OpAccessChain %\d+ {re.escape(outer_id)} %\d+",
            spv_code,
        )
        assert outer_member_accesses
        nested_member_accesses = [
            nested_access
            for member_access in outer_member_accesses
            for nested_access in re.findall(
                rf"(%\d+) = OpAccessChain %\d+ {re.escape(member_access)} %\d+",
                spv_code,
            )
        ]
        assert nested_member_accesses
        assert any(
            re.search(
                rf"%\d+ = OpAccessChain %\d+ {re.escape(nested_access)} %\d+",
                spv_code,
            )
            for nested_access in nested_member_accesses
        )

        assert spv_code.count("OpStore") >= 9
        assert spv_code.count("OpLoad") >= 8
        assert "OpFAdd" in spv_code
        assert "OpFMul" in spv_code
        assert "OpReturnValue" in spv_code
        assert "WARNING" not in spv_code

    def test_returned_local_nested_struct_array_writes_in_control_flow_use_spirv_access_chains(
        self,
    ):
        source_code = """
        struct InnerPayload {
            float values[3];
            float bias;
        };

        struct OuterPayload {
            InnerPayload inner;
            float scale;
        };

        shader ControlFlowReturnedPayloads {
            OuterPayload makeOuter(float first, float second, float bias, float scale) {
                OuterPayload outer;
                outer.inner.values[0] = first;
                outer.inner.values[1] = second;
                outer.inner.values[2] = first + second;
                outer.inner.bias = bias;
                outer.scale = scale;
                return outer;
            }

            float mutateReturnedLocalControl(int slot) {
                OuterPayload outer = makeOuter(1.0, 2.0, 0.25, 4.0);
                for (int i = 0; i < 3; i++) {
                    outer.inner.values[i] = outer.inner.values[i] + outer.inner.bias;
                    if (i == slot) {
                        outer.inner.values[i] = outer.inner.values[i] * outer.scale;
                    }
                }
                if (slot > 1) {
                    outer.scale = outer.inner.values[slot] + outer.inner.bias;
                } else {
                    outer.scale = outer.inner.values[0] - outer.inner.bias;
                }
                return outer.inner.values[slot] + outer.scale;
            }

            compute {
                void main() {
                    float value = mutateReturnedLocalControl(1);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        outer_ids = [
            outer_id
            for outer_id in spirv_named_ids(spv_code, "outer")
            if re.search(rf"{re.escape(outer_id)} = OpVariable %\d+ Function", spv_code)
        ]
        assert outer_ids
        outer_id = next(
            (
                candidate
                for candidate in outer_ids
                if re.search(
                    rf"(%\d+) = OpFunctionCall %\d+ %\d+ %\d+ %\d+ %\d+ %\d+\n"
                    rf"OpStore {re.escape(candidate)} \1",
                    spv_code,
                )
            ),
            None,
        )
        assert outer_id is not None

        outer_member_accesses = re.findall(
            rf"(%\d+) = OpAccessChain %\d+ {re.escape(outer_id)} %\d+",
            spv_code,
        )
        nested_member_accesses = [
            nested_access
            for member_access in outer_member_accesses
            for nested_access in re.findall(
                rf"(%\d+) = OpAccessChain %\d+ {re.escape(member_access)} %\d+",
                spv_code,
            )
        ]
        nested_array_accesses = [
            array_access
            for nested_access in nested_member_accesses
            for array_access in re.findall(
                rf"(%\d+) = OpAccessChain %\d+ {re.escape(nested_access)} %\d+",
                spv_code,
            )
        ]

        assert len(nested_array_accesses) >= 5
        assert any(
            re.search(rf"OpStore {re.escape(access)} %\d+", spv_code)
            for access in nested_array_accesses
        )
        assert any(
            re.search(rf"%\d+ = OpLoad %\d+ {re.escape(access)}", spv_code)
            for access in nested_array_accesses
        )
        assert "OpLoopMerge" in spv_code
        assert "OpSelectionMerge" in spv_code
        assert "OpSLessThan" in spv_code
        assert "OpIEqual" in spv_code
        assert "OpSGreaterThan" in spv_code
        assert "OpFAdd" in spv_code
        assert "OpFSub" in spv_code
        assert "OpFMul" in spv_code
        assert "OpReturnValue" in spv_code
        assert "WARNING" not in spv_code

    def test_nested_resource_array_access_operations_use_spirv_element_pointers(self):
        source_code = """
        shader Resources {
            sampler2d textureGrid[2][3];
            image2D imageGrid @rgba16f[2][4];
            uimage2D counterGrid @r32ui[2][2];

            vec4 sampleGrid(
                sampler2d paramGrid[2][3],
                int layer,
                int slot,
                vec2 uv
            ) {
                return texture(paramGrid[layer][slot], uv);
            }

            vec4 readGrid(
                image2D paramImages[2][4] @rgba16f,
                int layer,
                int slot,
                ivec2 pixel
            ) {
                return imageLoad(paramImages[layer][slot], pixel);
            }

            compute {
                void main() {
                    int layer = 1;
                    int slot = 1;
                    vec2 uv = vec2(0.5, 0.25);
                    ivec2 pixel = ivec2(0, 0);
                    vec4 sampled = texture(textureGrid[layer][slot], uv);
                    vec4 sampledParam = sampleGrid(textureGrid, layer, slot, uv);
                    vec4 color = imageLoad(imageGrid[layer][slot], pixel);
                    vec4 colorParam = readGrid(imageGrid, layer, slot, pixel);
                    imageStore(imageGrid[layer][slot], pixel, color);
                    uint count = imageLoad(counterGrid[layer][slot], pixel);
                    imageStore(counterGrid[layer][slot], pixel, count);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        float_type = re.search(r"(%\d+) = OpTypeFloat 32", spv_code)
        uint_type = re.search(r"(%\d+) = OpTypeInt 32 0", spv_code)
        assert float_type is not None
        assert uint_type is not None

        sampled_image_type = re.search(
            rf"(%\d+) = OpTypeImage {re.escape(float_type.group(1))} "
            r"2D 0 0 0 1 Unknown\n(%\d+) = OpTypeSampledImage \1",
            spv_code,
        )
        rgba_image_type = re.search(
            rf"(%\d+) = OpTypeImage {re.escape(float_type.group(1))} "
            r"2D 0 0 0 2 Rgba16f",
            spv_code,
        )
        r32ui_image_type = re.search(
            rf"(%\d+) = OpTypeImage {re.escape(uint_type.group(1))} "
            r"2D 0 0 0 2 R32ui",
            spv_code,
        )
        assert sampled_image_type is not None
        assert rgba_image_type is not None
        assert r32ui_image_type is not None

        sampled_type = sampled_image_type.group(2)
        rgba_type = rgba_image_type.group(1)
        r32ui_type = r32ui_image_type.group(1)

        def nested_array_types(element_type):
            inner = re.search(
                rf"(%\d+) = OpTypeArray {re.escape(element_type)} %\d+",
                spv_code,
            )
            assert inner is not None
            outer = re.search(
                rf"(%\d+) = OpTypeArray {re.escape(inner.group(1))} %\d+",
                spv_code,
            )
            assert outer is not None
            return inner.group(1), outer.group(1)

        sampled_inner, sampled_outer = nested_array_types(sampled_type)
        rgba_inner, rgba_outer = nested_array_types(rgba_type)
        r32ui_inner, r32ui_outer = nested_array_types(r32ui_type)

        def uniform_pointer(pointed_type):
            pointer = re.search(
                rf"(%\d+) = OpTypePointer UniformConstant "
                rf"{re.escape(pointed_type)}",
                spv_code,
            )
            assert pointer is not None
            return pointer.group(1)

        sampled_outer_pointer = uniform_pointer(sampled_outer)
        sampled_inner_pointer = uniform_pointer(sampled_inner)
        sampled_element_pointer = uniform_pointer(sampled_type)
        rgba_outer_pointer = uniform_pointer(rgba_outer)
        rgba_inner_pointer = uniform_pointer(rgba_inner)
        rgba_element_pointer = uniform_pointer(rgba_type)
        r32ui_outer_pointer = uniform_pointer(r32ui_outer)
        r32ui_inner_pointer = uniform_pointer(r32ui_inner)
        r32ui_element_pointer = uniform_pointer(r32ui_type)

        def named_variable(name, pointer_type):
            return spirv_named_variable(spv_code, name, pointer_type, "UniformConstant")

        texture_grid = named_variable("textureGrid", sampled_outer_pointer)
        image_grid = named_variable("imageGrid", rgba_outer_pointer)
        counter_grid = named_variable("counterGrid", r32ui_outer_pointer)

        param_grid = spirv_named_parameter(spv_code, "paramGrid", sampled_outer_pointer)
        param_images = spirv_named_parameter(
            spv_code, "paramImages", rgba_outer_pointer
        )

        def assert_nested_access(base_id, row_pointer, element_pointer):
            row_access = re.search(
                rf"(%\d+) = OpAccessChain {re.escape(row_pointer)} "
                rf"{re.escape(base_id)} %\d+",
                spv_code,
            )
            assert row_access is not None
            assert re.search(
                rf"OpAccessChain {re.escape(element_pointer)} "
                rf"{row_access.group(1)} %\d+",
                spv_code,
            )

        assert_nested_access(
            texture_grid, sampled_inner_pointer, sampled_element_pointer
        )
        assert_nested_access(param_grid, sampled_inner_pointer, sampled_element_pointer)
        assert_nested_access(image_grid, rgba_inner_pointer, rgba_element_pointer)
        assert_nested_access(param_images, rgba_inner_pointer, rgba_element_pointer)
        assert_nested_access(counter_grid, r32ui_inner_pointer, r32ui_element_pointer)

        assert re.search(
            rf"OpFunctionCall %\d+ %\d+ {re.escape(texture_grid)} %\d+ %\d+ %\d+",
            spv_code,
        )
        assert re.search(
            rf"OpFunctionCall %\d+ %\d+ {re.escape(image_grid)} %\d+ %\d+ %\d+",
            spv_code,
        )
        assert len(re.findall(r"\bOpImageSampleExplicitLod\b", spv_code)) == 2
        assert len(re.findall(r"\bOpImageRead\b", spv_code)) == 3
        assert len(re.findall(r"\bOpImageWrite\b", spv_code)) == 2
        assert f"OpImageRead {uint_type.group(1)}" in spv_code
        for array_type in [
            sampled_inner,
            sampled_outer,
            rgba_inner,
            rgba_outer,
            r32ui_inner,
            r32ui_outer,
        ]:
            assert f"OpLoad {array_type}" not in spv_code
        assert "texture(" not in spv_code
        assert "imageLoad" not in spv_code
        assert "imageStore" not in spv_code
        assert "WARNING" not in spv_code

    def test_nested_resource_array_shadow_gather_and_fetch_emit_spirv_ops(self):
        source_code = """
        shader Resources {
            sampler2d colorGrid[2][3];
            sampler2dms msGrid[2][2];
            sampler2dshadow shadowGrid[2][3];
            sampler cmpSampler;

            float compareGrid(
                sampler2dshadow paramShadow[2][3],
                int layer,
                int slot,
                vec2 uv,
                float depth
            ) {
                return textureCompare(paramShadow[layer][slot], uv, depth);
            }

            compute {
                void main() {
                    int layer = 1;
                    int slot = 1;
                    vec2 uv = vec2(0.5, 0.25);
                    ivec2 pixel = ivec2(0, 0);
                    float depth = 0.5;
                    float directCompare = textureCompare(
                        shadowGrid[layer][slot],
                        uv,
                        depth
                    );
                    float sampledCompare = textureCompare(
                        shadowGrid[layer][slot],
                        cmpSampler,
                        uv,
                        depth
                    );
                    float helperCompare = compareGrid(
                        shadowGrid,
                        layer,
                        slot,
                        uv,
                        depth
                    );
                    vec4 gathered = textureGather(colorGrid[layer][slot], uv, 1);
                    vec4 gatheredShadow = textureGatherCompare(
                        shadowGrid[layer][slot],
                        uv,
                        depth
                    );
                    vec4 fetched = texelFetch(colorGrid[layer][slot], pixel, 0);
                    vec4 fetchedMs = texelFetch(msGrid[layer][slot], pixel, 1);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        float_type = re.search(r"(%\d+) = OpTypeFloat 32", spv_code)
        assert float_type is not None

        sampled_image_type = re.search(
            rf"(%\d+) = OpTypeImage {re.escape(float_type.group(1))} "
            r"2D 0 0 0 1 Unknown\n(%\d+) = OpTypeSampledImage \1",
            spv_code,
        )
        ms_sampled_image_type = re.search(
            rf"(%\d+) = OpTypeImage {re.escape(float_type.group(1))} "
            r"2D 0 0 1 1 Unknown\n(%\d+) = OpTypeSampledImage \1",
            spv_code,
        )
        shadow_sampled_image_type = re.search(
            rf"(%\d+) = OpTypeImage {re.escape(float_type.group(1))} "
            r"2D 1 0 0 1 Unknown\n(%\d+) = OpTypeSampledImage \1",
            spv_code,
        )
        assert sampled_image_type is not None
        assert ms_sampled_image_type is not None
        assert shadow_sampled_image_type is not None

        sampled_type = sampled_image_type.group(2)
        ms_sampled_type = ms_sampled_image_type.group(2)
        shadow_type = shadow_sampled_image_type.group(2)

        def nested_array_types(element_type):
            inner = re.search(
                rf"(%\d+) = OpTypeArray {re.escape(element_type)} %\d+",
                spv_code,
            )
            assert inner is not None
            outer = re.search(
                rf"(%\d+) = OpTypeArray {re.escape(inner.group(1))} %\d+",
                spv_code,
            )
            assert outer is not None
            return inner.group(1), outer.group(1)

        color_inner, color_outer = nested_array_types(sampled_type)
        ms_inner, ms_outer = nested_array_types(ms_sampled_type)
        shadow_inner, shadow_outer = nested_array_types(shadow_type)

        def uniform_pointer(pointed_type):
            pointer = re.search(
                rf"(%\d+) = OpTypePointer UniformConstant "
                rf"{re.escape(pointed_type)}",
                spv_code,
            )
            assert pointer is not None
            return pointer.group(1)

        color_outer_pointer = uniform_pointer(color_outer)
        color_inner_pointer = uniform_pointer(color_inner)
        color_element_pointer = uniform_pointer(sampled_type)
        ms_outer_pointer = uniform_pointer(ms_outer)
        ms_inner_pointer = uniform_pointer(ms_inner)
        ms_element_pointer = uniform_pointer(ms_sampled_type)
        shadow_outer_pointer = uniform_pointer(shadow_outer)
        shadow_inner_pointer = uniform_pointer(shadow_inner)
        shadow_element_pointer = uniform_pointer(shadow_type)

        def named_variable(name, pointer_type):
            return spirv_named_variable(spv_code, name, pointer_type, "UniformConstant")

        color_grid = named_variable("colorGrid", color_outer_pointer)
        ms_grid = named_variable("msGrid", ms_outer_pointer)
        shadow_grid = named_variable("shadowGrid", shadow_outer_pointer)

        param_shadow = spirv_named_parameter(
            spv_code, "paramShadow", shadow_outer_pointer
        )

        def assert_nested_access(base_id, row_pointer, element_pointer):
            row_access = re.search(
                rf"(%\d+) = OpAccessChain {re.escape(row_pointer)} "
                rf"{re.escape(base_id)} %\d+",
                spv_code,
            )
            assert row_access is not None
            assert re.search(
                rf"OpAccessChain {re.escape(element_pointer)} "
                rf"{row_access.group(1)} %\d+",
                spv_code,
            )

        assert_nested_access(color_grid, color_inner_pointer, color_element_pointer)
        assert_nested_access(ms_grid, ms_inner_pointer, ms_element_pointer)
        assert_nested_access(shadow_grid, shadow_inner_pointer, shadow_element_pointer)
        assert_nested_access(param_shadow, shadow_inner_pointer, shadow_element_pointer)

        assert re.search(
            rf"OpFunctionCall %\d+ %\d+ {re.escape(shadow_grid)} "
            r"%\d+ %\d+ %\d+ %\d+",
            spv_code,
        )
        assert len(re.findall(r"\bOpImageSampleDrefExplicitLod\b", spv_code)) == 3
        assert len(re.findall(r"\bOpImageDrefGather\b", spv_code)) == 1
        assert len(re.findall(r"\bOpImageGather\b", spv_code)) == 1
        assert len(re.findall(r"\bOpImageFetch\b", spv_code)) == 2
        for array_type in [
            color_inner,
            color_outer,
            ms_inner,
            ms_outer,
            shadow_inner,
            shadow_outer,
        ]:
            assert f"OpLoad {array_type}" not in spv_code
        assert "textureCompare" not in spv_code
        assert "textureGather" not in spv_code
        assert "textureGatherCompare" not in spv_code
        assert "texelFetch" not in spv_code
        assert "WARNING" not in spv_code

    def test_texture_query_shapes_for_array_cube_and_multisample_samplers(self):
        source_code = """
        shader Resources {
            sampler2darray arrayTex;
            samplercube cubeTex;
            samplercubearray cubeArrayTex;
            sampler2dms msTex;
            sampler2dmsarray msArrayTex;
            sampler samplers[2];

            compute {
                void main() {
                    int layer = 1;
                    vec3 uvLayer = vec3(0.25, 0.75, 1.0);
                    vec3 direction = vec3(1.0, 0.0, 0.0);
                    ivec3 arraySize = textureSize(arrayTex, 1);
                    ivec2 cubeSize = textureSize(cubeTex, 0);
                    ivec3 cubeArraySize = textureSize(cubeArrayTex, 0);
                    ivec2 msSize = textureSize(msTex, 0);
                    ivec3 msArraySize = textureSize(msArrayTex, 0);
                    vec2 arrayLod = textureQueryLod(
                        arrayTex,
                        samplers[layer],
                        uvLayer
                    );
                    vec2 cubeLod = textureQueryLod(cubeTex, direction);
                    int cubeLevels = textureQueryLevels(cubeTex);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        int_type = re.search(r"(%\d+) = OpTypeInt 32 1", spv_code)
        float_type = re.search(r"(%\d+) = OpTypeFloat 32", spv_code)
        sampler_type = re.search(r"(%\d+) = OpTypeSampler", spv_code)
        assert int_type is not None
        assert float_type is not None
        assert sampler_type is not None

        ivec2_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(int_type.group(1))} 2",
            spv_code,
        )
        ivec3_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(int_type.group(1))} 3",
            spv_code,
        )
        vec2_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(float_type.group(1))} 2",
            spv_code,
        )
        sampler_array_type = re.search(
            rf"(%\d+) = OpTypeArray {re.escape(sampler_type.group(1))} %\d+",
            spv_code,
        )

        assert ivec2_type is not None
        assert ivec3_type is not None
        assert vec2_type is not None
        assert sampler_array_type is not None
        assert "OpTypeImage" in spv_code
        assert " 2D 0 1 0 1 Unknown" in spv_code
        assert " Cube 0 1 0 1 Unknown" in spv_code
        assert " 2D 0 0 1 1 Unknown" in spv_code
        assert " 2D 0 1 1 1 Unknown" in spv_code
        assert spv_code.count("OpImageQuerySizeLod") == 3
        assert len(re.findall(r"OpImageQuerySize %\d+ %\d+", spv_code)) == 2
        assert spv_code.count("OpImageQueryLod") == 2
        assert spv_code.count("OpImageQueryLevels") == 1
        assert f"OpImageQuerySizeLod {ivec2_type.group(1)}" in spv_code
        assert f"OpImageQuerySizeLod {ivec3_type.group(1)}" in spv_code
        assert f"OpImageQuerySize {ivec2_type.group(1)}" in spv_code
        assert f"OpImageQuerySize {ivec3_type.group(1)}" in spv_code
        assert f"OpImageQueryLod {vec2_type.group(1)}" in spv_code
        assert f"OpImageQueryLevels {int_type.group(1)}" in spv_code
        assert f"OpLoad {sampler_array_type.group(1)}" not in spv_code
        assert "textureSize" not in spv_code
        assert "textureQueryLod" not in spv_code
        assert "textureQueryLevels" not in spv_code
        assert "WARNING" not in spv_code

    def test_sampler_1d_array_sampling_and_queries_emit_image_ops(self):
        source_code = """
        shader Resources {
            sampler1DArray lineArray;

            compute {
                void main() {
                    vec2 uvLayer = vec2(0.25, 1.0);
                    vec4 sampled = texture(lineArray, uvLayer);
                    ivec2 size = textureSize(lineArray, 0);
                    int levels = textureQueryLevels(lineArray);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        int_type = re.search(r"(%\d+) = OpTypeInt 32 1", spv_code)
        assert int_type is not None
        ivec2_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(int_type.group(1))} 2",
            spv_code,
        )
        assert ivec2_type is not None

        assert " 1D 0 1 0 1 Unknown" in spv_code
        assert "OpCapability Sampled1D" in spv_code
        assert "OpImageSampleExplicitLod" in spv_code
        assert f"OpImageQuerySizeLod {ivec2_type.group(1)}" in spv_code
        assert f"OpImageQueryLevels {int_type.group(1)}" in spv_code
        assert "texture(" not in spv_code
        assert "textureSize" not in spv_code
        assert "textureQueryLevels" not in spv_code
        assert "WARNING" not in spv_code

    def test_texture_query_lod_trims_array_layer_coordinates(self):
        source_code = """
        shader Resources {
            sampler2darray arrayTex;
            samplercubearray cubeArrayTex;
            sampler querySampler;

            compute {
                void main() {
                    vec3 uvLayer = vec3(0.25, 0.75, 1.0);
                    vec4 cubeLayer = vec4(1.0, 0.0, 0.0, 2.0);
                    vec2 explicitArrayLod = textureQueryLod(
                        arrayTex,
                        querySampler,
                        uvLayer
                    );
                    vec2 implicitCubeArrayLod = textureQueryLod(
                        cubeArrayTex,
                        cubeLayer
                    );
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        float_type = re.search(r"(%\d+) = OpTypeFloat 32", spv_code)
        assert float_type is not None
        vec2_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(float_type.group(1))} 2",
            spv_code,
        )
        vec3_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(float_type.group(1))} 3",
            spv_code,
        )
        assert vec2_type is not None
        assert vec3_type is not None

        query_coord_ids = re.findall(
            rf"OpImageQueryLod {re.escape(vec2_type.group(1))} %\d+ (%\d+)",
            spv_code,
        )
        vec2_constructs = set(
            re.findall(
                rf"(%\d+) = OpCompositeConstruct "
                rf"{re.escape(vec2_type.group(1))} %\d+ %\d+\b",
                spv_code,
            )
        )
        vec3_constructs = set(
            re.findall(
                rf"(%\d+) = OpCompositeConstruct "
                rf"{re.escape(vec3_type.group(1))} %\d+ %\d+ %\d+\b",
                spv_code,
            )
        )

        assert " 2D 0 1 0 1 Unknown" in spv_code
        assert " Cube 0 1 0 1 Unknown" in spv_code
        assert len(query_coord_ids) == 2
        assert any(coord_id in vec2_constructs for coord_id in query_coord_ids)
        assert any(coord_id in vec3_constructs for coord_id in query_coord_ids)
        assert all(
            coord_id in vec2_constructs or coord_id in vec3_constructs
            for coord_id in query_coord_ids
        )
        assert re.search(r"OpCompositeExtract %\d+ %\d+ 2\b", spv_code)
        assert not re.search(r"OpCompositeExtract %\d+ %\d+ 3\b", spv_code)
        assert spv_code.count("OpImageQueryLod") == 2
        assert "textureQueryLod" not in spv_code
        assert "WARNING" not in spv_code

    def test_multisample_texture_samples_query_emits_image_query_samples(self):
        source_code = """
        shader Resources {
            sampler2dms colorMs;
            sampler2dmsarray arrayMs;

            compute {
                void main() {
                    int samples = textureSamples(colorMs);
                    int arraySamples = textureSamples(arrayMs);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        int_type = re.search(r"(%\d+) = OpTypeInt 32 1", spv_code)
        assert int_type is not None
        assert spv_code.count("OpCapability ImageQuery") == 1
        assert " 2D 0 0 1 1 Unknown" in spv_code
        assert " 2D 0 1 1 1 Unknown" in spv_code
        assert spv_code.count("OpImageQuerySamples") == 2
        assert f"OpImageQuerySamples {int_type.group(1)}" in spv_code
        assert len(re.findall(r"%\d+ = OpImage %\d+ %\d+", spv_code)) == 2
        assert "OpFunctionCall" not in spv_code
        assert "textureSamples" not in spv_code
        assert "WARNING" not in spv_code

    def test_multisample_storage_image_samples_query_emits_image_query_samples(self):
        source_code = """
        shader Resources {
            image2DMS msColor @rgba16f;
            image2DMSArray msLayers @format(rgba8);
            uimage2DMS counters @r32ui;
            iimage2DMSArray signedLayers @r32i;

            compute {
                void main() {
                    int colorSamples = imageSamples(msColor);
                    int layerSamples = imageSamples(msLayers);
                    int counterSamples = imageSamples(counters);
                    int signedSamples = imageSamples(signedLayers);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        int_type = re.search(r"(%\d+) = OpTypeInt 32 1", spv_code)
        assert int_type is not None
        assert spv_code.count("OpCapability ImageQuery") == 1
        assert " 2D 0 0 1 2 Rgba16f" in spv_code
        assert " 2D 0 1 1 2 Rgba8" in spv_code
        assert " 2D 0 0 1 2 R32ui" in spv_code
        assert " 2D 0 1 1 2 R32i" in spv_code
        assert "OpTypeInt 32 0" in spv_code
        assert spv_code.count("OpImageQuerySamples") == 4
        assert f"OpImageQuerySamples {int_type.group(1)}" in spv_code
        assert "OpTypeSampledImage" not in spv_code
        assert "OpFunctionCall" not in spv_code
        assert "imageSamples" not in spv_code
        assert "WARNING" not in spv_code

    def test_shadow_texture_compare_emits_dref_sampling(self):
        source_code = """
        shader Resources {
            sampler2dshadow shadowMap;
            sampler2dshadow shadowMaps[4];
            sampler compareSampler;

            float sampleLayer(sampler2DShadow maps[4], int layer, vec2 uv, float depth) {
                return textureCompare(maps[layer], uv, depth);
            }

            float sampleLayerWithSampler(
                sampler2DShadow maps[4],
                sampler compareSampler,
                int layer,
                vec2 uv,
                float depth
            ) {
                return textureCompare(maps[layer], compareSampler, uv, depth);
            }

            compute {
                void main() {
                    int layer = 1;
                    vec2 uv = vec2(0.25, 0.75);
                    float direct = textureCompare(shadowMap, uv, 0.5);
                    float indexed = textureCompare(
                        shadowMaps[layer],
                        compareSampler,
                        uv,
                        0.4
                    );
                    float helper = sampleLayer(shadowMaps, layer, uv, 0.3);
                    float helperSampler = sampleLayerWithSampler(
                        shadowMaps,
                        compareSampler,
                        layer,
                        uv,
                        0.2
                    );
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        float_type = re.search(r"(%\d+) = OpTypeFloat 32", spv_code)
        shadow_image_type = re.search(
            r"%\d+ = OpTypeImage %\d+ 2D 1 0 0 1 Unknown",
            spv_code,
        )
        sampled_image_type = re.search(r"(%\d+) = OpTypeSampledImage %\d+", spv_code)
        assert float_type is not None
        assert shadow_image_type is not None
        assert sampled_image_type is not None

        array_type = re.search(
            rf"(%\d+) = OpTypeArray " rf"{re.escape(sampled_image_type.group(1))} %\d+",
            spv_code,
        )
        assert array_type is not None
        assert spv_code.count("OpImageSampleDrefExplicitLod") == 4
        assert f"OpImageSampleDrefExplicitLod {float_type.group(1)}" in spv_code
        assert "OpImageSampleImplicitLod" not in spv_code
        assert f"OpLoad {array_type.group(1)}" not in spv_code
        assert "textureCompare" not in spv_code
        assert "WARNING" not in spv_code

    def test_shadow_texture_compare_explicit_variants_emit_dref_operations(self):
        source_code = """
        shader Resources {
            sampler2dshadow shadowMap;
            sampler2dshadow shadowMaps[4];
            sampler compareSampler;

            float sampleLod(
                sampler2DShadow maps[4],
                sampler compareSampler,
                int layer,
                vec2 uv,
                float depth
            ) {
                return textureCompareLod(
                    maps[layer],
                    compareSampler,
                    uv,
                    depth,
                    1.0
                );
            }

            compute {
                void main() {
                    int layer = 1;
                    vec2 uv = vec2(0.25, 0.75);
                    vec2 dx = vec2(0.1, 0.0);
                    vec2 dy = vec2(0.0, 0.1);
                    ivec2 offset = ivec2(1, 0);
                    float lod = textureCompareLod(shadowMap, uv, 0.5, 2.0);
                    float grad = textureCompareGrad(
                        shadowMap,
                        compareSampler,
                        uv,
                        0.4,
                        dx,
                        dy
                    );
                    float offsetShadow = textureCompareOffset(
                        shadowMaps[layer],
                        uv,
                        0.3,
                        offset
                    );
                    vec4 gathered = textureGatherCompare(shadowMap, uv, 0.6);
                    vec4 gatheredOffset = textureGatherCompareOffset(
                        shadowMaps[layer],
                        compareSampler,
                        uv,
                        0.2,
                        offset
                    );
                    float helper = sampleLod(
                        shadowMaps,
                        compareSampler,
                        layer,
                        uv,
                        0.1
                    );
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        float_type = re.search(r"(%\d+) = OpTypeFloat 32", spv_code)
        sampled_image_type = re.search(r"(%\d+) = OpTypeSampledImage %\d+", spv_code)
        assert float_type is not None
        assert sampled_image_type is not None

        vec4_type = re.search(
            rf"(%\d+) = OpTypeVector {re.escape(float_type.group(1))} 4",
            spv_code,
        )
        array_type = re.search(
            rf"(%\d+) = OpTypeArray " rf"{re.escape(sampled_image_type.group(1))} %\d+",
            spv_code,
        )
        assert vec4_type is not None
        assert array_type is not None
        assert spv_code.count("OpImageSampleDrefExplicitLod") == 4
        assert spv_code.count("OpImageSampleDrefImplicitLod") == 0
        assert spv_code.count("OpImageDrefGather") == 2
        assert spv_code.count("OpCapability ImageGatherExtended") == 1
        assert "ConstOffset" not in spv_code
        assert (
            len(re.findall(r"\bOpImageSampleDrefExplicitLod\b.*\bOffset\b", spv_code))
            == 1
        )
        assert len(re.findall(r"\bOpImageDrefGather\b.*\bOffset\b", spv_code)) == 1
        assert f"OpImageSampleDrefExplicitLod {float_type.group(1)}" in spv_code
        assert f"OpImageSampleDrefImplicitLod {float_type.group(1)}" not in spv_code
        assert f"OpImageDrefGather {vec4_type.group(1)}" in spv_code
        assert re.search(
            r"OpImageSampleDrefExplicitLod %\d+ %\d+ %\d+ %\d+ Lod %\d+",
            spv_code,
        )
        assert re.search(
            r"OpImageSampleDrefExplicitLod %\d+ %\d+ %\d+ %\d+ Grad %\d+ %\d+",
            spv_code,
        )
        assert f"OpLoad {array_type.group(1)}" not in spv_code
        assert "textureCompareLod" not in spv_code
        assert "textureCompareGrad" not in spv_code
        assert "textureCompareOffset" not in spv_code
        assert "textureGatherCompare" not in spv_code
        assert "textureGatherCompareOffset" not in spv_code
        assert "WARNING" not in spv_code

    def test_resource_array_access_uses_uniform_constant_element_pointers(self):
        source_code = """
        shader Resources {
            sampler2d textures[4];
            image2D images @rgba16f[2];
            uimage2D counters @r32ui[2];

            compute {
                void main() {
                    int layer = 1;
                    vec2 uv = vec2(0.5, 0.25);
                    ivec2 pixel = ivec2(0, 0);
                    vec4 sampled = texture(textures[layer], uv);
                    vec4 color = imageLoad(images[1], pixel);
                    imageStore(images[0], pixel, color);
                    uint count = imageLoad(counters[1], pixel);
                    imageStore(counters[0], pixel, count);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        sampled_image_type = re.search(r"(%\d+) = OpTypeSampledImage %\d+", spv_code)
        rgba_image_type = re.search(
            r"(%\d+) = OpTypeImage %\d+ 2D 0 0 0 2 Rgba16f",
            spv_code,
        )
        uint_type = re.search(r"(%\d+) = OpTypeInt 32 0", spv_code)
        r32ui_image_type = re.search(
            r"(%\d+) = OpTypeImage %\d+ 2D 0 0 0 2 R32ui",
            spv_code,
        )

        assert sampled_image_type is not None
        assert rgba_image_type is not None
        assert uint_type is not None
        assert r32ui_image_type is not None
        assert (
            f"{r32ui_image_type.group(1)} = OpTypeImage {uint_type.group(1)} "
            "2D 0 0 0 2 R32ui"
        ) in spv_code

        for resource_name, element_type in [
            ("textures", sampled_image_type.group(1)),
            ("images", rgba_image_type.group(1)),
            ("counters", r32ui_image_type.group(1)),
        ]:
            array_type = re.search(
                rf"(%\d+) = OpTypeArray {re.escape(element_type)} %\d+",
                spv_code,
            )
            variable = spirv_named_variable(
                spv_code, resource_name, storage_class="UniformConstant"
            )
            element_pointer = re.search(
                rf"(%\d+) = OpTypePointer UniformConstant {re.escape(element_type)}",
                spv_code,
            )

            assert array_type is not None
            assert element_pointer is not None
            assert f"OpAccessChain {element_pointer.group(1)} {variable}" in spv_code
            assert f"OpLoad {array_type.group(1)} {variable}" not in spv_code
            assert f"OpTypePointer Function {element_type}" not in spv_code

        assert spv_code.count("OpImageRead") == 2
        assert spv_code.count("OpImageWrite") == 2
        assert "OpImageSampleImplicitLod" not in spv_code
        assert "OpImageSampleExplicitLod" in spv_code
        assert f"OpImageRead {uint_type.group(1)}" in spv_code
        assert "WARNING" not in spv_code

    def test_resource_array_parameters_pass_and_index_uniform_constant_pointers(self):
        source_code = """
        shader Resources {
            sampler2d textures[4];
            image2D images @rgba16f[2];
            uimage2D counters @r32ui[2];

            vec4 sampleLayer(sampler2d texArray[4], int layer, vec2 uv) {
                return texture(texArray[layer], uv);
            }

            vec4 sampleForward(sampler2d texArray[4], int layer, vec2 uv) {
                return sampleLayer(texArray, layer, uv);
            }

            vec4 readImage(image2D imageArray[2] @rgba16f, ivec2 pixel) {
                return imageLoad(imageArray[1], pixel);
            }

            uint readCounter(uimage2D counterArray[2] @r32ui, ivec2 pixel) {
                return imageLoad(counterArray[0], pixel);
            }

            compute {
                void main() {
                    int layer = 1;
                    vec2 uv = vec2(0.5, 0.25);
                    ivec2 pixel = ivec2(0, 0);
                    vec4 sampled = sampleForward(textures, layer, uv);
                    vec4 color = readImage(images, pixel);
                    uint count = readCounter(counters, pixel);
                }
            }
        }
        """

        spv_code = VulkanSPIRVCodeGen().generate(
            Parser(Lexer(source_code).tokens).parse()
        )

        sampled_image_type = re.search(r"(%\d+) = OpTypeSampledImage %\d+", spv_code)
        rgba_image_type = re.search(
            r"(%\d+) = OpTypeImage %\d+ 2D 0 0 0 2 Rgba16f",
            spv_code,
        )
        uint_type = re.search(r"(%\d+) = OpTypeInt 32 0", spv_code)
        r32ui_image_type = re.search(
            r"(%\d+) = OpTypeImage %\d+ 2D 0 0 0 2 R32ui",
            spv_code,
        )

        assert sampled_image_type is not None
        assert rgba_image_type is not None
        assert uint_type is not None
        assert r32ui_image_type is not None

        for resource_name, element_type, parameter_name in [
            ("textures", sampled_image_type.group(1), "texArray"),
            ("images", rgba_image_type.group(1), "imageArray"),
            ("counters", r32ui_image_type.group(1), "counterArray"),
        ]:
            array_type = re.search(
                rf"(%\d+) = OpTypeArray {re.escape(element_type)} %\d+",
                spv_code,
            )
            assert array_type is not None

            array_pointer = re.search(
                rf"(%\d+) = OpTypePointer UniformConstant " rf"{array_type.group(1)}",
                spv_code,
            )
            assert array_pointer is not None

            global_variable = spirv_named_variable(
                spv_code, resource_name, array_pointer.group(1), "UniformConstant"
            )

            parameter = spirv_named_parameter(
                spv_code, parameter_name, array_pointer.group(1)
            )

            element_pointer = re.search(
                rf"(%\d+) = OpTypePointer UniformConstant "
                rf"{re.escape(element_type)}",
                spv_code,
            )
            assert element_pointer is not None
            assert f"OpAccessChain {element_pointer.group(1)} {parameter}" in spv_code
            assert re.search(
                rf"OpFunctionCall %\d+ %\d+ {re.escape(global_variable)}",
                spv_code,
            )
            assert f"OpLoad {array_type.group(1)} {global_variable}" not in spv_code
            assert f"OpLoad {array_type.group(1)} {parameter}" not in spv_code

        tex_array_params = spirv_named_parameters(spv_code, "texArray")
        assert len(tex_array_params) >= 2
        assert re.search(
            rf"OpFunctionCall %\d+ %\d+ {re.escape(tex_array_params[1])} " r"%\d+ %\d+",
            spv_code,
        )
        assert "OpImageSampleExplicitLod" in spv_code
        assert spv_code.count("OpImageRead") >= 2
        assert f"OpImageRead {uint_type.group(1)}" in spv_code
        assert "WARNING" not in spv_code

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
