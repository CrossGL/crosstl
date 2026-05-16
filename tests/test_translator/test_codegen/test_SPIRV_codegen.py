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
        assert "Unknown variable inputValue" not in spv_code
        assert "Unknown variable outputValue" not in spv_code
        assert "WARNING" not in spv_code

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
        normalized_var = re.search(
            r"(%\d+) = OpVariable %\d+ UniformConstant\nOpName \1 \"normalizedColor\"",
            spv_code,
        )

        assert rgba8_type is not None
        assert rgba16f_type is not None
        assert normalized_var is not None
        assert f"= OpLoad {rgba8_type.group(1)} {normalized_var.group(1)}" in spv_code
        assert (
            f"= OpLoad {rgba16f_type.group(1)} {normalized_var.group(1)}"
            not in spv_code
        )
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
        assert "OpImageSampleImplicitLod" in spv_code
        assert f"OpImageRead {uint_type.group(1)}" in spv_code
        assert f"OpImageSampleImplicitLod {vec4_type.group(1)}" in spv_code
        assert "OpFunctionCall" not in spv_code
        assert "imageLoad" not in spv_code
        assert "imageStore" not in spv_code
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

        sampled_image_type = re.search(r"(%\d+) = OpTypeSampledImage %\d+", spv_code)
        vec4_type = re.search(r"(%\d+) = OpTypeVector %\d+ 4", spv_code)
        assert sampled_image_type is not None
        assert vec4_type is not None

        array_type = re.search(
            rf"(%\d+) = OpTypeArray " rf"{re.escape(sampled_image_type.group(1))} %\d+",
            spv_code,
        )

        assert array_type is not None
        assert spv_code.count("OpImageSampleImplicitLod") == 2
        assert spv_code.count("ConstOffset") == 2
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
        assert spv_code.count("OpImageSampleImplicitLod") == 3
        assert spv_code.count("OpImageSampleExplicitLod") == 2
        assert spv_code.count("OpImageGather") == 3
        assert spv_code.count("OpImageFetch") == 1
        assert len(re.findall(r"\bConstOffset\b", spv_code)) == 2
        assert len(re.findall(r"\bConstOffsets\b", spv_code)) == 1
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
        assert f"OpImageQuerySizeLod {ivec2_type.group(1)}" in spv_code
        assert f"OpImageQuerySize {ivec2_type.group(1)}" in spv_code
        assert f"OpImageQuerySize {ivec3_type.group(1)}" in spv_code
        assert f"OpImageQueryLevels {int_type.group(1)}" in spv_code
        assert f"OpImageQueryLod {vec2_type.group(1)}" in spv_code
        assert f"OpLoad {array_type.group(1)}" not in spv_code
        assert "OpFunctionCall" not in spv_code
        assert "textureSize" not in spv_code
        assert "imageSize" not in spv_code
        assert "textureQueryLevels" not in spv_code
        assert "textureQueryLod" not in spv_code
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
        assert spv_code.count("OpImageSampleDrefImplicitLod") == 4
        assert f"OpImageSampleDrefImplicitLod {float_type.group(1)}" in spv_code
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
        assert spv_code.count("OpImageSampleDrefExplicitLod") == 3
        assert spv_code.count("OpImageSampleDrefImplicitLod") == 1
        assert spv_code.count("OpImageDrefGather") == 2
        assert spv_code.count("ConstOffset") == 2
        assert f"OpImageSampleDrefExplicitLod {float_type.group(1)}" in spv_code
        assert f"OpImageSampleDrefImplicitLod {float_type.group(1)}" in spv_code
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
            variable = re.search(
                rf"(%\d+) = OpVariable %\d+ UniformConstant\n"
                rf"OpName \1 \"{resource_name}\"",
                spv_code,
            )
            element_pointer = re.search(
                rf"(%\d+) = OpTypePointer UniformConstant {re.escape(element_type)}",
                spv_code,
            )

            assert array_type is not None
            assert variable is not None
            assert element_pointer is not None
            assert (
                f"OpAccessChain {element_pointer.group(1)} {variable.group(1)}"
                in spv_code
            )
            assert f"OpLoad {array_type.group(1)} {variable.group(1)}" not in spv_code
            assert f"OpTypePointer Function {element_type}" not in spv_code

        assert spv_code.count("OpImageRead") == 2
        assert spv_code.count("OpImageWrite") == 2
        assert "OpImageSampleImplicitLod" in spv_code
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

            global_variable = re.search(
                rf"(%\d+) = OpVariable {array_pointer.group(1)} "
                rf"UniformConstant\nOpName \1 \"{resource_name}\"",
                spv_code,
            )
            assert global_variable is not None

            parameter = re.search(
                rf"(%\d+) = OpFunctionParameter {array_pointer.group(1)}\n"
                rf"OpName \1 \"{parameter_name}\"",
                spv_code,
            )
            assert parameter is not None

            element_pointer = re.search(
                rf"(%\d+) = OpTypePointer UniformConstant "
                rf"{re.escape(element_type)}",
                spv_code,
            )
            assert element_pointer is not None
            assert (
                f"OpAccessChain {element_pointer.group(1)} {parameter.group(1)}"
                in spv_code
            )
            assert re.search(
                rf"OpFunctionCall %\d+ %\d+ {re.escape(global_variable.group(1))}",
                spv_code,
            )
            assert (
                f"OpLoad {array_type.group(1)} {global_variable.group(1)}"
                not in spv_code
            )
            assert f"OpLoad {array_type.group(1)} {parameter.group(1)}" not in spv_code

        tex_array_params = re.findall(
            r"(%\d+) = OpFunctionParameter %\d+\nOpName \1 \"texArray\"",
            spv_code,
        )
        assert len(tex_array_params) >= 2
        assert re.search(
            rf"OpFunctionCall %\d+ %\d+ {re.escape(tex_array_params[1])} " r"%\d+ %\d+",
            spv_code,
        )
        assert "OpImageSampleImplicitLod" in spv_code
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
