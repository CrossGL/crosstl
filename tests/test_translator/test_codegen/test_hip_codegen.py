"""Test HIP Code Generation from CrossGL"""

import pytest
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.ast import (
    AssignmentNode,
    BlockNode,
    ExecutionModel,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    LiteralNode,
    PrimitiveType,
    ShaderNode,
    VariableNode,
)
from crosstl.translator.codegen.hip_codegen import HipCodeGen


class TestHipCodeGen:
    def test_simple_function_generation(self):
        """Test generating a simple HIP function from CrossGL"""
        source_code = """
        shader TestShader {
            vertex {
                void main() {
                    int x;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "#include <hip/hip_runtime.h>" in hip_code
        assert "#include <hip/hip_runtime_api.h>" in hip_code
        assert "__device__ void main()" in hip_code

    def test_compute_shader_to_kernel(self):
        """Test converting a compute shader to HIP kernel"""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    int id;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "__global__ void main()" in hip_code

    def test_type_conversion(self):
        """Test CrossGL to HIP type conversion"""
        codegen = HipCodeGen()

        # Test basic types
        assert codegen.map_type("int") == "int"
        assert codegen.map_type("float") == "float"
        assert codegen.map_type("double") == "double"
        assert codegen.map_type("bool") == "bool"
        assert codegen.map_type("void") == "void"

        # Test vector types
        assert codegen.map_type("vec2") == "float2"
        assert codegen.map_type("vec3") == "float3"
        assert codegen.map_type("vec4") == "float4"
        assert codegen.map_type("ivec2") == "int2"
        assert codegen.map_type("ivec3") == "int3"
        assert codegen.map_type("ivec4") == "int4"
        assert codegen.map_type("vec2<f64>") == "double2"
        assert codegen.map_type("dvec3") == "double3"
        assert codegen.map_type("bvec2") == "uchar2"
        assert codegen.map_type("vec2<bool>") == "uchar2"
        assert codegen.map_type("bool3") == "uchar3"
        assert codegen.map_type("mat3x4") == "float3x4"
        assert codegen.map_type("dmat4x3") == "double4x3"

    def test_function_conversion(self):
        """Test CrossGL to HIP function conversion"""
        codegen = HipCodeGen()

        # Test math functions
        assert codegen.function_map.get("sqrt") == "sqrtf"
        assert codegen.function_map.get("pow") == "powf"
        assert codegen.function_map.get("sin") == "sinf"
        assert codegen.function_map.get("cos") == "cosf"

        # Test vector constructors
        assert codegen.function_map.get("vec2") == "make_float2"
        assert codegen.function_map.get("vec3") == "make_float3"
        assert codegen.function_map.get("vec4") == "make_float4"
        assert codegen.function_map.get("dvec2") == "make_double2"
        assert codegen.function_map.get("vec3<f64>") == "make_double3"
        assert codegen.function_map.get("bvec2") == "make_uchar2"
        assert codegen.function_map.get("vec2<bool>") == "make_uchar2"
        assert codegen.function_map.get("bool3") == "make_uchar3"
        assert codegen.function_map.get("mat3x4") == "float3x4"
        assert codegen.function_map.get("dmat2") == "double2x2"

    def test_vector_constructors_emit_hip_make_functions(self):
        """Test HIP maps parser-produced vector constructors."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    vec2 uv = vec2(0.5, 1.0);
                    dvec2 precise = dvec2(1.0, 2.0);
                    dvec4 weights = dvec4(1.0, 2.0, 3.0, 4.0);
                    ivec3 index = ivec3(1, 2, 3);
                    uvec4 mask = uvec4(1, 2, 3, 4);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "float2 uv = make_float2(0.5, 1.0);" in hip_code
        assert "double2 precise = make_double2(1.0, 2.0);" in hip_code
        assert "double4 weights = make_double4(1.0, 2.0, 3.0, 4.0);" in hip_code
        assert "int3 index = make_int3(1, 2, 3);" in hip_code
        assert "uint4 mask = make_uint4(1, 2, 3, 4);" in hip_code
        assert " = vec2(" not in hip_code
        assert " = dvec2(" not in hip_code
        assert " = dvec4(" not in hip_code
        assert " = ivec3(" not in hip_code
        assert " = uvec4(" not in hip_code

    def test_matrix_types_and_constructors_emit_hip_names(self):
        """Test HIP maps parser-produced matrix types and constructors."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    mat2 transform = mat2(1.0, 0.0, 0.0, 1.0);
                    mat3x4 affine;
                    dmat2 precise = dmat2(1.0, 0.0, 0.0, 1.0);
                    dmat4x3 jacobian;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "float2x2 transform = float2x2(1.0, 0.0, 0.0, 1.0);" in hip_code
        assert "float3x4 affine;" in hip_code
        assert "double2x2 precise = double2x2(1.0, 0.0, 0.0, 1.0);" in hip_code
        assert "double4x3 jacobian;" in hip_code
        assert "MatrixType(" not in hip_code
        assert " = mat2(" not in hip_code
        assert " = dmat2(" not in hip_code

    def test_bool_vector_constructors_emit_hip_names(self):
        """Test HIP maps boolean vectors to supported uchar vector types."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    bvec2 mask = bvec2(true, false);
                    bvec3 flags;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "uchar2 mask = make_uchar2(true, false);" in hip_code
        assert "uchar3 flags;" in hip_code
        assert "bvec2" not in hip_code
        assert "bool2 mask" not in hip_code
        assert "bool3 flags" not in hip_code

    def test_generic_vector_constructors_emit_hip_names(self):
        """Test HIP maps parser-produced generic vector forms."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    vec2<f64> precise = vec2<f64>(1.0, 2.0);
                    vec3<i32> index = vec3<i32>(1, 2, 3);
                    vec4<u32> mask = vec4<u32>(1, 2, 3, 4);
                    vec2<bool> flags = vec2<bool>(true, false);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "double2 precise = make_double2(1.0, 2.0);" in hip_code
        assert "int3 index = make_int3(1, 2, 3);" in hip_code
        assert "uint4 mask = make_uint4(1, 2, 3, 4);" in hip_code
        assert "uchar2 flags = make_uchar2(true, false);" in hip_code
        assert "vec2<" not in hip_code
        assert "vec3<" not in hip_code
        assert "vec4<" not in hip_code

    def test_builtin_variable_conversion(self):
        """Test built-in variable conversion"""
        codegen = HipCodeGen()

        # Test thread index mapping
        assert codegen.builtin_map.get("gl_LocalInvocationID.x") == "threadIdx.x"
        assert codegen.builtin_map.get("gl_LocalInvocationID.y") == "threadIdx.y"
        assert codegen.builtin_map.get("gl_LocalInvocationID.z") == "threadIdx.z"

        # Test workgroup index mapping
        assert codegen.builtin_map.get("gl_WorkGroupID.x") == "blockIdx.x"
        assert codegen.builtin_map.get("gl_WorkGroupID.y") == "blockIdx.y"
        assert codegen.builtin_map.get("gl_WorkGroupID.z") == "blockIdx.z"

        # Test workgroup size mapping
        assert codegen.builtin_map.get("gl_WorkGroupSize.x") == "blockDim.x"
        assert codegen.builtin_map.get("gl_WorkGroupSize.y") == "blockDim.y"
        assert codegen.builtin_map.get("gl_WorkGroupSize.z") == "blockDim.z"

    def test_builtin_invocation_ids_emit_hip_names(self):
        """Test HIP maps CrossGL invocation built-ins in member access form."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    int lx = gl_LocalInvocationID.x;
                    int gx = gl_GlobalInvocationID.x;
                    int wy = gl_WorkGroupID.y;
                    int bz = gl_WorkGroupSize.z;
                    int nz = gl_NumWorkGroups.z;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "int lx = threadIdx.x;" in hip_code
        assert "int gx = (blockIdx.x * blockDim.x + threadIdx.x);" in hip_code
        assert "int wy = blockIdx.y;" in hip_code
        assert "int bz = blockDim.z;" in hip_code
        assert "int nz = gridDim.z;" in hip_code
        assert "gl_LocalInvocationID" not in hip_code
        assert "gl_GlobalInvocationID" not in hip_code
        assert "gl_WorkGroupID" not in hip_code
        assert "gl_WorkGroupSize" not in hip_code
        assert "gl_NumWorkGroups" not in hip_code

    def test_direct_builtin_identifier_nodes_emit_hip_names(self):
        """Test HIP maps direct AST built-in identifiers with component suffixes."""
        ast = ShaderNode(
            name="DirectBuiltins",
            execution_model=ExecutionModel.GENERAL_PURPOSE,
            functions=[
                FunctionNode(
                    "main",
                    PrimitiveType("void"),
                    [],
                    BlockNode(
                        [
                            VariableNode(
                                "lx",
                                PrimitiveType("int"),
                                IdentifierNode("gl_LocalInvocationID.x"),
                            ),
                            VariableNode(
                                "gx",
                                PrimitiveType("int"),
                                IdentifierNode("gl_GlobalInvocationID.x"),
                            ),
                            VariableNode(
                                "wy",
                                PrimitiveType("int"),
                                IdentifierNode("gl_WorkGroupID.y"),
                            ),
                            VariableNode(
                                "bz",
                                PrimitiveType("int"),
                                IdentifierNode("gl_WorkGroupSize.z"),
                            ),
                            VariableNode(
                                "nz",
                                PrimitiveType("int"),
                                IdentifierNode("gl_NumWorkGroups.z"),
                            ),
                        ]
                    ),
                )
            ],
        )

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "int lx = threadIdx.x;" in hip_code
        assert "int gx = (blockIdx.x * blockDim.x + threadIdx.x);" in hip_code
        assert "int wy = blockIdx.y;" in hip_code
        assert "int bz = blockDim.z;" in hip_code
        assert "int nz = gridDim.z;" in hip_code
        assert "gl_LocalInvocationID" not in hip_code
        assert "gl_GlobalInvocationID" not in hip_code
        assert "gl_WorkGroupID" not in hip_code
        assert "gl_WorkGroupSize" not in hip_code
        assert "gl_NumWorkGroups" not in hip_code

    def test_struct_generation(self):
        """Test struct generation"""
        source_code = """
        shader TestShader {
            struct Vertex {
                vec3 position;
                vec3 normal;
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "struct Vertex" in hip_code
        assert "float3 position;" in hip_code
        assert "float3 normal;" in hip_code

    def test_variable_with_qualifiers(self):
        """Test variable generation with memory qualifiers"""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    float shared_data;
                    float constants;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        # For now, just check that basic variables are generated
        assert "float shared_data;" in hip_code
        assert "float constants;" in hip_code

    def test_atomic_operations(self):
        """Test atomic operations code generation"""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    int counter;
                    atomicAdd(counter, 1);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        # Check for basic structure
        assert "__global__" in hip_code or "__device__" in hip_code

    def test_synchronization_functions(self):
        """Test synchronization function generation"""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    barrier();
                    memoryBarrier();
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        # Check for basic structure
        assert "__global__" in hip_code or "__device__" in hip_code

    def test_vector_operations(self):
        """Test vector operations generation"""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    vec3 a = vec3(1.0, 2.0, 3.0);
                    vec3 b = vec3(4.0, 5.0, 6.0);
                    vec3 result = a + b;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        # Check for basic structure with HIP includes
        assert "#include <hip/hip_runtime.h>" in hip_code

    def test_math_functions(self):
        """Test math function generation"""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    float x = 1.0;
                    float result = sqrt(sin(x) + cos(x));
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        # Check for basic structure with HIP includes
        assert "#include <hip/hip_runtime.h>" in hip_code

    def test_texture_operations(self):
        """Test texture operation generation"""
        source_code = """
        shader TestShader {
            fragment {
                void main() {
                    sampler2D tex;
                    vec2 uv = vec2(0.5, 0.5);
                    vec4 color = texture(tex, uv);
                    vec4 lodColor = textureLod(tex, uv, 1.0);
                    vec4 gradColor = textureGrad(tex, uv, uv, uv);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "#include <hip/hip_runtime.h>" in hip_code
        assert "texture<float4, 2> tex;" in hip_code
        assert "float2 uv = make_float2(0.5, 0.5);" in hip_code
        assert "float4 color = tex2D(tex, uv);" in hip_code
        assert "float4 lodColor = tex2DLod(tex, uv, 1.0);" in hip_code
        assert "float4 gradColor = tex2DGrad(tex, uv, uv, uv);" in hip_code
        assert "sampler2D tex;" not in hip_code
        assert " = texture(" not in hip_code
        assert " = textureLod(" not in hip_code
        assert " = textureGrad(" not in hip_code

    def test_array_and_3d_texture_calls_emit_hip_texture_functions(self):
        """Test HIP maps array and 3D sampled texture calls by resource type."""
        source_code = """
        shader Resources {
            sampler2darray layers;
            sampler3d volumeMap;

            void sampleResources(
                sampler2darray paramLayers,
                sampler3d paramVolume,
                vec3 uvLayer,
                vec3 uvw,
                vec2 ddx2,
                vec2 ddy2,
                vec3 ddx3,
                vec3 ddy3
            ) {
                vec4 layerColor = texture(paramLayers, uvLayer);
                vec4 layerMip = textureLod(paramLayers, uvLayer, 2.0);
                vec4 layerGrad = textureGrad(paramLayers, uvLayer, ddx2, ddy2);
                vec4 volumeColor = texture(paramVolume, uvw);
                vec4 volumeMip = textureLod(paramVolume, uvw, 1.0);
                vec4 volumeGrad = textureGrad(paramVolume, uvw, ddx3, ddy3);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "hipTextureObject_t layers;" in hip_code
        assert "texture<float4, 3> volumeMap;" in hip_code
        assert (
            "float4 layerColor = tex2DLayered<float4>"
            "(paramLayers, uvLayer.x, uvLayer.y, uvLayer.z);" in hip_code
        )
        assert (
            "float4 layerMip = tex2DLayeredLod<float4>"
            "(paramLayers, uvLayer.x, uvLayer.y, uvLayer.z, 2.0);" in hip_code
        )
        assert (
            "float4 layerGrad = tex2DLayeredGrad<float4>"
            "(paramLayers, uvLayer.x, uvLayer.y, uvLayer.z, ddx2, ddy2);" in hip_code
        )
        assert (
            "float4 volumeColor = tex3D(paramVolume, uvw.x, uvw.y, uvw.z);" in hip_code
        )
        assert (
            "float4 volumeMip = tex3DLod(paramVolume, uvw.x, uvw.y, uvw.z, 1.0);"
            in hip_code
        )
        assert (
            "float4 volumeGrad = tex3DGrad"
            "(paramVolume, uvw.x, uvw.y, uvw.z, ddx3, ddy3);" in hip_code
        )
        assert " = texture(" not in hip_code
        assert " = textureLod(" not in hip_code
        assert " = textureGrad(" not in hip_code

    def test_1d_and_cube_texture_calls_emit_hip_texture_functions(self):
        """Test HIP maps 1D and cube sampled texture calls by resource type."""
        source_code = """
        shader Resources {
            sampler1d ramp;
            samplercube envMap;

            void sampleResources(
                sampler1d paramRamp,
                samplercube paramEnv,
                float u,
                vec3 dir,
                float du,
                vec4 ddxCube,
                vec4 ddyCube
            ) {
                vec4 rampColor = texture(paramRamp, u);
                vec4 rampMip = textureLod(paramRamp, u, 2.0);
                vec4 rampGrad = textureGrad(paramRamp, u, du, du);
                vec4 envColor = texture(paramEnv, dir);
                vec4 envMip = textureLod(paramEnv, dir, 1.0);
                vec4 envGrad = textureGrad(paramEnv, dir, ddxCube, ddyCube);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "texture<float4, 1> ramp;" in hip_code
        assert "textureCube<float4> envMap;" in hip_code
        assert "float4 rampColor = tex1D(paramRamp, u);" in hip_code
        assert "float4 rampMip = tex1DLod(paramRamp, u, 2.0);" in hip_code
        assert "float4 rampGrad = tex1DGrad(paramRamp, u, du, du);" in hip_code
        assert (
            "float4 envColor = texCubemap(paramEnv, dir.x, dir.y, dir.z);" in hip_code
        )
        assert (
            "float4 envMip = texCubemapLod(paramEnv, dir.x, dir.y, dir.z, 1.0);"
            in hip_code
        )
        assert (
            "float4 envGrad = texCubemapGrad"
            "(paramEnv, dir.x, dir.y, dir.z, ddxCube, ddyCube);" in hip_code
        )
        assert " = texture(" not in hip_code
        assert " = textureLod(" not in hip_code
        assert " = textureGrad(" not in hip_code

    def test_resource_type_keywords_emit_hip_resource_types(self):
        """Test HIP maps all parser resource keywords instead of leaking CrossGL types."""
        source_code = """
        shader Resources {
            sampler linearState;
            sampler1d ramp;
            sampler2d colorMap;
            sampler3d volumeMap;
            samplercube environmentMap;
            sampler2darray layers;
            sampler2dshadow shadowMap;
            sampler2darrayshadow cascades;
            samplercubeshadow cubeShadow;
            samplercubearray probes;
            samplercubearrayshadow shadowProbes;
            sampler2dms msTex;
            sampler2dmsarray msArray;
            iimage2d signedImage;
            iimage3d signedVolume;
            iimage2darray signedLayers;
            iimage2dms signedMs;
            iimage2dmsarray signedMsLayers;
            uimage2d unsignedImage;
            uimage3d unsignedVolume;
            uimage2darray unsignedLayers;
            uimage2dms unsignedMs;
            uimage2dmsarray unsignedMsLayers;
            image2D outputImage;
            image3D volumeImage;
            imageCube cubeImage;
            image2DArray layerImage;
            image2DMS outputMs;
            image2DMSArray layerMs;

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)
        declarations = {
            line.strip() for line in hip_code.splitlines() if line.strip().endswith(";")
        }

        assert "hipTextureObject_t linearState;" in declarations
        assert "texture<float4, 1> ramp;" in declarations
        assert "texture<float4, 2> colorMap;" in declarations
        assert "texture<float4, 3> volumeMap;" in declarations
        assert "textureCube<float4> environmentMap;" in declarations
        assert "hipTextureObject_t layers;" in declarations
        assert "hipTextureObject_t shadowMap;" in declarations
        assert "hipTextureObject_t cascades;" in declarations
        assert "hipTextureObject_t cubeShadow;" in declarations
        assert "hipTextureObject_t probes;" in declarations
        assert "hipTextureObject_t shadowProbes;" in declarations
        assert "hipTextureObject_t msTex;" in declarations
        assert "hipTextureObject_t msArray;" in declarations
        assert "hipSurfaceObject_t signedImage;" in declarations
        assert "hipSurfaceObject_t signedVolume;" in declarations
        assert "hipSurfaceObject_t signedLayers;" in declarations
        assert "hipSurfaceObject_t signedMs;" in declarations
        assert "hipSurfaceObject_t signedMsLayers;" in declarations
        assert "hipSurfaceObject_t unsignedImage;" in declarations
        assert "hipSurfaceObject_t unsignedVolume;" in declarations
        assert "hipSurfaceObject_t unsignedLayers;" in declarations
        assert "hipSurfaceObject_t unsignedMs;" in declarations
        assert "hipSurfaceObject_t unsignedMsLayers;" in declarations
        assert "hipSurfaceObject_t outputImage;" in declarations
        assert "hipSurfaceObject_t volumeImage;" in declarations
        assert "hipSurfaceObject_t cubeImage;" in declarations
        assert "hipSurfaceObject_t layerImage;" in declarations
        assert "hipSurfaceObject_t outputMs;" in declarations
        assert "hipSurfaceObject_t layerMs;" in declarations

        raw_resource_types = [
            "sampler",
            "sampler1D",
            "sampler2D",
            "sampler3D",
            "samplerCube",
            "sampler2DArray",
            "sampler2DShadow",
            "sampler2DArrayShadow",
            "samplerCubeShadow",
            "samplerCubeArray",
            "samplerCubeArrayShadow",
            "sampler2DMS",
            "sampler2DMSArray",
            "iimage2D",
            "iimage3D",
            "iimage2DArray",
            "iimage2DMS",
            "iimage2DMSArray",
            "uimage2D",
            "uimage3D",
            "uimage2DArray",
            "uimage2DMS",
            "uimage2DMSArray",
            "image2D",
            "image3D",
            "imageCube",
            "image2DArray",
            "image2DMS",
            "image2DMSArray",
        ]
        for raw_type in raw_resource_types:
            assert not any(
                declaration.startswith(f"{raw_type} ") for declaration in declarations
            )

    def test_resource_builtins_emit_hip_texture_and_surface_calls(self):
        """Test HIP lowers non-multisample resource calls with tracked resource types."""
        source_code = """
        shader Resources {
            sampler2d tex;
            sampler2darray layers;
            image2D colorImage;
            image3D volumeImage;
            image2DArray layerImage;
            iimage2d signedImage;

            void processResources(
                sampler2d paramTex,
                ivec2 pixel,
                ivec3 pixelLayer
            ) {
                vec4 fetched = texelFetch(paramTex, pixel, 1);
                vec4 fetchedLayer = texelFetch(layers, pixelLayer, 1);
                vec4 regular = imageLoad(colorImage, pixel);
                imageStore(colorImage, pixel, regular);
                vec4 voxel = imageLoad(volumeImage, pixelLayer);
                imageStore(volumeImage, pixelLayer, voxel);
                vec4 layered = imageLoad(layerImage, pixelLayer);
                imageStore(layerImage, pixelLayer, layered);
                int signedValue = imageLoad(signedImage, pixel);
                imageStore(signedImage, pixel, signedValue);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "hipSurfaceObject_t colorImage;" in hip_code
        assert "hipSurfaceObject_t volumeImage;" in hip_code
        assert "hipSurfaceObject_t layerImage;" in hip_code
        assert "hipSurfaceObject_t signedImage;" in hip_code
        assert (
            "__device__ T cgl_surf2Dread(hipSurfaceObject_t surfObj, int x, int y)"
            in hip_code
        )
        assert (
            "__device__ T cgl_surf3Dread(hipSurfaceObject_t surfObj, int x, int y, int z)"
            in hip_code
        )
        assert (
            "__device__ T cgl_surf2DLayeredread"
            "(hipSurfaceObject_t surfObj, int x, int y, int layer)" in hip_code
        )
        assert "float4 fetched = tex2D(paramTex, pixel.x, pixel.y);" in hip_code
        assert (
            "float4 fetchedLayer = tex2DLayered<float4>"
            "(layers, pixelLayer.x, pixelLayer.y, pixelLayer.z);" in hip_code
        )
        assert (
            "float4 regular = cgl_surf2Dread<float4>"
            "(colorImage, pixel.x * sizeof(float4), pixel.y);" in hip_code
        )
        assert (
            "surf2Dwrite(regular, colorImage, pixel.x * sizeof(float4), pixel.y);"
            in hip_code
        )
        assert (
            "float4 voxel = cgl_surf3Dread<float4>"
            "(volumeImage, pixelLayer.x * sizeof(float4), pixelLayer.y, pixelLayer.z);"
            in hip_code
        )
        assert (
            "surf3Dwrite(voxel, volumeImage, "
            "pixelLayer.x * sizeof(float4), pixelLayer.y, pixelLayer.z);" in hip_code
        )
        assert (
            "float4 layered = cgl_surf2DLayeredread<float4>"
            "(layerImage, pixelLayer.x * sizeof(float4), pixelLayer.y, pixelLayer.z);"
            in hip_code
        )
        assert (
            "surf2DLayeredwrite(layered, layerImage, "
            "pixelLayer.x * sizeof(float4), pixelLayer.y, pixelLayer.z);" in hip_code
        )
        assert (
            "int signedValue = cgl_surf2Dread<int>"
            "(signedImage, pixel.x * sizeof(int), pixel.y);" in hip_code
        )
        assert (
            "surf2Dwrite(signedValue, signedImage, pixel.x * sizeof(int), pixel.y);"
            in hip_code
        )
        assert "texelFetch(" not in hip_code
        assert "imageLoad(" not in hip_code
        assert "imageStore(" not in hip_code

    def test_multisample_resource_builtins_emit_hip_diagnostics(self):
        """Test HIP does not silently lower MS resource calls to non-MS functions."""
        source_code = """
        shader Resources {
            sampler2dms msTex;
            sampler2dmsarray msLayers;
            image2DMS msImage;
            image2DMSArray msImageLayers;

            void msResources(
                ivec2 pixel,
                ivec3 pixelLayer,
                vec2 uv,
                int sampleIndex
            ) {
                vec4 sampled = texture(msTex, uv);
                vec4 fetched = texelFetch(msTex, pixel, sampleIndex);
                vec4 fetchedLayer = texelFetch(msLayers, pixelLayer, sampleIndex);
                vec4 loaded = imageLoad(msImage, pixel, sampleIndex);
                imageStore(msImage, pixel, sampleIndex, loaded);
                vec4 loadedLayer = imageLoad(msImageLayers, pixelLayer, sampleIndex);
                imageStore(msImageLayers, pixelLayer, sampleIndex, loadedLayer);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "hipTextureObject_t msTex;" in hip_code
        assert "hipTextureObject_t msLayers;" in hip_code
        assert "hipSurfaceObject_t msImage;" in hip_code
        assert "hipSurfaceObject_t msImageLayers;" in hip_code
        assert (
            "/* unsupported HIP multisample resource call: texture on sampler2DMS */ "
            "texture(msTex, uv)" in hip_code
        )
        assert (
            "/* unsupported HIP multisample resource call: texelFetch on sampler2DMS */ "
            "texelFetch(msTex, pixel, sampleIndex)" in hip_code
        )
        assert (
            "/* unsupported HIP multisample resource call: texelFetch on sampler2DMSArray */ "
            "texelFetch(msLayers, pixelLayer, sampleIndex)" in hip_code
        )
        assert (
            "/* unsupported HIP multisample resource call: imageLoad on image2DMS */ "
            "imageLoad(msImage, pixel, sampleIndex)" in hip_code
        )
        assert (
            "/* unsupported HIP multisample resource call: imageStore on image2DMS */ "
            "imageStore(msImage, pixel, sampleIndex, loaded)" in hip_code
        )
        assert (
            "/* unsupported HIP multisample resource call: imageLoad on image2DMSArray */ "
            "imageLoad(msImageLayers, pixelLayer, sampleIndex)" in hip_code
        )
        assert (
            "/* unsupported HIP multisample resource call: imageStore on image2DMSArray */ "
            "imageStore(msImageLayers, pixelLayer, sampleIndex, loadedLayer)"
            in hip_code
        )
        assert "tex2D(msTex" not in hip_code
        assert "tex2DLayered<float4>(msLayers" not in hip_code
        assert "cgl_surf2Dread" not in hip_code
        assert "cgl_surf2DLayeredread" not in hip_code
        assert "surf2Dwrite" not in hip_code
        assert "surf2DLayeredwrite" not in hip_code

    def test_control_flow(self):
        """Test control flow generation"""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    int i;
                    if (i > 0) {
                        i = i + 1;
                    } else {
                        i = i - 1;
                    }
                    
                    for (int j = 0; j < 10; j++) {
                        i = i * 2;
                    }
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "#include <hip/hip_runtime.h>" in hip_code
        assert "for (int j = 0; (j < 10); j++)" in hip_code
        assert "int j = 0;\n    for" not in hip_code

    def test_compound_assignments_preserve_operator(self):
        """Test HIP preserves compound assignment operators."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    int total = 0;
                    total += 1;
                    total -= 2;
                    total *= 3;
                    total /= 4;
                    total %= 5;
                    for (int j = 0; j < 3; j += 1) {
                        total += j;
                    }
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "total += 1;" in hip_code
        assert "total -= 2;" in hip_code
        assert "total *= 3;" in hip_code
        assert "total /= 4;" in hip_code
        assert "total %= 5;" in hip_code
        assert "for (int j = 0; (j < 3); j += 1)" in hip_code
        assert "total += j;" in hip_code
        assert "total = 1;" not in hip_code
        assert "total = j;" not in hip_code

    def test_while_loop_generation(self):
        """Test HIP emits while loops for block and single-statement bodies."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    int value = 0;
                    while (value < 4) {
                        value += 1;
                    }
                    while (value < 8)
                        value += 2;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "while ((value < 4))" in hip_code
        assert "value += 1;" in hip_code
        assert "while ((value < 8))" in hip_code
        assert "value += 2;" in hip_code
        assert "NotImplemented" not in hip_code

    def test_switch_generation(self):
        """Test HIP emits switch, case, default, and case bodies."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    int mode = 1;
                    int value = 0;
                    switch (mode) {
                        case 0:
                            value = 10;
                            break;
                        case 1:
                            value += 20;
                            break;
                        default:
                            value = -1;
                    }
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "switch (mode)" in hip_code
        assert "case 0:" in hip_code
        assert "value = 10;" in hip_code
        assert "case 1:" in hip_code
        assert "value += 20;" in hip_code
        assert "default:" in hip_code
        assert "value = -1;" in hip_code
        assert hip_code.count("break;") == 2
        assert "NotImplemented" not in hip_code

    def test_direct_ast_expression_statements_are_emitted(self):
        """Test direct HIP AST function bodies emit expression-returning nodes."""
        ast = ShaderNode(
            name="DirectAst",
            execution_model=ExecutionModel.GENERAL_PURPOSE,
            functions=[
                FunctionNode(
                    "main",
                    PrimitiveType("void"),
                    [],
                    BlockNode(
                        [
                            VariableNode(
                                "value",
                                PrimitiveType("int"),
                                LiteralNode(0, PrimitiveType("int")),
                            ),
                            FunctionCallNode(
                                IdentifierNode("barrier"),
                                [],
                            ),
                            AssignmentNode(
                                IdentifierNode("value"),
                                LiteralNode(1, PrimitiveType("int")),
                                "+=",
                            ),
                        ]
                    ),
                )
            ],
        )

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "int value = 0;" in hip_code
        assert "__syncthreads();" in hip_code
        assert "value += 1;" in hip_code

    def test_bool_string_and_char_literals_emit_hip_syntax(self):
        """Test HIP literal output uses target-language spelling."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    bool enabled = true;
                    bool disabled = false;
                    string label = "debug";
                    char marker = 'x';
                    if (enabled && !disabled) {
                        label = "active";
                        marker = 'y';
                    }
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()
        entry_point = next(iter(ast.stages.values())).entry_point

        assert entry_point.body.statements[0].initial_value.value is True
        assert entry_point.body.statements[1].initial_value.value is False

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "bool enabled = true;" in hip_code
        assert "bool disabled = false;" in hip_code
        assert 'string label = "debug";' in hip_code
        assert "char marker = 'x';" in hip_code
        assert 'label = "active";' in hip_code
        assert "marker = 'y';" in hip_code
        assert "True" not in hip_code
        assert "False" not in hip_code

    def test_direct_literal_nodes_emit_hip_escaping(self):
        """Test direct HIP literal formatting escapes quotes."""
        codegen = HipCodeGen()

        assert codegen.visit(LiteralNode(True, PrimitiveType("bool"))) == "true"
        assert (
            codegen.visit(LiteralNode('debug"name', PrimitiveType("string")))
            == '"debug\\"name"'
        )
        assert codegen.visit(LiteralNode("'", PrimitiveType("char"))) == "'\\''"

    def test_array_operations(self):
        """Test array operation generation"""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    float data[256];
                    int index = 0;
                    data[index] = 1.0;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        # Check for basic structure
        assert "#include <hip/hip_runtime.h>" in hip_code

    def test_array_declarations_emit_c_style_declarators(self):
        """Test HIP array declarations place dimensions after the variable name."""
        source_code = """
        shader TestShader {
            struct Material {
                float weights[4];
                vec3 colors[2];
            };

            compute {
                void main(float input_weights[4], vec3 input_colors[2], float dynamic_weights[]) {
                    float data[256];
                    vec3 local_colors[2];
                    vec3 scratch[];
                    data[0] = input_weights[1];
                    local_colors[0] = input_colors[1];
                    data[1] = dynamic_weights[0];
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "float weights[4];" in hip_code
        assert "float3 colors[2];" in hip_code
        assert (
            "void main(float input_weights[4], float3 input_colors[2], "
            "float* dynamic_weights)"
        ) in hip_code
        assert "float data[256];" in hip_code
        assert "float3 local_colors[2];" in hip_code
        assert "float3* scratch;" in hip_code
        assert "data[0] = input_weights[1];" in hip_code
        assert "local_colors[0] = input_colors[1];" in hip_code
        assert "data[1] = dynamic_weights[0];" in hip_code
        assert "LiteralNode" not in hip_code
        assert "float[256] data" not in hip_code
        assert "float3[2] local_colors" not in hip_code

    def test_empty_shader(self):
        """Test empty shader generation"""
        source_code = ""

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "#include <hip/hip_runtime.h>" in hip_code
        assert "#include <hip/hip_runtime_api.h>" in hip_code

    def test_prefix_and_postfix_unary_operators_preserve_position(self):
        """Test HIP preserves prefix/postfix increment and decrement operators."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    int i = 0;
                    i++;
                    ++i;
                    i--;
                    --i;
                    for (; i < 2; i++) {
                        i++;
                    }
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "i++;" in hip_code
        assert "++i;" in hip_code
        assert "i--;" in hip_code
        assert "--i;" in hip_code
        assert "for (; (i < 2); i++)" in hip_code
        assert "++i++" not in hip_code

    def test_multiple_functions(self):
        """Test multiple function generation"""
        source_code = """
        shader TestShader {
            vertex {
                void vertex_main() {
                    int x;
                }
            }
            fragment {
                void fragment_main() {
                    float y;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        # Check for HIP includes
        assert "#include <hip/hip_runtime.h>" in hip_code
        assert "#include <hip/hip_runtime_api.h>" in hip_code
