"""Test CUDA Code Generation from CrossGL"""

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
from crosstl.translator.codegen.cuda_codegen import CudaCodeGen


class TestCudaCodeGen:
    def test_simple_function_generation(self):
        """Test generating a simple CUDA function from CrossGL"""
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

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "#include <cuda_runtime.h>" in cuda_code
        assert "#include <device_launch_parameters.h>" in cuda_code
        assert "__device__ void main()" in cuda_code

    def test_compute_shader_to_kernel(self):
        """Test converting a compute shader to CUDA kernel"""
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

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "__global__ void main()" in cuda_code
        assert "int3 threadIdx =" not in cuda_code
        assert "int3 blockIdx =" not in cuda_code
        assert "int3 blockDim =" not in cuda_code
        assert "int3 gridDim =" not in cuda_code

    def test_type_conversion(self):
        """Test CrossGL to CUDA type conversion"""
        codegen = CudaCodeGen()

        # Test basic types
        assert codegen.convert_crossgl_type_to_cuda("i32") == "int"
        assert codegen.convert_crossgl_type_to_cuda("f32") == "float"
        assert codegen.convert_crossgl_type_to_cuda("f64") == "double"
        assert codegen.convert_crossgl_type_to_cuda("bool") == "bool"
        assert codegen.convert_crossgl_type_to_cuda("void") == "void"
        assert codegen.convert_crossgl_type_to_cuda("sampler2D") == "texture<float4, 2>"

        # Test vector types
        assert codegen.convert_crossgl_type_to_cuda("vec2<f32>") == "float2"
        assert codegen.convert_crossgl_type_to_cuda("vec3<f32>") == "float3"
        assert codegen.convert_crossgl_type_to_cuda("vec4<f32>") == "float4"
        assert codegen.convert_crossgl_type_to_cuda("vec2<f64>") == "double2"
        assert codegen.convert_crossgl_type_to_cuda("dvec3") == "double3"
        assert codegen.convert_crossgl_type_to_cuda("vec2<i32>") == "int2"
        assert codegen.convert_crossgl_type_to_cuda("bvec2") == "uchar2"
        assert codegen.convert_crossgl_type_to_cuda("vec2<bool>") == "uchar2"
        assert codegen.convert_crossgl_type_to_cuda("bool3") == "uchar3"
        assert codegen.convert_crossgl_type_to_cuda("mat3x4") == "float3x4"
        assert codegen.convert_crossgl_type_to_cuda("dmat4x3") == "double4x3"

    def test_function_conversion(self):
        """Test CrossGL to CUDA function conversion"""
        codegen = CudaCodeGen()

        # Test math functions
        assert codegen.convert_builtin_function("sqrt") == "sqrtf"
        assert codegen.convert_builtin_function("pow") == "powf"
        assert codegen.convert_builtin_function("sin") == "sinf"
        assert codegen.convert_builtin_function("cos") == "cosf"
        assert codegen.convert_builtin_function("asin") == "asinf"
        assert codegen.convert_builtin_function("atan2") == "atan2f"
        assert codegen.convert_builtin_function("exp2") == "exp2f"
        assert codegen.convert_builtin_function("log2") == "log2f"
        assert codegen.convert_builtin_function("inversesqrt") == "rsqrtf"
        assert codegen.convert_builtin_function("round") == "roundf"
        assert codegen.convert_builtin_function("trunc") == "truncf"
        assert codegen.convert_builtin_function("mod") == "fmodf"

        # Test vector constructors
        assert codegen.convert_builtin_function("vec2") == "make_float2"
        assert codegen.convert_builtin_function("vec3") == "make_float3"
        assert codegen.convert_builtin_function("vec4") == "make_float4"
        assert codegen.convert_builtin_function("vec2<f32>") == "make_float2"
        assert codegen.convert_builtin_function("vec3<f32>") == "make_float3"
        assert codegen.convert_builtin_function("vec4<f32>") == "make_float4"
        assert codegen.convert_builtin_function("dvec2") == "make_double2"
        assert codegen.convert_builtin_function("vec3<f64>") == "make_double3"
        assert codegen.convert_builtin_function("ivec3") == "make_int3"
        assert codegen.convert_builtin_function("uvec4") == "make_uint4"
        assert codegen.convert_builtin_function("bvec2") == "make_uchar2"
        assert codegen.convert_builtin_function("vec2<bool>") == "make_uchar2"
        assert codegen.convert_builtin_function("bool3") == "make_uchar3"
        assert codegen.convert_builtin_function("mat3x4") == "float3x4"
        assert codegen.convert_builtin_function("dmat2") == "double2x2"

        # Test atomic operations
        assert codegen.convert_builtin_function("atomicAdd") == "atomicAdd"
        assert codegen.convert_builtin_function("atomicExchange") == "atomicExch"

        # Test synchronization
        assert codegen.convert_builtin_function("workgroupBarrier") == "__syncthreads"

        # Test texture functions
        assert codegen.convert_builtin_function("texture") == "tex2D"
        assert codegen.convert_builtin_function("textureLod") == "tex2DLod"
        assert codegen.convert_builtin_function("textureGrad") == "tex2DGrad"

    def test_builtin_invocation_ids_emit_cuda_names(self):
        """Test CUDA maps CrossGL invocation built-ins in member access form."""
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

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "int lx = threadIdx.x;" in cuda_code
        assert "int gx = (blockIdx.x * blockDim.x + threadIdx.x);" in cuda_code
        assert "int wy = blockIdx.y;" in cuda_code
        assert "int bz = blockDim.z;" in cuda_code
        assert "int nz = gridDim.z;" in cuda_code
        assert "int3 threadIdx =" not in cuda_code
        assert "int3 blockIdx =" not in cuda_code
        assert "int3 blockDim =" not in cuda_code
        assert "int3 gridDim =" not in cuda_code
        assert "gl_LocalInvocationID" not in cuda_code
        assert "gl_GlobalInvocationID" not in cuda_code
        assert "gl_WorkGroupID" not in cuda_code
        assert "gl_WorkGroupSize" not in cuda_code
        assert "gl_NumWorkGroups" not in cuda_code

    def test_direct_builtin_identifier_nodes_emit_cuda_names(self):
        """Test CUDA maps direct AST built-in identifiers with component suffixes."""
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

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "int lx = threadIdx.x;" in cuda_code
        assert "int gx = (blockIdx.x * blockDim.x + threadIdx.x);" in cuda_code
        assert "int wy = blockIdx.y;" in cuda_code
        assert "int bz = blockDim.z;" in cuda_code
        assert "int nz = gridDim.z;" in cuda_code
        assert "gl_LocalInvocationID" not in cuda_code
        assert "gl_GlobalInvocationID" not in cuda_code
        assert "gl_WorkGroupID" not in cuda_code
        assert "gl_WorkGroupSize" not in cuda_code
        assert "gl_NumWorkGroups" not in cuda_code

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

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "struct Vertex {" in cuda_code
        assert "float3 position;" in cuda_code
        assert "float3 normal;" in cuda_code

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

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        # For now, just check that basic variables are generated
        assert "float shared_data;" in cuda_code
        assert "float constants;" in cuda_code

    def test_vector_constructors_emit_cuda_make_functions(self):
        """Test CUDA maps parser-produced vector constructors."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    vec2 uv = vec2(0.5, 1.0);
                    vec3 normal = vec3(1.0, 2.0, 3.0);
                    vec4 color = vec4(1.0, 0.5, 0.25, 1.0);
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

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "float2 uv = make_float2(0.5, 1.0);" in cuda_code
        assert "float3 normal = make_float3(1.0, 2.0, 3.0);" in cuda_code
        assert "float4 color = make_float4(1.0, 0.5, 0.25, 1.0);" in cuda_code
        assert "double2 precise = make_double2(1.0, 2.0);" in cuda_code
        assert "double4 weights = make_double4(1.0, 2.0, 3.0, 4.0);" in cuda_code
        assert "int3 index = make_int3(1, 2, 3);" in cuda_code
        assert "uint4 mask = make_uint4(1, 2, 3, 4);" in cuda_code
        assert " = vec2(" not in cuda_code
        assert " = vec3(" not in cuda_code
        assert " = vec4(" not in cuda_code
        assert " = dvec2(" not in cuda_code
        assert " = dvec4(" not in cuda_code
        assert " = ivec3(" not in cuda_code
        assert " = uvec4(" not in cuda_code

    def test_vector_scalar_arithmetic_expands_cuda_components(self):
        """Test CUDA lowers vector math through component-wise helpers."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    float weight = 0.5;
                    vec4 base = vec4(1.0);
                    vec4 offset = vec4(0.25, 0.5, 0.75, 1.0);
                    vec4 scaled = base * weight;
                    vec4 biased = weight + base;
                    vec4 shifted = base - 1.0;
                    vec4 divided = base / weight;
                    vec4 combined = base + offset;
                    base *= weight;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "float4 base = make_float4(1.0, 1.0, 1.0, 1.0);" in cuda_code
        assert "__device__ inline float4 cgl_float4_mul_scalar" in cuda_code
        assert "__device__ inline float4 cgl_scalar_add_float4" in cuda_code
        assert "__device__ inline float4 cgl_float4_sub_scalar" in cuda_code
        assert "__device__ inline float4 cgl_float4_div_scalar" in cuda_code
        assert "__device__ inline float4 cgl_float4_add" in cuda_code
        assert "float4 scaled = cgl_float4_mul_scalar(base, weight);" in cuda_code
        assert "float4 biased = cgl_scalar_add_float4(weight, base);" in cuda_code
        assert "float4 shifted = cgl_float4_sub_scalar(base, 1.0);" in cuda_code
        assert "float4 divided = cgl_float4_div_scalar(base, weight);" in cuda_code
        assert "float4 combined = cgl_float4_add(base, offset);" in cuda_code
        assert "base = cgl_float4_mul_scalar(base, weight);" in cuda_code
        assert "(base * weight)" not in cuda_code
        assert "(weight + base)" not in cuda_code
        assert "make_float4(1.0);" not in cuda_code

    def test_matrix_types_and_constructors_emit_cuda_names(self):
        """Test CUDA maps parser-produced matrix types and constructors."""
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

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "float2x2 transform = float2x2(1.0, 0.0, 0.0, 1.0);" in cuda_code
        assert "float3x4 affine;" in cuda_code
        assert "double2x2 precise = double2x2(1.0, 0.0, 0.0, 1.0);" in cuda_code
        assert "double4x3 jacobian;" in cuda_code
        assert "MatrixType(" not in cuda_code
        assert " = mat2(" not in cuda_code
        assert " = dmat2(" not in cuda_code

    def test_bool_vector_constructors_emit_cuda_names(self):
        """Test CUDA maps boolean vectors to supported uchar vector types."""
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

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "uchar2 mask = make_uchar2(true, false);" in cuda_code
        assert "uchar3 flags;" in cuda_code
        assert "bvec2" not in cuda_code
        assert "bool2 mask" not in cuda_code
        assert "bool3 flags" not in cuda_code

    def test_generic_vector_constructors_emit_cuda_names(self):
        """Test CUDA maps parser-produced generic vector forms."""
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

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "double2 precise = make_double2(1.0, 2.0);" in cuda_code
        assert "int3 index = make_int3(1, 2, 3);" in cuda_code
        assert "uint4 mask = make_uint4(1, 2, 3, 4);" in cuda_code
        assert "uchar2 flags = make_uchar2(true, false);" in cuda_code
        assert "vec2<" not in cuda_code
        assert "vec3<" not in cuda_code
        assert "vec4<" not in cuda_code

    def test_extended_math_builtins_emit_cuda_functions(self):
        """Test CUDA maps parser-produced math builtins to CUDA functions."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    float x = 2.5;
                    float a = inversesqrt(x);
                    float b = round(x);
                    float c = trunc(x);
                    float d = mod(x, 2.0);
                    float e = asin(x);
                    float f = atan2(x, 1.0);
                    float g = exp2(x);
                    float h = log2(x);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "float a = rsqrtf(x);" in cuda_code
        assert "float b = roundf(x);" in cuda_code
        assert "float c = truncf(x);" in cuda_code
        assert "float d = fmodf(x, 2.0);" in cuda_code
        assert "float e = asinf(x);" in cuda_code
        assert "float f = atan2f(x, 1.0);" in cuda_code
        assert "float g = exp2f(x);" in cuda_code
        assert "float h = log2f(x);" in cuda_code
        assert "inversesqrt(" not in cuda_code
        assert "mod(" not in cuda_code
        assert "atan2(" not in cuda_code

    def test_texture_calls_emit_cuda_texture_functions(self):
        """Test CUDA maps parser-produced texture calls."""
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

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "texture<float4, 2> tex;" in cuda_code
        assert "float2 uv = make_float2(0.5, 0.5);" in cuda_code
        assert "float4 color = tex2D(tex, uv);" in cuda_code
        assert "float4 lodColor = tex2DLod(tex, uv, 1.0);" in cuda_code
        assert "float4 gradColor = tex2DGrad(tex, uv, uv, uv);" in cuda_code
        assert "sampler2D tex;" not in cuda_code
        assert " = texture(" not in cuda_code
        assert " = textureLod(" not in cuda_code
        assert " = textureGrad(" not in cuda_code

    def test_array_and_3d_texture_calls_emit_cuda_texture_functions(self):
        """Test CUDA maps array and 3D sampled texture calls by resource type."""
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

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "cudaTextureObject_t layers;" in cuda_code
        assert "texture<float4, 3> volumeMap;" in cuda_code
        assert (
            "float4 layerColor = tex2DLayered<float4>"
            "(paramLayers, uvLayer.x, uvLayer.y, uvLayer.z);" in cuda_code
        )
        assert (
            "float4 layerMip = tex2DLayeredLod<float4>"
            "(paramLayers, uvLayer.x, uvLayer.y, uvLayer.z, 2.0);" in cuda_code
        )
        assert (
            "float4 layerGrad = tex2DLayeredGrad<float4>"
            "(paramLayers, uvLayer.x, uvLayer.y, uvLayer.z, ddx2, ddy2);" in cuda_code
        )
        assert (
            "float4 volumeColor = tex3D(paramVolume, uvw.x, uvw.y, uvw.z);" in cuda_code
        )
        assert (
            "float4 volumeMip = tex3DLod(paramVolume, uvw.x, uvw.y, uvw.z, 1.0);"
            in cuda_code
        )
        assert (
            "float4 volumeGrad = tex3DGrad"
            "(paramVolume, uvw.x, uvw.y, uvw.z, ddx3, ddy3);" in cuda_code
        )
        assert " = texture(" not in cuda_code
        assert " = textureLod(" not in cuda_code
        assert " = textureGrad(" not in cuda_code

    def test_1d_and_cube_texture_calls_emit_cuda_texture_functions(self):
        """Test CUDA maps 1D and cube sampled texture calls by resource type."""
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

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "texture<float4, 1> ramp;" in cuda_code
        assert "textureCube<float4> envMap;" in cuda_code
        assert "float4 rampColor = tex1D(paramRamp, u);" in cuda_code
        assert "float4 rampMip = tex1DLod(paramRamp, u, 2.0);" in cuda_code
        assert "float4 rampGrad = tex1DGrad(paramRamp, u, du, du);" in cuda_code
        assert (
            "float4 envColor = texCubemap(paramEnv, dir.x, dir.y, dir.z);" in cuda_code
        )
        assert (
            "float4 envMip = texCubemapLod(paramEnv, dir.x, dir.y, dir.z, 1.0);"
            in cuda_code
        )
        assert (
            "float4 envGrad = texCubemapGrad"
            "(paramEnv, dir.x, dir.y, dir.z, ddxCube, ddyCube);" in cuda_code
        )
        assert " = texture(" not in cuda_code
        assert " = textureLod(" not in cuda_code
        assert " = textureGrad(" not in cuda_code

    def test_cube_array_texture_calls_emit_cuda_texture_functions(self):
        """Test CUDA maps cube-array sampled texture calls by resource type."""
        source_code = """
        shader Resources {
            samplercubearray probes;

            void sampleResources(
                samplercubearray paramProbes,
                vec4 dirLayer,
                vec4 ddxCube,
                vec4 ddyCube
            ) {
                vec4 color = texture(paramProbes, dirLayer);
                vec4 mip = textureLod(paramProbes, dirLayer, 2.0);
                vec4 grad = textureGrad(paramProbes, dirLayer, ddxCube, ddyCube);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "cudaTextureObject_t probes;" in cuda_code
        assert (
            "float4 color = texCubemapLayered<float4>"
            "(paramProbes, dirLayer.x, dirLayer.y, dirLayer.z, dirLayer.w);"
            in cuda_code
        )
        assert (
            "float4 mip = texCubemapLayeredLod<float4>"
            "(paramProbes, dirLayer.x, dirLayer.y, dirLayer.z, dirLayer.w, 2.0);"
            in cuda_code
        )
        assert (
            "float4 grad = texCubemapLayeredGrad<float4>"
            "(paramProbes, dirLayer.x, dirLayer.y, dirLayer.z, dirLayer.w, "
            "ddxCube, ddyCube);" in cuda_code
        )
        assert " = texture(" not in cuda_code
        assert " = textureLod(" not in cuda_code
        assert " = textureGrad(" not in cuda_code

    def test_shadow_sampler_calls_emit_cuda_diagnostics(self):
        """Test CUDA makes unsupported shadow sampler calls explicit."""
        source_code = """
        shader Resources {
            sampler2dshadow shadowMap;
            sampler2darrayshadow cascades;
            samplercubeshadow cubeShadow;
            samplercubearrayshadow cubeArrayShadow;
            sampler cmpSampler;

            void sampleResources(
                sampler2dshadow paramShadow,
                vec3 uvw,
                vec4 uvwLayer,
                vec3 dirDepth,
                vec4 dirLayerDepth,
                vec2 uv,
                float depth,
                vec2 ddx,
                vec2 ddy,
                ivec2 offset
            ) {
                float paramValue = textureCompare(paramShadow, uv, depth);
                float shadow = texture(shadowMap, uvw);
                float cascade = texture(cascades, uvwLayer);
                float cube = texture(cubeShadow, dirDepth);
                float cubeArray = texture(cubeArrayShadow, dirLayerDepth);
                float compared = textureCompare(shadowMap, cmpSampler, uv, depth);
                float comparedLod = textureCompareLod(shadowMap, uv, depth, 2.0);
                float comparedGrad = textureCompareGrad(shadowMap, uv, depth, ddx, ddy);
                float comparedOffset = textureCompareOffset(shadowMap, uv, depth, offset);
                vec4 gathered = textureGatherCompare(shadowMap, cmpSampler, uv, depth);
                vec4 gatheredOffset = textureGatherCompareOffset(
                    shadowMap,
                    uv,
                    depth,
                    offset
                );
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert (
            "float paramValue = /* unsupported CUDA shadow resource call: "
            "textureCompare on sampler2DShadow */ 0.0f;" in cuda_code
        )
        assert (
            "float shadow = /* unsupported CUDA shadow resource call: "
            "texture on sampler2DShadow */ 0.0f;" in cuda_code
        )
        assert (
            "float cascade = /* unsupported CUDA shadow resource call: "
            "texture on sampler2DArrayShadow */ 0.0f;" in cuda_code
        )
        assert (
            "float cube = /* unsupported CUDA shadow resource call: "
            "texture on samplerCubeShadow */ 0.0f;" in cuda_code
        )
        assert (
            "float cubeArray = /* unsupported CUDA shadow resource call: "
            "texture on samplerCubeArrayShadow */ 0.0f;" in cuda_code
        )
        assert (
            "float compared = /* unsupported CUDA shadow resource call: "
            "textureCompare on sampler2DShadow */ 0.0f;" in cuda_code
        )
        assert (
            "float comparedLod = /* unsupported CUDA shadow resource call: "
            "textureCompareLod on sampler2DShadow */ 0.0f;" in cuda_code
        )
        assert (
            "float comparedGrad = /* unsupported CUDA shadow resource call: "
            "textureCompareGrad on sampler2DShadow */ 0.0f;" in cuda_code
        )
        assert (
            "float comparedOffset = /* unsupported CUDA shadow resource call: "
            "textureCompareOffset on sampler2DShadow */ 0.0f;" in cuda_code
        )
        assert (
            "float4 gathered = /* unsupported CUDA shadow resource call: "
            "textureGatherCompare on sampler2DShadow */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in cuda_code
        )
        assert (
            "float4 gatheredOffset = /* unsupported CUDA shadow resource call: "
            "textureGatherCompareOffset on sampler2DShadow */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in cuda_code
        )
        assert "textureCompare(" not in cuda_code
        assert "textureCompareLod(" not in cuda_code
        assert "textureCompareGrad(" not in cuda_code
        assert "textureCompareOffset(" not in cuda_code
        assert "textureGatherCompare(" not in cuda_code
        assert "textureGatherCompareOffset(" not in cuda_code
        assert "texture(shadowMap" not in cuda_code
        assert "texture(cascades" not in cuda_code
        assert "texture(cubeShadow" not in cuda_code
        assert "texture(cubeArrayShadow" not in cuda_code
        assert " = tex2D(shadowMap" not in cuda_code
        assert " = tex2D(cascades" not in cuda_code
        assert " = tex2D(cubeShadow" not in cuda_code
        assert " = tex2D(cubeArrayShadow" not in cuda_code

    def test_texture_gather_calls_emit_cuda_diagnostics(self):
        """Test CUDA makes unsupported gather sampled-resource calls explicit."""
        source_code = """
        shader Resources {
            sampler2d colorMap;
            sampler2darray layers;
            sampler3d volumeTex;
            sampler querySampler;

            void gatherResources(
                sampler2d paramTex,
                vec2 uv,
                vec3 uvLayer,
                ivec2 offset,
                ivec2 offsets[4]
            ) {
                vec4 gathered = textureGather(colorMap, uv);
                vec4 gatheredComponent = textureGather(colorMap, querySampler, uv, 1);
                vec4 gatheredParam = textureGather(paramTex, uv, 2);
                vec4 gatheredLayer = textureGather(layers, querySampler, uvLayer, 0);
                vec4 gatheredOffset = textureGatherOffset(
                    colorMap,
                    querySampler,
                    uv,
                    offset,
                    1
                );
                vec4 gatheredOffsets = textureGatherOffsets(
                    colorMap,
                    querySampler,
                    uv,
                    offsets,
                    2
                );
                vec4 gatheredVolume = textureGather(volumeTex, uvLayer);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert (
            "float4 gathered = /* unsupported CUDA sampled resource call: "
            "textureGather on sampler2D */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in cuda_code
        )
        assert (
            "float4 gatheredComponent = /* unsupported CUDA sampled resource call: "
            "textureGather on sampler2D */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in cuda_code
        )
        assert (
            "float4 gatheredParam = /* unsupported CUDA sampled resource call: "
            "textureGather on sampler2D */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in cuda_code
        )
        assert (
            "float4 gatheredLayer = /* unsupported CUDA sampled resource call: "
            "textureGather on sampler2DArray */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in cuda_code
        )
        assert (
            "float4 gatheredOffset = /* unsupported CUDA sampled resource call: "
            "textureGatherOffset on sampler2D */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in cuda_code
        )
        assert (
            "float4 gatheredOffsets = /* unsupported CUDA sampled resource call: "
            "textureGatherOffsets on sampler2D */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in cuda_code
        )
        assert (
            "float4 gatheredVolume = /* unsupported CUDA sampled resource call: "
            "textureGather on sampler3D */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in cuda_code
        )
        assert "textureGather(" not in cuda_code
        assert "textureGatherOffset(" not in cuda_code
        assert "textureGatherOffsets(" not in cuda_code

    def test_resource_type_keywords_emit_cuda_resource_types(self):
        """Test CUDA maps all parser resource keywords instead of leaking CrossGL types."""
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

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)
        declarations = {
            line.strip()
            for line in cuda_code.splitlines()
            if line.strip().endswith(";")
        }

        assert "cudaTextureObject_t linearState;" in declarations
        assert "texture<float4, 1> ramp;" in declarations
        assert "texture<float4, 2> colorMap;" in declarations
        assert "texture<float4, 3> volumeMap;" in declarations
        assert "textureCube<float4> environmentMap;" in declarations
        assert "cudaTextureObject_t layers;" in declarations
        assert "cudaTextureObject_t shadowMap;" in declarations
        assert "cudaTextureObject_t cascades;" in declarations
        assert "cudaTextureObject_t cubeShadow;" in declarations
        assert "cudaTextureObject_t probes;" in declarations
        assert "cudaTextureObject_t shadowProbes;" in declarations
        assert "cudaTextureObject_t msTex;" in declarations
        assert "cudaTextureObject_t msArray;" in declarations
        assert "cudaSurfaceObject_t signedImage;" in declarations
        assert "cudaSurfaceObject_t signedVolume;" in declarations
        assert "cudaSurfaceObject_t signedLayers;" in declarations
        assert "cudaSurfaceObject_t signedMs;" in declarations
        assert "cudaSurfaceObject_t signedMsLayers;" in declarations
        assert "cudaSurfaceObject_t unsignedImage;" in declarations
        assert "cudaSurfaceObject_t unsignedVolume;" in declarations
        assert "cudaSurfaceObject_t unsignedLayers;" in declarations
        assert "cudaSurfaceObject_t unsignedMs;" in declarations
        assert "cudaSurfaceObject_t unsignedMsLayers;" in declarations
        assert "cudaSurfaceObject_t outputImage;" in declarations
        assert "cudaSurfaceObject_t volumeImage;" in declarations
        assert "cudaSurfaceObject_t cubeImage;" in declarations
        assert "cudaSurfaceObject_t layerImage;" in declarations
        assert "cudaSurfaceObject_t outputMs;" in declarations
        assert "cudaSurfaceObject_t layerMs;" in declarations

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

    def test_resource_builtins_emit_cuda_texture_and_surface_calls(self):
        """Test CUDA lowers non-multisample resource calls with tracked resource types."""
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

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "cudaSurfaceObject_t colorImage;" in cuda_code
        assert "cudaSurfaceObject_t volumeImage;" in cuda_code
        assert "cudaSurfaceObject_t layerImage;" in cuda_code
        assert "cudaSurfaceObject_t signedImage;" in cuda_code
        assert "float4 fetched = tex2D(paramTex, pixel.x, pixel.y);" in cuda_code
        assert (
            "float4 fetchedLayer = tex2DLayered<float4>"
            "(layers, pixelLayer.x, pixelLayer.y, pixelLayer.z);" in cuda_code
        )
        assert (
            "float4 regular = surf2Dread<float4>"
            "(colorImage, pixel.x * sizeof(float4), pixel.y);" in cuda_code
        )
        assert (
            "surf2Dwrite(regular, colorImage, pixel.x * sizeof(float4), pixel.y);"
            in cuda_code
        )
        assert (
            "float4 voxel = surf3Dread<float4>"
            "(volumeImage, pixelLayer.x * sizeof(float4), pixelLayer.y, pixelLayer.z);"
            in cuda_code
        )
        assert (
            "surf3Dwrite(voxel, volumeImage, "
            "pixelLayer.x * sizeof(float4), pixelLayer.y, pixelLayer.z);" in cuda_code
        )
        assert (
            "float4 layered = surf2DLayeredread<float4>"
            "(layerImage, pixelLayer.x * sizeof(float4), pixelLayer.y, pixelLayer.z);"
            in cuda_code
        )
        assert (
            "surf2DLayeredwrite(layered, layerImage, "
            "pixelLayer.x * sizeof(float4), pixelLayer.y, pixelLayer.z);" in cuda_code
        )
        assert (
            "int signedValue = surf2Dread<int>"
            "(signedImage, pixel.x * sizeof(int), pixel.y);" in cuda_code
        )
        assert (
            "surf2Dwrite(signedValue, signedImage, pixel.x * sizeof(int), pixel.y);"
            in cuda_code
        )
        assert "texelFetch(" not in cuda_code
        assert "imageLoad(" not in cuda_code
        assert "imageStore(" not in cuda_code
        assert "CglResourceQueryInfo" not in cuda_code

    def test_nested_resource_arrays_emit_cuda_texture_and_surface_calls(self):
        """Test CUDA preserves nested resource-array indices in resource calls."""
        source_code = """
        shader Resources {
            sampler2d textureGrid[2][3];
            sampler2darray layerGrid[2][2];
            sampler2dms msGrid[2][2];
            image2D imageGrid[2][4];
            uimage2D counterGrid[2][2];

            void processNested(
                sampler2d paramTextures[2][3],
                image2D paramImages[2][4],
                int layer,
                int slot,
                vec2 uv,
                vec3 uvLayer,
                ivec2 pixel,
                ivec3 pixelLayer,
                int sampleIndex
            ) {
                vec4 sampled = texture(textureGrid[layer][slot], uv);
                vec4 sampledLayer = texture(layerGrid[layer][slot], uvLayer);
                vec4 mip = textureLod(textureGrid[layer][slot], uv, 1.0);
                vec4 grad = textureGrad(textureGrid[layer][slot], uv, uv, uv);
                vec4 sampledParam = texture(paramTextures[layer][slot], uv);
                vec4 gathered = textureGather(textureGrid[layer][slot], uv, 1);
                vec4 fetched = texelFetch(textureGrid[layer][slot], pixel, 0);
                vec4 fetchedLayer = texelFetch(layerGrid[layer][slot], pixelLayer, 0);
                vec4 fetchedMs = texelFetch(msGrid[layer][slot], pixel, sampleIndex);
                vec4 color = imageLoad(imageGrid[layer][slot], pixel);
                vec4 colorParam = imageLoad(paramImages[layer][slot], pixel);
                imageStore(imageGrid[layer][slot], pixel, color);
                uint count = imageLoad(counterGrid[layer][slot], pixel);
                imageStore(counterGrid[layer][slot], pixel, count);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "texture<float4, 2> textureGrid[2][3];" in cuda_code
        assert "cudaTextureObject_t layerGrid[2][2];" in cuda_code
        assert "cudaTextureObject_t msGrid[2][2];" in cuda_code
        assert "cudaSurfaceObject_t imageGrid[2][4];" in cuda_code
        assert "cudaSurfaceObject_t counterGrid[2][2];" in cuda_code
        assert (
            "__device__ void processNested("
            "texture<float4, 2> paramTextures[2][3], "
            "cudaSurfaceObject_t paramImages[2][4], int layer, int slot, "
            "float2 uv, float3 uvLayer, int2 pixel, int3 pixelLayer, "
            "int sampleIndex)" in cuda_code
        )
        assert "float4 sampled = tex2D(textureGrid[layer][slot], uv);" in cuda_code
        assert (
            "float4 sampledLayer = tex2DLayered<float4>"
            "(layerGrid[layer][slot], uvLayer.x, uvLayer.y, uvLayer.z);" in cuda_code
        )
        assert "float4 mip = tex2DLod(textureGrid[layer][slot], uv, 1.0);" in cuda_code
        assert (
            "float4 grad = tex2DGrad(textureGrid[layer][slot], uv, uv, uv);"
            in cuda_code
        )
        assert (
            "float4 sampledParam = tex2D(paramTextures[layer][slot], uv);" in cuda_code
        )
        assert (
            "float4 gathered = /* unsupported CUDA sampled resource call: "
            "textureGather on sampler2D */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in cuda_code
        )
        assert (
            "float4 fetched = tex2D(textureGrid[layer][slot], pixel.x, pixel.y);"
            in cuda_code
        )
        assert (
            "float4 fetchedLayer = tex2DLayered<float4>"
            "(layerGrid[layer][slot], pixelLayer.x, pixelLayer.y, pixelLayer.z);"
            in cuda_code
        )
        assert (
            "float4 fetchedMs = /* unsupported CUDA multisample resource call: "
            "texelFetch on sampler2DMS */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in cuda_code
        )
        assert (
            "float4 color = surf2Dread<float4>"
            "(imageGrid[layer][slot], pixel.x * sizeof(float4), pixel.y);" in cuda_code
        )
        assert (
            "float4 colorParam = surf2Dread<float4>"
            "(paramImages[layer][slot], pixel.x * sizeof(float4), pixel.y);"
            in cuda_code
        )
        assert (
            "surf2Dwrite(color, imageGrid[layer][slot], "
            "pixel.x * sizeof(float4), pixel.y);" in cuda_code
        )
        assert (
            "uint count = surf2Dread<uint>"
            "(counterGrid[layer][slot], pixel.x * sizeof(uint), pixel.y);" in cuda_code
        )
        assert (
            "surf2Dwrite(count, counterGrid[layer][slot], "
            "pixel.x * sizeof(uint), pixel.y);" in cuda_code
        )
        assert "texture(" not in cuda_code
        assert "textureLod(" not in cuda_code
        assert "textureGrad(" not in cuda_code
        assert "textureGather(" not in cuda_code
        assert "texelFetch(" not in cuda_code
        assert "imageLoad(" not in cuda_code
        assert "imageStore(" not in cuda_code
        assert "CglResourceQueryInfo" not in cuda_code

    def test_dynamic_resource_array_params_emit_cuda_pointer_shapes(self):
        """Test CUDA preserves dynamic resource-array parameter shapes."""
        source_code = """
        shader Resources {
            sampler2d textures[4];
            sampler2darray layerTextures[3];
            image2D images[4];
            uimage2D counters[4];
            sampler2d textureGrid[2][3];
            image2D imageGrid[2][3];

            void useDynamic(
                sampler2d dynTextures[],
                sampler2darray dynLayers[],
                image2D dynImages[],
                uimage2D dynCounters[],
                int i,
                vec2 uv,
                vec3 uvLayer,
                ivec2 pixel
            ) {
                vec4 sampled = texture(dynTextures[i], uv);
                vec4 layered = texture(dynLayers[i], uvLayer);
                vec4 fetched = texelFetch(dynTextures[i], pixel, 0);
                vec4 color = imageLoad(dynImages[i], pixel);
                imageStore(dynImages[i], pixel, color);
                uint count = imageLoad(dynCounters[i], pixel);
                imageStore(dynCounters[i], pixel, count);
            }

            void useDynamicGrid(
                sampler2d dynGrid[][3],
                image2D dynGridImages[][3],
                int layer,
                int slot,
                vec2 uv,
                ivec2 pixel
            ) {
                vec4 sampled = texture(dynGrid[layer][slot], uv);
                vec4 color = imageLoad(dynGridImages[layer][slot], pixel);
                imageStore(dynGridImages[layer][slot], pixel, color);
            }

            void callDynamic(
                int i,
                int layer,
                int slot,
                vec2 uv,
                vec3 uvLayer,
                ivec2 pixel
            ) {
                useDynamic(textures, layerTextures, images, counters, i, uv, uvLayer, pixel);
                useDynamicGrid(textureGrid, imageGrid, layer, slot, uv, pixel);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert (
            "__device__ void useDynamic(texture<float4, 2>* dynTextures, "
            "cudaTextureObject_t* dynLayers, cudaSurfaceObject_t* dynImages, "
            "cudaSurfaceObject_t* dynCounters, int i, float2 uv, "
            "float3 uvLayer, int2 pixel)" in cuda_code
        )
        assert (
            "__device__ void useDynamicGrid("
            "texture<float4, 2> (*dynGrid)[3], "
            "cudaSurfaceObject_t (*dynGridImages)[3], int layer, int slot, "
            "float2 uv, int2 pixel)" in cuda_code
        )
        assert "float4 sampled = tex2D(dynTextures[i], uv);" in cuda_code
        assert (
            "float4 layered = tex2DLayered<float4>"
            "(dynLayers[i], uvLayer.x, uvLayer.y, uvLayer.z);" in cuda_code
        )
        assert "float4 fetched = tex2D(dynTextures[i], pixel.x, pixel.y);" in cuda_code
        assert (
            "float4 color = surf2Dread<float4>"
            "(dynImages[i], pixel.x * sizeof(float4), pixel.y);" in cuda_code
        )
        assert (
            "surf2Dwrite(color, dynImages[i], pixel.x * sizeof(float4), pixel.y);"
            in cuda_code
        )
        assert (
            "uint count = surf2Dread<uint>"
            "(dynCounters[i], pixel.x * sizeof(uint), pixel.y);" in cuda_code
        )
        assert (
            "surf2Dwrite(count, dynCounters[i], pixel.x * sizeof(uint), pixel.y);"
            in cuda_code
        )
        assert "float4 sampled = tex2D(dynGrid[layer][slot], uv);" in cuda_code
        assert (
            "float4 color = surf2Dread<float4>"
            "(dynGridImages[layer][slot], pixel.x * sizeof(float4), pixel.y);"
            in cuda_code
        )
        assert (
            "surf2Dwrite(color, dynGridImages[layer][slot], "
            "pixel.x * sizeof(float4), pixel.y);" in cuda_code
        )
        assert "texture<float4, 2>* dynGrid[3]" not in cuda_code
        assert "cudaSurfaceObject_t* dynGridImages[3]" not in cuda_code
        assert "texture(" not in cuda_code
        assert "texelFetch(" not in cuda_code
        assert "imageLoad(" not in cuda_code
        assert "imageStore(" not in cuda_code
        assert "CglResourceQueryInfo" not in cuda_code

    def test_forwarded_dynamic_resource_arrays_emit_cuda_metadata_arguments(self):
        """Test CUDA forwards query metadata sidecars through dynamic arrays."""
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

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "CglResourceQueryInfo textureGrid_metadata[2][3] = {};" in cuda_code
        assert "CglResourceQueryInfo imageGrid_metadata[2][3] = {};" in cuda_code
        assert "counterGrid_metadata" not in cuda_code
        assert (
            "__device__ float4 leafSample("
            "texture<float4, 2> (*dynGrid)[3], int layer, int slot, float2 uv)"
            in cuda_code
        )
        assert (
            "__device__ float4 forwardSample("
            "texture<float4, 2> (*dynGrid)[3], int layer, int slot, float2 uv)"
            in cuda_code
        )
        assert "return leafSample(dynGrid, layer, slot, uv);" in cuda_code
        assert (
            "__device__ float4 leafImage(cudaSurfaceObject_t (*dynImages)[3], "
            "int layer, int slot, int2 pixel)" in cuda_code
        )
        assert "return leafImage(dynImages, layer, slot, pixel);" in cuda_code
        assert (
            "__device__ uint leafCounter(cudaSurfaceObject_t (*dynCounters)[2], "
            "int layer, int slot, int2 pixel)" in cuda_code
        )
        assert "return leafCounter(dynCounters, layer, slot, pixel);" in cuda_code
        assert (
            "__device__ void leafQuery(texture<float4, 2> (*dynGrid)[3], "
            "CglResourceQueryInfo (*dynGrid_metadata)[3], "
            "cudaSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], int layer, int slot)"
            in cuda_code
        )
        assert (
            "leafQuery(dynGrid, dynGrid_metadata, dynImages, "
            "dynImages_metadata, layer, slot);" in cuda_code
        )
        assert (
            "forwardQuery(textureGrid, textureGrid_metadata, imageGrid, "
            "imageGrid_metadata, 1, 2);" in cuda_code
        )
        assert "return tex2D(dynGrid[layer][slot], uv);" in cuda_code
        assert (
            "float4 color = surf2Dread<float4>"
            "(dynImages[layer][slot], pixel.x * sizeof(float4), pixel.y);" in cuda_code
        )
        assert (
            "uint count = surf2Dread<uint>"
            "(dynCounters[layer][slot], pixel.x * sizeof(uint), pixel.y);" in cuda_code
        )
        assert (
            "int2 texSize = cgl_textureSize_sampler2D"
            "(dynGrid_metadata[layer][slot], 0);" in cuda_code
        )
        assert (
            "int2 imageSizeValue = cgl_imageSize_image2D"
            "(dynImages_metadata[layer][slot]);" in cuda_code
        )
        assert "leafSample(dynGrid, dynGrid_metadata" not in cuda_code
        assert "leafImage(dynImages, dynImages_metadata" not in cuda_code
        assert "leafCounter(dynCounters, dynCounters_metadata" not in cuda_code
        assert "texture(" not in cuda_code
        assert "textureSize(" not in cuda_code
        assert "imageSize(" not in cuda_code
        assert "imageLoad(" not in cuda_code
        assert "imageStore(" not in cuda_code

    def test_mixed_fixed_dynamic_resource_arrays_emit_cuda_metadata_arguments(self):
        """Test CUDA forwards fixed rows and dynamic grids with metadata sidecars."""
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

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "CglResourceQueryInfo textureGrid_metadata[2][3] = {};" in cuda_code
        assert "CglResourceQueryInfo imageGrid_metadata[2][3] = {};" in cuda_code
        assert (
            "__device__ void leafMixed(texture<float4, 2> (*dynamicGrid)[3], "
            "CglResourceQueryInfo (*dynamicGrid_metadata)[3], "
            "texture<float4, 2> fixedRow[3], "
            "CglResourceQueryInfo fixedRow_metadata[3], "
            "cudaSurfaceObject_t (*dynamicImages)[3], "
            "CglResourceQueryInfo (*dynamicImages_metadata)[3], "
            "cudaSurfaceObject_t fixedImages[3], "
            "CglResourceQueryInfo fixedImages_metadata[3], "
            "int layer, int slot, float2 uv, int2 pixel)" in cuda_code
        )
        assert (
            "leafMixed(dynamicGrid, dynamicGrid_metadata, fixedRow, "
            "fixedRow_metadata, dynamicImages, dynamicImages_metadata, "
            "fixedImages, fixedImages_metadata, layer, slot, uv, pixel);" in cuda_code
        )
        assert (
            "forwardMixed(textureGrid, textureGrid_metadata, textureGrid[1], "
            "textureGrid_metadata[1], imageGrid, imageGrid_metadata, imageGrid[1], "
            "imageGrid_metadata[1], 1, 2, uv, pixel);" in cuda_code
        )
        assert (
            "float4 dynamicSample = tex2D(dynamicGrid[layer][slot], uv);" in cuda_code
        )
        assert "float4 fixedSample = tex2D(fixedRow[slot], uv);" in cuda_code
        assert (
            "float4 dynamicColor = surf2Dread<float4>"
            "(dynamicImages[layer][slot], pixel.x * sizeof(float4), pixel.y);"
            in cuda_code
        )
        assert (
            "float4 fixedColor = surf2Dread<float4>"
            "(fixedImages[slot], pixel.x * sizeof(float4), pixel.y);" in cuda_code
        )
        assert (
            "int2 dynamicSize = cgl_textureSize_sampler2D"
            "(dynamicGrid_metadata[layer][slot], 0);" in cuda_code
        )
        assert (
            "int2 fixedSize = cgl_textureSize_sampler2D"
            "(fixedRow_metadata[slot], 0);" in cuda_code
        )
        assert (
            "int2 dynamicImageSize = cgl_imageSize_image2D"
            "(dynamicImages_metadata[layer][slot]);" in cuda_code
        )
        assert (
            "int2 fixedImageSize = cgl_imageSize_image2D"
            "(fixedImages_metadata[slot]);" in cuda_code
        )
        assert "texture(" not in cuda_code
        assert "textureSize(" not in cuda_code
        assert "imageSize(" not in cuda_code
        assert "imageLoad(" not in cuda_code
        assert "imageStore(" not in cuda_code

    def test_multihop_reordered_resource_arrays_emit_cuda_metadata_arguments(self):
        """Test CUDA forwards reordered rows/grids and duplicate metadata sidecars."""
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

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "CglResourceQueryInfo textureGrid_metadata[2][3] = {};" in cuda_code
        assert "CglResourceQueryInfo imageGrid_metadata[2][3] = {};" in cuda_code
        assert (
            "__device__ void leafShuffle(texture<float4, 2> (*dynA)[3], "
            "CglResourceQueryInfo (*dynA_metadata)[3], "
            "cudaSurfaceObject_t fixedImageRow[3], "
            "CglResourceQueryInfo fixedImageRow_metadata[3], "
            "texture<float4, 2> fixedTexRow[3], "
            "CglResourceQueryInfo fixedTexRow_metadata[3], "
            "cudaSurfaceObject_t (*dynImageGrid)[3], "
            "CglResourceQueryInfo (*dynImageGrid_metadata)[3], "
            "texture<float4, 2> (*dynAlias)[3], "
            "CglResourceQueryInfo (*dynAlias_metadata)[3], "
            "int layer, int slot, float2 uv, int2 pixel)" in cuda_code
        )
        assert (
            "__device__ void midForward(texture<float4, 2> (*firstDyn)[3], "
            "CglResourceQueryInfo (*firstDyn_metadata)[3], "
            "texture<float4, 2> firstFixed[3], "
            "CglResourceQueryInfo firstFixed_metadata[3], "
            "cudaSurfaceObject_t (*firstDynImages)[3], "
            "CglResourceQueryInfo (*firstDynImages_metadata)[3], "
            "cudaSurfaceObject_t firstFixedImages[3], "
            "CglResourceQueryInfo firstFixedImages_metadata[3], "
            "int layer, int slot, float2 uv, int2 pixel)" in cuda_code
        )
        assert (
            "__device__ void topForward(texture<float4, 2> topFixedTex[3], "
            "CglResourceQueryInfo topFixedTex_metadata[3], "
            "cudaSurfaceObject_t (*topDynImages)[3], "
            "CglResourceQueryInfo (*topDynImages_metadata)[3], "
            "texture<float4, 2> (*topDynTex)[3], "
            "CglResourceQueryInfo (*topDynTex_metadata)[3], "
            "cudaSurfaceObject_t topFixedImages[3], "
            "CglResourceQueryInfo topFixedImages_metadata[3], "
            "int layer, int slot, float2 uv, int2 pixel)" in cuda_code
        )
        assert (
            "leafShuffle(firstDyn, firstDyn_metadata, firstFixedImages, "
            "firstFixedImages_metadata, firstFixed, firstFixed_metadata, "
            "firstDynImages, firstDynImages_metadata, firstDyn, "
            "firstDyn_metadata, layer, slot, uv, pixel);" in cuda_code
        )
        assert (
            "midForward(topDynTex, topDynTex_metadata, topFixedTex, "
            "topFixedTex_metadata, topDynImages, topDynImages_metadata, "
            "topFixedImages, topFixedImages_metadata, layer, slot, uv, pixel);"
            in cuda_code
        )
        assert (
            "topForward(textureGrid[1], textureGrid_metadata[1], imageGrid, "
            "imageGrid_metadata, textureGrid, textureGrid_metadata, imageGrid[1], "
            "imageGrid_metadata[1], 1, 2, uv, pixel);" in cuda_code
        )
        assert "float4 sampledA = tex2D(dynA[layer][slot], uv);" in cuda_code
        assert "float4 sampledFixed = tex2D(fixedTexRow[slot], uv);" in cuda_code
        assert "float4 sampledAlias = tex2D(dynAlias[layer][slot], uv);" in cuda_code
        assert (
            "float4 dynamicColor = surf2Dread<float4>"
            "(dynImageGrid[layer][slot], pixel.x * sizeof(float4), pixel.y);"
            in cuda_code
        )
        assert (
            "float4 fixedColor = surf2Dread<float4>"
            "(fixedImageRow[slot], pixel.x * sizeof(float4), pixel.y);" in cuda_code
        )
        assert (
            "int2 dynSize = cgl_textureSize_sampler2D"
            "(dynA_metadata[layer][slot], 0);" in cuda_code
        )
        assert (
            "int2 fixedTexSize = cgl_textureSize_sampler2D"
            "(fixedTexRow_metadata[slot], 0);" in cuda_code
        )
        assert (
            "int2 aliasSize = cgl_textureSize_sampler2D"
            "(dynAlias_metadata[layer][slot], 0);" in cuda_code
        )
        assert (
            "int2 dynamicImageSize = cgl_imageSize_image2D"
            "(dynImageGrid_metadata[layer][slot]);" in cuda_code
        )
        assert (
            "int2 fixedImageSize = cgl_imageSize_image2D"
            "(fixedImageRow_metadata[slot]);" in cuda_code
        )
        assert "texture(" not in cuda_code
        assert "textureSize(" not in cuda_code
        assert "imageSize(" not in cuda_code
        assert "imageLoad(" not in cuda_code
        assert "imageStore(" not in cuda_code

    def test_resource_calls_in_control_flow_emit_cuda_metadata_arguments(self):
        """Test CUDA forwards metadata in assignments, returns, and ternaries."""
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

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert (
            "__device__ float4 chooseBranch(texture<float4, 2> (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "texture<float4, 2> fixedTex[3], "
            "CglResourceQueryInfo fixedTex_metadata[3], "
            "cudaSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], "
            "cudaSurfaceObject_t fixedImages[3], "
            "CglResourceQueryInfo fixedImages_metadata[3], "
            "bool useFixed, int layer, int slot, float2 uv, int2 pixel)" in cuda_code
        )
        assert (
            "float4 result = sampleDynamic(dynTex, dynTex_metadata, "
            "dynImages, dynImages_metadata, layer, slot, uv, pixel);" in cuda_code
        )
        assert (
            "result = sampleFixed(fixedTex, fixedTex_metadata, fixedImages, "
            "fixedImages_metadata, slot, uv, pixel);" in cuda_code
        )
        assert (
            "return sampleFixed(fixedTex, fixedTex_metadata, fixedImages, "
            "fixedImages_metadata, slot, uv, pixel);" in cuda_code
        )
        assert (
            "return sampleDynamic(dynTex, dynTex_metadata, dynImages, "
            "dynImages_metadata, layer, slot, uv, pixel);" in cuda_code
        )
        assert (
            "return (useFixed ? sampleFixed(fixedTex, fixedTex_metadata, "
            "fixedImages, fixedImages_metadata, slot, uv, pixel) : "
            "sampleDynamic(dynTex, dynTex_metadata, dynImages, "
            "dynImages_metadata, layer, slot, uv, pixel));" in cuda_code
        )
        assert (
            "chooseBranch(textureGrid, textureGrid_metadata, textureGrid[1], "
            "textureGrid_metadata[1], imageGrid, imageGrid_metadata, imageGrid[1], "
            "imageGrid_metadata[1], true, 1, 2, uv, pixel);" in cuda_code
        )
        assert (
            "chooseTernary(textureGrid, textureGrid_metadata, textureGrid[1], "
            "textureGrid_metadata[1], imageGrid, imageGrid_metadata, imageGrid[1], "
            "imageGrid_metadata[1], true, 1, 2, uv, pixel);" in cuda_code
        )
        assert (
            "int2 texSize = cgl_textureSize_sampler2D"
            "(dynTex_metadata[layer][slot], 0);" in cuda_code
        )
        assert (
            "int2 imageSizeValue = cgl_imageSize_image2D"
            "(dynImages_metadata[layer][slot]);" in cuda_code
        )
        assert (
            "int2 texSize = cgl_textureSize_sampler2D(fixedTex_metadata[slot], 0);"
            in cuda_code
        )
        assert (
            "int2 imageSizeValue = cgl_imageSize_image2D(fixedImages_metadata[slot]);"
            in cuda_code
        )
        assert "texture(" not in cuda_code
        assert "textureSize(" not in cuda_code
        assert "imageSize(" not in cuda_code
        assert "imageLoad(" not in cuda_code

    def test_resource_calls_in_loops_emit_cuda_metadata_arguments(self):
        """Test CUDA forwards metadata through loop-carried resource indices."""
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

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert (
            "__device__ float4 sampleLoop(texture<float4, 2> (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "texture<float4, 2> fixedTex[3], "
            "CglResourceQueryInfo fixedTex_metadata[3], "
            "cudaSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], "
            "cudaSurfaceObject_t fixedImages[3], "
            "CglResourceQueryInfo fixedImages_metadata[3], "
            "int layer, float2 uv, int2 pixel)" in cuda_code
        )
        assert "for (int slot = 0; (slot < 3); slot++)" in cuda_code
        assert "continue;" in cuda_code
        assert "break;" in cuda_code
        assert "while ((fixedSlot < 3))" in cuda_code
        assert "float4 sampled = tex2D(dynTex[layer][slot], uv);" in cuda_code
        assert (
            "float4 loaded = surf2Dread<float4>"
            "(dynImages[layer][slot], pixel.x * sizeof(float4), pixel.y);" in cuda_code
        )
        assert (
            "int2 texSize = cgl_textureSize_sampler2D"
            "(dynTex_metadata[layer][slot], 0);" in cuda_code
        )
        assert (
            "int2 imageSizeValue = cgl_imageSize_image2D"
            "(dynImages_metadata[layer][slot]);" in cuda_code
        )
        assert "__device__ inline float4 cgl_float4_add(float4 lhs, float4 rhs)" in cuda_code
        assert (
            "accum = cgl_float4_add(accum, tex2D(fixedTex[fixedSlot], uv));"
            in cuda_code
        )
        assert (
            "accum = cgl_float4_add(accum, surf2Dread<float4>"
            "(fixedImages[fixedSlot], pixel.x * sizeof(float4), pixel.y));"
            in cuda_code
        )
        assert (
            "int2 fixedTexSize = cgl_textureSize_sampler2D"
            "(fixedTex_metadata[fixedSlot], 0);" in cuda_code
        )
        assert (
            "int2 fixedImageSize = cgl_imageSize_image2D"
            "(fixedImages_metadata[fixedSlot]);" in cuda_code
        )
        assert (
            "sampleLoop(textureGrid, textureGrid_metadata, textureGrid[1], "
            "textureGrid_metadata[1], imageGrid, imageGrid_metadata, imageGrid[1], "
            "imageGrid_metadata[1], 1, uv, pixel);" in cuda_code
        )
        assert "texture(" not in cuda_code
        assert "textureSize(" not in cuda_code
        assert "imageSize(" not in cuda_code
        assert "imageLoad(" not in cuda_code

    def test_nested_loop_resource_indices_emit_cuda_metadata_arguments(self):
        """Test CUDA forwards metadata when both resource indices are loop-carried."""
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

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert (
            "__device__ float4 sampleNestedLoops("
            "texture<float4, 2> (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "cudaSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], float2 uv, int2 pixel)"
            in cuda_code
        )
        assert "for (int layer = 0; (layer < 2); layer++)" in cuda_code
        assert "for (int slot = 0; (slot < 3); slot++)" in cuda_code
        assert "break;" in cuda_code
        assert "continue;" in cuda_code
        assert "float4 sampled = tex2D(dynTex[layer][slot], uv);" in cuda_code
        assert (
            "float4 loaded = surf2Dread<float4>"
            "(dynImages[layer][slot], pixel.x * sizeof(float4), pixel.y);" in cuda_code
        )
        assert (
            "surf2Dwrite(loaded, dynImages[layer][slot], "
            "pixel.x * sizeof(float4), pixel.y);" in cuda_code
        )
        assert (
            "int2 texSize = cgl_textureSize_sampler2D"
            "(dynTex_metadata[layer][slot], 0);" in cuda_code
        )
        assert (
            "int2 imageSizeValue = cgl_imageSize_image2D"
            "(dynImages_metadata[layer][slot]);" in cuda_code
        )
        assert (
            "sampleNestedLoops(textureGrid, textureGrid_metadata, imageGrid, "
            "imageGrid_metadata, uv, pixel);" in cuda_code
        )
        assert "texture(" not in cuda_code
        assert "textureSize(" not in cuda_code
        assert "imageSize(" not in cuda_code
        assert "imageLoad(" not in cuda_code
        assert "imageStore(" not in cuda_code

    def test_struct_and_scalar_params_preserve_cuda_metadata_argument_order(self):
        """Test CUDA keeps metadata aligned around non-resource parameters."""
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

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "struct SampleParams {" in cuda_code
        assert (
            "__device__ float4 samplePacked(texture<float4, 2> (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], SampleParams params, "
            "cudaSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], float weight, "
            "texture<float4, 2> fixedTex[3], "
            "CglResourceQueryInfo fixedTex_metadata[3], "
            "cudaSurfaceObject_t fixedImages[3], "
            "CglResourceQueryInfo fixedImages_metadata[3])" in cuda_code
        )
        assert (
            "__device__ float4 passPacked(SampleParams params, "
            "texture<float4, 2> (*firstDyn)[3], "
            "CglResourceQueryInfo (*firstDyn_metadata)[3], float weight, "
            "cudaSurfaceObject_t (*firstImages)[3], "
            "CglResourceQueryInfo (*firstImages_metadata)[3], "
            "texture<float4, 2> fixedTex[3], "
            "CglResourceQueryInfo fixedTex_metadata[3], "
            "cudaSurfaceObject_t fixedImages[3], "
            "CglResourceQueryInfo fixedImages_metadata[3])" in cuda_code
        )
        assert (
            "return samplePacked(firstDyn, firstDyn_metadata, params, firstImages, "
            "firstImages_metadata, weight, fixedTex, fixedTex_metadata, fixedImages, "
            "fixedImages_metadata);" in cuda_code
        )
        assert (
            "passPacked(params, textureGrid, textureGrid_metadata, 0.5, imageGrid, "
            "imageGrid_metadata, textureGrid[1], textureGrid_metadata[1], "
            "imageGrid[1], imageGrid_metadata[1]);" in cuda_code
        )
        assert (
            "float4 dynamicSample = tex2D"
            "(dynTex[params.layer][params.slot], params.uv);" in cuda_code
        )
        assert (
            "float4 fixedSample = tex2D(fixedTex[params.slot], params.uv);" in cuda_code
        )
        assert (
            "float4 dynamicColor = surf2Dread<float4>"
            "(dynImages[params.layer][params.slot], "
            "params.pixel.x * sizeof(float4), params.pixel.y);" in cuda_code
        )
        assert (
            "float4 fixedColor = surf2Dread<float4>"
            "(fixedImages[params.slot], params.pixel.x * sizeof(float4), "
            "params.pixel.y);" in cuda_code
        )
        assert (
            "int2 dynSize = cgl_textureSize_sampler2D"
            "(dynTex_metadata[params.layer][params.slot], 0);" in cuda_code
        )
        assert (
            "int2 fixedSize = cgl_textureSize_sampler2D"
            "(fixedTex_metadata[params.slot], 0);" in cuda_code
        )
        assert (
            "int2 dynImageSize = cgl_imageSize_image2D"
            "(dynImages_metadata[params.layer][params.slot]);" in cuda_code
        )
        assert (
            "int2 fixedImageSize = cgl_imageSize_image2D"
            "(fixedImages_metadata[params.slot]);" in cuda_code
        )
        assert "texture(" not in cuda_code
        assert "textureSize(" not in cuda_code
        assert "imageSize(" not in cuda_code
        assert "imageLoad(" not in cuda_code

    def test_resource_values_in_struct_returns_emit_cuda_metadata_arguments(self):
        """Test CUDA forwards metadata through struct assignment and constructors."""
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

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "struct SampleResult {" in cuda_code
        assert (
            "__device__ SampleResult buildAssigned("
            "texture<float4, 2> (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "cudaSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], int layer, int slot, "
            "float2 uv, int2 pixel)" in cuda_code
        )
        assert (
            "__device__ SampleResult buildConstructed("
            "texture<float4, 2> (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "cudaSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], int layer, int slot, "
            "float2 uv, int2 pixel)" in cuda_code
        )
        assert (
            "__device__ float4 consumeResults(texture<float4, 2> (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "cudaSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], int layer, int slot, "
            "float2 uv, int2 pixel)" in cuda_code
        )
        assert "result.sampled = tex2D(dynTex[layer][slot], uv);" in cuda_code
        assert (
            "result.loaded = surf2Dread<float4>"
            "(dynImages[layer][slot], pixel.x * sizeof(float4), pixel.y);" in cuda_code
        )
        assert (
            "result.texSize = cgl_textureSize_sampler2D"
            "(dynTex_metadata[layer][slot], 0);" in cuda_code
        )
        assert (
            "result.imageSizeValue = cgl_imageSize_image2D"
            "(dynImages_metadata[layer][slot]);" in cuda_code
        )
        assert (
            "return SampleResult(tex2D(dynTex[layer][slot], uv), "
            "surf2Dread<float4>(dynImages[layer][slot], "
            "pixel.x * sizeof(float4), pixel.y), "
            "cgl_textureSize_sampler2D(dynTex_metadata[layer][slot], 0), "
            "cgl_imageSize_image2D(dynImages_metadata[layer][slot]));" in cuda_code
        )
        assert (
            "SampleResult assigned = buildAssigned(dynTex, dynTex_metadata, "
            "dynImages, dynImages_metadata, layer, slot, uv, pixel);" in cuda_code
        )
        assert (
            "SampleResult constructed = buildConstructed(dynTex, dynTex_metadata, "
            "dynImages, dynImages_metadata, layer, slot, uv, pixel);" in cuda_code
        )
        assert (
            "consumeResults(textureGrid, textureGrid_metadata, imageGrid, "
            "imageGrid_metadata, 1, 2, uv, pixel);" in cuda_code
        )
        assert "texture(" not in cuda_code
        assert "textureSize(" not in cuda_code
        assert "imageSize(" not in cuda_code
        assert "imageLoad(" not in cuda_code

    def test_nested_struct_resource_assignments_emit_cuda_metadata_arguments(self):
        """Test CUDA handles resource calls assigned into nested struct members."""
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

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "struct SampleEnvelope {" in cuda_code
        assert (
            "__device__ SampleEnvelope buildNestedAssigned("
            "texture<float4, 2> (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "cudaSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], int layer, int slot, "
            "float2 uv, int2 pixel)" in cuda_code
        )
        assert "envelope.result.sampled = tex2D(dynTex[layer][slot], uv);" in cuda_code
        assert (
            "envelope.result.loaded = surf2Dread<float4>"
            "(dynImages[layer][slot], pixel.x * sizeof(float4), pixel.y);" in cuda_code
        )
        assert (
            "envelope.result.texSize = cgl_textureSize_sampler2D"
            "(dynTex_metadata[layer][slot], 0);" in cuda_code
        )
        assert (
            "envelope.result.imageSizeValue = cgl_imageSize_image2D"
            "(dynImages_metadata[layer][slot]);" in cuda_code
        )
        assert "__device__ inline float4 cgl_float4_add(float4 lhs, float4 rhs)" in cuda_code
        assert (
            "envelope.bias = cgl_float4_add(tex2D(dynTex[layer][slot], uv), "
            "surf2Dread<float4>(dynImages[layer][slot], "
            "pixel.x * sizeof(float4), pixel.y));" in cuda_code
        )
        assert (
            "SampleEnvelope envelope = buildNestedAssigned("
            "dynTex, dynTex_metadata, dynImages, dynImages_metadata, "
            "layer, slot, uv, pixel);" in cuda_code
        )
        assert (
            "consumeNested(textureGrid, textureGrid_metadata, imageGrid, "
            "imageGrid_metadata, 1, 2, uv, pixel);" in cuda_code
        )
        assert "texture(" not in cuda_code
        assert "textureSize(" not in cuda_code
        assert "imageSize(" not in cuda_code
        assert "imageLoad(" not in cuda_code

    def test_struct_member_array_resource_assignments_emit_cuda_metadata_arguments(self):
        """Test CUDA handles resource calls assigned into struct member arrays."""
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

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "SampleResult results[3];" in cuda_code
        assert (
            "__device__ SampleEnvelope buildArrayEnvelope("
            "texture<float4, 2> (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "cudaSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], int layer, int slot, "
            "float2 uv, int2 pixel)"
            in cuda_code
        )
        assert (
            "envelope.results[slot].sampled = tex2D(dynTex[layer][slot], uv);"
            in cuda_code
        )
        assert (
            "envelope.results[slot].loaded = surf2Dread<float4>"
            "(dynImages[layer][slot], pixel.x * sizeof(float4), pixel.y);"
            in cuda_code
        )
        assert (
            "envelope.results[slot].texSize = cgl_textureSize_sampler2D"
            "(dynTex_metadata[layer][slot], 0);"
            in cuda_code
        )
        assert (
            "envelope.results[slot].imageSizeValue = cgl_imageSize_image2D"
            "(dynImages_metadata[layer][slot]);"
            in cuda_code
        )
        assert (
            "SampleEnvelope envelope = buildArrayEnvelope("
            "dynTex, dynTex_metadata, dynImages, dynImages_metadata, "
            "layer, slot, uv, pixel);"
            in cuda_code
        )
        assert (
            "consumeArrayEnvelope(textureGrid, textureGrid_metadata, imageGrid, "
            "imageGrid_metadata, 1, 2, uv, pixel);"
            in cuda_code
        )
        assert "texture(" not in cuda_code
        assert "textureSize(" not in cuda_code
        assert "imageSize(" not in cuda_code
        assert "imageLoad(" not in cuda_code

    def test_control_flow_struct_resource_assignments_emit_cuda_metadata_arguments(self):
        """Test CUDA keeps metadata aligned through branch and loop struct writes."""
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

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert (
            "__device__ SampleEnvelope fillControlled("
            "texture<float4, 2> (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "cudaSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], int layer, int slot, "
            "bool useAlternate, float2 uv, int2 pixel)"
            in cuda_code
        )
        assert "if (useAlternate)" in cuda_code
        assert "else {" in cuda_code
        assert "for (int i = 0; (i < 3); i++)" in cuda_code
        assert "continue;" in cuda_code
        assert "break;" in cuda_code
        assert "envelope.accum = make_float4(0.0, 0.0, 0.0, 0.0);" in cuda_code
        assert "envelope.chosen = slot;" in cuda_code
        assert (
            "envelope.result.sampled = tex2D(dynTex[layer][slot], uv);"
            in cuda_code
        )
        assert (
            "envelope.result.loaded = surf2Dread<float4>"
            "(dynImages[layer][slot], pixel.x * sizeof(float4), pixel.y);"
            in cuda_code
        )
        assert (
            "envelope.result.sampled = tex2D(dynTex[fallback][slot], uv);"
            in cuda_code
        )
        assert (
            "envelope.result.loaded = surf2Dread<float4>"
            "(dynImages[fallback][slot], pixel.x * sizeof(float4), pixel.y);"
            in cuda_code
        )
        assert (
            "envelope.result.texSize = cgl_textureSize_sampler2D"
            "(dynTex_metadata[layer][i], 0);"
            in cuda_code
        )
        assert (
            "envelope.result.imageSizeValue = cgl_imageSize_image2D"
            "(dynImages_metadata[layer][i]);"
            in cuda_code
        )
        assert (
            "envelope.accum = cgl_float4_add(envelope.accum, tex2D(dynTex[layer][i], uv));"
            in cuda_code
        )
        assert (
            "fillControlled(dynTex, dynTex_metadata, dynImages, "
            "dynImages_metadata, layer, slot, true, uv, pixel);"
            in cuda_code
        )
        assert (
            "consumeControlled(textureGrid, textureGrid_metadata, imageGrid, "
            "imageGrid_metadata, 1, 2, uv, pixel);"
            in cuda_code
        )
        assert "texture(" not in cuda_code
        assert "textureSize(" not in cuda_code
        assert "imageSize(" not in cuda_code
        assert "imageLoad(" not in cuda_code

    def test_struct_parameter_resource_values_round_trip_cuda_metadata_arguments(self):
        """Test CUDA forwards resource-derived structs through helper parameters."""
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

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert (
            "__device__ SampleResult buildResourceResult("
            "texture<float4, 2> (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "cudaSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], int layer, int slot, "
            "float2 uv, int2 pixel)"
            in cuda_code
        )
        assert "__device__ SampleResult adjustResult(SampleResult payload, float weight)" in cuda_code
        assert (
            "__device__ float4 consumeAdjusted(texture<float4, 2> (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "cudaSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], int layer, int slot, "
            "float2 uv, int2 pixel)"
            in cuda_code
        )
        assert "__device__ inline float4 cgl_float4_mul_scalar" in cuda_code
        assert "__device__ inline float4 cgl_float4_add(float4 lhs, float4 rhs)" in cuda_code
        assert "payload.sampled = cgl_float4_mul_scalar(payload.sampled, weight);" in cuda_code
        assert "payload.loaded = cgl_float4_add(payload.loaded, payload.sampled);" in cuda_code
        assert "return payload;" in cuda_code
        assert (
            "SampleResult raw = buildResourceResult("
            "dynTex, dynTex_metadata, dynImages, dynImages_metadata, "
            "layer, slot, uv, pixel);"
            in cuda_code
        )
        assert "SampleResult adjusted = adjustResult(raw, 0.5);" in cuda_code
        assert (
            "consumeAdjusted(textureGrid, textureGrid_metadata, imageGrid, "
            "imageGrid_metadata, 1, 2, uv, pixel);"
            in cuda_code
        )
        assert "texture(" not in cuda_code
        assert "textureSize(" not in cuda_code
        assert "imageSize(" not in cuda_code
        assert "imageLoad(" not in cuda_code

    def test_mutable_scalar_and_array_params_emit_cuda_updates(self):
        """Test CUDA emits scalar and array parameter updates directly."""
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

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "__device__ float bumpScalar(float weight)" in cuda_code
        assert "weight = (weight + 1.0);" in cuda_code
        assert "return (weight * 2.0);" in cuda_code
        assert "__device__ float bumpArray(float values[3], int slot)" in cuda_code
        assert "values[slot] = (values[slot] + 1.0);" in cuda_code
        assert "values[0] = (values[slot] * 0.5);" in cuda_code
        assert "return (values[slot] + values[0]);" in cuda_code
        assert "float values[3];" in cuda_code
        assert "float a = bumpScalar(0.5);" in cuda_code
        assert "float b = bumpArray(values, 1);" in cuda_code

    def test_nested_array_and_struct_member_params_emit_cuda_updates(self):
        """Test CUDA emits nested array and struct-member array parameter updates."""
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

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "struct Payload {" in cuda_code
        assert "float values[3];" in cuda_code
        assert "__device__ float bumpNested(float values[2][3], int row, int col)" in cuda_code
        assert "values[row][col] = (values[row][col] + 1.0);" in cuda_code
        assert "values[0][col] = (values[row][col] * 0.5);" in cuda_code
        assert "return (values[row][col] + values[0][col]);" in cuda_code
        assert "__device__ float bumpPayload(Payload payload, int slot)" in cuda_code
        assert (
            "payload.values[slot] = (payload.values[slot] + payload.bias);"
            in cuda_code
        )
        assert "payload.bias = (payload.values[slot] * 0.5);" in cuda_code
        assert "return (payload.values[slot] + payload.bias);" in cuda_code
        assert "float grid[2][3];" in cuda_code
        assert "float a = bumpNested(grid, 1, 2);" in cuda_code
        assert "float b = bumpPayload(payload, 1);" in cuda_code

    def test_returned_nested_struct_params_emit_cuda_updates(self):
        """Test CUDA mutates nested struct params built from returned values."""
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

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "__device__ OuterPayload adjustOuter(OuterPayload payload, int slot)" in cuda_code
        assert (
            "payload.inner.values[slot] = "
            "(payload.inner.values[slot] + payload.inner.bias);"
            in cuda_code
        )
        assert (
            "payload.inner.values[0] = (payload.inner.values[slot] * payload.scale);"
            in cuda_code
        )
        assert "payload.scale = (payload.inner.values[0] + payload.inner.bias);" in cuda_code
        assert "return payload;" in cuda_code
        assert "__device__ float consumeAdjustedOuter(int slot)" in cuda_code
        assert (
            "OuterPayload adjusted = adjustOuter("
            "makeOuter(1.0, 2.0, 0.25, 4.0), slot);"
            in cuda_code
        )
        assert (
            "return ((adjusted.inner.values[slot] + adjusted.inner.values[0]) + "
            "adjusted.scale);"
            in cuda_code
        )
        assert "float value = consumeAdjustedOuter(1);" in cuda_code

    def test_conditional_returned_nested_structs_emit_cuda_expressions(self):
        """Test CUDA preserves branch and ternary selected returned structs."""
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

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "__device__ OuterPayload chooseBranch(bool useSecond)" in cuda_code
        assert "return makeOuter(3.0, 4.0, 0.5, 5.0);" in cuda_code
        assert "return makeOuter(1.0, 2.0, 0.25, 4.0);" in cuda_code
        assert "__device__ OuterPayload chooseTernary(bool useSecond)" in cuda_code
        assert (
            "return (useSecond ? makeOuter(3.0, 4.0, 0.5, 5.0) : "
            "makeOuter(1.0, 2.0, 0.25, 4.0));"
            in cuda_code
        )
        assert "OuterPayload branchPayload = chooseBranch(useSecond);" in cuda_code
        assert "OuterPayload ternaryPayload = chooseTernary(useSecond);" in cuda_code
        assert (
            "ternaryPayload.inner.values[slot] = "
            "(ternaryPayload.inner.values[slot] + ternaryPayload.inner.bias);"
            in cuda_code
        )
        assert (
            "branchPayload.scale = "
            "(branchPayload.inner.values[slot] + ternaryPayload.scale);"
            in cuda_code
        )
        assert (
            "return (branchPayload.scale + ternaryPayload.inner.values[slot]);"
            in cuda_code
        )
        assert "float value = consumeConditional(1, true);" in cuda_code

    def test_temporary_struct_array_member_reads_emit_cuda_expressions(self):
        """Test CUDA emits array-field reads from returned struct temporaries."""
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

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "__device__ Payload makePayload(float first, float second, float bias)" in cuda_code
        assert "payload.values[2] = (first + second);" in cuda_code
        assert "__device__ float readTemporary(int slot)" in cuda_code
        assert (
            "float dynamicValue = makePayload(1.0, 2.0, 0.25).values[slot];"
            in cuda_code
        )
        assert "float fixedValue = makePayload(3.0, 4.0, 0.5).values[0];" in cuda_code
        assert "float biasValue = makePayload(5.0, 6.0, 0.75).bias;" in cuda_code
        assert "return ((dynamicValue + fixedValue) + biasValue);" in cuda_code
        assert "float value = readTemporary(1);" in cuda_code

    def test_nested_temporary_struct_array_member_reads_emit_cuda_expressions(self):
        """Test CUDA emits nested array-field reads from returned struct temporaries."""
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

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "struct InnerPayload {" in cuda_code
        assert "struct OuterPayload {" in cuda_code
        assert (
            "__device__ OuterPayload makeOuter("
            "float first, float second, float bias, float scale)"
            in cuda_code
        )
        assert "outer.inner.values[2] = (first + second);" in cuda_code
        assert "outer.inner.bias = bias;" in cuda_code
        assert "outer.scale = scale;" in cuda_code
        assert "__device__ float readNestedTemporary(int slot)" in cuda_code
        assert (
            "float dynamicValue = "
            "makeOuter(1.0, 2.0, 0.25, 4.0).inner.values[slot];"
            in cuda_code
        )
        assert (
            "float fixedValue = makeOuter(3.0, 4.0, 0.5, 5.0).inner.values[0];"
            in cuda_code
        )
        assert (
            "float biasValue = makeOuter(5.0, 6.0, 0.75, 6.0).inner.bias;"
            in cuda_code
        )
        assert "float scaleValue = makeOuter(7.0, 8.0, 1.0, 9.0).scale;" in cuda_code
        assert "return (((dynamicValue + fixedValue) + biasValue) + scaleValue);" in cuda_code

    def test_returned_local_nested_struct_array_writes_emit_cuda_expressions(self):
        """Test CUDA mutates locals initialized from returned nested structs."""
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

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "OuterPayload outer = makeOuter(1.0, 2.0, 0.25, 4.0);" in cuda_code
        assert (
            "outer.inner.values[slot] = "
            "(outer.inner.values[slot] + outer.inner.bias);"
            in cuda_code
        )
        assert (
            "outer.inner.values[0] = (outer.inner.values[slot] * outer.scale);"
            in cuda_code
        )
        assert "outer.scale = (outer.inner.values[0] + outer.inner.bias);" in cuda_code
        assert (
            "return ((outer.inner.values[slot] + outer.inner.values[0]) + outer.scale);"
            in cuda_code
        )
        assert "float value = mutateReturnedLocal(1);" in cuda_code

    def test_returned_local_nested_struct_array_writes_in_control_flow_emit_cuda(
        self,
    ):
        """Test CUDA preserves returned nested struct mutations inside control flow."""
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

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "__device__ float mutateReturnedLocalControl(int slot)" in cuda_code
        assert "OuterPayload outer = makeOuter(1.0, 2.0, 0.25, 4.0);" in cuda_code
        assert "for (int i = 0; (i < 3); i++)" in cuda_code
        assert (
            "outer.inner.values[i] = "
            "(outer.inner.values[i] + outer.inner.bias);"
            in cuda_code
        )
        assert "if ((i == slot))" in cuda_code
        assert (
            "outer.inner.values[i] = (outer.inner.values[i] * outer.scale);"
            in cuda_code
        )
        assert "if ((slot > 1))" in cuda_code
        assert "outer.scale = (outer.inner.values[slot] + outer.inner.bias);" in cuda_code
        assert "outer.scale = (outer.inner.values[0] - outer.inner.bias);" in cuda_code
        assert "return (outer.inner.values[slot] + outer.scale);" in cuda_code
        assert "float value = mutateReturnedLocalControl(1);" in cuda_code

    def test_image_atomic_builtins_emit_cuda_diagnostics(self):
        """Test CUDA makes unsupported storage image atomics explicit."""
        source_code = """
        shader Resources {
            iimage2d signedImage;
            uimage2d counters;
            iimage3d signedVolume;
            uimage2darray counterLayers;
            image2D vectorCounters @rg32ui;

            void atomicResources(ivec2 pixel, ivec3 voxel, ivec3 pixelLayer) {
                int added = imageAtomicAdd(signedImage, pixel, 1);
                uint exchanged = imageAtomicExchange(counters, pixel, 2);
                int compared = imageAtomicCompSwap(signedImage, pixel, 3, 4);
                int volumeMin = imageAtomicMin(signedVolume, voxel, 5);
                uint layerMax = imageAtomicMax(counterLayers, pixelLayer, 6);
                uint vectorAtomic = imageAtomicAdd(vectorCounters, pixel, 7);
                int missingArgs = imageAtomicAdd(signedImage);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert (
            "int added = /* unsupported CUDA image atomic resource call: "
            "imageAtomicAdd on iimage2D */ 0;" in cuda_code
        )
        assert (
            "uint exchanged = /* unsupported CUDA image atomic resource call: "
            "imageAtomicExchange on uimage2D */ 0u;" in cuda_code
        )
        assert (
            "int compared = /* unsupported CUDA image atomic resource call: "
            "imageAtomicCompSwap on iimage2D */ 0;" in cuda_code
        )
        assert (
            "int volumeMin = /* unsupported CUDA image atomic resource call: "
            "imageAtomicMin on iimage3D */ 0;" in cuda_code
        )
        assert (
            "uint layerMax = /* unsupported CUDA image atomic resource call: "
            "imageAtomicMax on uimage2DArray */ 0u;" in cuda_code
        )
        assert (
            "uint vectorAtomic = /* unsupported CUDA image atomic resource call: "
            "imageAtomicAdd on image2D */ 0;" in cuda_code
        )
        assert (
            "int missingArgs = /* unsupported CUDA image atomic resource call: "
            "imageAtomicAdd on iimage2D */ 0;" in cuda_code
        )
        assert "imageAtomicAdd(" not in cuda_code
        assert "imageAtomicExchange(" not in cuda_code
        assert "imageAtomicCompSwap(" not in cuda_code
        assert "imageAtomicMin(" not in cuda_code
        assert "imageAtomicMax(" not in cuda_code

    def test_resource_query_builtins_emit_cuda_metadata_helpers(self):
        """Test CUDA lowers resource queries through explicit metadata sidecars."""
        source_code = """
        shader Resources {
            sampler2d colorMap;
            sampler2darray layers;
            sampler3d volumeTex;
            samplercube cubeTex;
            samplercubearray cubeArrayTex;
            sampler2dms msTex;
            sampler2dmsarray msLayers;
            image2D colorImage;
            image3D volumeImage;
            image2DArray layerImage;
            image2DMS msImage;
            image2DMSArray msImageLayers;
            uimage2DMS counters;

            void queryParam(sampler2d paramTex) {
                ivec2 paramSize = textureSize(paramTex, 0);
                int paramLevels = textureQueryLevels(paramTex);
            }

            compute {
                void main() {
                    ivec2 texSize = textureSize(colorMap, 2);
                    ivec3 layerSize = textureSize(layers, 1);
                    ivec3 volumeSize = textureSize(volumeTex, 0);
                    ivec2 cubeSize = textureSize(cubeTex, 0);
                    ivec3 cubeArraySize = textureSize(cubeArrayTex, 0);
                    int texLevels = textureQueryLevels(colorMap);
                    int layerLevels = textureQueryLevels(layers);
                    int cubeLevels = textureQueryLevels(cubeTex);
                    int cubeArrayLevels = textureQueryLevels(cubeArrayTex);
                    ivec2 msSize = textureSize(msTex, 0);
                    ivec3 msLayerSize = textureSize(msLayers, 0);
                    ivec2 imageSize2d = imageSize(colorImage);
                    ivec3 imageSize3d = imageSize(volumeImage);
                    ivec3 imageSizeLayer = imageSize(layerImage);
                    ivec2 msImageSize = imageSize(msImage);
                    ivec3 msImageLayerSize = imageSize(msImageLayers);
                    int texSamples = textureSamples(msTex);
                    int texLayerSamples = textureSamples(msLayers);
                    int imageSamplesValue = imageSamples(msImage);
                    int counterSamples = imageSamples(counters);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "struct CglResourceQueryInfo" in cuda_code
        assert "__device__ inline int cgl_lod_extent" in cuda_code
        assert (
            "__device__ void queryParam(texture<float4, 2> paramTex, "
            "CglResourceQueryInfo paramTex_metadata)" in cuda_code
        )
        assert "CglResourceQueryInfo colorMap_metadata = {};" in cuda_code
        assert "CglResourceQueryInfo cubeArrayTex_metadata = {};" in cuda_code
        assert "CglResourceQueryInfo counters_metadata = {};" in cuda_code
        assert (
            "return make_int3(cgl_lod_extent(info.width, mipLevel), "
            "cgl_lod_extent(info.height, mipLevel), info.elements);" in cuda_code
        )
        assert "return info.levels > 0 ? info.levels : 1;" in cuda_code
        assert (
            "int2 paramSize = cgl_textureSize_sampler2D(paramTex_metadata, 0);"
            in cuda_code
        )
        assert (
            "int paramLevels = cgl_textureQueryLevels_sampler2D(paramTex_metadata);"
            in cuda_code
        )
        assert (
            "int2 texSize = cgl_textureSize_sampler2D(colorMap_metadata, 2);"
            in cuda_code
        )
        assert (
            "int3 layerSize = cgl_textureSize_sampler2DArray(layers_metadata, 1);"
            in cuda_code
        )
        assert (
            "int3 volumeSize = cgl_textureSize_sampler3D(volumeTex_metadata, 0);"
            in cuda_code
        )
        assert (
            "int2 cubeSize = cgl_textureSize_samplerCube(cubeTex_metadata, 0);"
            in cuda_code
        )
        assert (
            "int3 cubeArraySize = cgl_textureSize_samplerCubeArray"
            "(cubeArrayTex_metadata, 0);" in cuda_code
        )
        assert (
            "int texLevels = cgl_textureQueryLevels_sampler2D(colorMap_metadata);"
            in cuda_code
        )
        assert (
            "int layerLevels = cgl_textureQueryLevels_sampler2DArray"
            "(layers_metadata);" in cuda_code
        )
        assert (
            "int cubeLevels = cgl_textureQueryLevels_samplerCube(cubeTex_metadata);"
            in cuda_code
        )
        assert (
            "int cubeArrayLevels = cgl_textureQueryLevels_samplerCubeArray"
            "(cubeArrayTex_metadata);" in cuda_code
        )
        assert "int2 msSize = cgl_textureSize_sampler2DMS(msTex_metadata);" in cuda_code
        assert (
            "int3 msLayerSize = cgl_textureSize_sampler2DMSArray"
            "(msLayers_metadata);" in cuda_code
        )
        assert (
            "int2 imageSize2d = cgl_imageSize_image2D(colorImage_metadata);"
            in cuda_code
        )
        assert (
            "int3 imageSize3d = cgl_imageSize_image3D(volumeImage_metadata);"
            in cuda_code
        )
        assert (
            "int3 imageSizeLayer = cgl_imageSize_image2DArray(layerImage_metadata);"
            in cuda_code
        )
        assert (
            "int2 msImageSize = cgl_imageSize_image2DMS(msImage_metadata);" in cuda_code
        )
        assert (
            "int3 msImageLayerSize = cgl_imageSize_image2DMSArray"
            "(msImageLayers_metadata);" in cuda_code
        )
        assert (
            "int texSamples = cgl_textureSamples_sampler2DMS(msTex_metadata);"
            in cuda_code
        )
        assert (
            "int texLayerSamples = cgl_textureSamples_sampler2DMSArray"
            "(msLayers_metadata);" in cuda_code
        )
        assert (
            "int imageSamplesValue = cgl_imageSamples_image2DMS(msImage_metadata);"
            in cuda_code
        )
        assert (
            "int counterSamples = cgl_imageSamples_uimage2DMS(counters_metadata);"
            in cuda_code
        )
        assert "textureSize(" not in cuda_code
        assert "imageSize(" not in cuda_code
        assert "textureSamples(" not in cuda_code
        assert "imageSamples(" not in cuda_code
        assert "textureQueryLevels(" not in cuda_code

    def test_texture_query_lod_emits_cuda_diagnostics(self):
        """Test CUDA makes unsupported textureQueryLod explicit."""
        source_code = """
        shader Resources {
            sampler2d colorMap;
            sampler2darray layers;
            samplercube cubeTex;
            sampler querySampler;

            void queryParam(sampler2d paramTex, vec2 uv) {
                vec2 paramLod = textureQueryLod(paramTex, uv);
            }

            compute {
                void main() {
                    vec2 uv = vec2(0.25, 0.75);
                    vec3 uvLayer = vec3(0.25, 0.75, 1.0);
                    vec3 direction = vec3(1.0, 0.0, 0.0);
                    vec2 lod = textureQueryLod(colorMap, querySampler, uv);
                    vec2 arrayLod = textureQueryLod(layers, querySampler, uvLayer);
                    vec2 cubeLod = textureQueryLod(cubeTex, direction);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert (
            "float2 paramLod = /* unsupported CUDA resource query: "
            "textureQueryLod on sampler2D */ make_float2(0.0f, 0.0f);" in cuda_code
        )
        assert (
            "float2 lod = /* unsupported CUDA resource query: "
            "textureQueryLod on sampler2D */ "
            "make_float2(0.0f, 0.0f);" in cuda_code
        )
        assert (
            "float2 arrayLod = /* unsupported CUDA resource query: "
            "textureQueryLod on sampler2DArray */ "
            "make_float2(0.0f, 0.0f);" in cuda_code
        )
        assert (
            "float2 cubeLod = /* unsupported CUDA resource query: "
            "textureQueryLod on samplerCube */ make_float2(0.0f, 0.0f);" in cuda_code
        )
        assert "CglResourceQueryInfo" not in cuda_code
        assert "textureQueryLod(" not in cuda_code

    def test_resource_query_arrays_emit_indexed_cuda_metadata(self):
        """Test CUDA resource queries preserve resource-array metadata indexing."""
        source_code = """
        shader Resources {
            sampler2d textures[4];
            sampler2dms msTextures[2];
            image2D images[3];

            void queryArrays(sampler2d paramTextures[3], int i) {
                ivec2 texSize = textureSize(textures[1 + 2], 0);
                int texLevels = textureQueryLevels(textures[1 + 2]);
                int msSamples = textureSamples(msTextures[i]);
                ivec2 imageSizeValue = imageSize(images[i]);
                ivec2 paramSize = textureSize(paramTextures[i], 0);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "CglResourceQueryInfo textures_metadata[4] = {};" in cuda_code
        assert "CglResourceQueryInfo msTextures_metadata[2] = {};" in cuda_code
        assert "CglResourceQueryInfo images_metadata[3] = {};" in cuda_code
        assert (
            "__device__ void queryArrays(texture<float4, 2> paramTextures[3], "
            "CglResourceQueryInfo paramTextures_metadata[3], int i)" in cuda_code
        )
        assert (
            "int2 texSize = cgl_textureSize_sampler2D"
            "(textures_metadata[(1 + 2)], 0);" in cuda_code
        )
        assert (
            "int texLevels = cgl_textureQueryLevels_sampler2D"
            "(textures_metadata[(1 + 2)]);" in cuda_code
        )
        assert (
            "int msSamples = cgl_textureSamples_sampler2DMS(msTextures_metadata[i]);"
            in cuda_code
        )
        assert (
            "int2 imageSizeValue = cgl_imageSize_image2D(images_metadata[i]);"
            in cuda_code
        )
        assert (
            "int2 paramSize = cgl_textureSize_sampler2D"
            "(paramTextures_metadata[i], 0);" in cuda_code
        )
        assert "cgl_textureSize_sampler2D(textures_metadata, 0)" not in cuda_code
        assert "cgl_textureSamples_sampler2DMS(msTextures_metadata)" not in cuda_code
        assert "cgl_imageSize_image2D(images_metadata)" not in cuda_code

    def test_dynamic_and_nested_resource_query_arrays_emit_cuda_metadata(self):
        """Test CUDA metadata sidecars cover dynamic and nested resource arrays."""
        source_code = """
        shader Resources {
            sampler2d textureGrid[2][3];
            image2D imageGrid[2][4];

            void queryArrays(
                sampler2d dynamicTextures[],
                sampler2d fixedTextures[3],
                sampler2dms dynamicMsTextures[],
                sampler2d dynamicGrid[][3],
                image2D dynamicImageGrid[][4],
                int layer,
                int slot
            ) {
                ivec2 dynamicSize = textureSize(dynamicTextures[slot], 0);
                int fixedLevels = textureQueryLevels(fixedTextures[slot]);
                int dynamicSamples = textureSamples(dynamicMsTextures[slot]);
                ivec2 dynamicGridSize = textureSize(dynamicGrid[layer][slot], 0);
                ivec2 dynamicImageSize = imageSize(dynamicImageGrid[layer][slot]);
                ivec2 gridSize = textureSize(textureGrid[layer][slot], 0);
                ivec2 imageGridSize = imageSize(imageGrid[layer][slot]);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "CglResourceQueryInfo textureGrid_metadata[2][3] = {};" in cuda_code
        assert "CglResourceQueryInfo imageGrid_metadata[2][4] = {};" in cuda_code
        assert (
            "__device__ void queryArrays(texture<float4, 2>* dynamicTextures, "
            "CglResourceQueryInfo* dynamicTextures_metadata, "
            "texture<float4, 2> fixedTextures[3], "
            "CglResourceQueryInfo fixedTextures_metadata[3], "
            "cudaTextureObject_t* dynamicMsTextures, "
            "CglResourceQueryInfo* dynamicMsTextures_metadata, "
            "texture<float4, 2> (*dynamicGrid)[3], "
            "CglResourceQueryInfo (*dynamicGrid_metadata)[3], "
            "cudaSurfaceObject_t (*dynamicImageGrid)[4], "
            "CglResourceQueryInfo (*dynamicImageGrid_metadata)[4], "
            "int layer, int slot)" in cuda_code
        )
        assert (
            "int2 dynamicSize = cgl_textureSize_sampler2D"
            "(dynamicTextures_metadata[slot], 0);" in cuda_code
        )
        assert (
            "int fixedLevels = cgl_textureQueryLevels_sampler2D"
            "(fixedTextures_metadata[slot]);" in cuda_code
        )
        assert (
            "int dynamicSamples = cgl_textureSamples_sampler2DMS"
            "(dynamicMsTextures_metadata[slot]);" in cuda_code
        )
        assert (
            "int2 dynamicGridSize = cgl_textureSize_sampler2D"
            "(dynamicGrid_metadata[layer][slot], 0);" in cuda_code
        )
        assert (
            "int2 dynamicImageSize = cgl_imageSize_image2D"
            "(dynamicImageGrid_metadata[layer][slot]);" in cuda_code
        )
        assert (
            "int2 gridSize = cgl_textureSize_sampler2D"
            "(textureGrid_metadata[layer][slot], 0);" in cuda_code
        )
        assert (
            "int2 imageGridSize = cgl_imageSize_image2D"
            "(imageGrid_metadata[layer][slot]);" in cuda_code
        )
        assert "cgl_textureSize_sampler2D(dynamicTextures_metadata, 0)" not in cuda_code
        assert (
            "cgl_textureQueryLevels_sampler2D(fixedTextures_metadata)" not in cuda_code
        )
        assert "cgl_textureSize_sampler2D(textureGrid_metadata, 0)" not in cuda_code
        assert "cgl_textureSize_sampler2D(dynamicGrid_metadata, 0)" not in cuda_code
        assert "cgl_imageSize_image2D(dynamicImageGrid_metadata)" not in cuda_code
        assert "cgl_imageSize_image2D(imageGrid_metadata)" not in cuda_code

    def test_multisample_resource_builtins_emit_cuda_diagnostics(self):
        """Test CUDA does not silently lower MS resource calls to non-MS functions."""
        source_code = """
        shader Resources {
            sampler2dms msTex;
            sampler2dmsarray msLayers;
            image2DMS msImage;
            image2DMSArray msImageLayers;
            uimage2DMS msCounters;
            iimage2DMSArray msSignedLayers;

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
                uint loadedCounter = imageLoad(msCounters, pixel, sampleIndex);
                imageStore(msCounters, pixel, sampleIndex, loadedCounter);
                int loadedSignedLayer = imageLoad(msSignedLayers, pixelLayer, sampleIndex);
                imageStore(msSignedLayers, pixelLayer, sampleIndex, loadedSignedLayer);
                uint msAtomic = imageAtomicAdd(msCounters, pixel, sampleIndex, 1);
                int msLayerAtomic = imageAtomicCompSwap(
                    msSignedLayers,
                    pixelLayer,
                    sampleIndex,
                    2,
                    3
                );
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "cudaTextureObject_t msTex;" in cuda_code
        assert "cudaTextureObject_t msLayers;" in cuda_code
        assert "cudaSurfaceObject_t msImage;" in cuda_code
        assert "cudaSurfaceObject_t msImageLayers;" in cuda_code
        assert "cudaSurfaceObject_t msCounters;" in cuda_code
        assert "cudaSurfaceObject_t msSignedLayers;" in cuda_code
        assert (
            "float4 sampled = /* unsupported CUDA multisample resource call: "
            "texture on sampler2DMS */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in cuda_code
        )
        assert (
            "float4 fetched = /* unsupported CUDA multisample resource call: "
            "texelFetch on sampler2DMS */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in cuda_code
        )
        assert (
            "float4 fetchedLayer = /* unsupported CUDA multisample resource call: "
            "texelFetch on sampler2DMSArray */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in cuda_code
        )
        assert (
            "float4 loaded = /* unsupported CUDA multisample resource call: "
            "imageLoad on image2DMS */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in cuda_code
        )
        assert (
            "/* unsupported CUDA multisample resource call: imageStore on image2DMS */ "
            "((void)0);" in cuda_code
        )
        assert (
            "float4 loadedLayer = /* unsupported CUDA multisample resource call: "
            "imageLoad on image2DMSArray */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in cuda_code
        )
        assert (
            "/* unsupported CUDA multisample resource call: imageStore on image2DMSArray */ "
            "((void)0);" in cuda_code
        )
        assert (
            "uint loadedCounter = /* unsupported CUDA multisample resource call: "
            "imageLoad on uimage2DMS */ 0u;" in cuda_code
        )
        assert (
            "int loadedSignedLayer = /* unsupported CUDA multisample resource call: "
            "imageLoad on iimage2DMSArray */ 0;" in cuda_code
        )
        assert "imageStore on uimage2DMS */ ((void)0);" in cuda_code
        assert "imageStore on iimage2DMSArray */ ((void)0);" in cuda_code
        assert (
            "uint msAtomic = /* unsupported CUDA image atomic resource call: "
            "imageAtomicAdd on uimage2DMS */ 0u;" in cuda_code
        )
        assert (
            "int msLayerAtomic = /* unsupported CUDA image atomic resource call: "
            "imageAtomicCompSwap on iimage2DMSArray */ 0;" in cuda_code
        )
        assert "imageAtomicAdd(" not in cuda_code
        assert "imageAtomicCompSwap(" not in cuda_code
        assert "texture(msTex" not in cuda_code
        assert "texelFetch(msTex" not in cuda_code
        assert "texelFetch(msLayers" not in cuda_code
        assert "imageLoad(msImage" not in cuda_code
        assert "imageLoad(msImageLayers" not in cuda_code
        assert "imageStore(msImage" not in cuda_code
        assert "imageStore(msImageLayers" not in cuda_code
        assert "tex2D(msTex" not in cuda_code
        assert "tex2DLayered<float4>(msLayers" not in cuda_code
        assert "surf2Dread" not in cuda_code
        assert "surf2DLayeredread" not in cuda_code
        assert "surf2Dwrite" not in cuda_code
        assert "surf2DLayeredwrite" not in cuda_code

    def test_empty_shader(self):
        """Test empty shader generation"""
        source_code = ""

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "#include <cuda_runtime.h>" in cuda_code
        assert "#include <device_launch_parameters.h>" in cuda_code

    def test_prefix_and_postfix_unary_operators_preserve_position(self):
        """Test CUDA preserves prefix/postfix increment and decrement operators."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    int i = 0;
                    i++;
                    ++i;
                    i--;
                    --i;
                    for (int j = 0; j < 2; j++) {
                        i++;
                    }
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "int i = 0;" in cuda_code
        assert "i++;" in cuda_code
        assert "++i;" in cuda_code
        assert "i--;" in cuda_code
        assert "--i;" in cuda_code
        assert "for (int j = 0; (j < 2); j++)" in cuda_code
        assert "for (None;" not in cuda_code
        assert "++i++" not in cuda_code

    def test_bool_string_and_char_literals_emit_cuda_syntax(self):
        """Test CUDA literal output uses target-language spelling."""
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

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "bool enabled = true;" in cuda_code
        assert "bool disabled = false;" in cuda_code
        assert 'string label = "debug";' in cuda_code
        assert "char marker = 'x';" in cuda_code
        assert 'label = "active";' in cuda_code
        assert "marker = 'y';" in cuda_code
        assert "True" not in cuda_code
        assert "False" not in cuda_code

    def test_direct_literal_nodes_emit_cuda_escaping(self):
        """Test direct CUDA literal formatting escapes quotes."""
        codegen = CudaCodeGen()

        assert codegen.visit(LiteralNode(True, PrimitiveType("bool"))) == "true"
        assert (
            codegen.visit(LiteralNode('debug"name', PrimitiveType("string")))
            == '"debug\\"name"'
        )
        assert codegen.visit(LiteralNode("'", PrimitiveType("char"))) == "'\\''"

    def test_while_loop_generation(self):
        """Test CUDA emits while loops for block and single-statement bodies."""
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

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "while ((value < 4)) {" in cuda_code
        assert "value += 1;" in cuda_code
        assert "while ((value < 8)) {" in cuda_code
        assert "value += 2;" in cuda_code
        assert "WhileNode" not in cuda_code

    def test_break_and_continue_generation(self):
        """Test CUDA emits break and continue statements in loop bodies."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    int value = 0;
                    while (value < 8) {
                        value += 1;
                        if (value == 2) {
                            continue;
                        }
                        if (value == 4) {
                            break;
                        }
                    }
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "while ((value < 8)) {" in cuda_code
        assert "continue;" in cuda_code
        assert "break;" in cuda_code
        assert "if ((value == 2)) {\n            continue;\n        }" in cuda_code
        assert "if ((value == 4)) {\n            break;\n        }" in cuda_code

    def test_switch_generation(self):
        """Test CUDA emits switch, case, default, and case bodies."""
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

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "switch (mode) {" in cuda_code
        assert "case 0:" in cuda_code
        assert "value = 10;" in cuda_code
        assert "case 1:" in cuda_code
        assert "value += 20;" in cuda_code
        assert "default:" in cuda_code
        assert "value = -1;" in cuda_code
        assert cuda_code.count("break;") == 2
        assert "SwitchNode" not in cuda_code

    def test_direct_ast_expression_statements_are_emitted(self):
        """Test direct CUDA AST function bodies emit expression-returning nodes."""
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
                                IdentifierNode("workgroupBarrier"),
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

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "int value = 0;" in cuda_code
        assert "__syncthreads();" in cuda_code
        assert "value += 1;" in cuda_code

    def test_array_declarations_emit_c_style_declarators(self):
        """Test CUDA array declarations place dimensions after the variable name."""
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

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "float weights[4];" in cuda_code
        assert "float3 colors[2];" in cuda_code
        assert (
            "void main(float input_weights[4], float3 input_colors[2], "
            "float* dynamic_weights)"
        ) in cuda_code
        assert "float data[256];" in cuda_code
        assert "float3 local_colors[2];" in cuda_code
        assert "float3* scratch;" in cuda_code
        assert "data[0] = input_weights[1];" in cuda_code
        assert "local_colors[0] = input_colors[1];" in cuda_code
        assert "data[1] = dynamic_weights[0];" in cuda_code
        assert "LiteralNode" not in cuda_code
        assert "float[256] data" not in cuda_code
        assert "float3[2] local_colors" not in cuda_code
