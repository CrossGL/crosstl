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

    def test_multisample_resource_builtins_emit_cuda_diagnostics(self):
        """Test CUDA does not silently lower MS resource calls to non-MS functions."""
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

        codegen = CudaCodeGen()
        cuda_code = codegen.generate(ast)

        assert "cudaTextureObject_t msTex;" in cuda_code
        assert "cudaTextureObject_t msLayers;" in cuda_code
        assert "cudaSurfaceObject_t msImage;" in cuda_code
        assert "cudaSurfaceObject_t msImageLayers;" in cuda_code
        assert (
            "/* unsupported CUDA multisample resource call: texture on sampler2DMS */ "
            "texture(msTex, uv)" in cuda_code
        )
        assert (
            "/* unsupported CUDA multisample resource call: texelFetch on sampler2DMS */ "
            "texelFetch(msTex, pixel, sampleIndex)" in cuda_code
        )
        assert (
            "/* unsupported CUDA multisample resource call: texelFetch on sampler2DMSArray */ "
            "texelFetch(msLayers, pixelLayer, sampleIndex)" in cuda_code
        )
        assert (
            "/* unsupported CUDA multisample resource call: imageLoad on image2DMS */ "
            "imageLoad(msImage, pixel, sampleIndex)" in cuda_code
        )
        assert (
            "/* unsupported CUDA multisample resource call: imageStore on image2DMS */ "
            "imageStore(msImage, pixel, sampleIndex, loaded)" in cuda_code
        )
        assert (
            "/* unsupported CUDA multisample resource call: imageLoad on image2DMSArray */ "
            "imageLoad(msImageLayers, pixelLayer, sampleIndex)" in cuda_code
        )
        assert (
            "/* unsupported CUDA multisample resource call: imageStore on image2DMSArray */ "
            "imageStore(msImageLayers, pixelLayer, sampleIndex, loadedLayer)"
            in cuda_code
        )
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
