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

    def test_compute_stage_local_helper_functions_emit_before_kernel(self):
        """Test compute-stage helper functions are emitted before kernel calls."""
        source_code = """
        shader TestShader {
            compute {
                float helper(float x) {
                    return x + 1.0;
                }

                void main() {
                    float y = helper(1.0);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        assert len(next(iter(ast.stages.values())).local_functions) == 1

        hip_code = HipCodeGen().generate(ast)

        helper_signature = "__device__ float helper(float x)"
        kernel_signature = "__global__ void main()"
        assert helper_signature in hip_code
        assert kernel_signature in hip_code
        assert hip_code.index(helper_signature) < hip_code.index(kernel_signature)
        assert "float y = helper(1.0);" in hip_code

    def test_user_defined_mix_function_is_not_lowered_to_hip_builtin(self):
        """Test user-defined functions shadow HIP builtin lowering."""
        source_code = """
        shader TestShader {
            compute {
                float mix(float x, float y, float t) {
                    return x + y + t;
                }

                void main() {
                    float adjusted = mix(0.0, 1.0, 0.25);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "__device__ float mix(float x, float y, float t)" in hip_code
        assert "float adjusted = mix(0.0, 1.0, 0.25);" in hip_code
        assert "float adjusted = (0.0 + ((1.0 - 0.0) * 0.25));" not in hip_code

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
        assert codegen.function_map.get("atomicExchange") == "atomicExch"
        assert codegen.function_map.get("atomicCompareExchange") == "atomicCAS"
        assert codegen.function_map.get("atomicCompSwap") == "atomicCAS"

    def test_lambda_call_emits_hip_device_lambda(self):
        """Test CrossGL pseudo-lambda calls lower to HIP device lambdas."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    int scale = 2;
                    let mapped = map(values, lambda(x, { prepare(x); return (x + 1); }));
                    let folded = fold(values, 0, lambda(int acc, int x, (acc + x)));
                    let generic = map(values, lambda(Result<i32, i32> value, { return value; }));
                    let tupled = map(values, lambda((i32, i32) pair, { return pair; }));
                    let always = lambda(true);
                }
            }
        }
        """

        ast = Parser(Lexer(source_code).tokens).parse()
        hip_code = HipCodeGen().generate(ast)

        assert (
            "map(values, [&] __device__ (auto x) "
            "{ prepare(x); return (x + 1); })" in hip_code
        )
        assert (
            "fold(values, 0, [&] __device__ (int acc, int x) "
            "{ return (acc + x); })" in hip_code
        )
        assert "map(values, [&] __device__ (auto value) { return value; })" in hip_code
        assert "map(values, [&] __device__ (auto pair) { return pair; })" in hip_code
        assert "[&] __device__ () { return true; }" in hip_code
        assert "lambda(" not in hip_code

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

    def test_composite_vector_constructors_flatten_hip_lanes(self):
        """Test HIP vector constructors flatten vector and swizzle arguments."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    vec2 uv = vec2(0.25, 0.5);
                    vec4 color = vec4(uv, 0.75, 1.0);
                    vec3 normal = vec3(1.0, 2.0, 3.0);
                    vec4 packed = vec4(normal, 1.0);
                    vec3 rgb = vec3(color.rgb);
                    vec2 from_swizzle = vec2(color.xy);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "float4 color = make_float4(uv.x, uv.y, 0.75, 1.0);" in hip_code
        assert (
            "float4 packed = make_float4(normal.x, normal.y, normal.z, 1.0);"
            in hip_code
        )
        assert "float3 rgb = make_float3(color.x, color.y, color.z);" in hip_code
        assert "float2 from_swizzle = make_float2(color.x, color.y);" in hip_code
        assert "make_float4(uv, 0.75, 1.0)" not in hip_code
        assert "make_float4(normal, 1.0)" not in hip_code
        assert "make_float3(make_float3(color.x, color.y, color.z))" not in hip_code
        assert "make_float2(make_float2(color.x, color.y))" not in hip_code

    def test_complex_scalar_vector_constructor_splats_use_hip_helpers(self):
        """Test HIP scalar splats evaluate complex expressions once."""
        source_code = """
        shader TestShader {
            compute {
                float makeWeight() {
                    return 0.5;
                }

                int nextIndex() {
                    return 1;
                }

                bool nextFlag() {
                    return true;
                }

                void main() {
                    vec3 fromCall = vec3(makeWeight());
                    vec3 fromCast = vec3(nextIndex());
                    bvec4 fromBoolCall = bvec4(nextFlag());
                    bvec2 fromInt = bvec2(nextIndex());
                    vec3 simple = vec3(1.0);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "__device__ inline float3 cgl_float3_splat(float value)" in hip_code
        assert "__device__ inline uchar4 cgl_uchar4_splat(bool value)" in hip_code
        assert "__device__ inline uchar2 cgl_uchar2_splat(bool value)" in hip_code
        assert "float3 fromCall = cgl_float3_splat(makeWeight());" in hip_code
        assert "float3 fromCast = cgl_float3_splat(nextIndex());" in hip_code
        assert "uchar4 fromBoolCall = cgl_uchar4_splat(nextFlag());" in hip_code
        assert "uchar2 fromInt = cgl_uchar2_splat(nextIndex());" in hip_code
        assert "float3 simple = make_float3(1.0, 1.0, 1.0);" in hip_code
        assert "make_float3(makeWeight(), makeWeight(), makeWeight())" not in hip_code
        assert "make_float3(nextIndex(), nextIndex(), nextIndex())" not in hip_code
        assert "make_uchar4(nextFlag(), nextFlag()" not in hip_code
        assert "make_uchar2(nextIndex(), nextIndex())" not in hip_code

    def test_complex_composite_vector_constructors_use_hip_helpers(self):
        """Test HIP composite vector constructors evaluate vector inputs once."""
        source_code = """
        shader TestShader {
            compute {
                vec2 makeUv() {
                    return vec2(0.25, 0.5);
                }

                vec3 makeNormal() {
                    return vec3(0.0, 1.0, 0.0);
                }

                vec4 makeColor() {
                    return vec4(1.0, 2.0, 3.0, 4.0);
                }

                float makeWeight() {
                    return 0.75;
                }

                void main() {
                    vec4 color = vec4(makeUv(), 0.0, 1.0);
                    vec4 packed = vec4(makeNormal(), makeWeight());
                    vec3 rgb = vec3(makeColor().rgb);
                    vec2 uv = vec2(0.25, 0.5);
                    vec4 simple = vec4(uv, 0.0, 1.0);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert (
            "__device__ inline float4 "
            "cgl_float4_construct_float2_xy_float_float"
            "(float2 arg0, float arg1, float arg2)" in hip_code
        )
        assert "return make_float4(arg0.x, arg0.y, arg1, arg2);" in hip_code
        assert (
            "__device__ inline float4 cgl_float4_construct_float3_xyz_float"
            "(float3 arg0, float arg1)" in hip_code
        )
        assert "return make_float4(arg0.x, arg0.y, arg0.z, arg1);" in hip_code
        assert (
            "__device__ inline float3 cgl_float3_construct_float4_xyz(float4 arg0)"
            in hip_code
        )
        assert "return make_float3(arg0.x, arg0.y, arg0.z);" in hip_code
        assert (
            "float4 color = "
            "cgl_float4_construct_float2_xy_float_float(makeUv(), 0.0, 1.0);"
            in hip_code
        )
        assert (
            "float4 packed = "
            "cgl_float4_construct_float3_xyz_float(makeNormal(), makeWeight());"
            in hip_code
        )
        assert "float3 rgb = cgl_float3_construct_float4_xyz(makeColor());" in hip_code
        assert "float4 simple = make_float4(uv.x, uv.y, 0.0, 1.0);" in hip_code
        assert "make_float4(makeUv().x, makeUv().y, 0.0, 1.0)" not in hip_code
        assert "make_float4(makeNormal().x, makeNormal().y" not in hip_code
        assert (
            "make_float3(makeColor().x, makeColor().y, makeColor().z)" not in hip_code
        )

    def test_vector_swizzles_emit_hip_make_functions(self):
        """Test HIP lowers vector swizzles to scalar fields and constructors."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    vec4 base = vec4(1.0, 2.0, 3.0, 4.0);
                    ivec4 lanes = ivec4(1, 2, 3, 4);
                    dvec4 precise = dvec4(1.0, 2.0, 3.0, 4.0);
                    float red = base.r;
                    vec2 uv = base.xy;
                    vec3 rgb = base.rgb;
                    vec4 rgba = base.rgba;
                    ivec3 reversed = lanes.zyx;
                    dvec2 alphaGreen = precise.ag;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "float red = base.x;" in hip_code
        assert "float2 uv = make_float2(base.x, base.y);" in hip_code
        assert "float3 rgb = make_float3(base.x, base.y, base.z);" in hip_code
        assert "float4 rgba = make_float4(base.x, base.y, base.z, base.w);" in hip_code
        assert "int3 reversed = make_int3(lanes.z, lanes.y, lanes.x);" in hip_code
        assert "double2 alphaGreen = make_double2(precise.w, precise.y);" in hip_code
        assert ".xy" not in hip_code
        assert ".rgb" not in hip_code
        assert ".rgba" not in hip_code
        assert ".ag" not in hip_code

    def test_complex_vector_swizzles_use_hip_helpers(self):
        """Test HIP swizzles evaluate complex vector expressions once."""
        source_code = """
        shader TestShader {
            compute {
                vec4 makeColor() {
                    return vec4(1.0, 2.0, 3.0, 4.0);
                }

                vec2 makeUv() {
                    return vec2(0.25, 0.5);
                }

                void main() {
                    vec3 rgb = makeColor().rgb;
                    vec2 rg = makeColor().rg;
                    vec4 bgra = makeColor().bgra;
                    float red = makeColor().r;
                    vec2 uv = makeUv();
                    vec2 stable = uv.xy;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert (
            "__device__ inline float3 cgl_float3_construct_float4_xyz(float4 arg0)"
            in hip_code
        )
        assert "return make_float3(arg0.x, arg0.y, arg0.z);" in hip_code
        assert (
            "__device__ inline float2 cgl_float2_construct_float4_xy(float4 arg0)"
            in hip_code
        )
        assert "return make_float2(arg0.x, arg0.y);" in hip_code
        assert (
            "__device__ inline float4 cgl_float4_construct_float4_zyxw(float4 arg0)"
            in hip_code
        )
        assert "return make_float4(arg0.z, arg0.y, arg0.x, arg0.w);" in hip_code
        assert "float3 rgb = cgl_float3_construct_float4_xyz(makeColor());" in hip_code
        assert "float2 rg = cgl_float2_construct_float4_xy(makeColor());" in hip_code
        assert (
            "float4 bgra = cgl_float4_construct_float4_zyxw(makeColor());" in hip_code
        )
        assert "float red = makeColor().x;" in hip_code
        assert "float2 stable = make_float2(uv.x, uv.y);" in hip_code
        assert (
            "make_float3(makeColor().x, makeColor().y, makeColor().z)" not in hip_code
        )
        assert "make_float2(makeColor().x, makeColor().y)" not in hip_code
        assert "make_float4(makeColor().z, makeColor().y" not in hip_code

    def test_vector_array_accesses_infer_hip_element_types(self):
        """Test HIP infers vector element types through array/member chains."""
        source_code = """
        struct InnerPayload {
            vec3 color;
            vec3 values[2];
        };

        struct Payload {
            vec3 colors[2];
            InnerPayload inners[2];
            float grid[2][3];
        };

        shader TestShader {
            Payload makePayload() {
                Payload payload;
                payload.colors[0] = vec3(1.0, 2.0, 3.0);
                payload.colors[1] = vec3(4.0, 5.0, 6.0);
                payload.inners[0].color = vec3(7.0, 8.0, 9.0);
                payload.inners[0].values[0] = vec3(0.5, 0.25, 0.125);
                payload.grid[0][1] = 0.25;
                return payload;
            }

            compute {
                void main() {
                    int slot = 1;
                    int valueSlot = 0;
                    vec3 localColors[2];
                    InnerPayload localInners[2];
                    localColors[0] = vec3(0.0, 1.0, 2.0);
                    localInners[slot].values[valueSlot] = vec3(3.0, 4.0, 5.0);
                    vec2 localRg = localColors[slot].xy;
                    vec2 localNested = localInners[slot].values[valueSlot].xy;
                    vec2 rg = makePayload().colors[slot].xy;
                    vec3 rgb = vec3(makePayload().colors[0]);
                    vec2 nested = makePayload().inners[slot].values[valueSlot].xy;
                    float gridValue = makePayload().grid[0][1];
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "float3 values[2];" in hip_code
        assert "float3 colors[2];" in hip_code
        assert "InnerPayload inners[2];" in hip_code
        assert "float grid[2][3];" in hip_code
        assert (
            "__device__ inline float2 cgl_float2_construct_float3_xy(float3 arg0)"
            in hip_code
        )
        assert (
            "__device__ inline float3 cgl_float3_construct_float3_xyz(float3 arg0)"
            in hip_code
        )
        assert (
            "float2 localRg = make_float2(localColors[slot].x, "
            "localColors[slot].y);" in hip_code
        )
        assert (
            "float2 localNested = make_float2("
            "localInners[slot].values[valueSlot].x, "
            "localInners[slot].values[valueSlot].y);" in hip_code
        )
        assert (
            "float2 rg = cgl_float2_construct_float3_xy("
            "makePayload().colors[slot]);" in hip_code
        )
        assert (
            "float3 rgb = cgl_float3_construct_float3_xyz("
            "makePayload().colors[0]);" in hip_code
        )
        assert (
            "float2 nested = cgl_float2_construct_float3_xy("
            "makePayload().inners[slot].values[valueSlot]);" in hip_code
        )
        assert "float gridValue = makePayload().grid[0][1];" in hip_code
        assert "float2 localRg = localColors[slot].xy;" not in hip_code
        assert (
            "float2 localNested = localInners[slot].values[valueSlot].xy;"
            not in hip_code
        )
        assert "float2 rg = makePayload().colors[slot].xy;" not in hip_code
        assert (
            "float2 nested = makePayload().inners[slot].values[valueSlot].xy;"
            not in hip_code
        )
        assert "make_float3(makePayload().colors[0])" not in hip_code

    def test_vector_scalar_arithmetic_expands_hip_components(self):
        """Test HIP lowers vector math through component-wise helpers."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "float4 base = make_float4(1.0, 1.0, 1.0, 1.0);" in hip_code
        assert "__device__ inline float4 cgl_float4_mul_scalar" in hip_code
        assert "__device__ inline float4 cgl_scalar_add_float4" in hip_code
        assert "__device__ inline float4 cgl_float4_sub_scalar" in hip_code
        assert "__device__ inline float4 cgl_float4_div_scalar" in hip_code
        assert "__device__ inline float4 cgl_float4_add" in hip_code
        assert "float4 scaled = cgl_float4_mul_scalar(base, weight);" in hip_code
        assert "float4 biased = cgl_scalar_add_float4(weight, base);" in hip_code
        assert "float4 shifted = cgl_float4_sub_scalar(base, 1.0);" in hip_code
        assert "float4 divided = cgl_float4_div_scalar(base, weight);" in hip_code
        assert "float4 combined = cgl_float4_add(base, offset);" in hip_code
        assert "base = cgl_float4_mul_scalar(base, weight);" in hip_code
        assert "(base * weight)" not in hip_code
        assert "(weight + base)" not in hip_code
        assert "make_float4(1.0);" not in hip_code

    def test_vector_unary_negation_uses_hip_helpers(self):
        """Test HIP lowers vector unary negation through component-wise helpers."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    float x = 1.0;
                    float y = -x;
                    vec2 v2 = vec2(1.0, -2.0);
                    vec3 v3 = vec3(1.0, 2.0, 3.0);
                    vec4 v4 = vec4(1.0);
                    ivec3 iv = ivec3(1, -2, 3);
                    vec2 neg2 = -v2;
                    vec3 neg3 = -v3;
                    vec4 neg4 = -v4;
                    ivec3 negi = -iv;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "float y = -x;" in hip_code
        assert "__device__ inline float2 cgl_float2_neg(float2 value)" in hip_code
        assert "__device__ inline float3 cgl_float3_neg(float3 value)" in hip_code
        assert "__device__ inline float4 cgl_float4_neg(float4 value)" in hip_code
        assert "__device__ inline int3 cgl_int3_neg(int3 value)" in hip_code
        assert "return make_float3((-value.x), (-value.y), (-value.z));" in hip_code
        assert "return make_int3((-value.x), (-value.y), (-value.z));" in hip_code
        assert "float2 neg2 = cgl_float2_neg(v2);" in hip_code
        assert "float3 neg3 = cgl_float3_neg(v3);" in hip_code
        assert "float4 neg4 = cgl_float4_neg(v4);" in hip_code
        assert "int3 negi = cgl_int3_neg(iv);" in hip_code
        assert "float2 neg2 = -v2;" not in hip_code
        assert "float3 neg3 = -v3;" not in hip_code
        assert "float4 neg4 = -v4;" not in hip_code
        assert "int3 negi = -iv;" not in hip_code

    def test_bool_vector_logical_not_uses_hip_helpers(self):
        """Test HIP lowers bool-vector logical not through component-wise helpers."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    bool flag = true;
                    bool invFlag = !flag;
                    bvec2 m2 = bvec2(true, false);
                    bvec3 m3 = bvec3(true, false, true);
                    bvec4 m4 = bvec4(true, false, true, false);
                    bvec2 inv2 = !m2;
                    bvec3 inv3 = !m3;
                    bvec4 inv4 = !m4;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "bool invFlag = !flag;" in hip_code
        assert "__device__ inline uchar2 cgl_uchar2_not(uchar2 value)" in hip_code
        assert "__device__ inline uchar3 cgl_uchar3_not(uchar3 value)" in hip_code
        assert "__device__ inline uchar4 cgl_uchar4_not(uchar4 value)" in hip_code
        assert "return make_uchar3((!value.x), (!value.y), (!value.z));" in hip_code
        assert "uchar2 inv2 = cgl_uchar2_not(m2);" in hip_code
        assert "uchar3 inv3 = cgl_uchar3_not(m3);" in hip_code
        assert "uchar4 inv4 = cgl_uchar4_not(m4);" in hip_code
        assert "uchar2 inv2 = !m2;" not in hip_code
        assert "uchar3 inv3 = !m3;" not in hip_code
        assert "uchar4 inv4 = !m4;" not in hip_code

    def test_bool_vector_logical_binary_ops_expand_hip_components(self):
        """Test HIP lowers bool-vector logical binary operators component-wise."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    bool aFlag = true;
                    bool bFlag = false;
                    bool scalarAnd = aFlag && bFlag;
                    bool scalarOr = aFlag || bFlag;
                    bvec3 a = bvec3(true, false, true);
                    bvec3 b = bvec3(false, true, true);
                    bvec3 both = a && b;
                    bvec3 either = a || b;
                    bvec3 scalarRight = a && true;
                    bvec3 scalarLeft = false || b;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "bool scalarAnd = (aFlag && bFlag);" in hip_code
        assert "bool scalarOr = (aFlag || bFlag);" in hip_code
        assert (
            "uchar3 both = make_uchar3((a.x && b.x), (a.y && b.y), "
            "(a.z && b.z));" in hip_code
        )
        assert (
            "uchar3 either = make_uchar3((a.x || b.x), (a.y || b.y), "
            "(a.z || b.z));" in hip_code
        )
        assert (
            "uchar3 scalarRight = make_uchar3((a.x && true), "
            "(a.y && true), (a.z && true));" in hip_code
        )
        assert (
            "uchar3 scalarLeft = make_uchar3((false || b.x), "
            "(false || b.y), (false || b.z));" in hip_code
        )
        assert "uchar3 both = (a && b);" not in hip_code
        assert "uchar3 either = (a || b);" not in hip_code
        assert "uchar3 scalarRight = (a && true);" not in hip_code
        assert "uchar3 scalarLeft = (false || b);" not in hip_code

    def test_bool_vector_ternary_condition_expands_hip_components(self):
        """Test HIP lowers bool-vector ternary conditions component-wise."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    bool cond = true;
                    bvec3 mask = bvec3(true, false, true);
                    vec3 a = vec3(1.0, 2.0, 3.0);
                    vec3 b = vec3(4.0, 5.0, 6.0);
                    vec3 scalarSelected = cond ? a : b;
                    vec3 laneSelected = mask ? a : b;
                    vec3 scalarArms = mask ? 1.0 : 0.0;
                    vec3 mixedArm = mask ? a : 0.0;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "float3 scalarSelected = (cond ? a : b);" in hip_code
        assert (
            "float3 laneSelected = make_float3((mask.x ? a.x : b.x), "
            "(mask.y ? a.y : b.y), (mask.z ? a.z : b.z));" in hip_code
        )
        assert (
            "float3 scalarArms = make_float3((mask.x ? 1.0 : 0.0), "
            "(mask.y ? 1.0 : 0.0), (mask.z ? 1.0 : 0.0));" in hip_code
        )
        assert (
            "float3 mixedArm = make_float3((mask.x ? a.x : 0.0), "
            "(mask.y ? a.y : 0.0), (mask.z ? a.z : 0.0));" in hip_code
        )
        assert "float3 laneSelected = (mask ? a : b);" not in hip_code
        assert "float3 scalarArms = (mask ? 1.0 : 0.0);" not in hip_code
        assert "float3 mixedArm = (mask ? a : 0.0);" not in hip_code

    def test_integer_vector_bitwise_ops_expand_hip_components(self):
        """Test HIP lowers integer-vector bitwise operators component-wise."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    ivec3 a = ivec3(1, 2, 3);
                    ivec3 b = ivec3(4, 5, 6);
                    ivec3 anded = a & b;
                    ivec3 ored = a | b;
                    ivec3 xored = a ^ b;
                    ivec3 shiftLeftScalar = a << 1;
                    ivec3 shiftRightVec = b >> a;
                    uvec4 ua = uvec4(1, 2, 3, 4);
                    uvec4 ub = uvec4(4, 3, 2, 1);
                    uvec4 uanded = ua & ub;
                    uvec4 ushift = ua >> 1;
                    int scalar = 1 & 3;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "int scalar = (1 & 3);" in hip_code
        assert (
            "int3 anded = make_int3((a.x & b.x), (a.y & b.y), (a.z & b.z));" in hip_code
        )
        assert (
            "int3 ored = make_int3((a.x | b.x), (a.y | b.y), (a.z | b.z));" in hip_code
        )
        assert (
            "int3 xored = make_int3((a.x ^ b.x), (a.y ^ b.y), (a.z ^ b.z));" in hip_code
        )
        assert (
            "int3 shiftLeftScalar = make_int3((a.x << 1), "
            "(a.y << 1), (a.z << 1));" in hip_code
        )
        assert (
            "int3 shiftRightVec = make_int3((b.x >> a.x), "
            "(b.y >> a.y), (b.z >> a.z));" in hip_code
        )
        assert (
            "uint4 uanded = make_uint4((ua.x & ub.x), (ua.y & ub.y), "
            "(ua.z & ub.z), (ua.w & ub.w));" in hip_code
        )
        assert (
            "uint4 ushift = make_uint4((ua.x >> 1), (ua.y >> 1), "
            "(ua.z >> 1), (ua.w >> 1));" in hip_code
        )
        assert "int3 anded = (a & b);" not in hip_code
        assert "int3 shiftLeftScalar = (a << 1);" not in hip_code
        assert "uint4 uanded = (ua & ub);" not in hip_code
        assert "uint4 ushift = (ua >> 1);" not in hip_code

    def test_integer_vector_compound_bitwise_assignments_expand_hip_components(self):
        """Test HIP lowers vector compound bitwise assignments component-wise."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    ivec3 a = ivec3(1, 2, 3);
                    ivec3 b = ivec3(4, 5, 6);
                    a &= b;
                    a |= b;
                    a ^= b;
                    a <<= 1;
                    b >>= a;
                    uvec4 ua = uvec4(1, 2, 3, 4);
                    uvec4 ub = uvec4(4, 3, 2, 1);
                    ua &= ub;
                    ua >>= 1;
                    int scalar = 7;
                    scalar &= 3;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "scalar &= 3;" in hip_code
        assert "a = make_int3((a.x & b.x), (a.y & b.y), (a.z & b.z));" in hip_code
        assert "a = make_int3((a.x | b.x), (a.y | b.y), (a.z | b.z));" in hip_code
        assert "a = make_int3((a.x ^ b.x), (a.y ^ b.y), (a.z ^ b.z));" in hip_code
        assert "a = make_int3((a.x << 1), (a.y << 1), (a.z << 1));" in hip_code
        assert "b = make_int3((b.x >> a.x), (b.y >> a.y), (b.z >> a.z));" in hip_code
        assert (
            "ua = make_uint4((ua.x & ub.x), (ua.y & ub.y), "
            "(ua.z & ub.z), (ua.w & ub.w));" in hip_code
        )
        assert (
            "ua = make_uint4((ua.x >> 1), (ua.y >> 1), "
            "(ua.z >> 1), (ua.w >> 1));" in hip_code
        )
        assert "a &= b;" not in hip_code
        assert "a |= b;" not in hip_code
        assert "a ^= b;" not in hip_code
        assert "a <<= 1;" not in hip_code
        assert "b >>= a;" not in hip_code
        assert "ua &= ub;" not in hip_code
        assert "ua >>= 1;" not in hip_code

    def test_vector_modulo_uses_hip_helpers(self):
        """Test HIP lowers vector modulo expressions and assignments."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    vec3 a = vec3(5.5, 6.5, 7.5);
                    vec3 b = vec3(2.0, 2.5, 3.0);
                    vec3 vf = a % b;
                    vec3 vs = a % 2.0;
                    vec3 sv = 10.0 % b;
                    vec3 builtin = mod(a, b);
                    a %= b;
                    dvec2 da = dvec2(5.5, 6.5);
                    dvec2 db = dvec2(2.0, 2.5);
                    dvec2 dm = da % db;
                    da %= db;
                    ivec3 ia = ivec3(5, 6, 7);
                    ivec3 ib = ivec3(2, 3, 4);
                    ivec3 im = ia % ib;
                    ivec3 ims = ia % 2;
                    ia %= ib;
                    uvec4 ua = uvec4(5, 6, 7, 8);
                    uvec4 ub = uvec4(2, 3, 4, 5);
                    uvec4 um = ua % ub;
                    ua %= ub;
                    int scalar = 5 % 2;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "int scalar = (5 % 2);" in hip_code
        assert (
            "__device__ inline float3 cgl_float3_mod(float3 lhs, float3 rhs)"
            in hip_code
        )
        assert (
            "return make_float3(fmodf(lhs.x, rhs.x), fmodf(lhs.y, rhs.y), fmodf(lhs.z, rhs.z));"
            in hip_code
        )
        assert (
            "__device__ inline double2 cgl_double2_mod(double2 lhs, double2 rhs)"
            in hip_code
        )
        assert (
            "return make_double2(fmod(lhs.x, rhs.x), fmod(lhs.y, rhs.y));" in hip_code
        )
        assert "__device__ inline int3 cgl_int3_mod(int3 lhs, int3 rhs)" in hip_code
        assert "__device__ inline uint4 cgl_uint4_mod(uint4 lhs, uint4 rhs)" in hip_code
        assert "float3 vf = cgl_float3_mod(a, b);" in hip_code
        assert "float3 vs = cgl_float3_mod_scalar(a, 2.0);" in hip_code
        assert "float3 sv = cgl_scalar_mod_float3(10.0, b);" in hip_code
        assert "float3 builtin = cgl_float3_mod(a, b);" in hip_code
        assert "a = cgl_float3_mod(a, b);" in hip_code
        assert "double2 dm = cgl_double2_mod(da, db);" in hip_code
        assert "da = cgl_double2_mod(da, db);" in hip_code
        assert "int3 im = cgl_int3_mod(ia, ib);" in hip_code
        assert "int3 ims = cgl_int3_mod_scalar(ia, 2);" in hip_code
        assert "ia = cgl_int3_mod(ia, ib);" in hip_code
        assert "uint4 um = cgl_uint4_mod(ua, ub);" in hip_code
        assert "ua = cgl_uint4_mod(ua, ub);" in hip_code
        assert "float3 vf = (a % b);" not in hip_code
        assert "float3 builtin = fmodf(a, b);" not in hip_code
        assert "a %= b;" not in hip_code
        assert "da %= db;" not in hip_code
        assert "int3 im = (ia % ib);" not in hip_code
        assert "ia %= ib;" not in hip_code
        assert "uint4 um = (ua % ub);" not in hip_code
        assert "ua %= ub;" not in hip_code

    def test_scalar_float_modulo_uses_hip_fmod(self):
        """Test HIP lowers floating scalar modulo while preserving integer modulo."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    float fa = 5.5;
                    float fb = 2.0;
                    float fm = fa % fb;
                    float flit = 5.5 % 2.0;
                    fa %= fb;
                    double da = 5.5;
                    double db = 2.0;
                    double dm = da % db;
                    da %= db;
                    int ia = 5;
                    int ib = 2;
                    int im = ia % ib;
                    ia %= ib;
                    uint ua = 5;
                    uint ub = 2;
                    uint um = ua % ub;
                    ua %= ub;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "float fm = fmodf(fa, fb);" in hip_code
        assert "float flit = fmodf(5.5, 2.0);" in hip_code
        assert "fa = fmodf(fa, fb);" in hip_code
        assert "double dm = fmod(da, db);" in hip_code
        assert "da = fmod(da, db);" in hip_code
        assert "int im = (ia % ib);" in hip_code
        assert "ia %= ib;" in hip_code
        assert "unsigned int um = (ua % ub);" in hip_code
        assert "ua %= ub;" in hip_code
        assert "float fm = (fa % fb);" not in hip_code
        assert "double dm = (da % db);" not in hip_code
        assert "fa %= fb;" not in hip_code
        assert "da %= db;" not in hip_code

    def test_vector_comparisons_emit_hip_bool_vector_constructors(self):
        """Test HIP lowers vector comparisons component-wise."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    vec3 a = vec3(1.0, 2.0, 3.0);
                    vec3 b = vec3(1.0, 0.0, 4.0);
                    bvec3 lt = a < b;
                    bvec3 le = a <= b;
                    bvec3 gt = a > b;
                    bvec3 ge = a >= 0.0;
                    bvec3 scalarLeft = 1.0 < b;
                    vec4 c = vec4(1.0);
                    vec4 d = vec4(2.0);
                    bvec4 eq = c == d;
                    ivec2 ia = ivec2(1, 2);
                    ivec2 ib = ivec2(2, 2);
                    bvec2 ne = ia != ib;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert (
            "uchar3 lt = make_uchar3((a.x < b.x), (a.y < b.y), (a.z < b.z));"
            in hip_code
        )
        assert (
            "uchar3 le = make_uchar3((a.x <= b.x), (a.y <= b.y), (a.z <= b.z));"
            in hip_code
        )
        assert (
            "uchar3 gt = make_uchar3((a.x > b.x), (a.y > b.y), (a.z > b.z));"
            in hip_code
        )
        assert (
            "uchar3 ge = make_uchar3((a.x >= 0.0), (a.y >= 0.0), (a.z >= 0.0));"
            in hip_code
        )
        assert (
            "uchar3 scalarLeft = make_uchar3((1.0 < b.x), "
            "(1.0 < b.y), (1.0 < b.z));" in hip_code
        )
        assert (
            "uchar4 eq = make_uchar4((c.x == d.x), (c.y == d.y), "
            "(c.z == d.z), (c.w == d.w));" in hip_code
        )
        assert "uchar2 ne = make_uchar2((ia.x != ib.x), (ia.y != ib.y));" in hip_code
        assert "uchar3 lt = (a < b);" not in hip_code
        assert "uchar4 eq = (c == d);" not in hip_code
        assert "uchar2 ne = (ia != ib);" not in hip_code

    def test_complex_vector_conditionals_emit_hip_single_eval_helpers(self):
        """Test HIP vector logical/comparison/select operands evaluate once."""
        source_code = """
        shader TestShader {
            compute {
                bvec3 makeMask() {
                    return bvec3(true, false, true);
                }

                bvec3 makeOtherMask() {
                    return bvec3(false, true, true);
                }

                bool nextFlag() {
                    return true;
                }

                vec3 makeA() {
                    return vec3(1.0, 2.0, 3.0);
                }

                vec3 makeB() {
                    return vec3(4.0, 5.0, 6.0);
                }

                float nextWeight() {
                    return 0.5;
                }

                void main() {
                    bvec3 both = makeMask() && makeOtherMask();
                    bvec3 vectorScalar = makeMask() && nextFlag();
                    bvec3 scalarVector = nextFlag() || makeOtherMask();
                    bvec3 lt = makeA() < makeB();
                    bvec3 ge = makeA() >= nextWeight();
                    bvec3 scalarLeft = nextWeight() < makeB();
                    vec3 selected = makeMask() ? makeA() : makeB();
                    vec3 scalarBranch = makeMask() ? nextWeight() : 0.0;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert (
            "__device__ inline uchar3 "
            "cgl_uchar3_logical_and_uchar3_xyz_uchar3_xyz"
            "(uchar3 arg0, uchar3 arg1)" in hip_code
        )
        assert (
            "__device__ inline uchar3 cgl_uchar3_logical_and_uchar3_xyz_bool"
            "(uchar3 arg0, bool arg1)" in hip_code
        )
        assert (
            "__device__ inline uchar3 cgl_uchar3_logical_or_bool_uchar3_xyz"
            "(bool arg0, uchar3 arg1)" in hip_code
        )
        assert (
            "return make_uchar3((arg0.x && arg1.x), (arg0.y && arg1.y), "
            "(arg0.z && arg1.z));" in hip_code
        )
        assert (
            "__device__ inline uchar3 "
            "cgl_uchar3_compare_lt_float3_xyz_float3_xyz"
            "(float3 arg0, float3 arg1)" in hip_code
        )
        assert (
            "__device__ inline uchar3 cgl_uchar3_compare_ge_float3_xyz_float"
            "(float3 arg0, float arg1)" in hip_code
        )
        assert (
            "__device__ inline uchar3 cgl_uchar3_compare_lt_float_float3_xyz"
            "(float arg0, float3 arg1)" in hip_code
        )
        assert (
            "__device__ inline float3 "
            "cgl_float3_select_uchar3_xyz_float3_xyz_float3_xyz"
            "(uchar3 arg0, float3 arg1, float3 arg2)" in hip_code
        )
        assert (
            "__device__ inline float3 cgl_float3_select_uchar3_xyz_float_float"
            "(uchar3 arg0, float arg1, float arg2)" in hip_code
        )
        assert (
            "uchar3 both = "
            "cgl_uchar3_logical_and_uchar3_xyz_uchar3_xyz("
            "makeMask(), makeOtherMask());" in hip_code
        )
        assert (
            "uchar3 vectorScalar = cgl_uchar3_logical_and_uchar3_xyz_bool("
            "makeMask(), nextFlag());" in hip_code
        )
        assert (
            "uchar3 scalarVector = cgl_uchar3_logical_or_bool_uchar3_xyz("
            "nextFlag(), makeOtherMask());" in hip_code
        )
        assert (
            "uchar3 lt = cgl_uchar3_compare_lt_float3_xyz_float3_xyz("
            "makeA(), makeB());" in hip_code
        )
        assert (
            "uchar3 ge = cgl_uchar3_compare_ge_float3_xyz_float("
            "makeA(), nextWeight());" in hip_code
        )
        assert (
            "uchar3 scalarLeft = cgl_uchar3_compare_lt_float_float3_xyz("
            "nextWeight(), makeB());" in hip_code
        )
        assert (
            "float3 selected = "
            "cgl_float3_select_uchar3_xyz_float3_xyz_float3_xyz("
            "makeMask(), makeA(), makeB());" in hip_code
        )
        assert (
            "float3 scalarBranch = cgl_float3_select_uchar3_xyz_float_float("
            "makeMask(), nextWeight(), 0.0);" in hip_code
        )
        assert "makeMask().x && makeOtherMask().x" not in hip_code
        assert "makeA().x < makeB().x" not in hip_code
        assert "makeMask().x ? makeA().x" not in hip_code
        assert "makeMask().x ? nextWeight()" not in hip_code

    def test_vector_geometric_builtins_emit_hip_helpers(self):
        """Test HIP lowers vector geometric builtins to helper calls."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    vec3 a = vec3(1.0, 2.0, 3.0);
                    vec3 b = vec3(4.0, 5.0, 6.0);
                    float d = dot(a, b);
                    vec3 c = cross(a, b);
                    vec3 n = normalize(a);
                    float l = length(a);
                    dvec2 da = dvec2(3.0, 4.0);
                    double dl = length(da);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert (
            "__device__ inline float cgl_float3_dot(float3 lhs, float3 rhs)" in hip_code
        )
        assert (
            "__device__ inline float3 cgl_float3_cross(float3 lhs, float3 rhs)"
            in hip_code
        )
        assert "__device__ inline float cgl_float3_length(float3 value)" in hip_code
        assert "__device__ inline float3 cgl_float3_normalize(float3 value)" in hip_code
        assert "__device__ inline double cgl_double2_length(double2 value)" in hip_code
        assert "return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z);" in hip_code
        assert (
            "return make_float3((lhs.y * rhs.z - lhs.z * rhs.y), "
            "(lhs.z * rhs.x - lhs.x * rhs.z), "
            "(lhs.x * rhs.y - lhs.y * rhs.x));" in hip_code
        )
        assert (
            "return sqrtf((value.x * value.x) + "
            "(value.y * value.y) + (value.z * value.z));" in hip_code
        )
        assert "float inv_length = 1.0f / cgl_float3_length(value);" in hip_code
        assert "float d = cgl_float3_dot(a, b);" in hip_code
        assert "float3 c = cgl_float3_cross(a, b);" in hip_code
        assert "float3 n = cgl_float3_normalize(a);" in hip_code
        assert "float l = cgl_float3_length(a);" in hip_code
        assert "double dl = cgl_double2_length(da);" in hip_code
        assert "float d = dot(a, b);" not in hip_code
        assert "float3 c = cross(a, b);" not in hip_code
        assert "float3 n = normalize(a);" not in hip_code
        assert "float l = length(a);" not in hip_code

    def test_vector_atan2_builtin_lowers_componentwise(self):
        """Test HIP lowers vector atan2 through component-wise helpers."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    vec3 y = vec3(1.0, 2.0, 3.0);
                    vec3 x = vec3(4.0, 5.0, 6.0);
                    vec3 a = atan2(y, x);
                    dvec2 dy = dvec2(1.0, 2.0);
                    dvec2 dx = dvec2(3.0, 4.0);
                    dvec2 da = atan2(dy, dx);
                    float s = atan2(1.0, 2.0);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert (
            "__device__ inline float3 cgl_float3_atan2(float3 y, float3 x)" in hip_code
        )
        assert (
            "return make_float3(atan2f(y.x, x.x), atan2f(y.y, x.y), "
            "atan2f(y.z, x.z));" in hip_code
        )
        assert (
            "__device__ inline double2 cgl_double2_atan2(double2 y, double2 x)"
            in hip_code
        )
        assert "return make_double2(atan2(y.x, x.x), atan2(y.y, x.y));" in hip_code
        assert "float3 a = cgl_float3_atan2(y, x);" in hip_code
        assert "double2 da = cgl_double2_atan2(dy, dx);" in hip_code
        assert "float s = atan2f(1.0, 2.0);" in hip_code
        assert "float3 a = atan2f(" not in hip_code
        assert "double2 da = atan2f(" not in hip_code

    def test_sign_builtin_lowers_to_hip_expressions(self):
        """Test HIP lowers scalar and vector sign without a raw sign builtin."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    float fx = -1.0;
                    float fs = sign(fx);
                    double dx = -2.0;
                    double ds = sign(dx);
                    int ix = -3;
                    int is = sign(ix);
                    uint ux = 4;
                    uint us = sign(ux);
                    vec3 v = vec3(-1.0, 0.0, 1.0);
                    vec3 vs = sign(v);
                    dvec2 dv = dvec2(-1.0, 1.0);
                    dvec2 dvs = sign(dv);
                    ivec3 iv = ivec3(-1, 0, 1);
                    ivec3 ivs = sign(iv);
                    uvec4 uv = uvec4(0, 1, 2, 3);
                    uvec4 uvs = sign(uv);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert (
            "float fs = ((fx) > 0.0f ? 1.0f : "
            "((fx) < 0.0f ? -1.0f : 0.0f));" in hip_code
        )
        assert (
            "double ds = ((dx) > 0.0 ? 1.0 : " "((dx) < 0.0 ? -1.0 : 0.0));" in hip_code
        )
        assert "int is = ((ix) > 0 ? 1 : ((ix) < 0 ? -1 : 0));" in hip_code
        assert "unsigned int us = ((ux) > 0u ? 1u : 0u);" in hip_code
        assert "__device__ inline float3 cgl_float3_sign(float3 value)" in hip_code
        assert (
            "return make_float3(((value.x) > 0.0f ? 1.0f : "
            "((value.x) < 0.0f ? -1.0f : 0.0f)), " in hip_code
        )
        assert "__device__ inline double2 cgl_double2_sign(double2 value)" in hip_code
        assert "__device__ inline int3 cgl_int3_sign(int3 value)" in hip_code
        assert "__device__ inline uint4 cgl_uint4_sign(uint4 value)" in hip_code
        assert "float3 vs = cgl_float3_sign(v);" in hip_code
        assert "double2 dvs = cgl_double2_sign(dv);" in hip_code
        assert "int3 ivs = cgl_int3_sign(iv);" in hip_code
        assert "uint4 uvs = cgl_uint4_sign(uv);" in hip_code
        assert " = sign(" not in hip_code

    def test_vector_min_max_expands_hip_components(self):
        """Test HIP lowers vector min/max through component-wise helpers."""
        source_code = """
        shader TestShader {
            compute {
                float nextWeight() {
                    return 0.5;
                }

                void main() {
                    vec3 a = vec3(1.0, 2.0, 3.0);
                    vec3 b = vec3(3.0, 2.0, 1.0);
                    vec3 lo = min(a, b);
                    vec3 hi = max(a, b);
                    vec3 capped = min(a, 2.0);
                    vec3 raised = max(0.0, b);
                    vec3 complexCapped = min(a, nextWeight());
                    vec3 complexRaised = max(nextWeight(), b);
                    ivec3 ia = ivec3(1, 2, 3);
                    ivec3 ib = ivec3(3, 2, 1);
                    ivec3 ilo = min(ia, ib);
                    float scalar = min(0.25, 1.0);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert (
            "__device__ inline float3 cgl_float3_min(float3 lhs, float3 rhs)"
            in hip_code
        )
        assert (
            "__device__ inline float3 cgl_float3_max(float3 lhs, float3 rhs)"
            in hip_code
        )
        assert (
            "__device__ inline float3 cgl_float3_min_scalar(float3 lhs, float rhs)"
            in hip_code
        )
        assert (
            "__device__ inline float3 cgl_scalar_max_float3(float lhs, float3 rhs)"
            in hip_code
        )
        assert "__device__ inline int3 cgl_int3_min(int3 lhs, int3 rhs)" in hip_code
        assert (
            "return make_float3(fminf(lhs.x, rhs.x), fminf(lhs.y, rhs.y), fminf(lhs.z, rhs.z));"
            in hip_code
        )
        assert (
            "return make_float3(fmaxf(lhs.x, rhs.x), fmaxf(lhs.y, rhs.y), fmaxf(lhs.z, rhs.z));"
            in hip_code
        )
        assert "((lhs.x) < (rhs.x) ? (lhs.x) : (rhs.x))" in hip_code
        assert "float3 lo = cgl_float3_min(a, b);" in hip_code
        assert "float3 hi = cgl_float3_max(a, b);" in hip_code
        assert "float3 capped = cgl_float3_min_scalar(a, 2.0);" in hip_code
        assert "float3 raised = cgl_scalar_max_float3(0.0, b);" in hip_code
        assert (
            "float3 complexCapped = cgl_float3_min_scalar(a, nextWeight());" in hip_code
        )
        assert (
            "float3 complexRaised = cgl_scalar_max_float3(nextWeight(), b);" in hip_code
        )
        assert "int3 ilo = cgl_int3_min(ia, ib);" in hip_code
        assert "float scalar = fminf(0.25, 1.0);" in hip_code
        assert "float3 lo = fminf(a, b);" not in hip_code
        assert "float3 hi = fmaxf(a, b);" not in hip_code
        assert "float3 complexCapped = fminf(a, nextWeight());" not in hip_code
        assert "int3 ilo = fminf(ia, ib);" not in hip_code

    def test_vector_clamp_expands_hip_components(self):
        """Test HIP lowers vector clamp through a component-wise helper."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    vec3 v = vec3(2.0, -1.0, 0.5);
                    vec3 c = clamp(v, vec3(0.0), vec3(1.0));
                    float x = clamp(0.25, 0.0, 1.0);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "__device__ inline float3 cgl_float3_clamp" in hip_code
        assert "fmaxf(min_value.x, fminf(max_value.x, value.x))" in hip_code
        assert (
            "float3 c = cgl_float3_clamp("
            "v, make_float3(0.0, 0.0, 0.0), make_float3(1.0, 1.0, 1.0));" in hip_code
        )
        assert "float x = fmaxf(0.0, fminf(1.0, 0.25));" in hip_code
        assert "fmaxf(make_float3" not in hip_code
        assert "fminf(make_float3" not in hip_code
        assert " = clamp(" not in hip_code

    def test_scalar_clamp_uses_hip_type_appropriate_operations(self):
        """Test HIP scalar clamp preserves double and integer semantics."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    float x = clamp(0.25, 0.0, 1.0);
                    double v = 0.25;
                    double lo = 0.0;
                    double hi = 1.0;
                    double d = clamp(v, lo, hi);
                    int n = 2;
                    int i = clamp(n, 0, 1);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "float x = fmaxf(0.0, fminf(1.0, 0.25));" in hip_code
        assert "double d = fmax(lo, fmin(hi, v));" in hip_code
        assert "int i = ((n) < (0) ? (0) : ((n) > (1) ? (1) : (n)));" in hip_code
        assert "double d = fmaxf(" not in hip_code
        assert "int i = fmaxf(" not in hip_code
        assert " = clamp(" not in hip_code

    def test_complex_integer_clamp_uses_hip_single_eval_helper(self):
        """Test HIP integer clamp evaluates complex scalar operands once."""
        source_code = """
        shader TestShader {
            compute {
                int nextIndex() {
                    return 1;
                }

                int lowerBound() {
                    return 0;
                }

                int upperBound() {
                    return 2;
                }

                void main() {
                    int i = clamp(nextIndex(), lowerBound(), upperBound());
                    int n = 2;
                    int simple = clamp(n, 0, 1);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert (
            "__device__ inline int cgl_int_clamp"
            "(int value, int min_value, int max_value)" in hip_code
        )
        assert (
            "return ((value) < (min_value) ? (min_value) : "
            "((value) > (max_value) ? (max_value) : (value)));" in hip_code
        )
        assert (
            "int i = cgl_int_clamp(nextIndex(), lowerBound(), upperBound());"
            in hip_code
        )
        assert "int simple = ((n) < (0) ? (0) : ((n) > (1) ? (1) : (n)));" in hip_code
        assert "int i = ((nextIndex()) <" not in hip_code

    def test_saturate_builtin_lowers_through_hip_clamp(self):
        """Test HIP lowers shader-style saturate through scalar/vector clamp."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    float x = saturate(1.25);
                    double d = 1.25;
                    double y = saturate(d);
                    vec3 v = saturate(vec3(2.0, -1.0, 0.5));
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "float x = fmaxf(0.0, fminf(1.0, 1.25));" in hip_code
        assert "double y = fmax(0.0, fmin(1.0, d));" in hip_code
        assert (
            "__device__ inline float3 cgl_float3_clamp_scalar_min_scalar_max"
            in hip_code
        )
        assert (
            "return make_float3(fmaxf(min_value, fminf(max_value, value.x)), "
            "fmaxf(min_value, fminf(max_value, value.y)), "
            "fmaxf(min_value, fminf(max_value, value.z)));" in hip_code
        )
        assert (
            "float3 v = cgl_float3_clamp_scalar_min_scalar_max("
            "make_float3(2.0, -1.0, 0.5), 0.0, 1.0);" in hip_code
        )
        assert " = saturate(" not in hip_code

    def test_mix_builtin_lowers_to_hip_arithmetic(self):
        """Test HIP lowers mix without relying on a non-HIP lerp intrinsic."""
        source_code = """
        shader TestShader {
            compute {
                float nextX() {
                    return 0.0;
                }

                float nextY() {
                    return 1.0;
                }

                float nextT() {
                    return 0.25;
                }

                void main() {
                    float a = mix(0.0, 1.0, 0.25);
                    float complex = mix(nextX(), nextY(), nextT());
                    double x = 0.0;
                    double y = 1.0;
                    double t = 0.25;
                    double d = mix(x, y, t);
                    vec3 vx = vec3(1.0, 2.0, 3.0);
                    vec3 vy = vec3(4.0, 5.0, 6.0);
                    vec3 v = mix(vx, vy, 0.5);
                    vec3 wrapped = fract(mix(vx, vy, 0.5));
                    dvec2 px = dvec2(1.0, 2.0);
                    dvec2 py = dvec2(3.0, 4.0);
                    dvec2 p = mix(px, py, dvec2(0.25, 0.75));
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "float a = (0.0 + ((1.0 - 0.0) * 0.25));" in hip_code
        assert (
            "__device__ inline float cgl_float_mix(float x, float y, float a)"
            in hip_code
        )
        assert "return (x + ((y - x) * a));" in hip_code
        assert "float complex = cgl_float_mix(nextX(), nextY(), nextT());" in hip_code
        assert "double d = (x + ((y - x) * t));" in hip_code
        assert (
            "__device__ inline float3 cgl_float3_mix_scalar"
            "(float3 x, float3 y, float a)" in hip_code
        )
        assert (
            "return make_float3((x.x + ((y.x - x.x) * a)), "
            "(x.y + ((y.y - x.y) * a)), "
            "(x.z + ((y.z - x.z) * a)));" in hip_code
        )
        assert (
            "__device__ inline double2 cgl_double2_mix_vector"
            "(double2 x, double2 y, double2 a)" in hip_code
        )
        assert (
            "return make_double2((x.x + ((y.x - x.x) * a.x)), "
            "(x.y + ((y.y - x.y) * a.y)));" in hip_code
        )
        assert "float3 v = cgl_float3_mix_scalar(vx, vy, 0.5);" in hip_code
        assert (
            "float3 wrapped = cgl_float3_fract("
            "cgl_float3_mix_scalar(vx, vy, 0.5));" in hip_code
        )
        assert (
            "double2 p = cgl_double2_mix_vector(px, py, "
            "make_double2(0.25, 0.75));" in hip_code
        )
        assert "float complex = (nextX() + ((nextY() - nextX())" not in hip_code
        assert " = mix(" not in hip_code
        assert " = lerp(" not in hip_code

    def test_bool_vector_mix_lowers_to_hip_select(self):
        """Test HIP lowers bool-mask vector mix to component selection."""
        source_code = """
        shader TestShader {
            compute {
                vec3 makeA() {
                    return vec3(1.0, 2.0, 3.0);
                }

                vec3 makeB() {
                    return vec3(4.0, 5.0, 6.0);
                }

                bvec3 makeMask() {
                    return bvec3(true, false, true);
                }

                void main() {
                    vec3 vx = vec3(1.0, 2.0, 3.0);
                    vec3 vy = vec3(4.0, 5.0, 6.0);
                    bvec3 mask = bvec3(true, false, true);
                    vec3 selected = mix(vx, vy, mask);
                    vec3 complexSelected = mix(makeA(), makeB(), makeMask());
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert (
            "__device__ inline float3 "
            "cgl_float3_select_uchar3_xyz_float3_xyz_float3_xyz"
            "(uchar3 arg0, float3 arg1, float3 arg2)" in hip_code
        )
        assert (
            "return make_float3((arg0.x ? arg1.x : arg2.x), "
            "(arg0.y ? arg1.y : arg2.y), "
            "(arg0.z ? arg1.z : arg2.z));" in hip_code
        )
        assert (
            "float3 selected = make_float3((mask.x ? vy.x : vx.x), "
            "(mask.y ? vy.y : vx.y), (mask.z ? vy.z : vx.z));" in hip_code
        )
        assert (
            "float3 complexSelected = "
            "cgl_float3_select_uchar3_xyz_float3_xyz_float3_xyz("
            "makeMask(), makeB(), makeA());" in hip_code
        )
        assert "float3 selected = mix(vx, vy, mask);" not in hip_code
        assert "float3 complexSelected = mix(" not in hip_code
        assert " = lerp(" not in hip_code

    def test_bool_scalar_mix_lowers_to_hip_select(self):
        """Test HIP lowers scalar bool mix to selector semantics."""
        source_code = """
        shader TestShader {
            compute {
                float nextX() {
                    return 0.0;
                }

                float nextY() {
                    return 1.0;
                }

                bool nextFlag() {
                    return true;
                }

                void main() {
                    bool flag = true;
                    float literal = mix(0.0, 1.0, flag);
                    float complexValue = mix(nextX(), nextY(), nextFlag());
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert (
            "__device__ inline float cgl_float_select"
            "(bool condition, float true_value, float false_value)" in hip_code
        )
        assert "return condition ? true_value : false_value;" in hip_code
        assert "float literal = (flag ? 1.0 : 0.0);" in hip_code
        assert (
            "float complexValue = cgl_float_select("
            "nextFlag(), nextY(), nextX());" in hip_code
        )
        assert "float literal = (0.0 + ((1.0 - 0.0) * flag));" not in hip_code
        assert (
            "float complexValue = (nextX() + ((nextY() - nextX()) * "
            "nextFlag()));" not in hip_code
        )
        assert " = mix(" not in hip_code
        assert " = lerp(" not in hip_code

    def test_abs_builtin_uses_hip_type_appropriate_operations(self):
        """Test HIP abs preserves scalar types and lowers vector operands."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    float fx = -1.25;
                    float af = abs(fx);
                    double dx = -2.5;
                    double ad = abs(dx);
                    int ix = -3;
                    int ai = abs(ix);
                    vec3 v = vec3(-1.0, 2.0, -3.0);
                    vec3 av = abs(v);
                    ivec3 iv = ivec3(-1, 2, -3);
                    ivec3 aiv = abs(iv);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "float af = fabsf(fx);" in hip_code
        assert "double ad = fabs(dx);" in hip_code
        assert "int ai = abs(ix);" in hip_code
        assert "__device__ inline float3 cgl_float3_abs(float3 value)" in hip_code
        assert (
            "return make_float3(fabsf(value.x), fabsf(value.y), fabsf(value.z));"
            in hip_code
        )
        assert "__device__ inline int3 cgl_int3_abs(int3 value)" in hip_code
        assert "return make_int3(abs(value.x), abs(value.y), abs(value.z));" in hip_code
        assert "float3 av = cgl_float3_abs(v);" in hip_code
        assert "int3 aiv = cgl_int3_abs(iv);" in hip_code
        assert "double ad = fabsf(dx);" not in hip_code
        assert "int ai = fabsf(ix);" not in hip_code
        assert "float3 av = fabsf(v);" not in hip_code
        assert "int3 aiv = fabsf(iv);" not in hip_code

    def test_fract_scalar_builtin_lowers_to_hip_helper(self):
        """Test HIP lowers scalar fract without relying on a non-HIP fracf call."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    float x = fract(1.25);
                    float y = frac(1.75);
                    double d = 1.25;
                    double z = fract(d);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "__device__ inline float cgl_fract_float(float value)" in hip_code
        assert "return value - floorf(value);" in hip_code
        assert "__device__ inline double cgl_fract_double(double value)" in hip_code
        assert "return value - floor(value);" in hip_code
        assert "float x = cgl_fract_float(1.25);" in hip_code
        assert "float y = cgl_fract_float(1.75);" in hip_code
        assert "double z = cgl_fract_double(d);" in hip_code
        assert "fracf(" not in hip_code
        assert " = fract(" not in hip_code
        assert " = frac(" not in hip_code

    def test_fract_vector_builtin_lowers_componentwise(self):
        """Test HIP lowers vector fract through component-wise helpers."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    vec3 v = fract(vec3(1.25, 2.5, 3.75));
                    dvec2 p = fract(dvec2(1.25, 2.5));
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "__device__ inline float3 cgl_float3_fract(float3 value)" in hip_code
        assert (
            "return make_float3("
            "cgl_fract_float(value.x), "
            "cgl_fract_float(value.y), "
            "cgl_fract_float(value.z));" in hip_code
        )
        assert "__device__ inline double2 cgl_double2_fract(double2 value)" in hip_code
        assert (
            "return make_double2("
            "cgl_fract_double(value.x), "
            "cgl_fract_double(value.y));" in hip_code
        )
        assert (
            "float3 v = cgl_float3_fract(" "make_float3(1.25, 2.5, 3.75));" in hip_code
        )
        assert "double2 p = cgl_double2_fract(make_double2(1.25, 2.5));" in hip_code
        assert "fracf(" not in hip_code
        assert " = fract(" not in hip_code

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
            uniform float constants;

            compute {
                void main() {
                    shared float shared_data;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "__shared__ float shared_data;" in hip_code
        assert "__constant__ float constants;" in hip_code

    def test_for_in_statement_lowers_to_counted_hip_loops(self):
        """Test HIP lowers CrossGL for-in loops to counted integer loops."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
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
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "for (int i = 0; i < 4; ++i)" in hip_code
        assert "for (int j = 2; j < 5; ++j)" in hip_code
        assert "for (int k = 1; k <= 4; ++k)" in hip_code
        assert "total = (total + i);" in hip_code
        assert "total = (total + j);" in hip_code
        assert "total = (total + k);" in hip_code
        assert "ForInNode" not in hip_code
        assert "RangeNode" not in hip_code

    def test_atomic_operations(self):
        """Test atomic operations code generation"""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    int counter;
                    atomicAdd(counter, 1);
                    int oldValue = atomicExchange(counter, 2);
                    int exchanged = atomicCompareExchange(counter, 0, 3);
                    int swapped = atomicCompSwap(counter, exchanged, 4);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "__global__" in hip_code or "__device__" in hip_code
        assert "atomicAdd(counter, 1);" in hip_code
        assert "int oldValue = atomicExch(counter, 2);" in hip_code
        assert "int exchanged = atomicCAS(counter, 0, 3);" in hip_code
        assert "int swapped = atomicCAS(counter, exchanged, 4);" in hip_code
        assert "atomicExchange(" not in hip_code
        assert "atomicCompareExchange(" not in hip_code
        assert "atomicCompSwap(" not in hip_code

    def test_synchronization_functions(self):
        """Test synchronization function generation"""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    barrier();
                    memoryBarrier();
                    workgroupBarrier();
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "__syncthreads();" in hip_code
        assert hip_code.count("__syncthreads();") == 2
        assert "__threadfence();" in hip_code
        assert "barrier();" not in hip_code
        assert "memoryBarrier();" not in hip_code
        assert "workgroupBarrier();" not in hip_code

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

    def test_double_math_builtins_emit_hip_double_functions(self):
        """Test HIP preserves double precision for scalar math builtins."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    double x = 4.0;
                    double y = 2.0;
                    double a = sqrt(x);
                    double b = sin(x);
                    double c = cos(x);
                    double d = pow(x, y);
                    double e = exp(x);
                    double f = log(x);
                    double g = exp2(x);
                    double h = log2(x);
                    double i = floor(x);
                    double j = ceil(x);
                    double k = round(x);
                    double l = trunc(x);
                    double m = inversesqrt(x);
                    double n = sqrt(sin(x) + cos(x));
                    float fx = 4.0;
                    float fa = sqrt(fx);
                    float fb = pow(fx, 2.0);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "double a = sqrt(x);" in hip_code
        assert "double b = sin(x);" in hip_code
        assert "double c = cos(x);" in hip_code
        assert "double d = pow(x, y);" in hip_code
        assert "double e = exp(x);" in hip_code
        assert "double f = log(x);" in hip_code
        assert "double g = exp2(x);" in hip_code
        assert "double h = log2(x);" in hip_code
        assert "double i = floor(x);" in hip_code
        assert "double j = ceil(x);" in hip_code
        assert "double k = round(x);" in hip_code
        assert "double l = trunc(x);" in hip_code
        assert "double m = (1.0 / sqrt(x));" in hip_code
        assert "double n = sqrt((sin(x) + cos(x)));" in hip_code
        assert "float fa = sqrtf(fx);" in hip_code
        assert "float fb = powf(fx, 2.0);" in hip_code
        assert "sqrtf(x)" not in hip_code
        assert "sinf(x)" not in hip_code
        assert "cosf(x)" not in hip_code
        assert "powf(x, y)" not in hip_code
        assert "expf(x)" not in hip_code
        assert "logf(x)" not in hip_code
        assert "floorf(x)" not in hip_code
        assert "ceilf(x)" not in hip_code
        assert "roundf(x)" not in hip_code
        assert "truncf(x)" not in hip_code

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

    def test_cube_array_texture_calls_emit_hip_texture_functions(self):
        """Test HIP maps cube-array sampled texture calls by resource type."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "hipTextureObject_t probes;" in hip_code
        assert (
            "float4 color = texCubemapLayered<float4>"
            "(paramProbes, dirLayer.x, dirLayer.y, dirLayer.z, dirLayer.w);" in hip_code
        )
        assert (
            "float4 mip = texCubemapLayeredLod<float4>"
            "(paramProbes, dirLayer.x, dirLayer.y, dirLayer.z, dirLayer.w, 2.0);"
            in hip_code
        )
        assert (
            "float4 grad = texCubemapLayeredGrad<float4>"
            "(paramProbes, dirLayer.x, dirLayer.y, dirLayer.z, dirLayer.w, "
            "ddxCube, ddyCube);" in hip_code
        )
        assert " = texture(" not in hip_code
        assert " = textureLod(" not in hip_code
        assert " = textureGrad(" not in hip_code

    def test_shadow_sampler_calls_emit_hip_diagnostics(self):
        """Test HIP makes unsupported shadow sampler calls explicit."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert (
            "float paramValue = /* unsupported HIP shadow resource call: "
            "textureCompare on sampler2DShadow */ 0.0f;" in hip_code
        )
        assert (
            "float shadow = /* unsupported HIP shadow resource call: "
            "texture on sampler2DShadow */ 0.0f;" in hip_code
        )
        assert (
            "float cascade = /* unsupported HIP shadow resource call: "
            "texture on sampler2DArrayShadow */ 0.0f;" in hip_code
        )
        assert (
            "float cube = /* unsupported HIP shadow resource call: "
            "texture on samplerCubeShadow */ 0.0f;" in hip_code
        )
        assert (
            "float cubeArray = /* unsupported HIP shadow resource call: "
            "texture on samplerCubeArrayShadow */ 0.0f;" in hip_code
        )
        assert (
            "float compared = /* unsupported HIP shadow resource call: "
            "textureCompare on sampler2DShadow */ 0.0f;" in hip_code
        )
        assert (
            "float comparedLod = /* unsupported HIP shadow resource call: "
            "textureCompareLod on sampler2DShadow */ 0.0f;" in hip_code
        )
        assert (
            "float comparedGrad = /* unsupported HIP shadow resource call: "
            "textureCompareGrad on sampler2DShadow */ 0.0f;" in hip_code
        )
        assert (
            "float comparedOffset = /* unsupported HIP shadow resource call: "
            "textureCompareOffset on sampler2DShadow */ 0.0f;" in hip_code
        )
        assert (
            "float4 gathered = /* unsupported HIP shadow resource call: "
            "textureGatherCompare on sampler2DShadow */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert (
            "float4 gatheredOffset = /* unsupported HIP shadow resource call: "
            "textureGatherCompareOffset on sampler2DShadow */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert "textureCompare(" not in hip_code
        assert "textureCompareLod(" not in hip_code
        assert "textureCompareGrad(" not in hip_code
        assert "textureCompareOffset(" not in hip_code
        assert "textureGatherCompare(" not in hip_code
        assert "textureGatherCompareOffset(" not in hip_code
        assert "texture(shadowMap" not in hip_code
        assert "texture(cascades" not in hip_code
        assert "texture(cubeShadow" not in hip_code
        assert "texture(cubeArrayShadow" not in hip_code
        assert " = tex2D(shadowMap" not in hip_code
        assert " = tex2D(cascades" not in hip_code
        assert " = tex2D(cubeShadow" not in hip_code
        assert " = tex2D(cubeArrayShadow" not in hip_code

    def test_texture_gather_calls_emit_hip_diagnostics(self):
        """Test HIP makes unsupported gather sampled-resource calls explicit."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert (
            "float4 gathered = /* unsupported HIP sampled resource call: "
            "textureGather on sampler2D */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert (
            "float4 gatheredComponent = /* unsupported HIP sampled resource call: "
            "textureGather on sampler2D */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert (
            "float4 gatheredParam = /* unsupported HIP sampled resource call: "
            "textureGather on sampler2D */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert (
            "float4 gatheredLayer = /* unsupported HIP sampled resource call: "
            "textureGather on sampler2DArray */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert (
            "float4 gatheredOffset = /* unsupported HIP sampled resource call: "
            "textureGatherOffset on sampler2D */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert (
            "float4 gatheredOffsets = /* unsupported HIP sampled resource call: "
            "textureGatherOffsets on sampler2D */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert (
            "float4 gatheredVolume = /* unsupported HIP sampled resource call: "
            "textureGather on sampler3D */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert "textureGather(" not in hip_code
        assert "textureGatherOffset(" not in hip_code
        assert "textureGatherOffsets(" not in hip_code

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
        assert "CglResourceQueryInfo" not in hip_code

    def test_nested_resource_arrays_emit_hip_texture_and_surface_calls(self):
        """Test HIP preserves nested resource-array indices in resource calls."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "texture<float4, 2> textureGrid[2][3];" in hip_code
        assert "hipTextureObject_t layerGrid[2][2];" in hip_code
        assert "hipTextureObject_t msGrid[2][2];" in hip_code
        assert "hipSurfaceObject_t imageGrid[2][4];" in hip_code
        assert "hipSurfaceObject_t counterGrid[2][2];" in hip_code
        assert (
            "__device__ void processNested("
            "texture<float4, 2> paramTextures[2][3], "
            "hipSurfaceObject_t paramImages[2][4], int layer, int slot, "
            "float2 uv, float3 uvLayer, int2 pixel, int3 pixelLayer, "
            "int sampleIndex)" in hip_code
        )
        assert "float4 sampled = tex2D(textureGrid[layer][slot], uv);" in hip_code
        assert (
            "float4 sampledLayer = tex2DLayered<float4>"
            "(layerGrid[layer][slot], uvLayer.x, uvLayer.y, uvLayer.z);" in hip_code
        )
        assert "float4 mip = tex2DLod(textureGrid[layer][slot], uv, 1.0);" in hip_code
        assert (
            "float4 grad = tex2DGrad(textureGrid[layer][slot], uv, uv, uv);" in hip_code
        )
        assert (
            "float4 sampledParam = tex2D(paramTextures[layer][slot], uv);" in hip_code
        )
        assert (
            "float4 gathered = /* unsupported HIP sampled resource call: "
            "textureGather on sampler2D */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert (
            "float4 fetched = tex2D(textureGrid[layer][slot], pixel.x, pixel.y);"
            in hip_code
        )
        assert (
            "float4 fetchedLayer = tex2DLayered<float4>"
            "(layerGrid[layer][slot], pixelLayer.x, pixelLayer.y, pixelLayer.z);"
            in hip_code
        )
        assert (
            "float4 fetchedMs = /* unsupported HIP multisample resource call: "
            "texelFetch on sampler2DMS */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert (
            "float4 color = cgl_surf2Dread<float4>"
            "(imageGrid[layer][slot], pixel.x * sizeof(float4), pixel.y);" in hip_code
        )
        assert (
            "float4 colorParam = cgl_surf2Dread<float4>"
            "(paramImages[layer][slot], pixel.x * sizeof(float4), pixel.y);" in hip_code
        )
        assert (
            "surf2Dwrite(color, imageGrid[layer][slot], "
            "pixel.x * sizeof(float4), pixel.y);" in hip_code
        )
        assert (
            "unsigned int count = cgl_surf2Dread<uint>"
            "(counterGrid[layer][slot], pixel.x * sizeof(uint), pixel.y);" in hip_code
        )
        assert (
            "surf2Dwrite(count, counterGrid[layer][slot], "
            "pixel.x * sizeof(uint), pixel.y);" in hip_code
        )
        assert "texture(" not in hip_code
        assert "textureLod(" not in hip_code
        assert "textureGrad(" not in hip_code
        assert "textureGather(" not in hip_code
        assert "texelFetch(" not in hip_code
        assert "imageLoad(" not in hip_code
        assert "imageStore(" not in hip_code
        assert "CglResourceQueryInfo" not in hip_code

    def test_dynamic_resource_array_params_emit_hip_pointer_shapes(self):
        """Test HIP preserves dynamic resource-array parameter shapes."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert (
            "__device__ void useDynamic(texture<float4, 2>* dynTextures, "
            "hipTextureObject_t* dynLayers, hipSurfaceObject_t* dynImages, "
            "hipSurfaceObject_t* dynCounters, int i, float2 uv, "
            "float3 uvLayer, int2 pixel)" in hip_code
        )
        assert (
            "__device__ void useDynamicGrid("
            "texture<float4, 2> (*dynGrid)[3], "
            "hipSurfaceObject_t (*dynGridImages)[3], int layer, int slot, "
            "float2 uv, int2 pixel)" in hip_code
        )
        assert "float4 sampled = tex2D(dynTextures[i], uv);" in hip_code
        assert (
            "float4 layered = tex2DLayered<float4>"
            "(dynLayers[i], uvLayer.x, uvLayer.y, uvLayer.z);" in hip_code
        )
        assert "float4 fetched = tex2D(dynTextures[i], pixel.x, pixel.y);" in hip_code
        assert (
            "float4 color = cgl_surf2Dread<float4>"
            "(dynImages[i], pixel.x * sizeof(float4), pixel.y);" in hip_code
        )
        assert (
            "surf2Dwrite(color, dynImages[i], pixel.x * sizeof(float4), pixel.y);"
            in hip_code
        )
        assert (
            "unsigned int count = cgl_surf2Dread<uint>"
            "(dynCounters[i], pixel.x * sizeof(uint), pixel.y);" in hip_code
        )
        assert (
            "surf2Dwrite(count, dynCounters[i], pixel.x * sizeof(uint), pixel.y);"
            in hip_code
        )
        assert "float4 sampled = tex2D(dynGrid[layer][slot], uv);" in hip_code
        assert (
            "float4 color = cgl_surf2Dread<float4>"
            "(dynGridImages[layer][slot], pixel.x * sizeof(float4), pixel.y);"
            in hip_code
        )
        assert (
            "surf2Dwrite(color, dynGridImages[layer][slot], "
            "pixel.x * sizeof(float4), pixel.y);" in hip_code
        )
        assert "texture<float4, 2>* dynGrid[3]" not in hip_code
        assert "hipSurfaceObject_t* dynGridImages[3]" not in hip_code
        assert "texture(" not in hip_code
        assert "texelFetch(" not in hip_code
        assert "imageLoad(" not in hip_code
        assert "imageStore(" not in hip_code
        assert "CglResourceQueryInfo" not in hip_code

    def test_forwarded_dynamic_resource_arrays_emit_hip_metadata_arguments(self):
        """Test HIP forwards query metadata sidecars through dynamic arrays."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "CglResourceQueryInfo textureGrid_metadata[2][3] = {};" in hip_code
        assert "CglResourceQueryInfo imageGrid_metadata[2][3] = {};" in hip_code
        assert "counterGrid_metadata" not in hip_code
        assert (
            "__device__ float4 leafSample("
            "texture<float4, 2> (*dynGrid)[3], int layer, int slot, float2 uv)"
            in hip_code
        )
        assert (
            "__device__ float4 forwardSample("
            "texture<float4, 2> (*dynGrid)[3], int layer, int slot, float2 uv)"
            in hip_code
        )
        assert "return leafSample(dynGrid, layer, slot, uv);" in hip_code
        assert (
            "__device__ float4 leafImage(hipSurfaceObject_t (*dynImages)[3], "
            "int layer, int slot, int2 pixel)" in hip_code
        )
        assert "return leafImage(dynImages, layer, slot, pixel);" in hip_code
        assert (
            "__device__ unsigned int leafCounter("
            "hipSurfaceObject_t (*dynCounters)[2], int layer, int slot, int2 pixel)"
            in hip_code
        )
        assert "return leafCounter(dynCounters, layer, slot, pixel);" in hip_code
        assert (
            "__device__ void leafQuery(texture<float4, 2> (*dynGrid)[3], "
            "CglResourceQueryInfo (*dynGrid_metadata)[3], "
            "hipSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], int layer, int slot)"
            in hip_code
        )
        assert (
            "leafQuery(dynGrid, dynGrid_metadata, dynImages, "
            "dynImages_metadata, layer, slot);" in hip_code
        )
        assert (
            "forwardQuery(textureGrid, textureGrid_metadata, imageGrid, "
            "imageGrid_metadata, 1, 2);" in hip_code
        )
        assert "return tex2D(dynGrid[layer][slot], uv);" in hip_code
        assert (
            "float4 color = cgl_surf2Dread<float4>"
            "(dynImages[layer][slot], pixel.x * sizeof(float4), pixel.y);" in hip_code
        )
        assert (
            "unsigned int count = cgl_surf2Dread<uint>"
            "(dynCounters[layer][slot], pixel.x * sizeof(uint), pixel.y);" in hip_code
        )
        assert (
            "int2 texSize = cgl_textureSize_sampler2D"
            "(dynGrid_metadata[layer][slot], 0);" in hip_code
        )
        assert (
            "int2 imageSizeValue = cgl_imageSize_image2D"
            "(dynImages_metadata[layer][slot]);" in hip_code
        )
        assert "leafSample(dynGrid, dynGrid_metadata" not in hip_code
        assert "leafImage(dynImages, dynImages_metadata" not in hip_code
        assert "leafCounter(dynCounters, dynCounters_metadata" not in hip_code
        assert "texture(" not in hip_code
        assert "textureSize(" not in hip_code
        assert "imageSize(" not in hip_code
        assert "imageLoad(" not in hip_code
        assert "imageStore(" not in hip_code

    def test_mixed_fixed_dynamic_resource_arrays_emit_hip_metadata_arguments(self):
        """Test HIP forwards fixed rows and dynamic grids with metadata sidecars."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "CglResourceQueryInfo textureGrid_metadata[2][3] = {};" in hip_code
        assert "CglResourceQueryInfo imageGrid_metadata[2][3] = {};" in hip_code
        assert (
            "__device__ void leafMixed(texture<float4, 2> (*dynamicGrid)[3], "
            "CglResourceQueryInfo (*dynamicGrid_metadata)[3], "
            "texture<float4, 2> fixedRow[3], "
            "CglResourceQueryInfo fixedRow_metadata[3], "
            "hipSurfaceObject_t (*dynamicImages)[3], "
            "CglResourceQueryInfo (*dynamicImages_metadata)[3], "
            "hipSurfaceObject_t fixedImages[3], "
            "CglResourceQueryInfo fixedImages_metadata[3], "
            "int layer, int slot, float2 uv, int2 pixel)" in hip_code
        )
        assert (
            "leafMixed(dynamicGrid, dynamicGrid_metadata, fixedRow, "
            "fixedRow_metadata, dynamicImages, dynamicImages_metadata, "
            "fixedImages, fixedImages_metadata, layer, slot, uv, pixel);" in hip_code
        )
        assert (
            "forwardMixed(textureGrid, textureGrid_metadata, textureGrid[1], "
            "textureGrid_metadata[1], imageGrid, imageGrid_metadata, imageGrid[1], "
            "imageGrid_metadata[1], 1, 2, uv, pixel);" in hip_code
        )
        assert "float4 dynamicSample = tex2D(dynamicGrid[layer][slot], uv);" in hip_code
        assert "float4 fixedSample = tex2D(fixedRow[slot], uv);" in hip_code
        assert (
            "float4 dynamicColor = cgl_surf2Dread<float4>"
            "(dynamicImages[layer][slot], pixel.x * sizeof(float4), pixel.y);"
            in hip_code
        )
        assert (
            "float4 fixedColor = cgl_surf2Dread<float4>"
            "(fixedImages[slot], pixel.x * sizeof(float4), pixel.y);" in hip_code
        )
        assert (
            "int2 dynamicSize = cgl_textureSize_sampler2D"
            "(dynamicGrid_metadata[layer][slot], 0);" in hip_code
        )
        assert (
            "int2 fixedSize = cgl_textureSize_sampler2D"
            "(fixedRow_metadata[slot], 0);" in hip_code
        )
        assert (
            "int2 dynamicImageSize = cgl_imageSize_image2D"
            "(dynamicImages_metadata[layer][slot]);" in hip_code
        )
        assert (
            "int2 fixedImageSize = cgl_imageSize_image2D"
            "(fixedImages_metadata[slot]);" in hip_code
        )
        assert "texture(" not in hip_code
        assert "textureSize(" not in hip_code
        assert "imageSize(" not in hip_code
        assert "imageLoad(" not in hip_code
        assert "imageStore(" not in hip_code

    def test_multihop_reordered_resource_arrays_emit_hip_metadata_arguments(self):
        """Test HIP forwards reordered rows/grids and duplicate metadata sidecars."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "CglResourceQueryInfo textureGrid_metadata[2][3] = {};" in hip_code
        assert "CglResourceQueryInfo imageGrid_metadata[2][3] = {};" in hip_code
        assert (
            "__device__ void leafShuffle(texture<float4, 2> (*dynA)[3], "
            "CglResourceQueryInfo (*dynA_metadata)[3], "
            "hipSurfaceObject_t fixedImageRow[3], "
            "CglResourceQueryInfo fixedImageRow_metadata[3], "
            "texture<float4, 2> fixedTexRow[3], "
            "CglResourceQueryInfo fixedTexRow_metadata[3], "
            "hipSurfaceObject_t (*dynImageGrid)[3], "
            "CglResourceQueryInfo (*dynImageGrid_metadata)[3], "
            "texture<float4, 2> (*dynAlias)[3], "
            "CglResourceQueryInfo (*dynAlias_metadata)[3], "
            "int layer, int slot, float2 uv, int2 pixel)" in hip_code
        )
        assert (
            "__device__ void midForward(texture<float4, 2> (*firstDyn)[3], "
            "CglResourceQueryInfo (*firstDyn_metadata)[3], "
            "texture<float4, 2> firstFixed[3], "
            "CglResourceQueryInfo firstFixed_metadata[3], "
            "hipSurfaceObject_t (*firstDynImages)[3], "
            "CglResourceQueryInfo (*firstDynImages_metadata)[3], "
            "hipSurfaceObject_t firstFixedImages[3], "
            "CglResourceQueryInfo firstFixedImages_metadata[3], "
            "int layer, int slot, float2 uv, int2 pixel)" in hip_code
        )
        assert (
            "__device__ void topForward(texture<float4, 2> topFixedTex[3], "
            "CglResourceQueryInfo topFixedTex_metadata[3], "
            "hipSurfaceObject_t (*topDynImages)[3], "
            "CglResourceQueryInfo (*topDynImages_metadata)[3], "
            "texture<float4, 2> (*topDynTex)[3], "
            "CglResourceQueryInfo (*topDynTex_metadata)[3], "
            "hipSurfaceObject_t topFixedImages[3], "
            "CglResourceQueryInfo topFixedImages_metadata[3], "
            "int layer, int slot, float2 uv, int2 pixel)" in hip_code
        )
        assert (
            "leafShuffle(firstDyn, firstDyn_metadata, firstFixedImages, "
            "firstFixedImages_metadata, firstFixed, firstFixed_metadata, "
            "firstDynImages, firstDynImages_metadata, firstDyn, "
            "firstDyn_metadata, layer, slot, uv, pixel);" in hip_code
        )
        assert (
            "midForward(topDynTex, topDynTex_metadata, topFixedTex, "
            "topFixedTex_metadata, topDynImages, topDynImages_metadata, "
            "topFixedImages, topFixedImages_metadata, layer, slot, uv, pixel);"
            in hip_code
        )
        assert (
            "topForward(textureGrid[1], textureGrid_metadata[1], imageGrid, "
            "imageGrid_metadata, textureGrid, textureGrid_metadata, imageGrid[1], "
            "imageGrid_metadata[1], 1, 2, uv, pixel);" in hip_code
        )
        assert "float4 sampledA = tex2D(dynA[layer][slot], uv);" in hip_code
        assert "float4 sampledFixed = tex2D(fixedTexRow[slot], uv);" in hip_code
        assert "float4 sampledAlias = tex2D(dynAlias[layer][slot], uv);" in hip_code
        assert (
            "float4 dynamicColor = cgl_surf2Dread<float4>"
            "(dynImageGrid[layer][slot], pixel.x * sizeof(float4), pixel.y);"
            in hip_code
        )
        assert (
            "float4 fixedColor = cgl_surf2Dread<float4>"
            "(fixedImageRow[slot], pixel.x * sizeof(float4), pixel.y);" in hip_code
        )
        assert (
            "int2 dynSize = cgl_textureSize_sampler2D"
            "(dynA_metadata[layer][slot], 0);" in hip_code
        )
        assert (
            "int2 fixedTexSize = cgl_textureSize_sampler2D"
            "(fixedTexRow_metadata[slot], 0);" in hip_code
        )
        assert (
            "int2 aliasSize = cgl_textureSize_sampler2D"
            "(dynAlias_metadata[layer][slot], 0);" in hip_code
        )
        assert (
            "int2 dynamicImageSize = cgl_imageSize_image2D"
            "(dynImageGrid_metadata[layer][slot]);" in hip_code
        )
        assert (
            "int2 fixedImageSize = cgl_imageSize_image2D"
            "(fixedImageRow_metadata[slot]);" in hip_code
        )
        assert "texture(" not in hip_code
        assert "textureSize(" not in hip_code
        assert "imageSize(" not in hip_code
        assert "imageLoad(" not in hip_code
        assert "imageStore(" not in hip_code

    def test_resource_calls_in_control_flow_emit_hip_metadata_arguments(self):
        """Test HIP forwards metadata in assignments, returns, and ternaries."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert (
            "__device__ float4 chooseBranch(texture<float4, 2> (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "texture<float4, 2> fixedTex[3], "
            "CglResourceQueryInfo fixedTex_metadata[3], "
            "hipSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], "
            "hipSurfaceObject_t fixedImages[3], "
            "CglResourceQueryInfo fixedImages_metadata[3], "
            "bool useFixed, int layer, int slot, float2 uv, int2 pixel)" in hip_code
        )
        assert (
            "float4 result = sampleDynamic(dynTex, dynTex_metadata, "
            "dynImages, dynImages_metadata, layer, slot, uv, pixel);" in hip_code
        )
        assert (
            "result = sampleFixed(fixedTex, fixedTex_metadata, fixedImages, "
            "fixedImages_metadata, slot, uv, pixel);" in hip_code
        )
        assert (
            "return sampleFixed(fixedTex, fixedTex_metadata, fixedImages, "
            "fixedImages_metadata, slot, uv, pixel);" in hip_code
        )
        assert (
            "return sampleDynamic(dynTex, dynTex_metadata, dynImages, "
            "dynImages_metadata, layer, slot, uv, pixel);" in hip_code
        )
        assert (
            "return (useFixed ? sampleFixed(fixedTex, fixedTex_metadata, "
            "fixedImages, fixedImages_metadata, slot, uv, pixel) : "
            "sampleDynamic(dynTex, dynTex_metadata, dynImages, "
            "dynImages_metadata, layer, slot, uv, pixel));" in hip_code
        )
        assert (
            "chooseBranch(textureGrid, textureGrid_metadata, textureGrid[1], "
            "textureGrid_metadata[1], imageGrid, imageGrid_metadata, imageGrid[1], "
            "imageGrid_metadata[1], true, 1, 2, uv, pixel);" in hip_code
        )
        assert (
            "chooseTernary(textureGrid, textureGrid_metadata, textureGrid[1], "
            "textureGrid_metadata[1], imageGrid, imageGrid_metadata, imageGrid[1], "
            "imageGrid_metadata[1], true, 1, 2, uv, pixel);" in hip_code
        )
        assert (
            "int2 texSize = cgl_textureSize_sampler2D"
            "(dynTex_metadata[layer][slot], 0);" in hip_code
        )
        assert (
            "int2 imageSizeValue = cgl_imageSize_image2D"
            "(dynImages_metadata[layer][slot]);" in hip_code
        )
        assert (
            "int2 texSize = cgl_textureSize_sampler2D(fixedTex_metadata[slot], 0);"
            in hip_code
        )
        assert (
            "int2 imageSizeValue = cgl_imageSize_image2D(fixedImages_metadata[slot]);"
            in hip_code
        )
        assert "texture(" not in hip_code
        assert "textureSize(" not in hip_code
        assert "imageSize(" not in hip_code
        assert "imageLoad(" not in hip_code

    def test_resource_calls_in_loops_emit_hip_metadata_arguments(self):
        """Test HIP forwards metadata through loop-carried resource indices."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert (
            "__device__ float4 sampleLoop(texture<float4, 2> (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "texture<float4, 2> fixedTex[3], "
            "CglResourceQueryInfo fixedTex_metadata[3], "
            "hipSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], "
            "hipSurfaceObject_t fixedImages[3], "
            "CglResourceQueryInfo fixedImages_metadata[3], "
            "int layer, float2 uv, int2 pixel)" in hip_code
        )
        assert "for (int slot = 0; (slot < 3); slot++)" in hip_code
        assert "continue;" in hip_code
        assert "break;" in hip_code
        assert "while ((fixedSlot < 3))" in hip_code
        assert "float4 sampled = tex2D(dynTex[layer][slot], uv);" in hip_code
        assert (
            "float4 loaded = cgl_surf2Dread<float4>"
            "(dynImages[layer][slot], pixel.x * sizeof(float4), pixel.y);" in hip_code
        )
        assert (
            "int2 texSize = cgl_textureSize_sampler2D"
            "(dynTex_metadata[layer][slot], 0);" in hip_code
        )
        assert (
            "int2 imageSizeValue = cgl_imageSize_image2D"
            "(dynImages_metadata[layer][slot]);" in hip_code
        )
        assert (
            "__device__ inline float4 cgl_float4_add(float4 lhs, float4 rhs)"
            in hip_code
        )
        assert (
            "accum = cgl_float4_add(accum, tex2D(fixedTex[fixedSlot], uv));" in hip_code
        )
        assert (
            "accum = cgl_float4_add(accum, cgl_surf2Dread<float4>"
            "(fixedImages[fixedSlot], pixel.x * sizeof(float4), pixel.y));" in hip_code
        )
        assert (
            "int2 fixedTexSize = cgl_textureSize_sampler2D"
            "(fixedTex_metadata[fixedSlot], 0);" in hip_code
        )
        assert (
            "int2 fixedImageSize = cgl_imageSize_image2D"
            "(fixedImages_metadata[fixedSlot]);" in hip_code
        )
        assert (
            "sampleLoop(textureGrid, textureGrid_metadata, textureGrid[1], "
            "textureGrid_metadata[1], imageGrid, imageGrid_metadata, imageGrid[1], "
            "imageGrid_metadata[1], 1, uv, pixel);" in hip_code
        )
        assert "texture(" not in hip_code
        assert "textureSize(" not in hip_code
        assert "imageSize(" not in hip_code
        assert "imageLoad(" not in hip_code

    def test_nested_loop_resource_indices_emit_hip_metadata_arguments(self):
        """Test HIP forwards metadata when both resource indices are loop-carried."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert (
            "__device__ float4 sampleNestedLoops("
            "texture<float4, 2> (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "hipSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], float2 uv, int2 pixel)"
            in hip_code
        )
        assert "for (int layer = 0; (layer < 2); layer++)" in hip_code
        assert "for (int slot = 0; (slot < 3); slot++)" in hip_code
        assert "break;" in hip_code
        assert "continue;" in hip_code
        assert "float4 sampled = tex2D(dynTex[layer][slot], uv);" in hip_code
        assert (
            "float4 loaded = cgl_surf2Dread<float4>"
            "(dynImages[layer][slot], pixel.x * sizeof(float4), pixel.y);" in hip_code
        )
        assert (
            "surf2Dwrite(loaded, dynImages[layer][slot], "
            "pixel.x * sizeof(float4), pixel.y);" in hip_code
        )
        assert (
            "int2 texSize = cgl_textureSize_sampler2D"
            "(dynTex_metadata[layer][slot], 0);" in hip_code
        )
        assert (
            "int2 imageSizeValue = cgl_imageSize_image2D"
            "(dynImages_metadata[layer][slot]);" in hip_code
        )
        assert (
            "sampleNestedLoops(textureGrid, textureGrid_metadata, imageGrid, "
            "imageGrid_metadata, uv, pixel);" in hip_code
        )
        assert "texture(" not in hip_code
        assert "textureSize(" not in hip_code
        assert "imageSize(" not in hip_code
        assert "imageLoad(" not in hip_code
        assert "imageStore(" not in hip_code

    def test_struct_and_scalar_params_preserve_hip_metadata_argument_order(self):
        """Test HIP keeps metadata aligned around non-resource parameters."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "struct SampleParams" in hip_code
        assert (
            "__device__ float4 samplePacked(texture<float4, 2> (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], SampleParams params, "
            "hipSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], float weight, "
            "texture<float4, 2> fixedTex[3], "
            "CglResourceQueryInfo fixedTex_metadata[3], "
            "hipSurfaceObject_t fixedImages[3], "
            "CglResourceQueryInfo fixedImages_metadata[3])" in hip_code
        )
        assert (
            "__device__ float4 passPacked(SampleParams params, "
            "texture<float4, 2> (*firstDyn)[3], "
            "CglResourceQueryInfo (*firstDyn_metadata)[3], float weight, "
            "hipSurfaceObject_t (*firstImages)[3], "
            "CglResourceQueryInfo (*firstImages_metadata)[3], "
            "texture<float4, 2> fixedTex[3], "
            "CglResourceQueryInfo fixedTex_metadata[3], "
            "hipSurfaceObject_t fixedImages[3], "
            "CglResourceQueryInfo fixedImages_metadata[3])" in hip_code
        )
        assert (
            "return samplePacked(firstDyn, firstDyn_metadata, params, firstImages, "
            "firstImages_metadata, weight, fixedTex, fixedTex_metadata, fixedImages, "
            "fixedImages_metadata);" in hip_code
        )
        assert (
            "passPacked(params, textureGrid, textureGrid_metadata, 0.5, imageGrid, "
            "imageGrid_metadata, textureGrid[1], textureGrid_metadata[1], "
            "imageGrid[1], imageGrid_metadata[1]);" in hip_code
        )
        assert (
            "float4 dynamicSample = tex2D"
            "(dynTex[params.layer][params.slot], params.uv);" in hip_code
        )
        assert (
            "float4 fixedSample = tex2D(fixedTex[params.slot], params.uv);" in hip_code
        )
        assert (
            "float4 dynamicColor = cgl_surf2Dread<float4>"
            "(dynImages[params.layer][params.slot], "
            "params.pixel.x * sizeof(float4), params.pixel.y);" in hip_code
        )
        assert (
            "float4 fixedColor = cgl_surf2Dread<float4>"
            "(fixedImages[params.slot], params.pixel.x * sizeof(float4), "
            "params.pixel.y);" in hip_code
        )
        assert (
            "int2 dynSize = cgl_textureSize_sampler2D"
            "(dynTex_metadata[params.layer][params.slot], 0);" in hip_code
        )
        assert (
            "int2 fixedSize = cgl_textureSize_sampler2D"
            "(fixedTex_metadata[params.slot], 0);" in hip_code
        )
        assert (
            "int2 dynImageSize = cgl_imageSize_image2D"
            "(dynImages_metadata[params.layer][params.slot]);" in hip_code
        )
        assert (
            "int2 fixedImageSize = cgl_imageSize_image2D"
            "(fixedImages_metadata[params.slot]);" in hip_code
        )
        assert "texture(" not in hip_code
        assert "textureSize(" not in hip_code
        assert "imageSize(" not in hip_code
        assert "imageLoad(" not in hip_code

    def test_resource_values_in_struct_returns_emit_hip_metadata_arguments(self):
        """Test HIP forwards metadata through struct assignment and constructors."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "struct SampleResult" in hip_code
        assert (
            "__device__ SampleResult buildAssigned("
            "texture<float4, 2> (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "hipSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], int layer, int slot, "
            "float2 uv, int2 pixel)" in hip_code
        )
        assert (
            "__device__ SampleResult buildConstructed("
            "texture<float4, 2> (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "hipSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], int layer, int slot, "
            "float2 uv, int2 pixel)" in hip_code
        )
        assert (
            "__device__ float4 consumeResults(texture<float4, 2> (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "hipSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], int layer, int slot, "
            "float2 uv, int2 pixel)" in hip_code
        )
        assert "result.sampled = tex2D(dynTex[layer][slot], uv);" in hip_code
        assert (
            "result.loaded = cgl_surf2Dread<float4>"
            "(dynImages[layer][slot], pixel.x * sizeof(float4), pixel.y);" in hip_code
        )
        assert (
            "result.texSize = cgl_textureSize_sampler2D"
            "(dynTex_metadata[layer][slot], 0);" in hip_code
        )
        assert (
            "result.imageSizeValue = cgl_imageSize_image2D"
            "(dynImages_metadata[layer][slot]);" in hip_code
        )
        assert (
            "return SampleResult(tex2D(dynTex[layer][slot], uv), "
            "cgl_surf2Dread<float4>(dynImages[layer][slot], "
            "pixel.x * sizeof(float4), pixel.y), "
            "cgl_textureSize_sampler2D(dynTex_metadata[layer][slot], 0), "
            "cgl_imageSize_image2D(dynImages_metadata[layer][slot]));" in hip_code
        )
        assert (
            "SampleResult assigned = buildAssigned(dynTex, dynTex_metadata, "
            "dynImages, dynImages_metadata, layer, slot, uv, pixel);" in hip_code
        )
        assert (
            "SampleResult constructed = buildConstructed(dynTex, dynTex_metadata, "
            "dynImages, dynImages_metadata, layer, slot, uv, pixel);" in hip_code
        )
        assert (
            "consumeResults(textureGrid, textureGrid_metadata, imageGrid, "
            "imageGrid_metadata, 1, 2, uv, pixel);" in hip_code
        )
        assert "texture(" not in hip_code
        assert "textureSize(" not in hip_code
        assert "imageSize(" not in hip_code
        assert "imageLoad(" not in hip_code

    def test_nested_struct_resource_assignments_emit_hip_metadata_arguments(self):
        """Test HIP handles resource calls assigned into nested struct members."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "struct SampleEnvelope" in hip_code
        assert (
            "__device__ SampleEnvelope buildNestedAssigned("
            "texture<float4, 2> (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "hipSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], int layer, int slot, "
            "float2 uv, int2 pixel)" in hip_code
        )
        assert "envelope.result.sampled = tex2D(dynTex[layer][slot], uv);" in hip_code
        assert (
            "envelope.result.loaded = cgl_surf2Dread<float4>"
            "(dynImages[layer][slot], pixel.x * sizeof(float4), pixel.y);" in hip_code
        )
        assert (
            "envelope.result.texSize = cgl_textureSize_sampler2D"
            "(dynTex_metadata[layer][slot], 0);" in hip_code
        )
        assert (
            "envelope.result.imageSizeValue = cgl_imageSize_image2D"
            "(dynImages_metadata[layer][slot]);" in hip_code
        )
        assert (
            "__device__ inline float4 cgl_float4_add(float4 lhs, float4 rhs)"
            in hip_code
        )
        assert (
            "envelope.bias = cgl_float4_add(tex2D(dynTex[layer][slot], uv), "
            "cgl_surf2Dread<float4>(dynImages[layer][slot], "
            "pixel.x * sizeof(float4), pixel.y));" in hip_code
        )
        assert (
            "SampleEnvelope envelope = buildNestedAssigned("
            "dynTex, dynTex_metadata, dynImages, dynImages_metadata, "
            "layer, slot, uv, pixel);" in hip_code
        )
        assert (
            "consumeNested(textureGrid, textureGrid_metadata, imageGrid, "
            "imageGrid_metadata, 1, 2, uv, pixel);" in hip_code
        )
        assert "texture(" not in hip_code
        assert "textureSize(" not in hip_code
        assert "imageSize(" not in hip_code
        assert "imageLoad(" not in hip_code

    def test_struct_member_array_resource_assignments_emit_hip_metadata_arguments(self):
        """Test HIP handles resource calls assigned into struct member arrays."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "SampleResult results[3];" in hip_code
        assert (
            "__device__ SampleEnvelope buildArrayEnvelope("
            "texture<float4, 2> (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "hipSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], int layer, int slot, "
            "float2 uv, int2 pixel)" in hip_code
        )
        assert (
            "envelope.results[slot].sampled = tex2D(dynTex[layer][slot], uv);"
            in hip_code
        )
        assert (
            "envelope.results[slot].loaded = cgl_surf2Dread<float4>"
            "(dynImages[layer][slot], pixel.x * sizeof(float4), pixel.y);" in hip_code
        )
        assert (
            "envelope.results[slot].texSize = cgl_textureSize_sampler2D"
            "(dynTex_metadata[layer][slot], 0);" in hip_code
        )
        assert (
            "envelope.results[slot].imageSizeValue = cgl_imageSize_image2D"
            "(dynImages_metadata[layer][slot]);" in hip_code
        )
        assert (
            "SampleEnvelope envelope = buildArrayEnvelope("
            "dynTex, dynTex_metadata, dynImages, dynImages_metadata, "
            "layer, slot, uv, pixel);" in hip_code
        )
        assert (
            "consumeArrayEnvelope(textureGrid, textureGrid_metadata, imageGrid, "
            "imageGrid_metadata, 1, 2, uv, pixel);" in hip_code
        )
        assert "texture(" not in hip_code
        assert "textureSize(" not in hip_code
        assert "imageSize(" not in hip_code
        assert "imageLoad(" not in hip_code

    def test_control_flow_struct_resource_assignments_emit_hip_metadata_arguments(self):
        """Test HIP keeps metadata aligned through branch and loop struct writes."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert (
            "__device__ SampleEnvelope fillControlled("
            "texture<float4, 2> (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "hipSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], int layer, int slot, "
            "bool useAlternate, float2 uv, int2 pixel)" in hip_code
        )
        assert "if (useAlternate)" in hip_code
        assert "else" in hip_code
        assert "for (int i = 0; (i < 3); i++)" in hip_code
        assert "continue;" in hip_code
        assert "break;" in hip_code
        assert "envelope.accum = make_float4(0.0, 0.0, 0.0, 0.0);" in hip_code
        assert "envelope.chosen = slot;" in hip_code
        assert "envelope.result.sampled = tex2D(dynTex[layer][slot], uv);" in hip_code
        assert (
            "envelope.result.loaded = cgl_surf2Dread<float4>"
            "(dynImages[layer][slot], pixel.x * sizeof(float4), pixel.y);" in hip_code
        )
        assert (
            "envelope.result.sampled = tex2D(dynTex[fallback][slot], uv);" in hip_code
        )
        assert (
            "envelope.result.loaded = cgl_surf2Dread<float4>"
            "(dynImages[fallback][slot], pixel.x * sizeof(float4), pixel.y);"
            in hip_code
        )
        assert (
            "envelope.result.texSize = cgl_textureSize_sampler2D"
            "(dynTex_metadata[layer][i], 0);" in hip_code
        )
        assert (
            "envelope.result.imageSizeValue = cgl_imageSize_image2D"
            "(dynImages_metadata[layer][i]);" in hip_code
        )
        assert (
            "envelope.accum = cgl_float4_add(envelope.accum, tex2D(dynTex[layer][i], uv));"
            in hip_code
        )
        assert (
            "fillControlled(dynTex, dynTex_metadata, dynImages, "
            "dynImages_metadata, layer, slot, true, uv, pixel);" in hip_code
        )
        assert (
            "consumeControlled(textureGrid, textureGrid_metadata, imageGrid, "
            "imageGrid_metadata, 1, 2, uv, pixel);" in hip_code
        )
        assert "texture(" not in hip_code
        assert "textureSize(" not in hip_code
        assert "imageSize(" not in hip_code
        assert "imageLoad(" not in hip_code

    def test_struct_parameter_resource_values_round_trip_hip_metadata_arguments(self):
        """Test HIP forwards resource-derived structs through helper parameters."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert (
            "__device__ SampleResult buildResourceResult("
            "texture<float4, 2> (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "hipSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], int layer, int slot, "
            "float2 uv, int2 pixel)" in hip_code
        )
        assert (
            "__device__ SampleResult adjustResult(SampleResult payload, float weight)"
            in hip_code
        )
        assert (
            "__device__ float4 consumeAdjusted(texture<float4, 2> (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "hipSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], int layer, int slot, "
            "float2 uv, int2 pixel)" in hip_code
        )
        assert "__device__ inline float4 cgl_float4_mul_scalar" in hip_code
        assert (
            "__device__ inline float4 cgl_float4_add(float4 lhs, float4 rhs)"
            in hip_code
        )
        assert (
            "payload.sampled = cgl_float4_mul_scalar(payload.sampled, weight);"
            in hip_code
        )
        assert (
            "payload.loaded = cgl_float4_add(payload.loaded, payload.sampled);"
            in hip_code
        )
        assert "return payload;" in hip_code
        assert (
            "SampleResult raw = buildResourceResult("
            "dynTex, dynTex_metadata, dynImages, dynImages_metadata, "
            "layer, slot, uv, pixel);" in hip_code
        )
        assert "SampleResult adjusted = adjustResult(raw, 0.5);" in hip_code
        assert (
            "consumeAdjusted(textureGrid, textureGrid_metadata, imageGrid, "
            "imageGrid_metadata, 1, 2, uv, pixel);" in hip_code
        )
        assert "texture(" not in hip_code
        assert "textureSize(" not in hip_code
        assert "imageSize(" not in hip_code
        assert "imageLoad(" not in hip_code

    def test_mutable_scalar_and_array_params_emit_hip_updates(self):
        """Test HIP emits scalar and array parameter updates directly."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "__device__ float bumpScalar(float weight)" in hip_code
        assert "weight = (weight + 1.0);" in hip_code
        assert "return (weight * 2.0);" in hip_code
        assert "__device__ float bumpArray(float values[3], int slot)" in hip_code
        assert "values[slot] = (values[slot] + 1.0);" in hip_code
        assert "values[0] = (values[slot] * 0.5);" in hip_code
        assert "return (values[slot] + values[0]);" in hip_code
        assert "float values[3];" in hip_code
        assert "float a = bumpScalar(0.5);" in hip_code
        assert "float b = bumpArray(values, 1);" in hip_code

    def test_nested_array_and_struct_member_params_emit_hip_updates(self):
        """Test HIP emits nested array and struct-member array parameter updates."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "struct Payload" in hip_code
        assert "float values[3];" in hip_code
        assert (
            "__device__ float bumpNested(float values[2][3], int row, int col)"
            in hip_code
        )
        assert "values[row][col] = (values[row][col] + 1.0);" in hip_code
        assert "values[0][col] = (values[row][col] * 0.5);" in hip_code
        assert "return (values[row][col] + values[0][col]);" in hip_code
        assert "__device__ float bumpPayload(Payload payload, int slot)" in hip_code
        assert (
            "payload.values[slot] = (payload.values[slot] + payload.bias);" in hip_code
        )
        assert "payload.bias = (payload.values[slot] * 0.5);" in hip_code
        assert "return (payload.values[slot] + payload.bias);" in hip_code
        assert "float grid[2][3];" in hip_code
        assert "float a = bumpNested(grid, 1, 2);" in hip_code
        assert "float b = bumpPayload(payload, 1);" in hip_code

    def test_returned_nested_struct_params_emit_hip_updates(self):
        """Test HIP mutates nested struct params built from returned values."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert (
            "__device__ OuterPayload adjustOuter(OuterPayload payload, int slot)"
            in hip_code
        )
        assert (
            "payload.inner.values[slot] = "
            "(payload.inner.values[slot] + payload.inner.bias);" in hip_code
        )
        assert (
            "payload.inner.values[0] = (payload.inner.values[slot] * payload.scale);"
            in hip_code
        )
        assert (
            "payload.scale = (payload.inner.values[0] + payload.inner.bias);"
            in hip_code
        )
        assert "return payload;" in hip_code
        assert "__device__ float consumeAdjustedOuter(int slot)" in hip_code
        assert (
            "OuterPayload adjusted = adjustOuter("
            "makeOuter(1.0, 2.0, 0.25, 4.0), slot);" in hip_code
        )
        assert (
            "return ((adjusted.inner.values[slot] + adjusted.inner.values[0]) + "
            "adjusted.scale);" in hip_code
        )
        assert "float value = consumeAdjustedOuter(1);" in hip_code

    def test_conditional_returned_nested_structs_emit_hip_expressions(self):
        """Test HIP preserves branch and ternary selected returned structs."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "__device__ OuterPayload chooseBranch(bool useSecond)" in hip_code
        assert "return makeOuter(3.0, 4.0, 0.5, 5.0);" in hip_code
        assert "return makeOuter(1.0, 2.0, 0.25, 4.0);" in hip_code
        assert "__device__ OuterPayload chooseTernary(bool useSecond)" in hip_code
        assert (
            "return (useSecond ? makeOuter(3.0, 4.0, 0.5, 5.0) : "
            "makeOuter(1.0, 2.0, 0.25, 4.0));" in hip_code
        )
        assert "OuterPayload branchPayload = chooseBranch(useSecond);" in hip_code
        assert "OuterPayload ternaryPayload = chooseTernary(useSecond);" in hip_code
        assert (
            "ternaryPayload.inner.values[slot] = "
            "(ternaryPayload.inner.values[slot] + ternaryPayload.inner.bias);"
            in hip_code
        )
        assert (
            "branchPayload.scale = "
            "(branchPayload.inner.values[slot] + ternaryPayload.scale);" in hip_code
        )
        assert (
            "return (branchPayload.scale + ternaryPayload.inner.values[slot]);"
            in hip_code
        )
        assert "float value = consumeConditional(1, true);" in hip_code

    def test_temporary_struct_array_member_reads_emit_hip_expressions(self):
        """Test HIP emits array-field reads from returned struct temporaries."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert (
            "__device__ Payload makePayload(float first, float second, float bias)"
            in hip_code
        )
        assert "payload.values[2] = (first + second);" in hip_code
        assert "__device__ float readTemporary(int slot)" in hip_code
        assert (
            "float dynamicValue = makePayload(1.0, 2.0, 0.25).values[slot];" in hip_code
        )
        assert "float fixedValue = makePayload(3.0, 4.0, 0.5).values[0];" in hip_code
        assert "float biasValue = makePayload(5.0, 6.0, 0.75).bias;" in hip_code
        assert "return ((dynamicValue + fixedValue) + biasValue);" in hip_code
        assert "float value = readTemporary(1);" in hip_code

    def test_nested_temporary_struct_array_member_reads_emit_hip_expressions(self):
        """Test HIP emits nested array-field reads from returned struct temporaries."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "struct InnerPayload" in hip_code
        assert "struct OuterPayload" in hip_code
        assert (
            "__device__ OuterPayload makeOuter("
            "float first, float second, float bias, float scale)" in hip_code
        )
        assert "outer.inner.values[2] = (first + second);" in hip_code
        assert "outer.inner.bias = bias;" in hip_code
        assert "outer.scale = scale;" in hip_code
        assert "__device__ float readNestedTemporary(int slot)" in hip_code
        assert (
            "float dynamicValue = "
            "makeOuter(1.0, 2.0, 0.25, 4.0).inner.values[slot];" in hip_code
        )
        assert (
            "float fixedValue = makeOuter(3.0, 4.0, 0.5, 5.0).inner.values[0];"
            in hip_code
        )
        assert (
            "float biasValue = makeOuter(5.0, 6.0, 0.75, 6.0).inner.bias;" in hip_code
        )
        assert "float scaleValue = makeOuter(7.0, 8.0, 1.0, 9.0).scale;" in hip_code
        assert (
            "return (((dynamicValue + fixedValue) + biasValue) + scaleValue);"
            in hip_code
        )

    def test_returned_local_nested_struct_array_writes_emit_hip_expressions(self):
        """Test HIP mutates locals initialized from returned nested structs."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "OuterPayload outer = makeOuter(1.0, 2.0, 0.25, 4.0);" in hip_code
        assert (
            "outer.inner.values[slot] = "
            "(outer.inner.values[slot] + outer.inner.bias);" in hip_code
        )
        assert (
            "outer.inner.values[0] = (outer.inner.values[slot] * outer.scale);"
            in hip_code
        )
        assert "outer.scale = (outer.inner.values[0] + outer.inner.bias);" in hip_code
        assert (
            "return ((outer.inner.values[slot] + outer.inner.values[0]) + outer.scale);"
            in hip_code
        )
        assert "float value = mutateReturnedLocal(1);" in hip_code

    def test_returned_local_nested_struct_array_writes_in_control_flow_emit_hip(
        self,
    ):
        """Test HIP preserves returned nested struct mutations inside control flow."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "__device__ float mutateReturnedLocalControl(int slot)" in hip_code
        assert "OuterPayload outer = makeOuter(1.0, 2.0, 0.25, 4.0);" in hip_code
        assert "for (int i = 0; (i < 3); i++)" in hip_code
        assert (
            "outer.inner.values[i] = "
            "(outer.inner.values[i] + outer.inner.bias);" in hip_code
        )
        assert "if ((i == slot))" in hip_code
        assert (
            "outer.inner.values[i] = (outer.inner.values[i] * outer.scale);" in hip_code
        )
        assert "if ((slot > 1))" in hip_code
        assert (
            "outer.scale = (outer.inner.values[slot] + outer.inner.bias);" in hip_code
        )
        assert "outer.scale = (outer.inner.values[0] - outer.inner.bias);" in hip_code
        assert "return (outer.inner.values[slot] + outer.scale);" in hip_code
        assert "float value = mutateReturnedLocalControl(1);" in hip_code

    def test_image_atomic_builtins_emit_hip_diagnostics(self):
        """Test HIP makes unsupported storage image atomics explicit."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert (
            "int added = /* unsupported HIP image atomic resource call: "
            "imageAtomicAdd on iimage2D */ 0;" in hip_code
        )
        assert (
            "unsigned int exchanged = /* unsupported HIP image atomic resource call: "
            "imageAtomicExchange on uimage2D */ 0u;" in hip_code
        )
        assert (
            "int compared = /* unsupported HIP image atomic resource call: "
            "imageAtomicCompSwap on iimage2D */ 0;" in hip_code
        )
        assert (
            "int volumeMin = /* unsupported HIP image atomic resource call: "
            "imageAtomicMin on iimage3D */ 0;" in hip_code
        )
        assert (
            "unsigned int layerMax = /* unsupported HIP image atomic resource call: "
            "imageAtomicMax on uimage2DArray */ 0u;" in hip_code
        )
        assert (
            "unsigned int vectorAtomic = /* unsupported HIP image atomic resource call: "
            "imageAtomicAdd on image2D */ 0;" in hip_code
        )
        assert (
            "int missingArgs = /* unsupported HIP image atomic resource call: "
            "imageAtomicAdd on iimage2D */ 0;" in hip_code
        )
        assert "imageAtomicAdd(" not in hip_code
        assert "imageAtomicExchange(" not in hip_code
        assert "imageAtomicCompSwap(" not in hip_code
        assert "imageAtomicMin(" not in hip_code
        assert "imageAtomicMax(" not in hip_code

    def test_resource_query_builtins_emit_hip_metadata_helpers(self):
        """Test HIP lowers resource queries through explicit metadata sidecars."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "struct CglResourceQueryInfo" in hip_code
        assert "__device__ inline int cgl_lod_extent" in hip_code
        assert (
            "__device__ void queryParam(texture<float4, 2> paramTex, "
            "CglResourceQueryInfo paramTex_metadata)" in hip_code
        )
        assert "CglResourceQueryInfo colorMap_metadata = {};" in hip_code
        assert "CglResourceQueryInfo cubeArrayTex_metadata = {};" in hip_code
        assert "CglResourceQueryInfo counters_metadata = {};" in hip_code
        assert (
            "return make_int3(cgl_lod_extent(info.width, mipLevel), "
            "cgl_lod_extent(info.height, mipLevel), info.elements);" in hip_code
        )
        assert "return info.levels > 0 ? info.levels : 1;" in hip_code
        assert (
            "int2 paramSize = cgl_textureSize_sampler2D(paramTex_metadata, 0);"
            in hip_code
        )
        assert (
            "int paramLevels = cgl_textureQueryLevels_sampler2D(paramTex_metadata);"
            in hip_code
        )
        assert (
            "int2 texSize = cgl_textureSize_sampler2D(colorMap_metadata, 2);"
            in hip_code
        )
        assert (
            "int3 layerSize = cgl_textureSize_sampler2DArray(layers_metadata, 1);"
            in hip_code
        )
        assert (
            "int3 volumeSize = cgl_textureSize_sampler3D(volumeTex_metadata, 0);"
            in hip_code
        )
        assert (
            "int2 cubeSize = cgl_textureSize_samplerCube(cubeTex_metadata, 0);"
            in hip_code
        )
        assert (
            "int3 cubeArraySize = cgl_textureSize_samplerCubeArray"
            "(cubeArrayTex_metadata, 0);" in hip_code
        )
        assert (
            "int texLevels = cgl_textureQueryLevels_sampler2D(colorMap_metadata);"
            in hip_code
        )
        assert (
            "int layerLevels = cgl_textureQueryLevels_sampler2DArray"
            "(layers_metadata);" in hip_code
        )
        assert (
            "int cubeLevels = cgl_textureQueryLevels_samplerCube(cubeTex_metadata);"
            in hip_code
        )
        assert (
            "int cubeArrayLevels = cgl_textureQueryLevels_samplerCubeArray"
            "(cubeArrayTex_metadata);" in hip_code
        )
        assert "int2 msSize = cgl_textureSize_sampler2DMS(msTex_metadata);" in hip_code
        assert (
            "int3 msLayerSize = cgl_textureSize_sampler2DMSArray"
            "(msLayers_metadata);" in hip_code
        )
        assert (
            "int2 imageSize2d = cgl_imageSize_image2D(colorImage_metadata);" in hip_code
        )
        assert (
            "int3 imageSize3d = cgl_imageSize_image3D(volumeImage_metadata);"
            in hip_code
        )
        assert (
            "int3 imageSizeLayer = cgl_imageSize_image2DArray(layerImage_metadata);"
            in hip_code
        )
        assert (
            "int2 msImageSize = cgl_imageSize_image2DMS(msImage_metadata);" in hip_code
        )
        assert (
            "int3 msImageLayerSize = cgl_imageSize_image2DMSArray"
            "(msImageLayers_metadata);" in hip_code
        )
        assert (
            "int texSamples = cgl_textureSamples_sampler2DMS(msTex_metadata);"
            in hip_code
        )
        assert (
            "int texLayerSamples = cgl_textureSamples_sampler2DMSArray"
            "(msLayers_metadata);" in hip_code
        )
        assert (
            "int imageSamplesValue = cgl_imageSamples_image2DMS(msImage_metadata);"
            in hip_code
        )
        assert (
            "int counterSamples = cgl_imageSamples_uimage2DMS(counters_metadata);"
            in hip_code
        )
        assert "textureSize(" not in hip_code
        assert "imageSize(" not in hip_code
        assert "textureSamples(" not in hip_code
        assert "imageSamples(" not in hip_code
        assert "textureQueryLevels(" not in hip_code

    def test_texture_query_lod_emits_hip_diagnostics(self):
        """Test HIP makes unsupported textureQueryLod explicit."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert (
            "float2 paramLod = /* unsupported HIP resource query: "
            "textureQueryLod on sampler2D */ make_float2(0.0f, 0.0f);" in hip_code
        )
        assert (
            "float2 lod = /* unsupported HIP resource query: "
            "textureQueryLod on sampler2D */ "
            "make_float2(0.0f, 0.0f);" in hip_code
        )
        assert (
            "float2 arrayLod = /* unsupported HIP resource query: "
            "textureQueryLod on sampler2DArray */ "
            "make_float2(0.0f, 0.0f);" in hip_code
        )
        assert (
            "float2 cubeLod = /* unsupported HIP resource query: "
            "textureQueryLod on samplerCube */ make_float2(0.0f, 0.0f);" in hip_code
        )
        assert "CglResourceQueryInfo" not in hip_code
        assert "textureQueryLod(" not in hip_code

    def test_resource_query_arrays_emit_indexed_hip_metadata(self):
        """Test HIP resource queries preserve resource-array metadata indexing."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "CglResourceQueryInfo textures_metadata[4] = {};" in hip_code
        assert "CglResourceQueryInfo msTextures_metadata[2] = {};" in hip_code
        assert "CglResourceQueryInfo images_metadata[3] = {};" in hip_code
        assert (
            "__device__ void queryArrays(texture<float4, 2> paramTextures[3], "
            "CglResourceQueryInfo paramTextures_metadata[3], int i)" in hip_code
        )
        assert (
            "int2 texSize = cgl_textureSize_sampler2D"
            "(textures_metadata[(1 + 2)], 0);" in hip_code
        )
        assert (
            "int texLevels = cgl_textureQueryLevels_sampler2D"
            "(textures_metadata[(1 + 2)]);" in hip_code
        )
        assert (
            "int msSamples = cgl_textureSamples_sampler2DMS(msTextures_metadata[i]);"
            in hip_code
        )
        assert (
            "int2 imageSizeValue = cgl_imageSize_image2D(images_metadata[i]);"
            in hip_code
        )
        assert (
            "int2 paramSize = cgl_textureSize_sampler2D"
            "(paramTextures_metadata[i], 0);" in hip_code
        )
        assert "cgl_textureSize_sampler2D(textures_metadata, 0)" not in hip_code
        assert "cgl_textureSamples_sampler2DMS(msTextures_metadata)" not in hip_code
        assert "cgl_imageSize_image2D(images_metadata)" not in hip_code

    def test_dynamic_and_nested_resource_query_arrays_emit_hip_metadata(self):
        """Test HIP metadata sidecars cover dynamic and nested resource arrays."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "CglResourceQueryInfo textureGrid_metadata[2][3] = {};" in hip_code
        assert "CglResourceQueryInfo imageGrid_metadata[2][4] = {};" in hip_code
        assert (
            "__device__ void queryArrays(texture<float4, 2>* dynamicTextures, "
            "CglResourceQueryInfo* dynamicTextures_metadata, "
            "texture<float4, 2> fixedTextures[3], "
            "CglResourceQueryInfo fixedTextures_metadata[3], "
            "hipTextureObject_t* dynamicMsTextures, "
            "CglResourceQueryInfo* dynamicMsTextures_metadata, "
            "texture<float4, 2> (*dynamicGrid)[3], "
            "CglResourceQueryInfo (*dynamicGrid_metadata)[3], "
            "hipSurfaceObject_t (*dynamicImageGrid)[4], "
            "CglResourceQueryInfo (*dynamicImageGrid_metadata)[4], "
            "int layer, int slot)" in hip_code
        )
        assert (
            "int2 dynamicSize = cgl_textureSize_sampler2D"
            "(dynamicTextures_metadata[slot], 0);" in hip_code
        )
        assert (
            "int fixedLevels = cgl_textureQueryLevels_sampler2D"
            "(fixedTextures_metadata[slot]);" in hip_code
        )
        assert (
            "int dynamicSamples = cgl_textureSamples_sampler2DMS"
            "(dynamicMsTextures_metadata[slot]);" in hip_code
        )
        assert (
            "int2 dynamicGridSize = cgl_textureSize_sampler2D"
            "(dynamicGrid_metadata[layer][slot], 0);" in hip_code
        )
        assert (
            "int2 dynamicImageSize = cgl_imageSize_image2D"
            "(dynamicImageGrid_metadata[layer][slot]);" in hip_code
        )
        assert (
            "int2 gridSize = cgl_textureSize_sampler2D"
            "(textureGrid_metadata[layer][slot], 0);" in hip_code
        )
        assert (
            "int2 imageGridSize = cgl_imageSize_image2D"
            "(imageGrid_metadata[layer][slot]);" in hip_code
        )
        assert "cgl_textureSize_sampler2D(dynamicTextures_metadata, 0)" not in hip_code
        assert (
            "cgl_textureQueryLevels_sampler2D(fixedTextures_metadata)" not in hip_code
        )
        assert "cgl_textureSize_sampler2D(textureGrid_metadata, 0)" not in hip_code
        assert "cgl_textureSize_sampler2D(dynamicGrid_metadata, 0)" not in hip_code
        assert "cgl_imageSize_image2D(dynamicImageGrid_metadata)" not in hip_code
        assert "cgl_imageSize_image2D(imageGrid_metadata)" not in hip_code

    def test_multisample_resource_builtins_emit_hip_diagnostics(self):
        """Test HIP does not silently lower MS resource calls to non-MS functions."""
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

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "hipTextureObject_t msTex;" in hip_code
        assert "hipTextureObject_t msLayers;" in hip_code
        assert "hipSurfaceObject_t msImage;" in hip_code
        assert "hipSurfaceObject_t msImageLayers;" in hip_code
        assert "hipSurfaceObject_t msCounters;" in hip_code
        assert "hipSurfaceObject_t msSignedLayers;" in hip_code
        assert (
            "float4 sampled = /* unsupported HIP multisample resource call: "
            "texture on sampler2DMS */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert (
            "float4 fetched = /* unsupported HIP multisample resource call: "
            "texelFetch on sampler2DMS */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert (
            "float4 fetchedLayer = /* unsupported HIP multisample resource call: "
            "texelFetch on sampler2DMSArray */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert (
            "float4 loaded = /* unsupported HIP multisample resource call: "
            "imageLoad on image2DMS */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert (
            "/* unsupported HIP multisample resource call: imageStore on image2DMS */ "
            "((void)0);" in hip_code
        )
        assert (
            "float4 loadedLayer = /* unsupported HIP multisample resource call: "
            "imageLoad on image2DMSArray */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert (
            "/* unsupported HIP multisample resource call: imageStore on image2DMSArray */ "
            "((void)0);" in hip_code
        )
        assert (
            "unsigned int loadedCounter = /* unsupported HIP multisample resource call: "
            "imageLoad on uimage2DMS */ 0u;" in hip_code
        )
        assert (
            "int loadedSignedLayer = /* unsupported HIP multisample resource call: "
            "imageLoad on iimage2DMSArray */ 0;" in hip_code
        )
        assert "imageStore on uimage2DMS */ ((void)0);" in hip_code
        assert "imageStore on iimage2DMSArray */ ((void)0);" in hip_code
        assert (
            "unsigned int msAtomic = /* unsupported HIP image atomic resource call: "
            "imageAtomicAdd on uimage2DMS */ 0u;" in hip_code
        )
        assert (
            "int msLayerAtomic = /* unsupported HIP image atomic resource call: "
            "imageAtomicCompSwap on iimage2DMSArray */ 0;" in hip_code
        )
        assert "imageAtomicAdd(" not in hip_code
        assert "imageAtomicCompSwap(" not in hip_code
        assert "texture(msTex" not in hip_code
        assert "texelFetch(msTex" not in hip_code
        assert "texelFetch(msLayers" not in hip_code
        assert "imageLoad(msImage" not in hip_code
        assert "imageLoad(msImageLayers" not in hip_code
        assert "imageStore(msImage" not in hip_code
        assert "imageStore(msImageLayers" not in hip_code
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

    def test_do_while_loop_generation(self):
        """Test HIP emits do-while loops."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    int value = 0;
                    do {
                        value += 1;
                    } while (value < 4);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "\n    do\n    {\n" in hip_code
        assert "value += 1;" in hip_code
        assert "} while ((value < 4));" in hip_code
        assert "DoWhileNode" not in hip_code

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

    def test_match_literal_and_wildcard_arms_lower_to_switch(self):
        """Test HIP lowers simple match arms to switch cases."""
        source_code = """
        shader TestShader {
            compute {
                int main(int mode) {
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
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "switch (mode)" in hip_code
        assert "case 0:" in hip_code
        assert "value = 1;" in hip_code
        assert "default:" in hip_code
        assert "value = 2;" in hip_code
        assert hip_code.count("break;") == 2
        assert "MatchNode" not in hip_code

    def test_match_guarded_arm_rejected_for_hip_switch_lowering(self):
        """Test HIP rejects match forms that cannot be lowered to switch."""
        source_code = """
        shader TestShader {
            compute {
                int main(int mode) {
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
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        with pytest.raises(ValueError, match="Unsupported match arm for HIP"):
            codegen.generate(ast)

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

    def test_inferred_let_declarations_emit_inferred_hip_types(self):
        """Test typeless let declarations infer HIP types from literal initializers."""
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    let scalar = 1.0;
                    let flag = true;
                    let label = "debug";
                    let marker = 'x';
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "float scalar = 1.0;" in hip_code
        assert "bool flag = true;" in hip_code
        assert 'string label = "debug";' in hip_code
        assert "char marker = 'x';" in hip_code
        assert "None scalar" not in hip_code
        assert "None flag" not in hip_code

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

    def test_array_literals_emit_hip_brace_initializers(self):
        """Test HIP lowers parsed array literals to C-style initializers."""
        source_code = """
        float globalWeights[4] = {1.0, 2.0};

        float pickScalar(int index) {
            float values[4] = {1.0, 2.0, 3.0, 4.0};
            return values[index];
        }

        vec3 pickColor(int index) {
            vec3 colors[2] = {vec3(1.0, 2.0, 3.0), vec3(4.0, 5.0, 6.0)};
            return colors[index];
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "float globalWeights[4] = {1.0, 2.0};" in hip_code
        assert "float values[4] = {1.0, 2.0, 3.0, 4.0};" in hip_code
        assert (
            "float3 colors[2] = {make_float3(1.0, 2.0, 3.0), "
            "make_float3(4.0, 5.0, 6.0)};"
        ) in hip_code
        assert "ArrayLiteralNode" not in hip_code

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
