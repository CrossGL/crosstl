import shutil
import subprocess

import pytest

from crosstl.translator.ast import (
    ArrayAccessNode,
    AssignmentNode,
    AttributeNode,
    BlockNode,
    ConstructorPatternNode,
    ExecutionModel,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    LiteralNode,
    MatchArmNode,
    MatchNode,
    MemberAccessNode,
    ParameterNode,
    PrimitiveType,
    ReturnNode,
    ShaderNode,
    ShaderStage,
    StructNode,
    VariableNode,
    WaveOpNode,
    WildcardPatternNode,
)
from crosstl.translator.codegen.hip_codegen import HipCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def compile_hip_if_hipcc_available(hip_code, tmp_path):
    """Compile generated HIP when hipcc is available in the local environment."""
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip("hipcc is not installed")

    source_path = tmp_path / "generated.hip"
    object_path = tmp_path / "generated.o"
    source_path.write_text(hip_code, encoding="utf-8")

    result = subprocess.run(
        [hipcc, "-std=c++17", "-c", str(source_path), "-o", str(object_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + "\n\n" + hip_code


def compile_matrix_helpers_if_cxx_available(helper_code, tmp_path):
    """Compile CUDA/HIP-neutral matrix helper overloads with a C++ compiler."""
    compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        return

    source_path = tmp_path / "matrix_helpers.cpp"
    object_path = tmp_path / "matrix_helpers.o"
    source_path.write_text(
        "\n".join(
            [
                "#define __host__",
                "#define __device__",
                "struct float2 { float x; float y; };",
                "struct float3 { float x; float y; float z; };",
                "struct float4 { float x; float y; float z; float w; };",
                "struct double2 { double x; double y; };",
                "struct double3 { double x; double y; double z; };",
                "struct double4 { double x; double y; double z; double w; };",
                "inline float2 make_float2(float x, float y) { return {x, y}; }",
                "inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }",
                "inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }",
                "inline double2 make_double2(double x, double y) { return {x, y}; }",
                "inline double3 make_double3(double x, double y, double z) { return {x, y, z}; }",
                "inline double4 make_double4(double x, double y, double z, double w) { return {x, y, z, w}; }",
                helper_code,
                "int main() {",
                "    float2x2 a(1.0f);",
                "    float2x2 b(1.0f, 2.0f, 3.0f, 4.0f);",
                "    float2x2 c = a + b;",
                "    c += b;",
                "    c *= 2.0f;",
                "    float2x2 d = a * b;",
                "    float2 v = make_float2(1.0f, 2.0f);",
                "    float2 mv = a * v;",
                "    float2 vm = v * a;",
                "    double4x3 da(1.0);",
                "    double2x4 db(1.0);",
                "    double2x3 dc = da * db;",
                "    (void)c; (void)d; (void)mv; (void)vm; (void)dc;",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [compiler, "-std=c++17", "-c", str(source_path), "-o", str(object_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + "\n\n" + helper_code


class TestHipCodeGen:
    def test_simple_function_generation(self):
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

    def test_global_constants_are_emitted(self):
        source_code = """
        shader TestShader {
            const float PI = 3.14159;
            const int WARP_SIZE = 32;
            const vec3 UP_VECTOR = vec3(0.0, 1.0, 0.0);

            compute {
                void main(float *out) {
                    out[0] = PI + float(WARP_SIZE) + UP_VECTOR.y;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "const float PI = 3.14159;" in hip_code
        assert "const int WARP_SIZE = 32;" in hip_code
        assert "const float3 UP_VECTOR = make_float3(0.0, 1.0, 0.0);" in hip_code
        assert "out[0] = ((PI + float(WARP_SIZE)) + UP_VECTOR.y);" in hip_code

    def test_compute_shader_to_kernel(self):
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

        assert "__global__ void compute_main()" in hip_code

    def test_non_void_compute_entry_point_lowers_to_void_kernel(self, tmp_path):
        ast = ShaderNode(
            name="NonVoidHipKernel",
            execution_model=ExecutionModel.COMPUTE_KERNEL,
            functions=[
                FunctionNode(
                    "returnsValue",
                    PrimitiveType("float"),
                    [ParameterNode("out", "float *")],
                    BlockNode(
                        [
                            AssignmentNode(
                                ArrayAccessNode(
                                    IdentifierNode("out"),
                                    LiteralNode(0, PrimitiveType("int")),
                                ),
                                LiteralNode(1.0, PrimitiveType("float")),
                            ),
                            ReturnNode(LiteralNode(2.0, PrimitiveType("float"))),
                        ]
                    ),
                    qualifiers=["compute"],
                )
            ],
        )

        hip_code = HipCodeGen().generate(ast)

        assert "__global__ void returnsValue(float * out)" in hip_code
        assert "__global__ float returnsValue" not in hip_code
        assert "return 2.0;" not in hip_code
        assert "return;" in hip_code
        if shutil.which("hipcc") is not None:
            compile_hip_if_hipcc_available(hip_code, tmp_path)

    def test_mesh_task_stages_lower_to_hip_metadata_and_placeholder_hooks(self):
        """Mesh/task stages lower to deterministic HIP helper representation."""
        source_code = """
        shader HipMeshTaskPipeline {
            struct TaskPayload {
                uint meshlet;
            };

            mesh {
                void main()
                    @numthreads(4, 2, 1)
                    @outputtopology(triangle)
                    @max_vertices(3)
                    @max_primitives(1) {
                    SetMeshOutputCounts(3, 1);
                }
            }

            task {
                void main() @numthreads(2, 1, 1) {
                    TaskPayload payload;
                    payload.meshlet = 7u;
                    DispatchMesh(1, 2, 1, payload);
                }
            }
        }
        """

        hip_code = HipCodeGen().generate(Parser(Lexer(source_code).tokens).parse())

        assert "__device__ inline void cgl_hip_set_mesh_output_counts" in hip_code
        assert "__device__ inline void cgl_hip_dispatch_mesh" in hip_code
        assert "// CrossGL mesh stage" in hip_code
        assert "// CrossGL mesh/task numthreads: 4, 2, 1" in hip_code
        assert "// CrossGL mesh output topology: triangle" in hip_code
        assert "// CrossGL mesh max_vertices: 3" in hip_code
        assert "// CrossGL mesh max_primitives: 1" in hip_code
        assert "__device__ void mesh_main()" in hip_code
        assert "cgl_hip_set_mesh_output_counts(3, 1);" in hip_code
        assert "// CrossGL task stage" in hip_code
        assert "// CrossGL mesh/task numthreads: 2, 1, 1" in hip_code
        assert "__device__ void task_main()" in hip_code
        assert "cgl_hip_dispatch_mesh(1, 2, 1, payload);" in hip_code

    def test_mesh_task_stages_validate_hip_metadata_and_intrinsics(self):
        invalid_topology = """
        shader BadHipMeshTopology {
            mesh {
                void main() @outputtopology(strip) { }
            }
        }
        """
        with pytest.raises(
            ValueError, match="outputtopology.*point, line, or triangle"
        ):
            HipCodeGen().generate(Parser(Lexer(invalid_topology).tokens).parse())

        invalid_numthreads = """
        shader BadHipTaskNumThreads {
            task {
                void main() @numthreads(1, 0, 1) { }
            }
        }
        """
        with pytest.raises(ValueError, match="numthreads y dimension.*positive"):
            HipCodeGen().generate(Parser(Lexer(invalid_numthreads).tokens).parse())

        invalid_set_counts = """
        shader BadHipMeshCounts {
            mesh {
                void main() {
                    SetMeshOutputCounts(1);
                }
            }
        }
        """
        with pytest.raises(ValueError, match="SetMeshOutputCounts.*two arguments"):
            HipCodeGen().generate(Parser(Lexer(invalid_set_counts).tokens).parse())

        wrong_dispatch_stage = """
        shader BadHipDispatchStage {
            compute {
                void main() {
                    DispatchMesh(1, 1, 1);
                }
            }
        }
        """
        with pytest.raises(ValueError, match="DispatchMesh.*task, object"):
            HipCodeGen().generate(Parser(Lexer(wrong_dispatch_stage).tokens).parse())

    def test_geometry_stage_lowers_to_hip_metadata_and_stream_placeholders(self):
        """Geometry stages lower to deterministic HIP helper representation."""
        source_code = """
        shader HipGeometryExpansion {
            struct GSInput {
                vec4 position @ gl_Position;
                vec3 normal @ NORMAL;
            };

            struct GSOutput {
                vec4 position @ gl_Position;
                vec3 color @ COLOR;
            };

            geometry {
                void main(
                    triangle GSInput input[3],
                    inout TriangleStream<GSOutput> stream
                ) @maxvertexcount(3) {
                    GSOutput output;
                    output.position = input[0].position;
                    output.color = input[0].normal;
                    stream.Append(output);
                    stream.RestartStrip();
                }
            }
        }
        """

        hip_code = HipCodeGen().generate(Parser(Lexer(source_code).tokens).parse())

        assert "template <typename T>" in hip_code
        assert "struct CglHipTriangleStream" in hip_code
        assert "// CrossGL geometry stage: maxvertexcount=3" in hip_code
        assert "// CrossGL geometry input primitive: triangle:input" in hip_code
        assert (
            "// CrossGL geometry output stream: TriangleStream:stream->GSOutput"
            in hip_code
        )
        assert "__device__ void geometry_main(" in hip_code
        assert "GSInput input[3]" in hip_code
        assert "CglHipTriangleStream<GSOutput>& stream" in hip_code
        assert "stream.Append(output);" in hip_code
        assert "stream.RestartStrip();" in hip_code

    def test_tessellation_stages_lower_to_hip_metadata_and_patch_placeholders(self):
        """Tessellation stages lower to deterministic HIP helper representation."""
        source_code = """
        shader HipTessellationPipeline {
            struct HSInput {
                vec3 position;
            };

            struct HSOutput {
                vec3 position;
            };

            struct PatchConstants {
                float edge;
            };

            tessellation_control {
                PatchConstants HSConst(InputPatch<HSInput, 3> patch) {
                    PatchConstants constants;
                    return constants;
                }

                HSOutput main(InputPatch<HSInput, 3> patch)
                    @domain(tri)
                    @partitioning(fractional_odd)
                    @outputtopology(triangle_cw)
                    @outputcontrolpoints(3)
                    @patchconstantfunc(HSConst) {
                    HSOutput output;
                    output.position = patch[0].position;
                    return output;
                }
            }

            tessellation_evaluation {
                void main(OutputPatch<HSOutput, 3> patch) @domain(tri) {
                    vec3 position = patch[0].position;
                }
            }
        }
        """

        hip_code = HipCodeGen().generate(Parser(Lexer(source_code).tokens).parse())

        assert "template <typename T, int N>" in hip_code
        assert "struct CglHipInputPatch" in hip_code
        assert "struct CglHipOutputPatch" in hip_code
        assert (
            "// CrossGL tessellation control stage: outputcontrolpoints=3" in hip_code
        )
        assert "// CrossGL tessellation domain: tri" in hip_code
        assert "// CrossGL tessellation partitioning: fractional_odd" in hip_code
        assert "// CrossGL tessellation output topology: triangle_cw" in hip_code
        assert "// CrossGL tessellation patch constant function: HSConst" in hip_code
        assert "// CrossGL tessellation evaluation stage: domain=tri" in hip_code
        assert (
            "__device__ HSOutput tessellation_control_main"
            "(CglHipInputPatch<HSInput, 3> patch)" in hip_code
        )
        assert (
            "__device__ void tessellation_evaluation_main"
            "(CglHipOutputPatch<HSOutput, 3> patch)" in hip_code
        )
        assert "output.position = patch[0].position;" in hip_code
        assert "float3 position = patch[0].position;" in hip_code

    def test_tessellation_stages_validate_hip_metadata(self):
        missing_control_points = """
        shader MissingHipTessControlPoints {
            tessellation_control {
                void main() @domain(tri) { }
            }
        }
        """
        with pytest.raises(ValueError, match="requires outputcontrolpoints"):
            HipCodeGen().generate(Parser(Lexer(missing_control_points).tokens).parse())

        invalid_control_points = """
        shader BadHipTessControlPoints {
            tessellation_control {
                void main() @outputcontrolpoints(0) { }
            }
        }
        """
        with pytest.raises(ValueError, match="outputcontrolpoints.*positive"):
            HipCodeGen().generate(Parser(Lexer(invalid_control_points).tokens).parse())

        missing_domain = """
        shader MissingHipTessDomain {
            tessellation_evaluation {
                void main() { }
            }
        }
        """
        with pytest.raises(ValueError, match="requires domain"):
            HipCodeGen().generate(Parser(Lexer(missing_domain).tokens).parse())

        invalid_domain = """
        shader BadHipTessDomain {
            tessellation_evaluation {
                void main() @domain(hex) { }
            }
        }
        """
        with pytest.raises(ValueError, match="domain.*must be tri"):
            HipCodeGen().generate(Parser(Lexer(invalid_domain).tokens).parse())

    def test_geometry_stage_validates_maxvertexcount_and_input_shape_for_hip(self):
        missing_attribute = """
        shader MissingHipGeometryMax {
            struct GSInput { vec4 position @ gl_Position; };
            struct GSOutput { vec4 position @ gl_Position; };

            geometry {
                void main(
                    triangle GSInput input[3],
                    inout TriangleStream<GSOutput> stream
                ) { }
            }
        }
        """
        with pytest.raises(ValueError, match="requires maxvertexcount"):
            HipCodeGen().generate(Parser(Lexer(missing_attribute).tokens).parse())

        invalid_max = """
        shader BadHipGeometryMax {
            struct GSInput { vec4 position @ gl_Position; };
            struct GSOutput { vec4 position @ gl_Position; };

            geometry {
                void main(
                    triangle GSInput input[3],
                    inout TriangleStream<GSOutput> stream
                ) @maxvertexcount(0) { }
            }
        }
        """
        with pytest.raises(ValueError, match="maxvertexcount.*positive"):
            HipCodeGen().generate(Parser(Lexer(invalid_max).tokens).parse())

        wrong_arity = """
        shader BadHipGeometryInputArity {
            struct GSInput { vec4 position @ gl_Position; };
            struct GSOutput { vec4 position @ gl_Position; };

            geometry {
                void main(
                    triangle GSInput input[2],
                    inout TriangleStream<GSOutput> stream
                ) @maxvertexcount(3) { }
            }
        }
        """
        with pytest.raises(ValueError, match="triangle input primitive.*3 element"):
            HipCodeGen().generate(Parser(Lexer(wrong_arity).tokens).parse())

        missing_stream = """
        shader BadHipGeometryStream {
            struct GSInput { vec4 position @ gl_Position; };

            geometry {
                void main(triangle GSInput input[3]) @maxvertexcount(3) { }
            }
        }
        """
        with pytest.raises(ValueError, match="must include a PointStream"):
            HipCodeGen().generate(Parser(Lexer(missing_stream).tokens).parse())

    def test_mesh_function_stage_qualifier_lowers_hip_metadata(self):
        ast = ShaderNode(
            name="QualifiedHipMeshStage",
            execution_model=ExecutionModel.GRAPHICS_PIPELINE,
            functions=[
                FunctionNode(
                    name="main",
                    return_type=PrimitiveType("void"),
                    parameters=[],
                    body=BlockNode([]),
                    attributes=[
                        AttributeNode("outputtopology", [IdentifierNode("line")]),
                        AttributeNode("max_vertices", [2]),
                        AttributeNode("max_primitives", [1]),
                    ],
                    qualifiers=[ShaderStage.MESH],
                )
            ],
        )

        hip_code = HipCodeGen().generate(ast)

        assert "// CrossGL mesh stage" in hip_code
        assert "// CrossGL mesh output topology: line" in hip_code
        assert "__device__ void main()" in hip_code

    def test_ray_tracing_stages_emit_hip_metadata_and_hooks(self, tmp_path):
        source_code = """
        shader HipRayTracingStages {
            struct RayPayload {
                vec3 color;
            };

            struct HitAttributes {
                vec2 barycentrics;
            };

            struct CallableData {
                uint value;
            };

            accelerationStructureEXT scene @binding(3) @set(2);

            ray_generation {
                void main() {
                    uvec3 launch = gl_LaunchIDEXT;
                    uint launchWidth = gl_LaunchSizeEXT.x;
                    RayDesc ray = RayDesc(
                        vec3(0.0, 0.0, 0.0),
                        0.001,
                        vec3(0.0, 0.0, 1.0),
                        100.0
                    );
                    RayPayload payload;
                    CallableData data;
                    TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);
                    CallShader(0, data);
                }
            }

            ray_closest_hit {
                void main(
                    RayPayload payload @ payload,
                    HitAttributes attributes @ hit_attribute,
                    float hitT @ gl_HitTEXT
                ) {
                    payload.color = vec3(1.0, 0.0, hitT);
                }
            }

            ray_any_hit {
                void main(
                    RayPayload payload @ payload,
                    HitAttributes attributes @ hit_attribute,
                    uint hitKind @ gl_HitKindEXT
                ) {
                    payload.color = vec3(0.5, 0.0, float(hitKind));
                    IgnoreHit();
                    AcceptHitAndEndSearch();
                }
            }

            ray_miss {
                void main(RayPayload payload @ rayPayloadInEXT) {
                    payload.color = vec3(0.0, 0.0, 0.0);
                }
            }

            ray_callable {
                void main(CallableData data @ callableDataInEXT) {
                    data.value = data.value + 1u;
                }
            }

            ray_intersection {
                void main() {
                    HitAttributes attributes;
                    bool accepted = ReportHit(1.0, 0, attributes);
                    bool acceptedWithoutAttributes = ReportHit(2.0, 1);
                }
            }
        }
        """

        hip_code = HipCodeGen().generate(Parser(Lexer(source_code).tokens).parse())

        assert (
            "// CrossGL resource metadata: name=scene kind=acceleration_structure "
            "set=2 binding=3 binding_source=explicit" in hip_code
        )
        assert "struct CglRayTracingAccelerationStructure" in hip_code
        assert "struct CglRayDesc" in hip_code
        assert "// CrossGL ray stage: ray_generation" in hip_code
        assert "// CrossGL ray stage: closest_hit" in hip_code
        assert "// CrossGL ray stage: any_hit" in hip_code
        assert "// CrossGL ray stage: miss" in hip_code
        assert "// CrossGL ray stage: callable" in hip_code
        assert "// CrossGL ray stage: intersection" in hip_code
        assert "__device__ void ray_generation_main()" in hip_code
        assert "__device__ void ray_closest_hit_main(" in hip_code
        assert "// CrossGL parameter semantic: payload: ray_payload" in hip_code
        assert "// CrossGL parameter semantic: attributes: hit_attribute" in hip_code
        assert "// CrossGL parameter semantic: data: callable_data" in hip_code
        assert "uint3 launch = cgl_ray_launch_id();" in hip_code
        assert "unsigned int launchWidth = cgl_ray_launch_size().x;" in hip_code
        assert "cgl_trace_ray(scene, 0, 255, 0, 1, 0, ray, payload);" in hip_code
        assert "cgl_call_shader(0, data);" in hip_code
        assert "cgl_ignore_hit();" in hip_code
        assert "cgl_accept_hit_and_end_search();" in hip_code
        assert "bool accepted = cgl_report_hit(1.0, 0, attributes);" in hip_code
        assert (
            "bool acceptedWithoutAttributes = cgl_report_hit"
            "(2.0, 1, CglBuiltInTriangleIntersectionAttributes{});" in hip_code
        )

        compile_hip_if_hipcc_available(hip_code, tmp_path)

    def test_hip_ray_stage_parameter_semantics_validate_builtin_stage_type_and_duplicates(
        self,
    ):
        invalid_type = """
        shader InvalidHipRayBuiltinType {
            ray_generation {
                void main(float launch @ gl_LaunchIDEXT) {
                }
            }
        }
        """
        with pytest.raises(ValueError, match="gl_LaunchIDEXT.*expected uint3 type"):
            HipCodeGen().generate(Parser(Lexer(invalid_type).tokens).parse())

        invalid_stage = """
        shader InvalidHipRayBuiltinStage {
            fragment {
                void main(uvec3 launch @ gl_LaunchIDEXT) {
                }
            }
        }
        """
        with pytest.raises(ValueError, match="gl_LaunchIDEXT.*fragment stage"):
            HipCodeGen().generate(Parser(Lexer(invalid_stage).tokens).parse())

        duplicate = """
        shader DuplicateHipRayBuiltin {
            ray_generation {
                void main(
                    uvec3 launch @ gl_LaunchIDEXT,
                    uvec3 launchAgain @ SV_DispatchRaysIndex
                ) {
                }
            }
        }
        """
        with pytest.raises(
            ValueError, match="Duplicate HIP stage parameter semantic launch_id"
        ):
            HipCodeGen().generate(Parser(Lexer(duplicate).tokens).parse())

    def test_ray_query_methods_lower_to_hip_hooks(self):
        source_code = """
        void queryRay(RayQuery query) {
            bool active = query.Proceed();
        }
        """

        hip_code = HipCodeGen().generate(Parser(Lexer(source_code).tokens).parse())

        assert "struct CglRayQuery" in hip_code
        assert "__device__ void queryRay(CglRayQuery query)" in hip_code
        assert "__device__ inline bool cgl_ray_query_proceed" in hip_code
        assert "bool active = cgl_ray_query_proceed(query);" in hip_code

    def test_generated_basic_compute_kernel_smoke_compiles_with_hipcc(self, tmp_path):
        """Smoke compile a generated HIP kernel when hipcc is available."""
        smoke_ast = ShaderNode(
            "HipCompileSmoke",
            ExecutionModel.COMPUTE_KERNEL,
            functions=[
                FunctionNode(
                    "compileSmoke",
                    PrimitiveType("void"),
                    [ParameterNode("out", "float *")],
                    [
                        AssignmentNode(
                            ArrayAccessNode(
                                IdentifierNode("out"),
                                LiteralNode(0, PrimitiveType("int")),
                            ),
                            LiteralNode(1.0, PrimitiveType("float")),
                        )
                    ],
                    qualifiers=["compute"],
                )
            ],
        )

        hip_code = HipCodeGen().generate(smoke_ast)

        assert "__global__ void compileSmoke(float * out)" in hip_code
        assert "out[0] = 1.0;" in hip_code
        compile_hip_if_hipcc_available(hip_code, tmp_path)

    def test_generated_atomic_barrier_kernel_smoke_compiles_with_hipcc(self, tmp_path):
        """Smoke compile generated HIP atomics, barriers, fences, and builtins."""
        smoke_ast = ShaderNode(
            "HipAtomicBarrierSmoke",
            ExecutionModel.COMPUTE_KERNEL,
            functions=[
                FunctionNode(
                    "atomicBarrierSmoke",
                    PrimitiveType("void"),
                    [
                        ParameterNode("counter", "int *"),
                        ParameterNode("out", "int *"),
                    ],
                    [
                        VariableNode(
                            "lane",
                            PrimitiveType("int"),
                            IdentifierNode("gl_LocalInvocationID.x"),
                        ),
                        VariableNode(
                            "global",
                            PrimitiveType("int"),
                            IdentifierNode("gl_GlobalInvocationID.x"),
                        ),
                        VariableNode(
                            "old",
                            PrimitiveType("int"),
                            FunctionCallNode(
                                IdentifierNode("atomicAdd"),
                                [
                                    IdentifierNode("counter"),
                                    LiteralNode(1, PrimitiveType("int")),
                                ],
                            ),
                        ),
                        FunctionCallNode(IdentifierNode("barrier"), []),
                        FunctionCallNode(IdentifierNode("memoryBarrier"), []),
                        AssignmentNode(
                            ArrayAccessNode(
                                IdentifierNode("out"),
                                LiteralNode(0, PrimitiveType("int")),
                            ),
                            FunctionCallNode(
                                IdentifierNode("atomicCAS"),
                                [
                                    IdentifierNode("counter"),
                                    IdentifierNode("old"),
                                    IdentifierNode("global"),
                                ],
                            ),
                        ),
                        AssignmentNode(
                            ArrayAccessNode(
                                IdentifierNode("out"),
                                LiteralNode(1, PrimitiveType("int")),
                            ),
                            IdentifierNode("lane"),
                        ),
                    ],
                    qualifiers=["compute"],
                )
            ],
        )

        hip_code = HipCodeGen().generate(smoke_ast)

        assert "int lane = threadIdx.x;" in hip_code
        assert "int global = (blockIdx.x * blockDim.x + threadIdx.x);" in hip_code
        assert "int old = atomicAdd(counter, 1);" in hip_code
        assert "__syncthreads();" in hip_code
        assert "__threadfence();" in hip_code
        assert "out[0] = atomicCAS(counter, old, global);" in hip_code
        compile_hip_if_hipcc_available(hip_code, tmp_path)

    def test_generated_vector_constructor_kernel_smoke_compiles_with_hipcc(
        self, tmp_path
    ):
        """Smoke compile generated HIP vector constructors and member access."""
        smoke_ast = ShaderNode(
            "HipVectorSmoke",
            ExecutionModel.COMPUTE_KERNEL,
            functions=[
                FunctionNode(
                    "vectorSmoke",
                    PrimitiveType("void"),
                    [ParameterNode("out", "float *")],
                    [
                        VariableNode(
                            "uv",
                            PrimitiveType("vec2"),
                            FunctionCallNode(
                                IdentifierNode("vec2"),
                                [
                                    LiteralNode(1.0, PrimitiveType("float")),
                                    LiteralNode(2.0, PrimitiveType("float")),
                                ],
                            ),
                        ),
                        VariableNode(
                            "color",
                            PrimitiveType("vec4"),
                            FunctionCallNode(
                                IdentifierNode("vec4"),
                                [
                                    MemberAccessNode(IdentifierNode("uv"), "x"),
                                    MemberAccessNode(IdentifierNode("uv"), "y"),
                                    LiteralNode(3.0, PrimitiveType("float")),
                                    LiteralNode(4.0, PrimitiveType("float")),
                                ],
                            ),
                        ),
                        AssignmentNode(
                            ArrayAccessNode(
                                IdentifierNode("out"),
                                LiteralNode(0, PrimitiveType("int")),
                            ),
                            MemberAccessNode(IdentifierNode("color"), "w"),
                        ),
                    ],
                    qualifiers=["compute"],
                )
            ],
        )

        hip_code = HipCodeGen().generate(smoke_ast)

        assert "float2 uv = make_float2(1.0, 2.0);" in hip_code
        assert "float4 color = make_float4(uv.x, uv.y, 3.0, 4.0);" in hip_code
        assert "out[0] = color.w;" in hip_code
        compile_hip_if_hipcc_available(hip_code, tmp_path)

    def test_compute_stage_local_helper_functions_emit_before_kernel(self):
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
        kernel_signature = "__global__ void compute_main()"
        assert helper_signature in hip_code
        assert kernel_signature in hip_code
        assert hip_code.index(helper_signature) < hip_code.index(kernel_signature)
        assert "float y = helper(1.0);" in hip_code

    def test_user_defined_mix_function_is_not_lowered_to_hip_builtin(self):
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
        codegen = HipCodeGen()

        assert codegen.map_type("int") == "int"
        assert codegen.map_type("float") == "float"
        assert codegen.map_type("double") == "double"
        assert codegen.map_type("bool") == "bool"
        assert codegen.map_type("void") == "void"

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
        codegen = HipCodeGen()

        assert codegen.function_map.get("sqrt") == "sqrtf"
        assert codegen.function_map.get("pow") == "powf"
        assert codegen.function_map.get("sin") == "sinf"
        assert codegen.function_map.get("cos") == "cosf"

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
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    vec3 a = vec3(1.0, 2.0, 3.0);
                    vec3 b = vec3(4.0, 5.0, 6.0);
                    float d = dot(a, b);
                    vec3 c = cross(a, b);
                    vec3 n = normalize(a);
                    float weight = 0.5;
                    vec3 scaled = normalize(a) * weight;
                    vec3 wave = sin(a);
                    vec3 noisy = -1.0 + 2.0 * fract(sin(a) * 43758.5453);
                    vec3 fresnel = (1.0 - a) * pow(max(1.0 - weight, 0.0), 5.0);
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
        assert "__device__ inline float3 cgl_float3_sin(float3 value)" in hip_code
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
        assert (
            "float3 scaled = cgl_float3_mul_scalar(cgl_float3_normalize(a), weight);"
            in hip_code
        )
        assert "float3 wave = cgl_float3_sin(a);" in hip_code
        assert "cgl_float3_mul_scalar(cgl_float3_sin(a), 43758.5453)" in hip_code
        assert "cgl_scalar_mul_float3(2.0, cgl_float3_fract" in hip_code
        assert "cgl_float3_mul_scalar(cgl_scalar_sub_float3(1.0, a), powf" in hip_code
        assert "float l = cgl_float3_length(a);" in hip_code
        assert "double dl = cgl_double2_length(da);" in hip_code
        assert "float d = dot(a, b);" not in hip_code
        assert "float3 c = cross(a, b);" not in hip_code
        assert "float3 n = normalize(a);" not in hip_code
        assert "sinf(a)" not in hip_code
        assert "cgl_float3_normalize(a) * weight" not in hip_code
        assert "2.0 * cgl_float3_fract" not in hip_code
        assert "float l = length(a);" not in hip_code

    def test_vector_atan2_builtin_lowers_componentwise(self):
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

    def test_matrix_types_and_constructors_emit_hip_names(self, tmp_path):
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    mat2 transform = mat2(1.0, 0.0, 0.0, 1.0);
                    mat2 transform2 = mat2(1.0);
                    mat2 combined = transform + transform2;
                    mat2 product = transform * transform2;
                    mat2 scaled = 2.0 * transform;
                    vec2 direction = vec2(1.0, 2.0);
                    vec2 transformed = transform * direction;
                    vec2 projected = direction * transform;
                    mat3x4 affine;
                    dmat2 precise = dmat2(1.0, 0.0, 0.0, 1.0);
                    dmat2 preciseProduct = precise * dmat2(1.0);
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
        assert "float2x2 transform2 = float2x2(1.0);" in hip_code
        assert "float2x2 combined = (transform + transform2);" in hip_code
        assert "float2x2 product = (transform * transform2);" in hip_code
        assert "float2x2 scaled = (2.0 * transform);" in hip_code
        assert "float2 transformed = (transform * direction);" in hip_code
        assert "float2 projected = (direction * transform);" in hip_code
        assert "float3x4 affine;" in hip_code
        assert "double2x2 precise = double2x2(1.0, 0.0, 0.0, 1.0);" in hip_code
        assert "double2x2 preciseProduct = (precise * double2x2(1.0));" in hip_code
        assert "double4x3 jacobian;" in hip_code
        assert "__host__ __device__ explicit float2x2(float diagonal)" in hip_code
        assert (
            "inline float2x2 operator*(const float2x2& lhs, " "const float2x2& rhs)"
        ) in hip_code
        assert (
            "inline float2 operator*(const float2x2& lhs, const float2& rhs)"
            in hip_code
        )
        assert (
            "inline float2 operator*(const float2& lhs, const float2x2& rhs)"
            in hip_code
        )
        assert "MatrixType(" not in hip_code
        assert " = mat2(" not in hip_code
        assert " = dmat2(" not in hip_code
        compile_matrix_helpers_if_cxx_available(
            codegen.generate_hip_matrix_type_helpers(),
            tmp_path,
        )

    def test_bool_vector_constructors_emit_hip_names(self):
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
        codegen = HipCodeGen()

        assert codegen.builtin_map.get("gl_LocalInvocationID.x") == "threadIdx.x"
        assert codegen.builtin_map.get("gl_LocalInvocationID.y") == "threadIdx.y"
        assert codegen.builtin_map.get("gl_LocalInvocationID.z") == "threadIdx.z"

        assert codegen.builtin_map.get("gl_WorkGroupID.x") == "blockIdx.x"
        assert codegen.builtin_map.get("gl_WorkGroupID.y") == "blockIdx.y"
        assert codegen.builtin_map.get("gl_WorkGroupID.z") == "blockIdx.z"

        assert codegen.builtin_map.get("gl_WorkGroupSize.x") == "blockDim.x"
        assert codegen.builtin_map.get("gl_WorkGroupSize.y") == "blockDim.y"
        assert codegen.builtin_map.get("gl_WorkGroupSize.z") == "blockDim.z"

    def test_builtin_invocation_ids_emit_hip_names(self):
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    int lx = gl_LocalInvocationID.x;
                    int gx = gl_GlobalInvocationID.x;
                    ivec2 globalXY = ivec2(gl_GlobalInvocationID.xy);
                    uvec2 groupXY = gl_WorkGroupID.xy;
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
        assert (
            "int2 globalXY = make_int2("
            "(blockIdx.x * blockDim.x + threadIdx.x), "
            "(blockIdx.y * blockDim.y + threadIdx.y));"
        ) in hip_code
        assert "uint2 groupXY = make_uint2(blockIdx.x, blockIdx.y);" in hip_code
        assert "int wy = blockIdx.y;" in hip_code
        assert "int bz = blockDim.z;" in hip_code
        assert "int nz = gridDim.z;" in hip_code
        assert "gl_LocalInvocationID" not in hip_code
        assert "gl_GlobalInvocationID" not in hip_code
        assert "gl_WorkGroupID" not in hip_code
        assert "gl_WorkGroupSize" not in hip_code
        assert "gl_NumWorkGroups" not in hip_code
        assert ".xy" not in hip_code

    def test_hlsl_compute_builtins_emit_hip_names(self):
        source_code = """
        shader TestShader {
            compute {
                void main() {
                    int lx = SV_GroupThreadID.x;
                    int gy = SV_DispatchThreadID.y;
                    int wz = SV_GroupID.z;
                    uint lane = SV_GroupIndex;
                    uint localLane = gl_LocalInvocationIndex;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        linear_index = (
            "(threadIdx.z * blockDim.y * blockDim.x + "
            "threadIdx.y * blockDim.x + threadIdx.x)"
        )
        assert "int lx = threadIdx.x;" in hip_code
        assert "int gy = (blockIdx.y * blockDim.y + threadIdx.y);" in hip_code
        assert "int wz = blockIdx.z;" in hip_code
        assert f"unsigned int lane = {linear_index};" in hip_code
        assert f"unsigned int localLane = {linear_index};" in hip_code
        assert "SV_GroupThreadID" not in hip_code
        assert "SV_DispatchThreadID" not in hip_code
        assert "SV_GroupID" not in hip_code
        assert "SV_GroupIndex" not in hip_code
        assert "gl_LocalInvocationIndex" not in hip_code

    def test_hlsl_compute_builtin_parameters_lower_to_hip_expressions(self):
        source_code = """
        shader TestShader {
            compute {
                void main(
                    uvec3 dispatchId @ SV_DispatchThreadID,
                    uvec3 groupId @ SV_GroupID
                ) {
                    uvec3 full = dispatchId;
                    uint gy = dispatchId.y;
                    uint bx = groupId.x;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "__global__ void compute_main()" in hip_code
        assert "uint3 dispatchId" not in hip_code
        assert "uint3 groupId" not in hip_code
        assert (
            "uint3 full = make_uint3((blockIdx.x * blockDim.x + threadIdx.x), "
            "(blockIdx.y * blockDim.y + threadIdx.y), "
            "(blockIdx.z * blockDim.z + threadIdx.z));"
        ) in hip_code
        assert "unsigned int gy = (blockIdx.y * blockDim.y + threadIdx.y);" in hip_code
        assert "unsigned int bx = blockIdx.x;" in hip_code
        assert "SV_DispatchThreadID" not in hip_code
        assert "SV_GroupID" not in hip_code

    def test_direct_hlsl_compute_vector_builtins_emit_hip_vectors(self):
        ast = ShaderNode(
            name="DirectHlslBuiltins",
            execution_model=ExecutionModel.GENERAL_PURPOSE,
            functions=[
                FunctionNode(
                    "main",
                    PrimitiveType("void"),
                    [],
                    BlockNode(
                        [
                            VariableNode(
                                "local",
                                PrimitiveType("uvec3"),
                                IdentifierNode("SV_GroupThreadID"),
                            ),
                            VariableNode(
                                "global",
                                PrimitiveType("uvec3"),
                                IdentifierNode("SV_DispatchThreadID"),
                            ),
                            VariableNode(
                                "group",
                                PrimitiveType("uvec3"),
                                IdentifierNode("SV_GroupID"),
                            ),
                        ]
                    ),
                )
            ],
        )

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert (
            "uint3 local = make_uint3(threadIdx.x, threadIdx.y, threadIdx.z);"
            in hip_code
        )
        assert (
            "uint3 global = make_uint3((blockIdx.x * blockDim.x + threadIdx.x), "
            "(blockIdx.y * blockDim.y + threadIdx.y), "
            "(blockIdx.z * blockDim.z + threadIdx.z));"
        ) in hip_code
        assert (
            "uint3 group = make_uint3(blockIdx.x, blockIdx.y, blockIdx.z);" in hip_code
        )
        assert "SV_GroupThreadID" not in hip_code
        assert "SV_DispatchThreadID" not in hip_code
        assert "SV_GroupID" not in hip_code

    def test_direct_builtin_identifier_nodes_emit_hip_names(self):
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
        source_code = """
        shader TestShader {
            struct Counters {
                int active;
            };

            Counters counters;

            compute {
                void main() {
                    int counter;
                    atomicAdd(counter, 1);
                    atomicAdd(counters.active, 1);
                    int oldValue = atomicExchange(counter, 2);
                    int exchanged = atomicCompareExchange(counter, 0, 3);
                    int swapped = atomicCompSwap(counter, exchanged, 4);
                    atomicSub(counter, 1);
                    atomicMin(counter, 0);
                    atomicMax(counter, 7);
                    atomicAnd(counter, 1);
                    atomicOr(counter, 2);
                    atomicXor(counter, 3);
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
        assert "atomicAdd(&counter, 1);" in hip_code
        assert "atomicAdd(&counters.active, 1);" in hip_code
        assert "int oldValue = atomicExch(&counter, 2);" in hip_code
        assert "int exchanged = atomicCAS(&counter, 0, 3);" in hip_code
        assert "int swapped = atomicCAS(&counter, exchanged, 4);" in hip_code
        assert "atomicSub(&counter, 1);" in hip_code
        assert "atomicMin(&counter, 0);" in hip_code
        assert "atomicMax(&counter, 7);" in hip_code
        assert "atomicAnd(&counter, 1);" in hip_code
        assert "atomicOr(&counter, 2);" in hip_code
        assert "atomicXor(&counter, 3);" in hip_code
        assert "atomicExchange(" not in hip_code
        assert "atomicCompareExchange(" not in hip_code
        assert "atomicCompSwap(" not in hip_code

    def test_synchronization_functions(self):
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

        assert "#include <hip/hip_runtime.h>" in hip_code

    def test_math_functions(self):
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

        assert "#include <hip/hip_runtime.h>" in hip_code

    def test_double_math_builtins_emit_hip_double_functions(self):
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
        source_code = """
        shader TestShader {
            fragment {
                void main() {
                    sampler2D tex;
                    vec2 uv = vec2(0.5, 0.5);
                    vec4 color = texture(tex, uv);
                    vec4 lodColor = textureLod(tex, uv, 1.0);
                    vec4 gradColor = textureGrad(tex, uv, uv, uv);
                    vec4 directColor = tex2D(tex, uv);
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
        assert "hipTextureObject_t tex;" in hip_code
        assert "float2 uv = make_float2(0.5, 0.5);" in hip_code
        assert "float4 color = tex2D<float4>(tex, uv.x, uv.y);" in hip_code
        assert "float4 lodColor = tex2DLod<float4>(tex, uv.x, uv.y, 1.0);" in hip_code
        assert (
            "float4 gradColor = tex2DGrad<float4>(tex, uv.x, uv.y, uv, uv);" in hip_code
        )
        assert "float4 directColor = tex2D<float4>(tex, uv.x, uv.y);" in hip_code
        assert "sampler2D tex;" not in hip_code
        assert "tex2D(tex, uv)" not in hip_code
        assert " = texture(" not in hip_code
        assert " = textureLod(" not in hip_code
        assert " = textureGrad(" not in hip_code

    def test_split_sampler_texture_calls_use_hip_texture_object_coordinates(self):
        source_code = """
        shader SplitSampler {
            sampler2D colorMap @texture(0);
            sampler2D textures[4] @texture(1);
            sampler linearSampler @sampler(0);
            sampler samplers[4] @sampler(1);

            vec4 sampleColor(
                sampler2D tex,
                sampler state,
                vec2 uv,
                vec2 ddx,
                vec2 ddy
            ) {
                return texture(tex, state, uv)
                    + textureLod(tex, state, uv, 1.0)
                    + textureGrad(tex, state, uv, ddx, ddy);
            }

            vec4 sampleGlobal(int layer, vec2 uv) {
                return texture(colorMap, linearSampler, uv)
                    + texture(textures[layer], samplers[layer], uv);
            }

            compute {
                void main() {}
            }
        }
        """

        hip_code = HipCodeGen().generate(Parser(Lexer(source_code).tokens).parse())

        assert (
            "// CrossGL resource metadata: name=colorMap kind=texture set=0 "
            "binding=0 binding_source=explicit"
        ) in hip_code
        assert (
            "// CrossGL resource metadata: name=linearSampler kind=sampler set=0 "
            "binding=0 binding_source=explicit"
        ) in hip_code
        assert (
            "// CrossGL resource metadata: name=textures kind=texture set=0 "
            "binding=1 binding_source=explicit count=4"
        ) in hip_code
        assert (
            "// CrossGL resource metadata: name=samplers kind=sampler set=0 "
            "binding=1 binding_source=explicit count=4"
        ) in hip_code
        assert "hipTextureObject_t colorMap;" in hip_code
        assert "hipTextureObject_t linearSampler;" in hip_code
        assert "hipTextureObject_t textures[4];" in hip_code
        assert "hipTextureObject_t samplers[4];" in hip_code
        assert "tex2D<float4>(tex, uv.x, uv.y)" in hip_code
        assert "tex2DLod<float4>(tex, uv.x, uv.y, 1.0)" in hip_code
        assert "tex2DGrad<float4>(tex, uv.x, uv.y, ddx, ddy)" in hip_code
        assert "tex2D<float4>(colorMap, uv.x, uv.y)" in hip_code
        assert "tex2D<float4>(textures[layer], uv.x, uv.y)" in hip_code
        assert "linearSampler.x" not in hip_code
        assert "linearSampler.y" not in hip_code
        assert "samplers[layer].x" not in hip_code
        assert "samplers[layer].y" not in hip_code
        assert "texture(" not in hip_code
        assert "textureLod(" not in hip_code
        assert "textureGrad(" not in hip_code

    def test_hlsl_sampler_state_aliases_emit_hip_sampler_resources(self):
        source_code = """
        shader AliasSampler {
            Texture2D colorMap @register(t3, space2);
            SamplerState linearSampler @register(s3, space2);
            SamplerComparisonState compareSampler @register(s4, space2);

            vec4 sampleAlias(Texture2D tex, SamplerState state, vec2 uv) {
                return texture(tex, state, uv);
            }

            compute {
                void main() {}
            }
        }
        """

        hip_code = HipCodeGen().generate(Parser(Lexer(source_code).tokens).parse())

        assert (
            "// CrossGL resource metadata: name=colorMap kind=texture set=2 "
            "binding=3 binding_source=explicit register=t3,space2"
        ) in hip_code
        assert (
            "// CrossGL resource metadata: name=linearSampler kind=sampler set=2 "
            "binding=3 binding_source=explicit register=s3,space2"
        ) in hip_code
        assert (
            "// CrossGL resource metadata: name=compareSampler kind=sampler set=2 "
            "binding=4 binding_source=explicit register=s4,space2"
        ) in hip_code
        assert "hipTextureObject_t colorMap;" in hip_code
        assert "hipTextureObject_t linearSampler;" in hip_code
        assert "hipTextureObject_t compareSampler;" in hip_code
        assert (
            "__device__ float4 sampleAlias("
            "hipTextureObject_t tex, hipTextureObject_t state, float2 uv)"
        ) in hip_code
        assert "return tex2D<float4>(tex, uv.x, uv.y);" in hip_code
        assert "SamplerState" not in hip_code
        assert "SamplerComparisonState" not in hip_code
        assert "state.x" not in hip_code
        assert "state.y" not in hip_code

    def test_array_and_3d_texture_calls_emit_hip_texture_functions(self):
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
        assert "hipTextureObject_t volumeMap;" in hip_code
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
            "float4 volumeColor = tex3D<float4>(paramVolume, uvw.x, uvw.y, uvw.z);"
            in hip_code
        )
        assert (
            "float4 volumeMip = tex3DLod<float4>(paramVolume, uvw.x, uvw.y, uvw.z, 1.0);"
            in hip_code
        )
        assert (
            "float4 volumeGrad = tex3DGrad<float4>"
            "(paramVolume, uvw.x, uvw.y, uvw.z, "
            "make_float4(ddx3.x, ddx3.y, ddx3.z, 0.0f), "
            "make_float4(ddy3.x, ddy3.y, ddy3.z, 0.0f));" in hip_code
        )
        assert " = texture(" not in hip_code
        assert " = textureLod(" not in hip_code
        assert " = textureGrad(" not in hip_code

    def test_1d_and_cube_texture_calls_emit_hip_texture_functions(self):
        source_code = """
        shader Resources {
            sampler1d ramp;
            sampler1darray rampArray;
            samplercube envMap;

            void sampleResources(
                sampler1d paramRamp,
                sampler1darray paramRampArray,
                samplercube paramEnv,
                float u,
                vec2 uLayer,
                vec3 dir,
                float du,
                float duLayer,
                vec4 ddxCube,
                vec4 ddyCube
            ) {
                vec4 rampColor = texture(paramRamp, u);
                vec4 rampMip = textureLod(paramRamp, u, 2.0);
                vec4 rampGrad = textureGrad(paramRamp, u, du, du);
                vec4 rampArrayColor = texture(paramRampArray, uLayer);
                vec4 rampArrayMip = textureLod(paramRampArray, uLayer, 3.0);
                vec4 rampArrayGrad = textureGrad(
                    paramRampArray,
                    uLayer,
                    duLayer,
                    duLayer
                );
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

        assert "hipTextureObject_t ramp;" in hip_code
        assert "hipTextureObject_t rampArray;" in hip_code
        assert "hipTextureObject_t envMap;" in hip_code
        assert "float4 rampColor = tex1D<float4>(paramRamp, u);" in hip_code
        assert "float4 rampMip = tex1DLod<float4>(paramRamp, u, 2.0);" in hip_code
        assert "float4 rampGrad = tex1DGrad<float4>(paramRamp, u, du, du);" in hip_code
        assert (
            "float4 rampArrayColor = tex1DLayered<float4>"
            "(paramRampArray, uLayer.x, uLayer.y);" in hip_code
        )
        assert (
            "float4 rampArrayMip = tex1DLayeredLod<float4>"
            "(paramRampArray, uLayer.x, uLayer.y, 3.0);" in hip_code
        )
        assert (
            "float4 rampArrayGrad = tex1DLayeredGrad<float4>"
            "(paramRampArray, uLayer.x, uLayer.y, duLayer, duLayer);" in hip_code
        )
        assert (
            "float4 envColor = texCubemap<float4>(paramEnv, dir.x, dir.y, dir.z);"
            in hip_code
        )
        assert (
            "float4 envMip = texCubemapLod<float4>(paramEnv, dir.x, dir.y, dir.z, 1.0);"
            in hip_code
        )
        assert (
            "float4 envGrad = texCubemapGrad<float4>"
            "(paramEnv, dir.x, dir.y, dir.z, ddxCube, ddyCube);" in hip_code
        )
        assert " = texture(" not in hip_code
        assert " = textureLod(" not in hip_code
        assert " = textureGrad(" not in hip_code

    def test_cube_array_texture_calls_emit_hip_texture_functions(self):
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
                float comparedLodOffset = textureCompareLodOffset(
                    shadowMap,
                    uv,
                    depth,
                    2.0,
                    offset
                );
                float comparedGradOffset = textureCompareGradOffset(
                    shadowMap,
                    uv,
                    depth,
                    ddx,
                    ddy,
                    offset
                );
                vec4 gathered = textureGatherCompare(shadowMap, cmpSampler, uv, depth);
                vec4 gatheredOffset = textureGatherCompareOffset(
                    shadowMap,
                    uv,
                    depth,
                    offset
                );
                float projected = textureCompareProj(shadowMap, uvw, depth);
                float projectedOffset = textureCompareProjOffset(
                    shadowMap,
                    uvw,
                    depth,
                    offset
                );
                float projectedLod = textureCompareProjLod(
                    shadowMap,
                    uvw,
                    depth,
                    2.0
                );
                float projectedLodOffset = textureCompareProjLodOffset(
                    shadowMap,
                    uvw,
                    depth,
                    2.0,
                    offset
                );
                float projectedGrad = textureCompareProjGrad(
                    shadowMap,
                    uvw,
                    depth,
                    ddx,
                    ddy
                );
                float projectedGradOffset = textureCompareProjGradOffset(
                    shadowMap,
                    uvw,
                    depth,
                    ddx,
                    ddy,
                    offset
                );
                float projectedSum = textureCompareProj(shadowMap, uvw, depth) +
                    textureCompareProjGradOffset(
                        shadowMap,
                        uvw,
                        depth,
                        ddx,
                        ddy,
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
            "float comparedLodOffset = /* unsupported HIP shadow resource call: "
            "textureCompareLodOffset on sampler2DShadow */ 0.0f;" in hip_code
        )
        assert (
            "float comparedGradOffset = /* unsupported HIP shadow resource call: "
            "textureCompareGradOffset on sampler2DShadow */ 0.0f;" in hip_code
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
        assert (
            "float projected = /* unsupported HIP shadow resource call: "
            "textureCompareProj on sampler2DShadow */ 0.0f;" in hip_code
        )
        assert (
            "float projectedOffset = /* unsupported HIP shadow resource call: "
            "textureCompareProjOffset on sampler2DShadow */ 0.0f;" in hip_code
        )
        assert (
            "float projectedLod = /* unsupported HIP shadow resource call: "
            "textureCompareProjLod on sampler2DShadow */ 0.0f;" in hip_code
        )
        assert (
            "float projectedLodOffset = /* unsupported HIP shadow resource call: "
            "textureCompareProjLodOffset on sampler2DShadow */ 0.0f;" in hip_code
        )
        assert (
            "float projectedGrad = /* unsupported HIP shadow resource call: "
            "textureCompareProjGrad on sampler2DShadow */ 0.0f;" in hip_code
        )
        assert (
            "float projectedGradOffset = /* unsupported HIP shadow resource call: "
            "textureCompareProjGradOffset on sampler2DShadow */ 0.0f;" in hip_code
        )
        assert (
            "float projectedSum = "
            "(/* unsupported HIP shadow resource call: "
            "textureCompareProj on sampler2DShadow */ 0.0f + "
            "/* unsupported HIP shadow resource call: textureCompareProjGradOffset "
            "on sampler2DShadow */ 0.0f);" in hip_code
        )
        assert "textureCompare(" not in hip_code
        assert "textureCompareLod(" not in hip_code
        assert "textureCompareGrad(" not in hip_code
        assert "textureCompareOffset(" not in hip_code
        assert "textureCompareLodOffset(" not in hip_code
        assert "textureCompareGradOffset(" not in hip_code
        assert "textureCompareProj(" not in hip_code
        assert "textureCompareProjOffset(" not in hip_code
        assert "textureCompareProjLod(" not in hip_code
        assert "textureCompareProjLodOffset(" not in hip_code
        assert "textureCompareProjGrad(" not in hip_code
        assert "textureCompareProjGradOffset(" not in hip_code
        assert "textureGatherCompare(" not in hip_code
        assert "textureGatherCompareOffset(" not in hip_code
        assert "texture(shadowMap" not in hip_code
        assert "texture(cascades" not in hip_code
        assert "texture(cubeShadow" not in hip_code
        assert "texture(cubeArrayShadow" not in hip_code
        assert " = tex2D<float4>(shadowMap" not in hip_code
        assert " = tex2D<float4>(cascades" not in hip_code
        assert " = tex2D<float4>(cubeShadow" not in hip_code
        assert " = tex2D<float4>(cubeArrayShadow" not in hip_code

    def test_texture_gather_calls_emit_hip_intrinsics_and_diagnostics(self):
        source_code = """
        shader Resources {
            sampler2d colorMap;
            sampler1darray rampLayers;
            sampler2darray layers;
            sampler3d volumeTex;
            sampler querySampler;

            void gatherResources(
                sampler2d paramTex,
                vec2 uv,
                vec3 uvLayer,
                ivec2 offset,
                ivec2 offsets[4],
                int selector
            ) {
                vec4 gathered = textureGather(colorMap, uv);
                vec4 gatheredComponent = textureGather(colorMap, querySampler, uv, 1);
                vec4 gatheredParam = textureGather(paramTex, uv, 2);
                vec4 gatheredDynamic = textureGather(colorMap, uv, selector);
                vec4 gatheredBadComponent = textureGather(colorMap, uv, 4);
                vec4 gatheredBadCoord = textureGather(colorMap, uvLayer, 0);
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
            "float4 gathered = tex2Dgather<float4>(colorMap, uv.x, uv.y);" in hip_code
        )
        assert (
            "float4 gatheredComponent = "
            "tex2Dgather<float4>(colorMap, uv.x, uv.y, 1);" in hip_code
        )
        assert (
            "float4 gatheredParam = "
            "tex2Dgather<float4>(paramTex, uv.x, uv.y, 2);" in hip_code
        )
        assert (
            "float4 gatheredDynamic = "
            "tex2Dgather<float4>(colorMap, uv.x, uv.y, selector);" in hip_code
        )
        assert (
            "float4 gatheredBadComponent = /* unsupported HIP sampled resource call: "
            "textureGather component literal must be 0, 1, 2, or 3 on sampler2D */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert (
            "float4 gatheredBadCoord = /* unsupported HIP sampled resource call: "
            "textureGather coordinate rank on sampler2D */ "
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

    def test_offset_and_projected_texture_calls_emit_hip_diagnostics(self):
        source_code = """
        shader Resources {
            sampler2d colorMap;
            samplercube cubeMap;
            sampler2dms msTex;
            sampler2dshadow shadowMap;
            image2D colorImage;
            sampler querySampler;

            void sampleResources(vec2 uv, vec3 uvw, ivec2 offset, ivec4 cubePixel) {
                vec4 offsetSample = textureOffset(colorMap, uv, offset);
                vec4 lodOffsetSample = textureLodOffset(
                    colorMap,
                    uv,
                    1.0,
                    offset
                );
                vec4 gradOffsetSample = textureGradOffset(
                    colorMap,
                    uv,
                    uv,
                    uv,
                    offset
                );
                vec4 projected = textureProj(colorMap, uvw);
                vec4 projectedOffset = textureProjOffset(
                    colorMap,
                    uvw,
                    offset
                );
                vec4 projectedLod = textureProjLod(colorMap, uvw, 1.0);
                vec4 projectedLodOffset = textureProjLodOffset(
                    colorMap,
                    uvw,
                    1.0,
                    offset
                );
                vec4 projectedGrad = textureProjGrad(
                    colorMap,
                    uvw,
                    uv,
                    uv
                );
                vec4 projectedGradOffset = textureProjGradOffset(
                    colorMap,
                    uvw,
                    uv,
                    uv,
                    offset
                );
                vec4 fetchedOffset = texelFetchOffset(colorMap, offset, 0, offset);
                vec4 imageOffset = texelFetchOffset(colorImage, offset, 0, offset);
                vec4 cubeOffset = texelFetchOffset(cubeMap, cubePixel, 0, offset);
                vec4 msOffset = texelFetchOffset(msTex, offset, 0, offset);
                float shadowFetchOffset = texelFetchOffset(
                    shadowMap,
                    offset,
                    0,
                    offset
                );
                float shadowProjected = textureCompareProj(
                    shadowMap,
                    vec4(uv, 0.5, 1.0)
                );
                float shadowProjectedLodOffset = textureCompareProjLodOffset(
                    shadowMap,
                    vec4(uv, 0.5, 1.0),
                    1.0,
                    offset
                );
                float shadowLodOffset = textureCompareLodOffset(
                    shadowMap,
                    uv,
                    0.5,
                    1.0,
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

        sampled_diagnostics = [
            "textureOffset",
            "textureLodOffset",
            "textureGradOffset",
            "textureProjOffset",
            "textureProjLodOffset",
            "textureProjGradOffset",
        ]
        for func_name in sampled_diagnostics:
            assert (
                f"/* unsupported HIP sampled resource call: "
                f"{func_name} on sampler2D */ "
                "make_float4(0.0f, 0.0f, 0.0f, 0.0f)" in hip_code
            )

        assert (
            "float4 projected = tex2D<float4>"
            "(colorMap, (uvw.x / uvw.z), (uvw.y / uvw.z));" in hip_code
        )
        assert (
            "float4 projectedLod = tex2DLod<float4>"
            "(colorMap, (uvw.x / uvw.z), (uvw.y / uvw.z), 1.0);" in hip_code
        )
        assert (
            "float4 projectedGrad = tex2DGrad<float4>"
            "(colorMap, (uvw.x / uvw.z), (uvw.y / uvw.z), uv, uv);" in hip_code
        )
        assert (
            "float4 fetchedOffset = tex2D<float4>"
            "(colorMap, (offset.x + offset.x), (offset.y + offset.y));" in hip_code
        )
        assert (
            "float4 msOffset = /* unsupported HIP multisample resource call: "
            "texelFetchOffset on sampler2DMS */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert (
            "float4 imageOffset = /* unsupported HIP sampled resource call: "
            "texelFetchOffset on image2D */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert (
            "float4 cubeOffset = /* unsupported HIP sampled resource call: "
            "texelFetchOffset on samplerCube */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert (
            "float shadowFetchOffset = /* unsupported HIP shadow resource call: "
            "texelFetchOffset on sampler2DShadow */ 0.0f;" in hip_code
        )
        assert (
            "float shadowProjected = /* unsupported HIP shadow resource call: "
            "textureCompareProj on sampler2DShadow */ 0.0f;" in hip_code
        )
        assert (
            "float shadowProjectedLodOffset = "
            "/* unsupported HIP shadow resource call: "
            "textureCompareProjLodOffset on sampler2DShadow */ 0.0f;" in hip_code
        )
        assert (
            "float shadowLodOffset = /* unsupported HIP shadow resource call: "
            "textureCompareLodOffset on sampler2DShadow */ 0.0f;" in hip_code
        )
        assert "textureOffset(" not in hip_code
        assert "textureLodOffset(" not in hip_code
        assert "textureGradOffset(" not in hip_code
        assert "textureProj(" not in hip_code
        assert "textureProjOffset(" not in hip_code
        assert "textureProjLod(" not in hip_code
        assert "textureProjLodOffset(" not in hip_code
        assert "textureProjGrad(" not in hip_code
        assert "textureProjGradOffset(" not in hip_code
        assert "texelFetchOffset(" not in hip_code
        assert "textureCompareProj(" not in hip_code
        assert "textureCompareProjLodOffset(" not in hip_code
        assert "textureCompareLodOffset(" not in hip_code

    def test_resource_type_keywords_emit_hip_resource_types(self):
        source_code = """
        shader Resources {
            sampler linearState;
            sampler1d ramp;
            sampler1darray rampLayers;
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
        assert "hipTextureObject_t ramp;" in declarations
        assert "hipTextureObject_t rampLayers;" in declarations
        assert "hipTextureObject_t colorMap;" in declarations
        assert "hipTextureObject_t volumeMap;" in declarations
        assert "hipTextureObject_t environmentMap;" in declarations
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
            "sampler1DArray",
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
        assert "float4 fetched = tex2D<float4>(paramTex, pixel.x, pixel.y);" in hip_code
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

    def test_texture_coordinate_shapes_emit_hip_helpers_or_diagnostics(self):
        source_code = """
        shader TextureCoordinateShapes {
            sampler1d lineTex;
            sampler1darray lineLayers;
            sampler2d tex;
            sampler2darray layers;
            sampler3d volume;
            samplercube cubeTex;
            samplercubearray cubeLayers;

            void sampleShapes(
                float u,
                vec2 uv,
                vec3 uvw,
                vec4 uvqw,
                float du,
                vec2 ddx2,
                vec2 ddy2,
                vec3 ddx3,
                vec3 ddy3,
                vec4 ddx4,
                vec4 ddy4
            ) {
                vec4 line = texture(lineTex, u);
                vec4 lineLayer = texture(lineLayers, uv);
                vec4 color = texture(tex, uv);
                vec4 colorLod = textureLod(tex, uv, 1.0);
                vec4 colorGrad = textureGrad(tex, uv, ddx2, ddy2);
                vec4 layer = texture(layers, uvw);
                vec4 volumeColor = texture(volume, uvw);
                vec4 volumeGrad = textureGrad(volume, uvw, ddx3, ddy3);
                vec4 cube = texture(cubeTex, uvw);
                vec4 cubeLayer = texture(cubeLayers, uvqw);
                vec4 badLine = texture(lineTex, uv);
                vec4 badLineLayer = texture(lineLayers, u);
                vec4 badTex = texture(tex, u);
                vec4 badLayer = texture(layers, uv);
                vec4 badVolume = texture(volume, uv);
                vec4 badCube = texture(cubeTex, uv);
                vec4 badCubeArray = texture(cubeLayers, uvw);
                vec4 badTexLod = textureLod(tex, u, 1.0);
                vec4 badTexGradCoord = textureGrad(tex, u, ddx2, ddy2);
                vec4 badTexGradDerivative = textureGrad(tex, uv, du, du);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "float4 line = tex1D<float4>(lineTex, u);" in hip_code
        assert (
            "float4 lineLayer = tex1DLayered<float4>(lineLayers, uv.x, uv.y);"
            in hip_code
        )
        assert "float4 color = tex2D<float4>(tex, uv.x, uv.y);" in hip_code
        assert "float4 colorLod = tex2DLod<float4>(tex, uv.x, uv.y, 1.0);" in hip_code
        assert (
            "float4 colorGrad = tex2DGrad<float4>(tex, uv.x, uv.y, ddx2, ddy2);"
            in hip_code
        )
        assert (
            "float4 layer = tex2DLayered<float4>(layers, uvw.x, uvw.y, uvw.z);"
            in hip_code
        )
        assert (
            "float4 volumeColor = tex3D<float4>(volume, uvw.x, uvw.y, uvw.z);"
            in hip_code
        )
        assert (
            "float4 volumeGrad = tex3DGrad<float4>"
            "(volume, uvw.x, uvw.y, uvw.z, "
            "make_float4(ddx3.x, ddx3.y, ddx3.z, 0.0f), "
            "make_float4(ddy3.x, ddy3.y, ddy3.z, 0.0f));" in hip_code
        )
        assert (
            "float4 cube = texCubemap<float4>(cubeTex, uvw.x, uvw.y, uvw.z);"
            in hip_code
        )
        assert (
            "float4 cubeLayer = texCubemapLayered<float4>"
            "(cubeLayers, uvqw.x, uvqw.y, uvqw.z, uvqw.w);" in hip_code
        )
        for name, func_name, texture_type in (
            ("badLine", "texture", "sampler1D"),
            ("badLineLayer", "texture", "sampler1DArray"),
            ("badTex", "texture", "sampler2D"),
            ("badLayer", "texture", "sampler2DArray"),
            ("badVolume", "texture", "sampler3D"),
            ("badCube", "texture", "samplerCube"),
            ("badCubeArray", "texture", "samplerCubeArray"),
            ("badTexLod", "textureLod", "sampler2D"),
            ("badTexGradCoord", "textureGrad", "sampler2D"),
        ):
            assert (
                f"float4 {name} = /* unsupported HIP sampled resource call: "
                f"{func_name} coordinate rank on {texture_type} */ "
                "make_float4(0.0f, 0.0f, 0.0f, 0.0f);"
            ) in hip_code
        assert (
            "float4 badTexGradDerivative = "
            "/* unsupported HIP sampled resource call: "
            "textureGrad derivative rank on sampler2D */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert "tex1D<float4>(lineTex, uv)" not in hip_code
        assert "tex1DLayered<float4>(lineLayers, u." not in hip_code
        assert "tex2D<float4>(tex, u." not in hip_code
        assert "tex2DLayered<float4>(layers, uv.x, uv.y, uv.z)" not in hip_code
        assert "tex3D<float4>(volume, uv.x, uv.y, uv.z)" not in hip_code
        assert "texCubemap<float4>(cubeTex, uv.x, uv.y, uv.z)" not in hip_code
        assert "texture(" not in hip_code
        assert "textureLod(" not in hip_code
        assert "textureGrad(" not in hip_code

    def test_texel_fetch_coordinate_shapes_emit_hip_helpers_or_diagnostics(self):
        source_code = """
        shader TexelFetchShapes {
            sampler1d lineTex;
            sampler1darray lineLayers;
            sampler2d tex;
            sampler2darray layers;
            sampler3d volume;
            samplercube cubeTex;
            samplercubearray cubeLayers;
            sampler2dshadow shadowTex;
            image2D colorImage;

            void fetchShapes(
                int x,
                int dx,
                ivec2 pixel,
                ivec2 delta,
                ivec3 voxel,
                ivec3 delta3,
                ivec4 cubeLayerTexel
            ) {
                vec4 fetchedLine = texelFetch(lineTex, x, 0);
                vec4 fetchedLineLayer = texelFetch(lineLayers, pixel, 0);
                vec4 fetched = texelFetch(tex, pixel, 0);
                vec4 fetchedLayer = texelFetch(layers, voxel, 0);
                vec4 fetchedVolume = texelFetch(volume, voxel, 0);
                vec4 offsetLine = texelFetchOffset(lineTex, x, 0, dx);
                vec4 offsetLineLayer = texelFetchOffset(lineLayers, pixel, 0, dx);
                vec4 offset2D = texelFetchOffset(tex, pixel, 0, delta);
                vec4 offsetLayer = texelFetchOffset(layers, voxel, 0, delta);
                vec4 offsetVolume = texelFetchOffset(volume, voxel, 0, delta3);
                vec4 badOffsetRank = texelFetchOffset(tex, pixel, 0, dx);
                vec4 badLine = texelFetch(lineTex, pixel, 0);
                vec4 badLineLayer = texelFetch(lineLayers, x, 0);
                vec4 badTex = texelFetch(tex, x, 0);
                vec4 badVolume = texelFetch(volume, pixel, 0);
                vec4 badCube = texelFetch(cubeTex, voxel, 0);
                vec4 badCubeArray = texelFetch(cubeLayers, cubeLayerTexel, 0);
                vec4 badImage = texelFetch(colorImage, pixel, 0);
                float badShadow = texelFetch(shadowTex, pixel, 0);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "float4 fetchedLine = tex1Dfetch<float4>(lineTex, x);" in hip_code
        assert (
            "float4 fetchedLineLayer = tex1DLayered<float4>"
            "(lineLayers, pixel.x, pixel.y);" in hip_code
        )
        assert "float4 fetched = tex2D<float4>(tex, pixel.x, pixel.y);" in hip_code
        assert (
            "float4 fetchedLayer = tex2DLayered<float4>"
            "(layers, voxel.x, voxel.y, voxel.z);" in hip_code
        )
        assert (
            "float4 fetchedVolume = tex3D<float4>(volume, voxel.x, voxel.y, voxel.z);"
            in hip_code
        )
        assert "float4 offsetLine = tex1Dfetch<float4>(lineTex, (x + dx));" in hip_code
        assert (
            "float4 offsetLineLayer = tex1DLayered<float4>"
            "(lineLayers, (pixel.x + dx), pixel.y);" in hip_code
        )
        assert (
            "float4 offset2D = tex2D<float4>"
            "(tex, (pixel.x + delta.x), (pixel.y + delta.y));" in hip_code
        )
        assert (
            "float4 offsetLayer = tex2DLayered<float4>"
            "(layers, (voxel.x + delta.x), (voxel.y + delta.y), voxel.z);" in hip_code
        )
        assert (
            "float4 offsetVolume = tex3D<float4>"
            "(volume, (voxel.x + delta3.x), (voxel.y + delta3.y), "
            "(voxel.z + delta3.z));" in hip_code
        )
        assert (
            "float4 badOffsetRank = /* unsupported HIP sampled resource call: "
            "texelFetchOffset offset rank on sampler2D */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        for name, texture_type in (
            ("badLine", "sampler1D"),
            ("badLineLayer", "sampler1DArray"),
            ("badTex", "sampler2D"),
            ("badVolume", "sampler3D"),
        ):
            assert (
                f"float4 {name} = /* unsupported HIP sampled resource call: "
                f"texelFetch coordinate rank on {texture_type} */ "
                "make_float4(0.0f, 0.0f, 0.0f, 0.0f);"
            ) in hip_code
        for name, texture_type in (
            ("badCube", "samplerCube"),
            ("badCubeArray", "samplerCubeArray"),
            ("badImage", "image2D"),
        ):
            assert (
                f"float4 {name} = /* unsupported HIP sampled resource call: "
                f"texelFetch on {texture_type} */ "
                "make_float4(0.0f, 0.0f, 0.0f, 0.0f);"
            ) in hip_code
        assert (
            "float badShadow = /* unsupported HIP shadow resource call: "
            "texelFetch on sampler2DShadow */ 0.0f;" in hip_code
        )
        assert "tex1Dfetch<float4>(lineTex, pixel)" not in hip_code
        assert "tex1DLayered<float4>(lineLayers, x." not in hip_code
        assert "tex2D<float4>(tex, x." not in hip_code
        assert "tex3D<float4>(volume, pixel.x, pixel.y, pixel.z)" not in hip_code
        assert "texelFetch(" not in hip_code
        assert "texelFetchOffset(" not in hip_code

    def test_image_coordinate_shapes_emit_hip_surface_helpers_or_diagnostics(self):
        source_code = """
        shader ImageCoordinateShapes {
            image2D colorImage;
            image2DArray layerImage;
            image3D volumeImage;
            uimage2D counters;
            iimage2D signedImage;

            void imageShapes(int x, ivec2 pixel, ivec3 voxel) {
                vec4 regular = imageLoad(colorImage, pixel);
                imageStore(colorImage, pixel, regular);
                vec4 layered = imageLoad(layerImage, voxel);
                imageStore(layerImage, voxel, layered);
                vec4 volume = imageLoad(volumeImage, voxel);
                imageStore(volumeImage, voxel, volume);
                uint count = imageLoad(counters, pixel);
                imageStore(counters, pixel, count);
                int signedValue = imageLoad(signedImage, pixel);
                imageStore(signedImage, pixel, signedValue);
                vec4 badRegular = imageLoad(colorImage, x);
                imageStore(colorImage, x, regular);
                vec4 badLayer = imageLoad(layerImage, pixel);
                imageStore(layerImage, pixel, layered);
                vec4 badVolume = imageLoad(volumeImage, pixel);
                imageStore(volumeImage, pixel, volume);
                uint badCounter = imageLoad(counters, x);
                imageStore(counters, x, count);
                int badSigned = imageLoad(signedImage, x);
                imageStore(signedImage, x, signedValue);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert (
            "float4 regular = cgl_surf2Dread<float4>"
            "(colorImage, pixel.x * sizeof(float4), pixel.y);" in hip_code
        )
        assert (
            "surf2Dwrite(regular, colorImage, pixel.x * sizeof(float4), pixel.y);"
            in hip_code
        )
        assert (
            "float4 layered = cgl_surf2DLayeredread<float4>"
            "(layerImage, voxel.x * sizeof(float4), voxel.y, voxel.z);" in hip_code
        )
        assert (
            "surf2DLayeredwrite(layered, layerImage, "
            "voxel.x * sizeof(float4), voxel.y, voxel.z);" in hip_code
        )
        assert (
            "float4 volume = cgl_surf3Dread<float4>"
            "(volumeImage, voxel.x * sizeof(float4), voxel.y, voxel.z);" in hip_code
        )
        assert (
            "unsigned int count = cgl_surf2Dread<uint>"
            "(counters, pixel.x * sizeof(uint), pixel.y);" in hip_code
        )
        assert (
            "int signedValue = cgl_surf2Dread<int>"
            "(signedImage, pixel.x * sizeof(int), pixel.y);" in hip_code
        )
        assert (
            "float4 badRegular = /* unsupported HIP image resource call: "
            "imageLoad coordinate rank on image2D */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert "imageStore coordinate rank on image2D */ ((void)0);" in hip_code
        assert (
            "float4 badLayer = /* unsupported HIP image resource call: "
            "imageLoad coordinate rank on image2DArray */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert "imageStore coordinate rank on image2DArray */ ((void)0);" in hip_code
        assert (
            "float4 badVolume = /* unsupported HIP image resource call: "
            "imageLoad coordinate rank on image3D */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert "imageStore coordinate rank on image3D */ ((void)0);" in hip_code
        assert (
            "unsigned int badCounter = /* unsupported HIP image resource call: "
            "imageLoad coordinate rank on uimage2D */ 0u;" in hip_code
        )
        assert "imageStore coordinate rank on uimage2D */ ((void)0);" in hip_code
        assert (
            "int badSigned = /* unsupported HIP image resource call: "
            "imageLoad coordinate rank on iimage2D */ 0;" in hip_code
        )
        assert "imageStore coordinate rank on iimage2D */ ((void)0);" in hip_code
        assert "cgl_surf2Dread<float4>(colorImage, x." not in hip_code
        assert (
            "cgl_surf2DLayeredread<float4>(layerImage, pixel.x, pixel.y, pixel.z)"
            not in hip_code
        )
        assert (
            "cgl_surf3Dread<float4>(volumeImage, pixel.x, pixel.y, pixel.z)"
            not in hip_code
        )
        assert "imageLoad(" not in hip_code
        assert "imageStore(" not in hip_code

    def test_1d_and_cube_image_calls_emit_hip_surface_helpers_or_diagnostics(self):
        source_code = """
        shader ImageSurfaceShapes {
            image1D lineImage;
            image1DArray lineLayers;
            imageCube cubeImage;
            uimage1D counters;
            uimage1DArray counterLayers;

            void imageShapes(int x, ivec2 xLayer, ivec3 cubeCoord, ivec2 pixel) {
                vec4 line = imageLoad(lineImage, x);
                imageStore(lineImage, x, line);
                vec4 lineLayer = imageLoad(lineLayers, xLayer);
                imageStore(lineLayers, xLayer, lineLayer);
                vec4 cube = imageLoad(cubeImage, cubeCoord);
                imageStore(cubeImage, cubeCoord, cube);
                uint count = imageLoad(counters, x);
                imageStore(counters, x, count);
                uint layerCount = imageLoad(counterLayers, xLayer);
                imageStore(counterLayers, xLayer, layerCount);
                vec4 badLine = imageLoad(lineImage, pixel);
                imageStore(lineLayers, x, lineLayer);
                vec4 badCube = imageLoad(cubeImage, pixel);
                imageStore(cubeImage, pixel, cube);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "hipSurfaceObject_t lineImage;" in hip_code
        assert "hipSurfaceObject_t lineLayers;" in hip_code
        assert "hipSurfaceObject_t cubeImage;" in hip_code
        assert "hipSurfaceObject_t counters;" in hip_code
        assert "hipSurfaceObject_t counterLayers;" in hip_code
        assert (
            "__device__ T cgl_surf1Dread(hipSurfaceObject_t surfObj, int x)" in hip_code
        )
        assert (
            "__device__ T cgl_surf1DLayeredread"
            "(hipSurfaceObject_t surfObj, int x, int layer)" in hip_code
        )
        assert (
            "__device__ T cgl_surfCubemapread"
            "(hipSurfaceObject_t surfObj, int x, int y, int face)" in hip_code
        )
        assert (
            "float4 line = cgl_surf1Dread<float4>"
            "(lineImage, x * sizeof(float4));" in hip_code
        )
        assert "surf1Dwrite(line, lineImage, x * sizeof(float4));" in hip_code
        assert (
            "float4 lineLayer = cgl_surf1DLayeredread<float4>"
            "(lineLayers, xLayer.x * sizeof(float4), xLayer.y);" in hip_code
        )
        assert (
            "surf1DLayeredwrite(lineLayer, lineLayers, "
            "xLayer.x * sizeof(float4), xLayer.y);" in hip_code
        )
        assert (
            "float4 cube = cgl_surfCubemapread<float4>"
            "(cubeImage, cubeCoord.x * sizeof(float4), cubeCoord.y, "
            "cubeCoord.z);" in hip_code
        )
        assert (
            "surfCubemapwrite(cube, cubeImage, cubeCoord.x * sizeof(float4), "
            "cubeCoord.y, cubeCoord.z);" in hip_code
        )
        assert (
            "unsigned int count = cgl_surf1Dread<uint>"
            "(counters, x * sizeof(uint));" in hip_code
        )
        assert "surf1Dwrite(count, counters, x * sizeof(uint));" in hip_code
        assert (
            "unsigned int layerCount = cgl_surf1DLayeredread<uint>"
            "(counterLayers, xLayer.x * sizeof(uint), xLayer.y);" in hip_code
        )
        assert (
            "surf1DLayeredwrite(layerCount, counterLayers, "
            "xLayer.x * sizeof(uint), xLayer.y);" in hip_code
        )
        assert (
            "float4 badLine = /* unsupported HIP image resource call: "
            "imageLoad coordinate rank on image1D */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert "imageStore coordinate rank on image1DArray */ ((void)0);" in hip_code
        assert (
            "float4 badCube = /* unsupported HIP image resource call: "
            "imageLoad coordinate rank on imageCube */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert "imageStore coordinate rank on imageCube */ ((void)0);" in hip_code
        assert "image1D lineImage" not in hip_code
        assert "image1DArray lineLayers" not in hip_code
        assert "uimage1D counters" not in hip_code
        assert "imageLoad(" not in hip_code
        assert "imageStore(" not in hip_code

    def test_cube_array_image_ir_calls_emit_hip_surface_helpers_or_diagnostics(self):
        cube_array_ast = ShaderNode(
            "CubeArrayImages",
            ExecutionModel.COMPUTE_KERNEL,
            global_variables=[
                VariableNode("cubeLayers", PrimitiveType("imageCubeArray")),
                VariableNode("cubeCounters", PrimitiveType("uimageCubeArray")),
            ],
            functions=[
                FunctionNode(
                    "useCubeArrays",
                    PrimitiveType("void"),
                    [],
                    [
                        VariableNode("coord", PrimitiveType("ivec4")),
                        VariableNode("badCoord", PrimitiveType("ivec3")),
                        VariableNode(
                            "cubeLayer",
                            PrimitiveType("vec4"),
                            FunctionCallNode(
                                IdentifierNode("imageLoad"),
                                [IdentifierNode("cubeLayers"), IdentifierNode("coord")],
                            ),
                        ),
                        FunctionCallNode(
                            IdentifierNode("imageStore"),
                            [
                                IdentifierNode("cubeLayers"),
                                IdentifierNode("coord"),
                                IdentifierNode("cubeLayer"),
                            ],
                        ),
                        VariableNode(
                            "counter",
                            PrimitiveType("uint"),
                            FunctionCallNode(
                                IdentifierNode("imageLoad"),
                                [
                                    IdentifierNode("cubeCounters"),
                                    IdentifierNode("coord"),
                                ],
                            ),
                        ),
                        FunctionCallNode(
                            IdentifierNode("imageStore"),
                            [
                                IdentifierNode("cubeCounters"),
                                IdentifierNode("coord"),
                                IdentifierNode("counter"),
                            ],
                        ),
                        VariableNode(
                            "badLayer",
                            PrimitiveType("vec4"),
                            FunctionCallNode(
                                IdentifierNode("imageLoad"),
                                [
                                    IdentifierNode("cubeLayers"),
                                    IdentifierNode("badCoord"),
                                ],
                            ),
                        ),
                        FunctionCallNode(
                            IdentifierNode("imageStore"),
                            [
                                IdentifierNode("cubeLayers"),
                                IdentifierNode("badCoord"),
                                IdentifierNode("cubeLayer"),
                            ],
                        ),
                    ],
                    qualifiers=["compute"],
                )
            ],
        )

        hip_code = HipCodeGen().generate(cube_array_ast)

        assert "hipSurfaceObject_t cubeLayers;" in hip_code
        assert "hipSurfaceObject_t cubeCounters;" in hip_code
        assert (
            "__device__ T cgl_surfCubemapLayeredread"
            "(hipSurfaceObject_t surfObj, int x, int y, int face, int layer)"
            in hip_code
        )
        assert (
            "float4 cubeLayer = cgl_surfCubemapLayeredread<float4>"
            "(cubeLayers, coord.x * sizeof(float4), coord.y, coord.z, "
            "coord.w);" in hip_code
        )
        assert (
            "surfCubemapLayeredwrite(cubeLayer, cubeLayers, "
            "coord.x * sizeof(float4), coord.y, coord.z, coord.w);" in hip_code
        )
        assert (
            "unsigned int counter = cgl_surfCubemapLayeredread<uint>"
            "(cubeCounters, coord.x * sizeof(uint), coord.y, coord.z, "
            "coord.w);" in hip_code
        )
        assert (
            "surfCubemapLayeredwrite(counter, cubeCounters, "
            "coord.x * sizeof(uint), coord.y, coord.z, coord.w);" in hip_code
        )
        assert (
            "float4 badLayer = /* unsupported HIP image resource call: "
            "imageLoad coordinate rank on imageCubeArray */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert "imageStore coordinate rank on imageCubeArray */ ((void)0);" in hip_code
        assert "imageCubeArray cubeLayers" not in hip_code
        assert "uimageCubeArray cubeCounters" not in hip_code
        assert "imageLoad(" not in hip_code
        assert "imageStore(" not in hip_code

    def test_nested_resource_arrays_emit_hip_texture_and_surface_calls(self):
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

        assert "hipTextureObject_t textureGrid[2][3];" in hip_code
        assert "hipTextureObject_t layerGrid[2][2];" in hip_code
        assert "hipTextureObject_t msGrid[2][2];" in hip_code
        assert "hipSurfaceObject_t imageGrid[2][4];" in hip_code
        assert "hipSurfaceObject_t counterGrid[2][2];" in hip_code
        assert (
            "__device__ void processNested("
            "hipTextureObject_t paramTextures[2][3], "
            "hipSurfaceObject_t paramImages[2][4], int layer, int slot, "
            "float2 uv, float3 uvLayer, int2 pixel, int3 pixelLayer, "
            "int sampleIndex)" in hip_code
        )
        assert (
            "float4 sampled = tex2D<float4>(textureGrid[layer][slot], uv.x, uv.y);"
            in hip_code
        )
        assert (
            "float4 sampledLayer = tex2DLayered<float4>"
            "(layerGrid[layer][slot], uvLayer.x, uvLayer.y, uvLayer.z);" in hip_code
        )
        assert (
            "float4 mip = tex2DLod<float4>(textureGrid[layer][slot], uv.x, uv.y, 1.0);"
            in hip_code
        )
        assert (
            "float4 grad = tex2DGrad<float4>(textureGrid[layer][slot], uv.x, uv.y, uv, uv);"
            in hip_code
        )
        assert (
            "float4 sampledParam = tex2D<float4>(paramTextures[layer][slot], uv.x, uv.y);"
            in hip_code
        )
        assert (
            "float4 gathered = tex2Dgather<float4>"
            "(textureGrid[layer][slot], uv.x, uv.y, 1);" in hip_code
        )
        assert (
            "float4 fetched = tex2D<float4>(textureGrid[layer][slot], pixel.x, pixel.y);"
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
            "__device__ void useDynamic(hipTextureObject_t* dynTextures, "
            "hipTextureObject_t* dynLayers, hipSurfaceObject_t* dynImages, "
            "hipSurfaceObject_t* dynCounters, int i, float2 uv, "
            "float3 uvLayer, int2 pixel)" in hip_code
        )
        assert (
            "__device__ void useDynamicGrid("
            "hipTextureObject_t (*dynGrid)[3], "
            "hipSurfaceObject_t (*dynGridImages)[3], int layer, int slot, "
            "float2 uv, int2 pixel)" in hip_code
        )
        assert "float4 sampled = tex2D<float4>(dynTextures[i], uv.x, uv.y);" in hip_code
        assert (
            "float4 layered = tex2DLayered<float4>"
            "(dynLayers[i], uvLayer.x, uvLayer.y, uvLayer.z);" in hip_code
        )
        assert (
            "float4 fetched = tex2D<float4>(dynTextures[i], pixel.x, pixel.y);"
            in hip_code
        )
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
        assert (
            "float4 sampled = tex2D<float4>(dynGrid[layer][slot], uv.x, uv.y);"
            in hip_code
        )
        assert (
            "float4 color = cgl_surf2Dread<float4>"
            "(dynGridImages[layer][slot], pixel.x * sizeof(float4), pixel.y);"
            in hip_code
        )
        assert (
            "surf2Dwrite(color, dynGridImages[layer][slot], "
            "pixel.x * sizeof(float4), pixel.y);" in hip_code
        )
        assert "hipTextureObject_t* dynGrid[3]" not in hip_code
        assert "hipSurfaceObject_t* dynGridImages[3]" not in hip_code
        assert "texture(" not in hip_code
        assert "texelFetch(" not in hip_code
        assert "imageLoad(" not in hip_code
        assert "imageStore(" not in hip_code
        assert "CglResourceQueryInfo" not in hip_code

    def test_unsized_resource_arrays_emit_hip_pointer_shapes(self):
        source_code = """
        shader UnsizedHIPResources {
            sampler linearSamplers[];
            sampler2d textures[];
            image2D images[];
            RWStructuredBuffer<uint> counters[];
            ByteAddressBuffer bytes[];

            vec4 process(uint index, vec2 uv, ivec2 pixel) {
                vec4 sampled = texture(textures[index], uv);
                vec4 loaded = imageLoad(images[index], pixel);
                imageStore(images[index], pixel, sampled + loaded);
                uint count = counters[index].Load(0u);
                uint raw = bytes[index].Load(4u);
                return sampled + loaded + vec4(float(count + raw));
            }

            compute {
                void main() {}
            }
        }
        """

        hip_code = HipCodeGen().generate(Parser(Lexer(source_code).tokens).parse())

        assert "hipTextureObject_t* linearSamplers;" in hip_code
        assert "hipTextureObject_t* textures;" in hip_code
        assert "hipSurfaceObject_t* images;" in hip_code
        assert "unsigned int** counters;" in hip_code
        assert "const unsigned char** bytes;" in hip_code
        assert (
            "// CrossGL resource metadata: name=linearSamplers kind=sampler set=0 "
            "binding=0 binding_source=automatic"
        ) in hip_code
        assert (
            "// CrossGL resource metadata: name=textures kind=texture set=0 "
            "binding=0 binding_source=automatic"
        ) in hip_code
        assert (
            "// CrossGL resource metadata: name=images kind=image set=0 "
            "binding=0 binding_source=automatic"
        ) in hip_code
        assert (
            "// CrossGL resource metadata: name=counters kind=buffer set=0 "
            "binding=0 binding_source=automatic"
        ) in hip_code
        assert (
            "// CrossGL resource metadata: name=bytes kind=buffer set=0 "
            "binding=1 binding_source=automatic"
        ) in hip_code
        assert (
            "float4 sampled = tex2D<float4>(textures[index], uv.x, uv.y);" in hip_code
        )
        assert (
            "float4 loaded = cgl_surf2Dread<float4>"
            "(images[index], pixel.x * sizeof(float4), pixel.y);" in hip_code
        )
        assert (
            "surf2Dwrite(cgl_float4_add(sampled, loaded), images[index], "
            "pixel.x * sizeof(float4), pixel.y);" in hip_code
        )
        assert "unsigned int count = counters[index][0u];" in hip_code
        assert (
            "unsigned int raw = cgl_byte_address_load_uint(bytes[index], 4u);"
            in hip_code
        )
        assert "sampler linearSamplers[]" not in hip_code
        assert "sampler2d textures[]" not in hip_code
        assert "image2D images[]" not in hip_code
        assert "RWStructuredBuffer<uint> counters[]" not in hip_code
        assert "ByteAddressBuffer bytes[]" not in hip_code
        assert "texture(" not in hip_code
        assert "imageLoad(" not in hip_code
        assert "imageStore(" not in hip_code

    def test_resource_binding_metadata_comments_emit_hip_bindings(self):
        source_code = """
        shader HipResourceBindings {
            sampler2D colorMap @set(2) @binding(5);
            sampler linearSampler @binding(1);
            @binding(3) RWStructuredBuffer<int> counters[2];
            Texture2D hlslTexture @register(t7, space3);

            @binding(6)
            cbuffer Camera {
                float exposure;
            };

            image2D outImage;

            compute {
                void main() {}
            }
        }
        """

        hip_code = HipCodeGen().generate(Parser(Lexer(source_code).tokens).parse())

        assert (
            "// CrossGL resource metadata: name=colorMap kind=texture set=2 "
            "binding=5 binding_source=explicit" in hip_code
        )
        assert (
            "// CrossGL resource metadata: name=linearSampler kind=sampler set=0 "
            "binding=1 binding_source=explicit" in hip_code
        )
        assert (
            "// CrossGL resource metadata: name=counters kind=buffer set=0 "
            "binding=3 binding_source=explicit count=2" in hip_code
        )
        assert (
            "// CrossGL resource metadata: name=hlslTexture kind=texture set=3 "
            "binding=7 binding_source=explicit register=t7,space3" in hip_code
        )
        assert (
            "// CrossGL resource metadata: name=outImage kind=image set=0 "
            "binding=0 binding_source=automatic" in hip_code
        )
        assert (
            "// CrossGL resource metadata: name=Camera kind=cbuffer set=0 "
            "binding=6 binding_source=explicit" in hip_code
        )

    def test_duplicate_resource_bindings_are_rejected_for_hip_codegen(self):
        duplicate_texture_binding = """
        shader DuplicateHipTextureBindings {
            @binding(2) sampler2D firstTexture;
            @binding(2) sampler2D secondTexture;

            compute {
                void main() {}
            }
        }
        """

        with pytest.raises(ValueError, match="Conflicting HIP resource binding"):
            HipCodeGen().generate(
                Parser(Lexer(duplicate_texture_binding).tokens).parse()
            )

        overlapping_buffer_range = """
        shader DuplicateHipBufferBindings {
            @binding(3) RWStructuredBuffer<int> counters[2];
            @binding(4)
            cbuffer Camera {
                float exposure;
            };

            compute {
                void main() {}
            }
        }
        """

        with pytest.raises(ValueError, match="Conflicting HIP resource binding"):
            HipCodeGen().generate(
                Parser(Lexer(overlapping_buffer_range).tokens).parse()
            )

    def test_forwarded_dynamic_resource_arrays_emit_hip_metadata_arguments(self):
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
            "hipTextureObject_t (*dynGrid)[3], int layer, int slot, float2 uv)"
            in hip_code
        )
        assert (
            "__device__ float4 forwardSample("
            "hipTextureObject_t (*dynGrid)[3], int layer, int slot, float2 uv)"
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
            "__device__ void leafQuery(hipTextureObject_t (*dynGrid)[3], "
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
        assert "return tex2D<float4>(dynGrid[layer][slot], uv.x, uv.y);" in hip_code
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
            "__device__ void leafMixed(hipTextureObject_t (*dynamicGrid)[3], "
            "CglResourceQueryInfo (*dynamicGrid_metadata)[3], "
            "hipTextureObject_t fixedRow[3], "
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
        assert (
            "float4 dynamicSample = tex2D<float4>(dynamicGrid[layer][slot], uv.x, uv.y);"
            in hip_code
        )
        assert (
            "float4 fixedSample = tex2D<float4>(fixedRow[slot], uv.x, uv.y);"
            in hip_code
        )
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
            "__device__ void leafShuffle(hipTextureObject_t (*dynA)[3], "
            "CglResourceQueryInfo (*dynA_metadata)[3], "
            "hipSurfaceObject_t fixedImageRow[3], "
            "CglResourceQueryInfo fixedImageRow_metadata[3], "
            "hipTextureObject_t fixedTexRow[3], "
            "CglResourceQueryInfo fixedTexRow_metadata[3], "
            "hipSurfaceObject_t (*dynImageGrid)[3], "
            "CglResourceQueryInfo (*dynImageGrid_metadata)[3], "
            "hipTextureObject_t (*dynAlias)[3], "
            "CglResourceQueryInfo (*dynAlias_metadata)[3], "
            "int layer, int slot, float2 uv, int2 pixel)" in hip_code
        )
        assert (
            "__device__ void midForward(hipTextureObject_t (*firstDyn)[3], "
            "CglResourceQueryInfo (*firstDyn_metadata)[3], "
            "hipTextureObject_t firstFixed[3], "
            "CglResourceQueryInfo firstFixed_metadata[3], "
            "hipSurfaceObject_t (*firstDynImages)[3], "
            "CglResourceQueryInfo (*firstDynImages_metadata)[3], "
            "hipSurfaceObject_t firstFixedImages[3], "
            "CglResourceQueryInfo firstFixedImages_metadata[3], "
            "int layer, int slot, float2 uv, int2 pixel)" in hip_code
        )
        assert (
            "__device__ void topForward(hipTextureObject_t topFixedTex[3], "
            "CglResourceQueryInfo topFixedTex_metadata[3], "
            "hipSurfaceObject_t (*topDynImages)[3], "
            "CglResourceQueryInfo (*topDynImages_metadata)[3], "
            "hipTextureObject_t (*topDynTex)[3], "
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
        assert (
            "float4 sampledA = tex2D<float4>(dynA[layer][slot], uv.x, uv.y);"
            in hip_code
        )
        assert (
            "float4 sampledFixed = tex2D<float4>(fixedTexRow[slot], uv.x, uv.y);"
            in hip_code
        )
        assert (
            "float4 sampledAlias = tex2D<float4>(dynAlias[layer][slot], uv.x, uv.y);"
            in hip_code
        )
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
            "__device__ float4 chooseBranch(hipTextureObject_t (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "hipTextureObject_t fixedTex[3], "
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
            "__device__ float4 sampleLoop(hipTextureObject_t (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "hipTextureObject_t fixedTex[3], "
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
        assert (
            "float4 sampled = tex2D<float4>(dynTex[layer][slot], uv.x, uv.y);"
            in hip_code
        )
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
            "accum = cgl_float4_add(accum, tex2D<float4>(fixedTex[fixedSlot], uv.x, uv.y));"
            in hip_code
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
            "hipTextureObject_t (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "hipSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], float2 uv, int2 pixel)"
            in hip_code
        )
        assert "for (int layer = 0; (layer < 2); layer++)" in hip_code
        assert "for (int slot = 0; (slot < 3); slot++)" in hip_code
        assert "break;" in hip_code
        assert "continue;" in hip_code
        assert (
            "float4 sampled = tex2D<float4>(dynTex[layer][slot], uv.x, uv.y);"
            in hip_code
        )
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
            "__device__ float4 samplePacked(hipTextureObject_t (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], SampleParams params, "
            "hipSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], float weight, "
            "hipTextureObject_t fixedTex[3], "
            "CglResourceQueryInfo fixedTex_metadata[3], "
            "hipSurfaceObject_t fixedImages[3], "
            "CglResourceQueryInfo fixedImages_metadata[3])" in hip_code
        )
        assert (
            "__device__ float4 passPacked(SampleParams params, "
            "hipTextureObject_t (*firstDyn)[3], "
            "CglResourceQueryInfo (*firstDyn_metadata)[3], float weight, "
            "hipSurfaceObject_t (*firstImages)[3], "
            "CglResourceQueryInfo (*firstImages_metadata)[3], "
            "hipTextureObject_t fixedTex[3], "
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
            "float4 dynamicSample = tex2D<float4>"
            "(dynTex[params.layer][params.slot], params.uv.x, params.uv.y);" in hip_code
        )
        assert (
            "float4 fixedSample = tex2D<float4>(fixedTex[params.slot], params.uv.x, params.uv.y);"
            in hip_code
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
            "hipTextureObject_t (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "hipSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], int layer, int slot, "
            "float2 uv, int2 pixel)" in hip_code
        )
        assert (
            "__device__ SampleResult buildConstructed("
            "hipTextureObject_t (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "hipSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], int layer, int slot, "
            "float2 uv, int2 pixel)" in hip_code
        )
        assert (
            "__device__ float4 consumeResults(hipTextureObject_t (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "hipSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], int layer, int slot, "
            "float2 uv, int2 pixel)" in hip_code
        )
        assert (
            "result.sampled = tex2D<float4>(dynTex[layer][slot], uv.x, uv.y);"
            in hip_code
        )
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
            "return SampleResult(tex2D<float4>(dynTex[layer][slot], uv.x, uv.y), "
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
            "hipTextureObject_t (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "hipSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], int layer, int slot, "
            "float2 uv, int2 pixel)" in hip_code
        )
        assert (
            "envelope.result.sampled = tex2D<float4>(dynTex[layer][slot], uv.x, uv.y);"
            in hip_code
        )
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
            "envelope.bias = cgl_float4_add(tex2D<float4>(dynTex[layer][slot], uv.x, uv.y), "
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
            "hipTextureObject_t (*dynTex)[3], "
            "CglResourceQueryInfo (*dynTex_metadata)[3], "
            "hipSurfaceObject_t (*dynImages)[3], "
            "CglResourceQueryInfo (*dynImages_metadata)[3], int layer, int slot, "
            "float2 uv, int2 pixel)" in hip_code
        )
        assert (
            "envelope.results[slot].sampled = tex2D<float4>(dynTex[layer][slot], uv.x, uv.y);"
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
            "hipTextureObject_t (*dynTex)[3], "
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
        assert (
            "envelope.result.sampled = tex2D<float4>(dynTex[layer][slot], uv.x, uv.y);"
            in hip_code
        )
        assert (
            "envelope.result.loaded = cgl_surf2Dread<float4>"
            "(dynImages[layer][slot], pixel.x * sizeof(float4), pixel.y);" in hip_code
        )
        assert (
            "envelope.result.sampled = tex2D<float4>(dynTex[fallback][slot], uv.x, uv.y);"
            in hip_code
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
            "envelope.accum = cgl_float4_add(envelope.accum, tex2D<float4>(dynTex[layer][i], uv.x, uv.y));"
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
            "hipTextureObject_t (*dynTex)[3], "
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
            "__device__ float4 consumeAdjusted(hipTextureObject_t (*dynTex)[3], "
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

    def test_structured_and_byte_address_buffers_emit_hip_pointer_helpers(self):
        source_code = """
        shader HIPBufferResources {
            struct Particle {
                float weight;
            };

            StructuredBuffer<int> input;
            RWStructuredBuffer<int> values;
            RWStructuredBuffer<Particle> particles;
            StructuredBuffer<uint> readonlyCounts[2];
            ByteAddressBuffer readBytes;
            RWByteAddressBuffer writeBytes;
            RWByteAddressBuffer byteBuffers[2];

            void process(
                StructuredBuffer<float> inputValues,
                RWStructuredBuffer<float> outputValues,
                ByteAddressBuffer paramRead,
                RWByteAddressBuffer paramWrite,
                uint index,
                uint which,
                float value,
                Particle p,
                uint rawValue,
                uint2 pair,
                uint3 triple,
                uint4 quad
            ) {
                int g = input.Load(index);
                values.Store(index, g);
                values[index] = g + 1;
                int loadedWrite = values[index];
                float localValue = buffer_load(inputValues, index);
                buffer_store(outputValues, index, localValue + value);
                Particle particleValue = particles.Load(index);
                particles.Store(index, p);
                uint count = readonlyCounts[which].Load(index);
                uint raw = readBytes.Load(index);
                uint2 rawPair = paramRead.Load2(index + 4u);
                uint3 rawTriple = byteBuffers[which].Load3(index);
                uint4 rawQuad = writeBytes.Load4(index);
                uint rawGeneric = buffer_load(paramRead, index);
                writeBytes.Store(index, rawValue);
                paramWrite.Store2(index, pair);
                byteBuffers[which].Store3(index, triple);
                writeBytes.Store4(index, quad);
                buffer_store(paramWrite, index, rawGeneric);
            }

            void rejectReadOnlyStores(
                StructuredBuffer<int> readOnly,
                ByteAddressBuffer readOnlyBytes,
                uint index,
                int value,
                uint rawValue
            ) {
                readOnly.Store(index, value);
                buffer_store(readOnly, index, value);
                readOnlyBytes.Store(index, rawValue);
                buffer_store(readOnlyBytes, index, rawValue);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "const int* input;" in hip_code
        assert "int* values;" in hip_code
        assert "Particle* particles;" in hip_code
        assert "const unsigned int* readonlyCounts[2];" in hip_code
        assert "const unsigned char* readBytes;" in hip_code
        assert "unsigned char* writeBytes;" in hip_code
        assert "unsigned char* byteBuffers[2];" in hip_code
        assert (
            "__device__ void process(const float* inputValues, float* outputValues, "
            "const unsigned char* paramRead, unsigned char* paramWrite, "
            "unsigned int index, unsigned int which, float value, Particle p, "
            "unsigned int rawValue, uint2 pair, uint3 triple, uint4 quad)"
        ) in hip_code
        assert "int g = input[index];" in hip_code
        assert "values[index] = g;" in hip_code
        assert "values[index] = (g + 1);" in hip_code
        assert "int loadedWrite = values[index];" in hip_code
        assert "float localValue = inputValues[index];" in hip_code
        assert "outputValues[index] = (localValue + value);" in hip_code
        assert "Particle particleValue = particles[index];" in hip_code
        assert "particles[index] = p;" in hip_code
        assert "unsigned int count = readonlyCounts[which][index];" in hip_code
        assert (
            "__device__ inline uint cgl_byte_address_load_uint"
            "(const unsigned char* buffer, uint offset)"
        ) in hip_code
        assert "return *reinterpret_cast<const uint*>(buffer + offset);" in hip_code
        assert (
            "__device__ inline uint4 cgl_byte_address_load_uint4"
            "(const unsigned char* buffer, uint offset)"
        ) in hip_code
        assert (
            "__device__ inline void cgl_byte_address_store_uint4"
            "(unsigned char* buffer, uint offset, uint4 value)"
        ) in hip_code
        assert "unsigned int raw = cgl_byte_address_load_uint(readBytes, index);" in (
            hip_code
        )
        assert (
            "uint2 rawPair = cgl_byte_address_load_uint2(paramRead, (index + 4u));"
            in hip_code
        )
        assert (
            "uint3 rawTriple = cgl_byte_address_load_uint3(byteBuffers[which], index);"
            in hip_code
        )
        assert (
            "uint4 rawQuad = cgl_byte_address_load_uint4(writeBytes, index);"
            in hip_code
        )
        assert (
            "unsigned int rawGeneric = cgl_byte_address_load_uint(paramRead, index);"
            in hip_code
        )
        assert "cgl_byte_address_store_uint(writeBytes, index, rawValue);" in hip_code
        assert "cgl_byte_address_store_uint2(paramWrite, index, pair);" in hip_code
        assert (
            "cgl_byte_address_store_uint3(byteBuffers[which], index, triple);"
            in hip_code
        )
        assert "cgl_byte_address_store_uint4(writeBytes, index, quad);" in hip_code
        assert "cgl_byte_address_store_uint(paramWrite, index, rawGeneric);" in hip_code
        assert (
            "/* unsupported HIP structured buffer call: "
            "Store on StructuredBuffer<int> */ ((void)0);" in hip_code
        )
        assert (
            "/* unsupported HIP structured buffer call: "
            "buffer_store on StructuredBuffer<int> */ ((void)0);" in hip_code
        )
        assert (
            "/* unsupported HIP byte-address buffer call: "
            "Store on ByteAddressBuffer */ ((void)0);" in hip_code
        )
        assert (
            "/* unsupported HIP byte-address buffer call: "
            "buffer_store on ByteAddressBuffer */ ((void)0);" in hip_code
        )
        for raw_type in [
            "StructuredBuffer<int> input;",
            "StructuredBuffer<float> inputValues",
            "RWStructuredBuffer<int> values;",
            "RWStructuredBuffer<float> outputValues",
            "ByteAddressBuffer readBytes",
            "RWByteAddressBuffer writeBytes",
        ]:
            assert raw_type not in hip_code
        assert ".Load(" not in hip_code
        assert ".Store(" not in hip_code
        assert "buffer_load(" not in hip_code
        assert "buffer_store(" not in hip_code

    def test_structured_buffer_dimensions_emit_hip_length_sidecars(self):
        source_code = """
        shader StructuredBufferDimensionsHIP {
            RWStructuredBuffer<int> values;
            StructuredBuffer<uint> readonlyCounts[2];

            uint countOne(StructuredBuffer<uint> input) {
                return buffer_dimensions(input);
            }

            uint countInner(RWStructuredBuffer<int> localValues, uint index) {
                uint len = buffer_dimensions(localValues);
                return len + index;
            }

            uint countValues(RWStructuredBuffer<int> localValues) {
                uint len;
                buffer_dimensions(localValues, len);
                return countInner(localValues, 0);
            }

            void process(uint which) {
                uint globalLen = buffer_dimensions(values);
                uint arrayLen;
                buffer_dimensions(readonlyCounts[which], arrayLen);
                values.GetDimensions(globalLen);
                uint helperLen = countValues(values);
                uint oneLen = countOne(readonlyCounts[which]);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "int* values;" in hip_code
        assert "const uint* values_length;" in hip_code
        assert "const unsigned int* readonlyCounts[2];" in hip_code
        assert "const uint* readonlyCounts_length[2];" in hip_code
        assert (
            "__device__ unsigned int countOne(const unsigned int* input, "
            "const uint* input_length)"
        ) in hip_code
        assert (
            "__device__ unsigned int countInner(int* localValues, "
            "const uint* localValues_length, unsigned int index)"
        ) in hip_code
        assert (
            "__device__ unsigned int countValues(int* localValues, "
            "const uint* localValues_length)"
        ) in hip_code
        assert "return input_length[0];" in hip_code
        assert "unsigned int len = localValues_length[0];" in hip_code
        assert "len = localValues_length[0];" in hip_code
        assert "return countInner(localValues, localValues_length, 0);" in hip_code
        assert "unsigned int globalLen = values_length[0];" in hip_code
        assert "arrayLen = readonlyCounts_length[which][0];" in hip_code
        assert "globalLen = values_length[0];" in hip_code
        assert "unsigned int helperLen = countValues(values, values_length);" in (
            hip_code
        )
        assert (
            "unsigned int oneLen = countOne("
            "readonlyCounts[which], readonlyCounts_length[which]);"
        ) in hip_code
        assert "buffer_dimensions(" not in hip_code
        assert ".GetDimensions(" not in hip_code

    def test_glsl_buffer_blocks_emit_hip_structs_pointers_and_metadata(self):
        source_code = """
        layout(std430, binding = 3) readonly buffer float values[];

        layout(std140, binding = 4) buffer ParticleBlock {
            vec4 positions[2];
            uint count;
        } particles;

        float readValue(uint index) {
            return values[index];
        }

        void writeParticle(uint index, vec4 value) {
            particles.positions[index] = value;
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert (
            "// CrossGL resource metadata: name=values kind=glsl_buffer_block "
            "layout=std430 access=readonly"
        ) in hip_code
        assert "const float* values;" in hip_code
        assert "struct ParticleBlock" in hip_code
        assert "float4 positions[2];" in hip_code
        assert "unsigned int count;" in hip_code
        assert (
            "// CrossGL resource metadata: name=particles kind=glsl_buffer_block "
            "layout=std140"
        ) in hip_code
        assert "ParticleBlock particles;" in hip_code
        assert "return values[index];" in hip_code
        assert "particles.positions[index] = value;" in hip_code

    def test_glsl_buffer_block_runtime_arrays_and_atomics_emit_hip(self):
        source_code = """
        layout(std430, binding = 1) buffer CounterBlock {
            uint counters[];
            int signedCounters[];
        } counters;

        uint addCounter(uint index, uint value) {
            return atomicAdd(counters.counters[index], value);
        }

        uint swapCounter(uint index, uint expected, uint replacement) {
            return atomicCompSwap(counters.counters[index], expected, replacement);
        }

        int maxSigned(uint index, int value) {
            return atomicMax(counters.signedCounters[index], value);
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "unsigned int* counters;" in hip_code
        assert "int* signedCounters;" in hip_code
        assert "return atomicAdd(&counters.counters[index], value);" in hip_code
        assert (
            "return atomicCAS(&counters.counters[index], expected, replacement);"
        ) in hip_code
        assert "return atomicMax(&counters.signedCounters[index], value);" in hip_code
        assert "atomicAdd(counters.counters[index]" not in hip_code

    def test_glsl_buffer_block_access_diagnostics_emit_hip(self):
        source_code = """
        layout(std430, binding = 1) readonly buffer ReadonlyBlock {
            uint counters[];
            uint value;
        } readBlock;

        layout(std430, binding = 2) writeonly buffer WriteonlyBlock {
            uint counters[];
        } writeBlock;

        uint blockedAtomic(uint index, uint value) {
            return atomicAdd(readBlock.counters[index], value);
        }

        void blockedStore(uint value) {
            readBlock.value = value;
        }

        uint blockedRead(uint index) {
            return writeBlock.counters[index];
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert (
            "return /* unsupported HIP GLSL buffer block atomicAdd: resource "
            "'readBlock' is readonly */ 0u;"
        ) in hip_code
        assert (
            "/* unsupported HIP GLSL buffer block assignment: resource "
            "'readBlock' is readonly */ ((void)0);"
        ) in hip_code
        assert (
            "return /* unsupported HIP GLSL buffer block load: resource "
            "'writeBlock' is writeonly */ 0u;"
        ) in hip_code

    def test_glsl_buffer_block_arrays_emit_hip_contract_evidence(self):
        source_code = """
        struct RuntimeLeaf {
            float mass;
        };

        struct RuntimeBlock {
            uint count;
            RuntimeLeaf leaves[];
        };

        struct FixedBlock {
            mat4 transform;
            vec3 values[2];
        };

        RuntimeBlock readBlocks[] @glsl_buffer_block(std430) @binding(8) @readonly;
        RuntimeBlock writeBlocks[] @glsl_buffer_block(std430) @binding(9);
        FixedBlock fixedBlocks[2] @glsl_buffer_block(std140) @binding(10);
        FixedBlock outputBlocks[2] @glsl_buffer_block(std140)
            @binding(12) @writeonly;

        float readRuntime(uint blockIndex, uint itemIndex) {
            return readBlocks[blockIndex].leaves[itemIndex].mass;
        }

        void writeRuntime(uint blockIndex, uint itemIndex, RuntimeLeaf leaf) {
            writeBlocks[blockIndex].leaves[itemIndex] = leaf;
        }

        vec3 readFixed(uint blockIndex, uint itemIndex) {
            return fixedBlocks[blockIndex].values[itemIndex];
        }

        void writeFixed(uint blockIndex, mat4 transform) {
            fixedBlocks[blockIndex].transform = transform;
        }

        void blockedWrite(uint blockIndex, uint value) {
            readBlocks[blockIndex].count = value;
        }

        vec3 blockedRead(uint blockIndex, uint itemIndex) {
            return outputBlocks[blockIndex].values[itemIndex];
        }
        """

        hip_code = HipCodeGen().generate(Parser(Lexer(source_code).tokens).parse())

        assert "RuntimeLeaf* leaves;" in hip_code
        assert (
            "// CrossGL resource metadata: name=readBlocks kind=glsl_buffer_block "
            "layout=std430 access=readonly set=0 binding=8 binding_source=explicit"
        ) in hip_code
        assert "const RuntimeBlock* readBlocks;" in hip_code
        assert (
            "// CrossGL resource metadata: name=writeBlocks kind=glsl_buffer_block "
            "layout=std430 set=0 binding=9 binding_source=explicit"
        ) in hip_code
        assert "RuntimeBlock* writeBlocks;" in hip_code
        assert (
            "// CrossGL resource metadata: name=fixedBlocks kind=glsl_buffer_block "
            "layout=std140 set=0 binding=10 binding_source=explicit count=2"
        ) in hip_code
        assert "FixedBlock fixedBlocks[2];" in hip_code
        assert (
            "// CrossGL resource metadata: name=outputBlocks kind=glsl_buffer_block "
            "layout=std140 access=writeonly set=0 binding=12 "
            "binding_source=explicit count=2"
        ) in hip_code
        assert "return readBlocks[blockIndex].leaves[itemIndex].mass;" in hip_code
        assert "writeBlocks[blockIndex].leaves[itemIndex] = leaf;" in hip_code
        assert "return fixedBlocks[blockIndex].values[itemIndex];" in hip_code
        assert "fixedBlocks[blockIndex].transform = transform;" in hip_code
        assert (
            "/* unsupported HIP GLSL buffer block assignment: resource "
            "'readBlocks' is readonly */ ((void)0);"
        ) in hip_code
        assert (
            "return /* unsupported HIP GLSL buffer block load: resource "
            "'outputBlocks' is writeonly */ make_float3(0.0f, 0.0f, 0.0f);"
        ) in hip_code

    def test_byte_address_buffer_dimensions_emit_hip_length_sidecars(self, tmp_path):
        source_code = """
        shader ByteAddressBufferDimensionsHIP {
            ByteAddressBuffer rawBytes;
            RWByteAddressBuffer rawOutput;
            RWByteAddressBuffer rawArray[2];

            uint countRaw(ByteAddressBuffer input) {
                return buffer_dimensions(input);
            }

            uint countOutput(RWByteAddressBuffer output) {
                uint count;
                output.GetDimensions(count);
                return count;
            }

            void process(uint which) {
                uint globalBytes = buffer_dimensions(rawBytes);
                uint writtenBytes;
                buffer_dimensions(rawOutput, writtenBytes);
                uint arrayBytes;
                rawArray[which].GetDimensions(arrayBytes);
                uint directBytes = rawOutput.GetDimensions();
                uint helperBytes = countRaw(rawBytes);
                uint helperArrayBytes = countOutput(rawArray[which]);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "const unsigned char* rawBytes;" in hip_code
        assert "const uint* rawBytes_length;" in hip_code
        assert "unsigned char* rawOutput;" in hip_code
        assert "const uint* rawOutput_length;" in hip_code
        assert "unsigned char* rawArray[2];" in hip_code
        assert "const uint* rawArray_length[2];" in hip_code
        assert (
            "__device__ unsigned int countRaw(const unsigned char* input, "
            "const uint* input_length)"
        ) in hip_code
        assert (
            "__device__ unsigned int countOutput(unsigned char* output, "
            "const uint* output_length)"
        ) in hip_code
        assert "return input_length[0];" in hip_code
        assert "count = output_length[0];" in hip_code
        assert "unsigned int globalBytes = rawBytes_length[0];" in hip_code
        assert "writtenBytes = rawOutput_length[0];" in hip_code
        assert "arrayBytes = rawArray_length[which][0];" in hip_code
        assert "unsigned int directBytes = rawOutput_length[0];" in hip_code
        assert "unsigned int helperBytes = countRaw(rawBytes, rawBytes_length);" in (
            hip_code
        )
        assert (
            "unsigned int helperArrayBytes = countOutput("
            "rawArray[which], rawArray_length[which]);"
        ) in hip_code
        assert "buffer_dimensions(" not in hip_code
        assert ".GetDimensions(" not in hip_code
        if shutil.which("hipcc") is not None:
            compile_hip_if_hipcc_available(hip_code, tmp_path)

    def test_append_consume_structured_buffers_emit_hip_counter_helpers(self, tmp_path):
        source_code = """
        shader AppendConsumeHIP {
            struct Particle {
                float weight;
            };

            AppendStructuredBuffer<Particle> appendParticles;
            ConsumeStructuredBuffer<Particle> consumeParticles;
            AppendStructuredBuffer<Particle> appendQueues[2];

            void helper(
                AppendStructuredBuffer<Particle> outParticles,
                ConsumeStructuredBuffer<Particle> inParticles,
                Particle p
            ) {
                outParticles.Append(p);
                Particle q = inParticles.Consume();
                buffer_append(outParticles, q);
                Particle r = buffer_consume(inParticles);
            }

            void process(uint which, Particle p) {
                appendParticles.Append(p);
                buffer_append(appendQueues[which], p);
                Particle a = consumeParticles.Consume();
                Particle b = buffer_consume(consumeParticles);
                uint appendLen = buffer_dimensions(appendParticles);
                helper(appendParticles, consumeParticles, b);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "Particle* appendParticles;" in hip_code
        assert "const uint* appendParticles_length;" in hip_code
        assert "uint* appendParticles_counter;" in hip_code
        assert "const Particle* consumeParticles;" in hip_code
        assert "uint* consumeParticles_counter;" in hip_code
        assert "Particle* appendQueues[2];" in hip_code
        assert "uint* appendQueues_counter[2];" in hip_code
        assert (
            "__device__ void helper(Particle* outParticles, "
            "uint* outParticles_counter, const Particle* inParticles, "
            "uint* inParticles_counter, Particle p)"
        ) in hip_code
        assert (
            "template <typename T>\n"
            "__device__ inline void cgl_append_structured_buffer"
            "(T* buffer, uint* counter, const T& value)"
        ) in hip_code
        assert (
            "template <typename T>\n"
            "__device__ inline T cgl_consume_structured_buffer"
            "(const T* buffer, uint* counter)"
        ) in hip_code
        assert "uint index = atomicAdd(counter, 1u);" in hip_code
        assert "uint index = atomicSub(counter, 1u) - 1u;" in hip_code
        assert (
            "cgl_append_structured_buffer("
            "appendParticles, appendParticles_counter, p);"
        ) in hip_code
        assert (
            "cgl_append_structured_buffer("
            "appendQueues[which], appendQueues_counter[which], p);"
        ) in hip_code
        assert (
            "Particle a = cgl_consume_structured_buffer("
            "consumeParticles, consumeParticles_counter);"
        ) in hip_code
        assert (
            "Particle b = cgl_consume_structured_buffer("
            "consumeParticles, consumeParticles_counter);"
        ) in hip_code
        assert "unsigned int appendLen = appendParticles_length[0];" in hip_code
        assert (
            "helper(appendParticles, appendParticles_counter, consumeParticles, "
            "consumeParticles_counter, b);"
        ) in hip_code
        assert "AppendStructuredBuffer<" not in hip_code
        assert "ConsumeStructuredBuffer<" not in hip_code
        assert ".Append(" not in hip_code
        assert ".Consume(" not in hip_code
        assert "buffer_append(" not in hip_code
        assert "buffer_consume(" not in hip_code
        assert "buffer_dimensions(" not in hip_code
        if shutil.which("hipcc") is not None:
            compile_hip_if_hipcc_available(hip_code, tmp_path)

    def test_structured_buffer_element_atomics_emit_hip_pointer_atomics(self, tmp_path):
        source_code = """
        shader StructuredBufferAtomicsHIP {
            struct Particle {
                uint counter;
                float weight;
            };

            StructuredBuffer<uint> readonlyCounts;
            RWStructuredBuffer<uint> counters;
            RWStructuredBuffer<int> signedCounters;
            RWStructuredBuffer<uint2> vectorCounters;
            RWStructuredBuffer<Particle> particles;
            RWStructuredBuffer<uint> counterArrays[2];

            void process(uint index, uint which, uint value, int signedValue) {
                uint oldAdd = atomicAdd(counters[index], value);
                uint oldSub = atomicSub(counters[index], value);
                uint oldMin = atomicMin(counters[index], value);
                uint oldMax = atomicMax(counters[index], value);
                uint oldAnd = atomicAnd(counters[index], value);
                uint oldOr = atomicOr(counters[index], value);
                uint oldXor = atomicXor(counters[index], value);
                uint oldExchange = atomicExchange(counters[index], value);
                uint oldCas = atomicCompareExchange(counters[index], oldAdd, value);
                uint oldComp = atomicCompSwap(
                    counterArrays[which][index],
                    oldMin,
                    value
                );
                uint oldMember = atomicAdd(particles[index].counter, value);
                int oldSigned = atomicAdd(signedCounters[index], signedValue);
                int oldSignedSub = atomicSub(signedCounters[index], signedValue);
                int oldSignedAnd = atomicAnd(signedCounters[index], signedValue);
                int oldSignedOr = atomicOr(signedCounters[index], signedValue);
                int oldSignedXor = atomicXor(signedCounters[index], signedValue);
                int oldSignedCas = atomicCompareExchange(
                    signedCounters[index],
                    oldSigned,
                    signedValue
                );
                uint readOnlyOld = atomicAdd(readonlyCounts[index], value);
                uint2 vectorOld = atomicAdd(vectorCounters[index], uint2(1u, 2u));
                uint badAdd = atomicAdd(counters[index]);
                uint badCas = atomicCompareExchange(counters[index], oldAdd);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "const unsigned int* readonlyCounts;" in hip_code
        assert "unsigned int* counters;" in hip_code
        assert "int* signedCounters;" in hip_code
        assert "uint2* vectorCounters;" in hip_code
        assert "Particle* particles;" in hip_code
        assert "unsigned int* counterArrays[2];" in hip_code
        assert "unsigned int oldAdd = atomicAdd(&counters[index], value);" in hip_code
        assert "unsigned int oldSub = atomicSub(&counters[index], value);" in hip_code
        assert "unsigned int oldMin = atomicMin(&counters[index], value);" in hip_code
        assert "unsigned int oldMax = atomicMax(&counters[index], value);" in hip_code
        assert "unsigned int oldAnd = atomicAnd(&counters[index], value);" in hip_code
        assert "unsigned int oldOr = atomicOr(&counters[index], value);" in hip_code
        assert "unsigned int oldXor = atomicXor(&counters[index], value);" in hip_code
        assert (
            "unsigned int oldExchange = atomicExch(&counters[index], value);"
            in hip_code
        )
        assert (
            "unsigned int oldCas = atomicCAS(&counters[index], oldAdd, value);"
            in hip_code
        )
        assert (
            "unsigned int oldComp = atomicCAS("
            "&counterArrays[which][index], oldMin, value);"
        ) in hip_code
        assert (
            "unsigned int oldMember = atomicAdd(&particles[index].counter, value);"
            in hip_code
        )
        assert (
            "int oldSigned = atomicAdd(&signedCounters[index], signedValue);"
            in hip_code
        )
        assert (
            "int oldSignedSub = atomicSub(&signedCounters[index], signedValue);"
            in hip_code
        )
        assert (
            "int oldSignedAnd = atomicAnd(&signedCounters[index], signedValue);"
            in hip_code
        )
        assert (
            "int oldSignedOr = atomicOr(&signedCounters[index], signedValue);"
            in hip_code
        )
        assert (
            "int oldSignedXor = atomicXor(&signedCounters[index], signedValue);"
            in hip_code
        )
        assert (
            "int oldSignedCas = atomicCAS(&signedCounters[index], oldSigned, "
            "signedValue);" in hip_code
        )
        assert (
            "unsigned int readOnlyOld = /* unsupported HIP structured buffer "
            "atomic: atomicAdd on StructuredBuffer<uint> requires "
            "RWStructuredBuffer target */ 0u;" in hip_code
        )
        assert (
            "uint2 vectorOld = /* unsupported HIP structured buffer atomic: "
            "atomicAdd on RWStructuredBuffer<uint2> requires supported scalar "
            "int/uint/float target */ make_uint2(0u, 0u);" in hip_code
        )
        assert (
            "unsigned int badAdd = /* unsupported HIP structured buffer atomic: "
            "atomicAdd on RWStructuredBuffer<uint> requires 2 argument(s) */ 0u;"
            in hip_code
        )
        assert (
            "unsigned int badCas = /* unsupported HIP structured buffer atomic: "
            "atomicCompareExchange on RWStructuredBuffer<uint> requires "
            "3 argument(s) */ 0u;" in hip_code
        )
        assert "atomicExchange(" not in hip_code
        assert "atomicCompareExchange(" not in hip_code
        assert "atomicCompSwap(" not in hip_code
        if shutil.which("hipcc") is not None:
            compile_hip_if_hipcc_available(hip_code, tmp_path)

    def test_float_structured_buffer_add_and_exchange_atomics_emit_hip_atomics(
        self, tmp_path
    ):
        """Test HIP supports float add/exchange atomics on RWStructuredBuffer."""
        source_code = """
        shader FloatStructuredBufferAtomicsHIP {
            RWStructuredBuffer<float> values;

            void process(uint index, float value) {
                float oldAdd = atomicAdd(values[index], value);
                float oldExchange = atomicExchange(values[index], value);
                float oldMin = atomicMin(values[index], value);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "float* values;" in hip_code
        assert "float oldAdd = atomicAdd(&values[index], value);" in hip_code
        assert "float oldExchange = atomicExch(&values[index], value);" in hip_code
        assert (
            "float oldMin = /* unsupported HIP structured buffer atomic: "
            "atomicMin on RWStructuredBuffer<float> requires supported scalar "
            "int/uint target */ 0.0f;" in hip_code
        )
        assert "float oldAdd = /* unsupported HIP structured buffer atomic" not in (
            hip_code
        )
        assert (
            "float oldExchange = /* unsupported HIP structured buffer atomic"
            not in hip_code
        )
        if shutil.which("hipcc") is not None:
            compile_hip_if_hipcc_available(hip_code, tmp_path)

    def test_uint_structured_buffer_inc_dec_emit_hip_bounded_atomics(self, tmp_path):
        source_code = """
        shader BoundedStructuredBufferAtomicsHIP {
            RWStructuredBuffer<uint> counters;
            RWStructuredBuffer<int> signedCounters;
            RWStructuredBuffer<float> floatCounters;

            void process(uint index, uint limit, int signedLimit, float floatLimit) {
                uint oldInc = atomicInc(counters[index], limit);
                uint oldDec = atomicDec(counters[index], limit);
                int signedInc = atomicInc(signedCounters[index], signedLimit);
                float floatDec = atomicDec(floatCounters[index], floatLimit);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "unsigned int* counters;" in hip_code
        assert "int* signedCounters;" in hip_code
        assert "float* floatCounters;" in hip_code
        assert "unsigned int oldInc = atomicInc(&counters[index], limit);" in hip_code
        assert "unsigned int oldDec = atomicDec(&counters[index], limit);" in hip_code
        assert (
            "int signedInc = /* unsupported HIP structured buffer atomic: "
            "atomicInc on RWStructuredBuffer<int> requires supported scalar "
            "uint target */ 0;" in hip_code
        )
        assert (
            "float floatDec = /* unsupported HIP structured buffer atomic: "
            "atomicDec on RWStructuredBuffer<float> requires supported scalar "
            "uint target */ 0.0f;" in hip_code
        )
        assert "atomicAdd(&counters[index], limit)" not in hip_code
        assert "atomicSub(&counters[index], limit)" not in hip_code
        if shutil.which("hipcc") is not None:
            compile_hip_if_hipcc_available(hip_code, tmp_path)

    def test_byte_address_buffer_atomics_emit_hip_atomic_helpers(self, tmp_path):
        source_code = """
        shader HIPByteAddressBufferAtomics {
            ByteAddressBuffer readBytes;
            RWByteAddressBuffer writeBytes;
            RWByteAddressBuffer byteBuffers[2];

            uint helper(RWByteAddressBuffer output, uint offset, uint value) {
                uint oldValue = output.InterlockedAdd(offset, value);
                uint original;
                output.InterlockedExchange(offset, oldValue, original);
                return original;
            }

            void process(uint offset, uint which, uint value, uint compare) {
                uint oldAdd;
                writeBytes.InterlockedAdd(offset, value, oldAdd);
                uint oldMin = writeBytes.InterlockedMin(offset, value);
                uint oldMax = writeBytes.InterlockedMax(offset, value);
                uint oldAnd = writeBytes.InterlockedAnd(offset, value);
                uint oldOr = writeBytes.InterlockedOr(offset, value);
                uint oldXor = writeBytes.InterlockedXor(offset, value);
                uint oldExchange = writeBytes.InterlockedExchange(offset, value);
                uint oldCas;
                writeBytes.InterlockedCompareExchange(
                    offset,
                    compare,
                    value,
                    oldCas
                );
                uint oldCasExpr = byteBuffers[which].InterlockedCompareExchange(
                    offset,
                    oldCas,
                    value
                );
                uint helperOld = helper(byteBuffers[which], offset, value);
                readBytes.InterlockedAdd(offset, value, oldAdd);
            }

            void rejectMalformed(uint offset, uint value) {
                writeBytes.InterlockedAdd(offset);
                uint badCas = writeBytes.InterlockedCompareExchange(offset, value);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "const unsigned char* readBytes;" in hip_code
        assert "unsigned char* writeBytes;" in hip_code
        assert "unsigned char* byteBuffers[2];" in hip_code
        assert (
            "__device__ unsigned int helper(unsigned char* output, "
            "unsigned int offset, unsigned int value)"
        ) in hip_code
        assert (
            "__device__ inline uint cgl_byte_address_atomic_add_uint"
            "(unsigned char* buffer, uint offset, uint value)"
        ) in hip_code
        assert (
            "return atomicAdd(reinterpret_cast<unsigned int*>(buffer + offset), "
            "value);" in hip_code
        )
        assert "return atomicMin(reinterpret_cast<unsigned int*>(" in hip_code
        assert "return atomicMax(reinterpret_cast<unsigned int*>(" in hip_code
        assert "return atomicAnd(reinterpret_cast<unsigned int*>(" in hip_code
        assert "return atomicOr(reinterpret_cast<unsigned int*>(" in hip_code
        assert "return atomicXor(reinterpret_cast<unsigned int*>(" in hip_code
        assert "return atomicExch(reinterpret_cast<unsigned int*>(" in hip_code
        assert (
            "__device__ inline uint cgl_byte_address_atomic_compare_exchange_uint"
            "(unsigned char* buffer, uint offset, uint compare_value, uint value)"
            in hip_code
        )
        assert (
            "return atomicCAS(reinterpret_cast<unsigned int*>(buffer + offset), "
            "compare_value, value);" in hip_code
        )
        assert (
            "unsigned int oldValue = cgl_byte_address_atomic_add_uint("
            "output, offset, value);"
        ) in hip_code
        assert (
            "original = cgl_byte_address_atomic_exchange_uint("
            "output, offset, oldValue);"
        ) in hip_code
        assert (
            "oldAdd = cgl_byte_address_atomic_add_uint(writeBytes, offset, value);"
            in hip_code
        )
        assert (
            "unsigned int oldMin = cgl_byte_address_atomic_min_uint("
            "writeBytes, offset, value);"
        ) in hip_code
        assert (
            "unsigned int oldMax = cgl_byte_address_atomic_max_uint("
            "writeBytes, offset, value);"
        ) in hip_code
        assert (
            "unsigned int oldAnd = cgl_byte_address_atomic_and_uint("
            "writeBytes, offset, value);"
        ) in hip_code
        assert (
            "unsigned int oldOr = cgl_byte_address_atomic_or_uint("
            "writeBytes, offset, value);"
        ) in hip_code
        assert (
            "unsigned int oldXor = cgl_byte_address_atomic_xor_uint("
            "writeBytes, offset, value);"
        ) in hip_code
        assert (
            "unsigned int oldExchange = cgl_byte_address_atomic_exchange_uint("
            "writeBytes, offset, value);"
        ) in hip_code
        assert (
            "oldCas = cgl_byte_address_atomic_compare_exchange_uint("
            "writeBytes, offset, compare, value);"
        ) in hip_code
        assert (
            "unsigned int oldCasExpr = cgl_byte_address_atomic_compare_exchange_uint("
            "byteBuffers[which], offset, oldCas, value);"
        ) in hip_code
        assert (
            "unsigned int helperOld = helper(byteBuffers[which], offset, value);"
            in hip_code
        )
        assert (
            "/* unsupported HIP byte-address buffer call: "
            "InterlockedAdd on ByteAddressBuffer */ ((void)0);" in hip_code
        )
        assert (
            "/* unsupported HIP byte-address buffer call: "
            "InterlockedAdd on RWByteAddressBuffer */ 0u;" in hip_code
        )
        assert (
            "unsigned int badCas = /* unsupported HIP byte-address buffer call: "
            "InterlockedCompareExchange on RWByteAddressBuffer */ 0u;" in hip_code
        )
        assert "InterlockedAdd(" not in hip_code
        assert "InterlockedCompareExchange(" not in hip_code
        if shutil.which("hipcc") is not None:
            compile_hip_if_hipcc_available(hip_code, tmp_path)

    def test_hlsl_texture_aliases_emit_hip_resources_and_metadata(self):
        source_code = """
        shader Resources {
            Texture2D<float4> colorMap;
            Texture2DArray<float4> layers;
            Texture2DMS<float4> msTex;
            RWTexture2D<uint> counters;
            RWTexture3D<int> signedVolume;

            void queryParam(
                Texture2D<float4> paramTex,
                RWTexture2D<uint> paramCounters,
                vec2 uv,
                ivec2 pixel
            ) {
                ivec2 paramSize = textureSize(paramTex, 0);
                ivec2 counterSize = imageSize(paramCounters);
                vec4 sampled = texture(paramTex, uv);
                uint loaded = imageLoad(paramCounters, pixel);
            }

            compute {
                void main() {
                    vec2 uv;
                    vec3 uvLayer;
                    ivec2 pixel;
                    ivec3 voxel;
                    vec4 sampled = texture(colorMap, uv);
                    vec4 layered = texture(layers, uvLayer);
                    ivec2 texSize = textureSize(colorMap, 0);
                    int samples = textureSamples(msTex);
                    uint loaded = imageLoad(counters, pixel);
                    int signedValue = imageLoad(signedVolume, voxel);
                    imageStore(counters, pixel, loaded);
                    queryParam(colorMap, counters, uv, pixel);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "hipTextureObject_t colorMap;" in hip_code
        assert "CglResourceQueryInfo colorMap_metadata = {};" in hip_code
        assert "hipTextureObject_t layers;" in hip_code
        assert "hipTextureObject_t msTex;" in hip_code
        assert "CglResourceQueryInfo msTex_metadata = {};" in hip_code
        assert "hipSurfaceObject_t counters;" in hip_code
        assert "CglResourceQueryInfo counters_metadata = {};" in hip_code
        assert "hipSurfaceObject_t signedVolume;" in hip_code
        assert (
            "__device__ void queryParam(hipTextureObject_t paramTex, "
            "CglResourceQueryInfo paramTex_metadata, hipSurfaceObject_t "
            "paramCounters, CglResourceQueryInfo paramCounters_metadata, "
            "float2 uv, int2 pixel)"
        ) in hip_code
        assert (
            "int2 paramSize = cgl_textureSize_sampler2D(paramTex_metadata, 0);"
            in hip_code
        )
        assert (
            "int2 counterSize = cgl_imageSize_uimage2D(paramCounters_metadata);"
            in hip_code
        )
        assert "float4 sampled = tex2D<float4>(colorMap, uv.x, uv.y);" in hip_code
        assert (
            "float4 layered = tex2DLayered<float4>"
            "(layers, uvLayer.x, uvLayer.y, uvLayer.z);" in hip_code
        )
        assert (
            "int2 texSize = cgl_textureSize_sampler2D(colorMap_metadata, 0);"
            in hip_code
        )
        assert (
            "int samples = cgl_textureSamples_sampler2DMS(msTex_metadata);" in hip_code
        )
        assert (
            "unsigned int loaded = cgl_surf2Dread<uint>"
            "(counters, pixel.x * sizeof(uint), pixel.y);" in hip_code
        )
        assert (
            "int signedValue = cgl_surf3Dread<int>"
            "(signedVolume, voxel.x * sizeof(int), voxel.y, voxel.z);" in hip_code
        )
        assert (
            "surf2Dwrite(loaded, counters, pixel.x * sizeof(uint), pixel.y);"
            in hip_code
        )
        assert (
            "queryParam(colorMap, colorMap_metadata, counters, counters_metadata, "
            "uv, pixel);"
        ) in hip_code
        assert "Texture2D<" not in hip_code
        assert "RWTexture2D<" not in hip_code
        assert "RWTexture3D<" not in hip_code
        assert "textureSize(" not in hip_code
        assert "textureSamples(" not in hip_code
        assert "imageSize(" not in hip_code

    def test_image_atomic_builtins_emit_hip_diagnostics(self):
        source_code = """
        shader Resources {
            iimage2d signedImage;
            uimage2d counters;
            iimage3d signedVolume;
            uimage2darray counterLayers;
            image2D vectorCounters @rg32ui;
            readonly uimage2D readonlyCounters;
            writeonly iimage2D writeOnlySigned;

            void atomicResources(ivec2 pixel, ivec3 voxel, ivec3 pixelLayer) {
                int added = imageAtomicAdd(signedImage, pixel, 1);
                uint exchanged = imageAtomicExchange(counters, pixel, 2);
                int compared = imageAtomicCompSwap(signedImage, pixel, 3, 4);
                int volumeMin = imageAtomicMin(signedVolume, voxel, 5);
                uint layerMax = imageAtomicMax(counterLayers, pixelLayer, 6);
                uint bitAnd = imageAtomicAnd(counters, pixel, 255);
                uint bitOr = imageAtomicOr(counters, pixel, 8);
                uint bitXor = imageAtomicXor(counters, pixel, 4);
                uint vectorAtomic = imageAtomicAdd(vectorCounters, pixel, 7);
                uint readonlyAtomic = imageAtomicAdd(readonlyCounters, pixel, 1);
                int writeOnlyAtomic = imageAtomicAdd(writeOnlySigned, pixel, 1);
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
            "unsigned int bitAnd = /* unsupported HIP image atomic resource call: "
            "imageAtomicAnd on uimage2D */ 0u;" in hip_code
        )
        assert (
            "unsigned int bitOr = /* unsupported HIP image atomic resource call: "
            "imageAtomicOr on uimage2D */ 0u;" in hip_code
        )
        assert (
            "unsigned int bitXor = /* unsupported HIP image atomic resource call: "
            "imageAtomicXor on uimage2D */ 0u;" in hip_code
        )
        assert (
            "unsigned int vectorAtomic = /* unsupported HIP image atomic resource call: "
            "imageAtomicAdd on image2D */ 0;" in hip_code
        )
        assert (
            "unsigned int readonlyAtomic = "
            "/* unsupported HIP image atomic resource call: imageAtomicAdd "
            "requires readwrite image resource on uimage2D */ 0u;" in hip_code
        )
        assert (
            "int writeOnlyAtomic = /* unsupported HIP image atomic resource call: "
            "imageAtomicAdd requires readwrite image resource on iimage2D */ 0;"
            in hip_code
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
        assert "imageAtomicAnd(" not in hip_code
        assert "imageAtomicOr(" not in hip_code
        assert "imageAtomicXor(" not in hip_code

    def test_image_atomic_coordinate_shapes_emit_hip_diagnostics(self):
        source_code = """
        shader ImageAtomicCoordinateShapes {
            uimage2D counters;
            iimage2D signedImage;
            uimage2DArray layerCounters;
            iimage3D signedVolume;

            void atomicCoordinateShapes(int x, ivec2 pixel, ivec3 voxel) {
                uint validCounter = imageAtomicAdd(counters, pixel, 1);
                uint badCounter = imageAtomicAdd(counters, x, 1);
                int badSigned = imageAtomicMin(signedImage, x, 2);
                uint badLayer = imageAtomicExchange(layerCounters, pixel, 3);
                int badVolume = imageAtomicCompSwap(signedVolume, pixel, 4, 5);
                uint validLayer = imageAtomicMax(layerCounters, voxel, 6);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert (
            "unsigned int validCounter = "
            "/* unsupported HIP image atomic resource call: "
            "imageAtomicAdd on uimage2D */ 0u;" in hip_code
        )
        assert (
            "unsigned int badCounter = "
            "/* unsupported HIP image atomic resource call: "
            "imageAtomicAdd coordinate rank on uimage2D */ 0u;" in hip_code
        )
        assert (
            "int badSigned = /* unsupported HIP image atomic resource call: "
            "imageAtomicMin coordinate rank on iimage2D */ 0;" in hip_code
        )
        assert (
            "unsigned int badLayer = "
            "/* unsupported HIP image atomic resource call: "
            "imageAtomicExchange coordinate rank on uimage2DArray */ 0u;" in hip_code
        )
        assert (
            "int badVolume = /* unsupported HIP image atomic resource call: "
            "imageAtomicCompSwap coordinate rank on iimage3D */ 0;" in hip_code
        )
        assert (
            "unsigned int validLayer = "
            "/* unsupported HIP image atomic resource call: "
            "imageAtomicMax on uimage2DArray */ 0u;" in hip_code
        )
        assert "imageAtomicAdd(" not in hip_code
        assert "imageAtomicMin(" not in hip_code
        assert "imageAtomicExchange(" not in hip_code
        assert "imageAtomicCompSwap(" not in hip_code
        assert "imageAtomicMax(" not in hip_code

    def test_cubemap_image_atomic_builtins_emit_hip_diagnostics(self):
        source_code = """
        shader ImageAtomicCubeShapes {
            iimageCube signedCube;
            uimageCube counterCube;
            iimageCubeArray signedCubeLayers;
            uimageCubeArray counterCubeLayers;

            void atomicCubeShapes(
                int x,
                ivec2 pixel,
                ivec3 cubeCoord,
                ivec4 cubeLayerCoord
            ) {
                int signedAdd = imageAtomicAdd(signedCube, cubeCoord, 1);
                uint counterExchange =
                    imageAtomicExchange(counterCube, cubeCoord, 2);
                int signedLayerComp =
                    imageAtomicCompSwap(signedCubeLayers, cubeLayerCoord, 3, 4);
                uint counterLayerMax =
                    imageAtomicMax(counterCubeLayers, cubeLayerCoord, 5);
                int badSignedCube = imageAtomicMin(signedCube, pixel, 6);
                uint badCounterLayer =
                    imageAtomicOr(counterCubeLayers, cubeCoord, 7);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "hipSurfaceObject_t signedCube;" in hip_code
        assert "hipSurfaceObject_t counterCube;" in hip_code
        assert "hipSurfaceObject_t signedCubeLayers;" in hip_code
        assert "hipSurfaceObject_t counterCubeLayers;" in hip_code
        assert (
            "int signedAdd = /* unsupported HIP image atomic resource call: "
            "imageAtomicAdd on iimageCube */ 0;" in hip_code
        )
        assert (
            "unsigned int counterExchange = "
            "/* unsupported HIP image atomic resource call: "
            "imageAtomicExchange on uimageCube */ 0u;" in hip_code
        )
        assert (
            "int signedLayerComp = "
            "/* unsupported HIP image atomic resource call: "
            "imageAtomicCompSwap on iimageCubeArray */ 0;" in hip_code
        )
        assert (
            "unsigned int counterLayerMax = "
            "/* unsupported HIP image atomic resource call: "
            "imageAtomicMax on uimageCubeArray */ 0u;" in hip_code
        )
        assert (
            "int badSignedCube = /* unsupported HIP image atomic resource call: "
            "imageAtomicMin coordinate rank on iimageCube */ 0;" in hip_code
        )
        assert (
            "unsigned int badCounterLayer = "
            "/* unsupported HIP image atomic resource call: "
            "imageAtomicOr coordinate rank on uimageCubeArray */ 0u;" in hip_code
        )
        for function_name in [
            "imageAtomicAdd",
            "imageAtomicExchange",
            "imageAtomicCompSwap",
            "imageAtomicMax",
            "imageAtomicMin",
            "imageAtomicOr",
        ]:
            assert f"{function_name}(" not in hip_code

    def test_resource_query_builtins_emit_hip_metadata_helpers(self):
        source_code = """
        shader Resources {
            sampler2d colorMap;
            sampler1darray rampLayers;
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
                    ivec2 rampLayerSize = textureSize(rampLayers, 0);
                    ivec3 layerSize = textureSize(layers, 1);
                    ivec3 volumeSize = textureSize(volumeTex, 0);
                    ivec2 cubeSize = textureSize(cubeTex, 0);
                    ivec3 cubeArraySize = textureSize(cubeArrayTex, 0);
                    int texLevels = textureQueryLevels(colorMap);
                    int rampLayerLevels = textureQueryLevels(rampLayers);
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
            "__device__ void queryParam(hipTextureObject_t paramTex, "
            "CglResourceQueryInfo paramTex_metadata)" in hip_code
        )
        assert "CglResourceQueryInfo colorMap_metadata = {};" in hip_code
        assert "CglResourceQueryInfo rampLayers_metadata = {};" in hip_code
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
            "int2 rampLayerSize = cgl_textureSize_sampler1DArray"
            "(rampLayers_metadata, 0);" in hip_code
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
            "int rampLayerLevels = cgl_textureQueryLevels_sampler1DArray"
            "(rampLayers_metadata);" in hip_code
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

    def test_storage_image_query_shapes_emit_hip_metadata_helpers(self):
        source_code = """
        shader Resources {
            image1D lineImage;
            image1DArray lineLayers;
            imageCube cubeImage;
            imageCubeArray cubeLayers;
            iimageCube signedCube;
            iimageCubeArray signedCubeLayers;
            uimageCube counterCube;
            uimageCubeArray counterCubeLayers;
            RWTexture1D<uint> aliasLine;
            RWTextureCube<uint> aliasCube;

            void queryParam(
                imageCube paramCube,
                uimageCubeArray paramCubeCounters
            ) {
                ivec2 paramCubeSize = imageSize(paramCube);
                ivec3 paramCounterSize = imageSize(paramCubeCounters);
            }

            compute {
                void main() {
                    int lineSize = imageSize(lineImage);
                    ivec2 lineLayerSize = imageSize(lineLayers);
                    ivec2 cubeSize = imageSize(cubeImage);
                    ivec3 cubeLayerSize = imageSize(cubeLayers);
                    ivec2 signedCubeSize = imageSize(signedCube);
                    ivec3 signedLayerSize = imageSize(signedCubeLayers);
                    ivec2 counterCubeSize = imageSize(counterCube);
                    ivec3 counterLayerSize = imageSize(counterCubeLayers);
                    int aliasLineSize = imageSize(aliasLine);
                    ivec2 aliasCubeSize = imageSize(aliasCube);
                    queryParam(cubeImage, counterCubeLayers);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "__device__ inline int cgl_imageSize_image1D" in hip_code
        assert "__device__ inline int2 cgl_imageSize_image1DArray" in hip_code
        assert "__device__ inline int2 cgl_imageSize_imageCube" in hip_code
        assert "__device__ inline int3 cgl_imageSize_imageCubeArray" in hip_code
        assert "__device__ inline int2 cgl_imageSize_iimageCube" in hip_code
        assert "__device__ inline int3 cgl_imageSize_iimageCubeArray" in hip_code
        assert "__device__ inline int2 cgl_imageSize_uimageCube" in hip_code
        assert "__device__ inline int3 cgl_imageSize_uimageCubeArray" in hip_code
        assert "CglResourceQueryInfo lineImage_metadata = {};" in hip_code
        assert "CglResourceQueryInfo lineLayers_metadata = {};" in hip_code
        assert "CglResourceQueryInfo cubeImage_metadata = {};" in hip_code
        assert "CglResourceQueryInfo cubeLayers_metadata = {};" in hip_code
        assert "CglResourceQueryInfo signedCube_metadata = {};" in hip_code
        assert "CglResourceQueryInfo signedCubeLayers_metadata = {};" in hip_code
        assert "CglResourceQueryInfo counterCube_metadata = {};" in hip_code
        assert "CglResourceQueryInfo counterCubeLayers_metadata = {};" in hip_code
        assert "CglResourceQueryInfo aliasLine_metadata = {};" in hip_code
        assert "CglResourceQueryInfo aliasCube_metadata = {};" in hip_code
        assert (
            "__device__ void queryParam(hipSurfaceObject_t paramCube, "
            "CglResourceQueryInfo paramCube_metadata, hipSurfaceObject_t "
            "paramCubeCounters, CglResourceQueryInfo paramCubeCounters_metadata)"
        ) in hip_code
        assert "int lineSize = cgl_imageSize_image1D(lineImage_metadata);" in hip_code
        assert (
            "int2 lineLayerSize = cgl_imageSize_image1DArray"
            "(lineLayers_metadata);" in hip_code
        )
        assert (
            "int2 cubeSize = cgl_imageSize_imageCube(cubeImage_metadata);" in hip_code
        )
        assert (
            "int3 cubeLayerSize = cgl_imageSize_imageCubeArray"
            "(cubeLayers_metadata);" in hip_code
        )
        assert (
            "int2 signedCubeSize = cgl_imageSize_iimageCube"
            "(signedCube_metadata);" in hip_code
        )
        assert (
            "int3 signedLayerSize = cgl_imageSize_iimageCubeArray"
            "(signedCubeLayers_metadata);" in hip_code
        )
        assert (
            "int2 counterCubeSize = cgl_imageSize_uimageCube"
            "(counterCube_metadata);" in hip_code
        )
        assert (
            "int3 counterLayerSize = cgl_imageSize_uimageCubeArray"
            "(counterCubeLayers_metadata);" in hip_code
        )
        assert (
            "int aliasLineSize = cgl_imageSize_uimage1D(aliasLine_metadata);"
            in hip_code
        )
        assert (
            "int2 aliasCubeSize = cgl_imageSize_uimageCube(aliasCube_metadata);"
            in hip_code
        )
        assert (
            "queryParam(cubeImage, cubeImage_metadata, counterCubeLayers, "
            "counterCubeLayers_metadata);"
        ) in hip_code
        assert "imageSize(" not in hip_code
        assert "RWTexture1D<" not in hip_code
        assert "RWTextureCube<" not in hip_code

    def test_texture_query_lod_emits_hip_diagnostics(self):
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

    def test_target_invalid_resource_queries_emit_hip_diagnostics(self):
        source_code = """
        shader Resources {
            sampler2d colorMap;
            sampler2dms msTex;
            image2D colorImage;
            image2DMS msImage;
            sampler querySampler;

            compute {
                void main() {
                    ivec2 validTextureSize = textureSize(colorMap, 0);
                    ivec2 validImageSize = imageSize(colorImage);
                    int validTextureSamples = textureSamples(msTex);
                    int validImageSamples = imageSamples(msImage);
                    int validTextureLevels = textureQueryLevels(colorMap);
                    ivec2 badImageSizeFromTexture = imageSize(colorMap);
                    ivec2 badTextureSizeFromImage = textureSize(colorImage, 0);
                    int badTextureSamples = textureSamples(colorMap);
                    int badImageSamples = imageSamples(colorImage);
                    int badLevelsFromImage = textureQueryLevels(colorImage);
                    int badLevelsFromMsTexture = textureQueryLevels(msTex);
                    int badLevelsFromSamplerState = textureQueryLevels(querySampler);
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
            "int2 validTextureSize = cgl_textureSize_sampler2D"
            "(colorMap_metadata, 0);" in hip_code
        )
        assert (
            "int2 validImageSize = cgl_imageSize_image2D"
            "(colorImage_metadata);" in hip_code
        )
        assert (
            "int validTextureSamples = cgl_textureSamples_sampler2DMS"
            "(msTex_metadata);" in hip_code
        )
        assert (
            "int validImageSamples = cgl_imageSamples_image2DMS"
            "(msImage_metadata);" in hip_code
        )
        assert (
            "int validTextureLevels = cgl_textureQueryLevels_sampler2D"
            "(colorMap_metadata);" in hip_code
        )
        assert (
            "int2 badImageSizeFromTexture = /* unsupported HIP resource query: "
            "imageSize on sampler2D */ make_int2(0, 0);" in hip_code
        )
        assert (
            "int2 badTextureSizeFromImage = /* unsupported HIP resource query: "
            "textureSize on image2D */ make_int2(0, 0);" in hip_code
        )
        assert (
            "int badTextureSamples = /* unsupported HIP resource query: "
            "textureSamples on sampler2D */ 0;" in hip_code
        )
        assert (
            "int badImageSamples = /* unsupported HIP resource query: "
            "imageSamples on image2D */ 0;" in hip_code
        )
        assert (
            "int badLevelsFromImage = /* unsupported HIP resource query: "
            "textureQueryLevels on image2D */ 0;" in hip_code
        )
        assert (
            "int badLevelsFromMsTexture = /* unsupported HIP resource query: "
            "textureQueryLevels on sampler2DMS */ 0;" in hip_code
        )
        assert (
            "int badLevelsFromSamplerState = /* unsupported HIP resource query: "
            "textureQueryLevels on sampler */ 0;" in hip_code
        )
        assert "cgl_imageSize_sampler2D" not in hip_code
        assert "cgl_textureSize_image2D" not in hip_code
        assert "textureSize(" not in hip_code
        assert "imageSize(" not in hip_code
        assert "textureSamples(" not in hip_code
        assert "imageSamples(" not in hip_code
        assert "textureQueryLevels(" not in hip_code

    def test_resource_query_arrays_emit_indexed_hip_metadata(self):
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
            "__device__ void queryArrays(hipTextureObject_t paramTextures[3], "
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
            "__device__ void queryArrays(hipTextureObject_t* dynamicTextures, "
            "CglResourceQueryInfo* dynamicTextures_metadata, "
            "hipTextureObject_t fixedTextures[3], "
            "CglResourceQueryInfo fixedTextures_metadata[3], "
            "hipTextureObject_t* dynamicMsTextures, "
            "CglResourceQueryInfo* dynamicMsTextures_metadata, "
            "hipTextureObject_t (*dynamicGrid)[3], "
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
        assert "tex2D<float4>(msTex" not in hip_code
        assert "tex2DLayered<float4>(msLayers" not in hip_code
        assert "cgl_surf2Dread" not in hip_code
        assert "cgl_surf2DLayeredread" not in hip_code
        assert "surf2Dwrite" not in hip_code
        assert "surf2DLayeredwrite" not in hip_code

    def test_control_flow(self):
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

    def test_match_guarded_arm_lowers_to_hip_if_chain(self):
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
        hip_code = codegen.generate(ast)

        assert "switch (mode)" not in hip_code
        assert "if (((mode == 0) &&" in hip_code
        assert "mode > 0" in hip_code
        assert "else {" in hip_code
        assert "value = 1;" in hip_code
        assert "value = 2;" in hip_code
        assert "MatchNode" not in hip_code

    def test_match_identifier_binding_arm_lowers_to_hip_scoped_else_body(self):
        source_code = """
        shader TestShader {
            compute {
                int main(int mode) {
                    int value = 0;
                    match mode {
                        0 => {
                            value = 1;
                        }
                        other => {
                            value = other;
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

        hip_code = HipCodeGen().generate(ast)

        assert "switch (mode)" not in hip_code
        assert "if ((mode == 0))" in hip_code
        assert "else {" in hip_code
        assert "int other = mode;" in hip_code
        assert "value = other;" in hip_code
        assert "MatchNode" not in hip_code

    def test_match_guarded_identifier_binding_falls_through_to_later_hip_arm(self):
        source_code = """
        shader TestShader {
            compute {
                int main(int mode) {
                    int value = 0;
                    match mode {
                        0 => {
                            value = 1;
                        }
                        candidate if candidate > 2 => {
                            value = candidate;
                        }
                        _ => {
                            value = 7;
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

        hip_code = HipCodeGen().generate(ast)

        assert "switch (mode)" not in hip_code
        assert "if ((mode == 0))" in hip_code
        assert "else {" in hip_code
        assert "int candidate = mode;" in hip_code
        assert "candidate > 2" in hip_code
        assert "value = candidate;" in hip_code
        assert "value = 7;" in hip_code
        assert "MatchNode" not in hip_code

    def test_match_plain_struct_pattern_binds_fields_for_hip(self):
        source_code = """
        shader TestShader {
            struct Pair {
                int left;
                int right;
            };

            compute {
                int main(Pair pair) {
                    int value = 0;
                    match pair {
                        Pair { left, right } => {
                            value = left + right;
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

        hip_code = HipCodeGen().generate(ast)

        assert "__global__ void compute_main(Pair pair)" in hip_code
        assert "int left = pair.left;" in hip_code
        assert "int right = pair.right;" in hip_code
        assert "value = (left + right);" in hip_code
        assert "return value;" not in hip_code
        assert "MatchNode" not in hip_code

    def test_match_plain_enum_path_patterns_lower_to_hip_if_chain(self):
        source_code = """
        shader TestShader {
            enum Mode {
                Add,
                Sub
            }

            compute {
                void main(Mode mode, int *out) {
                    match mode {
                        Mode::Add => {
                            out[0] = 1;
                        }
                        Mode::Sub => {
                            out[0] = 2;
                        }
                    }
                }
            }
        }
        """

        hip_code = HipCodeGen().generate(Parser(Lexer(source_code).tokens).parse())

        assert "static const int Mode_Add = 0;" in hip_code
        assert "static const int Mode_Sub = 1;" in hip_code
        assert "__global__ void compute_main(int mode, int* out)" in hip_code
        assert "if ((mode == Mode_Add))" in hip_code
        assert "else if ((mode == Mode_Sub))" in hip_code
        assert "out[0] = 1;" in hip_code
        assert "out[0] = 2;" in hip_code
        assert "enum path patterns are not supported" not in hip_code
        assert "MatchNode" not in hip_code

    def test_match_generic_enum_constructor_pattern_binds_payload_for_hip(self):
        source_code = """
        shader TestShader {
            generic<T> struct Option {
                enum OptionType {
                    Some(T),
                    None
                }
                OptionType variant;
            }

            struct State {
                value: Option<int>;
            }

            compute {
                void main(State state, int *out) {
                    match state.value {
                        Option::Some(value) => {
                            out[0] = value;
                        }
                        Option::None => {
                            out[0] = 0;
                        }
                    }
                }
            }
        }
        """

        hip_code = HipCodeGen().generate(Parser(Lexer(source_code).tokens).parse())

        assert "static const int Option_Some = 0;" in hip_code
        assert "static const int Option_None = 1;" in hip_code
        assert "struct Option_int" in hip_code
        assert "Option_int value;" in hip_code
        assert "if ((state.value.variant == Option_Some))" in hip_code
        assert "int value = state.value.Some_0;" in hip_code
        assert "else if ((state.value.variant == Option_None))" in hip_code
        assert "out[0] = value;" in hip_code
        assert "out[0] = 0;" in hip_code
        assert "enum constructor patterns are not supported" not in hip_code
        assert "MatchNode" not in hip_code

    def test_match_enum_struct_pattern_binds_fields_for_hip(self):
        source_code = """
        shader TestShader {
            enum Command {
                Clear {
                    color: int,
                    depth: int
                },
                Draw {
                    id: int
                }
            }

            compute {
                void main(Command command, int *out) {
                    match command {
                        Command::Clear { color, depth } => {
                            out[0] = color + depth;
                        }
                        Command::Draw { id } => {
                            out[0] = id;
                        }
                    }
                }
            }
        }
        """

        hip_code = HipCodeGen().generate(Parser(Lexer(source_code).tokens).parse())

        assert "static const int Command_Clear = 0;" in hip_code
        assert "static const int Command_Draw = 1;" in hip_code
        assert "struct Command" in hip_code
        assert "if ((command.variant == Command_Clear))" in hip_code
        assert "int color = command.color;" in hip_code
        assert "int depth = command.depth;" in hip_code
        assert "else if ((command.variant == Command_Draw))" in hip_code
        assert "int id = command.id;" in hip_code
        assert "out[0] = (color + depth);" in hip_code
        assert "out[0] = id;" in hip_code
        assert "enum struct patterns are not supported" not in hip_code
        assert "MatchNode" not in hip_code

    def test_match_expression_initializes_hip_local_with_bindings(self):
        source_code = """
        shader TestShader {
            compute {
                int main(int mode) {
                    int value = match mode {
                        0 => 10,
                        candidate if candidate > 1 => candidate,
                        _ => -1,
                    };
                    return value;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "int value;" in hip_code
        assert "value = 10;" in hip_code
        assert "int candidate = mode;" in hip_code
        assert "candidate > 1" in hip_code
        assert "value = candidate;" in hip_code
        assert "value = -1;" in hip_code
        assert "MatchNode" not in hip_code

    def test_return_match_expression_in_hip_kernel_discards_return_value(self):
        source_code = """
        shader TestShader {
            compute {
                int main(int mode) {
                    return match mode {
                        0 => 10,
                        _ => -1,
                    };
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "__global__ void compute_main(int mode)" in hip_code
        assert "int __crossgl_match_value_0;" not in hip_code
        assert "__crossgl_match_value_0 = 10;" not in hip_code
        assert "__crossgl_match_value_0 = -1;" not in hip_code
        assert "return __crossgl_match_value_0;" not in hip_code
        assert "return;" in hip_code
        assert "MatchNode" not in hip_code

    def test_direct_ast_expression_statements_are_emitted(self):
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
        codegen = HipCodeGen()

        assert codegen.visit(LiteralNode(True, PrimitiveType("bool"))) == "true"
        assert (
            codegen.visit(LiteralNode('debug"name', PrimitiveType("string")))
            == '"debug\\"name"'
        )
        assert codegen.visit(LiteralNode("'", PrimitiveType("char"))) == "'\\''"

    def test_array_operations(self):
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

        assert "#include <hip/hip_runtime.h>" in hip_code

    def test_array_declarations_emit_c_style_declarators(self):
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
            "void compute_main(float input_weights[4], float3 input_colors[2], "
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
        source_code = ""

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "#include <hip/hip_runtime.h>" in hip_code
        assert "#include <hip/hip_runtime_api.h>" in hip_code

    def test_prefix_and_postfix_unary_operators_preserve_position(self):
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

        assert "#include <hip/hip_runtime.h>" in hip_code
        assert "#include <hip/hip_runtime_api.h>" in hip_code

    def test_barrier_emits_syncthreads(self):
        source_code = """
        shader BarrierShader {
            compute {
                void main() {
                    barrier();
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "__syncthreads();" in hip_code
        assert "barrier()" not in hip_code

    def test_memory_barrier_emits_threadfence(self):
        source_code = """
        shader MemoryBarrierShader {
            compute {
                void main() {
                    memoryBarrier();
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "__threadfence();" in hip_code
        assert "memoryBarrier()" not in hip_code

    def test_workgroup_barrier_emits_syncthreads(self):
        source_code = """
        shader WorkgroupBarrierShader {
            compute {
                void main() {
                    workgroupBarrier();
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "__syncthreads();" in hip_code
        assert "workgroupBarrier()" not in hip_code

    def test_hip_memory_barrier_variants_emit_sync_intrinsics(self):
        source_code = """
        shader HipMemoryBarrierVariants {
            compute {
                void main() {
                    groupMemoryBarrier();
                    memoryBarrierShared();
                    memoryBarrierBuffer();
                    memoryBarrierImage();
                    allMemoryBarrier();
                    deviceMemoryBarrier();
                }
            }
        }
        """

        ast = Parser(Lexer(source_code).tokens).parse()
        hip_code = HipCodeGen().generate(ast)

        assert hip_code.count("__threadfence_block();") == 2
        assert hip_code.count("__threadfence();") == 4
        assert "groupMemoryBarrier();" not in hip_code
        assert "memoryBarrierShared();" not in hip_code
        assert "memoryBarrierBuffer();" not in hip_code
        assert "memoryBarrierImage();" not in hip_code
        assert "allMemoryBarrier();" not in hip_code
        assert "deviceMemoryBarrier();" not in hip_code

    def test_hip_user_defined_synchronization_names_are_not_lowered(self):
        source_code = """
        shader HipSynchronizationShadowing {
            compute {
                void barrier() {
                    return;
                }

                void memoryBarrier() {
                    return;
                }

                void workgroupBarrier() {
                    return;
                }

                void groupMemoryBarrier() {
                    return;
                }

                void memoryBarrierShared() {
                    return;
                }

                void memoryBarrierBuffer() {
                    return;
                }

                void memoryBarrierImage() {
                    return;
                }

                void allMemoryBarrier() {
                    return;
                }

                void deviceMemoryBarrier() {
                    return;
                }

                void main() {
                    barrier();
                    memoryBarrier();
                    workgroupBarrier();
                    groupMemoryBarrier();
                    memoryBarrierShared();
                    memoryBarrierBuffer();
                    memoryBarrierImage();
                    allMemoryBarrier();
                    deviceMemoryBarrier();
                }
            }
        }
        """

        ast = Parser(Lexer(source_code).tokens).parse()
        hip_code = HipCodeGen().generate(ast)

        assert "void barrier()" in hip_code
        assert "void memoryBarrier()" in hip_code
        assert "void workgroupBarrier()" in hip_code
        assert "void groupMemoryBarrier()" in hip_code
        assert "void memoryBarrierShared()" in hip_code
        assert "void memoryBarrierBuffer()" in hip_code
        assert "void memoryBarrierImage()" in hip_code
        assert "void allMemoryBarrier()" in hip_code
        assert "void deviceMemoryBarrier()" in hip_code
        assert "barrier();" in hip_code
        assert "memoryBarrier();" in hip_code
        assert "workgroupBarrier();" in hip_code
        assert "groupMemoryBarrier();" in hip_code
        assert "memoryBarrierShared();" in hip_code
        assert "memoryBarrierBuffer();" in hip_code
        assert "memoryBarrierImage();" in hip_code
        assert "allMemoryBarrier();" in hip_code
        assert "deviceMemoryBarrier();" in hip_code
        assert "__syncthreads();" not in hip_code
        assert "__threadfence();" not in hip_code
        assert "__threadfence_block();" not in hip_code

    @pytest.mark.parametrize(
        "builtin",
        [
            "barrier",
            "groupMemoryBarrier",
            "memoryBarrier",
            "memoryBarrierShared",
            "memoryBarrierBuffer",
            "memoryBarrierImage",
            "allMemoryBarrier",
            "deviceMemoryBarrier",
            "workgroupBarrier",
        ],
    )
    def test_hip_synchronization_builtins_reject_arguments(self, builtin):
        source_code = f"""
        shader BadHipSynchronizationBuiltinArgs {{
            compute {{
                void main() {{
                    {builtin}(1);
                }}
            }}
        }}
        """

        ast = Parser(Lexer(source_code).tokens).parse()

        with pytest.raises(
            ValueError,
            match=rf"HIP synchronization builtin '{builtin}' requires 0 argument",
        ):
            HipCodeGen().generate(ast)

    def test_wave_intrinsics_lower_to_hip_subgroup_helpers(self):
        source_code = """
        shader HipWaveShader {
            compute {
                void main() {
                    uint lane = WaveGetLaneIndex();
                    uint count = WaveGetLaneCount();
                    bool first = WaveIsFirstLane();
                    uint sumValue = WaveActiveSum(lane);
                    bool anyLane = WaveActiveAnyTrue(sumValue > 0u);
                    uvec4 ballot = WaveActiveBallot(anyLane);
                    uint broadcast = WaveReadLaneAt(sumValue, 0u);
                }
            }
        }
        """

        hip_code = HipCodeGen().generate(Parser(Lexer(source_code).tokens).parse())

        assert "unsigned int lane = (threadIdx.x & (warpSize - 1));" in hip_code
        assert "unsigned int count = warpSize;" in hip_code
        assert "bool first = ((threadIdx.x & (warpSize - 1)) == 0);" in hip_code
        assert (
            "__device__ inline unsigned int "
            "cgl_hip_wave_active_sum_uint_uint(unsigned int value)" in hip_code
        )
        assert (
            "unsigned int sumValue = cgl_hip_wave_active_sum_uint_uint(lane);"
            in hip_code
        )
        assert (
            "bool anyLane = "
            "cgl_hip_wave_active_any_true_bool_bool((sumValue > 0u));" in hip_code
        )
        assert (
            "uint4 ballot = cgl_hip_wave_active_ballot_uint4_bool(anyLane);" in hip_code
        )
        assert (
            "unsigned int broadcast = "
            "cgl_hip_wave_read_lane_at_uint_uint_uint(sumValue, 0u);" in hip_code
        )
        assert "WaveOpNode(" not in hip_code
        assert "Unsupported HIP wave intrinsic" not in hip_code

    def test_direct_shfl_down_sync_lowers_to_guarded_hip_helper(self):
        source_code = """
        shader HipDirectShuffle {
            compute {
                void main() {
                    float partial = 1.0;
                    float reduced = __shfl_down_sync(0xFFFFFFFF, partial, 16);
                }
            }
        }
        """

        hip_code = HipCodeGen().generate(Parser(Lexer(source_code).tokens).parse())

        assert (
            "__device__ inline float "
            "cgl_hip_shfl_down_sync_float(unsigned long long mask, float value, "
            "int offset)" in hip_code
        )
        assert (
            "#if defined(__HIP_DEVICE_COMPILE__) && "
            "defined(HIP_ENABLE_WARP_SYNC_BUILTINS)" in hip_code
        )
        assert (
            "float reduced = cgl_hip_shfl_down_sync_float"
            "(4294967295, partial, 16);" in hip_code
        )

    def test_wave_match_count_quad_and_multi_prefix_lower_to_hip_helpers(self):
        source_code = """
        shader HipWaveExtended {
            compute {
                void main() {
                    uint lane = WaveGetLaneIndex();
                    bool equal = WaveActiveAllEqual(lane);
                    uint bitCount = WaveActiveCountBits(equal);
                    uint prefixCount = WavePrefixCountBits(equal);
                    uvec4 mask = WaveMatch(lane);
                    uint quadLane = QuadReadLaneAt(lane, 1u);
                    uint quadX = QuadReadAcrossX(lane);
                    uint quadY = QuadReadAcrossY(lane);
                    uint quadDiagonal = QuadReadAcrossDiagonal(lane);
                    uint sum = WaveMultiPrefixSum(lane, mask);
                    uint product = WaveMultiPrefixProduct(lane + 1u, mask);
                    uint count = WaveMultiPrefixCountBits(equal, mask);
                    uint bitAnd = WaveMultiPrefixBitAnd(product, mask);
                    uint bitOr = WaveMultiPrefixBitOr(bitAnd, mask);
                    uint bitXor = WaveMultiPrefixBitXor(bitOr, mask);
                }
            }
        }
        """

        hip_code = HipCodeGen().generate(Parser(Lexer(source_code).tokens).parse())

        assert "cgl_hip_wave_active_all_equal_bool_uint(lane)" in hip_code
        assert "cgl_hip_wave_active_count_bits_uint_bool(equal)" in hip_code
        assert "cgl_hip_wave_prefix_count_bits_uint_bool(equal)" in hip_code
        assert "cgl_hip_wave_match_uint4_uint(lane)" in hip_code
        assert "cgl_hip_quad_read_lane_at_uint_uint_uint(lane, 1u)" in hip_code
        assert "cgl_hip_quad_read_across_x_uint_uint(lane)" in hip_code
        assert "cgl_hip_quad_read_across_y_uint_uint(lane)" in hip_code
        assert "cgl_hip_quad_read_across_diagonal_uint_uint(lane)" in hip_code
        assert "cgl_hip_wave_multi_prefix_sum_uint_uint_uint4(lane, mask)" in hip_code
        assert (
            "cgl_hip_wave_multi_prefix_product_uint_uint_uint4((lane + 1u), mask)"
            in hip_code
        )
        assert (
            "cgl_hip_wave_multi_prefix_count_bits_uint_bool_uint4(equal, mask)"
            in hip_code
        )
        assert "cgl_hip_wave_multi_prefix_bit_and_uint_uint_uint4(product, mask)" in (
            hip_code
        )
        assert "cgl_hip_wave_multi_prefix_bit_or_uint_uint_uint4(bitAnd, mask)" in (
            hip_code
        )
        assert "cgl_hip_wave_multi_prefix_bit_xor_uint_uint_uint4(bitOr, mask)" in (
            hip_code
        )
        assert "Unsupported HIP wave intrinsic" not in hip_code

    def test_direct_wave_ir_node_lowers_to_hip_helper(self):
        codegen = HipCodeGen()

        generated_expr = codegen.visit(
            WaveOpNode("WaveActiveSum", [LiteralNode(1, PrimitiveType("uint"))])
        )

        assert generated_expr == "cgl_hip_wave_active_sum_uint_uint(1u)"
        assert "cgl_hip_wave_active_sum_uint_uint" in codegen.helper_functions

    def test_wave_intrinsics_emit_hip_diagnostics_for_invalid_argument_shapes(self):
        source_code = """
        shader HipWaveDiagnostics {
            compute {
                void main(
                    uint value,
                    uint predicateValue,
                    vec2 laneVector,
                    uvec2 badMask
                ) {
                    uint badLane = WaveReadLaneAt(value, laneVector);
                    uint badMaskResult = WaveMultiPrefixSum(value, badMask);
                    uvec4 badPredicate = WaveActiveBallot(predicateValue);
                }
            }
        }
        """

        hip_code = HipCodeGen().generate(Parser(Lexer(source_code).tokens).parse())

        assert (
            "unsigned int badLane = /* unsupported HIP wave intrinsic: "
            "WaveReadLaneAt lane index must be scalar int or uint, got float2 */ "
            "0u;"
        ) in hip_code
        assert (
            "unsigned int badMaskResult = /* unsupported HIP wave intrinsic: "
            "WaveMultiPrefixSum partition mask must be uvec4, got uint2 */ 0u;"
        ) in hip_code
        assert (
            "uint4 badPredicate = /* unsupported HIP wave intrinsic: "
            "WaveActiveBallot predicate must be bool, got unsigned int */ "
            "make_uint4(0u, 0u, 0u, 0u);"
        ) in hip_code
        assert "WaveReadLaneAt(value, laneVector)" not in hip_code
        assert "WaveMultiPrefixSum(value, badMask)" not in hip_code
        assert "WaveActiveBallot(predicateValue)" not in hip_code

    def test_wave_intrinsics_reject_hip_wrong_arity(self):
        source_code = """
        shader BadHipWaveArity {
            compute {
                uint main(uint value) {
                    return WaveReadLaneAt(value);
                }
            }
        }
        """

        ast = Parser(Lexer(source_code).tokens).parse()
        with pytest.raises(
            ValueError,
            match="HIP wave intrinsic WaveReadLaneAt requires 2 arguments, got 1",
        ):
            HipCodeGen().generate(ast)

    def test_wave_intrinsic_helpers_compile_with_hipcc_if_available(self, tmp_path):
        """Smoke compile generated HIP wave helpers when hipcc is available."""
        source_code = """
        uint waveHelpers(uint lane, bool active, uvec4 mask) {
            uint sumValue = WaveActiveSum(lane);
            bool anyLane = WaveActiveAnyTrue(active);
            uvec4 matched = WaveMatch(sumValue);
            uint multi = WaveMultiPrefixSum(sumValue, mask);
            uint quad = QuadReadLaneAt(multi, 1u);
            return WaveReadLaneAt(quad, 0u);
        }
        """

        hip_code = HipCodeGen().generate(Parser(Lexer(source_code).tokens).parse())

        assert "cgl_hip_wave_active_sum_uint_uint(lane)" in hip_code
        assert "cgl_hip_wave_match_uint4_uint(sumValue)" in hip_code
        assert "cgl_hip_wave_multi_prefix_sum_uint_uint_uint4(sumValue, mask)" in (
            hip_code
        )
        assert "cgl_hip_quad_read_lane_at_uint_uint_uint(multi, 1u)" in hip_code
        assert "cgl_hip_wave_read_lane_at_uint_uint_uint(quad, 0u)" in hip_code
        compile_hip_if_hipcc_available(hip_code, tmp_path)

    def test_cbuffer_members_lowered_to_constant_memory(self):
        source_code = """
        shader CbufferShader {
            cbuffer Params {
                float4x4 mvp;
                vec3 cameraPosition;
                float time;
                int frameCount;
            };
            compute {
                void main() {
                    float t = time;
                    int f = frameCount;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "// Constant buffer: Params" in hip_code
        assert "__constant__ float4x4 mvp;" in hip_code
        assert "__constant__ float3 cameraPosition;" in hip_code
        assert "__constant__ float time;" in hip_code
        assert "__constant__ int frameCount;" in hip_code
        assert "float t = time;" in hip_code
        assert "int f = frameCount;" in hip_code
        assert "struct Params" not in hip_code
        assert (
            "// CrossGL resource metadata: name=Params kind=cbuffer set=0 "
            "binding=0 binding_source=automatic"
        ) in hip_code
        assert "cbuffer Params" not in hip_code

    def test_uniform_and_multiple_cbuffers_lower_to_constant_memory(self):
        source_code = """
        shader MultiCbufferShader {
            uniform PerFrame {
                float deltaTime;
                vec4 colors[2];
            };

            cbuffer PerObject {
                vec3 objectPosition;
                float scale;
            };

            compute {
                void main() {
                    float dt = deltaTime;
                    float s = scale;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "// Constant buffer: PerFrame" in hip_code
        assert "// Constant buffer: PerObject" in hip_code
        assert "__constant__ float deltaTime;" in hip_code
        assert "__constant__ float4 colors[2];" in hip_code
        assert "__constant__ float3 objectPosition;" in hip_code
        assert "__constant__ float scale;" in hip_code
        assert (
            "// CrossGL resource metadata: name=PerFrame kind=cbuffer set=0 "
            "binding=0 binding_source=automatic"
        ) in hip_code
        assert (
            "// CrossGL resource metadata: name=PerObject kind=cbuffer set=0 "
            "binding=1 binding_source=automatic"
        ) in hip_code
        assert "struct PerFrame" not in hip_code
        assert "struct PerObject" not in hip_code
        assert "uniform PerFrame" not in hip_code
        assert "cbuffer PerObject" not in hip_code

    def test_cbuffer_struct_nodes_lower_to_constant_memory(self):
        cbuffer = StructNode(
            "Params",
            [
                VariableNode("time", PrimitiveType("float")),
            ],
        )
        cbuffer.is_cbuffer = True
        ast = ShaderNode(
            name="DirectCbufferStruct",
            execution_model=ExecutionModel.COMPUTE_KERNEL,
            cbuffers=[cbuffer],
        )

        hip_code = HipCodeGen().generate(ast)

        assert "// Constant buffer: Params" in hip_code
        assert "float time;" in hip_code
        assert "__constant__ float time;" in hip_code
        assert "struct Params" not in hip_code
        assert (
            "// CrossGL resource metadata: name=Params kind=cbuffer set=0 "
            "binding=0 binding_source=automatic"
        ) in hip_code

    def test_structured_buffer_maps_to_device_pointer(self):
        source_code = """
        shader StructuredBufferPointerShader {
            StructuredBuffer<float> inputData;

            void readBuffer(StructuredBuffer<int> src, uint idx) {
                int val = src[idx];
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "const float* inputData;" in hip_code
        assert "const int* src" in hip_code
        assert "StructuredBuffer" not in hip_code

    def test_rw_structured_buffer_maps_to_device_pointer_readwrite(self):
        source_code = """
        shader RWStructuredBufferShader {
            RWStructuredBuffer<int> outputData;

            void writeBuffer(RWStructuredBuffer<float> dst, uint idx, float value) {
                dst[idx] = value;
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "int* outputData;" in hip_code
        assert "float* dst" in hip_code
        assert "dst[idx] = value;" in hip_code
        assert "RWStructuredBuffer" not in hip_code

    def test_texture_sampling_tex3d_and_combined_sampler2d(self):
        source_code = """
        shader TextureSamplingShader {
            sampler3d volumeTex;
            sampler2D colorTex;

            void sampleTextures(
                sampler3d paramVolume,
                sampler2D paramColor,
                vec3 uvw,
                vec2 uv
            ) {
                vec4 volume = texture(paramVolume, uvw);
                vec4 volumeLod = textureLod(paramVolume, uvw, 2.0);
                vec4 color = texture(paramColor, uv);
                vec4 colorLod = textureLod(paramColor, uv, 1.0);
                vec4 direct = tex2D(paramColor, uv);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "hipTextureObject_t volumeTex;" in hip_code
        assert "hipTextureObject_t colorTex;" in hip_code
        assert (
            "float4 volume = tex3D<float4>(paramVolume, uvw.x, uvw.y, uvw.z);"
            in hip_code
        )
        assert (
            "float4 volumeLod = tex3DLod<float4>(paramVolume, uvw.x, uvw.y, uvw.z, 2.0);"
            in hip_code
        )
        assert "float4 color = tex2D<float4>(paramColor, uv.x, uv.y);" in hip_code
        assert (
            "float4 colorLod = tex2DLod<float4>(paramColor, uv.x, uv.y, 1.0);"
            in hip_code
        )
        assert "float4 direct = tex2D<float4>(paramColor, uv.x, uv.y);" in hip_code
        assert "sampler3d" not in hip_code
        assert "sampler2D" not in hip_code
        assert " = texture(" not in hip_code
        assert " = textureLod(" not in hip_code

    def test_struct_members_with_semantics_produce_hip_struct_fields(self):
        source_code = """
        struct VertexInput {
            vec3 position;
            vec3 normal;
            vec2 texCoord;
        };

        struct VertexOutput {
            vec4 clipPos;
            vec3 worldNormal;
            vec2 uv;
        };

        shader StructSemanticShader {
            compute {
                void main() {
                    VertexInput input;
                    input.position = vec3(1.0, 2.0, 3.0);
                    input.normal = vec3(0.0, 1.0, 0.0);
                    input.texCoord = vec2(0.5, 0.5);
                    VertexOutput output;
                    output.clipPos = vec4(input.position, 1.0);
                    output.worldNormal = input.normal;
                    output.uv = input.texCoord;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "struct VertexInput" in hip_code
        assert "float3 position;" in hip_code
        assert "float3 normal;" in hip_code
        assert "float2 texCoord;" in hip_code
        assert "struct VertexOutput" in hip_code
        assert "float4 clipPos;" in hip_code
        assert "float3 worldNormal;" in hip_code
        assert "float2 uv;" in hip_code
        assert "input.position = make_float3(1.0, 2.0, 3.0);" in hip_code
        assert "output.worldNormal = input.normal;" in hip_code
        assert "output.uv = input.texCoord;" in hip_code
        assert "VertexInput input;" in hip_code
        assert "VertexOutput output;" in hip_code

    def test_struct_semantics_validate_builtin_types_and_stage_context_for_hip(self):
        valid_code = """
        shader HipStructSemantics {
            struct FSOutput {
                vec4 color @ SV_Target1;
                float depth @ gl_FragDepth;
                vec2 uv @ TEXCOORD0;
            };

            fragment {
                FSOutput main() {
                    FSOutput output;
                    return output;
                }
            }
        }
        """
        hip_code = HipCodeGen().generate(Parser(Lexer(valid_code).tokens).parse())

        assert "float4 color; // CrossGL semantic: target(1)" in hip_code
        assert "float depth; // CrossGL semantic: depth(any)" in hip_code
        assert "float2 uv; // CrossGL semantic: texcoord(0)" in hip_code

        invalid_type = """
        shader BadHipStructSemanticType {
            struct FSOutput {
                vec3 color @ gl_FragColor;
            };
        }
        """
        with pytest.raises(ValueError, match="gl_FragColor.*vec4-compatible"):
            HipCodeGen().generate(Parser(Lexer(invalid_type).tokens).parse())

        invalid_stage = """
        shader BadHipStructSemanticStage {
            struct FSOutput {
                vec4 color @ gl_FragColor;
            };

            vertex {
                FSOutput main() {
                    FSOutput output;
                    return output;
                }
            }
        }
        """
        with pytest.raises(ValueError, match="gl_FragColor.*vertex stage"):
            HipCodeGen().generate(Parser(Lexer(invalid_stage).tokens).parse())

        invalid_input_only = """
        shader BadHipStructInputOnlySemantic {
            struct VSOutput {
                uint vertexId @ gl_VertexID;
            };

            vertex {
                VSOutput main() {
                    VSOutput output;
                    return output;
                }
            }
        }
        """
        with pytest.raises(ValueError, match="gl_VertexID.*input-only"):
            HipCodeGen().generate(Parser(Lexer(invalid_input_only).tokens).parse())

    def test_return_semantics_validate_builtin_types_and_stage_context_for_hip(self):
        invalid_depth_type = """
        shader BadHipDepthType {
            fragment {
                vec4 main() @ gl_FragDepth {
                    return vec4(0.0);
                }
            }
        }
        """
        with pytest.raises(ValueError, match="gl_FragDepth.*float type"):
            HipCodeGen().generate(Parser(Lexer(invalid_depth_type).tokens).parse())

        invalid_stage = """
        shader BadHipStage {
            vertex {
                vec4 main() @ gl_FragColor {
                    return vec4(0.0);
                }
            }
        }
        """
        with pytest.raises(ValueError, match="gl_FragColor.*vertex stage"):
            HipCodeGen().generate(Parser(Lexer(invalid_stage).tokens).parse())

        invalid_input_only = """
        shader BadHipInputOnlyReturn {
            vertex {
                uint main() @ gl_VertexID {
                    return uint(0);
                }
            }
        }
        """
        with pytest.raises(ValueError, match="gl_VertexID.*input-only"):
            HipCodeGen().generate(Parser(Lexer(invalid_input_only).tokens).parse())

    def test_direct_return_semantics_emit_metadata_for_hip(self, tmp_path):
        color_code = """
        shader HipDirectReturnSemantics {
            vertex {
                vec4 vertexMain() @ gl_Position {
                    return vec4(0.0, 0.0, 0.0, 1.0);
                }
            }

            fragment {
                vec4 fragmentMain() @ gl_FragColor {
                    return vec4(1.0, 0.5, 0.25, 1.0);
                }
            }
        }
        """
        hip_code = HipCodeGen().generate(Parser(Lexer(color_code).tokens).parse())

        assert "// CrossGL return semantic: position" in hip_code
        assert "__device__ float4 vertexMain()" in hip_code
        assert "// CrossGL return semantic: target(0)" in hip_code
        assert "__device__ float4 fragmentMain()" in hip_code

        depth_code = """
        shader HipDepthReturnSemantics {
            fragment {
                float depthMain() @ gl_FragDepth {
                    return 0.5;
                }
            }
        }
        """
        depth_hip_code = HipCodeGen().generate(Parser(Lexer(depth_code).tokens).parse())

        assert "// CrossGL return semantic: depth(any)" in depth_hip_code
        assert "__device__ float depthMain()" in depth_hip_code

        if shutil.which("hipcc") is not None:
            compile_hip_if_hipcc_available(hip_code, tmp_path)
            compile_hip_if_hipcc_available(depth_hip_code, tmp_path)

    def test_stage_parameter_semantics_emit_metadata_for_hip(self, tmp_path):
        vertex_code = """
        shader HipVertexStageParameterSemantics {
            vertex {
                vec4 main(
                    uint vertexId @ gl_VertexID,
                    uint instanceId @ SV_InstanceID
                ) @ gl_Position {
                    return vec4(0.0, 0.0, 0.0, 1.0);
                }
            }
        }
        """
        vertex_hip = HipCodeGen().generate(Parser(Lexer(vertex_code).tokens).parse())

        assert "// CrossGL parameter semantic: vertexId: vertex_id" in vertex_hip
        assert "// CrossGL parameter semantic: instanceId: instance_id" in vertex_hip
        assert "__device__ float4 main(unsigned int vertexId" in vertex_hip

        fragment_code = """
        shader HipFragmentStageParameterSemantics {

            fragment {
                vec4 main(
                    vec4 fragCoord @ gl_FragCoord,
                    bool front @ gl_FrontFacing
                ) @ gl_FragColor {
                    return vec4(1.0, 0.0, 0.0, 1.0);
                }
            }
        }
        """
        fragment_hip = HipCodeGen().generate(
            Parser(Lexer(fragment_code).tokens).parse()
        )

        assert "// CrossGL parameter semantic: fragCoord: position" in fragment_hip
        assert "// CrossGL parameter semantic: front: front_facing" in fragment_hip
        assert "__device__ float4 main(float4 fragCoord, bool front)" in fragment_hip

        compute_code = """
        shader HipComputeStageParameterSemantics {

            compute {
                void main(
                    uvec3 globalId @ gl_GlobalInvocationID,
                    uvec3 localId @ SV_GroupThreadID,
                    uint lane @ gl_LocalInvocationIndex
                ) {
                    uint gx = globalId.x;
                    uint lx = localId.x;
                    uint copy = lane;
                }
            }
        }
        """
        compute_hip = HipCodeGen().generate(Parser(Lexer(compute_code).tokens).parse())

        assert (
            "// CrossGL parameter semantic: globalId: global_invocation_id"
            in compute_hip
        )
        assert (
            "// CrossGL parameter semantic: localId: local_invocation_id" in compute_hip
        )
        assert (
            "// CrossGL parameter semantic: lane: local_invocation_index" in compute_hip
        )
        assert "__global__ void compute_main()" in compute_hip
        assert "uint3 globalId" not in compute_hip
        assert "uint3 localId" not in compute_hip
        assert "unsigned int lane" not in compute_hip
        assert (
            "unsigned int gx = (blockIdx.x * blockDim.x + threadIdx.x);" in compute_hip
        )
        assert "unsigned int lx = threadIdx.x;" in compute_hip
        assert (
            "unsigned int copy = (threadIdx.z * blockDim.y * blockDim.x + "
            "threadIdx.y * blockDim.x + threadIdx.x);"
        ) in compute_hip
        if shutil.which("hipcc") is not None:
            compile_hip_if_hipcc_available(vertex_hip, tmp_path)
            compile_hip_if_hipcc_available(fragment_hip, tmp_path)
            compile_hip_if_hipcc_available(compute_hip, tmp_path)

    def test_stage_parameter_semantics_validate_builtin_stage_type_for_hip(self):
        invalid_stage = """
        shader BadHipStageParameterStage {
            fragment {
                void main(uint vertexId @ gl_VertexID) {
                }
            }
        }
        """
        with pytest.raises(ValueError, match="gl_VertexID.*fragment stage"):
            HipCodeGen().generate(Parser(Lexer(invalid_stage).tokens).parse())

        invalid_type = """
        shader BadHipStageParameterType {
            fragment {
                void main(vec3 fragCoord @ gl_FragCoord) {
                }
            }
        }
        """
        with pytest.raises(ValueError, match="gl_FragCoord.*expected float4 type"):
            HipCodeGen().generate(Parser(Lexer(invalid_type).tokens).parse())

        output_only = """
        shader BadHipStageParameterOutputOnly {
            vertex {
                void main(vec4 color @ SV_Target0) {
                }
            }
        }
        """
        with pytest.raises(ValueError, match="SV_Target0.*output-only"):
            HipCodeGen().generate(Parser(Lexer(output_only).tokens).parse())

        duplicate_system_value = """
        shader BadHipStageParameterDuplicate {
            vertex {
                void main(uint vertexId @ gl_VertexID, uint alsoVertexId @ SV_VertexID) {
                }
            }
        }
        """
        with pytest.raises(ValueError, match="Duplicate HIP stage parameter semantic"):
            HipCodeGen().generate(Parser(Lexer(duplicate_system_value).tokens).parse())

    def test_stage_parameter_semantics_thread_position_in_grid(self):
        source_code = """
        shader ComputeBuiltinsShader {
            compute {
                void main() {
                    int localX = gl_LocalInvocationID.x;
                    int localY = gl_LocalInvocationID.y;
                    int localZ = gl_LocalInvocationID.z;
                    int globalX = gl_GlobalInvocationID.x;
                    int globalY = gl_GlobalInvocationID.y;
                    int globalZ = gl_GlobalInvocationID.z;
                    int groupX = gl_WorkGroupID.x;
                    int groupY = gl_WorkGroupID.y;
                    int groupZ = gl_WorkGroupID.z;
                    int sizeX = gl_WorkGroupSize.x;
                    int sizeY = gl_WorkGroupSize.y;
                    int sizeZ = gl_WorkGroupSize.z;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "int localX = threadIdx.x;" in hip_code
        assert "int localY = threadIdx.y;" in hip_code
        assert "int localZ = threadIdx.z;" in hip_code
        assert "int globalX = (blockIdx.x * blockDim.x + threadIdx.x);" in hip_code
        assert "int globalY = (blockIdx.y * blockDim.y + threadIdx.y);" in hip_code
        assert "int globalZ = (blockIdx.z * blockDim.z + threadIdx.z);" in hip_code
        assert "int groupX = blockIdx.x;" in hip_code
        assert "int groupY = blockIdx.y;" in hip_code
        assert "int groupZ = blockIdx.z;" in hip_code
        assert "int sizeX = blockDim.x;" in hip_code
        assert "int sizeY = blockDim.y;" in hip_code
        assert "int sizeZ = blockDim.z;" in hip_code
        assert "gl_LocalInvocationID" not in hip_code
        assert "gl_GlobalInvocationID" not in hip_code
        assert "gl_WorkGroupID" not in hip_code
        assert "gl_WorkGroupSize" not in hip_code

    def test_vertex_stage_with_position_output(self):
        source_code = """
        shader VertexPositionShader {
            vertex {
                void main() {
                    vec4 position = vec4(1.0, 2.0, 3.0, 1.0);
                    vec3 normal = vec3(0.0, 1.0, 0.0);
                    vec2 uv = vec2(0.5, 0.5);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "__device__ void main()" in hip_code
        assert "float4 position = make_float4(1.0, 2.0, 3.0, 1.0);" in hip_code
        assert "float3 normal = make_float3(0.0, 1.0, 0.0);" in hip_code
        assert "float2 uv = make_float2(0.5, 0.5);" in hip_code
        assert "vec4" not in hip_code
        assert "vec3" not in hip_code
        assert "vec2(" not in hip_code

    def test_texture_sampling_with_cubemap_and_1d_types(self):
        source_code = """
        shader CubeAndLinearShader {
            void sampleVariety(
                sampler1d paramRamp,
                samplercube paramEnv,
                float u,
                vec3 dir
            ) {
                vec4 ramp = texture(paramRamp, u);
                vec4 rampLod = textureLod(paramRamp, u, 1.0);
                vec4 env = texture(paramEnv, dir);
                vec4 envLod = textureLod(paramEnv, dir, 2.0);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "float4 ramp = tex1D<float4>(paramRamp, u);" in hip_code
        assert "float4 rampLod = tex1DLod<float4>(paramRamp, u, 1.0);" in hip_code
        assert (
            "float4 env = texCubemap<float4>(paramEnv, dir.x, dir.y, dir.z);"
            in hip_code
        )
        assert (
            "float4 envLod = texCubemapLod<float4>(paramEnv, dir.x, dir.y, dir.z, 2.0);"
            in hip_code
        )
        assert " = texture(" not in hip_code
        assert " = textureLod(" not in hip_code

    def test_match_multiple_literal_arms_emit_switch_cases(self):
        source_code = """
        shader TestShader {
            compute {
                int main(int code) {
                    int result = 0;
                    match code {
                        1 => {
                            result = 10;
                        }
                        2 => {
                            result = 20;
                        }
                        3 => {
                            result = 30;
                        }
                        _ => {
                            result = -1;
                        }
                    }
                    return result;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "switch (code)" in hip_code
        assert "case 1:" in hip_code
        assert "case 2:" in hip_code
        assert "case 3:" in hip_code
        assert "result = 10;" in hip_code
        assert "result = 20;" in hip_code
        assert "result = 30;" in hip_code
        assert "default:" in hip_code
        assert "result = -1;" in hip_code
        assert hip_code.count("break;") == 4

    def test_match_wildcard_only_emits_default_case(self):
        source_code = """
        shader TestShader {
            compute {
                int main(int x) {
                    int out = 0;
                    match x {
                        _ => {
                            out = 99;
                        }
                    }
                    return out;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "switch (x)" in hip_code
        assert "default:" in hip_code
        assert "out = 99;" in hip_code
        assert "case " not in hip_code

    def test_match_constructor_pattern_rejected_for_hip(self):
        match_ast = ShaderNode(
            name="ConstructorMatch",
            execution_model=ExecutionModel.COMPUTE_KERNEL,
            functions=[
                FunctionNode(
                    "main",
                    PrimitiveType("void"),
                    [],
                    BlockNode(
                        [
                            MatchNode(
                                expression=IdentifierNode("val"),
                                arms=[
                                    MatchArmNode(
                                        pattern=ConstructorPatternNode(
                                            "Some", [WildcardPatternNode()]
                                        ),
                                        guard=None,
                                        body=[
                                            AssignmentNode(
                                                IdentifierNode("x"),
                                                LiteralNode(1, PrimitiveType("int")),
                                                "=",
                                            )
                                        ],
                                    ),
                                    MatchArmNode(
                                        pattern=WildcardPatternNode(),
                                        guard=None,
                                        body=[
                                            AssignmentNode(
                                                IdentifierNode("x"),
                                                LiteralNode(0, PrimitiveType("int")),
                                                "=",
                                            )
                                        ],
                                    ),
                                ],
                            )
                        ]
                    ),
                    qualifiers=["compute"],
                )
            ],
        )

        codegen = HipCodeGen()
        with pytest.raises(ValueError, match="Unsupported match arm for HIP"):
            codegen.generate(match_ast)

    def test_shader_with_only_structs_produces_valid_output(self):
        struct_ast = ShaderNode(
            name="StructsOnly",
            execution_model=ExecutionModel.GENERAL_PURPOSE,
            structs=[
                StructNode(
                    "Particle",
                    [
                        VariableNode("position", PrimitiveType("float3")),
                        VariableNode("velocity", PrimitiveType("float3")),
                        VariableNode("mass", PrimitiveType("float")),
                    ],
                ),
            ],
            functions=[],
        )

        codegen = HipCodeGen()
        hip_code = codegen.generate(struct_ast)

        assert "#include <hip/hip_runtime.h>" in hip_code
        assert "struct Particle" in hip_code
        assert "float3 position;" in hip_code
        assert "float3 velocity;" in hip_code
        assert "float mass;" in hip_code
        assert "__global__" not in hip_code

    def test_invalid_mixed_stage_vertex_and_compute_still_generates(self):
        source_code = """
        shader MixedStageShader {
            vertex {
                void main() {
                    int a = 1;
                }
            }
            compute {
                void main() {
                    int b = 2;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "__device__" in hip_code
        assert "__global__" in hip_code
        assert "int a = 1;" in hip_code
        assert "int b = 2;" in hip_code

    def test_match_literals_without_wildcard_emits_no_default(self):
        source_code = """
        shader TestShader {
            compute {
                int main(int flag) {
                    int r = 0;
                    match flag {
                        0 => {
                            r = 100;
                        }
                        1 => {
                            r = 200;
                        }
                    }
                    return r;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        codegen = HipCodeGen()
        hip_code = codegen.generate(ast)

        assert "switch (flag)" in hip_code
        assert "case 0:" in hip_code
        assert "case 1:" in hip_code
        assert "r = 100;" in hip_code
        assert "r = 200;" in hip_code
        assert "default:" not in hip_code

    def test_readonly_qualifier_on_buffer_produces_const_pointer(self):
        source_code = """
        shader ReadonlyBufferShader {
            StructuredBuffer<float> weights;
            StructuredBuffer<int> indices[4];

            float sumWeights(StructuredBuffer<float> input, uint count) {
                float total = 0.0;
                for (uint i = 0; i < count; i = i + 1) {
                    total = total + input.Load(i);
                }
                return total;
            }

            compute {
                void main() {
                    float w = weights.Load(0);
                    int idx = indices[0].Load(0);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "const float* weights;" in hip_code
        assert "const int* indices[4];" in hip_code
        assert "const float* input" in hip_code
        assert "float w = weights[0];" in hip_code
        assert "int idx = indices[0][0];" in hip_code
        assert "float total = 0.0;" in hip_code
        assert "total = (total + input[i]);" in hip_code
        assert "StructuredBuffer" not in hip_code
        assert "RWStructuredBuffer" not in hip_code

    def test_writeonly_buffer_produces_mutable_pointer(self):
        source_code = """
        shader WriteBufferShader {
            struct Pixel {
                float r;
                float g;
                float b;
            };

            RWStructuredBuffer<Pixel> output;
            RWStructuredBuffer<int> counters;

            void writePixel(RWStructuredBuffer<Pixel> dest, uint index, Pixel value) {
                dest.Store(index, value);
            }

            compute {
                void main() {
                    Pixel p;
                    p.r = 1.0;
                    p.g = 0.5;
                    p.b = 0.0;
                    output.Store(0, p);
                    counters[0] = 1;
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "Pixel* output;" in hip_code
        assert "int* counters;" in hip_code
        assert "__device__ void writePixel(Pixel* dest," in hip_code
        assert "dest[index] = value;" in hip_code
        assert "output[0] = p;" in hip_code
        assert "counters[0] = 1;" in hip_code

    def test_resource_memory_qualifiers_map_hip_access_contracts(self):
        source_code = """
        shader HIPResourceMemoryQualifiers {
            readonly RWStructuredBuffer<int> readOnlyValues;
            writeonly StructuredBuffer<int> writeOnlyValues;
            readwrite StructuredBuffer<int> readWriteValues;
            RWStructuredBuffer<int> attrReadValues @access(read);
            StructuredBuffer<int> attrWriteValues @access(write);
            readonly RWByteAddressBuffer rawInput;
            writeonly ByteAddressBuffer rawOutput;
            readonly image2D source @rgba32f;
            writeonly image2D target @rgba32f;

            compute {
                void main(uint index) {
                    int value = buffer_load(readOnlyValues, index);
                    int attrValue = buffer_load(attrReadValues, index);
                    uint raw = buffer_load(rawInput, index);
                    buffer_store(writeOnlyValues, index, value + attrValue);
                    buffer_store(readWriteValues, index, value);
                    buffer_store(attrWriteValues, index, attrValue);
                    buffer_store(rawOutput, index, raw);
                    vec4 color = imageLoad(source, ivec2(0));
                    vec4 blockedLoad = imageLoad(target, ivec2(0));
                    imageStore(target, ivec2(0), color);
                    imageStore(source, ivec2(0), color);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "const int* readOnlyValues;" in hip_code
        assert "int* writeOnlyValues;" in hip_code
        assert "int* readWriteValues;" in hip_code
        assert "const int* attrReadValues;" in hip_code
        assert "int* attrWriteValues;" in hip_code
        assert "const unsigned char* rawInput;" in hip_code
        assert "unsigned char* rawOutput;" in hip_code
        assert "hipSurfaceObject_t source;" in hip_code
        assert "hipSurfaceObject_t target;" in hip_code
        assert "int value = readOnlyValues[index];" in hip_code
        assert "int attrValue = attrReadValues[index];" in hip_code
        assert "unsigned int raw = cgl_byte_address_load_uint(rawInput, index);" in (
            hip_code
        )
        assert "writeOnlyValues[index] = (value + attrValue);" in hip_code
        assert "readWriteValues[index] = value;" in hip_code
        assert "attrWriteValues[index] = attrValue;" in hip_code
        assert "cgl_byte_address_store_uint(rawOutput, index, raw);" in hip_code
        assert "float4 color = cgl_surf2Dread<float4>(source," in hip_code
        assert (
            "float4 blockedLoad = /* unsupported HIP image access: imageLoad "
            "requires readable image resource on image2D */ "
            "make_float4(0.0f, 0.0f, 0.0f, 0.0f);" in hip_code
        )
        assert "surf2Dwrite(color, target," in hip_code
        assert (
            "/* unsupported HIP image access: imageStore requires writable "
            "image resource on image2D */ ((void)0);" in hip_code
        )
        assert "readonly" not in hip_code
        assert "writeonly" not in hip_code
        assert "readwrite" not in hip_code
        assert "access(" not in hip_code

    def test_resource_memory_qualifiers_reject_writeonly_hip_buffer_reads(self):
        source_code = """
        shader HIPWriteOnlyBufferDiagnostics {
            writeonly StructuredBuffer<int> writeOnlyValues;
            writeonly ByteAddressBuffer writeOnlyBytes;
            writeonly RWStructuredBuffer<uint> atomicValues;
            writeonly RWByteAddressBuffer atomicBytes;

            compute {
                void main(uint index, uint amount) {
                    int blockedMemberRead = writeOnlyValues.Load(index);
                    int blockedIndexRead = writeOnlyValues[index];
                    int blockedFunctionRead = buffer_load(writeOnlyValues, index);
                    uint blockedRawMember = writeOnlyBytes.Load(index);
                    uint2 blockedRawPair = writeOnlyBytes.Load2(index);
                    uint blockedRawFunction = buffer_load(writeOnlyBytes, index);
                    uint blockedAtomic = atomicAdd(atomicValues[index], amount);
                    uint blockedRawAtomic =
                        atomicBytes.InterlockedAdd(index, amount);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "int* writeOnlyValues;" in hip_code
        assert "unsigned char* writeOnlyBytes;" in hip_code
        assert "unsigned int* atomicValues;" in hip_code
        assert "unsigned char* atomicBytes;" in hip_code
        assert (
            "int blockedMemberRead = /* unsupported HIP structured buffer "
            "access: Load requires readable buffer access on "
            "RWStructuredBuffer<int> */ 0;" in hip_code
        )
        assert (
            "int blockedIndexRead = /* unsupported HIP structured buffer "
            "access: load requires readable buffer access on "
            "RWStructuredBuffer<int> */ 0;" in hip_code
        )
        assert (
            "int blockedFunctionRead = /* unsupported HIP structured buffer "
            "access: buffer_load requires readable buffer access on "
            "RWStructuredBuffer<int> */ 0;" in hip_code
        )
        assert (
            "unsigned int blockedRawMember = /* unsupported HIP byte-address "
            "buffer access: Load requires readable buffer access on "
            "RWByteAddressBuffer */ 0u;" in hip_code
        )
        assert (
            "uint2 blockedRawPair = /* unsupported HIP byte-address buffer "
            "access: Load2 requires readable buffer access on "
            "RWByteAddressBuffer */ make_uint2(0u, 0u);" in hip_code
        )
        assert (
            "unsigned int blockedRawFunction = /* unsupported HIP "
            "byte-address buffer access: buffer_load requires readable "
            "buffer access on RWByteAddressBuffer */ 0u;" in hip_code
        )
        assert (
            "unsigned int blockedAtomic = /* unsupported HIP structured "
            "buffer access: atomicAdd requires readwrite buffer access on "
            "RWStructuredBuffer<uint> */ 0u;" in hip_code
        )
        assert (
            "unsigned int blockedRawAtomic = /* unsupported HIP byte-address "
            "buffer access: InterlockedAdd requires readwrite buffer access "
            "on RWByteAddressBuffer */ 0u;" in hip_code
        )
        assert ".Load(" not in hip_code
        assert "buffer_load(" not in hip_code
        assert "InterlockedAdd(" not in hip_code

    def test_resource_memory_qualifiers_emit_hip_pointer_qualifiers(self):
        source_code = """
        shader HIPNativeResourceMemoryQualifiers {
            struct BlockData {
                uint value;
            };

            RWStructuredBuffer<uint> volatileValues @volatile;
            RWStructuredBuffer<uint> restrictedValues @restrict;
            readonly RWStructuredBuffer<uint> readOnlyRestricted
                @volatile @restrict;
            RWByteAddressBuffer rawRestricted @volatile @restrict;

            void consume(
                readonly device BlockData* rawBlock @buffer(0),
                volatile restrict RWStructuredBuffer<uint> paramValues,
                RWByteAddressBuffer paramRaw @volatile @restrict
            ) {
                uint value = paramValues.Load(0);
                uint raw = paramRaw.Load(0);
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "volatile unsigned int* volatileValues;" in hip_code
        assert "unsigned int* __restrict__ restrictedValues;" in hip_code
        assert (
            "const volatile unsigned int* __restrict__ readOnlyRestricted;" in hip_code
        )
        assert "volatile unsigned char* __restrict__ rawRestricted;" in hip_code
        assert (
            "__device__ void consume(const BlockData* rawBlock, "
            "volatile unsigned int* __restrict__ paramValues, "
            "volatile unsigned char* __restrict__ paramRaw)"
        ) in hip_code
        assert "unsigned int value = paramValues[0];" in hip_code
        assert "unsigned int raw = cgl_byte_address_load_uint(paramRaw, 0);" in (
            hip_code
        )
        assert "@volatile" not in hip_code
        assert "@restrict" not in hip_code
        assert " readonly " not in hip_code
        assert " restrict " not in hip_code

    def test_explicit_binding_annotations_on_resources_parse_without_error(self):
        source_code = """
        shader ExplicitBindingShader {
            sampler2D albedoTex @texture(0);
            sampler2D normalTex @texture(1);
            sampler2D roughnessTex @binding(2);
            image2D outputImage @binding(3);

            compute {
                void main() {
                    vec4 color = texelFetch(albedoTex, ivec2(0, 0), 0);
                    vec4 normal = texelFetch(normalTex, ivec2(0, 0), 0);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "hipTextureObject_t albedoTex;" in hip_code
        assert "hipTextureObject_t normalTex;" in hip_code
        assert "hipSurfaceObject_t outputImage;" in hip_code
        assert "@binding" not in hip_code
        assert "@texture" not in hip_code
        assert "__global__ void compute_main()" in hip_code

    def test_automatic_sequential_binding_for_multiple_resources(self):
        source_code = """
        shader MultiResourceShader {
            StructuredBuffer<float> positionBuffer;
            StructuredBuffer<float> normalBuffer;
            StructuredBuffer<float> uvBuffer;
            RWStructuredBuffer<float> outputBuffer;

            void transformVertex(
                StructuredBuffer<float> positions,
                StructuredBuffer<float> normals,
                RWStructuredBuffer<float> dest,
                uint index
            ) {
                float pos = positions.Load(index);
                float nrm = normals.Load(index);
                dest.Store(index, pos + nrm);
            }

            compute {
                void main() {
                    float p = positionBuffer.Load(0);
                    float n = normalBuffer.Load(0);
                    float u = uvBuffer.Load(0);
                    outputBuffer.Store(0, p + n + u);
                }
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "const float* positionBuffer;" in hip_code
        assert "const float* normalBuffer;" in hip_code
        assert "const float* uvBuffer;" in hip_code
        assert "float* outputBuffer;" in hip_code
        assert (
            "__device__ void transformVertex("
            "const float* positions, const float* normals, "
            "float* dest, unsigned int index)"
        ) in hip_code
        assert "float pos = positions[index];" in hip_code
        assert "float nrm = normals[index];" in hip_code
        assert "dest[index] = (pos + nrm);" in hip_code

        pos_idx = hip_code.index("const float* positionBuffer;")
        norm_idx = hip_code.index("const float* normalBuffer;")
        uv_idx = hip_code.index("const float* uvBuffer;")
        out_idx = hip_code.index("float* outputBuffer;")
        assert pos_idx < norm_idx < uv_idx < out_idx

    def test_multiple_textures_with_explicit_binding_slots(self):
        source_code = """
        shader MultiTextureBindingShader {
            sampler2D diffuse @texture(0);
            sampler2D specular @texture(1);
            sampler2D ambient @texture(2);
            sampler samp @sampler(0);

            vec4 sampleAll(vec2 uv) {
                vec4 d = texture(diffuse, samp, uv);
                vec4 s = texture(specular, samp, uv);
                vec4 a = texture(ambient, samp, uv);
                return d + s + a;
            }

            compute {
                void main() {}
            }
        }
        """

        lexer = Lexer(source_code)
        parser = Parser(lexer.tokens)
        ast = parser.parse()

        hip_code = HipCodeGen().generate(ast)

        assert "hipTextureObject_t diffuse;" in hip_code
        assert "hipTextureObject_t specular;" in hip_code
        assert "hipTextureObject_t ambient;" in hip_code
        assert "@texture" not in hip_code
        assert "@sampler" not in hip_code
        assert "sampler2D" not in hip_code

        diff_idx = hip_code.index("hipTextureObject_t diffuse;")
        spec_idx = hip_code.index("hipTextureObject_t specular;")
        amb_idx = hip_code.index("hipTextureObject_t ambient;")
        assert diff_idx < spec_idx < amb_idx
