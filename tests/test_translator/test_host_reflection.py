import textwrap

from crosstl.project import host_reflection
from crosstl.project.host_reflection import reflect_target_host_interface


def _reflect_hlsl(tmp_path, source, *, stage=None):
    artifact = tmp_path / "kernel.hlsl"
    artifact.write_text(textwrap.dedent(source).strip(), encoding="utf-8")
    return reflect_target_host_interface(artifact, target="directx", stage=stage)


def test_hlsl_reflection_infers_compute_stage_from_numthreads(tmp_path):
    reflection = _reflect_hlsl(
        tmp_path,
        """
        [numthreads(4, 2, 1)]
        void CopyMain(uint3 tid : SV_DispatchThreadID) {}
        """,
    )

    assert reflection["entryPoints"] == [
        {
            "name": "CopyMain",
            "stage": "compute",
            "executionConfig": {"numthreads": [4, 2, 1]},
        }
    ]


def test_hlsl_reflection_preserves_entry_points_resources_and_constants(tmp_path):
    reflection = _reflect_hlsl(
        tmp_path,
        """
        cbuffer Frame : register(b2, space3) {
            float exposure;
        };
        Texture2D<float4> sourceTexture : register(t1, space3);
        RWStructuredBuffer<float> outputValues : register(u4);
        SamplerState linearSampler : register(s5);
        static const uint tileSize = 16u;

        #if ENABLE_GENERATED_KERNEL
        [shader("compute")]
        [numthreads(8, 4, 2)]
        void GeneratedKernel(uint3 tid : SV_DispatchThreadID) {
            outputValues[tid.x] = sourceTexture.Load(int3(tid.xy, 0)).x;
        }
        #endif

        float4 PSMain() : SV_Target0 {
            return float4(1.0, 1.0, 1.0, 1.0);
        }
        """,
    )

    assert reflection["entryPoints"] == [
        {
            "name": "GeneratedKernel",
            "stage": "compute",
            "executionConfig": {"numthreads": [8, 4, 2]},
        },
        {
            "name": "PSMain",
            "stage": "fragment",
            "executionConfig": {},
        },
    ]
    assert reflection["resources"] == [
        {
            "name": "Frame",
            "kind": "constant-buffer",
            "type": "Frame",
            "set": 3,
            "binding": 2,
            "access": "read",
            "scalarLayout": {
                "physicalType": "float",
                "elementType": "float32",
                "elementSizeBytes": 4,
                "elementStrideBytes": 4,
                "alignmentBytes": 16,
                "memberOffsetBytes": 0,
                "storageLayout": "hlsl-constant-buffer",
                "runtimeSized": False,
                "memberName": "exposure",
                "blockSizeBytes": 16,
            },
        },
        {
            "name": "sourceTexture",
            "kind": "texture",
            "type": "Texture2D<float4>",
            "set": 3,
            "binding": 1,
            "access": "read",
        },
        {
            "name": "outputValues",
            "kind": "buffer",
            "type": "RWStructuredBuffer<float>",
            "set": 0,
            "binding": 4,
            "access": "read_write",
            "scalarLayout": {
                "physicalType": "float",
                "elementType": "float32",
                "elementSizeBytes": 4,
                "elementStrideBytes": 4,
                "alignmentBytes": 4,
                "memberOffsetBytes": 0,
                "storageLayout": "hlsl-structured-buffer",
                "runtimeSized": True,
            },
        },
        {
            "name": "linearSampler",
            "kind": "sampler",
            "type": "SamplerState",
            "set": 0,
            "binding": 5,
            "access": None,
        },
    ]
    assert reflection["constants"] == [
        {
            "name": "tileSize",
            "kind": "scalar-constant",
            "dtype": "uint",
            "value": 16,
            "required": False,
            "source": "hlsl.const",
        }
    ]


def test_metal_reflection_records_entries_resources_and_function_constants(tmp_path):
    artifact = tmp_path / "kernel.metal"
    artifact.write_text(
        textwrap.dedent("""
            #include <metal_stdlib>
            using namespace metal;

            constant bool enabled [[function_constant(20)]];

            kernel void first(
                const device float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                texture2d<float, access::read> source [[texture(0)]],
                sampler linear_sampler [[sampler(0)]]) {
                output[0] = enabled ? input[0] : source.read(uint2(0)).x;
            }

            kernel void second(
                device uint* output [[buffer(0)]]) {
                output[0] = 1u;
            }
            """).strip(),
        encoding="utf-8",
    )

    reflection = reflect_target_host_interface(artifact, target="metal")

    assert reflection["status"] == "ready"
    assert reflection["parser"] == "metal-reflection"
    assert reflection["entryPoints"] == [
        {"name": "first", "stage": "compute", "executionConfig": {}},
        {"name": "second", "stage": "compute", "executionConfig": {}},
    ]
    assert [
        (
            resource["name"],
            resource["kind"],
            resource["binding"],
            resource["access"],
            resource["metadata"]["entryPoint"],
        )
        for resource in reflection["resources"]
    ] == [
        ("input", "buffer", 0, "read", "first"),
        ("output", "buffer", 1, "read_write", "first"),
        ("source", "texture", 0, "read", "first"),
        ("linear_sampler", "sampler", 0, "read", "first"),
        ("output", "buffer", 0, "read_write", "second"),
    ]
    assert reflection["specializationConstants"] == [
        {
            "name": "enabled",
            "kind": "function-constant",
            "id": 20,
            "dtype": "bool",
            "sourceType": "bool",
            "required": True,
            "overridden": False,
            "overrideStatus": "not-overridden",
            "status": "required",
            "source": "metal.function_constant",
        }
    ]
    assert reflection["diagnostics"] == []


def test_glsl_reflection_canonicalizes_c_family_specialization_ids(tmp_path):
    artifact = tmp_path / "kernel.comp"
    artifact.write_text(
        textwrap.dedent("""
            #version 450 core
            layout(local_size_x = 1) in;
            layout(constant_id = 00) const int zero = 0;
            layout(constant_id = 01) const int one = 1;
            layout(constant_id = 10u) const int decimal = 10;
            layout(constant_id = 0x10u) const int hexadecimal = 16;
            layout(constant_id = 0b10000u) const int binary = 16;
            void main() {}
            """).strip(),
        encoding="utf-8",
    )

    reflection = reflect_target_host_interface(
        artifact, target="opengl", stage="compute"
    )

    assert [
        (constant["name"], constant["id"])
        for constant in reflection["specializationConstants"]
    ] == [
        ("zero", 0),
        ("one", 1),
        ("decimal", 10),
        ("hexadecimal", 16),
        ("binary", 16),
    ]


def test_glsl_reflection_records_exact_mlx_scalar_block_layouts(tmp_path):
    artifact = tmp_path / "arangeuint32.glsl"
    artifact.write_text(
        textwrap.dedent("""
            #version 450 core
            layout(std430, binding = 2) buffer out_Buffer { uint out_[]; };
            layout(std140, binding = 0) uniform arangeuint32_start_Args {
                uint start;
            };
            layout(std140, binding = 1) uniform arangeuint32_step_Args {
                uint step;
            };
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {}
            """).strip(),
        encoding="utf-8",
    )

    reflection = reflect_target_host_interface(
        artifact, target="opengl", stage="compute"
    )

    by_name = {resource["name"]: resource for resource in reflection["resources"]}
    assert by_name["out_Buffer"]["scalarLayout"] == {
        "physicalType": "uint",
        "elementType": "uint32",
        "elementSizeBytes": 4,
        "elementStrideBytes": 4,
        "alignmentBytes": 4,
        "memberOffsetBytes": 0,
        "storageLayout": "std430",
        "runtimeSized": True,
        "memberName": "out_",
    }
    assert by_name["arangeuint32_start_Args"]["scalarLayout"] == {
        "physicalType": "uint",
        "elementType": "uint32",
        "elementSizeBytes": 4,
        "elementStrideBytes": 4,
        "alignmentBytes": 16,
        "memberOffsetBytes": 0,
        "storageLayout": "std140",
        "runtimeSized": False,
        "memberName": "start",
        "blockSizeBytes": 16,
    }
    assert by_name["arangeuint32_step_Args"]["scalarLayout"]["memberName"] == "step"


def test_glsl_reflection_records_exact_vector_block_layouts(tmp_path):
    artifact = tmp_path / "vectors.glsl"
    artifact.write_text(
        textwrap.dedent("""
            #version 450 core
            #extension GL_ARB_gpu_shader_int64 : require
            layout(std430, binding = 0) readonly buffer Inputs { vec2 values[]; };
            layout(std430, binding = 1) buffer Triples { vec3 values[]; };
            layout(std430, binding = 2) buffer Outputs { vec4 values[]; };
            layout(std430, binding = 3) buffer Integers { ivec2 values[]; };
            layout(std430, binding = 4) buffer Unsigned { uvec4 values[]; };
            layout(std430, binding = 5) buffer Flags { bvec2 values[]; };
            layout(std430, binding = 6) buffer WideIntegers { i64vec2 values[]; };
            layout(std430, binding = 7) buffer WideTriples { u64vec3 values[]; };
            layout(std140, binding = 8) uniform Scale { vec2 value; };
            layout(std430, binding = 9) buffer Unsupported { dvec2 values[]; };
            layout(local_size_x = 1) in;
            void main() {}
            """).strip(),
        encoding="utf-8",
    )

    reflection = reflect_target_host_interface(
        artifact, target="opengl", stage="compute"
    )
    by_name = {resource["name"]: resource for resource in reflection["resources"]}

    assert by_name["Inputs"]["scalarLayout"] == {
        "physicalType": "float2",
        "elementType": "float32",
        "elementSizeBytes": 8,
        "elementStrideBytes": 8,
        "alignmentBytes": 8,
        "memberOffsetBytes": 0,
        "storageLayout": "std430",
        "runtimeSized": True,
        "vectorWidth": 2,
        "memberName": "values",
    }
    assert by_name["Triples"]["scalarLayout"] == {
        "physicalType": "float3",
        "elementType": "float32",
        "elementSizeBytes": 12,
        "elementStrideBytes": 16,
        "alignmentBytes": 16,
        "memberOffsetBytes": 0,
        "storageLayout": "std430",
        "runtimeSized": True,
        "vectorWidth": 3,
        "memberName": "values",
    }
    assert by_name["Outputs"]["scalarLayout"] == {
        **by_name["Triples"]["scalarLayout"],
        "physicalType": "float4",
        "elementSizeBytes": 16,
        "vectorWidth": 4,
    }
    assert by_name["Integers"]["scalarLayout"] == {
        **by_name["Inputs"]["scalarLayout"],
        "physicalType": "int2",
        "elementType": "int32",
    }
    assert by_name["Unsigned"]["scalarLayout"] == {
        **by_name["Outputs"]["scalarLayout"],
        "physicalType": "uint4",
        "elementType": "uint32",
    }
    assert by_name["Flags"]["scalarLayout"] == {
        **by_name["Inputs"]["scalarLayout"],
        "physicalType": "uint2",
        "elementType": "uint32",
    }
    assert by_name["WideIntegers"]["scalarLayout"] == {
        **by_name["Inputs"]["scalarLayout"],
        "physicalType": "int64_t2",
        "elementType": "int64",
        "elementSizeBytes": 16,
        "elementStrideBytes": 16,
        "alignmentBytes": 16,
    }
    assert by_name["WideTriples"]["scalarLayout"] == {
        **by_name["Triples"]["scalarLayout"],
        "physicalType": "uint64_t3",
        "elementType": "uint64",
        "elementSizeBytes": 24,
        "elementStrideBytes": 32,
        "alignmentBytes": 32,
    }
    assert by_name["Scale"]["scalarLayout"] == {
        **by_name["Inputs"]["scalarLayout"],
        "alignmentBytes": 16,
        "storageLayout": "std140",
        "runtimeSized": False,
        "memberName": "value",
        "blockSizeBytes": 16,
    }
    assert "scalarLayout" not in by_name["Unsupported"]


def test_hlsl_reflection_records_exact_vector_resource_layouts(tmp_path):
    reflection = _reflect_hlsl(
        tmp_path,
        """
        StructuredBuffer<float2> in_ : register(t0);
        RWStructuredBuffer<float2> out_ : register(u1);
        cbuffer CrossGLDispatchInfo : register(b4) {
            uint3 crossglNumWorkGroups;
        };
        [numthreads(1, 1, 64)] void CSMain() {}
        """,
    )

    by_name = {resource["name"]: resource for resource in reflection["resources"]}
    vector_buffer_layout = {
        "physicalType": "float2",
        "elementType": "float32",
        "elementSizeBytes": 8,
        "elementStrideBytes": 8,
        "alignmentBytes": 4,
        "memberOffsetBytes": 0,
        "storageLayout": "hlsl-structured-buffer",
        "runtimeSized": True,
        "vectorWidth": 2,
    }
    assert by_name["in_"]["scalarLayout"] == vector_buffer_layout
    assert by_name["out_"]["scalarLayout"] == vector_buffer_layout
    assert by_name["CrossGLDispatchInfo"]["scalarLayout"] == {
        "physicalType": "uint3",
        "elementType": "uint32",
        "elementSizeBytes": 12,
        "elementStrideBytes": 12,
        "alignmentBytes": 16,
        "memberOffsetBytes": 0,
        "storageLayout": "hlsl-constant-buffer",
        "runtimeSized": False,
        "vectorWidth": 3,
        "memberName": "crossglNumWorkGroups",
        "blockSizeBytes": 16,
    }
    assert by_name["CrossGLDispatchInfo"]["provenance"] == {
        "kind": "generated-execution-input",
        "executionInput": {
            "kind": "dispatch-workgroup-count",
            "valueSource": "dispatch.workgroupCount",
            "coordinateSpace": "physical",
            "dimensions": 3,
            "memberName": "crossglNumWorkGroups",
        },
    }


def test_source_reflection_records_stored_bool_as_uint32_abi(tmp_path):
    hlsl = _reflect_hlsl(
        tmp_path,
        """
        StructuredBuffer<bool> inputFlags : register(t0);
        RWStructuredBuffer<bool> outputFlags : register(u1);
        [numthreads(1, 1, 1)] void CSMain() {}
        """,
    )
    hlsl_by_name = {resource["name"]: resource for resource in hlsl["resources"]}
    hlsl_layout = {
        "physicalType": "uint",
        "elementType": "uint32",
        "elementSizeBytes": 4,
        "elementStrideBytes": 4,
        "alignmentBytes": 4,
        "memberOffsetBytes": 0,
        "storageLayout": "hlsl-structured-buffer",
        "runtimeSized": True,
    }
    assert hlsl_by_name["inputFlags"]["scalarLayout"] == hlsl_layout
    assert hlsl_by_name["outputFlags"]["scalarLayout"] == hlsl_layout

    artifact = tmp_path / "bool.comp"
    artifact.write_text(
        textwrap.dedent("""
            #version 450 core
            layout(std430, binding = 0) readonly buffer InputFlags {
                bool inputFlags[];
            };
            layout(std430, binding = 1) buffer OutputFlags {
                bool outputFlags[];
            };
            layout(local_size_x = 1) in;
            void main() {}
            """).strip(),
        encoding="utf-8",
    )
    glsl = reflect_target_host_interface(artifact, target="opengl", stage="compute")
    glsl_by_name = {resource["name"]: resource for resource in glsl["resources"]}
    assert glsl_by_name["InputFlags"]["scalarLayout"] == {
        **hlsl_layout,
        "storageLayout": "std430",
        "memberName": "inputFlags",
    }
    assert glsl_by_name["OutputFlags"]["scalarLayout"] == {
        **hlsl_layout,
        "storageLayout": "std430",
        "memberName": "outputFlags",
    }


def test_source_reflection_records_exact_64_bit_scalar_layouts(tmp_path):
    hlsl = _reflect_hlsl(
        tmp_path,
        """
        StructuredBuffer<int64_t> inputStrides : register(t0);
        RWStructuredBuffer<uint64_t> outputOffsets : register(u1);
        cbuffer AxisStride : register(b2) { int64_t axisStride; };
        cbuffer AxisSize : register(b3) { uint64_t axisSize; };
        [numthreads(1, 1, 1)] void CSMain() {}
        """,
    )
    hlsl_by_name = {resource["name"]: resource for resource in hlsl["resources"]}
    assert hlsl_by_name["inputStrides"]["scalarLayout"] == {
        "physicalType": "int64_t",
        "elementType": "int64",
        "elementSizeBytes": 8,
        "elementStrideBytes": 8,
        "alignmentBytes": 8,
        "memberOffsetBytes": 0,
        "storageLayout": "hlsl-structured-buffer",
        "runtimeSized": True,
    }
    assert hlsl_by_name["outputOffsets"]["scalarLayout"]["elementType"] == ("uint64")
    assert hlsl_by_name["AxisStride"]["scalarLayout"] == {
        "physicalType": "int64_t",
        "elementType": "int64",
        "elementSizeBytes": 8,
        "elementStrideBytes": 8,
        "alignmentBytes": 16,
        "memberOffsetBytes": 0,
        "storageLayout": "hlsl-constant-buffer",
        "runtimeSized": False,
        "memberName": "axisStride",
        "blockSizeBytes": 16,
    }
    assert hlsl_by_name["AxisSize"]["scalarLayout"]["elementType"] == "uint64"

    artifact = tmp_path / "kernel.comp"
    artifact.write_text(
        textwrap.dedent("""
            #version 450 core
            #extension GL_ARB_gpu_shader_int64 : require
            layout(std430, binding = 0) readonly buffer InputStrides {
                int64_t inputStrides[];
            };
            layout(std430, binding = 1) buffer OutputOffsets {
                uint64_t outputOffsets[];
            };
            layout(std140, binding = 2) uniform AxisStride {
                int64_t axisStride;
            };
            layout(std140, binding = 3) uniform AxisSize {
                uint64_t axisSize;
            };
            layout(local_size_x = 1) in;
            void main() {}
            """).strip(),
        encoding="utf-8",
    )
    glsl = reflect_target_host_interface(artifact, target="opengl", stage="compute")
    glsl_by_name = {resource["name"]: resource for resource in glsl["resources"]}
    assert glsl_by_name["InputStrides"]["scalarLayout"] == {
        "physicalType": "int64_t",
        "elementType": "int64",
        "elementSizeBytes": 8,
        "elementStrideBytes": 8,
        "alignmentBytes": 8,
        "memberOffsetBytes": 0,
        "storageLayout": "std430",
        "runtimeSized": True,
        "memberName": "inputStrides",
    }
    assert glsl_by_name["OutputOffsets"]["scalarLayout"]["elementType"] == ("uint64")
    assert glsl_by_name["AxisStride"]["scalarLayout"] == {
        "physicalType": "int64_t",
        "elementType": "int64",
        "elementSizeBytes": 8,
        "elementStrideBytes": 8,
        "alignmentBytes": 16,
        "memberOffsetBytes": 0,
        "storageLayout": "std140",
        "runtimeSized": False,
        "memberName": "axisStride",
        "blockSizeBytes": 16,
    }
    assert glsl_by_name["AxisSize"]["scalarLayout"]["elementType"] == "uint64"


def test_source_reflection_does_not_guess_aggregate_or_implicit_layouts(tmp_path):
    hlsl = _reflect_hlsl(
        tmp_path,
        """
        struct Pair { uint first; uint second; };
        cbuffer Multiple : register(b0) {
            uint first;
            uint second;
        };
        RWStructuredBuffer<Pair> pairs : register(u0);
        ByteAddressBuffer bytes : register(t0);
        RWStructuredBuffer<uint64_t2> wideOffsets : register(u1);
        cbuffer Wide : register(b1) { int64_t2 wide; };
        [numthreads(1, 1, 1)] void CSMain() {}
        """,
    )
    assert all("scalarLayout" not in resource for resource in hlsl["resources"])

    artifact = tmp_path / "unsupported.comp"
    artifact.write_text(
        textwrap.dedent("""
            #version 450 core
            layout(std140, binding = 0) uniform Multiple {
                uint first;
                uint second;
            };
            layout(std430, binding = 1) buffer UnsupportedVector { dvec2 values[]; };
            layout(binding = 2) buffer Implicit { uint values[]; };
            layout(local_size_x = 1) in;
            void main() {}
            """).strip(),
        encoding="utf-8",
    )
    glsl = reflect_target_host_interface(artifact, target="opengl", stage="compute")
    assert all("scalarLayout" not in resource for resource in glsl["resources"])


def test_hlsl_reflection_excludes_malformed_function_declarations(tmp_path):
    reflection = _reflect_hlsl(
        tmp_path,
        """
        const uint sentinel = 1u;
        [shader("compute")] void MissingBody();
        [numthreads(1, 1, 1)] void MissingParen(uint3 tid {
        }
        [shader("compute"] void CSMain() {
        }
        void CSMain() : {
        }
        float CSMain = factory() {
        }
        """,
    )

    assert reflection["status"] == "ready"
    assert reflection["entryPoints"] == []
    assert reflection["constants"][0]["name"] == "sentinel"


def test_hlsl_function_scan_has_bounded_work_on_failed_declarations(monkeypatch):
    adversarial_prefix = " ".join(["GeneratedType"] * 4096)
    source = (
        f"{adversarial_prefix} candidate() : 123 {{}}\n"
        "[numthreads(8, 1, 1)] void CSMain() {}"
    )
    parsed_segments = 0
    parsed_characters = 0
    original_parser = host_reflection._parse_hlsl_function_declaration

    def counting_parser(header):
        nonlocal parsed_characters, parsed_segments
        parsed_segments += 1
        parsed_characters += len(header)
        return original_parser(header)

    monkeypatch.setattr(
        host_reflection, "_parse_hlsl_function_declaration", counting_parser
    )

    declarations = list(host_reflection._iter_hlsl_function_declarations(source))

    assert declarations == [("CSMain", "[numthreads(8, 1, 1)]")]
    assert parsed_segments == 2
    assert parsed_characters <= len(source)


def test_hlsl_reflection_distinguishes_register_namespaces(tmp_path):
    reflection = _reflect_hlsl(
        tmp_path,
        """
        cbuffer CrossGLDispatchInfo : register(b0) {
            uint3 crossglNumWorkGroups;
        };
        StructuredBuffer<float> inputValues : register(t0);
        RWStructuredBuffer<float> outputValues : register(u0);
        SamplerState linearSampler : register(s0);
        [numthreads(32, 1, 1)] void CSMain() {}
        """,
    )

    assert reflection["status"] == "ready"
    assert reflection["diagnostics"] == []


def test_hlsl_reflection_rejects_duplicate_binding_within_register_namespace(tmp_path):
    reflection = _reflect_hlsl(
        tmp_path,
        """
        StructuredBuffer<float> firstInput : register(t0);
        StructuredBuffer<float> secondInput : register(t0);
        [numthreads(1, 1, 1)] void CSMain() {}
        """,
    )

    assert reflection["status"] == "ambiguous"
    assert reflection["diagnostics"] == [
        "project.runtime-package-inspection."
        "host-interface-reflection-ambiguous-binding"
    ]
    assert reflection["diagnosticRecords"][0]["details"] == {
        "resource": "secondInput",
        "conflictingResource": "firstInput",
        "set": 0,
        "binding": 0,
        "bindingNamespace": "srv",
    }
