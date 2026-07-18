import textwrap

from crosstl.project import host_reflection
from crosstl.project.host_reflection import reflect_target_host_interface


def _reflect_hlsl(tmp_path, source, *, stage=None):
    artifact = tmp_path / "kernel.hlsl"
    artifact.write_text(textwrap.dedent(source).strip(), encoding="utf-8")
    return reflect_target_host_interface(artifact, target="directx", stage=stage)


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


def test_source_reflection_does_not_guess_aggregate_or_implicit_layouts(tmp_path):
    hlsl = _reflect_hlsl(
        tmp_path,
        """
        cbuffer Multiple : register(b0) {
            uint first;
            uint second;
        };
        RWStructuredBuffer<uint2> vectors : register(u0);
        ByteAddressBuffer bytes : register(t0);
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
            layout(std430, binding = 1) buffer Vectors { uvec2 values[]; };
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
