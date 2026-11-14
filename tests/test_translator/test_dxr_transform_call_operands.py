"""DXR transform calls used directly as matrix operands in Slang and HLSL output (#4558).

``ObjectToWorld3x4()``, ``ObjectToWorld4x3()``, ``WorldToObject3x4()``, ``WorldToObject4x3()``, ``ObjectToWorld()``
and ``WorldToObject()`` return matrices (DXR spec; the unsuffixed names are ``float3x4``). A product or an element
index on a local of that type became ``mul(...)`` and a swapped index, but the same expression on the call itself was
written verbatim: ``ObjectToWorld3x4() * p``, which slangc and DXC reject, and ``ObjectToWorld3x4()[2][1]``, which
reads the transposed element. The generators now know the calls' types, so the call converts like the local.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl

PROGRAM = textwrap.dedent("""
    shader main {
        struct Payload {
            vec4 color @ ray_payload_read(caller) @ ray_payload_write(anyhit, closesthit);
        }
        ray_closest_hit {
            void main(inout Payload payload @ payload, in BuiltInTriangleIntersectionAttributes attributes @ hit_attribute) {
                vec4 p = vec4(1.0, 2.0, 3.0, 1.0);
                mat4x3 r = ObjectToWorld3x4();
                vec3 viaLocal = r * p;
                float localElem = r[2][1];
                vec3 a = ObjectToWorld3x4() * p;
                vec3 b = WorldToObject3x4() * p;
                vec3 c = ObjectToWorld() * p;
                vec3 d = WorldToObject() * p;
                vec3 e = p * ObjectToWorld4x3();
                vec3 f = p * WorldToObject4x3();
                float g = ObjectToWorld3x4()[2][1];
                float h = WorldToObject4x3()[1][2];
                float k = WorldToObject()[3][0];
                payload.color = vec4(viaLocal + a + b + c + d + e + f, localElem + g + h + k);
            }
        }
    }
""")

# A function the program declares under a DXR name keeps its own return type.
SHADOWED = textwrap.dedent("""
    shader main {
        struct Payload {
            vec4 color @ ray_payload_read(caller) @ ray_payload_write(anyhit, closesthit);
        }
        ray_closest_hit {
            vec4 ObjectToWorld() {
                return vec4(1.0, 2.0, 3.0, 4.0);
            }
            void main(inout Payload payload @ payload, in BuiltInTriangleIntersectionAttributes attributes @ hit_attribute) {
                vec4 p = vec4(1.0, 2.0, 3.0, 1.0);
                vec4 q = ObjectToWorld() * p;
                payload.color = q;
            }
        }
    }
""")

EXPECTED = (
    "mul(r, p)",
    "r[1][2]",
    "mul(ObjectToWorld3x4(), p)",
    "mul(WorldToObject3x4(), p)",
    "mul(ObjectToWorld(), p)",
    "mul(WorldToObject(), p)",
    "mul(p, ObjectToWorld4x3())",
    "mul(p, WorldToObject4x3())",
    "(ObjectToWorld3x4())[1][2]",
    "(WorldToObject4x3())[2][1]",
    "(WorldToObject())[0][3]",
)


def translate(tmp_path, source, backend):
    path = tmp_path / "xform.cgl"
    path.write_text(source, "utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def compact(code):
    return "".join(code.split())


@pytest.mark.parametrize("backend", ["slang", "directx"])
def test_transform_calls_convert_like_a_local_of_their_type(tmp_path, backend):
    code = compact(translate(tmp_path, PROGRAM, backend))
    for text in EXPECTED:
        assert compact(text) in code, text
    for verbatim in (
        "ObjectToWorld3x4()*p",
        "WorldToObject3x4()*p",
        "ObjectToWorld()*p",
        "WorldToObject()*p",
        "p*ObjectToWorld4x3()",
        "p*WorldToObject4x3()",
        "ObjectToWorld3x4()[2][1]",
        "WorldToObject4x3()[1][2]",
        "WorldToObject()[3][0]",
    ):
        assert verbatim not in code, verbatim


@pytest.mark.parametrize("backend", ["slang", "directx"])
def test_a_declared_function_with_a_transform_name_keeps_its_type(tmp_path, backend):
    code = compact(translate(tmp_path, SHADOWED, backend))
    assert (
        "float4q=(ObjectToWorld()*p);" in code or "float4q=ObjectToWorld()*p;" in code
    )
    assert "mul(ObjectToWorld(),p)" not in code


def require_tool(name):
    path = shutil.which(name)
    if path is None:
        pytest.skip(f"{name} is not installed")
    return path


def run_tool(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr


def test_slang_output_compiles_with_slangc(tmp_path):
    slangc = require_tool("slangc")
    source = tmp_path / "xform.slang"
    source.write_text(translate(tmp_path, PROGRAM, "slang"), "utf-8")
    run_tool([slangc, str(source), "-target", "spirv", "-o", str(tmp_path / "x.spv")])


def test_hlsl_output_compiles_with_dxc(tmp_path):
    dxc = require_tool("dxc")
    source = tmp_path / "xform.hlsl"
    source.write_text(translate(tmp_path, PROGRAM, "directx"), "utf-8")
    run_tool([dxc, "-T", "lib_6_3", "-HV", "2021", str(source)])
