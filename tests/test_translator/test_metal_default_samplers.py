"""A Metal ``sampler s;`` local is the default fixed sampler state, unless the function reassigns it.

The importer kept ``sampler s;`` as an uninitialized local. HLSL then declared an uninitialized
``SamplerState`` local, which DXC rejects ("local resource not guaranteed to map to unique global
resource"), and SPIR-V failed with an untyped ``ValueError``.
"""

import json
import re
import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.backend.Metal.default_samplers import is_default_sampler_local
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.translator.semantics.sampler_states import (
    FixedSamplerStateUnsupportedError,
)

DEFAULT_SAMPLER = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;
    struct V { float4 position [[position]]; float2 uv; };
    fragment float4 shade(V in [[stage_in]], texture2d<float> t [[texture(0)]]) {
        sampler s;
        return t.sample(s, in.uv);
    }
    """)

REASSIGNED_SAMPLER = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;
    fragment float4 shade(texture2d<float> t [[texture(0)]], sampler bound [[sampler(0)]],
                          float4 position [[position]]) {
        sampler s;
        if (position.x > 1.0) {
            s = bound;
        }
        return t.sample(s, position.xy);
    }
    """)


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _fixed_samplers(hlsl):
    header = re.match(r"// crosstl-fixed-samplers: (\{.*\})\n", hlsl)
    assert header, hlsl
    return json.loads(header.group(1))["samplers"]


def test_default_sampler_imports_as_fixed_default_state(tmp_path):
    code = _translate(tmp_path, DEFAULT_SAMPLER, "cgl")
    assert "const sampler s = sampler();" in code, code


def test_directx_binds_the_default_state(tmp_path):
    code = _translate(tmp_path, DEFAULT_SAMPLER, "directx")
    (sampler,) = _fixed_samplers(code)
    state = sampler["state"]
    assert state["min_filter"] == state["mag_filter"] == "nearest"
    assert state["mip_filter"] == "none"
    assert state["s_address"] == state["t_address"] == "clamp_to_edge"
    assert re.search(rf"SamplerState {sampler['name']} : register\(s0\);", code), code
    # No sampler is declared inside a function.
    assert not re.search(r"\n\s+SamplerState ", code), code
    dxc = shutil.which("dxc")
    if dxc is None:
        return
    entry = re.search(r"float4 (\w+)\([^)]*\)\s*:\s*SV_TARGET", code, re.I)
    shader = tmp_path / "shader.hlsl"
    shader.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [dxc, "-T", "ps_6_0", "-E", entry.group(1), str(shader)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_vulkan_binds_the_default_state(tmp_path):
    code = _translate(tmp_path, DEFAULT_SAMPLER, "vulkan")
    assert "OpTypeSampler" in code
    spirv_as, spirv_val = shutil.which("spirv-as"), shutil.which("spirv-val")
    if spirv_as is None or spirv_val is None:
        return
    source, binary = tmp_path / "shader.spvasm", tmp_path / "shader.spv"
    source.write_text(code, encoding="utf-8")
    subprocess.run(
        [spirv_as, "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
        check=True,
    )
    result = subprocess.run(
        [spirv_val, "--target-env", "vulkan1.1", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_opengl_refuses_the_default_state(tmp_path):
    with pytest.raises(FixedSamplerStateUnsupportedError):
        _translate(tmp_path, DEFAULT_SAMPLER, "opengl")


def test_reassigned_sampler_stays_a_variable(tmp_path):
    code = _translate(tmp_path, REASSIGNED_SAMPLER, "cgl")
    assert "sampler()" not in code, code
    assert re.search(r"\bsampler s;", code), code
    assert re.search(r"\bs = bound;", code), code


def _first_local(source):
    program = MetalParser(MetalLexer(textwrap.dedent(source)).tokenize()).parse()
    function = program.functions[0]
    return function.body[0], function.body


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("sampler s; return t.sample(s, p);", True),
        ("sampler s[2]; return t.sample(s[0], p);", False),
        ("sampler s; s = b; return t.sample(s, p);", False),
        ("sampler s; { s = b; } return t.sample(s, p);", False),
        ("float4 s; return s;", False),
    ],
    ids=["default", "array", "assigned", "assigned-in-block", "not-a-sampler"],
)
def test_default_sampler_detection(body, expected):
    declaration, function_body = _first_local(f"""
        #include <metal_stdlib>
        using namespace metal;
        float4 f(texture2d<float> t, sampler b, float2 p) {{ {body} }}
        """)
    assert is_default_sampler_local(declaration, function_body) is expected
