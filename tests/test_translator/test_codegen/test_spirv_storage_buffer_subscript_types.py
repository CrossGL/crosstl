"""SPIR-V types a structured-buffer element read as a value of the element type.

``bump(q[i])`` with ``RWStructuredBuffer<uint> q`` deduces ``T = uint``, so it calls
``@explicit_specialization(bump, uint)`` when the program declares one, and an instantiation of the
primary for ``uint`` otherwise. The generator typed the subscript as nothing while it collected
specializations, and as the whole buffer while it emitted calls, so it either refused the call as an
unspecialized generic or called a ``bump_RWStructuredBuffer_uint`` it never defined.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from tests.spirv_friendly_names import friendly_spirv

KERNEL = """shader S {{
    struct P {{ float v; vec2 w; }}
{helpers}
    compute {{
{globals}
        @ stage_entry
        void k({params}uint id @gl_GlobalInvocationID) {{
{body}
        }}
    }}
}}
"""

BUMP = """
    generic<T> T bump(T x) { return x + T(1); }
    uint bump_u32(uint x) @explicit_specialization(bump, uint) { return x + 5u; }
"""

TWICE = """
    generic<T> T twice(T x) { return x + x; }
"""


def validate(tmp_path, output):
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if assembler is None or validator is None:
        return
    source, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
    source.write_text(output)
    for command in (
        [assembler, "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
        [validator, "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr


def translate(tmp_path, source, suffix):
    path = tmp_path / f"kernel{suffix}"
    path.write_text(source)
    # Friendly ids (``%uint``) whether or not SPIRV-Tools formats the output.
    output = friendly_spirv(
        crosstl.translate(str(path), backend="vulkan", format_output=False)
    )
    validate(tmp_path, output)
    return output


def translate_kernel(tmp_path, helpers, body, params="", globals_=""):
    source = KERNEL.format(helpers=helpers, globals=globals_, params=params, body=body)
    return translate(tmp_path, source, ".cgl")


def function_ids(output):
    return re.findall(r"^\s*(%\w+) = OpFunction ", output, re.MULTILINE)


def called_ids(output):
    return re.findall(r"= OpFunctionCall %\w+ (%\w+)", output)


def function_containing(output, instruction):
    """The result id of the function whose body contains ``instruction``."""
    current = None
    for line in output.splitlines():
        match = re.match(r"\s*(%\w+) = OpFunction ", line)
        if match:
            current = match.group(1)
        elif instruction in line:
            return current
    return None


@pytest.mark.parametrize(
    "body",
    [
        "            q[id] = bump(q[id]);",
        "            uint v = bump(q[id + 1u]);\n            q[id] = v;",
    ],
)
def test_structured_buffer_element_selects_explicit_specialization(tmp_path, body):
    output = translate_kernel(
        tmp_path,
        BUMP,
        body,
        params="RWStructuredBuffer<uint> q @buffer(0), ",
    )

    specialization = function_containing(output, "OpIAdd %uint %x %uint_5")
    assert specialization is not None
    assert called_ids(output) == [specialization]
    assert "RWStructuredBuffer" not in output


def test_structured_buffer_elements_instantiate_primary_per_element_type(tmp_path):
    output = translate_kernel(
        tmp_path,
        TWICE,
        "\n".join(
            [
                "            f[id] = twice(f[id]);",
                "            g[id] = twice(g[id]);",
                "            ps[id].v = twice(ps[id].v);",
                "            ps[id].w = twice(ps[id].w);",
            ]
        ),
        params=(
            "RWStructuredBuffer<float> f @buffer(0), "
            "RWStructuredBuffer<vec2> g @buffer(1), "
            "RWStructuredBuffer<P> ps @buffer(2), "
        ),
    )

    scalar = function_containing(output, "= OpFAdd %float %x")
    vector = function_containing(output, "= OpFAdd %v2float %x")
    assert scalar is not None and vector is not None and scalar != vector
    assert called_ids(output) == [scalar, vector, scalar, vector]
    assert len(function_ids(output)) == 3


def test_element_of_buffer_array_selects_explicit_specialization(tmp_path):
    output = translate_kernel(
        tmp_path,
        BUMP,
        "            bufs[1][id] = bump(bufs[0][id]);",
        globals_="        RWStructuredBuffer<uint> bufs[2] @buffer(0);",
    )

    specialization = function_containing(output, "OpIAdd %uint %x %uint_5")
    assert specialization is not None
    assert called_ids(output) == [specialization]


def test_metal_device_pointer_elements_instantiate_template(tmp_path):
    source = """#include <metal_stdlib>
using namespace metal;
template <typename T> T twice(T x) { return x + x; }
kernel void k(device float* p [[buffer(0)]], device uint* q [[buffer(1)]],
              uint id [[thread_position_in_grid]]) {
    p[id] = twice(p[id]);
    q[id] = twice(q[id]);
}
"""
    output = translate(tmp_path, source, ".metal")

    scalar = function_containing(output, "= OpFAdd %float %x")
    unsigned = function_containing(output, "= OpIAdd %uint %x")
    assert scalar is not None and unsigned is not None and scalar != unsigned
    assert called_ids(output) == [scalar, unsigned]
