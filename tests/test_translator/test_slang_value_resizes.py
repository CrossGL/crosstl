"""Slang output for matrix resizes, vector truncations, structured buffer lengths, scalar cull distances and
``MaybeReorderThread``.

Each case used to be a wrong row in the Slang output coverage: slangc rejected ``float3x3(float2x2)``,
``float2(float4)`` and ``RWStructuredBuffer.length()``, a scalar ``SV_CullDistance`` compiled to a non-array
``CullDistance`` built-in that spirv-val rejects, and Slang declares HLSL's ``MaybeReorderThread`` for HLSL only.
Every output is compiled with slangc and validated with spirv-val when they are installed. The compute cases also
run on a host Vulkan runtime and must compute what GLSL 4.60 section 5.4.2 defines.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess

import pytest

import crosstl

REQUIRE_ENV = "CROSSTL_REQUIRE_VULKAN_RUNTIME"


def translate(tmp_path, source):
    path = tmp_path / "probe.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="slang")


def run_tool(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def compile_slang(tmp_path, code):
    """Compile Slang to SPIR-V with slangc and validate it; skip when slangc is missing."""
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    source, binary = tmp_path / "out.slang", tmp_path / "out.spv"
    source.write_text(code, encoding="utf-8")
    run_tool([slangc, str(source), "-target", "spirv", "-o", str(binary)])
    if shutil.which("spirv-val"):
        run_tool(["spirv-val", "--target-env", "vulkan1.2", str(binary)])
    return binary


def vulkan_runtime():
    """A host Vulkan runtime and ``vkrun``, or ``None``: lavapipe, else MoltenVK or KosmicKrisp."""
    try:
        from tests.conformance.runtimes import provision
    except ImportError:
        return None
    for runtime in ("vulkan-lavapipe", "vulkan-moltenvk", "vulkan-kosmickrisp"):
        try:
            provisioned = provision(runtime)
        except Exception:
            continue
        if provisioned.available and provisioned.mode == "host":
            env = dict(provisioned.env)
            if env.get("CROSSTL_CONFORMANCE_VKRUN"):
                return env
    return None


def run_compute(tmp_path, binary, buffers):
    """Run ``binary`` once per element of the first buffer; return each buffer's words afterwards."""
    env = vulkan_runtime()
    if env is None:
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail(
                f"{REQUIRE_ENV}=1 but no host Vulkan runtime with vkrun is available"
            )
        pytest.skip(f"no host Vulkan runtime; set {REQUIRE_ENV}=1 to require one")
    disassembled = run_tool(["spirv-dis", str(binary)]).stdout
    entry = re.search(r'OpEntryPoint GLCompute %\w+ "(\w+)"', disassembled).group(1)
    paths = []
    for index, (fmt, values) in enumerate(buffers):
        path = tmp_path / f"buffer{index}.bin"
        path.write_bytes(struct.pack(f"<{len(values)}{fmt}", *values))
        paths.append((path, fmt, len(values)))
    invocations = len(buffers[0][1])
    command = [
        env["CROSSTL_CONFORMANCE_VKRUN"],
        str(binary),
        entry,
        str(invocations),
        "1",
        "1",
        *(f"{slot}:inout:{path}" for slot, (path, _, _) in enumerate(paths)),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=120, env={**os.environ, **env}
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return [
        list(struct.unpack(f"<{count}{fmt}", path.read_bytes()))
        for path, fmt, count in paths
    ]


MATRIX_RESIZE = """shader S {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            mat3 m = mat3(fin[i], 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
            mat2 a = mat2(m);
            mat3 b = mat3(a * 2.0);
            mat4 c = mat4(m);
            mat2x3 n = mat2x3(m);
            mat3x2 t = mat3x2(a);
            outp[i] = a[0][0] + a[0][1] * 2.0 + a[1][0] * 4.0 + a[1][1] * 8.0 + b[1][1] * 16.0
                + b[2][2] * 512.0 + b[0][2] * 1024.0 + b[2][0] * 2048.0 + c[2][2] * 4096.0
                + n[1][2] * 8192.0 + c[3][3] * 65536.0 + c[3][0] * 131072.0 + c[0][3] * 262144.0
                + t[2][0] * 524288.0 + t[2][1] * 1048576.0;
        }
    }
}
"""


def matrix_resize_expected(x):
    # Columns of mat3(x, 1, ..., 8): column c holds elements 3c .. 3c+2.
    m = [[x, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]

    def resize(source, columns, rows):
        return [
            [
                (
                    source[c][r]
                    if c < len(source) and r < len(source[0])
                    else (1.0 if c == r else 0.0)
                )
                for r in range(rows)
            ]
            for c in range(columns)
        ]

    a = resize(m, 2, 2)
    b = resize([[v * 2.0 for v in column] for column in a], 3, 3)
    c = resize(m, 4, 4)
    n = resize(m, 2, 3)
    t = resize(a, 3, 2)
    return (
        a[0][0]
        + a[0][1] * 2.0
        + a[1][0] * 4.0
        + a[1][1] * 8.0
        + b[1][1] * 16.0
        + b[2][2] * 512.0
        + b[0][2] * 1024.0
        + b[2][0] * 2048.0
        + c[2][2] * 4096.0
        + n[1][2] * 8192.0
        + c[3][3] * 65536.0
        + c[3][0] * 131072.0
        + c[0][3] * 262144.0
        + t[2][0] * 524288.0
        + t[2][1] * 1048576.0
    )


def test_matrix_resize_calls_one_helper_per_shape_pair(tmp_path):
    code = translate(tmp_path, MATRIX_RESIZE)
    assert "= float2x2(m);" not in code and "= float3x3(a" not in code
    assert "float2x2 a = cgl_matrix_resize_float3x3_to_float2x2(m);" in code
    assert (
        "return float3x3(float3(m[0], 0.0), float3(m[1], 0.0), float3(0.0, 0.0, 1.0));"
        in code
    )
    assert code.count("cgl_matrix_resize_float3x3_to_float2x2(float3x3 m)") == 1
    compile_slang(tmp_path, code)


def test_matrix_resize_computes_the_glsl_result(tmp_path):
    binary = compile_slang(tmp_path, translate(tmp_path, MATRIX_RESIZE))
    inputs = [0.0, 1.0, -3.0, 9.0]
    _, outputs = run_compute(
        tmp_path, binary, [("f", inputs), ("f", [0.0] * len(inputs))]
    )
    assert outputs == [matrix_resize_expected(x) for x in inputs]


VECTOR_TRUNCATION = """shader S {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer int* iin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            vec4 v = vec4(fin[i], 2.0, 3.0, 4.0);
            vec2 a = vec2(v);
            vec3 b = vec3(v);
            ivec3 k = ivec3(iin[i], 5, 6);
            ivec2 j = ivec2(k);
            vec2 c = vec2(ivec3(iin[i] * 2, 7, 8));
            outp[i] = a.x + a.y * 4.0 + b.z * 16.0 + float(j.x + j.y * 16) * 64.0
                + (c.x + c.y * 32.0) * 8192.0;
        }
    }
}
"""


def test_vector_truncation_keeps_the_leading_components(tmp_path):
    code = translate(tmp_path, VECTOR_TRUNCATION)
    assert "float2 a = float2(v.xy);" in code
    assert "float3 b = float3(v.xyz);" in code
    assert "int2 j = int2(k.xy);" in code
    assert "float2(int3(iin[i] * 2, 7, 8).xy)" in code
    compile_slang(tmp_path, code)


def test_vector_truncation_computes_the_glsl_result(tmp_path):
    binary = compile_slang(tmp_path, translate(tmp_path, VECTOR_TRUNCATION))
    floats, ints = [0.0, 1.5, -2.0], [0, 3, -4]
    *_, outputs = run_compute(
        tmp_path,
        binary,
        [("f", floats), ("i", ints), ("f", [0.0] * len(floats))],
    )
    expected = [
        x
        + 2.0 * 4.0
        + 3.0 * 16.0
        + float(k + 5 * 16) * 64.0
        + (k * 2 + 7.0 * 32.0) * 8192.0
        for x, k in zip(floats, ints)
    ]
    assert outputs == expected


def test_a_vector_of_the_same_width_is_not_swizzled(tmp_path):
    code = translate(
        tmp_path,
        """shader S {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() {
            ivec2 k = ivec2(int(gl_GlobalInvocationID.x), 1);
            vec2 a = vec2(k);
            outp[gl_GlobalInvocationID.x] = a.x + a.y;
        }
    }
}
""",
    )
    assert "float2 a = float2(k);" in code
    compile_slang(tmp_path, code)


BUFFER_LENGTH = """shader S {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        buffer uint* counts;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = float(outp.length());
            counts[i] = uint(counts.length());
        }
    }
}
"""


def test_structured_buffer_length_is_its_element_count(tmp_path):
    code = translate(tmp_path, BUFFER_LENGTH)
    assert ".length()" not in code
    assert "float(int(cgl_bufferDimensions_RWStructuredBuffer_float(outp)))" in code
    assert "buffer.GetDimensions(count, stride);" in code
    compile_slang(tmp_path, code)


def test_structured_buffer_length_counts_the_bound_elements(tmp_path):
    binary = compile_slang(tmp_path, translate(tmp_path, BUFFER_LENGTH))
    outputs, counts = run_compute(tmp_path, binary, [("f", [0.0] * 5), ("I", [0] * 7)])
    assert outputs == [5.0] * 5
    assert counts[:5] == [7] * 5


CULL_DISTANCE = """shader S {
    struct VertexInput {
        vec4 position;
    }
    struct VertexOutput {
        vec4 position @ gl_Position;
        float cull @ gl_CullDistance;
        float clip @ gl_ClipDistance;
    }
    vertex {
        VertexOutput main(VertexInput input) {
            VertexOutput output;
            output.position = input.position * 2.0;
            output.cull = input.position.x;
            output.cull += 1.0;
            output.clip = input.position.y;
            return output;
        }
    }
}
"""


def test_a_scalar_cull_distance_is_a_one_element_array(tmp_path):
    code = translate(tmp_path, CULL_DISTANCE)
    assert "float cull[1] : SV_CullDistance;" in code
    assert "output.cull[0] = input.position.x;" in code
    assert "output.cull[0] += 1.0;" in code
    # slangc wraps a scalar clip distance in an array itself.
    assert "float clip : SV_ClipDistance;" in code
    binary = compile_slang(tmp_path, code)
    disassembled = (
        run_tool(["spirv-dis", str(binary)]).stdout if shutil.which("spirv-dis") else ""
    )
    if disassembled:
        assert re.search(r"OpDecorate %\w+ BuiltIn CullDistance", disassembled)


def test_a_scalar_cull_distance_constructor_argument_is_an_initializer_list(tmp_path):
    code = translate(
        tmp_path,
        """shader S {
    struct VertexOutput {
        vec4 position @ gl_Position;
        float cull @ gl_CullDistance;
    }
    vertex {
        VertexOutput main(vec4 position @ POSITION) {
            return VertexOutput(position, position.x);
        }
    }
}
""",
    )
    assert "VertexOutput(position, {position.x})" in code
    compile_slang(tmp_path, code)


def test_maybe_reorder_thread_is_slangs_reorder_thread(tmp_path):
    code = translate(
        tmp_path,
        """shader S {
    ray_generation {
        buffer uint* uin;
        buffer uint* outp;
        void main() {
            uint a0 = uin[0u];
            uint a1 = uin[1u];
            MaybeReorderThread(a0, a1);
            outp[0u] = a0;
        }
    }
}
""",
    )
    assert "ReorderThread(a0, a1);" in code
    assert "MaybeReorderThread" not in code
    compile_slang(tmp_path, code)
