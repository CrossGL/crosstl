"""HLSL interlocked calls reach OpenGL, Metal and SPIR-V in the value-returning atomic forms.

``InterlockedAdd(dest, value, original)`` returns the previous value through its last argument, and
``InterlockedCompareStore`` returns nothing. The HLSL importer kept those argument lists for destinations other than
structured-buffer elements, which OpenGL printed unchanged (glslang rejects them) and Metal and SPIR-V refused. SPIR-V
also refused every atomic on group-shared memory, which Vulkan allows at workgroup scope.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics.atomic_forms import canonicalize_atomic_calls
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

HLSL = """
RWStructuredBuffer<uint> c : register(u0);
RWStructuredBuffer<int> d : register(u1);
RWStructuredBuffer<uint3> v : register(u2);
groupshared uint g;
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint o1, o2, o4, o5, o6, o7, o8, o9, o10;
    int o3;
    if (id.x == 0) { g = 0; }
    GroupMemoryBarrierWithGroupSync();
    InterlockedAdd(c[0], 1u, o1);
    InterlockedMin(c[1], id.x, o2);
    InterlockedMax(d[2], -1, o3);
    InterlockedAnd(c[3], 7u, o4);
    InterlockedOr(c[4], 8u, o5);
    InterlockedXor(c[5], 9u, o6);
    InterlockedExchange(c[6], 10u, o7);
    InterlockedCompareExchange(c[7], 0u, 11u, o8);
    InterlockedCompareStore(c[8], 0u, 12u);
    InterlockedAdd(g, 1u, o9);
    InterlockedAdd(v[id.x].z, 1u, o10);
    InterlockedAdd(c[9], 1u);
    c[10] = o1 + o2 + uint(o3) + o4 + o5 + o6 + o7 + o8 + o9 + o10;
}
"""


def _translate(tmp_path, backend, source=HLSL, name="shader.hlsl"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_opengl_receives_value_returning_atomics_and_compiles(tmp_path):
    code = re.sub(r"\s+", " ", _translate(tmp_path, "opengl"))
    assert "o9 = atomicAdd(g, 1u);" in code
    assert re.search(r"o10 = atomicAdd\(v\[[^]]+\]\.z, 1u\);", code)
    assert "atomicCompSwap(c[8], 0u, 12u);" in code
    assert "o8 = atomicCompSwap(c[7], 0u, 11u);" in code
    assert "Interlocked" not in code
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not on PATH")
    path = tmp_path / "shader.comp"
    path.write_text(_translate(tmp_path, "opengl"), encoding="utf-8")
    _run(["glslangValidator", "-S", "comp", str(path)])


def test_metal_output_compiles(tmp_path):
    code = _translate(tmp_path, "metal")
    assert "Interlocked" not in code
    if not shutil.which("xcrun"):
        pytest.skip("xcrun is not on PATH")
    path = tmp_path / "shader.metal"
    path.write_text(code, encoding="utf-8")
    _run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(path), "-o", str(path) + ".air"]
    )


def test_hlsl_output_keeps_the_interlocked_argument_lists(tmp_path):
    code = re.sub(r"\s+", " ", _translate(tmp_path, "directx"))
    assert "InterlockedAdd(g, 1u, o9);" in code
    assert "InterlockedCompareStore(c[8], 0u, 12u);" in code


GLSL_SHARED = """#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer B { uint v[]; };
shared uint g;
shared int h;
void main() {
    if (gl_LocalInvocationIndex == 0u) { g = 0u; h = 0; }
    barrier();
    uint o = atomicAdd(g, 1u);
    int m = atomicMax(h, int(gl_LocalInvocationIndex));
    uint s = atomicCompSwap(g, 64u, 0u);
    barrier();
    v[gl_GlobalInvocationID.x] = o + s + uint(m);
}
"""


def test_spirv_lowers_group_shared_atomics_at_workgroup_scope(tmp_path):
    assembly = _translate(tmp_path, "vulkan", GLSL_SHARED, "shader.comp")
    constants = dict(re.findall(r"(%\w+) = OpConstant %\w+ (\d+)", assembly))
    atomics = re.findall(r"= (OpAtomic\w+) %\w+ (%\w+) (%\w+) ", assembly)
    assert sorted(opcode for opcode, _pointer, _scope in atomics) == [
        "OpAtomicCompareExchange",
        "OpAtomicIAdd",
        "OpAtomicSMax",
    ]
    # Scope 2 is Workgroup (SPIR-V 1.6 section 3.27).
    assert {constants[scope] for _opcode, _pointer, scope in atomics} == {"2"}
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    path = tmp_path / "shader.spvasm"
    path.write_text(assembly, encoding="utf-8")
    binary = tmp_path / "shader.spv"
    _run(["spirv-as", "--target-env", "vulkan1.1", str(path), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


def test_hlsl_group_shared_atomics_validate_as_spirv(tmp_path):
    source = HLSL.replace(
        "    InterlockedAdd(v[id.x].z, 1u, o10);\n", "    o10 = 0u;\n"
    )
    assembly = _translate(tmp_path, "vulkan", source)
    assert "Interlocked" not in assembly
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    path = tmp_path / "shader.spvasm"
    path.write_text(assembly, encoding="utf-8")
    binary = tmp_path / "shader.spv"
    _run(["spirv-as", "--target-env", "vulkan1.1", str(path), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


def _program(text):
    register_default_sources()
    return SOURCE_REGISTRY.get("cgl").parse(text)


def test_a_program_function_named_like_an_atomic_is_left_alone():
    program = _program("""
        shader main {
            void atomicAdd(uint a, uint b, out uint c) { c = a + b; }
            compute {
                void main() {
                    uint r;
                    atomicAdd(1u, 2u, r);
                }
            }
        }
        """)
    assert canonicalize_atomic_calls(program, "opengl") == 0


def test_other_targets_are_not_rewritten():
    program = _program("""
        shader main {
            compute {
                void main() {
                    uint r;
                    atomicAdd(g, 2u, r);
                }
            }
        }
        """)
    assert canonicalize_atomic_calls(program, "directx") == 0
    assert canonicalize_atomic_calls(program, "opengl") == 1


def test_hlsl_writes_group_shared_atomics_as_interlocked_calls(tmp_path):
    code = re.sub(
        r"\s+", " ", _translate(tmp_path, "directx", GLSL_SHARED, "shader.comp")
    )
    assert "uint o; InterlockedAdd(g, 1u, o);" in code
    assert re.search(r"int m; InterlockedMax\(h, int\(\w+\), m\);", code)
    assert "uint s; InterlockedCompareExchange(g, 64u, 0u, s);" in code
    assert "atomic" not in code
    dxc = shutil.which("dxc")
    if not dxc:
        pytest.skip("dxc is not on PATH")
    path = tmp_path / "shader.hlsl"
    rendered = _translate(tmp_path, "directx", GLSL_SHARED, "shader.comp")
    path.write_text(rendered, encoding="utf-8")
    entry = re.search(r"\]\s*void\s+(\w+)\s*\(", rendered).group(1)
    _run([dxc, "-T", "cs_6_0", "-E", entry, str(path), "-Fo", str(tmp_path / "o")])


def test_hlsl_group_shared_atomics_from_hlsl_take_interlocked_names(tmp_path):
    source = HLSL.replace(
        "InterlockedAdd(g, 1u, o9);",
        "InterlockedAdd(g, 1u, o9); InterlockedMax(g, 3u);",
    )
    code = re.sub(r"\s+", " ", _translate(tmp_path, "directx", source))
    assert "InterlockedAdd(g, 1u, o9);" in code
    assert "InterlockedMax(g, 3u);" in code
