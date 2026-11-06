"""Metal free functions of one name declared in several namespaces keep their identity in flattened CrossGL."""

import re
import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.namespaced_functions import (
    declared_in_several_namespaces,
    nominated_namespaces_in_scope,
)

HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

SAME_SIGNATURE = HEADER + """
namespace a { inline float f(float x) { return x + 1.0f; } }
namespace b { inline float f(float x) { return x * 2.0f; } }
inline float f(float x) { return x - 3.0f; }
namespace b { inline float g(float x) { return f(x) + a::f(x) + ::f(x); } }
kernel void k(device float* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    float v = float(tid);
    out[tid] = a::f(v) + b::f(v) + f(v) + b::g(v);
}
"""

# ``a::f`` hides ``::f`` inside ``a`` although ``f(int)`` matches ``f(1)`` better. In ``b`` the using-directive
# makes ``a::f`` appear in the global namespace next to ``::f``, so ``f(2)`` takes the exact match ``::f(int)``.
HIDING = HEADER + """
inline float f(int x) { return float(x) * 100.0f; }
namespace a {
inline float f(float x) { return x + 0.5f; }
inline float g() { return f(1); }
}
namespace b {
using namespace a;
inline float h() { return f(2); }
}
kernel void k(device float* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = a::g() + f(3) + b::h();
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "unit.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _definition_name(crossgl, signature, body):
    match = re.search(
        rf"float (\w+)\({re.escape(signature)}\) @metal_inline \{{\s*return {re.escape(body)};",
        crossgl,
    )
    assert match, crossgl
    return match.group(1)


def _body(crossgl, function):
    match = re.search(
        rf"float {function}\([^)]*\) @metal_inline \{{\s*return ([^;]*);", crossgl
    )
    assert match, crossgl
    return match.group(1)


def _run(tool, *args):
    executable = shutil.which(tool)
    if executable is None:
        pytest.skip(f"{tool} is not installed")
    result = subprocess.run([executable, *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _run_metal_kernel(source, count):
    metal = pytest.importorskip("Metal")
    device = metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    library, error = device.newLibraryWithSource_options_error_(source, None, None)
    assert library is not None, error
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_("k"), None
    )
    assert pipeline is not None, error
    buffer = device.newBufferWithLength_options_(4 * count, 0)
    commands = device.newCommandQueue().commandBuffer()
    encoder = commands.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(buffer, 0, 0)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        metal.MTLSizeMake(count, 1, 1), metal.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    return struct.unpack(f"{count}f", buffer.contents().as_buffer(4 * count))


def test_same_signature_functions_get_distinct_names_and_bind_by_lookup(tmp_path):
    crossgl = _translate(tmp_path, SAME_SIGNATURE, "cgl")
    in_a = _definition_name(crossgl, "float x", "x + 1.0f")
    in_b = _definition_name(crossgl, "float x", "x * 2.0f")
    in_global = _definition_name(crossgl, "float x", "x - 3.0f")
    assert len({in_a, in_b, in_global}) == 3
    assert _body(crossgl, "g") == f"{in_b}(x) + {in_a}(x) + {in_global}(x)"
    assert f"{in_a}(v) + {in_b}(v) + {in_global}(v) + g(v)" in crossgl


def test_same_signature_functions_compile_on_every_target(tmp_path):
    shader = tmp_path / "unit.comp"
    shader.write_text(_translate(tmp_path, SAME_SIGNATURE, "opengl"), encoding="utf-8")
    _run("glslangValidator", "-S", "comp", str(shader))
    assembly = tmp_path / "unit.spvasm"
    assembly.write_text(
        _translate(tmp_path, SAME_SIGNATURE, "vulkan"), encoding="utf-8"
    )
    binary = tmp_path / "unit.spv"
    _run("spirv-as", "--target-env", "vulkan1.1", str(assembly), "-o", str(binary))
    _run("spirv-val", "--target-env", "vulkan1.1", str(binary))
    hlsl = _translate(tmp_path, SAME_SIGNATURE, "directx")
    definitions = re.findall(r"^float (\w+)\(float x\) \{", hlsl, re.MULTILINE)
    assert len(set(definitions) - {"g"}) == 3


def test_namespace_member_hides_better_global_overload(tmp_path):
    crossgl = _translate(tmp_path, HIDING, "cgl")
    in_a = _definition_name(crossgl, "float x", "x + 0.5f")
    in_global = _definition_name(crossgl, "int x", "float(x) * 100.0f")
    assert in_a != in_global
    assert _body(crossgl, "g") == f"{in_a}(1)"
    assert _body(crossgl, "h") == f"{in_global}(2)"
    assert f"g() + {in_global}(3) + h()" in crossgl


@pytest.mark.parametrize(
    "source", [SAME_SIGNATURE, HIDING], ids=["same-signature", "hiding"]
)
def test_metal_round_trip_computes_the_source_results(tmp_path, source):
    expected = _run_metal_kernel(source, 4)
    assert _run_metal_kernel(_translate(tmp_path, source, "metal"), 4) == expected


def test_declared_in_several_namespaces():
    class Function:
        def __init__(self, namespace):
            self.namespace = namespace

    assert declared_in_several_namespaces([Function("a"), Function("")])
    assert not declared_in_several_namespaces([Function("a"), Function("a")])
    assert not declared_in_several_namespaces([Function(None), Function("")])


def test_using_directive_members_appear_in_nearest_common_namespace():
    directives = {
        "outer::b": ["outer::a"],
        "outer::a": ["c"],
        "": ["metal"],
    }

    def nominated(scope):
        return directives.get(scope, [])

    # From ``outer::b``: ``outer::a`` appears in ``outer``; ``c`` (nominated transitively) and ``metal`` globally.
    assert nominated_namespaces_in_scope("outer::b", "outer::b", nominated) == []
    assert nominated_namespaces_in_scope("outer::b", "outer", nominated) == ["outer::a"]
    assert nominated_namespaces_in_scope("outer::b", "", nominated) == ["c", "metal"]
    # A directive of ``outer::b`` plays no part in lookups from other namespaces.
    assert nominated_namespaces_in_scope("outer", "outer", nominated) == []
    assert nominated_namespaces_in_scope("", "", nominated) == ["metal"]
