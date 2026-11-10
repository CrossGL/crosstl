"""Metal functions, parameters and locals named ``read``, ``write`` or ``read_write`` (#2144).

The lexer gives these names the access-mode tokens of ``access::read``; the parser takes them as names wherever a
name belongs and keeps them as access modes inside ``texture2d<T, access::...>``. Every output must compile with the
target's own compiler, and the helpers keep their own bodies (none of them is the identity).
"""

import re
import shutil
import subprocess

import pytest

import crosstl

ISSUE_SOURCE = """
#include <metal_stdlib>
using namespace metal;
struct Wrapper { uchar payload; };
namespace wrappers {
uint read(Wrapper value) { return uint(value.payload) + 3u; }
}
kernel void test(const device uchar* values [[buffer(0)]],
                 device uint* output [[buffer(1)]],
                 uint i [[thread_position_in_grid]]) {
    output[i] = wrappers::read(((const device Wrapper*)values)[i]);
}
"""

CONTEXTS_SOURCE = """
#include <metal_stdlib>
using namespace metal;

float read(float v);
float read(float v) { return v + 1.0; }
float read(float v, float s) { return v * s; }
float write(float v) { return v * 2.0; }
float read_write(float v) { return read(v) + write(v); }

namespace io {
float read(float v) { return v - 3.0; }
}

struct Pair { float read; float write; };

float scale(float read, float write) { return read * write; }

kernel void k(texture2d<float, access::read> src [[texture(0)]],
              device float* out [[buffer(0)]],
              uint2 gid [[thread_position_in_grid]]) {
    float4 c = src.read(gid);
    float read_value = read(c.x) + read(c.y, 2.0) + write(c.z) + read_write(c.w);
    Pair p;
    p.read = io::read(read_value);
    p.write = scale(p.read, 0.5);
    out[gid.x] = p.read + p.write;
}
"""

DXC_ENTRY = re.compile(r"\[numthreads\([^)]*\)\]\s*void\s+(\w+)\s*\(")


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.metal"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _compile(tmp_path, backend, code):
    """Compile ``code`` with the target's compiler when it is installed."""
    if backend == "metal":
        if shutil.which("xcrun") is None:
            pytest.skip("xcrun is not installed")
        path = tmp_path / "generated.metal"
        command = ["xcrun", "-sdk", "macosx", "metal", "-c", str(path)]
        command += ["-o", str(tmp_path / "generated.air")]
    elif backend == "opengl":
        compiler = shutil.which("glslangValidator")
        if compiler is None:
            pytest.skip("glslangValidator is not installed")
        path = tmp_path / "generated.comp"
        command = [compiler, "-S", "comp", str(path)]
    else:
        compiler = shutil.which("dxc")
        if compiler is None:
            pytest.skip("dxc is not installed")
        entry = DXC_ENTRY.search(code)
        assert entry, code
        path = tmp_path / "generated.hlsl"
        command = [compiler, "-T", "cs_6_0", "-E", entry.group(1), "-HV", "2021"]
        command += [str(path), "-Fo", str(tmp_path / "generated.dxil")]
    path.write_text(code)
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", ["metal", "opengl", "directx"])
def test_namespaced_read_function_from_the_issue_compiles(tmp_path, backend):
    code = _translate(tmp_path, ISSUE_SOURCE, backend)
    assert "uint read(Wrapper value)" in code
    assert re.search(r"\bread\(_crosstl_metal_load_value_Wrapper\(", code)
    _compile(tmp_path, backend, code)


@pytest.mark.parametrize("backend", ["metal", "opengl", "directx"])
def test_access_names_as_functions_parameters_and_members_compile(tmp_path, backend):
    code = _translate(tmp_path, CONTEXTS_SOURCE, backend)
    assert re.search(r"float scale\(float read, float write\)", code)
    assert re.search(r"\bfloat read;", code) and re.search(r"\bfloat write;", code)
    _compile(tmp_path, backend, code)


def test_access_names_keep_their_overloads_and_namespaces(tmp_path):
    # Each overload and the namespaced function keep their own body: none is
    # merged into another or dropped, and the prototype adds no definition.
    cgl = _translate(tmp_path, CONTEXTS_SOURCE, "cgl")
    definitions = re.findall(r"float (\w+)\(float v[^)]*\) \{\s*return ([^;]+);", cgl)
    bodies = sorted(body for _, body in definitions)
    for body in ("v + 1.0", "v * s", "v * 2.0", "v - 3.0"):
        assert bodies.count(body) == 1, cgl
    assert len(definitions) == 5, cgl
