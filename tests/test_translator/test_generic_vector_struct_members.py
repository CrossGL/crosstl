"""Generic struct members of type ``vecN<T>`` keep their concrete vector after specialisation.

``struct VBox<T> { vec2<T> v; T s; }`` specialized for ``float`` declared ``T2 v;``: the generators spell the member
``T2`` before substituting it, so Metal, HLSL, HIP and CUDA named an undeclared type and GLSL refused the program
(issue #4198). The same spelling broke an array of generic vectors, ``vec2<T> q[2]``. Each case below fails on that
code.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl

PROGRAM = textwrap.dedent("""
    shader main {
        struct VBox<T> { vec2<T> v; T s; };
        struct IVec<T> { vec3<T> p; vec2<T> q[2]; T s; };
        buffer float* results;
        buffer int* iresults;
        compute {
        @ numthreads(1, 1, 1)
        void main() {
            VBox<float> b;
            b.v = vec2(1.0, 2.0);
            b.s = 3.0;
            results[0] = b.v.x + b.v.y + b.s;
            IVec<int> u;
            u.p = vec3(1, 2, 3);
            u.q[0] = vec2(4, 5);
        u.q[1] = vec2(6, 7);
            u.s = 6;
            iresults[0] = u.p.x + u.p.y + u.p.z + u.q[0].x + u.q[1].y + u.s;
        }
        }
    }
""").strip()


def translate(tmp_path, backend):
    path = tmp_path / "vectors.cgl"
    path.write_text(PROGRAM)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _validate(tmp_path, backend, output):
    if backend == "opengl":
        tool = shutil.which("glslangValidator")
        if tool is None:
            pytest.skip("glslangValidator is not installed")
        path = tmp_path / "out.comp"
        path.write_text(output)
        command = [tool, "-S", "comp", str(path)]
    elif backend == "metal":
        xcrun = shutil.which("xcrun")
        if xcrun is None:
            pytest.skip("xcrun is not installed")
        path = tmp_path / "out.metal"
        path.write_text(output)
        command = [xcrun, "-sdk", "macosx", "metal", "-c", str(path), "-o", "/dev/null"]
    else:
        tool = shutil.which("dxc")
        if tool is None:
            pytest.skip("dxc is not installed")
        path = tmp_path / "out.hlsl"
        path.write_text(output)
        command = [tool, "-HV", "2021", "-T", "cs_6_2", "-E", "CSMain", str(path)]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("backend", "declarations"),
    [
        (
            "metal",
            (
                "struct VBox_float {\n    float2 v;\n    float s;\n};",
                "struct IVec_int {\n    int3 p;\n    int2 q[2];\n    int s;\n};",
            ),
        ),
        (
            "directx",
            (
                "struct VBox_float {\n    float2 v;\n    float s;\n};",
                "struct IVec_int {\n    int3 p;\n    int2 q[2];\n    int s;\n};",
            ),
        ),
        (
            "opengl",
            (
                "struct VBox_float {\n    vec2 v;\n    float s;\n};",
                "struct IVec_int {\n    ivec3 p;\n    ivec2 q[2];\n    int s;\n};",
            ),
        ),
        (
            "hip",
            (
                "struct VBox_float {\n    float2 v;\n    float s;\n};",
                "struct IVec_int {\n    int3 p;\n    int2 q[2];\n    int s;\n};",
            ),
        ),
    ],
)
def test_specialized_vector_struct_members_are_concrete(
    tmp_path, backend, declarations
):
    output = translate(tmp_path, backend)

    for declaration in declarations:
        assert declaration in output, declaration
    assert "T2" not in output


@pytest.mark.parametrize("backend", ["metal", "directx", "opengl"])
def test_specialized_vector_struct_members_translate_to_valid_targets(
    tmp_path, backend
):
    _validate(tmp_path, backend, translate(tmp_path, backend))
