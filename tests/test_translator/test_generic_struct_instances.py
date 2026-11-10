"""Instances of CrossGL generic structures on every target, including value parameters.

``generic<T> struct Box { T value; }`` used as ``Box<float>`` was refused by SPIR-V ("Unknown type Box<float>, using
float as default"), which declared a generic structure only when its members needed no argument; a generic function
taking ``Box<T>`` deduced nothing from a ``Box<float>`` argument on SPIR-V and WGSL; and the member ``T items[N]`` of
``Ring<float, 4>`` kept the undeclared ``N`` on GLSL, MSL and HLSL and named its WGSL structure after the literal's
``repr``. Each case below fails on that code.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from tests.spirv_friendly_names import friendly_spirv

PROGRAM = textwrap.dedent("""
    shader main {
        generic<T> struct Box {
            T value;
        }
        generic<T, N> struct Ring {
            T items[N];
            uint head;
        }
        generic<T> struct Pair {
            Box<T> first;
            T second;
        }
        generic<T> Box<T> make(T v) {
            Box<T> b;
            b.value = v;
            return b;
        }
        generic<T> T Box_twice(Box<T> self_) {
            return self_.value + self_.value;
        }
        compute {
            RWStructuredBuffer<float> f @buffer(0);
            RWStructuredBuffer<int> g @buffer(1);
            @numthreads(1, 1, 1)
            void main(uvec3 id @ gl_GlobalInvocationID) {
                float fx = f[1];
                int gx = g[1];
                Box<float> a = make(fx);
                Box<int> b = make(gx);
                Ring<float, 4> r;
                r.head = 0u;
                r.items[0] = Box_twice(a);
                Pair<float> p;
                p.first = a;
                p.second = r.items[0];
                Box<Box<int>> nested;
                nested.value = b;
                f[0] = p.first.value + p.second;
                g[0] = Box_twice(nested.value);
            }
        }
    }
""").strip()


def translate(tmp_path, backend):
    path = tmp_path / "generic_structs.cgl"
    path.write_text(PROGRAM)
    output = crosstl.translate(str(path), backend=backend, format_output=False)
    # Friendly SPIR-V ids (``%Box_float``) whether or not SPIRV-Tools formats the output.
    return friendly_spirv(output) if backend == "vulkan" else output


def _validate(tmp_path, backend, output):
    if backend == "opengl":
        tool = shutil.which("glslangValidator")
        if tool is None:
            pytest.skip("glslangValidator is not installed")
        path = tmp_path / "out.comp"
        path.write_text(output)
        command = [tool, "-S", "comp", str(path)]
    elif backend == "vulkan":
        assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
        if assembler is None or validator is None:
            pytest.skip("SPIRV-Tools are not installed")
        path, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
        path.write_text(output)
        subprocess.run(
            [assembler, "--target-env", "vulkan1.1", str(path), "-o", str(binary)],
            check=True,
            capture_output=True,
        )
        command = [validator, "--target-env", "vulkan1.1", str(binary)]
    elif backend == "wgsl":
        tool = shutil.which("naga")
        if tool is None:
            pytest.skip("naga is not installed")
        path = tmp_path / "out.wgsl"
        path.write_text(output)
        command = [tool, str(path)]
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
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", ["opengl", "metal", "directx", "vulkan", "wgsl"])
def test_generic_struct_instances_translate_to_valid_targets(tmp_path, backend):
    _validate(tmp_path, backend, translate(tmp_path, backend))


def test_spirv_declares_one_structure_per_instance(tmp_path):
    output = translate(tmp_path, "vulkan")

    assert "%Box_float = OpTypeStruct %float" in output
    assert "%Box_int = OpTypeStruct %int" in output
    assert "%Box_Box_int = OpTypeStruct %Box_int" in output
    assert "%Pair_float = OpTypeStruct %Box_float %float" in output
    assert "%Ring_float_4 = OpTypeStruct %_arr_float_int_4 %uint" in output
    assert "Unknown type" not in output


@pytest.mark.parametrize(
    ("backend", "member"),
    [
        ("opengl", "float items[4];"),
        ("metal", "float items[4];"),
        ("directx", "float items[4];"),
        ("wgsl", "items: array<f32, 4>,"),
    ],
)
def test_value_parameters_size_instance_members(tmp_path, backend, member):
    output = translate(tmp_path, backend)

    assert member in output
    assert "items[N]" not in output and "f32, N>" not in output
    assert "LiteralNode" not in output


def test_wgsl_deduces_type_parameters_from_generic_instances(tmp_path):
    output = translate(tmp_path, "wgsl")

    assert "fn Box_twice_f32(self_: Box_f32) -> f32" in output
    assert "fn Box_twice_i32(self_: Box_i32) -> i32" in output
    assert "fn make_f32(v: f32) -> Box_f32" in output


CONSTRUCTORS = textwrap.dedent("""
    shader main {
        generic<T, U> struct Pair {
            T first;
            U second;
        }
        compute {
            RWStructuredBuffer<float> f @buffer(0);
            @numthreads(1, 1, 1)
            void main(uvec3 id @ gl_GlobalInvocationID) {
                Pair<vec2, float> p = Pair<vec2, float>(vec2(f[1]), 2.0);
                Pair<int, float> q = Pair<int, float>(3, f[2]);
                f[0] = p.first.x + p.second + float(q.first) + q.second;
            }
        }
    }
""").strip()


@pytest.mark.parametrize("backend", ["opengl", "metal", "directx", "vulkan", "wgsl"])
def test_constructors_name_the_instance_their_type_declares(tmp_path, backend):
    path = tmp_path / "constructors.cgl"
    path.write_text(CONSTRUCTORS)

    output = crosstl.translate(str(path), backend=backend)

    if backend == "metal":
        # The constructor ``Pair<vec2, float>(...)`` and the type ``Pair<vec2, float>`` are one structure; the
        # constructor used to name an undeclared ``Pair_vec2_float``.
        assert output.count("struct Pair_float2_float {") == 1
        assert "Pair_float2_float p = Pair_float2_float{" in output
        assert "Pair_vec2_float" not in output
    if backend == "vulkan":
        # SPIR-V refused the constructor as a call through a callee expression.
        assert "OpCompositeConstruct" in output
    _validate(tmp_path, backend, output)
