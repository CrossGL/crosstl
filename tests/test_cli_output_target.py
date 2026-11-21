"""``crosstl FILE -o OUT`` takes the target language from the output file's extension."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from crosstl._crosstl import main
from crosstl.output_target import target_for_output
from crosstl.translator.refusals import RegistryRefusal

ROOT = Path(__file__).resolve().parents[1]

METAL_KERNEL = """#include <metal_stdlib>
using namespace metal;
kernel void add(device float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                uint i [[thread_position_in_grid]]) {
    a[i] = a[i] + b[i];
}
"""


@pytest.fixture
def kernel(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("shader.metal").write_text(METAL_KERNEL)
    return tmp_path


@pytest.mark.parametrize(
    "output, backend",
    [
        ("shader.hlsl", "directx"),
        ("shader.metal", "metal"),
        ("shader.glsl", "opengl"),
        ("out/Shader.WGSL", "wgsl"),
        ("shader.spvasm", "vulkan"),
        ("kernel.cu", "cuda"),
        ("kernel.hip", "hip"),
        ("shader.slang", "slang"),
        ("kernel.rs", "rust"),
        ("kernel.mojo", "mojo"),
        # A generator's compound extension wins over the GLSL one it ends with.
        ("shader.webgl.glsl", "webgl"),
        ("my.shader.hlsl", "directx"),
        # Extensions only the source frontends list name their language when CrossTL can write it.
        ("shader.frag", "opengl"),
        ("shader.comp", "opengl"),
        ("shader.msl", "metal"),
        ("shader.hlsli", "directx"),
        ("shader.cgl", "cgl"),
    ],
)
def test_output_extension_names_the_target(output, backend):
    assert target_for_output(output) == backend


@pytest.mark.parametrize("output", ["shader.spv", "shader.txt", "shader", "hlsl"])
def test_an_output_name_without_a_target_extension_is_refused(output):
    with pytest.raises(RegistryRefusal, match="Cannot tell the target language"):
        target_for_output(output)


def test_an_output_language_crosstl_only_reads_is_refused():
    with pytest.raises(RegistryRefusal, match="reads opencl sources but has no opencl"):
        target_for_output("kernel.cl")


def test_metal_to_hlsl_from_the_output_extension(kernel, capsys):
    assert main(["shader.metal", "-o", "shader.hlsl"]) == 0
    hlsl = Path("shader.hlsl").read_text()
    assert "RWStructuredBuffer<float> a" in hlsl
    assert "[numthreads(" in hlsl
    assert "shader main {" not in hlsl
    assert main(["translate", "shader.metal", "-o", "shader.wgsl"]) == 0
    assert "@compute" in Path("shader.wgsl").read_text()


def test_an_explicit_backend_wins_over_the_output_extension(kernel):
    assert main(["shader.metal", "--backend", "cgl", "-o", "shader.hlsl"]) == 0
    assert "shader main {" in Path("shader.hlsl").read_text()


def test_without_an_output_file_crossgl_is_written(kernel, capsys):
    assert main(["shader.metal"]) == 0
    assert "shader main {" in Path("shader.cgl").read_text()
    assert main(["shader.metal", "-o", "-"]) == 0
    assert "shader main {" in capsys.readouterr().out


def test_an_unknown_output_extension_fails_without_writing(kernel, capsys):
    assert main(["shader.metal", "-o", "shader.spv"]) == 1
    assert "Cannot tell the target language from the output file name 'shader.spv'" in (
        capsys.readouterr().out
    )
    assert not Path("shader.spv").exists()


def test_console_command_translates_metal_to_hlsl(kernel):
    result = subprocess.run(
        [sys.executable, "-m", "crosstl", "shader.metal", "-o", "shader.hlsl"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[numthreads(" in Path("shader.hlsl").read_text()
