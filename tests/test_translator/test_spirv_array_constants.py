"""SPIR-V constant composites of array and structure type, and the list assignments they import as.

glslang writes ``float v[4] = float[4](1.0, 2.0, 4.0, 8.0)`` as an ``OpConstantComposite`` of array type stored
into a function variable. The importer used to write it as a call to an unknown ``spirv_constant_composite``,
and structure constants the same way, because neither type has a constructor name when constants are read.
Arrays now import as initializer lists and structures as their constructors, which every generator lowers; Metal
stores a list assigned to an array element by element.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

import crosstl
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "conformance"
    / "cases"
    / "s_array_constants"
    / "src.spvasm"
)


def _run(command, cwd):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _translate(backend, path=FIXTURE):
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _compile(tmp_path, backend, code):
    if backend == "opengl" and shutil.which("glslangValidator"):
        (tmp_path / "out.comp").write_text(code)
        _run(["glslangValidator", "-S", "comp", "out.comp"], tmp_path)
    elif backend == "directx" and shutil.which("dxc"):
        (tmp_path / "out.hlsl").write_text(code)
        _run(["dxc", "-T", "cs_6_0", "-E", "CSMain", "out.hlsl"], tmp_path)
    elif backend == "metal" and shutil.which("xcrun"):
        (tmp_path / "out.metal").write_text(code)
        _run(
            ["xcrun", "-sdk", "macosx", "metal", "-c", "out.metal", "-o", "out.air"],
            tmp_path,
        )
    elif backend == "vulkan" and shutil.which("spirv-as") and shutil.which("spirv-val"):
        (tmp_path / "out.spvasm").write_text(code)
        _run(["spirv-as", "out.spvasm", "-o", "out.spv"], tmp_path)
        _run(["spirv-val", "--target-env", "vulkan1.1", "out.spv"], tmp_path)


def test_array_and_structure_constants_import_as_crossgl_values():
    crossgl = _translate("cgl")
    assert "spirv_constant_composite" not in crossgl
    assert "{1.0, 2.0, 4.0, 8.0}" in crossgl
    assert "{Entry(1.5, 3), Entry(-2.5, 7)}" in crossgl
    assert "{{float2(1.0, 0.5), float2(2.0, 0.25)}, {float2(3.0, 4.0)" in crossgl


@pytest.mark.parametrize("backend", ["opengl", "directx", "metal", "vulkan"])
def test_imported_array_constants_reach_every_target(tmp_path, backend):
    code = _translate(backend)
    assert "spirv_constant_composite" not in code
    _compile(tmp_path, backend, code)


def test_metal_stores_an_assigned_list_element_by_element():
    metal = _translate("metal")
    for line in (
        "weights[0] = 1.0;",
        "weights[3] = 8.0;",
        "table[1] = Entry{-2.5, 7};",
        "grid[1][0] = float2(3.0, 4.0);",
    ):
        assert line in metal


SWAP = """
shader Swap {
    compute {
        layout(local_size_x = 1) in;
        buffer float data[];
        void main() {
            float v[2];
            v = {data[0], data[1]};
            v = {v[1], v[0]};
            data[0] = v[0];
            data[1] = v[1];
        }
    }
}
"""


def test_metal_refuses_a_list_that_reads_the_assigned_array(tmp_path):
    # v = {v[1], v[0]} evaluates both elements first; storing v[0] first would make v[1] read the new v[0].
    path = tmp_path / "swap.cgl"
    path.write_text(SWAP)
    with pytest.raises(MetalLoweringRefused, match="reads the assigned array"):
        _translate("metal", path)


def test_metal_assigns_a_list_that_reads_other_arrays(tmp_path):
    path = tmp_path / "copy.cgl"
    path.write_text(SWAP.replace("v = {v[1], v[0]};", ""))
    metal = _translate("metal", path)
    assert "v[0] = data[0];" in metal and "v[1] = data[1];" in metal
