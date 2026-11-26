"""HLSL output of files with several CUDA, HIP or OpenCL kernels whose parameters share names.

Each kernel's parameters become program-scope declarations: a resource per buffer and a ``<kernel>_Args`` constant
buffer for the scalars. HLSL cbuffer members are global names, and a resource name is one register, so a second
kernel's ``width`` or a ``part`` bound elsewhere must not reach HLSL under the same name. A program scoped to one
entry keeps only that kernel's declarations, and a whole program renames the later kernel's.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.codegen.hlsl_entry_arguments import (
    ENTRY_ARGUMENT_ANNOTATION,
    separate_entry_argument_names,
)

SCALARS = """
__global__ void copy(float *odata, float *idata, int width, int height)
{ int i = threadIdx.x + width * blockIdx.x; odata[i] = idata[i] + height; }
__global__ void copy2(float *odata, float *idata, int width, int height)
{ int i = threadIdx.x + width * blockIdx.x; odata[i] = idata[i] * height; }
"""

BUFFERS = """
__global__ void a(unsigned int *part, const unsigned int *data)
{ part[threadIdx.x] = data[threadIdx.x]; }
__global__ void b(unsigned int *hist, const unsigned int *part)
{ hist[threadIdx.x] = part[threadIdx.x]; }
"""

SHADOWED = """
__global__ void first(float *out, int width) { out[threadIdx.x] = width; }
__global__ void second(float *out, int width)
{
    int i = threadIdx.x;
    { int width = 3; out[i] = width; }
    out[i + 1] = width;
}
"""

ENTRY = re.compile(r"void (CSMain\w*)\(")


def _translate(tmp_path, source, suffix=".cu", **options):
    path = tmp_path / f"kernels{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(
        str(path), backend="directx", format_output=False, **options
    )


def _body(code, entry):
    start = code.index(f"void {entry}(")
    end = code.find("[numthreads", start)
    return code[start : end if end != -1 else len(code)]


def _dxc(tmp_path, code, entries):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not on PATH")
    path = tmp_path / "kernels.hlsl"
    path.write_text(code, encoding="utf-8")
    sizes = [f"-DCROSSGL_WORKGROUP_SIZE_{axis}=1" for axis in "XYZ"]
    for entry in entries:
        result = subprocess.run(
            [dxc, "-T", "cs_6_0", "-E", entry, "-WX", *sizes, str(path)],
            capture_output=True,
            text=True,
            check=False,
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.parametrize("suffix", [".cu", ".hip"])
def test_whole_program_renames_the_later_kernels_scalars(tmp_path, suffix):
    code = _translate(tmp_path, SCALARS, suffix)
    assert re.search(r"cbuffer copy_Args[^{]*\{\s*int width;\s*int height;", code)
    assert re.search(
        r"cbuffer copy2_Args[^{]*\{\s*int copy2_width;\s*int copy2_height;", code
    )
    first, second = ENTRY.findall(code)
    assert "width" in _body(code, first) and "copy2_" not in _body(code, first)
    body = _body(code, second)
    assert "copy2_width" in body and "copy2_height" in body
    assert not re.search(r"(?<!copy2_)\b(width|height)\b", body)
    # Identical buffers of both kernels stay one resource.
    assert code.count("RWStructuredBuffer<float> odata") == 1
    _dxc(tmp_path, code, (first, second))


@pytest.mark.parametrize("entry", ["copy", "copy2"])
def test_one_entry_keeps_only_its_own_scalars(tmp_path, entry):
    code = _translate(tmp_path, SCALARS, entry_point=entry)
    assert f"cbuffer {entry}_Args" in code
    other = "copy2" if entry == "copy" else "copy"
    assert f"{other}_Args" not in code
    assert re.search(r"\{\s*int width;\s*int height;", code)
    assert len(ENTRY.findall(code)) == 1
    _dxc(tmp_path, code, ENTRY.findall(code))


def test_whole_program_renames_a_buffer_bound_elsewhere(tmp_path):
    code = _translate(tmp_path, BUFFERS)
    names = re.findall(r"RWStructuredBuffer<uint> (\w+) : register\((u\d+)\)", code)
    assert [name for name, _register in names] == ["part", "data", "hist", "b_part"]
    assert len({register for _name, register in names}) == 4
    first, second = ENTRY.findall(code)
    assert "part[" in _body(code, first) and "b_part" not in _body(code, first)
    assert "b_part[" in _body(code, second)
    assert not re.search(r"(?<!b_)\bpart\b", _body(code, second))
    _dxc(tmp_path, code, (first, second))


@pytest.mark.parametrize(
    ("entry", "expected"),
    [("a", [("part", "u0"), ("data", "u1")]), ("b", [("hist", "u0"), ("part", "u1")])],
)
def test_one_entry_keeps_its_own_buffers_and_bindings(tmp_path, entry, expected):
    code = _translate(tmp_path, BUFFERS, entry_point=entry)
    names = re.findall(r"RWStructuredBuffer<uint> (\w+) : register\((u\d+)\)", code)
    assert names == expected
    _dxc(tmp_path, code, ENTRY.findall(code))


def test_a_local_that_shadows_a_renamed_scalar_keeps_its_name(tmp_path):
    code = _translate(tmp_path, SHADOWED)
    _first, second = ENTRY.findall(code)
    body = _body(code, second)
    assert re.search(r"int width = 3;\s*\w+\[i\] = width;", body)
    assert re.search(r"\w+\[\(i \+ 1\)\] = second_width;", body)
    _dxc(tmp_path, code, ENTRY.findall(code))


def _hoisted_scalar_block(kernel, *names):
    block = A.StructNode(
        f"{kernel}_Args",
        [A.StructMemberNode(name, A.PrimitiveType("int")) for name in names],
    )
    block.is_cbuffer = True
    block.add_annotation(ENTRY_ARGUMENT_ANNOTATION, kernel)
    return block


def test_the_new_name_avoids_names_in_use():
    program = A.ShaderNode(
        "kernels",
        A.ExecutionModel.GRAPHICS_PIPELINE,
        global_variables=[A.VariableNode("second_n", A.PrimitiveType("int"))],
        cbuffers=[
            _hoisted_scalar_block("first", "n"),
            _hoisted_scalar_block("second", "n"),
        ],
    )
    renamed = separate_entry_argument_names(program)
    assert renamed is not program
    assert [member.name for member in renamed.cbuffers[1].members] == ["second_n_2"]
    # The input program is left as it was.
    assert [member.name for member in program.cbuffers[1].members] == ["n"]


def test_programs_without_hoisted_arguments_are_returned_unchanged():
    program = A.ShaderNode(
        "kernels",
        A.ExecutionModel.GRAPHICS_PIPELINE,
        cbuffers=[_hoisted_scalar_block("first", "n")],
    )
    program.cbuffers[0].annotations.pop(ENTRY_ARGUMENT_ANNOTATION)
    assert separate_entry_argument_names(program) is program
    program.cbuffers[0].add_annotation(ENTRY_ARGUMENT_ANNOTATION, "first")
    assert separate_entry_argument_names(program) is program
