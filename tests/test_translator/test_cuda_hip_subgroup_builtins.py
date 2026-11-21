"""GLSL subgroup built-ins and reserved ``gl_`` names in CUDA and HIP output.

Warps are formed from consecutive linear thread IDs, so the lane, subgroup ID and subgroup count read all three
block dimensions. The expressions are evaluated as host C++ over a two-dimensional block, where a lane taken from
``threadIdx.x`` alone would be wrong. A ``gl_`` built-in the generators do not map is refused.
"""

import shutil
import subprocess

import pytest

from crosstl.translator.codegen.cuda_codegen import CudaCodeGen
from crosstl.translator.codegen.cuda_refusals import CudaLoweringRefused
from crosstl.translator.codegen.gpu_subgroup_builtins import (
    subgroup_builtin_expression,
    warp_lane_index_expression,
)
from crosstl.translator.codegen.hip_codegen import HipCodeGen
from crosstl.translator.codegen.hip_refusals import HipLoweringRefused
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

GENERATORS = [
    pytest.param(CudaCodeGen, CudaLoweringRefused, "CUDA", id="cuda"),
    pytest.param(HipCodeGen, HipLoweringRefused, "HIP", id="hip"),
]

SUBGROUP_SOURCE = """
shader Subgroups {
    compute {
        void main() {
            uint size = gl_SubgroupSize;
            uint lane = gl_SubgroupInvocationID;
            uint id = gl_SubgroupID;
            uint count = gl_NumSubgroups;
            uint wave_lane = WaveGetLaneIndex();
        }
    }
}
"""


def _generate(generator_type, source):
    return generator_type().generate(Parser(Lexer(source).tokens).parse())


@pytest.mark.parametrize("generator_type, _refusal, _label", GENERATORS)
def test_subgroup_builtins_are_lowered(generator_type, _refusal, _label):
    code = " ".join(_generate(generator_type, SUBGROUP_SOURCE).split())

    for name in (
        "gl_SubgroupSize",
        "gl_SubgroupInvocationID",
        "gl_SubgroupID",
        "gl_NumSubgroups",
    ):
        assert name not in code
    assert "warpSize" in code
    assert "threadIdx.x & (warpSize - 1)" not in code


@pytest.mark.parametrize("generator_type, refusal, label", GENERATORS)
def test_unmapped_reserved_builtin_is_refused(generator_type, refusal, label):
    source = """
    shader Masks {
        compute {
            void main() {
                uvec4 mask = gl_SubgroupEqMask;
            }
        }
    }
    """
    with pytest.raises(
        refusal, match=f"built-in variable gl_SubgroupEqMask: no {label} equivalent"
    ):
        _generate(generator_type, source)


@pytest.mark.parametrize("generator_type, _refusal, _label", GENERATORS)
def test_program_identifiers_are_kept(generator_type, _refusal, _label):
    source = """
    shader Names {
        compute {
            void main() {
                uint glow = 3u;
                uint total = glow + 1u;
            }
        }
    }
    """
    code = _generate(generator_type, source)
    assert "glow + 1u" in code


def test_lane_expressions_follow_linear_thread_ids(tmp_path):
    compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("a C++ compiler is not installed")
    expressions = {
        "lane": warp_lane_index_expression(),
        "size": subgroup_builtin_expression("gl_SubgroupSize"),
        "id": subgroup_builtin_expression("gl_SubgroupID"),
        "count": subgroup_builtin_expression("gl_NumSubgroups"),
    }
    body = "\n".join(
        f"                unsigned int {name} = {expression};"
        for name, expression in expressions.items()
    )
    program = f"""
#include <cstdio>
struct dim {{ unsigned int x, y, z; }};
static dim threadIdx;
static dim blockDim = {{8, 6, 1}};
static const int warpSize = 32;
int main() {{
    for (unsigned int z = 0; z < blockDim.z; ++z)
        for (unsigned int y = 0; y < blockDim.y; ++y)
            for (unsigned int x = 0; x < blockDim.x; ++x) {{
                threadIdx = {{x, y, z}};
{body}
                std::printf("%u %u %u %u %u\\n", x + y * blockDim.x, lane, size, id, count);
            }}
    return 0;
}}
"""
    source_path = tmp_path / "lanes.cpp"
    binary_path = tmp_path / "lanes"
    source_path.write_text(program, encoding="utf-8")
    built = subprocess.run(
        [compiler, "-std=c++17", str(source_path), "-o", str(binary_path)],
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stderr + "\n\n" + program
    ran = subprocess.run([str(binary_path)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr

    rows = [
        tuple(int(value) for value in line.split()) for line in ran.stdout.splitlines()
    ]
    assert len(rows) == 48
    for linear, lane, size, subgroup, count in rows:
        assert (lane, size, subgroup, count) == (linear % 32, 32, linear // 32, 2)
