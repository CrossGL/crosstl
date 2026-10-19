"""CUDA warp shuffles in CrossGL programs: workgroup-memory emulation for one-warp workgroups, typed refusals otherwise.

The source-lane arithmetic is checked exhaustively against the CUDA C++ Programming Guide's description (the
``x_warp_shuffle`` conformance oracle), the emitted programs are checked for shape and, where the tools are
installed, validated; execution evidence is in the conformance scoreboard.
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator import parse
from crosstl.translator.semantics.warp import (
    WARP_TARGETS,
    CudaWarpIntrinsicUnsupportedError,
    WarpLegalizer,
)
from tests.conformance import warp_cases

EMULATING = ("metal", "directx", "opengl", "vulkan", "slang")


def program(
    body,
    layout="layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;",
    extra="",
):
    return f"""
shader WarpProbe {{
    {extra}
    compute probe {{
        {layout}
        uniform int count;
        buffer int* data;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            int v = data[i];
            {body}
            data[i] = v;
        }}
    }}
}}
"""


def translate(tmp_path, source, backend):
    path = tmp_path / "probe.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def refusal(tmp_path, source, backend="metal"):
    with pytest.raises(CudaWarpIntrinsicUnsupportedError) as raised:
        translate(tmp_path, source, backend)
    assert (
        raised.value.code
        == f"project.translate.{backend}-cuda-warp-intrinsic-unsupported"
    )
    return raised.value.reason_code


# Source-lane arithmetic -------------------------------------------------------------------------------------------
def _evaluate(node, env):
    if isinstance(node, A.LiteralNode):
        return int(node.value)
    if isinstance(node, A.IdentifierNode):
        return env[node.name]
    if isinstance(node, A.FunctionCallNode):
        assert node.function.name == "uint"
        return _evaluate(node.arguments[0], env) & 0xFFFFFFFF
    if isinstance(node, A.TernaryOpNode):
        branch = node.true_expr if _evaluate(node.condition, env) else node.false_expr
        return _evaluate(branch, env)
    left, right = _evaluate(node.left, env), _evaluate(node.right, env)
    return {
        "&": lambda: left & right,
        "|": lambda: left | right,
        "^": lambda: left ^ right,
        "+": lambda: (left + right) & 0xFFFFFFFF,
        "-": lambda: (left - right) & 0xFFFFFFFF,
        "<": lambda: int(left < right),
        "<=": lambda: int(left <= right),
        ">=": lambda: int(left >= right),
        "&&": lambda: int(bool(left) and bool(right)),
    }[node.operator]()


def _source_lanes(mode, operand, width, lanes):
    legalizer = WarpLegalizer(parse(program("")), "metal")
    legalizer.lanes = lanes
    prelude = []
    name = legalizer.source_lane(mode, A.IdentifierNode("operand"), width, prelude)
    result = []
    for lane in range(lanes):
        env = {"gl_LocalInvocationIndex": lane, "operand": operand}
        for declaration in prelude:
            env[declaration.name] = _evaluate(declaration.initial_value, env)
        result.append(env[name])
    return np.array(result)


ORACLES = {
    "down": warp_cases._down,
    "up": warp_cases._up,
    "bfly": warp_cases._xor,
    "idx": lambda values, operand, width: warp_cases._index(
        values, np.full(warp_cases.WARP, operand), width
    ),
}


@pytest.mark.parametrize("mode", sorted(ORACLES))
@pytest.mark.parametrize("width", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("lanes", [1, 7, 16, 32])
def test_source_lane_matches_cuda_semantics(mode, width, lanes):
    # Lane identity: the oracle's result names the CUDA source lane; a source past the workgroup is an inactive lane,
    # whose value CUDA leaves undefined, and the emulation returns the caller's own value there.
    identity = np.arange(warp_cases.WARP)
    for operand in range(0, 32 if mode != "idx" else 100):
        expected = ORACLES[mode](identity, operand, width)[:lanes]
        expected = np.where(expected < lanes, expected, np.arange(lanes))
        assert (_source_lanes(mode, operand, width, lanes) == expected).all(), operand


def test_delta_uses_low_five_bits_as_ptx_does():
    # PTX shfl.sync takes b[4:0]; the Programming Guide does not define deltas of 32 or more.
    assert (_source_lanes("down", 33, 32, 32) == _source_lanes("down", 1, 32, 32)).all()


# Emission ---------------------------------------------------------------------------------------------------------
REDUCTION = """
            for (int offset = 16; offset > 0; offset /= 2) {
                v += __shfl_down_sync(0xFFFFFFFF, v, offset);
            }
"""


@pytest.mark.parametrize("backend", EMULATING)
def test_shuffle_is_emulated_through_workgroup_memory(tmp_path, backend):
    generated = translate(tmp_path, program(REDUCTION), backend)
    assert "__shfl" not in generated and "shfl_down" not in generated
    assert "cgl_warp_exchange_0" in generated
    assert "cgl_warp_result_0" in generated


def test_vulkan_exchange_is_workgroup_storage_and_validates(tmp_path):
    generated = translate(tmp_path, program(REDUCTION), "vulkan")
    names = {
        line.split()[1]: line.split('"')[1]
        for line in generated.splitlines()
        if line.startswith("OpName ")
    }
    exchange = next(i for i, name in names.items() if name == "cgl_warp_exchange_0")
    variable = next(
        line
        for line in generated.splitlines()
        if line.startswith(f"{exchange} = OpVariable")
    )
    assert variable.endswith("Workgroup")
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        source, binary = tmp_path / "probe.spvasm", tmp_path / "probe.spv"
        source.write_text(generated)
        assembled = subprocess.run(
            ["spirv-as", "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
            capture_output=True,
            text=True,
        )
        assert assembled.returncode == 0, assembled.stderr
        validated = subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
            capture_output=True,
            text=True,
        )
        assert validated.returncode == 0, validated.stderr


def test_wgsl_emulates_in_uniform_control_flow(tmp_path):
    generated = translate(
        tmp_path,
        program(REDUCTION + "if (count > 2) { v = __shfl_sync(0xFFFFFFFF, v, 0); }"),
        "wgsl",
    )
    assert "workgroupBarrier()" in generated and "__shfl" not in generated
    if shutil.which("naga"):
        path = tmp_path / "probe.wgsl"
        path.write_text(generated)
        checked = subprocess.run(["naga", str(path)], capture_output=True, text=True)
        assert checked.returncode == 0, checked.stderr


def test_wgsl_refuses_after_a_divergent_break(tmp_path):
    body = """
            for (int k = 0; k < count; k++) {
                if (v > 3) break;
                v += __shfl_down_sync(0xFFFFFFFF, v, 1);
            }
"""
    assert refusal(tmp_path, program(body), "wgsl") == "non-uniform-control-flow"
    assert "cgl_warp_exchange_0" in translate(tmp_path, program(body), "metal")


def test_wgsl_refuses_under_a_lane_dependent_condition(tmp_path):
    body = "if (i > 4u) { v = __shfl_xor_sync(0xFFFFFFFF, v, 1); }"
    assert refusal(tmp_path, program(body), "wgsl") == "non-uniform-control-flow"


def test_partial_workgroup_reads_only_existing_lanes(tmp_path):
    layout = "layout(local_size_x = 8, local_size_y = 2, local_size_z = 1) in;"
    generated = translate(tmp_path, program(REDUCTION, layout), "metal")
    assert "cgl_warp_exchange_0[16]" in generated
    assert "< 16u" in generated


# Refusals ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "body,layout,reason",
    [
        (
            REDUCTION,
            "layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;",
            "multiple-warps",
        ),
        (REDUCTION, "", "workgroup-size"),
        ("v = __shfl_sync(0x0000FFFF, v, 0);", None, "mask"),
        ("v = __shfl_sync(__activemask(), v, 0);", None, "intrinsic"),
        ("v = __shfl_sync(0xFFFFFFFF, v, 0, 12);", None, "width"),
        ("v = __shfl_sync(0xFFFFFFFF, v, 0, count);", None, "width"),
        (
            "v = (v > 0 && __shfl_sync(0xFFFFFFFF, v, 0) > 0) ? 1 : 0;",
            None,
            "conditional-evaluation",
        ),
        (
            "v = v > 0 ? __shfl_sync(0xFFFFFFFF, v, 0) : 0;",
            None,
            "conditional-evaluation",
        ),
        (
            "while (__shfl_sync(0xFFFFFFFF, v, 0) > v) { v++; }",
            None,
            "conditional-evaluation",
        ),
        ("v = int(__ballot_sync(0xFFFFFFFF, v > 0));", None, "intrinsic"),
        ("__syncwarp(0xFFFFFFFF);", None, "intrinsic"),
        (
            "data[i + 1u] = v++ + __shfl_sync(0xFFFFFFFF, v, 0);",
            None,
            "evaluation-order",
        ),
        ("v = __shfl_sync(0xFFFFFFFF, v);", None, "arguments"),
    ],
)
def test_refusals_are_typed(tmp_path, body, layout, reason):
    source = program(body) if layout is None else program(body, layout)
    assert refusal(tmp_path, source) == reason


def test_shuffle_in_a_helper_function_is_refused(tmp_path):
    source = program(
        "v = helper(v);",
        extra="int helper(int x) { return __shfl_sync(0xFFFFFFFF, x, 0); }",
    )
    assert refusal(tmp_path, source) == "helper-function"


def test_shuffle_outside_compute_is_refused(tmp_path):
    source = """
shader FragmentProbe {
    fragment {
        vec4 main(float level) @ gl_FragColor {
            float shuffled = __shfl_sync(0xFFFFFFFF, level, 0);
            return vec4(shuffled);
        }
    }
}
"""
    assert refusal(tmp_path, source) == "stage"


def test_user_function_with_a_warp_name_is_left_alone(tmp_path):
    source = program(
        "v = __shfl_sync(1u, v, 0);",
        extra="int __shfl_sync(uint mask, int x, int lane) { return x + lane; }",
    )
    assert "cgl_warp" not in translate(tmp_path, source, "metal")


def test_every_target_without_cuda_warps_is_covered():
    assert set(WARP_TARGETS) == {
        "metal",
        "directx",
        "opengl",
        "webgl",
        "vulkan",
        "wgsl",
        "slang",
    }


def test_cuda_source_partial_mask_vote_is_refused_not_zeroed(tmp_path):
    # The CUDA importer used to replace a warp intrinsic it could not map with the constant 0.
    path = tmp_path / "vote.cu"
    path.write_text("""
__global__ void vote(const int* input, unsigned int* result, unsigned int mask) {
    int tx = threadIdx.x;
    result[tx] = __ballot_sync(mask, input[tx] > 0);
}
""")
    crossgl = crosstl.translate(str(path), backend="cgl", format_output=False)
    assert "__ballot_sync(mask" in crossgl
    for backend in ("metal", "vulkan"):
        with pytest.raises(CudaWarpIntrinsicUnsupportedError) as raised:
            crosstl.translate(str(path), backend=backend, format_output=False)
        assert (
            raised.value.code
            == f"project.translate.{backend}-cuda-warp-intrinsic-unsupported"
        )
