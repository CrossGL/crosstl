"""Constant blocks declared inside a stage are declared and bound like module-scope blocks.

The Metal, Slang, Rust, CUDA and HIP generators read only the module-scope block list, so a block declared at
stage scope was dropped and its members were emitted as undeclared names; the SPIR-V generator refused the program.
Rule carrying moves such blocks to module scope for those targets (``semantics/stage_constant_blocks.py``).
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import (
    StageConstantBlockConflictError,
    hoist_stage_constant_blocks,
)

COMPUTE_BLOCK = "uniform Params { Item p; };"
COMPUTE = """
shader Blocks {{
    struct Item {{ float a; vec4 b; }};
    {module_block}
    compute {{
        {stage_block}
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer float* outBuf;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            outBuf[i] = p.a + p.b.x;
        }}
    }}
}}
"""

GRAPHICS_BLOCKS = (
    "uniform Params { mat4 mvp; vec4 tint; };",
    "uniform FragmentParams { vec4 color; };",
)
GRAPHICS = """
shader Blocks {{
    struct VertexOutput {{ vec4 position; }};
    {module_blocks}
    vertex {{
        {vertex_block}
        VertexOutput main(vec3 position) {{
            VertexOutput output;
            output.position = mvp * vec4(position, 1.0) + tint;
            return output;
        }}
    }}
    fragment {{
        {fragment_block}
        vec4 main() {{
            return color;
        }}
    }}
}}
"""

COMPUTE_TARGETS = ("metal", "slang", "rust", "cuda", "hip", "vulkan")
GRAPHICS_TARGETS = ("metal", "slang", "vulkan")


def _translate(source, backend, tmp_path, name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _compute(stage_scope):
    if stage_scope:
        return COMPUTE.format(module_block="", stage_block=COMPUTE_BLOCK)
    return COMPUTE.format(module_block=COMPUTE_BLOCK, stage_block="")


def _graphics(stage_scope):
    if stage_scope:
        return GRAPHICS.format(
            module_blocks="",
            vertex_block=GRAPHICS_BLOCKS[0],
            fragment_block=GRAPHICS_BLOCKS[1],
        )
    return GRAPHICS.format(
        module_blocks="\n    ".join(GRAPHICS_BLOCKS),
        vertex_block="",
        fragment_block="",
    )


@pytest.mark.parametrize("backend", COMPUTE_TARGETS)
def test_compute_stage_block_matches_module_scope_block(backend, tmp_path):
    stage = _translate(_compute(True), backend, tmp_path, "stage.cgl")
    module = _translate(_compute(False), backend, tmp_path, "module.cgl")
    assert stage == module
    assert "Params" in stage


@pytest.mark.parametrize("backend", GRAPHICS_TARGETS)
def test_graphics_stage_blocks_match_module_scope_blocks(backend, tmp_path):
    stage = _translate(_graphics(True), backend, tmp_path, "stage.cgl")
    module = _translate(_graphics(False), backend, tmp_path, "module.cgl")
    assert stage == module
    assert "FragmentParams" in stage


def test_metal_entry_point_reads_stage_block_through_its_parameter(tmp_path):
    code = _translate(_compute(True), "metal", tmp_path)
    assert "struct Params {" in code
    assert "constant Params& params [[buffer(0)]]" in code
    assert "params.p.a + params.p.b.x" in code


def test_metal_stage_block_output_compiles(tmp_path):
    if not shutil.which("xcrun"):
        pytest.skip("xcrun is unavailable")
    probe = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip("the Metal compiler is unavailable")
    for name, source in (("compute", _compute(True)), ("graphics", _graphics(True))):
        shader = tmp_path / f"{name}.metal"
        shader.write_text(_translate(source, "metal", tmp_path, f"{name}.cgl"))
        result = subprocess.run(
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                "-c",
                str(shader),
                "-o",
                str(tmp_path / f"{name}.air"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


def test_slang_multiplies_by_a_block_matrix_with_mul(tmp_path):
    code = _translate(_graphics(True), "slang", tmp_path)
    assert "mul(mvp, float4(position, 1.0))" in code
    assert "mvp * " not in code


def test_identical_block_in_two_stages_is_declared_once(tmp_path):
    shared = "uniform Shared { vec4 tint; };"
    source = GRAPHICS.format(
        module_blocks="uniform Params { mat4 mvp; };",
        vertex_block=shared,
        fragment_block=shared,
    ).replace("return color;", "return tint;")
    program = _parse(source)
    assert hoist_stage_constant_blocks(program, "metal") == 1
    assert [block.name for block in program.cbuffers] == ["Params", "Shared"]
    assert all(not stage.local_cbuffers for stage in program.stages.values())
    code = _translate(source, "metal", tmp_path)
    assert code.count("struct Shared {") == 1


def test_different_blocks_with_one_name_are_refused():
    source = GRAPHICS.format(
        module_blocks="",
        vertex_block="uniform Params { mat4 mvp; vec4 tint; };",
        fragment_block="uniform Params { vec4 color; };",
    )
    with pytest.raises(StageConstantBlockConflictError) as raised:
        hoist_stage_constant_blocks(_parse(source), "metal")
    assert "Params" in str(raised.value)
    assert "stage-constant-block-conflict" in raised.value.code


def test_member_declared_by_another_stage_is_refused():
    source = GRAPHICS.format(
        module_blocks="uniform Params { mat4 mvp; vec4 tint; };",
        vertex_block="vec4 color = vec4(1.0);",
        fragment_block="uniform FragmentParams { vec4 color; };",
    )
    with pytest.raises(StageConstantBlockConflictError, match="vertex stage"):
        hoist_stage_constant_blocks(_parse(source), "slang")


def test_member_declared_at_module_scope_is_refused(tmp_path):
    source = _compute(True).replace(
        "struct Item { float a; vec4 b; };",
        "struct Item { float a; vec4 b; };\n    uniform float p;",
    )
    with pytest.raises(StageConstantBlockConflictError, match="module-scope"):
        _translate(source, "metal", tmp_path)


@pytest.mark.parametrize("target", ["opengl", "directx", "wgsl", "mojo"])
def test_generators_that_declare_stage_blocks_keep_them_in_their_stage(target):
    program = _parse(_compute(True))
    assert hoist_stage_constant_blocks(program, target) == 0
    (stage,) = program.stages.values()
    assert [block.name for block in stage.local_cbuffers] == ["Params"]
    assert not program.cbuffers
