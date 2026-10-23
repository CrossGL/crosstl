"""Unsized resource arrays get a size that covers every index computed at run time.

HLSL ``Texture2D t[]`` and ``RWStructuredBuffer<T> b[]`` hold as many resources as the host binds. Metal's
``array<T, N>`` arguments and OpenGL's resource arrays need a size, and the shared inference took it from constant
indexes only: ``t[1]`` with ``t[i % 5]`` gave ``t`` two elements, so ``t[4]`` read past the array. An access whose
index has no provable bound, such as ``t[material]`` with ``material`` from a constant buffer, used to give a single
element and is now refused.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.resource_array_bounds import (
    ResourceArrayIndexUnboundedError,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

TARGETS = ("metal", "opengl")

PREFIX = """
cbuffer Params : register(b0) { uint material; int offset; };
Texture2D<float4> texs[] : register(t0);
RWStructuredBuffer<float> blocks[] : register(u0);
RWStructuredBuffer<float4> results : register(u8);
RWStructuredBuffer<uint> counter : register(u9);
SamplerState samp : register(s0);
"""


def compute(body):
    return (
        PREFIX
        + """
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    float4 acc = float4(0.0, 0.0, 0.0, 0.0);
"""
        + body
        + """
    results[i] = acc;
}
"""
    )


def sample(index):
    return f"texs[{index}].SampleLevel(samp, float2(0.5, 0.5), 0)"


def translate(tmp_path, source, backend):
    path = tmp_path / "shader.hlsl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def texture_size(code, backend):
    if backend == "metal":
        marker = "array<texture2d<float>, "
        start = code.index(marker) + len(marker)
        return int(code[start : code.index(">", start)])
    marker = "uniform sampler2D texs["
    start = code.index(marker) + len(marker)
    return int(code[start : code.index("]", start)])


def block_size(code, backend):
    if backend == "metal":
        pattern = r"array<device float ?\*, (\d+)> blocks\b"
    else:
        pattern = r"\bblocks\[(\d+)\];"
    return int(re.search(pattern, code).group(1))


BOUNDED = [
    # A constant index and dynamic ones: i & 3 reaches 3, i % 5 reaches 4.
    (
        "acc += " + sample(1) + " + " + sample("i & 3") + " + " + sample("i % 5") + ";",
        5,
    ),
    # A canonical loop's induction variable.
    ("for (uint k = 0; k < 6; k++) { acc += " + sample("k") + "; }", 6),
    ("for (uint k = 2; k <= 6; k += 2) { acc += " + sample("k") + "; }", 7),
    # Locals declared once from a bounded value, and arithmetic on them.
    ("uint slot = min(i, 2u); acc += " + sample("slot + 1") + ";", 4),
    ("uint lane = i & 7; acc += " + sample("lane / 2") + ";", 4),
    ("uint lane = i & 7; acc += " + sample("(lane >> 1) * 2") + ";", 7),
    ("acc += " + sample("clamp(int(i), 0, 9)") + ";", 10),
    ("acc += " + sample("NonUniformResourceIndex(i % 3)") + ";", 3),
    ("acc += " + sample("i % 2 == 0 ? 1 : 6") + ";", 7),
]


@pytest.mark.parametrize("backend", TARGETS)
@pytest.mark.parametrize("body, size", BOUNDED)
def test_dynamic_texture_indexes_size_the_array(tmp_path, backend, body, size):
    code = translate(tmp_path, compute(body), backend)
    assert texture_size(code, backend) == size


@pytest.mark.parametrize("backend", TARGETS)
def test_dynamic_buffer_indexes_size_the_array(tmp_path, backend):
    body = "float v = blocks[3][i]; blocks[i % 5][i] = v; acc.x = blocks[0][i];"
    code = translate(tmp_path, compute(body), backend)
    assert block_size(code, backend) == 5


UNBOUNDED = [
    # A value from a constant buffer.
    "acc += " + sample("material") + ";",
    # The loop body changes the induction variable.
    "for (uint k = 0; k < 4; k++) { k += material; acc += " + sample("k") + "; }",
    # A loop whose limit is not a constant.
    "for (uint k = 0; k < material; k++) { acc += " + sample("k") + "; }",
    # A local an atomic writes through its original-value output.
    "uint slot = 0; InterlockedAdd(counter[0], 1, slot); acc += "
    + sample("slot")
    + ";",
    # A local assigned after its declaration.
    "uint slot = 1; slot = material; acc += " + sample("slot") + ";",
    # A product of values that may be negative: (-4) * (-4) exceeds 3 * 3.
    "int a = min(offset, 3); acc += " + sample("a * a") + ";",
    # A shift of a value that may have wrapped through an unsigned conversion.
    "uint w = min(offset, 3); acc += " + sample("w >> 30") + ";",
    # A difference whose subtrahend may be negative.
    "acc += " + sample("(i & 3) - offset") + ";",
]


@pytest.mark.parametrize("backend", TARGETS)
@pytest.mark.parametrize("body", UNBOUNDED)
def test_unbounded_dynamic_indexes_are_refused(tmp_path, backend, body):
    with pytest.raises(ResourceArrayIndexUnboundedError, match="texs"):
        translate(tmp_path, compute(body), backend)


def _metal_compiler_available():
    if not shutil.which("xcrun"):
        return False
    probe = subprocess.run(["xcrun", "--find", "metal"], capture_output=True, text=True)
    return probe.returncode == 0


def test_sized_texture_arrays_compile_natively(tmp_path):
    body = BOUNDED[0][0] + " " + BOUNDED[1][0] + " " + BOUNDED[3][0]
    source = compute(body).replace(
        "RWStructuredBuffer<float> blocks[] : register(u0);\n", ""
    )
    checked = False
    glslang = shutil.which("glslangValidator")
    if glslang:
        path = tmp_path / "shader.comp"
        path.write_text(translate(tmp_path, source, "opengl"), encoding="utf-8")
        run = subprocess.run(
            [glslang, str(path)], capture_output=True, text=True, timeout=60
        )
        assert run.returncode == 0, run.stdout
        checked = True
    if _metal_compiler_available():
        path = tmp_path / "shader.metal"
        path.write_text(translate(tmp_path, source, "metal"), encoding="utf-8")
        run = subprocess.run(
            ["xcrun", "metal", "-c", str(path), "-o", str(tmp_path / "shader.air")],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert run.returncode == 0, run.stderr
        checked = True
    if not checked:
        pytest.skip("neither glslangValidator nor the Metal compiler is available")


CROSSGL_ALIAS_HELPERS = {
    # A branch merges a literal with a value from the stage input.
    "branch": (
        """
        int layer = 2;
        if (choose) { layer = choose ? dynamicLayer : 5; } else { layer = 3; }
        return layer;
    """
    ),
    # A loop with a dynamic trip count carries the stage input into the alias.
    "loop": (
        """
        int layer = 1;
        for (int i = 0; i < dynamicLayer; i = i + 1) { layer = choose ? dynamicLayer : 6; }
        return layer;
    """
    ),
    # A switch case stores the stage input.
    "switch": (
        """
        int layer = 1;
        switch (dynamicLayer) {
            case 0: layer = choose ? dynamicLayer : 6; break;
            default: layer = 2; break;
        }
        return layer;
    """
    ),
    # Arithmetic on the stage input in a helper return.
    "arithmetic": (
        """
        int fallback = choose ? dynamicLayer : 5;
        return fallback + 2;
    """
    ),
}


def crossgl_alias_shader(helper_body, argument):
    return f"""
    shader AliasIndexes {{
        sampler2D textures[];
        sampler samplers[];

        int pickLayer(bool choose, int dynamicLayer) {{
            {helper_body}
        }}

        fragment {{
            vec4 main(vec2 uv, bool choose, int dynamicLayer) @ gl_FragColor {{
                int layer = pickLayer(choose, {argument});
                return texture(textures[layer], samplers[layer], uv);
            }}
        }}
    }}
    """


def crossgl_codegen(source, backend):
    ast = Parser(Lexer(source).get_tokens()).parse()
    generator = MetalCodeGen() if backend == "metal" else GLSLCodeGen()
    return generator.generate(ast)


@pytest.mark.parametrize("backend", TARGETS)
@pytest.mark.parametrize("pattern", sorted(CROSSGL_ALIAS_HELPERS))
def test_stage_input_index_aliases_are_refused(backend, pattern):
    source = crossgl_alias_shader(CROSSGL_ALIAS_HELPERS[pattern], "dynamicLayer")
    # Metal checks the sampler array first; GLSL folds samplers into the textures.
    with pytest.raises(
        ResourceArrayIndexUnboundedError, match="(textures|samplers) is indexed in main"
    ):
        crossgl_codegen(source, backend)


@pytest.mark.parametrize(
    "pattern, size",
    [("branch", 6), ("loop", 7), ("switch", 7), ("arithmetic", 8)],
)
@pytest.mark.parametrize("backend", TARGETS)
def test_clamped_stage_input_index_aliases_size_the_array(backend, pattern, size):
    source = crossgl_alias_shader(
        CROSSGL_ALIAS_HELPERS[pattern], "clamp(dynamicLayer, 0, 5)"
    )
    code = crossgl_codegen(source, backend)
    if backend == "metal":
        assert f"array<texture2d<float>, {size}> textures" in code
    else:
        assert f"uniform sampler2D textures[{size}];" in code
