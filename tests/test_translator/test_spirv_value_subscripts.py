"""SPIR-V output of a subscript of ``gl_WorkGroupSize``, a constant composite that has no variable."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused

SHADER = """shader Probe {
    compute {
        layout(LAYOUT) in;
        buffer uint* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = READ;
        }
    }
}
"""


def _vulkan(tmp_path, layout, read):
    path = tmp_path / "shader.cgl"
    path.write_text(SHADER.replace("LAYOUT", layout).replace("READ", read))
    return crosstl.translate(str(path), backend="vulkan", format_output=False)


def _builtin(code):
    """The id decorated BuiltIn WorkgroupSize."""
    return re.search(r"OpDecorate (%\w+) BuiltIn WorkgroupSize", code).group(1)


def _validate(tmp_path, code):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
    asm.write_text(code)
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "layout",
    ["local_size_x = 64", "local_size_x = 64, local_size_x_id = 4"],
    ids=["fixed", "specialized"],
)
def test_constant_subscript_extracts_the_component(tmp_path, layout):
    code = _vulkan(tmp_path, layout, "gl_WorkGroupSize[0] + gl_WorkGroupSize[2]")
    size = _builtin(code)

    assert re.search(rf"OpCompositeExtract %\w+ {size} 0\b", code)
    assert re.search(rf"OpCompositeExtract %\w+ {size} 2\b", code)
    assert not re.search(rf"OpAccessChain %\w+ {size}\b", code)
    _validate(tmp_path, code)


def test_dynamic_subscript_extracts_the_component(tmp_path):
    code = _vulkan(tmp_path, "local_size_x = 64", "gl_WorkGroupSize[i % 3u]")
    size = _builtin(code)

    assert re.search(rf"OpVectorExtractDynamic %\w+ {size} %\w+", code)
    assert not re.search(rf"OpAccessChain %\w+ {size}\b", code)
    _validate(tmp_path, code)


def test_subscript_matches_the_swizzle(tmp_path):
    indexed = _vulkan(tmp_path, "local_size_x = 64", "gl_WorkGroupSize[1]")
    swizzled = _vulkan(tmp_path, "local_size_x = 64", "gl_WorkGroupSize.y")

    def extracts(code):
        return re.findall(rf"OpCompositeExtract %\w+ {_builtin(code)} (\d)", code)

    assert extracts(indexed) == extracts(swizzled) == ["1"]


def test_out_of_range_subscript_is_refused(tmp_path):
    with pytest.raises(SpirvLoweringRefused, match="Component 3 .* is out of range"):
        _vulkan(tmp_path, "local_size_x = 64", "gl_WorkGroupSize[3]")


BLOCK_MEMBER_SHADER = """shader Probe {
    struct VertexOutput {
        vec4 gl_Position @ gl_Position;
    };
    cbuffer buf @binding(0) {
        mat4 MVP;
        vec4 position[4];
    };
    vertex {
        VertexOutput main() {
            VertexOutput output;
            output.gl_Position = MVP * position[gl_VertexIndex];
            return output;
        }
    }
}
"""


def test_block_member_subscript_emits_one_access_chain(tmp_path):
    """A bare uniform-block member keeps its access chain, and the lookup for a vector value adds none."""
    path = tmp_path / "shader.cgl"
    path.write_text(BLOCK_MEMBER_SHADER)
    code = crosstl.translate(str(path), backend="vulkan", format_output=False)

    chains = re.findall(r"= (OpAccessChain %\w+ %\w+ %\w+)$", code, re.MULTILINE)
    assert len(chains) == len(set(chains)), chains
    _validate(tmp_path, code)
