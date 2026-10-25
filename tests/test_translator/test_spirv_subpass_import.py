"""SPIR-V subpass-input reads import as ``subpassLoad`` and reach every target.

``OpImageRead`` of a ``SubpassData`` image reads the input attachment at the current fragment plus its coordinate
(SPIR-V 1.6 section 3.49.10). The importer used to turn it into ``imageLoad(image, ivec2(0, 0))``, which no emitter
accepts on a subpass input. These cases round-trip GLSL subpass reads through SPIR-V assembly.
"""

import pytest

import crosstl
from crosstl.backend.SPIRV.ray_tracing_import import SpirvInstructionUnsupportedError

SUBPASS_READS = """#version 450
layout (input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput inputColor;
layout (input_attachment_index = 1, set = 0, binding = 1) uniform subpassInputMS inputSamples;
layout(location = 0) out vec4 o_color;
void main() { o_color = subpassLoad(inputColor) + subpassLoad(inputSamples, 1); }
"""


@pytest.fixture(scope="module")
def assembly(tmp_path_factory):
    source = tmp_path_factory.mktemp("subpass") / "reads.frag"
    source.write_text(SUBPASS_READS)
    return crosstl.translate(str(source), backend="vulkan", format_output=False)


def _import(tmp_path, assembly, backend):
    path = tmp_path / "reads.spvasm"
    path.write_text(assembly)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_subpass_reads_import_as_subpass_loads(tmp_path, assembly):
    cgl = _import(tmp_path, assembly, "cgl")
    assert "subpassLoad(inputColor)" in cgl
    assert "subpassLoad(inputSamples, 1)" in cgl
    assert "imageLoad" not in cgl


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("directx", "inputColor.Load("),
        ("opengl", "texelFetch(inputColor"),
        ("vulkan", "OpImageRead"),
    ],
)
def test_imported_subpass_reads_reach_each_target(tmp_path, assembly, backend, expected):
    assert expected in _import(tmp_path, assembly, backend)


def test_a_subpass_read_at_a_nonzero_offset_is_refused(tmp_path, assembly):
    lines = assembly.splitlines()
    for index, line in enumerate(lines):
        if "OpConstantComposite" in line:
            result, _, rest = line.partition(" = ")
            composite_type, first, _second = rest.split()[1:4]
            lines[index] = f"{result} = OpConstantComposite {composite_type} {first} %offset_one"
            lines.insert(index, f"%offset_one = OpConstant {_scalar_type(lines, first)} 1")
            break
    with pytest.raises(SpirvInstructionUnsupportedError, match="not the constant"):
        _import(tmp_path, "\n".join(lines) + "\n", "cgl")


def _scalar_type(lines, constant_id):
    for line in lines:
        if line.startswith(f"{constant_id} = OpConstant "):
            return line.split()[3]
    raise AssertionError(f"no OpConstant {constant_id}")
