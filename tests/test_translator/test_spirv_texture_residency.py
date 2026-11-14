"""CrossGL texture residency queries lower to SPIR-V's ``OpImageSparse*`` instructions.

The read keeps its ordinary lowering, and its image instruction is replaced by the sparse twin, which returns the
``int`` residency code and the texel in a struct; ``textureResident`` is ``OpImageSparseTexelsResident``.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.spirv_texture_residency import SPIRV_SPARSE_OPCODES
from crosstl.translator.codegen.texture_residency import (
    TextureResidencyUnsupportedError,
)

FRAGMENT = """
shader Residency {
    struct FragmentInput { vec2 uv; }
    struct FragmentOutput { vec4 color; }
    fragment {
        uniform sampler2D colorMap;
        uniform sampler2DShadow shadowMap;
        uniform sampler2DArray layers;
        uniform samplerCube cube;
        uniform sampler3D volume;
        uniform sampler2DMS multisampled;
        layout(rgba8) uniform image2D image;

        bool residentSample(vec2 uv, out vec4 texel) {
            return textureResident(textureResidency(texture(colorMap, uv), texel));
        }

        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            vec4 a; vec4 b; vec4 c; vec4 d; float s; vec4 l; vec4 f; vec4 g; vec4 h; vec4 i; vec4 m;
            float n; vec4 o; vec4 q; vec4 r;
            int code = textureResidency(texture(colorMap, input.uv), a);
            code |= textureResidency(texture(colorMap, input.uv, 0.5), b);
            code |= textureResidency(texelFetch(colorMap, ivec2(input.uv), 0), c);
            code |= textureResidency(textureGather(colorMap, input.uv, 1), d);
            code |= textureResidency(textureCompare(shadowMap, input.uv, 0.5), s);
            code |= textureResidency(textureLod(layers, vec3(input.uv, 1.0), 0.0), l);
            code |= textureResidency(texture(cube, vec3(input.uv, 1.0)), f);
            code |= textureResidency(textureOffset(volume, vec3(input.uv, 1.0), ivec3(1)), g);
            code |= textureResidency(textureClamp(colorMap, input.uv, 1.0), h);
            code |= textureResidency(textureGatherCompare(shadowMap, input.uv, 0.5), i);
            code |= textureResidency(textureCompareLod(shadowMap, input.uv, 0.5, 0.0), n);
            code |= textureResidency(texelFetch(multisampled, ivec2(input.uv), 1), o);
            code |= textureResidency(imageLoad(image, ivec2(input.uv)), q);
            code |= textureResidency(textureGather(cube, vec3(input.uv, 1.0)), r);
            for (int it = 0; it < 2 && textureResident(textureResidency(texture(colorMap, input.uv + float(it)), m)); it++) {
                b += m;
            }
            bool helper = residentSample(input.uv, output.color);
            vec4 sum = a + b + c + d + vec4(s) + l + f + g + h + i + m + vec4(n) + o + q + r;
            output.color = textureResident(code) && helper ? sum + output.color : vec4(0.0);
            return output;
        }
    }
}
"""

COMPUTE = """
shader ResidencyCompute {
    compute {
        uniform sampler2D colorMap;
        buffer float results[];

        layout(local_size_x = 8) in;
        void main() {
            vec4 texel;
            int code = textureResidency(texture(colorMap, vec2(0.5)), texel);
            results[0] = textureResident(code) ? texel.x : -1.0;
        }
    }
}
"""


def _translate(tmp_path, source, backend="vulkan"):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _validate(tmp_path, code):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return
    asm = tmp_path / "out.spvasm"
    binary = tmp_path / "out.spv"
    asm.write_text(code, encoding="utf-8")
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr


def _image_instructions(code):
    return re.findall(r"= (OpImage(?:Sparse)?\w*(?:Lod|Fetch|Gather|Read)) ", code)


def test_every_wrapped_read_becomes_its_sparse_twin(tmp_path):
    code = _translate(tmp_path, FRAGMENT)

    assert "OpCapability SparseResidency" in code
    instructions = _image_instructions(code)
    # Every read in the shader is wrapped in a residency query.
    assert instructions and all("Sparse" in name for name in instructions)
    assert set(instructions) == set(SPIRV_SPARSE_OPCODES.values())
    # The helper's query (also inlined at its call, whose texel is a member), the loop condition's and the final
    # test.
    assert code.count("OpImageSparseTexelsResident") == 4
    assert re.search(r"OpImageSparseSampleImplicitLod %\d+ %\d+ %\d+ MinLod %\d+", code)
    assert re.search(r"OpImageSparseFetch %\d+ %\d+ %\d+ Sample %\d+", code)
    _validate(tmp_path, code)


def test_reads_return_the_residency_struct_and_extract_both_members(tmp_path):
    code = _translate(tmp_path, FRAGMENT)

    name = re.search(r'OpName %(\d+) "_crosstl_sparse_residency_v4float"', code)
    assert name, "the vec4 residency struct is named"
    struct = re.search(rf"%({name.group(1)}) = OpTypeStruct %(\d+) %(\d+)\n", code)
    assert struct, "the vec4 residency struct is declared"
    sparse = re.search(
        rf"%(\d+) = OpImageSparseSampleImplicitLod %{struct.group(1)} ", code
    )
    assert sparse
    result = sparse.group(1)
    assert re.search(rf"= OpCompositeExtract %{struct.group(3)} %{result} 1\b", code)
    assert re.search(rf"= OpCompositeExtract %{struct.group(2)} %{result} 0\b", code)
    assert re.search(
        rf"%{struct.group(2)} = OpTypeInt 32 1", code
    ), "the residency code is a signed 32-bit integer"


def test_compute_reads_use_the_explicit_lod_sparse_form(tmp_path):
    code = _translate(tmp_path, COMPUTE)

    assert re.search(r"= OpImageSparseSampleExplicitLod %\d+ %\d+ %\d+ Lod %\d+", code)
    assert "OpImageSparseTexelsResident" in code
    _validate(tmp_path, code)


def test_a_constant_splat_offset_is_a_constant_offset(tmp_path):
    source = """
shader Offsets {
    struct FragmentInput { vec2 uv; }
    struct FragmentOutput { vec4 color; }
    fragment {
        uniform sampler3D volume;
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            output.color = textureOffset(volume, vec3(input.uv, 1.0), ivec3(1));
            return output;
        }
    }
}
"""
    code = _translate(tmp_path, source)

    # Vulkan accepts a non-constant Offset only on gathers (VUID-RuntimeSpirv-Offset-10213).
    assert re.search(r"OpImageSampleImplicitLod %\d+ %\d+ %\d+ ConstOffset %\d+", code)
    assert " Offset %" not in code
    _validate(tmp_path, code)


def test_resident_refuses_a_non_integer_code(tmp_path):
    source = COMPUTE.replace("textureResident(code)", "textureResident(texel.x)")

    with pytest.raises(TextureResidencyUnsupportedError) as raised:
        _translate(tmp_path, source)
    assert raised.value.operation == "textureResident"
    assert "scalar integer residency code" in str(raised.value)
