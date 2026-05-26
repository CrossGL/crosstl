import shutil
import subprocess

import pytest

import crosstl.translator
from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser
from crosstl.backend.GLSL.openglCrossglCodegen import GLSLToCrossGLConverter
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen

FRAGMENT_SMOKE_SHADER = """
shader ExternalValidatorSmoke {
    fragment {
        vec4 main() @ gl_FragColor {
            return vec4(1.0, 0.0, 0.0, 1.0);
        }
    }
}
"""


CROSSGL_SYNCHRONIZATION_COMPUTE_SHADER = """
shader ExternalValidatorSynchronization {
    compute {
        void main() {
            barrier();
            memoryBarrier();
            workgroupBarrier();
        }
    }
}
"""


GLSL_MULTISAMPLE_STORAGE_COMPUTE_SHADER = """
shader GLSLMultisampleStorageValidator {
    image2DMS colorImage @rgba16f;
    uimage2DMS counters @r32ui;
    image2DMSArray layered @rgba16f;

    compute {
        void main() {
            ivec2 pixel = ivec2(0, 1);
            int sampleIndex = 2;
            vec4 oldColor = imageLoad(colorImage, pixel, sampleIndex);
            uint oldCount = imageLoad(counters, pixel, sampleIndex);
            imageStore(colorImage, pixel, sampleIndex, oldColor + vec4(1.0));
            imageStore(counters, pixel, sampleIndex, oldCount + 3u);
            uint atomicOld = imageAtomicAdd(counters, pixel, sampleIndex, 3u);
            uint exchanged = imageAtomicExchange(counters, pixel, sampleIndex, atomicOld + 1u);
            uint swapped = imageAtomicCompSwap(counters, pixel, sampleIndex, exchanged, oldCount);

            ivec3 pixelLayer = ivec3(2, 3, 1);
            vec4 oldLayer = imageLoad(layered, pixelLayer, 0);
            imageStore(layered, pixelLayer, 0, oldLayer + oldColor + vec4(float(swapped)));
        }
    }
}
"""


GLSL_PARAMETER_IMAGE_ATOMIC_COMPUTE_SHADER = """
shader GLSLParameterImageAtomicValidator {
    uimage2D counters @r32ui;

    uint addCounter(uimage2D image @r32ui, ivec2 pixel, uint value) {
        return imageAtomicAdd(image, pixel, value);
    }

    compute {
        void main() {
            uint oldValue = addCounter(counters, ivec2(0, 1), 2u);
            imageStore(counters, ivec2(0, 1), oldValue);
        }
    }
}
"""


MIXED_GLSL_PREPROCESSOR_COMPUTE_SHADER = """
#version 300 es
#extension GL_ARB_separate_shader_objects : enable
precision highp float;

void main() { }
"""


MIXED_GLSL_SSBO_UINT_ATOMICS_COMPUTE_SHADER = """
#version 450 core
layout(std430, binding = 17) buffer AtomicBlock {
    uint counter;
    uint bins[4];
} atomicBlock;

void main() {
    uint oldCounter = atomicAdd(atomicBlock.counter, 1u);
    uint oldBin = atomicExchange(atomicBlock.bins[2], oldCounter);
    uint minBin = atomicMin(atomicBlock.bins[0], 2u);
    uint maxBin = atomicMax(atomicBlock.bins[0], minBin);
    uint andBin = atomicAnd(atomicBlock.bins[1], 15u);
    uint orBin = atomicOr(atomicBlock.bins[1], andBin);
    uint xorBin = atomicXor(atomicBlock.bins[2], orBin);
    uint casBin = atomicCompSwap(atomicBlock.bins[3], xorBin, 7u);
    atomicAdd(atomicBlock.bins[1], casBin);
}
"""


MIXED_GLSL_SSBO_INT_ATOMICS_COMPUTE_SHADER = """
#version 450 core
layout(std430, binding = 18) buffer SignedAtomicBlock {
    int counter;
    int bins[4];
} signedAtomicBlock;

void main() {
    int oldCounter = atomicAdd(signedAtomicBlock.counter, -1);
    int oldBin = atomicExchange(signedAtomicBlock.bins[2], oldCounter);
    int minBin = atomicMin(signedAtomicBlock.bins[0], -2);
    int maxBin = atomicMax(signedAtomicBlock.bins[0], minBin);
    int andBin = atomicAnd(signedAtomicBlock.bins[1], 15);
    int orBin = atomicOr(signedAtomicBlock.bins[1], andBin);
    int xorBin = atomicXor(signedAtomicBlock.bins[2], orBin);
    int casBin = atomicCompSwap(signedAtomicBlock.bins[3], xorBin, -7);
    atomicAdd(signedAtomicBlock.bins[1], casBin);
}
"""


MIXED_GLSL_SSBO_RUNTIME_ARRAY_ATOMICS_COMPUTE_SHADER = """
#version 450 core
layout(std430, binding = 19) buffer RuntimeAtomicBlock {
    uint count;
    uint values[];
} runtimeAtomicBlock;
layout(std430, binding = 20) buffer RuntimeSignedAtomicBlock {
    int count;
    int values[];
} runtimeSignedAtomicBlock;

void main() {
    uint i = runtimeAtomicBlock.count;
    uint oldValue = atomicAdd(runtimeAtomicBlock.values[i], 1u);
    uint swapped = atomicCompSwap(runtimeAtomicBlock.values[i + 1u], oldValue, 7u);
    int j = runtimeSignedAtomicBlock.count;
    int oldSigned = atomicMin(runtimeSignedAtomicBlock.values[j], -2);
    int exchanged = atomicExchange(runtimeSignedAtomicBlock.values[j + 1], oldSigned);
    atomicAdd(runtimeSignedAtomicBlock.values[j], exchanged);
}
"""


MIXED_GLSL_SSBO_UNSUPPORTED_ATOMICS_COMPUTE_SHADER = """
#version 450 core
layout(std430, binding = 21) readonly buffer ReadAtomicBlock {
    uint value;
} readAtomicBlock;
layout(std430, binding = 22) buffer FloatAtomicBlock {
    float value;
} floatAtomicBlock;
layout(std430, binding = 23) buffer VectorAtomicBlock {
    uvec2 value;
} vectorAtomicBlock;
layout(std430, binding = 24) buffer MatrixAtomicBlock {
    mat2 value;
} matrixAtomicBlock;

void main() {
    uint readonlyOld = atomicAdd(readAtomicBlock.value, 1u);
    float floatOld = atomicAdd(floatAtomicBlock.value, 1.0);
    uint vectorOld = atomicAdd(vectorAtomicBlock.value, 1u);
    float matrixOld = atomicAdd(matrixAtomicBlock.value, 1.0);
}
"""


def _fragment_ast():
    return crosstl.translator.parse(FRAGMENT_SMOKE_SHADER)


def _mixed_glsl_ast(source, shader_type):
    tokens = GLSLLexer(source).tokenize()
    glsl_ast = GLSLParser(tokens, shader_type).parse()
    crossgl = GLSLToCrossGLConverter(shader_type=shader_type).generate(glsl_ast)
    return crosstl.translator.parse(crossgl)


def _require_tool(name):
    path = shutil.which(name)
    if not path:
        pytest.skip(f"{name} is not installed")
    return path


def _require_xcrun_tool(name):
    xcrun = _require_tool("xcrun")
    probe = subprocess.run(
        [xcrun, "-sdk", "macosx", "-f", name],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip()
        pytest.skip(f"xcrun cannot locate {name}: {detail}")
    return xcrun


def _run_validator(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=60,
    )
    diagnostics = "\n".join(
        part for part in (result.stdout, result.stderr) if part.strip()
    )
    assert result.returncode == 0, diagnostics


def test_generated_hlsl_fragment_compiles_with_dxc(tmp_path):
    dxc = _require_tool("dxc")
    shader_path = tmp_path / "validator_smoke.hlsl"
    output_path = tmp_path / "validator_smoke.dxil"

    shader_path.write_text(
        HLSLCodeGen().generate_stage(_fragment_ast(), "fragment"),
        encoding="utf-8",
    )

    _run_validator(
        [
            dxc,
            "-T",
            "ps_6_0",
            "-E",
            "PSMain",
            str(shader_path),
            "-Fo",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_generated_hlsl_compute_synchronization_compiles_with_dxc(tmp_path):
    dxc = _require_tool("dxc")
    shader_path = tmp_path / "synchronization.hlsl"
    output_path = tmp_path / "synchronization.dxil"

    code = HLSLCodeGen().generate(
        crosstl.translator.parse(CROSSGL_SYNCHRONIZATION_COMPUTE_SHADER)
    )
    assert code.count("GroupMemoryBarrierWithGroupSync();") == 2
    assert "AllMemoryBarrier();" in code
    assert "workgroupBarrier();" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            dxc,
            "-T",
            "cs_6_0",
            "-E",
            "CSMain",
            str(shader_path),
            "-Fo",
            str(output_path),
        ]
    )
    assert output_path.exists()


@pytest.mark.parametrize(
    ("case_name", "source", "expected_snippets", "forbidden_snippets"),
    [
        (
            "uint",
            MIXED_GLSL_SSBO_UINT_ATOMICS_COMPUTE_SHADER,
            (
                "RWByteAddressBuffer atomicBlock : register(u17);",
                "__crossgl_byteaddress_atomic_compare_exchange_uint",
                "InterlockedAdd",
            ),
            ("unsupported HLSL GLSL buffer block atomic", "#version"),
        ),
        (
            "int",
            MIXED_GLSL_SSBO_INT_ATOMICS_COMPUTE_SHADER,
            (
                "RWByteAddressBuffer signedAtomicBlock : register(u18);",
                "__crossgl_byteaddress_atomic_compare_exchange_int",
                "InterlockedMin",
            ),
            ("unsupported HLSL GLSL buffer block atomic", "#version"),
        ),
        (
            "runtime_array",
            MIXED_GLSL_SSBO_RUNTIME_ARRAY_ATOMICS_COMPUTE_SHADER,
            (
                "runtimeAtomicBlock, (4 + i * 4)",
                "runtimeSignedAtomicBlock, (4 + j * 4)",
                "__crossgl_byteaddress_atomic_compare_exchange_uint",
            ),
            ("unsupported HLSL GLSL buffer block atomic", "#version"),
        ),
        (
            "unsupported",
            MIXED_GLSL_SSBO_UNSUPPORTED_ATOMICS_COMPUTE_SHADER,
            (
                "ByteAddressBuffer readAtomicBlock : register(t21);",
                "unsupported HLSL GLSL buffer block atomic",
                "float floatOld = /* unsupported HLSL GLSL buffer block atomic",
                "*/ 0;",
            ),
            ("Interlocked", "__crossgl_byteaddress_atomic", "#version"),
        ),
    ],
)
def test_mixed_glsl_ssbo_atomics_hlsl_output_compiles_with_dxc(
    tmp_path,
    case_name,
    source,
    expected_snippets,
    forbidden_snippets,
):
    dxc = _require_tool("dxc")
    shader_path = tmp_path / f"mixed_glsl_ssbo_{case_name}_atomics.hlsl"
    output_path = tmp_path / f"mixed_glsl_ssbo_{case_name}_atomics.dxil"

    code = HLSLCodeGen().generate(_mixed_glsl_ast(source, "compute"))
    for snippet in expected_snippets:
        assert snippet in code
    for snippet in forbidden_snippets:
        assert snippet not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            dxc,
            "-T",
            "cs_6_0",
            "-E",
            "CSMain",
            str(shader_path),
            "-Fo",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_generated_glsl_fragment_validates_with_glslangvalidator(tmp_path):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "validator_smoke.frag"

    shader_path.write_text(
        GLSLCodeGen().generate_stage(_fragment_ast(), "fragment"),
        encoding="utf-8",
    )

    _run_validator([glslang, "-S", "frag", str(shader_path)])


def test_generated_glsl_compute_synchronization_validates_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "synchronization.comp"

    code = GLSLCodeGen().generate(
        crosstl.translator.parse(CROSSGL_SYNCHRONIZATION_COMPUTE_SHADER)
    )
    assert code.count("barrier();") == 2
    assert "memoryBarrier();" in code
    assert "workgroupBarrier();" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "comp", str(shader_path)])


def test_generated_glsl_multisample_storage_validates_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "multisample_storage.comp"

    shader_path.write_text(
        GLSLCodeGen().generate_stage(
            crosstl.translator.parse(GLSL_MULTISAMPLE_STORAGE_COMPUTE_SHADER),
            "compute",
        ),
        encoding="utf-8",
    )

    _run_validator([glslang, "-S", "comp", str(shader_path)])


def test_generated_glsl_parameter_image_atomic_specialization_validates_with_glslangvalidator(
    tmp_path,
):
    glslang = _require_tool("glslangValidator")
    shader_path = tmp_path / "parameter_image_atomic.comp"

    code = GLSLCodeGen().generate_stage(
        crosstl.translator.parse(GLSL_PARAMETER_IMAGE_ATOMIC_COMPUTE_SHADER),
        "compute",
    )
    assert "imageAtomicAdd(image, pixel, value)" not in code
    assert "imageAtomicAdd(counters, pixel, value)" in code
    assert "addCounter__glsl_image_counters(ivec2(0, 1), 2u)" in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator([glslang, "-S", "comp", str(shader_path)])


def test_generated_metal_fragment_compiles_with_xcrun_metal(tmp_path):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "validator_smoke.metal"
    output_path = tmp_path / "validator_smoke.air"

    shader_path.write_text(
        MetalCodeGen().generate_stage(_fragment_ast(), "fragment"),
        encoding="utf-8",
    )

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_generated_metal_compute_synchronization_compiles_with_xcrun_metal(tmp_path):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "synchronization.metal"
    output_path = tmp_path / "synchronization.air"

    code = MetalCodeGen().generate(
        crosstl.translator.parse(CROSSGL_SYNCHRONIZATION_COMPUTE_SHADER)
    )
    assert code.count("threadgroup_barrier(mem_flags::mem_threadgroup);") == 2
    assert "threadgroup_barrier(mem_flags::mem_device);" in code
    assert "workgroupBarrier();" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_preprocessor_metal_output_compiles_with_xcrun_metal(tmp_path):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "mixed_glsl_preprocessor.metal"
    output_path = tmp_path / "mixed_glsl_preprocessor.air"

    code = MetalCodeGen().generate(
        _mixed_glsl_ast(MIXED_GLSL_PREPROCESSOR_COMPUTE_SHADER, "compute")
    )
    assert "#version" not in code
    assert "#extension" not in code
    assert "precision highp float" not in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_uint_atomics_metal_output_compiles_with_xcrun_metal(
    tmp_path,
):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "mixed_glsl_ssbo_uint_atomics.metal"
    output_path = tmp_path / "mixed_glsl_ssbo_uint_atomics.air"

    code = MetalCodeGen().generate(
        _mixed_glsl_ast(MIXED_GLSL_SSBO_UINT_ATOMICS_COMPUTE_SHADER, "compute")
    )
    assert "#version" not in code
    assert "atomic_fetch_add_explicit" in code
    assert "__crossgl_buffer_atomic_compare_exchange_uint" in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_int_atomics_metal_output_compiles_with_xcrun_metal(
    tmp_path,
):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "mixed_glsl_ssbo_int_atomics.metal"
    output_path = tmp_path / "mixed_glsl_ssbo_int_atomics.air"

    code = MetalCodeGen().generate(
        _mixed_glsl_ast(MIXED_GLSL_SSBO_INT_ATOMICS_COMPUTE_SHADER, "compute")
    )
    assert "#version" not in code
    assert "atomic_fetch_add_explicit" in code
    assert "reinterpret_cast<device atomic_int*>" in code
    assert "__crossgl_buffer_atomic_compare_exchange_int" in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_runtime_array_atomics_metal_output_compiles_with_xcrun_metal(
    tmp_path,
):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "mixed_glsl_ssbo_runtime_array_atomics.metal"
    output_path = tmp_path / "mixed_glsl_ssbo_runtime_array_atomics.air"

    code = MetalCodeGen().generate(
        _mixed_glsl_ast(
            MIXED_GLSL_SSBO_RUNTIME_ARRAY_ATOMICS_COMPUTE_SHADER,
            "compute",
        )
    )
    assert "#version" not in code
    assert "runtimeAtomicBlock + (4 + i * 4)" in code
    assert "runtimeSignedAtomicBlock + (4 + j * 4)" in code
    assert "__crossgl_buffer_atomic_compare_exchange_uint" in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()


def test_mixed_glsl_ssbo_unsupported_atomics_metal_output_compiles_with_xcrun_metal(
    tmp_path,
):
    xcrun = _require_xcrun_tool("metal")
    shader_path = tmp_path / "mixed_glsl_ssbo_unsupported_atomics.metal"
    output_path = tmp_path / "mixed_glsl_ssbo_unsupported_atomics.air"

    code = MetalCodeGen().generate(
        _mixed_glsl_ast(
            MIXED_GLSL_SSBO_UNSUPPORTED_ATOMICS_COMPUTE_SHADER,
            "compute",
        )
    )
    assert "#version" not in code
    assert "unsupported Metal GLSL buffer block atomic" in code
    assert "atomic_fetch_" not in code
    assert "float floatOld = /* unsupported Metal GLSL buffer block atomic" in code
    assert "*/ 0;" in code
    shader_path.write_text(code, encoding="utf-8")

    _run_validator(
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(shader_path),
            "-o",
            str(output_path),
        ]
    )
    assert output_path.exists()
