"""Slang specialization constants keep their SpecId through CrossGL (issue #3792).

The importer dropped ``[vk::constant_id(N)]`` and ``[SpecializationConstant]``, so the pipeline's override no longer
reached the constant. Both now become CrossGL ``@constant_id(N)``. slangc numbers an automatic constant with the
lowest id that no explicit ``vk::constant_id`` and no earlier declaration holds. When slangc, glslangValidator and
spirv-dis are installed, the SpecIds of slangc's SPIR-V are compared with those of the CrossGL program's GLSL.
"""

import re
import shutil
import subprocess

import pytest

import crosstl.translator as cgl_translator
from crosstl.backend.import_refusals import SlangImportRefused
from crosstl.backend.slang import SlangCrossGLCodeGen, SlangLexer, SlangParser
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen

ENTRY = """
RWStructuredBuffer<float> outp;
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {{
    outp[id.x] = {value};
}}
"""


def to_crossgl(code):
    ast = SlangParser(SlangLexer(code).tokenize()).parse()
    return SlangCrossGLCodeGen.SlangToCrossGLConverter().generate(ast)


def module(declarations, value):
    return declarations + ENTRY.format(value=value)


def crossgl_ids(code):
    return {
        name: int(spec_id)
        for name, spec_id in re.findall(r"\b(\w+) @constant_id\((\d+)\)", code)
    }


def spirv_ids(disassembly):
    names = dict(re.findall(r'OpName %(\w+) "(\w+)"', disassembly))
    return {
        names.get(target, target): int(spec_id)
        for target, spec_id in re.findall(
            r"OpDecorate %(\w+) SpecId (\d+)", disassembly
        )
    }


PROGRAMS = {
    "explicit": (
        "[vk::constant_id(4)] const int scale = 3;\n"
        "[vk::constant_id(7)] const bool negate = true;\n"
        "[vk::constant_id(2)] const float gain = 0.5;\n"
        "[vk::constant_id(9)] const uint count = 2;\n",
        "(negate ? -1.0 : 1.0) * float(scale) * gain * float(count)",
        {"scale": 4, "negate": 7, "gain": 2, "count": 9},
    ),
    "automatic": (
        "[SpecializationConstant] const int a = 3;\n"
        "[vk::specialization_constant] const bool b = false;\n",
        "float(a) + (b ? 1.0 : 0.0)",
        {"a": 0, "b": 1},
    ),
    # Explicit ids are reserved first, and an unused automatic constant still takes its id.
    "mixed": (
        "[SpecializationConstant] const int a = 3;\n"
        "[SpecializationConstant] const int unused = 9;\n"
        "[vk::constant_id(1)] const int b = 2;\n"
        "[SpecializationConstant] const bool c = true;\n"
        "[vk::constant_id(0)] const int d = 5;\n",
        "(c ? 1.0 : 0.0) + float(b) + float(a) + float(d)",
        {"a": 2, "unused": 3, "b": 1, "c": 4, "d": 0},
    ),
    "declarators": (
        "[SpecializationConstant] const int a = 3, b = 5;\n",
        "float(a) + float(b)",
        {"a": 0, "b": 1},
    ),
    "dependent": (
        "[vk::constant_id(0)] const int scale = 3;\n"
        "static const int doubled = scale * 2 + 1;\n",
        "float(doubled)",
        {"scale": 0},
    ),
}


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_specialization_constants_keep_their_ids(name):
    declarations, value, expected = PROGRAMS[name]
    code = to_crossgl(module(declarations, value))

    assert crossgl_ids(code) == expected
    cgl_translator.parse(code)


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_spirv_output_decorates_the_slang_ids(name):
    declarations, value, expected = PROGRAMS[name]
    code = to_crossgl(module(declarations, value))
    spirv = VulkanSPIRVCodeGen().generate(cgl_translator.parse(code))

    assert spirv_ids(spirv) == expected


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_ids_match_slangc(tmp_path, name):
    tools = [shutil.which(tool) for tool in ("slangc", "glslangValidator", "spirv-dis")]
    if None in tools:
        pytest.skip("slangc, glslangValidator and spirv-dis are required")
    slangc, glslang, spirv_dis = tools
    declarations, value, _ = PROGRAMS[name]
    source = tmp_path / "probe.slang"
    source.write_text(module(declarations, value), encoding="utf-8")
    native = tmp_path / "native.spvasm"
    _run(
        [
            slangc,
            str(source),
            "-target",
            "spirv-asm",
            "-entry",
            "main",
            "-o",
            str(native),
        ]
    )

    glsl = tmp_path / "translated.comp"
    glsl.write_text(
        GLSLCodeGen().generate(cgl_translator.parse(to_crossgl(source.read_text()))),
        encoding="utf-8",
    )
    binary = tmp_path / "translated.spv"
    _run([glslang, "-V", str(glsl), "-o", str(binary)])
    translated = _run([spirv_dis, str(binary)])

    native_ids = spirv_ids(native.read_text(encoding="utf-8"))
    translated_ids = spirv_ids(translated)
    # slangc drops a constant the entry point never reads; glslang keeps it with the id it was given.
    assert native_ids and set(native_ids) <= set(translated_ids)
    assert {key: translated_ids[key] for key in native_ids} == native_ids
    assert translated_ids == PROGRAMS[name][2]


@pytest.mark.parametrize(
    "declarations,reason",
    [
        (
            "[vk::constant_id(-1)] const int a = 3;\n",
            "vk::constant_id(-1); CrossGL takes one integer id",
        ),
        (
            "static const int N = 2;\n[vk::constant_id(N)] const int a = 3;\n",
            "vk::constant_id(N)",
        ),
        (
            "[vk::constant_id(2147483648)] const int a = 3;\n",
            "vk::constant_id(2147483648)",
        ),
        (
            "[vk::constant_id(1)] [SpecializationConstant] const int a = 3;\n",
            "more than one vk::constant_id or SpecializationConstant attribute",
        ),
        (
            "[vk::constant_id(1)] static const int a = 3;\n",
            "a is not a non-static const global",
        ),
        (
            "[SpecializationConstant] int a = 3;\n",
            "a is not a non-static const global",
        ),
    ],
)
def test_misstated_specialization_constants_are_refused(declarations, reason):
    with pytest.raises(SlangImportRefused) as error:
        to_crossgl(module(declarations, "float(a)"))

    assert reason in str(error.value)
    assert "project.translate.slang-import-unsupported" in str(error.value)


@pytest.mark.parametrize(
    "declarations,numthreads,named",
    [
        (
            "[vk::constant_id(0)] const int size = 8;\n",
            "size, 1, 1",
            "x dimension size",
        ),
        (
            "[SpecializationConstant] const int size = 8;\n"
            "static const int twice = size * 2;\n",
            "4, twice, 1",
            "y dimension twice",
        ),
    ],
)
def test_specialized_workgroup_size_is_refused(declarations, numthreads, named):
    source = declarations + (
        "RWStructuredBuffer<float> outp;\n"
        '[shader("compute")]\n'
        f"[numthreads({numthreads})]\n"
        "void main(uint3 id : SV_DispatchThreadID) { outp[id.x] = 1.0; }\n"
    )
    with pytest.raises(SlangImportRefused) as error:
        to_crossgl(source)

    assert f"numthreads {named} depends on specialization constant" in str(error.value)


def test_numthreads_from_a_plain_constant_is_kept():
    code = to_crossgl(
        "static const int SIZE = 8;\n"
        "[vk::constant_id(0)] const int scale = 3;\n"
        "RWStructuredBuffer<float> outp;\n"
        '[shader("compute")]\n'
        "[numthreads(SIZE, 1, 1)]\n"
        "void main(uint3 id : SV_DispatchThreadID) { outp[id.x] = float(scale); }\n"
    )

    assert "layout(local_size_x = SIZE, local_size_y = 1, local_size_z = 1) in;" in code
    assert crossgl_ids(code) == {"scale": 0}
