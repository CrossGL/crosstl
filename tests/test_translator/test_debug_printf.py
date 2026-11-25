"""CrossGL ``printf``: shader debug printing in every direction.

``printf(format, values...)`` is CrossGL's debug print with the Vulkan debug printf format. The GLSL and SPIR-V
importers read GLSL's ``debugPrintfEXT`` and SPIR-V's ``NonSemantic.DebugPrintf`` as it, HLSL and Slang spell it
``printf`` already. SPIR-V emits ``NonSemantic.DebugPrintf``, GLSL ``debugPrintfEXT``, HLSL ``printf`` (which DXC
compiles to SPIR-V) and Metal ``os_log_default.log`` with vector conversions written out per component. WebGL 2 and
WGSL cannot print from a shader and refuse it with a typed diagnostic.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.metal_debug_printf import MetalDebugPrintfError
from crosstl.translator.codegen.spirv_debug_printf import (
    decode_format_literal,
    spirv_string_operand,
)
from crosstl.translator.diagnostics import TranslationDiagnostic
from crosstl.translator.semantics import (
    DebugPrintfUnsupportedError,
    DebugPrintfValueUnsupportedError,
)

GLSL_PRINT = """#version 450
#extension GL_EXT_debug_printf : require
layout(location = 0) in vec3 v;
layout(location = 0) out vec4 o;
void main() {
    debugPrintfEXT("v=%v3f i=%d", v, 3);
    o = vec4(v, 1.0);
}
"""

HLSL_PRINT = """struct VSOutput { float4 Pos : SV_POSITION; };
VSOutput main(float3 pos : POSITION)
{
    VSOutput o;
    o.Pos = float4(pos, 1.0);
    printf("Position = %v3f", pos);
    return o;
}
"""

CROSSGL_PRINT = r"""
shader DebugPrint {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            vec3 v = vec3(1.0, 2.0, 3.0);
            uint id = 4u;
            printf("plain %d %u 100%% \"q\" \\ \n", 3, id);
            printf("v=%v3f id=%u w=%5.2f", v * 2.0, id, v.x);
        }
    }
}
"""


def _compute(body, prelude=""):
    return f"""
shader DebugPrint {{
    {prelude}
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {{
            vec3 v = vec3(1.0, 2.0, 3.0);
            {body}
        }}
    }}
}}
"""


def _translate(tmp_path, name, source, backend):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command, cwd):
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("xcrun has no Metal compiler")
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _validate_spirv(tmp_path, assembly):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    (tmp_path / "out.spvasm").write_text(assembly, encoding="utf-8")
    _run(
        ["spirv-as", "--target-env", "vulkan1.1", "out.spvasm", "-o", "out.spv"],
        tmp_path,
    )
    _run(["spirv-val", "--target-env", "vulkan1.1", "out.spv"], tmp_path)


def _validate_metal(tmp_path, code):
    if not shutil.which("xcrun"):
        pytest.skip("xcrun is not on PATH")
    (tmp_path / "out.metal").write_text(code, encoding="utf-8")
    _run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", "out.metal", "-o", "out.air"],
        tmp_path,
    )


def test_importers_read_debug_prints_as_crossgl_printf(tmp_path):
    from_glsl = _translate(tmp_path, "print.frag", GLSL_PRINT, "cgl")
    from_hlsl = _translate(tmp_path, "print.hlsl", HLSL_PRINT, "cgl")
    assert 'printf("v=%v3f i=%d", input.v, 3);' in from_glsl
    assert "debugPrintfEXT" not in from_glsl
    assert 'printf("Position = %v3f", pos);' in from_hlsl


@pytest.mark.parametrize(
    "name, source",
    [
        ("print.frag", GLSL_PRINT),
        ("print.hlsl", HLSL_PRINT),
        ("print.cgl", CROSSGL_PRINT),
    ],
)
def test_spirv_prints_through_nonsemantic_debug_printf(tmp_path, name, source):
    assembly = _translate(tmp_path, name, source, "vulkan")
    assert 'OpExtension "SPV_KHR_non_semantic_info"' in assembly
    assert assembly.count('OpExtInstImport "NonSemantic.DebugPrintf"') == 1
    assert "printf" not in assembly.replace("DebugPrintf", "")
    _validate_spirv(tmp_path, assembly)


def test_spirv_format_strings_decode_c_escapes(tmp_path):
    assembly = _translate(tmp_path, "print.cgl", CROSSGL_PRINT, "vulkan")
    # glslang stores the decoded characters; SPIR-V assembly escapes only the quote and the backslash.
    assert '= OpString "plain %d %u 100%% \\"q\\" \\\\ \n"' in assembly
    assert '= OpString "v=%v3f id=%u w=%5.2f"' in assembly
    assert decode_format_literal(r"a\tb\x41\101\?") == "a\tbAA?"
    assert decode_format_literal(r"bad \q") is None
    assert spirv_string_operand('say "hi" \\') == '"say \\"hi\\" \\\\"'


@pytest.mark.parametrize(
    "body, message",
    [
        ('bool b = true; printf("%d", b);', "prints only 32- or 64-bit"),
        ('printf("bad \\q %d", 1);', "string-literal format"),
    ],
)
def test_spirv_refuses_prints_it_cannot_carry(tmp_path, body, message):
    with pytest.raises(TranslationDiagnostic, match=message):
        _translate(tmp_path, "print.cgl", _compute(body), "vulkan")


def test_metal_prints_through_os_log_with_vectors_by_component(tmp_path):
    code = _translate(tmp_path, "print.cgl", CROSSGL_PRINT, "metal")
    assert "#include <metal_logging>" in code
    assert 'os_log_default.log("plain %d %u 100%% \\"q\\" \\\\ \\n", 3, id);' in code
    assert "auto _crosstl_log0 = v * 2.0;" in code
    assert (
        'os_log_default.log("v=%f, %f, %f id=%u w=%5.2f", _crosstl_log0.x, '
        "_crosstl_log0.y, _crosstl_log0.z, _crosstl_log1, _crosstl_log2);"
    ) in code
    _validate_metal(tmp_path, code)


@pytest.mark.parametrize(
    "name, source", [("print.frag", GLSL_PRINT), ("print.hlsl", HLSL_PRINT)]
)
def test_metal_compiles_imported_debug_prints(tmp_path, name, source):
    code = _translate(tmp_path, name, source, "metal")
    assert "os_log_default.log(" in code
    _validate_metal(tmp_path, code)


@pytest.mark.parametrize(
    "body, message",
    [
        ('printf("%s", 1);', "no Metal os_log equivalent"),
        ('printf("%d %d", 1);', "has no value to print"),
        ('printf("%d", 1, 2);', "have no conversion"),
        ('printf("%*d", 1, 2);', "width from an argument"),
        ('printf("%v2f", v);', "prints a 3-component value"),
    ],
)
def test_metal_refuses_formats_os_log_cannot_print(tmp_path, body, message):
    with pytest.raises(MetalDebugPrintfError, match=message) as caught:
        _translate(tmp_path, "print.cgl", _compute(body), "metal")
    assert caught.value.code == "project.translate.metal-debug-printf-unsupported"


@pytest.mark.parametrize("backend", ["webgl", "wgsl"])
def test_targets_without_a_shader_print_refuse_it(tmp_path, backend):
    with pytest.raises(DebugPrintfUnsupportedError) as caught:
        _translate(tmp_path, "print.hlsl", HLSL_PRINT, backend)
    assert caught.value.code.endswith("-debug-printf-unsupported")


def _dxc_spirv(tmp_path, code, profile, entry):
    # DXC compiles HLSL printf only to SPIR-V; DXIL has no print operation.
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not on PATH")
    (tmp_path / "out.hlsl").write_text(code, encoding="utf-8")
    _run(
        [dxc, "-spirv", "-fspv-target-env=vulkan1.3", "-T", profile, "-E", entry]
        + ["-HV", "2021", "out.hlsl", "-Fo", "out.spv"],
        tmp_path,
    )


@pytest.mark.parametrize(
    "name, source, expected, profile, entry",
    [
        (
            "print.cgl",
            CROSSGL_PRINT,
            (
                'printf("plain %d %u 100%% \\"q\\" \\\\ \\n", 3, id);',
                'printf("v=%v3f id=%u w=%5.2f", (v * 2.0), id, v.x);',
            ),
            "cs_6_9",
            "CSMain",
        ),
        (
            "print.hlsl",
            HLSL_PRINT,
            ('printf("Position = %v3f", pos);',),
            "vs_6_9",
            "VSMain",
        ),
        (
            "print.frag",
            GLSL_PRINT,
            ('printf("v=%v3f i=%d", input.v, 3);',),
            "ps_6_9",
            "PSMain",
        ),
    ],
)
@pytest.mark.parametrize("backend", ["directx", "dx12"])
def test_hlsl_prints_through_its_printf_intrinsic(
    tmp_path, backend, name, source, expected, profile, entry
):
    code = _translate(tmp_path, name, source, backend)
    for line in expected:
        assert line in code
    assert "debugPrintfEXT" not in code
    _dxc_spirv(tmp_path, code, profile, entry)


def test_a_program_function_named_printf_is_an_ordinary_call(tmp_path):
    source = _compute(
        "float x = printf(v.x);",
        prelude="float printf(float value) { return value * 2.0; }",
    )
    hlsl = _translate(tmp_path, "print.cgl", source, "directx")
    # HLSL output renames the helper apart from the printf intrinsic.
    assert "printf_fn(v.x)" in hlsl


CUDA_PRINT_VALUE = r"""
__global__ void k(int* out) {
    int r = printf("v=%d\n", 1);
    out[threadIdx.x] = r;
}
"""


@pytest.mark.parametrize(
    "body",
    [
        'int r = printf("%d", 1);',
        'int r = 0; r = printf("%d", 1) + 1;',
        'if (printf("%d", 1) > 0) { v.x = 0.0; }',
    ],
)
@pytest.mark.parametrize(
    "backend", ["directx", "opengl", "vulkan", "metal", "slang", "cuda", "hip"]
)
def test_a_debug_print_whose_value_is_read_is_refused(tmp_path, body, backend):
    # Every target's debug print is a statement, and C's printf value is the source language's own.
    with pytest.raises(DebugPrintfValueUnsupportedError) as caught:
        _translate(tmp_path, "print.cgl", _compute(body), backend)
    assert caught.value.code == (
        f"project.translate.{backend}-debug-printf-value-unsupported"
    )


def test_a_cuda_print_value_is_kept_only_for_cuda(tmp_path):
    code = _translate(tmp_path, "print.cu", CUDA_PRINT_VALUE, "cuda")
    assert 'int r = printf("v=%d\\n", 1);' in code
    for backend in ("hip", "directx", "opengl", "vulkan"):
        with pytest.raises(DebugPrintfValueUnsupportedError, match="cannot be carried"):
            _translate(tmp_path, "print.cu", CUDA_PRINT_VALUE, backend)


def test_prints_evaluated_for_their_effects_are_statements(tmp_path):
    body = 'int i = 0; for (; i < 2; printf("i=%d", i), i++) { printf("%d", i); }'
    assembly = _translate(tmp_path, "print.cgl", _compute(body), "vulkan")
    assert assembly.count('OpExtInstImport "NonSemantic.DebugPrintf"') == 1
    _validate_spirv(tmp_path, assembly)
    hlsl = _translate(tmp_path, "print.cgl", _compute(body), "directx")
    assert 'printf("%d", i);' in hlsl
