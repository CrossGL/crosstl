"""Structures declared in a function body.

GLSL, HLSL, MSL, CUDA, HIP and Slang scope such a structure to its block, and WGSL and Mojo declare structures
only at module scope. The CrossGL parser used to reject the declaration, the HLSL importer dropped it and the Slang
importer stopped at it, so every target refused the function. CrossGL now reads it as a statement, and
``semantics.local_structs`` moves it to module scope under a fresh name for every target.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.import_refusals import DirectXImportRefused, SlangImportRefused
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import LocalStructUnsupportedError
from crosstl.translator.semantics.local_structs import hoist_local_structs

KERNEL = """
shader P {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;

        void main() {
            uint i = gl_GlobalInvocationID.x;
            struct Span {
                float lo;
                float hi;
            };
            Span s;
            s.lo = outp[i];
            s.hi = 2.0;
            outp[i] = s.lo + s.hi;
        }
    }
}
"""

SCOPES = """
shader P {
    struct Span {
        int lo;
    };

    float width(float a, float b) {
        struct Span {
            float lo;
            float hi;
        };
        struct Pair {
            Span first;
            Span second[2];
        } pair;
        pair.first = Span(a, b);
        pair.second[0] = pair.first;
        pair.second[1] = Span(b, a);
        float total = 0.0;
        for (int k = 0; k < 2; k++) {
            struct Span {
                int lo;
                int hi;
            };
            Span inner;
            inner.lo = k;
            inner.hi = k + 1;
            total += float(inner.hi - inner.lo);
        }
        return pair.first.hi - pair.first.lo + pair.second[1].lo + total;
    }

    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;

        void main() {
            uint i = gl_GlobalInvocationID.x;
            Span g;
            g.lo = 3;
            struct Span2 {
                float v;
            } a, b;
            a.v = width(1.0, 4.0);
            b.v = float(g.lo);
            outp[i] = a.v + b.v;
        }
    }
}
"""

HLSL_KERNEL = """
RWStructuredBuffer<float> outp : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    struct Span {
        float lo;
        float hi;
    } s, t;
    s.lo = outp[id.x];
    s.hi = 2.0;
    t = s;
    outp[id.x] = t.lo + t.hi;
}
"""

SLANG_KERNEL = """
RWStructuredBuffer<float> outp;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    struct Span {
        float lo;
        float hi;
    };
    Span s;
    s.lo = outp[id.x];
    s.hi = 2.0;
    outp[id.x] = s.lo + s.hi;
}
"""


def _translate(tmp_path, source, backend, suffix="cgl"):
    path = tmp_path / f"shader.{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _flat(code):
    return " ".join(code.split())


def _parse(source):
    return Parser(Lexer(source).get_tokens(), strict_function_bodies=True).parse()


def _run(command, cwd):
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=cwd, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _validate_glsl(tmp_path, code):
    validator = shutil.which("glslangValidator")
    if validator is not None:
        (tmp_path / "out.comp").write_text(code)
        _run([validator, "out.comp"], tmp_path)


def _validate_metal(tmp_path, code):
    if shutil.which("xcrun") is not None:
        (tmp_path / "out.metal").write_text(code)
        _run(["xcrun", "metal", "-c", "out.metal", "-o", "out.air"], tmp_path)


def test_parser_reads_a_structure_declared_in_a_block():
    shader = _parse(KERNEL)
    body = next(iter(shader.stages.values())).entry_point.body
    statements = getattr(body, "statements", body)
    struct = statements[1]
    assert type(struct).__name__ == "StructNode"
    assert struct.name == "Span"
    assert [member.name for member in struct.members] == ["lo", "hi"]


def test_scopes_and_declarators_are_renamed_where_they_apply():
    shader = _parse(SCOPES)
    assert hoist_local_structs(shader, "opengl") == 4
    assert [struct.name for struct in shader.structs] == [
        "Span",
        "width_Span",
        "width_Pair",
        "width_Span_1",
    ]
    assert [
        struct.name for struct in next(iter(shader.stages.values())).local_structs
    ] == ["main_Span2"]


def test_glsl_hoists_each_structure_under_a_fresh_name(tmp_path):
    code = _translate(tmp_path, SCOPES, "opengl")
    flat = _flat(code)
    assert "struct Span { int lo; };" in flat, code
    assert "struct width_Span { float lo; float hi; };" in flat, code
    assert "struct width_Pair { width_Span first; width_Span second[2]; };" in flat
    assert "struct width_Span_1 { int lo; int hi; };" in flat, code
    assert "width_Pair pair; pair.first = width_Span(a, b);" in flat, code
    assert "pair.second[1] = width_Span(b, a);" in flat, code
    assert "width_Span_1 inner;" in flat, code
    # The stage's own ``Span`` is the module-scope one; the declarators follow their structure.
    assert "Span g; g.lo = 3; main_Span2 a; main_Span2 b;" in flat, code
    _validate_glsl(tmp_path, code)


def test_wgsl_declares_the_structure_at_module_scope(tmp_path):
    code = _translate(tmp_path, KERNEL, "wgsl")
    flat = _flat(code)
    assert "struct main_Span { lo: f32, hi: f32, };" in flat, code
    assert "var s: main_Span;" in flat, code
    naga = shutil.which("naga")
    if naga is not None:
        (tmp_path / "out.wgsl").write_text(code)
        _run([naga, "out.wgsl"], tmp_path)


def test_metal_compiles_the_nested_scopes(tmp_path):
    code = _translate(tmp_path, SCOPES, "metal")
    assert re.search(r"struct width_Span_1 \{", code), code
    _validate_metal(tmp_path, code)


def test_spirv_declares_the_structure(tmp_path):
    code = _translate(tmp_path, SCOPES, "vulkan")
    assert 'OpName %width_Pair "width_Pair"' in code or '"width_Pair"' in code, code
    assembler = shutil.which("spirv-as")
    validator = shutil.which("spirv-val")
    if assembler is None or validator is None:
        return
    (tmp_path / "out.spvasm").write_text(code)
    _run(
        [assembler, "out.spvasm", "-o", "out.spv", "--target-env", "vulkan1.0"],
        tmp_path,
    )
    _run([validator, "out.spv"], tmp_path)


@pytest.mark.parametrize("backend", ["directx", "slang", "cuda", "hip", "rust", "mojo"])
def test_every_target_declares_the_hoisted_structure(tmp_path, backend):
    code = _translate(tmp_path, KERNEL, backend)
    assert "main_Span" in code, code
    assert "StructNode" not in code, code


def test_slang_writes_structures_declared_in_a_stage_block(tmp_path):
    source = KERNEL.replace(
        "        void main() {",
        "        struct Range {\n            float lo;\n        };\n        void main() {",
    ).replace(
        "outp[i] = s.lo + s.hi;", "Range r;\n r.lo = s.lo;\n outp[i] = r.lo + s.hi;"
    )
    code = _translate(tmp_path, source, "slang")
    flat = _flat(code)
    assert "struct Range { float lo; };" in flat, code
    assert "struct main_Span { float lo; float hi; };" in flat, code
    slangc = shutil.which("slangc")
    if slangc is not None:
        (tmp_path / "out.slang").write_text(code)
        _run([slangc, "out.slang", "-target", "spirv", "-o", "out.spv"], tmp_path)


def test_a_structure_using_a_function_local_name_is_refused(tmp_path):
    source = (
        KERNEL.replace(
            "            struct Span {\n                float lo;",
            "            const int N = 2;\n            struct Span {\n                float lo[N];",
        )
        .replace("s.lo = outp[i];", "s.lo[0] = outp[i];")
        .replace("s.lo + s.hi", "s.lo[0] + s.hi")
    )
    with pytest.raises(LocalStructUnsupportedError, match="uses 'N', which 'main'"):
        _translate(tmp_path, source, "wgsl")


def test_hlsl_import_keeps_the_structure_and_its_declarators(tmp_path):
    cgl = _translate(tmp_path, HLSL_KERNEL, "cgl", suffix="hlsl")
    flat = _flat(cgl)
    assert "struct Span { float lo; float hi; }; Span s; Span t;" in flat, cgl
    assert "Unhandled" not in cgl, cgl
    _validate_metal(tmp_path, _translate(tmp_path, HLSL_KERNEL, "metal", "hlsl"))
    _validate_glsl(tmp_path, _translate(tmp_path, HLSL_KERNEL, "opengl", "hlsl"))


def test_hlsl_import_refuses_a_local_structure_with_a_base_class(tmp_path):
    source = HLSL_KERNEL.replace(
        "    struct Span {",
        "    struct Base { float b; };\n    struct Span : Base {",
    )
    with pytest.raises(DirectXImportRefused, match="base classes"):
        _translate(tmp_path, source, "cgl", suffix="hlsl")


def test_slang_import_keeps_the_structure(tmp_path):
    cgl = _translate(tmp_path, SLANG_KERNEL, "cgl", suffix="slang")
    assert "struct Span { float lo; float hi; }; Span s;" in _flat(cgl), cgl
    _validate_metal(tmp_path, _translate(tmp_path, SLANG_KERNEL, "metal", "slang"))
    _validate_glsl(tmp_path, _translate(tmp_path, SLANG_KERNEL, "opengl", "slang"))


def test_slang_import_refuses_a_local_structure_with_methods(tmp_path):
    source = SLANG_KERNEL.replace(
        "        float hi;\n    };",
        "        float hi;\n        float width() { return hi - lo; }\n    };",
    )
    with pytest.raises(SlangImportRefused, match="member functions"):
        _translate(tmp_path, source, "cgl", suffix="slang")
