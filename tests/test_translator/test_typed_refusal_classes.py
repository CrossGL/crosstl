"""Site refusals carry their component's stable code and keep the built-in base and the message they had.

Every site in ``crosstl/translator`` and ``crosstl/backend`` that rejects a program raises a class from
:mod:`crosstl.translator.refusals` instead of a built-in exception (``tests/conformance/typed_refusals.py`` keeps it
that way). These tests pin the code of each class, check that ``except ValueError`` and ``except SyntaxError``
handlers and messages are unaffected, check how a site refusal combines with refusals recorded before it, and
translate one program per component to check that the code reaches the caller.
"""

import pickle

import pytest

import crosstl
from crosstl.backend.DirectX.DirectxCrossGLCodeGen import HLSLToCrossGLConverter
from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
from crosstl.translator import refusals
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.lowering_refusals import LoweringRefusals
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused
from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

SITE_REFUSALS = {
    "TranslationRefusal": "refused",
    "CrossGLRefusal": "crossgl-refused",
    "MetadataContractError": "metadata-contract-invalid",
    "RegistryRefusal": "registry-refused",
    "TargetRefusal": "target-refused",
    "DirectXRefusal": "directx-refused",
    "OpenGLRefusal": "opengl-refused",
    "MetalRefusal": "metal-refused",
    "VulkanRefusal": "vulkan-refused",
    "WGSLRefusal": "wgsl-refused",
    "SlangRefusal": "slang-refused",
    "CudaRefusal": "cuda-refused",
    "HipRefusal": "hip-refused",
    "MojoRefusal": "mojo-refused",
    "RustRefusal": "rust-refused",
    "DirectXSourceRefusal": "directx-source-refused",
    "OpenGLSourceRefusal": "opengl-source-refused",
    "MetalSourceRefusal": "metal-source-refused",
    "HipSourceRefusal": "hip-source-refused",
}
SYNTAX_ERRORS = {
    "SourceSyntaxError": "syntax-error",
    "CrossGLSyntaxError": "crossgl-syntax-error",
    "DirectXSyntaxError": "directx-syntax-error",
    "OpenGLSyntaxError": "opengl-syntax-error",
    "MetalSyntaxError": "metal-syntax-error",
    "SpirvSyntaxError": "spirv-syntax-error",
    "CudaSyntaxError": "cuda-syntax-error",
    "HipSyntaxError": "hip-syntax-error",
    "SlangSyntaxError": "slang-syntax-error",
    "MojoSyntaxError": "mojo-syntax-error",
    "RustSyntaxError": "rust-syntax-error",
    "OpenCLSyntaxError": "opencl-syntax-error",
}
OTHER_ERRORS = {
    "TranslatorArgumentError": ("invalid-argument", TypeError),
    "TranslatorKeyError": ("unknown-name", KeyError),
    "TranslatorIndexError": ("index-out-of-range", IndexError),
}


def test_every_exported_class_has_a_pinned_code():
    exported = {
        name
        for name in refusals.__all__
        if isinstance(getattr(refusals, name), type)
        and issubclass(getattr(refusals, name), BaseException)
    }
    assert exported == {*SITE_REFUSALS, *SYNTAX_ERRORS, *OTHER_ERRORS}
    codes = [
        *SITE_REFUSALS.values(),
        *SYNTAX_ERRORS.values(),
        *(code for code, _base in OTHER_ERRORS.values()),
    ]
    assert len(codes) == len(set(codes))


@pytest.mark.parametrize("name", sorted(SITE_REFUSALS))
def test_site_refusals_are_value_errors_with_the_message_unchanged(name):
    cls = getattr(refusals, name)
    error = cls("the reason, exactly as written")
    assert isinstance(error, ValueError)
    assert isinstance(error, TranslationDiagnostic)
    assert not isinstance(error, SyntaxError)
    assert str(error) == "the reason, exactly as written"
    assert error.reason == "the reason, exactly as written"
    assert error.project_diagnostic_code == CODE_PREFIX + SITE_REFUSALS[name]
    assert error.target == cls.default_target
    assert error.defers_to_pending
    assert not refusals.supersedes_pending_refusals(error)
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is cls and str(restored) == str(error)


@pytest.mark.parametrize("name", sorted(SYNTAX_ERRORS))
def test_syntax_errors_stay_syntax_errors_and_not_value_errors(name):
    cls = getattr(refusals, name)
    error = cls("Expected ';', got '}'")
    assert isinstance(error, SyntaxError)
    assert not isinstance(error, ValueError)
    assert str(error) == "Expected ';', got '}'"
    assert error.reason == "Expected ';', got '}'"
    assert error.project_diagnostic_code == CODE_PREFIX + SYNTAX_ERRORS[name]
    assert error.to_json()["code"] == error.project_diagnostic_code
    assert not refusals.supersedes_pending_refusals(error)
    located = cls("bad token", ("shader.src", 3, 7, "text"))
    assert (located.lineno, located.offset, located.reason) == (3, 7, "bad token")
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is cls and str(restored) == str(error)


@pytest.mark.parametrize("name", sorted(OTHER_ERRORS))
def test_api_errors_keep_their_builtin_base(name):
    code, base = OTHER_ERRORS[name]
    error = getattr(refusals, name)("message")
    assert isinstance(error, base)
    assert not isinstance(error, ValueError)
    assert error.project_diagnostic_code == CODE_PREFIX + code


def _metal_sink_with(raised):
    sink = LoweringRefusals(MetalLoweringRefused)

    def generate():
        sink.record("first construct")
        raise raised

    return sink, generate


def test_site_refusal_reports_with_the_refusals_recorded_before_it():
    sink, generate = _metal_sink_with(refusals.MetalRefusal("second construct"))
    with pytest.raises(MetalLoweringRefused) as info:
        sink.run(generate)
    assert info.value.reasons == (
        "first construct",
        "generation then failed: second construct",
    )
    assert isinstance(info.value.__cause__, refusals.MetalRefusal)


def test_diagnostic_with_its_own_code_replaces_recorded_refusals():
    own = TranslationDiagnostic(
        "names its construct", code=CODE_PREFIX + "example-unsupported"
    )
    sink, generate = _metal_sink_with(own)
    with pytest.raises(TranslationDiagnostic) as info:
        sink.run(generate)
    assert info.value is own


def test_site_refusal_without_recorded_refusals_propagates():
    sink = LoweringRefusals(MetalLoweringRefused)
    error = refusals.MetalRefusal("only construct")

    def generate():
        raise error

    with pytest.raises(refusals.MetalRefusal) as info:
        sink.run(generate)
    assert info.value is error


GEOMETRY = "shader S { geometry { void main() { } } }"
TESSELLATION_CONTROL = "shader S { tessellation_control { void main() { } } }"
MESH = "shader S { mesh { void main() { SetMeshOutputCounts(1); } } }"
DUPLICATE_UNIFORM = (
    "shader S { uniform float A; uniform int A; "
    "compute { void main() { float b = A; } } }"
)
DUPLICATE_SPEC_ID = """
shader S {
    const int SCALE @constant_id(1) @constant_id(2) = 2;
    compute { void main() { int value = SCALE; } }
}
"""
DUPLICATE_FUNCTION_CONSTANT = """
shader S {
    bool useFast @function_constant(0);
    int mode @constant_id(0);
    fragment {
        vec4 main() @gl_FragColor { return useFast ? vec4(float(mode)) : vec4(0.0); }
    }
}
"""

TRANSLATIONS = [
    # (file name, source, backend, code)
    (
        "bad.cgl",
        "shader S { vertex { void main( { } } }",
        "metal",
        "crossgl-syntax-error",
    ),
    ("geometry.cgl", GEOMETRY, "wgsl", "wgsl-refused"),
    ("geometry.cgl", GEOMETRY, "cuda", "cuda-refused"),
    ("geometry.cgl", GEOMETRY, "hip", "hip-refused"),
    ("geometry.cgl", GEOMETRY, "mojo", "mojo-refused"),
    ("geometry.cgl", GEOMETRY, "rust", "rust-refused"),
    ("control.cgl", TESSELLATION_CONTROL, "slang", "slang-refused"),
    ("mesh.cgl", MESH, "opengl", "opengl-refused"),
    ("duplicate.cgl", DUPLICATE_UNIFORM, "directx", "target-refused"),
    ("spec.cgl", DUPLICATE_SPEC_ID, "vulkan", "vulkan-refused"),
    ("constants.cgl", DUPLICATE_FUNCTION_CONSTANT, "metal", "metal-refused"),
    (
        "bad.hlsl",
        "float4 main( : SV_Target { return 0; }",
        "cgl",
        "directx-syntax-error",
    ),
    ("bad.glsl", "#version 450\nvoid main( { }", "cgl", "opengl-syntax-error"),
    ("bad.cu", "__global__ void k( { }", "cgl", "cuda-syntax-error"),
    ("bad.hip", "__global__ void k( { }", "cgl", "hip-syntax-error"),
    ("bad.cl", "__kernel void k( { }", "cgl", "opencl-syntax-error"),
    ("bad.slang", "void main( { }", "cgl", "slang-syntax-error"),
    ("bad.mojo", "fn main( :\n", "cgl", "mojo-syntax-error"),
    ("bad.rs", "fn main( {", "cgl", "rust-syntax-error"),
    (
        "virtual.hip",
        "struct A { int a; };\nstruct B : virtual A { int b; };\n"
        "__global__ void k(B* p) { p->b = 1; }\n",
        "cgl",
        "hip-source-refused",
    ),
    (
        "redeclared.vert",
        "#version 450\nlayout(location = 0) out float gl_PointSize;\n"
        "layout(location = 1) out float gl_PointSize;\n"
        "void main() { gl_Position = vec4(0.0); }\n",
        "cgl",
        "opengl-source-refused",
    ),
]


@pytest.mark.parametrize(
    "file_name, source, backend, code",
    TRANSLATIONS,
    ids=[f"{name}->{backend}" for name, _source, backend, _code in TRANSLATIONS],
)
def test_translation_reports_the_component_code(
    tmp_path, file_name, source, backend, code
):
    path = tmp_path / file_name
    path.write_text(source, encoding="utf-8")
    with pytest.raises(Exception) as info:
        crosstl.translate(str(path), backend=backend)
    assert info.value.project_diagnostic_code == CODE_PREFIX + code, str(info.value)


@pytest.mark.parametrize(
    "construct, code",
    [
        (lambda: HLSLCodeGen(target_profile="dx9"), "directx-refused"),
        (
            lambda: HLSLToCrossGLConverter(matrix_packing="diagonal"),
            "directx-source-refused",
        ),
        (
            lambda: MetalToCrossGLConverter(preserve_resource_origins="yes"),
            "metal-source-refused",
        ),
    ],
    ids=["hlsl-target-profile", "hlsl-matrix-packing", "metal-resource-origins"],
)
def test_option_refusals_report_the_component_code(construct, code):
    with pytest.raises(ValueError) as info:
        construct()
    assert info.value.project_diagnostic_code == CODE_PREFIX + code
