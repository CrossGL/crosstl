import shutil
import subprocess

import pytest

import crosstl
from crosstl.project import (
    load_project_config,
    translate_project,
    validate_project_report,
)
from crosstl.translator.codegen.GLSL_codegen import (
    GLSLCodeGen,
    OpenGLIndexTypeError,
)
from crosstl.translator.codegen.index_range_contracts import (
    INDEX_RANGE_STATIC,
    OPENGL_INDEX_PROFILE,
    WEBGL_INDEX_PROFILE,
    IntegerRange,
    decide_index_narrowing,
)
from crosstl.translator.codegen.webgl_codegen import (
    WebGLCodeGen,
    WebGLIndexTypeError,
)


def _parse(shader):
    return crosstl.translator.parse(shader)


def _validate_compute_if_available(source, tmp_path, name):
    validator = shutil.which("glslangValidator")
    if validator is None:
        return False
    path = tmp_path / f"{name}.comp"
    path.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [validator, "-S", "comp", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return True


def test_profiles_publish_backend_neutral_32_bit_scalar_contracts():
    scalar_contracts = [
        (item.name, item.signed, item.bits)
        for item in OPENGL_INDEX_PROFILE.scalar_types
    ]
    assert scalar_contracts == [
        ("int", True, 32),
        ("uint", False, 32),
    ]
    assert WEBGL_INDEX_PROFILE.scalar_types == OPENGL_INDEX_PROFILE.scalar_types

    accepted = decide_index_narrowing(
        source_signed=False,
        source_bits=64,
        source_width=1,
        profile=OPENGL_INDEX_PROFILE,
        value_range=IntegerRange(0, 15, INDEX_RANGE_STATIC),
    )
    rejected = decide_index_narrowing(
        source_signed=False,
        source_bits=64,
        source_width=1,
        profile=OPENGL_INDEX_PROFILE,
        value_range=None,
    )
    assert (accepted.action, accepted.target_type.name) == ("convert", "uint")
    assert (rejected.action, rejected.reason) == ("reject", "index-range-unproven")


def test_opengl_fails_closed_for_unproven_wide_runtime_index():
    shader = """
    shader UnprovenWideIndex {
        StructuredBuffer<uint> values @ binding(0);

        uint readValue(uint64_t index) {
            return buffer_load(values, index);
        }
    }
    """

    with pytest.raises(OpenGLIndexTypeError) as exc_info:
        GLSLCodeGen().generate(_parse(shader))

    diagnostic = exc_info.value
    assert diagnostic.index_type == "uint64_t"
    assert diagnostic.target_index_type == "uint"
    assert diagnostic.indexed_value == "values"
    assert diagnostic.reason == "index-range-unproven"


def test_opengl_rejects_negative_wide_constant_before_narrowing():
    shader = """
    shader NegativeWideIndex {
        uint readValue() {
            uint values[4];
            return values[int64_t(-1)];
        }
    }
    """

    with pytest.raises(OpenGLIndexTypeError) as exc_info:
        GLSLCodeGen().generate(_parse(shader))

    diagnostic = exc_info.value
    assert diagnostic.index_type == "int64_t"
    assert diagnostic.target_index_type == "int"
    assert diagnostic.indexed_value == "values"
    assert diagnostic.reason == "negative-index"


def test_opengl_normalizes_fixed_arrays_vectors_matrices_and_bounded_values(tmp_path):
    shader = """
    shader FixedIndexKinds {
        RWStructuredBuffer<uint> output @ binding(0);

        compute {
            [numthreads(1, 1, 1)]
            void main() {
                uint values[4];
                vec4 vectorValue = vec4(1.0);
                mat4 matrixValue = mat4(1.0);
                values[uint64_t(7u) & uint64_t(3u)] = uint(vectorValue[uint64_t(2u)]);
                uint result = values[uint64_t(1u)];
                result += uint(matrixValue[int64_t(2)][0].x);
                buffer_store(output, 0, result);
            }
        }
    }
    """

    generated = GLSLCodeGen().generate(_parse(shader))

    assert "values[uint((uint64_t(7u) & uint64_t(3u)))]" in generated
    assert "vectorValue[2u]" in generated
    assert "values[1u]" in generated
    assert "matrixValue[2][0]" in generated
    _validate_compute_if_available(generated, tmp_path, "fixed_index_kinds")


def test_opengl_asserted_runtime_ssbo_index_is_converted_and_evaluated_once(tmp_path):
    shader = """
    shader RuntimeIndex {
        StructuredBuffer<uint> values @ binding(0);
        RWStructuredBuffer<uint> output @ binding(1);

        compute {
            [numthreads(1, 1, 1)]
            void main() {
                uint64_t index = uint64_t(0u);
                buffer_store(output, 0, buffer_load(values, index++));
            }
        }
    }
    """
    generated = (
        GLSLCodeGen()
        .set_index_range_assertions(
            [{"expression": "index++", "function": "main", "minimum": 0, "maximum": 31}]
        )
        .generate(_parse(shader))
    )

    assert generated.count("index++") == 1
    assert "values[uint((index++))]" in generated
    _validate_compute_if_available(generated, tmp_path, "runtime_index_once")


def test_opengl_normalizes_texture_buffer_and_nested_resource_alias_indices(tmp_path):
    texture_shader = """
    shader TextureBufferIndex {
        samplerBuffer values @ binding(0);
        vec4 readValue() { return texelFetch(values, uint64_t(2u)); }
    }
    """
    texture_generated = GLSLCodeGen().generate(_parse(texture_shader))
    assert "texelFetch(values, 2)" in texture_generated
    _validate_compute_if_available(texture_generated, tmp_path, "texture_buffer_index")

    runtime_texture_shader = """
    shader RuntimeTextureBufferIndex {
        samplerBuffer values @ binding(0);
        vec4 readValue(uint64_t index) { return texelFetch(values, index); }
    }
    """
    runtime_texture_generated = (
        GLSLCodeGen()
        .set_index_range_assertions(
            [{"expression": "index", "minimum": 0, "maximum": 31}]
        )
        .generate(_parse(runtime_texture_shader))
    )
    assert "texelFetch(values, int(index))" in runtime_texture_generated
    _validate_compute_if_available(
        runtime_texture_generated,
        tmp_path,
        "runtime_texture_buffer_index",
    )

    alias_shader = """
    shader NestedResourceAliasIndex {
        RWStructuredBuffer<uint> values[2] @ binding(0);
        uint readLeaf(RWStructuredBuffer<uint> leaf[], uint64_t which) {
            return buffer_load(leaf[which], 0);
        }
        uint readNested(RWStructuredBuffer<uint> nested[], uint64_t which) {
            return readLeaf(nested, which);
        }
        uint run(uint64_t which) { return readNested(values, which); }
    }
    """
    generated = (
        GLSLCodeGen()
        .set_index_range_assertions(
            [{"expression": "which", "minimum": 0, "maximum": 1}]
        )
        .generate(_parse(alias_shader))
    )
    assert "uint(which)" in generated
    assert "buffer_load" not in generated


@pytest.mark.parametrize(
    ("index", "reason"),
    [
        ("int64_t(-1)", "negative-index"),
        ("uint64_t(4294967296)", "constant-index-out-of-range"),
        ("uint64_t(4u)", "constant-index-out-of-range"),
    ],
)
def test_opengl_rejects_negative_and_out_of_range_fixed_indices(index, reason):
    shader = f"""
    shader InvalidConstantIndex {{
        uint readValue() {{
            uint values[4];
            return values[{index}];
        }}
    }}
    """
    with pytest.raises(OpenGLIndexTypeError) as exc_info:
        GLSLCodeGen().generate(_parse(shader))
    assert exc_info.value.reason == reason
    assert exc_info.value.range_status == "out-of-range"


def test_webgl_uses_its_profile_and_fails_closed_for_runtime_wide_index():
    constant_shader = """
    shader WebGLConstantIndex {
        uint readValue() { uint values[4]; return values[uint64_t(2u)]; }
    }
    """
    generated = WebGLCodeGen().generate(_parse(constant_shader))
    assert "#version 300 es" in generated
    assert "values[2u]" in generated
    assert "uint64_t" not in generated

    runtime_shader = """
    shader WebGLRuntimeIndex {
        uint readValue(uint64_t index) {
            uint values[4];
            return values[index];
        }
    }
    """
    with pytest.raises(WebGLIndexTypeError) as exc_info:
        WebGLCodeGen().generate(_parse(runtime_shader))
    assert exc_info.value.target_profile == "WebGL 2 / GLSL ES 3.00"
    assert exc_info.value.reason == "index-range-unproven"


def test_project_range_assertion_and_unproven_diagnostic_contract(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = """shader ProjectIndex {
    StructuredBuffer<uint> values @ binding(0);
    uint readValue(uint64_t index) { return buffer_load(values, index); }
}
"""
    (repo / "index.cgl").write_text(source, encoding="utf-8")
    (repo / "crosstl.toml").write_text(
        """[project]
include = ["index.cgl"]
targets = ["opengl"]
output_dir = "out"

[[project.index_range_assertions]]
source = "index.cgl"
function = "readValue"
expression = "index"
minimum = 0
maximum = 31
""",
        encoding="utf-8",
    )

    config = load_project_config(repo)
    project_report = translate_project(config, format_output=False)
    report = project_report.to_json()
    assert report["project"]["indexRangeAssertions"] == [
        {
            "source": "index.cgl",
            "function": "readValue",
            "expression": "index",
            "minimum": 0,
            "maximum": 31,
        }
    ]
    assert report["project"]["indexRangeAssertionCount"] == 1
    report_path = repo / "report.json"
    project_report.write_json(report_path)
    assert validate_project_report(report_path)["success"] is True
    generated = (repo / report["artifacts"][0]["path"]).read_text(encoding="utf-8")
    assert "values[uint(index)]" in generated

    unproven = translate_project(
        type(config)(
            root=repo,
            include_patterns=("index.cgl",),
            targets=("opengl",),
            output_dir="unproven",
        ),
        format_output=False,
    ).to_json()
    diagnostic = next(
        item
        for item in unproven["diagnostics"]
        if item["code"] == "project.translate.opengl-index-type-unsupported"
    )
    conversion = diagnostic["details"]["indexConversion"]
    assert conversion["sourceType"] == "uint64_t"
    assert conversion["targetProfile"].startswith("OpenGL #version")
    assert conversion["indexedValue"] == "values"
    assert conversion["rangeStatus"] == "unproven"
    assert diagnostic["location"]["line"] == 3
    assert diagnostic["location"]["column"] > 1
