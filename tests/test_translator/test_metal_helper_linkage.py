"""Preserve independent helper definitions when Metal artifacts are linked."""

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

ROOT = Path(__file__).resolve().parents[2]
REQUIRE_ENV = "CROSTL_REQUIRE_METAL_HELPER_LINKAGE"


def _source(name, expression):
    return f"""
    #include <metal_stdlib>
    using namespace metal;
    static float transform(float x) {{ return {expression}; }}
    namespace {{
        float hidden(float x) {{ return {expression}; }}
        template <typename T> T local_transform(T x) {{ return T({expression}); }}
    }}
    inline float shared(float x) {{ return x + 3.0f; }}
    template <typename T> T twice(T x) {{ return x * T(2); }}
    struct Offset {{
        float bias;
        Offset(float x) : bias(x) {{}}
        float apply(float x) const {{ return bias + x; }}
    }};
    struct Op {{ float operator()(float x) const {{ return x + 4.0f; }} }};
    [[visible]] float exported_{name}(float x) {{ return x + 10.0f; }}
    kernel void {name}(device float* output [[buffer(0)]], uint lane [[thread_position_in_grid]]) {{
        output[0] = transform(2.0f);
        output[1] = hidden(2.0f);
        Offset offset(3.0f);
        output[2] = offset.apply(2.0f);
        output[3] = shared(2.0f);
        output[4] = exported_{name}(2.0f);
        output[5] = Op{{}}(2.0f);
        float2 values = float2(7.0f, 8.0f);
        output[6] = values[lane % 2u];
        output[7] = precise::acos(0.0f);
        output[8] = twice<float>(3.0f);
        output[9] = local_transform<float>(2.0f);
    }}
    """


def _translate(tmp_path, source, target="metal"):
    path = tmp_path / "source.metal"
    path.write_text(source, encoding="utf-8")
    return translate(str(path), backend=target, format_output=False)


@pytest.mark.parametrize(
    "qualifier", ["static", "inline", "static inline", "constexpr"]
)
def test_source_function_qualifiers_survive_intermediate_ir(tmp_path, qualifier):
    source = f"{qualifier} float helper(float x) {{ return x + 1.0f; }}"
    intermediate = _translate(tmp_path, source, "crossgl")
    ast = Parser(Lexer(intermediate).get_tokens()).parse()
    function = ast.functions[0]
    expected = qualifier.replace("constexpr", "inline").split()
    assert function.linkage == ("internal" if "static" in expected else "external")
    assert function.is_inline == ("inline" in expected)
    assert not function.qualifiers
    assert not function.attributes
    generated = _translate(tmp_path, source)
    assert f"{' '.join(expected)} float helper(" in generated


@pytest.mark.parametrize("qualifier", ["static", "inline"])
def test_generic_function_linkage_metadata_parses(tmp_path, qualifier):
    source = f"template <typename T> {qualifier} T helper(T x) {{ return x; }}"
    intermediate = _translate(tmp_path, source, "crossgl")
    ast = Parser(Lexer(intermediate).get_tokens()).parse()
    assert ast.functions[0].linkage == (
        "internal" if qualifier == "static" else "external"
    )
    assert ast.functions[0].is_inline is True


@pytest.mark.parametrize("qualifier", ["", "inline", "static", "constexpr"])
def test_materialized_template_helpers_keep_repeatable_linkage(tmp_path, qualifier):
    source = f"""
    template <typename T> {qualifier} T twice(T x) {{ return x * T(2); }}
    kernel void first(device float* output [[buffer(0)]]) {{
        output[0] = twice<float>(3.0f);
    }}
    """
    generated = _translate(tmp_path, source)
    prefix = "static inline" if qualifier == "static" else "inline"
    assert f"{prefix} float twice_float(" in generated
    assert "inline inline" not in generated
    assert "inline kernel" not in generated


def test_materialized_template_entry_remains_exported(tmp_path):
    source = """
    template <typename T> kernel void write(device T* output [[buffer(0)]]) {
        output[0] = T(2);
    }
    template [[host_name("write_float")]] kernel void write<float>(device float*);
    """
    generated = _translate(tmp_path, source)
    assert "kernel void write_float(" in generated
    assert "inline kernel" not in generated
    assert "static kernel" not in generated


@pytest.mark.parametrize(
    "scope, call",
    [
        ("namespace { BODY }", "local_transform"),
        ("namespace named { namespace { BODY } }", "named::local_transform"),
        ("namespace { namespace nested { BODY } }", "nested::local_transform"),
    ],
)
def test_materialized_anonymous_template_remains_private(tmp_path, scope, call):
    helper = "template <typename T> T local_transform(T x) { return x + T(1); }"
    source = scope.replace("BODY", helper) + f"""
    kernel void first(device float* output [[buffer(0)]]) {{
        output[0] = {call}<float>(2.0f);
    }}
    """
    generated = _translate(tmp_path, source)
    assert re.search(
        r"^static inline float \w*local_transform_float\(", generated, re.MULTILINE
    )


@pytest.mark.parametrize("qualifier", ["static", "inline"])
def test_function_linkage_metadata_rejects_arguments(qualifier):
    with pytest.raises(SyntaxError, match="does not accept arguments"):
        Parser(
            Lexer(
                f"float helper() @metal_{qualifier}(1) {{ return 1.0; }}"
            ).get_tokens()
        ).parse()


def test_anonymous_namespace_linkage_is_scoped():
    source = """
    namespace { namespace nested { float hidden() { return 1.0f; } } }
    namespace named { float exported() { return 2.0f; } }
    float public_helper() { return 3.0f; }
    """
    ast = MetalParser(MetalLexer(source).tokenize()).parse()
    assert [
        (function.name, function.internal_linkage) for function in ast.functions
    ] == [("hidden", True), ("exported", False), ("public_helper", False)]


def test_private_helpers_and_exported_functions_keep_distinct_linkage(tmp_path):
    generated = _translate(tmp_path, _source("first", "x + 1.0f"))
    for name in (
        "transform",
        "hidden",
        "crosstl_ctor_Offset_1",
        "Offset__apply",
        "Op__operator_call",
        "Op__operator_call__temporary",
    ):
        assert re.search(rf"^static \w+ {name}\(", generated, re.MULTILINE), generated
    assert "inline float shared(" in generated
    assert "inline float twice_float(" in generated
    assert "static inline float local_transform_float(" in generated
    assert "float exported_first(float x) [[visible]]" in generated
    assert "static float exported_first" not in generated
    assert "kernel void first(" in generated


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_metal_linkage_metadata_does_not_become_target_semantics(tmp_path, target):
    source = """
    static float transform(float x) { return x + 1.0f; }
    kernel void first(device float* output [[buffer(0)]]) { output[0] = transform(2.0f); }
    """
    generated = _translate(tmp_path, source, target)
    assert "metal_static" not in generated
    assert "metal_inline" not in generated
    assert "transform" in generated


def test_generated_wide_vector_helpers_have_internal_linkage(tmp_path):
    source = """
    using Wide = metal::vec<float, 8>;
    kernel void wide(device float* output [[buffer(0)]]) {
        Wide left = Wide(2.0f);
        Wide right = Wide(1.0f);
        Wide sum = left + right;
        sum += right;
        output[0] = sum[0];
    }
    """
    intermediate = _translate(tmp_path, source, "crossgl")
    ast = Parser(Lexer(intermediate).get_tokens()).parse()
    assert len(ast.functions) == 4
    assert all(function.linkage == "internal" for function in ast.functions)


def test_generated_native_support_helpers_have_internal_linkage():
    generator = MetalCodeGen()
    generator.required_metal_wave_ballot_helper = True
    generator.required_metal_wave_match_helper = True
    generator.required_metal_wave_mask_contains_helper = True
    generator.required_metal_wave_multi_prefix_helpers = set(
        generator.METAL_WAVE_MULTI_PREFIX_HELPERS
    )
    wave = generator.generate_metal_wave_helpers()
    definitions = re.findall(r"^((?:static )?\w+ __crossgl_\w+\()", wave, re.MULTILINE)
    assert len(definitions) == 3 + len(generator.METAL_WAVE_MULTI_PREFIX_HELPERS)
    assert all(line.startswith("static ") for line in definitions)
    generator.required_buffer_atomic_compare_helpers = {"uint", "int"}
    atomics = generator.generate_buffer_atomic_compare_helpers()
    assert "static uint __crossgl_buffer_atomic_compare_exchange_uint(" in atomics
    assert "static int __crossgl_buffer_atomic_compare_exchange_int(" in atomics


def _run(command, directory, label):
    result = subprocess.run(command, text=True, capture_output=True, timeout=90)
    (directory / f"{label}.json").write_text(
        json.dumps(
            {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_independent_metal_modules_link_and_execute_distinct_helpers(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native Metal linking")
    assert sys.platform == "darwin", "Native Metal linking requires macOS"
    runner = tmp_path / "readback"
    _run(
        [
            "swiftc",
            str(
                ROOT / "tests/fixtures/runtime_verification/metal_float_readback.swift"
            ),
            "-o",
            str(runner),
        ],
        tmp_path,
        "build-readback",
    )
    evidence = {}
    for mode in ("original", "translated"):
        objects = []
        for name, expression in (("first", "x + 1.0f"), ("second", "x * 2.0f")):
            source = _source(name, expression)
            path = tmp_path / f"{mode}-{name}.metal"
            path.write_text(
                _translate(tmp_path, source) if mode == "translated" else source,
                encoding="utf-8",
            )
            air = path.with_suffix(".air")
            _run(
                [
                    "xcrun",
                    "--sdk",
                    "macosx",
                    "metal",
                    "-Werror",
                    "-fno-fast-math",
                    "-c",
                    str(path),
                    "-o",
                    str(air),
                ],
                tmp_path,
                f"compile-{mode}-{name}",
            )
            objects.append(str(air))
        library = tmp_path / f"{mode}.metallib"
        _run(
            ["xcrun", "--sdk", "macosx", "metallib", *objects, "-o", str(library)],
            tmp_path,
            f"link-{mode}",
        )
        evidence[mode] = {}
        for name, expected in (("first", 3.0), ("second", 4.0)):
            result = json.loads(
                _run(
                    [str(runner), str(library), name, "10", f"exported_{name}"],
                    tmp_path,
                    f"execute-{mode}-{name}",
                )
            )
            assert result["values"][:7] == [
                expected,
                expected,
                5.0,
                5.0,
                12.0,
                6.0,
                7.0,
            ]
            assert result["values"][7] == pytest.approx(math.pi / 2.0, abs=2e-7)
            assert result["values"][8] == 6.0
            assert result["values"][9] == expected
            evidence[mode][name] = result
    (tmp_path / "outputs.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
