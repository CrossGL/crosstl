"""Primitive alias chains retain declaration ownership and logical types."""

import json
import os
import re
import struct
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalScalarAliasResolutionError,
    MetalToCrossGLConverter,
)
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.project import (
    build_native_loader_dispatch_request,
    load_project_config,
    translate_project,
)
from crosstl.project.runtime_verification import RuntimeAllocationView, RuntimeValue
from crosstl.translator import parse
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from tests.runtime_helpers import _prepare_native_package, _validate
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_precise_trig import GUARD, _execute
from tests.test_translator.test_native_loader_dispatch_integration import _executor


def _convert(source):
    ast = MetalParser(
        MetalLexer(source, preprocess=False).tokenize(), file_path="aliases.metal"
    ).parse()
    return MetalToCrossGLConverter().generate(ast)


def _alias(name, target, syntax):
    return (
        f"using {name} = {target};"
        if syntax == "using"
        else f"typedef {target} {name};"
    )


def _source(dtype, syntax):
    return f"""#include <metal_stdlib>
using namespace metal;
{_alias('Base', dtype, syntax)}
{_alias('Middle', 'Base', syntax)}
{_alias('Value', 'Middle', syntax)}
Value preserve(Value value) {{ return value; }}
kernel void aliases(device const uint* values [[buffer(0)]],
                    device uint* results [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {{
    {_alias('Local', 'Value', syntax)}
    float source = as_type<float>(values[i]);
    Local value = Local(source);
    results[4u * i] = as_type<uint>(float(value));
    results[4u * i + 1u] = as_type<uint>(float(preserve(Value(source))));
    {{
        {_alias('Base', 'int', syntax)}
        {_alias('Nested', 'Base', syntax)}
        Nested integer = Nested(source);
        results[4u * i + 2u] = as_type<uint>(float(integer));
        results[4u * i + 3u] = as_type<uint>(float(Value(source)));
    }}
}}
"""


@pytest.mark.parametrize("syntax", ["using", "typedef"])
@pytest.mark.parametrize(
    "dtype,canonical",
    [
        ("float", "float"),
        ("int", "int"),
        ("uint", "uint"),
        ("half", "f16"),
        ("bfloat", "bfloat16"),
    ],
)
def test_scalar_alias_chain_retains_every_declaration(dtype, canonical, syntax):
    generated = _convert(_source(dtype, syntax))
    for name in ("Base", "Middle", "Value"):
        assert f"typedef {canonical} {name};" in generated
    assert "Local value" not in generated


def test_scalar_alias_namespace_ownership_and_imports():
    generated = _convert("""
        namespace Left {
            using Base = float;
            using Value = Base;
            Value left(Value x) { return Value(x); }
        }
        namespace Right {
            using Base = int;
            using Value = Base;
            Value right(Value x) { return Value(x); }
        }
        using namespace Left;
        Value imported(Value x) { return Value(x); }
        Right::Value qualified(Right::Value x) { return Right::Value(x); }
    """)
    assert "float left(float x)" in generated
    assert "int right(int x)" in generated
    assert "float imported(float x)" in generated
    assert "int qualified(int x)" in generated
    assert "typedef" not in generated


def test_scalar_alias_dependencies_bind_before_local_shadowing():
    generated = _convert("""
        using Base = float;
        using Value = Base;
        float evaluate(float x) {
            using Base = int;
            Value original = Value(x);
            using Local = Base;
            { using Base = half; Local inner = Local(x); original += float(inner); }
            Local restored = Local(x);
            return original + float(restored);
        }
    """)
    assert "Value original = float(x);" in generated
    assert "int inner = int(x);" in generated
    assert "int restored = int(x);" in generated


def test_scalar_alias_chain_preserves_const_qualification():
    generated = _convert("""
        using Base = const float;
        using Value = Base;
        float evaluate(float x) { Value value = x; return value; }
    """)
    assert "const Value value = x;" in generated


def test_local_scalar_alias_chain_preserves_const_qualification():
    generated = _convert("""
        float evaluate(float x) {
            using Base = const float;
            using Value = Base;
            Value value = x;
            return value;
        }
    """)
    assert "const float value = x;" in generated


def test_scalar_alias_lookup_uses_nested_import_namespace():
    generated = _convert("""
        namespace Outer {
            namespace Inner { using Value = float; }
            using namespace Inner;
            Value evaluate(Value x) { return x; }
        }
    """)
    assert "float evaluate(float x)" in generated


@pytest.mark.parametrize(
    "declaration", ["struct Value { int member; };", "enum Value { One = 1 };"]
)
def test_scalar_alias_does_not_override_a_nearer_type_declaration(declaration):
    generated = _convert("""
        using Value = float;
        namespace Inner {
            DECLARATION
            Value identity(Value value) { return value; }
        }
        Value outer(Value value) { return value; }
    """.replace("DECLARATION", declaration))
    assert "Value identity(Value value)" in generated
    assert "float identity" not in generated
    assert "typedef float Value" not in generated
    assert "float outer(float value)" in generated


def test_scalar_alias_overloads_use_their_declaration_context():
    generated = _convert("""
        namespace Left {
            using Value = half;
            Value convert(Value x) { return x; }
        }
        namespace Right {
            using Value = float;
            Value convert(Value x) { return x; }
        }
        float evaluate(float x) {
            using Value = int;
            return float(Left::convert(half(x))) + Right::convert(x);
        }
    """)
    assert "float16 convert" in generated
    assert "float convert" in generated
    assert "__metal_overload_" in generated


@pytest.mark.parametrize("dtype", ["float", "int", "half", "bfloat"])
def test_scalar_alias_state_does_not_leak_between_generations(dtype):
    converter = MetalToCrossGLConverter()
    for target_type in (dtype, "uint"):
        source = f"using Base = {target_type}; using Value = Base; Value identity(Value x) {{ return x; }}"
        ast = MetalParser(MetalLexer(source, preprocess=False).tokenize()).parse()
        generated = converter.generate(ast)
        assert generated == _convert(source)


@pytest.mark.parametrize(
    "source,reason",
    [
        ("using A = A; A value;", "recursive"),
        ("using A = Missing; A value;", "no concrete visible declaration"),
        (
            "using A = Missing; using Missing = float; A value;",
            "no concrete visible declaration",
        ),
        ("using A = float; using A = int; A value;", "conflicting"),
        (
            "namespace L { using A = float; } namespace R { using A = int; } using namespace L; using namespace R; A value;",
            "ambiguous",
        ),
    ],
)
def test_invalid_scalar_aliases_report_source_locations(source, reason):
    with pytest.raises(MetalScalarAliasResolutionError) as error:
        _convert(source)
    assert (
        error.value.project_diagnostic_code
        == "project.translate.metal-scalar-alias-unresolved"
    )
    assert reason in error.value.reason
    assert error.value.source_location["file"] == "aliases.metal"


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
def test_project_reports_scalar_alias_failure_without_an_artifact(tmp_path, target):
    (tmp_path / "broken.metal").write_text(
        "using Value = Missing; Value evaluate(Value x) { return x; }", encoding="utf-8"
    )
    payload = translate_project(
        tmp_path, targets=[target], output_dir="out", format_output=False
    ).to_json()
    assert payload["summary"]["translatedCount"] == 0
    assert payload["summary"]["failedCount"] == 1
    (diagnostic,) = payload["diagnostics"]
    assert diagnostic["code"] == "project.translate.metal-scalar-alias-unresolved"
    assert diagnostic["location"]["file"] == "broken.metal"
    assert diagnostic["location"]["line"] == 1
    assert diagnostic["missingCapabilities"] == ["metal.scalar-alias-resolution"]
    assert diagnostic["details"]["scalarAlias"] == {
        "aliasName": "Value",
        "reason": "target 'Missing' has no concrete visible declaration",
    }
    (artifact,) = payload["artifacts"]
    assert artifact["status"] == "failed"
    assert not (tmp_path / artifact["path"]).exists()


def test_project_reports_recursive_alias_dependency_chain(tmp_path):
    (tmp_path / "cycle.metal").write_text("using Value = Value;", encoding="utf-8")
    payload = translate_project(
        tmp_path, targets=["metal"], output_dir="out", format_output=False
    ).to_json()
    (diagnostic,) = payload["diagnostics"]
    assert diagnostic["details"]["scalarAlias"]["dependencyChain"] == ["Value", "Value"]
    assert payload["summary"]["failedCount"] == 1


@pytest.mark.parametrize("local", [False, True])
def test_scalar_aliases_do_not_silently_drop_volatile_value_qualification(local):
    declaration = "using Base = volatile float; using Value = Base;"
    body = "float evaluate(float x) { LOCAL Value value = x; return value; }"
    source = body.replace("LOCAL", declaration if local else "")
    if not local:
        source = declaration + source
    with pytest.raises(
        MetalScalarAliasResolutionError, match="require resource storage"
    ):
        _convert(source)


def _translate(root, source, target):
    original = root / "aliases.metal"
    original.write_text(source, encoding="utf-8")
    saved = root / "aliases.cgl"
    saved.write_text(
        translate(str(original), backend="cgl", format_output=False), encoding="utf-8"
    )
    generated = translate(str(original), backend=target, format_output=False)
    assert generated == translate(str(saved), backend=target, format_output=False)
    return generated


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("syntax", ["using", "typedef"])
@pytest.mark.parametrize("dtype", ["float", "int", "uint", "half", "bfloat"])
def test_scalar_aliases_translate_and_compile(tmp_path, target, syntax, dtype):
    generated = _translate(tmp_path, _source(dtype, syntax), target)
    _compile(
        generated, target, tmp_path, directx_compile_flags=("-enable-16bit-types",)
    )


def _resource_source(dtype):
    return f"""#include <metal_stdlib>
using namespace metal;
using Base = {dtype};
using Value = Base;
using ReadOnly = const Value;
kernel void copy_values(device ReadOnly* values [[buffer(0)]],
                        device Value* results [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {{
    results[i] = values[i];
}}
"""


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("dtype", ["float", "int", "uint", "half", "bfloat"])
def test_scalar_alias_resource_elements_translate_and_compile(tmp_path, target, dtype):
    source = _resource_source(dtype)
    crossgl = _convert(source)
    assert "void copy_values(StructuredBuffer<" in crossgl
    generated = _translate(tmp_path, source, target)
    _compile(
        generated, target, tmp_path, directx_compile_flags=("-enable-16bit-types",)
    )


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _specialized_alias_source(syntax, specialization):
    return f"""#include <metal_stdlib>
using namespace metal;
{_alias('Element', 'half', syntax)}
{_alias('Alias', 'Element', syntax)}
namespace Types {{ {_alias('Value', 'half', syntax)} }}
template <typename T>
struct Identity {{ static constexpr constant uint value = 7u; }};
template <> struct Identity<{specialization}>;
template <>
struct Identity<{specialization}> {{
    static constexpr constant uint value = 42u;
}};
kernel void aliases(device const uint* values [[buffer(0)]],
                    device uint* results [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {{
    if (i != 0u) return;
    results[0] = Identity<Element>::value + values[0];
    results[1] = Identity<half>::value + values[0];
    results[2] = Identity<Alias>::value + values[0];
    results[3] = Identity<float>::value + values[0];
    {{
        {_alias('Element', 'float', syntax)}
        results[4] = Identity<Element>::value + values[0];
        results[5] = Identity<Alias>::value + values[0];
    }}
    results[6] = Identity<const half>::value + values[0];
    results[7] = Identity<thread half*>::value + values[0];
    results[8] = Identity<Types::Value>::value + values[0];
}}
"""


@pytest.mark.parametrize("syntax", ["using", "typedef"])
@pytest.mark.parametrize("specialization", ["half", "Element"])
@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
def test_explicit_struct_specialization_uses_canonical_aliases(
    tmp_path, syntax, specialization, target
):
    source = _specialized_alias_source(syntax, specialization)
    generated = _translate(tmp_path, source, target)
    for index, expected in enumerate((42, 42, 42, 7, 7, 42, 7, 7, 42)):
        assert re.search(
            rf"results\[{index}\]\s*=\s*\(?{expected}u?\s*\+", generated
        ), generated
    _compile(
        generated, target, tmp_path, directx_compile_flags=("-enable-16bit-types",)
    )


@pytest.mark.parametrize("syntax", ["using", "typedef"])
@pytest.mark.parametrize("specialization", ["half", "Element"])
def test_explicit_struct_specialization_aliases_execute(
    tmp_path, syntax, specialization
):
    if os.environ.get("CROSTL_REQUIRE_SCALAR_ALIASES") != "1":
        pytest.skip("set CROSTL_REQUIRE_SCALAR_ALIASES=1 for native alias checks")
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    source = _specialized_alias_source(syntax, specialization)
    generated = _translate(tmp_path, source, target)
    expected = [42, 42, 42, 7, 7, 42, 7, 7, 42, *GUARD]
    records = {}
    for name, code in (("generated", generated), ("original", source)):
        if name == "original" and target != "metal":
            continue
        records[name] = _execute(
            tmp_path / name,
            target,
            code,
            [0],
            expected,
            metal_entry="aliases",
            check_outputs=_check,
            directx_compile_flags=("-enable-16bit-types",),
        )
    (tmp_path / "evidence.json").write_text(
        json.dumps({"target": target, "records": records}, indent=2), encoding="utf-8"
    )


def _expected(dtype, values):
    expected = []
    for value in values:
        if dtype in {"int", "uint"}:
            converted = _bits(float(int(value)))
        elif dtype == "half":
            converted = _bits(struct.unpack("<e", struct.pack("<e", value))[0])
        elif dtype == "bfloat":
            word = _bits(value)
            converted = ((word + 0x7FFF + ((word >> 16) & 1)) >> 16) << 16
        else:
            converted = _bits(value)
        expected.extend([converted, converted, _bits(float(int(value))), converted])
    return expected + GUARD


def _check(actual, expected):
    assert actual == expected
    return 0


@pytest.mark.parametrize("syntax", ["using", "typedef"])
@pytest.mark.parametrize("dtype", ["float", "int", "uint", "half", "bfloat"])
def test_scalar_aliases_execute(tmp_path, dtype, syntax):
    if os.environ.get("CROSTL_REQUIRE_SCALAR_ALIASES") != "1":
        pytest.skip("set CROSTL_REQUIRE_SCALAR_ALIASES=1 for native alias checks")
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    source = _source(dtype, syntax)
    generated = _translate(tmp_path, source, target)
    # Exact binary32 inputs include values that round when narrowed to half.
    values = [index * 13 / 1024.0 for index in range(2048)]
    inputs = [_bits(value) for value in values]
    expected = _expected(dtype, values)
    if dtype in {"half", "bfloat"}:
        assert expected != _expected("float", values)
    for name, words in (("inputs", inputs), ("expected", expected)):
        (tmp_path / f"{name}.bin").write_bytes(struct.pack(f"<{len(words)}I", *words))
    records = {}
    for name, code in (("generated", generated), ("original", source)):
        if name == "original" and target != "metal":
            continue
        records[name] = _execute(
            tmp_path / name,
            target,
            code,
            inputs,
            expected,
            metal_entry="aliases",
            check_outputs=_check,
            metal_compile_flags=("-fno-fast-math",),
            directx_compile_flags=("-enable-16bit-types",),
        )
    (tmp_path / "evidence.json").write_text(
        json.dumps({"target": target, "records": records}, indent=2), encoding="utf-8"
    )


def _bitcast_source(syntax):
    return f"""#include <metal_stdlib>
using namespace metal;
{_alias('Base', 'bfloat', syntax)}
{_alias('Value', 'Base', syntax)}
{_alias('ReadOnly', 'const Value', syntax)}
Value decode(ushort bits) {{ return as_type<Value>(bits); }}
ushort record(thread uint& count, ushort bits) {{ count += 1u; return bits; }}
kernel void aliases(device const uint* values [[buffer(0)]],
                    device uint* results [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {{
    {_alias('Local', 'Value', syntax)}
    uint count = 0u;
    Value aliased = as_type<Value>(ushort(values[i]));
    bfloat explicit_value = as_type<bfloat>(ushort(values[i]));
    ReadOnly qualified = as_type<ReadOnly>(ushort(values[i]));
    results[7u * i] = uint(as_type<ushort>(aliased));
    results[7u * i + 1u] = uint(as_type<ushort>(explicit_value));
    results[7u * i + 2u] = uint(as_type<ushort>(decode(ushort(values[i]))));
    results[7u * i + 3u] = uint(as_type<ushort>(as_type<Value>(ushort(values[i]))));
    results[7u * i + 4u] = uint(as_type<ushort>(as_type<Local>(record(count, ushort(values[i])))));
    results[7u * i + 5u] = uint(as_type<ushort>(qualified));
    results[7u * i + 6u] = count;
}}
"""


@pytest.mark.parametrize("syntax", ["using", "typedef"])
@pytest.mark.parametrize("target", ["opengl", "directx", "metal"])
def test_bfloat_alias_bitcasts_translate_and_compile(tmp_path, syntax, target):
    source = _bitcast_source(syntax)
    generated = _translate(tmp_path, source, target)
    if target == "opengl":
        assert "float decode(uint bits)" in generated
        assert "return uintBitsToFloat((bits << 16u));" in generated
        assert "float aliased = uintBitsToFloat((" in generated
    _compile(
        generated, target, tmp_path, directx_compile_flags=("-enable-16bit-types",)
    )


@pytest.mark.parametrize("alias", ["Value", "Chained", "ReadOnly"])
def test_glsl_bfloat_bitcast_alias_keeps_its_logical_result_type(alias):
    generated = GLSLCodeGen().generate(parse(f"""shader AliasBits {{
            typedef bfloat Value;
            typedef Value Chained;
            typedef const Value ReadOnly;
            uint unpack(uint bits) {{ return asuint(as_type<{alias}>(bits)); }}
            uint pack({alias} value) {{ return asuint(value); }}
        }}"""))
    assert (
        "return (floatBitsToUint(uintBitsToFloat((bits << 16u))) >> 16u);" in generated
    )
    assert "return (floatBitsToUint(value) >> 16u);" in generated


@pytest.mark.parametrize("alias", ["Value", "Chained", "ReadOnly"])
def test_glsl_bfloat_bitcast_alias_rejects_a_float_payload(alias):
    with pytest.raises(ValueError, match="requires an integer payload"):
        GLSLCodeGen().generate(parse(f"""shader InvalidBits {{
                typedef bfloat Value;
                typedef Value Chained;
                typedef const Value ReadOnly;
                {alias} unpack(float value) {{ return as_type<{alias}>(value); }}
            }}"""))


@pytest.mark.parametrize("qualified", ["const B", "thread B&"])
def test_glsl_qualified_alias_cycles_are_rejected(qualified):
    generator = GLSLCodeGen()
    generator.current_type_aliases = {"A": qualified, "B": "A"}
    with pytest.raises(ValueError, match="Cyclic OpenGL type alias"):
        generator.glsl_normalized_source_type("A")


@pytest.mark.parametrize("syntax", ["using", "typedef"])
def test_bfloat_alias_bitcasts_execute(tmp_path, syntax):
    if os.environ.get("CROSTL_REQUIRE_SCALAR_ALIASES") != "1":
        pytest.skip("set CROSTL_REQUIRE_SCALAR_ALIASES=1 for native alias checks")
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    source = _bitcast_source(syntax)
    generated = _translate(tmp_path, source, target)
    records = {}
    # A single 65,536-workgroup dispatch exceeds the DirectX axis limit.
    for start in (0, 32768):
        directory = tmp_path / f"payloads-{start}"
        directory.mkdir()
        inputs = list(range(start, start + 32768))
        expected = [word for value in inputs for word in [value] * 6 + [1]] + GUARD
        for name, words in (("inputs", inputs), ("expected", expected)):
            (directory / f"{name}.bin").write_bytes(
                struct.pack(f"<{len(words)}I", *words)
            )
        records[str(start)] = {}
        for name, code in (("generated", generated), ("original", source)):
            if name == "original" and target != "metal":
                continue
            records[str(start)][name] = _execute(
                directory / name,
                target,
                code,
                inputs,
                expected,
                metal_entry="aliases",
                check_outputs=_check,
                metal_compile_flags=("-fno-fast-math",),
                directx_compile_flags=("-enable-16bit-types",),
            )
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "target": target,
                "syntax": syntax,
                "payloadCount": 65536,
                "bitcastPaths": 6,
                "oracle": (
                    "Exact 16-bit payloads, single operand evaluation and intact guards"
                ),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _namespace_bitcast_source(syntax):
    return f"""#include <metal_stdlib>
using namespace metal;
{_alias('Value', 'float', syntax)}
namespace Payload {{ {_alias('Value', 'bfloat', syntax)} }}
kernel void aliases(device const uint* values [[buffer(0)]],
                    device uint* results [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {{
    {_alias('Local', 'Payload::Value', syntax)}
    Payload::Value qualified = as_type<Payload::Value>(ushort(values[i]));
    Local local = as_type<Local>(ushort(values[i]));
    results[3u * i] = as_type<uint>(float(qualified));
    results[3u * i + 1u] = as_type<uint>(float(local));
    results[3u * i + 2u] = as_type<uint>(as_type<Value>(values[i] << 16u));
}}
"""


@pytest.mark.parametrize("syntax", ["using", "typedef"])
@pytest.mark.parametrize("target", ["opengl", "directx", "metal"])
def test_namespace_alias_bitcasts_translate_and_compile(tmp_path, syntax, target):
    source = _namespace_bitcast_source(syntax)
    canonical = _convert(source)
    assert "as_type<Payload::Value>" not in canonical
    assert "as_type<Local>" not in canonical
    generated = _translate(tmp_path, source, target)
    _compile(
        generated, target, tmp_path, directx_compile_flags=("-enable-16bit-types",)
    )


@pytest.mark.parametrize("syntax", ["using", "typedef"])
def test_namespace_alias_bitcasts_execute(tmp_path, syntax):
    if os.environ.get("CROSTL_REQUIRE_SCALAR_ALIASES") != "1":
        pytest.skip("set CROSTL_REQUIRE_SCALAR_ALIASES=1 for native alias checks")
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    source = _namespace_bitcast_source(syntax)
    generated = _translate(tmp_path, source, target)
    inputs = [0, 1, 0x3F80, 0x8000, 0xBF80, 0x7F80, 0xFF80]
    expected = [value << 16 for value in inputs for _ in range(3)] + GUARD
    records = {}
    for name, code in (("generated", generated), ("original", source)):
        if name == "original" and target != "metal":
            continue
        records[name] = _execute(
            tmp_path / name,
            target,
            code,
            inputs,
            expected,
            metal_entry="aliases",
            check_outputs=_check,
            metal_compile_flags=("-fno-fast-math",),
            directx_compile_flags=("-enable-16bit-types",),
        )
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "target": target,
                "syntax": syntax,
                "oracle": (
                    "Exact bfloat-to-binary32 widening despite a conflicting outer alias"
                ),
                "inputs": inputs,
                "expected": expected,
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


NAMESPACE_SOURCE = """#include <metal_stdlib>
using namespace metal;
namespace Left {
    using Base = half;
    using Value = Base;
    Value convert(Value x) { return x; }
}
namespace Right {
    using Base = float;
    using Value = Base;
    Value convert(Value x) { return x; }
}
kernel void aliases(device const uint* values [[buffer(0)]],
                    device uint* results [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {
    using Value = int;
    float source = as_type<float>(values[i]);
    results[2u * i] = as_type<uint>(float(Left::convert(half(source))));
    results[2u * i + 1u] = as_type<uint>(Right::convert(source) + float(Value(source)));
}
"""


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
def test_namespace_scalar_aliases_translate_and_compile(tmp_path, target):
    generated = _translate(tmp_path, NAMESPACE_SOURCE, target)
    _compile(
        generated, target, tmp_path, directx_compile_flags=("-enable-16bit-types",)
    )


def test_namespace_scalar_aliases_execute(tmp_path):
    if os.environ.get("CROSTL_REQUIRE_SCALAR_ALIASES") != "1":
        pytest.skip("set CROSTL_REQUIRE_SCALAR_ALIASES=1 for native alias checks")
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    generated = _translate(tmp_path, NAMESPACE_SOURCE, target)
    values = [index * 13 / 1024.0 for index in range(2048)]
    inputs = [_bits(value) for value in values]
    expected = [
        word
        for value in values
        for word in (
            _bits(struct.unpack("<e", struct.pack("<e", value))[0]),
            _bits(value + int(value)),
        )
    ] + GUARD
    assert any(expected[2 * index] != word for index, word in enumerate(inputs))
    for name, words in (("inputs", inputs), ("expected", expected)):
        (tmp_path / f"{name}.bin").write_bytes(struct.pack(f"<{len(words)}I", *words))
    records = {}
    for name, code in (("generated", generated), ("original", NAMESPACE_SOURCE)):
        if name == "original" and target != "metal":
            continue
        records[name] = _execute(
            tmp_path / name,
            target,
            code,
            inputs,
            expected,
            metal_entry="aliases",
            check_outputs=_check,
            metal_compile_flags=("-fno-fast-math",),
            directx_compile_flags=("-enable-16bit-types",),
        )
    (tmp_path / "evidence.json").write_text(
        json.dumps({"target": target, "records": records}, indent=2), encoding="utf-8"
    )


def test_ci_requires_scalar_alias_controls_without_an_additional_runner():
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow.split("      - name: Validate scalar alias resolution\n", 1)[
        1
    ].split("      - name:", 1)[0]
    assert 'CROSTL_REQUIRE_SCALAR_ALIASES: "1"' in step
    assert "tests/test_translator/test_scalar_aliases.py" in step
    assert "--timeout-seconds 120" in step
    assert "pytest -q -n auto" in step
    assert "if:" not in step and "continue-on-error" not in step


@pytest.mark.parametrize("generator", [MetalCodeGen, GLSLCodeGen])
@pytest.mark.parametrize(
    "declarations,parameters,body",
    [
        ("", "const device Value& sample @buffer(0)", ""),
        ("", "RWStructuredBuffer<Value> sample @buffer(0)", ""),
        ("", "", "Value sample = 3; int Value = 4;"),
        ("Value helper(Value sample) { return sample; }", "", "helper(3);"),
        ("Value retained = 7;", "", "int sample = retained;"),
        ("cbuffer Config { Value retained; };", "", "int sample = retained;"),
        ("struct Payload { Value field; };", "", "Payload sample;"),
        ("", "", "Value sample[2];"),
    ],
)
def test_entry_scope_preserves_transitive_type_dependencies(
    generator, declarations, parameters, body
):
    ast = parse(f"""shader Dependencies {{
    typedef int Base;
    typedef Base Value;
    typedef uint Unused;
    Unused unrelated = 1u;
    {declarations}
    compute {{ @stage_entry void selected({parameters}) {{ {body} }} }}
    compute {{ @stage_entry void other() {{ Unused value = unrelated; }} }}
}}""")
    before = [declaration.name for declaration in ast.global_variables]
    scoped = generator().entry_scoped_ast(ast, "selected")
    aliases = [
        declaration.name
        for declaration in scoped.global_variables
        if getattr(declaration, "is_type_alias", False)
    ]
    assert aliases == ["Base", "Value"]
    assert "unrelated" not in [node.name for node in scoped.global_variables]
    assert [declaration.name for declaration in ast.global_variables] == before


@pytest.mark.parametrize("generator", [MetalCodeGen, GLSLCodeGen])
def test_entry_scope_keeps_aliases_for_retained_structs(generator):
    ast = parse("""shader Dependencies {
    typedef int Base;
    typedef Base Value;
    struct Payload { Value field; };
    compute { @stage_entry void selected() {} }
}""")
    scoped = generator().entry_scoped_ast(ast, "selected")
    names = [node.name for node in scoped.global_variables]
    if generator is MetalCodeGen:
        assert scoped.structs == [] and names == []
    else:
        assert [node.name for node in scoped.structs] == ["Payload"]
        assert names == ["Base", "Value"]


def _reference_alias_project(tmp_path, target, syntax, address_space):
    (tmp_path / "aliases.metal").write_text(
        f"""#include <metal_stdlib>
using namespace metal;
{_alias('Base', 'int', syntax)}
{_alias('Value', 'Base', syntax)}
{_alias('Unused', 'uint', syntax)}
kernel void selected({address_space} const Value& value [[buffer(0)]],
                     device int* result [[buffer(1)]]) {{
    result[0] = value;
    result[1] = -value;
}}
kernel void other(device Unused* result [[buffer(0)]]) {{ result[0] = 7u; }}
""",
        encoding="utf-8",
    )
    (tmp_path / "crosstl.toml").write_text(
        f"""[project]
include = ["aliases.metal"]
targets = ["{target}"]
workgroup_size = [1, 1, 1]
[project.entry_points]
"aliases.metal" = "selected"
""",
        encoding="utf-8",
    )
    report = translate_project(load_project_config(tmp_path), format_output=False)
    assert report.to_json()["summary"]["failedCount"] == 0, report.to_json()
    return _prepare_native_package(report, tmp_path)


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("syntax", ["using", "typedef"])
@pytest.mark.parametrize("address_space", ["device", "constant"])
def test_entry_reference_aliases_translate_and_compile(
    tmp_path, target, syntax, address_space
):
    descriptor, package = _reference_alias_project(
        tmp_path, target, syntax, address_space
    )
    artifact = package / descriptor["artifact"]["packagePath"]
    generated = artifact.read_text(encoding="utf-8")
    assert "Unused" not in generated and "void other(" not in generated
    assert len(descriptor["bindings"]) == 2
    assert all(
        binding["scalarLayout"]["elementType"] == "int32"
        for binding in descriptor["bindings"]
    )
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("address_space", ["device", "constant"])
def test_entry_reference_aliases_execute(tmp_path, address_space):
    if os.environ.get("CROSTL_REQUIRE_SCALAR_ALIASES") != "1":
        pytest.skip("set CROSTL_REQUIRE_SCALAR_ALIASES=1 for native alias checks")
    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    descriptor, package = _reference_alias_project(
        tmp_path, target, "using", address_space
    )
    artifact = package / descriptor["artifact"]["packagePath"]
    module = _validate(artifact, tmp_path, target)
    assert module.stat().st_size > 0
    guard = [1037] * 8
    for index, sample in enumerate([-2147483647, -1, 0, 2147483647]):
        inputs, outputs = {}, {}
        for binding in descriptor["bindings"]:
            name = binding["name"]
            writable = binding["access"] == "read_write"
            values = [0, 0] + guard if writable else [sample]
            value = RuntimeValue(
                name=name,
                dtype="int32",
                shape=(len(values),),
                values=values,
                allocation=(
                    RuntimeAllocationView(name, 256, len(values) * 4, 1024)
                    if binding["kind"] == "buffer"
                    else None
                ),
            )
            inputs[name] = value
            if writable:
                outputs[name] = replace(value, values=[sample, -sample] + guard)
        assert len(outputs) == 1
        request = build_native_loader_dispatch_request(
            descriptor,
            package,
            inputs,
            outputs,
            {"workgroupCount": [1, 1, 1], "workgroupSize": [1, 1, 1]},
            expected_target=target,
        )
        executor = _executor(target)
        availability = executor.is_available(request)
        assert availability.available, availability.reason
        result = executor.run(request)
        (tmp_path / f"readback-{index}.json").write_text(
            json.dumps(
                {
                    "fixture": request.fixture.to_json(),
                    "executionPlan": request.execution_plan.to_json(),
                    "status": result.status,
                    "outputs": result.outputs,
                    "details": result.details,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        assert result.status == "ok", result.details
        for name, expected in outputs.items():
            assert result.outputs[name]["values"] == expected.values
