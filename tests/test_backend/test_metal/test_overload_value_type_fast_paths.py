"""Overload value types and alias resolution must agree on the plain-name fast paths."""

from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser


def convert(code: str) -> str:
    ast = MetalParser(MetalLexer(code).tokenize()).parse()
    return MetalToCrossGLConverter().generate(ast)


def test_plain_names_without_aliases_are_returned_unchanged():
    converter = MetalToCrossGLConverter()

    assert converter.metal_source_overload_value_type(" float4 ") == "float4"
    assert converter.metal_source_overload_value_type("auto") is None
    assert converter.metal_source_overload_value_type("const device float&") == (
        "float"
    )
    assert converter.resolve_type_alias("uint") == "uint"
    assert converter.resolve_type_alias("float4 *") == "float4*"
    assert converter.resolve_type_alias("") == ""


def test_plain_names_bound_by_aliases_or_templates_still_resolve():
    converter = MetalToCrossGLConverter()
    converter.type_aliases["real"] = "float"
    converter.template_type_bindings.append({"T": "half"})

    assert converter.resolve_type_alias("real*") == "float*"
    assert converter.metal_source_overload_value_type("real") == "float"
    assert converter.metal_source_overload_value_type("T") == "half"
    assert converter.metal_source_overload_value_type("T&") == "half"


def test_overloads_resolve_through_local_and_global_aliases():
    crossgl = convert("""
        typedef int count_t;

        float pick(float x) { return x * 2.0f; }
        int pick(int x) { return x + 1; }

        kernel void choose(device float* out_values [[buffer(0)]],
                           device int* out_counts [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
            typedef float real_t;
            real_t value = real_t(gid);
            count_t count = count_t(gid);
            out_values[gid] = pick(value);
            out_counts[gid] = pick(count);
        }
        """)

    assert "real_t" not in crossgl
    assert "float value = float(gid);" in crossgl
    assert "count = int(gid);" in crossgl
    assert "buffer_store(out_values, gid, pick(value));" in crossgl
    assert "buffer_store(out_counts, gid, pick(count));" in crossgl
