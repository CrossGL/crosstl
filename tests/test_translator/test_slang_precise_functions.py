"""A CrossGL ``@precise`` function keeps its floating-point arithmetic exact in Slang.

slangc ignores ``precise`` on a function (``precise void k()`` compiles to SPIR-V with no ``NoContraction``), but
carries a ``precise`` local backward through data flow. The Slang emitter therefore declares the function's value
locals ``precise`` and captures each value it stores elsewhere, returns or branches on in a precise local
(crosstl/translator/codegen/slang_precise_functions.py).
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.refusals import SlangRefusal

PRECISE_KERNEL = """shader S {
    buffer float a[];
    buffer float o[];
    compute {
        @precise
        @numthreads(64, 1, 1)
        void main(uvec3 gid @ gl_GlobalInvocationID) {
            uint i = gid.x;
            float acc = 0.0;
            for (int j = 0; a[i] * float(j) < a[i + 1u]; j++) {
                acc = acc + a[i] * a[i + 2u];
            }
            if (a[i] * a[i] - a[i + 3u] > 0.0) {
                o[i] = a[i] * a[i] - acc;
            } else {
                o[i] = acc > 0.0 ? acc * a[i] - a[i + 1u] : acc;
            }
        }
    }
}
"""

CONTRACTED_KERNEL = PRECISE_KERNEL.replace("@precise\n", "")

HELPER_AND_FRAGMENT = """shader S {
    struct FragmentInput { vec2 uv; }
    struct FragmentOutput { vec4 color; }
    buffer float a[];
    buffer float o[];
    @precise
    float helper(float x, float y) {
        return x * y + 1.0;
    }
    compute {
        @numthreads(64, 1, 1)
        void main(uvec3 gid @ gl_GlobalInvocationID) {
            o[gid.x] = helper(a[gid.x], a[gid.x + 1u]);
        }
    }
    fragment {
        uniform float u;
        @precise
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            output.color = vec4(input.uv.x * input.uv.y + u);
            return output;
        }
    }
}
"""

AGGREGATE_INITIALIZER = """shader S {
    struct Pair { float x; float y; }
    buffer float a[];
    buffer float o[];
    compute {
        @precise
        @numthreads(64, 1, 1)
        void main(uvec3 gid @ gl_GlobalInvocationID) {
            Pair p = Pair(a[gid.x] * a[gid.x + 1u] + 1.0, 0.0);
            o[gid.x] = p.x;
        }
    }
}
"""


def translate(tmp_path, source):
    path = tmp_path / "k.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend="slang", format_output=False)


def test_slang_precise_function_declares_its_value_locals_precise(tmp_path):
    slang = translate(tmp_path, PRECISE_KERNEL)
    assert re.search(r"\bprecise uint i = ", slang), slang
    assert re.search(r"\bprecise float acc = ", slang), slang


def test_slang_precise_function_captures_stores_and_conditions(tmp_path):
    slang = translate(tmp_path, PRECISE_KERNEL)
    stores = re.findall(r"o\[i\] = \((crossglPreciseValue\w*) = ", slang)
    conditions = re.findall(r"\((crossglPreciseCondition\w*) = ", slang)
    assert len(stores) == 2, slang
    # The loop condition, the if condition and the selection's condition.
    assert len(conditions) == 3, slang
    for name in stores:
        assert re.search(rf"\bprecise float {name};", slang)
    for name in conditions:
        assert re.search(rf"\bprecise bool {name};", slang)
    assert re.search(r"if \(\(crossglPreciseCondition\w* = ", slang)


def test_slang_precise_function_captures_returned_and_member_stores(tmp_path):
    slang = translate(tmp_path, HELPER_AND_FRAGMENT)
    helper = re.search(r"float helper\(float x, float y\)\s*\{(.*?)\n\}", slang, re.S)
    assert helper, slang
    assert re.search(r"precise float (crossglPreciseValue\w*);", helper.group(1))
    assert re.search(r"return \(crossglPreciseValue\w* = x \* y \+ 1\.0\);", slang)
    assert re.search(r"output\.color = \(crossglPreciseValue\w* = float4\(", slang)
    assert re.search(r"precise float4 crossglPreciseValue\w*;", slang)


def test_slang_function_that_is_not_precise_is_unchanged(tmp_path):
    slang = translate(tmp_path, CONTRACTED_KERNEL)
    assert "crossglPrecise" not in slang
    assert not re.search(r"\bprecise\b", slang)


def test_slang_precise_aggregate_initializer_is_refused(tmp_path):
    with pytest.raises(SlangRefusal, match="no precise Slang local"):
        translate(tmp_path, AGGREGATE_INITIALIZER)


def _spirv_float_operations(tmp_path, slang, entry, stage):
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    source = tmp_path / f"{entry}.slang"
    source.write_text(slang)
    listing = tmp_path / f"{entry}.spvasm"
    subprocess.run(
        [
            slangc,
            str(source),
            "-target",
            "spirv-asm",
            "-entry",
            entry,
            "-stage",
            stage,
            "-o",
            str(listing),
        ],
        check=True,
        capture_output=True,
    )
    text = listing.read_text()
    decorated = set(re.findall(r"OpDecorate (%\w+) NoContraction", text))
    results = re.findall(r"^\s*(%\w+) = Op(?:FMul|FAdd|FSub)\b", text, re.M)
    assert results, text
    return results, decorated


def test_slangc_decorates_every_operation_of_a_precise_function(tmp_path):
    slang = translate(tmp_path, PRECISE_KERNEL)
    entry = re.search(r"\bvoid\s+(\w+)\s*\(uint3", slang).group(1)
    results, decorated = _spirv_float_operations(tmp_path, slang, entry, "compute")
    assert set(results) <= decorated

    slang = translate(tmp_path, HELPER_AND_FRAGMENT)
    entry = re.search(r"\bFragmentOutput\s+(\w+)\s*\(FragmentInput", slang).group(1)
    results, decorated = _spirv_float_operations(tmp_path, slang, entry, "fragment")
    assert set(results) <= decorated


def test_slangc_leaves_a_function_that_is_not_precise_contractible(tmp_path):
    slang = translate(tmp_path, CONTRACTED_KERNEL)
    entry = re.search(r"\bvoid\s+(\w+)\s*\(uint3", slang).group(1)
    results, decorated = _spirv_float_operations(tmp_path, slang, entry, "compute")
    assert not set(results) & decorated
