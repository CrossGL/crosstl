"""Mojo output for CrossGL's numeric built-ins, scalar atomic loads and stores, and conditions with side effects.

The classifications, ``nextafter``, ``rcp``, ``tgamma``, ``nan``, ``clz``, ``ctz`` and ``mulhi`` become functions of
Mojo's ``std.math`` and ``std.bit``; ``atomicLoad`` and ``atomicStore`` become std.atomic's static ``Atomic`` load
and store; and an ``if``, loop or assignment target whose expression increments or assigns writes those side effects
to a prelude instead of being refused.
"""

from __future__ import annotations

import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.mojo_refusals import MojoLoweringRefused
from crosstl.translator.refusals import MojoRefusal
from tests.test_translator.test_codegen.test_mojo_codegen import find_mojo_compiler

KERNEL = """shader Probe {{
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer int* iin;
        buffer uint* uin;
        buffer int* outp;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            float a = fin[i];
            int k = iin[i];
            uint u = uin[i];
            {body}
        }}
    }}
}}
"""

CASES = {
    "classify": (
        "vec2 v = vec2(a, 1.0); bvec2 bv = isnan(v); double x = double(a); half h = half(a);"
        " if (isnan(x) || isinf(h) || !isfinite(a) || bv.y) { outp[i] = 1; }"
    ),
    "float_functions": (
        "vec3 w = vec3(a, 1.0, 2.0); vec3 r = rcp(w) + nextafter(w, vec3(0.0)) + tgamma(w);"
        " float n = nextafter(a, 1.0) + rcp(a) + tgamma(a) + nan(); double dn = nan();"
        " outp[i] = int(r.x + n + float(dn));"
    ),
    "bit_counts": (
        "int c = clz(u) + clz(k); uint d = ctz(k) + ctz(u); uvec2 cv = clz(uvec2(u, 3u));"
        " outp[i] = c + int(d) + int(cv.y);"
    ),
    "multiply_high": (
        "uint h = mulhi(u, 7u) + mulhi(uin[i + 1u], u); int s = mulhi(k, -5);"
        " ivec2 hv = mulhi(ivec2(k, 2), ivec2(3, k)); uint64_t w = uint64_t(u) << 32u;"
        " uint64_t wh = mulhi(w, w); outp[i] = int(h) + s + hv.x + int(wh);"
    ),
    "atomics": (
        "atomicStore(outp[i], atomicLoad(iin[i + 1u]) + 2); uin[i] = atomicLoad(uin[i + 2u]);"
        " atomicLoad(outp[i + 1u]);"
    ),
    "conditions": (
        "int n = 0; while (k++ < 6) { n += 2; } int j = k; if (j > 10) { n = 1; }"
        " else if (++j > 5) { n += 4; } else if (j-- > 1) { n += 8; } else { n += 16; }"
        " for (int q = 0; q++ < 6; q += 1) { if (q == 2) { continue; } n += q; }"
        " int t = 20; while ((t -= 3) > 0) { n += 64; } int m = 3; do { m += 1; } while (m-- > 5);"
        " int vals[3]; int x = 0; vals[x++] = 1; vals[x++] = 2; vals[x] = x; outp[i] = n + j + vals[1] + m;"
    ),
    "per_lane_operands": (
        "float up = nextafter(abs(a), INFINITY); bool big = isinf(abs(a)) || isnan(sqrt(a));"
        " vec2 v = vec2(a, 2.0); vec2 w = nextafter(floor(v), vec2(sin(a))); bvec2 f = isfinite(abs(v));"
        " outp[i] = (big || f.y) ? int(up) : int(w.x);"
    ),
}


def translate(tmp_path, body, name="probe"):
    source = tmp_path / f"{name}.cgl"
    source.write_text(KERNEL.format(body=body))
    return crosstl.translate(str(source), backend="mojo")


def kernel_lines(code):
    """The statements of the kernel body, stripped, from the first local declaration on."""
    lines = [line.strip() for line in code.splitlines()]
    start = next(index for index, line in enumerate(lines) if line.startswith("var a:"))
    return lines[start:]


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_case_builds_with_mojo(tmp_path, name):
    mojo = find_mojo_compiler()
    code = translate(tmp_path, CASES[name], name)
    source = tmp_path / f"{name}.mojo"
    source.write_text(code)

    result = subprocess.run(
        [
            mojo,
            "build",
            "--emit",
            "llvm",
            "--disable-warnings",
            str(source),
            "-o",
            str(tmp_path / f"{name}.ll"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]


def test_classifications_keep_their_names_and_return_a_bool_per_lane(tmp_path):
    code = translate(tmp_path, CASES["classify"])

    assert "var bv: SIMD[DType.bool, 2] = isnan(v)" in code
    assert "if (((isnan(x) or isinf(h)) or (not isfinite(a))) or bv[1]):" in code


def test_float_functions_use_mojo_names_and_nan_takes_the_context_type(tmp_path):
    code = translate(tmp_path, CASES["float_functions"])

    assert "recip(w)" in code and "gamma(w)" in code and "nextafter(w, " in code
    assert "nan[DType.float32]()" in code
    assert "var dn: Float64 = Float64(nan[DType.float64]())" in code
    assert "rcp(" not in code and "tgamma(" not in code


def test_per_lane_built_in_results_are_operands_of_the_operand_type(tmp_path):
    code = translate(tmp_path, CASES["per_lane_operands"])

    assert "var up: Float32 = nextafter(abs(a), " in code
    assert "isinf(abs(a))" in code and "isnan(sqrt(a))" in code
    assert "nextafter(floor(v), " in code
    assert "var f: SIMD[DType.bool, 2] = isfinite(abs(v))" in code


def test_a_lone_underscore_name_is_renamed_because_mojo_discards_it(tmp_path):
    source = tmp_path / "underscore.cgl"
    source.write_text(
        KERNEL.replace("buffer float* fin;", "buffer float* _;")
        .replace("fin[i]", "_[i]")
        .format(body="outp[i] = int(a) + k + int(u);")
    )
    code = crosstl.translate(str(source), backend="mojo")

    assert "__: UnsafePointer[Float32" in code and "__[Int(i)]" in code
    assert "_:" not in code.replace("__:", "")


def test_bit_counts_are_std_bit_counts_of_the_operand_type(tmp_path):
    code = translate(tmp_path, CASES["bit_counts"])

    assert "from std.bit import count_leading_zeros, count_trailing_zeros" in code
    # clz of a uint is a UInt32 count; CrossGL adds the int count to it as a uint, which the declaration converts.
    assert (
        "var c: Int32 = Int32((count_leading_zeros(u) + UInt32(count_leading_zeros(k))))"
        in code
    )
    assert (
        "var cv: SIMD[DType.uint32, 2] = count_leading_zeros(SIMD[DType.uint32, 2](u, 3))"
        in code
    )


def test_multiply_high_widens_32_bit_operands_to_64_bits(tmp_path):
    code = translate(tmp_path, CASES["multiply_high"])

    assert (
        "((UInt32(u).cast[DType.uint64]() * UInt32(7).cast[DType.uint64]()) >> 32)"
        ".cast[DType.uint32]()" in code
    )
    # A signed product shifts arithmetically, which keeps the sign of the high word.
    assert (
        "var s: Int32 = ((Int32(k).cast[DType.int64]() * Int32((-5)).cast[DType.int64]()) >> 32)"
        ".cast[DType.int32]()" in code
    )
    assert "mulhi(u" not in code and "mulhi(k" not in code


def test_multiply_high_of_64_bit_scalars_is_std_math_mulhi(tmp_path):
    code = translate(tmp_path, CASES["multiply_high"])

    assert "var wh: UInt64 = mulhi(UInt64(w), UInt64(w))" in code


def test_scalar_atomic_load_and_store_use_the_element_address(tmp_path):
    code = translate(tmp_path, CASES["atomics"])

    assert (
        "Atomic[Int32].store(UnsafePointer(to=outp[Int(i)]), "
        "(Atomic[Int32].load(UnsafePointer(to=iin[Int((i + 1))])) + 2))" in code
    )
    assert (
        "uin[Int(i)] = Atomic[UInt32].load(UnsafePointer(to=uin[Int((i + 2))]))" in code
    )
    assert "from std.atomic import Atomic" in code


def test_loop_conditions_with_side_effects_run_them_before_every_test(tmp_path):
    lines = kernel_lines(translate(tmp_path, CASES["conditions"]))
    start = lines.index("while True:")

    assert lines[start : start + 6] == [
        "while True:",
        "var __cgl_increment_0: Int32 = k",
        "k += 1",
        "if not ((__cgl_increment_0 < 6)):",
        "break",
        "n += 2",
    ]
    assert "t -= 3" in lines
    assert lines[lines.index("t -= 3") + 1] == "if not ((t > 0)):"


def test_an_else_if_condition_with_side_effects_nests_in_else(tmp_path):
    lines = kernel_lines(translate(tmp_path, CASES["conditions"]))
    start = lines.index("if (j > 10):")

    assert lines[start : start + 8] == [
        "if (j > 10):",
        "n = 1",
        "else:",
        "j += 1",
        "var __cgl_increment_1: Int32 = j",
        "if (__cgl_increment_1 > 5):",
        "n += 4",
        "else:",
    ]


def test_an_assignment_target_with_an_increment_evaluates_it_once(tmp_path):
    lines = kernel_lines(translate(tmp_path, CASES["conditions"]))
    start = lines.index("var x: Int32 = 0")

    assert lines[start + 1 : start + 7] == [
        "var __cgl_increment_5: Int32 = x",
        "x += 1",
        "vals[__cgl_increment_5] = 1",
        "var __cgl_increment_6: Int32 = x",
        "x += 1",
        "vals[__cgl_increment_6] = 2",
    ]


@pytest.mark.parametrize(
    "body, name",
    [
        ("uint64_t w = uint64_t(u); uint64_t c = clz(w); outp[i] = int(c);", "clz"),
        ("float n = nan(3u); outp[i] = int(n);", "nan"),
        ("float r = rcp(k); outp[i] = int(r);", "rcp"),
    ],
)
def test_operands_of_crossgl_only_names_are_refused_as_foreign_calls(
    tmp_path, body, name
):
    with pytest.raises(MojoLoweringRefused, match=f"call to '{name}'"):
        translate(tmp_path, body)


@pytest.mark.parametrize(
    "body, message",
    [
        (
            "bool b = isnan(k > 2); outp[i] = b ? 1 : 0;",
            "no isnan for operands of type DType.bool",
        ),
        (
            "float m = mulhi(a, a); outp[i] = int(m);",
            "no mulhi for operands of type DType.float32",
        ),
        ("uvec2 m = mulhi(uvec2(u), uvec3(u)); outp[i] = int(m.x);", "no mulhi"),
        (
            "double x = double(a); vec2 y = nextafter(vec2(a), dvec2(x)); outp[i] = int(y.x);",
            "no nextafter",
        ),
    ],
)
def test_operands_mojo_functions_do_not_take_are_refused(tmp_path, body, message):
    with pytest.raises(MojoRefusal, match=message):
        translate(tmp_path, body)


def test_integer_operands_are_classified_as_finite_numbers(tmp_path):
    code = translate(
        tmp_path,
        "bool p = isnan(k) || isinf(u) || isfinite(ivec2(k, 1)).y; outp[i] = p ? 1 : 0;",
    )

    assert "isnan(k)" in code and "isinf(u)" in code
    assert "isfinite(SIMD[DType.int32, 2](k, 1))[1]" in code


def test_mixed_signedness_multiplies_high_in_the_unsigned_type(tmp_path):
    code = translate(
        tmp_path,
        "uint a = mulhi(k, u); uvec2 b = mulhi(uvec2(u, 2u), u); outp[i] = int(a + b.y);",
    )

    assert (
        "((UInt32(k).cast[DType.uint64]() * UInt32(u).cast[DType.uint64]()) >> 32)"
        ".cast[DType.uint32]()" in code
    )
    assert "SIMD[DType.uint32, 2](u).cast[DType.uint64]()" in code


def test_a_program_function_named_like_a_numeric_built_in_stays_a_call(tmp_path):
    source = tmp_path / "user.cgl"
    source.write_text(
        KERNEL.format(body="outp[i] = clz(k);").replace(
            "        void main() {",
            "        int clz(int x) { return x + 1; }\n        void main() {",
        )
    )

    code = crosstl.translate(str(source), backend="mojo")

    assert "count_leading_zeros" not in code
    assert "clz(k)" in code
