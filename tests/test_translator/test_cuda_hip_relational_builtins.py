"""CUDA and HIP output spells GLSL's vector relational built-ins as the componentwise operators they are.

CrossGL accepts GLSL 4.60 section 8.7's ``lessThan`` ... ``notEqual`` and ``not``. Those names exist only in GLSL, and
the CUDA and HIP emitters wrote them through verbatim, so nvcc and hipcc rejected every call as an undeclared
identifier. Both emitters now lower them into the componentwise operators first, as the HLSL, MSL, WGSL and Slang
emitters do.
"""

import pytest

import crosstl

KERNEL = """shader RelationalBuiltins {
    compute {
        layout(local_size_x = 64) in;
        shared vec4 values[65];

        void main() {
            uint i = gl_LocalInvocationID.x;
            vec3 a = values[i].xyz;
            vec3 b = values[i + 1u].xyz;
            bvec3 lt = lessThan(a, b);
            bvec3 le = lessThanEqual(a, b);
            bvec3 gt = greaterThan(a, b);
            bvec3 eq = equal(a, b);
            bvec3 ne = notEqual(a, b);
            bvec3 n = not(lt);
            ivec2 p = ivec2(1, 2);
            ivec2 q = ivec2(3, 0);
            bvec2 ge = greaterThanEqual(p, q);
            bvec3 direct = a < b;
            if (lt.x && le.y && gt.z && eq.x && ne.y && n.z && ge.x && direct.y) {
                values[i] = vec4(1.0);
            }
        }
    }
}
"""

GLSL_RELATIONAL_BUILTINS = (
    "lessThan(",
    "lessThanEqual(",
    "greaterThan(",
    "greaterThanEqual(",
    "equal(",
    "notEqual(",
    "not(",
)


def translate(tmp_path, backend, source=KERNEL):
    path = tmp_path / "kernel.cgl"
    path.write_text(source)
    code = crosstl.translate(str(path), backend=backend, format_output=False)
    return code[code.index("void compute_main") :]


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_relational_builtins_are_componentwise_operators(tmp_path, backend):
    lines = {line.strip() for line in translate(tmp_path, backend).splitlines()}

    for statement in (
        "uchar3 lt = make_uchar3((a.x < b.x), (a.y < b.y), (a.z < b.z));",
        "uchar3 le = make_uchar3((a.x <= b.x), (a.y <= b.y), (a.z <= b.z));",
        "uchar3 gt = make_uchar3((a.x > b.x), (a.y > b.y), (a.z > b.z));",
        "uchar3 eq = make_uchar3((a.x == b.x), (a.y == b.y), (a.z == b.z));",
        "uchar3 ne = make_uchar3((a.x != b.x), (a.y != b.y), (a.z != b.z));",
        "uchar3 n = cgl_uchar3_not(lt);",
        "uchar2 ge = make_uchar2((p.x >= q.x), (p.y >= q.y));",
    ):
        assert statement in lines


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_no_glsl_relational_builtin_reaches_the_output(tmp_path, backend):
    code = translate(tmp_path, backend)

    for builtin in GLSL_RELATIONAL_BUILTINS:
        assert f" {builtin}" not in code and f"({builtin}" not in code


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_builtin_matches_the_operator_it_names(tmp_path, backend):
    lines = {line.strip() for line in translate(tmp_path, backend).splitlines()}

    lowered = next(line for line in lines if line.startswith("uchar3 lt ="))
    direct = next(line for line in lines if line.startswith("uchar3 direct ="))
    assert lowered.split("=", 1)[1] == direct.split("=", 1)[1]


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_user_function_named_like_a_builtin_is_called(tmp_path, backend):
    source = KERNEL.replace(
        "    compute {",
        "    bool equal(vec3 x, vec3 y) { return x.x == y.x; }\n    compute {",
    ).replace("bvec3 eq = equal(a, b);", "bvec3 eq = bvec3(equal(a, b));")
    lines = {line.strip() for line in translate(tmp_path, backend, source).splitlines()}

    assert any("equal(a, b)" in line for line in lines if line.startswith("uchar3 eq"))
    assert "uchar3 lt = make_uchar3((a.x < b.x), (a.y < b.y), (a.z < b.z));" in lines
