"""The preprocessor subset that reads headers whose definitions depend on macros."""

import pytest

from tools.coverage.features import preprocess


def test_keep_evaluates_conditions_with_the_given_macros():
    header = "\n".join(
        [
            "#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 300",
            "int sm30();",
            "#if __CUDA_ARCH__ >= 900",
            "int sm90();",
            "#elif CUDA_VERSION >= 0x2328 && !defined(OFF)",
            "int cuda9();",
            "#else",
            "int other();",
            "#endif",
            "#endif",
            "#ifndef GUARD",
            "#define GUARD",
            "int once();",
            "#endif",
            "#ifdef GUARD",
            "int guarded(); // a comment",
            "#endif",
        ]
    )
    kept = preprocess.keep(header, {"__CUDA_ARCH__": 800, "CUDA_VERSION": 13000})
    assert [line for line in kept.splitlines() if "(" in line] == [
        "int sm30();",
        "int cuda9();",
        "int once();",
        "int guarded(); ",
    ]


def test_keep_drops_nested_blocks_of_a_dropped_branch():
    header = "#if 0\n#if 1\nint a();\n#else\nint b();\n#endif\n#else\nint c();\n#endif"
    assert preprocess.keep(header, {}).split() == ["int", "c();"]


def test_keep_rejects_a_condition_it_cannot_evaluate():
    with pytest.raises(ValueError):
        preprocess.keep("#if __has_builtin(__nvvm_activemask)\n#endif", {})


def test_expand_substitutes_pastes_and_rescans():
    header = "\n".join(
        [
            "#define ONE(__Name, __Type) __Type __Name(const __Type *p) { return #__Name; }",
            "#define FAMILY(__Mode) \\",
            "  ONE(__ld##__Mode, int) \\",
            "  ONE(__ld##__Mode##x, float)",
            "FAMILY(cg)",
            '#pragma pop_macro("ONE")',
        ]
    )
    assert preprocess.expand(header).split() == [
        *'int __ldcg(const int *p) { return "__ldcg"; }'.split(),
        *'float __ldcgx(const float *p) { return "__ldcgx"; }'.split(),
    ]


def test_expand_keeps_commas_inside_parentheses_and_strings_in_one_argument():
    header = '#define CALL(f, x) f(x)\nCALL(g, h(1, ","))'
    assert preprocess.expand(header).strip() == 'g(h(1, ","))'
