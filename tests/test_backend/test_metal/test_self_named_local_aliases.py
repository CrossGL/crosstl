"""A function-local alias may reuse the name of the template it specializes.

``using Op = Op<T>;`` is valid C++: the alias is declared after its type-id, so ``Op<T>`` names the class template.
Template materialization must resolve such an alias exactly once.
"""

import pytest

import crosstl
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.project.pipeline import (
    _metal_resolve_type_identifiers,
    _metal_rewrite_embedded_identifier_parts,
)

SELF_NAMED_ALIAS_TEMPLATE = """
#include <metal_stdlib>
using namespace metal;

template <typename T>
struct SumOp {
    T operator()(T a, T b) const { return a + b; }
};

template <typename T, typename OP>
struct Combiner {
    T run(T a, T b) const { OP op; return op(a, b); }
};

template <typename T>
void combine_values(device T* out, uint gid) {
    DECLARATION
    Combiner<T, SumOp> combiner;
    out[gid] = combiner.run(out[gid], T(1));
}

kernel void combine_float(device float* out [[buffer(0)]],
                          uint gid [[thread_position_in_grid]]) {
    combine_values<float>(out, gid);
}

kernel void combine_int(device int* values [[buffer(0)]],
                        uint gid [[thread_position_in_grid]]) {
    combine_values<int>(values, gid);
}
"""

DECLARATIONS = {
    "using": "using SumOp = SumOp<T>;",
    "typedef": "typedef SumOp<T> SumOp;",
}


@pytest.mark.parametrize("declaration", sorted(DECLARATIONS))
@pytest.mark.parametrize("backend", ["directx", "opengl", "vulkan", "cgl"])
def test_self_named_local_alias_specializes_the_template_once(
    tmp_path, declaration, backend
):
    source = tmp_path / "combine.metal"
    source.write_text(
        SELF_NAMED_ALIAS_TEMPLATE.replace("DECLARATION", DECLARATIONS[declaration])
    )

    output = crosstl.translate(str(source), backend=backend, format_output=False)

    assert "SumOp_float_float" not in output
    assert "SumOp_int_int" not in output
    if backend != "vulkan":
        assert "Combiner_float_SumOp_float" in output
        assert "Combiner_int_SumOp_int" in output


def test_type_identifier_rewrite_leaves_the_aliased_template_name():
    resolved = _metal_resolve_type_identifiers(
        MetalPreprocessor(),
        "Combiner<float, SumOp, Indexer>",
        aliases={"SumOp": "SumOp<float>", "Indexer": "Index<uint>"},
        constants={},
    )

    assert resolved == "Combiner<float,SumOp<float>,Index<uint>>"


def test_preprocessor_alias_resolution_leaves_the_aliased_template_name():
    preprocessor = MetalPreprocessor()
    code = "void f() {\n    using SumOp = SumOp<float>;\n    Combiner<SumOp> c;\n}\n"
    aliases = preprocessor._collect_local_type_alias_bindings(code, [(0, len(code))])
    position = code.index("Combiner")

    assert (
        preprocessor._resolve_type_aliases_at("Combiner<SumOp>", aliases, position)
        == "Combiner<SumOp<float>>"
    )
    assert (
        preprocessor._resolve_type_aliases_at("SumOp<float>", aliases, position)
        == "SumOp<float>"
    )


def test_local_type_aliases_are_not_unresolved_integral_constants():
    preprocessor = MetalPreprocessor()
    code = (
        "void f() {\n"
        "    using SumOp = SumOp<float>;\n"
        "    typedef SumOp<int> IntOp;\n"
        "    int count = 3;\n"
        "}\n"
    )
    bindings = preprocessor._collect_local_integral_constant_bindings(
        code, [(0, len(code))]
    )

    assert bindings["SumOp"][0].type_alias
    assert bindings["IntOp"][0].type_alias
    assert not bindings["count"][0].type_alias
    assert bindings["count"][0].value is None


def test_embedded_alias_parts_already_followed_by_their_arguments_are_kept():
    replacements = {"SumOp": "SumOp_float", "N": "4"}

    assert _metal_rewrite_embedded_identifier_parts(
        "Combiner_float_SumOp_float", replacements
    ) == ("Combiner_float_SumOp_float", False)
    assert _metal_rewrite_embedded_identifier_parts(
        "Combiner_float_SumOp", replacements
    ) == ("Combiner_float_SumOp_float", True)
    assert _metal_rewrite_embedded_identifier_parts("Tile_N", replacements) == (
        "Tile_4",
        True,
    )
