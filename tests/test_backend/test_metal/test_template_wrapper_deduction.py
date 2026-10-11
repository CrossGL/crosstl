"""Primary-template argument identity survives partial struct materialization."""

import pytest

from crosstl.backend.Metal.preprocessor import MetalPreprocessor, MetalStructMethodError

CASES = (
    "default",
    "explicit",
    "alias",
    "named",
    "dependent",
    "non-type",
    "primary",
    "dependent-plain",
    "non-type-plain",
)


def source(
    case="default", *, argument_type=None, pointer="device", value="values[tid]"
):
    parameter, partial = {
        "named": ("typename Enable = void", "T, enable_if_t<is_same_v<T, int>>"),
        "dependent-plain": ("typename Other = T", "T, T"),
        "non-type-plain": ("int Width = 2", "T, 2"),
        "dependent": (
            "typename Other = T, typename = void",
            "T, T, enable_if_t<is_same_v<T, int>>",
        ),
        "non-type": (
            "int Width = 2, typename = void",
            "T, 2, enable_if_t<is_same_v<T, int>>",
        ),
    }.get(case, ("typename = void", "T, enable_if_t<is_same_v<T, int>>"))
    specialization = (
        ""
        if case == "primary"
        else f"template <typename T> struct Box<{partial}> {{ T value; }};"
    )
    concrete = argument_type or ("Box<int, void>" if case == "explicit" else "Box<int>")
    aliases = ""
    if case == "alias":
        aliases = f"using output_t = {concrete};"
        concrete = "output_t"
    return f"""#include <metal_stdlib>
using namespace metal;
template <typename T, {parameter}> struct Box {{ T value; }};
{specialization}
{aliases}
struct Store {{
    template <typename T>
    void put(device Box<T>* results, T value, uint index) thread {{
        results[index].value = value;
    }}
}};
kernel void copy_wrapped(const device int* values [[buffer(0)]],
                         {pointer} {concrete}* results [[buffer(1)]],
                         uint tid [[thread_position_in_grid]]) {{
    Store store;
    store.put(results, {value}, tid + 2);
}}
"""


@pytest.mark.parametrize("case", CASES)
def test_template_wrapper_deduction_retains_default_slots(case):
    preprocessor = MetalPreprocessor()
    result = preprocessor.preprocess(source(case))
    assert "Store__put__int(store, results, values[tid], tid + 2)" in result
    assert "results[index].value = value;" in result
    if not case.endswith("-plain"):
        assert preprocessor._materialized_struct_primary_templates


@pytest.mark.parametrize(
    "change",
    (
        {"argument_type": "Box<float>"},
        {"argument_type": "Box<int, float>"},
        {"argument_type": "Box<int, void, int>"},
        {"pointer": "const device"},
        {"pointer": "constant"},
        {"value": "float(values[tid])"},
    ),
)
def test_template_wrapper_deduction_rejects_conflicts(change):
    with pytest.raises(MetalStructMethodError, match="did not bind consistently"):
        MetalPreprocessor().preprocess(source(**change))


def test_template_wrapper_metadata_does_not_leak_between_sources():
    preprocessor = MetalPreprocessor()
    preprocessor.preprocess(source())
    assert preprocessor._materialized_struct_primary_templates
    preprocessor.preprocess("kernel void empty() {}")
    assert not preprocessor._materialized_struct_primary_templates


def test_template_binding_does_not_consume_an_extra_argument_slot():
    preprocessor = MetalPreprocessor()
    preprocessor._materialized_struct_specializations["Box_int_void"] = (
        "Box",
        ("int", "void"),
    )
    bindings = {}
    preprocessor._infer_template_parameter_bindings_from_type(
        "Box<T>", "Box_int_void", {"T"}, bindings
    )
    assert bindings == {}


def test_template_wrapper_default_expansion_keeps_other_namespaces_distinct():
    preprocessor = MetalPreprocessor()
    preprocessor.preprocess(source())
    name = next(iter(preprocessor._materialized_struct_primary_templates))
    assert (
        preprocessor._canonical_template_binding_pointee_type(
            "Other::Box<int>", context=name
        )
        == "Other::Box<int>"
    )


@pytest.mark.parametrize("argument", ("int", "metal::vec<int, 2>", "Pair<int, float>"))
def test_template_wrapper_deduction_preserves_nested_argument_boundaries(argument):
    preprocessor = MetalPreprocessor()
    primary = preprocessor._find_template_structs(source())[0]
    preprocessor._materialized_struct_primary_templates["Concrete"] = primary
    preprocessor._materialized_struct_specializations["Concrete"] = (
        "Box",
        (argument, "void"),
    )
    bindings = {}
    preprocessor._infer_template_parameter_bindings_from_type(
        "Box<T>", "Concrete", {"T"}, bindings
    )
    assert bindings == {"T": preprocessor._normalize_template_argument_text(argument)}
