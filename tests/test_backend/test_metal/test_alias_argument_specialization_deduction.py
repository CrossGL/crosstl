"""Template deduction through a struct specialization named from a type alias.

A pointer argument of type ``device Box<float16_t>*`` names the specialization
``Box<half>`` when ``float16_t`` aliases ``half`` ([dcl.typedef]); deduction of
``T`` in a parameter ``device Box<T>*`` must compare the two types after the
same alias canonicalization ([temp.deduct.call]). The materialized struct keeps
the spelling it was named from, so the comparison has to canonicalize again
after the specialization is expanded.
"""

import pytest

from crosstl.backend.Metal.preprocessor import MetalPreprocessor, MetalStructMethodError

ALIASES = {
    "typedef": "typedef half float16_t;",
    "using": "using float16_t = half;",
}


def source(alias="typedef", *, value="values[tid]", argument="float16_t"):
    return f"""#include <metal_stdlib>
using namespace metal;
{ALIASES[alias]}
template <typename T, typename = void> struct Box {{ uint bits; }};
template <typename T> struct Box<T, enable_if_t<is_same_v<T, float>>> {{ T value; }};
struct Store {{
    template <typename T>
    void put(device Box<T>* results, T value, uint index) thread {{
        results[index].bits = as_type<ushort>(value);
    }}
}};
template <typename T, typename Op>
void put_all(device Box<T>* results, const device T* values, uint tid) {{
    Op store;
    store.put(results, {value}, tid);
}}
kernel void store_halves(const device float16_t* values [[buffer(0)]],
                         device Box<{argument}>* results [[buffer(1)]],
                         uint tid [[thread_position_in_grid]]) {{
    put_all<{argument}, Store>(results, values, tid);
}}
"""


@pytest.mark.parametrize("alias", sorted(ALIASES))
@pytest.mark.parametrize("argument", ("float16_t", "half"))
def test_member_template_deduces_through_aliased_specialization(alias, argument):
    result = MetalPreprocessor().preprocess(source(alias, argument=argument))
    assert "Store__put__" in result
    assert "results[index].bits = as_type<ushort>(value);" in result
    assert "uint bits;" in result


@pytest.mark.parametrize("alias", sorted(ALIASES))
def test_member_template_rejects_conflicting_aliased_specialization(alias):
    with pytest.raises(MetalStructMethodError, match="did not bind consistently"):
        MetalPreprocessor().preprocess(source(alias, value="float(values[tid])"))
