"""A struct template with a short upper-case name is a type, not an unresolved template placeholder."""

import pytest

import crosstl
from crosstl.project.pipeline import _unresolved_metal_template_placeholders_in_type

SHORT_NAMED_TEMPLATE = """
#include <metal_stdlib>
using namespace metal;

template <typename T>
struct NAME {
    T m;
    T d;
};

template <typename T, typename R>
struct Holder {
    R value;
    T scale;
    R get() const { return value; }
};

template <typename T>
void store_pair(device T* out, uint gid) {
    Holder<T, NAME<T>> h;
    h.value.m = T(1);
    h.value.d = T(2);
    h.scale = T(3);
    out[gid] = h.get().m + h.get().d + h.scale;
}

kernel void store_float(device float* out [[buffer(0)]],
                        uint gid [[thread_position_in_grid]]) {
    store_pair<float>(out, gid);
}
"""


@pytest.mark.parametrize("name", ["MD", "RMS", "Vec"])
@pytest.mark.parametrize("backend", ["directx", "opengl", "vulkan"])
def test_short_struct_template_names_translate(tmp_path, name, backend):
    path = tmp_path / "holder.metal"
    path.write_text(SHORT_NAMED_TEMPLATE.replace("NAME", name))

    output = crosstl.translate(str(path), backend=backend, format_output=False)

    if backend != "vulkan":
        assert f"Holder_float_{name}_float" in output


def test_declared_type_names_are_not_placeholders():
    assert (
        _unresolved_metal_template_placeholders_in_type(
            "Holder_float_MD_float",
            set(),
            {"Holder_float_MD_float"},
            {"Holder", "MD"},
        )
        == []
    )
    assert (
        _unresolved_metal_template_placeholders_in_type(
            "Holder<float, RGB>", set(), {"RGB"}
        )
        == []
    )


def test_unbound_placeholders_are_still_reported():
    assert _unresolved_metal_template_placeholders_in_type(
        "Holder_float_BN", set(), {"Holder_float_BN"}, {"Holder"}
    ) == ["BN"]
    assert _unresolved_metal_template_placeholders_in_type(
        "Holder<T, MD<T>>", {"T"}, set(), {"Holder", "MD"}
    ) == ["T"]
