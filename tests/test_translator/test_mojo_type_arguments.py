"""The Mojo generator refuses a type argument that names no CrossGL type (mojo_generic_types.py).

An importer that writes a native generic type verbatim, such as Mojo's ``Array[SIMD[DType.float32, 4], 4]``, gives
the CrossGL parser an expression (``DType.float32``) among the type's arguments. The generator raised a ``TypeError``
on it; it now raises the typed refusal every other unspellable type gets.
"""

import pytest

import crosstl
from crosstl.translator.refusals import MojoRefusal


def test_an_expression_among_type_arguments_is_refused(tmp_path):
    path = tmp_path / "probe.cgl"
    path.write_text("""
struct Params {
    Array[SIMD[DType.float32, 4], 4] w;
}

@compute
@workgroup_size(1, 1, 1)
fn probe(@group(0) @binding(0) var<storage, read_write> outp: array<float>, Params params) {
    outp[0] = 1.0;
}
""")
    with pytest.raises(MojoRefusal, match="names no CrossGL type"):
        crosstl.translate(str(path), backend="mojo")
