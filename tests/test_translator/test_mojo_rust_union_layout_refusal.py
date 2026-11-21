"""Mojo and Rust output refuse a CrossGL ``@union_layout`` struct.

Neither generator lowers a union, and a plain struct gives each overlapping member its own storage, so a program
that writes one member and reads another would read a field that was never written (issue #4774).
"""

import pytest

import crosstl
from crosstl.translator.refusals import MojoRefusal, RustRefusal

UNION_SOURCE = """
shader main {
    @union_layout(4, 4, little_endian, metal)
    struct Bits {
        @union_member_layout(0, 4, 4) float f;
        @union_member_layout(0, 4, 4) uint u;
    }

    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* inp;
        buffer uint* out_;

        void main() {
            uint i = gl_GlobalInvocationID.x;
            Bits b;
            b.f = inp[i];
            out_[i] = b.u;
        }
    }
}
"""

PLAIN_STRUCT_SOURCE = UNION_SOURCE.replace(
    "@union_layout(4, 4, little_endian, metal)\n", ""
).replace("@union_member_layout(0, 4, 4) ", "")


@pytest.fixture
def union_file(tmp_path):
    path = tmp_path / "union.cgl"
    path.write_text(UNION_SOURCE)
    return path


@pytest.mark.parametrize(
    ("backend", "refusal"), [("mojo", MojoRefusal), ("rust", RustRefusal)]
)
def test_union_layout_struct_is_refused(union_file, backend, refusal):
    with pytest.raises(refusal, match=r"struct 'Bits' has a @union_layout"):
        crosstl.translate(str(union_file), backend=backend)


@pytest.mark.parametrize("backend", ["mojo", "rust"])
def test_struct_without_union_layout_is_still_written(tmp_path, backend):
    path = tmp_path / "plain.cgl"
    path.write_text(PLAIN_STRUCT_SOURCE)
    code = crosstl.translate(str(path), backend=backend)
    assert "struct Bits" in code
