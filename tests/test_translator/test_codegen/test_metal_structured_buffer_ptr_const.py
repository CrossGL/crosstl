"""ptr<T> locals initialised from StructuredBuffer params carry const.

The ``points_into_readonly_metal_storage`` check only consulted
``current_readonly_metal_parameters`` and ``current_readonly_raw_buffer_parameters``,
missing the case where a ``ptr<T>`` local is initialised from a ``StructuredBuffer<T>``
parameter whose read-only status is tracked via ``local_variable_types``.  The fix adds
a third check that inspects the tracked local type and returns ``True`` when it is a
``StructuredBuffer``, so the resulting pointer local gets the required ``const``
qualifier in Metal output.
"""

import textwrap

import crosstl

READONLY_SB = textwrap.dedent(
    """\
    shader Test {
        compute {
            @stage_entry
            void main(StructuredBuffer<float> in_ @buffer(0),
                      RWStructuredBuffer<float> out_ @buffer(1),
                      uint id @gl_GlobalInvocationID) {
                float* s = in_;
                out_[id] = s[id];
            }
        }
    }
    """
)

CHAINED_SB = textwrap.dedent(
    """\
    shader Test {
        compute {
            @stage_entry
            void main(StructuredBuffer<float> in_ @buffer(0),
                      RWStructuredBuffer<float> out_ @buffer(1),
                      uint id @gl_GlobalInvocationID) {
                float* s = in_;
                float* t = s;
                out_[id] = t[id];
            }
        }
    }
    """
)

READWRITE_SB = textwrap.dedent(
    """\
    shader Test {
        compute {
            @stage_entry
            void main(RWStructuredBuffer<float> in_ @buffer(0),
                      RWStructuredBuffer<float> out_ @buffer(1),
                      uint id @gl_GlobalInvocationID) {
                float* s = in_;
                out_[id] = s[id];
            }
        }
    }
    """
)


def _cgl(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="metal", format_output=False)


def test_readonly_structured_buffer_ptr_local_is_const(tmp_path):
    """ptr<T> local from StructuredBuffer<T> must carry const (issue #3722)."""
    code = _cgl(tmp_path, READONLY_SB)
    assert "const device float* s = in_;" in code, code
    assert "device float* s = in_;" not in code.replace("const ", ""), code


def test_chained_ptr_local_keeps_const(tmp_path):
    """A ptr<T> local derived from a const ptr<T> also carries const."""
    code = _cgl(tmp_path, CHAINED_SB)
    assert "const device float* s = in_;" in code, code
    assert "const device float* t = s;" in code, code


def test_readwrite_structured_buffer_ptr_local_has_no_const(tmp_path):
    """ptr<T> local from RWStructuredBuffer<T> must not carry const."""
    code = _cgl(tmp_path, READWRITE_SB)
    assert "const device float* s" not in code, code
    assert "device float* s = in_;" in code, code
