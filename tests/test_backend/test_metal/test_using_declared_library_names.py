"""Library overloads a using-declaration brings in beside a source function of the same name.

``using metal::max;`` adds the library's ``max`` overloads to the global overload set, where the source declares
``bfloat max(bfloat, bfloat)`` too ([namespace.udecl]/1). ``max(h0, h1)`` on ``half`` values then selects the
library's exact match, and only ``bfloat`` arguments select the source function ([over.match.best]). The importer
used to bind every call to the source function, converting ``half`` and ``float`` operands to ``bfloat``.
"""

import crosstl
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser

SOURCE = """#include <metal_stdlib>
using metal::max;
bfloat max(bfloat a, bfloat b) { return a > b ? a : b; }
kernel void k(device half* h [[buffer(0)]], device bfloat* b [[buffer(1)]],
              device float* f [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  h[i] = max(h[i], h[i + 1]);
  b[i] = max(b[i], b[i + 1]);
  f[i] = max(f[i], 2.0f);
}
"""


def _translate(tmp_path, backend):
    path = tmp_path / "kernel.metal"
    path.write_text(SOURCE)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


def test_parser_records_using_declarations():
    source = "using metal::max;\nnamespace n { using metal::min; }\n"
    ast = MetalParser(MetalLexer(source).tokenize()).parse()
    assert ast.using_declarations == [
        {"namespace": "", "target": "metal::max"},
        {"namespace": "n", "target": "metal::min"},
    ]


def test_exact_library_matches_select_the_library_overload(tmp_path):
    cgl = _translate(tmp_path, "cgl")
    assert "buffer_store(h, i, max(buffer_load(h, i), buffer_load(h, i + 1)));" in cgl
    assert "buffer_store(f, i, max(buffer_load(f, i), 2.0f));" in cgl


def test_source_overload_keeps_its_own_arguments(tmp_path):
    cgl = _translate(tmp_path, "cgl")
    assert "bfloat16 max__metal_overload_1(bfloat16 a, bfloat16 b)" in cgl
    assert (
        "buffer_store(b, i, max__metal_overload_1(buffer_load(b, i), "
        "buffer_load(b, i + 1)));" in cgl
    )


def test_metal_round_trip_keeps_half_and_float_operands(tmp_path):
    metal = _translate(tmp_path, "metal")
    assert "h[i] = max(h[i], h[i + 1]);" in metal
    assert "b[i] = max__metal_overload_1(b[i], b[i + 1]);" in metal
    assert "bfloat(h[i])" not in metal
