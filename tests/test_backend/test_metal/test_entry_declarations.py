"""A Metal function is a stage entry point only when its declaration says so.

``kernel``, ``vertex`` and the other stage qualifiers, or ``[[kernel]]``, precede the parameter list. Reachability
starts from the entry points, so a helper whose parameter is named ``object`` or ``mesh``, or whose header follows a
comment that mentions kernels, must not become a root: everything it calls would then count as reachable, and the
template materialization of an entry-scoped translation would record specializations no entry point uses.
"""

import pytest

from crosstl.backend.Metal.preprocessor import MetalPreprocessor

SOURCE = """#include <metal_stdlib>
using namespace metal;
inline void store_object(device atomic_uint* object, uint value) {
  atomic_store_explicit(object, value, memory_order_relaxed);
}
inline uint mesh_count(uint mesh, uint vertex) { return mesh * vertex; }
// Reduction kernels call this helper.
inline uint reduce_helper(uint value) { return value + 1u; }
[[host_name("vertex_copy")]] inline uint named_helper(uint value) { return value; }
kernel void k(device atomic_uint* out [[buffer(0)]], device uint* values [[buffer(1)]],
              uint tid [[thread_position_in_grid]]) {
  values[tid] = reduce_helper(values[tid]);
}
[[kernel]] void attributed(device uint* values [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  values[tid] = 0u;
}
"""


def _functions():
    preprocessor = MetalPreprocessor()
    spans = preprocessor._find_template_declaration_spans(SOURCE)
    functions = preprocessor._find_non_template_function_definitions(SOURCE, spans)
    return preprocessor, spans, {function.name: function for function in functions}


@pytest.mark.parametrize(
    "name,is_entry",
    (
        ("store_object", False),
        ("mesh_count", False),
        ("reduce_helper", False),
        ("named_helper", False),
        ("k", True),
        ("attributed", True),
    ),
)
def test_only_declared_stage_functions_are_entries(name, is_entry):
    _preprocessor, _spans, functions = _functions()
    assert functions[name].is_entry is is_entry


def test_helpers_reachable_from_no_entry_are_unreachable():
    preprocessor, spans, functions = _functions()
    reachable = set(preprocessor._reachable_function_spans(SOURCE, spans))
    assert functions["reduce_helper"].span in reachable
    for name in ("store_object", "mesh_count", "named_helper"):
        assert functions[name].span not in reachable, name
