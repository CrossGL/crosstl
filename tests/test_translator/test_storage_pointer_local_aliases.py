"""Local storage pointers that reuse one name in several scopes.

OpenGL decides which access a helper's storage-pointer parameter needs by following the local pointers derived from
it. A body that declares the same pointer name in two branches, each from a different offset, used to rebind that
name back and forth forever. The bindings now settle, a name declared several ways counts as an alias of the
parameter whenever one of its declarations is, and pointers defined through each other in a cycle stop after a
bounded number of passes, so the helper is given the most demanding access.
"""

import threading

import crosstl
from crosstl.translator.codegen.storage_pointer_local_aliases import (
    bind_local_storage_pointers,
)
from tests.test_backend.test_metal.test_codegen import (
    assert_opengl_compute_validates_if_available,
)

HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

BRANCH_POINTERS = """
float helper(device float* data, int mode) {
    float total = 0.0f;
    if (mode == 0) {
        device float* p = data + 1;
        total += p[0];
    } else {
        device float* p = data + 2;
        total += p[0];
    }
    return total;
}

kernel void branch_pointers(device float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                            uint id [[thread_position_in_grid]]) {
    dst[id] = helper(src, int(id));
}
"""


def _translate_within(tmp_path, source, seconds=120):
    """Translate ``source`` to OpenGL, failing instead of hanging when it does not finish in time."""
    path = tmp_path / "pointers.metal"
    path.write_text(HEADER + source)
    result = {}

    def run():
        try:
            result["code"] = crosstl.translate(
                str(path), backend="opengl", format_output=False
            )
        except BaseException as error:  # reported on the test thread
            result["error"] = error

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(seconds)
    assert not worker.is_alive(), "OpenGL translation did not finish"
    if "error" in result:
        raise result["error"]
    return result["code"]


def test_branch_local_pointers_with_one_name_translate(tmp_path):
    generated = _translate_within(tmp_path, BRANCH_POINTERS)

    assert "data_offset + 1" in generated
    assert "data_offset + 2" in generated
    assert_opengl_compute_validates_if_available(generated, tmp_path, "pointers")


def test_store_through_one_of_the_pointers_keeps_the_buffer_writable(tmp_path):
    source = BRANCH_POINTERS.replace(
        "total += p[0];\n    }\n    return", "p[0] = 3.0f;\n    }\n    return"
    )
    generated = _translate_within(tmp_path, source)

    assert "data_offset + 2" in generated
    assert "= 3.0" in generated
    assert "readonly buffer srcBuffer" not in generated
    assert_opengl_compute_validates_if_available(generated, tmp_path, "pointers")


class _Declaration:
    def __init__(self, name, initial_value):
        self.name = name
        self.initial_value = initial_value


def _offset_binding(expression, aliases):
    """``(base, delta)`` points ``delta`` elements past ``base``; a bare name is its alias."""
    if isinstance(expression, str):
        return aliases.get(expression)
    base, delta = expression
    binding = aliases.get(base)
    if binding is None:
        return None
    return {**binding, "offset": binding["offset"] + delta}


def _always(node, aliases):
    return True


def _parameter(name):
    return {name: {"kind": "storage-pointer", "root": name, "offset": 0}}


def test_one_declaration_per_name_binds_exactly():
    aliases = _parameter("data")
    declarations = [
        _Declaration("p", ("data", 1)),
        _Declaration("q", ("p", 2)),
    ]

    assert bind_local_storage_pointers(
        declarations, aliases, _always, _offset_binding, preferred_root="data"
    )
    assert aliases["p"] == {"kind": "storage-pointer", "root": "data", "offset": 1}
    assert aliases["q"] == {"kind": "storage-pointer", "root": "data", "offset": 3}


def test_later_declaration_is_resolved_on_a_following_pass():
    aliases = _parameter("data")
    declarations = [
        _Declaration("q", ("p", 2)),
        _Declaration("p", ("data", 1)),
    ]

    assert bind_local_storage_pointers(declarations, aliases, _always, _offset_binding)
    assert aliases["q"]["offset"] == 3


def test_disagreeing_declarations_settle_on_the_preferred_root():
    aliases = {**_parameter("data"), **_parameter("other")}
    declarations = [
        _Declaration("p", ("other", 1)),
        _Declaration("p", ("data", 2)),
        _Declaration("r", ("p", 1)),
    ]

    assert bind_local_storage_pointers(
        declarations, aliases, _always, _offset_binding, preferred_root="data"
    )
    assert aliases["p"]["root"] == "data"
    assert aliases["r"]["root"] == "data"


def test_disagreeing_declarations_without_the_preferred_root_take_the_first():
    aliases = {**_parameter("data"), **_parameter("other")}
    declarations = [
        _Declaration("p", ("other", 1)),
        _Declaration("p", ("other", 2)),
    ]

    assert bind_local_storage_pointers(
        declarations, aliases, _always, _offset_binding, preferred_root="data"
    )
    assert aliases["p"] == {"kind": "storage-pointer", "root": "other", "offset": 1}


def test_declaration_without_a_binding_keeps_the_name():
    aliases = _parameter("data")
    declarations = [
        _Declaration("p", ("data", 1)),
        _Declaration("p", ("untracked", 1)),
    ]

    assert bind_local_storage_pointers(declarations, aliases, _always, _offset_binding)
    assert aliases["p"]["offset"] == 1


def test_non_pointer_declarations_are_skipped():
    aliases = _parameter("data")
    declarations = [_Declaration("count", ("data", 1))]

    assert bind_local_storage_pointers(
        declarations, aliases, lambda node, table: False, _offset_binding
    )
    assert "count" not in aliases


def test_pointer_defined_through_itself_stops_unsettled():
    # ``device float* data = data + 1;`` shadowing the parameter: every pass moves the offset again.
    aliases = _parameter("data")
    declarations = [_Declaration("data", ("data", 1))]

    assert not bind_local_storage_pointers(
        declarations, aliases, _always, _offset_binding, preferred_root="data"
    )
