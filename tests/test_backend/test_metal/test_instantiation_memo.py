"""The process-wide memo of explicit Metal template instantiations."""

import ast
import inspect
import threading
from pathlib import Path

import pytest

from crosstl.backend.Metal import instantiation_memo
from crosstl.backend.Metal import preprocessor as preprocessor_module
from crosstl.backend.Metal.preprocessor import MetalPreprocessor

SOURCE = """
template <typename T>
[[kernel]] void scale(device T* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    out[i] = out[i] * T(2);
}
template [[host_name("scale_float")]] [[kernel]] void scale<float>(device float*, uint);
template [[host_name("scale_int")]] [[kernel]] void scale<int>(device int*, uint);
"""

DEDUCED = (
    "template <typename T> void consume(Box<T> value) {}\n"
    "template void consume(Box_float value);\n"
)


@pytest.fixture(autouse=True)
def isolated_memo():
    instantiation_memo.clear_memo()
    yield
    instantiation_memo.clear_memo()


def _summary(instantiations):
    return [
        (item.host_name, item.function_name, list(item.template_arguments), item.span)
        for item in instantiations
    ]


def test_fresh_preprocessors_reuse_an_equal_scan():
    expected = _summary(
        MetalPreprocessor()._scan_project_template_instantiations(SOURCE)
    )
    assert [host for host, *_ in expected] == ["scale_float", "scale_int"]

    first = MetalPreprocessor()._find_project_template_instantiations(SOURCE)
    # An equal but distinct string object must hit the same entry.
    second = MetalPreprocessor()._find_project_template_instantiations(
        SOURCE.encode().decode()
    )
    assert _summary(first) == _summary(second) == expected
    info = instantiation_memo.memo_info()
    assert (info["misses"], info["hits"], info["size"]) == (1, 1, 1)


def test_results_are_detached_from_the_memo():
    first = MetalPreprocessor()._find_project_template_instantiations(SOURCE)
    first[0].template_arguments.append("Unexpected")
    first.clear()
    again = MetalPreprocessor()._find_project_template_instantiations(SOURCE)
    assert again[0].template_arguments == ["float"]
    again[1].template_arguments[0] = "half"
    assert MetalPreprocessor()._find_project_template_instantiations(SOURCE)[
        1
    ].template_arguments == ["int"]


def test_changed_source_is_scanned_again():
    MetalPreprocessor()._find_project_template_instantiations(SOURCE)
    changed = SOURCE.replace('"scale_int"', '"scale_i32"')
    result = MetalPreprocessor()._find_project_template_instantiations(changed)
    assert [item.host_name for item in result] == ["scale_float", "scale_i32"]
    assert instantiation_memo.memo_info()["misses"] == 2


def test_struct_specialization_state_is_part_of_the_key():
    preprocessor = MetalPreprocessor()
    assert preprocessor._find_project_template_instantiations(DEDUCED) == []
    preprocessor._materialized_struct_specializations["Box_float"] = (
        "Box",
        ("float",),
    )
    entries = preprocessor._find_project_template_instantiations(DEDUCED)
    assert [item.template_arguments for item in entries] == [["float"]]
    # A fresh preprocessor without that specialization must not see it.
    assert MetalPreprocessor()._find_project_template_instantiations(DEDUCED) == []
    preprocessor._materialized_struct_specializations.clear()
    assert preprocessor._find_project_template_instantiations(DEDUCED) == []
    assert instantiation_memo.memo_info()["hits"] == 2


def test_instance_overrides_and_subclasses_always_scan(monkeypatch):
    MetalPreprocessor()._find_project_template_instantiations(SOURCE)
    patched = MetalPreprocessor()
    monkeypatch.setattr(patched, "_find_raw_mlx_kernel_instantiations", lambda code: [])
    assert len(patched._find_project_template_instantiations(SOURCE)) == 2

    class Narrow(MetalPreprocessor):
        def _find_declared_template_instantiations(self, code):
            return []

        def _find_declared_mlx_kernel_instantiations(self, code):
            return []

    assert Narrow()._find_project_template_instantiations(SOURCE) == []
    info = instantiation_memo.memo_info()
    assert (info["misses"], info["hits"], info["bypassed"]) == (1, 0, 2)


def test_replaced_class_methods_do_not_reuse_older_results(monkeypatch):
    assert len(MetalPreprocessor()._find_project_template_instantiations(SOURCE)) == 2
    monkeypatch.setattr(
        MetalPreprocessor,
        "_find_declared_template_instantiations",
        lambda self, code: [],
    )
    monkeypatch.setattr(
        MetalPreprocessor,
        "_find_declared_mlx_kernel_instantiations",
        lambda self, code: [],
    )
    assert MetalPreprocessor()._find_project_template_instantiations(SOURCE) == []
    monkeypatch.undo()
    assert len(MetalPreprocessor()._find_project_template_instantiations(SOURCE)) == 2
    assert instantiation_memo.memo_info()["hits"] == 0


def test_failed_scans_are_not_stored(monkeypatch):
    calls = []
    original = MetalPreprocessor._find_declared_template_instantiations

    def fail_once(self, code):
        calls.append(code)
        if len(calls) == 1:
            raise ValueError("scan failed")
        return original(self, code)

    monkeypatch.setattr(
        MetalPreprocessor, "_find_declared_template_instantiations", fail_once
    )
    with pytest.raises(ValueError, match="scan failed"):
        MetalPreprocessor()._find_project_template_instantiations(SOURCE)
    assert instantiation_memo.memo_info()["size"] == 0
    assert len(MetalPreprocessor()._find_project_template_instantiations(SOURCE)) == 2


def test_memo_is_bounded():
    limit = instantiation_memo.MEMO_ENTRY_LIMIT
    for index in range(limit + 1):
        MetalPreprocessor()._find_project_template_instantiations(
            SOURCE.replace("scale_int", f"scale_{index}")
        )
    assert instantiation_memo.memo_info()["size"] == limit
    # The oldest entry was evicted, the newest is still present.
    MetalPreprocessor()._find_project_template_instantiations(
        SOURCE.replace("scale_int", "scale_0")
    )
    MetalPreprocessor()._find_project_template_instantiations(
        SOURCE.replace("scale_int", f"scale_{limit}")
    )
    info = instantiation_memo.memo_info()
    assert (info["misses"], info["hits"]) == (limit + 2, 1)


def test_concurrent_scans_agree():
    expected = _summary(
        MetalPreprocessor()._scan_project_template_instantiations(SOURCE)
    )
    results = [None] * 8

    def scan(index):
        results[index] = _summary(
            MetalPreprocessor()._find_project_template_instantiations(SOURCE)
        )

    threads = [threading.Thread(target=scan, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == [expected] * 8


def _class_methods(tree):
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MetalPreprocessor"
    )
    return {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}


def _self_attributes(function):
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            yield node


def test_memo_key_names_every_state_the_scan_reads():
    """Guard the purity assumption against future edits of the scan.

    Every attribute that a method reachable from the scan reads must be a
    method, a class constant, a cache or one of the memo's state attributes,
    and no reachable method may assign preprocessor state.
    """
    tree = ast.parse(Path(inspect.getfile(preprocessor_module)).read_text("utf-8"))
    methods = _class_methods(tree)
    reachable, pending = set(), ["_scan_project_template_instantiations"]
    while pending:
        name = pending.pop()
        if name in reachable or name not in methods:
            continue
        reachable.add(name)
        pending.extend(node.attr for node in _self_attributes(methods[name]))
    assert "_find_declared_template_instantiations" in reachable

    state = set(instantiation_memo.METAL_INSTANTIATION_STATE)
    unexpected = set()
    for name in sorted(reachable):
        for node in _self_attributes(methods[name]):
            attribute = node.attr
            if isinstance(node.ctx, ast.Store) and not attribute.endswith("_cache"):
                unexpected.add(f"{name} assigns {attribute}")
            if (
                attribute in state
                or attribute.endswith("_cache")
                or attribute.lstrip("_").isupper()
                or callable(getattr(MetalPreprocessor, attribute, None))
            ):
                continue
            unexpected.add(f"{name} reads {attribute}")
    assert unexpected == set()
    preprocessor = MetalPreprocessor()
    assert all(hasattr(preprocessor, name) for name in state)
    assert set(instantiation_memo.METAL_INSTANTIATION_GUARDS) <= reachable
