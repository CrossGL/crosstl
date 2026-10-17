"""DirectX can keep ``groupshared`` bools in ``uint`` words.

DXC loads a ``groupshared bool`` word with ``!range [0, 2)``. MLX's
contiguous scan writes one ``simdgroup_sums`` slot per SIMD-group and then
reads all 32 slots, so loads of words the dispatch never wrote are undefined
in DXIL rather than unspecified values. ``groupshared_bool_storage = "uint"``
declares such variables as ``uint`` words, which HLSL converts to and from
``bool`` at every use.
"""

import pytest

from crosstl.project import ProjectConfig, translate_project
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from tests.test_translator.test_metal_builtin_ownership import _compile

SOURCE = """#include <metal_stdlib>
using namespace metal;

[[kernel]] void partial_flags(
    const device bool* input [[buffer(0)]],
    device bool* output [[buffer(1)]],
    device float* totals [[buffer(2)]],
    uint lid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]) {
  threadgroup bool slots[32];
  threadgroup bool grid[4][8];
  threadgroup bool any_set;
  threadgroup float weights[32];
  if (lid < 4) {
    slots[lid] = input[gid * 32 + lid];
    grid[lid][lid] = !input[gid * 32 + lid];
  }
  if (lid == 0) {
    any_set = input[gid * 32];
  }
  weights[lid] = float(lid);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Every lane reads its slot, written or not; only lane 0's result is kept.
  bool mine = slots[lid];
  bool other = grid[lid % 4][lid % 8] || any_set;
  if (lid == 0) {
    output[gid] = mine && other;
    totals[gid] = slots[0] ? weights[1] : weights[2];
  }
}
"""


def _translate(tmp_path, target, storage=None):
    (tmp_path / "flags.metal").write_text(SOURCE, encoding="utf-8")
    options = {}
    if storage is not None:
        options = {"target_options": {target: {"groupshared_bool_storage": storage}}}
    config = ProjectConfig(
        root=tmp_path,
        include_patterns=("flags.metal",),
        targets=(target,),
        output_dir="out",
        entry_points={"flags.metal": ("partial_flags",)},
        workgroup_size=(32, 1, 1),
        source_options={"metal": options} if options else {},
    )
    payload = translate_project(config, format_output=False).to_json()
    return payload


def _generated(tmp_path, payload):
    assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
    (artifact,) = payload["artifacts"]
    return (tmp_path / artifact["path"]).read_text(encoding="utf-8")


def _groupshared(generated):
    return sorted(
        line.strip()
        for line in generated.splitlines()
        if line.startswith("groupshared ") and "__crossgl" not in line
    )


def test_default_keeps_groupshared_bool_declarations(tmp_path):
    generated = _generated(tmp_path, _translate(tmp_path, "directx"))
    declarations = _groupshared(generated)
    assert [item.split()[1] for item in declarations] == [
        "bool",
        "bool",
        "bool",
        "float",
    ]
    _compile(generated, "directx", tmp_path)


def test_uint_storage_declares_bool_words(tmp_path):
    generated = _generated(tmp_path, _translate(tmp_path, "directx", "uint"))
    declarations = _groupshared(generated)
    # Scalars and arrays of any rank change; other element types do not.
    assert [item.split()[1] for item in declarations] == [
        "float",
        "uint",
        "uint",
        "uint",
    ]
    assert any(item.endswith("[4][8];") for item in declarations), declarations
    assert any(
        item.endswith("[32];") and " uint " in f" {item} " for item in declarations
    )
    assert "groupshared bool" not in generated
    # Everything else is the default translation.
    default = _generated(tmp_path / "default", _translate_into(tmp_path / "default"))
    assert _without_groupshared(generated) == _without_groupshared(default)
    _compile(generated, "directx", tmp_path)


def _translate_into(directory):
    directory.mkdir()
    return _translate(directory, "directx")


def _without_groupshared(generated):
    return [
        line for line in generated.splitlines() if not line.startswith("groupshared ")
    ]


def test_bool_storage_option_is_directx_only(tmp_path):
    payload = _translate(tmp_path, "opengl", "uint")
    assert payload["summary"]["failedCount"] == 1
    assert any(
        "groupshared_bool_storage is supported only by the DirectX target" in item[
            "message"
        ]
        for item in payload["diagnostics"]
    ), payload["diagnostics"]


@pytest.mark.parametrize("storage", ["int", "UINT ", "", "bits"])
def test_bool_storage_option_rejects_other_words(tmp_path, storage):
    payload = _translate(tmp_path, "directx", storage)
    assert payload["summary"]["failedCount"] == 1
    assert any(
        "groupshared_bool_storage must be 'bool' or 'uint'" in item["message"]
        for item in payload["diagnostics"]
    ), payload["diagnostics"]


def test_codegen_validates_bool_storage():
    assert HLSLCodeGen().groupshared_bool_storage == "bool"
    assert HLSLCodeGen(groupshared_bool_storage=" UINT ").groupshared_bool_storage == (
        "uint"
    )
    with pytest.raises(TypeError, match="must be a string"):
        HLSLCodeGen(groupshared_bool_storage=True)
    with pytest.raises(ValueError, match="must be one of: bool, uint"):
        HLSLCodeGen(groupshared_bool_storage="int")
