"""A local pointer into the constant address space can be advanced."""

import textwrap

from crosstl.project import translate_project
from tests.test_translator.test_project_translation import (
    assert_compute_glsl_validates_if_available,
    assert_directx_compute_validates_if_available,
    assert_metal_validates_if_available,
)

SOURCE = """
#include <metal_stdlib>
using namespace metal;

[[kernel]] void advance_strides(
    device int* out [[buffer(0)]],
    const constant int* strides [[buffer(1)]],
    const constant int& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  const constant auto* cursor = strides + 2 * count;
  if (count > 1) {
    cursor += count;
  }
  out[gid] = cursor[0];
}
"""


def test_constant_address_space_local_pointer_alias_can_be_advanced(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "kernel.metal").write_text(textwrap.dedent(SOURCE).strip() + "\n")

    payload = translate_project(
        repo,
        targets=["directx", "opengl", "metal"],
        output_dir="out",
        format_output=False,
    ).to_json()

    assert payload["diagnostics"] == []
    outputs = {
        artifact["target"]: (repo / artifact["path"]).read_text(encoding="utf-8")
        for artifact in payload["artifacts"]
    }
    assert set(outputs) == {"directx", "opengl", "metal"}
    assert all("unsupported" not in output for output in outputs.values())
    assert "cursor += count;" in outputs["metal"]
    assert "cursor_offset += int(count);" in outputs["opengl"]
    assert_directx_compute_validates_if_available(outputs["directx"], tmp_path)
    assert_compute_glsl_validates_if_available(outputs["opengl"], tmp_path)
    assert_metal_validates_if_available(outputs["metal"], tmp_path)
