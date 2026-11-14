"""Constructor initializers that read pointer parameters or pointer members.

A struct whose pointer members are promoted out keeps its scalar members,
and the rebuilt constructor keeps their initializers. An initializer such as
``ld(params_[0])`` reads a pointer parameter, and ``step(params[1] + 1)``
reads a promoted member initialized directly from one. The rebuilt
constructor keeps those parameters, and every construction passes them, so
the initializer never names a dropped parameter. A member initialized from a
derived pointer is not promoted, so the struct keeps its ordinary lowering.

A ``constant`` pointer member is an assignable pointer: ``constant`` names
the pointee's address space.
"""

import pytest

from crosstl.backend.Metal.identifier_rewrites import (
    qualified_identifiers,
    replace_unqualified_identifiers,
    unqualified_identifiers,
)
from crosstl.project import ProjectConfig, translate_project
from tests.test_translator.test_metal_builtin_ownership import _compile

READER = """
#include <metal_stdlib>
using namespace metal;

struct Params {
  int scale;
  int strides[2];
};

template <typename T>
struct Reader {
  const device T* src;
  const constant PARAM* params;
  const int ld;
  const int step;

  Reader(const device T* src_, const constant PARAM* params_) thread
      : INITIALIZERS {}

  T read(int i) const thread {
    return src[i * ld] * T(SCALE) + T(step);
  }
};

template <typename T>
[[kernel]] void run(const device T* in [[buffer(0)]],
                    device T* out [[buffer(1)]],
                    const constant PARAM* params [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
  thread Reader<T> reader(in, params);
  out[i] = reader.read(int(i));
}

template [[host_name("run_int")]] [[kernel]] void run<int>(
    const device int*, device int*, const constant PARAM*, uint);
"""

CASES = {
    # Promoted: the initializers read a parameter and a direct member.
    "parameter-and-member": (
        "int",
        "src(src_), params(params_), ld(params_[0]), step(params[1] + 1)",
        "params[2]",
    ),
    "struct-parameter": (
        "Params",
        "src(src_), params(params_), ld(params_->strides[0]), "
        "step(params->scale + 1)",
        "params->scale",
    ),
    # Not promoted: ``params`` is derived from its parameter.
    "derived-member": (
        "int",
        "src(src_ + 1), params(params_ + 1), ld(params_[0]), step(params[0] + 1)",
        "params[1]",
    ),
    # Not promoted: ``this->params`` cannot be mapped to the parameter.
    "this-member": (
        "int",
        "src(src_), params(params_), ld(params_[0]), step(this->params[1] + 1)",
        "params[2]",
    ),
}


def _source(case):
    parameter, initializers, scale = CASES[case]
    return (
        READER.replace("PARAM", parameter)
        .replace("INITIALIZERS", initializers)
        .replace("SCALE", scale)
    )


def _translate(tmp_path, backend, case, source_options):
    (tmp_path / "reader.metal").write_text(_source(case), encoding="utf-8")
    config = ProjectConfig(
        root=tmp_path,
        source_roots=(".",),
        include_patterns=("reader.metal",),
        targets=(backend,),
        output_dir="out",
        entry_points={"reader.metal": ("run_int",)},
        workgroup_size_rules={"reader.metal": (4, 1, 1)},
        source_options={"metal": source_options},
    )
    return translate_project(config, format_output=False).to_json()


def _artifact(tmp_path, payload):
    assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
    assert not payload["diagnostics"]
    (artifact,) = payload["artifacts"]
    return (tmp_path / artifact["path"]).read_text(encoding="utf-8")


def test_unqualified_identifiers_skip_member_and_scope_access():
    text = "params->C / params.groups + other->params + Owner::params + params_[0]"
    assert unqualified_identifiers(text) == [
        "params",
        "params",
        "other",
        "Owner",
        "params_",
    ]
    assert qualified_identifiers(text) == ["C", "groups", "params", "params"]
    assert replace_unqualified_identifiers(text, {"params": "params_"}) == (
        "params_->C / params_.groups + other->params + Owner::params + params_[0]"
    )
    assert replace_unqualified_identifiers("a + b", {}) == "a + b"


@pytest.mark.parametrize("case", ["parameter-and-member", "struct-parameter"])
@pytest.mark.parametrize("backend", ["metal", "directx", "opengl"])
def test_promoted_constructor_keeps_parameters_its_initializers_read(
    tmp_path, backend, case
):
    generated = _artifact(tmp_path, _translate(tmp_path, backend, case, {}))
    factory = next(
        line for line in generated.splitlines() if "crosstl_ctor_Reader_int_1" in line
    )
    # The factory takes the constant buffer but not the promoted device one.
    assert "params_" in factory and "src_" not in factory
    assert "params_" in generated.split("crosstl_ctor_value.step", 1)[1].split(";")[0]
    _compile(generated, backend, tmp_path)


@pytest.mark.parametrize("case", ["derived-member", "this-member"])
@pytest.mark.parametrize("backend", ["metal", "directx"])
def test_unmapped_member_read_keeps_ordinary_constructor_lowering(
    tmp_path, backend, case
):
    payload = _translate(
        tmp_path,
        backend,
        case,
        {"promote_derived_pointer_members": True},
    )
    generated = _artifact(tmp_path, payload)
    factory = next(
        line for line in generated.splitlines() if "crosstl_ctor_Reader_int_1" in line
    )
    assert "src_" in factory and "params_" in factory
    _compile(generated, backend, tmp_path)


@pytest.mark.parametrize("backend", ["metal", "directx"])
def test_constant_pointer_member_is_assigned_in_ordinary_lowering(tmp_path, backend):
    payload = _translate(
        tmp_path, backend, "derived-member", {"preserve_resource_origins": True}
    )
    generated = _artifact(tmp_path, payload)
    assert "crosstl_ctor_value.params = " in generated
    _compile(generated, backend, tmp_path)
