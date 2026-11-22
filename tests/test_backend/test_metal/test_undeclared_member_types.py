"""Metal input of ``Owner::member`` types whose owner is a concrete struct (#4651).

A qualified type names a member of the struct its nested-name-specifier denotes ([class.qual]), including a member
of a base class ([class.member.lookup]) and of a nested struct. The importer passed a member that ``Owner`` does not
declare, ``Plain::missing_t<float>``, to every target unchanged, and wrote ``Plain::missing`` as ``missing``; it now
refuses both with a typed diagnostic. It also wrote an inherited alias (``Derived::scalar``) or the alias of a nested
struct (``Plain::Inner::t``) without its owner, which names nothing in CrossGL; those now resolve to the type the
alias names.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.scoped_member_types import MetalUndeclaredMemberTypeError

_PLAIN = """
#include <metal_stdlib>
using namespace metal;
struct Plain {
  int x;
};
kernel void k(device float* r [[buffer(0)]]) {
  %s
  r[0] = v[0];
}
"""

_UNDECLARED = {
    "template-member": (
        "typename Plain::template missing_t<float> v = float4(1.0f);",
        "Plain::missing_t",
    ),
    "member": ("Plain::missing v = float4(1.0f);", "Plain::missing"),
    "pointer-member": (
        "thread Plain::missing* p = nullptr; float4 v = float4(1.0f);",
        "Plain::missing",
    ),
    "alias-owner": (
        "using C = Plain; typename C::template missing_t<float> v = float4(1.0f);",
        "C::missing_t",
    ),
}


@pytest.mark.parametrize("backend", ["cgl", "metal", "directx", "opengl", "vulkan"])
@pytest.mark.parametrize("case", sorted(_UNDECLARED))
def test_undeclared_member_type_is_refused_on_every_target(tmp_path, case, backend):
    declaration, spelling = _UNDECLARED[case]
    path = tmp_path / "undeclared_member_type.metal"
    path.write_text(_PLAIN % declaration, encoding="utf-8")
    with pytest.raises(MetalUndeclaredMemberTypeError) as raised:
        crosstl.translate(str(path), backend=backend)
    error = raised.value
    assert error.spelling == spelling
    assert error.struct_name == "Plain"
    assert error.function_name == "k"
    assert (
        error.project_diagnostic_code
        == "project.translate.metal-undeclared-member-type"
    )
    assert f"'{spelling}'" in str(error) and "struct 'Plain'" in str(error)


@pytest.mark.parametrize("backend", ["cgl", "metal", "directx", "opengl", "vulkan"])
@pytest.mark.parametrize(
    "member", ["template missing_t<float>", "missing", "template vec_t<float>"]
)
def test_member_type_of_an_out_of_scope_alias_is_refused(tmp_path, member, backend):
    # ``C`` is declared in a block that has ended, so ``C::...`` names nothing at the use ([basic.scope.block]),
    # whether or not ``Plain`` declares the member.
    source = _PLAIN.replace(
        "int x;", "int x;\n  template <typename U> using vec_t = vec<U, 4>;"
    )
    declaration = "{ using C = Plain; }\n  typename C::%s v = float4(1.0f);" % member
    path = tmp_path / "out_of_scope_alias.metal"
    path.write_text(source % declaration, encoding="utf-8")
    with pytest.raises(MetalUndeclaredMemberTypeError) as raised:
        crosstl.translate(str(path), backend=backend)
    error = raised.value
    assert error.struct_name is None and error.owner == "C"
    assert error.function_name == "k"
    assert "alias declaration of 'C' is out of scope" in str(error)


def test_alias_in_scope_in_an_enclosing_block_still_resolves(tmp_path):
    source = _PLAIN.replace("int x;", "int x;\n  using scalar = float;")
    declaration = (
        "using C = Plain;\n  float4 v = float4(1.0f);\n"
        "  { typename C::scalar s = 2.0f; v.x = s; }"
    )
    path = tmp_path / "enclosing_alias.metal"
    path.write_text(source % declaration, encoding="utf-8")
    crossgl = " ".join(crosstl.translate(str(path), backend="cgl").split())
    assert "float s = 2.0f;" in crossgl, crossgl


_DECLARED = """
#include <metal_stdlib>
using namespace metal;
struct Plain {
  int x;
  using scalar = float;
  template <typename U> using vec_t = vec<U, 4>;
  struct Inner {
    int y;
    using t = int;
  };
};
struct Derived : Plain {
  float z;
};
kernel void k(device float* r [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  Plain::scalar s = 1.5f;
  Derived::scalar d = 2.0f;
  Plain::Inner::t q = 3;
  Plain::Inner inner;
  inner.y = 4;
  typename Plain::template vec_t<float> v = float4(5.0f);
  r[i] = s + d + float(q) + float(inner.y) + v[1];
}
"""


def test_declared_member_types_resolve_through_bases_and_nested_structs(tmp_path):
    path = tmp_path / "declared_member_types.metal"
    path.write_text(_DECLARED, encoding="utf-8")
    crossgl = " ".join(crosstl.translate(str(path), backend="cgl").split())
    for expected in (
        "float s = 1.5f;",
        "float d = 2.0f;",
        "int q = 3;",
        "vec4 v = vec4(5.0f);",
    ):
        assert expected in crossgl, crossgl
    assert "scalar d" not in crossgl and " t q" not in crossgl

    metal = crosstl.translate(str(path), backend="metal")
    if shutil.which("xcrun") is None:
        pytest.skip("xcrun is not installed")
    (tmp_path / "out.metal").write_text(metal, encoding="utf-8")
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", "out.metal", "-o", "out.air"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("the Metal toolchain is not installed")
    assert result.returncode == 0, result.stderr[-2000:]
