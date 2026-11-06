"""A generic's parameter that names no type parameter keeps its type node in every specialization.

Specializing ``generic<T> void widen(const device float16* src, inout thread T out)`` rewrote each parameter
type through its spelling. ``src`` names no type parameter, but its pointer node became the string
``const device float16*``, and the Metal generator then declared it as a reference to a pointer
(``const device half *&src``), which an rvalue argument such as ``src + gid`` cannot bind to.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator.ast import PointerType, PrimitiveType
from crosstl.translator.codegen.generic_vector_deduction import (
    substituted_generic_type,
)

PROGRAM = textwrap.dedent("""
    shader main {
        compute {
            @ stage_entry
            void k(StructuredBuffer<float16> src @buffer(0),
                   RWStructuredBuffer<float> dst @buffer(1),
                   uint gid @gl_GlobalInvocationID) {
                float value;
                widen(src + gid, value);
                buffer_store(dst, gid, value);
            }
        }
        generic<T> void widen(const device float16* src, inout thread T out_) {
            out_ = T(*src);
        }
    }
    """)


class _Spelling:
    @staticmethod
    def type_name_string(type_value):
        if isinstance(type_value, PointerType):
            return f"{type_value.pointee_type.name}*"
        return getattr(type_value, "name", type_value)


def test_unchanged_pointer_keeps_its_node():
    pointer = PointerType(PrimitiveType("float16"))
    assert substituted_generic_type(_Spelling(), pointer, {"T": "float"}) is pointer


def test_changed_pointer_is_substituted():
    pointer = PointerType(PrimitiveType("T"))
    concrete = substituted_generic_type(_Spelling(), pointer, {"T": "float"})
    assert isinstance(concrete, PointerType)
    assert concrete.pointee_type.name == "float"


def test_metal_specialization_keeps_the_pointer_parameter(tmp_path):
    source = tmp_path / "widen.cgl"
    source.write_text(PROGRAM, encoding="utf-8")
    # Unformatted, so the spelling does not depend on whether clang-format is installed.
    code = crosstl.translate(str(source), backend="metal", format_output=False)
    assert "widen_float(const device half* src, thread float& out_)" in code
    assert "*&" not in code
    if shutil.which("xcrun") is None:
        pytest.skip("xcrun not installed")
    (tmp_path / "widen.metal").write_text(code, encoding="utf-8")
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", "widen.metal", "-o", "w.air"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
