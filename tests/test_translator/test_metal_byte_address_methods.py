"""CrossGL programs that call the HLSL methods of a byte-address buffer (#3646).

Metal and GLSL output bind a ``RWByteAddressBuffer`` as ``uint`` words
(``crosstl/translator/semantics/byte_address_buffers.py``). A method call such as ``obj.Load(0u)`` is the CrossGL call
the HLSL importer writes for it, ``buffer_load(obj, 0u)``, so each target writes the same code for both spellings.
Metal output used to copy the methods onto the ``device uint*`` it declares, which the Metal compiler rejects.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl

_PROGRAM = """
shader P {{
    RWByteAddressBuffer obj @register(u0);
    RWStructuredBuffer<int> iin @register(u1);
    compute {{
        @numthreads(1, 1, 1)
        void main(uvec3 tid @gl_GlobalInvocationID) {{
            uint o1;
            uint o2;
            uint n;
            {body}
        }}
    }}
}}
"""

# (method spelling, the CrossGL call the HLSL importer writes for it)
_FORMS = [
    (
        "uint x = obj.Load(0u); obj.Store(4u, x + 1u);",
        "uint x = buffer_load(obj, 0u); buffer_store(obj, 4u, x + 1u);",
    ),
    (
        "uvec2 y = obj.Load2(8u); obj.Store2(16u, y);",
        "uvec2 y = buffer_load2(obj, 8u); buffer_store2(obj, 16u, y);",
    ),
    (
        "uvec3 y = obj.Load3(8u); obj.Store3(16u, y);",
        "uvec3 y = buffer_load3(obj, 8u); buffer_store3(obj, 16u, y);",
    ),
    (
        "uvec4 y = obj.Load4(8u); obj.Store4(32u, y);",
        "uvec4 y = buffer_load4(obj, 8u); buffer_store4(obj, 32u, y);",
    ),
    ("obj.InterlockedAdd(8u, 1u);", "atomicAdd(obj, 8u, 1u);"),
    (
        "obj.InterlockedAdd(8u, 1u, o1); obj.Store(0u, o1);",
        "o1 = atomicAdd(obj, 8u, 1u); buffer_store(obj, 0u, o1);",
    ),
    ("obj.InterlockedOr(8u, 6u, o1);", "o1 = atomicOr(obj, 8u, 6u);"),
    ("obj.InterlockedMax(8u, iin[tid.x]);", "atomicMax(obj, 8u, iin[tid.x]);"),
    ("obj.InterlockedMin(8u, 3u, o1);", "o1 = atomicMin(obj, 8u, 3u);"),
    ("obj.InterlockedExchange(8u, 3u, o2);", "o2 = atomicExchange(obj, 8u, 3u);"),
    (
        "obj.InterlockedCompareExchange(8u, 1u, 2u, o1);",
        "o1 = atomicCompareExchange(obj, 8u, 1u, 2u);",
    ),
    (
        "obj.InterlockedCompareStore(8u, 1u, 2u);",
        "atomicCompareExchange(obj, 8u, 1u, 2u);",
    ),
    (
        "obj.GetDimensions(n); obj.Store(0u, n);",
        "buffer_dimensions(obj, n); buffer_store(obj, 0u, n);",
    ),
    (
        "obj.Store(0u, obj.Load(4u) + obj.Load(8u));",
        "buffer_store(obj, 0u, buffer_load(obj, 4u) + buffer_load(obj, 8u));",
    ),
]
_IDS = [
    "load-store",
    "load2",
    "load3",
    "load4",
    "add",
    "add-original",
    "or-original",
    "signed-max",
    "min-original",
    "exchange",
    "compare-exchange",
    "compare-store",
    "dimensions",
    "nested-loads",
]


def _generate(backend: str, body: str, tmp_path) -> str:
    path = tmp_path / "program.cgl"
    path.write_text(_PROGRAM.format(body=body), encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("backend", ["metal", "opengl"])
@pytest.mark.parametrize("methods, calls", _FORMS, ids=_IDS)
def test_byte_address_methods_translate_like_the_crossgl_calls(
    backend, methods, calls, tmp_path
):
    written = _generate(backend, methods, tmp_path)
    assert written == _generate(backend, calls, tmp_path)
    assert ".Load" not in written and ".Store" not in written
    assert ".Interlocked" not in written and ".GetDimensions" not in written


_XCRUN = shutil.which("xcrun") is not None and (
    subprocess.run(
        ["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True, check=False
    ).returncode
    == 0
)


@pytest.mark.skipif(not _XCRUN, reason="the Metal compiler is not installed")
def test_every_method_form_compiles_with_the_metal_compiler(tmp_path):
    body = " ".join("{ " + methods + " }" for methods, _ in _FORMS)
    (tmp_path / "methods.metal").write_text(_generate("metal", body, tmp_path))
    compiled = subprocess.run(
        [
            *("xcrun", "-sdk", "macosx", "metal", "-c", "methods.metal"),
            *("-o", "methods.air"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr


@pytest.mark.parametrize("backend", ["metal", "opengl"])
@pytest.mark.parametrize(
    "body, reason",
    [
        (
            "uint x = obj.Load<uint>(0u);",
            r"method 'Load' has no lowering|generic member call 'obj\.Load<uint>'",
        ),
        ("uint x = obj.Sample(0u);", "method 'Sample' has no lowering"),
        ("obj.InterlockedExchange(8u, 3u);", "without its original-value output"),
        (
            "obj.InterlockedCompareExchange(8u, 1u, 2u);",
            "without its original-value output",
        ),
        ("uint s; uint x = obj.Load(0u, s);", "takes a buffer and a byte address"),
        (
            "o1 = obj.Load(0u) + obj.Store(4u, 1u);",
            "buffer_store on a byte-address buffer inside an expression",
        ),
    ],
    ids=[
        "generic-load",
        "unknown",
        "exchange",
        "compare-exchange",
        "status",
        "store-in-expression",
    ],
)
def test_byte_address_methods_without_a_lowering_are_refused(
    backend, body, reason, tmp_path
):
    with pytest.raises(Exception, match=reason):
        _generate(backend, body, tmp_path)
