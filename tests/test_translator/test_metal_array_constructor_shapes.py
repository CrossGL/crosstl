"""Array constructors with several dimensions or a size that names a constant (#3679, #3553).

Metal, HLSL and the SPIR-V generator build an array only in an aggregate initializer, so
``crosstl/translator/semantics/array_values.py`` rewrites each GLSL array constructor into an initializer list. It
recognised only ``element[literal](...)``: ``float[2][2](float[2](...), ...)``, which the parser reads as the type
``float`` indexed twice, and ``float[N](...)`` reached Metal and HLSL output verbatim, which neither compiler accepts,
and the SPIR-V generator refused them as calls through a callee expression.
"""

from __future__ import annotations

import re
import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.semantics.array_values import (
    array_constructor_dimensions,
    array_constructor_shape,
)

_PROGRAM = """
shader P {{
    {scope}
    buffer float* outp;
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            {body}
        }}
    }}
}}
"""

_NESTED = "float[2][2](float[2](1.0, 2.0), float[2](3.0, 4.0))"

# (shader-scope declarations, entry body, the Metal and HLSL text each must contain)
_CASES = {
    "nested-declaration": (
        "",
        f"float L[2][2] = {_NESTED}; outp[i] = L[1][0];",
        "float L[2][2] = {{1.0, 2.0}, {3.0, 4.0}};",
    ),
    "nested-assignment": (
        "",
        f"float L[2][2]; L = {_NESTED}; outp[i] = L[1][0];",
        "cgl_array_value_0[2][2] = {{1.0, 2.0}, {3.0, 4.0}};",
    ),
    "two-by-three": (
        "",
        "float L[2][3] = float[2][3](float[3](1.0, 2.0, 3.0), float[3](4.0, 5.0, 6.0)); outp[i] = L[1][2];",
        "float L[2][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};",
    ),
    "named-size-declaration": (
        "const int N = 3;",
        "float a[N] = float[N](1.0, 2.0, 3.0); outp[i] = a[N - 1];",
        "a[N] = {1.0, 2.0, 3.0};",
    ),
    "named-size-argument": (
        "const int N = 2; float total(float v[2]) { return v[0] + v[1]; }",
        "outp[i] = total(float[N](1.0, 2.0));",
        "cgl_array_value_0[2] = {1.0, 2.0};",
    ),
    # A short constructor zero-fills the rest of its full-size local, as a declaration does (#3639).
    "short-assignment": (
        "",
        "float a[3]; a = float[3](1.0); outp[i] = a[0] + a[2];",
        "cgl_array_value_0[3] = {1.0",
    ),
    "short-nested-assignment": (
        "",
        "float b[2][2]; b = float[2][2](float[2](4.0, 5.0)); outp[i] = b[0][1] + b[1][1];",
        "cgl_array_value_0[2][2] = {{4.0, 5.0}",
    ),
    "short-named-size-argument": (
        "const int N = 3; float total(float v[3]) { return v[0] + v[1] + v[2]; }",
        "outp[i] = total(float[N](2.0));",
        "cgl_array_value_0[3] = {2.0",
    ),
    "named-inner-size": (
        "const int N = 2;",
        "float L[2][N] = float[2][N](float[N](1.0, 2.0), float[N](3.0, 4.0)); outp[i] = L[1][0];",
        "{{1.0, 2.0}, {3.0, 4.0}};",
    ),
}


def _translate(tmp_path, backend: str, scope: str, body: str) -> str:
    path = tmp_path / "program.cgl"
    path.write_text(_PROGRAM.format(scope=scope, body=body))
    return crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize("backend", ["metal", "directx"])
@pytest.mark.parametrize("case", sorted(_CASES))
def test_array_constructors_become_initializer_lists(tmp_path, backend, case):
    scope, body, expected = _CASES[case]
    written = _translate(tmp_path, backend, scope, body)
    assert expected in written
    assert "float[2](" not in written and "float[N](" not in written


def test_a_renamed_size_constant_keeps_its_declaration_and_list(tmp_path):
    # Metal renames a constant named like a Metal function (``using namespace metal`` declares ``level``); the
    # constructor's own spelling is gone, so it cannot keep the old name.
    written = _translate(
        tmp_path,
        "metal",
        "const int level = 2;",
        "float w[level] = float[level](1.0, 2.0); outp[i] = w[1];",
    )
    assert "float w[level_] = {1.0, 2.0};" in written


def _indexed(base: str, *sizes) -> A.ASTNode:
    callee = A.IdentifierNode(base)
    for size in sizes:
        index = (
            A.LiteralNode(size, A.PrimitiveType("int"))
            if isinstance(size, int)
            else A.IdentifierNode(size)
        )
        callee = A.ArrayAccessNode(callee, index)
    return callee


@pytest.mark.parametrize(
    "callee, arguments, shape",
    [
        (A.IdentifierNode("float[2]"), 2, ("float", 2)),
        (A.IdentifierNode("float[N]"), 3, ("float", "N")),
        (_indexed("float", 2, 3), 2, ("float[3]", 2)),
        (_indexed("vec3", "N", 4), 2, ("vec3[4]", "N")),
        (_indexed("float[2]", 3), 3, ("float[3]", 2)),
        # A call through an indexed value, or a type index that is not a size, is not an array constructor.
        (_indexed("table", 0), 1, None),
        (_indexed("table", 0, 1), 1, None),
        (
            A.ArrayAccessNode(
                A.IdentifierNode("float"),
                A.BinaryOpNode(A.IdentifierNode("n"), "+", A.IdentifierNode("m")),
            ),
            1,
            None,
        ),
    ],
)
def test_array_constructor_shapes(callee, arguments, shape):
    call = A.FunctionCallNode(
        callee, [A.LiteralNode(1.0, A.PrimitiveType("float"))] * arguments
    )
    assert array_constructor_dimensions(call) == shape
    # A generator spells a one-dimensional, literal-sized shape back as ``element[size]``; the others have no
    # such spelling, so they are not one.
    one_dimensional = (
        shape is not None and isinstance(shape[1], int) and "[" not in shape[0]
    )
    assert array_constructor_shape(call) == (shape if one_dimensional else None)


@pytest.mark.parametrize("backend", ["metal", "directx"])
def test_a_constructor_sized_by_a_computed_constant_is_refused(tmp_path, backend):
    # The local the constructor is built in needs the number of elements, which ``2 + 1`` does not give here.
    with pytest.raises(
        Exception, match="array-constructor-size|not an integer literal constant"
    ):
        _translate(
            tmp_path,
            backend,
            "const int M = 2 + 1; float total(float v[3]) { return v[0] + v[2]; }",
            "outp[i] = total(float[M](1.0, 2.0, 3.0));",
        )


@pytest.mark.parametrize("backend", ["metal", "directx", "vulkan"])
def test_a_constructor_with_more_elements_than_its_size_is_refused(tmp_path, backend):
    with pytest.raises(Exception, match="has 3 elements, more than its 2"):
        _translate(
            tmp_path,
            backend,
            "const int N = 2; float total(float v[2]) { return v[0] + v[1]; }",
            "outp[i] = total(float[N](1.0, 2.0, 3.0));",
        )


_SHORT_PROGRAM = (
    "const int N = 3; float total(float v[3]) { return v[0] + v[1] + v[2]; }",
    "float a[3]; a = float[3](1.0); float b[2][2]; b = float[2][2](float[2](4.0, 5.0)); "
    "outp[i] = a[0] + a[2] + total(float[N](2.0)) + b[0][1] + b[1][1] + float(i);",
)


def test_short_constructors_compute_the_zero_filled_values_on_the_gpu(tmp_path):
    # a = {1, 0, 0}, total({2, 0, 0}) = 2 and b = {{4, 5}, {0, 0}}: 1 + 0 + 2 + 5 + 0 = 8 plus the lane.
    mtl = pytest.importorskip("Metal")
    device = mtl.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    metal = _translate(tmp_path, "metal", *_SHORT_PROGRAM)
    library, error = device.newLibraryWithSource_options_error_(metal, None, None)
    assert library is not None, error
    entry = re.search(r"\bkernel\s+void\s+(\w+)\s*\(", metal).group(1)
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_(entry), None
    )
    assert pipeline is not None, error
    index = int(re.search(r"\*\s*outp\s*\[\[buffer\((\d+)\)\]\]", metal).group(1))
    count = 4
    output = device.newBufferWithLength_options_(4 * count, 0)
    commands = device.newCommandQueue().commandBuffer()
    encoder = commands.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(output, 0, index)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        mtl.MTLSizeMake(count, 1, 1), mtl.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    values = struct.unpack(f"<{count}f", output.contents().as_buffer(4 * count))
    assert values == tuple(8.0 + lane for lane in range(count)), values


_SPIRV_TOOLS = all(shutil.which(tool) for tool in ("spirv-as", "spirv-val"))


@pytest.mark.skipif(not _SPIRV_TOOLS, reason="spirv-as and spirv-val are not installed")
@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_spirv_output_validates(tmp_path, case):
    scope, body, _ = _CASES[case]
    (tmp_path / "program.spvasm").write_text(
        _translate(tmp_path, "vulkan", scope, body)
    )
    for command in (
        ["spirv-as", "--target-env", "vulkan1.0", "program.spvasm", "-o", "p.spv"],
        ["spirv-val", "--target-env", "vulkan1.0", "p.spv"],
    ):
        result = subprocess.run(
            command, cwd=tmp_path, capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr


_XCRUN = shutil.which("xcrun") is not None and (
    subprocess.run(
        ["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True, check=False
    ).returncode
    == 0
)


@pytest.mark.skipif(not _XCRUN, reason="the Metal compiler is not installed")
@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_metal_output_compiles(tmp_path, case):
    scope, body, _ = _CASES[case]
    (tmp_path / "program.metal").write_text(_translate(tmp_path, "metal", scope, body))
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", "program.metal", "-o", "p.air"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
