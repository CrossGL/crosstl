"""Metal ``#import`` includes a header once, and directives the preprocessor cannot interpret are refused.

Apple's samples ``#import`` headers shared with Objective-C. The preprocessor used to copy the directive into the
translated program, where DXC and glslang reject it, and the header's declarations were missing.
"""

from __future__ import annotations

import pytest

import crosstl
from crosstl.backend.Metal.directives import MetalPreprocessorDirectiveError
from crosstl.backend.Metal.preprocessor import MetalPreprocessor

HEADER = """#pragma once
struct FragData { half4 color [[color(0)]]; };
"""

PLAIN_HEADER = "struct Shared { float value; };\n"


def preprocess(tmp_path, source, header=HEADER, name="Common.h"):
    (tmp_path / name).write_text(header)
    path = tmp_path / "shader.metal"
    path.write_text(source)
    return MetalPreprocessor().preprocess(source, file_path=str(path))


@pytest.mark.parametrize(
    "directives",
    [
        '#import "Shared.h"\n#import "Shared.h"\n',
        '#include "Shared.h"\n#import "Shared.h"\n',
        '#import "Shared.h"\n#include "Shared.h"\n',
    ],
)
def test_import_enters_a_header_at_most_once(tmp_path, directives):
    processed = preprocess(tmp_path, directives, PLAIN_HEADER, "Shared.h")

    assert processed.count("struct Shared") == 1
    assert "#import" not in processed


def test_repeated_include_without_import_still_includes_twice(tmp_path):
    processed = preprocess(
        tmp_path, '#include "Shared.h"\n#include "Shared.h"\n', PLAIN_HEADER, "Shared.h"
    )

    assert processed.count("struct Shared") == 2


def test_import_respects_inactive_branches(tmp_path):
    processed = preprocess(tmp_path, '#if 0\n#import "Common.h"\n#endif\n')

    assert "FragData" not in processed


@pytest.mark.parametrize("directive", ['#ident "v1"', "#include_next <metal_stdlib>"])
def test_unknown_active_directives_are_refused(tmp_path, directive):
    with pytest.raises(MetalPreprocessorDirectiveError) as raised:
        preprocess(tmp_path, f"{directive}\nkernel void k() {{}}\n")

    assert (
        raised.value.project_diagnostic_code
        == "project.translate.metal-preprocessor-directive-unsupported"
    )


def test_unknown_directives_in_inactive_branches_and_pragmas_are_kept_out_of_the_way(
    tmp_path,
):
    processed = preprocess(
        tmp_path,
        '#if 0\n#ident "v1"\n#endif\n#pragma clang diagnostic ignored "-Wunused"\n',
    )

    assert "#ident" not in processed
    assert "#pragma clang diagnostic" in processed


def test_imported_header_declarations_reach_every_target(tmp_path):
    (tmp_path / "Common.h").write_text(HEADER)
    path = tmp_path / "shader.metal"
    path.write_text("""#include <metal_stdlib>
using namespace metal;
#import "Common.h"
struct VOut { float4 position [[position]]; };
fragment FragData shade(VOut in [[stage_in]]) {
    FragData out;
    out.color = half4(1.0h);
    return out;
}
""")

    crossgl = crosstl.translate(str(path), backend="cgl")
    hlsl = crosstl.translate(str(path), backend="directx")

    assert "#import" not in crossgl
    assert "struct FragData" in crossgl
    # A half-precision color attachment stays a 16-bit render target value.
    assert "float16_t4 color : SV_TARGET" in hlsl
    assert "#import" not in hlsl
