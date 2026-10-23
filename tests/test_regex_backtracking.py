import subprocess
import sys
from pathlib import Path

import pytest

# Each check runs in a child process with a deadline, so a regular expression that backtracks
# exponentially fails the test instead of stalling the suite. Every input below takes longer than the
# deadline with the expression it replaced.
BACKTRACKING_CHECKS = {
    "rust-string": (
        "from crosstl.backend.Rust.RustCrossGLCodeGen import RUST_STRING_RE; "
        "literal = chr(34) + (chr(92) + chr(10)) * 64; "
        "assert RUST_STRING_RE.match(literal) is None"
    ),
    "metal-angle-operator": (
        "from crosstl.backend.Metal.preprocessor import METAL_ANGLE_OPERATOR_RE; "
        "text = 'operator' + ' /**/' * 64 + ' x'; "
        "assert METAL_ANGLE_OPERATOR_RE.match(text) is None"
    ),
    "hlsl-function-attributes": (
        "from crosstl.project.pipeline import DIRECTX_HLSL_FUNCTION_RE; "
        "text = '[a]' + ' ' * 5000 + '[b]' + ' ' * 5000 + ';'; "
        "assert DIRECTX_HLSL_FUNCTION_RE.search(text) is None"
    ),
    "hlsl-sampler-array-suffix": (
        "from crosstl.translator.codegen.directx_codegen import HLSLCodeGen; "
        "name = 'SamplerState' + '[ ]' * 64 + ' x'; "
        "assert HLSLCodeGen().is_hlsl_global_resource_type(name) is False"
    ),
}


def run_bounded(code):
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        timeout=30,
    )


def test_unterminated_rust_string_does_not_backtrack():
    run_bounded(BACKTRACKING_CHECKS["rust-string"])


@pytest.mark.parametrize(
    "check",
    [name for name in BACKTRACKING_CHECKS if name != "rust-string"],
)
def test_shader_source_expression_does_not_backtrack(check):
    run_bounded(BACKTRACKING_CHECKS[check])


def test_hlsl_sampler_array_suffixes_still_match():
    from crosstl.translator.codegen.directx_codegen import HLSLCodeGen

    generator = HLSLCodeGen()
    assert generator.is_hlsl_global_resource_type("SamplerState")
    assert generator.is_hlsl_global_resource_type("SamplerState[4]")
    assert generator.is_hlsl_global_resource_type("SamplerComparisonState [ N ][2]")
    assert generator.is_hlsl_global_resource_type("SamplerState[]")
    assert not generator.is_hlsl_global_resource_type("SamplerState[a[1]]")
    assert not generator.is_hlsl_global_resource_type("SamplerStateX[4]")
