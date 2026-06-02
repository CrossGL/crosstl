import pytest

from crosstl.backend.CUDA.CudaLexer import CudaLexer
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.HIP.HipLexer import HipLexer
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Mojo.MojoLexer import MojoLexer
from crosstl.backend.Rust.RustLexer import RustLexer
from crosstl.backend.slang.SlangLexer import SlangLexer
from crosstl.backend.SPIRV.VulkanLexer import VulkanLexer
from crosstl.translator.lexer import Lexer as CrossGLLexer

IGNORED_TOKEN_TYPES = {"EOF", "NEWLINE", "NL", "INDENT", "DEDENT"}


def _tokenize(lexer_cls, source):
    lexer = lexer_cls("\ufeff" + source)
    if hasattr(lexer, "get_tokens"):
        return lexer.get_tokens()
    return lexer.tokenize()


def _first_token_type(tokens):
    for token in tokens:
        token_type = token[0] if isinstance(token, tuple) else token.type
        if token_type not in IGNORED_TOKEN_TYPES:
            return token_type
    raise AssertionError("expected at least one non-ignored token")


@pytest.mark.parametrize(
    ("lexer_cls", "source", "expected_type"),
    [
        (CrossGLLexer, "shader Bom { compute { } }", "SHADER"),
        (HLSLLexer, "float4 main() : SV_Target { return float4(1.0); }", "FVECTOR"),
        (GLSLLexer, "#version 450\nvoid main() {}", "HASH"),
        (MetalLexer, "kernel void main0() {}", "KERNEL"),
        (SlangLexer, "float4 main() : SV_Target { return float4(1.0); }", "FVECTOR"),
        (VulkanLexer, "#version 450\nvoid main() {}", "VOID"),
        (CudaLexer, "__global__ void kernel0() {}", "GLOBAL"),
        (HipLexer, "__global__ void kernel0() {}", "__GLOBAL__"),
        (RustLexer, "fn main() {}", "FN"),
        (MojoLexer, "fn main():\n    return\n", "FN"),
    ],
)
def test_native_lexers_ignore_leading_utf8_bom(lexer_cls, source, expected_type):
    tokens = _tokenize(lexer_cls, source)

    assert _first_token_type(tokens) == expected_type
