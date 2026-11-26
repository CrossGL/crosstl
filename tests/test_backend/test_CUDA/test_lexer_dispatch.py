"""The CUDA and HIP lexers try only the token patterns that can begin at a character.

Each lexer keeps the first pattern of its ordered table that matches, so first-character dispatch must pick the
same pattern and the same end offset as trying the whole table, at every position of every CUDA and HIP source
in the repository and of random texts built from the tables' own spellings.
"""

import importlib
import random
import re
import subprocess
from pathlib import Path

import pytest

from crosstl.translator.token_dispatch import ASCII_LIMIT, ordered_pattern_dispatch

REPO = Path(__file__).resolve().parents[3]
LEXERS = {
    "cuda": importlib.import_module("crosstl.backend.CUDA.CudaLexer"),
    "hip": importlib.import_module("crosstl.backend.HIP.HipLexer"),
}

KERNEL = r"""
#include <cuda_runtime.h>
// line comment
/* block
   comment */
texture<float4, 2, cudaReadModeElementType> tref;
template <typename T> __device__ __forceinline__ T twice(T v) { return v + v; }
__global__ void __launch_bounds__(256) k(const float* __restrict__ a, float4* o, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !(a[i] != 0.5e-3f)) return;
    o[i] = make_float4(a[i], 1.0, 0x1Fu, 017ull) * twice(2.0f) - tex2D(tref, .5f, 1.f);
    o[i].x += a[i] << 2 >> 1 & ~3 | 4 ^ 5 % 6;
    const char* s = "str\"ing"; char c = '\n'; auto r = R"x(raw)x";
    i++; --i; i <<= 1; i >>= 1; i &= 1; i |= 2; i ^= 3; i %= 4; i ->* 0;
    __syncthreads(); asm volatile("bar.sync 0;" ::: "memory");
}
"""


def _sources():
    listed = subprocess.run(
        ["git", "ls-files", "*.cu", "*.cuh", "*.hip"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    texts = [KERNEL]
    texts += [
        (REPO / name).read_text(encoding="utf-8", errors="replace") for name in listed
    ]
    return texts


def _fuzz(table, count=40, length=400):
    # Pieces of the table's own spellings, every ASCII character, and a few
    # non-ASCII characters that exercise the full-table fallback.
    pieces = [re.sub(r"\\[bB]|\\(.)", r"\1", pattern)[:12] for _name, pattern in table]
    pieces += [chr(code) for code in range(ASCII_LIMIT)]
    pieces += ["\u00a0", "\u2028", "\u00e9", "\u03bb", "\u0663"]
    generator = random.Random(4231)
    return [
        "".join(generator.choice(pieces) for _ in range(length // 4))
        for _ in range(count)
    ]


def _first_match(compiled, text, pos):
    for name, pattern in compiled:
        match = pattern.match(text, pos)
        if match:
            return name, match.end()
    return None


@pytest.fixture(scope="module")
def sources():
    return _sources()


@pytest.mark.parametrize("language", sorted(LEXERS))
def test_dispatch_picks_the_whole_tables_first_match(language, sources):
    table = list(LEXERS[language].TOKENS)
    compiled = [(name, re.compile(pattern)) for name, pattern in table]
    dispatch = ordered_pattern_dispatch(table)
    for text in sources + _fuzz(table):
        for pos in range(len(text)):
            got = dispatch.match(text, pos)
            assert (got and (got[0], got[1].end())) == _first_match(
                compiled, text, pos
            ), (language, text[pos : pos + 30])


def _reference_tokens(module, lexer_class, code):
    """Tokens of ``code`` with the whole table tried at each position."""

    lexer = lexer_class(code, preprocess=False)
    compiled = [(name, re.compile(pattern)) for name, pattern in module.TOKENS]

    def whole_table(pos):
        found = _first_match(compiled, lexer.code, pos)
        return found and (found[1], found[0], lexer.code[pos : found[1]])

    lexer._next_token = whole_table
    return _spelled(lexer.tokenize())


def _spelled(tokens):
    # CUDA tokens are (type, text) pairs; HIP tokens also carry line and column.
    return [
        token if isinstance(token, tuple) else vars(token).copy() for token in tokens
    ]


@pytest.mark.parametrize("language", sorted(LEXERS))
def test_token_streams_are_unchanged(language, sources):
    module = LEXERS[language]
    lexer_class = module.CudaLexer if language == "cuda" else module.HipLexer
    for text in sources:
        tokens = lexer_class(text, preprocess=False).tokenize()
        assert tokens, "expected a token stream"
        assert _spelled(tokens) == _reference_tokens(module, lexer_class, text)
