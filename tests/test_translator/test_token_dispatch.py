"""First-character token dispatch must pick the same token as the full table."""

import importlib
import random
import re
import subprocess
from pathlib import Path

import pytest

from crosstl.translator.token_dispatch import (
    ASCII_LIMIT,
    alternation_dispatch,
    ordered_pattern_dispatch,
    pattern_first_chars,
)

REPO = Path(__file__).resolve().parents[2]
ALL_ASCII = frozenset(range(ASCII_LIMIT))

LEXER_MODULES = (
    "crosstl.translator.lexer",
    "crosstl.backend.Metal.MetalLexer",
    "crosstl.backend.DirectX.DirectxLexer",
    "crosstl.backend.GLSL.OpenglLexer",
)
SHADER_SUFFIXES = (".cgl", ".metal", ".hlsl", ".glsl", ".vert", ".frag", ".comp")


def _token_table(module_name):
    tokens = importlib.import_module(module_name).TOKENS
    return list(tokens.items() if hasattr(tokens, "items") else tokens)


def _shader_texts():
    listed = subprocess.run(
        ["git", "ls-files", *(f"*{suffix}" for suffix in SHADER_SUFFIXES)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    paths = [REPO / name for name in listed] or sorted(
        path
        for suffix in SHADER_SUFFIXES
        for path in (REPO / "examples").rglob(f"*{suffix}")
    )
    return [path.read_text(encoding="utf-8", errors="replace") for path in paths]


def _fuzz_texts(table, count=40, length=400):
    # Fragments of the table's own literal spellings, plus characters that
    # stress the analysis: ASCII controls Unicode treats as whitespace,
    # non-ASCII whitespace, letters and digits, and every printable ASCII.
    fragments = [
        re.sub(r"\\[bB]|\\(.)", r"\1", pattern)[:12] for _name, pattern in table
    ]
    fragments += [chr(code) for code in range(ASCII_LIMIT)]
    fragments += ["\u00a0", "\u2028", "\u00e9", "\u03bb", "\u0663", "\x1c", "\x1f"]
    generator = random.Random(1729)
    return [
        "".join(generator.choice(fragments) for _ in range(length // 4))
        for _ in range(count)
    ]


def _reference_first_match(compiled_table, text, pos):
    for name, pattern in compiled_table:
        match = pattern.match(text, pos)
        if match:
            return name, match.end()
    return None


@pytest.fixture(scope="module")
def shader_texts():
    texts = _shader_texts()
    assert texts, "expected checked-in shader sources to compare against"
    return texts


@pytest.mark.parametrize("module_name", LEXER_MODULES)
def test_dispatch_matches_full_table_at_every_position(module_name, shader_texts):
    table = _token_table(module_name)
    compiled = [(name, re.compile(pattern)) for name, pattern in table]
    ordered = ordered_pattern_dispatch(table)
    # Only the CrossGL lexer matches through one named-group alternation; the
    # other tables repeat token names, which an alternation cannot hold.
    alternation = (
        alternation_dispatch(table)
        if module_name == "crosstl.translator.lexer"
        else None
    )
    for text in shader_texts + _fuzz_texts(table):
        for pos in range(len(text)):
            expected = _reference_first_match(compiled, text, pos)
            got = ordered.match(text, pos)
            assert (got and (got[0], got[1].end())) == expected, (
                module_name,
                text[pos : pos + 30],
            )
            if alternation is not None:
                match = alternation.match(text, pos)
                assert (match and (match.lastgroup, match.end())) == expected


@pytest.mark.parametrize("module_name", LEXER_MODULES)
def test_first_chars_cover_every_observed_match_start(module_name, shader_texts):
    for name, pattern in _token_table(module_name):
        first = pattern_first_chars(pattern)
        compiled = re.compile(pattern)
        for text in shader_texts[:40]:
            for match in compiled.finditer(text):
                start = text[match.start()] if match.end() > match.start() else ""
                if start and ord(start) < ASCII_LIMIT:
                    assert ord(start) in first, (name, pattern, start)


@pytest.mark.parametrize(
    "pattern, expected",
    [
        (r"\bshader\b", {ord("s")}),
        (r"[a-cX]+", {ord("a"), ord("b"), ord("c"), ord("X")}),
        (r"(?:ab|cd)e", {ord("a"), ord("c")}),
        (r"x?y", {ord("x"), ord("y")}),
        (r"(?=q)qz", {ord("q")}),
        (r"\d+[uU]?", set(range(ord("0"), ord("9") + 1))),
        (r"\u00e9x", set()),
    ],
)
def test_first_chars_of_modelled_constructs(pattern, expected):
    assert pattern_first_chars(pattern) == frozenset(expected)


def test_whitespace_category_includes_unicode_ascii_separators():
    first = pattern_first_chars(r"\s+")
    assert {ord(" "), ord("\t"), ord("\n"), 0x1C, 0x1F} <= first


@pytest.mark.parametrize(
    "pattern",
    [
        r"a*",  # can match the empty string
        r"(?i)abc",  # case-insensitive
        r"(?i:abc)",
        r"[^a]",
        r".x",
        r"(a)\1",
    ],
)
def test_unmodelled_or_unanchored_patterns_assume_every_character(pattern):
    first = pattern_first_chars(pattern)
    if pattern == r"(a)\1":
        assert ord("a") in first
    else:
        assert first == ALL_ASCII


def test_non_ascii_positions_use_the_full_table():
    table = [("WORD", r"\w+"), ("SPACE", r"\s+"), ("OTHER", r".")]
    dispatch = ordered_pattern_dispatch(table)
    assert dispatch.candidates("\u00e9") == dispatch.patterns
    assert dispatch.match("\u00a0x", 0)[0] == "SPACE"
    assert dispatch.match("\u03bbx", 0)[0] == "WORD"


def test_table_order_breaks_ties_within_a_bucket():
    table = [("KW", r"\bfor\b"), ("ID", r"[a-z]+"), ("DUP", r"\bfor\b")]
    dispatch = ordered_pattern_dispatch(table)
    assert [name for name, _ in dispatch.candidates("f")] == ["KW", "ID", "DUP"]
    assert dispatch.match("for x", 0)[0] == "KW"
    assert dispatch.match("fork", 0)[0] == "ID"
    assert dispatch.match("+", 0) is None


def test_crossgl_lexer_tokens_are_unchanged(shader_texts):
    from crosstl.translator import lexer as crossgl_lexer

    full = alternation_dispatch(crossgl_lexer.TOKENS).full
    for text in [t for t in shader_texts if "shader" in t][:20]:
        try:
            lexer = crossgl_lexer.Lexer(text)
        except SyntaxError:
            continue
        tokens = lexer.get_tokens()
        text = lexer.code
        expected = []
        pos = 0
        while pos < len(text):
            match = full.match(text, pos)
            assert match is not None
            kind = match.lastgroup
            value = match.group(kind)
            if kind == "IDENTIFIER" and value in crossgl_lexer.KEYWORDS:
                kind = crossgl_lexer.KEYWORDS[value]
            if kind not in crossgl_lexer.SKIP_TOKENS:
                expected.append((kind, value))
            pos = match.end(0)
        expected.append(("EOF", None))
        assert tokens == expected
