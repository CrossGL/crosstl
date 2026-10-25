"""The Metal preprocessor's delimiter scans return exactly what their character-by-character versions returned.

Each reference below is the previous implementation, kept verbatim apart from its name, and is compared with the
current one on generated C++-like text full of the characters the scans treat specially.
"""

import random

import pytest

from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.backend.Metal.text_scanners import mask_comments_and_literals
from crosstl.project.pipeline import _masked_metal_non_code_text

FRAGMENTS = [
    "int",
    "x",
    " ",
    "\t",
    "\n",
    "(",
    ")",
    "{",
    "}",
    "[",
    "]",
    ",",
    ";",
    "=",
    "==",
    "!=",
    "+=",
    "<=",
    "<",
    ">",
    "<<",
    ">>",
    "/",
    "*",
    "*/",
    "//",
    "/*",
    "\\",
    '"',
    "'",
    '"a,=(\\"b"',
    "'c'",
    "// d, = (\n",
    "/* e, = ( */",
    "operator()",
]


def generated_sources(seed, count=400, length=30):
    rng = random.Random(seed)
    for _ in range(count):
        yield "".join(rng.choice(FRAGMENTS) for _ in range(rng.randint(0, length)))


def reference_split_top_level_assignment(self, text):
    # Split at the first top-level `=` that is a plain assignment (not part of
    # `==`/`<=`/`>=`/`!=` or a compound operator), returning (lhs, rhs) with
    # rhs None when there is no such assignment. Used to peel a data member's
    # default initializer from its declarator.
    paren = bracket = brace = angle = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in "\"'":
            _literal, consumed = self._read_string(text, i)
            i += consumed
            continue
        if text.startswith("//", i):
            break
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = len(text) if end == -1 else end + 2
            continue
        if ch == "(":
            paren += 1
        elif ch == ")":
            paren = max(0, paren - 1)
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket = max(0, bracket - 1)
        elif ch == "{":
            brace += 1
        elif ch == "}":
            brace = max(0, brace - 1)
        elif ch == "<":
            angle += 1
        elif ch == ">":
            angle = max(0, angle - 1)
        elif ch == "=" and paren == bracket == brace == angle == 0:
            nxt = text[i + 1] if i + 1 < len(text) else ""
            prv = text[i - 1] if i > 0 else ""
            if nxt != "=" and prv not in "=<>!+-*/%&|^~":
                return text[:i].strip(), text[i + 1 :].strip()
        i += 1
    return text.strip(), None


def reference_strip_top_level_default_value(self, text):
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    angle_depth = 0
    i = 0
    while i < len(text):
        if text[i] in "\"'":
            _literal, consumed = self._read_string(text, i)
            i += consumed
            continue
        if text.startswith("//", i):
            break
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end == -1:
                break
            i = end + 2
            continue

        ch = text[i]
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth = max(0, paren_depth - 1)
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth = max(0, brace_depth - 1)
        elif ch == "<":
            angle_depth += 1
        elif ch == ">":
            angle_depth = max(0, angle_depth - 1)
        elif (
            ch == "="
            and paren_depth == 0
            and bracket_depth == 0
            and brace_depth == 0
            and angle_depth == 0
        ):
            return text[:i].strip()
        i += 1
    return text.strip()


def reference_strip_template_argument_comments(self, value):
    result = ""
    i = 0
    while i < len(value):
        if value[i] in "\"'":
            literal, consumed = self._read_string(value, i)
            result += literal
            i += consumed
            continue
        if value.startswith("//", i):
            end = value.find("\n", i)
            if end == -1:
                break
            i = end + 1
            continue
        if value.startswith("/*", i):
            end = value.find("*/", i + 2)
            if end == -1:
                break
            i = end + 2
            continue
        result += value[i]
        i += 1
    return result


def reference_function_parameter_start(self, header):
    paren_depth = 0
    bracket_depth = 0
    angle_depth = 0
    i = 0
    while i < len(header):
        if header[i] in "\"'":
            _literal, consumed = self._read_string(header, i)
            i += consumed
            continue
        if header.startswith("//", i):
            end = header.find("\n", i)
            if end == -1:
                return None
            i = end + 1
            continue
        if header.startswith("/*", i):
            end = header.find("*/", i + 2)
            if end == -1:
                return None
            i = end + 2
            continue

        ch = header[i]
        if ch == "(" and paren_depth == 0 and bracket_depth == 0 and angle_depth == 0:
            return i
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth = max(0, paren_depth - 1)
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif ch == "<":
            angle_depth += 1
        elif ch == ">":
            angle_depth = max(0, angle_depth - 1)
        i += 1
    return None


def reference_split_top_level_commas(self, text):
    parts = []
    current = ""
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    angle_depth = 0
    i = 0
    while i < len(text):
        if text[i] in "\"'":
            literal, consumed = self._read_string(text, i)
            current += literal
            i += consumed
            continue
        if text.startswith("//", i):
            end = text.find("\n", i)
            if end == -1:
                current += text[i:]
                break
            current += text[i:end]
            i = end
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end == -1:
                current += text[i:]
                break
            current += text[i : end + 2]
            i = end + 2
            continue

        ch = text[i]
        if text.startswith("<<", i):
            current += "<<"
            i += 2
            continue
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth = max(0, paren_depth - 1)
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth = max(0, brace_depth - 1)
        elif ch == "<" and not (paren_depth or bracket_depth or brace_depth):
            angle_depth += 1
        elif ch == ">" and not (paren_depth or bracket_depth or brace_depth):
            angle_depth = max(0, angle_depth - 1)
        elif (
            ch == ","
            and paren_depth == 0
            and bracket_depth == 0
            and brace_depth == 0
            and angle_depth == 0
        ):
            parts.append(current.strip())
            current = ""
            i += 1
            continue
        current += ch
        i += 1
    if current.strip():
        parts.append(current.strip())
    return parts


def reference_masked_metal_non_code_text(source):
    chars = list(source)
    i = 0
    while i < len(chars):
        if source.startswith("//", i):
            end = source.find("\n", i)
            if end == -1:
                end = len(chars)
            for index in range(i, end):
                chars[index] = " "
            i = end
            continue
        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            if end == -1:
                end = len(chars) - 2
            for index in range(i, min(end + 2, len(chars))):
                if chars[index] != "\n":
                    chars[index] = " "
            i = end + 2
            continue
        if chars[i] in {"'", '"'}:
            quote = chars[i]
            chars[i] = " "
            i += 1
            while i < len(chars):
                current = chars[i]
                if current != "\n":
                    chars[i] = " "
                if current == "\\":
                    i += 2
                    continue
                i += 1
                if current == quote:
                    break
            continue
        i += 1
    return "".join(chars)


SCANS = [
    ("_split_top_level_commas", reference_split_top_level_commas),
    ("_split_top_level_assignment", reference_split_top_level_assignment),
    ("_strip_top_level_default_value", reference_strip_top_level_default_value),
    ("_strip_template_argument_comments", reference_strip_template_argument_comments),
    ("_function_parameter_start", reference_function_parameter_start),
]


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("name, reference", SCANS, ids=[name for name, _ in SCANS])
def test_preprocessor_scan_matches_reference(name, reference, seed):
    preprocessor = MetalPreprocessor()
    scan = getattr(preprocessor, name)
    for source in generated_sources(seed):
        assert scan(source) == reference(preprocessor, source), source


@pytest.mark.parametrize("seed", range(3))
def test_comment_and_literal_mask_matches_reference(seed):
    for source in generated_sources(seed + 5):
        expected = reference_masked_metal_non_code_text(source)
        assert mask_comments_and_literals(source) == expected, source
        assert _masked_metal_non_code_text(source) == expected, source
        assert len(expected) == len(source)


def test_scans_split_declarators_and_initializers():
    preprocessor = MetalPreprocessor()
    assert preprocessor._split_top_level_commas(
        "float a = f(1, 2), vec<int, 2> b /* , */, c[3] = {1, 2, 3}"
    ) == ["float a = f(1, 2)", "vec<int, 2> b /* , */", "c[3] = {1, 2, 3}"]
    assert preprocessor._split_top_level_commas("a << 1, b") == ["a << 1", "b"]
    assert preprocessor._split_top_level_assignment("x += 1") == ("x += 1", None)
    assert preprocessor._split_top_level_assignment("T x = a == b") == ("T x", "a == b")
    assert preprocessor._strip_top_level_default_value("int n = 4") == "int n"
    assert preprocessor._strip_top_level_default_value("int n // = 4") == "int n // = 4"
    assert (
        preprocessor._strip_template_argument_comments('a /* b */ "c//d" // e\nf')
        == 'a  "c//d" f'
    )
    assert (
        preprocessor._function_parameter_start("vec<int, (2)> f /* ( */ (int x)") == 24
    )
