"""Source floating-point controls of Metal: contraction and reassociation directives.

Metal compiles with Clang, which accepts these directives:

- ``#pragma clang fp contract(on|off|fast|fast-honor-pragmas)``, ``reassociate(on|off)``,
  ``exceptions(ignore|maytrap|strict)`` and ``eval_method(source|double|extended)`` (Clang language extensions,
  "Extensions to specify floating-point flags");
- ``#pragma METAL fp contract(on|off|fast)`` (MSL 4.1 section 1.6.3);
- ``#pragma STDC FP_CONTRACT ON|OFF|DEFAULT`` (C11 7.12.2) and ``#pragma OPENCL FP_CONTRACT ...``;

each as a ``#pragma`` line or as a ``_Pragma("...")`` operator.

CrossGL has no per-scope contraction control, but it has ``@precise`` functions, which evaluate all of their
floating-point arithmetic as written: no contraction and no reassociation. Contraction and reassociation are only
ever permitted, never required, so evaluating code without them is a valid evaluation of every setting above. A
directive that forbids contraction (``off``), limits it to one expression (``on``, ``ON``, ``DEFAULT``), or
forbids reassociation therefore makes the importer mark every function of the translation unit ``@precise``
(:func:`contraction_requirement`). That is coarser than the directive's scope, never less strict. ``fast`` and
``reassociate(on)`` permit what Metal's default fast-math mode already permits and need nothing.

Settings CrossGL cannot express, floating-point exception semantics other than ``ignore`` and evaluation methods
other than ``source``, are refused with :class:`MetalContractionDirectiveError`.
"""

import re

__all__ = [
    "NO_CONTRACTION_PRAGMA",
    "MetalContractionDirectiveError",
    "contraction_directive_requires_precise",
    "contraction_requirement",
]

# The directive the preprocessor restates when the unit requires @precise functions.
NO_CONTRACTION_PRAGMA = "clang fp contract(off)"

_COMMENTS = re.compile(r"/\*[\s\S]*?\*/|//[^\n]*")
_CLANG_FP = re.compile(r"clang\s+fp\b(?P<options>[\s\S]*)$")
_METAL_FP = re.compile(r"METAL\s+fp\s+contract\s*\(\s*(?P<value>[A-Za-z_-]+)\s*\)\s*$")
_FP_CONTRACT = re.compile(r"(?:STDC|OPENCL)\s+FP_CONTRACT\b(?P<value>[\s\S]*)$")
_OPTION = re.compile(r"\s*(?P<name>[A-Za-z_]+)\s*\(\s*(?P<value>[A-Za-z_-]+)\s*\)\s*")

# Each recognized clang fp option value: True when it requires @precise, False when it needs nothing.
_CLANG_FP_OPTIONS = {
    "contract": {"off": True, "on": True, "fast": False, "fast-honor-pragmas": False},
    "reassociate": {"off": True, "on": False},
    "exceptions": {"ignore": False},
    "eval_method": {"source": False},
}


class MetalContractionDirectiveError(ValueError):
    """A source floating-point directive whose setting CrossGL cannot express."""

    project_diagnostic_code = "project.translate.metal-contraction-unsupported"
    missing_capabilities = ("metal.source-expression-contraction",)

    def __init__(self, directive, detail="its setting has no CrossGL equivalent"):
        self.directive = directive
        self.reason = detail
        self.suggested_action = (
            "Remove the setting or use the original source backend; contraction and reassociation controls are "
            "preserved, floating-point exception semantics and evaluation methods are not"
        )
        super().__init__(
            f"Cannot preserve Metal floating-point directive '{directive}': {detail}. "
            f"{self.suggested_action}."
        )


def contraction_directive_requires_precise(content):
    """Classify the text of a ``#pragma``: ``True`` if it requires ``@precise``, ``False`` if it is a
    floating-point directive that needs nothing, ``None`` if it is no floating-point directive.

    Raises :class:`MetalContractionDirectiveError` for a floating-point directive with a setting CrossGL cannot
    express.
    """
    content = _COMMENTS.sub(" ", content).strip()
    match = _CLANG_FP.match(content)
    if match is not None:
        return _clang_fp_requires_precise(content, match.group("options"))
    match = _METAL_FP.match(content)
    if match is not None:
        value = match.group("value")
        if value not in {"off", "on", "fast"}:
            raise MetalContractionDirectiveError(
                content, f"'{value}' is not a contraction setting"
            )
        return value != "fast"
    if re.match(r"METAL\s+fp\s+contract\b", content):
        raise MetalContractionDirectiveError(content, "the directive is malformed")
    match = _FP_CONTRACT.match(content)
    if match is not None:
        value = match.group("value").strip()
        if value not in {"ON", "OFF", "DEFAULT"}:
            raise MetalContractionDirectiveError(
                content, f"'{value}' is not an FP_CONTRACT setting"
            )
        # ON limits contraction to one expression and DEFAULT restores the build's setting; neither can be
        # expressed per scope, and evaluating without contraction is valid for both.
        return True
    return None


def _clang_fp_requires_precise(content, options):
    requires_precise = False
    position = 0
    found = False
    while position < len(options):
        option = _OPTION.match(options, position)
        if option is None:
            if options[position:].strip():
                raise MetalContractionDirectiveError(
                    content, "the directive is malformed"
                )
            break
        found = True
        position = option.end()
        name, value = option.group("name"), option.group("value")
        values = _CLANG_FP_OPTIONS.get(name)
        if values is None:
            raise MetalContractionDirectiveError(
                content, f"the option '{name}' is not supported"
            )
        if value not in values:
            raise MetalContractionDirectiveError(
                content, f"'{name}({value})' has no CrossGL equivalent"
            )
        requires_precise = requires_precise or values[value]
    if not found:
        raise MetalContractionDirectiveError(content, "the directive names no option")
    return requires_precise


def contraction_requirement(tokens):
    """Return whether any active floating-point directive among ``tokens`` requires ``@precise`` functions.

    ``tokens`` are Metal lexer tokens; ``#pragma`` lines arrive as ``PREPROCESSOR`` tokens and ``_Pragma``
    operators as their own tokens. Raises :class:`MetalContractionDirectiveError` for a setting CrossGL cannot
    express.
    """
    requires_precise = False
    tokens = iter(tokens)
    for token in tokens:
        kind, text = token
        if kind == "PREPROCESSOR":
            text = re.sub(r"/\*[\s\S]*?\*/", " ", text)
            match = re.match(r"#\s*pragma\b(.*)", text, re.DOTALL)
            if match:
                requires_precise |= bool(
                    contraction_directive_requires_precise(match.group(1))
                )
        elif token == ("IDENTIFIER", "_Pragma"):
            if next(tokens, None) != ("LPAREN", "("):
                continue
            literal = next(tokens, None)
            if literal == ("IDENTIFIER", "L"):
                literal = next(tokens, None)
            if literal is None or literal[0] != "STRING":
                continue
            text = literal[1]
            if text.startswith('"') and text.endswith('"'):
                # _Pragma destringifies only escaped quotes and backslashes.
                content = re.sub(r'\\([\\"])', r"\1", text[1:-1])
                requires_precise |= bool(
                    contraction_directive_requires_precise(content)
                )
    return requires_precise
