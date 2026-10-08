"""Source floating-point controls that require explicit translation support."""

import re


class MetalContractionDirectiveError(ValueError):
    """A source contraction directive must not be silently discarded."""

    project_diagnostic_code = "project.translate.metal-contraction-unsupported"
    missing_capabilities = ("metal.source-expression-contraction",)

    def __init__(self, directive):
        self.directive = directive
        self.reason = "source contraction directives are not preserved by translation"
        self.suggested_action = (
            "Retain the source directive and use the original source backend until "
            "its lexical contraction policy is supported; scalar arithmetic "
            "profiles and -fno-fast-math are not equivalent replacements"
        )
        super().__init__(
            f"Cannot preserve Metal floating-point directive '{directive}': "
            f"{self.reason}. {self.suggested_action}."
        )


def reject_contraction_directive(content):
    """Reject recognized contraction controls without matching comments."""
    content = re.sub(r"/\*[\s\S]*?\*/|//[^\n]*", " ", content).strip()
    if re.match(r"clang\s+fp\b[\s\S]*\bcontract\s*\(", content) or re.match(
        r"(?:STDC|OPENCL)\s+FP_CONTRACT\b", content
    ):
        raise MetalContractionDirectiveError(content)


def reject_contraction_tokens(tokens):
    """Check active tokens before template rewriting can move or erase bodies."""
    tokens = iter(tokens)
    for token in tokens:
        kind, text = token
        if kind == "PREPROCESSOR":
            text = re.sub(r"/\*[\s\S]*?\*/", " ", text)
            match = re.match(r"#\s*pragma\b(.*)", text, re.DOTALL)
            if match:
                reject_contraction_directive(match.group(1))
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
                reject_contraction_directive(content)
