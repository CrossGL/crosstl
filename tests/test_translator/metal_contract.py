"""Assertions shared by native Metal corpus proofs."""

import re


def operator_implementations(source: str, operator: str) -> tuple[str, ...]:
    pattern = re.compile(
        rf"(?m)^(?:(?:static|inline)\s+)*[A-Za-z_][A-Za-z0-9_]*\s+"
        rf"({re.escape(operator)}__operator_call(?:__[A-Za-z0-9_]+)*)\("
    )
    return tuple(pattern.findall(source))
