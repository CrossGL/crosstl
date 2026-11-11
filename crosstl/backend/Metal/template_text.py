"""Template-argument substitution in the type and value text the Metal importer materializes.

A template's parameters are bound to the arguments of one instantiation, and the text of a default argument, an
alias template or a dependent type is rewritten with those bindings. An identifier the bindings do not bind is not
a parameter of the template being instantiated, so it names the same entity after substitution and is copied as
written.
"""

from __future__ import annotations

import re

__all__ = ["TEMPLATE_IDENTIFIER", "substitute_template_identifiers"]

TEMPLATE_IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*\b")


def substitute_template_identifiers(text, bindings):
    """``text`` with every identifier that ``bindings`` binds replaced by the text of its binding."""

    def replace(match):
        name = match.group(0)
        if name in bindings:
            return str(bindings[name])
        return name

    return TEMPLATE_IDENTIFIER.sub(replace, str(text))
