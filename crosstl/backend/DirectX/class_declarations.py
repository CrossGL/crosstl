"""Read HLSL ``class`` definitions as ``struct`` definitions.

HLSL gives ``class`` and ``struct`` the same meaning: DXC accepts data members, member functions, static members and
base lists in both, and HLSL has no access control, so every member of a class is accessible like a struct's. The
importer therefore parses a class definition with the struct parser, which imports its members and methods, instead
of skipping it, which would leave every use of the class without a declaration.

Only a definition or forward declaration is rewritten: the keyword followed by a name and then ``{``, ``:``, ``;`` or
an explicit specialization's ``<``. The ``class`` of a template parameter (``template <class T>``) and of a scoped
enumeration (``enum class E``) keep their meaning. ``interface`` declarations are left to the parser, which skips
them: an interface declares only member function prototypes, which a class that implements it defines.
"""

__all__ = ["class_definitions_as_structs"]

_DEFINITION_FOLLOWERS = frozenset({"LBRACE", "COLON", "SEMICOLON", "LESS_THAN"})


def _is_class_keyword(token):
    return token[0] in {"IDENTIFIER", "CLASS"} and token[1] == "class"


def class_definitions_as_structs(tokens):
    """``tokens`` with the ``class`` keyword of each class definition or forward declaration lexed as ``struct``."""
    rewritten = None
    for index, token in enumerate(tokens):
        if not _is_class_keyword(token):
            continue
        if index > 0 and tokens[index - 1][0] == "ENUM":
            continue
        if index + 2 >= len(tokens) or tokens[index + 1][0] != "IDENTIFIER":
            continue
        if tokens[index + 2][0] not in _DEFINITION_FOLLOWERS:
            continue
        if rewritten is None:
            rewritten = list(tokens)
        rewritten[index] = ("STRUCT", "class")
    return tokens if rewritten is None else rewritten
