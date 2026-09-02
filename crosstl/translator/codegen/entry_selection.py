"""Shared preparation for entry-scoped target generation."""


def prepare_entry_scoped_target(codegen, ast, entry_point):
    """Return the AST and residual selector for one target entry.

    Backends exposing ``entry_scoped_ast`` can prune unrelated declarations
    before target capability validation. Backends that only expose
    ``generate_entry`` retain the selector and their existing generation path.
    """

    if entry_point is None:
        return ast, None
    scope_entry = getattr(codegen, "entry_scoped_ast", None)
    if not callable(scope_entry):
        return ast, entry_point
    return scope_entry(ast, entry_point), None
