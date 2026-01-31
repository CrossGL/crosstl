import os


def _backend_root():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "crosstl", "backend")
    )


def _backend_dirs():
    backend_root = _backend_root()
    if not os.path.isdir(backend_root):
        return []
    return sorted(
        [
            name
            for name in os.listdir(backend_root)
            if os.path.isdir(os.path.join(backend_root, name))
            and not name.startswith(".")
            and not name.startswith("__")
        ]
    )


def _backend_files(backend_dir: str):
    return [
        name
        for name in os.listdir(backend_dir)
        if os.path.isfile(os.path.join(backend_dir, name))
    ]


def test_backend_directories_follow_contract():
    missing = []

    for backend in _backend_dirs():
        backend_dir = os.path.join(_backend_root(), backend)
        files = _backend_files(backend_dir)
        lower_files = [name.lower() for name in files]

        has_init = "__init__.py" in lower_files
        has_lexer = any(name.endswith("lexer.py") for name in lower_files)
        has_parser = any(name.endswith("parser.py") for name in lower_files)
        has_ast = any(name.endswith("ast.py") for name in lower_files)
        has_crossgl = any(
            ("crossgl" in name.lower() or "crossglcodegen" in name.lower())
            for name in files
        )

        if not (has_init and has_lexer and has_parser and has_ast and has_crossgl):
            missing.append(
                {
                    "backend": backend,
                    "init": has_init,
                    "lexer": has_lexer,
                    "parser": has_parser,
                    "ast": has_ast,
                    "crossgl": has_crossgl,
                }
            )

    assert not missing, f"Backend contract violations: {missing}"
