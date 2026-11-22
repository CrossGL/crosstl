"""The translator does not depend on the runtime and host tooling (docs/source/adr/0006-host-tooling-package.md).

Every module under ``crosstl/translator`` and ``crosstl/backend`` is parsed, and each import is resolved to an
absolute name: ``import``, ``from ... import`` with any number of leading dots, and calls of
``importlib.import_module`` or ``__import__`` with a constant name. The check fails on an import of:

- the host tooling, ``crosstl.project``, or the command-line module, ``crosstl._crosstl``;
- the package root itself (``import crosstl``), or a name of the root that is not one of its modules or packages
  (``from crosstl import translate``), because the root re-exports the command-line module's functions.

Submodules of the root such as ``crosstl.translator`` and ``crosstl.formatter`` remain importable.
"""

import ast
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
ROOTS = ("crosstl/translator", "crosstl/backend")
FORBIDDEN_PREFIXES = ("crosstl.project", "crosstl._crosstl")


def _package_of(path: Path, repository: Path) -> list[str]:
    # ``a/b/c.py`` is in package ``a.b``; ``a/b/__init__.py`` is package ``a.b`` itself.
    return list(path.relative_to(repository).with_suffix("").parts[:-1])


def _resolve(package: list[str], module: str | None, level: int) -> str:
    if level == 0:
        return module or ""
    base = package[: len(package) - (level - 1)]
    return ".".join([*base, *([module] if module else [])])


def imported_names(path: Path, repository: Path = REPOSITORY):
    """``(line, absolute name)`` for every module, or attribute of the package root, that ``path`` imports."""
    package = _package_of(path, repository)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve(package, node.module, node.level)
            if resolved == "crosstl":
                for alias in node.names:
                    yield node.lineno, f"crosstl.{alias.name}"
            else:
                yield node.lineno, resolved
        elif isinstance(node, ast.Call) and node.args:
            function = node.func
            name = (
                function.attr
                if isinstance(function, ast.Attribute)
                else getattr(function, "id", None)
            )
            first = node.args[0]
            if (
                name in ("import_module", "__import__")
                and isinstance(first, ast.Constant)
                and isinstance(first.value, str)
            ):
                yield node.lineno, first.value


def forbidden(name: str, repository: Path = REPOSITORY) -> bool:
    if name == "crosstl":
        return True
    if any(
        name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_PREFIXES
    ):
        return True
    parts = name.split(".")
    if len(parts) == 2 and parts[0] == "crosstl":
        root = repository / "crosstl"
        member = parts[1]
        return not ((root / member).is_dir() or (root / f"{member}.py").is_file())
    return False


def violations(roots=ROOTS, repository: Path = REPOSITORY) -> list[str]:
    found = []
    for root in roots:
        for path in sorted((repository / root).rglob("*.py")):
            relative = path.relative_to(repository).as_posix()
            for line, name in sorted(imported_names(path, repository)):
                if forbidden(name, repository):
                    found.append(f"{relative}:{line} imports {name}")
    return found


def test_translator_and_backends_do_not_import_host_tooling():
    assert violations() == []


def test_the_check_resolves_every_import_form(tmp_path):
    package = tmp_path / "crosstl" / "translator" / "codegen"
    package.mkdir(parents=True)
    (tmp_path / "crosstl" / "formatter.py").write_text("")
    (package / "probe.py").write_text(
        "import importlib\n"
        "import crosstl.project.pipeline\n"
        "from ...project import pipeline\n"
        "from ... import _crosstl\n"
        "from crosstl import translate\n"
        "from .registry import BACKEND_REGISTRY\n"
        "importlib.import_module('crosstl.project.runtime_graph')\n"
        "import crosstl\n"
        "import crosstl.formatter\n"
        "from crosstl import translator, formatter\n"
        "from .. import semantics\n"
        "def lazy():\n"
        "    from ..semantics import layouts\n"
        "    __import__('crosstl._crosstl')\n"
    )
    assert violations(("crosstl/translator",), tmp_path) == [
        "crosstl/translator/codegen/probe.py:2 imports crosstl.project.pipeline",
        "crosstl/translator/codegen/probe.py:3 imports crosstl.project",
        "crosstl/translator/codegen/probe.py:4 imports crosstl._crosstl",
        "crosstl/translator/codegen/probe.py:5 imports crosstl.translate",
        "crosstl/translator/codegen/probe.py:7 imports crosstl.project.runtime_graph",
        "crosstl/translator/codegen/probe.py:8 imports crosstl",
        "crosstl/translator/codegen/probe.py:14 imports crosstl._crosstl",
    ]
