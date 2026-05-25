import importlib
import inspect
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _catalog_backends():
    catalog = json.loads(
        (ROOT / "support" / "backends.json").read_text(encoding="utf-8")
    )
    return catalog["backends"]


def _module_name(path):
    return ".".join(path.with_suffix("").relative_to(ROOT).parts)


def _classes_declared_in(module):
    return [
        name
        for name, value in inspect.getmembers(module, inspect.isclass)
        if value.__module__ == module.__name__
    ]


def test_native_backend_modules_are_importable():
    failures = []

    for backend in _catalog_backends():
        backend_path = ROOT / backend["native_backend"]
        for module_path in sorted(backend_path.glob("*.py")):
            if module_path.name == "__init__.py":
                continue
            module = _module_name(module_path)
            try:
                importlib.import_module(module)
            except Exception as exc:
                failures.append(f"{module}: {type(exc).__name__}: {exc}")

    assert not failures, "Native backend import failures:\n" + "\n".join(failures)


def test_native_backend_packages_expose_core_frontend_classes():
    missing = []

    for backend in _catalog_backends():
        backend_id = backend["id"]
        backend_path = ROOT / backend["native_backend"]
        modules = {
            module_path.name.lower(): importlib.import_module(_module_name(module_path))
            for module_path in sorted(backend_path.glob("*.py"))
            if module_path.name != "__init__.py"
        }

        lexer_classes = [
            cls
            for module_name, module in modules.items()
            if "lexer" in module_name
            for cls in _classes_declared_in(module)
            if cls.endswith("Lexer")
        ]
        parser_classes = [
            cls
            for module_name, module in modules.items()
            if "parser" in module_name
            for cls in _classes_declared_in(module)
            if cls.endswith("Parser")
        ]
        converter_classes = [
            cls
            for module_name, module in modules.items()
            if "crossgl" in module_name
            for cls in _classes_declared_in(module)
            if cls.endswith("ToCrossGLConverter")
        ]

        if not lexer_classes:
            missing.append(f"{backend_id}: lexer class")
        if not parser_classes:
            missing.append(f"{backend_id}: parser class")
        if not converter_classes:
            missing.append(f"{backend_id}: reverse CrossGL converter class")

    assert not missing, "Missing native backend package contracts: " + ", ".join(missing)
