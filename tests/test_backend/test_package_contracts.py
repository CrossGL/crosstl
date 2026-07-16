import ast
import importlib
import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _setup_keyword(name):
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    setup_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    )
    keyword = next(item for item in setup_call.keywords if item.arg == name)
    return ast.literal_eval(keyword.value)


def _catalog_backends():
    catalog = json.loads(
        (ROOT / "support" / "backends.json").read_text(encoding="utf-8")
    )
    return catalog["backends"]


def _native_source_backends():
    return [
        backend
        for backend in _catalog_backends()
        if backend.get("source_kind", "native") == "native"
    ]


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

    for backend in _native_source_backends():
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


def test_directx_runtime_extra_declares_supported_compushady_release():
    extras = _setup_keyword("extras_require")

    assert extras["directx-runtime"] == [
        "compushady>=0.17.5,<0.18; platform_system=='Windows'"
    ]


def test_native_backend_packages_expose_core_frontend_classes():
    missing = []

    for backend in _native_source_backends():
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

    assert not missing, "Missing native backend package contracts: " + ", ".join(
        missing
    )
