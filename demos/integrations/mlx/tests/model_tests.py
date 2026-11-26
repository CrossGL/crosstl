"""Find the MLX tests that only exercise pure-Python models of expectation data.

Many kernel gates check their own reference data before any translation runs:
a Python model of the kernel recomputes the pinned expectations, every seeded
defect must change them, and the OpenGL index assertions must name real buffer
elements. Those tests never call CrossTL, so their outcome can only change
with ``demos/integrations/mlx/**``. They also take most of the MLX share of the
complete test suite.

:func:`mark_model_tests` gives such tests the ``mlx_model`` marker, so a CI
tier can deselect them on changes that cannot affect them. A test is marked
only when both of these hold:

* its name follows one of the model-check conventions in
  :data:`MODEL_TEST_NAME`;
* :func:`foreign_references` finds nothing outside this demo that its code can
  reach: no import of, call into or object from another package of the
  repository (``crosstl``, ``tests``, ``tools`` or another demo), no
  ``subprocess``, and no fixture argument other than its parametrization.

The reachability walk is static and conservative: it follows every global,
closure, default, container element, class member and module attribute that
the test's code names, through every function defined under this demo.
"""

from __future__ import annotations

import dis
import functools
import inspect
import re
import types
from collections.abc import Iterable, Iterator

import pytest

MARKER = "mlx_model"
MARKER_DESCRIPTION = (
    f"{MARKER}: pure-Python model of MLX expectation data; its outcome only "
    "changes with demos/integrations/mlx/**"
)

LANE_PACKAGE = "demos.integrations.mlx"
REPOSITORY_PACKAGES = ("crosstl", "tests", "tools", "demos")
FORBIDDEN_MODULES = ("subprocess",)

# Name conventions of the model checks in demos/integrations/mlx/tests/kernels.
MODEL_TEST_NAME = re.compile(
    r"^test_\w*_(?:"
    r"rejects?_defects"
    r"|model_matches_\w+"
    r"|opengl_assertions_name_buffer_elements"
    r"|keep_\w+_exact"
    r"|reference_matches_\w+"
    r"|references_are_exact\w*"
    r"|values_are_exact\w*"
    r"|bounds_stay_meaningful"
    r"|reach_every_path"
    r"|cases_follow_mlx_dispatch"
    r")$"
)


def _in_package(name: str | None, package: str) -> bool:
    return bool(name) and (name == package or name.startswith(package + "."))


def _foreign_module_name(name: str | None) -> bool:
    if _in_package(name, LANE_PACKAGE):
        return False
    return any(_in_package(name, package) for package in REPOSITORY_PACKAGES) or any(
        _in_package(name, module) for module in FORBIDDEN_MODULES
    )


def _code_objects(code: types.CodeType) -> Iterator[types.CodeType]:
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _code_objects(const)


@functools.lru_cache(maxsize=None)
def _foreign_imports(code: types.CodeType) -> tuple[str, ...]:
    """The repository or forbidden modules that ``code`` imports."""
    found = []
    for nested in _code_objects(code):
        # IMPORT_NAME takes its module name from co_names, so only code that
        # names a candidate module needs disassembling.
        candidates = [name for name in nested.co_names if _foreign_module_name(name)]
        if not candidates:
            continue
        for instruction in dis.get_instructions(nested):
            if instruction.opname == "IMPORT_NAME" and _foreign_module_name(
                instruction.argval
            ):
                found.append(instruction.argval)
    return tuple(found)


@functools.lru_cache(maxsize=None)
def _names(code: types.CodeType) -> frozenset[str]:
    names: set[str] = set()
    for nested in _code_objects(code):
        names.update(nested.co_names)
    return frozenset(names)


class _Walk:
    """One reachability walk from a test function."""

    def __init__(self) -> None:
        self.seen: set[int] = set()
        self.foreign: list[str] = []
        # Names a lane function's code uses. A lane module's attributes are
        # followed for each of these names, whichever function used it.
        self.names: set[str] = set()
        self.modules: dict[str, tuple[types.ModuleType, set[str]]] = {}

    def run(self, function: types.FunctionType) -> list[str]:
        self.visit(function, function.__qualname__)
        changed = True
        while changed:
            changed = False
            for name, (module, followed) in sorted(self.modules.items()):
                for attribute in sorted(self.names - followed):
                    followed.add(attribute)
                    changed = True
                    if attribute in vars(module):
                        self.visit(vars(module)[attribute], f"{name}.{attribute}")
        return self.foreign

    def visit(self, value: object, via: str) -> None:
        if id(value) in self.seen:
            return
        self.seen.add(id(value))
        if isinstance(value, types.ModuleType):
            self._module(value, via)
        elif isinstance(value, (types.FunctionType, types.MethodType)):
            self._function(getattr(value, "__func__", value), via)
        elif isinstance(value, functools.partial):
            self.visit(value.func, via)
            self._elements(value.args, via)
            self._elements(value.keywords.values(), via)
        elif hasattr(value, "__wrapped__") and callable(value):
            self.visit(value.__wrapped__, via)
        elif isinstance(value, type):
            self._class(value, via)
        elif isinstance(value, dict):
            self._elements(value.keys(), via)
            self._elements(value.values(), via)
        elif isinstance(value, (list, tuple, set, frozenset)):
            self._elements(value, via)
        else:
            self._instance(value, via)

    def _elements(self, values: Iterable[object], via: str) -> None:
        for value in values:
            if not isinstance(value, (str, bytes, int, float, complex, bool)):
                self.visit(value, via)

    def _module(self, module: types.ModuleType, via: str) -> None:
        if _foreign_module_name(module.__name__):
            self.foreign.append(f"{via}: module {module.__name__}")
        elif _in_package(module.__name__, LANE_PACKAGE):
            self.modules.setdefault(module.__name__, (module, set()))

    def _function(self, function: types.FunctionType, via: str) -> None:
        module = function.__module__
        if _foreign_module_name(module):
            self.foreign.append(f"{via}: {module}.{function.__qualname__}")
            return
        if not _in_package(module, LANE_PACKAGE):
            return
        code = function.__code__
        for name in _foreign_imports(code):
            self.foreign.append(f"{via}: imports {name}")
        names = _names(code)
        self.names |= names
        scope = function.__globals__
        for name in sorted(names):
            if name in scope:
                self.visit(scope[name], f"{module}.{name}")
        for cell in function.__closure__ or ():
            try:
                self.visit(cell.cell_contents, via)
            except ValueError:  # an unfilled cell
                continue
        self._elements(function.__defaults__ or (), via)
        self._elements((function.__kwdefaults__ or {}).values(), via)

    def _class(self, cls: type, via: str) -> None:
        if _foreign_module_name(cls.__module__):
            self.foreign.append(f"{via}: class {cls.__module__}.{cls.__qualname__}")
        elif _in_package(cls.__module__, LANE_PACKAGE):
            for base in cls.__mro__[1:]:
                self.visit(base, via)
            for name, member in sorted(vars(cls).items()):
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                elif isinstance(member, property):
                    self._elements((member.fget, member.fset, member.fdel), via)
                    continue
                self.visit(member, f"{via}.{name}")

    def _instance(self, value: object, via: str) -> None:
        cls = type(value)
        if _foreign_module_name(cls.__module__) or _in_package(
            cls.__module__, LANE_PACKAGE
        ):
            self.visit(cls, via)
        if _in_package(cls.__module__, LANE_PACKAGE) and hasattr(value, "__dict__"):
            self._elements(vars(value).values(), via)


def foreign_references(function: types.FunctionType) -> list[str]:
    """Everything outside this demo that ``function``'s code can reach."""
    return _Walk().run(function)


@functools.lru_cache(maxsize=None)
def _pure(function: types.FunctionType) -> bool:
    return not foreign_references(function)


def is_model_test(item: pytest.Item) -> bool:
    function = getattr(item, "obj", None)
    if not isinstance(function, types.FunctionType):
        return False
    if not _in_package(function.__module__, LANE_PACKAGE):
        return False
    if not MODEL_TEST_NAME.match(function.__name__):
        return False
    callspec = getattr(item, "callspec", None)
    parameters = set(callspec.params) if callspec is not None else set()
    if set(inspect.signature(function).parameters) - parameters:
        return False  # a fixture: model checks need nothing but their cases
    return _pure(function)


def mark_model_tests(items: Iterable[pytest.Item]) -> None:
    for item in items:
        if is_model_test(item):
            item.add_marker(MARKER)
