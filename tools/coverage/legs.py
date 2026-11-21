"""Which optional Coverage legs a pull request or merge group runs.

The Linux leg measures its languages on every change. The Metal leg needs a macOS runner, and the CUDA, HIP and Mojo
leg and the Rust leg install toolchains that take minutes, so a change runs one of them only when it can move that
leg's rows: it touches the coverage engine, a file under the leg's own front ends or generators, a runtime file the
leg executes with, or a CrossTL module that only the leg's languages import. A module that a Linux-leg language
imports as well, such as the CrossGL parser, is measured through that language on the change, and through every
language on each push to ``main``.

The import graph comes from the ``import`` statements of every module under ``crosstl/``, including those inside
functions, without running any of them, so the planner needs nothing but Python.

The Direct3D 12 leg runs the HLSL probes, which the Linux leg measures, on WARP on a Windows runner as well. A pull
request runs it only when it touches the coverage engine, HLSL's own front end or generator modules, or the files the
D3D12 runtime executes with. Coverage gaps never fail a check, so the merge queue leaves the leg out
(:data:`OUTSIDE_QUEUE`) and the push to ``main`` that follows measures it.

``git diff --name-only HEAD^1 HEAD | python3 -m tools.coverage.legs [--merge-group]`` reads the changed paths and prints
one ``<leg>=true`` or ``<leg>=false`` line per leg, in the format of ``$GITHUB_OUTPUT``.
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

PACKAGE = "crosstl"
PIPELINE = "crosstl/__init__.py"
CODEGEN = "crosstl/translator/codegen/"
BACKEND = "crosstl/backend/"
ENGINE = re.compile(
    r"^(tools/coverage/|tests/coverage/|\.github/workflows/coverage\.yml$)"
)


@dataclass(frozen=True)
class Leg:
    """An optional leg: the prefixes of its generators, its front-end directories and its runtime files.

    ``alone`` says that only this leg measures its languages. A leg that runs probes of a Linux-leg language on another
    runtime is not alone: the modules its languages import stay shared, and it runs only for its own files.
    """

    generators: tuple[str, ...]
    frontends: tuple[str, ...]
    runtime: str = r"(?!)"
    alone: bool = True

    def owns(self, path: str) -> bool:
        return (
            path.startswith(self.frontends)
            or path.startswith(self.generators)
            or bool(re.fullmatch(self.runtime, path))
        )


LEGS = {
    "metal": Leg(
        generators=(CODEGEN + "metal",),
        frontends=(BACKEND + "Metal/",),
        runtime=r"tools/ensure_metal_toolchain\.py|tests/conformance/(runner|runtimes|native|interface|worker)\.py",
    ),
    "compute": Leg(
        generators=(CODEGEN + "cuda_", CODEGEN + "hip_", CODEGEN + "mojo_"),
        frontends=(BACKEND + "CUDA/", BACKEND + "HIP/", BACKEND + "Mojo/"),
    ),
    "rust": Leg(
        generators=(CODEGEN + "rust_",),
        frontends=(BACKEND + "Rust/",),
        runtime=r"tests/conformance/(runner\.py|runtimes\.py|interface\.py|vkrun\.c)",
    ),
    "d3d12": Leg(
        generators=(CODEGEN + "directx_", CODEGEN + "hlsl_"),
        frontends=(BACKEND + "DirectX/",),
        runtime=r"tests/conformance/(runner|runtimes|interface|toolchains|validators|worker)\.py"
        r"|crosstl/project/directx_(runtime|runtime_worker|toolchain)\.(py|cpp)"
        r"|\.github/actions/install-windows-d3d12/action\.yml",
        alone=False,
    ),
}
# The legs the merge queue leaves out.
OUTSIDE_QUEUE = frozenset({"d3d12"})


def _module_name(path: str) -> str:
    name = path.removesuffix(".py").replace("/", ".")
    return name.removesuffix(".__init__")


def _modules(root: Path) -> dict[str, str]:
    """Every module under the package, by dotted name, mapped to its path relative to ``root``."""
    found = {}
    for file in sorted((root / PACKAGE).rglob("*.py")):
        path = file.relative_to(root).as_posix()
        found[_module_name(path)] = path
    return found


def _imports(
    source: str, name: str, is_package: bool, modules: dict[str, str]
) -> set[str]:
    """The package's own modules that ``source``, the module ``name``, imports."""
    package = name.split(".") if is_package else name.split(".")[:-1]
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else []
            stem = ".".join([*base, *(node.module.split(".") if node.module else [])])
            # ``from a import b`` names the module a.b when there is one, and otherwise something defined in a.
            targets = [
                f"{stem}.{alias.name}" if f"{stem}.{alias.name}" in modules else stem
                for alias in node.names
            ]
        else:
            continue
        found.update(target for target in targets if target in modules)
    return found


def import_graph(root: Path) -> dict[str, set[str]]:
    """For each module path under ``crosstl/``, the module paths it imports."""
    modules = _modules(root)
    graph = {}
    for name, path in modules.items():
        source = (root / path).read_text(encoding="utf-8")
        imported = _imports(source, name, path.endswith("__init__.py"), modules)
        graph[path] = {modules[target] for target in imported}
    return graph


def _reachable(
    graph: dict[str, set[str]], roots: Iterable[str], stop: frozenset[str] = frozenset()
) -> set[str]:
    """The modules ``roots`` import, directly or not, without passing through ``stop``."""
    seen = set()
    pending = [root for root in roots if root in graph]
    while pending:
        path = pending.pop()
        if path not in seen:
            seen.add(path)
            pending.extend(graph[path] - stop)
    return seen


def exclusive_modules(root: Path) -> dict[str, set[str] | None]:
    """For each optional leg, the modules only its languages import, or ``None`` when it has no generator left.

    A language's modules are its generator (``codegen/<language>_codegen.py``) and its front end
    (``backend/<Language>/``). The translation pipeline, which ``crosstl/__init__.py`` reaches without passing through
    them, serves every language, so it counts as shared like the modules of the Linux leg's languages.
    """
    graph = import_graph(root)
    generators = {
        path
        for path in graph
        if path.startswith(CODEGEN) and path.endswith("_codegen.py")
    }
    frontends = {
        path for path in graph if path.startswith(BACKEND) and path.count("/") > 2
    }
    languages = generators | frontends
    own = {
        name: {path for path in languages if leg.owns(path)}
        for name, leg in LEGS.items()
        if leg.alone
    }
    linux = languages - set().union(*own.values())
    shared = _reachable(graph, linux) | _reachable(
        graph, [PIPELINE], stop=frozenset(languages)
    )
    return {
        name: (_reachable(graph, paths) - shared) if generators & paths else None
        for name, paths in own.items()
    }


def plan(
    changed: Iterable[str], root: Path, merge_group: bool = False
) -> dict[str, bool]:
    """Whether each optional leg runs for a change to the paths ``changed``, given the tree at ``root``, in a merge
    group when ``merge_group``."""
    decision = _planned(changed, root)
    if merge_group:
        decision.update(dict.fromkeys(OUTSIDE_QUEUE, False))
    return decision


def _planned(changed: Iterable[str], root: Path) -> dict[str, bool]:
    changed = [path.strip() for path in changed if path.strip()]
    if any(ENGINE.match(path) for path in changed):
        return {name: True for name in LEGS}
    decision = {
        name: any(leg.owns(path) for path in changed) for name, leg in LEGS.items()
    }
    if all(decision.values()) or not any(p.startswith(PACKAGE + "/") for p in changed):
        return decision
    for name, exclusive in exclusive_modules(root).items():
        # A leg whose generators have all been renamed can no longer be planned, so it runs.
        decision[name] |= exclusive is None or any(p in exclusive for p in changed)
    return decision


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    merge_group = "--merge-group" in arguments
    arguments = [a for a in arguments if a != "--merge-group"]
    root = Path(arguments[0]) if arguments else Path.cwd()
    for name, runs in plan(sys.stdin, root, merge_group).items():
        print(f"{name}={'true' if runs else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
