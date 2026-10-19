#!/usr/bin/env python3
"""Record the predefined macros of the Metal compiler and of DXC for the shared preprocessor's dialects.

The shared preprocessor (``crosstl.translator.preprocessor``) must take the same ``#if`` branches as the compiler a
shader was written for, so the MSL and HLSL dialects need that compiler's predefined macros. Neither compiler
documents them completely; both are read from the toolchain itself and committed as package data:

``msl``
    ``metal -dM -E -fno-modules`` on an empty translation unit for every language standard the compiler lists in
    its ``-std`` diagnostic, on macOS and iOS (the macros of the implicitly included ``<metal_types>`` included).
    Also recorded: the deployment-version macros and ``__AIR_VERSION__`` for a grid of OS versions, the effect of
    each macro-changing compiler option, and, for every public header of the toolchain's ``include/metal``
    directory, whether it can be included per standard, the macros it adds, the public headers it pulls in and
    the macro names its own ``#if`` lines test (a user definition of one of those would change the header).

``hlsl``
    DXC has no ``-dM``. The candidate names are every macro name in ``InitPreprocessor.cpp`` of DXC's source at
    the recorded release (fetched, sha256-pinned); each candidate is probed with ``dxc -P`` and its replacement
    list read with every other candidate pushed and undefined, so nested macros are recorded unexpanded. The
    profile-, version- and option-dependent macros are probed over every stage, shader model, ``-HV`` value and
    ``-fspv-target-env`` the compiler recognizes.

Usage, with the recorded toolchains (``--check`` exits non-zero when the committed data differs)::

    python tools/preprocessor_macros.py msl --check
    python tools/preprocessor_macros.py hlsl --dxc "docker run --rm -i --platform linux/amd64 IMAGE" --check
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import re
import shlex
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crosstl.translator.preprocessor.lexer import (  # noqa: E402
    IDENT,
    NEWLINE,
    LexMode,
    SourceFile,
    lex,
)

DATA = ROOT / "crosstl" / "translator" / "preprocessor" / "data"
SCHEMA = 1
WORKERS = 8

# Deployment targets recorded per platform (the newest is the default); the OS version changes __AIR_VERSION__,
# the deployment-version macros and, for old standards, the __MAX_*__ resource limits.
OS_GRID = {
    "macos": (
        "10.11",
        "10.12",
        "10.13",
        "10.14",
        "10.15",
        "11.0",
        "12.0",
        "13.0",
        "14.0",
        "15.0",
        "26.0",
        "27.0",
    ),
    "ios": tuple(f"{major}.0" for major in range(8, 19)) + ("26.0", "27.0"),
}
TRIPLE = {"macos": "air64-apple-macos{}", "ios": "air64-apple-ios{}"}
# Compiler options that change predefined macros, by the dialect's option name.
MSL_OPTIONS = {
    "math-mode=relaxed": ("-fmetal-math-mode=relaxed",),
    "math-mode=safe": ("-fmetal-math-mode=safe",),
    "fp32-functions=precise": ("-fmetal-math-fp32-functions=precise",),
    "O0": ("-O0",),
    "Os": ("-Os",),
    "enable-logging": ("-fmetal-enable-logging",),
}
FEATURE_TESTS = {
    "defined",
    "__has_include",
    "__has_include_next",
    "__has_feature",
    "__has_extension",
    "__has_attribute",
    "__has_cpp_attribute",
    "__has_c_attribute",
    "__has_declspec_attribute",
    "__has_builtin",
    "__has_constexpr_builtin",
    "__has_warning",
    "__is_identifier",
    "__is_target_arch",
    "__is_target_os",
    "__is_target_environment",
    "__is_target_vendor",
    "__is_target_variant_os",
    "__is_target_variant_environment",
}
CONDITIONAL_DIRECTIVES = {"if", "ifdef", "ifndef", "elif", "elifdef", "elifndef"}


class GenerationError(RuntimeError):
    pass


# Shared helpers ------------------------------------------------------------------------------------------
def parse_defines(text: str) -> dict[str, str]:
    """``{name: definition}`` from ``-dM`` output; the definition is the text after ``#define ``."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("#define "):
            continue
        definition = line[len("#define ") :]
        name = re.match(r"[A-Za-z_$][A-Za-z0-9_$]*", definition).group()
        found[name] = definition
    return found


class Pool:
    """Interns definitions and encodes macro sets as deltas against a base set."""

    def __init__(self):
        self.sets: dict[str, dict[str, str]] = {}
        self.bases: dict[str, str | None] = {}

    def add(self, key: str, macros: dict[str, str], base: str | None) -> None:
        self.sets[key] = macros
        self.bases[key] = base

    def encode(self) -> dict:
        definitions = sorted(
            {value for macros in self.sets.values() for value in macros.values()}
        )
        index = {value: position for position, value in enumerate(definitions)}
        encoded = {}
        for key in sorted(self.sets):
            current = set(self.sets[key].values())
            base = self.bases[key]
            previous = set(self.sets[base].values()) if base else set()
            encoded[key] = {
                "base": base,
                "add": sorted(index[value] for value in current - previous),
                "remove": sorted(index[value] for value in previous - current),
            }
        return {"definitions": definitions, "sets": encoded}


def dumps(data: dict) -> str:
    """JSON with one line per dictionary entry and every list on one line: compact, and readable in a diff."""

    def render(value, depth: int) -> str:
        if isinstance(value, dict) and value:
            pad = " " * (depth + 1)
            items = [
                f"{pad}{json.dumps(key)}: {render(value[key], depth + 1)}"
                for key in sorted(value)
            ]
            return "{\n" + ",\n".join(items) + "\n" + " " * depth + "}"
        flat = json.dumps(
            value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        )
        if (
            isinstance(value, list)
            and len(flat) > 120
            and not all(isinstance(item, int) for item in value)
        ):
            pad = " " * (depth + 1)
            return (
                "[\n"
                + ",\n".join(pad + render(item, depth + 1) for item in value)
                + "\n"
                + " " * depth
                + "]"
            )
        return flat

    return render(data, 0) + "\n"


def finish(name: str, data: dict, write: bool) -> int:
    path = DATA / f"{name}.json"
    text = dumps(data)
    if write:
        DATA.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"Wrote {path.relative_to(ROOT)} ({len(text) // 1024} KiB)")
        return 0
    if path.is_file() and path.read_text() == text:
        print(f"{path.relative_to(ROOT)} matches the toolchain")
        return 0
    print(
        f"{path.relative_to(ROOT)} differs from the toolchain; rerun with --write after reviewing",
        file=sys.stderr,
    )
    return 1


# Metal ---------------------------------------------------------------------------------------------------
class Metal:
    def __init__(self, command: Sequence[str]):
        self.command = list(command)

    def run(self, args: Sequence[str], source: str = "") -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                *self.command,
                "-x",
                "metal",
                "-fno-modules",
                "-fno-color-diagnostics",
                *args,
                "-",
            ],
            input=source,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )

    def macros(
        self, std: str, platform: str, os_version: str, extra=(), source: str = ""
    ) -> dict[str, str]:
        result = self.run(
            [
                "-dM",
                "-E",
                f"-std={std}",
                "-target",
                TRIPLE[platform].format(os_version),
                *extra,
            ],
            source,
        )
        if result.returncode != 0:
            raise GenerationError(
                f"metal -std={std} {' '.join(extra)} failed: {result.stderr[-400:]}"
            )
        return parse_defines(result.stdout)

    def version(self) -> str:
        lines = subprocess.run(
            [*self.command, "--version"], capture_output=True, text=True, check=False
        ).stdout.splitlines()
        return lines[0].strip() if lines else ""

    def standards(self) -> list[tuple[str, str | None, tuple[str, ...]]]:
        """``(std, platform or None for both, aliases)`` from the compiler's own list of ``-std`` values."""
        result = self.run(["-E", "-std=crosstl-probe"])
        found = []
        for line in result.stderr.splitlines():
            match = re.match(r"note: use (.+) for '([^']+)' standard", line)
            if match is None:
                continue
            names = re.findall(r"'([^']+)'", match.group(1))
            title = match.group(2)
            platform = (
                "ios" if "(iOS)" in title else "macos" if "(macOS)" in title else None
            )
            found.append((names[0], platform, tuple(names[1:])))
        if not found:
            raise GenerationError(
                f"cannot read the -std values of the Metal compiler: {result.stderr[-400:]}"
            )
        return found

    def default_standard(self, standards: Sequence[str]) -> str:
        """The macOS ``-std`` whose ``__METAL_VERSION__`` the compiler uses without one."""
        version = parse_defines(self.run(["-dM", "-E"]).stdout)["__METAL_VERSION__"]
        for std in standards:
            if (
                self.macros(std, "macos", OS_GRID["macos"][-1])["__METAL_VERSION__"]
                == version
            ):
                return std
        raise GenerationError(f"no -std gives the default {version}")

    def dependencies(
        self, std: str, platform: str, os_version: str, header: str | None
    ) -> list[str] | None:
        """Files read for ``#include <header>`` (for an empty translation unit when ``header`` is None)."""
        source = f"#include <{header}>\n" if header else ""
        result = self.run(
            ["-M", f"-std={std}", "-target", TRIPLE[platform].format(os_version)],
            source,
        )
        if result.returncode != 0:
            return None
        text = result.stdout.replace("\\\n", " ")
        return [item for item in text.split()[1:] if item != "-"]

    def include_root(self) -> Path:
        files = self.dependencies(
            "metal3.0", "macos", OS_GRID["macos"][-1], "metal_stdlib"
        )
        for item in files or ():
            if item.endswith("/metal_stdlib"):
                return Path(item).resolve().parent
        raise GenerationError("cannot locate <metal_stdlib>")


def public_headers(root: Path) -> list[str]:
    """Every header a shader can include: the files of ``include/metal`` and of its ``simd`` directory."""
    headers = [
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != "module.modulemap"
    ]
    simd = root / "simd"
    if simd.is_dir():
        headers += [f"simd/{path.name}" for path in simd.iterdir() if path.is_file()]
    return sorted(headers)


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(
                f"{path.relative_to(root).as_posix()}\0{hashlib.sha256(path.read_bytes()).hexdigest()}\n".encode()
            )
    return digest.hexdigest()


def conditional_names(paths: Iterable[str], mode: LexMode) -> set:
    """Identifiers tested by the conditional directives of ``paths``, except feature-test names and arguments."""
    names = set()
    for path in paths:
        tokens = lex(
            SourceFile(
                path, Path(path).read_text(errors="replace"), splice_whitespace=True
            ),
            mode=mode,
        )
        line: list = []
        for token in tokens:
            if token.kind != NEWLINE:
                line.append(token)
                continue
            if (
                len(line) > 2
                and line[0].text == "#"
                and line[1].text in CONDITIONAL_DIRECTIVES
            ):
                depth, skip = 0, False
                for index, item in enumerate(line[2:], start=2):
                    if item.text in FEATURE_TESTS and item.text != "defined":
                        skip, depth = True, 0
                        continue
                    if skip:
                        depth += {"(": 1, ")": -1}.get(item.text, 0)
                        skip = depth > 0
                        continue
                    if (
                        item.kind == IDENT
                        and item.text not in FEATURE_TESTS
                        and item.text not in ("true", "false")
                    ):
                        names.add(item.text)
            line = []
    return names


def msl_platform_standards(standards, platform):
    return [std for std, owner, _ in standards if owner in (None, platform)]


class MslRecorder:
    """Probes one Metal toolchain; :meth:`record` returns the ``msl.json`` document."""

    def __init__(self, compiler: Sequence[str]):
        self.metal = Metal(compiler)
        self.standards = self.metal.standards()
        self.root = self.metal.include_root()
        self.headers = public_headers(self.root)
        self.lex_mode = LexMode(raw_strings=False)
        self.pool = Pool()
        self.lists = Interner()
        self.names = Interner()

    def chains(self):
        """``(platform, reference OS, [std, ...])`` for each platform."""
        for platform, grid in OS_GRID.items():
            yield platform, grid[-1], msl_platform_standards(self.standards, platform)

    def probe(self, executor) -> dict:
        jobs = {}
        for platform, reference, stds in self.chains():
            for std in stds:
                submit = executor.submit
                jobs[("base", platform, std)] = submit(
                    self.metal.macros, std, platform, reference
                )
                for version in OS_GRID[platform][:-1]:
                    jobs[("os", platform, std, version)] = submit(
                        self.metal.macros, std, platform, version
                    )
                for option, flags in MSL_OPTIONS.items():
                    jobs[("option", platform, std, option)] = submit(
                        self.metal.macros, std, platform, reference, flags
                    )
                jobs[("implicit", platform, std)] = submit(
                    self.metal.dependencies, std, platform, reference, None
                )
                for header in self.headers:
                    jobs[("deps", platform, std, header)] = submit(
                        self.metal.dependencies, std, platform, reference, header
                    )
        results = {key: future.result() for key, future in jobs.items()}
        option_macros = {option: set() for option in MSL_OPTIONS}
        for key, value in results.items():
            if key[0] == "option":
                base = results[("base", key[1], key[2])]
                option_macros[key[3]] |= {
                    name
                    for name in set(base) | set(value)
                    if base.get(name) != value.get(name)
                }
        jobs = {}
        for platform, reference, stds in self.chains():
            for std in stds:
                for header in self.headers:
                    files = results[("deps", platform, std, header)]
                    if files is None:
                        continue
                    source = f"#include <{header}>\n"
                    jobs[("header", platform, std, header)] = executor.submit(
                        self.metal.macros, std, platform, reference, (), source
                    )
                    conditions = conditional_names(files, self.lex_mode)
                    results[("conditions", platform, std, header)] = conditions
                    for option, flags in MSL_OPTIONS.items():
                        if option_macros[option] & conditions:
                            jobs[("header-option", platform, std, header, option)] = (
                                executor.submit(
                                    self.metal.macros,
                                    std,
                                    platform,
                                    reference,
                                    flags,
                                    source,
                                )
                            )
        results.update({key: future.result() for key, future in jobs.items()})
        return results

    def record(self) -> dict:
        with concurrent.futures.ThreadPoolExecutor(WORKERS) as executor:
            results = self.probe(executor)
        options = self.record_options(results)
        implicit = self.record_implicit(results)
        headers = {
            header: self.record_header(header, results, options)
            for header in self.headers
        }
        self.check_consistency(headers)
        encoded = self.pool.encode()
        return {
            "schema": SCHEMA,
            "source": {
                "by": "tools/preprocessor_macros.py msl",
                "toolchain": self.metal.version(),
                "headersSha256": tree_digest(self.root),
                "command": (
                    "metal -x metal -fno-modules -dM -E -std=<std> -target <triple> -"
                ),
            },
            "defaultStandard": self.metal.default_standard(
                msl_platform_standards(self.standards, "macos")
            ),
            "standards": {platform: stds for platform, _, stds in self.chains()},
            "aliases": {
                alias: std for std, _, aliases in self.standards for alias in aliases
            },
            "osVersions": {platform: list(grid) for platform, grid in OS_GRID.items()},
            "options": options,
            "implicit": implicit,
            "headers": headers,
            "headerLists": self.lists.values,
            "conditionNames": self.names.values,
            "definitions": encoded["definitions"],
            "sets": encoded["sets"],
        }

    def check_consistency(self, headers) -> None:
        """A header's macro set must contain the macro set of every header it includes, unchanged."""
        for header, entry in headers.items():
            for platform, _, stds in self.chains():
                for std, index in zip(stds, entry["closure"][platform]):
                    if index is None:
                        continue
                    own = set(
                        self.pool.sets[f"header:{header}:{platform}/{std}"].items()
                    )
                    for other in self.lists.values[index]:
                        other_set = self.pool.sets.get(
                            f"header:{other}:{platform}/{std}"
                        )
                        if (
                            other != header
                            and other_set is not None
                            and not set(other_set.items()) <= own
                        ):
                            raise GenerationError(
                                f"<{header}> and <{other}> disagree on a macro at {platform}/{std}"
                            )

    def record_options(self, results) -> dict:
        options = {}
        for option in MSL_OPTIONS:
            deltas = set()
            for platform, _, stds in self.chains():
                for std in stds:
                    delta = macro_delta(
                        results[("base", platform, std)],
                        results[("option", platform, std, option)],
                    )
                    deltas.add(json.dumps(delta, sort_keys=True))
            if len(deltas) != 1:
                raise GenerationError(
                    f"option {option} changes different macros per standard"
                )
            options[option] = dict(json.loads(deltas.pop()), changesHeaders=[])
        return options

    def record_implicit(self, results) -> dict:
        """The headers the driver includes before the source, and the macro names their conditionals test."""
        implicit = {"headers": {}, "conditions": set()}
        for platform, reference, stds in self.chains():
            previous = None
            row = []
            for std in stds:
                key = f"{platform}/{std}"
                base = results[("base", platform, std)]
                self.pool.add(
                    f"predefined:{key}",
                    base,
                    f"predefined:{previous}" if previous else None,
                )
                for version in OS_GRID[platform][:-1]:
                    self.pool.add(
                        f"predefined:{key}@{version}",
                        results[("os", platform, std, version)],
                        f"predefined:{key}",
                    )
                files = results[("implicit", platform, std)] or []
                row.append(
                    self.lists.add(
                        sorted(
                            {header_name(self.root, item) for item in files} - {None}
                        )
                    )
                )
                implicit["conditions"] |= conditional_names(files, self.lex_mode)
                previous = key
            implicit["headers"][platform] = row
        implicit["conditions"] = sorted(
            self.names.add(name) for name in sorted(implicit["conditions"])
        )
        return implicit

    def record_header(self, header: str, results, options) -> dict:
        """Per platform, one entry per standard: ``null`` when the header cannot be included, else its closure."""
        entry = {"closure": {}, "conditions": set()}
        for platform, _, stds in self.chains():
            previous = None
            row = []
            for std in stds:
                key = f"{platform}/{std}"
                files = results[("deps", platform, std, header)]
                if files is None:
                    row.append(None)
                    continue
                base = results[("base", platform, std)]
                after = results[("header", platform, std, header)]
                if set(base) - set(after):
                    raise GenerationError(
                        f"<{header}> undefines predefined macros at {key}"
                    )
                added = {
                    name: value
                    for name, value in after.items()
                    if base.get(name) != value
                }
                for option, flags in MSL_OPTIONS.items():
                    probed = results.get(
                        ("header-option", platform, std, header, option)
                    )
                    if probed is None:
                        continue
                    optioned = results[("option", platform, std, option)]
                    changed = {
                        name: value
                        for name, value in probed.items()
                        if optioned.get(name) != value
                    }
                    if (
                        changed != added
                        and header not in options[option]["changesHeaders"]
                    ):
                        options[option]["changesHeaders"].append(header)
                set_key = f"header:{header}:{key}"
                self.pool.add(set_key, added, previous)
                previous = set_key
                row.append(
                    self.lists.add(
                        sorted(
                            {header_name(self.root, item) for item in files} - {None}
                        )
                    )
                )
                entry["conditions"] |= results[("conditions", platform, std, header)]
            entry["closure"][platform] = row
        entry["conditions"] = sorted(
            self.names.add(name) for name in sorted(entry["conditions"])
        )
        for option in options.values():
            option["changesHeaders"].sort()
        return entry


class Interner:
    """A list of distinct JSON values, referenced by index."""

    def __init__(self):
        self.values: list = []
        self._index: dict[str, int] = {}

    def add(self, value) -> int:
        key = json.dumps(value, sort_keys=True)
        if key not in self._index:
            self._index[key] = len(self.values)
            self.values.append(value)
        return self._index[key]


def generate_msl(compiler: Sequence[str]) -> dict:
    return MslRecorder(compiler).record()


def header_name(root: Path, path: str) -> str | None:
    try:
        relative = Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        return None
    first = relative.split("/")[0]
    if first.startswith("__"):
        return None
    return relative


def macro_delta(base: dict[str, str], other: dict[str, str]) -> dict[str, object]:
    return {
        "define": sorted(
            value for name, value in other.items() if base.get(name) != value
        ),
        "undefine": sorted(name for name in base if name not in other),
    }


# DXC -----------------------------------------------------------------------------------------------------
DXC_RELEASE = "v1.9.2602.24"
DXC_COMMIT = "d355aa8364d34df3f0822ba0de8d1dfc75ae6f48"
DXC_RAW = (
    f"https://raw.githubusercontent.com/microsoft/DirectXShaderCompiler/{DXC_COMMIT}/"
)
DXC_TREE = f"https://api.github.com/repos/microsoft/DirectXShaderCompiler/git/trees/{DXC_COMMIT}?recursive=1"
INIT_PREPROCESSOR = "tools/clang/lib/Frontend/InitPreprocessor.cpp"
INIT_PREPROCESSOR_SHA256 = (
    "56d3965b7975f9e7b55b89c760dfd15ddaff2c33bae753942087da183dddd1c4"
)
BUILTIN_HEADER_ROOT = "tools/clang/lib/Headers/hlsl/"
HLSL_STAGES = ("ps", "vs", "gs", "hs", "ds", "cs", "lib", "ms", "as")
HLSL_MODELS = ("5_0", "5_1") + tuple(f"6_{minor}" for minor in range(0, 11))
HLSL_VERSIONS = ("2015", "2016", "2017", "2018", "2021", "202x", "202y")
SPIRV_ENVIRONMENTS = (
    "vulkan1.0",
    "vulkan1.1",
    "vulkan1.1spirv1.4",
    "vulkan1.2",
    "vulkan1.3",
    "vulkan1.4",
    "universal1.5",
)
REFERENCE_PROFILE = ("cs_6_0", "2021")
# Macros that depend on the profile and options; every other predefined macro must be the same everywhere.
DYNAMIC = {
    "__SHADER_TARGET_STAGE",
    "__SHADER_TARGET_MAJOR",
    "__SHADER_TARGET_MINOR",
    "__HLSL_VERSION",
    "__HLSL_ENABLE_16_BIT",
    "__spirv__",
    "__SPIRV_MAJOR_VERSION__",
    "__SPIRV_MINOR_VERSION__",
}
ENGINE_BUILTINS = {
    "__LINE__",
    "__FILE__",
    "__COUNTER__",
    "__INCLUDE_LEVEL__",
    "__BASE_FILE__",
    "__DATE__",
    "__TIME__",
    "__TIMESTAMP__",
    "__VA_ARGS__",
    "__VA_OPT__",
} | FEATURE_TESTS


def fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "crosstl-preprocessor-macros"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


class Dxc:
    """Runs ``dxc -P`` on probe sources written to a scratch directory the command can read."""

    def __init__(self, command: Sequence[str], scratch: Path):
        self.command = list(command)
        self.scratch = scratch
        self.counter = itertools.count(1)

    def preprocess(self, source: str, args: Sequence[str]) -> tuple[str | None, str]:
        """``(output, stderr)``; output is ``None`` when DXC reports an error (``dxc -P`` exits 0 on errors)."""
        path = self.scratch / f"probe{next(self.counter)}.hlsl"
        out = path.with_suffix(".i")
        path.write_text(source)
        result = subprocess.run(
            [*self.command, *args, "-P", "-Fi", str(out), str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        text = out.read_text() if out.is_file() else None
        failed = (
            result.returncode != 0
            or re.search(r"error|failed", result.stdout + result.stderr) is not None
        )
        return (None if failed else text), result.stdout + result.stderr

    def version(self) -> str:
        result = subprocess.run(
            [*self.command, "--version"], capture_output=True, text=True, check=False
        )
        text = (result.stdout + result.stderr).strip().splitlines()
        return text[0].split(": ", 1)[-1].strip() if text else ""


def dxc_candidates(source: str) -> list[str]:
    """Every macro name ``InitPreprocessor.cpp`` can define, over-approximated."""
    names = set(re.findall(r'"(__[A-Za-z0-9_]+)"', source))
    names |= set(re.findall(r'defineMacro\(\s*"([A-Za-z_][A-Za-z0-9_]*)"', source))
    suffixes = set(re.findall(r'DefPrefix \+ "([A-Z0-9_]+)"', source))
    for prefix in ("FLT", "DBL", "LDBL", "HALF", "FLT16"):
        names |= {f"__{prefix}_{suffix}" for suffix in suffixes}
    return sorted(
        name
        for name in names - ENGINE_BUILTINS
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
    )


def values_probe(names: Sequence[str]) -> str:
    """Each defined macro's expansion, marked by its index (a name in the marker would be expanded too)."""
    return "".join(
        f"#ifdef {name}\n@@ {index} @@ {name} @@\n#endif\n"
        for index, name in enumerate(names)
    )


def body_probe(names: Sequence[str], name: str) -> str:
    """``name``'s replacement list with every other macro undefined, so nothing nested in it expands. (DXC's
    ``#pragma pop_macro`` does not restore predefined macros, so each body needs its own probe.)
    """
    undefines = "".join(f"#undef {other}\n" for other in names if other != name)
    return f"{undefines}@@ {names.index(name)} @@ {name} @@\n"


def parse_probe(text: str, names: Sequence[str]) -> dict[str, str]:
    found = {}
    for match in re.finditer(r"@@ ([0-9]+) @@(.*?)@@", text):
        found[names[int(match.group(1))]] = match.group(2).strip()
    return found


def profile_args(
    profile: str, version: str, sixteen: bool = False, spirv: str | None = None
) -> list[str]:
    args = ["-T", profile, "-HV", version]
    if sixteen:
        args.append("-enable-16bit-types")
    if spirv:
        args += ["-spirv", f"-fspv-target-env={spirv}"]
    return args


class HlslRecorder:
    def __init__(self, command: Sequence[str], scratch: Path):
        self.dxc = Dxc(command, scratch)
        source = fetch(DXC_RAW + INIT_PREPROCESSOR)
        digest = hashlib.sha256(source).hexdigest()
        if digest != INIT_PREPROCESSOR_SHA256:
            raise GenerationError(
                f"{INIT_PREPROCESSOR} has sha256 {digest}, not the pinned {INIT_PREPROCESSOR_SHA256}"
            )
        self.candidates = dxc_candidates(source.decode())
        tree = json.loads(fetch(DXC_TREE))["tree"]
        self.builtin_headers = sorted(
            item["path"][len(BUILTIN_HEADER_ROOT) :]
            for item in tree
            if item["path"].startswith(BUILTIN_HEADER_ROOT)
            and item["path"].endswith(".h")
        )

    def values(self, args: Sequence[str]) -> dict[str, str] | None:
        text, _ = self.dxc.preprocess(values_probe(self.candidates), args)
        return None if text is None else parse_probe(text, self.candidates)

    def record(self) -> dict:
        version = self.dxc.version()
        reference = self.values(profile_args(*REFERENCE_PROFILE))
        if reference is None:
            raise GenerationError("DXC rejects the reference profile")
        names = sorted(reference)
        jobs = {}
        with concurrent.futures.ThreadPoolExecutor(4) as executor:
            body_jobs = {
                name: executor.submit(
                    self.dxc.preprocess,
                    body_probe(names, name),
                    profile_args(*REFERENCE_PROFILE),
                )
                for name in names
            }
            bodies = {}
            for name, future in body_jobs.items():
                text, errors = future.result()
                if text is None:
                    raise GenerationError(
                        f"the body probe of {name} failed: {errors[-400:]}"
                    )
                bodies[name] = parse_probe(text, names)[name]
                if bodies[name] == name:
                    raise GenerationError(
                        f"{name} is function-like or self-referential; record it by hand"
                    )
            for stage in HLSL_STAGES:
                for model in HLSL_MODELS:
                    jobs[("profile", stage, model)] = executor.submit(
                        self.values, profile_args(f"{stage}_{model}", "2021")
                    )
            for hlsl_version in HLSL_VERSIONS:
                jobs[("version", hlsl_version)] = executor.submit(
                    self.values, profile_args("cs_6_0", hlsl_version)
                )
                for minor in range(0, 10):
                    jobs[("16bit", hlsl_version, minor)] = executor.submit(
                        self.values,
                        profile_args(f"cs_6_{minor}", hlsl_version, sixteen=True),
                    )
            for environment in SPIRV_ENVIRONMENTS:
                jobs[("spirv", environment)] = executor.submit(
                    self.values, profile_args("cs_6_0", "2021", spirv=environment)
                )
            results = {key: future.result() for key, future in jobs.items()}
        for key, probed in results.items():
            if probed is not None:
                changed = {
                    name
                    for name in set(reference) | set(probed)
                    if reference.get(name) != probed.get(name)
                }
                if changed - DYNAMIC:
                    raise GenerationError(f"{key} changes {sorted(changed - DYNAMIC)}")
        static = {
            name: f"{name} {bodies[name]}" for name in names if name not in DYNAMIC
        }
        pool = Pool()
        pool.add("predefined", static, None)
        encoded = pool.encode()
        return {
            "schema": SCHEMA,
            "source": {
                "by": "tools/preprocessor_macros.py hlsl",
                "toolchain": version,
                "release": DXC_RELEASE,
                "commit": DXC_COMMIT,
                "initPreprocessorSha256": INIT_PREPROCESSOR_SHA256,
                "candidates": len(self.candidates),
            },
            **self.tables(reference, results),
            "builtinHeaders": self.builtin_headers,
            "definitions": encoded["definitions"],
            "sets": encoded["sets"],
        }

    def tables(self, reference, results) -> dict:
        stage_kinds = {
            int(reference[name])
            for name in reference
            if name.startswith("__SHADER_STAGE_") and reference[name].isdigit()
        }
        stages, models = {}, {}
        for stage in HLSL_STAGES:
            for model in HLSL_MODELS:
                probed = results[("profile", stage, model)]
                if (
                    probed is None
                    or int(probed["__SHADER_TARGET_STAGE"]) not in stage_kinds
                ):
                    continue
                major, minor = model.split("_")
                if (
                    probed["__SHADER_TARGET_MAJOR"],
                    probed["__SHADER_TARGET_MINOR"],
                ) != (major, minor):
                    raise GenerationError(
                        f"{stage}_{model} reports another shader model"
                    )
                kind = int(probed["__SHADER_TARGET_STAGE"])
                if stages.setdefault(stage, kind) != kind:
                    raise GenerationError(f"{stage} profiles report different stages")
                models.setdefault(stage, []).append(model)
        versions = {
            hlsl_version: int(results[("version", hlsl_version)]["__HLSL_VERSION"])
            for hlsl_version in HLSL_VERSIONS
            if results[("version", hlsl_version)] is not None
        }
        accepted = sorted(
            (versions[hlsl_version], minor)
            for (kind, hlsl_version, minor), probed in (
                (key, value) for key, value in results.items() if key[0] == "16bit"
            )
            if probed is not None
            and hlsl_version in versions
            and probed.get("__HLSL_ENABLE_16_BIT") == "1"
        )
        minimum_version = min(version for version, _ in accepted)
        minimum_minor = min(minor for _, minor in accepted)
        expected = sorted(
            (value, minor)
            for value in versions.values()
            for minor in range(0, 10)
            if value >= minimum_version and minor >= minimum_minor
        )
        if accepted != expected:
            raise GenerationError(
                "-enable-16bit-types is not accepted exactly from one shader model and -HV on"
            )
        spirv = {}
        for environment in SPIRV_ENVIRONMENTS:
            probed = results[("spirv", environment)]
            if probed is None:
                continue
            if probed.get("__spirv__") != "1":
                raise GenerationError(
                    f"-spirv does not define __spirv__ as 1 for {environment}"
                )
            spirv[environment] = [
                int(probed["__SPIRV_MAJOR_VERSION__"]),
                int(probed["__SPIRV_MINOR_VERSION__"]),
            ]
        return {
            "stages": stages,
            "models": models,
            "hlslVersions": versions,
            "sixteenBit": {
                "minimumModel": [6, minimum_minor],
                "minimumVersion": minimum_version,
            },
            "spirvEnvironments": spirv,
        }


def generate_hlsl(command: Sequence[str], scratch: Path | None = None) -> dict:
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        return HlslRecorder(command, Path(directory).resolve()).record()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dialect", choices=("msl", "hlsl"))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--metal", default="xcrun metal", help="Metal compiler command")
    parser.add_argument(
        "--dxc",
        default="dxc",
        help="DXC command; '{scratch}' is replaced by the scratch directory (for a container's volume mount)",
    )
    parser.add_argument(
        "--scratch",
        type=Path,
        help="directory for DXC probe files (default: the system temp)",
    )
    args = parser.parse_args(argv)
    if args.dialect == "msl":
        return finish("msl", generate_msl(shlex.split(args.metal)), args.write)
    scratch = (args.scratch or Path(tempfile.gettempdir())).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    command = shlex.split(args.dxc.replace("{scratch}", str(scratch)))
    return finish("hlsl", generate_hlsl(command, scratch), args.write)


if __name__ == "__main__":
    sys.exit(main())
