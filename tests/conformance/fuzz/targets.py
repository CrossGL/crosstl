"""The importers under fuzz: how each one's inputs are generated and how a case becomes a worker job.

Each source language CrossTL imports gets three kinds of input, drawn by hypothesis:

- a well-formed program from :mod:`.program`, printed in the language (``valid``);
- that program after token mutations from :mod:`.mutate` (``mutant``);
- for languages with no printer, a mutated seed file from the repository's own test fixtures (``seed``).

The preprocessor is fuzzed directly with the units of :mod:`.preprocessor`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from hypothesis import strategies as st

from . import mutate, preprocessor
from .outcomes import REPOSITORY_ROOT
from .printers import PRINTERS, GLSLPrinter
from .program import Config, programs
from .spirv import SpirvPrinter

GLSL_VERSIONS = ("450", "430", "460", "310 es", "320 es")
CORE_BACKENDS = ("metal", "directx", "opengl", "vulkan")


@dataclass(frozen=True)
class Importer:
    name: str
    extension: str
    printer: Callable | None = None
    config: Config = Config()
    seeds: tuple[str, ...] = ()  # glob patterns relative to the repository root
    backends: tuple[str, ...] = ("cgl",)

    def seed_files(self):
        root = Path(REPOSITORY_ROOT)
        found = set()
        for pattern in self.seeds:
            found.update(path for path in root.glob(pattern) if path.is_file())
        return sorted(str(path.relative_to(root)) for path in found)

    def printed(self):
        if self.name == "opengl":
            versions = st.sampled_from(GLSL_VERSIONS)
            return st.tuples(versions, programs(self.config)).map(
                lambda pair: GLSLPrinter(pair[0]).print(pair[1])
            )
        return programs(self.config).map(lambda program: self.printer().print(program))

    def strategy(self):
        """Cases as ``(kind, extension, text)``; the kind separates valid inputs from mutants in statistics."""
        sources = []
        if self.printer is not None:
            valid = self.printed()
            sources += [
                valid.map(lambda text: ("valid", self.extension, text)),
                mutate.mutated(valid, self.name).map(
                    lambda text: ("mutant", self.extension, text)
                ),
            ]
        seeds = self.seed_files()
        if seeds:
            sources.append(st.sampled_from(seeds).flatmap(self._mutated_seed))
        return st.one_of(*sources)

    def _mutated_seed(self, path):
        text = (Path(REPOSITORY_ROOT) / path).read_text(
            encoding="utf-8", errors="replace"
        )
        return mutate.mutated(st.just(text), self.name).map(
            lambda mutant: ("seed", Path(path).suffix, mutant)
        )

    def job(self, case, backend):
        kind, extension, text = case
        return {
            "kind": "translate",
            "importer": self.name,
            "extension": extension,
            "backend": backend,
            "source": text,
            "input": kind,
        }


class PreprocessorTarget:
    name = "preprocessor"
    extension = ".pp.json"
    backends = ("preprocess",)

    def strategy(self):
        return preprocessor.units().map(
            lambda unit: ("valid", self.extension, unit.to_json())
        )

    def job(self, case, backend):
        kind, _, unit = case
        return {
            "kind": "preprocess",
            "importer": self.name,
            "backend": backend,
            "unit": unit,
            "input": kind,
        }


def _fixtures(*extensions):
    """The repository's own test inputs and examples with these extensions."""
    return tuple(
        f"{root}/**/*{extension}"
        for root in ("tests", "examples")
        for extension in extensions
    )


SCALAR = Config(vectors=False)

IMPORTERS = {
    "metal": Importer("metal", ".metal", PRINTERS["metal"], seeds=_fixtures(".metal")),
    "directx": Importer(
        "directx", ".hlsl", PRINTERS["directx"], seeds=_fixtures(".hlsl")
    ),
    "opengl": Importer(
        "opengl",
        ".comp",
        PRINTERS["opengl"],
        seeds=_fixtures(".comp", ".glsl", ".vert", ".frag"),
    ),
    "vulkan": Importer("vulkan", ".spvasm", SpirvPrinter, seeds=_fixtures(".spvasm")),
    "cgl": Importer(
        "cgl", ".cgl", PRINTERS["cgl"], seeds=_fixtures(".cgl"), backends=CORE_BACKENDS
    ),
    "slang": Importer("slang", ".slang", PRINTERS["slang"], seeds=_fixtures(".slang")),
    "cuda": Importer("cuda", ".cu", PRINTERS["cuda"], SCALAR, seeds=_fixtures(".cu")),
    "hip": Importer("hip", ".hip", PRINTERS["hip"], SCALAR, seeds=_fixtures(".hip")),
    "opencl": Importer(
        "opencl", ".cl", PRINTERS["opencl"], SCALAR, seeds=_fixtures(".cl")
    ),
    "mojo": Importer("mojo", ".mojo", seeds=_fixtures(".mojo")),
    "rust": Importer("rust", ".rs", seeds=_fixtures(".rs")),
    "preprocessor": PreprocessorTarget(),
}


def case_text(case) -> str:
    """The text a reproducer file holds for ``case``."""
    payload = case[2]
    if isinstance(payload, dict):
        return json.dumps(payload, indent=1, sort_keys=True) + "\n"
    return payload
