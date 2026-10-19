"""Upstream feature sources, pinned by commit and checked by digest.

Moving a pin changes the feature lists, so update ``commit`` and ``sha256`` together and say so in the pull request.
Downloads are cached under ``CROSSTL_COVERAGE_CACHE`` (default ``~/.cache/crosstl-coverage``).
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    repository: str
    commit: str
    path: str
    sha256: str
    describes: str

    @property
    def url(self) -> str:
        return f"https://raw.githubusercontent.com/{self.repository}/{self.commit}/{self.path}"

    @property
    def label(self) -> str:
        return f"{self.repository}@{self.commit[:12]}:{self.path}"


SOURCES = {
    "glsl": Source(
        "KhronosGroup/GLSL",
        "0099bf83b0288469917fc2e2972cb0a2b74d6711",
        "chapters/builtinfunctions.adoc",
        "95c602702118f5aabba3a5e8569abfca9865d4c86e02a210021eed16b5dd6bc5",
        "GLSL 4.60 specification, chapter 8 (built-in functions)",
    ),
    "hlsl": Source(
        "microsoft/DirectXShaderCompiler",
        "d355aa8364d34df3f0822ba0de8d1dfc75ae6f48",
        "utils/hct/gen_intrin_main.txt",
        "24bc62cd69825c062f6527b783d2089c669c643b2df9336a6b5c7895a0ad7eec",
        "DXC v1.9.2602.24 intrinsic table",
    ),
    "spirv": Source(
        "KhronosGroup/SPIRV-Headers",
        "496543121ce6419f23d6fa5d7194ba66c36212d2",
        "include/spirv/unified1/extinst.glsl.std.450.grammar.json",
        "5d72e48247569f9b77a17a849db624350abc9d26f7f72b153bbcf15c37238239",
        "SPIR-V GLSL.std.450 extended instruction grammar",
    ),
    "wgsl": Source(
        "google/dawn",
        "ff7c6960f62d4249f967a78448918eec21705a86",
        "src/tint/lang/wgsl/wgsl.def",
        "c8721d70259c8195d7366a8d7feb9c076f05a13de2f566db66e2213ffc14a06e",
        "Tint's WGSL built-in function definitions",
    ),
}


class UpstreamError(RuntimeError):
    pass


def cache_dir() -> Path:
    root = os.environ.get("CROSSTL_COVERAGE_CACHE")
    return Path(root) if root else Path.home() / ".cache" / "crosstl-coverage"


def fetch(name: str, *, offline: bool = False) -> str:
    """Return the text of one pinned source, downloading it on first use."""
    source = SOURCES[name]
    path = cache_dir() / "upstream" / source.commit / source.path
    if path.is_file():
        data = path.read_bytes()
    elif offline:
        raise UpstreamError(f"{source.label} is not cached and --offline was given")
    else:
        try:
            with urllib.request.urlopen(source.url, timeout=60) as response:
                data = response.read()
        except OSError as error:
            raise UpstreamError(f"cannot download {source.url}: {error}") from error
    digest = hashlib.sha256(data).hexdigest()
    if digest != source.sha256:
        raise UpstreamError(
            f"{source.label} has sha256 {digest}, expected {source.sha256}"
        )
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data.decode("utf-8")
