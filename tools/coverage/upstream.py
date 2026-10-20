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
    "cuda": Source(
        "llvm/llvm-project",
        "0d261d1ca552c95a8f007e061c787ac7132fbcbc",
        "clang/lib/Headers/__clang_cuda_math.h",
        "debe73cf7191627b94fea9e1bfe13a4318b1616fed38b2e94f5eefe2a8637c72",
        "LLVM 23.1.3 device-side CUDA Math API wrappers",
    ),
    "cuda-device": Source(
        "llvm/llvm-project",
        "0d261d1ca552c95a8f007e061c787ac7132fbcbc",
        "clang/lib/Headers/__clang_cuda_device_functions.h",
        "216f56ec0925602aaf8cab05bf226d0efcc531083993cc664d025d1737b3469a",
        "LLVM 23.1.3 CUDA intrinsic and atomic wrappers",
    ),
    "glsl": Source(
        "KhronosGroup/GLSL",
        "0099bf83b0288469917fc2e2972cb0a2b74d6711",
        "chapters/builtinfunctions.adoc",
        "95c602702118f5aabba3a5e8569abfca9865d4c86e02a210021eed16b5dd6bc5",
        "GLSL 4.60 specification, chapter 8 (built-in functions)",
    ),
    # HIP is pinned to the versions Ubuntu 24.04 packages (hipcc 5.7.1 on clang 17.0.6), the compiler the probes are
    # validated with, so the list names exactly what that compiler's headers define.
    "hip": Source(
        "llvm/llvm-project",
        "6009708b4367171ccdbf4b5905cb6a803753fe18",
        "clang/lib/Headers/__clang_hip_math.h",
        "218b1f09b2327d7bbd11534a4a6bba93ea990d1cc218177dbb36e2ff74b362ac",
        "LLVM 17.0.6 device-side HIP math functions and intrinsics",
    ),
    "hip-atomic": Source(
        "ROCm/clr",
        "1949b1621a802ffb1492616adbae6154bfbe64ef",
        "hipamd/include/hip/amd_detail/amd_hip_atomic.h",
        "44e62e2adac1ccd40bfdf007b91131f125cf572b908b7f9a770a1d502008262d",
        "ROCm 5.7.1 HIP atomic functions",
    ),
    "hip-device": Source(
        "ROCm/clr",
        "1949b1621a802ffb1492616adbae6154bfbe64ef",
        "hipamd/include/hip/amd_detail/amd_device_functions.h",
        "1cc905b32475f129990c64892514a822fb18fafa3e5b6f2a1ddb5eb093e3453c",
        "ROCm 5.7.1 HIP device functions (integer intrinsics, conversions, barriers)",
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
    "spirv-core": Source(
        "KhronosGroup/SPIRV-Headers",
        "496543121ce6419f23d6fa5d7194ba66c36212d2",
        "include/spirv/unified1/spirv.core.grammar.json",
        "c125a0fbf6730fde4625fef2618c4f7611ab42edf9b2ad5bb9291002b10bfad3",
        "SPIR-V core grammar (execution models, capabilities, storage classes, instructions)",
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
