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
    # Mojo's standard library at the release CI installs (mojo 1.1.0, tagged mojo/v1.1.0 and max/v26.6.0).
    "mojo-bit": Source(
        "modular/modular",
        "c6fa49f712fe15c99d0461ad2baaf2a15cbc58ce",
        "Mojo/stdlib/std/bit/bit.mojo",
        "a449dab898731d371e00a3373034de7bb3210de1cb4d7c01289a10538c2685ed",
        "Mojo 1.1.0 std.bit functions",
    ),
    "mojo-bit-exports": Source(
        "modular/modular",
        "c6fa49f712fe15c99d0461ad2baaf2a15cbc58ce",
        "Mojo/stdlib/std/bit/__init__.mojo",
        "0cabf5f5474056e435ac69082c42556ed049e8130c94f6a33e91e81cc7b8d5e8",
        "Mojo 1.1.0 std.bit public API",
    ),
    "mojo-math": Source(
        "modular/modular",
        "c6fa49f712fe15c99d0461ad2baaf2a15cbc58ce",
        "Mojo/stdlib/std/math/math.mojo",
        "765c598e516790d1814df7a40b19a1ecb01f748719755becc669620d7722a314",
        "Mojo 1.1.0 std.math functions",
    ),
    "mojo-math-exports": Source(
        "modular/modular",
        "c6fa49f712fe15c99d0461ad2baaf2a15cbc58ce",
        "Mojo/stdlib/std/math/__init__.mojo",
        "54ad6d83767414e9e2f9a049297cdb9ed37507d0c972f4e056984edef18f0b1d",
        "Mojo 1.1.0 std.math public API",
    ),
    "mojo-numerics": Source(
        "modular/modular",
        "c6fa49f712fe15c99d0461ad2baaf2a15cbc58ce",
        "Mojo/stdlib/std/utils/numerics.mojo",
        "d2978a288ad44a3ffc87fa6d4cb4f28564da1ec1fed2f4f31ecd11450d148e20",
        "Mojo 1.1.0 floating-point predicates std.math re-exports",
    ),
    # OpenCL C is pinned to LLVM 18.1.3, the clang-18 of the Ubuntu 24.04 runner image that validates the probes.
    "opencl": Source(
        "llvm/llvm-project",
        "c13b7485b87909fcf739f62cfa382b55407433c0",
        "clang/lib/Headers/opencl-c.h",
        "92425955d27a4f703fc88671c4ae6722fcdbf77d61c3b0f1df13c4310bec6b37",
        "LLVM 18.1.3 OpenCL C built-in function declarations",
    ),
    # rust-gpu v0.11.0, the release CI builds the probes with, and the num-traits and glam versions its lock file
    # resolves for spirv-std.
    "rust-arch": Source(
        "Rust-GPU/rust-gpu",
        "62401280ab4b512c734ad1aceee7553385e17418",
        "crates/spirv-std/src/arch.rs",
        "80f1ccbc43e66fc4cf61ce3868ed5de479805183e38be44a50766663f24acd50",
        "rust-gpu 0.11.0 spirv_std::arch",
    ),
    "rust-atomics": Source(
        "Rust-GPU/rust-gpu",
        "62401280ab4b512c734ad1aceee7553385e17418",
        "crates/spirv-std/src/arch/atomics.rs",
        "8801890a3ed72af6199661310086216e29cd184ed5f1160f7d7006ec00bf05c3",
        "rust-gpu 0.11.0 spirv_std::arch, module atomics",
    ),
    "rust-barrier": Source(
        "Rust-GPU/rust-gpu",
        "62401280ab4b512c734ad1aceee7553385e17418",
        "crates/spirv-std/src/arch/barrier.rs",
        "fdf0eea2a714080971a83460d38d0a2d961d2f4a9d5914158e4b627e3a2cdf56",
        "rust-gpu 0.11.0 spirv_std::arch, module barrier",
    ),
    "rust-demote_to_helper_invocation_ext": Source(
        "Rust-GPU/rust-gpu",
        "62401280ab4b512c734ad1aceee7553385e17418",
        "crates/spirv-std/src/arch/demote_to_helper_invocation_ext.rs",
        "65556003258bcb4ecbb3a5bbdd0367930591ab9b05b970c2cbcb27b568777d39",
        "rust-gpu 0.11.0 spirv_std::arch, module demote_to_helper_invocation_ext",
    ),
    "rust-derivative": Source(
        "Rust-GPU/rust-gpu",
        "62401280ab4b512c734ad1aceee7553385e17418",
        "crates/spirv-std/src/arch/derivative.rs",
        "feba79db9af20bcf4d330e13490209f1c5d26a8566bedc80407ae8959381b6a9",
        "rust-gpu 0.11.0 spirv_std::arch, module derivative",
    ),
    "rust-mesh_shading": Source(
        "Rust-GPU/rust-gpu",
        "62401280ab4b512c734ad1aceee7553385e17418",
        "crates/spirv-std/src/arch/mesh_shading.rs",
        "575d603529d6132824d7bb9785e680c6e226da69f9c673f5007bc72a2c8cd8e4",
        "rust-gpu 0.11.0 spirv_std::arch, module mesh_shading",
    ),
    "rust-primitive": Source(
        "Rust-GPU/rust-gpu",
        "62401280ab4b512c734ad1aceee7553385e17418",
        "crates/spirv-std/src/arch/primitive.rs",
        "f03aa30086f74c807514a656f156e6bb11c1092b7cb016b6758e7231eebb39d4",
        "rust-gpu 0.11.0 spirv_std::arch, module primitive",
    ),
    "rust-ray_tracing": Source(
        "Rust-GPU/rust-gpu",
        "62401280ab4b512c734ad1aceee7553385e17418",
        "crates/spirv-std/src/arch/ray_tracing.rs",
        "931e76c2f9b949161cebdfa88700931b4aa826f1ae4bcd1bcad49c2b14dc3231",
        "rust-gpu 0.11.0 spirv_std::arch, module ray_tracing",
    ),
    "rust-subgroup": Source(
        "Rust-GPU/rust-gpu",
        "62401280ab4b512c734ad1aceee7553385e17418",
        "crates/spirv-std/src/arch/subgroup.rs",
        "18d92eb06353bf9080c1ae6654777ff22988f5a0d8ba80e951fe257ece749063",
        "rust-gpu 0.11.0 spirv_std::arch, module subgroup",
    ),
    "rust-float": Source(
        "rust-num/num-traits",
        "7ec3d41d39b28190ec1d42db38021107b3951f3a",
        "src/float.rs",
        "2caf20ca3227422dcd8981accc79898e889bbda97fce1fc7f881c98e7091d8c3",
        "num-traits 0.2.19 Float trait",
    ),
    "rust-vec3": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/f32/vec3.rs",
        "e0b239b6c6ab095f2607d9668aac2018dad1829c3019a91b63eb6a4e57b7323b",
        "glam 0.34.1 Vec3",
    ),
    # Slang's standard library is the core module slangc is built from, pinned to the release CI installs.
    "slang": Source(
        "shader-slang/slang",
        "4685b35be78104ae007013cb749a1e906c414600",
        "source/slang/hlsl.meta.slang",
        "419f659ed73658539e6d962640c51cc748a7abefe1b76d37bacc6d3c38c470b0",
        "Slang v2026.9.1 core module: HLSL-compatible built-in functions",
    ),
    "slang-core": Source(
        "shader-slang/slang",
        "4685b35be78104ae007013cb749a1e906c414600",
        "source/slang/core.meta.slang",
        "983a544e6e666b02dc8966bf557363966bd6d7abb82a9177707cef6ebf04b357",
        "Slang v2026.9.1 core module: core built-in functions",
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
    "webgl": Source(
        "google/angle",
        "1ff891807e391d842c33d552ea26ab26b5341ef4",
        "src/compiler/translator/builtin_function_declarations.txt",
        "d644775414f12a0cf61d008c06a6fcac9f29fdcd138ea9334acfd3929fce91ba",
        "ANGLE's GLSL ES built-in function table",
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
