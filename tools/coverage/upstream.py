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
    "cuda-intrinsics": Source(
        "llvm/llvm-project",
        "0d261d1ca552c95a8f007e061c787ac7132fbcbc",
        "clang/lib/Headers/__clang_cuda_intrinsics.h",
        "399ffff6ad1c738839aaf5b0b0e189f52a30a35f8c9a64c82b5f5d1d1c2dd7f5",
        "LLVM 23.1.3 CUDA warp, cache-hinted load and store, and funnel shift intrinsics",
    ),
    "glsl": Source(
        "KhronosGroup/GLSL",
        "0099bf83b0288469917fc2e2972cb0a2b74d6711",
        "chapters/builtinfunctions.adoc",
        "95c602702118f5aabba3a5e8569abfca9865d4c86e02a210021eed16b5dd6bc5",
        "GLSL 4.60 specification, chapter 8 (built-in functions)",
    ),
    "glsl-subgroup": Source(
        "KhronosGroup/GLSL",
        "0099bf83b0288469917fc2e2972cb0a2b74d6711",
        "extensions/khr/GL_KHR_shader_subgroup.txt",
        "fc653cfd7e9392e3a42824ebddb7edbd19678fae974b0a925969d4c4a191d2d8",
        "GL_KHR_shader_subgroup (section 8.18, shader invocation group functions)",
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
    "hip-warp": Source(
        "ROCm/clr",
        "1949b1621a802ffb1492616adbae6154bfbe64ef",
        "hipamd/include/hip/amd_detail/amd_warp_functions.h",
        "37df4b765b0546538b07fbaeb3142145273e80326c37f046b64476f1e9751bbd",
        "ROCm 5.7.1 HIP warp shuffles and lane permutes",
    ),
    "hlsl": Source(
        "microsoft/DirectXShaderCompiler",
        "d355aa8364d34df3f0822ba0de8d1dfc75ae6f48",
        "utils/hct/gen_intrin_main.txt",
        "24bc62cd69825c062f6527b783d2089c669c643b2df9336a6b5c7895a0ad7eec",
        "DXC v1.9.2602.24 intrinsic table",
    ),
    # Mojo's standard library at the release CI installs (mojo 1.2.0, tagged mojo/v1.2.0 and max/v26.7.0).
    "mojo-atomic": Source(
        "modular/modular",
        "527d069ee11a41ac5320af1f9402db0a66b2b868",
        "Mojo/stdlib/std/atomic/atomic.mojo",
        "33e7ae26d1407df2234bcf156da75d9106d49eb842bdbb8476961644c40bbbae",
        "Mojo 1.2.0 std.atomic functions and Atomic methods",
    ),
    "mojo-atomic-exports": Source(
        "modular/modular",
        "527d069ee11a41ac5320af1f9402db0a66b2b868",
        "Mojo/stdlib/std/atomic/__init__.mojo",
        "0931118d39830baed30e88d7569a7b8b94c9b73a250b905c746ac9b1e969da88",
        "Mojo 1.2.0 std.atomic public API",
    ),
    "mojo-bit": Source(
        "modular/modular",
        "527d069ee11a41ac5320af1f9402db0a66b2b868",
        "Mojo/stdlib/std/bit/bit.mojo",
        "c794f2c70b78003bf18f375de99af439f1c6775b691f757ac978cf1d5ea629dc",
        "Mojo 1.2.0 std.bit functions",
    ),
    "mojo-bit-exports": Source(
        "modular/modular",
        "527d069ee11a41ac5320af1f9402db0a66b2b868",
        "Mojo/stdlib/std/bit/__init__.mojo",
        "0cabf5f5474056e435ac69082c42556ed049e8130c94f6a33e91e81cc7b8d5e8",
        "Mojo 1.2.0 std.bit public API",
    ),
    # MAX's GPU package at the same tag, as max-mojo-libs 26.7.0 installs it.
    "mojo-gpu-block": Source(
        "modular/modular",
        "527d069ee11a41ac5320af1f9402db0a66b2b868",
        "max/mojo/max/gpu/primitives/block.mojo",
        "303b46b1ab87ff6dc3e3bfa09a452e3248934bf238bcb281698451e208a47a79",
        "MAX 26.7.0 max.gpu.primitives.block functions",
    ),
    "mojo-gpu-id": Source(
        "modular/modular",
        "527d069ee11a41ac5320af1f9402db0a66b2b868",
        "max/mojo/max/gpu/primitives/id.mojo",
        "44256257727011951760602c866a8b3a0bcd1c0fe6c2a513f4efd64eec9b1f75",
        "MAX 26.7.0 max.gpu.primitives.id functions",
    ),
    "mojo-gpu-lanes": Source(
        "modular/modular",
        "527d069ee11a41ac5320af1f9402db0a66b2b868",
        "Mojo/stdlib/std/_gpu/primitives/id.mojo",
        "b3b415a9328be5ec176092001297123eae6756e44ed604748298c8c294da63a9",
        "Mojo 1.2.0 GPU index functions max.gpu re-exports",
    ),
    "mojo-gpu-sync": Source(
        "modular/modular",
        "527d069ee11a41ac5320af1f9402db0a66b2b868",
        "max/mojo/max/gpu/sync/sync.mojo",
        "54a2271e7cae969490fd3fa49f46c64fbd5a7a44c0d069ece7765dd861f46686",
        "MAX 26.7.0 max.gpu synchronization functions",
    ),
    "mojo-gpu-sync-exports": Source(
        "modular/modular",
        "527d069ee11a41ac5320af1f9402db0a66b2b868",
        "max/mojo/max/gpu/sync/__init__.mojo",
        "708174965f4d43d3958b44477ead7952702d692cb161d751c8ce5c00533d476e",
        "MAX 26.7.0 max.gpu.sync public API",
    ),
    "mojo-gpu-warp": Source(
        "modular/modular",
        "527d069ee11a41ac5320af1f9402db0a66b2b868",
        "max/mojo/max/gpu/primitives/warp.mojo",
        "9d8c07709bf313a81b11ec4568d0aa6ef4a5168853b690379690e359ec79791f",
        "MAX 26.7.0 max.gpu.primitives.warp functions",
    ),
    "mojo-math": Source(
        "modular/modular",
        "527d069ee11a41ac5320af1f9402db0a66b2b868",
        "Mojo/stdlib/std/math/math.mojo",
        "e91a2b68a59a59e5c651a96df0c4a57d40d4edcaa3ddb914ee105bd1bf8c95e4",
        "Mojo 1.2.0 std.math functions",
    ),
    "mojo-math-exports": Source(
        "modular/modular",
        "527d069ee11a41ac5320af1f9402db0a66b2b868",
        "Mojo/stdlib/std/math/__init__.mojo",
        "f9d964682e0a3e4e4be7b2d278c948972c7ca8323d6b9d5090e075631fec0201",
        "Mojo 1.2.0 std.math public API",
    ),
    "mojo-numerics": Source(
        "modular/modular",
        "527d069ee11a41ac5320af1f9402db0a66b2b868",
        "Mojo/stdlib/std/utils/numerics.mojo",
        "463caac0f3caba3f2441b33c6a5e15ba2bc3daaf9cf134a9af5ebb09ac17c07f",
        "Mojo 1.2.0 floating-point predicates std.math re-exports",
    ),
    # OpenCL C is pinned to LLVM 18.1.3, the clang-18 of the Ubuntu 24.04 runner image that validates the probes.
    "opencl": Source(
        "llvm/llvm-project",
        "c13b7485b87909fcf739f62cfa382b55407433c0",
        "clang/lib/Headers/opencl-c.h",
        "92425955d27a4f703fc88671c4ae6722fcdbf77d61c3b0f1df13c4310bec6b37",
        "LLVM 18.1.3 OpenCL C built-in function declarations",
    ),
    "opencl-base": Source(
        "llvm/llvm-project",
        "c13b7485b87909fcf739f62cfa382b55407433c0",
        "clang/lib/Headers/opencl-c-base.h",
        "a27cb3b58c585ebf103ef3055f1da8739d586ce1b7ff1cfc16459b0eadc9a760",
        "LLVM 18.1.3 OpenCL C as_type operators and printf",
    ),
    "opencl-builtins": Source(
        "llvm/llvm-project",
        "c13b7485b87909fcf739f62cfa382b55407433c0",
        "clang/include/clang/Basic/Builtins.def",
        "8b10e3289fee9e585d33ee61e12007a9a60a3c71890527a2d7235c27810a7002",
        "LLVM 18.1.3 OpenCL C pipe, enqueue and address-space built-ins",
    ),
    "opencl-keywords": Source(
        "llvm/llvm-project",
        "c13b7485b87909fcf739f62cfa382b55407433c0",
        "clang/include/clang/Basic/TokenKinds.def",
        "c0aa3d79961ac2484d2ccba6abd6ca95483e65c537619daa0b256174db5e5bff",
        "LLVM 18.1.3 OpenCL C keyword operators",
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
    "rust-packing": Source(
        "Rust-GPU/rust-gpu",
        "62401280ab4b512c734ad1aceee7553385e17418",
        "crates/spirv-std/src/float.rs",
        "036c6d8def3fa645e88650f206e57a39b33ed5f6e66bc76bd47ebd10619b09b0",
        "rust-gpu 0.11.0 spirv_std::float packing functions",
    ),
    "rust-image": Source(
        "Rust-GPU/rust-gpu",
        "62401280ab4b512c734ad1aceee7553385e17418",
        "crates/spirv-std/src/image.rs",
        "3f2404046458f4cb85f0ff0b04f325daf310acfc418c5d8aaefef18ac2256b7e",
        "rust-gpu 0.11.0 spirv_std::image Image and SampledImage methods",
    ),
    "rust-byte-buffer": Source(
        "Rust-GPU/rust-gpu",
        "62401280ab4b512c734ad1aceee7553385e17418",
        "crates/spirv-std/src/byte_addressable_buffer.rs",
        "b6d37e572839a57753c6a7ebcf59514f69247b1111d6e396f05b40f4e7124ea5",
        "rust-gpu 0.11.0 spirv_std::ByteAddressableBuffer methods",
    ),
    "rust-ray-query": Source(
        "Rust-GPU/rust-gpu",
        "62401280ab4b512c734ad1aceee7553385e17418",
        "crates/spirv-std/src/ray_tracing.rs",
        "ecfe33226a34b62005ec160623cf396a93d75adb8169cebcd0133b1844a259a2",
        "rust-gpu 0.11.0 spirv_std::ray_tracing RayQuery and AccelerationStructure methods",
    ),
    "rust-runtime-array": Source(
        "Rust-GPU/rust-gpu",
        "62401280ab4b512c734ad1aceee7553385e17418",
        "crates/spirv-std/src/runtime_array.rs",
        "bf04833fbb32af3e0e888665cf6cca0e03bfb1ed80a742231801ab3cc90b4c38",
        "rust-gpu 0.11.0 spirv_std::RuntimeArray methods",
    ),
    "rust-macros": Source(
        "Rust-GPU/rust-gpu",
        "62401280ab4b512c734ad1aceee7553385e17418",
        "crates/spirv-std/macros/src/lib.rs",
        "7ba4af405c9ea061842fdbd41f295a588edec455a6294b07699480c739569f2c",
        "rust-gpu 0.11.0 spirv_std debug_printf macros",
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
    "rust-vec2": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/f32/vec2.rs",
        "4a05b0de47e517f3daf4e067ede5fc0fb7b1599c0be68a81c58707796eae2338",
        "glam 0.34.1 Vec2",
    ),
    "rust-vec4": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/f32/scalar/vec4.rs",
        "54939f281628e3b7202d30d8915f6e781eada64be47dda0a51d50389a3013db1",
        "glam 0.34.1 Vec4",
    ),
    "rust-vec3a": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/f32/scalar/vec3a.rs",
        "b97d602fb61d749d3830d59e7f7bf1fcad936aecdf17e598ea8339c4ed653d40",
        "glam 0.34.1 Vec3A",
    ),
    "rust-ivec2": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/i32/ivec2.rs",
        "e1e44bbd0efa6d682813c46b35faedd540b4b82898f968cc1aecf7fc507b7d78",
        "glam 0.34.1 IVec2",
    ),
    "rust-ivec3": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/i32/ivec3.rs",
        "ad39c81a3480f198fdb2f9425586de42e19f499ac253f087eb2c99ec165271e8",
        "glam 0.34.1 IVec3",
    ),
    "rust-ivec4": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/i32/ivec4.rs",
        "2341141d0683d8114ed3b7a34657d89ae51d408f99cbfd2272372758d4895826",
        "glam 0.34.1 IVec4",
    ),
    "rust-uvec2": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/u32/uvec2.rs",
        "624b4f05ae253d08eee3c969a7c190ec3d7901073fad3978df161071eaf4c46b",
        "glam 0.34.1 UVec2",
    ),
    "rust-uvec3": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/u32/uvec3.rs",
        "e108882c55d7e9214c8e3fb2f49b76242f0e28217b23b30d71e2f028c8b90a48",
        "glam 0.34.1 UVec3",
    ),
    "rust-uvec4": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/u32/uvec4.rs",
        "ace6ce0010b6e34fb2d0f879d32ebfcbd5368da4cfa653b9debcbb98280c0331",
        "glam 0.34.1 UVec4",
    ),
    "rust-dvec2": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/f64/dvec2.rs",
        "f1bfb137d926db7a29e61754a17c459657cdf9c7234c6bd6bdcd4d6740d1b451",
        "glam 0.34.1 DVec2",
    ),
    "rust-dvec3": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/f64/dvec3.rs",
        "703accd0e4564f815352c8f83bab7b57bfe7c5abd6aed4b6aeb8bbdae4a23445",
        "glam 0.34.1 DVec3",
    ),
    "rust-dvec4": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/f64/dvec4.rs",
        "18177949137dee05543624a5217dce1b79126719810ca028d0a0905ac16af465",
        "glam 0.34.1 DVec4",
    ),
    "rust-bvec2": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/bool/bvec2.rs",
        "ae10a93d312b53c9ce293293fb279bc516267be3969fcee06d0a32f73ed5539c",
        "glam 0.34.1 BVec2",
    ),
    "rust-bvec3": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/bool/bvec3.rs",
        "fa794b33fcafd2e45922bf14502185d591e596b3213ca81f4b758d78738d6757",
        "glam 0.34.1 BVec3",
    ),
    "rust-bvec4": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/bool/bvec4.rs",
        "f61b96a41d0a97d9b3c1792ad332a2a88031adce8af66e4fe5f0c3be49793667",
        "glam 0.34.1 BVec4",
    ),
    "rust-mat2": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/f32/scalar/mat2.rs",
        "81525fd8c458f9d7b45d8dc156d4a55d7ba88cea64c8231e02021b874cb8e33d",
        "glam 0.34.1 Mat2",
    ),
    "rust-mat3": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/f32/mat3.rs",
        "a1b6dcd758145a2b92828b64b46205d3fb78b32454601687b4330a0a94132a3d",
        "glam 0.34.1 Mat3",
    ),
    "rust-mat3a": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/f32/scalar/mat3a.rs",
        "b57dee871b68bef1ed541564f33a6a6810dc4e10a59a396f82c5beeedacf0d9d",
        "glam 0.34.1 Mat3A",
    ),
    "rust-mat4": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/f32/scalar/mat4.rs",
        "57c997cb4d3c38fd698051fe7de45b8d270c84d824300305c26040ab5c85acf8",
        "glam 0.34.1 Mat4",
    ),
    "rust-dmat2": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/f64/dmat2.rs",
        "5dfdc9e9e5890539fc89a480369816a99d842f2eacfc6632c95d50c8128e7a26",
        "glam 0.34.1 DMat2",
    ),
    "rust-dmat3": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/f64/dmat3.rs",
        "10b7f56d9908bc35a5ce43ad722aad839ef8094dab3be54a6631550aceafa864",
        "glam 0.34.1 DMat3",
    ),
    "rust-dmat4": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/f64/dmat4.rs",
        "6f30504927e9bab9780186f69642b357966ebe36ae410a283509aa20a7e1f3b2",
        "glam 0.34.1 DMat4",
    ),
    "rust-quat": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/f32/scalar/quat.rs",
        "d7d3395516d6dfe1b84eaeaabad91e724303172d8a72816036917f2e49946153",
        "glam 0.34.1 Quat",
    ),
    "rust-dquat": Source(
        "bitshifter/glam-rs",
        "f5c5a6b635c5be1b7e904808175470ed0f29f4ff",
        "src/f64/dquat.rs",
        "0499b77ff842f397dc443e044f8053dafc3b16d1d2e6a090618b7ca6d71fd692",
        "glam 0.34.1 DQuat",
    ),
    # Slang's standard library is the core module slangc is built from, pinned to the release CI installs.
    "slang": Source(
        "shader-slang/slang",
        "d9ede2dd5836afc0e9586da18d9236b171d0faa8",
        "source/slang/hlsl.meta.slang",
        "da5f4febdc674f42a43679cbc4b37544457bf93d0831920af6c3a276e4e14cb4",
        "Slang v2026.20 core module: HLSL-compatible built-in functions",
    ),
    "slang-core": Source(
        "shader-slang/slang",
        "d9ede2dd5836afc0e9586da18d9236b171d0faa8",
        "source/slang/core.meta.slang",
        "634c1397f6e4e689d7621495e9ea50693cdace2db8935eef93d48ce427ba198e",
        "Slang v2026.20 core module: core built-in functions",
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
