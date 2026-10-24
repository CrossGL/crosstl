# CrossTL coverage

Measured on commit [`fbca7ea`](https://github.com/CrossGL/crosstl/commit/fbca7eaacace6f6c14c7f68738b4d3f9f27de060) at 2026-10-24T23:27:16Z (CrossTL 3.4.0) by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml), which publishes this file on every push to `main` and nightly. Run `python -m tools.coverage run` to measure locally.

Every feature in a pinned upstream feature list gets a generated probe. For output, CrossTL translates a CrossGL probe to the language; for input, it imports a native probe and translates it back. The native SPIR-V probe is the module glslang compiles the GLSL probe to, and its round trip must keep the feature's instruction. The language's reference tool then validates the result:

- **supported**: the reference tool accepts the translation and, where it runs, the translation computes what the native probe computes;
- **refused**: CrossTL rejects the feature with a diagnostic;
- **wrong**: CrossTL produces output the reference tool rejects or that computes other results, or fails without a diagnostic;
- **untested**: no probe can be built for the feature yet, or the reference tool rejects the probe itself.

Score = supported / (supported + refused + wrong). Untested features are shown separately and do not count. The CrossGL IR row covers the union of every language's output features, merged into semantic features through the SPIR-V each native probe lowers to; a semantic feature is expressible when its CrossGL probe translates correctly to at least one language that has it.

Where Core's conformance harness has a runtime for the language, a supported compute probe that reads one input buffer also runs: the native probe and CrossTL's translation get the same inputs, and a translation that writes other results is wrong. HLSL, Slang and WGSL probes run as the SPIR-V their reference compiler writes for them. Executed: GLSL output, 112 probes on opengl-llvmpipe; GLSL input, 112 probes on opengl-llvmpipe; HLSL output, 131 probes on vulkan-lavapipe; HLSL input, 131 probes on vulkan-lavapipe; Metal output, 179 probes on metal-apple; Metal input, 187 probes on metal-apple; Slang input, 171 probes on vulkan-lavapipe; SPIR-V output, 92 probes on vulkan-lavapipe; SPIR-V input, 91 probes on vulkan-lavapipe; WGSL output, 76 probes on vulkan-lavapipe.

| Language | Direction | Score | Supported | Refused | Wrong | Untested |
|---|---|---:|---:|---:|---:|---:|
| CUDA | output | 0.0 % | 0 | 4 | 500 | 0 |
| CUDA | input | 98.2 % | 495 | 2 | 7 | 0 |
| GLSL | output | 93.2 % | 207 | 2 | 13 | 0 |
| GLSL | input | 99.1 % | 220 | 0 | 2 | 0 |
| HIP | output | 0.0 % | 0 | 4 | 384 | 0 |
| HIP | input | 97.2 % | 377 | 2 | 9 | 0 |
| HLSL | output | 95.1 % | 506 | 22 | 4 | 11 |
| HLSL | input | 82.5 % | 439 | 61 | 32 | 11 |
| Metal | output | 93.6 % | 250 | 12 | 5 | 0 |
| Metal | input | 90.6 % | 242 | 3 | 22 | 0 |
| Mojo | output | 0.0 % | 0 | 1 | 111 | 0 |
| Mojo | input | 0.0 % | 0 | 112 | 0 | 0 |
| OpenCL C | output | not offered | | | | |
| OpenCL C | input | 34.9 % | 178 | 47 | 285 | 0 |
| Rust | output | 0.0 % | 0 | 12 | 349 | 3 |
| Rust | input | 0.0 % | 0 | 28 | 332 | 4 |
| Slang | output | 87.4 % | 327 | 9 | 38 | 16 |
| Slang | input | 86.9 % | 325 | 17 | 32 | 16 |
| SPIR-V | output | 93.1 % | 121 | 8 | 1 | 3 |
| SPIR-V | input | 90.9 % | 120 | 8 | 4 | 1 |
| WebGL | output | 100.0 % | 113 | 0 | 0 | 0 |
| WebGL | input | not offered | | | | |
| WGSL | output | 64.2 % | 115 | 1 | 63 | 1 |
| WGSL | input | not offered | | | | |
| CrossGL IR | expressiveness | 41.7 % | 610 | 48 | 804 | 30 |

OpenCL C output is not offered: CrossTL has no OpenCL C generator.
WebGL input is not offered: CrossTL has no WebGL source frontend.
WGSL input is not offered: CrossTL has no WGSL source frontend.

## Trend

![CrossTL coverage over time per language and direction](trend.svg)

## By category

Each cell counts the supported features of a category over its supported, refused and wrong ones. Untested features are left out, and a blank cell means none of the category's features could be measured.

| Language | Direction | compute | control-flow | function | method | resource | stage | type |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CUDA | output | 0/5 | 0/13 | 0/469 | | 0/4 | 0/1 | 0/12 |
| CUDA | input | 5/5 | 13/13 | 463/469 | | 4/4 | 1/1 | 9/12 |
| GLSL | output | 5/5 | 14/14 | 159/170 | | 4/4 | 10/14 | 15/15 |
| GLSL | input | 5/5 | 14/14 | 169/170 | | 4/4 | 13/14 | 15/15 |
| HIP | output | 0/5 | 0/13 | 0/353 | | 0/4 | 0/1 | 0/12 |
| HIP | input | 5/5 | 13/13 | 345/353 | | 4/4 | 1/1 | 9/12 |
| HLSL | output | 4/4 | 14/14 | 195/198 | 263/285 | 4/4 | 13/14 | 13/13 |
| HLSL | input | 4/4 | 14/14 | 195/198 | 195/285 | 4/4 | 14/14 | 13/13 |
| Metal | output | 5/5 | 14/14 | 209/224 | | 4/4 | 5/7 | 13/13 |
| Metal | input | 5/5 | 14/14 | 203/224 | | 4/4 | 3/7 | 13/13 |
| Mojo | output | 0/5 | 0/10 | 0/81 | | 0/2 | 0/1 | 0/13 |
| Mojo | input | 0/5 | 0/10 | 0/81 | | 0/2 | 0/1 | 0/13 |
| OpenCL C | input | 4/5 | 11/13 | 147/474 | | 2/4 | 1/1 | 13/13 |
| Rust | output | 0/5 | 0/13 | 0/310 | | 0/4 | 0/14 | 0/15 |
| Rust | input | 0/5 | 0/13 | 0/309 | | 0/4 | 0/14 | 0/15 |
| Slang | output | 5/5 | 14/14 | 278/322 | | 4/4 | 12/14 | 14/15 |
| Slang | input | 5/5 | 12/14 | 281/322 | | 4/4 | 9/14 | 14/15 |
| SPIR-V | output | 5/5 | 14/14 | 72/78 | | 4/4 | 12/14 | 14/15 |
| SPIR-V | input | 5/5 | 14/14 | 74/80 | | 4/4 | 13/14 | 10/15 |
| WebGL | output | | 14/14 | 89/89 | | 2/2 | 2/2 | 6/6 |
| WGSL | output | 5/5 | 11/14 | 85/145 | | 4/4 | 3/3 | 7/8 |
| CrossGL IR | expressiveness | 5/5 | 14/14 | 468/1304 | 90/106 | 4/4 | 14/14 | 15/15 |

## Feature lists

| Language | Feature list | Pinned at |
|---|---|---|
| CUDA | [LLVM 23.1.3 device-side CUDA Math API wrappers](https://github.com/llvm/llvm-project/blob/0d261d1ca552c95a8f007e061c787ac7132fbcbc/clang/lib/Headers/__clang_cuda_math.h) | `0d261d1ca552` |
| CUDA | [LLVM 23.1.3 CUDA intrinsic and atomic wrappers](https://github.com/llvm/llvm-project/blob/0d261d1ca552c95a8f007e061c787ac7132fbcbc/clang/lib/Headers/__clang_cuda_device_functions.h) | `0d261d1ca552` |
| GLSL | [GLSL 4.60 specification, chapter 8 (built-in functions)](https://github.com/KhronosGroup/GLSL/blob/0099bf83b0288469917fc2e2972cb0a2b74d6711/chapters/builtinfunctions.adoc) | `0099bf83b028` |
| HIP | [LLVM 17.0.6 device-side HIP math functions and intrinsics](https://github.com/llvm/llvm-project/blob/6009708b4367171ccdbf4b5905cb6a803753fe18/clang/lib/Headers/__clang_hip_math.h) | `6009708b4367` |
| HIP | [ROCm 5.7.1 HIP atomic functions](https://github.com/ROCm/clr/blob/1949b1621a802ffb1492616adbae6154bfbe64ef/hipamd/include/hip/amd_detail/amd_hip_atomic.h) | `1949b1621a80` |
| HIP | [ROCm 5.7.1 HIP device functions (integer intrinsics, conversions, barriers)](https://github.com/ROCm/clr/blob/1949b1621a802ffb1492616adbae6154bfbe64ef/hipamd/include/hip/amd_detail/amd_device_functions.h) | `1949b1621a80` |
| HLSL | [DXC v1.9.2602.24 intrinsic table](https://github.com/microsoft/DirectXShaderCompiler/blob/d355aa8364d34df3f0822ba0de8d1dfc75ae6f48/utils/hct/gen_intrin_main.txt) | `d355aa8364d3` |
| Metal | [Metal Shading Language 4.1 standard library, from Apple metal version 32023.918 (metalfe-32023.918.1)](https://github.com/CrossGL/crosstl/blob/fbca7eaacace6f6c14c7f68738b4d3f9f27de060/tools/coverage/features/metal.txt) | this repository |
| Mojo | [Mojo 1.1.0 std.bit functions](https://github.com/modular/modular/blob/c6fa49f712fe15c99d0461ad2baaf2a15cbc58ce/Mojo/stdlib/std/bit/bit.mojo) | `c6fa49f712fe` |
| Mojo | [Mojo 1.1.0 std.bit public API](https://github.com/modular/modular/blob/c6fa49f712fe15c99d0461ad2baaf2a15cbc58ce/Mojo/stdlib/std/bit/__init__.mojo) | `c6fa49f712fe` |
| Mojo | [Mojo 1.1.0 std.math functions](https://github.com/modular/modular/blob/c6fa49f712fe15c99d0461ad2baaf2a15cbc58ce/Mojo/stdlib/std/math/math.mojo) | `c6fa49f712fe` |
| Mojo | [Mojo 1.1.0 std.math public API](https://github.com/modular/modular/blob/c6fa49f712fe15c99d0461ad2baaf2a15cbc58ce/Mojo/stdlib/std/math/__init__.mojo) | `c6fa49f712fe` |
| Mojo | [Mojo 1.1.0 floating-point predicates std.math re-exports](https://github.com/modular/modular/blob/c6fa49f712fe15c99d0461ad2baaf2a15cbc58ce/Mojo/stdlib/std/utils/numerics.mojo) | `c6fa49f712fe` |
| OpenCL C | [LLVM 18.1.3 OpenCL C built-in function declarations](https://github.com/llvm/llvm-project/blob/c13b7485b87909fcf739f62cfa382b55407433c0/clang/lib/Headers/opencl-c.h) | `c13b7485b879` |
| Rust | [rust-gpu 0.11.0 spirv_std::arch](https://github.com/Rust-GPU/rust-gpu/blob/62401280ab4b512c734ad1aceee7553385e17418/crates/spirv-std/src/arch.rs) | `62401280ab4b` |
| Rust | [rust-gpu 0.11.0 spirv_std::arch, module atomics](https://github.com/Rust-GPU/rust-gpu/blob/62401280ab4b512c734ad1aceee7553385e17418/crates/spirv-std/src/arch/atomics.rs) | `62401280ab4b` |
| Rust | [rust-gpu 0.11.0 spirv_std::arch, module barrier](https://github.com/Rust-GPU/rust-gpu/blob/62401280ab4b512c734ad1aceee7553385e17418/crates/spirv-std/src/arch/barrier.rs) | `62401280ab4b` |
| Rust | [rust-gpu 0.11.0 spirv_std::arch, module demote_to_helper_invocation_ext](https://github.com/Rust-GPU/rust-gpu/blob/62401280ab4b512c734ad1aceee7553385e17418/crates/spirv-std/src/arch/demote_to_helper_invocation_ext.rs) | `62401280ab4b` |
| Rust | [rust-gpu 0.11.0 spirv_std::arch, module derivative](https://github.com/Rust-GPU/rust-gpu/blob/62401280ab4b512c734ad1aceee7553385e17418/crates/spirv-std/src/arch/derivative.rs) | `62401280ab4b` |
| Rust | [num-traits 0.2.19 Float trait](https://github.com/rust-num/num-traits/blob/7ec3d41d39b28190ec1d42db38021107b3951f3a/src/float.rs) | `7ec3d41d39b2` |
| Rust | [rust-gpu 0.11.0 spirv_std::arch, module mesh_shading](https://github.com/Rust-GPU/rust-gpu/blob/62401280ab4b512c734ad1aceee7553385e17418/crates/spirv-std/src/arch/mesh_shading.rs) | `62401280ab4b` |
| Rust | [rust-gpu 0.11.0 spirv_std::arch, module primitive](https://github.com/Rust-GPU/rust-gpu/blob/62401280ab4b512c734ad1aceee7553385e17418/crates/spirv-std/src/arch/primitive.rs) | `62401280ab4b` |
| Rust | [rust-gpu 0.11.0 spirv_std::arch, module ray_tracing](https://github.com/Rust-GPU/rust-gpu/blob/62401280ab4b512c734ad1aceee7553385e17418/crates/spirv-std/src/arch/ray_tracing.rs) | `62401280ab4b` |
| Rust | [rust-gpu 0.11.0 spirv_std::arch, module subgroup](https://github.com/Rust-GPU/rust-gpu/blob/62401280ab4b512c734ad1aceee7553385e17418/crates/spirv-std/src/arch/subgroup.rs) | `62401280ab4b` |
| Rust | [glam 0.34.1 Vec3](https://github.com/bitshifter/glam-rs/blob/f5c5a6b635c5be1b7e904808175470ed0f29f4ff/src/f32/vec3.rs) | `f5c5a6b635c5` |
| Slang | [Slang v2026.9.1 core module: HLSL-compatible built-in functions](https://github.com/shader-slang/slang/blob/4685b35be78104ae007013cb749a1e906c414600/source/slang/hlsl.meta.slang) | `4685b35be781` |
| Slang | [Slang v2026.9.1 core module: core built-in functions](https://github.com/shader-slang/slang/blob/4685b35be78104ae007013cb749a1e906c414600/source/slang/core.meta.slang) | `4685b35be781` |
| SPIR-V | [SPIR-V GLSL.std.450 extended instruction grammar](https://github.com/KhronosGroup/SPIRV-Headers/blob/496543121ce6419f23d6fa5d7194ba66c36212d2/include/spirv/unified1/extinst.glsl.std.450.grammar.json) | `496543121ce6` |
| SPIR-V | [SPIR-V core grammar (execution models, capabilities, storage classes, instructions)](https://github.com/KhronosGroup/SPIRV-Headers/blob/496543121ce6419f23d6fa5d7194ba66c36212d2/include/spirv/unified1/spirv.core.grammar.json) | `496543121ce6` |
| WebGL | [ANGLE's GLSL ES built-in function table](https://github.com/google/angle/blob/1ff891807e391d842c33d552ea26ab26b5341ef4/src/compiler/translator/builtin_function_declarations.txt) | `1ff891807e39` |
| WGSL | [Tint's WGSL built-in function definitions](https://github.com/google/dawn/blob/ff7c6960f62d4249f967a78448918eec21705a86/src/tint/lang/wgsl/wgsl.def) | `ff7c6960f62d` |

Reference tools: clang Ubuntu clang version 18.1.3 (1ubuntu1), dxc libdxcompiler.so: 1.9(5191-d355aa83)(1.9.0.5191), glslangValidator Glslang Version: 11:16.6.0, hipcc HIP version: 5.7.31921-0, metal Apple metal version 32023.921 (metalfe-32023.921.6), mojo Mojo 1.1.0 (8189361e), naga 30.0.1, nvcc nvcc: NVIDIA (R) Cuda compiler driver, rust-gpu rust-gpu 0.11.0 (rustc 1.99.0-nightly (d453bdd8f 2026-08-14)), slangc 2026.9.1, spirv-as SPIRV-Tools v2026.4 ef96ed763b43b59b33b31b362f09a02b729fa1c9, spirv-dis SPIRV-Tools v2026.4 ef96ed763b43b59b33b31b362f09a02b729fa1c9, spirv-val SPIRV-Tools v2026.4 ef96ed763b43b59b33b31b362f09a02b729fa1c9.

## Gaps

Each row's wrong and refused features are grouped by diagnostic. A diagnostic drops its line numbers and writes the probe's own variables as `<probe>`, and one that several features share writes each feature's name as `<feature>`. Untested features are grouped by the reason no probe could measure them.

### CUDA output

<details>
<summary>504 gaps: 4 refused, 500 wrong</summary>

Wrong, by cause ([open bug issues](https://github.com/CrossGL/crosstl/issues?q=is:issue%20is:open%20label:coverage%20label:bug%20label:backend:cuda%20label:direction:output)):

- `identifier "<probe>" is undefined in device code` (443): `__brev`, `__byte_perm`, `__clz`, `__clzll`, `__cosf`, `__dadd_rd`, `__dadd_rn`, `__dadd_ru`, `__dadd_rz`, `__ddiv_rd`, `__ddiv_rn`, `__ddiv_ru`, `__ddiv_rz`, `__dmul_rd`, `__dmul_rn`, `__dmul_ru`, `__dmul_rz`, `__double2float_rd`, `__double2float_rn`, `__double2float_ru`, `__double2float_rz`, `__double2hiint`, `__double2int_rd`, `__double2int_rn`, `__double2int_ru`, `__double2int_rz`, `__double2ll_rd`, `__double2ll_rn`, `__double2ll_ru`, `__double2ll_rz`, `__double2loint`, `__double2uint_rd`, `__double2uint_rn`, `__double2uint_ru`, `__double2uint_rz`, `__double_as_longlong`, `__drcp_rd`, `__drcp_rn`, `__drcp_ru`, `__drcp_rz`, `__dsqrt_rd`, `__dsqrt_rn`, `__dsqrt_ru`, `__dsqrt_rz`, `__dsub_rd`, `__dsub_rn`, `__dsub_ru`, `__dsub_rz`, `__exp10f`, `__expf`, `__fadd_rd`, `__fadd_rn`, `__fadd_ru`, `__fadd_rz`, `__fdiv_rd`, `__fdiv_rn`, `__fdiv_ru`, `__fdiv_rz`, `__fdividef`, `__ffs`, `__ffsll`, `__finite`, `__finitef`, `__float2int_rd`, `__float2int_rn`, `__float2int_ru`, `__float2int_rz`, `__float2ll_rd`, `__float2ll_rn`, `__float2ll_ru`, `__float2ll_rz`, `__float2uint_rd`, `__float2uint_rn`, `__float2uint_ru`, `__float2uint_rz`, `__float_as_int`, `__float_as_uint`, `__fma_rd`, `__fma_rn`, `__fma_ru`, `__fma_rz`, `__fmaf_ieee_rd`, `__fmaf_ieee_rn`, `__fmaf_ieee_ru`, `__fmaf_ieee_rz`, `__fmaf_rd`, `__fmaf_rn`, `__fmaf_ru`, `__fmaf_rz`, `__fmul_rd`, `__fmul_rn`, `__fmul_ru`, `__fmul_rz`, `__frcp_rd`, `__frcp_rn`, `__frcp_ru`, `__frcp_rz`, `__frsqrt_rn`, `__fsqrt_rd`, `__fsqrt_rn`, `__fsqrt_ru`, `__fsqrt_rz`, `__fsub_rd`, `__fsub_rn`, `__fsub_ru`, `__fsub_rz`, `__hadd`, `__hiloint2double`, `__int2double_rn`, `__int2float_rd`, `__int2float_rn`, `__int2float_ru`, `__int2float_rz`, `__int_as_float`, `__isinf`, `__isinff`, `__isnan`, `__isnanf`, `__ll2double_rd`, `__ll2double_rn`, `__ll2double_ru`, `__ll2double_rz`, `__ll2float_rd`, `__ll2float_rn`, `__ll2float_ru`, `__ll2float_rz`, `__log10f`, `__log2f`, `__logf`, `__longlong_as_double`, `__mul24`, `__mul64hi`, `__mulhi`, `__popc`, `__powf`, `__rhadd`, `__sad`, `__saturatef`, `__signbitf`, `__sinf`, `__syncthreads_and`, `__syncthreads_count`, `__syncthreads_or`, `__tanf`, `__threadfence`, `__threadfence_block`, `__threadfence_system`, `__uhadd`, `__uint2double_rn`, `__uint2float_rd`, `__uint2float_rn`, `__uint2float_ru`, `__uint2float_rz`, `__uint_as_float`, `__umul24`, `__umulhi`, `__urhadd`, `__usad`, `__vabs2`, `__vabs4`, `__vabsdiffs2`, `__vabsdiffs4`, `__vabsdiffu2`, `__vabsdiffu4`, `__vabsss2`, `__vabsss4`, `__vadd2`, `__vadd4`, `__vaddss2`, `__vaddss4`, `__vaddus2`, `__vaddus4`, `__vavgs2`, `__vavgs4`, `__vavgu2`, `__vavgu4`, `__vcmpeq2`, `__vcmpeq4`, `__vcmpges2`, `__vcmpges4`, `__vcmpgeu2`, `__vcmpgeu4`, `__vcmpgts2`, `__vcmpgts4`, `__vcmpgtu2`, `__vcmpgtu4`, `__vcmples2`, `__vcmples4`, `__vcmpleu2`, `__vcmpleu4`, `__vcmplts2`, `__vcmplts4`, `__vcmpltu2`, `__vcmpltu4`, `__vcmpne2`, `__vcmpne4`, `__vhaddu2`, `__vhaddu4`, `__vmaxs2`, `__vmaxs4`, `__vmaxu2`, `__vmaxu4`, `__vmins2`, `__vmins4`, `__vminu2`, `__vminu4`, `__vneg2`, `__vneg4`, `__vnegss2`, `__vnegss4`, `__vsads2`, `__vsads4`, `__vsadu2`, `__vsadu4`, `__vseteq2`, `__vseteq4`, `__vsetges2`, `__vsetges4`, `__vsetgeu2`, `__vsetgeu4`, `__vsetgts2`, `__vsetgts4`, `__vsetgtu2`, `__vsetgtu4`, `__vsetles2`, `__vsetles4`, `__vsetleu2`, `__vsetleu4`, `__vsetlts2`, `__vsetlts4`, `__vsetltu2`, `__vsetltu4`, `__vsetne2`, `__vsetne4`, `__vsub2`, `__vsub4`, `__vsubss2`, `__vsubss4`, `__vsubus2`, `__vsubus4`, `abs`, `acos`, `acosf`, `acosh`, `acoshf`, `array`, `array-of-structs`, `asin`, `asinf`, `asinh`, `asinhf`, `atan`, `atan2`, `atan2f`, `atanf`, `atanh`, `atanhf`, `atomicAdd`, `atomicAnd`, `atomicCAS`, `atomicDec`, `atomicExch`, `atomicInc`, `atomicMax`, `atomicMin`, `atomicOr`, `atomicXor`, `break`, `cbrt`, `cbrtf`, `ceil`, `ceilf`, `clock`, `clock64`, `compute`, `conditional-expression`, `continue`, `copysign`, `copysignf`, `cos`, `cosf`, `cosh`, `coshf`, `cyl_bessel_i0`, `cyl_bessel_i0f`, `cyl_bessel_i1`, `cyl_bessel_i1f`, `do-while-loop`, `double`, `early-return`, `erf`, `erfc`, `erfcf`, `erfcinv`, `erfcinvf`, `erfcx`, `erfcxf`, `erff`, `erfinv`, `erfinvf`, `exp`, `exp10`, `exp10f`, `exp2`, `exp2f`, `expf`, `expm1`, `expm1f`, `fabs`, `fabsf`, `fdim`, `fdimf`, `fdivide`, `fdividef`, `floor`, `floorf`, `fma`, `fmaf`, `fmax`, `fmaxf`, `fmin`, `fminf`, `fmod`, `fmodf`, `for-loop`, `function-call`, `global-constant`, `half`, `hypot`, `hypotf`, `if-else`, `ilogb`, `ilogbf`, `inout-parameter`, `int16`, `int64`, `int8`, `j0`, `j0f`, `j1`, `j1f`, `jn`, `jnf`, `labs`, `ldexp`, `ldexpf`, `lgamma`, `lgammaf`, `llabs`, `llmax`, `llmin`, `llrint`, `llrintf`, `llround`, `llroundf`, `local-invocation-id`, `local-invocation-index`, `log`, `log10`, `log10f`, `log1p`, `log1pf`, `log2`, `log2f`, `logb`, `logbf`, `logf`, `lrint`, `lrintf`, `lround`, `lroundf`, `max`, `min`, `nearbyint`, `nearbyintf`, `nextafter`, `nextafterf`, `norm`, `norm3d`, `norm3df`, `norm4d`, `norm4df`, `normcdf`, `normcdff`, `normcdfinv`, `normcdfinvf`, `num-workgroups`, `out-parameter`, `pow`, `powf`, `rcbrt`, `rcbrtf`, `read-only-buffer`, `remainder`, `remainderf`, `rhypot`, `rhypotf`, `rint`, `rintf`, `rnorm`, `rnorm3d`, `rnorm3df`, `rnorm4d`, `rnorm4df`, `round`, `roundf`, `rsqrt`, `rsqrtf`, `sampled-texture`, `scalbln`, `scalblnf`, `scalbn`, `scalbnf`, `shared-memory`, `short-circuit-logic`, `sin`, `sinf`, `sinh`, `sinhf`, `sqrt`, `sqrtf`, `storage-image`, `struct`, `switch`, `tan`, `tanf`, `tanh`, `tanhf`, `tgamma`, `tgammaf`, `trunc`, `truncf`, `umax`, `umin`, `uniform-buffer`, `while-loop`, `workgroup-id`, `y0`, `y0f`, `y1`, `y1f`, `yn`, `ynf`
- `no instance of overloaded function "<feature>" matches the argument list` (24): `atomicAdd_block`, `atomicAdd_system`, `atomicAnd_block`, `atomicAnd_system`, `atomicCAS_block`, `atomicCAS_system`, `atomicExch_block`, `atomicExch_system`, `atomicMax_block`, `atomicMax_system`, `atomicMin_block`, `atomicMin_system`, `atomicOr_block`, `atomicOr_system`, `atomicXor_block`, `atomicXor_system`, `frexp`, `frexpf`, `modf`, `modff`, `remquo`, `remquof`, `sincos`, `sincospi`
- `identifier "uint64_t" is undefined` (22): `__brevll`, `__double2ull_rd`, `__double2ull_rn`, `__double2ull_ru`, `__double2ull_rz`, `__float2ull_rd`, `__float2ull_rn`, `__float2ull_ru`, `__float2ull_rz`, `__popcll`, `__ull2double_rd`, `__ull2double_rn`, `__ull2double_ru`, `__ull2double_rz`, `__ull2float_rd`, `__ull2float_rn`, `__ull2float_ru`, `__ull2float_rz`, `__umul64hi`, `uint64`, `ullmax`, `ullmin`
- `argument of type "uint" (aka "unsigned int") is incompatible with parameter of type "unsigned int *"` (4): `atomicDec_block`, `atomicDec_system`, `atomicInc_block`, `atomicInc_system`
- `argument of type "float" is incompatible with parameter of type "float *"` (3): `__sincosf`, `sincosf`, `sincospif`
- `argument of type "float *" is incompatible with parameter of type "const double *"` (2): `normf`, `rnormf`
- `identifier "uint16_t" is undefined`: `uint16`
- `identifier "uint8_t" is undefined`: `uint8`

Refused, by diagnostic:

- `project.translate.cuda-metal-math-function-unsupported: MetalMathFunctionUnsupportedError: MSL <feature> is exact at integers and half-integers, which cuda has no function for, and CrossTL does not lower it with an exact range reduction yet (target cuda) [project.translate.cuda-metal-math-function-unsupported]` (4): `cospi`, `cospif`, `sinpi`, `sinpif`

</details>

### CUDA input

<details>
<summary>9 gaps: 2 refused, 7 wrong</summary>

Wrong, by cause ([open bug issues](https://github.com/CrossGL/crosstl/issues?q=is:issue%20is:open%20label:coverage%20label:bug%20label:backend:cuda%20label:direction:input)):

- `identifier "countLeadingZeros" is undefined` (2): `__clz`, `__clzll`
- `invalid specifier for structured binding declaration` (2): `array`, `array-of-structs`
- `identifier "SCALE" is undefined in device code`: `global-constant`
- `identifier "doubleBitsToLong" is undefined`: `__double_as_longlong`
- `identifier "longBitsToDouble" is undefined`: `__longlong_as_double`

Refused, by diagnostic:

- `project.translate.cuda-metal-math-function-unsupported: MetalMathFunctionUnsupportedError: MSL <feature> is exact at integers and half-integers, which cuda has no function for, and CrossTL does not lower it with an exact range reduction yet (target cuda) [project.translate.cuda-metal-math-function-unsupported]` (2): `cospi`, `sinpi`

</details>

### GLSL output

<details>
<summary>15 gaps: 2 refused, 13 wrong</summary>

Wrong, by cause ([open bug issues](https://github.com/CrossGL/crosstl/issues?q=is:issue%20is:open%20label:coverage%20label:bug%20label:backend:glsl%20label:direction:output)):

- `'<feature>' : no matching overloaded function found` (9): `atomicCounterAdd`, `atomicCounterAnd`, `atomicCounterCompSwap`, `atomicCounterExchange`, `atomicCounterMax`, `atomicCounterMin`, `atomicCounterOr`, `atomicCounterSubtract`, `atomicCounterXor`
- `ERROR: Linking mesh stage: At least one shader must specify a layout(max_vertices = value)`: `mesh`
- `'textureSamples' : no matching overloaded function found`: `imageSamples`
- `'location' : SPIR-V requires location for user input/output`: `subpassLoad`
- `'taskPayloadSharedEXT' : too many storage qualifiers`: `task`

Refused, by diagnostic:

- `project.translate.opengl-tessellation-patch-unsupported: TessellationPatchUnsupportedError: domain entry point '<probe>' needs [domain("tri"|"quad"|"isoline")]; opengl tessellation shaders exchange control points through per-vertex arrays, patch variables and gl_TessLevelOuter/gl_TessLevelInner (target opengl) [project.translate.opengl-tessellation-patch-unsupported]`: `tessellation-evaluation`
- `project.translate.opengl-tessellation-patch-unsupported: TessellationPatchUnsupportedError: patch-constant function 'IdentifierNode(name=constants)' is not one function of the program; opengl tessellation shaders exchange control points through per-vertex arrays, patch variables and gl_TessLevelOuter/gl_TessLevelInner (target opengl) [project.translate.opengl-tessellation-patch-unsupported`: `tessellation-control`

</details>

### GLSL input

<details>
<summary>2 gaps: 0 refused, 2 wrong</summary>

Wrong, by cause ([open bug issues](https://github.com/CrossGL/crosstl/issues?q=is:issue%20is:open%20label:coverage%20label:bug%20label:backend:glsl%20label:direction:input)):

- `ValueError: GLSL DispatchMesh payload argument type uint does not match taskPayloadSharedEXT payload type Payload`: `task`
- `'textureSamples' : no matching overloaded function found`: `imageSamples`

</details>

### HIP output

<details>
<summary>388 gaps: 4 refused, 384 wrong</summary>

Wrong, by cause ([open bug issues](https://github.com/CrossGL/crosstl/issues?q=is:issue%20is:open%20label:coverage%20label:bug%20label:backend:hip%20label:direction:output)):

- `reference to __host__ variable '<probe>' in __global__ function` (384): `__all`, `__any`, `__ballot`, `__ballot64`, `__bitextract_u32`, `__bitextract_u64`, `__bitinsert_u32`, `__bitinsert_u64`, `__brev`, `__brevll`, `__byte_perm`, `__clz`, `__clzll`, `__cosf`, `__dadd_rn`, `__ddiv_rn`, `__dmul_rn`, `__double2float_rd`, `__double2float_rn`, `__double2float_ru`, `__double2float_rz`, `__double2hiint`, `__double2int_rd`, `__double2int_rn`, `__double2int_ru`, `__double2int_rz`, `__double2ll_rd`, `__double2ll_rn`, `__double2ll_ru`, `__double2ll_rz`, `__double2loint`, `__double2uint_rd`, `__double2uint_rn`, `__double2uint_ru`, `__double2uint_rz`, `__double2ull_rd`, `__double2ull_rn`, `__double2ull_ru`, `__double2ull_rz`, `__double_as_longlong`, `__drcp_rn`, `__dsqrt_rn`, `__dsub_rn`, `__exp10f`, `__expf`, `__fadd_rn`, `__fdiv_rn`, `__fdividef`, `__ffs`, `__ffsll`, `__finite`, `__finitef`, `__float2int_rd`, `__float2int_rn`, `__float2int_ru`, `__float2int_rz`, `__float2ll_rd`, `__float2ll_rn`, `__float2ll_ru`, `__float2ll_rz`, `__float2uint_rd`, `__float2uint_rn`, `__float2uint_ru`, `__float2uint_rz`, `__float2ull_rd`, `__float2ull_rn`, `__float2ull_ru`, `__float2ull_rz`, `__float_as_int`, `__float_as_uint`, `__fma_rn`, `__fmaf_rn`, `__fmul_rn`, `__fns32`, `__fns64`, `__frcp_rn`, `__frsqrt_rn`, `__fsqrt_rn`, `__fsub_rn`, `__funnelshift_l`, `__funnelshift_lc`, `__funnelshift_r`, `__funnelshift_rc`, `__hadd`, `__hiloint2double`, `__int2double_rn`, `__int2float_rd`, `__int2float_rn`, `__int2float_ru`, `__int2float_rz`, `__int_as_float`, `__isinf`, `__isinff`, `__isnan`, `__isnanf`, `__lane_id`, `__lanemask_eq`, `__lanemask_gt`, `__lanemask_lt`, `__ll2double_rd`, `__ll2double_rn`, `__ll2double_ru`, `__ll2double_rz`, `__ll2float_rd`, `__ll2float_rn`, `__ll2float_ru`, `__ll2float_rz`, `__log10f`, `__log2f`, `__logf`, `__longlong_as_double`, `__mbcnt_hi`, `__mbcnt_lo`, `__mul24`, `__mul64hi`, `__mulhi`, `__popc`, `__popcll`, `__powf`, `__rhadd`, `__sad`, `__saturatef`, `__signbit`, `__signbitf`, `__sincosf`, `__sinf`, `__smid`, `__syncthreads`, `__syncthreads_and`, `__syncthreads_count`, `__syncthreads_or`, `__tanf`, `__threadfence`, `__threadfence_block`, `__threadfence_system`, `__uhadd`, `__uint2double_rn`, `__uint2float_rd`, `__uint2float_rn`, `__uint2float_ru`, `__uint2float_rz`, `__uint_as_float`, `__ull2double_rd`, `__ull2double_rn`, `__ull2double_ru`, `__ull2double_rz`, `__ull2float_rd`, `__ull2float_rn`, `__ull2float_ru`, `__ull2float_rz`, `__umul24`, `__umul64hi`, `__umulhi`, `__urhadd`, `__usad`, `abs`, `acos`, `acosf`, `acosh`, `acoshf`, `array`, `array-of-structs`, `asin`, `asinf`, `asinh`, `asinhf`, `atan`, `atan2`, `atan2f`, `atanf`, `atanh`, `atanhf`, `atomicAdd`, `atomicAddNoRet`, `atomicAdd_system`, `atomicAnd`, `atomicAnd_system`, `atomicCAS`, `atomicCAS_system`, `atomicDec`, `atomicExch`, `atomicExch_system`, `atomicInc`, `atomicMax`, `atomicMax_system`, `atomicMin`, `atomicMin_system`, `atomicOr`, `atomicOr_system`, `atomicSub`, `atomicSub_system`, `atomicXor`, `atomicXor_system`, `break`, `cbrt`, `cbrtf`, `ceil`, `ceilf`, `clock`, `clock64`, `compute`, `conditional-expression`, `continue`, `copysign`, `copysignf`, `cos`, `cosf`, `cosh`, `coshf`, `cyl_bessel_i0`, `cyl_bessel_i0f`, `cyl_bessel_i1`, `cyl_bessel_i1f`, `do-while-loop`, `double`, `early-return`, `erf`, `erfc`, `erfcf`, `erfcinv`, `erfcinvf`, `erfcx`, `erfcxf`, `erff`, `erfinv`, `erfinvf`, `exp`, `exp10`, `exp10f`, `exp2`, `exp2f`, `expf`, `expm1`, `expm1f`, `fabs`, `fabsf`, `fdim`, `fdimf`, `fdividef`, `floor`, `floorf`, `fma`, `fmaf`, `fmax`, `fmaxf`, `fmin`, `fminf`, `fmod`, `fmodf`, `for-loop`, `frexp`, `frexpf`, `function-call`, `global-constant`, `half`, `hypot`, `hypotf`, `if-else`, `ilogb`, `ilogbf`, `inout-parameter`, `int16`, `int64`, `int8`, `j0`, `j0f`, `j1`, `j1f`, `jn`, `jnf`, `labs`, `ldexp`, `ldexpf`, `lgamma`, `lgammaf`, `llabs`, `llrint`, `llrintf`, `llround`, `llroundf`, `local-invocation-id`, `local-invocation-index`, `log`, `log10`, `log10f`, `log1p`, `log1pf`, `log2`, `log2f`, `logb`, `logbf`, `logf`, `lrint`, `lrintf`, `lround`, `lroundf`, `max`, `min`, `modf`, `modff`, `nan`, `nanf`, `nearbyint`, `nearbyintf`, `nextafter`, `nextafterf`, `norm`, `norm3d`, `norm3df`, `norm4d`, `norm4df`, `normcdf`, `normcdff`, `normcdfinv`, `normcdfinvf`, `normf`, `num-workgroups`, `out-parameter`, `pow`, `powf`, `powi`, `powif`, `rcbrt`, `rcbrtf`, `read-only-buffer`, `remainder`, `remainderf`, `remquo`, `remquof`, `rhypot`, `rhypotf`, `rint`, `rintf`, `rnorm`, `rnorm3d`, `rnorm3df`, `rnorm4d`, `rnorm4df`, `rnormf`, `round`, `roundf`, `rsqrt`, `rsqrtf`, `sampled-texture`, `scalbln`, `scalblnf`, `scalbn`, `scalbnf`, `shared-memory`, `short-circuit-logic`, `sin`, `sincos`, `sincosf`, `sincospi`, `sincospif`, `sinf`, `sinh`, `sinhf`, `sqrt`, `sqrtf`, `storage-image`, `struct`, `switch`, `tan`, `tanf`, `tanh`, `tanhf`, `tgamma`, `tgammaf`, `trunc`, `truncf`, `uint16`, `uint64`, `uint8`, `uniform-buffer`, `wall_clock64`, `while-loop`, `workgroup-id`, `y0`, `y0f`, `y1`, `y1f`, `yn`, `ynf`

Refused, by diagnostic:

- `project.translate.hip-metal-math-function-unsupported: MetalMathFunctionUnsupportedError: MSL <feature> is exact at integers and half-integers, which hip has no function for, and CrossTL does not lower it with an exact range reduction yet (target hip) [project.translate.hip-metal-math-function-unsupported]` (4): `cospi`, `cospif`, `sinpi`, `sinpif`

</details>

### HIP input

<details>
<summary>11 gaps: 2 refused, 9 wrong</summary>

Wrong, by cause ([open bug issues](https://github.com/CrossGL/crosstl/issues?q=is:issue%20is:open%20label:coverage%20label:bug%20label:backend:hip%20label:direction:input)):

- `invalid operands to binary expression ('unsigned long long' and 'float')` (2): `__funnelshift_lc`, `__funnelshift_rc`
- `use of undeclared identifier 'countLeadingZeros'` (2): `__clz`, `__clzll`
- `no template named 'array'; did you mean 'std::array'?`: `array-of-structs`
- `reference to __host__ variable 'SCALE' in __global__ function`: `global-constant`
- `use of undeclared identifier 'doubleBitsToLong'`: `__double_as_longlong`
- `use of undeclared identifier 'f32'`: `array`
- `use of undeclared identifier 'longBitsToDouble'`: `__longlong_as_double`

Refused, by diagnostic:

- `project.translate.hip-metal-math-function-unsupported: MetalMathFunctionUnsupportedError: MSL <feature> is exact at integers and half-integers, which hip has no function for, and CrossTL does not lower it with an exact range reduction yet (target hip) [project.translate.hip-metal-math-function-unsupported]` (2): `cospi`, `sinpi`

</details>

### HLSL output

<details>
<summary>26 gaps: 22 refused, 4 wrong</summary>

Wrong, by cause ([open bug issues](https://github.com/CrossGL/crosstl/issues?q=is:issue%20is:open%20label:coverage%20label:bug%20label:backend:hlsl%20label:direction:output)):

- `ValueError: DirectX ByteAddressBuffer interlocked member '<feature>' value argument must be scalar uint, got int` (2): `RWByteAddressBuffer.InterlockedMax`, `RWByteAddressBuffer.InterlockedMin`
- `type 'Payload' used as payload requires that it is annotated with the [raypayload] attribute` (2): `DxHitObject.Invoke`, `DxHitObject.TraceRay`

Refused, by diagnostic:

- `CrossGL has no work-graph node stage` (11): `EmptyNodeInput.Count`, `EmptyNodeOutput.GroupIncrementOutputCount`, `EmptyNodeOutput.IsValid`, `EmptyNodeOutput.ThreadIncrementOutputCount`, `GetRemainingRecursionLevels`, `GroupNodeInputRecords.Count`, `GroupOrThreadNodeOutputRecords.OutputComplete`, `NodeOutput.GetGroupNodeOutputRecords`, `NodeOutput.GetThreadNodeOutputRecords`, `NodeOutput.IsValid`, `RWDispatchNodeInputRecord.FinishedCrossGroupSharing`
- `project.translate.undeclared-type: UndeclaredTypeError: type 'SAMPLER_FEEDBACK_MIN_MIP' is not declared: it is neither a CrossGL built-in type nor a structure, enumeration, type alias or generic parameter of the program (target directx) [project.translate.undeclared-type]` (8): `FeedbackTexture2D.WriteSamplerFeedback`, `FeedbackTexture2D.WriteSamplerFeedbackBias`, `FeedbackTexture2D.WriteSamplerFeedbackGrad`, `FeedbackTexture2D.WriteSamplerFeedbackLevel`, `FeedbackTexture2DArray.WriteSamplerFeedback`, `FeedbackTexture2DArray.WriteSamplerFeedbackBias`, `FeedbackTexture2DArray.WriteSamplerFeedbackGrad`, `FeedbackTexture2DArray.WriteSamplerFeedbackLevel`
- `project.translate.directx-contextual-conversion-unrepresentable: DirectXContextualConversionError: DirectX canonical select requires a resolved boolean scalar or vector condition`: `select`
- `project.translate.directx-stage-interface-unsupported: HlslStageInterfaceUnsupportedError: DirectX task stage: 'payload' is a taskPayloadSharedEXT variable; HLSL passes the payload to DispatchMesh and the mesh entry receives it as an in payload parameter, which is not implemented yet (target directx) [project.translate.directx-stage-interface-unsupported]`: `task`
- `project.translate.directx-stage-interface-unsupported: HlslStageInterfaceUnsupportedError: DirectX task stage: 'payload' is a taskPayloadSharedEXT variable; HLSL passes the payload to DispatchMesh and the mesh entry receives it as an in payload parameter, which is not implemented yet (target directx) [project.translate.directx-stage-interface-unsupported]`: `DispatchMesh`

</details>

<details>
<summary>11 untested: function 5, method 6</summary>

- `needs shader model 6.10` (10): `ClusterID`, `DxHitObject.ClusterID`, `DxHitObject.TriangleObjectPosition`, `GetGroupWaveCount`, `GetGroupWaveIndex`, `RayQuery.CandidateClusterID`, `RayQuery.CandidateTriangleObjectPosition`, `RayQuery.CommittedClusterID`, `RayQuery.CommittedTriangleObjectPosition`, `TriangleObjectPosition`
- `DXC compiles printf only to SPIR-V; DXIL has no print operation`: `printf`

</details>

### HLSL input

<details>
<summary>93 gaps: 61 refused, 32 wrong</summary>

Wrong, by cause ([open bug issues](https://github.com/CrossGL/crosstl/issues?q=is:issue%20is:open%20label:coverage%20label:bug%20label:backend:hlsl%20label:direction:input)):

- `ValueError: DirectX compute function '<probe>' cannot use return semantic 'NodeLaunch' with void return type` (7): `GetRemainingRecursionLevels`, `GroupNodeInputRecords.Count`, `GroupOrThreadNodeOutputRecords.OutputComplete`, `NodeOutput.GetGroupNodeOutputRecords`, `NodeOutput.GetThreadNodeOutputRecords`, `NodeOutput.IsValid`, `RWDispatchNodeInputRecord.FinishedCrossGroupSharing`
- `use of undeclared identifier '<feature>'` (5): `RWByteAddressBuffer.InterlockedAnd`, `RWByteAddressBuffer.InterlockedMax`, `RWByteAddressBuffer.InterlockedMin`, `RWByteAddressBuffer.InterlockedOr`, `RWByteAddressBuffer.InterlockedXor`
- `ValueError: DirectX texture operation 'textureCompareLod' requires a declared texture or image resource argument: obj` (4): `Texture1D.SampleCmpLevel`, `Texture1D.SampleCmpLevelZero`, `Texture1DArray.SampleCmpLevel`, `Texture1DArray.SampleCmpLevelZero`
- `ValueError: DirectX texture operation 'textureCompare' requires a declared texture or image resource argument: obj` (2): `Texture1D.SampleCmp`, `Texture1DArray.SampleCmp`
- `ValueError: DirectX texture operation 'textureCompareGrad' requires a declared texture or image resource argument: obj` (2): `Texture1D.SampleCmpGrad`, `Texture1DArray.SampleCmpGrad`
- `ValueError: DirectX texture query operation 'textureQueryLod' requires a 2D floating coordinate for Texture1DArray: a1 has type float` (2): `Texture1DArray.CalculateLevelOfDetail`, `Texture1DArray.CalculateLevelOfDetailUnclamped`
- `ValueError: DirectX texture query operation 'textureQueryLod' requires a 3D floating coordinate for Texture2DArray: a1 has type float2` (2): `Texture2DArray.CalculateLevelOfDetail`, `Texture2DArray.CalculateLevelOfDetailUnclamped`
- `ValueError: DirectX texture query operation 'textureQueryLod' requires a 4D floating coordinate for TextureCubeArray: a1 has type float3` (2): `TextureCUBEArray.CalculateLevelOfDetail`, `TextureCUBEArray.CalculateLevelOfDetailUnclamped`
- `cannot <feature> from resource containing uint4` (2): `Texture2D.GatherRaw`, `Texture2DArray.GatherRaw`
- `no matching function for call to 'InterlockedAdd'`: `RWByteAddressBuffer.InterlockedAdd`
- `use of undeclared identifier 'atomicCompareExchange'`: `RWByteAddressBuffer.InterlockedCompareExchange`
- `use of undeclared identifier 'atomicExchange'`: `RWByteAddressBuffer.InterlockedExchange`
- `use of undeclared identifier 'gl_HelperInvocation'`: `IsHelperLane`

Refused, by diagnostic:

- `project.translate.undeclared-type: UndeclaredTypeError: type 'dx_HitObject' is not declared: it is neither a CrossGL built-in type nor a structure, enumeration, type alias or generic parameter of the program (target directx) [project.translate.undeclared-type]` (28): `DxHitObject.FromRayQuery`, `DxHitObject.GetAttributes`, `DxHitObject.GetGeometryIndex`, `DxHitObject.GetHitKind`, `DxHitObject.GetInstanceID`, `DxHitObject.GetInstanceIndex`, `DxHitObject.GetObjectRayDirection`, `DxHitObject.GetObjectRayOrigin`, `DxHitObject.GetObjectToWorld3x4`, `DxHitObject.GetObjectToWorld4x3`, `DxHitObject.GetPrimitiveIndex`, `DxHitObject.GetRayFlags`, `DxHitObject.GetRayTCurrent`, `DxHitObject.GetRayTMin`, `DxHitObject.GetShaderTableIndex`, `DxHitObject.GetWorldRayDirection`, `DxHitObject.GetWorldRayOrigin`, `DxHitObject.GetWorldToObject3x4`, `DxHitObject.GetWorldToObject4x3`, `DxHitObject.Invoke`, `DxHitObject.IsHit`, `DxHitObject.IsMiss`, `DxHitObject.IsNop`, `DxHitObject.LoadLocalRootTableConstant`, `DxHitObject.MakeMiss`, `DxHitObject.MakeNop`, `DxHitObject.SetShaderTableIndex`, `DxHitObject.TraceRay`
- `project.translate.undeclared-type: UndeclaredTypeError: type 'SAMPLER_FEEDBACK_MIN_MIP' is not declared: it is neither a CrossGL built-in type nor a structure, enumeration, type alias or generic parameter of the program (target directx) [project.translate.undeclared-type]` (8): `FeedbackTexture2D.WriteSamplerFeedback`, `FeedbackTexture2D.WriteSamplerFeedbackBias`, `FeedbackTexture2D.WriteSamplerFeedbackGrad`, `FeedbackTexture2D.WriteSamplerFeedbackLevel`, `FeedbackTexture2DArray.WriteSamplerFeedback`, `FeedbackTexture2DArray.WriteSamplerFeedbackBias`, `FeedbackTexture2DArray.WriteSamplerFeedbackGrad`, `FeedbackTexture2DArray.WriteSamplerFeedbackLevel`
- `project.translate.directx-import-unsupported: DirectXImportRefused: CrossGL cannot represent N HLSL construct: DirectX texture overload extras for <feature>: CrossGL has no operand for LOD bias (target cgl) [project.translate.directx-import-unsupported]` (6): `Texture1D.SampleCmpBias`, `Texture1DArray.SampleCmpBias`, `Texture2D.SampleCmpBias`, `Texture2DArray.SampleCmpBias`, `TextureCUBE.SampleCmpBias`, `TextureCUBEArray.SampleCmpBias`
- `project.translate.directx-import-unsupported: DirectXImportRefused: CrossGL cannot represent N HLSL construct: DirectX texture gather-compare component for Texture2D: <feature> requires a compare-gather component selector; CrossGL only represents the default/red compare gather (target cgl) [project.translate.directx-import-unsupported]` (3): `Texture2D.GatherCmpAlpha`, `Texture2D.GatherCmpBlue`, `Texture2D.GatherCmpGreen`
- `project.translate.directx-import-unsupported: DirectXImportRefused: CrossGL cannot represent N HLSL construct: DirectX texture gather-compare component for Texture2DArray: <feature> requires a compare-gather component selector; CrossGL only represents the default/red compare gather (target cgl) [project.translate.directx-import-unsupported]` (3): `Texture2DArray.GatherCmpAlpha`, `Texture2DArray.GatherCmpBlue`, `Texture2DArray.GatherCmpGreen`
- `project.translate.directx-import-unsupported: DirectXImportRefused: CrossGL cannot represent N HLSL construct: DirectX texture gather-compare component for TextureCube: <feature> requires a compare-gather component selector; CrossGL only represents the default/red compare gather (target cgl) [project.translate.directx-import-unsupported]` (3): `TextureCUBE.GatherCmpAlpha`, `TextureCUBE.GatherCmpBlue`, `TextureCUBE.GatherCmpGreen`
- `project.translate.directx-import-unsupported: DirectXImportRefused: CrossGL cannot represent N HLSL construct: DirectX texture gather-compare component for TextureCubeArray: <feature> requires a compare-gather component selector; CrossGL only represents the default/red compare gather (target cgl) [project.translate.directx-import-unsupported]` (3): `TextureCUBEArray.GatherCmpAlpha`, `TextureCUBEArray.GatherCmpBlue`, `TextureCUBEArray.GatherCmpGreen`
- `project.translate.undeclared-type: UndeclaredTypeError: type 'EmptyNodeOutput' is not declared: it is neither a CrossGL built-in type nor a structure, enumeration, type alias or generic parameter of the program (target directx) [project.translate.undeclared-type]` (3): `EmptyNodeOutput.GroupIncrementOutputCount`, `EmptyNodeOutput.IsValid`, `EmptyNodeOutput.ThreadIncrementOutputCount`
- `project.translate.directx-import-unsupported: DirectXImportRefused: CrossGL cannot represent N HLSL construct: DirectX texture sample-position query for RWTexture2DMS: GetSamplePosition is only available on sampled multisample textures (target cgl) [project.translate.directx-import-unsupported]`: `RWTexture2DMS.GetSamplePosition`
- `project.translate.directx-import-unsupported: DirectXImportRefused: CrossGL cannot represent N HLSL construct: DirectX texture sample-position query for RWTexture2DMSArray: GetSamplePosition is only available on sampled multisample textures (target cgl) [project.translate.directx-import-unsupported]`: `RWTexture2DMSArray.GetSamplePosition`
- `project.translate.directx-import-unsupported: DirectXImportRefused: CrossGL cannot represent N HLSL construct: DirectX tiled-resource status check CheckAccessFullyMapped: CrossGL has no residency status, so the mapped result is unknown (target cgl) [project.translate.directx-import-unsupported]`: `CheckAccessFullyMapped`
- `project.translate.undeclared-type: UndeclaredTypeError: type 'EmptyNodeInput' is not declared: it is neither a CrossGL built-in type nor a structure, enumeration, type alias or generic parameter of the program (target directx) [project.translate.undeclared-type]`: `EmptyNodeInput.Count`

</details>

<details>
<summary>11 untested: function 5, method 6</summary>

- `needs shader model 6.10` (10): `ClusterID`, `DxHitObject.ClusterID`, `DxHitObject.TriangleObjectPosition`, `GetGroupWaveCount`, `GetGroupWaveIndex`, `RayQuery.CandidateClusterID`, `RayQuery.CandidateTriangleObjectPosition`, `RayQuery.CommittedClusterID`, `RayQuery.CommittedTriangleObjectPosition`, `TriangleObjectPosition`
- `DXC compiles printf only to SPIR-V; DXIL has no print operation`: `printf`

</details>

### Metal output

<details>
<summary>17 gaps: 12 refused, 5 wrong</summary>

Wrong, by cause ([open bug issues](https://github.com/CrossGL/crosstl/issues?q=is:issue%20is:open%20label:coverage%20label:bug%20label:backend:metal%20label:direction:output)):

- `no matching function for call to '<feature>'` (5): `atomic_fetch_sub`, `atomic_load`, `atomic_max`, `atomic_min`, `atomic_store`

Refused, by diagnostic:

- `CrossGL has no spelling for this operation: CrossGL has no operation that takes a memory order or memory flags` (7): `atomic_fetch_sub_explicit`, `atomic_load_explicit`, `atomic_max_explicit`, `atomic_min_explicit`, `atomic_store_explicit`, `atomic_thread_fence`, `simdgroup_barrier`
- `CrossGL has no type for Metal's quad_vote mask` (2): `quad_active_threads_mask`, `quad_ballot`
- `CrossGL has no type for Metal's simd_vote mask`: `simd_active_threads_mask`
- `project.translate.metal-lowering-unsupported: MetalLoweringRefused: Metal cannot lower N construct: bounding_box intersection stage: a void entry that reports hits with ReportHit has no lowering to a Metal intersection function result yet (target metal) [project.translate.metal-lowering-unsupported]`: `intersection`
- `project.translate.metal-stage-unsupported: MetalStageUnsupportedError: Metal evaluates tessellated patches in a [[patch]] post-tessellation vertex function that reads its control points and per-patch data through [[stage_in]] (Metal Shading Language Specification N.N.N); lowering a tessellation evaluation (domain) stage to that function is not implemented (stage tessel`: `tessellation-evaluation`

</details>

### Metal input

<details>
<summary>25 gaps: 3 refused, 22 wrong</summary>

Wrong, by cause ([open bug issues](https://github.com/CrossGL/crosstl/issues?q=is:issue%20is:open%20label:coverage%20label:bug%20label:backend:metal%20label:direction:input)):

- `no matching function for call to '<feature>'` (16): `atomic_compare_exchange_weak`, `atomic_exchange`, `atomic_fetch_add`, `atomic_fetch_and`, `atomic_fetch_max`, `atomic_fetch_min`, `atomic_fetch_or`, `atomic_fetch_sub`, `atomic_fetch_sub_explicit`, `atomic_fetch_xor`, `atomic_load`, `atomic_max`, `atomic_max_explicit`, `atomic_min`, `atomic_min_explicit`, `atomic_store`
- `use of undeclared identifier 'vote_t'; did you mean 'metal::quad_vote::vote_t'?` (2): `quad_active_threads_mask`, `quad_ballot`
- `use of undeclared identifier 'vote_t'; did you mean 'metal::simd_vote::vote_t'?` (2): `simd_active_threads_mask`, `simd_ballot`
- `'intersection' attribute cannot be applied to types`: `intersection`
- `use of undeclared identifier 'make_filled_matrix_u3cfloat_u2c3_u3e'`: `make_filled_matrix`

Refused, by diagnostic:

- `project.translate.metal-indexed-component-type-unresolved: MetalIndexedComponentTypeResolutionError: Cannot resolve Metal indexed component type for 'patch_control_point<ControlPoint>' (aggregate-subscript) at index 'N': the user-defined aggregate subscript result type cannot be inferred`: `tessellation-evaluation`
- `project.translate.metal-parse-failed: MetalParserError: Expected type, got MESH in function probe at line N, column N`: `mesh`
- `project.translate.undeclared-type: UndeclaredTypeError: type 'mesh_grid_properties' is not declared: it is neither a CrossGL built-in type nor a structure, enumeration, type alias or generic parameter of the program (target metal) [project.translate.undeclared-type]`: `task`

</details>

### Mojo output

<details>
<summary>112 gaps: 1 refused, 111 wrong</summary>

Wrong, by cause ([open bug issues](https://github.com/CrossGL/crosstl/issues?q=is:issue%20is:open%20label:coverage%20label:bug%20label:backend:mojo%20label:direction:output)):

- `global variables are not supported; move this into a function body or use 'comptime' to declare a constant` (110): `abs`, `acos`, `acosh`, `align_down`, `align_up`, `array`, `array-of-structs`, `asin`, `asinh`, `atan`, `atan2`, `atanh`, `bit_not`, `bit_reverse`, `bit_width`, `break`, `byte_swap`, `cbrt`, `ceil`, `ceildiv`, `clamp`, `comb`, `compute`, `conditional-expression`, `continue`, `copysign`, `cos`, `cosh`, `count_leading_zeros`, `count_trailing_zeros`, `double`, `early-return`, `erf`, `erfc`, `exp`, `exp2`, `expm1`, `factorial`, `floor`, `fma`, `for-loop`, `frexp`, `function-call`, `gamma`, `gcd`, `half`, `hypot`, `if-else`, `inf`, `inout-parameter`, `int16`, `int64`, `int8`, `iota`, `isclose`, `isfinite`, `isinf`, `isnan`, `j0`, `j1`, `lcm`, `ldexp`, `lgamma`, `local-invocation-id`, `local-invocation-index`, `log`, `log10`, `log1p`, `log2`, `log2_ceil`, `log2_floor`, `logb`, `max`, `min`, `modf`, `nan`, `next_power_of_two`, `nextafter`, `num-workgroups`, `perm`, `pop_count`, `pow`, `prev_power_of_two`, `read-only-buffer`, `recip`, `remainder`, `rotate_bits_left`, `rotate_bits_right`, `round`, `rsqrt`, `scalb`, `shared-memory`, `short-circuit-logic`, `sin`, `sinh`, `sqrt`, `struct`, `swizzle`, `tan`, `tanh`, `trunc`, `uint16`, `uint64`, `uint8`, `ulp`, `uniform-buffer`, `while-loop`, `workgroup-id`, `y0`, `y1`
- `expressions must not appear at file scope; move this into a function body`: `global-constant`

Refused, by diagnostic:

- `CrossGL has no spelling for this operation: no CrossGL function returns a quotient and a remainder together`: `divmod`

</details>

### Mojo input

<details>
<summary>112 gaps: 112 refused, 0 wrong</summary>

Refused, by diagnostic:

- `project.translate.undeclared-type: UndeclaredTypeError: type 'MutAnyOrigin' is not declared: it is neither a CrossGL built-in type nor a structure, enumeration, type alias or generic parameter of the program (target mojo) [project.translate.undeclared-type]` (112): `abs`, `acos`, `acosh`, `align_down`, `align_up`, `array`, `array-of-structs`, `asin`, `asinh`, `atan`, `atan2`, `atanh`, `bit_not`, `bit_reverse`, `bit_width`, `break`, `byte_swap`, `cbrt`, `ceil`, `ceildiv`, `clamp`, `comb`, `compute`, `conditional-expression`, `continue`, `copysign`, `cos`, `cosh`, `count_leading_zeros`, `count_trailing_zeros`, `divmod`, `double`, `early-return`, `erf`, `erfc`, `exp`, `exp2`, `expm1`, `factorial`, `floor`, `fma`, `for-loop`, `frexp`, `function-call`, `gamma`, `gcd`, `global-constant`, `half`, `hypot`, `if-else`, `inf`, `inout-parameter`, `int16`, `int64`, `int8`, `iota`, `isclose`, `isfinite`, `isinf`, `isnan`, `j0`, `j1`, `lcm`, `ldexp`, `lgamma`, `local-invocation-id`, `local-invocation-index`, `log`, `log10`, `log1p`, `log2`, `log2_ceil`, `log2_floor`, `logb`, `max`, `min`, `modf`, `nan`, `next_power_of_two`, `nextafter`, `num-workgroups`, `perm`, `pop_count`, `pow`, `prev_power_of_two`, `read-only-buffer`, `recip`, `remainder`, `rotate_bits_left`, `rotate_bits_right`, `round`, `rsqrt`, `scalb`, `shared-memory`, `short-circuit-logic`, `sin`, `sinh`, `sqrt`, `struct`, `swizzle`, `tan`, `tanh`, `trunc`, `uint16`, `uint64`, `uint8`, `ulp`, `uniform-buffer`, `while-loop`, `workgroup-id`, `y0`, `y1`

</details>

### OpenCL C input

<details>
<summary>332 gaps: 47 refused, 285 wrong</summary>

Wrong, by cause ([open bug issues](https://github.com/CrossGL/crosstl/issues?q=is:issue%20is:open%20label:coverage%20label:bug%20label:backend:opencl%20label:direction:input)):

- `use of undeclared identifier '<feature>'` (209): `abs_diff`, `add_sat`, `bit_reverse`, `bitfield_extract_signed`, `bitfield_extract_unsigned`, `bitfield_insert`, `bitselect`, `cbrt`, `clz`, `ctz`, `dot_4x8packed_ss_int`, `dot_4x8packed_su_int`, `dot_4x8packed_us_int`, `dot_4x8packed_uu_uint`, `dot_acc_sat`, `dot_acc_sat_4x8packed_ss_int`, `dot_acc_sat_4x8packed_su_int`, `dot_acc_sat_4x8packed_us_int`, `dot_acc_sat_4x8packed_uu_uint`, `erf`, `erfc`, `expm1`, `fast_distance`, `fast_length`, `fast_normalize`, `fdim`, `get_enqueued_local_size`, `get_enqueued_num_sub_groups`, `get_fence`, `get_global_id`, `get_global_linear_id`, `get_global_offset`, `get_global_size`, `get_group_id`, `get_image_array_size`, `get_image_channel_data_type`, `get_image_channel_order`, `get_image_depth`, `get_image_dim`, `get_image_height`, `get_image_num_mip_levels`, `get_image_width`, `get_local_id`, `get_local_size`, `get_max_sub_group_size`, `get_num_groups`, `get_num_sub_groups`, `get_sub_group_eq_mask`, `get_sub_group_ge_mask`, `get_sub_group_gt_mask`, `get_sub_group_id`, `get_sub_group_le_mask`, `get_sub_group_local_id`, `get_sub_group_lt_mask`, `get_sub_group_size`, `get_work_dim`, `hadd`, `half_cos`, `half_divide`, `half_exp`, `half_exp10`, `half_exp2`, `half_log`, `half_log10`, `half_log2`, `half_powr`, `half_recip`, `half_rsqrt`, `half_sin`, `half_sqrt`, `half_tan`, `hypot`, `ilogb`, `isequal`, `isgreater`, `isgreaterequal`, `isless`, `islessequal`, `islessgreater`, `isnotequal`, `isordered`, `isunordered`, `lgamma`, `log1p`, `logb`, `mad24`, `mad_hi`, `mad_sat`, `maxmag`, `minmag`, `mul24`, `mul_hi`, `nan`, `native_cos`, `native_divide`, `native_exp`, `native_exp10`, `native_exp2`, `native_log`, `native_log10`, `native_log2`, `native_powr`, `native_recip`, `native_rsqrt`, `native_sin`, `native_sqrt`, `native_tan`, `nextafter`, `popcount`, `pown`, `read_imageh`, `read_imagei`, `read_imageui`, `remainder`, `rhadd`, `rootn`, `rotate`, `shuffle`, `shuffle2`, `sub_group_all`, `sub_group_any`, `sub_group_ballot`, `sub_group_ballot_bit_count`, `sub_group_ballot_bit_extract`, `sub_group_ballot_exclusive_scan`, `sub_group_ballot_find_lsb`, `sub_group_ballot_find_msb`, `sub_group_ballot_inclusive_scan`, `sub_group_broadcast`, `sub_group_broadcast_first`, `sub_group_clustered_reduce_add`, `sub_group_clustered_reduce_and`, `sub_group_clustered_reduce_logical_and`, `sub_group_clustered_reduce_logical_or`, `sub_group_clustered_reduce_logical_xor`, `sub_group_clustered_reduce_max`, `sub_group_clustered_reduce_min`, `sub_group_clustered_reduce_mul`, `sub_group_clustered_reduce_or`, `sub_group_clustered_reduce_xor`, `sub_group_clustered_rotate`, `sub_group_elect`, `sub_group_inverse_ballot`, `sub_group_non_uniform_all`, `sub_group_non_uniform_all_equal`, `sub_group_non_uniform_any`, `sub_group_non_uniform_broadcast`, `sub_group_non_uniform_reduce_add`, `sub_group_non_uniform_reduce_and`, `sub_group_non_uniform_reduce_logical_and`, `sub_group_non_uniform_reduce_logical_or`, `sub_group_non_uniform_reduce_logical_xor`, `sub_group_non_uniform_reduce_max`, `sub_group_non_uniform_reduce_min`, `sub_group_non_uniform_reduce_mul`, `sub_group_non_uniform_reduce_or`, `sub_group_non_uniform_reduce_xor`, `sub_group_non_uniform_scan_exclusive_add`, `sub_group_non_uniform_scan_exclusive_and`, `sub_group_non_uniform_scan_exclusive_logical_and`, `sub_group_non_uniform_scan_exclusive_logical_or`, `sub_group_non_uniform_scan_exclusive_logical_xor`, `sub_group_non_uniform_scan_exclusive_max`, `sub_group_non_uniform_scan_exclusive_min`, `sub_group_non_uniform_scan_exclusive_mul`, `sub_group_non_uniform_scan_exclusive_or`, `sub_group_non_uniform_scan_exclusive_xor`, `sub_group_non_uniform_scan_inclusive_add`, `sub_group_non_uniform_scan_inclusive_and`, `sub_group_non_uniform_scan_inclusive_logical_and`, `sub_group_non_uniform_scan_inclusive_logical_or`, `sub_group_non_uniform_scan_inclusive_logical_xor`, `sub_group_non_uniform_scan_inclusive_max`, `sub_group_non_uniform_scan_inclusive_min`, `sub_group_non_uniform_scan_inclusive_mul`, `sub_group_non_uniform_scan_inclusive_or`, `sub_group_non_uniform_scan_inclusive_xor`, `sub_group_reduce_add`, `sub_group_reduce_max`, `sub_group_reduce_min`, `sub_group_rotate`, `sub_group_scan_exclusive_add`, `sub_group_scan_exclusive_max`, `sub_group_scan_exclusive_min`, `sub_group_scan_inclusive_add`, `sub_group_scan_inclusive_max`, `sub_group_scan_inclusive_min`, `sub_group_shuffle`, `sub_group_shuffle_down`, `sub_group_shuffle_up`, `sub_group_shuffle_xor`, `sub_sat`, `tgamma`, `upsample`, `work_group_all`, `work_group_any`, `work_group_broadcast`, `work_group_reduce_add`, `work_group_reduce_max`, `work_group_reduce_min`, `work_group_scan_exclusive_add`, `work_group_scan_exclusive_max`, `work_group_scan_exclusive_min`, `work_group_scan_inclusive_add`, `work_group_scan_inclusive_max`, `work_group_scan_inclusive_min`, `write_imageh`, `write_imagei`, `write_imageui`
- `operator is not supported` (44): `atom_add`, `atom_and`, `atom_cmpxchg`, `atom_dec`, `atom_inc`, `atom_max`, `atom_min`, `atom_or`, `atom_sub`, `atom_xchg`, `atom_xor`, `atomic_compare_exchange_strong`, `atomic_compare_exchange_weak`, `atomic_init`, `atomic_load`, `atomic_load_explicit`, `atomic_store`, `atomic_store_explicit`, `fract`, `frexp`, `lgamma_r`, `modf`, `remquo`, `sincos`, `vload_half`, `vload_halfn`, `vloada_halfn`, `vloadn`, `vstore_half`, `vstore_half_rte`, `vstore_half_rtn`, `vstore_half_rtp`, `vstore_half_rtz`, `vstore_halfn`, `vstore_halfn_rte`, `vstore_halfn_rtn`, `vstore_halfn_rtp`, `vstore_halfn_rtz`, `vstorea_halfn`, `vstorea_halfn_rte`, `vstorea_halfn_rtn`, `vstorea_halfn_rtp`, `vstorea_halfn_rtz`, `vstoren`
- `ValueError: DirectX typed buffer atomic '<feature>' original argument must be scalar int, got memory_order_relaxed` (7): `atomic_exchange_explicit`, `atomic_fetch_add_explicit`, `atomic_fetch_and_explicit`, `atomic_fetch_max_explicit`, `atomic_fetch_min_explicit`, `atomic_fetch_or_explicit`, `atomic_fetch_xor_explicit`
- `use of undeclared identifier 'create_user_event'` (6): `capture_event_profiling_info`, `create_user_event`, `is_valid_event`, `release_event`, `retain_event`, `set_user_event_status`
- `use of undeclared identifier 'CLK_LOCAL_MEM_FENCE'` (4): `atomic_work_item_fence`, `read_mem_fence`, `sub_group_barrier`, `write_mem_fence`
- `use of undeclared identifier 'atomicSub'` (3): `atomic_dec`, `atomic_fetch_sub`, `atomic_sub`
- `ValueError: DirectX typed buffer atomic 'atomicCompareExchange' requires N or N argument(s), got N` (2): `atomic_compare_exchange_strong_explicit`, `atomic_compare_exchange_weak_explicit`
- `use of undeclared identifier 'get_local_linear_id'` (2): `get_local_linear_id`, `local-invocation-index`
- `use of undeclared identifier 'ptr'` (2): `async_work_group_strided_copy`, `prefetch`
- `use of undeclared identifier 'read_imagef'` (2): `read_imagef`, `sampled-texture`
- `use of undeclared identifier 'write_imagef'` (2): `storage-image`, `write_imagef`
- `cannot initialize a variable of type 'vector<float, 4>' with an rvalue of type 'vector<float, 3>'`: `cross`
- `use of undeclared identifier 'memory_order_relaxed'`: `atomic_fetch_sub_explicit`

Refused, by diagnostic:

- `project.translate.opencl-conversion-unsupported: OpenCLConversionUnsupportedError: <feature> saturates or rounds to a N-bit integer, which the convert_sat rule family does not cover [project.translate.opencl-conversion-unsupported]` (16): `convert_long_rte`, `convert_long_rtn`, `convert_long_rtp`, `convert_long_sat`, `convert_long_sat_rte`, `convert_long_sat_rtn`, `convert_long_sat_rtp`, `convert_long_sat_rtz`, `convert_ulong_rte`, `convert_ulong_rtn`, `convert_ulong_rtp`, `convert_ulong_sat`, `convert_ulong_sat_rte`, `convert_ulong_sat_rtn`, `convert_ulong_sat_rtp`, `convert_ulong_sat_rtz`
- `project.translate.directx-metal-math-function-unsupported: MetalMathFunctionUnsupportedError: MSL <feature> is exact at integers and half-integers, which directx has no function for, and CrossTL does not lower it with an exact range reduction yet (target directx) [project.translate.directx-metal-math-function-unsupported]` (7): `acospi`, `asinpi`, `atan2pi`, `atanpi`, `cospi`, `sinpi`, `tanpi`
- `project.translate.undeclared-type: UndeclaredTypeError: type 'atomic_flag' is not declared: it is neither a CrossGL built-in type nor a structure, enumeration, type alias or generic parameter of the program (target directx) [project.translate.undeclared-type]` (4): `atomic_flag_clear`, `atomic_flag_clear_explicit`, `atomic_flag_test_and_set`, `atomic_flag_test_and_set_explicit`
- `project.translate.opencl-conversion-unsupported: OpenCLConversionUnsupportedError: <feature> rounds to a floating-point type in mode _rtn; CrossGL conversions to floating-point types round to nearest even [project.translate.opencl-conversion-unsupported]` (3): `convert_double_rtn`, `convert_float_rtn`, `convert_half_rtn`
- `project.translate.opencl-conversion-unsupported: OpenCLConversionUnsupportedError: <feature> rounds to a floating-point type in mode _rtp; CrossGL conversions to floating-point types round to nearest even [project.translate.opencl-conversion-unsupported]` (3): `convert_double_rtp`, `convert_float_rtp`, `convert_half_rtp`
- `project.translate.opencl-conversion-unsupported: OpenCLConversionUnsupportedError: <feature> rounds to a floating-point type in mode _rtz; CrossGL conversions to floating-point types round to nearest even [project.translate.opencl-conversion-unsupported]` (3): `convert_double_rtz`, `convert_float_rtz`, `convert_half_rtz`
- `project.translate.undeclared-type: UndeclaredTypeError: type 'ndrange_t' is not declared: it is neither a CrossGL built-in type nor a structure, enumeration, type alias or generic parameter of the program (target directx) [project.translate.undeclared-type]` (3): `ndrange_1D`, `ndrange_2D`, `ndrange_3D`
- `project.translate.undeclared-type: UndeclaredTypeError: type 'queue_t' is not declared: it is neither a CrossGL built-in type nor a structure, enumeration, type alias or generic parameter of the program (target directx) [project.translate.undeclared-type]` (2): `enqueue_marker`, `get_default_queue`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to directx because target lowering would produce invalid artifacts: OpenCL event/local-memory builtins require semantics not preserved by target lowering: async_work_group_copy`: `async_work_group_copy`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to directx because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: bump(a: ptr<f32>)`: `inout-parameter`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to directx because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: split(a: ptr<f32>), split(b: ptr<f32>)`: `out-parameter`
- `project.translate.directx-contextual-conversion-unrepresentable: DirectXContextualConversionError: DirectX canonical select requires a resolved boolean scalar or vector condition`: `select`
- `project.translate.undeclared-type: UndeclaredTypeError: type 'image2d_msaa_t' is not declared: it is neither a CrossGL built-in type nor a structure, enumeration, type alias or generic parameter of the program (target directx) [project.translate.undeclared-type]`: `get_image_num_samples`
- `project.translate.undeclared-type: UndeclaredTypeError: type 'reserve_id_t' is not declared: it is neither a CrossGL built-in type nor a structure, enumeration, type alias or generic parameter of the program (target directx) [project.translate.undeclared-type]`: `is_valid_reserve_id`

</details>

### Rust output

<details>
<summary>361 gaps: 12 refused, 349 wrong</summary>

Wrong, by cause ([open bug issues](https://github.com/CrossGL/crosstl/issues?q=is:issue%20is:open%20label:coverage%20label:bug%20label:backend:rust%20label:direction:output)):

- `error[E0463]: can't find crate for 'std'` (343): `Derivative::dfdx`, `Derivative::dfdx_coarse`, `Derivative::dfdx_fine`, `Derivative::dfdy`, `Derivative::dfdy_coarse`, `Derivative::dfdy_fine`, `Derivative::fwidth`, `Derivative::fwidth_coarse`, `Derivative::fwidth_fine`, `Float::abs`, `Float::acos`, `Float::acosh`, `Float::asin`, `Float::asinh`, `Float::atan`, `Float::atan2`, `Float::atanh`, `Float::cbrt`, `Float::ceil`, `Float::clamp`, `Float::copysign`, `Float::cos`, `Float::cosh`, `Float::epsilon`, `Float::exp`, `Float::exp2`, `Float::exp_m1`, `Float::floor`, `Float::infinity`, `Float::is_finite`, `Float::is_infinite`, `Float::is_nan`, `Float::is_normal`, `Float::is_sign_negative`, `Float::is_sign_positive`, `Float::is_subnormal`, `Float::ln`, `Float::ln_1p`, `Float::log10`, `Float::log2`, `Float::max`, `Float::max_value`, `Float::min`, `Float::min_positive_value`, `Float::min_value`, `Float::mul_add`, `Float::nan`, `Float::neg_infinity`, `Float::neg_zero`, `Float::powf`, `Float::powi`, `Float::recip`, `Float::round`, `Float::signum`, `Float::sin`, `Float::sin_cos`, `Float::sinh`, `Float::sqrt`, `Float::tan`, `Float::tanh`, `Float::to_degrees`, `Float::to_radians`, `Float::trunc`, `Vec3::abs`, `Vec3::abs_diff_eq`, `Vec3::angle_between`, `Vec3::angle_to`, `Vec3::any_orthogonal_vector`, `Vec3::any_orthonormal_vector`, `Vec3::as_dvec3`, `Vec3::as_ivec3`, `Vec3::as_uvec3`, `Vec3::ceil`, `Vec3::clamp`, `Vec3::clamp_length`, `Vec3::clamp_length_max`, `Vec3::clamp_length_min`, `Vec3::cmpeq`, `Vec3::cmpge`, `Vec3::cmpgt`, `Vec3::cmple`, `Vec3::cmplt`, `Vec3::cmpne`, `Vec3::copysign`, `Vec3::cos`, `Vec3::cross`, `Vec3::distance`, `Vec3::distance_squared`, `Vec3::div_euclid`, `Vec3::dot`, `Vec3::dot_into_vec`, `Vec3::element_product`, `Vec3::element_sum`, `Vec3::exp`, `Vec3::exp2`, `Vec3::extend`, `Vec3::floor`, `Vec3::fract_gl`, `Vec3::from_array`, `Vec3::from_homogeneous`, `Vec3::from_slice`, `Vec3::is_finite`, `Vec3::is_finite_mask`, `Vec3::is_nan`, `Vec3::is_nan_mask`, `Vec3::is_negative_bitmask`, `Vec3::is_negative_mask`, `Vec3::is_normalized`, `Vec3::length`, `Vec3::length_recip`, `Vec3::length_squared`, `Vec3::lerp`, `Vec3::lerp_monotonic`, `Vec3::ln`, `Vec3::log2`, `Vec3::max`, `Vec3::max_element`, `Vec3::midpoint`, `Vec3::min`, `Vec3::min_element`, `Vec3::move_towards`, `Vec3::mul_add`, `Vec3::new`, `Vec3::normalize`, `Vec3::normalize_or`, `Vec3::normalize_or_zero`, `Vec3::powf`, `Vec3::project_onto`, `Vec3::project_onto_normalized`, `Vec3::recip`, `Vec3::reflect`, `Vec3::refract`, `Vec3::reject_from`, `Vec3::reject_from_normalized`, `Vec3::rem_euclid`, `Vec3::rotate_axis`, `Vec3::rotate_towards`, `Vec3::rotate_x`, `Vec3::rotate_y`, `Vec3::rotate_z`, `Vec3::round`, `Vec3::saturate`, `Vec3::select`, `Vec3::signum`, `Vec3::sin`, `Vec3::sin_cos`, `Vec3::slerp`, `Vec3::smoothstep`, `Vec3::splat`, `Vec3::sqrt`, `Vec3::step`, `Vec3::to_homogeneous`, `Vec3::to_vec3a`, `Vec3::trunc`, `Vec3::truncate`, `Vec3::with_x`, `Vec3::with_y`, `Vec3::with_z`, `all`, `all_memory_barrier`, `all_memory_barrier_with_group_sync`, `any`, `any-hit`, `array`, `array-of-structs`, `atomic_and`, `atomic_compare_exchange`, `atomic_exchange`, `atomic_f_add`, `atomic_f_max`, `atomic_f_min`, `atomic_i_add`, `atomic_i_decrement`, `atomic_i_increment`, `atomic_i_sub`, `atomic_load`, `atomic_or`, `atomic_s_max`, `atomic_s_min`, `atomic_store`, `atomic_u_max`, `atomic_u_min`, `atomic_xor`, `break`, `callable`, `closest-hit`, `compute`, `conditional-expression`, `continue`, `control_barrier`, `demote_to_helper_invocation`, `device_memory_barrier`, `device_memory_barrier_with_group_sync`, `discard`, `double`, `early-return`, `emit_mesh_tasks_ext`, `emit_mesh_tasks_ext_payload`, `execute_callable`, `for-loop`, `fragment`, `function-call`, `global-constant`, `half`, `if-else`, `ignore_intersection`, `inout-parameter`, `int16`, `int64`, `int8`, `intersection`, `is_helper_invocation`, `kill`, `local-invocation-id`, `local-invocation-index`, `matrix-vector-product`, `memory_barrier`, `mesh`, `miss`, `num-workgroups`, `out-parameter`, `ray-generation`, `read-only-buffer`, `read_clock_khr`, `read_clock_uvec2_khr`, `report_intersection`, `runtime-array-length`, `sampled-texture`, `set_mesh_outputs_ext`, `shared-memory`, `short-circuit-logic`, `signed_max`, `signed_min`, `storage-image`, `struct`, `subgroup_all`, `subgroup_all_equal`, `subgroup_and`, `subgroup_any`, `subgroup_ballot`, `subgroup_ballot_bit_count`, `subgroup_ballot_bit_extract`, `subgroup_ballot_exclusive_bit_count`, `subgroup_ballot_find_lsb`, `subgroup_ballot_find_msb`, `subgroup_ballot_inclusive_bit_count`, `subgroup_barrier`, `subgroup_broadcast`, `subgroup_broadcast_const`, `subgroup_broadcast_first`, `subgroup_clustered_and`, `subgroup_clustered_f_add`, `subgroup_clustered_f_max`, `subgroup_clustered_f_min`, `subgroup_clustered_f_mul`, `subgroup_clustered_i_add`, `subgroup_clustered_i_mul`, `subgroup_clustered_logical_and`, `subgroup_clustered_logical_or`, `subgroup_clustered_logical_xor`, `subgroup_clustered_or`, `subgroup_clustered_s_max`, `subgroup_clustered_s_min`, `subgroup_clustered_u_max`, `subgroup_clustered_u_min`, `subgroup_clustered_xor`, `subgroup_elect`, `subgroup_exclusive_and`, `subgroup_exclusive_f_add`, `subgroup_exclusive_f_max`, `subgroup_exclusive_f_min`, `subgroup_exclusive_f_mul`, `subgroup_exclusive_i_add`, `subgroup_exclusive_i_mul`, `subgroup_exclusive_logical_and`, `subgroup_exclusive_logical_or`, `subgroup_exclusive_logical_xor`, `subgroup_exclusive_or`, `subgroup_exclusive_s_max`, `subgroup_exclusive_s_min`, `subgroup_exclusive_u_max`, `subgroup_exclusive_u_min`, `subgroup_exclusive_xor`, `subgroup_f_add`, `subgroup_f_max`, `subgroup_f_min`, `subgroup_f_mul`, `subgroup_i_add`, `subgroup_i_mul`, `subgroup_inclusive_and`, `subgroup_inclusive_f_add`, `subgroup_inclusive_f_max`, `subgroup_inclusive_f_min`, `subgroup_inclusive_f_mul`, `subgroup_inclusive_i_add`, `subgroup_inclusive_i_mul`, `subgroup_inclusive_logical_and`, `subgroup_inclusive_logical_or`, `subgroup_inclusive_logical_xor`, `subgroup_inclusive_or`, `subgroup_inclusive_s_max`, `subgroup_inclusive_s_min`, `subgroup_inclusive_u_max`, `subgroup_inclusive_u_min`, `subgroup_inclusive_xor`, `subgroup_inverse_ballot`, `subgroup_logical_and`, `subgroup_logical_or`, `subgroup_logical_xor`, `subgroup_memory_barrier`, `subgroup_memory_barrier_buffer`, `subgroup_memory_barrier_image`, `subgroup_memory_barrier_shared`, `subgroup_or`, `subgroup_quad_broadcast`, `subgroup_quad_swap`, `subgroup_s_max`, `subgroup_s_min`, `subgroup_shuffle`, `subgroup_shuffle_down`, `subgroup_shuffle_up`, `subgroup_shuffle_xor`, `subgroup_u_max`, `subgroup_u_min`, `subgroup_xor`, `switch`, `swizzle`, `task`, `terminate_ray`, `tessellation-control`, `uint16`, `uint64`, `uint8`, `uniform-buffer`, `unsigned_max`, `unsigned_min`, `vector_extract_dynamic`, `vector_insert_dynamic`, `vertex`, `while-loop`, `workgroup-id`, `workgroup_memory_barrier`, `workgroup_memory_barrier_with_group_sync`
- `ValueError: Rust geometry stage requires maxvertexcount attribute` (5): `emit_stream_vertex`, `emit_vertex`, `end_primitive`, `end_stream_primitive`, `geometry`
- `ValueError: Unsupported SV_Position struct return semantic 'DomainOutput.position' for Rust tessellation_evaluation stage; valid stage is vertex`: `tessellation-evaluation`

Refused, by diagnostic:

- `CrossGL has no spelling for this operation: Rust's fract is x - trunc(x); CrossGL's is x - floor(x)` (2): `Float::fract`, `Vec3::fract`
- `CrossGL has no spelling for this operation: Float::log takes a base; CrossGL's log is the natural logarithm`: `Float::log`
- `CrossGL has no spelling for this operation: no CrossGL function converts a vector to an array`: `Vec3::to_array`
- `CrossGL has no spelling for this operation: no CrossGL function returns a float's category`: `Float::classify`
- `CrossGL has no spelling for this operation: no CrossGL function returns a float's mantissa, exponent and sign as integers`: `Float::integer_decode`
- `CrossGL has no spelling for this operation: no CrossGL function returns a vector's direction and length together`: `Vec3::normalize_and_length`
- `CrossGL has no spelling for this operation: no CrossGL function returns an optional value`: `Vec3::try_normalize`
- `CrossGL has no spelling for this operation: no CrossGL function returns the index of a vector's largest element`: `Vec3::max_position`
- `CrossGL has no spelling for this operation: no CrossGL function returns the index of a vector's smallest element`: `Vec3::min_position`
- `CrossGL has no spelling for this operation: no CrossGL function returns two vectors orthonormal to a vector`: `Vec3::any_orthonormal_pair`
- `CrossGL has no spelling for this operation: no CrossGL function writes a vector's elements to an array`: `Vec3::write_to_slice`

</details>

<details>
<summary>3 untested: function 3</summary>

- `no probe for parameters of type F`: `Vec3::map`
- `the probe itself is rejected: error: Fdim not supported yet`: `Float::abs_sub`
- `the probe itself is rejected: error: Hypot not supported yet`: `Float::hypot`

</details>

### Rust input

<details>
<summary>360 gaps: 28 refused, 332 wrong</summary>

Wrong, by cause ([open bug issues](https://github.com/CrossGL/crosstl/issues?q=is:issue%20is:open%20label:coverage%20label:bug%20label:backend:rust%20label:direction:input)):

- `error[E0463]: can't find crate for 'std'` (303): `Derivative::dfdx`, `Derivative::dfdx_coarse`, `Derivative::dfdx_fine`, `Derivative::dfdy`, `Derivative::dfdy_coarse`, `Derivative::dfdy_fine`, `Derivative::fwidth`, `Derivative::fwidth_coarse`, `Derivative::fwidth_fine`, `Float::abs`, `Float::acos`, `Float::acosh`, `Float::asin`, `Float::asinh`, `Float::atan`, `Float::atan2`, `Float::atanh`, `Float::cbrt`, `Float::ceil`, `Float::clamp`, `Float::classify`, `Float::copysign`, `Float::cos`, `Float::cosh`, `Float::epsilon`, `Float::exp`, `Float::exp2`, `Float::exp_m1`, `Float::floor`, `Float::fract`, `Float::infinity`, `Float::is_finite`, `Float::is_infinite`, `Float::is_nan`, `Float::is_normal`, `Float::is_sign_negative`, `Float::is_sign_positive`, `Float::is_subnormal`, `Float::ln`, `Float::ln_1p`, `Float::log`, `Float::log10`, `Float::log2`, `Float::max`, `Float::max_value`, `Float::min`, `Float::min_positive_value`, `Float::min_value`, `Float::mul_add`, `Float::nan`, `Float::neg_infinity`, `Float::neg_zero`, `Float::powf`, `Float::powi`, `Float::recip`, `Float::round`, `Float::signum`, `Float::sin`, `Float::sinh`, `Float::sqrt`, `Float::tan`, `Float::tanh`, `Float::to_degrees`, `Float::to_radians`, `Float::trunc`, `Vec3::abs`, `Vec3::abs_diff_eq`, `Vec3::angle_between`, `Vec3::angle_to`, `Vec3::any_orthogonal_vector`, `Vec3::any_orthonormal_vector`, `Vec3::as_ivec3`, `Vec3::as_uvec3`, `Vec3::ceil`, `Vec3::clamp`, `Vec3::clamp_length`, `Vec3::clamp_length_max`, `Vec3::clamp_length_min`, `Vec3::cmpeq`, `Vec3::cmpge`, `Vec3::cmpgt`, `Vec3::cmple`, `Vec3::cmplt`, `Vec3::cmpne`, `Vec3::copysign`, `Vec3::cos`, `Vec3::cross`, `Vec3::distance`, `Vec3::distance_squared`, `Vec3::div_euclid`, `Vec3::dot`, `Vec3::dot_into_vec`, `Vec3::element_product`, `Vec3::element_sum`, `Vec3::exp`, `Vec3::exp2`, `Vec3::extend`, `Vec3::floor`, `Vec3::fract`, `Vec3::fract_gl`, `Vec3::from_array`, `Vec3::from_homogeneous`, `Vec3::from_slice`, `Vec3::is_finite`, `Vec3::is_finite_mask`, `Vec3::is_nan`, `Vec3::is_nan_mask`, `Vec3::is_negative_bitmask`, `Vec3::is_negative_mask`, `Vec3::is_normalized`, `Vec3::length`, `Vec3::length_recip`, `Vec3::length_squared`, `Vec3::lerp`, `Vec3::lerp_monotonic`, `Vec3::ln`, `Vec3::log2`, `Vec3::max`, `Vec3::max_element`, `Vec3::max_position`, `Vec3::midpoint`, `Vec3::min`, `Vec3::min_element`, `Vec3::min_position`, `Vec3::move_towards`, `Vec3::mul_add`, `Vec3::new`, `Vec3::normalize`, `Vec3::normalize_or`, `Vec3::normalize_or_zero`, `Vec3::powf`, `Vec3::project_onto`, `Vec3::project_onto_normalized`, `Vec3::recip`, `Vec3::reflect`, `Vec3::refract`, `Vec3::reject_from`, `Vec3::reject_from_normalized`, `Vec3::rem_euclid`, `Vec3::rotate_axis`, `Vec3::rotate_towards`, `Vec3::rotate_x`, `Vec3::rotate_y`, `Vec3::rotate_z`, `Vec3::round`, `Vec3::saturate`, `Vec3::select`, `Vec3::signum`, `Vec3::sin`, `Vec3::slerp`, `Vec3::smoothstep`, `Vec3::splat`, `Vec3::sqrt`, `Vec3::step`, `Vec3::to_array`, `Vec3::to_homogeneous`, `Vec3::to_vec3a`, `Vec3::trunc`, `Vec3::truncate`, `Vec3::try_normalize`, `Vec3::with_x`, `Vec3::with_y`, `Vec3::with_z`, `all`, `all_memory_barrier`, `all_memory_barrier_with_group_sync`, `any`, `any-hit`, `array`, `array-of-structs`, `break`, `callable`, `closest-hit`, `compute`, `conditional-expression`, `continue`, `demote_to_helper_invocation`, `device_memory_barrier`, `device_memory_barrier_with_group_sync`, `discard`, `double`, `early-return`, `emit_mesh_tasks_ext`, `emit_mesh_tasks_ext_payload`, `for-loop`, `fragment`, `function-call`, `global-constant`, `half`, `if-else`, `ignore_intersection`, `inout-parameter`, `int16`, `int64`, `int8`, `intersection`, `is_helper_invocation`, `kill`, `local-invocation-id`, `local-invocation-index`, `matrix-vector-product`, `miss`, `num-workgroups`, `out-parameter`, `ray-generation`, `read-only-buffer`, `report_intersection`, `runtime-array-length`, `sampled-texture`, `shared-memory`, `short-circuit-logic`, `signed_max`, `signed_min`, `storage-image`, `struct`, `subgroup_all`, `subgroup_all_equal`, `subgroup_and`, `subgroup_any`, `subgroup_ballot`, `subgroup_ballot_bit_count`, `subgroup_ballot_bit_extract`, `subgroup_ballot_exclusive_bit_count`, `subgroup_ballot_find_lsb`, `subgroup_ballot_find_msb`, `subgroup_ballot_inclusive_bit_count`, `subgroup_barrier`, `subgroup_broadcast`, `subgroup_broadcast_first`, `subgroup_elect`, `subgroup_exclusive_and`, `subgroup_exclusive_f_add`, `subgroup_exclusive_f_max`, `subgroup_exclusive_f_min`, `subgroup_exclusive_f_mul`, `subgroup_exclusive_i_add`, `subgroup_exclusive_i_mul`, `subgroup_exclusive_logical_and`, `subgroup_exclusive_logical_or`, `subgroup_exclusive_logical_xor`, `subgroup_exclusive_or`, `subgroup_exclusive_s_max`, `subgroup_exclusive_s_min`, `subgroup_exclusive_u_max`, `subgroup_exclusive_u_min`, `subgroup_exclusive_xor`, `subgroup_f_add`, `subgroup_f_max`, `subgroup_f_min`, `subgroup_f_mul`, `subgroup_i_add`, `subgroup_i_mul`, `subgroup_inclusive_and`, `subgroup_inclusive_f_add`, `subgroup_inclusive_f_max`, `subgroup_inclusive_f_min`, `subgroup_inclusive_f_mul`, `subgroup_inclusive_i_add`, `subgroup_inclusive_i_mul`, `subgroup_inclusive_logical_and`, `subgroup_inclusive_logical_or`, `subgroup_inclusive_logical_xor`, `subgroup_inclusive_or`, `subgroup_inclusive_s_max`, `subgroup_inclusive_s_min`, `subgroup_inclusive_u_max`, `subgroup_inclusive_u_min`, `subgroup_inclusive_xor`, `subgroup_inverse_ballot`, `subgroup_logical_and`, `subgroup_logical_or`, `subgroup_logical_xor`, `subgroup_memory_barrier`, `subgroup_memory_barrier_buffer`, `subgroup_memory_barrier_image`, `subgroup_memory_barrier_shared`, `subgroup_or`, `subgroup_quad_broadcast`, `subgroup_s_max`, `subgroup_s_min`, `subgroup_shuffle`, `subgroup_shuffle_down`, `subgroup_shuffle_up`, `subgroup_shuffle_xor`, `subgroup_u_max`, `subgroup_u_min`, `subgroup_xor`, `switch`, `swizzle`, `task`, `terminate_ray`, `uint16`, `uint64`, `uint8`, `uniform-buffer`, `unsigned_max`, `unsigned_min`, `vector_extract_dynamic`, `vector_insert_dynamic`, `while-loop`, `workgroup-id`, `workgroup_memory_barrier`, `workgroup_memory_barrier_with_group_sync`
- `expected one of '.', ';', '?', '}', or an operator, found ','` (18): `execute_callable`, `subgroup_broadcast_const`, `subgroup_clustered_and`, `subgroup_clustered_f_add`, `subgroup_clustered_f_max`, `subgroup_clustered_f_min`, `subgroup_clustered_f_mul`, `subgroup_clustered_i_add`, `subgroup_clustered_i_mul`, `subgroup_clustered_logical_and`, `subgroup_clustered_logical_or`, `subgroup_clustered_logical_xor`, `subgroup_clustered_or`, `subgroup_clustered_s_max`, `subgroup_clustered_s_min`, `subgroup_clustered_u_max`, `subgroup_clustered_u_min`, `subgroup_clustered_xor`
- `ValueError: Rust geometry stage requires maxvertexcount attribute` (5): `emit_stream_vertex`, `emit_vertex`, `end_primitive`, `end_stream_primitive`, `geometry`
- `ValueError: Rust mesh stage requires outputtopology attribute` (2): `mesh`, `set_mesh_outputs_ext`
- `ValueError: Rust tessellation_control stage requires domain attribute`: `tessellation-control`
- `ValueError: Rust tessellation_evaluation stage requires domain attribute`: `tessellation-evaluation`
- `ValueError: Unsupported gl_Position stage parameter semantic for Rust vertex stage; valid stage is fragment`: `vertex`
- `expected one of '.', ';', '?', 'else', or an operator, found ','`: `subgroup_quad_swap`

Refused, by diagnostic:

- `project.translate.crossgl-function-body-parse-failed: CrossGLFunctionBodyParseError: CrossGL could not parse the body of function '<probe>' at COMMA ',': Expected SEMICOLON, got COMMA ','` (18): `atomic_and`, `atomic_compare_exchange`, `atomic_exchange`, `atomic_f_add`, `atomic_f_max`, `atomic_f_min`, `atomic_i_add`, `atomic_i_decrement`, `atomic_i_increment`, `atomic_i_sub`, `atomic_load`, `atomic_or`, `atomic_s_max`, `atomic_s_min`, `atomic_store`, `atomic_u_max`, `atomic_u_min`, `atomic_xor`
- `project.translate.crossgl-function-body-parse-failed: CrossGLFunctionBodyParseError: CrossGL could not parse the body of function '<probe>' at FLOAT_NUMBER '.N': Expected SEMICOLON, got FLOAT_NUMBER '.N'` (5): `Float::integer_decode`, `Float::sin_cos`, `Vec3::any_orthonormal_pair`, `Vec3::normalize_and_length`, `Vec3::sin_cos`
- `project.translate.crossgl-function-body-parse-failed: CrossGLFunctionBodyParseError: CrossGL could not parse the body of function '<probe>' at AS 'as': Expected RBRACE, got AS 'as'` (4): `control_barrier`, `memory_barrier`, `read_clock_khr`, `read_clock_uvec2_khr`
- `project.translate.undeclared-type: UndeclaredTypeError: type 'DVec3' is not declared: it is neither a CrossGL built-in type nor a structure, enumeration, type alias or generic parameter of the program (target rust) [project.translate.undeclared-type]`: `Vec3::as_dvec3`

</details>

<details>
<summary>4 untested: function 4</summary>

- `no probe for parameters of type F`: `Vec3::map`
- `the probe itself is rejected: error: Fdim not supported yet`: `Float::abs_sub`
- `the probe itself is rejected: error: Hypot not supported yet`: `Float::hypot`
- `the probe itself is rejected: error: cannot cast between pointer types`: `Vec3::write_to_slice`

</details>

### Slang output

<details>
<summary>47 gaps: 9 refused, 38 wrong</summary>

Wrong, by cause ([open bug issues](https://github.com/CrossGL/crosstl/issues?q=is:issue%20is:open%20label:coverage%20label:bug%20label:backend:slang%20label:direction:output)):

- `unknown type name 'Ptr'` (15): `GetRenderTargetSampleCount`, `GetRenderTargetSamplePosition`, `InterlockedCompareExchangeFloatBitwise`, `MaybeReorderThread`, `Process2DQuadTessFactorsAvg`, `Process2DQuadTessFactorsMax`, `Process2DQuadTessFactorsMin`, `ProcessIsolineTessFactors`, `ProcessQuadTessFactorsAvg`, `ProcessQuadTessFactorsMax`, `ProcessQuadTessFactorsMin`, `ProcessTriTessFactorsAvg`, `ProcessTriTessFactorsMax`, `ProcessTriTessFactorsMin`, `dst`
- `error[E30015]: undefined identifier '<feature>'` (12): `InterlockedAdd`, `InterlockedAnd`, `InterlockedMax`, `InterlockedMin`, `InterlockedOr`, `InterlockedXor`, `countbits`, `firstbithigh`, `firstbitlow`, `fwidth_coarse`, `fwidth_fine`, `reversebits`
- `ValueError: Slang DispatchMesh payload argument requires a mesh stage @mesh_payload parameter` (2): `DispatchMesh`, `task`
- `error[E30015]: undefined identifier 'i16vec4'` (2): `unpackInt4x8ToInt16`, `unpack_s8s16`
- `error[E30015]: undefined identifier 'u16vec4'` (2): `unpackUint4x8ToUint16`, `unpack_u8u16`
- `ValueError: Slang ray_intersection ReportHit hit attribute argument must use a user-defined struct type, got int`: `ReportHit`
- `error[E30015]: undefined identifier 'gl_Position'`: `geometry`
- `error[E30015]: undefined identifier 'i16vec2'`: `asint16`
- `error[E30015]: undefined identifier 'u16vec2'`: `asuint16`
- `error[E30027]: member not found: 'length' is not a member of 'float'`: `runtime-array-length`

Refused, by diagnostic:

- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: Slang wave intrinsic: <feature> is not recognized by the Slang backend (target slang) [project.translate.slang-lowering-unsupported]` (6): `QuadAll`, `QuadAny`, `WaveActiveAllEqual`, `WaveActiveCountBits`, `WaveMultiPrefixCountBits`, `WavePrefixCountBits`
- `project.translate.slang-metal-math-function-unsupported: MetalMathFunctionUnsupportedError: MSL <feature> is exact at integers and half-integers, which slang has no function for, and CrossTL does not lower it with an exact range reduction yet (target slang) [project.translate.slang-metal-math-function-unsupported]` (3): `cospi`, `sinpi`, `tanpi`

</details>

<details>
<summary>16 untested: function 16</summary>

- `needs a target other than SPIR-V and HLSL` (11): `GetInstanceTraversable`, `GetTransformListHandle`, `GetTransformListSize`, `GetTraversableChild`, `GetTraversableInstanceId`, `GetTraversableObjectToWorld`, `GetTraversableTransformType`, `GetTraversableWorldToObject`, `cudaBlockDim`, `cudaBlockIdx`, `cudaThreadIdx`
- `no probe for parameters of type DescriptorHandle<T>`: `nonuniform`
- `no probe for parameters of type IOpaqueDescriptor`: `defaultGetDescriptorFromHandle`
- `the probe itself is rejected: error: line 12: Operand 3 of Decorate requires one of these capabilities: RayTracingMotionBlurNV`: `RayCurrentTime`
- `the probe itself is rejected: error: slangc emits /* debugBreak() not currently supported for HLSL */`: `debugBreak`
- `the probe itself is rejected: error: slangc exited with status -11 and no message`: `createDynamicObject`

</details>

### Slang input

<details>
<summary>49 gaps: 17 refused, 32 wrong</summary>

Wrong, by cause ([open bug issues](https://github.com/CrossGL/crosstl/issues?q=is:issue%20is:open%20label:coverage%20label:bug%20label:backend:slang%20label:direction:input)):

- `ValueError: Slang tessellation_control stage requires outputcontrolpoints` (11): `Process2DQuadTessFactorsAvg`, `Process2DQuadTessFactorsMax`, `Process2DQuadTessFactorsMin`, `ProcessIsolineTessFactors`, `ProcessQuadTessFactorsAvg`, `ProcessQuadTessFactorsMax`, `ProcessQuadTessFactorsMin`, `ProcessTriTessFactorsAvg`, `ProcessTriTessFactorsMax`, `ProcessTriTessFactorsMin`, `tessellation-control`
- `error[E30019]: type mismatch in expression: expected an expression of type 'vector<float,4>', got 'vector<float,3>'` (4): `ObjectToWorld`, `ObjectToWorld3x4`, `WorldToObject`, `WorldToObject3x4`
- `on vulkan-lavapipe as SPIR-V from slangc, the translation computes different results from the native probe` (3): `inout-parameter`, `matrix-vector-product`, `out-parameter`
- `ValueError: Slang DispatchMesh payload argument requires a mesh stage @mesh_payload parameter` (2): `DispatchMesh`, `task`
- `error[E30015]: undefined identifier '<feature>'` (2): `fwidth_coarse`, `fwidth_fine`
- `error[E30019]: type mismatch in expression: expected an expression of type 'vector<float,3>', got 'vector<float,4>'` (2): `ObjectToWorld4x3`, `WorldToObject4x3`
- `error[E38047]: mesh shader missing [outputtopology] attribute: mesh shader entry point '<probe>' must have an '[outputtopology("point"|"line"|"triangle")]' attribute` (2): `SetMeshOutputCounts`, `mesh`
- `invalid semantic 'OUT_COLOR' for ps N.N` (2): `GetRenderTargetSampleCount`, `GetRenderTargetSamplePosition`
- `ValueError: Slang geometry stage parameters must include an input primitive parameter qualified as point, line, triangle, lineadj, or triangleadj`: `geometry`
- `ValueError: Slang ray_intersection ReportHit hit attribute argument must use a user-defined struct type, got int`: `ReportHit`
- `ValueError: Slang tessellation_evaluation stage requires a domain attribute`: `tessellation-evaluation`
- `error[E30019]: type mismatch in expression: expected an expression of type 'vector<float,4>', got 'vector<float,2>'`: `GetLssPositionsAndRadii`

Refused, by diagnostic:

- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: Slang wave intrinsic: <feature> is not recognized by the Slang backend (target slang) [project.translate.slang-lowering-unsupported]` (6): `QuadAll`, `QuadAny`, `WaveActiveAllEqual`, `WaveActiveCountBits`, `WaveMultiPrefixCountBits`, `WavePrefixCountBits`
- `project.translate.undeclared-type: UndeclaredTypeError: type 'int8_t4_packed' is not declared: it is neither a CrossGL built-in type nor a structure, enumeration, type alias or generic parameter of the program (target slang) [project.translate.undeclared-type]` (4): `pack_clamp_s8`, `pack_s8`, `unpack_s8s16`, `unpack_s8s32`
- `project.translate.undeclared-type: UndeclaredTypeError: type 'uint8_t4_packed' is not declared: it is neither a CrossGL built-in type nor a structure, enumeration, type alias or generic parameter of the program (target slang) [project.translate.undeclared-type]` (4): `pack_clamp_u8`, `pack_u8`, `unpack_u8u16`, `unpack_u8u32`
- `project.translate.slang-metal-math-function-unsupported: MetalMathFunctionUnsupportedError: MSL <feature> is exact at integers and half-integers, which slang has no function for, and CrossTL does not lower it with an exact range reduction yet (target slang) [project.translate.slang-metal-math-function-unsupported]` (3): `cospi`, `sinpi`, `tanpi`

</details>

<details>
<summary>16 untested: function 16</summary>

- `needs a target other than SPIR-V and HLSL` (11): `GetInstanceTraversable`, `GetTransformListHandle`, `GetTransformListSize`, `GetTraversableChild`, `GetTraversableInstanceId`, `GetTraversableObjectToWorld`, `GetTraversableTransformType`, `GetTraversableWorldToObject`, `cudaBlockDim`, `cudaBlockIdx`, `cudaThreadIdx`
- `no probe for parameters of type DescriptorHandle<T>`: `nonuniform`
- `no probe for parameters of type IOpaqueDescriptor`: `defaultGetDescriptorFromHandle`
- `the probe itself is rejected: error: line 12: Operand 3 of Decorate requires one of these capabilities: RayTracingMotionBlurNV`: `RayCurrentTime`
- `the probe itself is rejected: error: slangc emits /* debugBreak() not currently supported for HLSL */`: `debugBreak`
- `the probe itself is rejected: error: slangc exited with status -11 and no message`: `createDynamicObject`

</details>

### SPIR-V output

<details>
<summary>9 gaps: 8 refused, 1 wrong</summary>

Wrong, by cause ([open bug issues](https://github.com/CrossGL/crosstl/issues?q=is:issue%20is:open%20label:coverage%20label:bug%20label:backend:spirv%20label:direction:output)):

- `on vulkan-lavapipe, the translation computes different results from the native probe`: `int8`

Refused, by diagnostic:

- `project.translate.vulkan-builtin-unsupported: SPIRVBuiltinLoweringError: Cannot lower call to '<feature>' to SPIR-V: it is neither a declared function nor a built-in with a SPIR-V lowering` (3): `InterpolateAtCentroid`, `InterpolateAtOffset`, `InterpolateAtSample`
- `CrossGL has no spelling for this operation: NaN-aware clamp has no GLSL built-in`: `NClamp`
- `CrossGL has no spelling for this operation: NaN-aware max has no GLSL built-in`: `NMax`
- `CrossGL has no spelling for this operation: NaN-aware min has no GLSL built-in`: `NMin`
- `project.translate.vulkan-tessellation-patch-unsupported: TessellationPatchUnsupportedError: domain entry point '<probe>' needs [domain("tri"|"quad"|"isoline")]; vulkan tessellation shaders exchange control points through per-vertex arrays, patch variables and gl_TessLevelOuter/gl_TessLevelInner (target vulkan) [project.translate.vulkan-tessellation-patch-unsupported]`: `tessellation-evaluation`
- `project.translate.vulkan-tessellation-patch-unsupported: TessellationPatchUnsupportedError: patch-constant function 'IdentifierNode(name=constants)' is not one function of the program; vulkan tessellation shaders exchange control points through per-vertex arrays, patch variables and gl_TessLevelOuter/gl_TessLevelInner (target vulkan) [project.translate.vulkan-tessellation-patch-unsupported`: `tessellation-control`

</details>

<details>
<summary>3 untested: function 3</summary>

- `alternative encoding of Frexp, chosen by the compiler`: `FrexpStruct`
- `alternative encoding of Modf, chosen by the compiler`: `ModfStruct`
- `reserved by the GLSL.std.450 specification, not a valid instruction`: `IMix`

</details>

### SPIR-V input

<details>
<summary>12 gaps: 8 refused, 4 wrong</summary>

Wrong, by cause ([open bug issues](https://github.com/CrossGL/crosstl/issues?q=is:issue%20is:open%20label:coverage%20label:bug%20label:backend:spirv%20label:direction:input)):

- `the round trip no longer calls <feature>` (3): `NClamp`, `NMax`, `NMin`
- `on vulkan-lavapipe, the translation computes different results from the native probe`: `int8`

Refused, by diagnostic:

- `project.translate.vulkan-builtin-unsupported: SPIRVBuiltinLoweringError: Cannot lower call to '<feature>' to SPIR-V: it is neither a declared function nor a built-in with a SPIR-V lowering` (3): `InterpolateAtCentroid`, `InterpolateAtOffset`, `InterpolateAtSample`
- `project.translate.vulkan-builtin-unsupported: SPIRVBuiltinLoweringError: Cannot lower call to 'spirvArrayLength' to SPIR-V: it is neither a declared function nor a built-in with a SPIR-V lowering`: `runtime-array-length`
- `project.translate.vulkan-builtin-unsupported: SPIRVBuiltinLoweringError: Cannot lower call to 'spirvBitcast_u16' to SPIR-V: it is neither a declared function nor a built-in with a SPIR-V lowering`: `uint16`
- `project.translate.vulkan-builtin-unsupported: SPIRVBuiltinLoweringError: Cannot lower call to 'spirvBitcast_u64' to SPIR-V: it is neither a declared function nor a built-in with a SPIR-V lowering`: `uint64`
- `project.translate.vulkan-builtin-unsupported: SPIRVBuiltinLoweringError: Cannot lower call to 'spirvBitcast_u8' to SPIR-V: it is neither a declared function nor a built-in with a SPIR-V lowering`: `uint8`
- `project.translate.vulkan-lowering-unsupported: SpirvLoweringRefused: SPIR-V cannot lower N construct: Cannot assign to member value of payload: it does not name storage (target vulkan) [project.translate.vulkan-lowering-unsupported]`: `task`

</details>

<details>
<summary>1 untested: function 1</summary>

- `reserved by the GLSL.std.450 specification, not a valid instruction`: `IMix`

</details>

### WGSL output

<details>
<summary>64 gaps: 1 refused, 63 wrong</summary>

Wrong, by cause ([open bug issues](https://github.com/CrossGL/crosstl/issues?q=is:issue%20is:open%20label:coverage%20label:bug%20label:backend:wgsl%20label:direction:output)):

- `ValueError: WGSL target does not support expression WaveOpNode` (20): `subgroupAdd`, `subgroupAll`, `subgroupAnd`, `subgroupAny`, `subgroupBallot`, `subgroupBroadcast`, `subgroupBroadcastFirst`, `subgroupExclusiveAdd`, `subgroupExclusiveMul`, `subgroupInclusiveAdd`, `subgroupInclusiveMul`, `subgroupMax`, `subgroupMin`, `subgroupMul`, `subgroupOr`, `subgroupShuffle`, `subgroupShuffleDown`, `subgroupShuffleUp`, `subgroupShuffleXor`, `subgroupXor`
- `no definition in scope for identifier: '<feature>'` (18): `atomicCompareExchangeWeak`, `countOneBits`, `extractBits`, `faceForward`, `firstLeadingBit`, `firstTrailingBit`, `insertBits`, `pack2x16float`, `pack2x16snorm`, `pack2x16unorm`, `pack4x8snorm`, `pack4x8unorm`, `reverseBits`, `unpack2x16float`, `unpack2x16snorm`, `unpack2x16unorm`, `unpack4x8snorm`, `unpack4x8unorm`
- `[ERROR naga::front::wgsl::lower] Type Scalar(Scalar { kind: Sint, width: N }) passed to atomic op` (10): `atomicAdd`, `atomicAnd`, `atomicExchange`, `atomicLoad`, `atomicMax`, `atomicMin`, `atomicOr`, `atomicStore`, `atomicSub`, `atomicXor`
- `ValueError: WGSL target does not support CrossGL texture function <feature> yet` (3): `textureGather`, `textureGatherCompare`, `textureNumLevels`
- `too many arguments passed to '<feature>'` (3): `atan2`, `frexp`, `modf`
- `ValueError: WGSL target does not support CrossGL resource type sampler2DMS yet; split texture/sampler/storage bindings are required`: `textureNumSamples`
- `ValueError: WGSL target does not support do-while statements`: `do-while-loop`
- `Entry point compute_main at Compute is invalid`: `switch`
- `Entry point fragment_main at Fragment is invalid`: `textureSampleBias`
- `expected ')', found "("`: `runtime-array-length`
- `expected assignment or increment/decrement, found ";"`: `discard`
- `the type of '<probe>' is expected to be 'vec2<i32>', but got 'vec2<u32>'`: `textureDimensions`
- `the type of '<probe>' is expected to be 'vec3<i32>', but got 'vec2<u32>'`: `textureNumLayers`
- `wrong type passed as argument #N to 'mix'`: `select`

Refused, by diagnostic:

- `CrossGL has no spelling for this operation: no CrossGL sample clamps its coordinates half a texel inside the edge`: `textureSampleBaseClampToEdge`

</details>

<details>
<summary>1 untested: function 1</summary>

- `the probe itself is rejected: error: no definition in scope for identifier: 'subgroupElect'`: `subgroupElect`

</details>

### CrossGL IR expressiveness

<details>
<summary>852 gaps: 48 refused, 804 wrong</summary>

Wrong, by cause:

- `cuda: identifier "<probe>" is undefined in device code` (291): `cuda/__byte_perm`, `cuda/__clz`, `cuda/__clzll`, `cuda/__cosf`, `cuda/__dadd_rd`, `cuda/__dadd_rn`, `cuda/__dadd_ru`, `cuda/__dadd_rz`, `cuda/__ddiv_rd`, `cuda/__ddiv_rn`, `cuda/__ddiv_ru`, `cuda/__ddiv_rz`, `cuda/__dmul_rd`, `cuda/__dmul_rn`, `cuda/__dmul_ru`, `cuda/__dmul_rz`, `cuda/__double2float_rd`, `cuda/__double2float_rn`, `cuda/__double2float_ru`, `cuda/__double2float_rz`, `cuda/__double2hiint`, `cuda/__double2int_rd`, `cuda/__double2int_rn`, `cuda/__double2int_ru`, `cuda/__double2int_rz`, `cuda/__double2ll_rd`, `cuda/__double2ll_rn`, `cuda/__double2ll_ru`, `cuda/__double2ll_rz`, `cuda/__double2loint`, `cuda/__double2uint_rd`, `cuda/__double2uint_rn`, `cuda/__double2uint_ru`, `cuda/__double2uint_rz`, `cuda/__double_as_longlong`, `cuda/__drcp_rd`, `cuda/__drcp_rn`, `cuda/__drcp_ru`, `cuda/__drcp_rz`, `cuda/__dsqrt_rd`, `cuda/__dsqrt_rn`, `cuda/__dsqrt_ru`, `cuda/__dsqrt_rz`, `cuda/__dsub_rd`, `cuda/__dsub_rn`, `cuda/__dsub_ru`, `cuda/__dsub_rz`, `cuda/__exp10f`, `cuda/__expf`, `cuda/__fadd_rd`, `cuda/__fadd_rn`, `cuda/__fadd_ru`, `cuda/__fadd_rz`, `cuda/__fdiv_rd`, `cuda/__fdiv_rn`, `cuda/__fdiv_ru`, `cuda/__fdiv_rz`, `cuda/__fdividef`, `cuda/__ffs`, `cuda/__ffsll`, `cuda/__finite`, `cuda/__finitef`, `cuda/__float2int_rd`, `cuda/__float2int_rn`, `cuda/__float2int_ru`, `cuda/__float2int_rz`, `cuda/__float2ll_rd`, `cuda/__float2ll_rn`, `cuda/__float2ll_ru`, `cuda/__float2ll_rz`, `cuda/__float2uint_rd`, `cuda/__float2uint_rn`, `cuda/__float2uint_ru`, `cuda/__float2uint_rz`, `cuda/__fma_rd`, `cuda/__fma_rn`, `cuda/__fma_ru`, `cuda/__fma_rz`, `cuda/__fmaf_ieee_rd`, `cuda/__fmaf_ieee_rn`, `cuda/__fmaf_ieee_ru`, `cuda/__fmaf_ieee_rz`, `cuda/__fmaf_rd`, `cuda/__fmaf_rn`, `cuda/__fmaf_ru`, `cuda/__fmaf_rz`, `cuda/__fmul_rd`, `cuda/__fmul_rn`, `cuda/__fmul_ru`, `cuda/__fmul_rz`, `cuda/__frcp_rd`, `cuda/__frcp_rn`, `cuda/__frcp_ru`, `cuda/__frcp_rz`, `cuda/__frsqrt_rn`, `cuda/__fsqrt_rd`, `cuda/__fsqrt_rn`, `cuda/__fsqrt_ru`, `cuda/__fsqrt_rz`, `cuda/__fsub_rd`, `cuda/__fsub_rn`, `cuda/__fsub_ru`, `cuda/__fsub_rz`, `cuda/__hadd`, `cuda/__hiloint2double`, `cuda/__int2double_rn`, `cuda/__int2float_rd`, `cuda/__int2float_rn`, `cuda/__int2float_ru`, `cuda/__int2float_rz`, `cuda/__isinf`, `cuda/__isnan`, `cuda/__ll2double_rd`, `cuda/__ll2double_rn`, `cuda/__ll2double_ru`, `cuda/__ll2double_rz`, `cuda/__ll2float_rd`, `cuda/__ll2float_rn`, `cuda/__ll2float_ru`, `cuda/__ll2float_rz`, `cuda/__log10f`, `cuda/__log2f`, `cuda/__logf`, `cuda/__longlong_as_double`, `cuda/__mul24`, `cuda/__mul64hi`, `cuda/__mulhi`, `cuda/__powf`, `cuda/__rhadd`, `cuda/__sad`, `cuda/__saturatef`, `cuda/__signbitf`, `cuda/__sinf`, `cuda/__syncthreads_and`, `cuda/__syncthreads_count`, `cuda/__syncthreads_or`, `cuda/__tanf`, `cuda/__threadfence_system`, `cuda/__uhadd`, `cuda/__uint2double_rn`, `cuda/__uint2float_rd`, `cuda/__uint2float_rn`, `cuda/__uint2float_ru`, `cuda/__uint2float_rz`, `cuda/__umul24`, `cuda/__umulhi`, `cuda/__urhadd`, `cuda/__usad`, `cuda/__vabs2`, `cuda/__vabs4`, `cuda/__vabsdiffs2`, `cuda/__vabsdiffs4`, `cuda/__vabsdiffu2`, `cuda/__vabsdiffu4`, `cuda/__vabsss2`, `cuda/__vabsss4`, `cuda/__vadd2`, `cuda/__vadd4`, `cuda/__vaddss2`, `cuda/__vaddss4`, `cuda/__vaddus2`, `cuda/__vaddus4`, `cuda/__vavgs2`, `cuda/__vavgs4`, `cuda/__vavgu2`, `cuda/__vavgu4`, `cuda/__vcmpeq2`, `cuda/__vcmpeq4`, `cuda/__vcmpges2`, `cuda/__vcmpges4`, `cuda/__vcmpgeu2`, `cuda/__vcmpgeu4`, `cuda/__vcmpgts2`, `cuda/__vcmpgts4`, `cuda/__vcmpgtu2`, `cuda/__vcmpgtu4`, `cuda/__vcmples2`, `cuda/__vcmples4`, `cuda/__vcmpleu2`, `cuda/__vcmpleu4`, `cuda/__vcmplts2`, `cuda/__vcmplts4`, `cuda/__vcmpltu2`, `cuda/__vcmpltu4`, `cuda/__vcmpne2`, `cuda/__vcmpne4`, `cuda/__vhaddu2`, `cuda/__vhaddu4`, `cuda/__vmaxs2`, `cuda/__vmaxs4`, `cuda/__vmaxu2`, `cuda/__vmaxu4`, `cuda/__vmins2`, `cuda/__vmins4`, `cuda/__vminu2`, `cuda/__vminu4`, `cuda/__vneg2`, `cuda/__vneg4`, `cuda/__vnegss2`, `cuda/__vnegss4`, `cuda/__vsads2`, `cuda/__vsads4`, `cuda/__vsadu2`, `cuda/__vsadu4`, `cuda/__vseteq2`, `cuda/__vseteq4`, `cuda/__vsetges2`, `cuda/__vsetges4`, `cuda/__vsetgeu2`, `cuda/__vsetgeu4`, `cuda/__vsetgts2`, `cuda/__vsetgts4`, `cuda/__vsetgtu2`, `cuda/__vsetgtu4`, `cuda/__vsetles2`, `cuda/__vsetles4`, `cuda/__vsetleu2`, `cuda/__vsetleu4`, `cuda/__vsetlts2`, `cuda/__vsetlts4`, `cuda/__vsetltu2`, `cuda/__vsetltu4`, `cuda/__vsetne2`, `cuda/__vsetne4`, `cuda/__vsub2`, `cuda/__vsub4`, `cuda/__vsubss2`, `cuda/__vsubss4`, `cuda/__vsubus2`, `cuda/__vsubus4`, `cuda/atomicDec`, `cuda/atomicInc`, `cuda/cbrt`, `cuda/clock`, `cuda/clock64`, `cuda/copysign`, `cuda/cyl_bessel_i0`, `cuda/cyl_bessel_i1`, `cuda/erf`, `cuda/erfc`, `cuda/erfcinv`, `cuda/erfcx`, `cuda/erfinv`, `cuda/exp10`, `cuda/expm1`, `cuda/fdim`, `cuda/fdivide`, `cuda/fmod`, `cuda/hypot`, `cuda/ilogb`, `cuda/ilogbf`, `cuda/j0`, `cuda/j1`, `cuda/jn`, `cuda/labs`, `cuda/lgamma`, `cuda/llabs`, `cuda/llmax`, `cuda/llmin`, `cuda/llrint`, `cuda/llrintf`, `cuda/llround`, `cuda/llroundf`, `cuda/log10`, `cuda/log1p`, `cuda/logb`, `cuda/lrint`, `cuda/lrintf`, `cuda/lround`, `cuda/lroundf`, `cuda/nearbyint`, `cuda/nextafter`, `cuda/norm`, `cuda/norm3d`, `cuda/norm4d`, `cuda/normcdf`, `cuda/normcdfinv`, `cuda/rcbrt`, `cuda/remainder`, `cuda/rhypot`, `cuda/rnorm`, `cuda/rnorm3d`, `cuda/rnorm4d`, `cuda/scalbln`, `cuda/scalbn`, `cuda/tgamma`, `cuda/umax`, `cuda/umin`, `cuda/y0`, `cuda/y1`, `cuda/yn`
- `hip: reference to __host__ variable '<probe>' in __global__ function` (220): `hip/__all`, `hip/__any`, `hip/__ballot`, `hip/__ballot64`, `hip/__bitextract_u32`, `hip/__bitextract_u64`, `hip/__bitinsert_u32`, `hip/__bitinsert_u64`, `hip/__brevll`, `hip/__byte_perm`, `hip/__clz`, `hip/__clzll`, `hip/__cosf`, `hip/__dadd_rn`, `hip/__ddiv_rn`, `hip/__dmul_rn`, `hip/__double2float_rd`, `hip/__double2float_rn`, `hip/__double2float_ru`, `hip/__double2float_rz`, `hip/__double2hiint`, `hip/__double2int_rd`, `hip/__double2int_rn`, `hip/__double2int_ru`, `hip/__double2int_rz`, `hip/__double2ll_rd`, `hip/__double2ll_rn`, `hip/__double2ll_ru`, `hip/__double2ll_rz`, `hip/__double2loint`, `hip/__double2uint_rd`, `hip/__double2uint_rn`, `hip/__double2uint_ru`, `hip/__double2uint_rz`, `hip/__double2ull_rd`, `hip/__double2ull_rn`, `hip/__double2ull_ru`, `hip/__double2ull_rz`, `hip/__double_as_longlong`, `hip/__drcp_rn`, `hip/__dsqrt_rn`, `hip/__dsub_rn`, `hip/__exp10f`, `hip/__expf`, `hip/__fadd_rn`, `hip/__fdiv_rn`, `hip/__fdividef`, `hip/__ffs`, `hip/__ffsll`, `hip/__finite`, `hip/__finitef`, `hip/__float2int_rd`, `hip/__float2int_rn`, `hip/__float2int_ru`, `hip/__float2int_rz`, `hip/__float2ll_rd`, `hip/__float2ll_rn`, `hip/__float2ll_ru`, `hip/__float2ll_rz`, `hip/__float2uint_rd`, `hip/__float2uint_rn`, `hip/__float2uint_ru`, `hip/__float2uint_rz`, `hip/__float2ull_rd`, `hip/__float2ull_rn`, `hip/__float2ull_ru`, `hip/__float2ull_rz`, `hip/__fma_rn`, `hip/__fmaf_rn`, `hip/__fmul_rn`, `hip/__fns32`, `hip/__fns64`, `hip/__frcp_rn`, `hip/__frsqrt_rn`, `hip/__fsqrt_rn`, `hip/__fsub_rn`, `hip/__funnelshift_l`, `hip/__funnelshift_lc`, `hip/__funnelshift_r`, `hip/__funnelshift_rc`, `hip/__hadd`, `hip/__hiloint2double`, `hip/__int2double_rn`, `hip/__int2float_rd`, `hip/__int2float_rn`, `hip/__int2float_ru`, `hip/__int2float_rz`, `hip/__isinf`, `hip/__isnan`, `hip/__lane_id`, `hip/__lanemask_eq`, `hip/__lanemask_gt`, `hip/__lanemask_lt`, `hip/__ll2double_rd`, `hip/__ll2double_rn`, `hip/__ll2double_ru`, `hip/__ll2double_rz`, `hip/__ll2float_rd`, `hip/__ll2float_rn`, `hip/__ll2float_ru`, `hip/__ll2float_rz`, `hip/__log10f`, `hip/__log2f`, `hip/__logf`, `hip/__longlong_as_double`, `hip/__mbcnt_hi`, `hip/__mbcnt_lo`, `hip/__mul24`, `hip/__mul64hi`, `hip/__mulhi`, `hip/__popcll`, `hip/__powf`, `hip/__rhadd`, `hip/__sad`, `hip/__saturatef`, `hip/__signbit`, `hip/__signbitf`, `hip/__sincosf`, `hip/__sinf`, `hip/__smid`, `hip/__syncthreads_and`, `hip/__syncthreads_count`, `hip/__syncthreads_or`, `hip/__tanf`, `hip/__threadfence_system`, `hip/__uhadd`, `hip/__uint2double_rn`, `hip/__uint2float_rd`, `hip/__uint2float_rn`, `hip/__uint2float_ru`, `hip/__uint2float_rz`, `hip/__ull2double_rd`, `hip/__ull2double_rn`, `hip/__ull2double_ru`, `hip/__ull2double_rz`, `hip/__ull2float_rd`, `hip/__ull2float_rn`, `hip/__ull2float_ru`, `hip/__ull2float_rz`, `hip/__umul24`, `hip/__umul64hi`, `hip/__umulhi`, `hip/__urhadd`, `hip/__usad`, `hip/atomicAddNoRet`, `hip/atomicAdd_system`, `hip/atomicAnd_system`, `hip/atomicCAS_system`, `hip/atomicDec`, `hip/atomicExch_system`, `hip/atomicInc`, `hip/atomicMax_system`, `hip/atomicMin_system`, `hip/atomicOr_system`, `hip/atomicSub`, `hip/atomicSub_system`, `hip/atomicXor_system`, `hip/cbrt`, `hip/clock`, `hip/clock64`, `hip/copysign`, `hip/cyl_bessel_i0`, `hip/cyl_bessel_i1`, `hip/erf`, `hip/erfc`, `hip/erfcinv`, `hip/erfcx`, `hip/erfinv`, `hip/exp10`, `hip/expm1`, `hip/fdim`, `hip/fdividef`, `hip/fmod`, `hip/hypot`, `hip/ilogb`, `hip/ilogbf`, `hip/j0`, `hip/j1`, `hip/jn`, `hip/labs`, `hip/lgamma`, `hip/llabs`, `hip/llrint`, `hip/llrintf`, `hip/llround`, `hip/llroundf`, `hip/log10`, `hip/log1p`, `hip/logb`, `hip/lrint`, `hip/lrintf`, `hip/lround`, `hip/lroundf`, `hip/nan`, `hip/nearbyint`, `hip/nextafter`, `hip/norm`, `hip/norm3d`, `hip/norm4d`, `hip/normcdf`, `hip/normcdfinv`, `hip/powi`, `hip/rcbrt`, `hip/remainder`, `hip/remquo`, `hip/rhypot`, `hip/rnorm`, `hip/rnorm3d`, `hip/rnorm4d`, `hip/scalbln`, `hip/scalbn`, `hip/sincos`, `hip/sincosf`, `hip/sincospi`, `hip/sincospif`, `hip/tgamma`, `hip/wall_clock64`, `hip/y0`, `hip/y1`, `hip/yn`
- `rust: error[E0463]: can't find crate for 'std'` (161): `GLSL.std.450:Exp+OpFSub`, `GLSL.std.450:FAbs+OpFOrdLessThan`, `GLSL.std.450:Log+OpFAdd`, `GLSL.std.450:Log+OpFMul`, `OpAll+OpFOrdGreaterThan`, `OpAny+OpFOrdGreaterThan`, `OpAtomicFMaxEXT`, `OpAtomicFMinEXT`, `OpAtomicIDecrement`, `OpAtomicIIncrement`, `OpAtomicISub`, `OpAtomicLoad`, `OpAtomicStore`, `OpBitwiseAnd+OpBitwiseOr`, `OpBitwiseAnd+OpBitwiseOr+OpFUnordNotEqual`, `OpBitwiseAnd+OpIEqual+OpSwitch`, `OpBitwiseAnd+OpINotEqual`, `OpBitwiseAnd+OpINotEqual+OpLogicalNot`, `OpControlBarrier(3,3,0)`, `OpControlBarrier(3,3,2376)`, `OpDemoteToHelperInvocation`, `OpEmitMeshTasksEXT`, `OpFOrdEqual+OpLogicalOr`, `OpFOrdGreaterThan+OpFOrdLessThan+OpFOrdLessThanEqual`, `OpFOrdGreaterThan+OpGroupNonUniformAll`, `OpFOrdGreaterThan+OpGroupNonUniformAny`, `OpFOrdGreaterThan+OpGroupNonUniformBallot`, `OpFOrdGreaterThan+OpGroupNonUniformLogicalAnd.ClusteredReduce`, `OpFOrdGreaterThan+OpGroupNonUniformLogicalAnd.ExclusiveScan`, `OpFOrdGreaterThan+OpGroupNonUniformLogicalAnd.InclusiveScan`, `OpFOrdGreaterThan+OpGroupNonUniformLogicalAnd.Reduce`, `OpFOrdGreaterThan+OpGroupNonUniformLogicalOr.ClusteredReduce`, `OpFOrdGreaterThan+OpGroupNonUniformLogicalOr.ExclusiveScan`, `OpFOrdGreaterThan+OpGroupNonUniformLogicalOr.InclusiveScan`, `OpFOrdGreaterThan+OpGroupNonUniformLogicalOr.Reduce`, `OpFOrdGreaterThan+OpGroupNonUniformLogicalXor.ClusteredReduce`, `OpFOrdGreaterThan+OpGroupNonUniformLogicalXor.ExclusiveScan`, `OpFOrdGreaterThan+OpGroupNonUniformLogicalXor.InclusiveScan`, `OpFOrdGreaterThan+OpGroupNonUniformLogicalXor.Reduce`, `OpFOrdGreaterThanEqual+OpFUnordNotEqual`, `OpFOrdLessThanEqual+OpFUnordNotEqual`, `OpGroupNonUniformAllEqual+OpLogicalAnd`, `OpGroupNonUniformBallotBitCount.ExclusiveScan`, `OpGroupNonUniformBallotBitCount.InclusiveScan`, `OpGroupNonUniformBallotBitCount.Reduce`, `OpGroupNonUniformBallotBitExtract`, `OpGroupNonUniformBallotFindLSB`, `OpGroupNonUniformBallotFindMSB`, `OpGroupNonUniformBitwiseAnd.ClusteredReduce`, `OpGroupNonUniformBitwiseAnd.InclusiveScan`, `OpGroupNonUniformBitwiseOr.ClusteredReduce`, `OpGroupNonUniformBitwiseOr.InclusiveScan`, `OpGroupNonUniformBitwiseXor.ClusteredReduce`, `OpGroupNonUniformBitwiseXor.InclusiveScan`, `OpGroupNonUniformFAdd.ClusteredReduce`, `OpGroupNonUniformFAdd.InclusiveScan`, `OpGroupNonUniformFMax.ClusteredReduce`, `OpGroupNonUniformFMax.InclusiveScan`, `OpGroupNonUniformFMin.ClusteredReduce`, `OpGroupNonUniformFMin.InclusiveScan`, `OpGroupNonUniformFMul.ClusteredReduce`, `OpGroupNonUniformFMul.InclusiveScan`, `OpGroupNonUniformIAdd.ClusteredReduce`, `OpGroupNonUniformIAdd.ExclusiveScan`, `OpGroupNonUniformIAdd.InclusiveScan`, `OpGroupNonUniformIAdd.Reduce`, `OpGroupNonUniformIMul.ClusteredReduce`, `OpGroupNonUniformIMul.ExclusiveScan`, `OpGroupNonUniformIMul.InclusiveScan`, `OpGroupNonUniformIMul.Reduce`, `OpGroupNonUniformInverseBallot`, `OpGroupNonUniformSMax.ClusteredReduce`, `OpGroupNonUniformSMax.ExclusiveScan`, `OpGroupNonUniformSMax.InclusiveScan`, `OpGroupNonUniformSMax.Reduce`, `OpGroupNonUniformSMin.ClusteredReduce`, `OpGroupNonUniformSMin.ExclusiveScan`, `OpGroupNonUniformSMin.InclusiveScan`, `OpGroupNonUniformSMin.Reduce`, `OpGroupNonUniformShuffleDown`, `OpGroupNonUniformShuffleUp`, `OpGroupNonUniformShuffleXor`, `OpGroupNonUniformUMax.ClusteredReduce`, `OpGroupNonUniformUMax.ExclusiveScan`, `OpGroupNonUniformUMax.InclusiveScan`, `OpGroupNonUniformUMax.Reduce`, `OpGroupNonUniformUMin.ClusteredReduce`, `OpGroupNonUniformUMin.ExclusiveScan`, `OpGroupNonUniformUMin.InclusiveScan`, `OpGroupNonUniformUMin.Reduce`, `OpKill`, `OpMemoryBarrier(3,2056)`, `OpMemoryBarrier(3,2376)`, `OpMemoryBarrier(3,264)`, `OpMemoryBarrier(3,72)`, `OpVectorExtractDynamic`, `OpVectorInsertDynamic`, `rust/abs_diff_eq`, `rust/angle_between`, `rust/angle_to`, `rust/any_orthogonal_vector`, `rust/any_orthonormal_vector`, `rust/as_dvec3`, `rust/as_ivec3`, `rust/as_uvec3`, `rust/clamp_length`, `rust/clamp_length_max`, `rust/clamp_length_min`, `rust/copysign`, `rust/distance_squared`, `rust/div_euclid`, `rust/dot_into_vec`, `rust/element_product`, `rust/element_sum`, `rust/epsilon`, `rust/extend`, `rust/from_homogeneous`, `rust/infinity`, `rust/is_finite`, `rust/is_finite_mask`, `rust/is_nan`, `rust/is_negative_bitmask`, `rust/is_negative_mask`, `rust/is_normalized`, `rust/length_recip`, `rust/length_squared`, `rust/lerp_monotonic`, `rust/max_element`, `rust/max_value`, `rust/midpoint`, `rust/min_element`, `rust/min_positive_value`, `rust/min_value`, `rust/move_towards`, `rust/nan`, `rust/neg_infinity`, `rust/neg_zero`, `rust/normalize_or`, `rust/normalize_or_zero`, `rust/project_onto`, `rust/project_onto_normalized`, `rust/recip`, `rust/reject_from`, `rust/reject_from_normalized`, `rust/rem_euclid`, `rust/rotate_axis`, `rust/rotate_towards`, `rust/rotate_x`, `rust/rotate_y`, `rust/rotate_z`, `rust/saturate`, `rust/select`, `rust/signum`, `rust/sincos`, `rust/slerp`, `rust/to_homogeneous`, `rust/truncate`, `rust/vec3`, `rust/with_x`, `rust/with_y`, `rust/with_z`
- `mojo: global variables are not supported; move this into a function body or use 'comptime' to declare a constant` (44): `mojo/align_down`, `mojo/align_up`, `mojo/bit_not`, `mojo/bit_width`, `mojo/byte_swap`, `mojo/cbrt`, `mojo/ceildiv`, `mojo/comb`, `mojo/copysign`, `mojo/count_leading_zeros`, `mojo/count_trailing_zeros`, `mojo/erf`, `mojo/erfc`, `mojo/expm1`, `mojo/factorial`, `mojo/gamma`, `mojo/gcd`, `mojo/hypot`, `mojo/inf`, `mojo/iota`, `mojo/isclose`, `mojo/isfinite`, `mojo/j0`, `mojo/j1`, `mojo/lcm`, `mojo/lgamma`, `mojo/log10`, `mojo/log1p`, `mojo/log2_ceil`, `mojo/log2_floor`, `mojo/logb`, `mojo/nan`, `mojo/next_power_of_two`, `mojo/nextafter`, `mojo/perm`, `mojo/prev_power_of_two`, `mojo/recip`, `mojo/remainder`, `mojo/rotate_bits_left`, `mojo/rotate_bits_right`, `mojo/scalb`, `mojo/ulp`, `mojo/y0`, `mojo/y1`
- `cuda: identifier "uint64_t" is undefined` (21): `cuda/__brevll`, `cuda/__double2ull_rd`, `cuda/__double2ull_rn`, `cuda/__double2ull_ru`, `cuda/__double2ull_rz`, `cuda/__float2ull_rd`, `cuda/__float2ull_rn`, `cuda/__float2ull_ru`, `cuda/__float2ull_rz`, `cuda/__popcll`, `cuda/__ull2double_rd`, `cuda/__ull2double_rn`, `cuda/__ull2double_ru`, `cuda/__ull2double_rz`, `cuda/__ull2float_rd`, `cuda/__ull2float_rn`, `cuda/__ull2float_ru`, `cuda/__ull2float_rz`, `cuda/__umul64hi`, `cuda/ullmax`, `cuda/ullmin`
- `slang: unknown type name 'Ptr'` (14): `slang/GetRenderTargetSampleCount`, `slang/GetRenderTargetSamplePosition`, `slang/MaybeReorderThread`, `slang/Process2DQuadTessFactorsAvg`, `slang/Process2DQuadTessFactorsMax`, `slang/Process2DQuadTessFactorsMin`, `slang/ProcessIsolineTessFactors`, `slang/ProcessQuadTessFactorsAvg`, `slang/ProcessQuadTessFactorsMax`, `slang/ProcessQuadTessFactorsMin`, `slang/ProcessTriTessFactorsAvg`, `slang/ProcessTriTessFactorsMax`, `slang/ProcessTriTessFactorsMin`, `slang/dst`
- `cuda: argument of type "uint" (aka "unsigned int") is incompatible with parameter of type "unsigned int *"` (4): `cuda/atomicDec_block`, `cuda/atomicDec_system`, `cuda/atomicInc_block`, `cuda/atomicInc_system`
- `cuda: argument of type "float" is incompatible with parameter of type "float *"` (3): `cuda/__sincosf`, `cuda/sincosf`, `cuda/sincospif`
- `hlsl: type 'Payload' used as payload requires that it is annotated with the [raypayload] attribute` (2): `hlsl/Invoke`, `hlsl/TraceRay`
- `cuda: no instance of overloaded function "atomicAdd_block" matches the argument list`: `cuda/atomicAdd_block`
- `cuda: no instance of overloaded function "atomicAdd_system" matches the argument list`: `cuda/atomicAdd_system`
- `cuda: no instance of overloaded function "atomicAnd_block" matches the argument list`: `cuda/atomicAnd_block`
- `cuda: no instance of overloaded function "atomicAnd_system" matches the argument list`: `cuda/atomicAnd_system`
- `cuda: no instance of overloaded function "atomicCAS_block" matches the argument list`: `cuda/atomicCAS_block`
- `cuda: no instance of overloaded function "atomicCAS_system" matches the argument list`: `cuda/atomicCAS_system`
- `cuda: no instance of overloaded function "atomicExch_block" matches the argument list`: `cuda/atomicExch_block`
- `cuda: no instance of overloaded function "atomicExch_system" matches the argument list`: `cuda/atomicExch_system`
- `cuda: no instance of overloaded function "atomicMax_block" matches the argument list`: `cuda/atomicMax_block`
- `cuda: no instance of overloaded function "atomicMax_system" matches the argument list`: `cuda/atomicMax_system`
- `cuda: no instance of overloaded function "atomicMin_block" matches the argument list`: `cuda/atomicMin_block`
- `cuda: no instance of overloaded function "atomicMin_system" matches the argument list`: `cuda/atomicMin_system`
- `cuda: no instance of overloaded function "atomicOr_block" matches the argument list`: `cuda/atomicOr_block`
- `cuda: no instance of overloaded function "atomicOr_system" matches the argument list`: `cuda/atomicOr_system`
- `cuda: no instance of overloaded function "atomicXor_block" matches the argument list`: `cuda/atomicXor_block`
- `cuda: no instance of overloaded function "atomicXor_system" matches the argument list`: `cuda/atomicXor_system`
- `cuda: no instance of overloaded function "remquo" matches the argument list`: `cuda/remquo`
- `cuda: no instance of overloaded function "sincos" matches the argument list`: `cuda/sincos`
- `cuda: no instance of overloaded function "sincospi" matches the argument list`: `cuda/sincospi`
- `glsl: 'atomicCounterAdd' : no matching overloaded function found`: `glsl/atomicCounterAdd`
- `glsl: 'atomicCounterAnd' : no matching overloaded function found`: `glsl/atomicCounterAnd`
- `glsl: 'atomicCounterCompSwap' : no matching overloaded function found`: `glsl/atomicCounterCompSwap`
- `glsl: 'atomicCounterExchange' : no matching overloaded function found`: `glsl/atomicCounterExchange`
- `glsl: 'atomicCounterMax' : no matching overloaded function found`: `glsl/atomicCounterMax`
- `glsl: 'atomicCounterMin' : no matching overloaded function found`: `glsl/atomicCounterMin`
- `glsl: 'atomicCounterOr' : no matching overloaded function found`: `glsl/atomicCounterOr`
- `glsl: 'atomicCounterSubtract' : no matching overloaded function found`: `glsl/atomicCounterSubtract`
- `glsl: 'atomicCounterXor' : no matching overloaded function found`: `glsl/atomicCounterXor`
- `glsl: 'textureSamples' : no matching overloaded function found`: `OpImageQuerySamples`
- `glsl: 'location' : SPIR-V requires location for user input/output`: `glsl/subpassLoad`
- `hlsl: ValueError: DirectX ByteAddressBuffer interlocked member 'InterlockedMax' value argument must be scalar uint, got int`: `OpAtomicSMax+OpShiftRightLogical`
- `hlsl: ValueError: DirectX ByteAddressBuffer interlocked member 'InterlockedMin' value argument must be scalar uint, got int`: `OpAtomicSMin+OpShiftRightLogical`
- `metal: no matching function for call to 'atomic_fetch_sub'`: `metal/atomic_fetch_sub`
- `metal: no matching function for call to 'atomic_load'`: `metal/atomic_load`
- `metal: no matching function for call to 'atomic_max'`: `metal/atomic_max`
- `metal: no matching function for call to 'atomic_min'`: `metal/atomic_min`
- `metal: no matching function for call to 'atomic_store'`: `metal/atomic_store`
- `slang: error[E30015]: undefined identifier 'i16vec2'`: `OpSConvert`
- `slang: error[E30015]: undefined identifier 'u16vec2'`: `OpUConvert`
- `wgsl: ValueError: WGSL target does not support CrossGL texture function textureQueryLevels yet`: `OpImageQueryLevels`
- `wgsl: [ERROR naga::front::wgsl::lower] Type Scalar(Scalar { kind: Sint, width: N }) passed to atomic op`: `wgsl/atomicLoad`
- `wgsl: no definition in scope for identifier: 'bitfieldExtract'`: `GLSL.std.450:UMin+OpBitFieldSExtract+OpISub`
- `wgsl: no definition in scope for identifier: 'bitfieldInsert'`: `GLSL.std.450:UMin+OpBitFieldInsert+OpISub`
- `wgsl: wrong type passed as argument #N to 'mix'`: `OpCompositeConstruct+OpSelect`

Refused, by diagnostic:

- `hlsl: CrossGL has no work-graph node stage` (9): `hlsl/Count`, `hlsl/FinishedCrossGroupSharing`, `hlsl/GetGroupNodeOutputRecords`, `hlsl/GetRemainingRecursionLevels`, `hlsl/GetThreadNodeOutputRecords`, `hlsl/GroupIncrementOutputCount`, `hlsl/IsValid`, `hlsl/OutputComplete`, `hlsl/ThreadIncrementOutputCount`
- `metal: CrossGL has no spelling for this operation: CrossGL has no operation that takes a memory order or memory flags` (7): `metal/atomic_fetch_sub_explicit`, `metal/atomic_load_explicit`, `metal/atomic_max_explicit`, `metal/atomic_min_explicit`, `metal/atomic_store_explicit`, `metal/atomic_thread_fence`, `metal/simdgroup_barrier`
- `hlsl: project.translate.undeclared-type: UndeclaredTypeError: type 'SAMPLER_FEEDBACK_MIN_MIP' is not declared: it is neither a CrossGL built-in type nor a structure, enumeration, type alias or generic parameter of the program (target directx) [project.translate.undeclared-type]` (4): `hlsl/WriteSamplerFeedback`, `hlsl/WriteSamplerFeedbackBias`, `hlsl/WriteSamplerFeedbackGrad`, `hlsl/WriteSamplerFeedbackLevel`
- `metal: CrossGL has no type for Metal's quad_vote mask` (2): `metal/quad_active_threads_mask`, `metal/quad_ballot`
- `rust: CrossGL has no spelling for this operation: Rust's fract is x - trunc(x); CrossGL's is x - floor(x)` (2): `GLSL.std.450:Trunc+OpFSub`, `rust/Vec3::fract`
- `cuda: project.translate.cuda-metal-math-function-unsupported: MetalMathFunctionUnsupportedError: MSL cospi is exact at integers and half-integers, which cuda has no function for, and CrossTL does not lower it with an exact range reduction yet (target cuda) [project.translate.cuda-metal-math-function-unsupported]`: `cuda/cospi`
- `cuda: project.translate.cuda-metal-math-function-unsupported: MetalMathFunctionUnsupportedError: MSL sinpi is exact at integers and half-integers, which cuda has no function for, and CrossTL does not lower it with an exact range reduction yet (target cuda) [project.translate.cuda-metal-math-function-unsupported]`: `cuda/sinpi`
- `hip: project.translate.hip-metal-math-function-unsupported: MetalMathFunctionUnsupportedError: MSL cospi is exact at integers and half-integers, which hip has no function for, and CrossTL does not lower it with an exact range reduction yet (target hip) [project.translate.hip-metal-math-function-unsupported]`: `hip/cospi`
- `hip: project.translate.hip-metal-math-function-unsupported: MetalMathFunctionUnsupportedError: MSL sinpi is exact at integers and half-integers, which hip has no function for, and CrossTL does not lower it with an exact range reduction yet (target hip) [project.translate.hip-metal-math-function-unsupported]`: `hip/sinpi`
- `hlsl: project.translate.directx-stage-interface-unsupported: HlslStageInterfaceUnsupportedError: DirectX task stage: 'payload' is a taskPayloadSharedEXT variable; HLSL passes the payload to DispatchMesh and the mesh entry receives it as an in payload parameter, which is not implemented yet (target directx) [project.translate.directx-stage-interface-unsupported]`: `hlsl/DispatchMesh`
- `metal: CrossGL has no type for Metal's simd_vote mask`: `metal/simd_active_threads_mask`
- `mojo: CrossGL has no spelling for this operation: no CrossGL function returns a quotient and a remainder together`: `mojo/divmod`
- `rust: CrossGL has no spelling for this operation: Float::log takes a base; CrossGL's log is the natural logarithm`: `GLSL.std.450:Log+OpFDiv`
- `rust: CrossGL has no spelling for this operation: no CrossGL function converts a vector to an array`: `rust/Vec3::to_array`
- `rust: CrossGL has no spelling for this operation: no CrossGL function returns a float's mantissa, exponent and sign as integers`: `OpBitwiseAnd+OpBitwiseOr+OpIEqual+OpISub+OpShiftLeftLogical+OpShiftRightLogical`
- `rust: CrossGL has no spelling for this operation: no CrossGL function returns a vector's direction and length together`: `rust/Vec3::normalize_and_length`
- `rust: CrossGL has no spelling for this operation: no CrossGL function returns an optional value`: `rust/Vec3::try_normalize`
- `rust: CrossGL has no spelling for this operation: no CrossGL function returns the index of a vector's largest element`: `rust/Vec3::max_position`
- `rust: CrossGL has no spelling for this operation: no CrossGL function returns the index of a vector's smallest element`: `rust/Vec3::min_position`
- `rust: CrossGL has no spelling for this operation: no CrossGL function returns two vectors orthonormal to a vector`: `rust/Vec3::any_orthonormal_pair`
- `rust: CrossGL has no spelling for this operation: no CrossGL function writes a vector's elements to an array`: `rust/Vec3::write_to_slice`
- `slang: project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: Slang wave intrinsic: QuadAll is not recognized by the Slang backend (target slang) [project.translate.slang-lowering-unsupported]`: `OpGroupNonUniformQuadAllKHR`
- `slang: project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: Slang wave intrinsic: QuadAny is not recognized by the Slang backend (target slang) [project.translate.slang-lowering-unsupported]`: `OpGroupNonUniformQuadAnyKHR`
- `slang: project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: Slang wave intrinsic: WaveMultiPrefixCountBits is not recognized by the Slang backend (target slang) [project.translate.slang-lowering-unsupported]`: `slang/WaveMultiPrefixCountBits`
- `slang: project.translate.slang-metal-math-function-unsupported: MetalMathFunctionUnsupportedError: MSL cospi is exact at integers and half-integers, which slang has no function for, and CrossTL does not lower it with an exact range reduction yet (target slang) [project.translate.slang-metal-math-function-unsupported]`: `GLSL.std.450:Cos+OpVectorTimesScalar`
- `slang: project.translate.slang-metal-math-function-unsupported: MetalMathFunctionUnsupportedError: MSL sinpi is exact at integers and half-integers, which slang has no function for, and CrossTL does not lower it with an exact range reduction yet (target slang) [project.translate.slang-metal-math-function-unsupported]`: `GLSL.std.450:Sin+OpVectorTimesScalar`
- `slang: project.translate.slang-metal-math-function-unsupported: MetalMathFunctionUnsupportedError: MSL tanpi is exact at integers and half-integers, which slang has no function for, and CrossTL does not lower it with an exact range reduction yet (target slang) [project.translate.slang-metal-math-function-unsupported]`: `GLSL.std.450:Tan+OpVectorTimesScalar`
- `spirv: CrossGL has no spelling for this operation: NaN-aware clamp has no GLSL built-in`: `GLSL.std.450:NClamp`
- `wgsl: CrossGL has no spelling for this operation: no CrossGL sample clamps its coordinates half a texel inside the edge`: `GLSL.std.450:NClamp+OpFDiv+OpFSub+OpImageQuerySizeLod+OpImageSampleExplicitLod+OpSampledImage`

</details>

<details>
<summary>30 untested: function 26, method 4</summary>

- `slang: needs a target other than SPIR-V and HLSL` (11): `slang/GetInstanceTraversable`, `slang/GetTransformListHandle`, `slang/GetTransformListSize`, `slang/GetTraversableChild`, `slang/GetTraversableInstanceId`, `slang/GetTraversableObjectToWorld`, `slang/GetTraversableTransformType`, `slang/GetTraversableWorldToObject`, `slang/cudaBlockDim`, `slang/cudaBlockIdx`, `slang/cudaThreadIdx`
- `hlsl: needs shader model 6.10` (8): `hlsl/CandidateClusterID`, `hlsl/CandidateTriangleObjectPosition`, `hlsl/ClusterID`, `hlsl/CommittedClusterID`, `hlsl/CommittedTriangleObjectPosition`, `hlsl/GetGroupWaveCount`, `hlsl/GetGroupWaveIndex`, `hlsl/TriangleObjectPosition`
- `hlsl: DXC compiles printf only to SPIR-V; DXIL has no print operation`: `hlsl/printf`
- `rust: no probe for parameters of type F`: `rust/map`
- `rust: the probe itself is rejected: error: Fdim not supported yet`: `rust/abs_sub`
- `rust: the probe itself is rejected: error: Hypot not supported yet`: `rust/hypot`
- `slang: no probe for parameters of type DescriptorHandle<T>`: `slang/nonuniform`
- `slang: no probe for parameters of type IOpaqueDescriptor`: `slang/defaultGetDescriptorFromHandle`
- `slang: the probe itself is rejected: error: line 12: Operand 3 of Decorate requires one of these capabilities: RayTracingMotionBlurNV`: `BuiltIn:CurrentRayTimeNV`
- `slang: the probe itself is rejected: error: slangc emits /* debugBreak() not currently supported for HLSL */`: `slang/debugBreak`
- `slang: the probe itself is rejected: error: slangc exited with status -11 and no message`: `slang/createDynamicObject`
- `spirv: reserved by the GLSL.std.450 specification, not a valid instruction`: `GLSL.std.450:IMix`
- `wgsl: the probe itself is rejected: error: no definition in scope for identifier: 'subgroupElect'`: `wgsl/subgroupElect`

</details>
