<!-- Written from coverage.json by tools/coverage/docs.py when the Coverage workflow publishes. Do not edit. -->

# Slang

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z (CrossTL 3.9.0) by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). The Coverage workflow writes this page from `coverage.json` each time it publishes; [COVERAGE.md](../COVERAGE.md) defines each outcome. Score = supported / (supported + refused + wrong).

The [backend guide](https://github.com/CrossGL/crosstl/blob/main/docs/source/backends/slang.rst) describes how CrossTL translates Slang.

| Direction | Score | Supported | Refused | Wrong | Untested | Executed | Reference tool |
|---|---:|---:|---:|---:|---:|---|---|
| output | 93.8 % | 655 | 43 | 0 | 9 | 443 on vulkan-lavapipe | slangc (2026.20) |
| input | 93.0 % | 649 | 48 | 1 | 9 | 439 on vulkan-lavapipe | slangc (2026.20) |

## Feature lists

- [Slang v2026.20 core module: HLSL-compatible built-in functions](https://github.com/shader-slang/slang/blob/d9ede2dd5836afc0e9586da18d9236b171d0faa8/source/slang/hlsl.meta.slang), pinned at `d9ede2dd5836`
- [Slang v2026.20 core module: core built-in functions](https://github.com/shader-slang/slang/blob/d9ede2dd5836afc0e9586da18d9236b171d0faa8/source/slang/core.meta.slang), pinned at `d9ede2dd5836`

## By category

Each cell counts the supported features of a category over its supported, refused and wrong ones.

| Category | output | input |
|---|---:|---:|
| compute | 24/24 | 24/24 |
| control-flow | 62/62 | 62/62 |
| function | 305/343 | 298/343 |
| graphics | 66/67 | 65/67 |
| operator | 51/51 | 51/51 |
| resource | 65/67 | 67/67 |
| stage | 14/14 | 14/14 |
| type | 68/70 | 68/70 |

## Translation pairs

The cells of the translation matrix that translate Slang straight to another language or another language straight to it, measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z. Each cell measures the features of the source's input row that the target has too, and its page lists the gaps.

| Source | Target | Score | Supported | Refused | Wrong | Untested | Executed |
|---|---|---:|---:|---:|---:|---:|---|
| Slang | CUDA | [97.6 %](../pairs/slang-to-cuda.md) | 247 | 6 | 0 | 0 | compile-only |
| Slang | GLSL ES | [93.9 %](../pairs/slang-to-gles.md) | 428 | 21 | 7 | 7 | compile-only |
| Slang | GLSL | [96.4 %](../pairs/slang-to-glsl.md) | 488 | 8 | 10 | 0 | yes |
| Slang | HIP | [97.3 %](../pairs/slang-to-hip.md) | 248 | 6 | 1 | 0 | compile-only |
| Slang | HLSL | [92.7 %](../pairs/slang-to-hlsl.md) | 493 | 11 | 28 | 0 | yes |
| Slang | Metal | [98.6 %](../pairs/slang-to-metal.md) | 426 | 5 | 1 | 0 | compile-only |
| Slang | Mojo | [93.9 %](../pairs/slang-to-mojo.md) | 216 | 11 | 3 | 0 | compile-only |
| Slang | Rust | [79.2 %](../pairs/slang-to-rust.md) | 361 | 83 | 12 | 0 | yes |
| Slang | SPIR-V | [91.0 %](../pairs/slang-to-spirv.md) | 443 | 12 | 32 | 0 | yes |
| Slang | WebGL | [64.7 %](../pairs/slang-to-webgl.md) | 11 | 5 | 1 | 0 | compile-only |
| Slang | WGSL | [86.0 %](../pairs/slang-to-wgsl.md) | 325 | 18 | 35 | 1 | yes |
| CUDA | Slang | [96.8 %](../pairs/cuda-to-slang.md) | 275 | 8 | 1 | 0 | compile-only |
| GLSL ES | Slang | [96.8 %](../pairs/gles-to-slang.md) | 423 | 11 | 3 | 7 | compile-only |
| GLSL | Slang | [95.9 %](../pairs/glsl-to-slang.md) | 470 | 17 | 3 | 0 | yes |
| HIP | Slang | [96.8 %](../pairs/hip-to-slang.md) | 276 | 8 | 1 | 0 | compile-only |
| HLSL | Slang | [96.7 %](../pairs/hlsl-to-slang.md) | 503 | 8 | 9 | 0 | yes |
| Metal | Slang | [92.5 %](../pairs/metal-to-slang.md) | 395 | 27 | 5 | 0 | compile-only |
| Mojo | Slang | [73.8 %](../pairs/mojo-to-slang.md) | 169 | 59 | 1 | 0 | compile-only |
| OpenCL C | Slang | [89.7 %](../pairs/opencl-to-slang.md) | 227 | 26 | 0 | 0 | compile-only |
| Rust | Slang | [56.9 %](../pairs/rust-to-slang.md) | 269 | 115 | 89 | 0 | yes |
| SPIR-V | Slang | [90.5 %](../pairs/spirv-to-slang.md) | 427 | 35 | 10 | 0 | yes |
| WebGL | Slang | [99.6 %](../pairs/webgl-to-slang.md) | 265 | 0 | 1 | 0 | compile-only |
| WGSL | Slang | [96.4 %](../pairs/wgsl-to-slang.md) | 346 | 2 | 11 | 2 | yes |

## Gaps

A diagnostic drops its line numbers and writes the probe's own variables as `<probe>`, and one that several features share writes each feature's name as `<feature>`.

### Output

Refused, by diagnostic:

- `project.translate.undeclared-function: UndeclaredFunctionError: function '<feature>' is not declared: it is neither a CrossGL built-in function nor a function, structure method, type or callable value of the program (target slang) [project.translate.undeclared-function]` (30): `GetClusterID`, `GetLssPositionsAndRadii`, `GetSpherePositionAndRadius`, `InterlockedAddF16Emulated`, `InterlockedAddF16x2`, `IsLssHit`, `IsSphereHit`, `RayCurrentTime`, `ReportHitOptix`, `TraceMotionRay`, `bit_cast`, `coopVecLoad`, `coopVecLoadCoherent`, `coopVecLoadGroupshared`, `debugBreak`, `defaultGetDescriptorFromHandle`, `f32tof16_`, `floatCast`, `getDescriptorFromDescriptorHeapEXT`, `getDescriptorFromDynamicResourceHeap`, `getDescriptorFromExplicitHeapIndex`, `getStringHash`, `loadAligned`, `makeTuple`, `nonuniform`, `reinterpret`, `storeAligned`, `unmodified`, `unused`, `usesNativeSpirvDescriptorHandle`
- `project.translate.undeclared-function: UndeclaredFunctionError: function 'CoopVec' is not declared: it is neither a CrossGL built-in function nor a function, structure method, type or callable value of the program (target slang) [project.translate.undeclared-function]` (6): `coopVecMatMul`, `coopVecMatMulAdd`, `coopVecMatMulAddPacked`, `coopVecMatMulPacked`, `coopVecOuterProductAccumulate`, `coopVecReduceSumAccumulate`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: Slang image access: imageStore requires an image resource (target slang) [project.translate.slang-lowering-unsupported]`: `storage-texel-buffer`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: Slang sampled texture: texelFetch requires a non-shadow texel-fetchable sampled texture resource (target slang) [project.translate.slang-lowering-unsupported]`: `texel-buffer`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: pointer declaration: 'main.p' points into thread memory, and Slang defines no pointer to a local variable (Slang User's Guide, "Pointers (limited)") (target slang) [project.translate.slang-lowering-unsupported]`: `pointer-local`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N constructs: pointer declaration: 'main.p' points into thread memory, and Slang defines no pointer to a local variable (Slang User's Guide, "Pointers (limited)"); pointer declaration: 'main.q' points into thread memory, and Slang defines no pointer to a local variable (Slang User`: `pointer-arithmetic`
- `project.translate.slang-refused: SlangRefusal: Slang geometry stage must declare at most one stream output parameter`: `geometry-streams`
- `project.translate.undeclared-function: UndeclaredFunctionError: function 'Tuple' is not declared: it is neither a CrossGL built-in function nor a function, structure method, type or callable value of the program (target slang) [project.translate.undeclared-function]`: `concat`
- `project.translate.workgroup-uniform-load-unsupported: WorkgroupUniformLoadError: workgroupUniformLoad: '<probe>' is not a shared variable declared at program or stage scope for slang (target slang) [project.translate.workgroup-uniform-load-unsupported]`: `workgroupUniformLoad`

Untested, by reason:

- `needs the OptiX SDK headers, which the CUDA toolkit does not include` (8): `GetInstanceTraversable`, `GetTransformListHandle`, `GetTransformListSize`, `GetTraversableChild`, `GetTraversableInstanceId`, `GetTraversableObjectToWorld`, `GetTraversableTransformType`, `GetTraversableWorldToObject`
- `the probe itself is rejected: error: slangc exited with status -11 and no message`: `createDynamicObject`

### Input

Wrong, by cause ([open bug issues](https://github.com/CrossGL/crosstl/issues?q=is:issue%20is:open%20label:coverage%20label:bug%20label:backend:slang%20label:direction:input)):

- `attribute patchconstantfunc only valid for HS.`: `tessellation-patch-input`

Refused, by diagnostic:

- `project.translate.undeclared-function: UndeclaredFunctionError: function '<feature>' is not declared: it is neither a CrossGL built-in function nor a function, structure method, type or callable value of the program (target slang) [project.translate.undeclared-function]` (30): `GetClusterID`, `GetLssPositionsAndRadii`, `GetSpherePositionAndRadius`, `HitTriangleVertexPosition`, `IsLssHit`, `IsSphereHit`, `RayCurrentTime`, `ReorderThread`, `ReportHitOptix`, `TraceMotionRay`, `bit_cast`, `coopVecLoad`, `coopVecLoadCoherent`, `coopVecLoadGroupshared`, `debugBreak`, `defaultGetDescriptorFromHandle`, `f32tof16_`, `floatCast`, `getDescriptorFromDescriptorHeapEXT`, `getDescriptorFromDynamicResourceHeap`, `getDescriptorFromExplicitHeapIndex`, `getStringHash`, `loadAligned`, `makeTuple`, `nonuniform`, `reinterpret`, `storeAligned`, `unmodified`, `unused`, `usesNativeSpirvDescriptorHandle`
- `project.translate.undeclared-function: UndeclaredFunctionError: function 'CoopVec' is not declared: it is neither a CrossGL built-in function nor a function, structure method, type or callable value of the program (target slang) [project.translate.undeclared-function]` (6): `coopVecMatMul`, `coopVecMatMulAdd`, `coopVecMatMulAddPacked`, `coopVecMatMulPacked`, `coopVecOuterProductAccumulate`, `coopVecReduceSumAccumulate`
- `project.translate.slang-metal-math-function-unsupported: MetalMathFunctionUnsupportedError: <feature> is exact at integers and half-integers, which slang has no function for, and CrossTL does not lower it with an exact range reduction yet (target slang) [project.translate.slang-metal-math-function-unsupported]` (3): `cospi`, `sinpi`, `tanpi`
- `project.translate.slang-import-unsupported: SlangImportRefused: CrossGL cannot represent N Slang construct: Slang <feature> has no CrossGL atomic operation (target cgl) [project.translate.slang-import-unsupported]` (2): `InterlockedAddF16Emulated`, `InterlockedAddF16x2`
- `project.translate.slang-import-unsupported: SlangImportRefused: CrossGL cannot represent N Slang construct: Slang InterlockedCompareStoreFloatBitwise needs a float location (target cgl) [project.translate.slang-import-unsupported]`: `InterlockedCompareStoreFloatBitwise`
- `project.translate.slang-import-unsupported: SlangImportRefused: CrossGL cannot represent N Slang construct: Slang InterlockedCompareExchangeFloatBitwise takes N arguments, not N (target cgl) [project.translate.slang-import-unsupported]`: `InterlockedCompareExchangeFloatBitwise`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: pointer declaration: 'main.p' points into thread memory, and Slang defines no pointer to a local variable (Slang User's Guide, "Pointers (limited)") (target slang) [project.translate.slang-lowering-unsupported]`: `pointer-local`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N constructs: pointer declaration: 'main.p' points into thread memory, and Slang defines no pointer to a local variable (Slang User's Guide, "Pointers (limited)"); pointer declaration: 'main.q' points into thread memory, and Slang defines no pointer to a local variable (Slang User`: `pointer-arithmetic`
- `project.translate.slang-refused: SlangRefusal: Slang geometry stage must declare at most one stream output parameter`: `geometry-streams`
- `project.translate.undeclared-function: UndeclaredFunctionError: function 'Tuple' is not declared: it is neither a CrossGL built-in function nor a function, structure method, type or callable value of the program (target slang) [project.translate.undeclared-function]`: `concat`
- `project.translate.workgroup-uniform-load-unsupported: WorkgroupUniformLoadError: workgroupUniformLoad: '<probe>' is not a shared variable declared at program or stage scope for slang (target slang) [project.translate.workgroup-uniform-load-unsupported]`: `workgroupUniformLoad`

Untested, by reason:

- `needs the OptiX SDK headers, which the CUDA toolkit does not include` (8): `GetInstanceTraversable`, `GetTransformListHandle`, `GetTransformListSize`, `GetTraversableChild`, `GetTraversableInstanceId`, `GetTraversableObjectToWorld`, `GetTraversableTransformType`, `GetTraversableWorldToObject`
- `the probe itself is rejected: error: slangc exited with status -11 and no message`: `createDynamicObject`
