<!-- Written from coverage.json by tools/coverage/docs.py when the Coverage workflow publishes. Do not edit. -->

# CrossGL IR

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z (CrossTL 3.9.0) by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). The Coverage workflow writes this page from `coverage.json` each time it publishes; [COVERAGE.md](../COVERAGE.md) defines each outcome. Score = supported / (supported + refused + wrong).

The CrossGL IR row counts the semantic features of every measured output language that CrossGL can express: a feature is expressible when its CrossGL probe translates correctly to at least one language that has it.

| Direction | Score | Supported | Refused | Wrong | Untested | Executed | Reference tool |
|---|---:|---:|---:|---:|---:|---|---|
| expressiveness | 94.1 % | 1570 | 98 | 0 | 18 |  |  |

## By category

Each cell counts the supported features of a category over its supported, refused and wrong ones.

| Category | expressiveness |
|---|---:|
| compute | 28/28 |
| control-flow | 62/62 |
| function | 1089/1187 |
| graphics | 73/73 |
| method | 113/113 |
| operator | 52/52 |
| resource | 67/67 |
| stage | 14/14 |
| type | 72/72 |

## Gaps

A diagnostic drops its line numbers and writes the probe's own variables as `<probe>`, and one that several features share writes each feature's name as `<feature>`.

### Expressiveness

Refused, by diagnostic:

- `slang: project.translate.undeclared-function: UndeclaredFunctionError: function '<feature>' is not declared: it is neither a CrossGL built-in function nor a function, structure method, type or callable value of the program (target slang) [project.translate.undeclared-function]` (23): `BuiltIn:ClusterIDNV`, `BuiltIn:CurrentRayTimeNV`, `BuiltIn:HitIsLSSNV`, `BuiltIn:HitIsSphereNV`, `BuiltIn:HitLSSPositionsNV+BuiltIn:HitLSSRadiiNV`, `BuiltIn:HitSpherePositionNV+BuiltIn:HitSphereRadiusNV`, `OpAtomicFAddEXT+OpBitwiseAnd+OpConvertPtrToU+OpIEqual`, `OpBufferPointerEXT+OpUntypedAccessChainKHR`, `OpCompositeExtract`, `OpCooperativeVectorLoadNV`, `OpTraceRayMotionNV`, `slang/bit_cast`, `slang/debugBreak`, `slang/floatCast`, `slang/getDescriptorFromExplicitHeapIndex`, `slang/getStringHash`, `slang/loadAligned`, `slang/makeTuple`, `slang/reinterpret`, `slang/storeAligned`, `slang/unmodified`, `slang/unused`, `slang/usesNativeSpirvDescriptorHandle`
- `mojo: project.translate.undeclared-function: UndeclaredFunctionError: function '<feature>' is not declared: it is neither a CrossGL built-in function nor a function, structure method, type or callable value of the program (target mojo) [project.translate.undeclared-function]` (18): `mojo/async_copy_arrive`, `mojo/cp_async_bulk_commit_group`, `mojo/cp_async_bulk_wait_group`, `mojo/mbarrier_arrive`, `mojo/mbarrier_arrive_expect_tx_relaxed`, `mojo/mbarrier_arrive_expect_tx_shared`, `mojo/mbarrier_init`, `mojo/mbarrier_test_wait`, `mojo/mbarrier_try_wait_parity_shared`, `mojo/named_barrier`, `mojo/named_barrier_arrive`, `mojo/s_waitcnt`, `mojo/s_waitcnt_barrier`, `mojo/scalb`, `mojo/schedule_barrier`, `mojo/sm_id`, `mojo/umma_arrive_leader_cta`, `mojo/umma_arrive_peer_cta`
- `rust: CrossGL has no spelling for this operation: no CrossGL built-in computes it` (16): `GLSL.std.450:Atan2+GLSL.std.450:Sqrt+OpFAdd+OpFDiv+OpFMul+OpFOrdGreaterThanEqual`, `GLSL.std.450:Atan2+GLSL.std.450:Sqrt+OpFAdd+OpFMul+OpFNegate+OpFOrdGreaterThan+OpFSub+OpIEqual+OpISub+OpUGreaterThan+OpUMod`, `GLSL.std.450:Atan2+GLSL.std.450:Sqrt+OpFAdd+OpFMul+OpFNegate+OpFOrdGreaterThan+OpIEqual+OpISub+OpUGreaterThan+OpUMod`, `GLSL.std.450:Cos+GLSL.std.450:Sin+GLSL.std.450:Sqrt+OpBitwiseAnd+OpBitwiseOr+OpFAdd+OpFDiv+OpFMul+OpFNegate+OpFOrdGreaterThan+OpFOrdLessThan+OpFSub+OpFUnordNotEqual`, `GLSL.std.450:Cos+GLSL.std.450:Sin+OpFAdd+OpFMul+OpFNegate+OpFSub+OpIAdd+OpIEqual+OpISub+OpUGreaterThan+OpULessThan+OpUMod`, `GLSL.std.450:FAbs+GLSL.std.450:Sin+GLSL.std.450:Sqrt+OpFAdd+OpFDiv+OpFMul+OpFNegate+OpFOrdGreaterThan+OpFOrdGreaterThanEqual+OpFOrdLessThan+OpFOrdLessThanEqual+OpFSub`, `GLSL.std.450:FAbs+GLSL.std.450:Sin+GLSL.std.450:Sqrt+OpFAdd+OpFDiv+OpFMul+OpFNegate+OpFOrdGreaterThan+OpFOrdGreaterThanEqual+OpFOrdLessThan+OpFSub`, `GLSL.std.450:FAbs+GLSL.std.450:Sin+GLSL.std.450:Sqrt+OpFAdd+OpFDiv+OpFMul+OpFOrdGreaterThan+OpFOrdGreaterThanEqual+OpFOrdLessThan+OpFSub`, `GLSL.std.450:FAbs+GLSL.std.450:Sqrt+OpFAdd+OpFMul+OpFOrdGreaterThanEqual+OpFOrdLessThan+OpFSub`, `GLSL.std.450:FAbs+OpFOrdGreaterThan`, `GLSL.std.450:Sqrt+OpBitwiseAnd+OpBitwiseOr+OpFAdd+OpFDiv+OpFMul+OpFOrdLessThanEqual+OpFSub+OpFUnordNotEqual`, `GLSL.std.450:Sqrt+OpFAdd+OpFDiv+OpFMul+OpFNegate+OpFOrdGreaterThan+OpFOrdLessThan+OpFSub`, `GLSL.std.450:Sqrt+OpFAdd+OpFDiv+OpFMul+OpFOrdLessThanEqual+OpFSub`, `OpCompositeConstruct+OpCompositeExtract`, `OpCompositeExtract+OpCompositeInsert`, `OpIAdd+OpLogicalEqual+OpSLessThan+OpULessThan`
- `rust: CrossGL has no spelling for this operation: no CrossGL function returns an optional value` (8): `OpIAdd+OpINotEqual+OpISub+OpSLessThan+OpULessThan`, `OpIAdd+OpINotEqual+OpLogicalEqual+OpLogicalNotEqual+OpSGreaterThan+OpSLessThan`, `OpIAdd+OpINotEqual+OpLogicalEqual+OpSGreaterThan+OpSLessThan`, `OpIAdd+OpINotEqual+OpLogicalNotEqual+OpSLessThan+OpULessThan`, `OpIEqual+OpINotEqual+OpSDiv`, `OpINotEqual+OpISub+OpLogicalEqual+OpLogicalNotEqual+OpSLessThan`, `OpINotEqual+OpISub+OpLogicalEqual+OpSLessThan`, `rust/Vec3::try_normalize`
- `rust: project.translate.undeclared-function: UndeclaredFunctionError: function '<feature>' is not declared: it is neither a CrossGL built-in function nor a function, structure method, type or callable value of the program (target rust) [project.translate.undeclared-function]` (4): `OpAtomicIDecrement`, `OpAtomicIIncrement`, `OpControlBarrier(3,3,0)`, `OpDemoteToHelperInvocation`
- `slang: project.translate.undeclared-function: UndeclaredFunctionError: function 'CoopVec' is not declared: it is neither a CrossGL built-in function nor a function, structure method, type or callable value of the program (target slang) [project.translate.undeclared-function]` (4): `OpCooperativeVectorMatrixMulAddNV+OpIAdd`, `OpCooperativeVectorMatrixMulNV+OpIAdd`, `OpCooperativeVectorOuterProductAccumulateNV+OpIAdd`, `OpCooperativeVectorReduceSumAccumulateNV+OpIAdd`
- `rust: CrossGL has no spelling for this operation: CrossGL declares a byte-address buffer as a resource, and makes none from a storage buffer or from another` (3): `rust/ByteAddressableBuffer::as_ref`, `rust/ByteAddressableBuffer::from_mut_slice`, `rust/ByteAddressableBuffer::from_slice`
- `mojo: CrossGL has no spelling for this operation: no CrossGL warp reduction applies a function the kernel passes` (2): `mojo/warp.lane_group_reduce`, `mojo/warp.reduce`
- `rust: CrossGL has no spelling for this operation: no CrossGL function converts a matrix to an array` (2): `rust/Mat4::to_cols_array`, `rust/Mat4::to_rows_array`
- `rust: CrossGL has no spelling for this operation: no CrossGL function converts a vector to an array` (2): `rust/Quat::to_array`, `rust/Vec3::to_array`
- `rust: project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N constructs: resource: uniform 'u0': 'ByteAddressBuffer' has type ByteAddressBuffer, which has no rust-gpu buffer layout; function: 'u0.Load' has no rust-gpu lowering for these operands (target rust) [project.translate.rust-lowering-unsupported]` (2): `OpArrayLength+OpIEqual+OpIMul+OpShiftRightLogical+OpUGreaterThan+OpUMod`, `OpIMul+OpShiftRightLogical`
- `mojo: CrossGL has no AMD instruction scheduling groups`: `mojo/schedule_group_barrier`
- `mojo: CrossGL has no spelling for this operation: CrossGL's compare-and-swap returns the value it found, and compare_exchange returns whether it stored and writes the value it found to its expected argument`: `mojo/Atomic.compare_exchange`
- `mojo: CrossGL has no spelling for this operation: CrossGL's lane match over the whole warp is CUDA's __match_all_sync, which also returns whether the lanes agree through a pointer`: `mojo/warp.match_all`
- `mojo: CrossGL has no spelling for this operation: no CrossGL function returns a quotient and a remainder together`: `mojo/divmod`
- `rust: CrossGL has no spelling for this operation: CrossGL declares an acceleration structure as a resource, and makes none from an address`: `OpConvertUToAccelerationStructureKHR`
- `rust: CrossGL has no spelling for this operation: CrossGL has no closures or function values`: `rust/Vec3::map`
- `rust: CrossGL has no spelling for this operation: no CrossGL function returns a float's mantissa, exponent and sign as integers`: `OpBitwiseAnd+OpBitwiseOr+OpIEqual+OpISub+OpShiftLeftLogical+OpShiftRightLogical`
- `rust: CrossGL has no spelling for this operation: no CrossGL function returns a vector's direction and length together`: `rust/Vec3::normalize_and_length`
- `rust: CrossGL has no spelling for this operation: no CrossGL function returns two vectors orthonormal to a vector`: `rust/Vec3::any_orthonormal_pair`
- `rust: project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: function: 'printf' has no rust-gpu lowering for these operands (target rust) [project.translate.rust-lowering-unsupported]`: `rust/debug_printfln!`
- `rust: project.translate.undeclared-function: UndeclaredFunctionError: function 'angle_between' is not declared: it is neither a CrossGL built-in function nor a function, structure method, type or callable value of the program (target rust) [project.translate.undeclared-function]`: `rust/angle_between`
- `rust: project.translate.undeclared-function: UndeclaredFunctionError: function 'rotate_axis' is not declared: it is neither a CrossGL built-in function nor a function, structure method, type or callable value of the program (target rust) [project.translate.undeclared-function]`: `rust/rotate_axis`
- `rust: project.translate.undeclared-function: UndeclaredFunctionError: function 'rotate_towards' is not declared: it is neither a CrossGL built-in function nor a function, structure method, type or callable value of the program (target rust) [project.translate.undeclared-function]`: `rust/rotate_towards`
- `rust: project.translate.undeclared-function: UndeclaredFunctionError: function 'slerp' is not declared: it is neither a CrossGL built-in function nor a function, structure method, type or callable value of the program (target rust) [project.translate.undeclared-function]`: `rust/slerp`

Untested, by reason:

- `slang: needs the OptiX SDK headers, which the CUDA toolkit does not include` (8): `slang/GetInstanceTraversable`, `slang/GetTransformListHandle`, `slang/GetTransformListSize`, `slang/GetTraversableChild`, `slang/GetTraversableInstanceId`, `slang/GetTraversableObjectToWorld`, `slang/GetTraversableTransformType`, `slang/GetTraversableWorldToObject`
- `rust: the probe itself is rejected: error: cannot cast between pointer types` (4): `rust/IVec3::chebyshev_distance`, `rust/Mat4::write_cols_to_slice`, `rust/Quat::write_to_slice`, `rust/Vec3::write_to_slice`
- `rust: the probe itself is rejected: error: checked mul is not supported yet` (2): `rust/IVec3::checked_mul`, `rust/IVec3::saturating_mul`
- `rust: the probe itself is rejected: error: Fdim not supported yet`: `rust/abs_sub`
- `rust: the probe itself is rejected: error: unsupported constant of type 'u128'`: `rust/Mat4::try_inverse`
- `slang: the probe itself is rejected: error: slangc exited with status -11 and no message`: `slang/createDynamicObject`
- `spirv: reserved by the GLSL.std.450 specification, not a valid instruction`: `GLSL.std.450:IMix`
