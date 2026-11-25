Metal output writes CrossGL's `atomicCompSwap` on a `float`, which HLSL's
`InterlockedCompareExchangeFloatBitwise` and `InterlockedCompareStoreFloatBitwise`
import to, for device, group-shared and buffer-block destinations. The helper
exchanges the destination's word through `atomic_uint` and compares
`as_type<uint>` of the operands, so the comparison is of bits as in shader model
6.6: `-0.0` does not match a stored `0.0`, and a NaN matches its own bits. Metal
output used to refuse a `float` destination.
