Slang input imports `InterlockedCompareExchangeFloatBitwise` and `InterlockedCompareStoreFloatBitwise` on a float
buffer element or groupshared float as the HLSL importer does, so every target lowers them as it lowers the HLSL
calls. The Slang importer used to refuse both.
