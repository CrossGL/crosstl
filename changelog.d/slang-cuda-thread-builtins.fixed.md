The Slang importer imports `cudaThreadIdx()`, `cudaBlockIdx()` and
`cudaBlockDim()`, which Slang declares for its CUDA target only, as
`gl_LocalInvocationID`, `gl_WorkGroupID` and `gl_WorkGroupSize`. It copied the
calls unchanged, so every target refused the program as calling an undeclared
function (#4636).
