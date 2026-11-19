SPIR-V input no longer refuses a module whose storage-buffer block is named like the resource type that wraps
it. slangc names the block of every `RWStructuredBuffer<T>` `RWStructuredBuffer`, which used to be imported as
`RWStructuredBuffer<RWStructuredBuffer>` and refused as an unsized array that ends no storage buffer; the block
now takes a suffixed name instead, so the module translates exactly as a renamed block does (#4531).
