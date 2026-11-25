When a call like `sel(Foo, f)` names a struct type as its leading argument, every target now strips that leading
type argument from the emitted call, so `sel(Foo, f)` emits the same call as `sel<Foo>(f)` on every C-style
target. WGSL and SPIR-V still raise for calls they cannot infer concrete type arguments for (#4430).
