GLSL output keeps the rename of a stage-local helper inside its stage. A vertex-stage helper named like a built-in
(`void subgroupBarrier()`) is renamed, and the fragment stage's call to the real built-in used to be renamed with it,
so the fragment section called a function it never declares (#5026).
