HLSL output writes a CrossGL `node` stage as a `[Shader("node")]` entry of a
Shader Model 6.8 library, with its launch, dispatch-grid, recursion, node-id,
input-sharing and record attributes in their HLSL spelling, `globallycoherent`
read-write input records and the thread-position system values the launch mode
provides. HLSL input imports a `[Shader("node")]` entry as a CrossGL node stage
with the same attributes, so HLSL work graphs round-trip (#4794).
