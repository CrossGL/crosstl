SPIR-V output reads a `QuadReadLaneAt` lane computed at run time, which it used to refuse. Each of the four quad
lanes is read with `OpGroupNonUniformQuadBroadcast` and a constant index, and the lane `lane & 3` names is
selected, as GLSL output does. HLSL `QuadReadLaneAt` and Metal `quad_broadcast`/`quad_shuffle` with a lane that
differs per invocation now reach SPIR-V. A literal lane above 3 is still refused.
