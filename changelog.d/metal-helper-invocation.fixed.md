Metal output reads CrossGL's `gl_HelperInvocation` fragment input through `simd_is_helper_thread()`. A
fragment-input structure member or entry-point parameter bound to `@ gl_HelperInvocation` used to be refused,
since Metal has no attribute for it, and a bare read of the built-in (as GLSL and HLSL `IsHelperLane()` sources
import it) was written verbatim and did not compile. An assignment to the built-in is refused.
