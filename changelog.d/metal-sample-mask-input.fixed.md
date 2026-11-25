Metal output reads GLSL's and SPIR-V's `int` coverage-mask input (`int gl_SampleMaskIn[1]`, or a scalar `int`) from a `uint [[sample_mask]]` entry parameter instead of refusing it.
