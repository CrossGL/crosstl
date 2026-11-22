HLSL, Metal, Slang and HIP output number an enum variant without a value as one
more than the previous variant also after a computed initializer such as `-5`,
`1 << 3` or `W::P + 3`, writing `V_B = V_A + 1`. They used to restart at 0, so
the program compiled and computed with the wrong values. Metal input writes a
scoped enum's `Q = P + 3` as `Q = W::P + 3`, so the earlier enumerator resolves.
