WGSL output writes a negative integer literal stored to a `uvecN`, in a
declaration, an assignment or a compound assignment, as `vecN<u32>(u32(-1i))`,
as it already did for a scalar `uint`. It used to write `u32(-1)`, which naga
rejected (#4504).
