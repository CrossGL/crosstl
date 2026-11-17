SPIR-V input translates `OpPtrAccessChain` the way SPIR-V defines it: its Element operand advances the base
pointer, so it adds to the subscript that selects the object the pointer points at instead of becoming an
extra subscript of its own. It used to be imported as another traversal index, so `&outp[0] + i` became
`outp[0].__member0[0][i]` — a subscript on a float — and the SPIR-V generator re-emitted it as an
`OpAccessChain` that spirv-val rejects for reaching a non-composite type while indexes remain. That only
showed once the #4531 block rename let such a module past the unsized-array refusal, turning a clean refusal
of the SPIR-V input `pointer-arithmetic` coverage probe into invalid SPIR-V on the round trip; the probe now
translates and round-trips to valid SPIR-V. A bare pointer advanced by a nonzero Element still has no CrossGL
form and is refused with a typed error, and the `OpPtrAccessChain` fixture in the backend tests now keeps the
valid upstream shape instead of the reduced form that dropped an index (#4531).
