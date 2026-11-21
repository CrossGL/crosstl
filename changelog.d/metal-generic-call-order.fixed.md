Metal output defines a generic function before a function that calls it with explicit type arguments, such as
`twice<float>(x)`. It used to write the caller first, where the Metal compiler rejected the call to the
not-yet-declared `twice_float` (#4816).
