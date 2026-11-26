HLSL output compares two pointers into the same buffer with the same element view by their offsets. It used to
write `p != q` with neither name declared in HLSL. A pointer used as a truth value, compared with null, with an
integer or with a pointer into another buffer is refused with a `DirectXLoweringRefused` diagnostic that names the
comparison.
