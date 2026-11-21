CUDA and HIP input refuse device-side `new`, `new[]`, `malloc`, `free` and `std::make_unique` with a diagnostic, as
they already refused `delete`, because CrossGL has no device heap. They used to write the calls into the CrossGL, so
`--backend cgl` produced code that does not translate further. Host code and a program's own functions named
`malloc` or `free` are unchanged (#4684).
