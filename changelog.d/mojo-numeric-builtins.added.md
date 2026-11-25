Mojo output writes CrossGL's `isnan`, `isinf`, `isfinite`, `nextafter`, `rcp`, `tgamma`, `nan`, `clz`, `ctz` and
`mulhi` with the functions of Mojo's `std.math` and `std.bit`, and `atomicLoad` and `atomicStore` as `std.atomic`'s
static `Atomic` load and store. An `if`, `else if`, loop condition or assignment target that increments, decrements
or assigns, such as `while (k++ < n)` or `vals[k++] = x`, runs those side effects before its test or store instead
of being refused.
