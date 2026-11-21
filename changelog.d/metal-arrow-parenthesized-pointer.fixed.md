Metal output keeps the parentheses around the pointer of a `->` access. `(s + 1)->a` used to be written as `s + 1->a`, which Clang reads as `s + (1->a)` and the Metal compiler rejects (#4750).
