Mojo output writes a pointer dereference `*p` as `p[]` and a pointer member access
`p->m` as `p[].m`, Mojo's dereference subscript. It used to write `(*p)`, which Mojo
reads as a starred expression, and printed `p->m` as the parser's internal
`PointerAccessNode(...)` text. An expression the Mojo generator has no rule for is
now refused with a diagnostic that names it, instead of being printed as that text.
A bare `return` and a `for` loop with an empty condition are written as `return` and
`while True:` instead of `return None` and `while None:` (#4752).
