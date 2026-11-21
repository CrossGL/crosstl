Metal input renames a structure whose name CrossGL reads as a keyword, such as `let`, `var`, `vec2` or `task`. The
importer used to write `struct let`, which the CrossGL parser refused as a syntax error; the structure now takes the
suffixed name `let_` at its declaration and at every use, and a structure the program itself names `let_` keeps that
name.
