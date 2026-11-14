CUDA and HIP importers now rename structs, enums, and type aliases whose names
collide with any CrossGL lexer keyword (e.g. `half`, `fn`, `mesh`, `kernel`)
so that the imported CrossGL compiles without syntax errors.
