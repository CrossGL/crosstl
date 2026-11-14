SPIR-V assembly import treats functions named `main` as ordinary helpers unless
they are declared in `OpEntryPoint`. The importer renames those helpers and
their calls so they do not collide with the GLSL entry point (#4600).
