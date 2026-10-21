Seed inputs for the importer fuzz campaign (`tests/conformance/fuzz/targets.py`), which mutates every file under
`tests/` and `examples/` with an importer's extension. The Mojo and Rust importers have no program printer, so these
files are their only inputs, and Slang has no other fixture. They are CrossTL's Mojo, Rust and Slang output for
`examples/` programs at the time they were added; they are fixed inputs and are never regenerated.
