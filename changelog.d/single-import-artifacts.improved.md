`crosstl.translate_artifacts` translates a file into the artifacts its target needs, importing the source once.
When the target's compilation unit holds one entry point per stage (GLSL, GLSL ES) and the source declares two
entries of one stage, it translates the discovered entries one artifact each from a single import, instead of
importing the whole file only to learn its entries from the generator's refusal. The corpus worker uses it; on the
Metal-source corpus slices this cuts GLSL translation CPU time by about a fifth.
