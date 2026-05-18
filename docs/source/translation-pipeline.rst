Translation Pipeline
====================

The public ``crosstl.translate`` function is a thin orchestration layer around
the source registry, parser frontends, and target code generators.

Pipeline steps
--------------

1. Register built-in source frontends with ``register_default_sources()``.
2. Discover backend plugin specs with ``discover_backend_plugins()``.
3. Select the source frontend by input file extension.
4. Parse source text into a source AST.
5. If the input is not CrossGL and the requested target is not CrossGL, emit an
   intermediate CrossGL program and parse that into the canonical AST.
6. Resolve the target backend through the backend registry.
7. Generate target source from the canonical AST.
8. Optionally format and save the generated source.

Source selection
----------------

``SOURCE_REGISTRY.get_by_extension(path)`` maps the input extension to a
``SourceSpec``. Each spec owns a lazy lexer/parser loader so importing
``crosstl`` does not eagerly import every backend.

CrossGL as the interchange form
-------------------------------

Native source frontends can provide a reverse code generator that emits
CrossGL. CrossGL then becomes the interchange form for native-to-native
translation. For example, HLSL-to-Metal is handled as HLSL AST to CrossGL text,
then CrossGL AST to Metal.

Target dispatch
---------------

``get_codegen(backend)`` resolves the requested backend name or alias and
instantiates the registered code generator class. Generators consume the
canonical CrossGL AST and return target source text.

Validation and formatting
-------------------------

The parser finalizes shader nodes through validation hooks such as constant
buffer validation. After generation, ``format_shader_code`` can format target
source when a formatter is available for that backend.
