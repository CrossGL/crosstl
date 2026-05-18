Architecture
============

CrossGL Translator uses registries to connect source parsers, reverse
generators, and target code generators.

Frontend
--------

The CrossGL lexer and parser produce the canonical AST in
``crosstl.translator.ast``. That AST represents shader stages, declarations,
statements, expressions, resources, attributes, and backend-specific metadata.

Source backends
---------------

Native shader inputs are registered through
``crosstl.translator.source_registry``. Each source spec owns parsing and, when
available, reverse code generation back into CrossGL.

Target backends
---------------

Target code generators are registered through
``crosstl.translator.codegen.registry``. The public ``crosstl.translate`` entry
point normalizes the requested backend name, loads plugins, and dispatches to
the matching generator.
