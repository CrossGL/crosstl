Extending CrossGL Translator
============================

CrossGL Translator extension points are registry based. New backends should be
registered through small specs rather than by branching the public
``translate`` entry point.

Add a target backend
--------------------

Target generators implement a class with a ``generate(ast)`` method, then
register a ``BackendSpec``.

.. code-block:: python

   from crosstl.translator.codegen import BackendSpec, register_backend

   class MyBackendCodeGen:
       def generate(self, ast):
           return "..."

   register_backend(
       BackendSpec(
           name="my-backend",
           codegen_class=MyBackendCodeGen,
           aliases=("myb",),
           file_extensions=(".myb",),
           format_backend="my-backend",
       )
   )

Add a source backend
--------------------

Source backends register a lexer/parser loader and, when supported, a reverse
code generator that emits CrossGL.

.. code-block:: python

   from crosstl.translator.source_registry import SourceSpec, SOURCE_REGISTRY

   def load_lexer_parser():
       return MyLexer, MyParser

   SOURCE_REGISTRY.register(
       SourceSpec(
           name="my-source",
           extensions=(".mysrc",),
           load_lexer_parser=load_lexer_parser,
           reverse_codegen_factory=MyToCrossGL,
           aliases=("mysrc",),
       )
   )

Plugin discovery
----------------

``discover_backend_plugins()`` looks for ``backend_spec`` and ``source_spec``
modules inside backend packages under ``crosstl.backend``. A package can expose
``BACKEND_SPEC``, ``BACKEND_SPECS``, ``SOURCE_SPEC``, ``SOURCE_SPECS``, or a
``register`` function.

This keeps new source and target integrations isolated from the core
translator package.
