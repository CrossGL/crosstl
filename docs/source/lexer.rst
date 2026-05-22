Lexer
=====

The CrossGL lexer turns source text into ``(token_type, text)`` pairs consumed
by ``crosstl.translator.parser.Parser``.

Token ordering
--------------

Token patterns are stored in an ``OrderedDict``. More specific patterns must
appear before broader patterns because the combined regular expression reports
the first matching named group. For example, compound operators such as ``+=``
and ``<<=`` are listed before their single-character forms, and identifiers are
listed near the end.

Keyword handling
----------------

Identifier text is checked against ``KEYWORDS`` after token matching. This lets
the lexer share one identifier regex while still emitting parser-friendly token
types such as ``STRUCT``, ``COMPUTE``, ``VEC4``, and ``SAMPLER2D``.

Comments and whitespace
-----------------------

Whitespace is skipped. Comment tokens are preserved so parser helpers can
consume them consistently with ``skip_comments()`` while still allowing source
frontends to retain comment awareness if needed.

Errors
------

If no token pattern matches the current character, the lexer raises a
``SyntaxError`` with a line, column, source line, and caret pointer. This keeps
syntax errors actionable before parsing begins.

Debugging
---------

Use ``Lexer(code).debug_print()`` to print token indexes, token types, and raw
text. This is useful when adding grammar support or diagnosing a parser
lookahead bug.
