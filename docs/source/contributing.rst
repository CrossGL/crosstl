Contributing
============

We welcome contributions to CrossTL! This guide will help you get started.

Development Setup
-----------------

1. Clone the repository::

    git clone https://github.com/CrossGL/crosstl.git
    cd crosstl

2. Create a virtual environment::

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install development dependencies::

    pip install -e .
    pip install -r requirements.txt

Running Tests
-------------

CrossTL uses pytest for testing::

    # Run all tests
    pytest

    # Run tests with coverage
    pytest --cov=crosstl

    # Run specific test file
    pytest tests/test_translator/test_lexer.py

Code Style
----------

CrossTL follows PEP 8 style guidelines. Use the following tools to ensure
code quality:

* **black** for code formatting
* **isort** for import sorting
* **flake8** for linting

Building Documentation
----------------------

To build the documentation locally::

    cd docs
    pip install -r ../requirements-docs.txt
    make html

The built documentation will be available in ``docs/_build/html/``.

Architecture Overview
---------------------

CrossTL follows a modular architecture:

1. **Translator Module** (``crosstl/translator/``)

   * ``lexer.py`` - Tokenizes CrossGL source code
   * ``parser.py`` - Builds AST from tokens
   * ``ast.py`` - AST node definitions
   * ``codegen/`` - Code generators for each backend

2. **Backend Module** (``crosstl/backend/``)

   * Contains parsers for reverse translation
   * Each backend has: Lexer, Parser, AST, CrossGLCodeGen

3. **Formatter Module** (``crosstl/formatter.py``)

   * Code formatting using external tools

Adding a New Backend
--------------------

To add support for a new shader language:

1. Create a new directory under ``crosstl/backend/YourBackend/``
2. Implement the lexer (``YourBackendLexer.py``)
3. Implement the parser (``YourBackendParser.py``)
4. Implement AST nodes (``YourBackendAst.py``)
5. Implement reverse codegen (``YourBackendCrossGLCodeGen.py``)
6. Create a codegen in ``crosstl/translator/codegen/yourbackend_codegen.py``
7. Register the backend in ``crosstl/translator/codegen/__init__.py``
8. Add tests in ``tests/test_backend/test_yourbackend/``
