Quickstart
==========

Install the package in editable mode:

.. code-block:: bash

   pip install -e .

Translate a CrossGL shader:

.. code-block:: python

   import crosstl

   output = crosstl.translate("examples/graphics/SimpleShader.cgl", backend="metal")
   print(output)

Use the command-line entry point:

.. code-block:: bash

   python -m crosstl._crosstl examples/graphics/SimpleShader.cgl --backend metal

Audit a repository before writing translated project artifacts:

.. code-block:: bash

   python -m crosstl._crosstl scan /path/to/repo \
     --target metal \
     --output scan-report.json

   python -m crosstl._crosstl translate-project /path/to/repo \
     --target metal \
     --output-dir crosstl-out \
     --report crosstl-out/portability-report.json

   python -m crosstl._crosstl validate-project \
     crosstl-out/portability-report.json \
     --format text

Project reports focus on shader and kernel source translation. Host runtime
API migration, build-system rewrites, and framework integration remain manual
migration work tracked in the report. See :doc:`project-porting` for the full
workflow and report contract.

Inspect registered sources and targets:

.. code-block:: python

   import crosstl

   print(crosstl.supported_sources())
   print(crosstl.supported_backends())
