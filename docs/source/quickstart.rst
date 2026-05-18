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

Inspect registered sources and targets:

.. code-block:: python

   import crosstl

   print(crosstl.supported_sources())
   print(crosstl.supported_backends())
