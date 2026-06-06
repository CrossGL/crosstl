Doxygen
-------

The repository includes a Doxygen configuration for generating XML from the
Python package. Sphinx can consume that XML through Breathe when deeper
cross-reference pages are needed.

Build Doxygen XML:

.. code-block:: bash

   make -C docs doxygen

Build HTML documentation:

.. code-block:: bash

   make -C docs html

Generated Doxygen output is written to ``docs/doxygen/build`` and is ignored by
git.
