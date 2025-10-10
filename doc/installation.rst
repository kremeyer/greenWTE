Installation
=============

Requirements
------------
greenWTE requires Python 3.10 or higher and the `NVIDIA CUDA toolkit`_ 12 or 13.
Note that currently numpy 2.0 or higher is not compatible with CUDA.
Python package requirements can be found `in the pyproject.toml file`_.

.. _NVIDIA CUDA toolkit: https://developer.nvidia.com/cuda-toolkit
.. _in the pyproject.toml file: https://github.com/kremeyer/greenWTE/blob/main/pyproject.toml

Installation
------------
A PyPI package is in preparation for version 1.0.
In the meantime, you can install greenWTE directly from the GitHub repository using pip:

.. code-block:: bash

    python3 -m pip install "greenWTE[cuda13x] @ git+ssh://git@github.com/kremeyer/greenWTE.git"

or for CUDA 12.x:

.. code-block:: bash

    python3 -m pip install "greenWTE[cuda12x] @ git+ssh://git@github.com/kremeyer/greenWTE.git"

Testing
-------
To run the tests we clone the repository and set up a virtual environment.
After that we install the dependencies and run the tests::

    python3 -m pytest --pyargs greenWTE

When running the tests for the first time, some test data will be downloaded into the `test` folder.
They contain material properties that will also be used in the examples.
