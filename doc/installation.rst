Installation
=============

Requirements
------------
greenWTE requires Python 3.10 or higher and the `NVIDIA CUDA toolkit`_ 12 or 13 are recommended to achieve
reasonable performance when using a GPU. Note that currently numpy 2.0 or higher is not compatible with CUDA.
Python package requirements can be found `in the pyproject.toml file`_.
This file also contains information about what python and CUDA versions this package has been tested with.

.. _NVIDIA CUDA toolkit: https://developer.nvidia.com/cuda-toolkit
.. _in the pyproject.toml file: https://github.com/kremeyer/greenWTE/blob/main/pyproject.toml

Installation
------------
A PyPI package is available for greenWTE. You can install it using pip:

.. code-block:: bash

    python3 -m pip install greenWTE[cuda13x]

or for CUDA 12.x:

.. code-block:: bash

    python3 -m pip install greenWTE[cuda12x]

You can also install greenWTE directly from the GitHub repository using pip:

.. code-block:: bash

    python3 -m pip install "greenWTE[cuda{12,13}x] @ git+ssh://git@github.com/kremeyer/greenWTE.git"

Replacing ``{12,13}`` with your desired CUDA version.

Testing
-------
To run the tests we clone the repository and set up a virtual environment.
After that we install the dependencies and run the tests using pytest:

.. code-block:: bash

    python3 -m pip install pytest
    python3 -m pytest --pyargs greenWTE

When running the tests for the first time, some test data will be downloaded into the `test` folder. Some
tests are skipped, if a compatible GPU is not available. They contain material properties that will also
be used in the examples.
