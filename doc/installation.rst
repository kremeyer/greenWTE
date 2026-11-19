Installation
=============

Requirements
------------
greenWTE requires Python 3.10 or higher and NumPy 2. A GPU is optional, but recommended to achieve reasonable
performance. GPU support is provided by `CuPy`_ and requires an NVIDIA GPU with a recent driver supporting
CUDA 12 or 13. The CUDA toolkit itself does not have to be installed, see below.
Python package requirements can be found `in the pyproject.toml file`_.
This file also contains information about what python and CUDA versions this package has been tested with.

.. _CuPy: https://cupy.dev
.. _in the pyproject.toml file: https://github.com/kremeyer/greenWTE/blob/main/pyproject.toml

Installation
------------
A PyPI package is available for greenWTE. Pick **one** of the CUDA extras below; installing more than one
of them into the same environment leads to conflicting CuPy installations.

.. code-block:: bash

    python3 -m pip install greenWTE[cuda13x]

or for CUDA 12.x:

.. code-block:: bash

    python3 -m pip install greenWTE[cuda12x]

These extras also install the required CUDA libraries from PyPI (about 1.7 GB), so only the NVIDIA driver needs
to be present on your system. Run ``nvidia-smi`` to check which CUDA version your driver supports.

Which CUDA version should I use?
    CUDA 13 does not support GPUs older than the Turing generation (compute capability 7.5). If you have a
    Maxwell, Pascal or Volta GPU (e.g. GTX 900/1000 series, Tesla P100, V100), use ``cuda12x``. For newer GPUs
    either works, ``cuda13x`` is recommended if your driver supports it.

Using an existing CUDA toolkit
    If a CUDA toolkit is already provided on your system, e.g. via environment modules on an HPC cluster, you
    can skip downloading the CUDA libraries by using the ``cuda13x-system`` or ``cuda12x-system`` extras instead:

    .. code-block:: bash

        module load cuda/13  # or however your system provides CUDA
        python3 -m pip install greenWTE[cuda13x-system]

    CuPy then has to be able to find the toolkit, e.g. via the ``CUDA_PATH`` environment variable.

CPU only
    Installing greenWTE without any extra (``python3 -m pip install greenWTE``) runs all calculations on the CPU.
    If CuPy is installed but cannot be loaded, greenWTE warns and falls back to the CPU. The CPU backend can
    also be forced by setting the environment variable ``GREENWTE_BACKEND=CPU``.

You can also install greenWTE directly from the GitHub repository using pip:

.. code-block:: bash

    python3 -m pip install "greenWTE[cuda13x] @ git+ssh://git@github.com/kremeyer/greenWTE.git"

Replacing ``cuda13x`` with the extra you want to use.

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
