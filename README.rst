.. BEGIN-INTRO

greenWTE - Frquency-domain solver for the phonon Wiger Transport Equation with arbitrary heating using Green's functions
========================================================================================================================

greenWTE is a Python package to solve the Wigner Transport Equation (WTE) in spatial and temporal Fourier space
for arbitrary source terms. This allows to compute thermal conductivities from bulk to nanoscale, from static to
high frequency regimes. Beyond that it can be used to study the response of materials to arbitrary heat sources.
A showcase of the capabilities of greenWTE can be found in the arXiv preprint `"Transition from Population to
Coherence-dominated Non-diffusive Thermal Transport" [arXiv:2512.13616 (2025)] <WTE_showcase_>`_.

Derived from the Wigner formulation of quantum mechanics, the WTE describes heat transport in terms of particlelike
and wavelike conduction mechanisms. The full and very detailed derivation can be found in the work by Simoncelli,
Marzari and Mauri in their paper `"Wigner Formulation of Thermal Transport in Solids" [Phys. Rev. X 12 (2022)]
<WTE_paper_>`_.

.. _WTE_showcase: https://arxiv.org/abs/2512.13616
.. _WTE_paper: https://journals.aps.org/prx/abstract/10.1103/PhysRevX.12.041011

.. END-INTRO

----

.. image:: https://github.com/kremeyer/greenWTE/actions/workflows/ci.yml/badge.svg?branch=main
    :target: https://github.com/kremeyer/greenWTE/actions/workflows/ci.yml?query=branch%3Amain
    :alt: CI status

.. image:: https://codecov.io/github/kremeyer/greenWTE/branch/main/graph/badge.svg?token=3W1D1HOSW2
    :target: https://codecov.io/github/kremeyer/greenWTE
    :alt: Codecov status

.. image:: https://img.shields.io/pypi/v/greenWTE
   :alt: PyPI - Version


- The full documentation including installation instructions, tutorials and API reference is hosted on `Read the Docs`_.
- Releases are available on `PyPI`_ and can be installed via ``pip install greenWTE[cuda12x,cuda13x]``.
- The source code is available on `GitHub`_.

.. _Read the Docs: https://greenwte.readthedocs.io/
.. _PyPI: https://pypi.org/project/greenWTE/
.. _GitHub: https://github.com/kremeyer/greenWTE
