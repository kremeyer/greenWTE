---
title: "greenWTE: Frequency-domain solver for the phonon Wigner Transport Equation with arbitrary heating using Green's functions"
tags:
  - Python
  - Thermal Transport
  - Phonons
  - Wigner Transport Equation
  - Boltzmann Transport Equation
  - GPU
authors:
  - name: Laurenz Kremeyer
    orcid: 0000-0002-7091-2887
    affiliation: 1
  - name: Bradley J. Siwick
    affiliation: 1
  - name: Samuel Huberman
    affiliation: 1
affiliations:
  - index: 1
    name: Department of Physics and Centre for the Physics of Materials, McGill University, Canada
date: 22 December 2025
bibliography: paper.bib

---

# Summary

Heat transport has been traditionally described in a phenomenological way via Fourier's law.
In the 1960s the popular phonon Boltzmann Transport equation emerged and thermal transport phenomena started to be described in a microscopic way[@ziman2001electrons].
A more recent breakthrough takes a quantum-mechanical approach to phonon transport and includes phonon tunnelling processes, so-called coherences[@simoncelli2022wigner].
The here presented Python package solves the Wigner Transport Equation (WTE) in spatial and temporal Fourier space for arbitrary source terms.
This allows to compute thermal conductivities from bulk to nanoscale, from static to high frequency regimes.
Beyond that it can be used to study the response of phonon systems in materials to arbitrary heat sources[@kremeyer2025transition].

# Statement of need

A solver for the WTE for static linear temperature gradients exists, that is phono3py[@phono3py]. Additionally, closed-source implementations that solve the BTE in Fourier space with arbitrary source terms using a Green's function approach[@chiloyan2021green]. Thus we present software that fills the gap and supports microscopic thermal transport calculations in the WTE framework that is able to capture size- and frequency-effects by supporting the user to choose arbitrary source terms.
Beyond computing bulk thermal conductivities, solving the WTE in spatial and temporal Fourier space provides a direct bridge to common ultrafast and spatially patterned heating experiments. In particular, the frequency-domain Green's function formulation naturally connects to ultrafast pump-probe experiments, where the heating is instantaneous and the subsequent thermal response is measured as a function of delay time[@kremeyer2024ultrafast]. Spatial thermal gratings used in transient thermal grating experiments are directly modelled by our approach as well[@ding2022observation;@kremeyer2024ultrafast;].
Showcase predictions and physical interpretations that can be made using this code can be taken from the research presented in [@kremeyer2025transition].

# Current functionality

The formulation of the WTE in spatial and temporal Fourier space reads
$$
\begin{split}
-\mathrm{i}\,\omega\,\tilde{\mathbf{N}}\!\left(\mathbf{k},\omega,\mathbf{q}\right)
+\mathrm{i}\Big[\Omega\!\left(\mathbf{q}\right),\,\tilde{\mathbf{N}}\!\left(\mathbf{k},\omega,\mathbf{q}\right)\Big]
+\frac{\mathrm{i}\,\mathbf{k}}{2}\Big\{\mathbf{V}\!\left(\mathbf{q}\right),\,\tilde{\mathbf{N}}\!\left(\mathbf{k},\omega,\mathbf{q}\right)\Big\}
= \\
\mathcal{F}\left\{\left.\frac{\partial}{\partial t}\mathbf{N}\!\left(\mathbf{R},t,\mathbf{q}\right)\right|_{\mathrm{H}^{\mathrm{col}}}\right\}
+\tilde{\mathbf{Q}}\!\left(\mathbf{k},\omega,\mathbf{q}\right).
\end{split}
$$
Here $\omega$ and $\mathbf{k}$ are the temporal and spatial Fourier variables, $\mathbf{q}$ is the phonon wavevector, $\tilde{\mathbf{N}}$ is the Wigner distribution, $\Omega = \delta_{s,s'}\omega_{s}$ is the phonon frequency matrix, $\mathbf{V}$ is the velocity operator, $\mathcal{F}\left\{\left.\frac{\partial}{\partial t}\mathbf{N}\!\left(\mathbf{R},t,\mathbf{q}\right)\right|_{\mathrm{H}^{\mathrm{col}}}\right\}$ is the collision term and $\tilde{\mathbf{Q}}\!\left(\mathbf{k},\omega,\mathbf{q}\right)$ is the source term in Fourier space. For a pair of $\left(\mathbf{k},\omega\right)$ we can rewrite this compactly as
$$
\mathcal{L}\!\left(\mathbf{k},\omega\right)\,\tilde{\mathbf{N}}\!\left(\mathbf{k},\omega\right) = \tilde{\mathbf{Q}}\!\left(\mathbf{k},\omega\right).
$$
The inverse of the operator $\mathcal{L}\!\left(\mathbf{k},\omega\right)$ is the Green's function $\mathcal{G}\!\left(\mathbf{k},\omega\right)$, which allows to solve for the Wigner distribution. We currently implement two approaches to solve the presented system.
A direct solver fully inverts the operator $\mathcal{L}$ at each $\left(\mathbf{k},\omega\right)$ point. The Green's function is computed once and can be reused for arbitrary source terms. The first computation is expensive, but subsequent reuse for different source terms is cheap. The second approach solves the linear system $\mathcal{L}\,\tilde{\mathbf{N}} = \tilde{\mathbf{Q}}$ directly for a given source term using an iterative solver. This avoids the expensive computation of the full Green's function, but requires to solve the system from scratch for each source term. The output of a greenWTE calculation is the Wigner distribution $\tilde{\mathbf{N}}\!\left(\mathbf{k},\omega,\mathbf{q}\right)$ and derived integral quantities such as the heat flux and the thermal conductivity.

greenWTE supports a NumPy (CPU) and a CuPy (GPU) backend. If available, the linear-algebra-heavy parts of the code run on the GPU. While it is in principle possible to run the code on a CPU, a GPU is highly recommended to achieve reasonable performance in real-world scenarios.

# Documentation
A detailed documentation including installation instructions, tutorials, command-line interface options and API reference is available via [ReadTheDocs](https://greenwte.readthedocs.io).

# Acknowledgements
L. K. acknowledges support from a Fonds de Recherche du Québec-Nature et Technologies (FRQNT) Merit fellowship. B. J. S. acknowledges support from the NRC Quantum Sensors Challenge Program and the Canada Research Chairs program. S. H. acknowledges support from the NSERC Discovery Grants Program under Grant No. RGPIN-2021-02957 and FRQNT Nouveau Chercheur No. 341503.

# References