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
    affiliation: 1, 2
  - name: Bradley J. Siwick
    affiliation: 1, 2, 3
  - name: Samuel Huberman
    orcid: 0000-0003-0865-8096
    affiliation: 1, 4
affiliations:
  - index: 1
    name: Department of Physics, McGill University, Canada
  - index: 2
    name: Centre for the Physics of Materials, McGill University, Canada
  - index: 3
    name: Department of Chemistry, McGill University, Canada
  - index: 4
    name: Department of Chemical Engineering, McGill University, Canada
date: 22 December 2025
bibliography: paper.bib

---

# Summary

Heat transport in solids is usually described in a phenomenological way via Fourier's law.
However, at small time and length scales, a microscopic description is needed.
This is traditionally done by invoking the phonon Boltzmann Transport equation (BTE) [@ziman2001electrons], which has been successful in describing transport in high thermal conductivity materials.
Recently, a generalization of the BTE, the Wigner Transport Equation, was developed to be able to microscopically describe transport in low thermal conductivity materials by taking into account not only phonon scattering processes but also phonon tunnelling processes, so-called coherences  [@simoncelli2022wigner].
We present a Python package that solves the Wigner Transport Equation (WTE) in spatial and temporal Fourier space for arbitrary source terms, thereby enabling the study to heat transport at small space and time scales.
This allows one to compute both static and dynamic thermal conductivities from bulk to nanoscale.
Beyond that, it can be used to study the response of phonon systems in materials to arbitrary heat sources [@kremeyer2025transition].

# Statement of need

There exists open-source code to determine the bulk thermal conductivity by solving the WTE for a static linear temperature gradient [@phono3py].
Additionally, there exist closed-source implementations that solve the BTE in spatial and temporal Fourier space with arbitrary source terms using a Green's function approach [@chiloyan2021green].
Thus, we present a solver that fills the gap and supports microscopic thermal transport calculations in the WTE framework beyond the static limit.
Specifically, the solver is able to capture size- and frequency-effects, including non-diffusive effects, by allowing the user to choose arbitrary source terms.
By solving the WTE in spatial and temporal Fourier space, a bridge to common ultrafast and spatially patterned heating experiments is built.
In particular, the frequency-domain Green's function formulation naturally connects to ultrafast pump-probe experiments, where the heating is instantaneous and the subsequent thermal response is measured as a function of delay time [@kremeyer2024ultrafast].
Spatially periodic temperature gratings used in transient thermal grating experiments are directly modelled by our approach as well [@ding2022observation;@kremeyer2024ultrafast].
The approach presented here is easily generalizable and can be applied to arbitrary experimental and device geometries.

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
Here $\omega$ and $\mathbf{k}$ are the temporal and spatial Fourier variables, $\mathbf{q}$ is the phonon wavevector, $\tilde{\mathbf{N}}$ is the Wigner distribution, $\Omega = \delta_{s,s'}\omega_{s}$ is the phonon frequency matrix, $\mathbf{V}$ is the velocity operator, $\mathcal{F}\left\{\left.\frac{\partial}{\partial t}\mathbf{N}\!\left(\mathbf{R},t,\mathbf{q}\right)\right|_{\mathrm{H}^{\mathrm{col}}}\right\}$ is the collision term and $\tilde{\mathbf{Q}}\!\left(\mathbf{k},\omega,\mathbf{q}\right)$ is the source term in Fourier space.
For a given pair of $\left(\mathbf{k},\omega\right)$ we can rewrite this compactly as
$$
\mathcal{L}\!\left(\mathbf{k},\omega\right)\,\tilde{\mathbf{N}}\!\left(\mathbf{k},\omega\right) = \tilde{\mathbf{Q}}\!\left(\mathbf{k},\omega\right).
$$
The inverse of the operator $\mathcal{L}\!\left(\mathbf{k},\omega\right)$ is the Green's function $\mathcal{G}\!\left(\mathbf{k},\omega\right)$, which allows to solve for the Wigner distribution.
We currently implement two approaches to solve the presented system.
The first approach is a direct solver that fully inverts the operator $\mathcal{L}$ at each $\left(\mathbf{k},\omega\right)$ point.
The determination of Green's function is computationally expensive, but can subsequently be reused for the construction of the temperature response to user-chosen source terms in a computationally efficient manner.
The second approach solves the linear system $\mathcal{L}\,\tilde{\mathbf{N}} = \tilde{\mathbf{Q}}$ directly for a user-determined source term using an iterative solver.
This avoids the expensive computation of the full Green's function, but requires one to solve the system from scratch for each source term.
The output of a greenWTE calculation is the Wigner distribution $\tilde{\mathbf{N}}\!\left(\mathbf{k},\omega,\mathbf{q}\right)$ and derived integral quantities such as the heat flux and the thermal conductivity.

greenWTE supports a NumPy [@numpy] (CPU) and a CuPy [@cupy] (GPU) backend.
If available, the linear-algebra-heavy parts of the code will run on supported GPUs.
While it is in principle possible to run the code on a CPU, a GPU is highly recommended to achieve reasonable performance in real-world scenarios.

# Documentation
Detailed documentation including installation instructions, tutorials, command-line interface options and API reference is available via [ReadTheDocs](https://greenwte.readthedocs.io).

# Acknowledgements
L. K. acknowledges support from a Fonds de Recherche du Québec-Nature et Technologies (FRQNT) Merit fellowship. B. J. S. acknowledges support from the NRC Quantum Sensors Challenge Program and the Canada Research Chairs program. S. H. acknowledges support from the NSERC Discovery Grants Program under Grant No. RGPIN-2021-02957 and FRQNT Nouveau Chercheur No. 341503.

# References