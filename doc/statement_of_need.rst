Statement of need
=================

There exists open-source code to determine the bulk thermal conductivity by solving the WTE for a static linear temperature gradient `[J.Phys. Condens. Matter 35 (2023)] <phono3py_>`_.
Additionally, there exist closed-source implementations that solve the BTE in spatial and temporal Fourier space with arbitrary source terms using a Green's function approach `[Phys. Rev. B 104 (2021)] <chiloyan2021green>`_.
Thus, we present a solver that fills the gap and supports microscopic thermal transport calculations in the WTE framework beyond the static limit.
Specifically, the solver is able to capture size- and frequency-effects, including non-diffusive effects, by allowing the user to choose arbitrary source terms.
By solving the WTE in spatial and temporal Fourier space, a bridge to common ultrafast and spatially patterned heating experiments is built.
In particular, the frequency-domain Green's function formulation naturally connects to ultrafast pump-probe experiments, where the heating is instantaneous and the subsequent thermal response is measured as a function of delay time `[Struct. Dyn. 11 (2024)] <kremeyer2024ultrafast>`_.
Spatially periodic temperature gratings used in transient thermal grating experiments are directly modelled by our approach as well [`Nat. Comm. 13 (2022) <ding2022observation>`_ and `Struct. Dyn. 11 (2024) <kremeyer2024ultrafast>`_].
The approach presented here is easily generalizable and can be applied to arbitrary experimental and device geometries.

.. _phono3py: https://doi.org/10.1088/1361-648X/acd831
.. _chiloyan2021green: https://doi.org/10.1103/PhysRevB.104.245424
.. _kremeyer2024ultrafast: https://doi.org/10.1063/4.0000224
.. _ding2022observation: https://doi.org/10.1038/s41467-021-27907-z
