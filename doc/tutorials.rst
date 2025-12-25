Tutorials
=========

This section contains tutorials on how to use the greenWTE package for different applications.

Input Data
----------
As a starting point for any calculating with greenWTE we need to provide the following material properties as input data:

* background temperature
* velocity operator
* phonon frequencies
* heat capacities
* unit cell volume

Note that the greenWTE cannot treat symmetries at this point, so the full Brillouin zone data is required!

The :class:`~greenWTE.base.Material` class is used to store these properties.
We can create an instance of this class by providing the properties as numpy arrays or using it's :meth:`~greenWTE.base.Material.from_phono3py` method to read the data from a phono3py HDF output file.
When using the first option all units are assumed to be in SI::

    from greenWTE.base import Material

    # pass the properties as arrays

    material = Material(
        temperature=300,
        velocities=velocities,
        frequencies=frequencies,
        heat_capacities=heat_capacities,
        volume=volume,
    )

If data from phono3py is used, the units are automatically converted to SI.
Make sure to save the velocity operator when running phono3py and run it with the ``--nosym`` flag to get the full Brillouin zone data.
The following code snippet shows how to read the data from a phono3py HDF file::

    from greenWTE.base import Material

    material = Material.from_phono3py("kappa-m191919.hdf5", temperature=300)

For sake of following these tutorials you can use the Silicon and CsPbBr\ :sub:`3` data provided for testing.
Note that the 5x5x5 and 4x4x3 q-point meshes used in the tests are not sufficient to obtain converged results!
The data can be downloaded by running the tests for the first time and will appear in the `test` folder. You can
create symlinks to the files to be able to easily access them from your working directory with:

.. code-block:: bash

    ln -sf "$(python -c 'import importlib.resources as r; print(r.files("greenWTE.tests").joinpath("Si-kappa-m555.hdf5"))')" .
    ln -sf "$(python -c 'import importlib.resources as r; print(r.files("greenWTE.tests").joinpath("CsPbBr3-kappa-m443.hdf5"))')" .

Static thermal conductivity in bulk
-----------------------------------
The most basic application of greenWTE is to compute the static thermal conductivity of a bulk material.
From here on out we will use the Silicon data provided for testing.
First we will use the CLI module :mod:`~greenWTE.solve_iter` and then the scripted interface.
A full list of the CLI arguments can be found in the :doc:`cli-arguments` section of the documentation.

Command-line interface
^^^^^^^^^^^^^^^^^^^^^^

To compute the static thermal conductivity using the CLI we can run the following command in the terminal:

.. code-block:: bash

    python3 -m greenWTE.solve_iter Si-kappa-m555.hdf5 static_kappa.h5 -k 2.5 -w -300 -t 300 -os none -s gradT

This will compute the thermal conductivity for a spatial frequency of 2.5 rad/m at a temperature of 300 K at a frequency of 1e-300 Hz (i.e. static)
using a temperature gradient as the source term, which does not require and outer solver.
The results will be saved in the file `static_kappa.h5` and the consolue output should look like::

    frequency                 κ_xx               κ_xx_P               κ_xx_C     |κ_xx|   |κ_xx_P|   |κ_xx_C|
    -----------------------------------------------------------------------------------------------------------
    1.00e-300   6.26e+01+4.98e-12j   6.25e+01+4.99e-12j   1.82e-01-1.73e-14j   6.26e+01   6.25e+01   1.82e-01
    -----------------------------------------------------------------------------------------------------------

.. only:: builder_doctest

    .. testsetup:: cli

        import os, sys, subprocess, tempfile, pathlib
        from greenWTE.tests.defaults import SI_INPUT_PATH
        SIPATH = str(SI_INPUT_PATH)       # provided by doctest_global_setup
        TMPDIR = pathlib.Path(tempfile.mkdtemp())
        OUT = TMPDIR / "static_kappa.h5"

    .. testcode:: cli

        # Run the CLI as a module so we use the same interpreter/env as Sphinx
        cmd = [
            sys.executable, "-m", "greenWTE.solve_iter",
            SIPATH, str(OUT),
            "-k", "2.5", "-w", "-300", "-t", "300", "-os", "none", "-s", "gradT",
        ]
        res = subprocess.run(cmd, text=True, capture_output=True, check=True)
        print("PROGRESS", "[1/1] k=3.16e+02 w=1.00e-300 dT= 1.00e+00+1.00e+00j n_it=1" in res.stdout)
        print("KAPPA_VALUES_GOOD", "6.26e+01" in res.stdout and "6.25e+01" in res.stdout and "1.82e-01" in res.stdout)
        print("OUT_EXISTS", os.path.exists(OUT))

    .. testoutput:: cli

        PROGRESS True
        KAPPA_VALUES_GOOD True
        OUT_EXISTS True

We can see that the total thermal conductivity is only 63 W/m/K, which is far from the experimental value, because of the very coarse q-point mesh.
However, the fact that the population contribution is much larger than the coherence contribution is already correctly captured. Let's see what happens
when we do the same thing for CsPbBr\ :sub:`3`! Running the following command:

.. code-block:: bash

    python3 -m greenWTE.solve_iter CsPbBr3-kappa-m443.hdf5 static_kappa.h5 -k 2.5 -w -300 -t 300 -os none -s gradT

we now get a different picture::

    frequency                 κ_xx               κ_xx_P               κ_xx_C     |κ_xx|   |κ_xx_P|   |κ_xx_C|
    -----------------------------------------------------------------------------------------------------------
    1.00e-300   3.13e-01+2.23e-12j   1.03e-01+1.91e-12j   2.10e-01+3.19e-13j   3.13e-01   1.03e-01   2.10e-01
    -----------------------------------------------------------------------------------------------------------

The imaginary parts are again very small, but now the coherence contribution is about twice as large as the population contribution! This shows that in
materials with strong anharmonicity and complex crystal structures the coherence effects can dominate the thermal transport even at room temperature.

.. only:: builder_doctest

    .. testsetup:: cli

        import os, sys, subprocess, tempfile, pathlib
        from greenWTE.tests.defaults import CSPBBR3_INPUT_PATH
        CPBPATH = str(CSPBBR3_INPUT_PATH)       # provided by doctest_global_setup
        TMPDIR = pathlib.Path(tempfile.mkdtemp())
        OUT = TMPDIR / "static_kappa.h5"

    .. testcode:: cli

        # Run the CLI as a module so we use the same interpreter/env as Sphinx
        cmd = [
            sys.executable, "-m", "greenWTE.solve_iter",
            CPBPATH, str(OUT),
            "-k", "2.5", "-w", "-300", "-t", "300", "-os", "none", "-s", "gradT",
        ]
        res = subprocess.run(cmd, text=True, capture_output=True, check=True)
        print("PROGRESS", "[1/1] k=3.16e+02 w=1.00e-300 dT= 1.00e+00+1.00e+00j n_it=1" in res.stdout)
        print("KAPPA_VALUES_GOOD", "3.13e-01" in res.stdout and "1.03e-01" in res.stdout and "2.10e-01" in res.stdout)
        print("OUT_EXISTS", os.path.exists(OUT))

    .. testoutput:: cli

        PROGRESS True
        KAPPA_VALUES_GOOD True
        OUT_EXISTS True

There is a lot more data saved in the output file which is described in :func:`~greenWTE.io.save_solver_result`.

Scripted interface
^^^^^^^^^^^^^^^^^^
To compute the same result using the scripted interface we can use the following code snippet:

.. testcode::

    from greenWTE.base import Material
    from greenWTE.iterative import IterativeWTESolver
    from greenWTE.sources import source_term_gradT
    from greenWTE.tests.defaults import SI_INPUT_PATH

    K_FT = 10 ** 2.5  # spatial frequency in rad/m
    material = Material.from_phono3py(SI_INPUT_PATH, temperature=300)
    source = source_term_gradT(
        K_FT,
        material.velocity_operator,
        material.phonon_freq,
        material.linewidth,
        material.heat_capacity,
        material.volume,
    )
    solver = IterativeWTESolver(
        [0],  # static case
        K_FT,
        material,
        source,
        source_type="gradient",
        outer_solver="none",
    )
    solver.run()

    print(f"{solver.kappa[0]:.1f}, {solver.kappa_p[0]:.1f}, {solver.kappa_c[0]:.1f}")

And as expected we get the same result of 63 W/m/K:

.. testoutput::

    62.6+0.0j, 62.5+0.0j, 0.2+0.0j

Try it with the example data for CsPbBr\ :sub:`3` provided and see how the coherence contribution dominates the thermal conductivity in this material at 300K!

Dynamical thermal conductivity in bulk
--------------------------------------
When using iterative solver we will stick to the scripted interface from now on, but the CLI can be used in the same way.
The code snippet from above can be easily adapted to compute the dynamical thermal conductivity by providing an array of temporal frequencies instead of just the static case.
To verify the results we can plot the real and imaginary part of the thermal conductivity said frequency range. Assuming you have matplotlib installed, you can run the following code:

.. plot::
    :caption: Real and imaginary part of the dynamical thermal conductivity of Silicon at 300K

    from greenWTE import to_cpu, xp
    from greenWTE.base import Material
    from greenWTE.iterative import IterativeWTESolver
    from greenWTE.sources import source_term_gradT
    from greenWTE.tests.defaults import SI_INPUT_PATH
    import matplotlib.pyplot as plt

    K_FT = 10 ** 2.5  # spatial frequency in rad/m
    omegas = xp.logspace(7, 14, 20)  # temporal frequencies in rad/s
    material = Material.from_phono3py(SI_INPUT_PATH, temperature=300)
    source = source_term_gradT(
        K_FT,
        material.velocity_operator,
        material.phonon_freq,
        material.linewidth,
        material.heat_capacity,
        material.volume,
    )
    solver = IterativeWTESolver(
        omegas,
        K_FT,
        material,
        source,
        source_type="gradient",
        outer_solver="none",
        print_progress=True,
    )
    solver.run()

    f, ax = plt.subplots()

    ax.set_xlim(to_cpu(omegas[0]), to_cpu(omegas[-1]))
    ax.plot(to_cpu(omegas), to_cpu(xp.real(solver.kappa_p)), "o-", label="$\Re(\kappa_\mathrm{P})$", mec="k")
    ax.plot(to_cpu(omegas), to_cpu(xp.imag(solver.kappa_p)), "o-", label="$\Im(\kappa_\mathrm{P})$", mec="k")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("temporal frequency [rad/s]")
    ax.set_ylabel("thermal conductivity [W/mK]")
    ax.legend()

    f.tight_layout()
    # plt.show()

We can see that the real part of the thermal conductivity decreases with increasing frequency, while the imaginary part first increases and then decreases again.
The phase lag between the heat flux and the applied temperature gradient is captured by the imaginary part of the thermal conductivity. Try this for CsPbBr\ :sub:`3`
as well and see how the population contributions are suppressed even more strongly at higher frequencies compared to the coherence contributions!

This short tutorial should give you a good starting point to use the greenWTE package for your own calculations. Note that the grid for the dataset is not converged,
but will give the right trends for many applications. An idea of more sophisticated  applications of greenWTE is given in the arXiv preprint `"Transition from
Population to Coherence-dominated Non-diffusive Thermal Transport" [arXiv:2512.13616 (2025)] <https://arxiv.org/abs/2512.13616>`_. The calculations fully rely on scripts similar to the
ones shown in these tutorials.
