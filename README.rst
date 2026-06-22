=========
Advection
=========


.. image:: https://img.shields.io/pypi/v/advection.svg
        :target: https://pypi.python.org/pypi/advection

.. image:: https://readthedocs.org/projects/advection/badge/?version=latest
        :target: https://advection.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

A lightweight Python toolkit for diagnosing **horizontal and vertical
advection** and correcting the **energy-balance closure gap** in
eddy-covariance data.

* Free software: MIT license
* Documentation: https://advection.readthedocs.io.


Purpose
-------

Eddy-covariance towers routinely under-sum the available energy—the measured
sensible (H) plus latent (LE) heat flux is typically 10–20% lower than net
radiation minus ground heat flux (Rn - G). This package provides functions
to calculate advective fluxes and apply corrections to improve energy balance
closure, based on methods described by Wang et al. (2024).

.. important::

    **This library assumes any open-path LE/CO2 flux you provide has ALREADY
    been WPL (Webb-Pearman-Leuning 1980) density-corrected.** The WPL correction
    accounts for the dry-air density fluctuations that contaminate an open-path
    vapour/CO2 covariance. It is a **mandatory, separate pre-processing step —
    not an advection correction** — and this library does **not** apply it for
    you. Run it (or confirm it has been run) in your eddy-covariance processing
    chain *before* passing ``LE`` to any function here.

    A thin convenience helper, ``advection.wpl_latent_heat_flux``, implements the
    simplified form ``E = (1 + mu*MR) * [w'rho_v' + (rho_v/T)*w'T']`` with
    ``mu = 1.6077``. It is intended for teaching and quick checks only; for
    production work prefer an established processing package such as EddyPro,
    EasyFlux, or your logger's online WPL routine, which also handle the
    pressure term, coordinate rotation, and spectral corrections this helper
    omits.

Physics & assumptions
---------------------

This library encodes a small, explicit physics contract (see ``CLAUDE.md``).
Everything below is what the functions assume and obey.

**Surface energy balance.** Without storage, ``Rn - G = H + LE``; with air heat
storage ``J`` (Wang 2024 Eq. 11), ``Rn - G - J = H + LE``. The closure residual
is ``Residual = Rn - G - H - LE`` and the evaporative fraction is
``EF = LE / (Rn - G)``. ``EF > 1`` (equivalently ``LE > Rn - G``) is the
**advective-input fingerprint** of the oasis regime.

**Sign convention (Moderow et al. 2021 — OUT-positive).** A **positive** flux is
energy **out** of the control volume; a **negative** flux is energy **into** it.
In the oasis case (warm, dry air advected onto a cool, transpiring surface) this
gives a downward (**negative**) ``H`` and **negative** horizontal/vertical heat
advection — heat carried *into* the field.

**Advection is computed from gradients, never from the residual.** Horizontal
heat/moisture advection use the measured along-wind gradients (Wang 2024
Eqs. 5a/5b; Moderow Term IV)::

    HA_T = rho * Cp     * u_bar * (dT/dx) * (zm - h)     # W/m^2  (Eq. 5a)
    HA_Q = rho * lambda * u_bar * (dq/dx) * (zm - h)     # W/m^2  (Eq. 5b)

and vertical heat advection uses the **planar-fit** mean vertical velocity
``w_bar`` (Lee 1998; Wang 2024 Eq. 6)::

    VAT = rho * Cp * w_bar * (T_zm - <T>)               # W/m^2  (Eq. 6)

The energy-balance residual ``(H + LE) - (Rn - G)`` is returned **only as a
closure diagnostic** — it is never relabelled as an advective flux. If the
inputs needed for a real advection term (an upwind tower, ``w_bar``, the
column-mean ``<T>``) are missing, the functions **raise** rather than
back-filling a meaningless zero or the residual.

**Conditional-inclusion rule (Wang 2024).** Advective fluxes are folded into the
budget **only** where **both** (1) ``Rn > 75 W/m^2`` **and** (2) the
spectrally-corrected ``(H + LE) < (Rn - G)`` hold. Applying this gate raised
closure from 0.89 to 0.97 in the Wang et al. (2024) alfalfa study. Steps that
fail the gate are left exactly uncorrected.

**Bowen-ratio closure is wrong in the oasis case.** When ``LE > (Rn - G)``,
Bowen-ratio closure would shrink ``LE`` toward the available energy, which is
physically wrong — the surplus is genuine advective input. The closure helpers
in ``advection.closure`` warn in this regime; prefer adding the *measured*
advective fluxes instead.

**WPL pre-step.** Every open-path ``LE``/CO2 flux is assumed to be **already**
WPL (Webb-Pearman-Leuning 1980) density-corrected — a mandatory, *separate*
pre-processing step (see the note above). This library does **not** apply it.

Worked oasis example
~~~~~~~~~~~~~~~~~~~~~~

Warm, dry air (30 °C, 5 g/kg) advects 100 m onto a cool, wet field
(25 °C, 10 g/kg). The **signs** are the oasis fingerprint — heat advected *into*
the field horizontally and vertically, and a *drying* moisture advection:

.. code-block:: python

    import numpy as np
    from advection import compute_advection_fluxes, apply_advection_correction

    main = {
        "H": np.array([-30.0]),   # downward H -> oasis fingerprint
        "LE": np.array([400.0]),
        "Rn": np.array([300.0]), "G": np.array([20.0]),
        "T": 25.0, "q": 0.010, "u": 2.0, "zm": 2.0, "h": 0.5,
        "w_bar": -0.03, "T_col": 23.0,   # planar-fit subsidence; warm air aloft
    }
    upwind = {"T": 30.0, "q": 0.005}      # warm, dry upwind air

    flux = compute_advection_fluxes(main, upwind_data=upwind, tower_distance=100.0)
    print(flux["HA_T"], flux["HA_Q"], flux["VAT"])
    # -> HA_T ~ -179 (heat INTO field), HA_Q ~ +431 (drying), VAT ~ -72 (warm air down)

Note that *this* step has ``EF = 400/280 > 1``: the turbulent sum already
*exceeds* the available energy, so there is no under-closure gap and the
conditional-inclusion gate correctly **declines** to add advection here. The
gate is built for the common *under-closure* case ``(H + LE) < (Rn - G)``. The
following daytime step has such a gap (available 380, measured 150), and a net
advective input of +120 W/m^2 moves the budget toward closure:

.. code-block:: python

    under = {"H": np.array([30.0]), "LE": np.array([120.0]),   # sum 150
             "Rn": np.array([420.0]), "G": np.array([40.0])}   # available 380
    corr = apply_advection_correction(
        under, np.array([70.0]), np.array([40.0]), HA_Q=np.array([10.0]),
    )
    print(corr["included"])             # [ True ]  (Rn>75 AND H+LE<Rn-G)
    print(corr["H_plus_LE_corrected"])  # [270.]    (150 + 120 folded in)
    print(corr["residual_corrected"])   # [110.]    (was 230 -> closer to 0)

Setup
-----

To install the latest version from PyPI:

.. code-block:: bash

    pip install advection

For development, clone the repository and install in editable mode:

.. code-block:: bash

    git clone https://github.com/inkenbrandt/advection.git
    cd advection
    pip install -e ".[dev]"

Usage
-----

Basic example of detecting advection and computing corrections:

.. code-block:: python

    import numpy as np
    from advection import advect_detect, advection

    # Main (downwind) tower. Horizontal advection is a gradient term, so the
    # main tower must also carry T, q (or RH), wind speed u, and the heights
    # zm (measurement) and h (canopy); an upwind reference and the tower
    # separation are required as well.
    main_data = {
        'H': [50, 60, -10],
        'LE': [200, 220, 50],
        'Rn': [300, 310, 100],
        'G': [20, 25, 10],
        'T': [25, 26, 24],      # air temperature [°C or K]
        'q': [0.010, 0.010, 0.011],  # specific humidity [kg/kg] (or 'RH' in %)
        'u': [2.0, 2.5, 1.8],   # mean horizontal wind speed [m/s]
        'zm': 2.0,              # measurement height [m]
        'h': 0.3,               # canopy height [m]
    }
    # Upwind (warmer, drier) reference tower.
    upwind_data = {'T': [29, 30, 28], 'q': [0.005, 0.005, 0.006]}

    # Detect horizontal advection
    flags_h = advect_detect.detect_horizontal_advection(
        main_flux=main_data['H'],
        le_main=main_data['LE'],
        rn=main_data['Rn'],
        g=main_data['G']
    )

    # Compute advection fluxes (HA_T < 0 means heat advected INTO the field)
    out = advection.compute_advection_fluxes(
        main_data=main_data,
        upwind_data=upwind_data,
        detect_horizontal=flags_h,
        tower_distance=100.0,   # m between the main and upwind towers
    )

    print(out['HA_T'], out['HA_Q'])

Features
--------

* Detect horizontal and vertical advection using multi-tower or single-tower criteria.
* Compute advective heat and moisture fluxes.
* Apply corrections to energy balance components.
* Utility functions for atmospheric physics (air density, specific humidity, etc.).
* Convenience WPL density-correction pre-step (``wpl_latent_heat_flux``); see the
  note above — prefer EddyPro / established processing for production use.

Credits
-------

Based on work by Wang and others, 2024 (10.1016/j.agrformet.2024.110196)

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
