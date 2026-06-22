Quick-start
===========

This tutorial shows how to

1. Read a 30-minute *TOA5* file.
2. Compute the energy-balance closure ratio.
3. Flag candidate periods of horizontal / vertical advection.
4. Compute advection fluxes needed to close the balance.

Prerequisites
-------------

.. code-block:: bash

   pip install pandas matplotlib numpy

Step 1 – Load the data
----------------------

.. code-block:: python
    
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = (
        pd.read_csv("EC_2024.CSV", skiprows=4,  # TOA5 header
                    parse_dates=["TIMESTAMP"], index_col="TIMESTAMP")
        .rename(columns={"Rn": "Rn", "G": "G", "H": "H", "LE": "LE"})
    )

Step 2 – Calculate closure metrics
--------------------------------------

.. code-block:: python

    # Compute energy balance closure ratio
    df["AE"]            = df["Rn"] - df["G"]
    df["Flux_sum"]      = df["H"]  + df["LE"]
    df["closure_ratio"] = df["Flux_sum"] / df["AE"]

Step 3 – Flag advection
---------------------------


.. code-block:: python

    from advection import advect_detect
    flags_h = advect_detect.detect_horizontal_advection(
        main_flux   = df["H"],
        le_main     = df["LE"],
        rn          = df["Rn"],
        g           = df["G"],
    )
    flags_v = advect_detect.detect_vertical_advection(
        vertical_w = df["w_bar"],   # planar-fit mean w (NOT raw sonic w)
        main_H     = df["H"],
        rn         = df["Rn"],
        g          = df["G"],
    )
    df["adv_h"] = flags_h
    df["adv_v"] = flags_v

Step 4 – Compute & apply flux corrections
-----------------------------------------------

.. code-block:: python
    
    from advection import advection

    # Horizontal advection is the gradient term rho*Cp*u*(dT/dx)*(zm-h), so the
    # main tower must also carry T, q (or RH), wind speed u and the heights
    # zm/h, plus an upwind reference tower and the tower separation.
    main = df[["H", "LE", "Rn", "G", "T", "q", "u"]].to_dict("list")
    main["zm"] = 2.0   # measurement height [m]
    main["h"] = 0.3    # canopy height [m]
    # Vertical advection (VAT) is the MEASURED term rho*Cp*w_bar*(T_zm - <T>)
    # (Lee 1998; Wang Eq. 6) -- NOT a closure residual. It is engaged when you
    # supply the PLANAR-FIT mean vertical velocity w_bar (never the raw sonic w)
    # and the column-mean temperature <T> ("T_col", or a "T_profile"/"z_profile"
    # pair). If detect_vertical is passed but these are missing, the call RAISES.
    main["w_bar"] = df["w_bar"].tolist()  # planar-fit / tilt-corrected w [m/s]
    main["T_col"] = df["T_col"].tolist()  # column-mean temperature [°C or K]
    upwind = df_upwind[["T", "q"]].to_dict("list")  # warmer/drier upwind tower

    out = advection.compute_advection_fluxes(
        main_data         = main,
        upwind_data       = upwind,
        detect_horizontal = flags_h,
        detect_vertical   = flags_v,
        tower_distance    = 100.0,   # m between the main and upwind towers
    )
    # out["HA_T"] (heat) and out["HA_Q"] (moisture) in W/m^2; HA_T < 0 means
    # energy advected INTO the field (oasis, warm upwind air). out["VAT"] is the
    # measured vertical heat advection (None if no vertical inputs were given).
    # out["residual"] = (H+LE) - (Rn-G) is a closure DIAGNOSTIC, not advection.

    corrected = advection.apply_advection_correction(
        main_data = df[["H", "LE", "Rn", "G"]].to_dict("list"),
        H_adv = out["H_adv"], V_adv = out["VAT"], HA_Q = out["HA_Q"],
        rn_min = 75.0,                # Wang (2024) conditional-inclusion gate
    )
    # Advective terms are folded onto the turbulent-sum side
    #   (Rn - G = H + LE + HA_T + HA_Q + VAT)
    # but ONLY at timesteps where Wang's gate passes:
    #   Rn > rn_min AND (H + LE) < (Rn - G).
    # corrected["H_plus_LE_corrected"] vs ["H_plus_LE_orig"], the residual before
    # /after (["residual_corrected"] vs ["residual_orig"]) and the boolean
    # ["included"] mask report exactly which steps were corrected.

---


Energy-balance closure & advection
==================================

Detection strategy implemented
------------------------------

The :py:mod:`advection.advect_detect`
module applies four empirically proven criteria:

1. **Up-/down-wind flux divergence** (requires a reference tower).
2. **LE > AE** by >5 %.
3. Daytime negative H.
4. Temperature / humidity gradients.

Vertical advection uses canopy inversions plus mean subsidence tests.

Flux computation
----------------

:py:func:`advection.advection.compute_advection_fluxes`
returns

* *HA\_T* (alias *H\_adv*) – horizontal **heat** advection,
  ``rho*Cp*u*(dT/dx)*(zm-h)`` [W/m²] (Wang 2024 Eq. 5a; Moderow Term IV),
* *HA\_Q* – horizontal **moisture** advection,
  ``rho*lambda*u*(dq/dx)*(zm-h)`` [W/m²] (Wang 2024 Eq. 5b),
* *VAT* (alias *V\_adv*) – **measured** vertical **heat** advection,
  ``rho*Cp*w_bar*(T_zm - <T>)`` [W/m²] (Lee 1998; Wang 2024 Eq. 6), computed
  only when the planar-fit ``w_bar`` (or a ``detect_vertical`` mask) is
  supplied — otherwise ``None``,
* *VFD\_T* – optional vertical **heat-flux divergence**,
  ``-rho*Cp*(wT|zm - wT|h)`` [W/m²] (Wang 2024 Eq. 12), returned when the
  two-level ``wT_zm`` / ``wT_h`` fluxes are supplied — otherwise ``None``,
* *residual* (deprecated alias *adv\_in*) – the closure imbalance
  ``(H+LE) - (Rn-G)`` [W/m²], a **diagnostic only**, **not** an advective flux.

The horizontal terms are computed from the two-tower temperature/humidity
gradient (the wind direction selects the relevant upwind tower when several are
supplied). A negative *HA\_T* means heat advected **into** the field (the oasis
fingerprint). ``w_bar`` **must** come from planar fit / tilt correction, never
the raw sonic *w*; if the vertical term is requested but ``w_bar`` or the
column-mean temperature is missing, the function **raises** rather than
back-filling the energy-balance residual (the term is *never* a closure
residual).

For rigorous background, consult Prueger 2012, Dhungel 2022,
Moderow 2021, and Wang 2024 (see *References* section of the API docs).
