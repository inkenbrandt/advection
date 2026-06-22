=======
History
=======

History of Advective-Flux Research
==================================

This page traces the **evolution of scientific understanding** that led to the
modules packaged here.  It is not an exhaustive literature review; rather, it
highlights the pivotal ideas, field campaigns, and methodological innovations
that shaped today’s best practices for diagnosing and correcting **energy-balance
closure gaps** in eddy-covariance (EC) measurement of evapotranspiration.

-----------------------------------------------------------------
1. 1950 s – 1980 s  |  Early recognition of imbalance
-----------------------------------------------------------------

* **Bowen-ratio studies** on irrigated grass (Sweden, Nebraska) first reported
  that *λE + H* systematically fell short of *Rₙ − G* by 10–30 %.
* Explanations focused on **instrument drift** and **heat storage** in soil and
  canopy, but the role of advection was largely untested—few towers measured
  horizontal gradients.

-----------------------------------------------------------------
2. 1990 s  |  Rise of eddy-covariance networks
-----------------------------------------------------------------

* Widespread deployment of **open-path IRGAs** (LI-7500, OP-2) revealed that
  closure errors persisted even with high-frequency turbulence resolved.
* Foken & Oncley (1995) introduced **stationarity and quality-control tests**
  that became standard QC flags in FLUXNET, yet did not fully explain the gap.
* The first **ogive (cumulative cospectral) analyses** showed that missing
  energy resided in frequencies lower than the typical 30-min averaging window,
  hinting at **large-eddy transport**.

-----------------------------------------------------------------
3. 2000 – 2005  |  Field campaigns isolate advection
-----------------------------------------------------------------

* **EBEX-2000 (California, cotton)** deployed *multi-tower transects* across an
  irrigated–dry interface.  Horizontal heat advection by the mean flow closed
  roughly half the energy deficit and introduced cases where *λE > Rₙ − G*.
* **ADVEX (Europe, conifer forests)** quantified both horizontal and vertical
  advection terms.  Results were site-specific: one site achieved near-closure
  after advective corrections; others remained imbalanced, underscoring the
  complexity of canopy storage and terrain.

-----------------------------------------------------------------
4. 2006 – 2015  |  Agricultural mosaics & low-frequency eddies
-----------------------------------------------------------------

* **BEAREX08 and BEFLUX** (US Great Plains) demonstrated that warm, dry air
  advected over irrigated cotton can *enhance ET* by up to 25 %.
* Spectral work (Prueger et al. 2012) confirmed that **low-frequency (≫ 30 min)
  motions** carry a non-negligible fraction of sensible and latent heat.
* New **soft-rotation and planar-fit algorithms** improved H and λE estimates
  but left the advective component unmeasured.

-------------------------------------------------------------------
5. 2016 – Present  |  Integrated sensing, LES, and machine learning
-------------------------------------------------------------------

* **Wang et al. 2024** combined *multi-tower EC*, Doppler lidar, and UAV
  imagery over irrigated alfalfa.  Explicitly adding horizontal & vertical
  advective fluxes raised closure from 0.89 to 0.97.
* **Large-eddy simulations (LES)** now reproduce heterogeneous irrigation
  patterns, allowing virtual towers that quantify dispersive and advective
  transport.
* **Machine-learning residual models** (e.g. random forests) use meteorology,
  footprint heterogeneity, and wind fields to predict closure gaps in
  near-real-time.

-----------------------------------------------------------------
6. How this history shaped the package
-----------------------------------------------------------------

The modules in :py:mod:`advection` encode three decades of insight:

* :py:mod:`advection.advect_detect`
  implements empirical rules distilled from EBEX, BEAREX, and Wang 2024 to
  flag periods prone to **horizontal / vertical advection** using only tower
  data plus optional upwind references.
* :py:mod:`advection.advection`
  translates those flags into **flux corrections** that reconcile *λE + H* with
  *Rₙ − G*, following the energy-balance bookkeeping formalized by Foken & Oncley
  and expanded by recent campaign protocols.

-----------------------------------------------------------------
Key references
-----------------------------------------------------------------

* Foken, T. & Oncley, S. **(1995)** *Bulletin of the AMS* — stationarity tests  
* Wilson, K. et al. **(2002)** *Agric. For. Meteorol.* — closure statistics  
* Prueger, J. et al. **(2012)** *Agric. For. Meteorol.* — BEAREX low-freq. eddies  
* Moderow, U. et al. **(2021)** *Biogeosciences* — ADVEX advective budgets  
* Dhungel, S. et al. **(2022)** *Water Resources Research* — wind–closure link  
* Wang, T. et al. **(2024)** *J. Hydrometeorology* — multi-tower advection & closure  



0.2.0 (2026-06-21)
------------------

Physics-correctness overhaul aligning the library with the ``CLAUDE.md`` contract
(Wang et al. 2024; Moderow et al. 2021; Lee 1998; Twine et al. 2000; WPL 1980).
Several changes are **behavior-breaking** relative to 0.1.0.

* **Variance Bowen ratio is now signed (Wang 2024 Eq. 8).**
  ``compute_bowen_ratio_variance`` returns ``sign(corr(T', q')) * |beta|`` when a
  correlation, covariance, or fluctuation series is supplied — a **negative beta
  is the oasis fingerprint**. With no sign source it returns the unsigned
  magnitude and warns (backward compatible).
* **Horizontal advection is gradient-based, not a flux difference**
  (Wang 2024 Eqs. 5a/5b; Moderow Term IV). ``HA_T = rho*Cp*u*(dT/dx)*(zm-h)`` and
  ``HA_Q = rho*lambda*u*(dq/dx)*(zm-h)`` are computed from an upwind reference; a
  new ``HA_Q`` moisture-advection term is returned. **Breaking:**
  ``compute_advection_fluxes`` now **raises** without an upwind tower /
  ``tower_distance`` instead of silently returning ``H_adv = 0``.
* **Vertical advection uses the planar-fit mean vertical velocity**
  (Lee 1998; Wang 2024 Eq. 6): ``VAT = rho*Cp*w_bar*(T_zm - <T>)``. **Breaking:**
  ``VAT``/``V_adv`` is now this *measured* term and is ``None`` when no vertical
  inputs are given; if engaged without ``w_bar`` or the column-mean ``<T>`` the
  function **raises** rather than back-filling the residual.
* **Removed residual-as-advection behavior (hard rule).** The closure imbalance
  ``(H + LE) - (Rn - G)`` is returned only as the diagnostic ``residual``
  (``adv_in`` kept as a deprecated alias); it is never relabelled as an advective
  flux.
* **Real correction with a conditional-inclusion gate (Wang 2024).**
  ``apply_advection_correction`` folds the measured advective terms onto the
  turbulent side **only** where ``Rn > 75 W/m^2`` **and** ``(H + LE) < (Rn - G)``
  — the gate that lifted closure from 0.89 to 0.97 in the alfalfa study. Gated-out
  steps are left exactly uncorrected; ``NaN`` advective terms contribute 0.
* **New ``advection.closure`` module.** Twine et al. (2000) Bowen-ratio and
  residual-LE closure forcing plus EBR / residual / closure-slope diagnostics,
  kept deliberately separate from advection accounting. Bowen-ratio closure
  **warns when ``LE > (Rn - G)``** (the oasis case where forcing closure is
  physically wrong).
* **Singularity guards.** ``correct_sonic_heat_flux`` (small-negative-beta band)
  and ``latent_heat_flux_bowen`` (``beta -> -1``) now return ``nan`` with a
  warning instead of an unphysically large flux.
* **Robustness & vectorization.** Detection (``detect_horizontal_advection``,
  ``detect_vertical_advection``) is fully vectorized with documented, keyword
  thresholds, a wind-sector fetch gate, and ``np.isnan`` masking of gaps (the old
  ``is None`` test silently let ``NaN`` through).
* **WPL pre-step.** Added the convenience helper ``wpl_latent_heat_flux`` and
  documented throughout that open-path ``LE`` is assumed **already** WPL
  density-corrected (a mandatory, separate pre-step this library does not apply).
* **Docs.** Added a README "Physics & assumptions" section with a worked oasis
  example, a closure API reference page, and corrected equation citations in the
  ``latent_heat_flux_*`` / ``compute_sensible_heat_flux`` docstrings.

0.1.0 (2025-04-23)
------------------

* First release on PyPI.
