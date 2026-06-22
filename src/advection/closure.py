"""Energy-balance *closure* methods and diagnostics.

This module is deliberately kept **separate from the direct advection
accounting** in :mod:`advection.advection` and :mod:`advection.advect_detect`.
Closure *forcing* (Twine et al. 2000) and physical *advection accounting* (Wang
et al. 2024; Moderow et al. 2021) are two fundamentally different responses to
eddy-covariance under-closure and must not be conflated:

* **Advection accounting** *adds the measured advective fluxes* (``HA_T``,
  ``HA_Q``, ``VAT``) to the budget — it explains *where the missing energy went*.
  This is the physically correct treatment in the **oasis regime** and is what
  this library exists to do (see :func:`advection.compute_advection_fluxes` and
  :func:`advection.apply_advection_correction`).
* **Closure forcing** *rescales the measured turbulent fluxes* so the budget
  shuts by construction. It is a pragmatic post-processing convention, not a
  physical correction, and it is **wrong in the oasis case** — see the caveat on
  :func:`bowen_ratio_closure`.

The two are offered here side by side so a user can compute either, but the
synthesis recommendation (``CLAUDE.md``) stands: **prefer adding measured
advective fluxes; do not force Bowen-ratio closure when ``LE > (Rn - G)``.**

WPL assumption
--------------
Every ``LE`` consumed or produced here is assumed to be **already WPL
(Webb-Pearman-Leuning 1980) density-corrected**. WPL is a mandatory, *separate*
pre-processing step for open-path ``LE``/CO2 — not an advection or closure
correction — and this library does not apply it. See
:func:`advection.advection.wpl_latent_heat_flux` (a convenience helper; prefer
EddyPro / established processing) and the package README.

Sign / storage conventions (``CLAUDE.md``)
------------------------------------------
* Surface energy balance with storage: ``Rn - G - J = H + LE``.
* Available energy: ``Rn - G - J`` (``J`` = air heat storage, Wang Eq. 11;
  default ``0``). ``S`` is used as the storage symbol in the closure-forcing
  signatures below to match the Twine et al. (2000) presentation; it plays the
  same role as ``J``.
* Closure residual: ``Residual = Rn - G - J - H - LE`` (positive ⇒ available
  energy exceeds the turbulent sum, the usual under-closure gap). This is the
  ``CLAUDE.md`` convention and is the **negative** of the legacy ``'residual'``
  diagnostic returned by :func:`advection.compute_advection_fluxes`, which uses
  ``(H + LE) - (Rn - G)``; the sign difference is intentional and documented.
* Energy Balance Ratio: ``EBR = sum(H + LE) / sum(Rn - G - J)``.

References
----------
Twine, T. E., et al. (2000), *Correcting eddy-covariance flux underestimates
over a grassland*, Agric. For. Meteorol. 103, 279-300 — the Bowen-ratio and
residual closure methods.
Wilson, K., et al. (2002), *Energy balance closure at FLUXNET sites*, Agric.
For. Meteorol. 113, 223-243 — the closure-slope regression diagnostic.
Wang et al. (2024); Moderow et al. (2021) — advection accounting and the
OUT-positive sign convention.
"""

import warnings

import numpy as np

__all__ = [
    "bowen_ratio_closure",
    "residual_le_closure",
    "energy_balance_residual",
    "energy_balance_ratio",
    "closure_slope",
]


def _scalarize(arr):
    """Return a Python float for a 0-d result, else the array unchanged.

    Closure helpers accept scalars or series; this keeps a scalar call returning
    a scalar rather than a 0-d :class:`numpy.ndarray`.
    """
    arr = np.asarray(arr)
    return float(arr) if arr.ndim == 0 else arr


def bowen_ratio_closure(Rn, G, H, LE, S=0.0, *, warn_oasis=True, singular_tol=1e-12):
    r"""Force energy-balance closure by Bowen-ratio-preserving rescaling.

    Implements the **Bowen-ratio (BR) closure** of Twine et al. (2000): both
    turbulent fluxes are multiplied by the **same** scale factor so their sum
    equals the available energy while their ratio — the Bowen ratio
    ``beta = H / LE`` — is left **unchanged**::

        f          = (Rn - G - S) / (H + LE)
        H_closed   = f * H
        LE_closed  = f * LE
        H_closed + LE_closed = Rn - G - S          (closure by construction)
        H_closed / LE_closed = H / LE = beta       (beta preserved)

    ``S`` is the storage term (air heat storage ``J`` and/or any other storage);
    pass ``S=0`` (default) for the storage-free balance ``Rn - G = H + LE``.

    .. caution:: **Do NOT use Bowen-ratio closure in the oasis / advection
       regime** — i.e. whenever ``LE > (Rn - G - S)`` (equivalently the
       evaporative fraction ``EF = LE / (Rn - G) > 1``). There, warm dry air
       advects *extra* energy into the control volume, so the true ``LE``
       legitimately **exceeds** the available energy and no rescaling of the
       *measured* fluxes can reproduce it: BR closure would shrink ``LE`` toward
       the available energy and is **physically wrong**. This is a ``CLAUDE.md``
       hard rule (*never force Bowen-ratio closure when ``LE > (Rn - G)``*). In
       that regime, add the **measured advective fluxes** instead
       (:func:`advection.compute_advection_fluxes`,
       :func:`advection.apply_advection_correction`); never close the budget by
       forcing the turbulent fluxes. A :class:`UserWarning` is emitted (unless
       ``warn_oasis=False``) when any step has ``LE > (Rn - G - S)``.

    Parameters
    ----------
    Rn : float or array-like
        Net radiation [W/m^2].
    G : float or array-like
        Ground heat flux (storage-corrected) [W/m^2].
    H : float or array-like
        Measured sensible heat flux [W/m^2].
    LE : float or array-like
        Measured latent heat flux [W/m^2]. Assumed **already WPL
        (Webb-Pearman-Leuning 1980) density-corrected** (a mandatory separate
        pre-step, not a closure fix; see :func:`advection.advection.wpl_latent_heat_flux`).
    S : float or array-like, optional
        Storage term [W/m^2] (air heat storage ``J`` etc.); default ``0``.
    warn_oasis : bool, optional
        If ``True`` (default), emit a :class:`UserWarning` when any step has
        ``LE > (Rn - G - S)`` — the oasis case where BR closure is invalid.
    singular_tol : float, optional
        Magnitude of the turbulent sum ``H + LE`` at or below which the scale
        factor is singular; those steps return ``nan`` (default 1e-12).

    Returns
    -------
    dict
        Keys (each a float for scalar input, else a length-matched array):

        - ``'H'``      : closed sensible heat flux ``f * H`` [W/m^2].
        - ``'LE'``     : closed latent heat flux ``f * LE`` [W/m^2].
        - ``'factor'`` : the common scale factor ``f`` (``nan`` where singular).
        - ``'beta'``   : the preserved Bowen ratio ``H / LE`` (``nan`` where
          ``LE == 0``). ``H_closed / LE_closed`` equals this by construction.

    Warns
    -----
    UserWarning
        When ``LE > (Rn - G - S)`` for any step (oasis regime; BR closure
        invalid), unless ``warn_oasis=False``; and when the turbulent sum
        ``H + LE`` is within ``singular_tol`` of zero (factor undefined → ``nan``).

    References
    ----------
    Twine, T. E., et al. (2000), Agric. For. Meteorol. 103, 279-300 (BR closure).
    """
    Rn = np.asarray(Rn, dtype=float)
    G = np.asarray(G, dtype=float)
    H = np.asarray(H, dtype=float)
    LE = np.asarray(LE, dtype=float)
    S = np.asarray(S, dtype=float)

    available = Rn - G - S
    turbulent = H + LE

    # Oasis guard: BR closure is physically invalid when LE exceeds the
    # available energy (EF > 1). Warn rather than silently producing a wrong,
    # LE-shrinking "closure" (CLAUDE.md hard rule).
    if warn_oasis:
        with np.errstate(invalid="ignore"):
            oasis = np.asarray(LE > available, dtype=bool)
        if np.any(oasis):
            warnings.warn(
                "bowen_ratio_closure: LE > (Rn - G - S) for "
                f"{int(np.count_nonzero(oasis))} step(s) — the oasis/advection "
                "regime (EF > 1). Bowen-ratio closure is physically invalid "
                "there: it would shrink LE toward the available energy. Add the "
                "MEASURED advective fluxes instead (compute_advection_fluxes / "
                "apply_advection_correction); do not force closure here "
                "(CLAUDE.md hard rule).",
                UserWarning,
                stacklevel=2,
            )

    # Singular factor where the turbulent sum vanishes -> nan (cannot rescale).
    singular = np.abs(turbulent) <= singular_tol
    if np.any(singular):
        warnings.warn(
            "bowen_ratio_closure: the turbulent sum (H + LE) is within "
            f"singular_tol ({singular_tol:.1e}) of zero for "
            f"{int(np.count_nonzero(singular))} step(s); the Bowen-ratio scale "
            "factor (Rn - G - S)/(H + LE) is undefined there. Returning NaN for "
            "those steps.",
            UserWarning,
            stacklevel=2,
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(singular, np.nan, available / turbulent)
        beta = np.where(LE == 0.0, np.nan, H / LE)

    H_closed = factor * H
    LE_closed = factor * LE

    return {
        "H": _scalarize(H_closed),
        "LE": _scalarize(LE_closed),
        "factor": _scalarize(factor),
        "beta": _scalarize(beta),
    }


def residual_le_closure(Rn, G, H, S=0.0):
    r"""Force closure by assigning the entire residual to latent heat (LE).

    Implements the **residual-LE closure** of Twine et al. (2000): the measured
    sensible heat flux ``H`` is **trusted as-is** and the latent heat flux is set
    to whatever closes the budget::

        LE = Rn - G - S - H

    Unlike :func:`bowen_ratio_closure`, this does **not** preserve the Bowen
    ratio — ``H`` is held fixed and ``LE`` absorbs the full closure gap. It is
    the appropriate choice when ``H`` is believed more reliable than ``LE`` (e.g.
    open-path ``LE`` with WPL/spectral uncertainty), and the two methods bracket
    the plausible partition of the missing energy.

    .. note:: This is still **closure forcing**, not advection accounting. In the
       oasis regime the missing energy is genuine advective input, and attributing
       all of it to ``LE`` inflates the latent flux rather than identifying the
       advection. Prefer the measured advective fluxes
       (:func:`advection.compute_advection_fluxes`) there. (Residual-LE closure is
       *less* pathological than Bowen-ratio closure in the oasis case — it does
       not collapse ``LE`` — but it is still a forced, non-physical partition.)

    Parameters
    ----------
    Rn : float or array-like
        Net radiation [W/m^2].
    G : float or array-like
        Ground heat flux (storage-corrected) [W/m^2].
    H : float or array-like
        Measured sensible heat flux [W/m^2] (held fixed).
    S : float or array-like, optional
        Storage term [W/m^2] (air heat storage ``J`` etc.); default ``0``.

    Returns
    -------
    float or numpy.ndarray
        Closed latent heat flux ``LE = Rn - G - S - H`` [W/m^2] (float for scalar
        input, else a length-matched array). To be comparable with a *measured*
        open-path ``LE``, that measurement must be **already WPL
        (Webb-Pearman-Leuning 1980) density-corrected** (a mandatory separate
        pre-step; see :func:`advection.advection.wpl_latent_heat_flux`).

    References
    ----------
    Twine, T. E., et al. (2000), Agric. For. Meteorol. 103, 279-300 (residual
    closure).
    """
    Rn = np.asarray(Rn, dtype=float)
    G = np.asarray(G, dtype=float)
    H = np.asarray(H, dtype=float)
    S = np.asarray(S, dtype=float)
    return _scalarize(Rn - G - S - H)


def energy_balance_residual(Rn, G, H, LE, J=0.0):
    r"""Compute the energy-balance closure **residual** (a diagnostic, not a flux).

    Uses the ``CLAUDE.md`` convention::

        Residual = Rn - G - J - H - LE

    A **positive** residual means the available energy ``Rn - G - J`` exceeds the
    turbulent sum ``H + LE`` — the usual eddy-covariance *under-closure* gap. A
    negative residual means over-closure (turbulent sum exceeds available energy),
    which in the oasis regime signals advective input (``EF > 1``).

    .. note:: This is the **negative** of the legacy ``'residual'`` returned by
       :func:`advection.compute_advection_fluxes` (which uses
       ``(H + LE) - (Rn - G)``). The sign here follows ``CLAUDE.md``; the
       difference is intentional. Either way the residual is a **closure
       diagnostic only** and must never be relabelled as an advective flux
       (``CLAUDE.md`` hard rule).

    Parameters
    ----------
    Rn : float or array-like
        Net radiation [W/m^2].
    G : float or array-like
        Ground heat flux (storage-corrected) [W/m^2].
    H : float or array-like
        Sensible heat flux [W/m^2].
    LE : float or array-like
        Latent heat flux [W/m^2].
    J : float or array-like, optional
        Air heat storage [W/m^2] (Wang Eq. 11); default ``0`` for the
        storage-free balance ``Rn - G = H + LE``.

    Returns
    -------
    float or numpy.ndarray
        Closure residual [W/m^2] (float for scalar input, else array).
    """
    Rn = np.asarray(Rn, dtype=float)
    G = np.asarray(G, dtype=float)
    H = np.asarray(H, dtype=float)
    LE = np.asarray(LE, dtype=float)
    J = np.asarray(J, dtype=float)
    return _scalarize(Rn - G - J - H - LE)


def energy_balance_ratio(H, LE, Rn, G, J=0.0):
    r"""Compute the Energy Balance Ratio (EBR) over a series.

    Implements the ``CLAUDE.md`` definition::

        EBR = sum(H + LE) / sum(Rn - G - J)

    EBR is the bulk closure fraction over an averaging window: ``EBR == 1`` is
    perfect closure, ``EBR < 1`` under-closure (the typical 0.8-0.9 of
    eddy-covariance towers), and ``EBR > 1`` over-closure (turbulent sum exceeds
    available energy — the oasis advective-input signature).

    Only timesteps where **all** of ``H``, ``LE``, ``Rn``, ``G`` and ``J`` are
    finite are summed (complete-case masking), so the numerator and denominator
    are always formed from the same sample and a gap in one term cannot bias the
    ratio.

    Parameters
    ----------
    H, LE : float or array-like
        Sensible and latent heat flux series [W/m^2].
    Rn, G : float or array-like
        Net radiation and ground heat flux series [W/m^2].
    J : float or array-like, optional
        Air heat storage series [W/m^2]; default ``0``.

    Returns
    -------
    float
        The energy balance ratio (dimensionless). ``nan`` if no timestep has all
        components finite, or if the summed available energy is zero.

    Warns
    -----
    UserWarning
        When no complete-case timesteps remain, or the summed available energy
        ``sum(Rn - G - J)`` is zero (EBR undefined → ``nan``).
    """
    H, LE, Rn, G, J = np.broadcast_arrays(
        np.asarray(H, dtype=float),
        np.asarray(LE, dtype=float),
        np.asarray(Rn, dtype=float),
        np.asarray(G, dtype=float),
        np.asarray(J, dtype=float),
    )
    mask = (
        np.isfinite(H)
        & np.isfinite(LE)
        & np.isfinite(Rn)
        & np.isfinite(G)
        & np.isfinite(J)
    )
    if not np.any(mask):
        warnings.warn(
            "energy_balance_ratio: no timestep has all of H, LE, Rn, G, J "
            "finite; EBR is undefined. Returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    turbulent = float(np.sum(H[mask] + LE[mask]))
    available = float(np.sum(Rn[mask] - G[mask] - J[mask]))
    if available == 0.0:
        warnings.warn(
            "energy_balance_ratio: summed available energy sum(Rn - G - J) is "
            "zero; EBR is undefined. Returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")
    return turbulent / available


def closure_slope(H, LE, Rn, G, J=0.0, *, force_origin=False):
    r"""Closure-slope diagnostic via ordinary least-squares regression.

    Regresses the turbulent flux on the available energy across the series
    (Wilson et al. 2002)::

        y = H + LE                 (dependent / turbulent sum)
        x = Rn - G - J             (independent / available energy)
        y = slope * x + intercept

    The **slope** is the standard scalar measure of energy-balance closure:
    ``slope == 1`` with ``intercept == 0`` is perfect closure; eddy-covariance
    towers typically report slopes of ~0.8 (under-closure). A slope **above 1**
    indicates over-closure — the oasis advective-input signature.

    Non-finite pairs (a ``nan`` in either ``x`` or ``y``) are dropped before the
    fit.

    Parameters
    ----------
    H, LE : array-like
        Sensible and latent heat flux series [W/m^2].
    Rn, G : array-like
        Net radiation and ground heat flux series [W/m^2].
    J : float or array-like, optional
        Air heat storage series [W/m^2]; default ``0``.
    force_origin : bool, optional
        If ``True``, constrain the fit through the origin (``intercept == 0``,
        ``slope = sum(x*y) / sum(x*x)``). Default ``False`` (free intercept).

    Returns
    -------
    dict
        Keys:

        - ``'slope'``     : regression slope (dimensionless).
        - ``'intercept'`` : regression intercept [W/m^2] (``0.0`` when
          ``force_origin=True``).
        - ``'r_squared'`` : coefficient of determination of the fit.
        - ``'n'``         : number of finite ``(x, y)`` pairs used.

    Raises
    ------
    ValueError
        If fewer than two finite ``(x, y)`` pairs are available — a slope is then
        undefined.

    References
    ----------
    Wilson, K., et al. (2002), Agric. For. Meteorol. 113, 223-243 (closure
    regression). Twine, T. E., et al. (2000).
    """
    H, LE, Rn, G, J = np.broadcast_arrays(
        np.asarray(H, dtype=float),
        np.asarray(LE, dtype=float),
        np.asarray(Rn, dtype=float),
        np.asarray(G, dtype=float),
        np.asarray(J, dtype=float),
    )
    x = Rn - G - J
    y = H + LE
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    if n < 2:
        raise ValueError(
            "closure_slope: need at least two finite (available-energy, "
            f"turbulent-sum) pairs to fit a slope; got {n}."
        )

    if force_origin:
        sxx = float(np.sum(x * x))
        if sxx == 0.0:
            raise ValueError(
                "closure_slope: all available-energy values are zero; an "
                "origin-forced slope is undefined."
            )
        slope = float(np.sum(x * y) / sxx)
        intercept = 0.0
        y_pred = slope * x
    else:
        slope, intercept = (float(v) for v in np.polyfit(x, y, 1))
        y_pred = slope * x + intercept

    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "n": n,
    }
