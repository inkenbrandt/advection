"""Functions for detecting horizontal and vertical advection."""

import numpy as np

# Air properties are not fixed constants: use the physically-based helpers from
# ``advection`` (which depend on P, T, q) instead of the hard-coded AIR_DENSITY,
# SPECIFIC_HEAT_AIR and LATENT_HEAT_VAP that previously lived here. Genuine fixed
# constants come from the shared ``_constants`` module. These back the physical
# advection terms and are imported here for that use.
from ._constants import G_OVER_CP, MU, VON_KARMAN  # noqa: F401
from .advection import (  # noqa: F401
    _to_kelvin,
    air_density,
    latent_heat_vaporization,
    rh_to_specific_humidity,
    specific_heat_moist_air,
)


def _as_float_series(value, n):
    """Return ``value`` as a length-``n`` float array, or ``None`` if absent.

    Scalars are broadcast to length ``n``; Python ``None`` entries inside a
    list become ``np.nan`` (because ``np.asarray(..., dtype=float)`` maps
    ``None`` to ``nan``). Missing data is therefore represented uniformly as
    ``nan`` so it can be masked with :func:`numpy.isnan` -- never with an
    ``is None`` test, which silently never fires on a float ``ndarray``.
    """
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    return arr


def detect_horizontal_advection(
    main_flux,
    *,
    rn=None,
    g=None,
    le_main=None,
    temp_main=None,
    temp_upwind=None,
    humidity_main=None,
    humidity_upwind=None,
    wind_dir=None,
    upwind_dir=None,
    upwind_flux=None,
    rn_high=50.0,
    h_neg_threshold=0.0,
    ef_tol=1.05,
    temp_diff_threshold=1.0,
    humidity_diff_threshold=0.001,
    wind_sector_deg=45.0,
    upwind_h_excess=20.0,
    return_components=False,
):
    """Detect periods of horizontal advection influencing the main tower.

    Fully **vectorized** (no Python per-timestep loop) and built from the
    documented oasis-regime detection signals. Each signal is an independent,
    parameterized boolean flag; the returned mask is their logical **OR**.
    Missing data is carried as ``np.nan`` and excluded from every comparison via
    :func:`numpy.isnan` (an ``x is None`` test, used by the previous version,
    never fires on a float ``ndarray`` and so silently failed to mask gaps).

    Detection signals
    -----------------
    1. **Negative midday sensible heat / negative Bowen ratio** (oasis
       fingerprint; Wang 2024 §2.2). Fires where ``H < h_neg_threshold`` while
       ``Rn > rn_high`` -- warm dry air advected onto a cool transpiring surface
       drives a downward (negative) ``H`` in full daytime. Needs ``main_flux``
       and ``rn``.
    2. **Evaporative fraction ``EF = LE / (Rn - G) > 1``** (advective input;
       synthesis §2.3). Fires where ``LE > (Rn - G) * ef_tol`` and the available
       energy ``Rn - G`` is positive (``EF`` is only meaningful by day). Needs
       ``le_main``, ``rn`` and ``g``.
    3. **Horizontal temperature / humidity gradient** between the main and an
       upwind tower (warm and/or dry air upwind). The warm-upwind flag fires
       where ``temp_upwind > temp_main + temp_diff_threshold``; the dry-upwind
       flag where ``humidity_main - humidity_upwind > humidity_diff_threshold``
       (main moister than upwind). Both are evaluated **only when the wind is
       blowing from the upwind tower** -- i.e. ``wind_dir`` is within
       ``±wind_sector_deg`` of the bearing ``upwind_dir`` (a ±45° sector by
       default; ±20° is a common stricter choice). If either ``wind_dir`` or
       ``upwind_dir`` is omitted the sector gate is not applied.

    In addition, when an upwind sensible-heat series ``upwind_flux`` is supplied,
    an **upwind sensible-heat excess** flag fires where
    ``upwind_flux > main_flux + upwind_h_excess`` (also subject to the wind-sector
    gate). ``upwind_h_excess`` is an **absolute W/m^2** difference -- this is the
    single, named, documented threshold that replaces the previous version's
    contradiction between a docstring that said "20% higher" and code that added
    an absolute ``20 W/m^2``. The absolute reading is the one kept.

    Parameters
    ----------
    main_flux : array-like
        Sensible heat flux ``H`` at the main tower [W/m^2].
    rn : array-like, optional
        Net radiation ``Rn`` at the main site [W/m^2].
    g : array-like, optional
        Soil/ground heat flux ``G`` at the main site [W/m^2].
    le_main : array-like, optional
        Latent heat flux ``LE`` at the main tower [W/m^2].
    temp_main, temp_upwind : array-like, optional
        Air temperature at the main and upwind towers [°C or K] (same units).
    humidity_main, humidity_upwind : array-like, optional
        Specific humidity at the main and upwind towers [kg/kg].
    wind_dir : array-like, optional
        Wind direction at the main tower [deg from north].
    upwind_dir : float, optional
        Bearing from the main tower toward the upwind tower [deg from north].
        Together with ``wind_dir`` this gates the gradient and upwind-flux
        signals to a ``±wind_sector_deg`` sector about the fetch direction.
    upwind_flux : array-like, optional
        Sensible heat flux ``H`` at an upwind reference tower [W/m^2].
    rn_high : float, default 50.0
        Net-radiation threshold [W/m^2] above which a period is treated as
        midday for the negative-``H`` signal.
    h_neg_threshold : float, default 0.0
        Sensible-heat threshold [W/m^2]; ``H`` below this (while ``Rn > rn_high``)
        flags the oasis negative-``H`` signal.
    ef_tol : float, default 1.05
        Evaporative-fraction tolerance. ``LE > (Rn - G) * ef_tol`` flags
        advective input; the 5 % margin guards against measurement noise.
    temp_diff_threshold : float, default 1.0
        Minimum upwind-minus-main air-temperature excess [°C or K] for the
        warm-upwind gradient flag.
    humidity_diff_threshold : float, default 0.001
        Minimum main-minus-upwind specific-humidity excess [kg/kg] for the
        dry-upwind gradient flag.
    wind_sector_deg : float, default 45.0
        Half-width [deg] of the wind-direction sector about ``upwind_dir`` within
        which the gradient and upwind-flux signals are evaluated.
    upwind_h_excess : float, default 20.0
        Absolute amount [W/m^2] by which ``upwind_flux`` must exceed ``main_flux``
        to flag an upwind sensible-heat excess. Absolute W/m^2, not a percentage.
    return_components : bool, default False
        If True, also return a dict mapping each criterion name to its boolean
        mask (plus ``'wind_aligned'``), suitable for ``pandas.DataFrame(...)``.

    Returns
    -------
    numpy.ndarray
        Boolean mask (length ``n``); True where any signal detects advection.
    dict, optional
        Returned only when ``return_components`` is True, as
        ``(mask, components)``. Keys: ``'negative_H'``, ``'ef_gt_1'``,
        ``'warm_upwind'``, ``'dry_upwind'``, ``'upwind_H_excess'`` and
        ``'wind_aligned'`` (the sector gate), each a length-``n`` boolean array.

    References
    ----------
    Wang et al. (2024) §2.2 (negative Bowen / oasis fingerprint) and the EF > 1
    advective-input synthesis §2.3. Moderow et al. (2021) OUT-positive sign
    convention.
    """
    main_flux = np.atleast_1d(np.asarray(main_flux, dtype=float))
    n = main_flux.shape[0]
    main_valid = ~np.isnan(main_flux)

    rn = _as_float_series(rn, n)
    g = _as_float_series(g, n)
    le_main = _as_float_series(le_main, n)
    temp_main = _as_float_series(temp_main, n)
    temp_upwind = _as_float_series(temp_upwind, n)
    humidity_main = _as_float_series(humidity_main, n)
    humidity_upwind = _as_float_series(humidity_upwind, n)
    wind_dir = _as_float_series(wind_dir, n)
    upwind_flux = _as_float_series(upwind_flux, n)

    # --- wind-sector gate for the upwind-referenced signals ------------------
    # Aligned == wind blowing from the bearing to the upwind tower (within
    # ±wind_sector_deg). With no directional information the gate is open (all
    # True); a NaN wind_dir is treated as "not aligned" -- fetch cannot be
    # confirmed, so the upwind-referenced signals are withheld for that step.
    if wind_dir is not None and upwind_dir is not None:
        ang_diff = np.abs(((wind_dir - upwind_dir + 180.0) % 360.0) - 180.0)
        wind_aligned = ~np.isnan(wind_dir) & (ang_diff <= wind_sector_deg)
    else:
        wind_aligned = np.ones(n, dtype=bool)

    # --- Signal 1: negative midday H (oasis fingerprint, Wang 2024 §2.2) ------
    negative_H = np.zeros(n, dtype=bool)
    if rn is not None:
        valid = main_valid & ~np.isnan(rn)
        negative_H = valid & (main_flux < h_neg_threshold) & (rn > rn_high)

    # --- Signal 2: EF = LE/(Rn-G) > 1 (advective input, synthesis §2.3) -------
    ef_gt_1 = np.zeros(n, dtype=bool)
    if le_main is not None and rn is not None and g is not None:
        avail_energy = rn - g  # Rn - G
        valid = ~np.isnan(le_main) & ~np.isnan(avail_energy) & (avail_energy > 0.0)
        ef_gt_1 = valid & (le_main > avail_energy * ef_tol)

    # --- Signal 3: horizontal T / q gradients (warm/dry upwind), wind-gated ---
    warm_upwind = np.zeros(n, dtype=bool)
    if temp_main is not None and temp_upwind is not None:
        valid = ~np.isnan(temp_main) & ~np.isnan(temp_upwind)
        warm_upwind = (
            wind_aligned & valid & (temp_upwind > temp_main + temp_diff_threshold)
        )

    dry_upwind = np.zeros(n, dtype=bool)
    if humidity_main is not None and humidity_upwind is not None:
        valid = ~np.isnan(humidity_main) & ~np.isnan(humidity_upwind)
        dry_upwind = (
            wind_aligned
            & valid
            & ((humidity_main - humidity_upwind) > humidity_diff_threshold)
        )

    # --- Optional: upwind sensible-heat excess (absolute W/m^2), wind-gated ---
    upwind_H_excess = np.zeros(n, dtype=bool)
    if upwind_flux is not None:
        valid = main_valid & ~np.isnan(upwind_flux)
        upwind_H_excess = (
            wind_aligned & valid & (upwind_flux > main_flux + upwind_h_excess)
        )

    mask = negative_H | ef_gt_1 | warm_upwind | dry_upwind | upwind_H_excess

    if return_components:
        components = {
            "negative_H": negative_H,
            "ef_gt_1": ef_gt_1,
            "warm_upwind": warm_upwind,
            "dry_upwind": dry_upwind,
            "upwind_H_excess": upwind_H_excess,
            "wind_aligned": wind_aligned,
        }
        return mask, components
    return mask


def detect_vertical_advection(
    vertical_w=None,
    *,
    temp_profile_lower=None,
    temp_profile_upper=None,
    main_H=None,
    rn=None,
    g=None,
    w_threshold=0.05,
    rn_g_threshold=50.0,
    temp_grad_threshold=0.5,
    h_anomaly_threshold=20.0,
    use_h_anomaly=True,
    return_components=False,
):
    """Detect periods of energetically significant **vertical heat advection**.

    Centred on the **planar-fit mean vertical velocity** ``w_bar`` (Lee 1998),
    not on the ad hoc "inverted profile" heuristic the previous version used as a
    gate. Lee (1998) shows that as ``|w_bar|`` approaches ~0.05 m/s the vertical
    advection term

    .. math:: VAT \\approx \\rho\\, C_p\\, \\bar{w}\\, (T_{zm} - \\langle T\\rangle)

    becomes energetically significant (of order ``-100 W/m^2`` at midday in the
    oasis regime; CLAUDE.md "Vertical heat advection", Wang 2024 Eq. 6). A small
    mean vertical velocity is therefore, on its own, sufficient grounds to flag a
    period -- it no longer has to co-occur with a temperature inversion.

    Fully **vectorized** (no per-timestep Python loop). Missing data is carried as
    ``np.nan`` and excluded from every comparison with :func:`numpy.isnan` (an
    ``x is None`` test, used by the previous version, never fires on a float
    ``ndarray`` and so silently failed to mask gaps). The returned mask is the
    logical **OR** of the independent signals below.

    .. warning::
       ``vertical_w`` **must** be the **planar-fit (or otherwise tilt-corrected)
       mean vertical velocity** ``w_bar``. The **raw sonic** ``w`` must **not** be
       used: instrument tilt relative to the mean streamline biases ``w`` by far
       more than the ~0.05 m/s threshold here, so an uncorrected ``w`` makes this
       detector fire on tilt, not on real subsidence/uplift.

    Detection signals
    -----------------
    1. **Primary -- significant mean vertical velocity** (Lee 1998). Fires where
       ``|w_bar| > w_threshold`` (default 0.05 m/s). This is the dominant signal
       and the basis of the rewrite. Needs ``vertical_w``.
    2. **Supporting -- daytime vertical temperature gradient of the advective
       sign.** Fires where the period is daytime (``Rn - G > rn_g_threshold``)
       **and** temperature increases with height
       (``temp_profile_upper > temp_profile_lower + temp_grad_threshold``). Warm
       air aloft over a cooler surface gives ``(T_zm - <T>) > 0``; paired with the
       typical oasis subsidence (``w_bar < 0``) this yields a **negative**
       (downward, energy-IN) ``VAT`` -- the oasis advective sign under the
       Moderow 2021 OUT-positive convention. Needs the two temperatures plus
       ``rn`` and ``g``.
    3. **Optional, weak/secondary -- anomalous daytime sensible heat.** Fires
       where it is daytime and ``main_H < h_anomaly_threshold`` (very low or
       downward ``H`` when it would normally be positive). This is a **screening
       heuristic only**: anomalously low daytime ``H`` has many possible causes
       (cloud, advection, instrument issues), so it is corroborating evidence at
       best, never a quantitative ``VAT``. Disable it with ``use_h_anomaly=False``.

    Parameters
    ----------
    vertical_w : array-like, optional
        **Planar-fit mean vertical velocity** ``w_bar`` [m/s] -- see the warning
        above. Scalars are broadcast. The primary detection signal.
    temp_profile_lower : array-like, optional
        Temperature near the surface/canopy [°C or K].
    temp_profile_upper : array-like, optional
        Temperature at (or above) the measurement height [°C or K], same units as
        ``temp_profile_lower``.
    main_H : array-like, optional
        Sensible heat flux ``H`` at the main tower [W/m^2] (H-anomaly signal).
    rn : array-like, optional
        Net radiation ``Rn`` [W/m^2] (daytime gate).
    g : array-like, optional
        Soil/ground heat flux ``G`` [W/m^2] (daytime gate).
    w_threshold : float, default 0.05
        Magnitude of ``w_bar`` [m/s] above which vertical advection is treated as
        energetically significant (Lee 1998).
    rn_g_threshold : float, default 50.0
        Available-energy threshold ``Rn - G`` [W/m^2] separating daytime from
        nighttime/low-energy periods for the supporting and H-anomaly signals.
    temp_grad_threshold : float, default 0.5
        Minimum upper-minus-lower temperature difference [°C or K] counted as a
        gradient of the advective sign (warm air aloft).
    h_anomaly_threshold : float, default 20.0
        Daytime ``H`` [W/m^2] below which the weak H-anomaly signal fires.
    use_h_anomaly : bool, default True
        Whether to include the weak/secondary H-anomaly signal in the mask.
    return_components : bool, default False
        If ``True`` also return the per-signal boolean arrays (see Returns).

    Returns
    -------
    numpy.ndarray
        Boolean mask of detected vertical-advection periods (length matches the
        input series).
    tuple of (numpy.ndarray, dict)
        If ``return_components`` is ``True``, ``(mask, components)`` with keys
        ``'w_significant'``, ``'temp_gradient'``, ``'supporting'``,
        ``'h_anomaly'`` and ``'daytime'`` -- each a length-``n`` boolean array.

    References
    ----------
    Lee, X. (1998), on the significance of planar-fit mean vertical velocity for
    vertical advection. Wang et al. (2024) Eq. 6 (``VAT``). Moderow et al. (2021)
    OUT-positive sign convention.
    """
    # Length is taken from the longest supplied series; scalars (length 1) and
    # absent inputs (None) do not constrain it.
    n = 0
    for series in (
        vertical_w,
        temp_profile_lower,
        temp_profile_upper,
        main_H,
        rn,
        g,
    ):
        if series is not None:
            n = max(n, np.atleast_1d(np.asarray(series, dtype=float)).shape[0])

    vertical_w = _as_float_series(vertical_w, n)
    temp_profile_lower = _as_float_series(temp_profile_lower, n)
    temp_profile_upper = _as_float_series(temp_profile_upper, n)
    main_H = _as_float_series(main_H, n)
    rn = _as_float_series(rn, n)
    g = _as_float_series(g, n)

    # --- daytime gate: available energy Rn - G above threshold ----------------
    daytime = np.zeros(n, dtype=bool)
    if rn is not None and g is not None:
        avail_energy = rn - g
        daytime = ~np.isnan(avail_energy) & (avail_energy > rn_g_threshold)

    # --- Signal 1 (primary): significant planar-fit mean vertical velocity ----
    w_significant = np.zeros(n, dtype=bool)
    if vertical_w is not None:
        w_significant = ~np.isnan(vertical_w) & (np.abs(vertical_w) > w_threshold)

    # --- Signal 2 (supporting): daytime vertical T gradient of advective sign -
    temp_gradient = np.zeros(n, dtype=bool)
    if temp_profile_lower is not None and temp_profile_upper is not None:
        valid = ~np.isnan(temp_profile_lower) & ~np.isnan(temp_profile_upper)
        temp_gradient = valid & (
            temp_profile_upper > temp_profile_lower + temp_grad_threshold
        )
    supporting = daytime & temp_gradient

    # --- Signal 3 (optional, weak): anomalously low/downward daytime H --------
    h_anomaly = np.zeros(n, dtype=bool)
    if use_h_anomaly and main_H is not None:
        h_anomaly = daytime & ~np.isnan(main_H) & (main_H < h_anomaly_threshold)

    mask = w_significant | supporting | h_anomaly

    if return_components:
        components = {
            "w_significant": w_significant,
            "temp_gradient": temp_gradient,
            "supporting": supporting,
            "h_anomaly": h_anomaly,
            "daytime": daytime,
        }
        return mask, components
    return mask


def _broadcast_series(value, n, *, name):
    """Return ``value`` as a length-``n`` float array (scalars are broadcast).

    Heights, distances and the wind speed may be supplied either as a single
    site constant or as a per-timestep series; this normalizes both to a float
    array of length ``n`` so the per-timestep arithmetic is uniform.
    """
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.shape[0] != n:
        raise ValueError(
            f"compute_advection_fluxes: '{name}' has length {arr.shape[0]}, "
            f"expected {n} to match the length of the main-tower 'H' series."
        )
    return arr.astype(float)


def _lookup(data, names):
    """Return the first present, non-None value among ``names`` keys (else None)."""
    for name in names:
        value = data.get(name)
        if value is not None:
            return value
    return None


def _require_temperature(data, n, *, who):
    """Extract the air-temperature series [°C or K] required for the dT/dx term."""
    T = _lookup(data, ("T", "T_air", "temp"))
    if T is None:
        raise ValueError(
            "compute_advection_fluxes: horizontal heat advection (HA_T) needs "
            f"air temperature 'T' [°C or K] for the {who}, but it is missing. "
            "HA_T is a temperature-gradient term (rho*Cp*u*dT/dx*(zm-h)), not a "
            "difference of sensible-heat fluxes; provide main_data['T'] and "
            "upwind_data['T']."
        )
    return _broadcast_series(T, n, name=f"{who} 'T'")


def _require_specific_humidity(data, T_celsius, P, n, *, who):
    """Extract specific humidity [kg/kg] for the dq/dx term (RH is converted)."""
    q = _lookup(data, ("q",))
    if q is not None:
        return _broadcast_series(q, n, name=f"{who} 'q'")
    RH = _lookup(data, ("RH", "rh"))
    if RH is not None:
        RH = _broadcast_series(RH, n, name=f"{who} 'RH'")
        return np.array(
            [rh_to_specific_humidity(RH[i], T_celsius[i], P[i]) for i in range(n)]
        )
    raise ValueError(
        "compute_advection_fluxes: horizontal moisture advection (HA_Q) needs "
        f"specific humidity 'q' [kg/kg] or relative humidity 'RH' [%] for the "
        f"{who}, but neither is present. HA_Q is a moisture-gradient term "
        "(rho*lambda*u*dq/dx*(zm-h)); provide 'q' (or 'RH') on both towers."
    )


def _choose_upwind_index(main_data, upwind_list, i):
    """Pick the upwind tower best aligned with the wind direction at step ``i``.

    When the main tower reports ``wind_dir`` and the upwind towers carry a
    ``bearing`` (degrees from north, as seen from the main tower), the tower
    whose bearing is closest to the wind direction is selected so the gradient
    is taken along the actual fetch. With no directional information the first
    tower is used.
    """
    wind_dir = main_data.get("wind_dir")
    has_bearing = any(t.get("bearing") is not None for t in upwind_list)
    if wind_dir is not None and has_bearing and wind_dir[i] is not None:
        wd = wind_dir[i]
        best_diff = 361.0
        best_idx = 0
        for idx, tower in enumerate(upwind_list):
            bearing = tower.get("bearing")
            if bearing is None:
                continue
            diff = abs(((wd - bearing + 180) % 360) - 180)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx
        return best_idx
    return 0


def _trapz(y, x):
    """Trapezoidal integral of ``y`` over ``x`` (both 1-D, ``x`` ascending).

    A local implementation is used instead of ``np.trapz`` / ``np.trapezoid``
    so the result is independent of the NumPy version's spelling of that
    function (``np.trapz`` is deprecated in NumPy 2.x).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum((y[:-1] + y[1:]) / 2.0 * np.diff(x)))


def _require_w_bar(main_data, n):
    """Extract the **planar-fit** mean vertical velocity ``w_bar`` [m/s] for VAT.

    The vertical-advection term (Lee 1998; Wang Eq. 6) is driven by the
    *residual* mean vertical velocity that survives planar-fit / tilt
    correction. The **raw sonic ``w`` must never be used**: it is biased by
    sensor tilt and flow distortion, so an un-corrected ``w`` would manufacture
    a spurious vertical advection. Accepted keys: ``'w_bar'`` (preferred),
    ``'w_planar'``, ``'w_mean'``.

    Raises
    ------
    ValueError
        If no planar-fit ``w_bar`` is present. The function does **not** fall
        back to the energy-balance residual (``CLAUDE.md`` hard rule).
    """
    w = _lookup(main_data, ("w_bar", "w_planar", "w_mean"))
    if w is None:
        raise ValueError(
            "compute_advection_fluxes: vertical heat advection (VAT; Lee 1998, "
            "Wang Eq. 6) needs the PLANAR-FIT mean vertical velocity 'w_bar' "
            "[m/s] on main_data, but it is missing. Provide the planar-fit / "
            "tilt-corrected w_bar -- never the raw sonic w, which is tilt-biased. "
            "VAT is NOT a closure residual: the function RAISES rather than "
            "falling back to (H+LE)-(Rn-G)."
        )
    return _broadcast_series(w, n, name="main 'w_bar'")


def _require_column_mean_temperature(main_data, n):
    r"""Extract the column-mean temperature ``<T>`` [°C or K] for VAT.

    ``<T> = (1/zm) \int_0^{zm} T dz`` may be supplied two ways:

    - **Directly** via ``'T_col'`` (aliases ``'T_column_mean'``,
      ``'T_mean_col'``) -- a scalar (broadcast) or a length-``n`` series.
    - **Approximated from a profile** via ``'T_profile'`` (temperatures) at
      heights ``'z_profile'`` [m], integrated by the trapezoidal rule and
      divided by the profile depth. ``'T_profile'`` may be 1-D (a constant
      profile broadcast over time) or 2-D ``(n, levels)`` (per-timestep).

    Raises
    ------
    ValueError
        If neither a direct ``<T>`` nor a usable profile is present. The VAT
        term is **never** back-filled from the energy-balance residual.
    """
    T_col = _lookup(main_data, ("T_col", "T_column_mean", "T_mean_col"))
    if T_col is not None:
        return _broadcast_series(T_col, n, name="main 'T_col'")

    T_profile = _lookup(main_data, ("T_profile",))
    z_profile = _lookup(main_data, ("z_profile",))
    if T_profile is not None and z_profile is not None:
        z = np.asarray(z_profile, dtype=float)
        if z.ndim != 1 or z.shape[0] < 2:
            raise ValueError(
                "compute_advection_fluxes: 'z_profile' must be a 1-D array of at "
                "least two heights [m] to approximate <T> = (1/zm) integral T dz."
            )
        order = np.argsort(z)
        z_sorted = z[order]
        span = z_sorted[-1] - z_sorted[0]
        if span <= 0:
            raise ValueError(
                "compute_advection_fluxes: 'z_profile' heights must span a "
                "positive depth to integrate the column-mean temperature."
            )
        Tp = np.asarray(T_profile, dtype=float)
        if Tp.ndim == 1:
            if Tp.shape[0] != z_sorted.shape[0]:
                raise ValueError(
                    "compute_advection_fluxes: 1-D 'T_profile' length "
                    f"{Tp.shape[0]} must match 'z_profile' length "
                    f"{z_sorted.shape[0]}."
                )
            mean = _trapz(Tp[order], z_sorted) / span
            return np.full(n, mean)
        if Tp.ndim == 2:
            if Tp.shape != (n, z.shape[0]):
                raise ValueError(
                    "compute_advection_fluxes: 2-D 'T_profile' must have shape "
                    f"(n, levels) = ({n}, {z.shape[0]}); got {Tp.shape}."
                )
            return np.array([_trapz(Tp[i][order], z_sorted) / span for i in range(n)])
        raise ValueError(
            "compute_advection_fluxes: 'T_profile' must be 1-D (constant) or "
            "2-D (n, levels)."
        )

    raise ValueError(
        "compute_advection_fluxes: vertical heat advection (VAT) needs the "
        "column-mean temperature <T>: provide 'T_col' (or 'T_column_mean') "
        "[°C or K], or a vertical profile via 'T_profile' + 'z_profile' to "
        "approximate <T> = (1/zm) integral T dz. VAT is never computed from the "
        "energy-balance residual."
    )


def compute_advection_fluxes(
    main_data,
    upwind_data=None,
    detect_horizontal=None,
    detect_vertical=None,
    tower_distance=None,
):
    r"""
    Compute horizontal and vertical advection flux series for closure.

    The horizontal terms are the **physical gradient-based advection** of Wang
    et al. (2024), Eqs. 5a/5b (Moderow et al. 2021, Term IV) -- *not* a
    difference of two sensible-heat fluxes::

        HA_T = rho * Cp     * u_bar * (dT/dx) * (zm - h)     # W/m^2  (Eq. 5a)
        HA_Q = rho * lambda * u_bar * (dq/dx) * (zm - h)     # W/m^2  (Eq. 5b)

    where ``rho`` is moist-air density (:func:`air_density`), ``Cp`` the
    moist-air specific heat (:func:`specific_heat_moist_air`), ``lambda`` the
    latent heat of vaporization (:func:`latent_heat_vaporization`) -- all
    evaluated from the **main-tower** state ``(P, T, q)`` -- ``u_bar`` the mean
    horizontal wind speed [m/s], and ``(zm - h)`` the depth [m] of the air layer
    between the canopy/lower height ``h`` and the measurement height ``zm``.

    Vertical mean-advection term (replaces the old residual fallback)
    ----------------------------------------------------------------
    When the caller supplies the **planar-fit** mean vertical velocity
    ``w_bar`` (or a ``detect_vertical`` mask), the vertical heat advection is
    computed as the **measured** Lee (1998) / Wang et al. (2024) Eq. 6 term::

        VAT = rho * Cp * w_bar * (T_zm - <T>)                # W/m^2  (Eq. 6)

    where ``T_zm`` is the air temperature at the measurement height (the main
    tower ``'T'``) and ``<T> = (1/zm) integral T dz`` is the column-mean
    temperature (``'T_col'`` or a ``'T_profile'``/``'z_profile'`` pair). This is
    **not** a closure residual: ``w_bar`` **must** be the planar-fit /
    tilt-corrected mean vertical velocity, never the raw sonic ``w`` (which is
    tilt-biased). If vertical advection is requested but ``w_bar`` or the
    column-mean temperature is missing, the function **raises** rather than
    falling back to the energy-balance residual (``CLAUDE.md`` hard rule:
    *never compute an advection term as the energy-balance residual*).

    Optionally, when two-level kinematic heat fluxes ``'wT_zm'`` and ``'wT_h'``
    [K m/s] are provided, the vertical heat-flux divergence (Wang Eq. 12) is
    also returned::

        VFD_T = -rho * Cp * (wT|zm - wT|h)                   # W/m^2  (Eq. 12)

    Diagnostic residual (NOT an advective flux)
    -------------------------------------------
    The energy-balance closure imbalance ``residual = (H + LE) - (Rn - G)`` is
    still computed and returned, but **only as a closure diagnostic** -- it is
    *not* an advective flux and must never be added to the budget as one.

    Sign derivation (read carefully)
    --------------------------------
    Let the streamwise coordinate ``x`` increase **downwind**, i.e. point from
    the upwind tower toward the main tower. The along-wind temperature gradient
    is approximated by the two-tower finite difference::

        dT/dx ~= (T_main - T_upwind) / tower_distance        # K/m

    In the **oasis case** warm dry air sits upwind of a cool transpiring field,
    so ``T_upwind > T_main`` and therefore ``dT/dx < 0``. With ``u_bar > 0`` and
    ``(zm - h) > 0`` this makes ``HA_T < 0``. Under the Moderow et al. (2021)
    convention (**positive = energy OUT of the control volume**), a negative
    ``HA_T`` means sensible-heat energy is advected **INTO** the field -- exactly
    the expected oasis behaviour. Defining ``dT/dx`` as
    ``(T_main - T_upwind)/tower_distance`` therefore yields the correct
    into-field (negative) sign with **no extra negation**; warmer upwind air
    gives ``HA_T < 0``. The moisture gradient uses the identical downwind-positive
    convention, ``dq/dx = (q_main - q_upwind)/tower_distance``, so drier upwind
    air (``q_upwind < q_main``) gives ``dq/dx > 0`` and ``HA_Q > 0`` -- the
    drying signal of dry-air advection.

    For the **vertical** term the canonical oasis has a *stable* internal
    boundary layer (cool wet surface, warm air aloft), so the measurement-height
    temperature exceeds the column mean (``T_zm > <T>``, i.e. ``T_zm - <T> > 0``)
    while the planar-fit mean motion is subsidence (``w_bar < 0``). Then
    ``VAT = rho*Cp*w_bar*(T_zm - <T>) < 0``: warm air is carried **DOWN** into the
    control volume, i.e. heat **INTO** the field -- negative under the OUT-positive
    convention, the expected oasis vertical-advection sign.

    Finally the diagnostic ``residual`` is defined here as ``(H + LE) - (Rn - G)``
    -- the *negative* of the ``Rn - G - H - LE`` imbalance written in CLAUDE.md,
    chosen so a **positive** value flags net energy advected **IN** (the legacy
    ``adv_in`` reading). In the oasis the latent flux exceeds the available energy
    (``EF = LE/(Rn-G) > 1``, the advective-input fingerprint), so
    ``H + LE > Rn - G`` and ``residual > 0``. It is a closure *diagnostic*, never
    an advective flux.

    Required inputs (a missing one RAISES, never silently returns zero)
    ------------------------------------------------------------------
    Both ``main_data`` and each upwind tower must carry the fields the gradient
    terms need; ``tower_distance`` (or a per-tower ``'distance'``) is required.
    If any required field is missing a :class:`ValueError` is raised rather than
    producing a meaningless zero or a flux difference.

    Parameters
    ----------
    main_data : dict
        Main-tower series. Required keys:

        - ``'H'``  : sensible heat flux [W/m^2]
        - ``'LE'`` : latent heat flux [W/m^2]
        - ``'Rn'`` : net radiation [W/m^2]
        - ``'G'``  : ground heat flux [W/m^2]
        - ``'T'``  : air temperature [°C or K]
        - ``'q'`` (or ``'RH'`` [%]) : specific humidity [kg/kg]
        - ``'u'`` (or ``'wind_speed'``) : mean horizontal wind speed ``u_bar`` [m/s]
        - ``'zm'`` : measurement height [m]
        - ``'h'``  : canopy / lower height [m] (must satisfy ``zm > h``)

        Optional keys: ``'P'`` ambient pressure [Pa] (default 101325),
        ``'wind_dir'`` [deg from north] for multi-tower fetch selection.

        Optional **vertical-advection** keys (needed only when ``w_bar`` or
        ``detect_vertical`` is supplied; missing required ones then *raise*):

        - ``'w_bar'`` (aliases ``'w_planar'``, ``'w_mean'``) : **planar-fit /
          tilt-corrected** mean vertical velocity [m/s]. Never the raw sonic
          ``w``. Required to compute ``VAT``.
        - ``'T_col'`` (aliases ``'T_column_mean'``, ``'T_mean_col'``) :
          column-mean temperature ``<T>`` [°C or K]; **or** ``'T_profile'`` +
          ``'z_profile'`` (temperatures at heights [m]) from which ``<T>`` is
          approximated by trapezoidal integration. Required to compute ``VAT``.
        - ``'wT_zm'`` and ``'wT_h'`` : two-level kinematic heat fluxes [K m/s]
          at the measurement and canopy heights. When **both** are present the
          optional vertical heat-flux divergence ``VFD_T`` (Eq. 12) is returned.

        ``'T'``, ``'q'``/``'RH'``, ``'u'``, ``'zm'``, ``'h'``, ``'P'``,
        ``'w_bar'``, ``'T_col'``, ``'wT_zm'``, ``'wT_h'`` may each be a scalar
        (broadcast) or a length-``n`` series.
    upwind_data : dict or list of dicts
        Upwind tower(s). Each must carry ``'T'`` and ``'q'`` (or ``'RH'``) for
        the gradients; may carry ``'bearing'`` [deg from north] for fetch
        selection and a per-tower ``'distance'`` [m] overriding ``tower_distance``.
        **Required** -- horizontal advection cannot be computed without an
        upwind reference.
    detect_horizontal : np.ndarray, optional
        Boolean mask from :func:`detect_horizontal_advection`. Where ``False``,
        ``HA_T`` and ``HA_Q`` are set to 0 for that step (no advection event).
        Where ``None`` (default), the gradient terms are computed at every step.
    detect_vertical : np.ndarray, optional
        Boolean mask from :func:`detect_vertical_advection`. Supplying it
        **engages** the vertical term: ``VAT`` is then required (so ``w_bar``
        and the column-mean temperature must be present, else a
        :class:`ValueError` is raised), and where the mask is ``False`` ``VAT``
        is set to 0 for that step (no vertical-advection event).
    tower_distance : float, optional
        Separation [m] between the main and upwind tower, used as ``dx`` in the
        gradients. Required unless every upwind dict supplies its own
        ``'distance'``.

    Returns
    -------
    dict
        Keys:

        - ``'HA_T'``  : horizontal heat advection [W/m^2] (Eq. 5a). Negative =
          INTO the field (oasis, warm upwind air). length-``n`` array.
        - ``'HA_Q'``  : horizontal moisture (latent-energy) advection [W/m^2]
          (Eq. 5b). length-``n`` array.
        - ``'H_adv'`` : alias of ``'HA_T'`` [W/m^2], kept for backward
          compatibility.
        - ``'VAT'``   : **measured** vertical heat advection
          ``rho*Cp*w_bar*(T_zm - <T>)`` [W/m^2] (Lee 1998; Wang Eq. 6) as a
          length-``n`` array, or ``None`` when no vertical inputs were supplied
          (horizontal-only call). This is the proper mean-advection term, **not**
          a closure residual.
        - ``'V_adv'`` : backward-compatible alias of ``'VAT'``. (Previously a
          closure-residual estimate; it is now the measured Eq. 6 term.)
        - ``'VFD_T'`` : vertical heat-flux divergence ``-rho*Cp*(wT|zm - wT|h)``
          [W/m^2] (Wang Eq. 12) as a length-``n`` array when both ``'wT_zm'``
          and ``'wT_h'`` were supplied; otherwise ``None``.
        - ``'residual'`` : **diagnostic** energy-balance closure imbalance
          ``(H + LE) - (Rn - G)`` [W/m^2], length-``n`` array. This is a closure
          *diagnostic*, **NOT** an advective flux -- never add it to the budget
          as advection.
        - ``'adv_in'`` : deprecated alias of ``'residual'`` (the old,
          misleadingly named key). Identical values.

    Raises
    ------
    ValueError
        If ``upwind_data`` is omitted, if ``tower_distance`` (and a per-tower
        ``'distance'``) is missing, if any required gradient field (``'T'``,
        ``'q'``/``'RH'``, ``'u'``, ``'zm'``, ``'h'``) is absent, or if the layer
        depth ``zm - h`` is not strictly positive. Additionally, when the
        vertical term is engaged (``w_bar`` or ``detect_vertical`` supplied),
        if the planar-fit ``w_bar`` or the column-mean temperature
        (``T_col`` / ``T_profile``) is missing -- the vertical term is **never**
        back-filled from the energy-balance residual.

    References
    ----------
    Wang et al. (2024), Eqs. 5a, 5b, 6 and 12. Moderow et al. (2021), Term IV
    and the OUT-positive sign convention. Lee (1998), vertical mean-advection
    term.
    """
    # --- main-tower energy-balance series (for the 'residual' diagnostic) ----
    H_main = np.asarray(main_data["H"], dtype=float)
    LE_main = np.asarray(main_data["LE"], dtype=float)
    Rn_main = np.asarray(main_data["Rn"], dtype=float)
    G_main = np.asarray(main_data["G"], dtype=float)
    n = len(H_main)

    # --- the horizontal gradient terms require an upwind reference -----------
    if upwind_data is None:
        raise ValueError(
            "compute_advection_fluxes: horizontal advection (HA_T/HA_Q) needs an "
            "upwind tower to form the dT/dx and dq/dx gradients, but upwind_data "
            "is None. Pass upwind_data (a dict or list of dicts) carrying 'T' and "
            "'q'/'RH'."
        )
    upwind_list = upwind_data if isinstance(upwind_data, list) else [upwind_data]
    if not upwind_list:
        raise ValueError(
            "compute_advection_fluxes: upwind_data is an empty list; at least one "
            "upwind tower (with 'T' and 'q'/'RH') is required."
        )

    # --- main-tower state for the gradient coefficients and humidity ---------
    P_value = _lookup(main_data, ("P",))
    P_main = _broadcast_series(
        P_value if P_value is not None else 101325.0, n, name="main 'P'"
    )
    T_main = _require_temperature(main_data, n, who="main tower")
    T_main_C = np.where(T_main > 150.0, T_main - 273.15, T_main)
    q_main = _require_specific_humidity(
        main_data, T_main_C, P_main, n, who="main tower"
    )

    u_value = _lookup(main_data, ("u", "wind_speed", "U"))
    if u_value is None:
        raise ValueError(
            "compute_advection_fluxes: horizontal advection needs the mean "
            "horizontal wind speed 'u' (alias 'wind_speed') [m/s] on main_data, "
            "but it is missing."
        )
    u_bar = _broadcast_series(u_value, n, name="main 'u'")

    zm_value = _lookup(main_data, ("zm",))
    h_value = _lookup(main_data, ("h",))
    if zm_value is None or h_value is None:
        raise ValueError(
            "compute_advection_fluxes: horizontal advection needs the layer depth "
            "(zm - h): provide both measurement height 'zm' [m] and canopy/lower "
            "height 'h' [m] on main_data."
        )
    zm = _broadcast_series(zm_value, n, name="main 'zm'")
    h = _broadcast_series(h_value, n, name="main 'h'")
    layer = zm - h
    if np.any(layer <= 0):
        raise ValueError(
            "compute_advection_fluxes: layer depth (zm - h) must be strictly "
            "positive (measurement height above canopy); got zm - h <= 0. Check "
            "that 'zm' and 'h' are not swapped."
        )

    # State-dependent coefficients, evaluated at the main tower (the control
    # volume). air_density / latent_heat_vaporization accept °C or K internally.
    rho = np.array([air_density(P_main[i], T_main[i], q_main[i]) for i in range(n)])
    Cp = np.array([specific_heat_moist_air(q_main[i]) for i in range(n)])
    Lv = np.array([latent_heat_vaporization(T_main[i]) for i in range(n)])

    # --- per-upwind temperature, humidity and gradient distance --------------
    upwind_T = []
    upwind_q = []
    upwind_dist = []
    for k, tower in enumerate(upwind_list):
        Tk = _require_temperature(tower, n, who=f"upwind tower {k}")
        Tk_C = np.where(Tk > 150.0, Tk - 273.15, Tk)
        qk = _require_specific_humidity(tower, Tk_C, P_main, n, who=f"upwind tower {k}")
        dist = tower.get("distance", tower_distance)
        if dist is None:
            raise ValueError(
                "compute_advection_fluxes: tower_distance [m] (or a per-tower "
                f"'distance') is required for the dT/dx gradient; upwind tower {k} "
                "has neither."
            )
        if dist <= 0:
            raise ValueError(
                f"compute_advection_fluxes: tower separation for upwind tower {k} "
                f"must be positive, got {dist}."
            )
        upwind_T.append(Tk)
        upwind_q.append(qk)
        upwind_dist.append(float(dist))

    # --- horizontal gradient terms (Wang Eqs. 5a/5b) -------------------------
    HA_T = np.zeros(n)
    HA_Q = np.zeros(n)
    for i in range(n):
        if detect_horizontal is not None and not detect_horizontal[i]:
            # Detection says no advection event here -> zero (not a missing-data
            # zero, which would have raised above).
            continue
        k = _choose_upwind_index(main_data, upwind_list, i)
        dx = upwind_dist[k]
        # Gradients along the downwind x-axis (upwind -> main). Differencing in
        # Kelvin makes the result robust to mixed °C/K inputs (a K difference
        # equals a °C difference); see the sign derivation above.
        dT_dx = (_to_kelvin(T_main[i]) - _to_kelvin(upwind_T[k][i])) / dx
        dq_dx = (q_main[i] - upwind_q[k][i]) / dx
        HA_T[i] = rho[i] * Cp[i] * u_bar[i] * dT_dx * layer[i]
        HA_Q[i] = rho[i] * Lv[i] * u_bar[i] * dq_dx * layer[i]

    # --- diagnostic energy-balance residual (NOT an advective flux) ----------
    # The closure imbalance (H + LE) - (Rn - G) is a diagnostic only; it must
    # never be relabelled as advection (CLAUDE.md hard rule). Kept and returned
    # under the unambiguous name 'residual' (with 'adv_in' as a deprecated
    # alias for backward compatibility).
    #
    # SIGN: this is the *negative* of CLAUDE.md's "Residual = Rn - G - H - LE".
    # The sign here is deliberate and matches the legacy 'adv_in' reading: a
    # POSITIVE value means the turbulent fluxes exceed the available energy
    # (H + LE > Rn - G), i.e. energy was advected IN. The oasis fingerprint is
    # EF = LE/(Rn - G) > 1 (latent flux above available energy), which drives
    # H + LE above Rn - G and therefore makes this residual POSITIVE. It is a
    # diagnostic of advective input, not a flux subject to the OUT/IN sign rule.
    residual = (H_main + LE_main) - (Rn_main - G_main)

    # --- vertical mean heat advection VAT (Lee 1998; Wang Eq. 6) -------------
    # Engaged only when the caller supplies the planar-fit w_bar or a
    # detect_vertical mask. VAT is the MEASURED term rho*Cp*w_bar*(T_zm - <T>),
    # never the energy-balance residual: if w_bar or the column-mean
    # temperature is missing the helpers RAISE rather than fall back.
    wants_vertical = (
        _lookup(main_data, ("w_bar", "w_planar", "w_mean")) is not None
        or detect_vertical is not None
    )
    if wants_vertical:
        w_bar = _require_w_bar(main_data, n)
        T_col = _require_column_mean_temperature(main_data, n)
        VAT = np.zeros(n)
        for i in range(n):
            if detect_vertical is not None and not detect_vertical[i]:
                # No vertical-advection event at this step.
                continue
            # T_zm is the main-tower air temperature at the measurement height;
            # difference against the column mean <T> in Kelvin (a K difference
            # equals a °C difference, so mixed °C/K inputs are handled).
            #
            # SIGN (Moderow OUT-positive): VAT = rho*Cp*w_bar*(T_zm - <T>). In the
            # oasis the internal boundary layer is stable (cool wet surface, warm
            # air aloft) so T_zm > <T> (dT > 0), and the planar-fit mean motion is
            # subsidence (w_bar < 0); the product is therefore NEGATIVE -- warm air
            # carried DOWN, heat INTO the field. No extra negation is applied: the
            # OUT-positive sign falls straight out of the signed w_bar and dT.
            dT = _to_kelvin(T_main[i]) - _to_kelvin(T_col[i])
            VAT[i] = rho[i] * Cp[i] * w_bar[i] * dT
    else:
        VAT = None

    # --- optional vertical heat-flux divergence VFD_T (Wang Eq. 12) ----------
    # Computed only when both two-level kinematic heat fluxes are provided.
    wT_zm = _lookup(main_data, ("wT_zm", "wT_top", "wT_upper"))
    wT_h = _lookup(main_data, ("wT_h", "wT_bottom", "wT_lower"))
    if wT_zm is not None and wT_h is not None:
        wT_zm = _broadcast_series(wT_zm, n, name="main 'wT_zm'")
        wT_h = _broadcast_series(wT_h, n, name="main 'wT_h'")
        VFD_T = -rho * Cp * (wT_zm - wT_h)
    else:
        VFD_T = None

    return {
        "HA_T": HA_T,
        "HA_Q": HA_Q,
        "H_adv": HA_T,
        "VAT": VAT,
        "V_adv": VAT,  # backward-compatible alias (now the measured Eq. 6 term)
        "VFD_T": VFD_T,
        "residual": residual,
        "adv_in": residual,  # deprecated alias of 'residual'
    }


def apply_advection_correction(main_data, H_adv, V_adv, HA_Q=None, rn_min=75.0):
    r"""
    Fold measured advective fluxes into the surface energy balance.

    Energy-balance bookkeeping (Moderow et al. 2021; Wang et al. 2024)
    -----------------------------------------------------------------
    The advection-augmented surface energy balance, written in the Moderow et al.
    (2021) **OUT-positive** convention (positive flux = energy *out* of the
    control volume), places the advective terms on the **turbulent-sum side**
    alongside ``H`` and ``LE`` -- the available-energy side ``(Rn - G)`` is left
    untouched::

        Rn - G = H + LE + HA_T + HA_Q + VAT

    so the corrected turbulent + advective sum is::

        (H + LE)_corrected = H + LE + HA_T + HA_Q + VAT

    Each advective term carries its own sign from the upstream computation
    (:func:`compute_advection_fluxes`): a **negative** term is energy advected
    **INTO** the field (the oasis fingerprint), a **positive** term is energy
    advected **OUT**. This function neither re-signs nor re-derives the terms --
    it simply adds them on the turbulent side. The residual is reported in the
    ``CLAUDE.md`` convention ``Residual = Rn - G - H - LE`` (positive = available
    energy exceeds the turbulent sum), so the correction moves a gated step's
    residual toward zero.

    Conditional inclusion gate (Wang et al. 2024)
    ---------------------------------------------
    The advective fluxes are folded in **only** at timesteps where **both**
    Wang's conditions hold (this is what lifted closure from 89 % to 97 % in the
    alfalfa study):

    1. ``Rn > rn_min`` (default 75 W/m^2) -- sufficient daytime radiative
       forcing, **AND**
    2. ``(H + LE) < (Rn - G)`` -- the (spectrally-corrected) turbulent sum is
       *below* the available energy, i.e. there is an under-closure gap for the
       advected energy to fill.

    Where the gate fails (night, low ``Rn``, or already-over-closed steps) the
    timestep is left **exactly uncorrected**: ``(H + LE)_corrected == H + LE``.
    The boolean ``'included'`` mask records which timesteps were corrected.

    ``NaN`` handling
    ----------------
    A ``NaN`` in any of ``Rn``/``G``/``H``/``LE`` makes the gate comparisons
    ``False`` for that step, so it is left uncorrected (never silently forced
    closed). A ``NaN`` in an advective term contributes **0** to the corrected
    sum (an unmeasured advective component adds nothing) rather than poisoning
    the whole corrected sum with ``NaN``.

    Parameters
    ----------
    main_data : dict
        Main-tower energy balance. Required keys ``'H'``, ``'LE'``, ``'Rn'``,
        ``'G'`` [all W/m^2], each a length-``n`` series.
    H_adv : array-like
        Horizontal **heat** advection ``HA_T`` [W/m^2] (Wang Eq. 5a; the
        ``'H_adv'`` / ``'HA_T'`` output of :func:`compute_advection_fluxes`).
    V_adv : array-like or None
        Vertical **heat** advection ``VAT`` [W/m^2] (Wang Eq. 6; the ``'V_adv'``
        / ``'VAT'`` output of :func:`compute_advection_fluxes`). ``None`` (e.g. a
        horizontal-only call) is treated as all-zero (no vertical advection).
    HA_Q : array-like or None, optional
        Horizontal **moisture** advection ``HA_Q`` [W/m^2] (Wang Eq. 5b). ``None``
        (default) is treated as all-zero, preserving the legacy three-argument
        call signature.
    rn_min : float, optional
        Net-radiation threshold [W/m^2] for Wang's gate condition (1). Default
        75 W/m^2.

    Returns
    -------
    dict
        Keys:

        - ``'Rn'``, ``'G'``, ``'H'``, ``'LE'`` : the input series (float arrays).
        - ``'HA_T'`` / ``'H_adv'`` : horizontal heat advection [W/m^2].
        - ``'HA_Q'`` : horizontal moisture advection [W/m^2].
        - ``'VAT'`` / ``'V_adv'`` : vertical heat advection [W/m^2].
        - ``'H_plus_LE_orig'`` : uncorrected turbulent sum ``H + LE`` [W/m^2].
        - ``'H_plus_LE_corrected'`` : turbulent sum with the gated advective
          terms folded in [W/m^2].
        - ``'available_energy'`` : ``Rn - G`` [W/m^2] (unchanged by the
          correction; the target the turbulent sum is compared against).
        - ``'residual_orig'`` : ``Rn - G - H - LE`` [W/m^2] before correction.
        - ``'residual_corrected'`` : ``Rn - G - (H + LE)_corrected`` [W/m^2]
          after correction.
        - ``'included'`` : boolean mask, ``True`` where Wang's gate passed and
          the advective terms were applied.

    References
    ----------
    Moderow et al. (2021), OUT-positive sign convention and advection terms.
    Wang et al. (2024), conditional-inclusion rule and Eqs. 5a/5b/6.
    """
    H_main = np.asarray(main_data["H"], dtype=float)
    LE_main = np.asarray(main_data["LE"], dtype=float)
    Rn_main = np.asarray(main_data["Rn"], dtype=float)
    G_main = np.asarray(main_data["G"], dtype=float)
    n = len(H_main)

    # Advective terms: None -> all-zero (term not supplied for this call);
    # scalars are broadcast to length n by the helper.
    def _adv_series(value):
        arr = _as_float_series(value, n)
        return np.zeros(n) if arr is None else arr

    HA_T = _adv_series(H_adv)
    HA_Q_series = _adv_series(HA_Q)
    VAT = _adv_series(V_adv)

    available_energy = Rn_main - G_main
    H_plus_LE_orig = H_main + LE_main

    # --- Wang (2024) conditional-inclusion gate ------------------------------
    # Include advection only where BOTH (1) Rn > rn_min and (2) the turbulent
    # sum is below the available energy. NaN comparisons evaluate False, so any
    # missing budget component leaves the step uncorrected.
    with np.errstate(invalid="ignore"):
        included = (Rn_main > rn_min) & (H_plus_LE_orig < available_energy)
    included = np.asarray(included, dtype=bool)

    # --- fold the advective terms onto the turbulent-sum side ----------------
    # NaN advective components contribute 0 (an unmeasured term adds nothing),
    # so a single missing term cannot poison the whole corrected sum.
    adv_total = (
        np.nan_to_num(HA_T, nan=0.0)
        + np.nan_to_num(HA_Q_series, nan=0.0)
        + np.nan_to_num(VAT, nan=0.0)
    )
    applied = np.where(included, adv_total, 0.0)
    H_plus_LE_corrected = H_plus_LE_orig + applied

    # Residuals in the CLAUDE.md convention: Residual = Rn - G - H - LE.
    residual_orig = available_energy - H_plus_LE_orig
    residual_corrected = available_energy - H_plus_LE_corrected

    return {
        "Rn": Rn_main,
        "G": G_main,
        "H": H_main,
        "LE": LE_main,
        "HA_T": HA_T,
        "H_adv": HA_T,  # backward-compatible alias of HA_T
        "HA_Q": HA_Q_series,
        "VAT": VAT,
        "V_adv": VAT,  # backward-compatible alias of VAT
        "H_plus_LE_orig": H_plus_LE_orig,
        "H_plus_LE_corrected": H_plus_LE_corrected,
        "available_energy": available_energy,
        "residual_orig": residual_orig,
        "residual_corrected": residual_corrected,
        "included": included,
    }
