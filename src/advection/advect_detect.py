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


def detect_horizontal_advection(
    main_flux,
    upwind_flux=None,
    wind_dir=None,
    upwind_dir=None,
    le_main=None,
    rn=None,
    g=None,
    temp_main=None,
    temp_upwind=None,
    humidity_main=None,
    humidity_upwind=None,
    wind_speed=None,
):
    """
    Detect periods of significant horizontal advection influencing the main tower.

    Parameters
    ----------
    main_flux : array-like
        Time series of sensible heat flux (H) at the main tower (W/m^2).
    upwind_flux : array-like, optional
        Time series of H at an upwind reference tower. Required for direct flux divergence detection.
    wind_dir : array-like, optional
        Time series of wind direction at the main tower (degrees from north).
    upwind_dir : float, optional
        The bearing (direction from main tower) toward the upwind reference tower (degrees from north).
        If provided, horizontal advection is only considered when wind_dir is within ±45° of upwind_dir.
    le_main : array-like, optional
        Time series of latent heat flux (LE) at main tower (W/m^2). Used to check LE/(Rn-G) ratio.
    rn : array-like, optional
        Time series of net radiation (R_n) at main site (W/m^2).
    g : array-like, optional
        Time series of soil heat flux (G) at main site (W/m^2).
    temp_main : array-like, optional
        Air temperature at the main tower (°C or K).
    temp_upwind : array-like, optional
        Air temperature at upwind tower (same units as temp_main).
    humidity_main : array-like, optional
        Air humidity (e.g. specific humidity or RH) at main tower.
    humidity_upwind : array-like, optional
        Air humidity at upwind tower.
    wind_speed : array-like, optional
        Wind speed at the main tower (m/s).

    Returns
    -------
    np.ndarray
        Boolean mask array where True indicates detected horizontal advection events.
    """
    main_flux = np.array(main_flux)
    n = len(main_flux)
    adv_flag = np.zeros(n, dtype=bool)
    # Compute available energy if possible
    avail_energy = None
    if rn is not None and g is not None:
        rn = np.array(rn)
        g = np.array(g)
        avail_energy = rn - g  # Rn - G
    # Loop through each time step (vectorized operations could be used as well)
    for i in range(n):
        # Check wind direction alignment for upwind tower if specified
        if upwind_dir is not None and wind_dir is not None:
            if wind_dir[i] is None:
                continue  # skip if no wind data
            # Calculate angular difference (taking care of circular wrap-around)
            ang_diff = None
            try:
                ang_diff = abs(((wind_dir[i] - upwind_dir + 180) % 360) - 180)
            except:
                ang_diff = abs(wind_dir[i] - upwind_dir)
            if ang_diff > 45:
                # Wind not coming from the direction of the reference tower, so skip marking adv from that tower
                continue
        # Criteria 1: If upwind flux is provided and significantly greater than main flux
        if upwind_flux is not None:
            upwind_val = upwind_flux[i]
            main_val = main_flux[i]
            if upwind_val is not None and main_val is not None:
                # Mark if upwind H >> main H (e.g. 20% higher or more) indicating extra heat available upwind
                if (
                    upwind_val > main_val + 20
                ):  # threshold = 20 W/m^2 difference (can be tuned)
                    adv_flag[i] = True
                # Also mark if upwind and main H have opposite signs (e.g. upwind positive, main negative)
                if main_val < 0 < upwind_val:
                    adv_flag[i] = True
        # Criteria 2: If available energy and LE are given, and LE exceeds available energy (LE/(Rn-G) > 1)
        if avail_energy is not None and le_main is not None:
            # Use a tolerance to account for measurement uncertainty
            if avail_energy[i] is not None and le_main[i] is not None:
                if (
                    le_main[i] > avail_energy[i] * 1.05
                ):  # LE is 5% greater than Rn-G (beyond typical error)
                    adv_flag[i] = True
        # Criteria 3: Main H is negative (downward) during daytime (suggesting advected warm air causing downward heat flux)
        # We consider daytime if Rn > 50 W/m^2 or so.
        if rn is not None:
            if rn[i] is not None and rn[i] > 50:
                if main_flux[i] is not None and main_flux[i] < 0:
                    adv_flag[i] = True
        # Criteria 4: Temperature/humidity differences indicating horizontal gradients
        if temp_main is not None and temp_upwind is not None:
            if temp_main[i] is not None and temp_upwind[i] is not None:
                # If upwind is significantly warmer than main (e.g. >1°C), indicates potential warm advection
                if temp_upwind[i] > temp_main[i] + 1.0:
                    adv_flag[i] = True
        if humidity_main is not None and humidity_upwind is not None:
            if humidity_main[i] is not None and humidity_upwind[i] is not None:
                # If upwind is much drier (e.g. lower specific humidity by >1 g/kg or 0.001 in kg/kg)
                if (humidity_main[i] - humidity_upwind[i]) > 0.001:
                    # Main is moister than upwind -> dry air advection likely
                    adv_flag[i] = True
        # (Optional spectral criterion could be implemented here: e.g., check if low-frequency variance is high during this period)
    return adv_flag


def detect_vertical_advection(
    temp_profile_lower=None,
    temp_profile_upper=None,
    vertical_w=None,
    main_H=None,
    rn=None,
    g=None,
):
    """
    Detect periods of vertical advection (vertical flux divergence) affecting the energy balance.

    Parameters
    ----------
    temp_profile_lower : array-like, optional
        Temperature near the surface or canopy (°C or K).
    temp_profile_upper : array-like, optional
        Temperature at the measurement height or above (°C or K).
    vertical_w : array-like, optional
        Mean vertical wind speed (m/s) at the site (if available; usually small).
    main_H : array-like, optional
        Time series of sensible heat flux at the main tower (W/m^2).
    rn : array-like, optional
        Net radiation (W/m^2) for context (to distinguish daytime).
    g : array-like, optional
        Soil heat flux (W/m^2) for context.

    Returns
    -------
    np.ndarray
        Boolean mask of detected vertical advection periods.
    """
    n = 0
    if temp_profile_lower is not None:
        n = len(temp_profile_lower)
    elif main_H is not None:
        n = len(main_H)
    vert_flag = np.zeros(n, dtype=bool)
    for i in range(n):
        # Daytime check
        if rn is not None and g is not None:
            if rn[i] is None or g[i] is None:
                continue
            if rn[i] - g[i] < 50:
                continue  # skip nighttime or low-energy periods
        # Check for inverted temperature profile (surface cooler than air above)
        inv_profile = False
        if temp_profile_lower is not None and temp_profile_upper is not None:
            if temp_profile_lower[i] is not None and temp_profile_upper[i] is not None:
                if (
                    temp_profile_lower[i] < temp_profile_upper[i] - 0.5
                ):  # >0.5°C inversion
                    inv_profile = True
        # Check for mean subsidence or upward transport
        vertical_motion = False
        if vertical_w is not None:
            if vertical_w[i] is not None:
                # If there's a consistent downward mean wind (negative w) or upward (positive w) outside a small range
                if vertical_w[i] < -0.05 or vertical_w[i] > 0.05:
                    vertical_motion = True
        # Check for unusual H (e.g. H near zero or negative when it normally would be positive)
        H_anomaly = False
        if main_H is not None:
            if main_H[i] is not None and rn is not None and rn[i] is not None:
                if rn[i] > 50:  # daytime
                    if (
                        main_H[i] < 20
                    ):  # very low or downward sensible heat during daytime
                        H_anomaly = True
        # Decide vertical advection flag:
        # We require an inverted profile plus either some vertical motion or an H anomaly as evidence
        if inv_profile and (vertical_motion or H_anomaly):
            vert_flag[i] = True
    return vert_flag


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


def apply_advection_correction(main_data, H_adv, V_adv):
    """
    Apply advection correction to the main tower energy balance components.

    This function adds the advection terms to the energy balance and returns an updated dataset.

    Parameters
    ----------
    main_data : dict
        Dictionary of main tower data (must contain 'H', 'LE', 'Rn', 'G').
    H_adv : array-like
        Horizontal advection flux time series (W/m^2).
    V_adv : array-like
        Vertical advection flux time series (W/m^2).

    Returns
    -------
    dict
        Corrected energy balance components, including:
        'Rn', 'G', 'H', 'LE', 'H_adv', 'V_adv', 'H_plus_LE_orig', 'H_plus_LE_corrected'.
    """
    H_main = np.array(main_data["H"])
    LE_main = np.array(main_data["LE"])
    Rn_main = np.array(main_data["Rn"])
    G_main = np.array(main_data["G"])
    H_adv = np.array(H_adv)
    V_adv = np.array(V_adv)
    # Original and corrected turbulent flux sums
    H_plus_LE = H_main + LE_main
    H_plus_LE_corrected = (
        H_main + LE_main
    )  # (We'll use separate terms rather than adjusting H or LE)
    # In the "corrected" sense, H+LE_corrected is conceptually H+LE plus any included adv fluxes (though we keep them separate)
    # We output H_adv and V_adv separately rather than lumping into H or LE.
    return {
        "Rn": Rn_main,
        "G": G_main,
        "H": H_main,
        "LE": LE_main,
        "H_adv": H_adv,
        "V_adv": V_adv,
        "H_plus_LE_orig": H_plus_LE,
        "H_plus_LE_corrected": H_plus_LE_corrected,  # (same numeric values as original; adv terms separate)
    }
