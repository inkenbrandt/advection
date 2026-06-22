"""Main module."""

import warnings

import numpy as np

# Shared fixed constants live in a single module so both ``advection`` and
# ``advect_detect`` import them rather than redefining literals. They back the
# physical advection/lapse-rate terms and are imported here for that use.
from ._constants import G_OVER_CP, MU, VON_KARMAN  # noqa: F401


def _to_kelvin(T):
    """Return an absolute temperature in Kelvin.

    A value greater than 150 is assumed to already be in Kelvin and is returned
    unchanged; otherwise it is treated as Celsius and 273.15 is added. The 150
    threshold sits far above any plausible near-surface Celsius air temperature
    and far below any plausible Kelvin value, so the heuristic is unambiguous
    for meteorological data. Use this wherever an *absolute* temperature is
    required so callers may pass either unit.

    Parameters
    ----------
    T : float
        Temperature in degrees Celsius or Kelvin.

    Returns
    -------
    float
        Temperature in Kelvin.
    """
    return T if T > 150 else T + 273.15


def compute_soil_heat_storage_flux(Csoil, dT_dt, dz=0.02):
    """
    Compute soil heat storage flux Gs (W/m^2) from soil volumetric heat capacity and temperature change rate.

    Implements Equation 1a: ``Gs = Cs * dz * (dTsoil/dt)``.

    Parameters
    ----------
    Csoil : float
        Volumetric heat capacity of the soil layer [J/(m^3 K)].
    dT_dt : float
        Time derivative of soil temperature [K/s] (temperature change rate),
        averaged over the storage layer.
    dz : float, optional
        **Thickness of the soil layer ABOVE the heat-flux plate** [m] — i.e. the
        depth interval ``dz`` between the surface and the plate over which heat is
        stored, *not* an absolute installation depth. Heat-flux plates are
        typically buried at 0.05-0.08 m, so the storage layer thickness usually
        lies in that range; the 0.02 m default is a placeholder, not a universal
        value, and should be set to the actual plate-burial depth at your site.

    Returns
    -------
    float
        Soil heat storage flux Gs [W/m^2].
    """
    return Csoil * dz * dT_dt


def total_ground_heat_flux(Gd, Gs):
    """
    Compute the storage-corrected ground heat flux G by adding raw ground flux Gd and the storage term Gs.

    Implements Equation 1b: G = Gd + Gs.

    Parameters
    ----------
    Gd : float
        Ground heat flux at the sensor depth (plate measurement) [W/m^2].
    Gs : float
        Soil heat storage flux (above the sensor) [W/m^2].

    Returns
    -------
    float
        Storage-corrected ground heat flux G [W/m^2].
    """
    return Gd + Gs


def air_heat_storage(rho, Cp, dT_dt, zm, h):
    """
    Compute the air-column sensible-heat storage term J (W/m^2).

    Implements Wang et al. (2024), Eq. 11::

        T_storage = rho * Cp * (dT/dt) * (zm - h)

    This is the rate at which sensible heat is stored in (positive) or released
    from (negative) the air layer between the canopy top ``h`` and the
    measurement height ``zm``. It is the storage term ``J`` in the
    storage-resolved surface energy balance ``Rn - G - J = H + LE`` and is
    **distinct from the soil heat storage** ``Gs`` (see
    :func:`compute_soil_heat_storage_flux`), which is folded into ``G``.

    Parameters
    ----------
    rho : float
        Air density [kg/m^3] during the period.
    Cp : float
        Specific heat capacity of air [J/(kg K)] (use the moist-air value if
        available; see :func:`specific_heat_moist_air`).
    dT_dt : float
        Time derivative of air temperature [K/s], representative of the
        ``h``-to-``zm`` layer.
    zm : float
        Measurement (sensor) height [m].
    h : float
        Canopy height [m].

    Returns
    -------
    float
        Air-column sensible-heat storage J [W/m^2].
    """
    return rho * Cp * dT_dt * (zm - h)


def compute_bowen_ratio_variance(
    sigma_T=None,
    sigma_q=None,
    Cp=1005.0,
    Lv=None,
    *,
    T_prime=None,
    q_prime=None,
    cov_Tq=None,
    corr_Tq=None,
    T=None,
    zero_cov_tol=1e-12,
):
    r"""
    Compute the variance (flux-variance) Bowen ratio ``beta`` **with sign**.

    Implements Wang et al. (2024), Eq. 8::

        |beta|     = (Cp / Lv) * (sigma_T / sigma_q)
        sign(beta) = sign(corr(T', q')) = sign(cov(T', q'))
        beta       = sign(beta) * |beta|

    The variance method recovers only the **magnitude** of the Bowen ratio from
    the ratio of the temperature and humidity standard deviations; the sign must
    come from the temperature-humidity correlation (Wang Eq. 8). A **negative
    beta is the oasis fingerprint**: warm, dry air (positive ``T'``) arrives as
    the surface evaporates and moistens the air (negative ``q'``), so
    ``corr(T', q') < 0``. Per the Moderow et al. (2021) sign convention
    (``CLAUDE.md``), a negative beta corresponds to **downward (negative)
    sensible heat flux H**, i.e. sensible heat advected *into* the field (energy
    INTO the control volume). The classic daytime convective case has warm
    rising air carrying moisture upward, giving ``corr(T', q') > 0`` and
    ``beta > 0``.

    The sign source may be supplied three ways, in order of precedence: a
    precomputed correlation ``corr_Tq``; a precomputed covariance ``cov_Tq``; or
    the raw fluctuation series ``T_prime``/``q_prime`` (from which the covariance
    -- and, when not given explicitly, the standard deviations -- are derived).
    When the sign source lies within ``zero_cov_tol`` of zero the sign is taken
    as **+1** (documented tie-break), since a vanishing T-q correlation carries
    no advective signal.

    Parameters
    ----------
    sigma_T : float, optional
        Standard deviation of air temperature fluctuations [K]. If omitted it is
        computed from ``T_prime``.
    sigma_q : float, optional
        Standard deviation of specific humidity fluctuations [kg/kg]. If omitted
        it is computed from ``q_prime``.
    Cp : float, optional
        Specific heat capacity of air [J/(kg K)] (use the moist-air value if
        available; default 1005 J/(kg K) for dry air).
    Lv : float, optional
        Latent heat of vaporization [J/kg]. If ``None`` and a temperature ``T``
        is supplied, ``latent_heat_vaporization(T)`` is used; otherwise it falls
        back to ~2.45e6 J/kg (~20 °C).
    T_prime, q_prime : array_like, optional
        Time series of temperature [K] and specific-humidity [kg/kg]
        fluctuations. Used to derive ``cov(T', q')`` (for the sign) and, when not
        supplied explicitly, ``sigma_T``/``sigma_q``.
    cov_Tq : float, optional
        Precomputed covariance of ``T'`` and ``q'``. Only its sign is used.
    corr_Tq : float, optional
        Precomputed correlation of ``T'`` and ``q'``. Only its sign is used.
        Takes precedence over ``cov_Tq`` and the fluctuation series.
    T : float, optional
        Air temperature [°C or K] used to evaluate ``Lv`` when ``Lv is None``.
    zero_cov_tol : float, optional
        Magnitude at or below which the sign source is treated as zero and the
        sign is set to +1 (default 1e-12).

    Returns
    -------
    float
        Signed Bowen ratio ``beta`` (dimensionless). A negative value indicates
        downward H / advective input (the oasis fingerprint).

    Warns
    -----
    UserWarning
        When no sign source (``corr_Tq``, ``cov_Tq`` or ``T_prime``/``q_prime``)
        is supplied: the **unsigned magnitude** is returned and the sign is
        flagged as undetermined (backward-compatible behaviour).

    References
    ----------
    Wang et al. (2024), Eq. 8. Moderow et al. (2021), sign convention.
    """
    # Derive the standard deviations from the fluctuation series when the scalar
    # values are not supplied directly.
    if sigma_T is None:
        if T_prime is None:
            raise ValueError("Provide either sigma_T or T_prime.")
        sigma_T = float(np.std(np.asarray(T_prime, dtype=float)))
    if sigma_q is None:
        if q_prime is None:
            raise ValueError("Provide either sigma_q or q_prime.")
        sigma_q = float(np.std(np.asarray(q_prime, dtype=float)))

    # Latent heat: prefer the temperature-dependent value when T is available,
    # falling back to the ~20 °C constant only when no temperature is known.
    if Lv is None:
        Lv = latent_heat_vaporization(T) if T is not None else 2.45e6

    # Eq. 8 magnitude. Standard deviations are non-negative by definition; take
    # the absolute value so the sign comes solely from the T-q correlation.
    magnitude = (Cp / Lv) * abs(sigma_T / sigma_q)

    # Resolve the sign from the T'-q' correlation/covariance (Wang Eq. 8).
    if corr_Tq is not None:
        sign_source = float(corr_Tq)
    elif cov_Tq is not None:
        sign_source = float(cov_Tq)
    elif T_prime is not None and q_prime is not None:
        tp = np.asarray(T_prime, dtype=float)
        qp = np.asarray(q_prime, dtype=float)
        sign_source = float(np.mean((tp - tp.mean()) * (qp - qp.mean())))
    else:
        warnings.warn(
            "compute_bowen_ratio_variance: no T'-q' correlation/covariance or "
            "fluctuation series supplied; returning the UNSIGNED Bowen-ratio "
            "magnitude. The sign -- and thus the oasis/advection fingerprint of "
            "a negative beta -- is undetermined.",
            UserWarning,
            stacklevel=2,
        )
        return magnitude

    if sign_source > zero_cov_tol:
        sign = 1.0
    elif sign_source < -zero_cov_tol:
        sign = -1.0
    else:
        # Near-zero correlation: documented +1 tie-break (no advective signal).
        sign = 1.0

    return sign * magnitude


def correct_sonic_heat_flux(w_Ts, T_mean, beta, Cp=1005.0, Lv=None, singular_tol=1e-6):
    r"""
    Convert sonic (virtual) temperature flux w'Ts' to the true kinematic
    sensible-heat flux w'T' by removing the humidity contribution.

    Implements Wang et al. (2024), Eq. 9::

        w'T' = w'Ts' / (1 + 0.51 * Cp * T_mean / (Lv * beta))    # T_mean in KELVIN

    where ``T_mean`` **must** be an absolute temperature in Kelvin and ``beta``
    is the signed Bowen ratio for the period.

    ``T_mean`` is passed through :func:`_to_kelvin`, so the caller may supply
    either Celsius or Kelvin (a value > 150 is assumed already Kelvin); Eq. 9
    is then evaluated with the absolute temperature it requires.

    Wang Eq. 7 carries an additional crosswind contribution,
    ``-(2 * T * q / cs**2) * u'w'`` (``cs`` = speed of sound, ``q`` = specific
    humidity, ``u'w'`` = momentum flux). It is **intentionally omitted here**,
    matching Wang et al. (2024), Eq. 9: the sonic anemometer's crosswind
    correction is already applied internally to the raw covariances, so
    re-applying it would double-count. This omission is deliberate, not an
    oversight.

    A **negative beta is valid** -- it is the oasis/advection fingerprint
    (downward H; see :func:`compute_bowen_ratio_variance` and ``CLAUDE.md``) --
    and is handled exactly like a positive beta. The denominator factor
    ``1 + 0.51 * Cp * T_mean / (Lv * beta)`` is order-unity for physically
    typical periods, but for a narrow band of *small negative* beta (around
    ``beta = -0.51 * Cp * T_mean / Lv``) it passes through zero. Dividing there
    would yield an unphysically huge corrected flux, so that singular case is
    detected and reported as ``nan`` (see *Warns*) rather than returned.

    Parameters
    ----------
    w_Ts : float
        Sonic (virtual) temperature flux, w'Ts' [K m/s].
    T_mean : float
        Mean air temperature during the period [°C or K]. Eq. 9 requires
        Kelvin; the value is converted via :func:`_to_kelvin`.
    beta : float
        Signed Bowen ratio (dimensionless). Negative values (advection) are
        valid and handled.
    Cp : float, optional
        Specific heat capacity of air [J/(kg K)] (use the moist-air value if
        available; default 1005).
    Lv : float, optional
        Latent heat of vaporization ``lambda`` [J/kg] (if None, uses
        ~2.45e6 J/kg).
    singular_tol : float, optional
        Magnitude below which the denominator factor is treated as singular
        (default 1e-6). The factor is order-unity for valid data, so a value
        this close to zero indicates the pathological small-negative-beta band.

    Returns
    -------
    float
        Corrected kinematic sensible-heat flux w'T' [K m/s]. Returns ``w_Ts``
        unchanged when ``beta == 0`` (the Bowen-ratio humidity correction is
        undefined, so it is skipped), and ``nan`` when the denominator factor is
        within ``singular_tol`` of zero.

    Warns
    -----
    UserWarning
        When the denominator factor ``1 + 0.51 * Cp * T_mean / (Lv * beta)`` is
        within ``singular_tol`` of zero (the small-negative-beta singularity):
        ``nan`` is returned instead of an unphysically large flux.

    References
    ----------
    Wang et al. (2024), Eq. 9 (and Eq. 7 for the deliberately dropped crosswind
    term). Moderow et al. (2021), sign convention.
    """
    if Lv is None:
        Lv = 2.45e6
    # Eq. 9 requires the mean temperature in Kelvin; accept Celsius or Kelvin.
    T_mean = _to_kelvin(T_mean)

    # beta == 0: the Bowen-ratio humidity correction is undefined (H = 0 in the
    # ratio's numerator drives the factor to infinity). Skip the correction and
    # return w'Ts' unchanged as a documented fallback.
    if beta == 0:
        return w_Ts

    factor = 1 + 0.51 * (Cp * T_mean) / (Lv * beta)

    # Guard the near-singular denominator. beta < 0 is valid (advection); only a
    # factor *at* the zero crossing -- a narrow band of small negative beta --
    # is rejected, since dividing there yields an unphysically huge flux.
    if abs(factor) < singular_tol:
        warnings.warn(
            "correct_sonic_heat_flux: the sonic-correction denominator "
            f"(1 + 0.51*Cp*T/(Lv*beta)) = {factor:.3e} is within singular_tol "
            f"({singular_tol:.1e}) of zero for beta={beta:.4g}; the correction "
            "is singular in this small-negative-beta band and would yield an "
            "unphysically large flux. Returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    return w_Ts / factor


def compute_sensible_heat_flux(w_T_prime, rho_air, Cp=1005.0):
    r"""
    Compute the sensible heat flux H (W/m^2) from the kinematic heat flux w'T'.

    Implements the ``CLAUDE.md`` "Sensible heat" relation::

        H = rho * Cp * w'T'                                   # W/m^2

    Sign convention (Moderow et al. 2021, **OUT-positive**): a positive ``H`` is
    energy carried *out* of the control volume (upward kinematic heat flux), a
    negative ``H`` is energy *into* it (downward) -- the oasis fingerprint. The
    sign is inherited directly from ``w'T'``; no extra negation is applied. Use
    the WPL/humidity-corrected kinematic flux from
    :func:`correct_sonic_heat_flux` as ``w_T_prime``.

    Parameters
    ----------
    w_T_prime : float
        Corrected kinematic sensible heat flux w'T' [K m/s].
    rho_air : float
        Air density [kg/m^3] during the period.
    Cp : float, optional
        Specific heat capacity of air [J/(kg K)] (use moist-air value if available).

    Returns
    -------
    float
        Sensible heat flux H [W/m^2].

    References
    ----------
    Wang et al. (2024); Moderow et al. (2021), OUT-positive sign convention.
    """
    return rho_air * Cp * w_T_prime


def latent_heat_flux_residual(Rnet, G, H):
    r"""
    Compute latent heat flux (λE) as the residual of the energy balance.

    Implements the **residual-closure** estimate (Twine et al. 2000)::

        λE = R_net - G - H                                    # W/m^2

    i.e. ``H`` is trusted and ``LE`` absorbs the closure gap, the storage-free
    rearrangement of ``Rn - G = H + LE``. This is the *latent-heat-as-residual*
    method (a legitimate standard closure choice), and is **distinct from** the
    ``CLAUDE.md`` prohibition on computing an **advection** term as a residual.
    See :func:`advection.closure.residual_le_closure` for the storage-aware form
    and the closure caveats.

    .. note::

        Any **measured** open-path ``LE`` you compare against this estimate is
        assumed to be **already WPL (Webb-Pearman-Leuning 1980) density-corrected**.
        WPL is a mandatory, separate pre-step (not an advection fix); see
        :func:`wpl_latent_heat_flux` and the package README.

    Parameters
    ----------
    Rnet : float
        Net radiation [W/m^2].
    G : float
        Ground heat flux (storage-corrected) [W/m^2].
    H : float
        Sensible heat flux [W/m^2].

    Returns
    -------
    float
        Latent heat flux λE [W/m^2].

    References
    ----------
    Twine, T. E., et al. (2000), Agric. For. Meteorol. 103, 279-300 (residual
    closure).
    """
    return Rnet - G - H


def latent_heat_flux_bowen(Rnet, G, beta, singular_tol=1e-6):
    r"""
    Compute latent heat flux (λE) using the Bowen ratio method (no fast data needed).

    Implements the **Bowen-ratio partition** of the available energy
    (Twine et al. 2000; Bowen 1926)::

        λE = (R_net - G) / (1 + beta)                         # W/m^2

    where ``beta = H / λE`` is the Bowen ratio. (This is *closure forcing*, not
    advection accounting; see the caution below and
    :func:`advection.closure.bowen_ratio_closure`.)

    .. note::

        Any **measured** open-path ``LE`` you compare against this estimate is
        assumed to be **already WPL (Webb-Pearman-Leuning 1980) density-corrected**.
        WPL is a mandatory, separate pre-step (not an advection fix); see
        :func:`wpl_latent_heat_flux` and the package README.

    The denominator ``1 + beta`` is order-unity for typical periods, but it
    passes through zero as ``beta -> -1`` -- a value that can occur in the
    **oasis/advection regime** (negative beta is the oasis fingerprint; see
    :func:`compute_bowen_ratio_variance` and ``CLAUDE.md``). Dividing there would
    yield an unphysically huge flux, so that singular case is detected and
    reported as ``nan`` (see *Warns*) rather than returned.

    .. caution::

        Bowen-ratio closure is **physically invalid when ``LE > (Rn - G)``**
        (the oasis/advection case). The method partitions the *available*
        energy ``R_net - G`` using the measured ``beta``, which forces the
        residual to share that ratio; when warm dry air advects extra energy
        into the control volume, the true ``LE`` exceeds the available energy
        and no real ``beta`` reproduces it. Forcing closure here drives ``beta``
        toward ``-1`` and the estimate diverges or changes sign (Twine et al.
        2000; Wang et al. 2024). In that regime **do not force Bowen-ratio
        closure** -- instead add the *measured* advective fluxes
        (:func:`horizontal_heat_advection`, :func:`vertical_heat_advection`,
        etc.) to close the budget, gated by the conditional-inclusion rule
        (``R_net > 75 W/m^2`` AND spectrally-corrected ``H + LE < R_net - G``;
        see ``CLAUDE.md``). Never compute an advection term as the
        energy-balance residual.

    Parameters
    ----------
    Rnet : float
        Net radiation [W/m^2].
    G : float
        Ground heat flux (storage-corrected) [W/m^2].
    beta : float
        Bowen ratio (dimensionless). Negative values occur in the oasis regime;
        values near ``-1`` are singular (see *Warns*).
    singular_tol : float, optional
        Magnitude below which the denominator ``1 + beta`` is treated as
        singular (default 1e-6). The denominator is order-unity for valid data,
        so a value this close to zero indicates the pathological ``beta ≈ -1``
        band.

    Returns
    -------
    float
        Latent heat flux λE [W/m^2]. Returns ``nan`` when ``1 + beta`` is within
        ``singular_tol`` of zero.

    Warns
    -----
    UserWarning
        When ``1 + beta`` is within ``singular_tol`` of zero (``beta ≈ -1``):
        ``nan`` is returned instead of an unphysically large flux.

    References
    ----------
    Twine, T. E., et al. (2000), Agric. For. Meteorol. 103, 279-300 -- on the
    breakdown of Bowen-ratio closure when ``LE > (Rn - G)``. Wang et al. (2024).
    Moderow et al. (2021), sign convention.
    """
    denom = 1 + beta

    # Guard the near-singular denominator. beta < 0 is valid (advection); only a
    # denominator *at* the zero crossing -- the narrow band around beta = -1 --
    # is rejected, since dividing there yields an unphysically huge flux.
    if abs(denom) < singular_tol:
        warnings.warn(
            "latent_heat_flux_bowen: the denominator (1 + beta) = "
            f"{denom:.3e} is within singular_tol ({singular_tol:.1e}) of zero "
            f"for beta={beta:.4g}; Bowen-ratio closure is singular near "
            "beta = -1 and would yield an unphysically large flux. This is the "
            "oasis/advection regime where LE > (Rn - G) and Bowen-ratio closure "
            "is invalid -- add measured advective fluxes instead (see "
            "CLAUDE.md conditional-inclusion rule). Returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    return (Rnet - G) / denom


def wpl_latent_heat_flux(
    w_rhov,
    w_T,
    rho_v,
    T,
    mixing_ratio,
    Lv=None,
    mu=MU,
):
    r"""Convenience WPL (Webb-Pearman-Leuning 1980) density-corrected latent heat flux.

    .. important::

        This is a **convenience pre-step**, *not* part of the advection
        accounting this library performs. The rest of :mod:`advection` assumes
        any open-path ``LE``/CO2 flux it is given has **already** been WPL
        density-corrected (see :func:`latent_heat_flux_residual`,
        :func:`latent_heat_flux_bowen`,
        :func:`advection.closure.bowen_ratio_closure` and the package README).
        WPL is a mandatory, *separate* pre-processing step — it accounts for the
        density fluctuations of dry air that contaminate an open-path vapour
        covariance, and it is **not** an advection correction.

    .. caution::

        Prefer an established eddy-covariance processing package (e.g. EddyPro,
        EasyFlux, or your logger's online WPL routine) for production work. This
        helper implements only the **simplified** form below — it omits the
        ambient-pressure fluctuation term and assumes the inputs are already
        coordinate-rotated, despiked, and frequency-response corrected. Use it
        for teaching, quick checks, or when you have the raw covariances but no
        full processing chain, not as a substitute for validated software.

    Implements the simplified Webb et al. (1980) water-vapour flux::

        E = (1 + mu * MR) * [ w'rho_v' + (rho_v / T) * w'T' ]     # kg m^-2 s^-1
        LE = Lv * E                                               # W/m^2

    with ``mu = M_d / M_v = 1.6077`` (:data:`advection._constants.MU`). The first
    bracket term is the raw (uncorrected) vapour covariance; the second adds the
    thermal-expansion contribution ``(rho_v / T) * w'T'``; the ``(1 + mu*MR)``
    prefactor is the dry-air dilution correction. ``T`` is passed through
    :func:`_to_kelvin`, so Celsius or Kelvin may be supplied (the ratio
    ``rho_v / T`` requires an absolute temperature).

    Parameters
    ----------
    w_rhov : float
        Raw covariance of vertical wind and water-vapour density,
        ``w'rho_v'`` [kg m^-2 s^-1] — the *uncorrected* open-path vapour flux.
    w_T : float
        Kinematic sensible-heat flux ``w'T'`` [K m/s].
    rho_v : float
        Mean water-vapour (mass) density [kg/m^3].
    T : float
        Mean air temperature [°C or K]; converted to Kelvin via
        :func:`_to_kelvin`.
    mixing_ratio : float
        Water-vapour **mass mixing ratio** ``MR = rho_v / rho_d`` [kg/kg]
        (vapour mass per unit mass of *dry* air).
    Lv : float, optional
        Latent heat of vaporization [J/kg]. If ``None``, uses
        ``latent_heat_vaporization(T)``.
    mu : float, optional
        Ratio of molar masses ``M_d / M_v`` [dimensionless]; default
        :data:`~advection._constants.MU` (1.6077).

    Returns
    -------
    float
        WPL density-corrected latent heat flux ``LE`` [W/m^2].

    References
    ----------
    Webb, E. K., Pearman, G. I., & Leuning, R. (1980), *Correction of flux
    measurements for density effects due to heat and water vapour transfer*,
    Q. J. R. Meteorol. Soc. 106, 85-100.
    """
    T = _to_kelvin(T)
    if Lv is None:
        Lv = latent_heat_vaporization(T)
    # Eq.: dry-air dilution prefactor times (raw vapour covariance + thermal
    # expansion term). E is the corrected vapour mass flux [kg m^-2 s^-1].
    E = (1.0 + mu * mixing_ratio) * (w_rhov + (rho_v / T) * w_T)
    return Lv * E


def compute_std(series):
    """
    Compute the standard deviation of a time series.

    Suitable for computing σ_T or σ_q over an averaging period.

    Parameters
    ----------
    series : iterable
        Iterable of data points (list or NumPy array).

    Returns
    -------
    float
        Standard deviation of the series.
    """
    data = np.array(series, dtype=float)
    return float(np.nanstd(data, ddof=0))


def rh_to_specific_humidity(RH, T, P=101325):
    """
    Convert relative humidity to specific humidity.

    Parameters
    ----------
    RH : float
        Relative humidity [% (0-100) or fraction (0-1)].
    T : float
        Air temperature [°C].
    P : float, optional
        Ambient pressure [Pa] (default 101325 Pa, sea level).

    Returns
    -------
    float
        Specific humidity q [kg/kg].
    """
    # Convert RH to a 0-1 fraction if given in %
    RH_frac = RH / 100.0 if RH > 1.0 else RH
    # Saturation vapor pressure (Pa) over water at temperature T (Bolton 1980 formula)
    Esat = 611.2 * np.exp(17.67 * T / (T + 243.5))
    # Actual vapor pressure (Pa)
    e = RH_frac * Esat
    # Mixing ratio w = mass_vapor/mass_dry = 0.622 * e / (P - e)
    w = 0.622 * e / (P - e)
    # Specific humidity q = w / (1 + w)
    return w / (1 + w)


def virtual_temperature(T, q):
    """
    Calculate virtual temperature T_v (K) for moist air.

    T_v = T * (1 + 0.61 * q), where q is specific humidity.

    Parameters
    ----------
    T : float
        Actual air temperature [K].
    q : float
        Specific humidity [kg/kg].

    Returns
    -------
    float
        Virtual temperature T_v [K].
    """
    T = _to_kelvin(T)
    return T * (1 + 0.61 * q)


def air_density(P, T, q, R_dry=287.05, R_vap=461.5):
    """
    Calculate moist air density [kg/m^3] given pressure, temperature, and humidity.

    Parameters
    ----------
    P : float
        Ambient pressure [Pa].
    T : float
        Air temperature [K].
    q : float
        Specific humidity [kg/kg].
    R_dry : float, optional
        Gas constant for dry air [J/(kg K)] (default 287.05).
    R_vap : float, optional
        Gas constant for water vapor [J/(kg K)] (default 461.5).

    Returns
    -------
    float
        Air density [kg/m^3].
    """
    # Ideal-gas law requires an absolute temperature.
    T = _to_kelvin(T)
    # Compute vapor pressure e from specific humidity (invert q formula):
    # q = w/(1+w), w = q/(1-q), and w = 0.622 * e/(P - e) -> solve for e:
    w = q / max(1e-9, (1 - q))  # mixing ratio (avoid division by zero if q ~1)
    e = (w * P) / (0.622 + w)  # vapor partial pressure (Pa)
    # Dry air partial pressure
    P_d = P - e
    # Calculate densities
    rho_dry = P_d / (R_dry * T)
    rho_vap = e / (R_vap * T)
    return rho_dry + rho_vap


def latent_heat_vaporization(T):
    """
    Compute latent heat of vaporization of water (Lv) at temperature T.

    Uses a polynomial fit for 0 <= T <= 40°C (from literature).
    T can be in °C or K (if K, it is converted to °C internally).

    Parameters
    ----------
    T : float
        Air temperature [°C or K].

    Returns
    -------
    float
        Latent heat of vaporization Lv [J/kg].
    """
    # Convert K to °C if necessary
    T_C = T - 273.15 if T > 100 else T  # assume T>100 means Kelvin
    # Polynomial fit as per standard formula (Wikipedia or literature)
    Lv = (2500.8 - 2.36 * T_C + 0.0016 * (T_C**2) - 0.00006 * (T_C**3)) * 1000.0
    return Lv


def specific_heat_moist_air(q):
    """
    Calculate the specific heat capacity of moist air [J/(kg K)] given specific humidity.

    Parameters
    ----------
    q : float
        Specific humidity [kg/kg].

    Returns
    -------
    float
        Cp of moist air [J/(kg K)].
    """
    Cp_dry = 1005.0  # J/(kg K) for dry air
    Cp_vap = 1860.0  # J/(kg K) for water vapor (at ~300 K)
    return (1 - q) * Cp_dry + q * Cp_vap
