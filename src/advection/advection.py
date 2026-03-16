"""Main module."""

import numpy as np


def compute_soil_heat_storage_flux(Csoil, dT_dt, depth=0.02):
    """
    Compute soil heat storage flux Gs (W/m^2) from soil volumetric heat capacity and temperature change rate.

    Implements Equation 1a: Gs = depth * Csoil * (dTsoil/dt).

    Parameters
    ----------
    Csoil : float
        Volumetric heat capacity of the soil layer [J/m^3 K].
    dT_dt : float
        Time derivative of soil temperature [K/s] (temperature change rate).
    depth : float, optional
        Depth of the heat flux plate [m] (default 0.02 m for 2 cm).

    Returns
    -------
    float
        Soil heat storage flux Gs [W/m^2].
    """
    return depth * Csoil * dT_dt


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


def compute_bowen_ratio_variance(sigma_T, sigma_q, Cp=1005.0, Lv=None):
    """
    Compute the Bowen ratio (beta) using the variance method (Wang et al. 2023).

    Implements Equation 3: beta = (Cp / Lv) * (sigma_T / sigma_q), assuming equal diffusion of heat and moisture.

    Parameters
    ----------
    sigma_T : float
        Standard deviation of air temperature fluctuations [K].
    sigma_q : float
        Standard deviation of specific humidity fluctuations [kg/kg].
    Cp : float, optional
        Specific heat capacity of air [J/(kg K)] (use moist-air value if available; default 1005 J/(kg K) for dry air).
    Lv : float, optional
        Latent heat of vaporization [J/kg] at the given temperature (if None, uses ~2.45e6 J/kg for ~20°C).

    Returns
    -------
    float
        Bowen ratio beta (dimensionless).
    """
    if Lv is None:
        Lv = 2.45e6  # default latent heat of vaporization (approx. at 20°C)
    return (Cp / Lv) * (sigma_T / sigma_q)


def correct_sonic_heat_flux(w_Ts, T_mean, beta, Cp=1005.0, Lv=None):
    """
    Convert sonic temperature flux (w'Ts') to true sensible heat flux (w'T') by accounting for humidity.

    Implements Eq. 4: w'T' = w'Ts' / [1 + 0.51 * (Cp * T_mean) / (Lv * beta)].

    Parameters
    ----------
    w_Ts : float
        Sonic (virtual) temperature flux, w'Ts' [K m/s].
    T_mean : float
        Mean air temperature during the period [K].
    beta : float
        Bowen ratio (dimensionless) for the period.
    Cp : float, optional
        Specific heat capacity of air [J/(kg K)] (use moist-air value if available; default 1005).
    Lv : float, optional
        Latent heat of vaporization [J/kg] (if None, uses ~2.45e6 J/kg).

    Returns
    -------
    float
        Corrected kinematic sensible heat flux w'T' [K m/s].
    """
    if Lv is None:
        Lv = 2.45e6
    # Avoid division by zero in extreme case beta=0 (no latent heat flux)
    factor = 1 + 0.51 * (Cp * T_mean) / (Lv * beta) if beta != 0 else 1.0
    return w_Ts / factor


def compute_sensible_heat_flux(w_T_prime, rho_air, Cp=1005.0):
    """
    Compute the sensible heat flux H (W/m^2) from the kinematic heat flux w'T'.

    Implements Eq. 5: H = rho * Cp * w'T'.

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
    """
    return rho_air * Cp * w_T_prime


def latent_heat_flux_residual(Rnet, G, H):
    """
    Compute latent heat flux (λE) as the residual of the energy balance.

    Implements Eq. 6: λE = R_net - G - H.

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
    """
    return Rnet - G - H


def latent_heat_flux_bowen(Rnet, G, beta):
    """
    Compute latent heat flux (λE) using the Bowen ratio method (no fast data needed).

    Implements Eq. 7: λE = (R_net - G) / (1 + beta).

    Parameters
    ----------
    Rnet : float
        Net radiation [W/m^2].
    G : float
        Ground heat flux (storage-corrected) [W/m^2].
    beta : float
        Bowen ratio (dimensionless).

    Returns
    -------
    float
        Latent heat flux λE [W/m^2].
    """
    return (Rnet - G) / (1 + beta)


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
