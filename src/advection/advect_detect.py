"""Functions for detecting horizontal and vertical advection."""

import numpy as np

# Physical constants for air (could be parameterized if needed)
AIR_DENSITY = 1.225  # kg/m^3 (approximate, at sea level and 15°C)
SPECIFIC_HEAT_AIR = 1005  # J/(kg·K), specific heat of air at constant pressure
LATENT_HEAT_VAP = 2.45e6  # J/kg, latent heat of vaporization of water


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


def compute_advection_fluxes(
    main_data,
    upwind_data=None,
    detect_horizontal=None,
    detect_vertical=None,
    tower_distance=None,
):
    """
    Compute horizontal and vertical advection flux time series for energy balance closure.

    Parameters
    ----------
    main_data : dict
        Dictionary of main tower data series. Must contain:
        'H' (sensible heat flux, W/m^2),
        'LE' (latent heat flux, W/m^2),
        'Rn' (net radiation, W/m^2),
        'G' (ground heat flux, W/m^2).
    upwind_data : dict or list of dicts, optional
        Data for upwind tower(s), with similar keys as main_data.
    detect_horizontal : np.ndarray, optional
        Boolean array from detect_horizontal_advection (to limit times of computation).
    detect_vertical : np.ndarray, optional
        Boolean array from detect_vertical_advection.
    tower_distance : float, optional
        Distance (m) between the main tower and the upwind tower used for advection.

    Returns
    -------
    dict
        A dictionary with keys:
        'H_adv' (horizontal advective heat flux, W/m^2, positive = energy entering control volume),
        'V_adv' (vertical advective heat flux, W/m^2, positive = energy entering from above (downward)),
        'adv_in' (net advective energy term used to close balance, W/m^2).
    """
    # Extract main data
    H_main = np.array(main_data["H"])
    LE_main = np.array(main_data["LE"])
    Rn_main = np.array(main_data["Rn"])
    G_main = np.array(main_data["G"])
    n = len(H_main)
    # Prepare upwind arrays
    use_multi_upwind = False
    if upwind_data is not None:
        if isinstance(upwind_data, list):
            use_multi_upwind = True
            # If multiple upwind towers, ensure each has necessary data
            # We assume all upwind dicts have 'H'; may also have 'T', 'q', and maybe their position or bearing.
        else:
            # single upwind tower
            upwind_data = [upwind_data]
            use_multi_upwind = True
    # Initialize arrays for adv fluxes
    H_adv = np.zeros(n)
    V_adv = np.zeros(n)
    adv_in = np.zeros(n)
    # Determine horizontal adv flux for each timestep
    for i in range(n):
        # net imbalance (H+LE - (Rn-G)) at this time
        adv_in[i] = (H_main[i] + LE_main[i]) - (Rn_main[i] - G_main[i])
        # Default adv contributions
        horiz_flux = 0.0
        vert_flux = 0.0
        # Only calculate adv flux if flagged (if detect arrays given), otherwise do for all times
        if detect_horizontal is not None and not detect_horizontal[i]:
            # No horizontal advection flagged
            pass
        else:
            # Choose appropriate upwind tower data if available
            if use_multi_upwind:
                # If multiple upwind towers, pick the one aligned with wind direction (if data available)
                chosen_upwind = None
                if upwind_data:
                    if "wind_dir" in main_data and "bearing" in upwind_data[0]:
                        # Example: each upwind_data dict could have a 'bearing' key for its direction relative to main
                        wind_dir = main_data.get("wind_dir")
                        if wind_dir is not None:
                            wd = wind_dir[i]
                            # Find tower with minimal angle difference to wind direction
                            best_diff = 361
                            for tower in upwind_data:
                                bearing = tower.get("bearing")
                                if bearing is None:
                                    continue
                                diff = (
                                    abs(((wd - bearing + 180) % 360) - 180)
                                    if wd is not None
                                    else 360
                                )
                                if diff < best_diff:
                                    best_diff = diff
                                    chosen_upwind = tower
                    # If no wind direction or bearing info, default to first tower
                    if chosen_upwind is None:
                        chosen_upwind = upwind_data[0]
                else:
                    chosen_upwind = None
                if chosen_upwind is not None and "H" in chosen_upwind:
                    H_up = chosen_upwind["H"][i]
                    # Horizontal advective heat flux: difference between upwind and main H
                    if H_up is not None:
                        horiz_flux = H_up - H_main[i]
            else:
                horiz_flux = 0.0
        # Vertical advection: only if flagged (or if horizontal adv doesn't fully explain imbalance)
        if detect_vertical is not None and detect_vertical[i]:
            # If we explicitly detected vertical advection, we'll compute it as residual needed for closure.
            vert_flux = adv_in[i] - horiz_flux
        else:
            # If not explicitly detected, we set to 0 unless a residual remains significant after horizontal correction
            # If horizontal flux is accounted and a large imbalance remains, attribute it to vertical.
            residual = adv_in[i] - horiz_flux
            if abs(residual) > 1e-6:
                # We assign vertical adv only if it is substantial (e.g., more than 10% of available energy)
                if Rn_main[i] - G_main[i] != 0:
                    if abs(residual) > 0.1 * (Rn_main[i] - G_main[i]):
                        vert_flux = residual
        # Assign outputs (note: positive horiz_flux means energy entering the main site from upwind;
        # positive vert_flux means energy entering from above (downward heat advection))
        H_adv[i] = horiz_flux
        V_adv[i] = vert_flux
    return {"H_adv": H_adv, "V_adv": V_adv, "adv_in": adv_in}


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
