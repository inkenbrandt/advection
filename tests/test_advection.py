#!/usr/bin/env python

"""Tests for `advection` package."""

import pytest
import math
import numpy as np
import sys
import os

sys.path.append("..")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import advection as ax

# -----------------------------------------------------------------------------
# advection.py tests
# -----------------------------------------------------------------------------


def test_compute_soil_heat_storage_flux():
    """Gs should equal Csoil * dz * dT/dt, where dz is the layer thickness."""
    Csoil = 2_000_000.0  # J/m³·K
    dT_dt = 1.0e-3  # K/s
    dz = 0.05  # m thickness of soil above the heat-flux plate
    expected = Csoil * dz * dT_dt
    out = ax.compute_soil_heat_storage_flux(Csoil, dT_dt, dz)
    assert pytest.approx(out) == expected


def test_total_ground_heat_flux():
    """Storage‐corrected flux should be additive."""
    Gd = -15.0
    Gs = 5.0
    assert ax.total_ground_heat_flux(Gd, Gs) == Gd + Gs


def test_air_heat_storage():
    """J should equal rho * Cp * dT/dt * (zm - h) [W/m^2] (Wang 2024 Eq. 11)."""
    rho = 1.2  # kg/m^3
    Cp = 1005.0  # J/(kg K)
    dT_dt = 1.0e-3  # K/s
    zm = 2.0  # m
    h = 0.2  # m
    expected = rho * Cp * dT_dt * (zm - h)
    out = ax.air_heat_storage(rho, Cp, dT_dt, zm, h)
    assert pytest.approx(out) == expected

    # Unit sanity: [kg/m^3] * [J/(kg K)] * [K/s] * [m] = [W/m^2], and a warming
    # air column (dT/dt > 0) stores energy, giving a positive J.
    assert out > 0.0
    # No temperature change => no storage.
    assert ax.air_heat_storage(rho, Cp, 0.0, zm, h) == 0.0


def test_compute_bowen_ratio_variance():
    sigma_T = 0.6  # K
    sigma_q = 0.003  # kg/kg
    Cp = 1005.0
    Lv = 2.45e6
    expected = (Cp / Lv) * (sigma_T / sigma_q)
    # With only the standard deviations supplied the sign is undetermined: the
    # function returns the unsigned magnitude and warns.
    with pytest.warns(UserWarning):
        out = ax.compute_bowen_ratio_variance(sigma_T, sigma_q, Cp=Cp, Lv=Lv)
    assert pytest.approx(out) == expected


def test_compute_bowen_ratio_variance_oasis_negative_beta():
    """Anticorrelated warm-dry / cool-moist T,q must yield a negative beta.

    This is the oasis / heat-advection fingerprint (Wang 2024 Eq. 8): warm dry
    air (T' > 0) coincides with drier air (q' < 0), so corr(T', q') < 0 and beta
    is negative, corresponding to downward H (energy into the control volume).
    """
    rng = np.random.default_rng(0)
    base = rng.standard_normal(2000)
    T_prime = 0.5 * base + 0.05 * rng.standard_normal(2000)  # warm dry excursions
    q_prime = -0.002 * base + 2e-4 * rng.standard_normal(2000)  # anticorrelated

    beta = ax.compute_bowen_ratio_variance(T_prime=T_prime, q_prime=q_prime)
    assert beta < 0
    # Magnitude should match the std-dev ratio form (sign aside).
    Lv = 2.45e6
    expected_mag = (1005.0 / Lv) * (np.std(T_prime) / np.std(q_prime))
    assert beta == pytest.approx(-expected_mag)


def test_compute_bowen_ratio_variance_daytime_positive_beta():
    """Classic daytime positive correlation of T,q must yield a positive beta."""
    rng = np.random.default_rng(1)
    base = rng.standard_normal(2000)
    T_prime = 0.5 * base + 0.05 * rng.standard_normal(2000)
    q_prime = 0.002 * base + 2e-4 * rng.standard_normal(2000)  # positively correlated

    beta = ax.compute_bowen_ratio_variance(T_prime=T_prime, q_prime=q_prime)
    assert beta > 0


def test_compute_bowen_ratio_variance_sign_sources_and_zero_tol():
    """corr_Tq / cov_Tq override series; near-zero covariance falls back to +1."""
    # Precomputed correlation sign wins.
    neg = ax.compute_bowen_ratio_variance(0.6, 0.003, corr_Tq=-0.7)
    assert neg < 0
    pos = ax.compute_bowen_ratio_variance(0.6, 0.003, cov_Tq=1.2e-5)
    assert pos > 0
    # Covariance within tolerance is treated as +1 (no advective signal).
    tie = ax.compute_bowen_ratio_variance(0.6, 0.003, cov_Tq=0.0)
    assert tie > 0


def test_compute_bowen_ratio_variance_lv_from_temperature():
    """When Lv is None and T is given, Lv = latent_heat_vaporization(T)."""
    T = 30.0  # deg C
    expected = (1005.0 / ax.latent_heat_vaporization(T)) * (0.6 / 0.003)
    out = ax.compute_bowen_ratio_variance(0.6, 0.003, T=T, corr_Tq=0.9)
    assert out == pytest.approx(expected)


def test_correct_sonic_heat_flux():
    w_Ts = 0.12  # K·m s⁻¹
    T_mean = 293.15  # K
    beta = 1.5
    Lv = 2.45e6
    Cp = 1005.0
    denom = 1 + 0.51 * (Cp * T_mean) / (Lv * beta)
    expected = w_Ts / denom
    out = ax.correct_sonic_heat_flux(w_Ts, T_mean, beta, Cp=Cp, Lv=Lv)
    assert math.isfinite(out)
    assert out < w_Ts  # positive beta -> factor > 1 -> reduced flux
    assert pytest.approx(out) == expected


def test_correct_sonic_heat_flux_negative_beta():
    # A negative Bowen ratio (oasis/advection fingerprint) is VALID and must be
    # handled, not rejected. Here the factor is < 1, amplifying w'Ts'.
    w_Ts = 0.12
    T_mean = 293.15  # K
    beta = -1.5
    Lv = 2.45e6
    Cp = 1005.0
    denom = 1 + 0.51 * (Cp * T_mean) / (Lv * beta)
    expected = w_Ts / denom
    out = ax.correct_sonic_heat_flux(w_Ts, T_mean, beta, Cp=Cp, Lv=Lv)
    assert math.isfinite(out)
    assert out > w_Ts  # factor < 1 amplifies the flux
    assert pytest.approx(out) == expected


def test_correct_sonic_heat_flux_zero_beta():
    # If beta is 0 the humidity correction is undefined; w'Ts' is returned
    # unchanged (documented fallback).
    w_Ts = 0.12
    T_mean = 293.15
    beta = 0.0
    out = ax.correct_sonic_heat_flux(w_Ts, T_mean, beta)
    assert out == w_Ts


def test_correct_sonic_heat_flux_celsius_input_converted():
    # Eq. 9 requires Kelvin; passing Celsius must yield the same result as the
    # equivalent Kelvin value (the _to_kelvin helper is applied internally).
    w_Ts = 0.12
    beta = 1.5
    out_celsius = ax.correct_sonic_heat_flux(w_Ts, 20.0, beta)  # 20 C
    out_kelvin = ax.correct_sonic_heat_flux(w_Ts, 293.15, beta)  # 293.15 K
    assert pytest.approx(out_celsius) == out_kelvin


def test_correct_sonic_heat_flux_near_singular_denominator():
    # For a small negative beta the denominator factor passes through zero. The
    # function must flag this and return NaN rather than a huge number.
    w_Ts = 0.12
    T_mean = 293.15  # K
    Cp = 1005.0
    Lv = 2.45e6
    # beta driving 1 + 0.51*Cp*T/(Lv*beta) exactly to 0 (~ -0.061, a small
    # negative beta within the valid-but-singular advection band).
    beta_singular = -0.51 * (Cp * T_mean) / Lv
    with pytest.warns(UserWarning, match="singular"):
        out = ax.correct_sonic_heat_flux(w_Ts, T_mean, beta_singular, Cp=Cp, Lv=Lv)
    assert math.isnan(out)


def test_compute_sensible_heat_flux():
    w_T_prime = 0.05  # K·m s⁻¹
    rho = 1.2  # kg m⁻³
    Cp = 1005.0
    expected = rho * Cp * w_T_prime
    out = ax.compute_sensible_heat_flux(w_T_prime, rho, Cp=Cp)
    assert pytest.approx(out) == expected


def test_latent_heat_flux_methods_agree():
    Rn = 450.0
    G = 50.0
    beta = 0.7
    # Pick H such that residual λE matches Bowen ratio method
    LE_bowen = ax.latent_heat_flux_bowen(Rn, G, beta)
    H = (Rn - G) - LE_bowen
    LE_residual = ax.latent_heat_flux_residual(Rn, G, H)
    assert pytest.approx(LE_residual) == LE_bowen


def test_latent_heat_flux_bowen_near_minus_one_beta():
    # As beta -> -1 the denominator (1 + beta) passes through zero (the oasis
    # regime where LE > Rn - G). The function must flag this and return NaN
    # rather than an unphysically large flux.
    Rn = 450.0
    G = 50.0
    beta_singular = -1.0 + 1e-9  # within the default singular_tol of -1
    with pytest.warns(UserWarning, match="singular"):
        out = ax.latent_heat_flux_bowen(Rn, G, beta_singular)
    assert math.isnan(out)


def test_compute_std_handles_nan():
    series = [1, 2, np.nan, 4]
    out = ax.compute_std(series)
    expected = np.nanstd(np.array(series), ddof=0)
    assert math.isclose(out, expected, rel_tol=1e-12)


def test_rh_to_specific_humidity():
    RH = 50.0  # %
    T = 20.0  # °C
    # Manual calc matching function implementation
    Esat = 611.2 * math.exp(17.67 * T / (T + 243.5))
    e = 0.5 * Esat
    w = 0.622 * e / (101325 - e)
    expected = w / (1 + w)
    out = ax.rh_to_specific_humidity(RH, T)
    assert pytest.approx(out, rel=1e-6) == expected


def test_virtual_temperature():
    T = 300.0  # K
    q = 0.01  # kg kg⁻¹
    expected = T * (1 + 0.61 * q)
    assert pytest.approx(ax.virtual_temperature(T, q)) == expected


def test_air_density():
    P = 101325.0  # Pa
    T = 298.15  # K
    q = 0.008  # kg kg⁻¹
    R_dry = 287.05
    R_vap = 461.0
    rho = ax.air_density(P, T, q, R_dry, R_vap)
    # Ideal‑gas check: density should be within ±10% of dry‑air density lower bound
    rho_dry = P / (R_dry * T)
    assert 0.9 * rho_dry <= rho <= 1.1 * rho_dry


def test_air_density_high_humidity():
    # q near 1
    P = 101325.0
    T = 300.0
    q = 0.99
    R_dry = 287.05
    R_vap = 461.0
    rho = ax.air_density(P, T, q, R_dry, R_vap)
    assert rho > 0


def test_latent_heat_vaporization_decreases_with_temp():
    Lv_0 = ax.latent_heat_vaporization(0.0)
    Lv_30 = ax.latent_heat_vaporization(30.0)
    assert Lv_30 < Lv_0  # Lv decreases with temperature


def test_specific_heat_moist_air_bounds():
    q = 0.02  # kg/kg
    Cp = ax.specific_heat_moist_air(q)
    assert 1005.0 <= Cp <= 1860.0  # Should be bounded by dry/vapor values


# -----------------------------------------------------------------------------
# advect_detect.py tests
# -----------------------------------------------------------------------------


def test_detect_horizontal_advection_flux_difference():
    main_flux = [50.0, 50.0]
    upwind_flux = [80.0, 40.0]  # Large diff then small diff
    flags = ax.detect_horizontal_advection(main_flux, upwind_flux=upwind_flux)
    assert np.array_equal(flags, np.array([True, False]))


def test_detect_horizontal_advection_le_exceeds_rn_minus_g():
    main_flux = [50.0]
    le = [110.0]
    rn = [100.0]
    g = [0.0]
    flags = ax.detect_horizontal_advection(main_flux, le_main=le, rn=rn, g=g)
    assert bool(flags[0]) is True


def test_detect_horizontal_advection_negative_midday_H():
    # H < 0 while Rn > rn_high (50) -> oasis negative-H fingerprint.
    flags = ax.detect_horizontal_advection([-10.0, 5.0], rn=[300.0, 300.0])
    assert flags.tolist() == [True, False]


def test_detect_horizontal_advection_ef_requires_positive_available_energy():
    # Nighttime: Rn - G < 0 so EF is meaningless and must NOT flag, even though
    # LE numerically exceeds (Rn - G) * ef_tol (a negative number).
    flags = ax.detect_horizontal_advection([-5.0], le_main=[10.0], rn=[-20.0], g=[10.0])
    assert flags.tolist() == [False]


def test_detect_horizontal_advection_nan_inputs_are_masked():
    # np.nan must be excluded via np.isnan, never flagged. The old `is None`
    # test silently let NaNs through; here every signal sees a NaN and the
    # result must stay False without raising.
    flags = ax.detect_horizontal_advection(
        [np.nan, -10.0],
        rn=[300.0, 300.0],
        le_main=[np.nan, 50.0],
        g=[0.0, 0.0],
        temp_main=[20.0, 20.0],
        temp_upwind=[np.nan, 18.0],
        humidity_main=[np.nan, 0.01],
        humidity_upwind=[0.005, 0.005],
    )
    # t0: H is NaN, all upwind/LE/temp/humidity comparisons hit a NaN -> False.
    # t1: H=-10 < 0 and Rn=300 > 50 -> negative-H fires.
    assert flags.tolist() == [False, True]


def test_detect_horizontal_advection_thresholds_are_keyword_args():
    # Raising rn_high above Rn suppresses the negative-H signal; lowering it
    # restores it. Same for ef_tol and upwind_h_excess.
    assert ax.detect_horizontal_advection([-10.0], rn=[60.0]).tolist() == [True]
    assert ax.detect_horizontal_advection(
        [-10.0], rn=[60.0], rn_high=75.0
    ).tolist() == [False]
    # EF tolerance: LE = 104 is below 1.05 * 100 by default, above 1.0 * 100 if
    # the tolerance is relaxed.
    base = dict(le_main=[104.0], rn=[100.0], g=[0.0])
    assert ax.detect_horizontal_advection([50.0], **base).tolist() == [False]
    assert ax.detect_horizontal_advection([50.0], ef_tol=1.0, **base).tolist() == [True]
    # upwind_h_excess is an absolute W/m^2 difference, not a percentage.
    assert ax.detect_horizontal_advection(
        [50.0], upwind_flux=[65.0], upwind_h_excess=10.0
    ).tolist() == [True]


def test_detect_horizontal_advection_wind_sector_gates_gradients():
    # Warm/dry upwind gradients fire only when wind blows from the upwind tower.
    aligned = ax.detect_horizontal_advection(
        [50.0],
        temp_main=[20.0],
        temp_upwind=[24.0],
        wind_dir=[185.0],
        upwind_dir=180.0,
    )
    assert aligned.tolist() == [True]
    off_sector = ax.detect_horizontal_advection(
        [50.0],
        temp_main=[20.0],
        temp_upwind=[24.0],
        wind_dir=[10.0],
        upwind_dir=180.0,
    )
    assert off_sector.tolist() == [False]
    # A tighter sector can exclude a borderline alignment.
    assert ax.detect_horizontal_advection(
        [50.0],
        temp_main=[20.0],
        temp_upwind=[24.0],
        wind_dir=[210.0],
        upwind_dir=180.0,
        wind_sector_deg=20.0,
    ).tolist() == [False]


def test_detect_horizontal_advection_return_components():
    mask, comps = ax.detect_horizontal_advection(
        [-10.0, 50.0],
        rn=[300.0, 300.0],
        le_main=[10.0, 350.0],  # t1: EF = 350/300 > 1.05 -> advective input
        g=[0.0, 0.0],
        return_components=True,
    )
    assert mask.tolist() == [True, True]
    assert comps["negative_H"].tolist() == [True, False]
    assert comps["ef_gt_1"].tolist() == [False, True]
    # No upwind tower / wind info supplied -> those signals all False, gate open.
    assert comps["warm_upwind"].tolist() == [False, False]
    assert comps["wind_aligned"].tolist() == [True, True]
    # Every component is a length-n boolean array (DataFrame-ready).
    for arr in comps.values():
        assert arr.dtype == bool
        assert arr.shape == (2,)


def test_detect_horizontal_advection_is_vectorized():
    # A longer series returns a single boolean array, not a Python loop result.
    n = 1000
    rng = np.random.default_rng(0)
    H = rng.normal(0.0, 50.0, n)
    Rn = np.full(n, 300.0)
    flags = ax.detect_horizontal_advection(H, rn=Rn)
    assert isinstance(flags, np.ndarray)
    assert flags.shape == (n,)
    assert np.array_equal(flags, (H < 0.0) & (Rn > 50.0))


def test_detect_vertical_advection_w_bar_threshold():
    # Primary signal: |w_bar| above vs below the 0.05 m/s threshold (Lee 1998).
    # t0: |w_bar| = 0.08 > 0.05 -> flagged purely on the mean vertical velocity,
    #     even with no temperature inversion and a healthy positive daytime H.
    # t1: |w_bar| = 0.02 < 0.05, no inversion, normal H -> not flagged.
    vertical_w = [-0.08, 0.02]
    temp_lower = [20.0, 20.0]
    temp_upper = [19.0, 19.0]  # no inversion either step
    main_H = [120.0, 120.0]
    rn = [300.0, 300.0]
    g = [50.0, 50.0]
    flags, comp = ax.detect_vertical_advection(
        vertical_w=vertical_w,
        temp_profile_lower=temp_lower,
        temp_profile_upper=temp_upper,
        main_H=main_H,
        rn=rn,
        g=g,
        return_components=True,
    )
    assert np.array_equal(flags, np.array([True, False]))
    # t0 fires only on the w_bar signal; nothing else.
    assert comp["w_significant"].tolist() == [True, False]
    assert comp["supporting"].tolist() == [False, False]
    assert comp["h_anomaly"].tolist() == [False, False]


def test_detect_vertical_advection_w_threshold_kwarg():
    # The threshold is exposed as a kwarg: raising it past |w_bar| suppresses
    # the flag; lowering it (default 0.05) keeps it.
    vertical_w = [0.06]
    assert ax.detect_vertical_advection(vertical_w=vertical_w).tolist() == [True]
    assert ax.detect_vertical_advection(
        vertical_w=vertical_w, w_threshold=0.1
    ).tolist() == [False]


def test_detect_vertical_advection_supporting_gradient():
    # Supporting signal: daytime (Rn - G > 50) AND a vertical T gradient of the
    # advective sign (warm air aloft) flags even with a sub-threshold w_bar.
    flags = ax.detect_vertical_advection(
        vertical_w=[0.01],
        temp_profile_lower=[15.0],
        temp_profile_upper=[17.0],  # +2 K inversion -> advective sign
        rn=[300.0],
        g=[50.0],
    )
    assert flags.tolist() == [True]
    # Same gradient but at night (Rn - G <= 50) -> supporting signal withheld.
    flags_night = ax.detect_vertical_advection(
        vertical_w=[0.01],
        temp_profile_lower=[15.0],
        temp_profile_upper=[17.0],
        rn=[40.0],
        g=[10.0],
    )
    assert flags_night.tolist() == [False]


def test_detect_vertical_advection_h_anomaly_secondary():
    # Weak/secondary H-anomaly: low daytime H flags by default but can be
    # switched off with use_h_anomaly=False.
    kwargs = dict(
        temp_profile_lower=[20.0],
        temp_profile_upper=[19.0],
        main_H=[10.0],
        rn=[300.0],
        g=[50.0],
    )
    assert ax.detect_vertical_advection(**kwargs).tolist() == [True]
    assert ax.detect_vertical_advection(use_h_anomaly=False, **kwargs).tolist() == [
        False
    ]


def test_detect_vertical_advection_no_data():
    # NaN/None handling: missing entries (None -> nan) are masked via np.isnan
    # and never flag.
    temp_lower = [15.0, None]
    temp_upper = [16.0, 17.0]
    rn = [300.0, 300.0]
    g = [50.0, None]
    flags = ax.detect_vertical_advection(
        temp_profile_lower=temp_lower, temp_profile_upper=temp_upper, rn=rn, g=g
    )
    assert len(flags) == 2
    # t1 has a NaN lower temperature and a NaN g -> no signal can fire.
    assert bool(flags[1]) is False


def test_compute_advection_fluxes_balances():
    # Zero horizontal gradient (main == upwind in T and q) => HA_T = HA_Q = 0,
    # regardless of the energy-balance imbalance captured by adv_in.
    main = {
        "H": np.array([54.0, 64.0]),
        "LE": np.array([30.0, 40.0]),
        "Rn": np.array([90.0, 110.0]),
        "G": np.array([5.0, 5.0]),
        "T": 22.0,
        "q": 0.009,
        "u": 2.5,
        "zm": 2.0,
        "h": 0.3,
    }
    upwind = {"T": 22.0, "q": 0.009}
    res = ax.compute_advection_fluxes(
        main_data=main, upwind_data=upwind, tower_distance=100.0
    )
    adv_in_expected = (main["H"] + main["LE"]) - (main["Rn"] - main["G"])
    np.testing.assert_allclose(res["adv_in"], adv_in_expected)
    np.testing.assert_allclose(res["HA_T"], [0.0, 0.0])
    np.testing.assert_allclose(res["HA_Q"], [0.0, 0.0])
    np.testing.assert_allclose(res["H_adv"], res["HA_T"])


def test_compute_advection_fluxes_oasis_hand_computed():
    """Oasis case: warm, dry air upwind must give HA_T < 0 (energy INTO field).

    Hand inputs: dT/dx = (25 - 30) / 100 = -0.05 K/m, dq/dx = (0.010 - 0.005)/100
    = 5e-5 (kg/kg)/m, u_bar = 2 m/s, layer (zm - h) = 1.5 m. With rho ~ 1.18,
    Cp ~ 1014, this is HA_T ~ -1.18*1014*2*0.05*1.5 ~ -179 W/m^2 (negative =
    advected INTO the field, per the Moderow OUT-positive convention).
    """
    distance = 100.0
    main = {
        "H": np.array([-30.0]),
        "LE": np.array([400.0]),
        "Rn": np.array([300.0]),
        "G": np.array([20.0]),
        "T": 25.0,  # cool transpiring field
        "q": 0.010,  # moist field
        "u": 2.0,
        "zm": 2.0,
        "h": 0.5,
        "P": 101325.0,
    }
    upwind = {"T": 30.0, "q": 0.005}  # warm, dry air upwind
    res = ax.compute_advection_fluxes(
        main_data=main, upwind_data=upwind, tower_distance=distance
    )

    # Independently reconstruct the expected W/m^2 from the same helpers.
    rho = ax.air_density(101325.0, 25.0, 0.010)
    Cp = ax.specific_heat_moist_air(0.010)
    Lv = ax.latent_heat_vaporization(25.0)
    layer = 2.0 - 0.5
    dT_dx = (25.0 - 30.0) / distance  # K/m (Kelvin diff == Celsius diff)
    dq_dx = (0.010 - 0.005) / distance
    expected_HA_T = rho * Cp * 2.0 * dT_dx * layer
    expected_HA_Q = rho * Lv * 2.0 * dq_dx * layer

    np.testing.assert_allclose(res["HA_T"], [expected_HA_T])
    np.testing.assert_allclose(res["HA_Q"], [expected_HA_Q])
    np.testing.assert_allclose(res["H_adv"], res["HA_T"])
    # Sign checks: warm upwind -> HA_T INTO field (negative); dry upwind -> HA_Q > 0.
    assert res["HA_T"][0] < 0.0
    assert res["HA_Q"][0] > 0.0
    # Sanity on the magnitude (~ -179 W/m^2).
    assert -200.0 < res["HA_T"][0] < -150.0


def test_oasis_signs():
    """Integration: a canonical oasis must produce a self-consistent set of signs.

    Oasis = warm dry air advected onto a cool wet transpiring surface. Built from
    a single input dict so all four signs are asserted for the *same* case:

    - warm upwind air (``T_upwind > T_main``) -> ``dT/dx < 0`` -> ``HA_T < 0``:
      sensible heat advected INTO the field (Moderow OUT-positive => negative).
    - dry upwind air (``q_upwind < q_main``) -> ``dq/dx > 0`` -> ``HA_Q > 0``:
      the drying signal of dry-air advection.
    - a stable internal boundary layer (cool surface, warm aloft) so the
      measurement-height temperature exceeds the column mean (``T_zm > <T>``),
      together with mean subsidence (``w_bar < 0``), gives
      ``VAT = rho*Cp*w_bar*(T_zm - <T>) < 0``: warm air carried DOWN, heat INTO
      the field.
    - latent heat above available energy (``EF = LE/(Rn-G) > 1``, the
      advective-input fingerprint) so ``H + LE > Rn - G`` and the raw closure
      residual ``(H + LE) - (Rn - G) > 0``.

    H is downward (negative) throughout -- the oasis sensible-heat signature.
    """
    distance = 100.0
    main = {
        "H": np.array([-30.0]),  # downward (negative) H -- oasis fingerprint
        "LE": np.array([400.0]),  # LE > Rn-G (EF > 1): advective input
        "Rn": np.array([300.0]),
        "G": np.array([20.0]),
        "T": 25.0,  # cool transpiring field; also T_zm at the measurement height
        "q": 0.010,  # moist field
        "u": 2.0,
        "zm": 2.0,
        "h": 0.5,
        "P": 101325.0,
        "w_bar": -0.03,  # planar-fit mean subsidence (downward)
        "T_col": 23.0,  # column mean cooler than T_zm -> warm aloft (stable IBL)
    }
    upwind = {"T": 30.0, "q": 0.005}  # warm, dry air upwind
    res = ax.compute_advection_fluxes(
        main_data=main, upwind_data=upwind, tower_distance=distance
    )

    # 1) Horizontal heat advection: warm upwind -> energy INTO field -> negative.
    assert res["HA_T"][0] < 0.0
    # 2) Horizontal moisture advection: dry upwind -> drying -> positive.
    assert res["HA_Q"][0] > 0.0
    # 3) Vertical heat advection: subsidence of warm-aloft air -> INTO -> negative.
    assert res["VAT"][0] < 0.0
    # 4) Raw closure residual (H+LE)-(Rn-G): EF > 1 advective input -> positive.
    assert res["residual"][0] > 0.0

    # Cross-checks: residual matches the explicit budget and its deprecated alias;
    # H_adv/V_adv alias HA_T/VAT. EF > 1 confirms the advective-input regime.
    assert res["residual"][0] == pytest.approx((-30.0 + 400.0) - (300.0 - 20.0))
    np.testing.assert_allclose(res["adv_in"], res["residual"])
    np.testing.assert_allclose(res["H_adv"], res["HA_T"])
    np.testing.assert_allclose(res["V_adv"], res["VAT"])
    assert main["LE"][0] > (main["Rn"][0] - main["G"][0])  # EF > 1


def test_compute_advection_fluxes_multi_upwind():
    distance = 50.0
    main = {
        "H": np.array([10.0, 10.0]),
        "LE": np.array([30.0, 30.0]),
        "Rn": np.array([40.0, 40.0]),
        "G": np.array([0.0, 0.0]),
        "wind_dir": np.array([0.0, 180.0]),
        "T": 20.0,
        "q": 0.008,
        "u": 3.0,
        "zm": 2.0,
        "h": 0.2,
    }
    up1 = {"T": 25.0, "q": 0.006, "bearing": 0.0}  # warm, dry  (N)
    up2 = {"T": 15.0, "q": 0.010, "bearing": 180.0}  # cool, moist (S)

    # t=0 wind from 0 deg -> up1 (warm upwind) -> HA_T < 0 (into field)
    # t=1 wind from 180 deg -> up2 (cool upwind) -> HA_T > 0 (out of field)
    res = ax.compute_advection_fluxes(
        main_data=main, upwind_data=[up1, up2], tower_distance=distance
    )

    rho = ax.air_density(101325.0, 20.0, 0.008)
    Cp = ax.specific_heat_moist_air(0.008)
    layer = 2.0 - 0.2
    exp_t0 = rho * Cp * 3.0 * ((20.0 - 25.0) / distance) * layer
    exp_t1 = rho * Cp * 3.0 * ((20.0 - 15.0) / distance) * layer
    np.testing.assert_allclose(res["HA_T"], [exp_t0, exp_t1])
    assert res["HA_T"][0] < 0.0 < res["HA_T"][1]


def test_compute_advection_fluxes_with_masks():
    distance = 80.0
    main = {
        "H": np.array([10.0, 10.0]),
        "LE": np.array([30.0, 30.0]),
        "Rn": np.array([40.0, 40.0]),
        "G": np.array([0.0, 0.0]),
        "T": 24.0,
        "q": 0.009,
        "u": 2.0,
        "zm": 2.0,
        "h": 0.4,
    }
    upwind = {"T": 28.0, "q": 0.005}
    det_h = np.array([True, False])

    res = ax.compute_advection_fluxes(
        main_data=main,
        upwind_data=upwind,
        detect_horizontal=det_h,
        tower_distance=distance,
    )

    # t=0 flagged -> compute gradient term; t=1 NOT flagged -> HA_T = HA_Q = 0.
    rho = ax.air_density(101325.0, 24.0, 0.009)
    Cp = ax.specific_heat_moist_air(0.009)
    layer = 2.0 - 0.4
    exp_t0 = rho * Cp * 2.0 * ((24.0 - 28.0) / distance) * layer
    np.testing.assert_allclose(res["HA_T"], [exp_t0, 0.0])
    np.testing.assert_allclose(res["HA_Q"][1], 0.0)
    assert res["HA_T"][0] < 0.0  # warm upwind -> into field


def test_compute_advection_fluxes_vertical_advection_hand_computed():
    """VAT = rho*Cp*w_bar*(T_zm - <T>) with known w_bar and a known gradient.

    T_zm = 25 C at the measurement height, column mean <T> = 27 C (warmer aloft,
    a stable/inversion profile), w_bar = +0.03 m/s (planar-fit). dT = -2 K, so
    VAT = rho*Cp*0.03*(-2) < 0 -- energy advected INTO the control volume
    (Moderow OUT-positive convention), the downward vertical-advection signal of
    the oasis regime.
    """
    main = {
        "H": np.array([-30.0]),
        "LE": np.array([400.0]),
        "Rn": np.array([300.0]),
        "G": np.array([20.0]),
        "T": 25.0,  # T_zm at the measurement height
        "q": 0.010,
        "u": 2.0,
        "zm": 2.0,
        "h": 0.5,
        "P": 101325.0,
        "w_bar": 0.03,  # planar-fit / tilt-corrected mean vertical velocity
        "T_col": 27.0,  # column-mean temperature <T>
    }
    upwind = {"T": 30.0, "q": 0.005}
    res = ax.compute_advection_fluxes(
        main_data=main, upwind_data=upwind, tower_distance=100.0
    )

    rho = ax.air_density(101325.0, 25.0, 0.010)
    Cp = ax.specific_heat_moist_air(0.010)
    expected_VAT = rho * Cp * 0.03 * (25.0 - 27.0)  # K diff == C diff
    np.testing.assert_allclose(res["VAT"], [expected_VAT])
    np.testing.assert_allclose(res["V_adv"], res["VAT"])  # backward-compat alias
    assert res["VAT"][0] < 0.0  # warmer aloft + upward w_bar -> energy INTO field
    # VAT must be the MEASURED term, NOT the closure residual.
    residual = (main["H"] + main["LE"]) - (main["Rn"] - main["G"])
    assert not np.allclose(res["VAT"], residual)


def test_compute_advection_fluxes_vertical_advection_from_profile():
    """<T> may be approximated from a vertical T profile (trapezoidal mean)."""
    main = {
        "H": np.array([20.0]),
        "LE": np.array([100.0]),
        "Rn": np.array([200.0]),
        "G": np.array([15.0]),
        "T": 24.0,  # T_zm
        "q": 0.009,
        "u": 2.5,
        "zm": 2.0,
        "h": 0.3,
        "w_bar": -0.04,
        # Linear profile 22 C (surface) -> 26 C (top) over 0..2 m => <T> = 24 C.
        "z_profile": np.array([0.0, 1.0, 2.0]),
        "T_profile": np.array([22.0, 24.0, 26.0]),
    }
    upwind = {"T": 27.0, "q": 0.006}
    res = ax.compute_advection_fluxes(
        main_data=main, upwind_data=upwind, tower_distance=120.0
    )
    rho = ax.air_density(101325.0, 24.0, 0.009)
    Cp = ax.specific_heat_moist_air(0.009)
    # Trapezoidal column mean of the linear profile is exactly 24 C, so
    # T_zm - <T> = 0 and VAT = 0 regardless of w_bar.
    expected_VAT = rho * Cp * (-0.04) * (24.0 - 24.0)
    np.testing.assert_allclose(res["VAT"], [expected_VAT])
    np.testing.assert_allclose(res["VAT"], [0.0])


def test_compute_advection_fluxes_vertical_flux_divergence():
    """VFD_T = -rho*Cp*(wT_zm - wT_h) when the two-level w'T' values are given."""
    main = {
        "H": np.array([40.0]),
        "LE": np.array([150.0]),
        "Rn": np.array([250.0]),
        "G": np.array([20.0]),
        "T": 22.0,
        "q": 0.008,
        "u": 3.0,
        "zm": 2.0,
        "h": 0.4,
        "wT_zm": 0.05,  # kinematic heat flux at the measurement height [K m/s]
        "wT_h": 0.08,  # kinematic heat flux at the canopy height [K m/s]
    }
    upwind = {"T": 24.0, "q": 0.006}
    res = ax.compute_advection_fluxes(
        main_data=main, upwind_data=upwind, tower_distance=90.0
    )
    rho = ax.air_density(101325.0, 22.0, 0.008)
    Cp = ax.specific_heat_moist_air(0.008)
    expected_VFD = -rho * Cp * (0.05 - 0.08)
    np.testing.assert_allclose(res["VFD_T"], [expected_VFD])
    assert res["VFD_T"][0] > 0.0  # flux decreasing with height -> convergence sign
    # No w_bar / detect_vertical supplied, so VAT itself is not engaged.
    assert res["VAT"] is None


def test_compute_advection_fluxes_residual_is_diagnostic_not_advection():
    """'residual' = (H+LE) - (Rn-G) is a diagnostic; 'adv_in' is a kept alias."""
    main = {
        "H": np.array([54.0, 64.0]),
        "LE": np.array([30.0, 40.0]),
        "Rn": np.array([90.0, 110.0]),
        "G": np.array([5.0, 5.0]),
        "T": 22.0,
        "q": 0.009,
        "u": 2.5,
        "zm": 2.0,
        "h": 0.3,
    }
    upwind = {"T": 22.0, "q": 0.009}
    res = ax.compute_advection_fluxes(
        main_data=main, upwind_data=upwind, tower_distance=100.0
    )
    expected_residual = (main["H"] + main["LE"]) - (main["Rn"] - main["G"])
    np.testing.assert_allclose(res["residual"], expected_residual)
    # 'adv_in' is retained as a deprecated alias with identical values.
    np.testing.assert_allclose(res["adv_in"], res["residual"])
    # No vertical inputs -> the residual is NOT relabelled as vertical advection.
    assert res["VAT"] is None
    assert res["V_adv"] is None


def test_compute_advection_fluxes_vertical_mask_zeroes_non_events():
    """detect_vertical engages VAT and zeroes the steps it flags False."""
    main = {
        "H": np.array([10.0, 10.0]),
        "LE": np.array([30.0, 30.0]),
        "Rn": np.array([40.0, 40.0]),
        "G": np.array([0.0, 0.0]),
        "T": 26.0,
        "q": 0.009,
        "u": 2.0,
        "zm": 2.0,
        "h": 0.4,
        "w_bar": 0.05,
        "T_col": 24.0,
    }
    upwind = {"T": 28.0, "q": 0.005}
    res = ax.compute_advection_fluxes(
        main_data=main,
        upwind_data=upwind,
        detect_vertical=np.array([False, True]),
        tower_distance=80.0,
    )
    rho = ax.air_density(101325.0, 26.0, 0.009)
    Cp = ax.specific_heat_moist_air(0.009)
    vat_event = rho * Cp * 0.05 * (26.0 - 24.0)
    np.testing.assert_allclose(res["VAT"], [0.0, vat_event])
    assert res["VAT"][1] > 0.0


def test_compute_advection_fluxes_raises_on_missing_fields():
    base = {
        "H": np.array([10.0]),
        "LE": np.array([30.0]),
        "Rn": np.array([40.0]),
        "G": np.array([0.0]),
        "T": 24.0,
        "q": 0.009,
        "u": 2.0,
        "zm": 2.0,
        "h": 0.4,
    }
    upwind = {"T": 28.0, "q": 0.005}

    # Missing upwind_data entirely -> raise (never silently zero).
    with pytest.raises(ValueError, match="upwind"):
        ax.compute_advection_fluxes(main_data=dict(base))

    # Missing tower_distance -> raise.
    with pytest.raises(ValueError, match="tower_distance"):
        ax.compute_advection_fluxes(main_data=dict(base), upwind_data=upwind)

    # Missing main temperature -> raise (HA_T cannot be a flux difference).
    no_T = {k: v for k, v in base.items() if k != "T"}
    with pytest.raises(ValueError, match="temperature"):
        ax.compute_advection_fluxes(
            main_data=no_T, upwind_data=upwind, tower_distance=80.0
        )

    # Missing humidity on the upwind tower -> raise.
    with pytest.raises(ValueError, match="humidity|'q'|'RH'"):
        ax.compute_advection_fluxes(
            main_data=dict(base), upwind_data={"T": 28.0}, tower_distance=80.0
        )

    # Swapped zm/h (non-positive layer depth) -> raise.
    bad_layer = dict(base)
    bad_layer["zm"], bad_layer["h"] = 0.4, 2.0
    with pytest.raises(ValueError, match="layer depth"):
        ax.compute_advection_fluxes(
            main_data=bad_layer, upwind_data=upwind, tower_distance=80.0
        )

    # Vertical engaged via detect_vertical but planar-fit w_bar missing -> raise
    # (never fall back to the residual).
    with pytest.raises(ValueError, match="w_bar"):
        ax.compute_advection_fluxes(
            main_data=dict(base),
            upwind_data=upwind,
            detect_vertical=np.array([True]),
            tower_distance=80.0,
        )

    # w_bar supplied but the column-mean temperature <T> missing -> raise.
    with_w = dict(base)
    with_w["w_bar"] = 0.05
    with pytest.raises(ValueError, match="column-mean|T_col|column"):
        ax.compute_advection_fluxes(
            main_data=with_w, upwind_data=upwind, tower_distance=80.0
        )


def test_apply_advection_correction_returns_keys():
    main = {
        "H": np.array([10.0, 20.0]),
        "LE": np.array([30.0, 40.0]),
        "Rn": np.array([60.0, 70.0]),
        "G": np.array([5.0, 5.0]),
    }
    zeros = np.zeros(2)
    out = ax.apply_advection_correction(main, zeros, zeros)
    expected_keys = {
        "Rn",
        "G",
        "H",
        "LE",
        "HA_T",
        "H_adv",
        "HA_Q",
        "VAT",
        "V_adv",
        "H_plus_LE_orig",
        "H_plus_LE_corrected",
        "available_energy",
        "residual_orig",
        "residual_corrected",
        "included",
    }
    assert expected_keys.issubset(out.keys())


def test_apply_advection_correction_gate_fails_leaves_step_unchanged():
    # Two steps, both with a turbulent under-closure gap (H+LE < Rn-G) and the
    # SAME nonzero advective terms. Step 0 has Rn below rn_min (gate condition 1
    # fails); step 1 has Rn above it. Only step 1 may be corrected.
    main = {
        "H": np.array([20.0, 20.0]),
        "LE": np.array([40.0, 40.0]),  # H+LE = 60
        "Rn": np.array([70.0, 200.0]),  # step 0: Rn < 75 (gate fails)
        "G": np.array([10.0, 10.0]),  # Rn-G = 60 (step0), 190 (step1)
    }
    HA_T = np.array([-15.0, -15.0])
    HA_Q = np.array([5.0, 5.0])
    VAT = np.array([20.0, 20.0])  # adv_total = +10 at every step

    out = ax.apply_advection_correction(main, HA_T, VAT, HA_Q=HA_Q, rn_min=75.0)

    # Step 0: Rn=70 < 75 -> gate fails -> left exactly uncorrected.
    assert out["included"][0] == False  # noqa: E712
    assert out["H_plus_LE_corrected"][0] == out["H_plus_LE_orig"][0] == 60.0
    assert out["residual_corrected"][0] == out["residual_orig"][0]

    # Step 1: Rn=200 > 75 and H+LE=60 < Rn-G=190 -> included; adv_total=+10
    # folded onto the turbulent side.
    assert out["included"][1] == True  # noqa: E712
    np.testing.assert_allclose(out["H_plus_LE_corrected"][1], 70.0)


def test_apply_advection_correction_oasis_step_improves_closure():
    # An advective-input ("oasis") daytime step with an under-closure gap:
    # warm/dry air advects energy the EC system did not capture in H+LE, so the
    # measured turbulent sum sits below the available energy. Folding the gated
    # advective terms in must move (H+LE) toward (Rn-G) and shrink |residual|.
    main = {
        "H": np.array([30.0]),
        "LE": np.array([120.0]),  # H+LE = 150
        "Rn": np.array([420.0]),  # Rn > rn_min
        "G": np.array([40.0]),  # Rn-G = 380 -> gap of 230 (under-closure)
    }
    # Net advective input of +120 W/m^2 (still short of fully closing the gap,
    # so no overshoot): HA_T + HA_Q + VAT = 70 + 10 + 40 = 120.
    HA_T = np.array([70.0])
    HA_Q = np.array([10.0])
    VAT = np.array([40.0])

    out = ax.apply_advection_correction(main, HA_T, VAT, HA_Q=HA_Q)

    assert out["included"][0] == True  # noqa: E712
    # Available energy is untouched; only the turbulent sum moves.
    np.testing.assert_allclose(out["available_energy"][0], 380.0)
    np.testing.assert_allclose(out["H_plus_LE_orig"][0], 150.0)
    np.testing.assert_allclose(out["H_plus_LE_corrected"][0], 270.0)
    # Corrected sum is strictly closer to available energy than the original.
    gap_before = abs(out["available_energy"][0] - out["H_plus_LE_orig"][0])
    gap_after = abs(out["available_energy"][0] - out["H_plus_LE_corrected"][0])
    assert gap_after < gap_before
    # Equivalently, |residual| shrinks toward zero.
    assert abs(out["residual_corrected"][0]) < abs(out["residual_orig"][0])


def test_apply_advection_correction_nan_term_does_not_poison_sum():
    # A NaN in one advective component contributes 0 rather than NaN-ing the
    # whole corrected sum; the other terms still apply at a gated step.
    main = {
        "H": np.array([30.0]),
        "LE": np.array([120.0]),
        "Rn": np.array([420.0]),
        "G": np.array([40.0]),
    }
    out = ax.apply_advection_correction(
        main,
        np.array([70.0]),  # HA_T
        np.array([40.0]),  # VAT
        HA_Q=np.array([np.nan]),  # missing moisture term
    )
    assert out["included"][0] == True  # noqa: E712
    # Only HA_T + VAT = 110 is folded in (HA_Q NaN -> 0).
    np.testing.assert_allclose(out["H_plus_LE_corrected"][0], 260.0)
    assert np.isfinite(out["H_plus_LE_corrected"][0])
