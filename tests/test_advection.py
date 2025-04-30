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
    """Gs should equal depth * Csoil * dT/dt."""
    Csoil = 2_000_000.0  # J/m³·K
    dT_dt = 1.0e-3  # K/s
    depth = 0.05  # m
    expected = depth * Csoil * dT_dt
    out = ax.compute_soil_heat_storage_flux(Csoil, dT_dt, depth)
    assert pytest.approx(out) == expected


def test_total_ground_heat_flux():
    """Storage‐corrected flux should be additive."""
    Gd = -15.0
    Gs = 5.0
    assert ax.total_ground_heat_flux(Gd, Gs) == Gd + Gs


def test_compute_bowen_ratio_variance():
    sigma_T = 0.6  # K
    sigma_q = 0.003  # kg/kg
    Cp = 1005.0
    Lv = 2.45e6
    expected = (Cp / Lv) * (sigma_T / sigma_q)
    out = ax.compute_bowen_ratio_variance(sigma_T, sigma_q, Cp=Cp, Lv=Lv)
    assert pytest.approx(out) == expected


def test_correct_sonic_heat_flux():
    w_Ts = 0.12  # K·m s⁻¹
    T_mean = 293.15  # K
    beta = 1.5
    Lv = 2.45e6
    Cp = 1005.0
    denom = 1 + 0.51 * (Cp * T_mean) / (Lv * beta)
    expected = w_Ts / denom
    out = ax.correct_sonic_heat_flux(w_Ts, T_mean, beta, Cp=Cp, Lv=Lv)
    assert pytest.approx(out) == expected


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
    assert flags[0] is True


def test_detect_vertical_advection():
    temp_lower = [15.0, 20.0]
    temp_upper = [17.0, 19.0]  # inversion only first time step
    vertical_w = [-0.1, 0.0]
    main_H = [5.0, 100.0]
    rn = [300.0, 300.0]
    g = [50.0, 50.0]
    flags = ax.detect_vertical_advection(
        temp_profile_lower=temp_lower,
        temp_profile_upper=temp_upper,
        vertical_w=vertical_w,
        main_H=main_H,
        rn=rn,
        g=g,
    )
    assert np.array_equal(flags, np.array([True, False]))


def test_compute_advection_fluxes_balances():
    main = {
        "H": np.array([10.0, 20.0]),
        "LE": np.array([30.0, 40.0]),
        "Rn": np.array([60.0, 70.0]),
        "G": np.array([5.0, 5.0]),
    }
    res = ax.compute_advection_fluxes(main_data=main)
    adv_in_expected = (main["H"] + main["LE"]) - (main["Rn"] - main["G"])
    np.testing.assert_allclose(res["adv_in"], adv_in_expected)
    assert np.all(res["H_adv"] == 0.0)
    assert np.all(res["V_adv"] == 0.0)


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
        "H_adv",
        "V_adv",
        "H_plus_LE_orig",
        "H_plus_LE_corrected",
    }
    assert expected_keys.issubset(out.keys())
