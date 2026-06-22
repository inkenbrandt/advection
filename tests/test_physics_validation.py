#!/usr/bin/env python
"""Physics-validation suite for the ``advection`` package.

Where ``test_advection.py`` checks each function against its own formula and
``test_golden_characterization.py`` pins exact numeric snapshots, this module
validates the *physics contract* in ``CLAUDE.md`` at a higher level:

1. **Unit / dimension checks** -- every flux function returns a finite value in
   a physically plausible ``W/m^2`` envelope for representative **SI** inputs,
   and is **dimensionally homogeneous**: scaling a dimensional input scales the
   output by the same factor (e.g. ``H = rho*Cp*w'T'`` is linear in ``rho``,
   ``Cp`` and ``w'T'``, confirming ``[kg/m^3][J/(kg K)][K m/s] = [W/m^2]``).
2. **Sign checks** -- the canonical oasis (warm dry air advected onto a cool
   transpiring surface) yields **negative H**, **negative HA_T** and
   **negative VAT** under the Moderow et al. (2021) OUT-positive convention.
3. **Conservation** -- with zero advection on a *closed* synthetic dataset the
   residual is ~0 and the advection correction is the identity
   (``corrected == uncorrected``).
4. **Reference-value checks** -- afternoon horizontal heat advection is on the
   order of *tens of negative* ``W/m^2`` (Wang et al. 2024 scale), and a gated
   synthetic correction lifts closure from ~0.89 toward ~0.97 (the alfalfa-study
   result that motivates the conditional-inclusion rule).

The import path to the in-tree package is set up in ``tests/conftest.py``.
"""

import numpy as np
import pytest

import advection as ax

# -----------------------------------------------------------------------------
# Representative SI inputs (one consistent near-surface midday state).
# -----------------------------------------------------------------------------
P_SI = 101325.0  # Pa
T_K = 298.15  # K  (25 C)
Q_SI = 0.010  # kg/kg
RHO_SI = ax.air_density(P_SI, T_K, Q_SI)  # kg/m^3  (~1.18)
CP_SI = ax.specific_heat_moist_air(Q_SI)  # J/(kg K) (~1015)
LV_SI = ax.latent_heat_vaporization(T_K)  # J/kg     (~2.44e6)
ZM_SI = 2.0  # m
H_CANOPY_SI = 0.5  # m
LAYER_SI = ZM_SI - H_CANOPY_SI  # m

# A generous magnitude envelope for surface-energy / advective fluxes. Real
# terms rarely leave +-1000 W/m^2 for representative inputs; 2000 keeps the
# bound from being flaky while still catching a unit blunder (e.g. forgetting
# Cp would put H near 0.06 W/m^2; using J instead of W would explode it).
FLUX_ENVELOPE = 2000.0


def _is_wm2(value):
    """A scalar is a plausible W/m^2 flux: finite and within the envelope."""
    v = float(value)
    return np.isfinite(v) and abs(v) <= FLUX_ENVELOPE


def _oasis_main(**overrides):
    """Canonical oasis main-tower state: cool wet surface, downward H, EF > 1.

    Warm dry air aloft/upwind, mean subsidence (``w_bar < 0``) and a stable
    internal boundary layer (``T_zm > <T>``) -- the configuration that must
    produce negative ``HA_T`` and negative ``VAT``.
    """
    main = {
        "H": np.array([-30.0]),  # downward sensible heat (oasis fingerprint)
        "LE": np.array([400.0]),  # LE > Rn - G  (EF > 1, advective input)
        "Rn": np.array([300.0]),
        "G": np.array([20.0]),
        "T": 25.0,  # cool transpiring field; also T_zm
        "q": 0.010,  # moist field
        "u": 2.0,
        "zm": 2.0,
        "h": 0.5,
        "P": P_SI,
        "w_bar": -0.03,  # planar-fit mean subsidence (downward)
        "T_col": 23.0,  # column mean cooler than T_zm -> warm aloft
    }
    main.update(overrides)
    return main


# =============================================================================
# 1. UNIT / DIMENSION CHECKS -- every flux function returns W/m^2
# =============================================================================


def test_units_soil_heat_storage_flux_is_wm2():
    # Gs = Cs * dz * dT/dt : [J/(m^3 K)] * [m] * [K/s] = [W/m^2].
    Csoil, dz, dT_dt = 2.1e6, 0.05, 1.0e-3
    gs = ax.compute_soil_heat_storage_flux(Csoil, dT_dt, dz)
    assert _is_wm2(gs)
    # Homogeneity: linear in each of the three dimensional factors.
    assert ax.compute_soil_heat_storage_flux(2 * Csoil, dT_dt, dz) == pytest.approx(
        2 * gs
    )
    assert ax.compute_soil_heat_storage_flux(Csoil, 2 * dT_dt, dz) == pytest.approx(
        2 * gs
    )
    assert ax.compute_soil_heat_storage_flux(Csoil, dT_dt, 2 * dz) == pytest.approx(
        2 * gs
    )


def test_units_total_ground_heat_flux_is_wm2():
    # G = Gd + Gs : sum of two W/m^2 terms is W/m^2.
    g = ax.total_ground_heat_flux(-15.0, 5.0)
    assert _is_wm2(g)
    assert g == pytest.approx(-10.0)


def test_units_air_heat_storage_is_wm2():
    # J = rho*Cp*dT/dt*(zm-h) : [kg/m^3][J/(kg K)][K/s][m] = [W/m^2].
    j = ax.air_heat_storage(RHO_SI, CP_SI, 1.0e-3, ZM_SI, H_CANOPY_SI)
    assert _is_wm2(j)
    # Homogeneity in rho, Cp, dT/dt and the layer depth.
    assert ax.air_heat_storage(2 * RHO_SI, CP_SI, 1.0e-3, ZM_SI, H_CANOPY_SI) == (
        pytest.approx(2 * j)
    )
    assert ax.air_heat_storage(RHO_SI, CP_SI, 2.0e-3, ZM_SI, H_CANOPY_SI) == (
        pytest.approx(2 * j)
    )
    # Doubling the layer depth (zm - h) doubles the storage.
    j2 = ax.air_heat_storage(
        RHO_SI, CP_SI, 1.0e-3, H_CANOPY_SI + 2 * LAYER_SI, H_CANOPY_SI
    )
    assert j2 == pytest.approx(2 * j)


def test_units_sensible_heat_flux_is_wm2_and_linear():
    # H = rho * Cp * w'T' : [kg/m^3][J/(kg K)][K m/s] = [W/m^2].
    wT = 0.05
    h = ax.compute_sensible_heat_flux(wT, RHO_SI, Cp=CP_SI)
    assert _is_wm2(h)
    assert h > 0.0  # upward kinematic flux -> positive (OUT) H
    # Dimensional homogeneity: linear in rho, Cp and w'T'.
    assert ax.compute_sensible_heat_flux(wT, 2 * RHO_SI, Cp=CP_SI) == pytest.approx(
        2 * h
    )
    assert ax.compute_sensible_heat_flux(wT, RHO_SI, Cp=2 * CP_SI) == pytest.approx(
        2 * h
    )
    assert ax.compute_sensible_heat_flux(2 * wT, RHO_SI, Cp=CP_SI) == pytest.approx(
        2 * h
    )


def test_units_latent_heat_flux_residual_is_wm2():
    le = ax.latent_heat_flux_residual(450.0, 50.0, 200.0)
    assert _is_wm2(le)
    assert le == pytest.approx(200.0)


def test_units_latent_heat_flux_bowen_is_wm2():
    le = ax.latent_heat_flux_bowen(450.0, 50.0, 0.7)
    assert _is_wm2(le)
    assert le > 0.0


def test_units_wpl_latent_heat_flux_is_wm2_and_scales_with_Lv():
    # LE = Lv * (1 + mu*MR) * [w'rho_v' + (rho_v/T) w'T'] : Lv [J/kg] times a
    # mass flux [kg m^-2 s^-1] = [W/m^2].
    w_rhov, wT, rho_v, MR = 1.0e-4, 0.05, 0.011, 0.008
    le = ax.wpl_latent_heat_flux(w_rhov, wT, rho_v, T_K, MR, Lv=LV_SI)
    assert _is_wm2(le)
    # Homogeneity: LE is exactly linear in Lv.
    le2 = ax.wpl_latent_heat_flux(w_rhov, wT, rho_v, T_K, MR, Lv=2 * LV_SI)
    assert le2 == pytest.approx(2 * le)


def test_units_horizontal_and_vertical_advection_terms_are_wm2():
    # All of HA_T, HA_Q, VAT, VFD_T must come back in W/m^2.
    main = _oasis_main(wT_zm=0.05, wT_h=0.08)
    res = ax.compute_advection_fluxes(
        main_data=main, upwind_data={"T": 30.0, "q": 0.005}, tower_distance=100.0
    )
    for key in ("HA_T", "HA_Q", "VAT", "VFD_T"):
        arr = np.asarray(res[key], dtype=float)
        assert arr.shape == (1,)
        assert all(_is_wm2(v) for v in arr)


def test_units_horizontal_advection_linear_in_wind_and_layer_depth():
    # HA_T = rho*Cp*u*(dT/dx)*(zm-h): doubling u or the layer depth doubles HA_T.
    base = dict(main_data=_oasis_main(u=2.0), upwind_data={"T": 30.0, "q": 0.005})
    ha = ax.compute_advection_fluxes(tower_distance=100.0, **base)["HA_T"][0]

    faster = ax.compute_advection_fluxes(
        main_data=_oasis_main(u=4.0),
        upwind_data={"T": 30.0, "q": 0.005},
        tower_distance=100.0,
    )["HA_T"][0]
    assert faster == pytest.approx(2 * ha)

    # Doubling the layer depth (zm - h): h fixed, zm so that (zm - h) doubles.
    deeper = ax.compute_advection_fluxes(
        main_data=_oasis_main(u=2.0, zm=0.5 + 2 * (2.0 - 0.5), h=0.5),
        upwind_data={"T": 30.0, "q": 0.005},
        tower_distance=100.0,
    )["HA_T"][0]
    assert deeper == pytest.approx(2 * ha)

    # Doubling the fetch (tower_distance) halves the gradient and so halves HA_T.
    farther = ax.compute_advection_fluxes(
        main_data=_oasis_main(u=2.0),
        upwind_data={"T": 30.0, "q": 0.005},
        tower_distance=200.0,
    )["HA_T"][0]
    assert farther == pytest.approx(ha / 2.0)


def test_units_closure_methods_return_wm2():
    # Twine (2000) closure fluxes are W/m^2. Use a non-oasis (LE < avail) case so
    # bowen_ratio_closure does not (correctly) warn.
    out = ax.bowen_ratio_closure(500.0, 50.0, 100.0, 300.0)
    assert _is_wm2(out["H"]) and _is_wm2(out["LE"])
    le = ax.residual_le_closure(500.0, 50.0, 120.0)
    assert _is_wm2(le)
    resid = ax.energy_balance_residual(500.0, 50.0, 100.0, 300.0)
    assert _is_wm2(resid)


# =============================================================================
# 2. SIGN CHECKS -- the oasis yields negative H, negative HA_T, negative VAT
# =============================================================================


def test_sign_oasis_negative_sensible_heat_flux():
    # Oasis: warm dry air advects DOWN onto a cool surface -> downward (negative)
    # kinematic heat flux -> negative H (energy INTO the control volume).
    wT_down = -0.02  # K m/s, downward
    h = ax.compute_sensible_heat_flux(wT_down, RHO_SI, Cp=CP_SI)
    assert h < 0.0


def test_sign_oasis_negative_bowen_ratio_fingerprint():
    # Anticorrelated warm-dry / cool-moist T',q' (corr < 0) -> negative beta,
    # the oasis fingerprint (Wang 2024 Eq. 8).
    rng = np.random.default_rng(0)
    base = rng.standard_normal(4000)
    T_prime = 0.5 * base + 0.05 * rng.standard_normal(4000)
    q_prime = -0.002 * base + 2e-4 * rng.standard_normal(4000)  # anticorrelated
    beta = ax.compute_bowen_ratio_variance(T_prime=T_prime, q_prime=q_prime)
    assert beta < 0.0


def test_sign_oasis_negative_HA_T_and_negative_VAT():
    # One canonical oasis case -> both advective heat terms are negative
    # (energy INTO the field) under the Moderow OUT-positive convention.
    res = ax.compute_advection_fluxes(
        main_data=_oasis_main(),
        upwind_data={"T": 30.0, "q": 0.005},  # warm, dry upwind
        tower_distance=100.0,
    )
    assert res["HA_T"][0] < 0.0  # warm upwind -> dT/dx < 0 -> HA_T < 0
    assert res["VAT"][0] < 0.0  # subsidence of warm-aloft air -> VAT < 0
    # The moisture term has the opposite (drying) sign: dry upwind -> HA_Q > 0.
    assert res["HA_Q"][0] > 0.0


def test_sign_oasis_full_set_is_self_consistent():
    # H (input), HA_T and VAT all negative for the SAME oasis state; the closure
    # residual (H+LE)-(Rn-G) is positive (EF > 1 advective input).
    main = _oasis_main()
    res = ax.compute_advection_fluxes(
        main_data=main, upwind_data={"T": 30.0, "q": 0.005}, tower_distance=100.0
    )
    assert main["H"][0] < 0.0  # negative (downward) sensible heat
    assert res["HA_T"][0] < 0.0
    assert res["VAT"][0] < 0.0
    assert res["residual"][0] > 0.0  # EF > 1: H + LE > Rn - G
    assert main["LE"][0] > (main["Rn"][0] - main["G"][0])  # EF > 1 explicitly


# =============================================================================
# 3. CONSERVATION -- closed dataset, zero advection
# =============================================================================


def _closed_dataset(seed, n=48):
    """Synthetic, perfectly closed dataset: H + LE == Rn - G at every step."""
    rng = np.random.default_rng(seed)
    Rn = rng.uniform(100.0, 600.0, n)
    G = rng.uniform(20.0, 80.0, n)
    available = Rn - G
    H = 0.3 * available
    LE = 0.7 * available  # H + LE == available exactly
    return Rn, G, H, LE


def test_conservation_closed_dataset_residual_is_zero():
    Rn, G, H, LE = _closed_dataset(seed=42)
    # CLAUDE.md residual = Rn - G - H - LE == 0 for a closed budget.
    resid = ax.energy_balance_residual(Rn, G, H, LE)
    np.testing.assert_allclose(resid, 0.0, atol=1e-9)
    # EBR == 1 and the closure slope is exactly 1 (intercept 0, R^2 == 1).
    assert ax.energy_balance_ratio(H, LE, Rn, G) == pytest.approx(1.0)
    cs = ax.closure_slope(H, LE, Rn, G)
    assert cs["slope"] == pytest.approx(1.0, abs=1e-9)
    assert cs["intercept"] == pytest.approx(0.0, abs=1e-6)
    assert cs["r_squared"] == pytest.approx(1.0, abs=1e-9)


def test_conservation_zero_advection_correction_is_identity():
    Rn, G, H, LE = _closed_dataset(seed=7)
    n = Rn.size
    main = {"H": H, "LE": LE, "Rn": Rn, "G": G}
    out = ax.apply_advection_correction(
        main, np.zeros(n), np.zeros(n), HA_Q=np.zeros(n)
    )
    # corrected == uncorrected, and the residual stays ~0 (closed dataset).
    np.testing.assert_allclose(
        out["H_plus_LE_corrected"], out["H_plus_LE_orig"], atol=1e-12
    )
    np.testing.assert_allclose(
        out["residual_corrected"], out["residual_orig"], atol=1e-12
    )
    np.testing.assert_allclose(out["residual_corrected"], 0.0, atol=1e-9)


def test_conservation_zero_horizontal_gradient_gives_zero_advection():
    # main == upwind in T and q -> dT/dx = dq/dx = 0 -> HA_T = HA_Q = 0,
    # independent of the (nonzero) closure imbalance.
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
    upwind = {"T": 22.0, "q": 0.009}  # identical state -> no gradient
    res = ax.compute_advection_fluxes(
        main_data=main, upwind_data=upwind, tower_distance=100.0
    )
    np.testing.assert_allclose(res["HA_T"], 0.0, atol=1e-12)
    np.testing.assert_allclose(res["HA_Q"], 0.0, atol=1e-12)


def test_conservation_zero_vertical_velocity_gives_zero_VAT():
    # w_bar = 0 -> VAT = rho*Cp*0*(T_zm - <T>) = 0 regardless of the gradient.
    main = _oasis_main(w_bar=0.0)
    res = ax.compute_advection_fluxes(
        main_data=main, upwind_data={"T": 30.0, "q": 0.005}, tower_distance=100.0
    )
    np.testing.assert_allclose(res["VAT"], 0.0, atol=1e-12)


# =============================================================================
# 4. REFERENCE-VALUE CHECKS -- anchored to the literature
# =============================================================================


def test_reference_afternoon_horizontal_advection_tens_of_negative_wm2():
    # A realistic afternoon oasis fetch: warm dry air 2 K warmer 200 m upwind of
    # a cool transpiring field, u = 3 m/s. Wang et al. (2024) report horizontal
    # heat advection on the order of tens of W/m^2; the sign is negative (energy
    # INTO the field) for warm upwind air.
    main = {
        "H": np.array([-20.0]),
        "LE": np.array([450.0]),
        "Rn": np.array([500.0]),
        "G": np.array([60.0]),
        "T": 24.0,  # cool field
        "q": 0.012,
        "u": 3.0,
        "zm": 2.0,
        "h": 0.5,
        "P": P_SI,
    }
    upwind = {"T": 26.0, "q": 0.008}  # warm, dry upwind (2 K over 200 m)
    res = ax.compute_advection_fluxes(
        main_data=main, upwind_data=upwind, tower_distance=200.0
    )
    HA_T = res["HA_T"][0]
    assert HA_T < 0.0
    assert -100.0 < HA_T < -10.0  # tens of negative W/m^2


def test_reference_closure_improves_from_089_toward_097():
    # Synthetic daytime dataset that is under-closed at EBR ~ 0.89 (the typical
    # eddy-covariance gap). Folding in the gated MEASURED advective input lifts
    # closure toward ~0.97 -- the alfalfa-study improvement (Wang et al. 2024)
    # that motivates the conditional-inclusion rule (Rn > 75 AND H+LE < Rn-G).
    n = 24
    available = np.full(n, 400.0)  # Rn - G  [W/m^2]
    Rn = np.full(n, 450.0)  # all daytime, Rn > rn_min (75)
    G = Rn - available  # 50
    H_plus_LE = 0.89 * available  # under-closure: EBR = 0.89
    H = 0.30 * H_plus_LE
    LE = 0.70 * H_plus_LE
    main = {"H": H, "LE": LE, "Rn": Rn, "G": G}

    ebr_before = ax.energy_balance_ratio(H, LE, Rn, G)
    assert ebr_before == pytest.approx(0.89, abs=1e-9)

    # Measured advective input summing to +32 W/m^2 per step (0.08 * 400), the
    # amount that lifts the turbulent sum from 0.89 to 0.97 of available energy.
    HA_T = np.full(n, 18.0)
    HA_Q = np.full(n, 6.0)
    VAT = np.full(n, 8.0)
    out = ax.apply_advection_correction(main, HA_T, VAT, HA_Q=HA_Q, rn_min=75.0)

    # Every step is daytime and under-closed -> all gated in.
    assert np.all(out["included"])

    ebr_after = np.sum(out["H_plus_LE_corrected"]) / np.sum(out["available_energy"])
    assert ebr_after == pytest.approx(0.97, abs=1e-9)
    # Closure strictly improved and moved closer to perfect (EBR = 1).
    assert ebr_before < ebr_after <= 1.0
    assert abs(1.0 - ebr_after) < abs(1.0 - ebr_before)


def test_reference_gate_blocks_correction_outside_conditional_inclusion():
    # The 0.89 -> 0.97 improvement depends on Wang's gate. A night/low-Rn step
    # (Rn < rn_min) must be left exactly uncorrected even with nonzero advection,
    # so closure is NOT spuriously "improved" outside the daytime regime.
    main = {
        "H": np.array([20.0, 20.0]),
        "LE": np.array([40.0, 40.0]),  # H + LE = 60
        "Rn": np.array([70.0, 200.0]),  # step 0: Rn < 75 (gate fails)
        "G": np.array([10.0, 10.0]),  # available = 60, 190
    }
    HA_T = np.array([18.0, 18.0])
    HA_Q = np.array([6.0, 6.0])
    VAT = np.array([8.0, 8.0])
    out = ax.apply_advection_correction(main, HA_T, VAT, HA_Q=HA_Q, rn_min=75.0)

    assert out["included"].tolist() == [False, True]
    # Step 0 untouched; step 1 gets the +32 W/m^2 advective input.
    np.testing.assert_allclose(out["H_plus_LE_corrected"][0], 60.0)
    np.testing.assert_allclose(out["H_plus_LE_corrected"][1], 92.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
