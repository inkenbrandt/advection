#!/usr/bin/env python
"""Characterization (golden) tests for the CURRENT public API.

These tests *snapshot* the present numeric outputs of the public functions in
``advection.advection`` and ``advection.advect_detect`` exactly as they behave
today. They intentionally hard-code literal expected values (rather than
re-deriving them from the same formula the source uses) so that any future
refactor that changes a result will surface as an explicit, reviewable delta.

Each assertion carries an inline note on whether the snapshotted value looks
**physically correct** or **SUSPECT** when judged against the physics contract
in ``CLAUDE.md`` (sign convention, Wang 2024 equations, "never compute an
advection term as a residual", "never force Bowen closure when LE > Rn-G").

NOTE: a "golden" value being asserted here does NOT mean it is endorsed as
correct physics. Where the current behavior contradicts the contract, the
SUSPECT comment records the disagreement; the test still pins today's output so
the eventual fix shows up as an intended change.

The package is installed in editable mode (see pyproject ``package-dir = src``),
so a plain ``import advection`` resolves to ``src/advection``.
"""

import math

import numpy as np
import pytest

import advection as ax

# =============================================================================
# advection.py
# =============================================================================


def test_golden_compute_bowen_ratio_variance():
    # |beta| = (Cp/Lv) * (sigma_T/sigma_q) = (1005/2.45e6) * (0.6/0.003).
    # No sign source supplied, so the UNSIGNED magnitude is returned (and a
    # warning is emitted that the sign is undetermined).
    with pytest.warns(UserWarning):
        out = ax.compute_bowen_ratio_variance(0.6, 0.003, Cp=1005.0, Lv=2.45e6)
    assert out == pytest.approx(0.08204081632653061)
    # PHYSICS: magnitude matches Wang 2024 Eq. 8. The sign is now derived from
    # sign(corr(T',q')) when a correlation/covariance or fluctuation series is
    # supplied (see test_golden_compute_bowen_ratio_variance_sign_from_*); the
    # backward-compatible magnitude-only path warns rather than guessing.


def test_golden_compute_bowen_ratio_variance_default_Lv():
    # Lv defaults to 2.45e6 when neither Lv nor T is given, so this must match
    # the explicit-Lv call above. Still magnitude-only -> warns.
    with pytest.warns(UserWarning):
        out = ax.compute_bowen_ratio_variance(0.6, 0.003)
    assert out == pytest.approx(0.08204081632653061)
    # PHYSICS: correct — the documented default Lv (~20 C) is applied.


def test_golden_compute_bowen_ratio_variance_magnitude_is_unsigned():
    # A negative sigma_q is unphysical (a std dev is >= 0); the function now
    # returns the non-negative magnitude (abs) rather than propagating the sign,
    # because the sign must come from corr(T', q'), not the std-dev inputs.
    with pytest.warns(UserWarning):
        out = ax.compute_bowen_ratio_variance(0.5, -0.002)
    assert out == pytest.approx(0.10255102040816326)
    # PHYSICS: correct — magnitude only; sign is undetermined without a
    # correlation/covariance, hence the warning.


def test_golden_compute_bowen_ratio_variance_sign_from_correlation():
    # With an anticorrelated T,q (oasis fingerprint) the sign is negative.
    out = ax.compute_bowen_ratio_variance(
        0.6, 0.003, Cp=1005.0, Lv=2.45e6, corr_Tq=-0.8
    )
    assert out == pytest.approx(-0.08204081632653061)
    # PHYSICS: correct — negative beta = downward H = advection into the field,
    # per CLAUDE.md / Moderow 2021.


def test_golden_correct_sonic_heat_flux():
    # wT = wTs / (1 + 0.51 * Cp * T / (Lv * beta)), T in KELVIN (Wang Eq. 9).
    out = ax.correct_sonic_heat_flux(0.12, 293.15, 1.5, Cp=1005.0, Lv=2.45e6)
    assert out == pytest.approx(0.11528646104368233)
    # PHYSICS: correct. Factor ~1.041 reduces wTs slightly, as expected for a
    # positive Bowen ratio.


def test_golden_correct_sonic_heat_flux_zero_beta():
    # beta == 0 hits the guard branch -> factor forced to 1.0 -> wTs returned
    # unchanged.
    out = ax.correct_sonic_heat_flux(0.12, 293.15, 0.0)
    assert out == 0.12
    # SUSPECT: the true mathematical limit as beta -> 0 is factor -> +inf, hence
    # wT -> 0 (an all-latent surface). The guard instead returns the *fully
    # uncorrected* flux (factor = 1), the opposite extreme — a discontinuity at
    # beta = 0.


def test_golden_correct_sonic_heat_flux_negative_beta():
    # Oasis-type negative beta makes the correction factor < 1, amplifying wTs.
    out = ax.correct_sonic_heat_flux(0.10, 300.0, -0.5, Cp=1005.0, Lv=2.45e6)
    assert out == pytest.approx(0.11435399328812074)
    # PHYSICS: arithmetic correct per Eq. 9. Note the singularity at the beta
    # value where the denominator factor crosses 0 is not guarded.


def test_golden_compute_sensible_heat_flux():
    # H = rho * Cp * wT = 1.2 * 1005 * 0.05
    out = ax.compute_sensible_heat_flux(0.05, 1.2, Cp=1005.0)
    assert out == pytest.approx(60.3)
    # PHYSICS: correct. Straightforward H = rho*Cp*w'T'.


def test_golden_latent_heat_flux_residual():
    # LE = Rn - G - H = 450 - 50 - 200
    out = ax.latent_heat_flux_residual(450.0, 50.0, 200.0)
    assert out == 200.0
    # PHYSICS: correct as the residual-closure LE. (This is LE-as-residual, a
    # legitimate standard method — distinct from the contract's prohibition on
    # computing an *advection* term as a residual.)


def test_golden_latent_heat_flux_bowen():
    # LE = (Rn - G) / (1 + beta) = 400 / 1.7
    out = ax.latent_heat_flux_bowen(450.0, 50.0, 0.7)
    assert out == pytest.approx(235.29411764705884)
    # PHYSICS: correct for a positive Bowen ratio.


def test_golden_latent_heat_flux_bowen_oasis_beta():
    # Negative (oasis) beta drives LE above the available energy (Rn - G = 400).
    out = ax.latent_heat_flux_bowen(450.0, 50.0, -0.3)
    assert out == pytest.approx(571.4285714285714)
    # SUSPECT: this yields LE > (Rn - G), exactly the case CLAUDE.md says must
    # NOT be forced via Bowen closure. Also numerically unstable: as beta -> -1
    # the result diverges. Pinned here to document the unguarded behavior.


def test_golden_air_density_moist():
    out = ax.air_density(101325.0, 298.15, 0.008)
    assert out == pytest.approx(1.1781969712523759)
    # PHYSICS: correct (~1.18 kg/m^3 at 25 C, sea-level pressure).


def test_golden_air_density_moist_vs_dry():
    moist = ax.air_density(101325.0, 298.15, 0.008)
    dry = ax.air_density(101325.0, 298.15, 0.0)
    assert dry == pytest.approx(1.1839251532625141)
    assert moist < dry
    # PHYSICS: correct — moist air is less dense than dry air at the same T, P
    # (water vapor is lighter than dry air). Good sanity sign.


def test_golden_latent_heat_vaporization_celsius():
    # Lv(20 C) via the polynomial fit.
    out = ax.latent_heat_vaporization(20.0)
    assert out == pytest.approx(2453760.0)
    # PHYSICS: correct (~2.454e6 J/kg at 20 C).


def test_golden_latent_heat_vaporization_kelvin_autoconvert():
    # T > 100 is treated as Kelvin and converted to C, so 293.15 K == 20 C.
    out = ax.latent_heat_vaporization(293.15)
    assert out == pytest.approx(2453760.0)
    # PHYSICS: correct, but the T>100 heuristic is brittle — a genuine 150 C
    # would be misread as Kelvin. Fine inside the documented 0-40 C range.


def test_golden_latent_heat_vaporization_zero_celsius():
    out = ax.latent_heat_vaporization(0.0)
    assert out == pytest.approx(2500800.0)
    # PHYSICS: correct (~2.501e6 J/kg at 0 C) and > Lv(20 C), i.e. Lv decreases
    # with temperature as expected.


def test_golden_rh_to_specific_humidity_percent():
    # 50% RH at 20 C.
    out = ax.rh_to_specific_humidity(50.0, 20.0)
    assert float(out) == pytest.approx(0.007204269021259108)
    # PHYSICS: correct (~7.2 g/kg; saturation q at 20 C is ~14.6 g/kg, so ~50%
    # of that). Returns a numpy float64.


def test_golden_rh_to_specific_humidity_fraction_equiv():
    # RH given as a 0-1 fraction must match RH given in percent (the >1.0
    # branch normalizes percent inputs).
    out = ax.rh_to_specific_humidity(0.5, 20.0)
    assert float(out) == pytest.approx(0.007204269021259108)
    # PHYSICS: correct. Edge case: an exactly 100% input expressed as 1.0 would
    # be treated as a fraction (also correct); 1.0 < RH would be percent.


# =============================================================================
# advect_detect.py
# =============================================================================


def test_golden_detect_horizontal_flux_difference():
    # t0: upwind 80 > main 50 + upwind_h_excess(20) -> flagged; t1: 40 > 70 is
    # False. The threshold is now the named, documented `upwind_h_excess`
    # keyword (absolute W/m^2), defaulting to the historical 20 W/m^2.
    flags = ax.detect_horizontal_advection([50.0, 50.0], upwind_flux=[80.0, 40.0])
    assert flags.tolist() == [True, False]
    # PHYSICS: coarse heuristic. The W/m^2 threshold is a tuned magic number, not
    # a physically derived gradient term; the gradient signals are preferred.


def test_golden_detect_horizontal_opposite_sign():
    # main -5 < 0 and upwind 30 > -5 + upwind_h_excess(20) = 15 -> the
    # upwind-sensible-heat-excess signal fires.
    flags = ax.detect_horizontal_advection([-5.0], upwind_flux=[30.0])
    assert flags.tolist() == [True]
    # PHYSICS: consistent with the oasis fingerprint (downward/negative H at the
    # main tower while upwind H is positive).


def test_golden_detect_horizontal_le_exceeds_available_energy():
    # LE 110 > 1.05 * (Rn - G) = 1.05 * 100 = 105 -> flagged.
    flags = ax.detect_horizontal_advection([50.0], le_main=[110.0], rn=[100.0], g=[0.0])
    assert flags.tolist() == [True]
    # PHYSICS: correct — EF > 1 (LE > Rn-G) is the advective-input signal from
    # CLAUDE.md. The 5% tolerance is a sensible noise guard.


def test_golden_detect_horizontal_negative_H_daytime():
    # Rn 300 > 50 (daytime) and main H -10 < 0 -> flagged.
    flags = ax.detect_horizontal_advection([-10.0], rn=[300.0])
    assert flags.tolist() == [True]
    # PHYSICS: consistent with the contract (advection produces negative H).


def test_golden_detect_horizontal_wind_direction_gate():
    # wind_dir 10 deg vs upwind_dir 180 deg -> angular diff 170 deg > 45 deg
    # (the default ±wind_sector_deg), so the upwind-referenced signal is gated
    # off and nothing else applies.
    flags = ax.detect_horizontal_advection(
        [50.0], upwind_flux=[80.0], wind_dir=[10.0], upwind_dir=180.0
    )
    assert flags.tolist() == [False]
    # PHYSICS: correct gating — wind is not blowing from the reference tower, so
    # that tower's gradient should not be attributed as advection here.


def test_golden_detect_horizontal_temp_and_humidity_gradients():
    # Upwind warmer by 2 C (> 1 C) AND main moister than upwind by 0.005 kg/kg
    # (> 0.001) both fire.
    flags = ax.detect_horizontal_advection(
        [50.0],
        temp_main=[20.0],
        temp_upwind=[22.0],
        humidity_main=[0.01],
        humidity_upwind=[0.005],
    )
    assert flags.tolist() == [True]
    # PHYSICS: warm-dry-air-onto-moist-surface gradient is the oasis setup. The
    # fixed 1 C / 0.001 kg/kg thresholds are heuristic, not gradient * distance.


def test_golden_detect_vertical_full_case():
    # t0: |w_bar| = 0.1 > 0.05 m/s -> primary signal fires -> True.
    # t1: |w_bar| = 0, no inversion (19 vs 20), healthy H -> no signal -> False.
    flags = ax.detect_vertical_advection(
        vertical_w=[-0.1, 0.0],
        temp_profile_lower=[15.0, 20.0],
        temp_profile_upper=[17.0, 19.0],
        main_H=[5.0, 100.0],
        rn=[300.0, 300.0],
        g=[50.0, 50.0],
    )
    assert flags.tolist() == [True, False]
    # PHYSICS: the planar-fit mean vertical velocity is the dominant signal for
    # vertical advection (Lee 1998); ~0.05 m/s is energetically significant.


def test_golden_detect_vertical_supporting_branch():
    # No w_bar: daytime (Rn - G = 250 > 50) AND a vertical T gradient of the
    # advective sign (16 > 15 + 0.5, warm air aloft) fire the supporting signal.
    flags = ax.detect_vertical_advection(
        temp_profile_lower=[15.0],
        temp_profile_upper=[16.0],
        main_H=[10.0],
        rn=[300.0],
        g=[50.0],
    )
    assert flags.tolist() == [True]
    # PHYSICS: warm air aloft over a cooler surface gives (T_zm - <T>) > 0;
    # paired with oasis subsidence this is the negative (energy-IN) VAT sign.


def test_golden_compute_advection_fluxes_requires_upwind_and_distance():
    main = {
        "H": np.array([54.0, 64.0]),
        "LE": np.array([30.0, 40.0]),
        "Rn": np.array([90.0, 110.0]),
        "G": np.array([5.0, 5.0]),
    }
    # CONTRACT CHANGE (was: silently returned H_adv = 0 with no upwind tower).
    # Horizontal advection is now the gradient term rho*Cp*u*(dT/dx)*(zm-h),
    # which has no meaning without an upwind reference, so the function RAISES
    # rather than emitting a meaningless zero (CLAUDE.md: never compute an
    # advection term as a flux difference / silent zero).
    with pytest.raises(ValueError, match="upwind"):
        ax.compute_advection_fluxes(main_data=main)


def test_golden_compute_advection_fluxes_multi_upwind_gradient():
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
    up1 = {"T": 25.0, "q": 0.006, "bearing": 0.0}  # warm/dry to the north
    up2 = {"T": 15.0, "q": 0.010, "bearing": 180.0}  # cool/moist to the south
    res = ax.compute_advection_fluxes(
        main_data=main, upwind_data=[up1, up2], tower_distance=50.0
    )
    # 'residual' = (H+LE) - (Rn-G) = (10+30) - (40-0) = 0. 'adv_in' is a
    # deprecated alias of 'residual'. Both are diagnostics, NOT advection.
    np.testing.assert_allclose(res["residual"], [0.0, 0.0])
    np.testing.assert_allclose(res["adv_in"], res["residual"])
    # PHYSICS (FIXED): H_adv is now the gradient term HA_T, NOT a flux
    # difference. Wind-direction selection: t0 picks up1 (warm upwind ->
    # dT/dx<0 -> HA_T<0, energy INTO field); t1 picks up2 (cool upwind ->
    # HA_T>0). Literals are rho*Cp*u*(dT/dx)*(zm-h) with the main-tower
    # rho/Cp(q=0.008), dx=50 m, (zm-h)=1.8 m.
    np.testing.assert_allclose(res["HA_T"], [-654.7393167761763, 654.7393167761763])
    np.testing.assert_allclose(res["H_adv"], res["HA_T"])
    # Moisture gradient term (Eq. 5b): dry upwind (t0) -> dq/dx>0 -> HA_Q>0.
    np.testing.assert_allclose(res["HA_Q"], [635.109561168845, -635.109561168845])
    # PHYSICS (FIXED): no vertical inputs (w_bar / detect_vertical) were
    # supplied, so the measured VAT term is NOT engaged and is returned as None
    # -- the function no longer back-fills the closure residual as vertical
    # advection (CLAUDE.md: never compute an advection term as a residual).
    assert res["VAT"] is None
    assert res["V_adv"] is None  # backward-compat alias of VAT
    assert res["VFD_T"] is None  # no two-level w'T' supplied


def test_golden_compute_advection_fluxes_with_masks():
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
        # Vertical-advection inputs (engaged because detect_vertical is passed):
        # planar-fit w_bar and the column-mean temperature <T>.
        "w_bar": 0.05,
        "T_col": 22.0,
    }
    up = {"T": 28.0, "q": 0.005}  # warm, dry upwind (oasis)
    res = ax.compute_advection_fluxes(
        main_data=main,
        upwind_data=up,
        detect_horizontal=np.array([True, False]),
        detect_vertical=np.array([False, True]),
        tower_distance=80.0,
    )
    # t0: horiz flagged -> HA_T = rho*Cp*u*(dT/dx)*(zm-h) with dT/dx<0 (warm
    #     upwind) -> negative (energy INTO field). t1: horiz NOT flagged -> 0.
    np.testing.assert_allclose(res["HA_T"], [-191.4313460469186, 0.0])
    np.testing.assert_allclose(res["H_adv"], res["HA_T"])
    np.testing.assert_allclose(res["HA_Q"], [462.04087219438054, 0.0])
    # PHYSICS (FIXED): V_adv is now the MEASURED VAT = rho*Cp*w_bar*(T_zm - <T>)
    # (Lee 1998; Wang Eq. 6), NOT the closure residual. The detect_vertical mask
    # [False, True] zeroes t0 (no event) and computes t1 with T_zm=24, <T>=22 ->
    # dT=+2 K, w_bar=0.05 -> positive (energy OUT, per Moderow OUT-positive).
    np.testing.assert_allclose(res["VAT"], [0.0, 119.64459127932412])
    np.testing.assert_allclose(res["V_adv"], res["VAT"])  # backward-compat alias


def test_golden_apply_advection_correction_is_identity_on_sum():
    main = {
        "H": np.array([10.0, 20.0]),
        "LE": np.array([30.0, 40.0]),
        "Rn": np.array([60.0, 70.0]),
        "G": np.array([5.0, 5.0]),
    }
    out = ax.apply_advection_correction(main, np.zeros(2), np.zeros(2))
    assert set(out.keys()) == {
        "Rn",
        "G",
        "H",
        "LE",
        "H_adv",
        "V_adv",
        "H_plus_LE_orig",
        "H_plus_LE_corrected",
    }
    np.testing.assert_allclose(out["H_plus_LE_orig"], [40.0, 60.0])
    np.testing.assert_allclose(out["H_plus_LE_corrected"], [40.0, 60.0])
    np.testing.assert_allclose(out["H_adv"], [0.0, 0.0])
    np.testing.assert_allclose(out["V_adv"], [0.0, 0.0])
    # SUSPECT: "corrected" == "original" by construction — the function never
    # folds H_adv/V_adv into the energy-balance sum, so it applies NO actual
    # correction. Even with nonzero advection terms, H_plus_LE_corrected would
    # be unchanged. The name over-promises relative to the behavior.


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# A non-numpy sanity import guard so a broken install fails loudly here too.
assert math is not None
