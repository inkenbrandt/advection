# CLAUDE.md — Physics Contract for the `advection` Library

This file defines the physics this library **must** obey. Every edit to the code,
docstrings, tests, and documentation must remain consistent with the conventions,
equations, and rules below. When in doubt, this file wins.

## SCOPE

Detecting and correcting **horizontal/vertical advection** in eddy-covariance
**surface energy balance**, primarily the irrigated low-canopy **"oasis" regime**
where warm dry air advects onto a cool transpiring surface.

## SIGN CONVENTION (Moderow et al. 2021)

- **Positive flux = energy OUT of the control volume.**
- **Negative flux = energy INTO the control volume.**
- Consequence for the oasis case: advection produces **negative (downward) H** and
  **negative vertical heat advection**.

## SURFACE ENERGY BALANCE

- Without storage: `Rn - G = H + LE`
- With storage: `Rn - G - J = H + LE`
- Residual (imbalance): `Residual = Rn - G - H - LE`
- Energy Balance Ratio: `EBR = sum(H + LE) / sum(Rn - G - J)`
- Evaporative fraction: `EF = LE / (Rn - G)`
  - `EF > 1` (equivalently `LE > Rn - G`) signals **advective input**.

## KEY EQUATIONS (cite by name in docstrings)

### Variance Bowen ratio (Wang 2024, Eq. 8)

```
beta = (Cp / lambda) * (sigma_T / sigma_q)
sign(beta) = sign(corr(T', q'))
```

A **negative beta is the oasis fingerprint.**

### Sonic-T flux correction (Wang 2024, Eq. 9)

```
wT = wTs / (1 + 0.51 * Cp * T / (lambda * beta))      # T in KELVIN
```

The crosswind term of Eq. 7 is dropped because the sonic crosswind correction is
applied internally.

### Sensible heat

```
H = rho * Cp * wT
```

### Horizontal heat advection (Wang 2024, Eq. 5a; Moderow Term IV)

```
HA_T ≈ rho * Cp * u_bar * (dT/dx) * (zm - h)          # W/m^2
```

### Horizontal moisture advection (Wang 2024, Eq. 5b)

```
HA_Q ≈ rho * lambda * u_bar * (dq/dx) * (zm - h)      # W/m^2
```

### Vertical heat advection (Lee 1998; Wang 2024, Eq. 6)

```
VAT ≈ rho * Cp * w_bar * (T_zm - <T>)
```

- `<T>` = column-mean temperature.
- `w_bar` = planar-fit mean vertical velocity.
- **This is NOT a closure residual.**

### Vertical heat flux divergence (Wang 2024, Eq. 12)

```
VFD_T ≈ -rho * Cp * (wT|zm - wT|h)
```

### Air heat storage (Wang 2024, Eq. 11)

```
T_storage ≈ rho * Cp * (dT/dt) * (zm - h)
```

### Soil heat storage (Eq. 1a) and total ground heat flux (Eq. 1b)

```
Gs = Cs * dz * (dTsoil/dt)
G  = Gd + Gs
```

- `dz` = thickness of the soil layer **ABOVE** the heat-flux plate.

## CONDITIONAL INCLUSION RULE (Wang 2024)

Only add advective fluxes to close the budget when **BOTH** conditions hold:

1. `Rn > 75 W/m^2`, **AND**
2. spectrally-corrected `(H + LE) < (Rn - G)`.

This improved closure from **89% to 97%** in the alfalfa study.

## HARD RULES

- **Never** compute an advection term as the energy-balance residual.
- **Never** force Bowen-ratio closure when `LE > (Rn - G)` (oasis case) — it is wrong.
- **WPL density correction** is a mandatory, separate pre-step for open-path
  LE/CO2. This library assumes LE is **already WPL-corrected** — state this in
  docstrings.
- All advective heat/vapour terms are in **W/m^2** and must be unit-consistent.
