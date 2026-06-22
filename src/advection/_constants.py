"""Shared fixed physical constants for the :mod:`advection` package.

These are genuine constants (no temperature/pressure dependence). Anything that
varies with the air state—density, specific heat, latent heat of
vaporization—must be obtained from the physically-based helpers in
:mod:`advection.advection` (``air_density``, ``specific_heat_moist_air``,
``latent_heat_vaporization``) rather than hard-coded here.

Centralizing the fixed values means both :mod:`advection.advection` and
:mod:`advection.advect_detect` import them from a single source instead of
duplicating literals.
"""

# Dry adiabatic lapse rate, g / c_p  [K/m].
G_OVER_CP = 0.0098

# Ratio of the molar mass of dry air to that of water vapor (M_d / M_v); the
# "mu" factor in the WPL density correction [dimensionless].
MU = 1.6077

# von Karman constant [dimensionless].
VON_KARMAN = 0.41
