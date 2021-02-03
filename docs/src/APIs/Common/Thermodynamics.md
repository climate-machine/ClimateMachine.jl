# Thermodynamics

```@meta
CurrentModule = ClimateMachine.Thermodynamics
```
```@docs
Thermodynamics
ThermodynamicState
```

## Thermodynamic State Constructors

```@docs
PhaseDry
PhaseDry_pT
PhaseDry_pθ
PhaseDry_ρθ
PhaseDry_ρT
PhaseDry_ρp
PhaseEquil
PhaseEquil_ρTq
PhaseEquil_pTq
PhaseEquil_pθq
PhaseEquil_ρθq
PhaseEquil_ρpq
PhaseNonEquil
PhaseNonEquil_ρTq
PhaseNonEquil_ρθq
PhaseNonEquil_pθq
PhaseNonEquil_ρpq
```

## Thermodynamic state methods

```@docs
air_density
air_pressure
air_temperature
condensate
cp_m
cv_m
dry_pottemp
exner
gas_constant_air
gas_constants
has_condensate
ice_specific_humidity
internal_energy
internal_energy_dry
internal_energy_vapor
internal_energy_liquid
internal_energy_ice
internal_energy_sat
latent_heat_fusion
latent_heat_liq_ice
latent_heat_sublim
latent_heat_vapor
liquid_fraction
liquid_ice_pottemp
liquid_ice_pottemp_sat
liquid_specific_humidity
mixing_ratios
moist_static_energy
q_vap_saturation
q_vap_saturation_liquid
q_vap_saturation_ice
q_vap_saturation_generic
relative_humidity
saturated
saturation_adjustment
saturation_excess
saturation_vapor_pressure
soundspeed_air
shum_to_mixing_ratio
specific_enthalpy
specific_volume
supersaturation
total_energy
total_specific_enthalpy
total_specific_humidity
vapor_specific_humidity
virtual_pottemp
virtual_temperature
```

## Additional types/constructors

```@docs
Liquid
Ice
PhasePartition
PhasePartition_equil
DryAdiabaticProcess
```
