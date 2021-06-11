# Balance Laws

```@meta
CurrentModule = ClimateMachine.BalanceLaws
```

## The balance law

```@docs
BalanceLaw
```

## Tendency types and methods

```@docs
AbstractPrognosticVariable
AbstractOrder
FirstOrder
SecondOrder
AbstractTendencyType
Flux
Source
TendencyDef
eq_tends
prognostic_vars
get_prog_state
projection
precompute
prognostic_var_source_map
show_tendencies
```

## Methods for fluxes and sources

```@docs
flux
source
Σfluxes
Σsources
```

## State variable types

```@docs
AbstractStateType
Prognostic
Primitive
Entropy
Auxiliary
Gradient
GradientFlux
GradientLaplacian
Hyperdiffusive
UpwardIntegrals
DownwardIntegrals
```

## Interface

```@docs
sub_model
```

## Variable specification methods

```@docs
vars_state
```

## Initial condition methods

```@docs
init_state_prognostic!
init_state_auxiliary!
nodal_init_state_auxiliary!
```

## Source term kernels

```@docs
flux_first_order!
flux_second_order!
source!
```

## Integral kernels

```@docs
indefinite_stack_integral!
reverse_indefinite_stack_integral!
integral_load_auxiliary_state!
integral_set_auxiliary_state!
reverse_integral_load_auxiliary_state!
reverse_integral_set_auxiliary_state!
```

## Gradient/Laplacian kernels

```@docs
compute_gradient_flux!
compute_gradient_argument!
transform_post_gradient_laplacian!
```

## Boundary conditions

```@docs
used_bcs
DefaultBC
DefaultBCValue
DefaultBCFlux
boundary_value
boundary_flux
set_boundary_values!
set_boundary_fluxes!
boundary_conditions
boundary_state!
```

## Auxiliary kernels

```@docs
wavespeed
update_auxiliary_state!
update_auxiliary_state_gradient!
nodal_update_auxiliary_state!
```
