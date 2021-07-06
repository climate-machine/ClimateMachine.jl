# Atmosphere model configurations

The struct `AtmosModel` defines a specific subtype of a balance law
(i.e. conservation equations) specific to atmospheric modeling. A
complete description of a `model` is provided by the fields listed
below. In this implementation of the `AtmosModel` we concern ourselves
with the conservative form of the compressible equations of moist fluid
motion given a set of initial, boundary and forcing(source) conditions.

First, we construct the atmospheric physics via `AtmosPhysics`:

```
physics = AtmosPhysics{FT}(;
    param_set::AbstractParameterSet;
    ref_state = HydrostaticState(DecayingTemperatureProfile{FT}(param_set),)
    turbulence = SmagorinskyLilly{FT}(0.21),
    hyperdiffusion = NoHyperDiffusion(),
    moisture = EquilMoist(),
    precipitation = NoPrecipitation(),
    radiation = NoRadiation(),
    tracers = NoTracers(),
)
```

### LES Configuration (with defaults)
Default field values for the LES `AtmosModel` definition are included
below. Users are directed to the model subcomponent pages to view the
possible options for each subcomponent.
```
atmos = AtmosModel{FT}(
    ::Type{AtmosLESConfigType},
    physics::AtmosPhysics;
    orientation = FlatOrientation(),
    source = (Gravity(), Coriolis(), GeostrophicForcing{FT}(7.62e-5, 0, 0)),
    problem = AtmosBC(physics),
    init_state_prognostic = nothing,
    data_config = nothing,
)
```

!!! note

    Most AtmosModel subcomponents are common to both LES / GCM
    configurations.  Equation sets are written in vector-invariant form and
    solved in Cartesian coordinates.  The component `orientation` determines
    whether the problem is solved in a `box (LES)` or a `sphere (GCM)`)
    
!!! warning

    The above defaults do not consider any potential interaction with the time stepping
    components. The user must take appropriate care when setting up the driver configurations.
    The default hydrostatic reference state may not be suitable to any initial conditions and 
    it recommended to use a reference state corresponding to the initial condition's background.
    It is also recommended to use the NoReferenceState type when using a single rate explicit 
    time stepper.


### GCM Configuration (with defaults)
Default field values for the GCM `AtmosModel` definition are included
below. Users are directed to the model subcomponent pages to view the
possible options for each subcomponent.

```
    ::Type{AtmosGCMConfigType},
    physics::AtmosPhysics;
    source::S = (Gravity(), Coriolis()),
    boundarycondition::BC = AtmosBC(physics),
    init_state_prognostic::IS = nothing,
    data_config::DC = nothing,
```
