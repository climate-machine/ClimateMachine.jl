#####
##### Tendency specification
#####

#####
##### Sources
#####

eq_tends(pv::AbstractPrognosticVariable, m::AtmosModel, tt::Source) =
    (m.source[pv]..., eq_tends(pv, turbconv_model(m), tt)...)

#####
##### First order fluxes
#####

# Dispatch to flux differencing form
eq_tends(
    pv::AbstractPrognosticVariable,
    atmos::AtmosModel,
    tt::Flux{FirstOrder},
) = eq_tends(atmos.equations_form, pv, atmos, tt)
eq_tends(
    pv::AbstractPrognosticVariable,
    atmos::AtmosModel,
    tt::FluxDifferencing{FirstOrder},
) = eq_tends(atmos.equations_form, pv, atmos, tt)

# Default is no flux differencing

eq_tends(
    ::AbstractEquationsForm,
    ::AbstractPrognosticVariable,
    ::AtmosModel,
    ::FluxDifferencing{FirstOrder},
) = ()

# helpers

# Mass
eq_tends(
    ::Mass,
    ::Anelastic1D,
    ::Union{Flux{FirstOrder}, FluxDifferencing{FirstOrder}},
) = ()
eq_tends(
    ::Mass,
    ::Compressible,
    ::Union{Flux{FirstOrder}, FluxDifferencing{FirstOrder}},
) = (Advect(),)

# Momentum
eq_tends(
    pv::Momentum,
    ::Compressible,
    ::Union{Flux{FirstOrder}, FluxDifferencing{FirstOrder}},
) = (PressureGradient(),)
eq_tends(
    pv::Momentum,
    ::Anelastic1D,
    ::Union{Flux{FirstOrder}, FluxDifferencing{FirstOrder}},
) = ()

# Energy
eq_tends(
    ::Energy,
    m::TotalEnergyModel,
    ::Union{Flux{FirstOrder}, FluxDifferencing{FirstOrder}},
) = (Advect(), Pressure())
eq_tends(
    ::ρθ_liq_ice,
    m::θModel,
    ::Union{Flux{FirstOrder}, FluxDifferencing{FirstOrder}},
) = (Advect(),)

##### Unsplit flux first order tendencies

# Mass
eq_tends(::Unsplit, pv::Mass, atmos::AtmosModel, tt::Flux{FirstOrder}) =
    (eq_tends(pv, compressibility_model(atmos), tt))

# Momentum
eq_tends(::Unsplit, pv::Momentum, m::AtmosModel, tt::Flux{FirstOrder}) =
    (Advect(), eq_tends(pv, compressibility_model(m), tt)...)

# Energy
# TODO: make radiation aware of which energy formulation is used:
# eq_tends(pv::PV, m::AtmosModel, tt::Flux{FirstOrder}) where {PV <: AbstractEnergyVariable} =
#     (eq_tends(pv, energy_model(m), tt)..., eq_tends(pv, energy_model(m), radiation_model(m), tt)...)
eq_tends(
    ::Unsplit,
    pv::AbstractEnergyVariable,
    m::AtmosModel,
    tt::Flux{FirstOrder},
) = (
    eq_tends(pv, energy_model(m), tt)...,
    eq_tends(pv, radiation_model(m), tt)...,
)

# AbstractMoistureVariable
eq_tends(
    ::Unsplit,
    ::AbstractMoistureVariable,
    ::AtmosModel,
    ::Flux{FirstOrder},
) = (Advect(),)

# AbstractPrecipitationVariable
eq_tends(
    ::Unsplit,
    pv::AbstractPrecipitationVariable,
    m::AtmosModel,
    tt::Flux{FirstOrder},
) = (eq_tends(pv, precipitation_model(m), tt)...,)

# Tracers
eq_tends(
    ::Unsplit,
    pv::Tracers{N},
    ::AtmosModel,
    ::Flux{FirstOrder},
) where {N} = (Advect(),)

##### CentralSplitForm flux first order tendencies
# treat everything using flux differencing

eq_tends(
    ::CentralSplitForm,
    pv::AbstractPrognosticVariable,
    atmos::AtmosModel,
    ::FluxDifferencing{FirstOrder},
) = eq_tends(Unsplit(), pv, atmos, Flux{FirstOrder}())

eq_tends(
    ::CentralSplitForm,
    ::AbstractPrognosticVariable,
    ::AtmosModel,
    ::Flux{FirstOrder},
) = ()


##### Kennedy Gruber split form

# default to no splitting
eq_tends(
    ::AbstractKennedyGruberSplitForm,
    pv::AbstractPrognosticVariable,
    atmos::AtmosModel,
    tt::Flux{FirstOrder},
) = eq_tends(Unsplit(), pv, atmos, tt)

# Mass
eq_tends(
    ::AbstractKennedyGruberSplitForm,
    pv::Mass,
    atmos::AtmosModel,
    tt::Flux{FirstOrder},
) = ()
eq_tends(
    ::AbstractKennedyGruberSplitForm,
    pv::Mass,
    atmos::AtmosModel,
    tt::FluxDifferencing{FirstOrder},
) = (eq_tends(pv, compressibility_model(atmos), tt))

# Momentum
eq_tends(
    ::AbstractKennedyGruberSplitForm,
    pv::Momentum,
    m::AtmosModel,
    tt::Flux{FirstOrder},
) = eq_tends(pv, compressibility_model(m), tt)
eq_tends(
    ::KennedyGruberSplitForm,
    pv::Momentum,
    m::AtmosModel,
    ::FluxDifferencing{FirstOrder},
) = (Advect(),)
eq_tends(
    ::KennedyGruberGravitySplitForm,
    pv::Momentum,
    m::AtmosModel,
    ::FluxDifferencing{FirstOrder},
) = (Advect(), GravityFluctuation())

# Energy
eq_tends(
    ::AbstractKennedyGruberSplitForm,
    pv::AbstractEnergyVariable,
    m::AtmosModel,
    tt::Flux{FirstOrder},
) = eq_tends(pv, radiation_model(m), tt)
eq_tends(
    ::AbstractKennedyGruberSplitForm,
    pv::AbstractEnergyVariable,
    m::AtmosModel,
    tt::FluxDifferencing{FirstOrder},
) = eq_tends(pv, energy_model(m), tt)

# AbstractMoistureVariable
eq_tends(
    ::AbstractKennedyGruberSplitForm,
    ::AbstractMoistureVariable,
    ::AtmosModel,
    ::Flux{FirstOrder},
) = ()
eq_tends(
    ::AbstractKennedyGruberSplitForm,
    ::AbstractMoistureVariable,
    ::AtmosModel,
    ::FluxDifferencing{FirstOrder},
) = (Advect(),)

# TODO: split tracers

#####
##### Second order fluxes
#####

eq_tends(
    ::Union{Mass, Momentum, AbstractMoistureVariable},
    ::DryModel,
    ::Flux{SecondOrder},
) = ()
eq_tends(
    ::Union{Mass, Momentum, AbstractMoistureVariable},
    ::AbstractMoistureModel,
    ::Flux{SecondOrder},
) = (MoistureDiffusion(),)

# Mass
eq_tends(pv::Mass, m::AtmosModel, tt::Flux{SecondOrder}) =
    (eq_tends(pv, moisture_model(m), tt)...,)

# Momentum
eq_tends(pv::Momentum, m::AtmosModel, tt::Flux{SecondOrder}) = (
    ViscousStress(),
    eq_tends(pv, moisture_model(m), tt)...,
    eq_tends(pv, turbconv_model(m), tt)...,
    eq_tends(pv, hyperdiffusion_model(m), tt)...,
)

# Energy
eq_tends(::Energy, m::TotalEnergyModel, tt::Flux{SecondOrder}) =
    (ViscousFlux(), DiffEnthalpyFlux())

eq_tends(::ρθ_liq_ice, m::θModel, tt::Flux{SecondOrder}) = (ViscousFlux(),)

eq_tends(pv::AbstractEnergyVariable, m::AtmosModel, tt::Flux{SecondOrder}) = (
    eq_tends(pv, energy_model(m), tt)...,
    eq_tends(pv, turbconv_model(m), tt)...,
    eq_tends(pv, hyperdiffusion_model(m), tt)...,
)

# AbstractMoistureVariable
eq_tends(pv::AbstractMoistureVariable, m::AtmosModel, tt::Flux{SecondOrder}) = (
    eq_tends(pv, moisture_model(m), tt)...,
    eq_tends(pv, turbconv_model(m), tt)...,
    eq_tends(pv, hyperdiffusion_model(m), tt)...,
)

# AbstractPrecipitationVariable
eq_tends(
    pv::AbstractPrecipitationVariable,
    m::AtmosModel,
    tt::Flux{SecondOrder},
) = (eq_tends(pv, precipitation_model(m), tt)...,)

# Tracers
eq_tends(pv::Tracers{N}, m::AtmosModel, tt::Flux{SecondOrder}) where {N} =
    (eq_tends(pv, tracer_model(m), tt)...,)
