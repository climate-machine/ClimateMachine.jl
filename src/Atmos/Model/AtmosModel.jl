module Atmos

export AtmosModel,
    AtmosPhysics,
    AtmosAcousticLinearModel,
    AtmosAcousticGravityLinearModel,
    HLLCNumericalFlux,
    RoeNumericalFlux,
    RoeNumericalFluxMoist,
    LMARSNumericalFlux,
    Compressible,
    Anelastic1D,
    reference_state,
    energy_model,
    moisture_model,
    compressibility_model,
    turbulence_model,
    turbconv_model,
    hyperdiffusion_model,
    viscoussponge_model,
    precipitation_model,
    radiation_model,
    tracer_model,
    lsforcing_model,
    parameter_set

using UnPack
using DispatchedTuples
using CLIMAParameters
using CLIMAParameters.Planet: grav, cp_d, R_v, LH_v0, e_int_v0
using CLIMAParameters.Atmos.SubgridScale: C_smag
using DocStringExtensions
using LinearAlgebra, StaticArrays
using ..ConfigTypes
using ..Orientations
import ..Orientations:
    vertical_unit_vector,
    altitude,
    latitude,
    longitude,
    projection_normal,
    gravitational_potential,
    ∇gravitational_potential,
    projection_tangential

using ..VariableTemplates
using ..Thermodynamics
using ..TemperatureProfiles

using ..TurbulenceClosures
import ..TurbulenceClosures: turbulence_tensors
using ..TurbulenceConvection

import ..Thermodynamics: internal_energy, soundspeed_air
using ..MPIStateArrays: MPIStateArray
using ..Mesh.Grids:
    VerticalDirection,
    HorizontalDirection,
    min_node_distance,
    EveryDirection,
    Direction

using ..Mesh.Filters: AbstractFilterTarget
import ..Mesh.Filters:
    vars_state_filtered, compute_filter_argument!, compute_filter_result!

using ..BalanceLaws
using ClimateMachine.Problems

import ..BalanceLaws:
    vars_state,
    projection,
    sub_model,
    prognostic_vars,
    get_prog_state,
    get_specific_state,
    flux_first_order!,
    total_flux_first_order!,
    flux_second_order!,
    source!,
    eq_tends,
    flux,
    two_point_flux,
    precompute,
    parameter_set,
    source,
    wavespeed,
    boundary_conditions,
    boundary_state!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    transform_post_gradient_laplacian!,
    prognostic_to_primitive!,
    primitive_to_prognostic!,
    init_state_auxiliary!,
    construct_face_auxiliary_state!,
    init_state_prognostic!,
    update_auxiliary_state!,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!

import ClimateMachine.DGMethods:
    LocalGeometry, lengthscale, resolutionmetric, DGModel

import ..DGMethods.NumericalFluxes:
    boundary_state!,
    boundary_flux_second_order!,
    normal_boundary_flux_second_order!,
    NumericalFluxFirstOrder,
    NumericalFluxGradient,
    NumericalFluxSecondOrder,
    CentralNumericalFluxHigherOrder,
    CentralNumericalFluxDivergence,
    CentralNumericalFluxFirstOrder,
    numerical_flux_first_order!,
    NumericalFluxFirstOrder
using ..DGMethods.NumericalFluxes:
    RoeNumericalFlux,
    HLLCNumericalFlux,
    RusanovNumericalFlux,
    RoeNumericalFluxMoist,
    LMARSNumericalFlux

import ..Courant: advective_courant, nondiffusive_courant, diffusive_courant

"""
    AtmosPhysics

An `AtmosPhysics` for atmospheric physics

# Usage

    AtmosPhysics(
        param_set,
        ref_state,
        energy,
        moisture,
        compressibility,
        turbulence,
        turbconv,
        hyperdiffusion,
        precipitation,
        radiation,
        tracers,
        lsforcing,
    )

# Fields
$(DocStringExtensions.FIELDS)
"""
struct AtmosPhysics{FT, PS, RS, E, M, C, T, TC, HD, VS, P, R, TR, LF}
    "Parameter Set (type to dispatch on, e.g., planet parameters. See CLIMAParameters.jl package)"
    param_set::PS
    "Reference State (For initial conditions, or for linearisation when using implicit solvers)"
    ref_state::RS
    "Energy sub-model, can be energy-based or θ_liq_ice-based"
    energy::E
    "Moisture Model (Equations for dynamics of moist variables)"
    moisture::M
    "Compressibility switch"
    compressibility::C
    "Turbulence Closure (Equations for dynamics of under-resolved turbulent flows)"
    turbulence::T
    "Turbulence Convection Closure (e.g., EDMF)"
    turbconv::TC
    "Hyperdiffusion Model (Equations for dynamics of high-order spatial wave attenuation)"
    hyperdiffusion::HD
    "Viscous sponge layers"
    viscoussponge::VS
    "Precipitation Model (Equations for dynamics of precipitating species)"
    precipitation::P
    "Radiation Model (Equations for radiative fluxes)"
    radiation::R
    "Tracer Terms (Equations for dynamics of active and passive tracers)"
    tracers::TR
    "Large-scale forcing (Forcing information from GCMs, reanalyses, or observations)"
    lsforcing::LF
end

"""
    AtmosPhysics{FT}()

Constructor for `AtmosPhysics`.
"""
function AtmosPhysics{FT}(
    param_set::AbstractParameterSet;
    energy = TotalEnergyModel(),
    ref_state = HydrostaticState(DecayingTemperatureProfile{FT}(param_set),),
    turbulence = SmagorinskyLilly{FT}(C_smag(param_set)),
    turbconv = NoTurbConv(),
    hyperdiffusion = NoHyperDiffusion(),
    viscoussponge = NoViscousSponge(),
    moisture = EquilMoist(),
    precipitation = NoPrecipitation(),
    radiation = NoRadiation(),
    tracers = NoTracers(),
    lsforcing = NoLSForcing(),
    compressibility = Compressible(),
) where {FT <: AbstractFloat}

    args = (
        param_set,
        ref_state,
        energy,
        moisture,
        compressibility,
        turbulence,
        turbconv,
        hyperdiffusion,
        viscoussponge,
        precipitation,
        radiation,
        tracers,
        lsforcing,
    )
    return AtmosPhysics{FT, typeof.(args)...}(args...)
end


"""
    AtmosModel <: BalanceLaw

A `BalanceLaw` for atmosphere modeling. Users may over-ride prescribed
default values for each field.

# Usage

    AtmosModel(
        physics,
        problem,
        orientation,
        source,
        data_config,
    )

# Fields
$(DocStringExtensions.FIELDS)
"""
struct AtmosModel{FT, PH, PR, O, S, DC, EF} <: BalanceLaw
    "Atmospheric physics"
    physics::PH
    "Problem (initial and boundary conditions)"
    problem::PR
    "An orientation model"
    orientation::O
    "Source Terms (Problem specific source terms)"
    source::S
    "Data Configuration (Helper field for experiment configuration)"
    data_config::DC
    "Form of equations: (Unsplit, split form, or entropy stable)"
    equations_form::EF
end

parameter_set(atmos::AtmosModel) = parameter_set(atmos.physics)
moisture_model(atmos::AtmosModel) = moisture_model(atmos.physics)
energy_model(atmos::AtmosModel) = energy_model(atmos.physics)
compressibility_model(atmos::AtmosModel) = compressibility_model(atmos.physics)
reference_state(atmos::AtmosModel) = reference_state(atmos.physics)
turbulence_model(atmos::AtmosModel) = turbulence_model(atmos.physics)
turbconv_model(atmos::AtmosModel) = turbconv_model(atmos.physics)
hyperdiffusion_model(atmos::AtmosModel) = hyperdiffusion_model(atmos.physics)
viscoussponge_model(atmos::AtmosModel) = viscoussponge_model(atmos.physics)
precipitation_model(atmos::AtmosModel) = precipitation_model(atmos.physics)
radiation_model(atmos::AtmosModel) = radiation_model(atmos.physics)
tracer_model(atmos::AtmosModel) = tracer_model(atmos.physics)
lsforcing_model(atmos::AtmosModel) = lsforcing_model(atmos.physics)

parameter_set(physics::AtmosPhysics) = physics.param_set
moisture_model(physics::AtmosPhysics) = physics.moisture
energy_model(physics::AtmosPhysics) = physics.energy
compressibility_model(physics::AtmosPhysics) = physics.compressibility
reference_state(physics::AtmosPhysics) = physics.ref_state
turbulence_model(physics::AtmosPhysics) = physics.turbulence
turbconv_model(physics::AtmosPhysics) = physics.turbconv
hyperdiffusion_model(physics::AtmosPhysics) = physics.hyperdiffusion
viscoussponge_model(physics::AtmosPhysics) = physics.viscoussponge
precipitation_model(physics::AtmosPhysics) = physics.precipitation
radiation_model(physics::AtmosPhysics) = physics.radiation
tracer_model(physics::AtmosPhysics) = physics.tracers
lsforcing_model(physics::AtmosPhysics) = physics.lsforcing

abstract type Compressibilty end

"""
    Compressible <: Compressibilty

Dispatch on compressible model (default)

 - Density is prognostic
"""
struct Compressible <: Compressibilty end

"""
    Anelastic1D <: Compressibilty

Dispatch on Anelastic1D model

 - The state density is taken constant in time and equal to the reference density. This
    constant density profile is used in all equations and conversions from conservative to specific
    variables per unit mass. The density can be accessed using the dispatch function
    `density(atmos, state, aux)`.
 - The thermodynamic state is constructed from the reference pressure (constant in time),
    and the internal energy (which evolves in time).
 - The state density is not consistent with the thermodynamic state, since we neglect 
    buoyancy perturbations on all equations except in the vertical buoyancy flux.
 - The density obtained from the thermodynamic state, `air_density(ts)`, recovers the full density, which
    should only be used to compute buoyancy and buoyancy fluxes, and in the FV reconstruction.
 - Removes momentum z-component tendencies, assuming balance between the pressure gradient and buoyancy
    forces.
"""
struct Anelastic1D <: Compressibilty end

"""
    AtmosModel{FT}()

Constructor for `AtmosModel` (where `AtmosModel <: BalanceLaw`).
"""
function AtmosModel{FT}(
    orientation::Orientation,
    physics::AtmosPhysics;
    init_state_prognostic = nothing,
    problem = AtmosProblem(;
        physics = physics,
        init_state_prognostic = init_state_prognostic,
    ),
    source = (
        Gravity(),
        Coriolis(),
        GeostrophicForcing{FT}(7.62e-5, 0, 0),
        turbconv_sources(turbconv_model(physics))...,
    ),
    data_config = nothing,
    equations_form = Unsplit(),
) where {FT <: AbstractFloat}

    # hack, with Kennedy Gruber split form gravity is treated
    # in fluctation form, not as a source
    if equations_form isa KennedyGruberSplitForm && any(s -> s isa Gravity, source)
        source = filter(s -> !(s isa Gravity), source)
        equations_form = KennedyGruberGravitySplitForm()
    end

    atmos = (
        physics,
        problem,
        orientation,
        prognostic_var_source_map(source),
        data_config,
        equations_form,
    )

    return AtmosModel{FT, typeof.(atmos)...}(atmos...)
end

"""
    AtmosModel{FT}()

Constructor for `AtmosModel` (where `AtmosModel <: BalanceLaw`) for LES
and single stack configurations.
"""
function AtmosModel{FT}(
    ::Union{Type{AtmosLESConfigType}, Type{SingleStackConfigType}},
    physics::AtmosPhysics;
    orientation = FlatOrientation(),
    kwargs...,
) where {FT <: AbstractFloat}
    return AtmosModel{FT}(orientation, physics; kwargs...)
end

"""
    AtmosModel{FT}()

Constructor for `AtmosModel` (where `AtmosModel <: BalanceLaw`) for GCM
configurations.
"""
function AtmosModel{FT}(
    ::Type{AtmosGCMConfigType},
    physics::AtmosPhysics;
    orientation = SphericalOrientation(),
    kwargs...,
) where {FT <: AbstractFloat}
    return AtmosModel{FT}(orientation, physics; kwargs...)
end

"""
    vars_state(m::AtmosModel, ::Prognostic, FT)

Conserved state variables (prognostic variables).

!!! warning

    The order of the fields for `AtmosModel` needs to match the one for
    `AtmosLinearModel` since a shared state is used
"""
function vars_state(m::AtmosModel, st::Prognostic, FT)
    @vars begin
        # start of inclusion in `AtmosLinearModel`
        ρ::FT
        ρu::SVector{3, FT}
        energy::vars_state(energy_model(m), st, FT) # TODO: adjust linearmodel
        turbulence::vars_state(turbulence_model(m), st, FT)
        hyperdiffusion::vars_state(hyperdiffusion_model(m), st, FT)
        moisture::vars_state(moisture_model(m), st, FT)
        # end of inclusion in `AtmosLinearModel`
        precipitation::vars_state(precipitation_model(m), st, FT)
        turbconv::vars_state(turbconv_model(m), st, FT)
        radiation::vars_state(radiation_model(m), st, FT)
        tracers::vars_state(tracer_model(m), st, FT)
        lsforcing::vars_state(lsforcing_model(m), st, FT)
    end
end

function vars_state(m::AtmosModel, st::Primitive, FT)
    @vars begin
        ρ::FT
        u::SVector{3, FT}
        p::FT
        moisture::vars_state(moisture_model(m), st, FT)
        turbconv::vars_state(turbconv_model(m), st, FT)
    end
end

"""
    vars_state(m::AtmosModel, ::Gradient, FT)

Pre-transform gradient variables.
"""
function vars_state(m::AtmosModel, st::Gradient, FT)
    @vars begin
        u::SVector{3, FT}
        energy::vars_state(energy_model(m), st, FT)
        turbulence::vars_state(turbulence_model(m), st, FT)
        turbconv::vars_state(turbconv_model(m), st, FT)
        hyperdiffusion::vars_state(hyperdiffusion_model(m), st, FT)
        moisture::vars_state(moisture_model(m), st, FT)
        lsforcing::vars_state(lsforcing_model(m), st, FT)
        precipitation::vars_state(precipitation_model(m), st, FT)
        tracers::vars_state(tracer_model(m), st, FT)
    end
end

"""
    vars_state(m::AtmosModel, ::GradientFlux, FT)

Post-transform gradient variables.
"""
function vars_state(m::AtmosModel, st::GradientFlux, FT)
    @vars begin
        energy::vars_state(energy_model(m), st, FT)
        turbulence::vars_state(turbulence_model(m), st, FT)
        turbconv::vars_state(turbconv_model(m), st, FT)
        hyperdiffusion::vars_state(hyperdiffusion_model(m), st, FT)
        moisture::vars_state(moisture_model(m), st, FT)
        lsforcing::vars_state(lsforcing_model(m), st, FT)
        precipitation::vars_state(precipitation_model(m), st, FT)
        tracers::vars_state(tracer_model(m), st, FT)
    end
end

"""
    vars_state(m::AtmosModel, ::GradientLaplacian, FT)

Pre-transform hyperdiffusive variables.
"""
function vars_state(m::AtmosModel, st::GradientLaplacian, FT)
    @vars begin
        hyperdiffusion::vars_state(hyperdiffusion_model(m), st, FT)
    end
end

"""
    vars_state(m::AtmosModel, ::Hyperdiffusive, FT)

Post-transform hyperdiffusive variables.
"""
function vars_state(m::AtmosModel, st::Hyperdiffusive, FT)
    @vars begin
        hyperdiffusion::vars_state(hyperdiffusion_model(m), st, FT)
    end
end

"""
    vars_state(m::AtmosModel, ::Auxiliary, FT)

Auxiliary variables, such as vertical (stack) integrals, coordinates,
orientation information, reference states, subcomponent auxiliary vars,
debug variables.
"""
function vars_state(m::AtmosModel, st::Auxiliary, FT)
    @vars begin
        ∫dz::vars_state(m, UpwardIntegrals(), FT)
        ∫dnz::vars_state(m, DownwardIntegrals(), FT)
        coord::SVector{3, FT}
        orientation::vars_state(m.orientation, st, FT)
        ref_state::vars_state(reference_state(m), st, FT)
        turbulence::vars_state(turbulence_model(m), st, FT)
        turbconv::vars_state(turbconv_model(m), st, FT)
        hyperdiffusion::vars_state(hyperdiffusion_model(m), st, FT)
        moisture::vars_state(moisture_model(m), st, FT)
        precipitation::vars_state(precipitation_model(m), st, FT)
        tracers::vars_state(tracer_model(m), st, FT)
        radiation::vars_state(radiation_model(m), st, FT)
        lsforcing::vars_state(lsforcing_model(m), st, FT)
    end
end

"""
    vars_state(m::AtmosModel, ::UpwardIntegrals, FT)
"""
function vars_state(m::AtmosModel, st::UpwardIntegrals, FT)
    @vars begin
        radiation::vars_state(radiation_model(m), st, FT)
        turbconv::vars_state(turbconv_model(m), st, FT)
    end
end

"""
    vars_state(m::AtmosModel, ::DownwardIntegrals, FT)
"""
function vars_state(m::AtmosModel, st::DownwardIntegrals, FT)
    @vars begin
        radiation::vars_state(radiation_model(m), st, FT)
    end
end

function vars_state_filtered(m::AtmosModel, FT)
    @vars begin
        ρ::FT
        u::SVector{3, FT}
        energy::vars_state_filtered(energy_model(m), FT)
        moisture::vars_state_filtered(moisture_model(m), FT)
        turbconv::vars_state_filtered(turbconv_model(m), FT)
    end
end


####
#### Forward orientation methods
####
projection_normal(bl, aux, u⃗) =
    projection_normal(bl.orientation, parameter_set(bl), aux, u⃗)
projection_tangential(bl, aux, u⃗) =
    projection_tangential(bl.orientation, parameter_set(bl), aux, u⃗)
latitude(bl, aux) = latitude(bl.orientation, aux)
longitude(bl, aux) = longitude(bl.orientation, aux)
altitude(bl, aux) = altitude(bl.orientation, parameter_set(bl), aux)
vertical_unit_vector(bl, aux) =
    vertical_unit_vector(bl.orientation, parameter_set(bl), aux)
gravitational_potential(bl, aux) = gravitational_potential(bl.orientation, aux)
∇gravitational_potential(bl, aux) =
    ∇gravitational_potential(bl.orientation, aux)

turbulence_tensors(atmos::AtmosModel, args...) = turbulence_tensors(
    turbulence_model(atmos),
    viscoussponge_model(atmos),
    atmos,
    args...,
)

"""
    density(atmos::AtmosModel, state::Vars, aux::Vars)

Density used in the conservative form of the prognostic equations.
In the Compressible case, it is equal to the prognostic density,
whereas in the Anelastic1D case it is the reference density,
which is constant in time.
"""
density(atmos::AtmosModel, state::Vars, aux::Vars) =
    density(compressibility_model(atmos), state, aux)
density(::Compressible, state, aux) = state.ρ
density(::Anelastic1D, state, aux) = aux.ref_state.ρ

"""
    pressure(atmos::AtmosModel, ts, aux::Vars)

Diagnostic pressure consistent with the given thermodynamic state ts.
In the Anelastic1D case it is the reference pressure,
which is constant in time.
"""
pressure(atmos::AtmosModel, ts, aux::Vars) =
    pressure(compressibility_model(atmos), ts, aux)
pressure(::Compressible, ts, aux) = air_pressure(ts)
pressure(::Anelastic1D, ts, aux) = aux.ref_state.p

include("declare_prognostic_vars.jl") # declare prognostic variables
include("equations_form.jl")           # types for split forms, etc.
include("multiphysics_types.jl")      # types for multi-physics tendencies
include("tendencies_mass.jl")         # specify mass tendencies
include("tendencies_momentum.jl")     # specify momentum tendencies
include("tendencies_energy.jl")       # specify energy tendencies
include("tendencies_moisture.jl")     # specify moisture tendencies
include("tendencies_precipitation.jl")# specify precipitation tendencies
include("tendencies_tracers.jl")      # specify tracer tendencies

include("problem.jl")
include("ref_state.jl")
include("moisture.jl")
include("energy.jl")
include("precipitation.jl")
include("thermo_states.jl")
include("thermo_states_anelastic.jl")
include("radiation.jl")
include("tracers.jl")
include("lsforcing.jl")
include("linear.jl")
include("courant.jl")
include("filters.jl")
include("prog_prim_conversion.jl")   # prognostic<->primitive conversion
include("reconstructions.jl")   # finite-volume method reconstructions
include("projections.jl")            # include term-by-term projectinos

include("linear_tendencies.jl")
include("linear_atmos_tendencies.jl")

include("atmos_tendencies.jl")        # specify atmos tendencies
include("get_prognostic_vars.jl")     # get tuple of prognostic variables


sub_model(atmos::AtmosModel, ::Type{<:AbstractEnergyModel}) =
    energy_model(atmos)
sub_model(atmos::AtmosModel, ::Type{<:AbstractMoistureModel}) =
    moisture_model(atmos)


function precompute(atmos::AtmosModel, args, tt::Flux{FirstOrder})
    ts = recover_thermo_state(atmos, args.state, args.aux)
    turbconv = precompute(turbconv_model(atmos), atmos, args, ts, tt)
    return (; ts, turbconv)
end

function compute_gradient_argument!(
    atmos::AtmosModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    ρinv = 1 / state.ρ
    transform.u = ρinv * state.ρu

    compute_gradient_argument!(
        energy_model(atmos),
        atmos,
        transform,
        state,
        aux,
        t,
    )
    compute_gradient_argument!(moisture_model(atmos), transform, state, aux, t)
    compute_gradient_argument!(
        precipitation_model(atmos),
        transform,
        state,
        aux,
        t,
    )
    compute_gradient_argument!(
        turbulence_model(atmos),
        transform,
        state,
        aux,
        t,
    )
    compute_gradient_argument!(
        hyperdiffusion_model(atmos),
        atmos,
        transform,
        state,
        aux,
        t,
    )
    compute_gradient_argument!(tracer_model(atmos), transform, state, aux, t)
    compute_gradient_argument!(lsforcing_model(atmos), transform, state, aux, t)
    compute_gradient_argument!(
        turbconv_model(atmos),
        atmos,
        transform,
        state,
        aux,
        t,
    )
end

function compute_gradient_flux!(
    atmos::AtmosModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    compute_gradient_flux!(
        energy_model(atmos),
        diffusive,
        ∇transform,
        state,
        aux,
        t,
    )

    # diffusion terms required for SGS turbulence computations
    compute_gradient_flux!(
        turbulence_model(atmos),
        atmos.orientation,
        diffusive,
        ∇transform,
        state,
        aux,
        t,
    )
    # diffusivity of moisture components
    compute_gradient_flux!(
        moisture_model(atmos),
        diffusive,
        ∇transform,
        state,
        aux,
        t,
    )
    compute_gradient_flux!(
        lsforcing_model(atmos),
        diffusive,
        ∇transform,
        state,
        aux,
        t,
    )
    compute_gradient_flux!(
        precipitation_model(atmos),
        diffusive,
        ∇transform,
        state,
        aux,
        t,
    )
    compute_gradient_flux!(
        tracer_model(atmos),
        diffusive,
        ∇transform,
        state,
        aux,
        t,
    )
    compute_gradient_flux!(
        turbconv_model(atmos),
        atmos,
        diffusive,
        ∇transform,
        state,
        aux,
        t,
    )
end

function transform_post_gradient_laplacian!(
    atmos::AtmosModel,
    hyperdiffusive::Vars,
    hypertransform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform_post_gradient_laplacian!(
        hyperdiffusion_model(atmos),
        atmos,
        hyperdiffusive,
        hypertransform,
        state,
        aux,
        t,
    )
end

function precompute(atmos::AtmosModel, args, tt::Flux{SecondOrder})
    @unpack state, diffusive, aux, t = args
    ts = recover_thermo_state(atmos, state, aux)
    ν, D_t, τ = turbulence_tensors(atmos, state, diffusive, aux, t)
    turbulence = (ν = ν, D_t = D_t, τ = τ)
    turbconv = precompute(turbconv_model(atmos), atmos, args, ts, tt)
    return (; ts, turbconv, turbulence)
end

soundspeed_air(ts::ThermodynamicState, ::Anelastic1D) = 0
soundspeed_air(ts::ThermodynamicState, ::Compressible) = soundspeed_air(ts)
@inline function wavespeed(
    m::AtmosModel,
    nM,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    ρinv = 1 / state.ρ
    u = ρinv * state.ρu
    uN = abs(dot(nM, u))
    ts = recover_thermo_state(m, state, aux)
    ss = soundspeed_air(ts, compressibility_model(m))
    FT = typeof(state.ρ)
    ws = fill(uN + ss, MVector{number_states(m, Prognostic()), FT})
    vars_ws = Vars{vars_state(m, Prognostic(), FT)}(ws)

    wavespeed_tracers!(tracer_model(m), vars_ws, nM, state, aux, t)

    return ws
end


function update_auxiliary_state!(
    dg::DGModel,
    m::AtmosModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    FT = eltype(Q)
    state_auxiliary = dg.state_auxiliary

    if number_states(m, UpwardIntegrals()) > 0
        indefinite_stack_integral!(dg, m, Q, state_auxiliary, t, elems)
        reverse_indefinite_stack_integral!(dg, m, Q, state_auxiliary, t, elems)
    end

    update_auxiliary_state!(nodal_update_auxiliary_state!, dg, m, Q, t, elems)

    # TODO: Remove this hook. This hook was added for implementing
    # the first draft of EDMF, and should be removed so that we can
    # rely on a single vertical element traversal. This hook allows
    # us to compute globally vertical quantities specific to EDMF
    # until we're able to remove them or somehow incorporate them
    # into a higher level hierarchy.
    update_auxiliary_state!(dg, turbconv_model(m), m, Q, t, elems)

    return true
end

function nodal_update_auxiliary_state!(
    m::AtmosModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    atmos_nodal_update_auxiliary_state!(moisture_model(m), m, state, aux, t)
    atmos_nodal_update_auxiliary_state!(
        precipitation_model(m),
        m,
        state,
        aux,
        t,
    )
    atmos_nodal_update_auxiliary_state!(radiation_model(m), m, state, aux, t)
    atmos_nodal_update_auxiliary_state!(tracer_model(m), m, state, aux, t)
    turbconv_nodal_update_auxiliary_state!(turbconv_model(m), m, state, aux, t)
end

function integral_load_auxiliary_state!(
    m::AtmosModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
)
    integral_load_auxiliary_state!(radiation_model(m), integ, state, aux)
    integral_load_auxiliary_state!(turbconv_model(m), m, integ, state, aux)
end

function integral_set_auxiliary_state!(m::AtmosModel, aux::Vars, integ::Vars)
    integral_set_auxiliary_state!(radiation_model(m), aux, integ)
    integral_set_auxiliary_state!(turbconv_model(m), m, aux, integ)
end

function reverse_integral_load_auxiliary_state!(
    m::AtmosModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
)
    reverse_integral_load_auxiliary_state!(
        radiation_model(m),
        integ,
        state,
        aux,
    )
end

function reverse_integral_set_auxiliary_state!(
    m::AtmosModel,
    aux::Vars,
    integ::Vars,
)
    reverse_integral_set_auxiliary_state!(radiation_model(m), aux, integ)
end

function atmos_nodal_init_state_auxiliary!(
    m::AtmosModel,
    aux::Vars,
    tmp::Vars,
    geom::LocalGeometry,
)
    aux.coord = geom.coord
    init_aux_turbulence!(turbulence_model(m), m, aux, geom)
    init_aux_hyperdiffusion!(hyperdiffusion_model(m), m, aux, geom)
    atmos_init_aux!(tracer_model(m), m, aux, geom)
    init_aux_turbconv!(turbconv_model(m), m, aux, geom)
    m.problem.init_state_auxiliary(m.problem, m, aux, geom)
end

"""
    init_state_auxiliary!(
        m::AtmosModel,
        aux::Vars,
        grid,
        direction
    )

Initialise auxiliary variables for each AtmosModel subcomponent.
Store Cartesian coordinate information in `aux.coord`.
"""
function init_state_auxiliary!(
    m::AtmosModel,
    state_auxiliary::MPIStateArray,
    grid,
    direction,
)
    # update the geopotential Φ in state_auxiliary.orientation.Φ
    init_aux!(m, m.orientation, state_auxiliary, grid, direction)
    atmos_init_aux!(m, reference_state(m), state_auxiliary, grid, direction)

    init_state_auxiliary!(
        m,
        atmos_nodal_init_state_auxiliary!,
        state_auxiliary,
        grid,
        direction,
    )
end

function precompute(atmos::AtmosModel, args, tt::Source)
    ts = recover_thermo_state(atmos, args.state, args.aux)
    precipitation = precompute(precipitation_model(atmos), atmos, args, ts, tt)
    turbconv = precompute(turbconv_model(atmos), atmos, args, ts, tt)
    return (; ts, turbconv, precipitation)
end

"""
    init_state_prognostic!(
        m::AtmosModel,
        state::Vars,
        aux::Vars,
        localgeo,
        t,
        args...,
    )

Initialise state variables. `args...` provides an option to include
configuration data (current use cases include problem constants,
spline-interpolants).
"""
function init_state_prognostic!(
    m::AtmosModel,
    state::Vars,
    aux::Vars,
    localgeo,
    t,
    args...,
)
    m.problem.init_state_prognostic(
        m.problem,
        m,
        state,
        aux,
        localgeo,
        t,
        args...,
    )
end

roe_average(ρ⁻, ρ⁺, var⁻, var⁺) =
    (sqrt(ρ⁻) * var⁻ + sqrt(ρ⁺) * var⁺) / (sqrt(ρ⁻) + sqrt(ρ⁺))

function numerical_flux_first_order!(
    numerical_flux::RoeNumericalFlux,
    balance_law::AtmosModel,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}
    @assert moisture_model(balance_law) isa DryModel

    numerical_flux_first_order!(
        CentralNumericalFluxFirstOrder(),
        balance_law,
        fluxᵀn,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )

    FT = eltype(fluxᵀn)
    param_set = parameter_set(balance_law)
    _cv_d::FT = cv_d(param_set)
    _T_0::FT = T_0(param_set)

    Φ = gravitational_potential(balance_law, state_auxiliary⁻)

    ρ⁻ = state_prognostic⁻.ρ
    ρu⁻ = state_prognostic⁻.ρu
    ρe⁻ = state_prognostic⁻.energy.ρe
    ts⁻ = recover_thermo_state(balance_law, state_prognostic⁻, state_auxiliary⁻)

    u⁻ = ρu⁻ / ρ⁻
    uᵀn⁻ = u⁻' * normal_vector
    e⁻ = ρe⁻ / ρ⁻
    h⁻ = total_specific_enthalpy(ts⁻, e⁻)
    p⁻ = air_pressure(ts⁻)
    c⁻ = soundspeed_air(ts⁻)

    ρ⁺ = state_prognostic⁺.ρ
    ρu⁺ = state_prognostic⁺.ρu
    ρe⁺ = state_prognostic⁺.energy.ρe

    # TODO: state_auxiliary⁺ is not up-to-date
    # with state_prognostic⁺ on the boundaries
    ts⁺ = recover_thermo_state(balance_law, state_prognostic⁺, state_auxiliary⁺)

    u⁺ = ρu⁺ / ρ⁺
    uᵀn⁺ = u⁺' * normal_vector
    e⁺ = ρe⁺ / ρ⁺
    h⁺ = total_specific_enthalpy(ts⁺, e⁺)
    p⁺ = air_pressure(ts⁺)
    c⁺ = soundspeed_air(ts⁺)

    ρ̃ = sqrt(ρ⁻ * ρ⁺)
    ũ = roe_average(ρ⁻, ρ⁺, u⁻, u⁺)
    h̃ = roe_average(ρ⁻, ρ⁺, h⁻, h⁺)
    c̃ = sqrt(roe_average(ρ⁻, ρ⁺, c⁻^2, c⁺^2))

    ũᵀn = ũ' * normal_vector

    Δρ = ρ⁺ - ρ⁻
    Δp = p⁺ - p⁻
    Δu = u⁺ - u⁻
    Δuᵀn = Δu' * normal_vector

    w1 = abs(ũᵀn - c̃) * (Δp - ρ̃ * c̃ * Δuᵀn) / (2 * c̃^2)
    w2 = abs(ũᵀn + c̃) * (Δp + ρ̃ * c̃ * Δuᵀn) / (2 * c̃^2)
    w3 = abs(ũᵀn) * (Δρ - Δp / c̃^2)
    w4 = abs(ũᵀn) * ρ̃

    fluxᵀn.ρ -= (w1 + w2 + w3) / 2
    fluxᵀn.ρu -=
        (
            w1 * (ũ - c̃ * normal_vector) +
            w2 * (ũ + c̃ * normal_vector) +
            w3 * ũ +
            w4 * (Δu - Δuᵀn * normal_vector)
        ) / 2
    fluxᵀn.energy.ρe -=
        (
            w1 * (h̃ - c̃ * ũᵀn) +
            w2 * (h̃ + c̃ * ũᵀn) +
            w3 * (ũ' * ũ / 2 + Φ - _T_0 * _cv_d) +
            w4 * (ũ' * Δu - ũᵀn * Δuᵀn)
        ) / 2

    if !(tracer_model(balance_law) isa NoTracers)
        ρχ⁻ = state_prognostic⁻.tracers.ρχ
        χ⁻ = ρχ⁻ / ρ⁻

        ρχ⁺ = state_prognostic⁺.tracers.ρχ
        χ⁺ = ρχ⁺ / ρ⁺

        χ̃ = roe_average(ρ⁻, ρ⁺, χ⁻, χ⁺)
        Δρχ = ρχ⁺ - ρχ⁻

        wt = abs(ũᵀn) * (Δρχ - χ̃ * Δp / c̃^2)

        fluxᵀn.tracers.ρχ -= ((w1 + w2) * χ̃ + wt) / 2
    end
end

"""
    NumericalFluxFirstOrder()
        ::HLLCNumericalFlux,
        balance_law::AtmosModel,
        fluxᵀn,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )

An implementation of the numerical flux based on the HLLC method for
the AtmosModel. For more information on this particular implementation,
see Chapter 10.4 in the provided reference below.

## References

 - [Toro2013](@cite)

"""
function numerical_flux_first_order!(
    ::HLLCNumericalFlux,
    balance_law::AtmosModel,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}
    FT = eltype(fluxᵀn)
    num_state_prognostic = number_states(balance_law, Prognostic())
    param_set = parameter_set(balance_law)

    # Extract the first-order fluxes from the AtmosModel (underlying BalanceLaw)
    # and compute normals on the positive + and negative - sides of the
    # interior facets
    flux⁻ = similar(parent(fluxᵀn), Size(3, num_state_prognostic))
    fill!(flux⁻, -zero(FT))
    total_flux_first_order!(
        balance_law,
        Grad{S}(flux⁻),
        state_prognostic⁻,
        state_auxiliary⁻,
        t,
        direction,
    )
    fluxᵀn⁻ = flux⁻' * normal_vector

    flux⁺ = similar(flux⁻)
    fill!(flux⁺, -zero(FT))
    total_flux_first_order!(
        balance_law,
        Grad{S}(flux⁺),
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )
    fluxᵀn⁺ = flux⁺' * normal_vector

    # Extract relevant fields and thermodynamic variables defined on
    # the positive + and negative - sides of the interior facets
    ρ⁻ = state_prognostic⁻.ρ
    ρu⁻ = state_prognostic⁻.ρu
    ρe⁻ = state_prognostic⁻.energy.ρe
    ts⁻ = recover_thermo_state(balance_law, state_prognostic⁻, state_auxiliary⁻)

    u⁻ = ρu⁻ / ρ⁻
    c⁻ = soundspeed_air(ts⁻)

    uᵀn⁻ = u⁻' * normal_vector
    e⁻ = ρe⁻ / ρ⁻
    p⁻ = air_pressure(ts⁻)

    ρ⁺ = state_prognostic⁺.ρ
    ρu⁺ = state_prognostic⁺.ρu
    ρe⁺ = state_prognostic⁺.energy.ρe
    ts⁺ = recover_thermo_state(balance_law, state_prognostic⁺, state_auxiliary⁺)

    u⁺ = ρu⁺ / ρ⁺
    uᵀn⁺ = u⁺' * normal_vector
    e⁺ = ρe⁺ / ρ⁺
    p⁺ = air_pressure(ts⁺)
    c⁺ = soundspeed_air(ts⁺)

    # Wave speeds estimates S⁻ and S⁺
    S⁻ = min(uᵀn⁻ - c⁻, uᵀn⁺ - c⁺)
    S⁺ = max(uᵀn⁻ + c⁻, uᵀn⁺ + c⁺)

    # Compute the middle wave speed S⁰ in the contact/star region
    S⁰ =
        (p⁺ - p⁻ + ρ⁻ * uᵀn⁻ * (S⁻ - uᵀn⁻) - ρ⁺ * uᵀn⁺ * (S⁺ - uᵀn⁺)) /
        (ρ⁻ * (S⁻ - uᵀn⁻) - ρ⁺ * (S⁺ - uᵀn⁺))

    p⁰ =
        (
            p⁺ +
            p⁻ +
            ρ⁻ * (S⁻ - uᵀn⁻) * (S⁰ - uᵀn⁻) +
            ρ⁺ * (S⁺ - uᵀn⁺) * (S⁰ - uᵀn⁺)
        ) / 2

    # Compute p * D = p * (0, n₁, n₂, n₃, S⁰)
    pD = @MVector zeros(FT, num_state_prognostic)
    ref_state = reference_state(balance_law)
    if ref_state isa HydrostaticState && ref_state.subtract_off
        # pressure should be continuous but it doesn't hurt to average
        ref_p⁻ = state_auxiliary⁻.ref_state.p
        ref_p⁺ = state_auxiliary⁺.ref_state.p
        ref_p⁰ = (ref_p⁻ + ref_p⁺) / 2

        momentum_p = p⁰ - ref_p⁰
    else
        momentum_p = p⁰
    end

    pD[2] = momentum_p * normal_vector[1]
    pD[3] = momentum_p * normal_vector[2]
    pD[4] = momentum_p * normal_vector[3]
    pD[5] = p⁰ * S⁰

    # Computes both +/- sides of intermediate flux term flux⁰
    flux⁰⁻ =
        (S⁰ * (S⁻ * parent(state_prognostic⁻) - fluxᵀn⁻) + S⁻ * pD) / (S⁻ - S⁰)
    flux⁰⁺ =
        (S⁰ * (S⁺ * parent(state_prognostic⁺) - fluxᵀn⁺) + S⁺ * pD) / (S⁺ - S⁰)

    if 0 <= S⁻
        parent(fluxᵀn) .= fluxᵀn⁻
    elseif S⁻ < 0 <= S⁰
        parent(fluxᵀn) .= flux⁰⁻
    elseif S⁰ < 0 <= S⁺
        parent(fluxᵀn) .= flux⁰⁺
    else # 0 > S⁺
        parent(fluxᵀn) .= fluxᵀn⁺
    end
end

function numerical_flux_first_order!(
    numerical_flux::RoeNumericalFluxMoist,
    balance_law::AtmosModel,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}
    moisture_model(balance_law) isa EquilMoist ||
        error("Must use a EquilMoist model for RoeNumericalFluxMoist")
    numerical_flux_first_order!(
        CentralNumericalFluxFirstOrder(),
        balance_law,
        fluxᵀn,
        normal_vector,
        state_prognostic⁻,
        state_auxiliary⁻,
        state_prognostic⁺,
        state_auxiliary⁺,
        t,
        direction,
    )

    FT = eltype(fluxᵀn)
    param_set = parameter_set(balance_law)
    _cv_d::FT = cv_d(param_set)
    _T_0::FT = T_0(param_set)
    γ::FT = cp_d(param_set) / cv_d(param_set)
    _e_int_v0::FT = e_int_v0(param_set)
    Φ = gravitational_potential(balance_law, state_auxiliary⁻)

    ρ⁻ = state_prognostic⁻.ρ
    ρu⁻ = state_prognostic⁻.ρu
    ρe⁻ = state_prognostic⁻.energy.ρe
    ρq_tot⁻ = state_prognostic⁻.moisture.ρq_tot

    u⁻ = ρu⁻ / ρ⁻
    e⁻ = ρe⁻ / ρ⁻
    ts⁻ = recover_thermo_state(balance_law, state_prognostic⁻, state_auxiliary⁻)
    h⁻ = total_specific_enthalpy(ts⁻, e⁻)
    qt⁻ = ρq_tot⁻ / ρ⁻
    c⁻ = soundspeed_air(ts⁻)

    ρ⁺ = state_prognostic⁺.ρ
    ρu⁺ = state_prognostic⁺.ρu
    ρe⁺ = state_prognostic⁺.energy.ρe
    ρq_tot⁺ = state_prognostic⁺.moisture.ρq_tot

    u⁺ = ρu⁺ / ρ⁺
    e⁺ = ρe⁺ / ρ⁺
    ts⁺ = recover_thermo_state(balance_law, state_prognostic⁺, state_auxiliary⁺)
    h⁺ = total_specific_enthalpy(ts⁺, e⁺)
    qt⁺ = ρq_tot⁺ / ρ⁺
    c⁺ = soundspeed_air(ts⁺)
    ũ = roe_average(ρ⁻, ρ⁺, u⁻, u⁺)
    e_tot = roe_average(ρ⁻, ρ⁺, e⁻, e⁺)
    h̃ = roe_average(ρ⁻, ρ⁺, h⁻, h⁺)
    qt = roe_average(ρ⁻, ρ⁺, qt⁻, qt⁺)
    ρ = sqrt(ρ⁻ * ρ⁺)
    e_int⁻ = internal_energy(ts⁻)
    e_int⁺ = internal_energy(ts⁺)
    e_int = roe_average(ρ⁻, ρ⁺, e_int⁻, e_int⁺)
    ts = PhaseEquil(
        param_set,
        e_int,
        ρ,
        qt,
        moisture_model(balance_law).maxiter,
        moisture_model(balance_law).tolerance,
    )
    c̃ = sqrt((γ - 1) * (h̃ - (ũ[1]^2 + ũ[2]^2 + ũ[3]^2) / 2))
    (R_m, _cp_m, _cv_m, gamma) = gas_constants(ts)
    # chosen by fair dice roll
    # guaranteed to be random
    ω = FT(π) / 3
    δ = FT(π) / 5
    random_unit_vector = SVector(sin(ω) * cos(δ), cos(ω) * cos(δ), sin(δ))
    # tangent space basis
    τ1 = random_unit_vector × normal_vector
    τ2 = τ1 × normal_vector
    ũᵀn⁻ = u⁻' * normal_vector
    ũᵀn⁺ = u⁺' * normal_vector
    ũᵀn = ũ' * normal_vector
    ũc̃⁻ = ũ + c̃ * normal_vector
    ũc̃⁺ = ũ - c̃ * normal_vector
    e_kin_pot = h̃ - _e_int_v0 * qt - _cp_m * c̃^2 / R_m
    if (numerical_flux.LM == true)
        Mach⁺ = sqrt(u⁺' * u⁺) / c⁺
        Mach⁻ = sqrt(u⁻' * u⁻) / c⁻
        Mach = (Mach⁺ + Mach⁻) / 2
        c̃_LM = c̃ * min(Mach * sqrt(4 + (1 - Mach^2)^2) / (1 + Mach^2), 1)
    else
        c̃_LM = c̃
    end
    #Standard Roe
    Λ = SDiagonal(
        abs(ũᵀn - c̃_LM),
        abs(ũᵀn),
        abs(ũᵀn),
        abs(ũᵀn),
        abs(ũᵀn + c̃_LM),
        abs(ũᵀn),
    )
    #Harten Hyman
    if (numerical_flux.HH == true)
        Λ = SDiagonal(
            max(
                abs(ũᵀn - c̃_LM),
                max(
                    0,
                    ũᵀn - c̃_LM - (u⁻' * normal_vector - c⁻),
                    u⁺' * normal_vector - c⁺ - (ũᵀn - c̃_LM),
                ),
            ),
            max(
                abs(ũᵀn),
                max(
                    0,
                    ũᵀn - (u⁻' * normal_vector),
                    u⁺' * normal_vector - (ũᵀn),
                ),
            ),
            max(
                abs(ũᵀn),
                max(
                    0,
                    ũᵀn - (u⁻' * normal_vector),
                    u⁺' * normal_vector - (ũᵀn),
                ),
            ),
            max(
                abs(ũᵀn),
                max(
                    0,
                    ũᵀn - (u⁻' * normal_vector),
                    u⁺' * normal_vector - (ũᵀn),
                ),
            ),
            max(
                abs(ũᵀn + c̃_LM),
                max(
                    0,
                    ũᵀn + c̃_LM - (u⁻' * normal_vector + c⁻),
                    u⁺' * normal_vector + c⁺ - (ũᵀn + c̃_LM),
                ),
            ),
            max(
                abs(ũᵀn),
                max(
                    0,
                    ũᵀn - (u⁻' * normal_vector),
                    u⁺' * normal_vector - (ũᵀn),
                ),
            ),
        )
    end
    if (numerical_flux.LV == true)
        #Pseudo LeVeque Fix
        δ_L_1 = max(0, ũᵀn - ũᵀn⁻)
        δ_L_2 = max(0, ũᵀn - c̃_LM - (ũᵀn⁻ - c⁻))
        δ_L_3 = max(0, ũᵀn + c̃_LM - (ũᵀn⁻ + c⁻))
        δ_R_1 = max(0, ũᵀn⁺ - ũᵀn)
        δ_R_2 = max(0, ũᵀn⁺ - c⁺ - (ũᵀn - c̃_LM))
        δ_R_3 = max(0, ũᵀn⁺ + c⁺ - (ũᵀn + c̃_LM))
        if (ũᵀn < δ_L_1 && ũᵀn > -δ_R_1)
            qa1 = ((δ_L_1 - δ_R_1) * ũᵀn + 2 * δ_L_1 * δ_R_1) / (δ_L_1 + δ_R_1)
        else
            qa1 = abs(ũᵀn)
        end
        if (ũᵀn - c̃ < δ_L_2 && ũᵀn - c̃_LM > -δ_R_2)
            qa2 =
                ((δ_L_2 - δ_R_2) * (ũᵀn - c̃_LM) + 2 * δ_L_2 * δ_R_2) /
                (δ_L_2 + δ_R_2)
        else
            qa2 = abs(ũᵀn - c̃_LM)
        end
        if (ũᵀn + c̃_LM < δ_L_3 && ũᵀn + c̃ > -δ_R_3)
            qa3 =
                ((δ_L_3 - δ_R_3) * (ũᵀn + c̃_LM) + 2 * δ_R_3 * δ_R_3) /
                (δ_L_3 + δ_R_3)
        else
            qa3 = abs(ũᵀn + c̃_LM)
        end
        Λ = SDiagonal(qa2, qa1, qa1, qa1, qa3, qa1)
    end
    if (numerical_flux.LVPP == true)
        #PosPreserving with LeVeque
        b_L = min(ũᵀn - c̃_LM, ũᵀn⁻ - c⁻)
        b_R = max(ũᵀn + c̃_LM, ũᵀn⁺ + c⁺)
        b⁻ = min(0, b_L)
        b⁺ = max(0, b_R)
        δ_L_1 = max(0, ũᵀn - b⁻)
        δ_L_2 = max(0, ũᵀn - c̃_LM - b⁻)
        δ_L_3 = max(0, ũᵀn + c̃_LM - b⁻)
        δ_R_1 = max(0, b⁺ - ũᵀn)
        δ_R_2 = max(0, b⁺ - (ũᵀn - c̃_LM))
        δ_R_3 = max(0, b⁺ - (ũᵀn + c̃_LM))
        if (ũᵀn < δ_L_1 && ũᵀn > -δ_R_1)
            qa1 = ((δ_L_1 - δ_R_1) * ũᵀn + 2 * δ_L_1 * δ_R_1) / (δ_L_1 + δ_R_1)
        else
            qa1 = abs(ũᵀn)
        end
        if (ũᵀn - c̃_LM < δ_L_2 && ũᵀn - c̃_LM > -δ_R_2)
            qa2 =
                ((δ_L_2 - δ_R_2) * (ũᵀn - c̃) + 2 * δ_L_2 * δ_R_2) /
                (δ_L_2 + δ_R_2)
        else
            qa2 = abs(ũᵀn - c̃_LM)
        end
        if (ũᵀn + c̃_LM < δ_L_3 && ũᵀn + c̃_LM > -δ_R_3)
            qa3 =
                ((δ_L_3 - δ_R_3) * (ũᵀn + c̃_LM) + 2 * δ_R_3 * δ_R_3) /
                (δ_L_3 + δ_R_3)
        else
            qa3 = abs(ũᵀn + c̃_LM)
        end
        Λ = SDiagonal(qa2, qa1, qa1, qa1, qa3, qa1)
    end

    M = hcat(
        SVector(1, ũc̃⁺[1], ũc̃⁺[2], ũc̃⁺[3], h̃ - c̃ * ũᵀn, qt),
        SVector(0, τ1[1], τ1[2], τ1[3], τ1' * ũ, 0),
        SVector(0, τ2[1], τ2[2], τ2[3], τ2' * ũ, 0),
        SVector(1, ũ[1], ũ[2], ũ[3], ũ' * ũ / 2 + Φ - _T_0 * _cv_m, 0),
        SVector(1, ũc̃⁻[1], ũc̃⁻[2], ũc̃⁻[3], h̃ + c̃ * ũᵀn, qt),
        SVector(0, 0, 0, 0, _e_int_v0, 1),
    )
    Δρ = ρ⁺ - ρ⁻
    Δρu = ρu⁺ - ρu⁻
    Δρe = ρe⁺ - ρe⁻
    Δρq_tot = ρq_tot⁺ - ρq_tot⁻
    Δstate = SVector(Δρ, Δρu[1], Δρu[2], Δρu[3], Δρe, Δρq_tot)
    parent(fluxᵀn) .-= M * Λ * (M \ Δstate) / 2
end

function numerical_flux_first_order!(
    numerical_flux::LMARSNumericalFlux,
    balance_law::AtmosModel,
    fluxᵀn::Vars{S},
    normal_vector::SVector,
    state_prognostic⁻::Vars{S},
    state_auxiliary⁻::Vars{A},
    state_prognostic⁺::Vars{S},
    state_auxiliary⁺::Vars{A},
    t,
    direction,
) where {S, A}


    @assert moisture_model(balance_law) isa DryModel ||
            moisture_model(balance_law) isa EquilMoist

    FT = eltype(fluxᵀn)
    param_set = parameter_set(balance_law)

    ρ⁻ = state_prognostic⁻.ρ
    ρu⁻ = state_prognostic⁻.ρu
    ρe⁻ = state_prognostic⁻.energy.ρe
    ts⁻ = recover_thermo_state(balance_law, state_prognostic⁻, state_auxiliary⁻)

    u⁻ = ρu⁻ / ρ⁻
    e⁻ = ρe⁻ / ρ⁻
    uᵀn⁻ = u⁻' * normal_vector
    p⁻ = air_pressure(ts⁻)
    ref_state = reference_state(balance_law)
    if ref_state isa HydrostaticState && ref_state.subtract_off
        p⁻ -= state_auxiliary⁻.ref_state.p
    end
    c⁻ = soundspeed_air(ts⁻)
    h⁻ = total_specific_enthalpy(ts⁻, e⁻)

    ρ⁺ = state_prognostic⁺.ρ
    ρu⁺ = state_prognostic⁺.ρu
    ρe⁺ = state_prognostic⁺.energy.ρe
    ts⁺ = recover_thermo_state(balance_law, state_prognostic⁺, state_auxiliary⁺)
    u⁺ = ρu⁺ / ρ⁺
    e⁺ = ρe⁺ / ρ⁺
    uᵀn⁺ = u⁺' * normal_vector
    p⁺ = air_pressure(ts⁺)
    if ref_state isa HydrostaticState && ref_state.subtract_off
        p⁺ -= state_auxiliary⁺.ref_state.p
    end
    c⁺ = soundspeed_air(ts⁺)
    h⁺ = total_specific_enthalpy(ts⁺, e⁺)

    # Eqn (49), (50), β the tuning parameter
    β = FT(1)
    u_half = 1 / 2 * (uᵀn⁺ + uᵀn⁻) - β * 1 / (ρ⁻ + ρ⁺) / c⁻ * (p⁺ - p⁻)
    p_half = 1 / 2 * (p⁺ + p⁻) - β * ((ρ⁻ + ρ⁺) * c⁻) / 4 * (uᵀn⁺ - uᵀn⁻)

    # Eqn (46), (47)
    ρ_b = u_half > FT(0) ? ρ⁻ : ρ⁺
    ρu_b = u_half > FT(0) ? ρu⁻ : ρu⁺
    ρh_b = u_half > FT(0) ? ρ⁻ * h⁻ : ρ⁺ * h⁺

    # Update fluxes Eqn (18)
    fluxᵀn.ρ = ρ_b * u_half
    fluxᵀn.ρu = ρu_b * u_half .+ p_half * normal_vector
    fluxᵀn.energy.ρe = ρh_b * u_half

    if moisture_model(balance_law) isa EquilMoist
        ρq⁻ = state_prognostic⁻.moisture.ρq_tot
        q⁻ = ρq⁻ / ρ⁻
        ρq⁺ = state_prognostic⁺.moisture.ρq_tot
        q⁺ = ρq⁺ / ρ⁺
        ρq_b = u_half > FT(0) ? ρq⁻ : ρq⁺
        fluxᵀn.moisture.ρq_tot = ρq_b * u_half
    end
    if !(tracer_model(balance_law) isa NoTracers)
        ρχ⁻ = state_prognostic⁻.tracers.ρχ
        χ⁻ = ρχ⁻ / ρ⁻
        ρχ⁺ = state_prognostic⁺.tracers.ρχ
        χ⁺ = ρχ⁺ / ρ⁺
        ρχ_b = u_half > FT(0) ? ρχ⁻ : ρχ⁺
        fluxᵀn.tracers.ρχ = ρχ_b * u_half
    end
end

end # module
