#!/usr/bin/env julia --project

include("eddying_channel.jl")
ClimateMachine.init(parse_clargs = true)

using CLIMAParameters.Planet: grav

# Float type
const FT = Float64

# simulation time
const timestart = FT(0)      # s
const timestep = FT(240)     # s
const timeend = FT(20 * 86400) # s
timespan = (timestart, timeend)

function config_eddying_channel()
    # DG polynomial order
    N = Int(5)

    # Domain resolution
    Nˣ = Int(40)
    Nʸ = Int(40)
    Nᶻ = Int(10)
    resolution = (Nˣ, Nʸ, Nᶻ)

    # Domain size
    Lˣ = 1e6    # m
    Lʸ = 1e6    # m
    H = 2000   # m
    dimensions = (Lˣ, Lʸ, H)

    BC = (
        OceanBC(Impenetrable(FreeSlip()), Insulating()), # south wall
        OceanBC(Impenetrable(FreeSlip()), TemperatureFlux()), # north wall
        OceanBC(Impenetrable(NoSlip()), Insulating()), # floor
        OceanBC(Penetrable(KinematicStress()), Insulating()), # surface
    )

    problem = OceanGyre{FT}(dimensions...; BC = BC)

    _grav::FT = grav(param_set)
    cʰ = sqrt(_grav * problem.H) # m/s
    model = HydrostaticBoussinesqModel{FT}(
        param_set,
        problem,
        cʰ = cʰ,
        νʰ = 100,
        νᶻ = 0.02,
        κʰ = 100,
        κᶻ = 0.02,
    )

    config = ClimateMachine.OceanBoxGCMConfiguration(
        "eddying_channel",
        N,
        resolution,
        param_set,
        model;
        periodicity = (true, false, false),
        boundary = ((0, 0), (1, 2), (3, 4)),
    )

    return config
end

run_eddying_channel(config_eddying_channel, timespan, Δt = nothing)
