#!/usr/bin/env julia --project

include("simple_box.jl")
ClimateMachine.init(parse_clargs = true)

# Float type
const FT = Float64

# simulation time
const timestart = FT(0)      # s
const timestep = FT(240)     # s
const timeend = FT(86400) # s
timespan = (timestart, timeend)

# DG polynomial order
const N = Int(4)

# Domain resolution
const Nˣ = Int(20)
const Nʸ = Int(20)
const Nᶻ = Int(20)
resolution = (N, Nˣ, Nʸ, Nᶻ)

# Domain size
const Lˣ = 4e6    # m
const Lʸ = 4e6    # m
const H = 1000   # m
dimensions = (Lˣ, Lʸ, H)

BC = (
    OceanBC(Impenetrable(NoSlip()), Insulating()),
    OceanBC(Impenetrable(NoSlip()), Insulating()),
    OceanBC(Penetrable(KinematicStress()), TemperatureFlux()),
)

run_simple_box(
    "ocean_gyre",
    resolution,
    dimensions,
    timespan,
    OceanGyre,
    imex = false,
    Δt = timestep,
    BC = BC,
)
