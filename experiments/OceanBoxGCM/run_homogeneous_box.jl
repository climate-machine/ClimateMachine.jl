#!/usr/bin/env julia --project

include("simple_box.jl")
ClimateMachine.init(parse_clargs = true)

# Float type
const FT = Float64

# simulation time
const timestart = FT(0)      # s
const timestep = FT(55)     # s
const timeend = FT(6 * 3600) # s
timespan = (timestart, timeend)

# DG polynomial order
const N = Int(4)

# Domain resolution
const Nˣ = Int(20)
const Nʸ = Int(20)
const Nᶻ = Int(50)
resolution = (N, Nˣ, Nʸ, Nᶻ)

# Domain size
const Lˣ = 4e6    # m
const Lʸ = 4e6    # m
const H = 400   # m
dimensions = (Lˣ, Lʸ, H)

BC = (
    OceanBC(Impenetrable(NoSlip()), Insulating()),
    OceanBC(Impenetrable(NoSlip()), Insulating()),
    OceanBC(Penetrable(KinematicStress()), Insulating()),
)

run_simple_box(
    "homogeneous_box",
    resolution,
    dimensions,
    timespan,
    HomogeneousBox,
    imex = true,
    Δt = timestep,
    BC = BC,
)
