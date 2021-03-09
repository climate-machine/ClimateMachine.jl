using Test
using MPI

using ClimateMachine

# To test coupling
using ClimateMachine.Coupling

# To create meshes (borrowed from Ocean for now!)
using ClimateMachine.Ocean.Domains

# To setup some callbacks
using ClimateMachine.GenericCallbacks

# To invoke timestepper
using ClimateMachine.ODESolvers

using QuickVizExample
using GLMakie

ClimateMachine.init()

# Use toy balance law for now
include("CplTestingBL.jl")

# TODO: Use RecordState to produce plots every X steps, 
# create time slider for components
mutable struct RecordState
    atmos::Vector{Any}
    ocean::Vector{Any}
end
RecordState() = RecordState(Any[], Any[])
function (rs::RecordState)()
    println("***record callback***")
    push!(rs.atmos, deepcopy(mA.state))
    push!(rs.ocean, deepcopy(mO.state))
end
GenericCallbacks.init!(f::RecordState, solver, Q, param, t) = nothing
function GenericCallbacks.call!(f::RecordState, solver, Q, param, t)
    f()
    return nothing
end
GenericCallbacks.fini!(f::RecordState, solver, Q, param, t) = nothing

rs = RecordState()
recordcallback = EveryXSimulationSteps(rs, 10)

## Visualization
"""
    coupledviz(modellist, plottime)

Create a 3d volume plots for coupled model components.

Assumes Atmos, Ocean, Land ordering in the model list.
"""
function coupledviz(modellist, plottime = 0.0, np_def = 100; title = "")
    states = Array{Any, 1}(nothing, length(modellist))
    statenames = similar(states)
    names = ["θAtmos, t=$plottime", "θOcean, t=$plottime", "θLand, t=$plottime"]
    for (i, model) in enumerate(modellist)
        gh = GridHelper(model.grid)
        x, y, z = coordinates(model.grid)
        xr = range(minimum(x), maximum(x), length = np_def)
        yr = range(minimum(y), maximum(y), length = np_def)
        zr = range(minimum(z), maximum(z), length = np_def)

        # interpolate data
        fld = zeros(np_def, np_def, np_def)
        ϕ = ScalarField(copy(x), gh)
        ϕ .= view(model.state, :, 1, :)
        fld[:, :, :] .= view(ϕ(xr, yr, zr), :, :, :)
        states[i] = fld
        statenames[i] = names[i]
    end
    volumeslice(states, statenames = statenames, title = title)
    return states
end

include("simple_3testcomp.jl")
models = map(m -> m.component_model, cC.component_list)
coupledviz(models, 10.0, 100; title = "Test Plot")
