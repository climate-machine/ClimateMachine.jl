#=
const minimum_collect_interval = 0.01

"""
    DiagnosticsGroupParams

Base type for any extra parameters needed by a diagnostics group.
"""
abstract type DiagnosticsGroupParams end

"""
    DiagnosticsGroup

A diagnostics group contains a set of diagnostic variables that are
collected at the same interval and written to the same file. One or more
diagnostics groups are placed in a [`DiagnosticsConfiguration`](@ref)
which is used by [`ClimateMachine.invoke!`](@ref).

A `DiagnosticsGroup` can be easily created with
[`@diagnostics_group`](@ref).
"""
mutable struct DiagnosticsGroup{
    DGP <: Union{Nothing, DiagnosticsGroupParams},
    DGI <: Union{Nothing, InterpolationTopology},
}
    name::String
    init::Function
    collect::Function
    fini::Function
    last_collect_time::Float64
    interval::String
    out_prefix::String
    writer::AbstractWriter
    interpol::DGI
    onetime::Bool
    params::DGP

    DiagnosticsGroup(
        name,
        init,
        fini,
        collect,
        interval,
        out_prefix,
        writer,
        interpol,
        onetime,
        params,
    ) = new{typeof(params), typeof(interpol)}(
        name,
        init,
        collect,
        0.0,
        fini,
        interval,
        out_prefix,
        writer,
        interpol,
        onetime,
        params,
    )
end

# `GenericCallbacks` implementations for `DiagnosticsGroup`s
function GenericCallbacks.init!(dgngrp::DiagnosticsGroup, solver, Q, param, t)
    @info @sprintf(
        """
    Diagnostics: %s
        initializing at %8.2f""",
        dgngrp.name,
        t,
    )
    dgngrp.init(dgngrp, t)
    dgngrp.collect(dgngrp, t)
    dgngrp.last_collect_time = t
    return nothing
end
function GenericCallbacks.call!(dgngrp::DiagnosticsGroup, solver, Q, param, t)
    @tic diagnostics
    @info @sprintf(
        """
    Diagnostics: %s
        collecting at %8.2f""",
        dgngrp.name,
        t,
    )
    dgngrp.collect(dgngrp, t)
    dgngrp.last_collect_time = t
    @toc diagnostics
    return nothing
end
function GenericCallbacks.fini!(dgngrp::DiagnosticsGroup, solver, Q, param, t)
    @info @sprintf(
        """
    Diagnostics: %s
        finishing at %8.2f""",
        dgngrp.name,
        t,
    )
    if t - last_collect_time >= minimum_collect_interval
        dgngrp.collect(dgngrp, t)
    end
    dgngrp.fini(dgngrp, t)
    return nothing
end
=#
abstract type InterpolationType end
struct NoInterpolation <: InterpolationType end
struct InterpolateAfterCollection <: InterpolationType end
struct InterpolateBeforeCollection <: InterpolationType end

"""
    @diagnostics_group

Generate the functions needed to establish and
use a [`DiagnosticsGroup`](@ref) containing the named
[`DiagnosticVar`](@ref)s. In particular, this creates `$(name)()`,
which creates the diagnostics group.

# Arguments
- `name`: a string that uniquely identifies the group.
- `config_type`: a `ClimateMachineConfigType`.
- `params_type`: a subtype of `DiagnosticsGroupParams` or `Nothing`.
  An instance of this type is created in `dgngrp.params` when the
  group is set up.
- `init_fun`: a function that is called when the group is initialized
  (called with `(dgngrp, curr_time); `may be `(_...) -> nothing`). Use
  this to initialize any required state, such as in `dgngrp.params`.
- `interpolate`: one of `InterpolateBeforeCollection`,
  `InterpolateAfterCollection` or `NoInterpolation`.
- `dvarnames`: one or more diagnostic variables. Together with the
  `config_type`, identify the [`DiagnosticVar`](@ref)s to be included
  in the group.
"""
macro diagnostics_group(
    name,
    config_type,
    params_type,
    init_fun,
    interpolate,
    dvarnames..., # TODO: need to separately specify which ones to output
)
    CT = getfield(ConfigTypes, config_type)
    dvars = DiagnosticVar[AllDiagnosticVars[CT][String(dv)] for dv in dvarnames]

    # Partition the diagnostic variables by type.
    dvtype_dvars_map = Dict{DataType, Array{DiagnosticVar, 1}}()
    for dvar in dvars
        push!(
            get!(dvtype_dvars_map, supertype(typeof(dvar)), DiagnosticVar[]),
            dvar,
        )
    end

    gen_params = (
        name,
        config_type,
        params_type,
        init_fun,
        interpolate,
        dvtype_dvars_map,
    )

    using_exprs = quote
        using KernelAbstractions
        using OrderedCollections
        using ClimateMachine.BalanceLaws
        using ClimateMachine.DGMethods
        using ClimateMachine.Mesh.Interpolation
        using ClimateMachine.Mesh.Topologies
        using ClimateMachine.MPIStateArrays
        using ClimateMachine.VariableTemplates
        using ClimateMachine.Writers
    end
    init_fun = esc(prewalk(unblock, generate_init_fun(gen_params...)))
    fini_fun = esc(prewalk(unblock, generate_fini_fun(gen_params...)))
    collect_fun = esc(prewalk(unblock, generate_collect_fun(gen_params...)))
    setup_fun = esc(prewalk(unblock, generate_setup_fun(gen_params...)))

    return Expr(:block, using_exprs, init_fun, fini_fun, collect_fun, setup_fun)
end

include("group_gen.jl")
