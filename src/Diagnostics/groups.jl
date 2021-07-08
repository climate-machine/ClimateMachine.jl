const minimum_collect_interval = 0.01

abstract type DiagnosticsGroupParams end

"""
    DiagnosticsGroup

Holds a set of diagnostics that share a collection interval, a filename
prefix, an output writer, an interpolation, and any extra parameters.
"""
mutable struct DiagnosticsGroup{DGP <: Union{Nothing, DiagnosticsGroupParams}}
    name::String
    init::Function
    fini::Function
    collect::Function
    last_collect_time::Float64
    interval::String
    out_prefix::String
    writer::AbstractWriter
    interpol::Union{Nothing, InterpolationTopology}
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
        params = nothing,
    ) = new{typeof(params)}(
        name,
        init,
        fini,
        collect,
        0.0,
        interval,
        out_prefix,
        writer,
        interpol,
        params,
    )
end

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
    if t - dgngrp.last_collect_time >= minimum_collect_interval
        dgngrp.collect(dgngrp, t)
    end
    dgngrp.fini(dgngrp, t)
    return nothing
end

include("atmos_les_default.jl")
include("atmos_gcm_default.jl")
include("atmos_les_core.jl")
include("atmos_les_default_perturbations.jl")
include("atmos_refstate_perturbations.jl")
include("atmos_turbulence_stats.jl")
include("atmos_mass_energy_loss.jl")
include("atmos_les_spectra.jl")
include("atmos_gcm_spectra.jl")
include("dump_init.jl")
include("dump_state.jl")
include("dump_aux.jl")
include("dump_tendencies.jl")
