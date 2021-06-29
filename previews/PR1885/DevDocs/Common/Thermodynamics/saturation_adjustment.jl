using ClimateMachine
using ClimateMachine.Thermodynamics
using ClimateMachine.TemperatureProfiles
using UnPack
using CLIMAParameters
using RootSolvers
using CLIMAParameters.Planet
using Plots
import ClimateMachine.Thermodynamics
Thermodynamics.print_warning() = false
TD = Thermodynamics

struct EarthParameterSet <: AbstractEarthParameterSet end;
const param_set = EarthParameterSet();
FT = Float64;

const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(clima_dir, "test", "Common", "Thermodynamics", "profiles.jl"))
include(joinpath(clima_dir, "docs", "plothelpers.jl"));
profiles = PhaseEquilProfiles(param_set, Array{FT});
@unpack ρ, q_tot = profiles
T_true = profiles.T
prof_pts = (ρ, T_true, q_tot)

dims = (10, 10, 10);
ρ = range(min(ρ...), stop = max(ρ...), length = dims[1]);
T_true = range(min(T_true...), stop = max(T_true...), length = dims[2]);
q_tot = range(min(q_tot...), stop = max(q_tot...), length = dims[3]);

ρ_all = Array{FT}(undef, prod(dims));
T_true_all = Array{FT}(undef, prod(dims));
q_tot_all = Array{FT}(undef, prod(dims));

linear_indices = LinearIndices((1:dims[1], 1:dims[2], 1:dims[3]));

numerical_methods =
    (SecantMethod, NewtonsMethod, NewtonsMethodAD, RegulaFalsiMethod)

ts = Dict(
    NM => Array{Union{ThermodynamicState, Nothing}}(undef, prod(dims))
    for NM in numerical_methods
)
ts_no_err = Dict(
    NM => Array{ThermodynamicState}(undef, prod(dims))
    for NM in numerical_methods
)

@inbounds for i in linear_indices.indices[1]
    @inbounds for j in linear_indices.indices[2]
        @inbounds for k in linear_indices.indices[3]
            n = linear_indices[i, j, k]
            ρ_all[n] = ρ[i]
            T_true_all[n] = T_true[j]
            q_tot_all[n] = q_tot[k]
            e_int =
                internal_energy(param_set, T_true[j], PhasePartition(q_tot[k]))

            @inbounds for NM in numerical_methods
                Thermodynamics.error_on_non_convergence() = false
                ts_no_err[NM][n] = TD.PhaseEquil_dev_only(
                    param_set,
                    e_int,
                    ρ[i],
                    q_tot[k];
                    sat_adjust_method = NM,
                    maxiter = 10,
                )
                Thermodynamics.error_on_non_convergence() = true
                # @show n/prod(linear_indices.indices)*100
                try
                    ts[NM][n] = TD.PhaseEquil_dev_only(
                        param_set,
                        e_int,
                        ρ[i],
                        q_tot[k];
                        sat_adjust_method = NM,
                        maxiter = 10,
                    )
                catch
                    ts[NM][n] = nothing
                end
            end
        end
    end
end

# folder = "sat_adjust_analysis"
folder = @__DIR__
mkpath(folder)

let
    # Full 3D scatter plot
    function plot3D(ts_no_err, ts, NM; converged)
        mask = converged ? ts .≠ nothing : ts .== nothing

        c_name = converged ? "converged" : "non_converged"
        label = converged ? "converged" : "non-converged"
        casename = converged ? "converged" : "non-converged"
        nm_name = nameof(NM)
        filename = "3DSpace_$(c_name)_$nm_name.svg"

        ρ_mask = ρ_all[mask]
        q_tot_mask = q_tot_all[mask]
        T_mask = T_true_all[mask]
        pts = (ρ_mask, T_mask, q_tot_mask)
        Plots.plot(
            pts...,
            color = "blue",
            seriestype = :scatter,
            markersize = 7,
            label = casename,
        )
        Plots.plot!(
            prof_pts...,
            color = "red",
            seriestype = :scatter,
            markersize = 7,
            label = "tested thermo profiles",
        )
        plot!(
            xlabel = "Density",
            ylabel = "Temperature",
            zlabel = "Total specific humidity",
            title = "3D input to PhaseEquil",
            xlims = (min(ρ_all...), max(ρ_all...)),
            ylims = (min(T_true_all...), max(T_true_all...)),
            zlims = (min(q_tot_all...), max(q_tot_all...)),
            camera = (50, 50),
            # camera = (50,70),
        )
        savefig(joinpath(folder, filename))
    end

    # 2D binned scatter plots
    function plot2D_slices(ts_no_err, ts, NM; converged)
        mask = converged ? ts .≠ nothing : ts .== nothing
        ρ_mask = ρ_all[mask]
        T_mask = T_true_all[mask]
        q_tot_mask = q_tot_all[mask]
        c_name = converged ? "converged" : "non_converged"
        label = converged ? "converged" : "non-converged"
        short_name = converged ? "C" : "NC"
        nm_name = nameof(NM)
        filename = "2DSlice_$(c_name)_$nm_name.svg"
        filename = joinpath(folder, filename)
        save_binned_surface_plots(
            ρ_mask,
            T_mask,
            q_tot_mask,
            short_name,
            filename;
            xlims = (min(ρ_all...), max(ρ_all...)),
            ylims = (min(T_true_all...), max(T_true_all...)),
            label = label,
            ref_points = prof_pts,
        )
    end

    for NM in numerical_methods
        plot3D(ts_no_err[NM], ts[NM], NM; converged = false)
        plot3D(ts_no_err[NM], ts[NM], NM; converged = true)
        plot2D_slices(ts_no_err[NM], ts[NM], NM; converged = true)
        plot2D_slices(ts_no_err[NM], ts[NM], NM; converged = false)
    end

    convergence_percent = Dict()
    for NM in numerical_methods
        convergence_percent[NM] = count(ts[NM] .≠ nothing) / length(ts[NM])
    end
    println("Convergence percentages:")
    for (k, v) in convergence_percent
        println("$k = $v")
    end
end
