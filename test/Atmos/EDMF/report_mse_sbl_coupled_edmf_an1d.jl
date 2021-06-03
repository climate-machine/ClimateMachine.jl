using ClimateMachine
const clima_dir = dirname(dirname(pathof(ClimateMachine)));

if parse(Bool, get(ENV, "CLIMATEMACHINE_PLOT_EDMF_COMPARISON", "false"))
    plot_dir = joinpath(clima_dir, "output", "sbl_edmf", "pycles_comparison")
else
    plot_dir = nothing
end

include(joinpath(@__DIR__, "compute_mse.jl"))

data_file = Dataset(joinpath(PyCLES_output_dataset_path, "Gabls.nc"), "r")

#! format: off
best_mse = OrderedDict()
best_mse["prog_ρ"] = 9.3808142287632006e-03
best_mse["prog_ρu_1"] = 6.7427140307178524e+03
best_mse["prog_ρu_2"] = 7.2306841961112966e-01
best_mse["prog_turbconv_environment_ρatke"] = 2.9810806322235749e+02
best_mse["prog_turbconv_environment_ρaθ_liq_cv"] = 8.1270487249851797e+01
best_mse["prog_turbconv_updraft_1_ρa"] = 2.7223017351724246e+02
best_mse["prog_turbconv_updraft_1_ρaw"] = 5.5909371368198686e+02
best_mse["prog_turbconv_updraft_1_ρaθ_liq"] = 2.7933498788454813e+02
#! format: on

computed_mse = compute_mse(
    solver_config.dg.grid,
    solver_config.dg.balance_law,
    time_data,
    dons_arr,
    data_file,
    "Gabls",
    best_mse,
    1800,
    plot_dir,
)

@testset "SBL Coupled EDMF Solution Quality Assurance (QA) tests" begin
    #! format: off
    test_mse(computed_mse, best_mse, "prog_ρ")
    test_mse(computed_mse, best_mse, "prog_ρu_1")
    test_mse(computed_mse, best_mse, "prog_ρu_2")
    test_mse(computed_mse, best_mse, "prog_turbconv_environment_ρatke")
    test_mse(computed_mse, best_mse, "prog_turbconv_environment_ρaθ_liq_cv")
    test_mse(computed_mse, best_mse, "prog_turbconv_updraft_1_ρa")
    test_mse(computed_mse, best_mse, "prog_turbconv_updraft_1_ρaw")
    test_mse(computed_mse, best_mse, "prog_turbconv_updraft_1_ρaθ_liq")
    #! format: on
end
