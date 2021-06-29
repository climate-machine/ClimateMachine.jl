using ClimateMachine
const clima_dir = dirname(dirname(pathof(ClimateMachine)));
include(joinpath(clima_dir, "tutorials", "Numerics", "DGMethods", "Box1D.jl"))

const FT = Float64

output_dir = @__DIR__;
mkpath(output_dir);

run_box1D(
    4,
    FT(0.0),
    FT(1.0),
    FT(1.0),
    joinpath(output_dir, "box_1D_4_no_filter.svg"),
)

run_box1D(
    4,
    FT(0.0),
    FT(1.0),
    FT(1.0),
    joinpath(output_dir, "box_1D_4_no_filter_upwind.svg"),
    numerical_flux_first_order = RusanovNumericalFlux(),
)

run_box1D(
    4,
    FT(0.0),
    FT(1.0),
    FT(1.0),
    joinpath(output_dir, "box_1D_4_tmar.svg");
    tmar_filter = true,
)

run_box1D(
    4,
    FT(0.0),
    FT(1.0),
    FT(1.0),
    joinpath(output_dir, "box_1D_4_tmar_upwind.svg");
    tmar_filter = true,
    numerical_flux_first_order = RusanovNumericalFlux(),
)

run_box1D(
    4,
    FT(0.0),
    FT(1.0),
    FT(1.0),
    joinpath(output_dir, "box_1D_4_cutoff_1.svg");
    cutoff_filter = true,
    cutoff_param = 1,
)

run_box1D(
    4,
    FT(0.0),
    FT(1.0),
    FT(1.0),
    joinpath(output_dir, "box_1D_4_cutoff_3.svg");
    cutoff_filter = true,
    cutoff_param = 3,
)

run_box1D(
    4,
    FT(0.0),
    FT(1.0),
    FT(1.0),
    joinpath(output_dir, "box_1D_4_exp_1_4.svg");
    exp_filter = true,
    exp_param_1 = 1,
    exp_param_2 = 4,
)

run_box1D(
    4,
    FT(0.0),
    FT(1.0),
    FT(1.0),
    joinpath(output_dir, "box_1D_4_exp_1_8.svg");
    exp_filter = true,
    exp_param_1 = 1,
    exp_param_2 = 8,
)

run_box1D(
    4,
    FT(0.0),
    FT(1.0),
    FT(1.0),
    joinpath(output_dir, "box_1D_4_exp_1_32.svg");
    exp_filter = true,
    exp_param_1 = 1,
    exp_param_2 = 32,
)

run_box1D(
    4,
    FT(0.0),
    FT(1.0),
    FT(1.0),
    joinpath(output_dir, "box_1D_4_boyd_1_4.svg");
    boyd_filter = true,
    boyd_param_1 = 1,
    boyd_param_2 = 4,
)

run_box1D(
    4,
    FT(0.0),
    FT(1.0),
    FT(1.0),
    joinpath(output_dir, "box_1D_4_boyd_1_8.svg");
    boyd_filter = true,
    boyd_param_1 = 1,
    boyd_param_2 = 8,
)

run_box1D(
    4,
    FT(0.0),
    FT(1.0),
    FT(1.0),
    joinpath(output_dir, "box_1D_4_boyd_1_32.svg");
    boyd_filter = true,
    boyd_param_1 = 1,
    boyd_param_2 = 32,
)

run_box1D(
    4,
    FT(0.0),
    FT(1.0),
    FT(1.0),
    joinpath(output_dir, "box_1D_4_tmar_exp_1_8.svg");
    exp_filter = true,
    tmar_filter = true,
    exp_param_1 = 1,
    exp_param_2 = 8,
)

run_box1D(
    4,
    FT(0.0),
    FT(1.0),
    FT(1.0),
    joinpath(output_dir, "box_1D_4_tmar_boyd_1_8.svg");
    boyd_filter = true,
    tmar_filter = true,
    boyd_param_1 = 1,
    boyd_param_2 = 8,
)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

