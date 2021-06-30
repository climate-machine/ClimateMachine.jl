####
#### Defines list of tutorials given `GENERATED_DIR`
####

generate_tutorials = false
    # parse(Bool, get(ENV, "CLIMATEMACHINE_DOCS_GENERATE_TUTORIALS", "true"))

tutorials = []


# Allow flag to skip generated
# tutorials since this is by
# far the slowest part of the
# docs build.
if generate_tutorials

    # generate tutorials

    include("pages_helper.jl")

    tutorials = [
        "Home" => "TutorialList.jl",
        "BalanceLaws" => [
            "Tendency specification" =>
                "BalanceLaws/tendency_specification_layer.jl",
        ],
        "Atmos" => [
            "Dry Idealized GCM (Held-Suarez)" => "Atmos/heldsuarez.jl",
            "Single Element Stack Experiment (Burgers Equation)" =>
                "Atmos/burgers_single_stack.jl",
            "Finite Volume Single Element Stack Experiment (Burgers Equation)" =>
                "Atmos/burgers_single_stack_fvm.jl",
            "HEVI Single Element Stack Experiment (Burgers Equation)" =>
                "Atmos/burgers_single_stack_bjfnk.jl",
            "LES Experiment (Density Current)" => "Atmos/densitycurrent.jl",
            "LES Experiment (Rising Thermal Bubble)" =>
                "Atmos/risingbubble.jl",
            "Linear Hydrostatic Mountain (Topography)" =>
                "Atmos/agnesi_hs_lin.jl",
            "Linear Non-Hydrostatic Mountain (Topography)" =>
                "Atmos/agnesi_nh_lin.jl",
        ],
        "Land" => [
            "Heat" => ["Heat Equation" => "Land/Heat/heat_equation.jl"],
            "Soil" => [
                "Hydraulic Functions" =>
                    "Land/Soil/Water/hydraulic_functions.jl",
                "Soil Heat Equation" =>
                    "Land/Soil/Heat/bonan_heat_tutorial.jl",
                "Richards Equation" =>
                    "Land/Soil/Water/equilibrium_test.jl",
                "Coupled Water and Heat" =>
                    "Land/Soil/Coupled/equilibrium_test.jl",
                "Phase Change I" =>
                    "Land/Soil/PhaseChange/freezing_front.jl",
                "Phase Change II" =>
                    "Land/Soil/PhaseChange/phase_change_analytic_test.jl",
            ],
        ],
        "Ocean" => [
            "One-dimensional geostrophic adjustment" =>
                "Ocean/geostrophic_adjustment.jl",
            "Propagating mode-1 internal wave" => "Ocean/internal_wave.jl",
            "Shear instability" => "Ocean/shear_instability.jl",
        ],
        "Numerics" => [
            "System Solvers" => [
                "Conjugate Gradient" => "Numerics/SystemSolvers/cg.jl",
                "Batched Generalized Minimal Residual" =>
                    "Numerics/SystemSolvers/bgmres.jl",
            ],
            "DG Methods" =>
                ["Filters" => "Numerics/DGMethods/showcase_filters.jl"],
            "Time-Stepping" => [
                "Introduction" => "Numerics/TimeStepping/ts_intro.jl",
                "Explicit Runge-Kutta methods" => [
                    "Numerics/TimeStepping/explicit_lsrk.jl",
                    "Numerics/TimeStepping/tutorial_risingbubble_config.jl",
                ],
                "Implicit-Explicit (IMEX) Additive Runge-Kutta methods" =>
                    [
                        "Numerics/TimeStepping/imex_ark.jl",
                        "Numerics/TimeStepping/tutorial_acousticwave_config.jl",
                    ],
                "Multirate Runge-Kutta methods" => [
                    "Numerics/TimeStepping/multirate_rk.jl",
                    "Numerics/TimeStepping/tutorial_risingbubble_config.jl",
                ],
                "MIS methods" => [
                    "Numerics/TimeStepping/mis.jl",
                    "Numerics/TimeStepping/tutorial_acousticwave_config.jl",
                ],
            ],
        ],
        "Diagnostics" => [
            "Debug" => [
                "State Statistics Regression" =>
                    "Diagnostics/Debug/StateCheck.jl",
            ],
        ],
    ]

    # Prepend tutorials_dir
    tutorials_jl = flatten_to_array_of_strings(get_second(tutorials))
    if run_single_tutorial
        tutorials_jl =
            filter(x -> occursin(tutorial_string_match, x), tutorials_jl)
        tutorials = ["Single tutorial mode" => tutorials_jl[1]]
    end
    println("Building literate tutorials...")

    @everywhere function generate_tutorial(tutorials_dir, tutorial)
        rpath = relpath(dirname(tutorial), tutorials_dir)
        rpath = rpath == "." ? "" : rpath
        gen_dir = joinpath(GENERATED_DIR, rpath)
        mkpath(gen_dir)

        cd(gen_dir) do
            # change the Edit on GitHub link:
            path = relpath(clima_dir, pwd())
            content = """
            # ```@meta
            # EditURL = "https://github.com/CliMA/ClimateMachine.jl/$(path)"
            # ```
            """
            mdpre(str) = content * str
            input = abspath(tutorial)
            Literate.markdown(
                input;
                execute = true,
                documenter = false,
                preprocess = mdpre,
            )
        end
    end

    tutorials_dir = joinpath(@__DIR__, "..", "tutorials")
    tutorials_jl = map(x -> joinpath(tutorials_dir, x), tutorials_jl)
    pmap(t -> generate_tutorial(tutorials_dir, t), tutorials_jl)

    # update list of rendered markdown tutorial output for mkdocs
    ext_jl2md(x) = joinpath(basename(GENERATED_DIR), replace(x, ".jl" => ".md"))
    tutorials = transform_second(x -> ext_jl2md(x), tutorials)
end
