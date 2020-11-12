####
#### Defines list of tutorials given `GENERATED_DIR`
####

generate_tutorials =
    parse(Bool, get(ENV, "CLIMATEMACHINE_DOCS_GENERATE_TUTORIALS", "true"))

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
        "Balance Law" => "how_to_make_a_balance_law.jl",
        "Atmos" => [
            "Dry Idealized GCM (Held-Suarez)" => "Atmos/heldsuarez.jl",
            "Single Element Stack Experiment (Burgers Equation)" =>
                "Atmos/burgers_single_stack.jl",
            "LES Experiment (Density Current)" => "Atmos/densitycurrent.jl",
            "LES Experiment (Rising Thermal Bubble)" =>
                "Atmos/risingbubble.jl",
            "Linear Hydrostatic Mountain (Topography)" =>
                "Atmos/agnesi_hs_lin.jl",
            "Linear Non-Hydrostatic Mountain (Topography)" =>
                "Atmos/agnesi_nh_lin.jl",
        ],
        "Ocean" => [],
        "Land" => [
            "Heat" => ["Heat Equation" => "Land/Heat/heat_equation.jl",
                "Heat Equation - Compare steppers" =>
                    "Land/Heat/compare_steppers.jl"],
            "Soil" => [
                "Hydraulic Functions" =>
                    "Land/Soil/Water/hydraulic_functions.jl",
                "Soil Heat Equation" =>
                    "Land/Soil/Heat/bonan_heat_tutorial.jl",
                "Richards Equation" =>
                    "Land/Soil/Water/equilibrium_test.jl",
                "Coupled Water and Heat" =>
                    "Land/Soil/Coupled/equilibrium_test.jl",
            ],
        ],
        "Numerics" => [
            "System Solvers" => [
                "Conjugate Gradient" => "Numerics/SystemSolvers/cg.jl",
                "Batched Generalized Minimal Residual" =>
                    "Numerics/SystemSolvers/bgmres.jl",
            ],
            "DG Methods" =>
                ["Filters" => "Numerics/DGMethods/showcase_filters.jl"],
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
    println("Building literate tutorials...")

    @everywhere function generate_tutorial(tutorials_dir, tutorial)
        gen_dir =
            joinpath(GENERATED_DIR, relpath(dirname(tutorial), tutorials_dir))
        input = abspath(tutorial)
        script = Literate.script(input, gen_dir)
        code = strip(read(script, String))
        mdpost(str) = replace(str, "@__CODE__" => code)
        Literate.markdown(input, gen_dir, postprocess = mdpost)
        Literate.notebook(input, gen_dir, execute = true)
    end

    tutorials_dir = joinpath(@__DIR__, "..", "tutorials")
    tutorials_jl = map(x -> joinpath(tutorials_dir, x), tutorials_jl)
    pmap(t -> generate_tutorial(tutorials_dir, t), tutorials_jl)

    # update list of rendered markdown tutorial output for mkdocs
    ext_jl2md(x) = joinpath(basename(GENERATED_DIR), replace(x, ".jl" => ".md"))
    tutorials = transform_second(x -> ext_jl2md(x), tutorials)
end
