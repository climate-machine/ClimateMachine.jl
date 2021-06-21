using Test, Pkg

@testset "Parameterizations" begin
    all_tests = isempty(ARGS) || "all" in ARGS ? true : false
    for submodule in ["Microphysics", "Activation"]
        if all_tests ||
           "$submodule" in ARGS ||
           "Parameterizations" in ARGS ||
           "Atmos" in ARGS
            include_test(submodule)
        end
    end

end
