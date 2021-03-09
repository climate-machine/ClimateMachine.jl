module ClimateMachine

using Pkg.TOML

const CLIMATEMACHINE_VERSION =
    VersionNumber(TOML.parsefile(joinpath(dirname(@__DIR__), "Project.toml"))["version"])

include(joinpath("Utilities", "TicToc", "TicToc.jl"))
include(joinpath("InputOutput", "ArtifactWrappers", "ArtifactWrappers.jl"))
include(joinpath("InputOutput", "Writers", "Writers.jl"))
include(joinpath("Driver", "ConfigTypes", "ConfigTypes.jl"))
include(joinpath("Utilities", "VariableTemplates", "VariableTemplates.jl"))
include(joinpath("Common", "Thermodynamics", "Thermodynamics.jl"))
include(joinpath("Atmos", "TemperatureProfiles", "TemperatureProfiles.jl"))
include(joinpath(
    "Atmos",
    "Parameterizations",
    "CloudPhysics",
    "Microphysics.jl",
))
include(joinpath(
    "Atmos",
    "Parameterizations",
    "CloudPhysics",
    "Microphysics_0M.jl",
))
include(joinpath("Common", "SurfaceFluxes", "SurfaceFluxes.jl"))
include(joinpath("Arrays", "MPIStateArrays.jl"))
include(joinpath("Numerics", "Mesh", "Mesh.jl"))
include(joinpath("Numerics", "DGMethods", "Courant.jl"))
include(joinpath("BalanceLaws", "Problems.jl"))
include(joinpath("BalanceLaws", "BalanceLaws.jl"))
include(joinpath("Numerics", "DGMethods", "DGMethods.jl"))
include(joinpath("Common", "Orientations", "Orientations.jl"))
include(joinpath("Utilities", "SingleStackUtils", "SingleStackUtils.jl"))
include(joinpath("Numerics", "SystemSolvers", "SystemSolvers.jl"))
include(joinpath("Numerics", "ODESolvers", "GenericCallbacks.jl"))
include(joinpath("Numerics", "ODESolvers", "ODESolvers.jl"))
include(joinpath("Land", "Model", "LandModel.jl"))
include(joinpath("InputOutput", "VTK", "VTK.jl"))
include(joinpath("Common", "TurbulenceClosures", "TurbulenceClosures.jl"))
include(joinpath("Common", "TurbulenceConvection", "TurbulenceConvection.jl"))
include(joinpath("Atmos", "Model", "AtmosModel.jl"))
include(joinpath("Common", "Spectra", "Spectra.jl"))
include(joinpath("Diagnostics", "Diagnostics.jl"))
include(joinpath("Diagnostics", "Debug", "StateCheck.jl"))
include(joinpath("Driver", "Checkpoint", "Checkpoint.jl"))
include(joinpath("Driver", "Callbacks", "Callbacks.jl"))
include(joinpath("Driver", "Driver.jl"))

include(joinpath("Ocean", "Ocean.jl"))

include(joinpath("Coupling", "Coupling.jl"))

end
