####
#### Defines list of Application Programming Interface (APIs)
####

apis = [
    "Home" => "APIs/index.md",
    "Driver" => [
        "Top level interface" => "APIs/Driver/index.md",
        "Checkpoint" => "APIs/Driver/Checkpoint.md",
    ],
    "Atmos" => [
        "AtmosModel" => "APIs/Atmos/AtmosModel.md",
        "Microphysics_0M" => "APIs/Atmos/Microphysics_0M.md",
        "Microphysics" => "APIs/Atmos/Microphysics.md",
        "Temperature Profiles" => "APIs/Atmos/TemperatureProfiles.md",
    ],
    "Ocean" => "APIs/Ocean/Ocean.md",
    "Land" => [
        "Land Model" => "APIs/Land/LandModel.md",
        "Runoff Parameterizations" => "APIs/Land/Runoff.md",
        "Soil Water Parameterizations" =>
            "APIs/Land/SoilWaterParameterizations.md",
        "Soil Heat Parameterizations" =>
            "APIs/Land/SoilHeatParameterizations.md",
    ],
    "Coupling" => "APIs/Coupling/Coupling.md",
    "Common" => [
        "Orientations" => "APIs/Common/Orientations.md",
        "Spectra" => "APIs/Common/Spectra.md",
        "Surface Fluxes" => "APIs/Common/SurfaceFluxes.md",
        "Thermodynamics" => "APIs/Common/Thermodynamics.md",
        "Turbulence Closures" => "APIs/Common/TurbulenceClosures.md",
        "Turbulence Convection" => "APIs/Common/TurbulenceConvection.md",
    ],
    "Balance Laws" => [
        "Balance Laws" => "APIs/BalanceLaws/BalanceLaws.md",
        "Problems" => "APIs/BalanceLaws/Problems.md",
    ],
    "Arrays" => "APIs/Arrays/Arrays.md",
    "Diagnostics" => [
        "Diagnostics groups" => "APIs/Diagnostics/Diagnostics.md",
        "State Check" => "APIs/Diagnostics/StateCheck.md",
    ],
    "Input/Output" => ["Input/Output" => "APIs/InputOutput/index.md"],
    "Numerics" => [
        "Meshes" => "APIs/Numerics/Meshes/Mesh.md",
        "SystemSolvers" => "APIs/Numerics/SystemSolvers/SystemSolvers.md",
        "ODESolvers" => "APIs/Numerics/ODESolvers/ODESolvers.md",
        "DG Methods" => "APIs/Numerics/DGMethods/DGMethods.md",
        "Courant" => "APIs/Numerics/DGMethods/Courant.md",
        "Numerical Fluxes" => "APIs/Numerics/DGMethods/NumericalFluxes.md",
        "Finite Volume Reconstructions" =>
            "APIs/Numerics/DGMethods/FVReconstructions.md",
    ],
    "Utilities" => [
        "Variable Templates" => "APIs/Utilities/VariableTemplates.md",
        "Single Stack Utilities" => "APIs/Utilities/SingleStackUtils.md",
        "Tic Toc" => "APIs/Utilities/TicToc.md",
    ],
]
