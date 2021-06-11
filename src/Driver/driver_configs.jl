# ClimateMachine driver configurations
#
# Contains helper functions to establish simulation configurations to be
# used with the ClimateMachine driver. Currently:
# - AtmosLESConfiguration
# - AtmosGCMConfiguration
# - OceanBoxGCMConfiguration
# - SingleStackConfiguration
# - MultiColumnLandModel
#
# User-customized configurations can use these as templates.

using CLIMAParameters
using CLIMAParameters.Planet: planet_radius

abstract type ConfigSpecificInfo end
struct AtmosLESSpecificInfo <: ConfigSpecificInfo end
struct AtmosGCMSpecificInfo{FT} <: ConfigSpecificInfo
    domain_height::FT
    nelem_vert::Int
    nelem_horz::Int
    vert_range::Array{FT, 1}
end
struct OceanBoxGCMSpecificInfo <: ConfigSpecificInfo end
struct SingleStackSpecificInfo{FT} <: ConfigSpecificInfo
    zmax::FT
    hmax::FT
end
struct MultiColumnLandSpecificInfo <: ConfigSpecificInfo end

include("SolverTypes/SolverTypes.jl")

"""
    ArgParse.parse_item

Parses custom command line option for tuples of three or fewer integers.
"""
function ArgParse.parse_item(::Type{NTuple{2, Int}}, s::AbstractString)

    str_array = split(s, ",")
    int_1 = length(str_array) > 0 ? parse(Int, str_array[1]) : 0
    int_2 = length(str_array) > 1 ? parse(Int, str_array[2]) : 0

    return (int_1, int_2)
end

"""
    ArgParse.parse_item

Parses custom command line option for tuples of three or fewer integers.
"""
function ArgParse.parse_item(::Type{NTuple{3, Int}}, s::AbstractString)

    str_array = split(s, ",")
    int_1 = length(str_array) > 0 ? parse(Int, str_array[1]) : 0
    int_2 = length(str_array) > 1 ? parse(Int, str_array[2]) : 0
    int_3 = length(str_array) > 2 ? parse(Int, str_array[3]) : 0

    return (int_1, int_2, int_3)
end

"""
    ArgParse.parse_item

Parses custom command line option for tuples of three integers.
"""
function ArgParse.parse_item(::Type{NTuple{3, Float64}}, s::AbstractString)

    str_array = split(s, ",")
    ft_1 = length(str_array) > 0 ? parse(Float64, str_array[1]) : 0
    ft_2 = length(str_array) > 1 ? parse(Float64, str_array[2]) : 0
    ft_3 = length(str_array) > 2 ? parse(Float64, str_array[3]) : 0

    return (ft_1, ft_2, ft_3)
end

"""
    ArgParse.parse_item

Parses custom command line option for `DateTime`s.
"""
function ArgParse.parse_item(::Type{DateTime}, s::AbstractString)
    DateTime(s, "yyyy-mm-ddTHH:MM:SS")
end

"""
    get_polyorders

Utility function that gets the polynomial orders for the given configuration
either passed from command line or as default values
"""
function get_polyorders(N, setting = ClimateMachine.Settings.degree)

    # Check if polynomial degree was passed as a CL option
    if setting != (-1, -1)
        setting
    elseif N isa Int
        (N, N)
    else
        N
    end
end

"""
    ClimateMachine.DriverConfiguration

Collects all parameters necessary to set up a ClimateMachine simulation.
"""
struct DriverConfiguration{FT}
    config_type::ClimateMachineConfigType

    name::String
    # Polynomial order tuple (polyorder_horz, polyorder_vert)
    polyorders::NTuple{2, Int}
    array_type::Any
    #
    # Model details
    param_set::AbstractParameterSet
    bl::BalanceLaw
    #
    # Execution details
    mpicomm::MPI.Comm
    #
    # Mesh details
    grid::DiscontinuousSpectralElementGrid
    #
    # DGModel details
    numerical_flux_first_order::NumericalFluxFirstOrder
    numerical_flux_second_order::NumericalFluxSecondOrder
    numerical_flux_gradient::NumericalFluxGradient
    # DGFVModel details, used when polyorder_vert = 0
    fv_reconstruction::Union{Nothing, AbstractReconstruction}
    # Cutoff filter to emulate overintegration
    filter::Any
    #
    # Configuration-specific info
    config_info::ConfigSpecificInfo

    function DriverConfiguration(
        config_type,
        name::String,
        polyorders::NTuple{2, Int},
        FT,
        array_type,
        param_set::AbstractParameterSet,
        bl::BalanceLaw,
        mpicomm::MPI.Comm,
        grid::DiscontinuousSpectralElementGrid,
        numerical_flux_first_order::NumericalFluxFirstOrder,
        numerical_flux_second_order::NumericalFluxSecondOrder,
        numerical_flux_gradient::NumericalFluxGradient,
        fv_reconstruction::Union{Nothing, AbstractReconstruction},
        filter,
        config_info::ConfigSpecificInfo,
    )
        return new{FT}(
            config_type,
            name,
            polyorders,
            array_type,
            param_set,
            bl,
            mpicomm,
            grid,
            numerical_flux_first_order,
            numerical_flux_second_order,
            numerical_flux_gradient,
            fv_reconstruction,
            filter,
            config_info,
        )
    end
end

function print_model_info(model, mpicomm)
    mpirank = MPI.Comm_rank(mpicomm)
    if mpirank == 0
        @show ClimateMachine.array_type()
        msg = "Model composition\n"
        for key in fieldnames(typeof(model))
            msg =
                msg * @sprintf(
                    "    %s = %s\n",
                    string(key),
                    string((getproperty(model, key)))
                )
        end
        @info msg
        show_tendencies(model; table_complete = model isa AtmosModel)
        model isa AtmosModel && show_bcs(model)
    end
end

function AtmosLESConfiguration(
    name::String,
    N::Union{Int, NTuple{2, Int}},
    (Δx, Δy, Δz)::NTuple{3, FT},
    xmax::FT,
    ymax::FT,
    zmax::FT,
    param_set::AbstractParameterSet,
    init_LES!;
    xmin = zero(FT),
    ymin = zero(FT),
    zmin = zero(FT),
    array_type = ClimateMachine.array_type(),
    physics = AtmosPhysics{FT}(param_set;),
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        physics;
        init_state_prognostic = init_LES!,
    ),
    mpicomm = MPI.COMM_WORLD,
    boundary = ((0, 0), (0, 0), (1, 2)),
    periodicity = (true, true, false),
    meshwarp = (x...) -> identity(x),
    grid_stretching = (nothing, nothing, nothing),
    numerical_flux_first_order = RusanovNumericalFlux(),
    numerical_flux_second_order = CentralNumericalFluxSecondOrder(),
    numerical_flux_gradient = CentralNumericalFluxGradient(),
    fv_reconstruction = nothing,
    Ncutoff = get_polyorders(N),
) where {FT <: AbstractFloat}

    (polyorder_horz, polyorder_vert) = get_polyorders(N)
    (cutofforder_horz, cutofforder_vert) =
        get_polyorders(Ncutoff, ClimateMachine.Settings.cutoff_degree)

    # Check if the element resolution was passed as a CL option
    if ClimateMachine.Settings.resolution != (-1, -1, -1)
        (Δx, Δy, Δz) = ClimateMachine.Settings.resolution
    end

    # Check if the min of the domain was passed as a CL option
    if ClimateMachine.Settings.domain_min != (-1, -1, -1)
        (xmin, ymin, zmin) = ClimateMachine.Settings.domain_min
    end

    # Check if the max of the domain was passed as a CL option
    if ClimateMachine.Settings.domain_max != (-1, -1, -1)
        (xmax, ymax, zmax) = ClimateMachine.Settings.domain_max
    end

    print_model_info(model, mpicomm)

    brickrange = (
        grid1d(
            xmin,
            xmax,
            grid_stretching[1],
            elemsize = Δx * max(polyorder_horz, 1),
        ),
        grid1d(
            ymin,
            ymax,
            grid_stretching[2],
            elemsize = Δy * max(polyorder_horz, 1),
        ),
        grid1d(
            zmin,
            zmax,
            grid_stretching[3],
            elemsize = Δz * max(polyorder_vert, 1),
        ),
    )
    topology = StackedBrickTopology(
        mpicomm,
        brickrange,
        periodicity = periodicity,
        boundary = boundary,
    )

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = array_type,
        polynomialorder = (polyorder_horz, polyorder_vert),
        meshwarp = meshwarp,
    )

    if cutofforder_horz < polyorder_horz && cutofforder_vert < polyorder_vert
        filter = CutoffFilter(
            grid,
            (cutofforder_horz + 1, cutofforder_horz + 1, cutofforder_vert + 1),
        )
    else
        filter = nothing
        cutofforder_horz = cutofforder_vert = nothing
    end


    @info @sprintf(
        """
Establishing Atmos LES configuration for %s
    precision               = %s
    horiz polynomial order  = %d
    vert polynomial order   = %d
    horiz cutoff order      = %s
    vert cutoff order       = %s
    domain_min              = %.2f m, %.2f m, %.2f m
    domain_max              = %.2f m, %.2f m, %.2f m
    resolution              = %dx%dx%d
    MPI ranks               = %d
    min(Δ_horz)             = %.2f m
    min(Δ_vert)             = %.2f m""",
        name,
        FT,
        polyorder_horz,
        polyorder_vert,
        cutofforder_horz,
        cutofforder_vert,
        xmin,
        ymin,
        zmin,
        xmax,
        ymax,
        zmax,
        Δx,
        Δy,
        Δz,
        MPI.Comm_size(mpicomm),
        min_node_distance(grid, HorizontalDirection()),
        min_node_distance(grid, VerticalDirection())
    )

    return DriverConfiguration(
        AtmosLESConfigType(),
        name,
        (polyorder_horz, polyorder_vert),
        FT,
        array_type,
        param_set,
        model,
        mpicomm,
        grid,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
        fv_reconstruction,
        filter,
        AtmosLESSpecificInfo(),
    )
end

function AtmosGCMConfiguration(
    name::String,
    N::Union{Int, NTuple{2, Int}},
    (nelem_horz, nelem_vert)::NTuple{2, Int},
    domain_height::FT,
    param_set::AbstractParameterSet,
    init_GCM!;
    array_type = ClimateMachine.array_type(),
    physics = AtmosPhysics{FT}(param_set),
    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        physics;
        init_state_prognostic = init_GCM!,
    ),
    mpicomm = MPI.COMM_WORLD,
    grid_stretching = (nothing,),
    meshwarp::Function = equiangular_cubed_sphere_warp,
    numerical_flux_first_order = RusanovNumericalFlux(),
    numerical_flux_second_order = CentralNumericalFluxSecondOrder(),
    numerical_flux_gradient = CentralNumericalFluxGradient(),
    fv_reconstruction = nothing,
    Ncutoff = get_polyorders(N),
) where {FT <: AbstractFloat}

    (polyorder_horz, polyorder_vert) = get_polyorders(N)
    (cutofforder_horz, cutofforder_vert) =
        get_polyorders(Ncutoff, ClimateMachine.Settings.cutoff_degree)

    # Check if the number of elements was passed as a CL option
    if ClimateMachine.Settings.nelems != (-1, -1, -1)
        (nelem_horz, nelem_vert) = ClimateMachine.Settings.nelems
    end

    # Check if the domain height was passed as a CL option
    if ClimateMachine.Settings.domain_height != -1
        domain_height = ClimateMachine.Settings.domain_height
    end

    print_model_info(model, mpicomm)

    _planet_radius::FT = planet_radius(param_set)
    vert_range = grid1d(
        _planet_radius,
        FT(_planet_radius + domain_height),
        grid_stretching[1],
        nelem = nelem_vert,
    )

    topology = StackedCubedSphereTopology(
        mpicomm,
        nelem_horz,
        vert_range;
        boundary = (1, 2),
    )

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = array_type,
        polynomialorder = (polyorder_horz, polyorder_vert),
        meshwarp = meshwarp,
    )


    if cutofforder_horz < polyorder_horz && cutofforder_vert < polyorder_vert
        filter = CutoffFilter(
            grid,
            (cutofforder_horz + 1, cutofforder_horz + 1, cutofforder_vert + 1),
        )
    else
        filter = nothing
        cutofforder_horz = cutofforder_vert = nothing
    end

    @info @sprintf(
        """
Establishing Atmos GCM configuration for %s
    precision               = %s
    horiz polynomial order  = %d
    vert polynomial order   = %d
    horiz cutoff order      = %s
    vert cutoff order       = %s
    # horiz elem            = %d
    # vert elems            = %d
    domain height           = %.2e m
    MPI ranks               = %d
    min(Δ_horz)             = %.2f m
    min(Δ_vert)             = %.2f m""",
        name,
        FT,
        polyorder_horz,
        polyorder_vert,
        cutofforder_horz,
        cutofforder_vert,
        nelem_horz,
        nelem_vert,
        domain_height,
        MPI.Comm_size(mpicomm),
        min_node_distance(grid, HorizontalDirection()),
        min_node_distance(grid, VerticalDirection())
    )

    return DriverConfiguration(
        AtmosGCMConfigType(),
        name,
        (polyorder_horz, polyorder_vert),
        FT,
        array_type,
        param_set,
        model,
        mpicomm,
        grid,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
        fv_reconstruction,
        filter,
        AtmosGCMSpecificInfo(
            domain_height,
            nelem_vert,
            nelem_horz,
            collect(vert_range),
        ),
    )
end

function OceanBoxGCMConfiguration(
    name::String,
    N::Union{Int, NTuple{2, Int}},
    (Nˣ, Nʸ, Nᶻ)::NTuple{3, Int},
    param_set::AbstractParameterSet,
    model;
    FT = Float64,
    array_type = ClimateMachine.array_type(),
    mpicomm = MPI.COMM_WORLD,
    numerical_flux_first_order = RusanovNumericalFlux(),
    numerical_flux_second_order = CentralNumericalFluxSecondOrder(),
    numerical_flux_gradient = CentralNumericalFluxGradient(),
    fv_reconstruction = nothing,
    periodicity = (false, false, false),
    boundary = ((1, 1), (1, 1), (2, 3)),
)

    (polyorder_horz, polyorder_vert) = get_polyorders(N)

    # Check if the number of elements was passed as a CL option
    if ClimateMachine.Settings.nelems != (-1, -1, -1)
        (Nˣ, Nʸ, Nᶻ) = ClimateMachine.Settings.nelems
    end

    brickrange = (
        range(FT(0); length = Nˣ + 1, stop = model.problem.Lˣ),
        range(FT(0); length = Nʸ + 1, stop = model.problem.Lʸ),
        range(FT(-model.problem.H); length = Nᶻ + 1, stop = 0),
    )

    topology = StackedBrickTopology(
        mpicomm,
        brickrange;
        periodicity = periodicity,
        boundary = boundary,
    )

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = array_type,
        polynomialorder = (polyorder_horz, polyorder_vert),
    )

    @info @sprintf(
        """
Establishing Ocean Box GCM configuration for %s
    precision               = %s
    horiz polynomial order  = %d
    vert polynomial order   = %d
    Nˣ # elems              = %d
    Nʸ # elems              = %d
    Nᶻ # elems              = %d
    domain depth            = %.2e m
    MPI ranks               = %d""",
        name,
        FT,
        polyorder_horz,
        polyorder_vert,
        Nˣ,
        Nʸ,
        Nᶻ,
        -model.problem.H,
        MPI.Comm_size(mpicomm)
    )

    return DriverConfiguration(
        OceanBoxGCMConfigType(),
        name,
        (polyorder_horz, polyorder_vert),
        FT,
        array_type,
        param_set,
        model,
        mpicomm,
        grid,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
        fv_reconstruction,
        nothing, # filter
        OceanBoxGCMSpecificInfo(),
    )
end

function SingleStackConfiguration(
    name::String,
    N::Union{Int, NTuple{2, Int}},
    nelem_vert::Int,
    zmax::FT,
    param_set::AbstractParameterSet,
    model::BalanceLaw;
    zmin = zero(FT),
    hmax = one(FT),
    array_type = ClimateMachine.array_type(),
    mpicomm = MPI.COMM_WORLD,
    boundary = ((0, 0), (0, 0), (1, 2)),
    periodicity = (true, true, false),
    meshwarp = (x...) -> identity(x),
    numerical_flux_first_order = RusanovNumericalFlux(),
    numerical_flux_second_order = CentralNumericalFluxSecondOrder(),
    numerical_flux_gradient = CentralNumericalFluxGradient(),
    fv_reconstruction = nothing,
) where {FT <: AbstractFloat}

    (polyorder_horz, polyorder_vert) = get_polyorders(N)

    # Check if the number of elements was passed as a CL option
    if ClimateMachine.Settings.nelems != (-1, -1, -1)
        nE = ClimateMachine.Settings.nelems
        nelem_vert = nE[1]
    end

    # Check if the domain height was passed as a CL option
    if ClimateMachine.Settings.domain_height != -1
        zmax = ClimateMachine.Settings.domain_height
    end

    print_model_info(model, mpicomm)

    xmin, xmax = zero(FT), hmax
    ymin, ymax = zero(FT), hmax
    brickrange = (
        grid1d(xmin, xmax, nelem = 1),
        grid1d(ymin, ymax, nelem = 1),
        grid1d(zmin, zmax, nelem = nelem_vert),
    )
    topology = StackedBrickTopology(
        mpicomm,
        brickrange,
        periodicity = periodicity,
        boundary = boundary,
    )

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = array_type,
        polynomialorder = (polyorder_horz, polyorder_vert),
        meshwarp = meshwarp,
    )

    @info @sprintf(
        """
Establishing single stack configuration for %s
    precision               = %s
    horiz polynomial order  = %d
    vert polynomial order   = %d
    domain_min              = %.2f m, %.2f m, %.2f m
    domain_max              = %.2f m, %.2f m, %.2f m
    # vert elems            = %d
    MPI ranks               = %d
    min(Δ_horz)             = %.2f m
    min(Δ_vert)             = %.2f m""",
        name,
        FT,
        polyorder_horz,
        polyorder_vert,
        xmin,
        ymin,
        zmin,
        xmax,
        ymax,
        zmax,
        nelem_vert,
        MPI.Comm_size(mpicomm),
        min_node_distance(grid, HorizontalDirection()),
        min_node_distance(grid, VerticalDirection())
    )

    return DriverConfiguration(
        SingleStackConfigType(),
        name,
        (polyorder_horz, polyorder_vert),
        FT,
        array_type,
        param_set,
        model,
        mpicomm,
        grid,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
        fv_reconstruction,
        nothing, # filter
        SingleStackSpecificInfo(zmax, hmax),
    )
end

function MultiColumnLandModel(
    name::String,
    N::Union{Int, NTuple{2, Int}},
    (Δx, Δy, Δz)::NTuple{3, FT},
    xmax::FT,
    ymax::FT,
    zmax::FT,
    param_set::AbstractParameterSet,
    model::BalanceLaw;
    xmin = zero(FT),
    ymin = zero(FT),
    zmin = zero(FT),
    array_type = ClimateMachine.array_type(),
    mpicomm = MPI.COMM_WORLD,
    boundary = ((3, 4), (5, 6), (1, 2)),
    solver_type = ExplicitSolverType(),
    periodicity = (false, false, false),
    meshwarp = (x...) -> identity(x),
    numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
    numerical_flux_second_order = CentralNumericalFluxSecondOrder(),
    numerical_flux_gradient = CentralNumericalFluxGradient(),
    fv_reconstruction = nothing,
) where {FT <: AbstractFloat}

    (polyorder_horz, polyorder_vert) = get_polyorders(N)

    # Check if the element resolution was passed as a CL option
    if ClimateMachine.Settings.resolution != (-1, -1, -1)
        (Δx, Δy, Δz) = ClimateMachine.Settings.resolution
    end

    # Check if the min of the domain was passed as a CL option
    if ClimateMachine.Settings.domain_min != (-1, -1, -1)
        (xmin, ymin, zmin) = ClimateMachine.Settings.domain_min
    end

    # Check if the max of the domain was passed as a CL option
    if ClimateMachine.Settings.domain_max != (-1, -1, -1)
        (xmax, ymax, zmax) = ClimateMachine.Settings.domain_max
    end

    print_model_info(model, mpicomm)

    brickrange = (
        grid1d(xmin, xmax, elemsize = Δx * max(polyorder_horz, 1)),
        grid1d(ymin, ymax, elemsize = Δy * max(polyorder_horz, 1)),
        grid1d(zmin, zmax, elemsize = Δz * max(polyorder_vert, 1)),
    )

    topology = StackedBrickTopology(
        mpicomm,
        brickrange,
        periodicity = periodicity,
        boundary = boundary,
    )

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = array_type,
        polynomialorder = (polyorder_horz, polyorder_vert),
        meshwarp = meshwarp,
    )

    @info @sprintf(
        """
Establishing MultiColumnLandModel configuration for %s
    precision        = %s
    vert polyn order = %d
    horz polyn order = %d
    domain_min       = %.2f m, %.2f m, %.2f m
    domain_max       = %.2f m, %.2f m, %.2f m
    resolution       = %dx%dx%d
    MPI ranks        = %d
    min(Δ_horz)      = %.2f m
    min(Δ_vert)      = %.2f m""",
        name,
        FT,
        polyorder_vert,
        polyorder_horz,
        xmin,
        ymin,
        zmin,
        xmax,
        ymax,
        zmax,
        Δx,
        Δy,
        Δz,
        MPI.Comm_size(mpicomm),
        min_node_distance(grid, HorizontalDirection()),
        min_node_distance(grid, VerticalDirection())
    )

    return DriverConfiguration(
        MultiColumnLandConfigType(),
        name,
        (polyorder_horz, polyorder_vert),
        FT,
        array_type,
        param_set,
        model,
        mpicomm,
        grid,
        numerical_flux_first_order,
        numerical_flux_second_order,
        numerical_flux_gradient,
        fv_reconstruction,
        nothing, # filter
        MultiColumnLandSpecificInfo(),
    )
end

import ..DGMethods: DGModel

"""
    DGModel(driver_config; kwargs...)

Initialize a [`DGModel`](@ref) given a
[`DriverConfiguration`](@ref) and keyword
arguments supported by [`DGModel`](@ref).
"""
DGModel(driver_config; kwargs...) = DGModel(
    driver_config.bl,
    driver_config.grid,
    driver_config.numerical_flux_first_order,
    driver_config.numerical_flux_second_order,
    driver_config.numerical_flux_gradient;
    gradient_filter = driver_config.filter,
    tendency_filter = driver_config.filter,
    kwargs...,
)

"""
    SpaceDiscretization(driver_config; kwargs...)

Initialize a [`SpaceDiscretization`](@ref) given a
[`DriverConfiguration`](@ref) and keyword
arguments supported by [`SpaceDiscretization`](@ref).
"""
SpaceDiscretization(driver_config; kwargs...) =
    (driver_config.polyorders[2] == 0) ?
    DGFVModel(
        driver_config.bl,
        driver_config.grid,
        driver_config.fv_reconstruction,
        driver_config.numerical_flux_first_order,
        driver_config.numerical_flux_second_order,
        driver_config.numerical_flux_gradient;
        gradient_filter = driver_config.filter,
        tendency_filter = driver_config.filter,
        kwargs...,
    ) :
    DGModel(
        driver_config.bl,
        driver_config.grid,
        driver_config.numerical_flux_first_order,
        driver_config.numerical_flux_second_order,
        driver_config.numerical_flux_gradient;
        gradient_filter = driver_config.filter,
        tendency_filter = driver_config.filter,
        kwargs...,
    )
