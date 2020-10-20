abstract type OceanBoundaryCondition end

"""
    Defining dummy structs to dispatch on for boundary conditions.
"""
struct CoastlineFreeSlip <: OceanBoundaryCondition end
struct CoastlineNoSlip <: OceanBoundaryCondition end
struct OceanFloorFreeSlip <: OceanBoundaryCondition end
struct OceanFloorNoSlip <: OceanBoundaryCondition end
struct OceanSurfaceNoStressNoForcing <: OceanBoundaryCondition end
struct OceanSurfaceStressNoForcing <: OceanBoundaryCondition end
struct OceanSurfaceNoStressForcing <: OceanBoundaryCondition end
struct OceanSurfaceStressForcing <: OceanBoundaryCondition end

"""
    CoastlineFreeSlip

applies boundary condition ∇u = 0 and ∇θ = 0
"""

"""
    ocean_boundary_state!(::AbstractOceanModel, ::CoastlineFreeSlip, ::NumericalFluxFirstOrder)

apply free slip boundary conditions for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::NumericalFluxFirstOrder,
    ::CoastlineFreeSlip,
    ::AbstractOceanModel,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    u⁻ = Q⁻.u
    n = @SVector [n⁻[1], n⁻[2]]

    Q⁺.u = u⁻ - 2 * (n ∘ u⁻) * n

    return nothing
end

@inline function ocean_boundary_state!(
    ::NumericalFluxFirstOrder,
    ::CoastlineFreeSlip,
    ::BarotropicModel,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    U⁻ = Q⁻.U
    n = @SVector [n⁻[1], n⁻[2]]

    Q⁺.U = U⁻ - 2 * (n ∘ U⁻) * n

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::CoastlineFreeSlip, ::NumericalFluxGradient)

apply free slip boundary conditions for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::NumericalFluxGradient,
    ::CoastlineFreeSlip,
    ::AbstractOceanModel,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    u⁻ = Q⁻.u
    ud⁻ = A⁻.u_d
    n = @SVector [n⁻[1], n⁻[2]]

    # Q⁺.u = u⁻ - (n⋅u⁻) * n
    Q⁺.u = u⁻ - (n ∘ u⁻) * n
    A⁺.u_d = ud⁻ - (n ∘ ud⁻) * n

    return nothing
end

@inline function ocean_boundary_state!(
    ::NumericalFluxGradient,
    ::CoastlineFreeSlip,
    ::BarotropicModel,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    U⁻ = Q⁻.U
    n = @SVector [n⁻[1], n⁻[2]]

    # Q⁺.U = U⁻ - (n⋅U⁻) * n
    Q⁺.U = U⁻ - (n ∘ U⁻) * n

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::CoastlineFreeSlip, ::NumericalFluxSecondOrder)

apply free slip boundary conditions for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::CoastlineFreeSlip,
    ::AbstractOceanModel,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    D⁺.ν∇u = n⁻ * (@SVector [-0, -0])'
    D⁺.κ∇θ = n⁻ * -0

    return nothing
end

@inline function ocean_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::CoastlineFreeSlip,
    ::BarotropicModel,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    D⁺.ν∇U = -D⁻.ν∇U

    return nothing
end

"""
    CoastlineNoSlip

applies boundary condition u = 0 and ∇θ = 0
"""

"""
    ocean_boundary_state!(::AbstractOceanModel, ::CoastlineNoSlip, ::RusanovNumericalFlux)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::NumericalFluxFirstOrder,
    ::CoastlineNoSlip,
    ::AbstractOceanModel,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    Q⁺.u = -Q⁻.u

    return nothing
end

@inline function ocean_boundary_state!(
    ::NumericalFluxFirstOrder,
    ::CoastlineNoSlip,
    ::BarotropicModel,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    Q⁺.U = -Q⁻.U

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::CoastlineNoSlip, ::NumericalFluxGradient)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::NumericalFluxGradient,
    ::CoastlineNoSlip,
    ::AbstractOceanModel,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    FT = eltype(Q⁺)
    Q⁺.u = SVector(-zero(FT), -zero(FT))
    A⁺.u_d = SVector(-zero(FT), -zero(FT))

    return nothing
end

@inline function ocean_boundary_state!(
    ::NumericalFluxGradient,
    ::CoastlineNoSlip,
    ::BarotropicModel,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    FT = eltype(Q⁺)
    Q⁺.U = SVector(-zero(FT), -zero(FT))

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::CoastlineNoSlip, ::NumericalFluxSecondOrder)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::CoastlineNoSlip,
    ::AbstractOceanModel,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    Q⁺.u = -Q⁻.u
    A⁺.u_d = -A⁻.u_d

    D⁺.κ∇θ = n⁻ * -0

    return nothing
end

@inline function ocean_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::CoastlineNoSlip,
    ::BarotropicModel,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    Q⁺.U = -Q⁻.U

    return nothing
end

"""
    OceanFloorFreeSlip

applies boundary condition ∇u = 0 and ∇θ = 0
"""

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanFloorFreeSlip, ::RusanovNumericalFlux)

apply free slip boundary conditions for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::NumericalFluxFirstOrder,
    ::OceanFloorFreeSlip,
    ::AbstractOceanModel,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    A⁺.w = -A⁻.w

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanFloorFreeSlip, ::NumericalFluxGradient)

apply free slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::NumericalFluxGradient,
    ::OceanFloorFreeSlip,
    ::AbstractOceanModel,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    FT = eltype(Q⁺)
    A⁺.w = -zero(FT)

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanFloorFreeSlip, ::NumericalFluxSecondOrder)

apply free slip boundary conditions for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::OceanFloorFreeSlip,
    ::AbstractOceanModel,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    A⁺.w = -A⁻.w
    D⁺.ν∇u = n⁻ * (@SVector [-0, -0])'

    D⁺.κ∇θ = n⁻ * -0

    return nothing
end

"""
    OceanFloorNoSlip

applies boundary condition u = 0 and ∇θ = 0
"""

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanFloorNoSlip, ::RusanovNumericalFlux)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::NumericalFluxFirstOrder,
    ::OceanFloorNoSlip,
    ::AbstractOceanModel,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    Q⁺.u = -Q⁻.u
    A⁺.w = -A⁻.w

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanFloorNoSlip, ::NumericalFluxGradient)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::NumericalFluxGradient,
    ::OceanFloorNoSlip,
    ::AbstractOceanModel,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    FT = eltype(Q⁺)
    Q⁺.u = SVector(-zero(FT), -zero(FT))
    A⁺.w = -zero(FT)

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanFloorNoSlip, ::NumericalFluxSecondOrder)

apply no slip boundary condition for velocity
apply no penetration boundary for temperature
"""
@inline function ocean_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::OceanFloorNoSlip,
    ::AbstractOceanModel,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)

    Q⁺.u = -Q⁻.u
    A⁺.w = -A⁻.w

    D⁺.κ∇θ = n⁻ * -0

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::Union{OceanSurface*}, ::Union{RusanovNumericalFlux, NumericalFluxGradient})

applying neumann boundary conditions, so don't need to do anything for these numerical fluxes
"""
@inline function ocean_boundary_state!(
    ::Union{NumericalFluxFirstOrder, NumericalFluxGradient},
    ::Union{
        OceanSurfaceNoStressNoForcing,
        OceanSurfaceStressNoForcing,
        OceanSurfaceNoStressForcing,
        OceanSurfaceStressForcing,
    },
    ::AbstractOceanModel,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanSurfaceNoStressNoForcing, ::NumericalFluxSecondOrder)

apply no flux boundary condition for velocity
apply no flux boundary condition for temperature
"""
@inline function ocean_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::OceanSurfaceNoStressNoForcing,
    ::AbstractOceanModel,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    D⁺.ν∇u = n⁻ * (@SVector [-0, -0])'

    D⁺.κ∇θ = n⁻ * -0

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanSurfaceStressNoForcing, ::NumericalFluxSecondOrder)

apply wind-stress boundary condition for velocity
apply no flux boundary condition for temperature
"""
@inline function ocean_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::OceanSurfaceStressNoForcing,
    m::AbstractOceanModel,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    τᶻ = velocity_flux(m.problem, A⁻.y, m.ρₒ)
    τ = @SMatrix [-0 -0; -0 -0; τᶻ -0]
    D⁺.ν∇u = τ

    D⁺.κ∇θ = n⁻ * -0

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanSurfaceNoStressForcing, ::NumericalFluxSecondOrder)

apply no flux boundary condition for velocity
apply forcing boundary condition for temperature
"""
@inline function ocean_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::OceanSurfaceNoStressForcing,
    m::AbstractOceanModel,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    D⁺.ν∇u = n⁻ * (@SVector [-0, -0])'

    σᶻ = temperature_flux(m.problem, A⁻.y, Q⁻.θ)
    σ = @SVector [-0, -0, σᶻ]
    D⁺.κ∇θ = σ

    return nothing
end

"""
    ocean_boundary_state!(::AbstractOceanModel, ::OceanSurfaceStressForcing, ::NumericalFluxSecondOrder)

apply wind-stress boundary condition for velocity
apply forcing boundary condition for temperature
"""
@inline function ocean_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::OceanSurfaceStressForcing,
    m::AbstractOceanModel,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    τᶻ = velocity_flux(m.problem, A⁻.y, m.ρₒ)
    τ = @SMatrix [-0 -0; -0 -0; τᶻ -0]
    D⁺.ν∇u = τ

    σᶻ = temperature_flux(m.problem, A⁻.y, Q⁻.θ)
    σ = @SVector [-0, -0, σᶻ]
    D⁺.κ∇θ = σ

    return nothing
end

@inline velocity_flux(p::AbstractOceanProblem, y, ρ) =
    -(p.τₒ / ρ) * cos(y * π / p.Lʸ)

@inline function temperature_flux(p::AbstractOceanProblem, y, θ)
    θʳ = p.θᴱ * (1 - y / p.Lʸ)
    return p.λʳ * (θʳ - θ)
end
