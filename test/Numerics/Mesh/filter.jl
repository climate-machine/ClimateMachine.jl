using Test
import ClimateMachine
using ClimateMachine.VariableTemplates: @vars, Vars
using ClimateMachine.Mesh.Grids:
    EveryDirection, HorizontalDirection, VerticalDirection
using ClimateMachine.MPIStateArrays: weightedsum
using ClimateMachine.BalanceLaws

import GaussQuadrature
using MPI
using LinearAlgebra

ClimateMachine.init()

@testset "Exponential and Cutoff filter matrix" begin
    let
        # Values computed with:
        #   https://github.com/tcew/nodal-dg/blob/master/Codes1.1/Codes1D/Filter1D.m
        #! format: off
        W = [0x3fe98f3cd0d725e8  0x3fddfd863c6c9a44  0xbfe111110d0fd334  0x3fddbe357bce0b5c  0xbfc970267f929618
             0x3fb608a150f6f927  0x3fe99528b1a1cd8d  0x3fcd41d41f8bae45  0xbfc987d5fabab8d5  0x3fb5da1cd858af87
             0xbfb333332eb1cd92  0x3fc666666826f178  0x3fe999999798faaa  0x3fc666666826f176  0xbfb333332eb1cd94
             0x3fb5da1cd858af84  0xbfc987d5fabab8d4  0x3fcd41d41f8bae46  0x3fe99528b1a1cd8e  0x3fb608a150f6f924
             0xbfc970267f929618  0x3fddbe357bce0b5c  0xbfe111110d0fd333  0x3fddfd863c6c9a44  0x3fe98f3cd0d725e8]
        #! format: on
        W = reinterpret.(Float64, W)

        N = size(W, 1) - 1

        topology = ClimateMachine.Mesh.Topologies.BrickTopology(
            MPI.COMM_SELF,
            -1.0:2.0:1.0,
        )

        grid = ClimateMachine.Mesh.Grids.DiscontinuousSpectralElementGrid(
            topology;
            polynomialorder = N,
            FloatType = Float64,
            DeviceArray = Array,
        )

        filter = ClimateMachine.Mesh.Filters.ExponentialFilter(grid, 0, 32)
        nf = length(filter.filter_matrices)
        @test all(ntuple(i -> filter.filter_matrices[i] ≈ W, nf))
    end

    let
        # Values computed with:
        #   https://github.com/tcew/nodal-dg/blob/master/Codes1.1/Codes1D/Filter1D.m
        #! format: off
        W = [0x3fd822e5f54ecb62   0x3fedd204a0f08ef8   0xbfc7d3aa58fd6968   0xbfbf74682ac4d276
             0x3fc7db36e726d8c1   0x3fe59d16feee478b   0x3fc6745bfbb91e20   0xbfa30fbb7a645448
             0xbfa30fbb7a645455   0x3fc6745bfbb91e26   0x3fe59d16feee478a   0x3fc7db36e726d8c4
             0xbfbf74682ac4d280   0xbfc7d3aa58fd6962   0x3fedd204a0f08ef7   0x3fd822e5f54ecb62]
        #! format: on
        W = reinterpret.(Float64, W)

        N = size(W, 1) - 1

        topology = ClimateMachine.Mesh.Topologies.BrickTopology(
            MPI.COMM_SELF,
            -1.0:2.0:1.0,
        )
        grid = ClimateMachine.Mesh.Grids.DiscontinuousSpectralElementGrid(
            topology;
            polynomialorder = N,
            FloatType = Float64,
            DeviceArray = Array,
        )

        filter = ClimateMachine.Mesh.Filters.ExponentialFilter(grid, 1, 4)
        nf = length(filter.filter_matrices)
        @test all(ntuple(i -> filter.filter_matrices[i] ≈ W, nf))
    end

    let
        T = Float64
        N = (5, 3)
        Nc = (4, 2)

        topology = ClimateMachine.Mesh.Topologies.BrickTopology(
            MPI.COMM_SELF,
            -1.0:2.0:1.0,
        )
        grid = ClimateMachine.Mesh.Grids.DiscontinuousSpectralElementGrid(
            topology;
            polynomialorder = N,
            FloatType = T,
            DeviceArray = Array,
        )

        ξ = ClimateMachine.Mesh.Grids.referencepoints(grid)
        ξ1 = ξ[1]
        ξ2 = ξ[2]
        a1, b1 = GaussQuadrature.legendre_coefs(T, N[1])
        a2, b2 = GaussQuadrature.legendre_coefs(T, N[2])
        V1 = GaussQuadrature.orthonormal_poly(ξ1, a1, b1)
        V2 = GaussQuadrature.orthonormal_poly(ξ2, a2, b2)

        Σ1 = ones(T, N[1] + 1)
        Σ2 = ones(T, N[2] + 1)
        Σ1[(Nc[1]:N[1]) .+ 1] .= 0
        Σ2[(Nc[2]:N[2]) .+ 1] .= 0

        W1 = V1 * Diagonal(Σ1) / V1
        W2 = V2 * Diagonal(Σ2) / V2

        filter = ClimateMachine.Mesh.Filters.CutoffFilter(grid, Nc)
        @test filter.filter_matrices[1] ≈ W1
        @test filter.filter_matrices[2] ≈ W2
    end
end

struct FilterTestModel{N} <: ClimateMachine.BalanceLaws.BalanceLaw end
ClimateMachine.BalanceLaws.vars_state(::FilterTestModel, ::Auxiliary, FT) =
    @vars()
ClimateMachine.BalanceLaws.init_state_auxiliary!(::FilterTestModel, _...) =
    nothing

# Legendre Polynomials
l0(r) = 1
l1(r) = r
l2(r) = (3 * r^2 - 1) / 2
l3(r) = (5 * r^3 - 3r) / 2

low(x, y, z) = l0(x) * l0(y) + 4 * l1(x) * l1(y) + 5 * l1(z) + 6 * l1(z) * l1(x)

high(x, y, z) = l2(x) * l3(y) + l3(x) + l2(y) + l3(z) * l1(y)

filtered(::EveryDirection, dim, x, y, z) = high(x, y, z)
filtered(::VerticalDirection, dim, x, y, z) =
    (dim == 2) ? l2(x) * l3(y) + l2(y) : l3(z) * l1(y)
filtered(::HorizontalDirection, dim, x, y, z) =
    (dim == 2) ? l2(x) * l3(y) + l3(x) : l2(x) * l3(y) + l3(x) + l2(y)

ClimateMachine.BalanceLaws.vars_state(::FilterTestModel{4}, ::Prognostic, FT) =
    @vars(q1::FT, q2::FT, q3::FT, q4::FT)
function ClimateMachine.BalanceLaws.init_state_prognostic!(
    ::FilterTestModel{4},
    state::Vars,
    aux::Vars,
    localgeo,
    filter_direction,
    dim,
)
    (x, y, z) = localgeo.coord
    state.q1 = low(x, y, z) + high(x, y, z)
    state.q2 = low(x, y, z) + high(x, y, z)
    state.q3 = low(x, y, z) + high(x, y, z)
    state.q4 = low(x, y, z) + high(x, y, z)

    if !isnothing(filter_direction)
        state.q1 -= filtered(filter_direction, dim, x, y, z)
        state.q3 -= filtered(filter_direction, dim, x, y, z)
    end
end

@testset "Exponential and Cutoff filter application" begin
    N = 3
    Ne = (1, 1, 1)

    @testset for FT in (Float64, Float32)
        @testset for dim in 2:3
            @testset for direction in (
                EveryDirection,
                HorizontalDirection,
                VerticalDirection,
            )
                brickrange = ntuple(
                    j -> range(FT(-1); length = Ne[j] + 1, stop = 1),
                    dim,
                )
                topl = ClimateMachine.Mesh.Topologies.BrickTopology(
                    MPI.COMM_WORLD,
                    brickrange,
                    periodicity = ntuple(j -> true, dim),
                )

                grid =
                    ClimateMachine.Mesh.Grids.DiscontinuousSpectralElementGrid(
                        topl,
                        FloatType = FT,
                        DeviceArray = ClimateMachine.array_type(),
                        polynomialorder = N,
                    )

                filter = ClimateMachine.Mesh.Filters.CutoffFilter(grid, 2)

                model = FilterTestModel{4}()
                dg = ClimateMachine.DGMethods.DGModel(
                    model,
                    grid,
                    nothing,
                    nothing,
                    nothing;
                    state_gradient_flux = nothing,
                )

                @testset for target in ((1, 3), (:q1, :q3))
                    Q = ClimateMachine.DGMethods.init_ode_state(
                        dg,
                        nothing,
                        dim,
                    )
                    ClimateMachine.Mesh.Filters.apply!(
                        Q,
                        target,
                        grid,
                        filter,
                        direction = direction(),
                    )
                    P = ClimateMachine.DGMethods.init_ode_state(
                        dg,
                        direction(),
                        dim,
                    )
                    @test Array(Q.data) ≈ Array(P.data)
                end
            end
        end
    end
end

ClimateMachine.BalanceLaws.vars_state(
    ::FilterTestModel{1},
    ::Prognostic,
    FT,
) where {N} = @vars(q::FT)
function ClimateMachine.BalanceLaws.init_state_prognostic!(
    ::FilterTestModel{1},
    state::Vars,
    aux::Vars,
    localgeo,
)
    (x, y, z) = localgeo.coord
    state.q = abs(x) - 0.1
end

@testset "TMAR filter application" begin

    N = 4
    Ne = (2, 2, 2)

    @testset for FT in (Float64, Float32)
        @testset for dim in 2:3
            brickrange =
                ntuple(j -> range(FT(-1); length = Ne[j] + 1, stop = 1), dim)
            topl = ClimateMachine.Mesh.Topologies.BrickTopology(
                MPI.COMM_WORLD,
                brickrange,
                periodicity = ntuple(j -> true, dim),
            )

            grid = ClimateMachine.Mesh.Grids.DiscontinuousSpectralElementGrid(
                topl,
                FloatType = FT,
                DeviceArray = ClimateMachine.array_type(),
                polynomialorder = N,
            )

            model = FilterTestModel{1}()
            dg = ClimateMachine.DGMethods.DGModel(
                model,
                grid,
                nothing,
                nothing,
                nothing;
                state_gradient_flux = nothing,
            )

            @testset for target in ((1,), (:q,), :)
                Q = ClimateMachine.DGMethods.init_ode_state(dg)

                initialsumQ = weightedsum(Q)
                @test minimum(Q.realdata) < 0

                ClimateMachine.Mesh.Filters.apply!(
                    Q,
                    target,
                    grid,
                    ClimateMachine.Mesh.Filters.TMARFilter(),
                )

                sumQ = weightedsum(Q)

                @test minimum(Q.realdata) >= 0
                @test isapprox(initialsumQ, sumQ; rtol = 10 * eps(FT))
            end
        end
    end
end
