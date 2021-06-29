using ClimateMachine,
    ClimateMachine.LinearSolvers,
    ClimateMachine.BatchedGeneralizedMinimalResidualSolver
using LinearAlgebra, Random, Plots

A1 = [
    2.0 -1.0 0.0
    -1.0 2.0 -1.0
    0.0 -1.0 2.0
];

b1 = ones(typeof(1.0), 3);

x1_exact = [1.5, 2.0, 1.5];

A2 = [
    2.0 -1.0 0.0
    0.0 2.0 -1.0
    0.0 0.0 2.0
];

b2 = ones(typeof(1.0), 3);

x2_exact = [0.875, 0.75, 0.5];

function closure_linear_operator(A1, A2)
    function linear_operator!(x, y)
        mul!(view(x, :, 1), A1, view(y, :, 1))
        mul!(view(x, :, 2), A2, view(y, :, 2))
        return nothing
    end
    return linear_operator!
end;

linear_operator! = closure_linear_operator(A1, A2);

y1 = ones(typeof(1.0), 3);
y2 = ones(typeof(1.0), 3) * 2.0;
y = [y1 y2];
x = copy(y);
linear_operator!(x, y);
x

[A1 * y1 A2 * y2]

b = [b1 b2];

x_exact = [x1_exact x2_exact];

linearsolver = BatchedGeneralizedMinimalResidual(b);

x1 = ones(typeof(1.0), 3);
x2 = ones(typeof(1.0), 3);
x = [x1 x2];

iters = linearsolve!(linear_operator!, linearsolver, x, b)

x

x_exact

function closure_linear_operator!(A, tup)
    function linear_operator!(y, x)
        alias_x = reshape(x, tup)
        alias_y = reshape(y, tup)
        for i6 in 1:tup[6]
            for i4 in 1:tup[4]
                for i2 in 1:tup[2]
                    for i1 in 1:tup[1]
                        tmp = alias_x[i1, i2, :, i4, :, i6][:]
                        tmp2 = A[i1, i2, i4, i6] * tmp
                        alias_y[i1, i2, :, i4, :, i6] .=
                            reshape(tmp2, (tup[3], tup[5]))
                    end
                end
            end
        end
    end
end;

tup = (2, 2, 5, 2, 10, 2);

Random.seed!(1234);
B = [
    randn(tup[3] * tup[5], tup[3] * tup[5])
    for i1 in 1:tup[1], i2 in 1:tup[2], i4 in 1:tup[4], i6 in 1:tup[6]
];
columnwise_A = [
    B[i1, i2, i4, i6] + 3 * (i1 + i2 + i4 + i6) * I
    for i1 in 1:tup[1], i2 in 1:tup[2], i4 in 1:tup[4], i6 in 1:tup[6]
];

columnwise_inv_A = [
    inv(columnwise_A[i1, i2, i4, i6])
    for i1 in 1:tup[1], i2 in 1:tup[2], i4 in 1:tup[4], i6 in 1:tup[6]
];
columnwise_linear_operator! = closure_linear_operator!(columnwise_A, tup);
columnwise_inverse_linear_operator! =
    closure_linear_operator!(columnwise_inv_A, tup);

mpi_tup = (tup[1] * tup[2] * tup[3], tup[4], tup[5] * tup[6]);

b = randn(mpi_tup);

x = copy(b);
x += randn(mpi_tup) * 0.1;

reshape_tuple_f = tup;

permute_tuple_f = (5, 3, 4, 6, 1, 2);

ArrayType = Array;

gmres = BatchedGeneralizedMinimalResidual(
    b,
    ArrayType = ArrayType,
    m = tup[3] * tup[5] * tup[4],
    n = tup[1] * tup[2] * tup[6],
    reshape_tuple_f = reshape_tuple_f,
    permute_tuple_f = permute_tuple_f,
    atol = eps(Float64) * 10^2,
    rtol = eps(Float64) * 10^2,
);

iters = linearsolve!(
    columnwise_linear_operator!,
    gmres,
    x,
    b,
    max_iters = tup[3] * tup[5] * tup[4],
)

x_exact = copy(x);
columnwise_inverse_linear_operator!(x_exact, b);

norm(x - x_exact) / norm(x_exact)
columnwise_linear_operator!(x_exact, x);
norm(x_exact - b) / norm(b)

plot(log.(gmres.residual[1:iters, :]) / log(10));
plot!(legend = false, xlims = (1, iters), ylims = (-15, 2));
plot!(ylabel = "log10 residual", xlabel = "iterations")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

