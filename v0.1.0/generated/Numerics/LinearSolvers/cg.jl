using ClimateMachine,
    ClimateMachine.LinearSolvers, ClimateMachine.ConjugateGradientSolver
using LinearAlgebra, Random

A = [
    2.0 -1.0 0.0
    -1.0 2.0 -1.0
    0.0 -1.0 2.0
];

eigvals(A)

function closure_linear_operator!(A)
    function linear_operator!(x, y)
        mul!(x, A, y)
    end
    return linear_operator!
end;

linear_operator! = closure_linear_operator!(A)

b = ones(typeof(1.0), 3);

x_exact = [1.5, 2.0, 1.5];

linearsolver = ConjugateGradient(b);

x = ones(typeof(1.0), 3);

iters = linearsolve!(linear_operator!, linearsolver, x, b)

norm(x - x_exact) / norm(x_exact)

norm(A * x - b) / norm(b)

iters

A = [
    2.0 -1.0 0.0
    0.0 2.0 -1.0
    0.0 0.0 2.0
];

eigvals(A)

function closure_linear_operator!(A)
    function linear_operator!(x, y)
        mul!(x, A, y)
    end
    return linear_operator!
end;

linear_operator! = closure_linear_operator!(A)

b = ones(typeof(1.0), 3);

x_exact = [0.875, 0.75, 0.5];

linearsolver = ConjugateGradient(b, max_iter = 100);

x = ones(typeof(1.0), 3);

iters = linearsolve!(linear_operator!, linearsolver, x, b)

norm(x - x_exact) / norm(x_exact)

norm(A * x - b) / norm(b)

iters

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

tup = (3, 4, 7, 2, 20, 2);

Random.seed!(1235);
B = [
    randn(tup[3] * tup[5], tup[3] * tup[5])
    for i1 in 1:tup[1], i2 in 1:tup[2], i4 in 1:tup[4], i6 in 1:tup[6]
];
columnwise_A = [
    B[i1, i2, i4, i6] * B[i1, i2, i4, i6]' + 10I
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
x = randn(mpi_tup);

linearsolver = ConjugateGradient(
    x,
    max_iter = tup[3] * tup[5],
    dims = (3, 5),
    reshape_tuple = tup,
);

iters = linearsolve!(columnwise_linear_operator!, linearsolver, x, b);
x_exact = copy(x);
columnwise_inverse_linear_operator!(x_exact, b);

norm(x - x_exact) / norm(x_exact)

iters

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

