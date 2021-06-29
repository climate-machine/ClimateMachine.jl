function f()
    return x = [i * i for i in 1:10]
end

x = f()

f();

using Plots
plot(x)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

