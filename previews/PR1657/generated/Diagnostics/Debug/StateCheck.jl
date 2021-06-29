using MPI
using StaticArrays
using Random
using ClimateMachine
using ClimateMachine.VariableTemplates
using ClimateMachine.MPIStateArrays
using ClimateMachine.GenericCallbacks
using ClimateMachine.StateCheck

ClimateMachine.init()
FT = Float64

F1 = @vars begin
    ν∇u::SMatrix{3, 2, FT, 6}
    κ∇θ::SVector{3, FT}
end
F2 = @vars begin
    u::SVector{2, FT}
    θ::SVector{1, FT}
end
nothing # hide

Q1 = MPIStateArray{Float32, F1}(
    MPI.COMM_WORLD,
    ClimateMachine.array_type(),
    4,
    9,
    8,
)
Q2 = MPIStateArray{Float64, F2}(
    MPI.COMM_WORLD,
    ClimateMachine.array_type(),
    4,
    3,
    8,
)
nothing # hide

cb = ClimateMachine.StateCheck.sccreate(
    [(Q1, "My gradients"), (Q2, "My fields")],
    1;
    prec = 15,
)
GenericCallbacks.init!(cb, nothing, nothing, nothing, nothing)
nothing # hide

typeof(cb)

Q1.data .= rand(MersenneTwister(0), Float32, size(Q1.data))
Q2.data .= rand(MersenneTwister(0), Float64, size(Q2.data))
GenericCallbacks.call!(cb, nothing, nothing, nothing, nothing)

ClimateMachine.StateCheck.scprintref(cb)

#! format: off
varr = [
 [ "My gradients", "ν∇u[1]",  1.34348869323730468750e-04,  9.84732866287231445313e-01,  5.23545503616333007813e-01,  3.08209930764271777814e-01 ],
 [ "My gradients", "ν∇u[2]",  1.16317868232727050781e-01,  9.92088317871093750000e-01,  4.83800649642944335938e-01,  2.83350456014221541157e-01 ],
 [ "My gradients", "ν∇u[3]",  1.05845928192138671875e-03,  9.51775908470153808594e-01,  4.65474426746368408203e-01,  2.73615551085745090099e-01 ],
 [ "My gradients", "ν∇u[4]",  5.97668886184692382813e-02,  9.68048095703125000000e-01,  5.42618036270141601563e-01,  2.81570862027933854765e-01 ],
 [ "My gradients", "ν∇u[5]",  8.31030607223510742188e-02,  9.35931921005249023438e-01,  5.05405902862548828125e-01,  2.46073509972619536290e-01 ],
 [ "My gradients", "ν∇u[6]",  3.09681892395019531250e-02,  9.98341441154479980469e-01,  4.54375565052032470703e-01,  3.09461067853178561915e-01 ],
 [ "My gradients", "κ∇θ[1]",  8.47448110580444335938e-02,  9.94180679321289062500e-01,  5.27157366275787353516e-01,  2.92455951648181833313e-01 ],
 [ "My gradients", "κ∇θ[2]",  1.20514631271362304688e-02,  9.93527650833129882813e-01,  4.71063584089279174805e-01,  2.96449027197666359346e-01 ],
 [ "My gradients", "κ∇θ[3]",  8.14980268478393554688e-02,  9.55443382263183593750e-01,  5.05038917064666748047e-01,  2.77201022741208891187e-01 ],
 [    "My fields",   "u[1]",  4.31410233294131639781e-02,  9.97140933049696531754e-01,  4.62139750850942054861e-01,  3.23076684924287371725e-01 ],
 [    "My fields",   "u[2]",  1.01416659908237782872e-02,  9.14712023896926407218e-01,  4.76160523012988778913e-01,  2.71443440757963339038e-01 ],
 [    "My fields",   "θ[1]",  6.58965491052394547467e-02,  9.73216404386510802738e-01,  4.60007166313864512830e-01,  2.87310472114545079059e-01 ],
]
parr = [
 [ "My gradients", "ν∇u[1]",    16,     7,    16,     0 ],
 [ "My gradients", "ν∇u[2]",    16,     7,    16,     0 ],
 [ "My gradients", "ν∇u[3]",    16,     7,    16,     0 ],
 [ "My gradients", "ν∇u[4]",    16,     7,    16,     0 ],
 [ "My gradients", "ν∇u[5]",    16,     7,    16,     0 ],
 [ "My gradients", "ν∇u[6]",    16,     7,    16,     0 ],
 [ "My gradients", "κ∇θ[1]",    16,    16,    16,     0 ],
 [ "My gradients", "κ∇θ[2]",    16,    16,    16,     0 ],
 [ "My gradients", "κ∇θ[3]",    16,    16,    16,     0 ],
 [    "My fields",   "u[1]",    16,    16,    16,     0 ],
 [    "My fields",   "u[2]",    16,    16,    16,     0 ],
 [    "My fields",   "θ[1]",    16,    16,    16,     0 ],
]
#! format: on

ClimateMachine.StateCheck.scdocheck(cb, (varr, parr))
nothing # hide

varr[1][3] = varr[1][3] * 10.0
ClimateMachine.StateCheck.scdocheck(cb, (varr, parr))
nothing # hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

