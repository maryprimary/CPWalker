#=
测试反向传播
=#

include("../../CPWalker/src/CPWalker.jl")
using ..CPWalker

using LatticeHamiltonian
using LatticeHamiltonian.SlaterDeterminant
using LinearAlgebra


Δτ = 0.1
h0 = rand(8, 8)
h0 = h0+h0'

phi = eigvecs(h0)[:, 1:2]

ham = HamConfig2(h0, Δτ, copy(phi), copy(phi))

for idx=1:1:8
    push!(ham.Mzints, (idx, 1, idx, 2))
    push!(ham.Axflds, AuxiliaryField2("int"*string(idx), 1.0, Δτ))
end

walkers = Vector{HSWalker2}(undef, 1)
walkers[1] = HSWalker2("wlk1", ham, copy(phi), copy(phi), 1.0)

cpsim = CPSim2(ham, walkers, 10, 20)
igr = initialize_simulation!(cpsim, true)

relaxation_simulation(cpsim, 1000)
E_trial_simulation!(cpsim, 10, 10)

println(cpsim.E_trial)


initial_meaurements!(cpsim, 40)

println()

println(walkers)
premeas_simulation!(cpsim, 40)
println(walkers)


sla1 = Slater("backwlk"*walkers[1].Φ[1].name, copy(ham.Φt[1].V))
sla2 = Slater("backwlk"*walkers[1].Φ[2].name, copy(ham.Φt[2].V))
backwlk = HSWalker2(
    (sla1, sla2), 1.0, 1.0, missing, missing, missing
)

println(backwlk)
