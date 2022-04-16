#=
ni基超导
=#

include("../../CPWalker/src/CPWalker.jl")
using ..CPWalker

using LatticeHamiltonian
using LinearAlgebra

L = 6
Nup = 18
Δτ = 0.1

NWALKERS = 200

U = 0.

TA1 = -1

TB1 = -1

PCTRL_INTERVAL = 20
STBLZ_INTERVAL = 10

"""
从x，y获得编号
"""
function get_index_from_posi(L, x, y)
    x_ = x
    while x_ > L || x_ < 1
        x_ = x_ - sign(x_-1) * L
    end
    y_ = y
    while y_ > L || y_ < 1
        y_ = y_ - sign(y_-1) * L
    end
    return x_ + (y_ - 1)*L
end


"""
从编号获得xy
"""
function get_posi_from_index(L, idx)
    x = mod(idx, L)
    y = (idx - x) // L
    return x, y
end


"""
获得hopping矩阵
"""
function hopping_matrix(L, TA1, TB1)
    h0 = zeros(2*L^2, 2*L^2)
    for xidx=1:1:L; for yidx=1:1:L
        uidx = get_index_from_posi(L, xidx, yidx)
        aidx = 2*uidx-1
        bidx = 2*uidx
        #最近邻
        uridx = get_index_from_posi(L, xidx+1, yidx)
        aridx = 2*uridx-1
        bridx = 2*uridx
        h0[aidx, aridx] = TA1
        h0[aridx, aidx] = TA1
        h0[bidx, bridx] = TB1
        h0[bridx, bidx] = TB1
        #
        utidx = get_index_from_posi(L, xidx, yidx+1)
        atidx = 2*utidx-1
        btidx = 2*utidx
        h0[aidx, atidx] = TA1
        h0[atidx, aidx] = TA1
        h0[bidx, btidx] = TB1
        h0[btidx, bidx] = TB1
        #
        h0[bidx, bidx] -= -U/2
    end; end
    return h0
end


"""
试探波函数
"""
function trial_wavefunction(h0, Nup)
    eigvs = eigvecs(h0)
    phi = eigvs[:, 1:Nup]
    return copy(phi), copy(phi)
end


"""
增加相互作用
"""
function add_interactions(ham::HamConfig2, L, U, Δτ)
    for idx = 2:2:L
        push!(ham.Mzints, (idx, 1, idx, 2))
        push!(ham.Axflds, AuxiliaryField2("int"*string(idx), U, Δτ))
    end
end


"""
创建walkers
"""
function create_walkers(ham, Nup, NWALKERS)
    walkers = Vector{HSWalker2}(undef, NWALKERS)
    phiup, phidn = trial_wavefunction(ham.H0.V, Nup)
    walkers[1] = HSWalker2("wlk1", ham, phiup, phidn, 1.0)
    for iw = 2:1:NWALKERS
        walkers[iw] = clone(walkers[1], "wlk"*string(iw))
    end
    return walkers
end


"""
观测
"""
function measurements(ham, wlk, eqgr,
    whgt_bin, engr_bin)
    eng = cal_energy(eqgr, ham)
    whgt_bin[end].V += wlk.weight
    engr_bin[end].V += wlk.weight * eng
end


"""
入口
"""
function main()
    global L, U, Nup, Δτ, TA1, TB1, NWALKERS
    global PCTRL_INTERVAL, STBLZ_INTERVAL
    h0 = hopping_matrix(L, TA1, TB1)
    phiup, phidn = trial_wavefunction(h0, Nup)
    #println(phiup)
    ham = HamConfig2(h0, Δτ, phiup, phidn)
    add_interactions(ham, L, U, Δτ)
    walkers = create_walkers(ham, Nup, NWALKERS)
    cpsim = CPSim2(ham, walkers, STBLZ_INTERVAL, PCTRL_INTERVAL)
    #
    igr = initialize_simulation!(cpsim)
    println(cpsim.E_trial)
    relaxation_simulation(cpsim, 10)
    E_trial_simulation!(cpsim, 10, 10)
    println(cpsim.E_trial)
    #
    whgt_bin = CPMeasure{:SCALE, Float64}[]
    engr_bin = CPMeasure{:SCALE, Float64}[]
    #
    meas_bin = 10
    meas_time = 10
    initial_meaurements!(cpsim, 40)
    for bidx = 1:1:meas_bin
        push!(whgt_bin, CPMeasure{:SCALE, Float64}("weight", 0.0))
        push!(engr_bin, CPMeasure{:SCALE, Float64}("energy", 0.0))
        for tidx = 1:1:meas_time
            premeas_simulation!(cpsim, 40)
            #postmeas_simulation(cpsim)
            for widx = 1:1:length(cpsim.walkers)
                measurements(cpsim.hamiltonian, cpsim.walkers[widx], cpsim.eqgrs[widx],
                whgt_bin, engr_bin
                )
            end
            postmeas_simulation(cpsim)
        end
    end
    println(whgt_bin)
    println(engr_bin)
    #
    total_engr = 0.
    for bidx = 1:1:meas_bin
        total_engr += engr_bin[bidx].V / whgt_bin[bidx].V
    end
    total_engr = total_engr / meas_bin
    println("total energy = ", total_engr)
end


main()

