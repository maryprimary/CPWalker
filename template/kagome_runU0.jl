#=
测试三角格子
=#


include("../../CPWalker/src/CPWalker.jl")
include("uhf_den.jl")
using ..CPWalker

using LatticeHamiltonian
using LinearAlgebra
using JLD

L = 3
Nup = 11
Δτ = 0.01

NWALKERS = 20

U = 0.

HOPPING_T = -1

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
function hopping_matrix(L, HOPPING_T)
    h0 = zeros(3*L^2, 3*L^2)
    for x=1:1:L; for y=1:1:L
        uidx = get_index_from_posi(L, x, y)
        aidx = 3*uidx-2
        bidx = 3*uidx-1
        cidx = 3*uidx
        #A的两个
        h0[aidx, bidx] = HOPPING_T
        h0[bidx, aidx] = HOPPING_T
        h0[aidx, cidx] = HOPPING_T
        h0[cidx, aidx] = HOPPING_T
        #B的两个
        rtidx = get_index_from_posi(L, x, y+1)
        artidx = 3*rtidx-2
        h0[bidx, artidx] = HOPPING_T
        h0[artidx, bidx] = HOPPING_T
        h0[bidx, cidx] = HOPPING_T
        h0[cidx, bidx] = HOPPING_T
        #C的两个
        ltidx = get_index_from_posi(L, x+1, y)
        altidx = 3*ltidx-2
        h0[cidx, altidx] = HOPPING_T
        h0[altidx, cidx] = HOPPING_T
        lidx = get_index_from_posi(L, x+1, y-1)
        blidx = 3*lidx-1
        h0[cidx, blidx] = HOPPING_T
        h0[blidx, cidx] = HOPPING_T
    end; end
    return h0
end


"""
试探波函数
"""
function trial_wavefunction(h0, Nup)
    #return uhf_trial(h0, Nup, U)
    h0d = copy(h0)
    ##nsite = size(h0d)[1]
    ##for ni = 1:1:nsite
    ##    h0d[ni, ni] += 0.01*(rand()-0.5)
    ##end
    ##println("close shell ")
    ##println(eigvals(h0d))
    ##println("------")
    eigvs = eigvecs(h0d)
    phi = eigvs[:, 1:Nup]
    return copy(phi), copy(phi)
    #if isfile("etgr.jld")
    #    phiup, phidn = construct_frg_hamiltonian2(
    #        "etgr.jld", "ints_L2_mu0.000_U_6.0.jld", h0, 3*L^2, Nup
    #    )
    #else
    #    phiup, phidn = construct_frg_hamiltonian(
    #        "ints_L2_mu0.000_U_6.0.jld", h0, 3*L^2, Nup
    #    )
    #end
    #phiup, phidn = construct_frg_hamiltonian2(
    #    "etgr.jld", "ints_L2_mu-1.163_U_3.0.jld", h0, 3*L^2, Nup
    #)
    #return phiup, phidn#copy(phi), copy(phi)
end


"""
增加相互作用
"""
function add_interactions(ham::HamConfig2, L, U, Δτ)
    for idx = 1:1:3*L^2
        push!(ham.Mzints, (idx, 1, idx, 2))
        push!(ham.Axflds, AuxiliaryField2("int"*string(idx), U, Δτ))
    end
end


"""
创建walkers
"""
function create_walkers(ham, Nup, NWALKERS, phiup, phidn)
    walkers = Vector{HSWalker2}(undef, NWALKERS)
    #phiup, phidn = trial_wavefunction(ham.H0.V, Nup)
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
    whgt_bin, engr_bin, spcr_bin, etgr_bin)
    eng = cal_energy(eqgr, ham)
    whgt_bin[end].V += wlk.weight
    engr_bin[end].V += wlk.weight * eng
    #
    #
    szmat = zeros(Float64, 3*L^2, 3*L^2)
    for i=1:1:3*L^2; for j=1:1:3*L^2
        if i == j
            continue
        end
        # (niu - nid) (nju - njd)
        # niu nju
        szmat[i, j] += eqgr.V[i, i, 1] * eqgr.V[j, j, 1]
        szmat[i, j] += eqgr.V[i, j, 1] * (-eqgr.V[j, i, 1])
        # -n1u n2d
        szmat[i, j] -= eqgr.V[i, i, 1] * eqgr.V[j, j, 2]
        # -n1d n2u
        szmat[i, j] -= eqgr.V[i, i, 2] * eqgr.V[j, j, 1]
        # n1d n2d
        szmat[i, j] += eqgr.V[i, i, 2] * eqgr.V[j, j, 2]
        szmat[i, j] += eqgr.V[i, j, 2] * (-eqgr.V[j, i, 2])
    end; end
    spcr_bin[end].V += wlk.weight * szmat
    #
    etgr_bin[end].V += wlk.weight * eqgr.V
end


"""
入口
"""
function main()
    global L, U, Nup, Δτ, HOPPING_T, NWALKERS
    global PCTRL_INTERVAL, STBLZ_INTERVAL
    h0 = hopping_matrix(L, HOPPING_T)
    println(eigvals(h0))
    phiup, phidn = trial_wavefunction(h0, Nup)
    #println(phiup)
    ham = HamConfig2(h0, Δτ, phiup, phidn)
    add_interactions(ham, L, U, Δτ)
    walkers = create_walkers(ham, Nup, NWALKERS, copy(phiup), copy(phidn))
    cpsim = CPSim2(ham, walkers, STBLZ_INTERVAL, PCTRL_INTERVAL)
    #
    igr = initialize_simulation!(cpsim, true)
    println("igr_up", [igr.V[idx, idx, 1] for idx=1:1:12])
    println("igr_dn", [igr.V[idx, idx, 2] for idx=1:1:12])
    println(cpsim.E_trial)
    #exit()
    relaxation_simulation(cpsim, 50)
    E_trial_simulation!(cpsim, 10, 10)
    println(cpsim.E_trial)
    #
    whgt_bin = CPMeasure{:SCALE, Float64}[]
    engr_bin = CPMeasure{:SCALE, Float64}[]
    spcr_bin = CPMeasure{:MATRIX, Matrix{Float64}}[]
    etgr_bin = CPMeasure{:EQGR, Array{Float64, 3}}[]
    #
    meas_bin = 10
    meas_time = 10
    initial_meaurements!(cpsim, 40)
    for bidx = 1:1:meas_bin
        push!(whgt_bin, CPMeasure{:SCALE, Float64}("weight", 0.0))
        push!(engr_bin, CPMeasure{:SCALE, Float64}("energy", 0.0))
        push!(spcr_bin, CPMeasure{:MATRIX, Matrix{Float64}}("spin corr",
        zeros(Float64, 3*L^2, 3*L^2)))
        push!(etgr_bin, CPMeasure{:EQGR, Array{Float64, 3}}(
            "etgr", zeros(Float64, 3*L^2, 3*L^2, 2)
        ))
        for tidx = 1:1:meas_time
            premeas_simulation!(cpsim, 40)
            #postmeas_simulation(cpsim)
            for widx = 1:1:length(cpsim.walkers)
                measurements(cpsim.hamiltonian, cpsim.walkers[widx], cpsim.eqgrs[widx],
                whgt_bin, engr_bin, spcr_bin, etgr_bin
                )
            end
            postmeas_simulation(cpsim)
        end
    end
    println(whgt_bin)
    println(engr_bin)
    println([spcr.V[1, 2] for spcr in spcr_bin])
    println([spcr.V[1, 3] for spcr in spcr_bin])
    #
    total_engr = 0.
    for bidx = 1:1:meas_bin
        total_engr += engr_bin[bidx].V / whgt_bin[bidx].V
    end
    total_engr = total_engr / meas_bin
    println("total energy = ", total_engr)
    error_engr = 0.
    for bidx = 1:1:meas_bin
        eng = engr_bin[bidx].V / whgt_bin[bidx].V
        error_engr += (eng - total_engr)^2
    end
    error_engr = sqrt(error_engr / (meas_bin-1))
    println("error energy = ", error_engr)
    #
    for j=2:1:12
        tot_szcr = 0.
        for bidx = 1:1:meas_bin
            tot_szcr += spcr_bin[bidx].V[1, j] / whgt_bin[bidx].V
        end
        tot_szcr = tot_szcr / meas_bin
        err_szcr = 0.
        for bidx = 1:1:meas_bin
            szcr = spcr_bin[bidx].V[1, j] / whgt_bin[bidx].V
            err_szcr += (szcr - tot_szcr)^2
        end
        err_szcr = sqrt(err_szcr/(meas_bin - 1))
        println("sz(1,"*string(j)*") ", tot_szcr, " ", err_szcr)
    end
    #
    avg_eqgr = zeros(Float64, 3*L^2, 3*L^2, 2)
    for bidx = 1:1:meas_bin
        avg_eqgr += etgr_bin[bidx].V / whgt_bin[bidx].V
    end
    avg_eqgr = avg_eqgr / meas_bin
    totp = 0.
    for idx=1:1:12
        println(avg_eqgr[idx, idx, 1], " ", avg_eqgr[idx, idx, 2])
        totp += avg_eqgr[idx, idx, 1]
    end
    println(totp)
    #save("etgr.jld", "etgr", avg_eqgr)
end


main()

