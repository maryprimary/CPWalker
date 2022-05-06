#=
测试三角格子
=#


include("../../CPWalker/src/CPWalker.jl")
include("uhf_den.jl")
using ..CPWalker

using LatticeHamiltonian
using LinearAlgebra
using JLD

L1 = 4
L2 = 3
Nup = 5
Δτ = 0.1

NWALKERS = 400

U = 3.

HOPPING_T = -1

PCTRL_INTERVAL = 20
STBLZ_INTERVAL = 10

"""
从x，y获得编号
"""
function get_index_from_posi(L1, L2, x, y)
    x_ = x
    while x_ > L1 || x_ < 1
        x_ = x_ - sign(x_-1) * L1
    end
    y_ = y
    while y_ > L2 || y_ < 1
        y_ = y_ - sign(y_-1) * L2
    end
    return x_ + (y_ - 1)*L1
end


"""
从编号获得xy
"""
function get_posi_from_index(L1, L2, idx)
    x = mod(idx-1, L1)+1
    y = (idx - x) // L1
    return x, Int64(y+1)
end


"""
获得hopping矩阵
"""
function hopping_matrix(L1, L2, HOPPING_T)
    h0 = zeros(L1*L2, L1*L2)
    for y=1:1:L2; for x=1:1:L1
        uidx = get_index_from_posi(L1, L2, x, y)
        #A的两个
        rtidx = get_index_from_posi(L1, L2, x, y+1)
        h0[uidx, rtidx] = HOPPING_T
        h0[rtidx, uidx] = HOPPING_T
        #C的两个
        ltidx = get_index_from_posi(L1, L2, x+1, y)
        h0[uidx, ltidx] = HOPPING_T
        h0[ltidx, uidx] = HOPPING_T
        #
        lidx = get_index_from_posi(L1, L2, x-1, y+1)
        h0[uidx, lidx] = HOPPING_T
        h0[lidx, uidx] = HOPPING_T
    end; end
    return h0
end


"""
试探波函数
"""
function trial_wavefunction(h0, Nup)
    #return uhf_trial(h0, Nup, U)
    #h0d = copy(h0)
    ##nsite = size(h0d)[1]
    ##for ni = 1:1:nsite
    ##    h0d[ni, ni] += 0.01*(rand()-0.5)
    ##end
    ##println("close shell ")
    ##println(eigvals(h0d))
    ##println("------")
    #eigvs = eigvecs(h0d)
    #phi = eigvs[:, 1:Nup]
    #return copy(phi), copy(phi)
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
    #    "etgr.jld", "ints_L43_mu-0.346_U_3.0.jld", h0, L1*L2, Nup
    #)
    phiup, phidn = construct_frg_hamiltonian3(
        "ints_L43_mu-0.346_U_3.0.jld", h0, L1*L2, Nup
    )
    return phiup, phidn#copy(phi), copy(phi)
end


"""
增加相互作用
"""
function add_interactions(ham::HamConfig2, L1, L2, U, Δτ)
    for idx = 1:1:L1*L2
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
    szmat = zeros(Float64, L1*L2, L1*L2)
    for i=1:1:L1*L2; for j=1:1:L1*L2
        if i == j
            szmat[i, j] = eqgr.V[i, i, 1] + eqgr.V[i, i, 2] -
            eqgr.V[i, i, 1] * eqgr.V[i, i, 2] * 2
        else
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
        end
    end; end
    spcr_bin[end].V += wlk.weight * szmat
    #
    etgr_bin[end].V += wlk.weight * eqgr.V
    #println(eqgr.V[1, 3, 1], " ", wlk.weight)
end


"""
入口
"""
function main()
    global L1, L2, U, Nup, Δτ, HOPPING_T, NWALKERS
    global PCTRL_INTERVAL, STBLZ_INTERVAL
    h0 = hopping_matrix(L1, L2, HOPPING_T)
    println(eigvals(h0))
    phiup, phidn = trial_wavefunction(h0, Nup)
    #println(phiup)
    ham = HamConfig2(h0, Δτ, phiup, phidn)
    add_interactions(ham, L1, L2, U, Δτ)
    walkers = create_walkers(ham, Nup, NWALKERS, copy(phiup), copy(phidn))
    cpsim = CPSim2(ham, walkers, STBLZ_INTERVAL, PCTRL_INTERVAL)
    #
    igr = initialize_simulation!(cpsim, true)
    println("igr_up", [igr.V[idx, idx, 1] for idx=1:1:8])
    println("igr_dn", [igr.V[idx, idx, 2] for idx=1:1:8])
    println(cpsim.E_trial)
    #exit()
    relaxation_simulation(cpsim, 1000)
    E_trial_simulation!(cpsim, 10, 10)
    println(cpsim.E_trial)
    #
    whgt_bin = CPMeasure{:SCALE, Float64}[]
    engr_bin = CPMeasure{:SCALE, Float64}[]
    spcr_bin = CPMeasure{:MATRIX, Matrix{Float64}}[]
    etgr_bin = CPMeasure{:EQGR, Array{Float64, 3}}[]
    #
    meas_bin = 10
    meas_time = 20
    initial_meaurements!(cpsim, 20)
    for bidx = 1:1:meas_bin
        push!(whgt_bin, CPMeasure{:SCALE, Float64}("weight", 0.0))
        push!(engr_bin, CPMeasure{:SCALE, Float64}("energy", 0.0))
        push!(spcr_bin, CPMeasure{:MATRIX, Matrix{Float64}}("spin corr",
        zeros(Float64, L1*L2, L1*L2)))
        push!(etgr_bin, CPMeasure{:EQGR, Array{Float64, 3}}(
            "etgr", zeros(Float64, L1*L2, L1*L2, 2)
        ))
        for tidx = 1:1:meas_time
            premeas_simulation!(cpsim, 20)
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
    total_engr, error_engr = postprocess_measurements(engr_bin, whgt_bin)
    println("total energy = ", total_engr)
    println("error energy = ", error_engr)
    #
    total_spcr, error_spcr = postprocess_measurements(spcr_bin, whgt_bin)
    spcr_vec = zeros(L1, L2)
    spcr_err = zeros(L1, L2)
    for xidx=1:1:L1; for yidx=1:1:L2
        for st=1:1:L1*L2
            xst, yst = get_posi_from_index(L1, L2, st)
            est = get_index_from_posi(L1, L2, xst+xidx-1, yst+yidx-1)
            spcr_vec[xidx, yidx] += total_spcr[st, est]
            spcr_err[xidx, yidx] += error_spcr[st, est]
        end
    end; end
    for xidx=1:1:L1; for yidx=1:1:L2
        est = get_index_from_posi(L1, L2, xidx, yidx)
        println(est, " ", spcr_vec[xidx, yidx]/12, " ", spcr_err[xidx, yidx]/12)
    end; end
    avg_eqgr, err_eqgr = postprocess_measurements(etgr_bin, whgt_bin)
    totp = 0.
    for idx=1:1:L1*L2
        println(avg_eqgr[idx, idx, 1], " ", avg_eqgr[idx, idx, 1])
        totp += avg_eqgr[idx, idx, 1]
    end
    println(totp)
    #
    eqgr_mat = zeros(L1, L2, 2)
    for i=1:1:L1; for j=1:1:L2
        for idx=1:1:L1*L2
            x, y = get_posi_from_index(L1, L2, idx)
            eidx = get_index_from_posi(L1, L2, x+i-1, y+j-1)
            eqgr_mat[i, j, 1] += avg_eqgr[idx, eidx, 1]
            eqgr_mat[i, j, 2] += avg_eqgr[idx, eidx, 2]
        end
    end; end
    eqgr_mat = eqgr_mat / (L1*L2)
    save_eqgr = zeros(L1*L2, L1*L2)
    for idx=1:1:L1*L2
        x, y = get_posi_from_index(L1, L2, idx)
        for xidx=1:1:L1; for yidx=1:1:L2
            eidx = get_index_from_posi(L1, L2, x+xidx-1, y+yidx-1)
            save_eqgr[idx, eidx] = 0.5*(eqgr_mat[xidx, yidx, 1] + eqgr_mat[xidx, yidx, 2])
        end; end
    end
    println(eqgr_mat[3, 1, 1], " ", avg_eqgr[3, 1, 1], save_eqgr[4, 2])
    println(eqgr_mat[3, 1, 2], " ", avg_eqgr[3, 1, 2], save_eqgr[4, 2])
    println(eqgr_mat[1, 1, 1], " ", avg_eqgr[1, 1, 1], save_eqgr[2, 2])
    println(eqgr_mat[1, 1, 2], " ", avg_eqgr[1, 1, 2], save_eqgr[2, 2])
    save("etgr.jld", "etgr", save_eqgr)
end


main()

