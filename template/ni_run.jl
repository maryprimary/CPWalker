#=
ni基超导
=#

include("../../CPWalker/src/CPWalker.jl")
using ..CPWalker

using LatticeHamiltonian
using LinearAlgebra

L = 6
Nup = 36
Δτ = 0.01

NWALKERS = 100

U = 0.9498

ta0=-0.2289
ta1=-0.017
ta2=0.00
tb0=-0.3166
tb1=0.0706
tb2=-0.0
tab0=0.0205
tab1=0.0
tab2=0.0

mu1=0.127753
mu2=-0.268547

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
function hopping_matrix(L,ta0,tb0,tab0,ta1,tb1,ta2,tb2,tab2,mu1,mu2)
    h0 = zeros(2*L^2, 2*L^2)
    for xidx=1:1:L; for yidx=1:1:L
        uidx = get_index_from_posi(L, xidx, yidx)
        aidx = 2*uidx-1
        bidx = 2*uidx
        #  +1  0
        uridx = get_index_from_posi(L, xidx+1, yidx)
        aridx = 2*uridx-1
        bridx = 2*uridx
        h0[aidx, aridx] = ta0
        h0[aridx, aidx] = ta0
        h0[bidx, bridx] = tb0
        h0[bridx, bidx] = tb0
        h0[bidx, aidx] = tab0
        h0[aidx, bidx] = tab0        
        h0[bidx, aridx] = tab0
        h0[aridx, bidx] = tab0
        #  0  +1
        utidx = get_index_from_posi(L, xidx, yidx+1)
        atidx = 2*utidx-1
        btidx = 2*utidx
        h0[aidx, atidx] = ta0
        h0[atidx, aidx] = ta0
        h0[bidx, btidx] = tb0
        h0[btidx, bidx] = tb0
        h0[bidx, atidx] = tab0
        h0[atidx, bidx] = tab0        
        #  +1  +1
        uvidx = get_index_from_posi(L, xidx+1, yidx+1)
        avidx = 2*uvidx-1
        bvidx = 2*uvidx
        h0[bidx, avidx] = tab0
        h0[avidx, bidx] = tab0 
        h0[aidx, avidx] = ta1
        h0[avidx, aidx] = ta1       
        h0[bidx, bvidx] = tb1
        h0[bvidx, bidx] = tb1
        h0[aidx, bvidx] = tab2
        h0[bvidx, aidx] = tab2                      
        #  +1  -1
        uwidx = get_index_from_posi(L, xidx+1, yidx-1)
        awidx = 2*uwidx-1
        bwidx = 2*uwidx
        h0[aidx, awidx] = ta1
        h0[awidx, aidx] = ta1 
        h0[bidx, bwidx] = tb1
        h0[bwidx, bidx] = tb1                          
        #  +2  0
        ukidx = get_index_from_posi(L, xidx+2, yidx)
        akidx = 2*ukidx-1
        bkidx = 2*ukidx
        h0[aidx, akidx] = ta2
        h0[akidx, aidx] = ta2
        h0[bidx, bkidx] = tb2
        h0[bkidx, bidx] = tb2
        #  0  +2
        ugidx = get_index_from_posi(L, xidx, yidx+2)
        agidx = 2*ugidx-1
        bgidx = 2*ugidx
        h0[aidx, agidx] = ta2
        h0[agidx, aidx] = ta2
        h0[bidx, bgidx] = tb2
        h0[bgidx, bidx] = tb2
        #  -2  +1
        umidx = get_index_from_posi(L, xidx-2, yidx+1)
        amidx = 2*umidx-1
        bmidx = 2*umidx
        h0[aidx, bmidx] = tab2
        h0[bmidx, aidx] = tab2
        #  +1  -2
        uqidx = get_index_from_posi(L, xidx+1, yidx-2)
        aqidx = 2*uqidx-1
        bqidx = 2*uqidx
        h0[aidx, bqidx] = tab2
        h0[bqidx, aidx] = tab2        
        #  -2  -2
        unidx = get_index_from_posi(L, xidx-2, yidx-2)
        anidx = 2*unidx-1
        bnidx = 2*unidx
        h0[aidx, bnidx] = tab2
        h0[bnidx, aidx] = tab2        
        #       
        h0[aidx, aidx]  = mu1                 
        h0[bidx, bidx]  = mu2-U/2
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
    for idx = 2:2:2*L^2
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
    #
    #
    szmat = zeros(Float64, 2*L^2, 2*L^2)
    for i=1:1:2*L^2; for j=1:1:2*L^2
        if i == j
            #(niu - nid) (niu - nid)
            #niu + nid - 2nid niu
            szmat[i, j] = eqgr.V[i, i, 1] + eqgr.V[i, i, 2] -
            eqgr.V[i, i, 1] * eqgr.V[i, i, 2] * 2
        else
            # (niu - nid) (nju - njd)
            # niu nju
            szmat[i, j] = eqgr.V[i, i, 1] * eqgr.V[j, j, 1]
            szmat[i, j] = eqgr.V[i, j, 1] * (-eqgr.V[j, i, 1])
            # -n1u n2d
            szmat[i, j] = eqgr.V[i, i, 1] * eqgr.V[j, j, 2]
            # -n1d n2u
            szmat[i, j] = eqgr.V[i, i, 2] * eqgr.V[j, j, 1]
            # n1d n2d
            szmat[i, j] = eqgr.V[i, i, 2] * eqgr.V[j, j, 2]
            szmat[i, j] = eqgr.V[i, j, 2] * (-eqgr.V[j, i, 2])
        end
    end; end
    spcr_bin[end].V += wlk.weight * szmat
    #
    etgr_bin[end].V += wlk.weight * eqgr.V
end


"""
入口
"""
function main()
    global L, U, Nup, Δτ, ta0, tb0,tab0,ta1,tb1,ta2,tb2,tab2,mu1,mu2,NWALKERS
    global PCTRL_INTERVAL, STBLZ_INTERVAL
    h0 = hopping_matrix(L, ta0, tb0,tab0,ta1,tb1,ta2,tb2,tab2,mu1,mu2)
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
        zeros(Float64, 2*L^2, 2*L^2)))
        push!(etgr_bin, CPMeasure{:EQGR, Array{Float64, 3}}(
            "etgr", zeros(Float64, 2*L^2, 2*L^2, 2)
        ))
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
    error_engr = 0.
    for bidx = 1:1:meas_bin
        eng = engr_bin[bidx].V / whgt_bin[bidx].V
        error_engr += (eng - total_engr)^2
    end
    error_engr = sqrt(error_engr / (meas_bin-1))
    println("error energy = ", error_engr)
    #
    avg_eqgr = zeros(Float64, 2*L^2, 2*L^2, 2)
    for bidx = 1:1:meas_bin
        avg_eqgr += etgr_bin[bidx].V / whgt_bin[bidx].V
    end
    avg_eqgr = avg_eqgr / meas_bin
    err_eqgr = zeros(Float64, 2*L^2, 2*L^2, 2)
    for bidx = 1:1:meas_bin
        eqgrb = etgr_bin[bidx].V / whgt_bin[bidx].V
        err_eqgr += (eqgrb - avg_eqgr).^2
    end
    err_eqgr = sqrt.(err_eqgr / (meas_bin-1))
    for idx=1:1:2*L^2
        println(avg_eqgr[idx, idx, 1], " ", avg_eqgr[idx, idx, 2])
    end
end


main()

