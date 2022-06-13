#=
测试非厄米
=#

include("../../CPWalker/src/CPWalker.jl")
using ..CPWalker

using JLD
using MPI


using LatticeHamiltonian
using LatticeHamiltonian.SlaterDeterminant
using LinearAlgebra


"""
构造双子格子正方晶格
"""
function construc_ckb_lattice(L, nh)
    h0 = zeros(Float64, 8, 8)
    h0[1, 2] = -1
    h0[2, 1] = -1
    h0[2, 7] = -1
    h0[7, 2] = -1
    h0[7, 8] = -1
    h0[8, 7] = -1
    h0[1, 8] = -1
    h0[8, 1] = -1
    h0[3, 4] = -1
    h0[4, 3] = -1
    h0[5, 6] = -1
    h0[6, 5] = -1
    h0[6, 7] = -1-nh
    h0[7, 6] = -1+nh
    h0[7, 4] = -1-nh
    h0[4, 7] = -1+nh
    h0[5, 2] = -1-nh
    h0[2, 5] = -1+nh
    h0[2, 3] = -1-nh
    h0[3, 2] = -1+nh
    #for xidx=1:1:L; for yidx=1:1:L
    #    uidx = (yidx-1)*L + xidx
    #    aidx = uidx*2-1
    #    bidx = uidx*2
    #    xm1 = xidx==1 ? L : xidx-1
    #    ym1 = yidx==1 ? L : yidx-1
    #    #
    #    h0[aidx, bidx] = -1-nh
    #    h0[bidx, aidx] = -1+nh
    #    #左边
    #    uidx2 = (yidx-1)*L + xm1
    #    bidx2 = uidx2*2
    #    h0[aidx, bidx2] = -1-nh
    #    h0[bidx2, aidx] = -1+nh
    #    #左下
    #    uidx2 = (ym1-1)*L + xm1
    #    bidx2 = uidx2*2
    #    h0[aidx, bidx2] = -1+nh
    #    h0[bidx2, aidx] = -1-nh
    #    #下
    #    uidx2 = (ym1-1)*L + xidx
    #    bidx2 = uidx2*2
    #    h0[aidx, bidx2] = -1+nh
    #    h0[bidx2, aidx] = -1-nh
    #end; end
    return h0
end



"""
构造一个非厄米的链
"""
function construct_chain_lattice(L, Y, nh)
    h0 = zeros(Float64, L*Y, L*Y)
    #一种纯实本正值的构造方式
    #for idx = 1:2:(L-1)
    #    h0[idx, idx+1] = 1-nh
    #    h0[idx+1, idx] = 1+nh
    #end
    ##
    #for idx = 2:2:(L-1)
    #    h0[idx, idx+1] = 1+nh
    #    h0[idx+1, idx] = 1-nh
    #end
    #h0[L, 1] = 1+nh
    #h0[1, L] = 1-nh
    for idx = 1:1:(L-1); for yidx=1:1:Y
        stidx = (yidx-1)*L + idx
        h0[stidx, stidx+1] = -1+nh
        h0[stidx+1, stidx] = -1-nh
    end; end
    if Y < 2
        return h0
    end
    #竖直的
    for idx = 1:1:L; for yidx=1:1:Y
        stidx = (yidx-1)*L + idx
        yplus1 = yidx == Y ? 1 : yidx+1
        vtidx = (yplus1-1)*L + idx
        h0[stidx, vtidx] = -1
        h0[vtidx, stidx] = -1
    end; end
    return h0
end


"""
保存一个结果
"""
function save_bin(bidx, eng, hop, duo)
    jlh = jldopen("bdata"*bidx*".jld", "w")
    write(jlh, "eng", eng)
    write(jlh, "hop", hop)
    write(jlh, "duo", duo)
    close(jlh)
end


"""
运行
"""
function run(profilename, previousname)
    Lo = 8
    Lp = 1
    Δτ = 0.05
    np = 3
    nh = 0.1
    U = parse(Float64, ARGS[1])
    NWLK = 500
    CBT = 10.0
    #
    h0 = construct_chain_lattice(Lo, Lp, nh)
    println(h0)
    println(eigvals(h0))
    #
    ham3 = HamConfig3(h0, Δτ, np, np, CBT; denprf=previousname, mfu=U)
    #
    for idx=1:1:Lo*Lp
        push!(ham3.Mzints, (idx, 1, idx, 2))
        push!(ham3.Axflds, AuxiliaryField2("int"*string(idx), U, Δτ))
    end
    #
    wlks = Vector{HSWalker3}(undef, NWLK)
    wlk1 = HSWalker3("wlk1", ham3, copy(ham3.Φt[1].V), copy(ham3.Φt[2].V), 1.0)
    wlks[1] = wlk1
    println(wlk1)
    for idx=2:1:NWLK
        wlks[idx] = clone(wlk1, "wlk"*string(idx))
    end
    cpsim = CPSim3(ham3, wlks, 10, Int64(CBT*20), 40)#Int64(CBT*20))
    #println(cpsim)
    initialize_simulation!(cpsim)
    println(cpsim.E_trial)
    relaxation_simulation(cpsim, 5)
    println(wlk1)
    #
    #igr = get_eqgr_without_back(ham3, cpsim.walkers[1])
    #println("ngr ", igr)
    #for idx=1:1:Lo*Lp
    #    println(igr.V[idx, idx, 1])
    #end
    #
    E_trial_simulation!(cpsim)
    #
    println(cpsim.E_trial)
    #
    initial_meaurements!(cpsim)
    #
    wgt_bin::Vector{Float64} = []
    eng_bin::Vector{Float64} = []
    hop_bin::Vector{Float64} = []
    grn_bin::Vector{Matrix{Float64}} = []
    duo_bin::Vector{Vector{Float64}} = []
    #
    binnum = 10
    meanum = 300
    #观测格林函数
    for bidx = 1:1:binnum
        push!(wgt_bin, 0.0)
        push!(eng_bin, 0.0)
        push!(hop_bin, 0.0)
        push!(grn_bin, zeros(Lo*Lp, Lo*Lp))
        push!(duo_bin, zeros(Lo*Lp))
        #update_backwalkers!(cpsim, 1)
        for midx = 1:1:meanum
            #premeas_simulation!(cpsim, true)
            premeas_simulation!(cpsim)
            for widx = 1:1:NWLK
                wgt_bin[end] = wgt_bin[end] + cpsim.walkers[widx].weight
                grn_bin[end] = grn_bin[end] + (cpsim.eqgrs[widx].V[:, :, 1] + cpsim.eqgrs[widx].V[:, :, 2])*
                0.5 * cpsim.walkers[widx].weight
                duob = zeros(Lo*Lp)
                for duoidx=1:1:Lo*Lp
                    duob[duoidx] = cpsim.eqgrs[widx].V[duoidx, duoidx, 1] * cpsim.eqgrs[widx].V[duoidx, duoidx, 2]
                end
                duo_bin[end] = duo_bin[end] + duob * cpsim.walkers[widx].weight
                hop, eng = cal_energy(cpsim.eqgrs[widx], cpsim.hamiltonian)
                eng_bin[end] = eng_bin[end] + eng * cpsim.walkers[widx].weight
                hop_bin[end] = hop_bin[end] + hop * cpsim.walkers[widx].weight
            end
            postmeas_simulation(cpsim)
        end
        grn_bin[end] = grn_bin[end] / wgt_bin[end]
        duo_bin[end] = duo_bin[end] / wgt_bin[end]
        eng_bin[end] = eng_bin[end] / wgt_bin[end]
        hop_bin[end] = hop_bin[end] / wgt_bin[end]
        println(bidx)
        #
        comm = MPI.COMM_WORLD
        rank = MPI.Comm_rank(comm)
        bstr = string(rank)*string(bidx)
        save_bin(bstr, eng_bin[end], hop_bin[end], duo_bin[end])
    end
    #
    println("wgt")
    for idx = 1:1:binnum
        println(wgt_bin[idx])
    end
    #
    mean_grn = zeros(Lo*Lp, Lo*Lp)
    for bidx = 1:1:binnum
        mean_grn += grn_bin[bidx]
    end
    mean_grn /= binnum
    errn_grn = zeros(Lo*Lp, Lo*Lp)
    for bidx = 1:1:binnum
        errn_grn += (grn_bin[bidx] - mean_grn).^2
    end
    errn_grn /= (binnum-1)
    errn_grn = sqrt.(errn_grn)
    println("ngr1 ")
    for idx=1:1:Lo*Lp
        println(mean_grn[idx, idx], " ", errn_grn[idx, idx])
    end
    println("ngr2 ")
    for idx=1:1:(Lo*Lp-1)
        println(mean_grn[idx, idx+1], " ", errn_grn[idx, idx+1], " ", mean_grn[idx+1, idx])
    end
    #
    #
    mean_duo = zeros(Lo*Lp)
    for bidx = 1:1:binnum
        mean_duo += duo_bin[bidx]
    end
    mean_duo /= binnum
    errn_duo = zeros(Lo*Lp)
    for bidx = 1:1:binnum
        errn_duo += (duo_bin[bidx] - mean_duo).^2
    end
    errn_duo /= (binnum-1)
    errn_duo = sqrt.(errn_duo)
    println("duo")
    for idx=1:1:Lo*Lp
        println(mean_duo[idx], " ", errn_duo[idx])
    end
    #
    mean_eng = sum(eng_bin) / binnum
    errnseng = eng_bin .- mean_eng
    errn_eng = sum(errnseng.^2) / (binnum-1)
    errn_eng = sqrt(errn_eng)
    #
    mean_hop = sum(hop_bin) / binnum
    errnshop = hop_bin .- mean_hop
    errn_hop = sum(errnshop.^2) / (binnum-1)
    errn_hop = sqrt(errn_hop)
    println("eng ", mean_eng, " ", errn_eng)
    println("hop ", mean_hop, " ", errn_hop)
    #
    #保存参数
    save_density_profile(profilename, mean_grn, mean_duo)
end


MPI.Init()
run("iter0", missing)
#run("iter1", "iter0")
#run("iter2", "iter1")
#run("iter3", "iter2")
#run("iter4", "iter3")

#h0 = construc_ckb_lattice(2, 0.3)
#println(eigvals(h0))
