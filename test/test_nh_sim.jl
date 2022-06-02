#=
测试非厄米
=#

include("../../CPWalker/src/CPWalker.jl")
using ..CPWalker


using LatticeHamiltonian
using LatticeHamiltonian.SlaterDeterminant
using LinearAlgebra


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
运行
"""
function run(profilename, previousname)
    L = 8
    Δτ = 0.05
    np = 3
    nh = 0.1
    U = 4.0
    NWLK = 500
    CBT = 10.0
    #
    h0 = construct_chain_lattice(L, 1, nh)
    println(h0)
    #
    ham3 = HamConfig3(h0, Δτ, np, np, CBT; denprf=previousname, mfu=U)
    #
    for idx=1:1:L
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
    cpsim = CPSim3(ham3, wlks, 10, Int64(CBT*20))
    #println(cpsim)
    initialize_simulation!(cpsim)
    println(cpsim.E_trial)
    relaxation_simulation(cpsim, 5)
    println(wlk1)
    #
    #igr = get_eqgr_without_back(ham3, cpsim.walkers[1])
    #println("ngr ", igr)
    #for idx=1:1:L
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
    grn_bin::Vector{Array{Float64, 3}} = []
    duo_bin::Vector{Vector{Float64}} = []
    #
    binnum = 10
    meanum = 10
    #观测格林函数
    for bidx = 1:1:binnum
        push!(wgt_bin, 0.0)
        push!(grn_bin, zeros(L, L, 2))
        push!(duo_bin, zeros(L))
        #update_backwalkers!(cpsim, 1)
        for midx = 1:1:meanum
            premeas_simulation!(cpsim)
            for widx = 1:1:NWLK
                wgt_bin[end] = wgt_bin[end] + cpsim.walkers[widx].weight
                grn_bin[end] = grn_bin[end] + cpsim.eqgrs[widx].V * cpsim.walkers[widx].weight
            end
            postmeas_simulation(cpsim)
        end
        grn_bin[end] = grn_bin[end] / wgt_bin[end]
    end
    #
    mean_grn = zeros(L, L, 2)
    for bidx = 1:1:binnum
        mean_grn += grn_bin[bidx]
    end
    mean_grn /= binnum
    errn_grn = zeros(L, L, 2)
    for bidx = 1:1:binnum
        errn_grn += (grn_bin[bidx] - mean_grn).^2
    end
    errn_grn /= (binnum-1)
    errn_grn = sqrt.(errn_grn)
    println("ngr ")
    for idx=1:1:L
        println(0.5*(mean_grn[idx, idx, 1]+mean_grn[idx, idx, 2]),
        " ", 0.5*(errn_grn[idx, idx, 1] + errn_grn[idx, idx, 2]))
    end
    #保存参数
    save_density_profile(profilename, mean_grn)
end


run("iter0", missing)
#run("iter1", "iter0")
