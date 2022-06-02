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
run
"""
function run()
    L = 8
    Δτ = 0.1
    np = 3
    nh = 0.4
    U = 0.0
    #
    h0 = construct_chain_lattice(L, 1, nh)
    #
    println("close ", eigvals(h0))

    #一定要注意，HL的close shell问题
    ham3 = HamConfig3(h0, Δτ, np, np, 1.0)

    println(h0)
    #return
    
    for idx=1:1:L
        push!(ham3.Mzints, (idx, 1, idx, 2))
        push!(ham3.Axflds, AuxiliaryField2("int"*string(idx), U, Δτ))
    end
    #wlk =  copy(ham3.Φt[1].V)
    wlk = HSWalker3("wlk1", ham3, copy(ham3.Φt[1].V), copy(ham3.Φt[2].V), 1.0)
    println("wlkini ", wlk.Φ[1].V)
    println("wlkini overlap", adjoint(wlk.Φ[1].V)*wlk.Φ[1].V)
    #
    igr = get_eqgr_without_back(ham3, wlk)
    #
    for idx = 1:1:10
        start = (idx-1)*10+1
        ends = idx*10
        #println("wlk overlap a", adjoint(wlk.Φ[1].V)*wlk.Φ[1].V)
        step_slice!(wlk, ham3, Vector(start:1:ends); E_trial=-5.9871336883973574)
        #-6.846747818353414)
        #println("wlk overlap b", adjoint(wlk.Φ[1].V)*wlk.Φ[1].V)
        #
        if idx != 10
            decorate_stablize!(wlk, ham3)
        end
    end
    #println("ebhl ", ham3.ebhl.V)
    #println(ham3.S.V * adjoint(ham3.S.V) * ham3.exp_dτHnhd.V^100)
    #iterwlk = ham3.S.V * adjoint(ham3.S.V) * wlk.Φ[1].V
    multiply_left!(ham3.SSd.V, wlk.Φ[1])
    multiply_left!(ham3.SSd.V, wlk.Φ[2])
    update_overlap!(wlk, ham3, true)
    stablize!(wlk, ham3)
    println("iter1 ", wlk.Φ[1])
    println("iter2 ", wlk.Φ[2])
    igr = get_eqgr_without_back(ham3, wlk)
    println("ngr ", igr)
    for idx=1:1:L
        println(igr.V[idx, idx, 1])
    end
    #println(ham3.ebhl.V * ham3.Φt[1].V)
    #println(ham3.S.V * adjoint(ham3.S.V) * ham3.exp_dτHnhd.V^100 * ham3.Φt[1].V)
    #
    #wwdi = wlk.Φ[1].V*adjoint(wlk.Φ[1].V)
    #println(inv(wlk.Φ[1].V))
    #println(det(wwdi))
    #wwdi = inv(wwdi)
    #ssd = sqrt(wwdi)
    #println(ham3.S.V * adjoint(ham3.S.V))
    #println("ssd ", ssd)
    ###########
    #println(adjoint(ham3.Φt[1].V)*ham3.Φt[1].V)
    #println(adjoint(wlk.Φ[1].V) * wlk.Φ[1].V)
    #println(adjoint(wlk.Φ[2].V) * wlk.Φ[2].V)
    
    #
    #println(ham3.S.V * adjoint(ham3.S.V) * wlk)
    #
    #println(ham3.S.V * adjoint(ham3.S.V) * ham3.exp_dτHnhd.V^100)    
end


run()


function run2()
    L = 8
    Δτ = 0.1
    np = 4
    nh = 0.5
    U = 0.0
    #
    h0 = construct_chain_lattice(L, 1, nh)
    #
    cbt = 10
    eig = eigen(h0)
    evalmat = Diagonal(eig.values)
    Smat = eig.vectors
    denmat = exp(-cbt*evalmat)
    ebhlmat = Smat * denmat * adjoint(Smat)
    phitrial = eigvecs(-ebhlmat)
    #
    #phiup1 = copy(phitrial[:, 1])
    ##从s中去掉phiup1
    #for idx=1:1:L
    #    iprd = dot(phiup1, Smat[:, idx])
    #    Smat[:, idx] = Smat[:, idx] - iprd*phiup1
    #    tmp = 1.0/norm(Smat[:, idx])
    #    Smat[:, idx] = tmp*Smat[:, idx]
    #end
    #denmat = exp(-cbt*evalmat)
    #ebhlmat = Smat * denmat * adjoint(Smat)
    #phitrial = eigvecs(-ebhlmat)
    #phiup2 = copy(phitrial[:, 1])
    ##
    #phiup = zeros(L, np)
    #phiup[:, 1] = phiup1
    #phiup[:, 2] = phiup2
    phiup = copy(phitrial[:, 1:np])
    #
    adjv1 = adjoint(phiup)
    invovlp1 = inv(adjv1 * phiup)
    gf = phiup * invovlp1 * adjv1
    for idx=1:1:L
        println(gf[idx, idx])
    end
end


#run2()
