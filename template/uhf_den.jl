#=
在密度上做uhf
=#

using JLD
using LinearAlgebra


function uhf_trial(h0, Nup, U)
    #
    h0d = copy(h0)
    nsite = size(h0d)[1]
    for ni = 1:1:nsite
        h0d[ni, ni] += 0.01*(rand()-0.5)
    end
    println("close shell ")
    println(eigvals(h0d))
    println("------")
    #
    eigvs = eigvecs(h0d)
    phi = eigvs[:, 1:Nup]
    #
    phiup = phi
    phidn = phi
    #
    for iter=1:1:100
        h0up = copy(h0d)
        h0dn = copy(h0d)
        grup = phiup * adjoint(phiup)
        grdn = phidn * adjoint(phidn)
        #up
        for ni=1:1:nsite
            h0up[ni, ni] += U*grdn[ni, ni]
        end
        #dn
        for ni=1:1:nsite
            h0dn[ni, ni] += U*grup[ni, ni]
        end
        #
        phiup = eigvecs(h0up)[:, 1:Nup]
        phidn = eigvecs(h0dn)[:, 1:Nup]
    end
    return phiup, phidn
end



"""
构造frg的相互作用
"""
function construct_frg_hamiltonian(fname, h0, L, N)
    ints = load(fname)
    ints = ints["ints"]
    #println(ints)
    #初始的试探格林函数
    grup = zeros(Float64, L, L)
    grdn = zeros(Float64, L, L)
    for idx=1:1:L
        grup[idx, idx] = N/L
        grdn[idx, idx] = N/L
    end
    #哈密顿量
    hamup = zeros(Float64, L, L)
    hamdn = zeros(Float64, L, L)
    for inta in ints
        # c+_up c+_up c_up c_up
        hamup[inta[1], inta[3]] -= inta[5] * grup[inta[2], inta[4]]
        hamup[inta[2], inta[4]] -= inta[5] * grup[inta[1], inta[3]]
        hamup[inta[1], inta[4]] += inta[5] * grup[inta[2], inta[3]]
        hamup[inta[2], inta[3]] += inta[5] * grup[inta[1], inta[4]]
        # c+_dn c+_dn c_dn c_dn
        hamdn[inta[1], inta[3]] -= inta[5] * grdn[inta[2], inta[4]]
        hamdn[inta[2], inta[4]] -= inta[5] * grdn[inta[1], inta[3]]
        hamdn[inta[1], inta[4]] += inta[5] * grdn[inta[2], inta[3]]
        hamdn[inta[2], inta[3]] += inta[5] * grdn[inta[1], inta[4]]
        # c+_up c+_dn c_dn c_up
        hamup[inta[1], inta[4]] += inta[5] * grdn[inta[2], inta[3]]
        hamdn[inta[2], inta[3]] += inta[5] * grup[inta[1], inta[4]]
        # c+_dn c+_up c_up c_dn
        hamdn[inta[1], inta[4]] += inta[5] * grup[inta[2], inta[3]]
        hamup[inta[2], inta[3]] += inta[5] * grdn[inta[1], inta[4]]
    end
    hamup = h0 + 0.5hamup
    hamdn = h0 + 0.5hamdn
    for idx=1:1:12
        println(hamup[idx, idx], " ", hamup[idx, idx])
    end
    println(eigvals(hamup))
    phiup = eigvecs(hamup)[:, 1:N]
    phidn = eigvecs(hamdn)[:, 1:N]
    return phiup, phidn
end



"""
构造frg的相互作用
"""
function construct_frg_hamiltonian2(etgr, fname, h0, L, N)
    ints = load(fname)
    ints = ints["ints"]
    #println(ints)
    #初始的试探格林函数
    grs = load(etgr)
    grs = grs["etgr"]
    grup = grs[:, :, 1]
    grdn = grs[:, :, 1]
    #哈密顿量
    hamup = zeros(Float64, L, L)
    hamdn = zeros(Float64, L, L)
    for inta in ints
        # c+_up c+_up c_up c_up
        hamup[inta[1], inta[3]] -= inta[5] * grup[inta[2], inta[4]]
        hamup[inta[2], inta[4]] -= inta[5] * grup[inta[1], inta[3]]
        hamup[inta[1], inta[4]] += inta[5] * grup[inta[2], inta[3]]
        hamup[inta[2], inta[3]] += inta[5] * grup[inta[1], inta[4]]
        # c+_dn c+_dn c_dn c_dn
        hamdn[inta[1], inta[3]] -= inta[5] * grdn[inta[2], inta[4]]
        hamdn[inta[2], inta[4]] -= inta[5] * grdn[inta[1], inta[3]]
        hamdn[inta[1], inta[4]] += inta[5] * grdn[inta[2], inta[3]]
        hamdn[inta[2], inta[3]] += inta[5] * grdn[inta[1], inta[4]]
        # c+_up c+_dn c_dn c_up
        hamup[inta[1], inta[4]] += inta[5] * grdn[inta[2], inta[3]]
        hamdn[inta[2], inta[3]] += inta[5] * grup[inta[1], inta[4]]
        # c+_dn c+_up c_up c_dn
        hamdn[inta[1], inta[4]] += inta[5] * grup[inta[2], inta[3]]
        hamup[inta[2], inta[3]] += inta[5] * grdn[inta[1], inta[4]]
    end
    hamup = h0 + 0.5hamup
    hamdn = h0 + 0.5hamdn
    for idx=1:1:12
        println(hamup[idx, idx], " ", hamup[idx, idx])
    end
    println(eigvals(hamup))
    phiup = eigvecs(hamup)[:, 1:N]
    phidn = eigvecs(hamdn)[:, 1:N]
    return real(phiup), real(phidn)
end



"""
构造frg的相互作用
"""
function construct_frg_hamiltonian3(fname, h0, L, N)
    ints = load(fname)
    ints = ints["ints"]
    #println(ints)
    #初始的试探格林函数
    eigvs = eigvecs(h0)
    phi = eigvs[:, 1:N]
    #
    phiup = phi
    phidn = phi
    grup = phiup * adjoint(phiup)
    grdn = phidn * adjoint(phidn)
    #哈密顿量
    hamup = zeros(Float64, L, L)
    hamdn = zeros(Float64, L, L)
    for inta in ints
        # c+_up c+_up c_up c_up
        hamup[inta[1], inta[3]] -= inta[5] * grup[inta[2], inta[4]]
        hamup[inta[2], inta[4]] -= inta[5] * grup[inta[1], inta[3]]
        hamup[inta[1], inta[4]] += inta[5] * grup[inta[2], inta[3]]
        hamup[inta[2], inta[3]] += inta[5] * grup[inta[1], inta[4]]
        # c+_dn c+_dn c_dn c_dn
        hamdn[inta[1], inta[3]] -= inta[5] * grdn[inta[2], inta[4]]
        hamdn[inta[2], inta[4]] -= inta[5] * grdn[inta[1], inta[3]]
        hamdn[inta[1], inta[4]] += inta[5] * grdn[inta[2], inta[3]]
        hamdn[inta[2], inta[3]] += inta[5] * grdn[inta[1], inta[4]]
        # c+_up c+_dn c_dn c_up
        hamup[inta[1], inta[4]] += inta[5] * grdn[inta[2], inta[3]]
        hamdn[inta[2], inta[3]] += inta[5] * grup[inta[1], inta[4]]
        # c+_dn c+_up c_up c_dn
        hamdn[inta[1], inta[4]] += inta[5] * grup[inta[2], inta[3]]
        hamup[inta[2], inta[3]] += inta[5] * grdn[inta[1], inta[4]]
    end
    hamup = h0 + 0.5hamup
    hamdn = h0 + 0.5hamdn
    #for idx=1:1:12
    #    println(hamup[idx, idx], " ", hamup[idx, idx])
    #end
    println(hamup)
    println(eigvals(hamup))
    phiup = eigvecs(hamup)[:, 1:N]
    phidn = eigvecs(hamdn)[:, 1:N]
    return real(phiup), real(phidn)
end

