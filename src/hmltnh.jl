#=
哈密顿量
=#

using JLD


"""
hopping项非厄米的哈密顿
"""
struct HamConfig3
    Hnh :: DenseFermiOpt{Float64}
    HL :: DenseFermiOpt{Float64}
    dτ :: Float64
    Mzints :: Vector{NTuple{4, Int64}}
    Axflds :: Vector{AuxiliaryField2}
    npart :: Tuple{Int64, Int64}
    Φt :: Tuple{Slater{Float64}, Slater{Float64}}
    ΦtT :: Tuple{Slater{Float64}, Slater{Float64}}
    exp_halfdτHn :: DenseFermiOpt{Float64}
    exp_dτHn :: DenseFermiOpt{Float64}
    exp_halfdτHnhd :: DenseFermiOpt{Float64}
    exp_dτHnhd :: DenseFermiOpt{Float64}
    SSd :: DenseFermiOpt{Float64}
    iSSd :: DenseFermiOpt{Float64}
    ebhl :: DenseFermiOpt{Float64}
end



"""
获得S,本正值，格林函数
"""
function get_ham_info(hnh, np, cutoffbeta; verbose=false)
    eig = eigen(hnh)
    imeval = imag(eig.values)
    reeval = real(eig.values)
    if verbose
        println("E level ", reeval)
    end
    #只要所有的本正值都是实数
    #就能保证 S e^(-bH) S^+ 厄米
    @assert all(isapprox.(imeval, 0., atol=1e-10))
    Smat = eig.vectors
    @assert all(isapprox.(imag(Smat), 0., atol=1e-10))
    Smat = real(Smat)
    #
    cbt = cutoffbeta
    reemin = minimum(reeval)
    denmat2 = zeros(length(reeval), length(reeval))
    for idx=1:1:length(reeval)
        denmat2[idx, idx] = exp(-cbt*(reeval[idx] - reemin))
    end
    ebhlmat = Smat * denmat2 * adjoint(Smat)
    #println(denmat2)
    #println(ebhlmat)
    if verbose
        println("shell ", eigvals(ebhlmat))
    end
    #@assert issymmetric(ebhlmat)
    ebhlmat = 0.5*(ebhlmat + adjoint(ebhlmat))
    phitrial = eigvecs(-ebhlmat)
    #
    phiup = copy(phitrial[:, 1:np])
    #
    adjv1 = adjoint(phiup)
    invovlp1 = inv(adjv1 * phiup)
    eqgr = phiup * invovlp1 * adjv1
    return ebhlmat, Smat, phiup, eqgr
end


"""
获取有效的哈密顿
"""
function RHFRecur(h0, cutoffbeta, np, mfu)
    ssize = size(h0)[1]
    #
    ebhlmat, Smat, phiup, eqgr = get_ham_info(h0, np, cutoffbeta; verbose=true)
    #
    #
    for iidx = 1:1:100
        hiter = copy(h0)
        for sidx = 1:1:ssize
            hiter[sidx, sidx] += mfu * eqgr[sidx, sidx]
        end
        ebhlmat2, Smat2, phiup2, eqgr2 = get_ham_info(hiter, np, cutoffbeta)
        println("c ", iidx, " ", [eqgr2[idx, idx] for idx = 1:1:ssize])
        #如果收敛，所有结果利用收敛后的
        if all(isapprox.(eqgr, eqgr2, atol=1e-5))
            ebhlmat = ebhlmat2
            Smat = Smat2
            phiup = phiup2
            eqgr = eqgr2
            @info "converge ", iidx
            break
        end
        #格林函数进行迭代
        eqgr = 0.5*(eqgr2+eqgr)
        if iidx == 100
            @info "not converge"
        end
    end
    return ebhlmat, Smat, phiup, eqgr
end


"""
获取能生成最接近的哈密顿
"""
function density_recur(h0, cutoffbeta, np, mfu, denprf)
    #
    ebhlmat, Smat, phiup, eqgr = get_ham_info(h0, np, cutoffbeta)
    #
    denarr, duoc = load(denprf*".jld", "density", "duobocc")
    @info "den read ", denarr, duoc
    mindis = 1e10
    minueff = (0.0, 0.0)
    ueffl = 0.01
    ueffr = 0.01
    ssize = size(h0)[1]
    while ueffl < mfu*2; 
        ueffr = 0.01
        while ueffr < mfu*2
            heff = copy(h0)
            for sidx = 1:1:ssize
                if mod(sidx, 8) == 1
                    heff[sidx, sidx] += ueffl * denarr[sidx]
                elseif mod(sidx, 8) == 0
                    heff[sidx, sidx] += ueffr * denarr[sidx]
                else
                    heff[sidx, sidx] += mfu * denarr[sidx]
                end      
            end
            ebhlmat2, Smat2, phiup2, eqgr2 = get_ham_info(heff, np, cutoffbeta)
            denarr2 = [eqgr2[idx, idx] for idx = 1:1:ssize]
            dis2 = sum(abs.(denarr2 - denarr))
            #dis2 = sqrt(dis2)
            #dis2 = dot(denarr, denarr2)
            if dis2 < mindis
                mindis = dis2
                minueff = (ueffl, ueffr)
                ebhlmat = ebhlmat2
                Smat = Smat2
                phiup = phiup2
                eqgr = eqgr2
            end
            ueffr += 0.01
        end; 
        ueffl += 0.01 
    end
    denarrn = [eqgr[idx, idx] for idx = 1:1:ssize]
    @info "effa ", denarrn, minueff
    #exit()
    return ebhlmat, Smat, phiup, eqgr
end



"""
创建一个非厄米的哈密顿量
"""
function HamConfig3(h0::Matrix{Float64}, dtau::Float64,
    npart1::Int64, npart2::Int64, cutoffbeta::Float64; denprf=missing, mfu=0.)
    h0op = DenseFermiOpt("H_0", h0)
    #
    if ismissing(denprf)
        ebhlmat, Smat, phiup, eqgr = RHFRecur(h0, cutoffbeta, npart1, mfu)
    else
        ebhlmat, Smat, phiup, eqgr = density_recur(h0, cutoffbeta, npart1, mfu, denprf)
    end
    #
    ssd = Smat * adjoint(Smat)
    issd = inv(ssd)
    #println(log(ssd)*adjoint(h0) - adjoint(h0)*log(ssd))
    #exit()
    #println("Smat overlap")
    #println(inv(adjoint(Smat) * Smat))
    #println(Smat)
    Sop = DenseFermiOpt("SS^+", ssd)
    iSop = DenseFermiOpt("(SS^+)^-1", issd)
    #iSmat = inv(Smat)
    #println(iSmat*h0*Smat)
    #cutoff beta
    #cbt = 10# / maximum(abs.(eig.values))
    #denmat = exp(-cbt*evalmat)
    #ebhlmat = Hermitian(Smat * denmat * adjoint(Smat))
    ##println("e-bhl ", ebhlmat)
    #ebhl = DenseFermiOpt("exp(-betaHL", Matrix{Float64}(real(ebhlmat)))
    #hl = - (1/cbt) * log(ebhlmat)
    #hl = real(hl)
    #println(hl)
    #exit()
    #println(eigvecs(hl))
    #
    ebhl = DenseFermiOpt("exp(-betaHL", ebhlmat)
    #println(denmat2)
    #hl2 = -(1/cbt)*(
    #    log(Smat * denmat2 *adjoint(Smat)) - Diagonal(ones(length(reeval))*reemin*cbt)
    #    )
    #hl2 = real(hl2)
    #println(hl2)
    #println(eigvecs(hl2))
    hl3 = -(1/cutoffbeta)*log(ebhlmat)
    #println(hl3)
    hl3 = real(hl3)
    #println(eigvecs(hl3))
    #println(eigvecs(-ebhlmat))
    #return
    ##println("hl", hl)
    hl = 0.5*(hl3 + adjoint(hl3))
    hlop = DenseFermiOpt("H_L", hl)
    #
    #phiup = copy(phitrial[:, 1:npart1])
    phidn = copy(phiup)
    Φt1 = Slater{Float64}("Φ_T1", phiup)
    Φt2 = Slater{Float64}("Φ_T2", phidn)
    #
    exphalfhnhd = exp(-0.5*dtau*adjoint(h0))
    exphnhd = exphalfhnhd * exphalfhnhd
    exphalfhnhop = DenseFermiOpt("exp(-0.5*dτHL)", exphalfhnhd)
    exphnhop = DenseFermiOpt("exp(-dτHL)", exphnhd)
    #
    exphalfhn = exp(-0.5*dtau*h0)
    exphn = exphalfhn * exphalfhn
    exphalfhnop = DenseFermiOpt("exp(-0.5*dτHL)", exphalfhn)
    exphnop = DenseFermiOpt("exp(-dτHL)", exphn)
    return HamConfig3(
        h0op, hlop, dtau, NTuple{4, Int64}[], AuxiliaryField2[],
        (npart1, npart2), (Φt1, Φt2), (adjoint(Φt1), adjoint(Φt2)),
        exphalfhnop, exphnop,
        exphalfhnhop, exphnhop, Sop, iSop, ebhl
    )
end


"""
保存参数
"""
function save_density_profile(fname, eqgr, duop)
    jlf = jldopen(fname*".jld", "w")
    meaneqgr = eqgr
    ssize = size(eqgr)[1]
    write(jlf, "density", [meaneqgr[idx, idx] for idx = 1:1:ssize])
    write(jlf, "duobocc", duop)
    close(jlf)
end

