#=
非厄米的walker
=#

mutable struct HSWalker3
    Φ :: Tuple{Slater{Float64}, Slater{Float64}}
    overlap :: Float64
    weight :: Float64
    ovlpinv :: Union{Missing, Tuple{Slater{Float64}, Slater{Float64}}}
    hshist :: Union{Missing, Matrix{Int64}}#第一个编号是相互作用，第二个是时间
    Φcache :: Union{Missing, Tuple{Slater{Float64}, Slater{Float64}}}
    #用来存储反传时的slater
end



"""
计算和试探波函数的交叠
"""
function update_overlap!(wlk::HSWalker3, ham::HamConfig3, weight_update::Bool)
    #第一个
    #Np * Np  
    ovlp_inv1 = ham.ΦtT[1] * wlk.Φ[1]
    #ovlp
    detovlp1 = det(ovlp_inv1)
    if detovlp1 < 1e-5
        wlk.weight = 0.
        wlk.overlap = 0.
        return
    end
    invovlp1 = inv(ovlp_inv1)
    #第二个
    ovlp_inv2 = ham.ΦtT[2] * wlk.Φ[2]
    #ovlp
    detovlp2 = det(ovlp_inv2)
    if detovlp2 < 1e-5
        wlk.weight = 0.
        wlk.overlap = 0.
        return
    end
    #println(detovlp2)
    invovlp2 = inv(ovlp_inv2)
    #
    ovlpnew = detovlp1 * detovlp2
    #这个时候就停止了
    if weight_update
        if ovlpnew < 1e-8#innerprod(ovlpnew, wlk.overlap) < 1e-8
            wlk.weight = 0.
        else
            wlk.weight = ovlpnew / wlk.overlap * wlk.weight
        end
    end
    if !isfinite(wlk.weight)
        wlk.weight = 0.
        #println(ovlpnew, " ", wlk.overlap)
        #throw(error("b"))
    end
    wlk.overlap = ovlpnew
    wlk.ovlpinv = (invovlp1, invovlp2)
end


"""
构造walker
"""
function HSWalker3(name::String, ham::HamConfig3,
    phi1::Matrix{Float64}, phi2::Matrix{Float64},
    wgt::Float64)
    sla1 = Slater{Float64}(name*"_1", phi1)
    sla2 = Slater{Float64}(name*"_1", phi2)
    wlk = HSWalker3((sla1, sla2), 1.0, wgt, missing, missing, missing)
    update_overlap!(wlk, ham, true)
    return wlk
end


"""
先做用在正交归一
"""
function decorate_stablize!(wlk::HSWalker3, ham::HamConfig3; checkovlp=true)
    #@info wlk.weight
    if wlk.weight < 1e-8
        return
    end
    slasize = size(wlk.Φ[1].V)
    #作用S S^{dagger}
    multiply_left!(ham.SSd.V, wlk.Φ[1])
    multiply_left!(ham.SSd.V, wlk.Φ[2])
    #对结果正交归一
    for idx in 1:1:slasize[2]
        tmp = 1.0/norm(wlk.Φ[1].V[:, idx])
        wlk.Φ[1].V[:, idx] = tmp*wlk.Φ[1].V[:, idx]
        #对之后的所有轨道减去这个部分
        for idx2 in (idx+1):1:slasize[2]
            iprd = dot(wlk.Φ[1].V[:, idx], wlk.Φ[1].V[:, idx2])
            wlk.Φ[1].V[:, idx2] = wlk.Φ[1].V[:, idx2] - iprd*wlk.Φ[1].V[:, idx]
        end
    end
    #第二个
    slasize = size(wlk.Φ[2].V)
    #对每一个轨道进行正交归一
    for idx in 1:1:slasize[2]
        tmp = 1.0/norm(wlk.Φ[2].V[:, idx])
        wlk.Φ[2].V[:, idx] = tmp*wlk.Φ[2].V[:, idx]
        #对之后的所有轨道减去这个部分
        for idx2 in (idx+1):1:slasize[2]
            iprd = dot(wlk.Φ[2].V[:, idx], wlk.Φ[2].V[:, idx2])
            wlk.Φ[2].V[:, idx2] = wlk.Φ[2].V[:, idx2] - iprd*wlk.Φ[2].V[:, idx]
        end
    end
    #作用回inv(S S^{dagger})
    #@info adjoint(wlk.Φ[1].V) * wlk.Φ[1].V
    #@info adjoint(wlk.Φ[2].V) * wlk.Φ[2].V
    multiply_left!(ham.iSSd.V, wlk.Φ[1])
    multiply_left!(ham.iSSd.V, wlk.Φ[2])
    #更新overlap
    update_overlap!(wlk, ham, false)
end



"""
给walker重新归一化
"""
function noorthstablize!(wlk::HSWalker3, ham::HamConfig3; checkovlp=true)
    if wlk.weight < 1e-8
        return
    end
    tempwlkphi1 = copy(wlk.Φ[1].V)
    tempwlkphi2 = copy(wlk.Φ[2].V)
    #第一个flavour
    slasize = size(wlk.Φ[1].V)
    rescale = 1.0
    #对每一个轨道进行正交归一
    for idx in 1:1:slasize[2]
        tmp = 1.0/norm(wlk.Φ[1].V[:, idx])
        rescale *= tmp
        wlk.Φ[1].V[:, idx] = tmp*wlk.Φ[1].V[:, idx]
    end
    testoverlap = wlk.overlap * rescale
    #第二个
    slasize = size(wlk.Φ[2].V)
    rescale = 1.0
    #对每一个轨道进行正交归一
    for idx in 1:1:slasize[2]
        tmp = 1.0/norm(wlk.Φ[2].V[:, idx])
        rescale *= tmp
        wlk.Φ[2].V[:, idx] = tmp*wlk.Φ[2].V[:, idx]
    end
    testoverlap = testoverlap * rescale
    #stablize中更新overlap不更新weight
    update_overlap!(wlk, ham, false)
    println(testoverlap, wlk.overlap)
    if checkovlp && abs(testoverlap - wlk.overlap) > 1e-6
        println(testoverlap, " ", wlk.overlap)
        println(wlk)
        println(tempwlkphi1)
        println(tempwlkphi2)
        throw(error("precision low"))
    end
end



"""
给walker重新正交归一化
"""
function stablize!(wlk::HSWalker3, ham::HamConfig3; checkovlp=true)
    #@info wlk.weight
    if wlk.weight < 1e-8
        return
    end
    tempwlkphi1 = copy(wlk.Φ[1].V)
    tempwlkphi2 = copy(wlk.Φ[2].V)
    #第一个flavour
    slasize = size(wlk.Φ[1].V)
    rescale = 1.0
    #对每一个轨道进行正交归一
    for idx in 1:1:slasize[2]
        tmp = 1.0/norm(wlk.Φ[1].V[:, idx])
        rescale *= tmp
        wlk.Φ[1].V[:, idx] = tmp*wlk.Φ[1].V[:, idx]
        #对之后的所有轨道减去这个部分
        for idx2 in (idx+1):1:slasize[2]
            iprd = dot(wlk.Φ[1].V[:, idx], wlk.Φ[1].V[:, idx2])
            wlk.Φ[1].V[:, idx2] = wlk.Φ[1].V[:, idx2] - iprd*wlk.Φ[1].V[:, idx]
        end
    end
    testoverlap = wlk.overlap * rescale
    #第二个
    slasize = size(wlk.Φ[2].V)
    rescale = 1.0
    #对每一个轨道进行正交归一
    for idx in 1:1:slasize[2]
        tmp = 1.0/norm(wlk.Φ[2].V[:, idx])
        rescale *= tmp
        wlk.Φ[2].V[:, idx] = tmp*wlk.Φ[2].V[:, idx]
        #对之后的所有轨道减去这个部分
        for idx2 in (idx+1):1:slasize[2]
            iprd = dot(wlk.Φ[2].V[:, idx], wlk.Φ[2].V[:, idx2])
            wlk.Φ[2].V[:, idx2] = wlk.Φ[2].V[:, idx2] - iprd*wlk.Φ[2].V[:, idx]
        end
    end
    testoverlap = testoverlap * rescale
    #stablize中更新overlap不更新weight
    update_overlap!(wlk, ham, false)
    if checkovlp && abs(testoverlap - wlk.overlap) > 1e-6
        println(testoverlap, " ", wlk.overlap)
        println(wlk)
        println(tempwlkphi1)
        println(tempwlkphi2)
        throw(error("precision low"))
    end
end



"""
创建一个新的HSWalker
"""
function clone(wlk::HSWalker3, newname::String)
    sla1 = Slater{Float64}(newname*"_1", copy(wlk.Φ[1].V))
    sla2 = Slater{Float64}(newname*"_2", copy(wlk.Φ[2].V))
    wlk3 = HSWalker3((sla1, sla2), 1.0, wlk.weight, missing, missing, missing)
    wlk3.overlap = wlk3.weight
    wlk3.ovlpinv = (
        Slater{Float64}("copy-"*wlk.ovlpinv[1].name, copy(wlk.ovlpinv[1].V)),
        Slater{Float64}("copy-"*wlk.ovlpinv[2].name, copy(wlk.ovlpinv[2].V))
    )
    return wlk3
end


"""
将一个内容复制到另一个
"""
function copy_to(src::HSWalker3, dst::HSWalker3)
    dst.Φ[1].V .= src.Φ[1].V
    dst.Φ[2].V .= src.Φ[2].V
    dst.overlap = src.overlap
    dst.weight = src.weight
    if !ismissing(src.ovlpinv)
        dst.ovlpinv[1].V .= src.ovlpinv[1].V
        dst.ovlpinv[2].V .= src.ovlpinv[2].V
    end
    if !ismissing(src.hshist)
        dst.hshist .= src.hshist
    end
    if !ismissing(src.Φcache)
        dst.Φcache[1].V .= src.Φcache[1].V
        dst.Φcache[2].V .= src.Φcache[2].V
    end
end

