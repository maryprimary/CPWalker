#=
cpmc的游走波函数
=#


mutable struct HSWalker{T}
    Φ :: Slater{T}
    overlap :: T
    weight :: Float64
    ovlpinv :: Union{Missing, Matrix{T}}
    hshist :: Union{Missing, Matrix{Int64}}
end


mutable struct HSWalker2
    Φ :: Tuple{Slater{Float64}, Slater{Float64}}
    overlap :: Float64
    weight :: Float64
    ovlpinv :: Union{Missing, Tuple{Slater{Float64}, Slater{Float64}}}
    hshist :: Union{Missing, Matrix{Int64}}#第一个编号是相互作用，第二个是时间
end



#=
计算和试探波函数的交叠
=#


"""
计算初始的和试探波函数的内积
"""
function init_overlap(sla, ham::HamConfig)
    ovlp_inv = ham.Φ_trialT.V * sla
    detovlp = det(ovlp_inv)
    return detovlp^ham.ncolor
end




"""
计算和试探波函数的交叠
"""
function update_overlap!(wlk::HSWalker, ham::HamConfig)
    #Np * Np
    ovlp_inv = ham.Φ_trialT * wlk.Φ
    #ovlp
    detovlp = det(ovlp_inv)
    invovlp = inv(ovlp_inv)
    #
    ovlpnew = detovlp^ham.ncolor
    #这个时候就停止了
    if real(ovlpnew) < 1e-8#innerprod(ovlpnew, wlk.overlap) < 1e-8
        wlk.weight = 0.
    else
        wlk.weight = (abs(ovlpnew) / abs(wlk.overlap)) * wlk.weight
    end
    wlk.overlap = ovlpnew
    wlk.ovlpinv = invovlp.V
end


"""
计算和试探波函数的交叠
"""
function update_overlap!(wlk::HSWalker2, ham::HamConfig2, weight_update::Bool)
    #第一个
    #Np * Np
    ovlp_inv1 = ham.ΦtT[1] * wlk.Φ[1]
    #ovlp
    detovlp1 = det(ovlp_inv1)
    #if detovlp1 < 1e-5
    #    wlk.weight = 0.
    #    wlk.overlap = 0.
    #    return
    #end
    invovlp1 = inv(ovlp_inv1)
    #第二个
    ovlp_inv2 = ham.ΦtT[2] * wlk.Φ[2]
    #ovlp
    detovlp2 = det(ovlp_inv2)
    #if detovlp2 < 1e-5
    #    wlk.weight = 0.
    #    wlk.overlap = 0.
    #    return
    #end
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
    wlk.overlap = ovlpnew
    wlk.ovlpinv = (invovlp1, invovlp2)
end



"""
给walker重新正交归一化
"""
function stablize!(wlk::HSWalker, ham::HamConfig)
    if wlk.weight < 1e-8
        return
    end
    slasize = size(wlk.Φ.V)
    rescale = 1.0
    #对每一个轨道进行正交归一
    for idx in 1:1:slasize[2]
        tmp = 1.0/norm(wlk.Φ.V[:, idx])
        rescale *= tmp
        wlk.Φ.V[:, idx] = tmp*wlk.Φ.V[:, idx]
        #对之后的所有轨道减去这个部分
        for idx2 in (idx+1):1:slasize[2]
            iprd = dot(wlk.Φ.V[:, idx], wlk.Φ.V[:, idx2])
            wlk.Φ.V[:, idx2] = wlk.Φ.V[:, idx2] - iprd*wlk.Φ.V[:, idx]
        end
    end
    testoverlap = wlk.overlap * (rescale^ham.ncolor)
    update_overlap!(wlk, ham)
    if abs(testoverlap - wlk.overlap) > 1e-6
        throw(error("precision low"))
    end
end




"""
给walker重新正交归一化
"""
function stablize!(wlk::HSWalker2, ham::HamConfig2)
    if wlk.weight < 1e-8
        return
    end
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
    if abs(testoverlap - wlk.overlap) > 1e-6
        throw(error("precision low"))
    end
end



function HSWalker(
    name::String, mat::Matrix{T}, ovlp::T, wgt::Float64
    ) where T
    return HSWalker{T}(Slater{T}(name, mat), ovlp, wgt, missing, missing)
end



function HSWalker2(
    name::String, ham::HamConfig2, phi1::Matrix{Float64}, phi2::Matrix{Float64}, wgt::Float64
    )
    sla1 = Slater{Float64}(name*"_1", phi1)
    sla2 = Slater{Float64}(name*"_2", phi2)
    wlk = HSWalker2((sla1, sla2), 1.0, wgt, missing, missing)
    update_overlap!(wlk, ham, true)
    return wlk
end


"""
创建一个新的HSWalker
"""
function clone(wlk::HSWalker2, newname::String)
    sla1 = Slater{Float64}(newname*"_1", copy(wlk.Φ[1].V))
    sla2 = Slater{Float64}(newname*"_2", copy(wlk.Φ[2].V))
    wlk2 = HSWalker2((sla1, sla2), 1.0, wlk.weight, missing, missing)
    wlk2.overlap = wlk2.weight
    wlk2.ovlpinv = (
        Slater{Float64}("copy-"*wlk.ovlpinv[1].name, copy(wlk.ovlpinv[1].V)),
        Slater{Float64}("copy-"*wlk.ovlpinv[2].name, copy(wlk.ovlpinv[2].V))
    )
    return wlk2
end


"""
将一个内容复制到另一个
"""
function copy_to(src::HSWalker2, dst::HSWalker2)
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
end
