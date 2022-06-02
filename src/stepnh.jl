#=
向前一步
=#


"""
向前一个slice(stblz_interval个)
"""
function step_slice!(wlk::HSWalker3, ham::HamConfig3, tauidxs::Vector{Int64};
    E_trial::Union{Missing, Float64}=missing)
    if length(tauidxs) == 0
        return
    end
    if ismissing(E_trial)
        @info "missing E_trial" maxlog=1
        E_trial = 0.
    end
    # e^-0.5ΔtH0 e^ΔtV (e^-0.5ΔtH0 e^-0.5ΔtH0)  e^ΔtV
    #先做用半个H0
    wlk.weight = exp(0.5*ham.dτ*E_trial) * wlk.weight
    #作用
    multiply_left!(ham.exp_halfdτHnhd, wlk.Φ[1])
    multiply_left!(ham.exp_halfdτHnhd, wlk.Φ[2])
    update_overlap!(wlk, ham, true)
    if abs(wlk.weight) < 1e-5
        return
    end
    #除了最后一次，作用e^ΔtV (e^-0.5ΔtH0 e^-0.5ΔtH0)
    for tidx in tauidxs[1:end-1]
        for opidx in 1:1:length(ham.Mzints)
            step_int!(wlk, ham, opidx, tidx)
            if abs(wlk.weight) < 1e-5
                return
            end
        end
        wlk.weight = exp(ham.dτ*E_trial) * wlk.weight
        multiply_left!(ham.exp_dτHnhd, wlk.Φ[1])
        multiply_left!(ham.exp_dτHnhd, wlk.Φ[2])
        update_overlap!(wlk, ham, true)
        if abs(wlk.weight) < 1e-5
            return
        end
    end
    #最后一次，作用e^ΔtV e^-0.5ΔtH0
    for opidx in 1:1:length(ham.Mzints)
        step_int!(wlk, ham, opidx, tauidxs[end])
        if abs(wlk.weight) < 1e-5
            return
        end
    end
    wlk.weight = exp(0.5*ham.dτ*E_trial) * wlk.weight
    multiply_left!(ham.exp_halfdτHnhd, wlk.Φ[1])
    multiply_left!(ham.exp_halfdτHnhd, wlk.Φ[2])
    update_overlap!(wlk, ham, true)
    if abs(wlk.weight) < 1e-5
        return
    end
end



"""
向前推进一个相互作用
"""
function step_int!(wlk::HSWalker3, ham::HamConfig3,
    opidx::Int64, tauidx::Int64)
    fl1 = ham.Mzints[opidx][2]
    fl2 = ham.Mzints[opidx][4]
    if fl1 != fl2
        step_int_diff_fl!(wlk, ham, opidx, tauidx)
    else
        step_int_same_fl!(wlk, ham, opidx, tauidx)
    end
end




"""
当fl1!=fl2时
"""
function step_int_diff_fl!(wlk::HSWalker3, ham::HamConfig3,
    opidx::Int64, tauidx::Int64)
    axfld = ham.Axflds[opidx]
    #wlk.Φ[1].V[1, :] .= (axfld.ΔV[1, 1]+1)*wlk.Φ[1].V[1, :]
    #wlk.Φ[2].V[1, :] .= (axfld.ΔV[1, 2]+1)*wlk.Φ[2].V[1, :]
    #计算需要的格林函数
    st1 = ham.Mzints[opidx][1]
    fl1 = ham.Mzints[opidx][2]
    #
    #println(adjoint(wlk.Φ[fl1].V[st1, :]))
    gr1 = adjoint(wlk.Φ[fl1].V[st1, :]) * wlk.ovlpinv[fl1].V
    gf1 = dot(gr1, ham.ΦtT[fl1].V[:, st1])
    #
    st2 = ham.Mzints[opidx][3]
    fl2 = ham.Mzints[opidx][4]
    gr2 = adjoint(wlk.Φ[fl2].V[st2, :]) * wlk.ovlpinv[fl2].V
    gf2 = dot(gr2, ham.ΦtT[fl2].V[:, st2])
    #
    newovlp = Vector{Float64}(undef, 2)
    #计算两种辅助场的新ovlp
    rat1 = 1.0 + axfld.ΔV[1, fl1]*gf1
    rat2 = 1.0 + axfld.ΔV[1, fl2]*gf2
    newovlp[1] = wlk.overlap * rat1 * rat2
    #
    rat1 = 1.0 + axfld.ΔV[2, fl1]*gf1
    rat2 = 1.0 + axfld.ΔV[2, fl2]*gf2
    newovlp[2] = wlk.overlap * rat1 * rat2
    #
    pt1 = 0.5 * axfld.Coef[1]*newovlp[1]/wlk.overlap
    if pt1 < 0
        wlk.weight = wlk.weight / (1 - pt1)
        pt1 = 0
    end
    pt2 = 0.5 * axfld.Coef[2]*newovlp[2]/wlk.overlap
    if pt2 < 0
        wlk.weight = wlk.weight / (1 - pt2)
        pt2 = 0
    end
    #if isnan(wlk.weight)
    #    println(pt1, " ", pt2)
    #    throw(error("a"))
    #end
    #传统方法更新
    #newphi1 = copy(wlk.Φ[fl1].V)
    #newphi1[st1, :] .= (axfld.ΔV[1, fl1]+1) * newphi1[st1, :]
    #ovlp1 = det(ham.ΦtT[fl1].V * newphi1)
    #newphi2 = copy(wlk.Φ[fl2].V)
    #newphi2[st2, :] .= (axfld.ΔV[1, fl2]+1) * newphi2[st2, :]
    #ovlp2 = det(ham.ΦtT[fl2].V * newphi2)
    #println(newovlp[1], " ", ovlp1*ovlp2, " ", wlk.overlap, " ", tauidx)
    #@assert isapprox(ovlp1*ovlp2, newovlp[1], atol=1e-8)
    #throw(error("abs"))
    #println(axfld.ΔV[1, 1], " ", axfld.ΔV[1, 2])
    #println(wlk.Φ[1].V)
    #println(pt1, " ", pt2)
    #throw(error("a"))
    prob = rand()*(pt1+pt2)
    ichose = prob < pt1 ? 1 : 2
    #传统方法
    #newphi1 = copy(wlk.Φ[fl1].V)
    #newphi1[st1, :] .= (axfld.ΔV[ichose, fl1]+1) * newphi1[st1, :]
    #oovlp1 = ham.ΦtT[fl1].V * newphi1
    #newphi2 = copy(wlk.Φ[fl2].V)
    #newphi2[st2, :] .= (axfld.ΔV[ichose, fl2]+1) * newphi2[st2, :]
    #oovlp2 = ham.ΦtT[fl2].V * newphi2
    #newovlptrd = det(oovlp1) * det(oovlp2)
    ##println(newovlptrd, " ", newovlp[ichose])
    #if !isapprox(newovlptrd, newovlp[ichose], rtol=1e-8)
    #    println(newovlptrd, " ", newovlp[ichose])
    #    throw(error("precision"))
    #end
    #更新wlk中的内容
    wlk.Φ[fl1].V[st1, :] .= (axfld.ΔV[ichose, fl1]+1)*wlk.Φ[fl1].V[st1, :]
    wlk.Φ[fl2].V[st2, :] .= (axfld.ΔV[ichose, fl2]+1)*wlk.Φ[fl2].V[st2, :]
    wlk.overlap = newovlp[ichose]
    wlk.weight = wlk.weight * (pt1+pt2)
    #更新inv（ovlp）
    #更新fl1
    rat = 1.0 + axfld.ΔV[ichose, fl1]*gf1
    gl1 = wlk.ovlpinv[fl1].V * ham.ΦtT[fl1].V[:, st1]
    #novlpinv = wlk.ovlpinv[fl1].V - (axfld.ΔV[ichose, fl1]/rat)*(gl1*gr1)
    wlk.ovlpinv[fl1].V .-= (axfld.ΔV[ichose, fl1]/rat)*(gl1*gr1)
    #更新fl2
    rat = 1.0 + axfld.ΔV[ichose, fl2]*gf2
    gl2 = wlk.ovlpinv[fl2].V * ham.ΦtT[fl2].V[:, st2]
    #println(typeof(gl2))
    wlk.ovlpinv[fl2].V .-= (axfld.ΔV[ichose, fl2]/rat)*(gl2*gr2)
    #println("aaaaaaaaaaa")
    #println(wlk.ovlpinv[fl1].V)
    #println(oinovlp)
    if !isfinite(wlk.weight)
        wlk.weight = 0
    end
    if !ismissing(wlk.hshist)
        hssize = size(wlk.hshist)
        wlk.hshist[opidx, mod(tauidx, hssize[2])+1] = ichose
    end
end



"""
当fl1==fl2时
"""
function step_int_same_fl!(wlk::HSWalker3, ham::HamConfig3,
    opidx::Int64, tauidx::Int64)
    axfld = ham.Axflds[opidx]
    newinvos = Vector{Matrix{Float64}}(undef, 2)
    #计算需要的格林函数，第一个算符直接在原本walker上更新
    st1 = ham.Mzints[opidx][1]
    fl1 = ham.Mzints[opidx][2]
    gr1 = adjoint(wlk.Φ[fl1].V[st1, :]) * wlk.ovlpinv[fl1].V
    gl1 = wlk.ovlpinv[fl1].V * ham.ΦtT[fl1].V[:, st1]
    gf1 = dot(gr1, ham.ΦtT[fl1].V[:, st1])
    #第二个算符的作用在作用完以后更新，所以有两种情况
    st2 = ham.Mzints[opidx][3]
    fl2 = ham.Mzints[opidx][4]
    if fl1 != fl2
        throw(error("fl1 != fl2 in step_int_same_fl!"))
    end
    if st1 == st2
        throw(error("st1 == st2 in step_int_same_fl!"))
    end
    gr2 = Vector{Matrix{Float64}}(undef, 2)
    #在julia中，N×1的矩阵会被当成列向量
    gl2 = Vector{Vector{Float64}}(undef, 2)
    gf2 = Vector{Float64}(undef, 2)
    for ichose in [1, 2]
        #先更新一次inv（ovlp）
        rat = 1.0 + axfld.ΔV[ichose, fl1]*gf1
        newinvos[ichose] = wlk.ovlpinv[fl1].V - (axfld.ΔV[ichose, fl1]/rat)*(gl1*gr1)
        gr2[ichose] = adjoint(wlk.Φ[fl2].V[st2, :]) * newinvos[ichose]
        gf2[ichose] = dot(gr2[ichose], ham.ΦtT[fl2].V[:, st2])
        gl2[ichose] = newinvos[ichose] * ham.ΦtT[fl2].V[:, st2]
    end
    #
    testovlp = Vector{Float64}(undef, 2)
    #计算两种辅助场的新ovlp
    rat1 = 1.0 + axfld.ΔV[1, fl1]*gf1
    rat2 = 1.0 + axfld.ΔV[1, fl2]*gf2[1]
    testovlp[1] = wlk.overlap * rat1 * rat2
    #
    rat1 = 1.0 + axfld.ΔV[2, fl1]*gf1
    rat2 = 1.0 + axfld.ΔV[2, fl2]*gf2[2]
    testovlp[2] = wlk.overlap * rat1 * rat2
    #
    pt1 = max(0., axfld.Coef[1]*testovlp[1]/wlk.overlap)
    pt2 = max(0., axfld.Coef[2]*testovlp[2]/wlk.overlap)
    #
    prob = rand()*(pt1+pt2)
    ichose = prob < pt1 ? 1 : 2
    #
    #println(ichose)
    #传统方法
    #newphi1 = copy(wlk.Φ[fl1].V)
    #newphi1[st1, :] .= (axfld.ΔV[ichose, fl1]+1) * newphi1[st1, :]
    #newphi1[st2, :] .= (axfld.ΔV[ichose, fl2]+1) * newphi1[st2, :]
    #oinovlp = inv(ham.ΦtT[fl1].V * newphi1)
    #println(oinovlp)
    #println(newinvos[ichose])
    #ngr2 = adjoint(newphi1[st2, :]) * newinvos[ichose]
    #ngf2 = dot(ngr2, ham.ΦtT[fl2].V[:, st2])
    #rat = 1.0 + axfld.ΔV[ichose, fl2]*ngf2
    #ngl2 = newinvos[ichose] * ham.ΦtT[fl2].V[:, st2]
    #ninovlp = newinvos[ichose] - (axfld.ΔV[ichose, fl2]/rat)*(ngl2*ngr2)
    #newphi1[st2, :] .= (axfld.ΔV[ichose, fl2]+1) * newphi1[st2, :]
    #oinovlp = inv(ham.ΦtT[fl2].V * newphi1)
    #println(oinovlp)
    #println(ninovlp)
    #
    #更新wlk中的内容
    wlk.Φ[fl1].V[st1, :] .= (axfld.ΔV[ichose, fl1]+1)*wlk.Φ[fl1].V[st1, :]
    wlk.Φ[fl2].V[st2, :] .= (axfld.ΔV[ichose, fl2]+1)*wlk.Φ[fl2].V[st2, :]
    wlk.overlap = testovlp[ichose]
    wlk.weight = wlk.weight * (pt1+pt2) / 2
    ##更新inv（ovlp）
    ##直接实用更新完第一个算符后的inv（ovlp）
    rat = 1.0 + axfld.ΔV[ichose, fl2]*gf2[ichose]
    wlk.ovlpinv[fl2].V .= newinvos[ichose] - 
    (axfld.ΔV[ichose, fl2]/rat)*(gl2[ichose]*gr2[ichose])
    if !isfinite(wlk.weight)
        wlk.weight = 0
    end
    #
    #println("aaaaaaaaaaa")
    #println(wlk.ovlpinv[fl1].V)
    #println(newinvos[ichose])
    #println(oinovlp)
    if !ismissing(wlk.hshist)
        hssize = size(wlk.hshist)
        wlk.hshist[opidx, mod(tauidx, hssize[2])+1] = ichose
    end
end


