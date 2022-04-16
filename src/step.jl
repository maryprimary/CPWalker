#=
向前一步
=#


"""
向前一步
"""
function step_dtau!(wlk::HSWalker{T}, ham::HamConfig{T}, tauidx::Int64) where T
    if abs(wlk.weight) < 1e-5
        return
    end
    #multiply_left!(ham.exp_half_H0, wlk.Φ)
    update_overlap!(wlk, ham)
    step_int!(wlk, ham, 1, tauidx)
    update_overlap!(wlk, ham)
end


"""
向前一个slice(stblz_interval个)
"""
function step_slice!(wlk::HSWalker2, ham::HamConfig2, tauidxs::Vector{Int64};
    E_trial::Union{Missing, Float64}=missing)
    if length(tauidxs) == 0
        return
    end
    if ismissing(E_trial)
        @info "missing E_trial"
        E_trial = 0.
    end
    # e^-0.5ΔtH0 e^ΔtV (e^-0.5ΔtH0 e^-0.5ΔtH0)  e^ΔtV
    #先做用半个H0
    wlk.weight = exp(0.5*ham.dτ*E_trial) * wlk.weight
    multiply_left!(ham.exp_halfdτH0, wlk.Φ[1])
    multiply_left!(ham.exp_halfdτH0, wlk.Φ[2])
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
        multiply_left!(ham.exp_dτH0, wlk.Φ[1])
        multiply_left!(ham.exp_dτH0, wlk.Φ[2])
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
    multiply_left!(ham.exp_halfdτH0, wlk.Φ[1])
    multiply_left!(ham.exp_halfdτH0, wlk.Φ[2])
    update_overlap!(wlk, ham, true)
    if abs(wlk.weight) < 1e-5
        return
    end
end



"""
向前一步
"""
function step_dtau!(wlk::HSWalker2, ham::HamConfig2, tauidx::Int64;
    E_trial::Union{Missing, Float64}=missing)
    if abs(wlk.weight) < 1e-5
        return
    end
    if !ismissing(E_trial)
        wlk.weight = exp(ham.dτ*E_trial) * wlk.weight
    else
        @info "missing E_trial"
    end
    multiply_left!(ham.exp_halfdτH0, wlk.Φ[1])
    multiply_left!(ham.exp_halfdτH0, wlk.Φ[2])
    update_overlap!(wlk, ham, true)
    if abs(wlk.weight) < 1e-5
        return
    end
    for opidx in 1:1:length(ham.Mzints)
        step_int!(wlk, ham, opidx, tauidx)
        if abs(wlk.weight) < 1e-5
            return
        end
    end
    multiply_left!(ham.exp_halfdτH0, wlk.Φ[1])
    multiply_left!(ham.exp_halfdτH0, wlk.Φ[2])
    update_overlap!(wlk, ham, true)
    if abs(wlk.weight) < 1e-5
        return
    end
end


"""
向前作用一个相互作用
"""
function step_int!(wlk::HSWalker{T}, ham::HamConfig{T},
    opidx::Int64, tauidx::Int64) where T
    ovlp_old = wlk.overlap
    #(Φ'_T * Φ)^-1, Np*Np
    ovlp_inv = wlk.ovlpinv#inv(ham.Φ_trialT * wlk.Φ)
    #找到指数上的算符
    intop = ham.Hint[opidx]
    hsval = ham.Axflds[opidx]
    stidx = Vector{Int64}(undef, length(intop.rowi))
    for idx in 1:1:length(intop.rowi)
        if intop.rowi[idx] != intop.coli[idx]
            throw(error("只支持密度算符"))
        end
        stidx[idx] = intop.rowi[idx]
    end
    #gl = Npart × Nopdim
    #gl = ovlp_inv.V * ham.Φ_trialT.V[:, stidx]
    #gr = Nopdim * Np
    #gr = wlk.Φ.V[stidx, :] * ovlp_inv.V
    #g = gr × phi^†_T
    #gsite = dot(gr, ham.Φ_trialT.V[:, stidx])
    #每个格点上的格林函数
    gs = Vector{T}(undef, length(intop.rowi))
    for idx in 1:1:length(intop.rowi)
        gr = transpose(wlk.Φ.V[stidx[idx], :]) * ovlp_inv
        gs[idx] = dot(gr, ham.Φ_trialT.V[:, stidx[idx]])
    end
    dintop = sparse2dense(intop)
    #新的ovlp计算
    ovlplist = Vector{T}(undef, 4)
    for hsc in 1:1:4
        val = Dict(1=>-2, 2=>-1, 3=>1, 4=>2)[hsc]
        bmat = exp(hsval.g*dintop.V*LatticeHamiltonian.FourComponentEta(Val{val}()))
        println(bmat)
        matnew = bmat * wlk.Φ.V
        println(init_overlap(matnew, ham), "np")
        ovlp_new = ovlp_old
        for idx in 1:1:length(intop.rowi)
            ovlp_new *= (1.0 + hsval.ΔV[hsc]*gs[idx])^ham.ncolor
        end
        ovlplist[hsc] = ovlp_new
        println(ovlp_new)
        #throw(error())
    end
    #新的概率计算
    problist = Vector{Float64}(undef, 4)
    for hsc in 1:1:4
        ipd = real(ovlplist[hsc])#innerprod(ovlplist[hsc], ovlp_old)
        println("ipd  ", ipd)
        if ipd < 1e-8
            problist[hsc] = 0
        else
            problist[hsc] = abs((hsval.Coef[hsc]^ham.ncolor)*ovlplist[hsc]) / abs(ovlp_old)
        end
    end
    println(problist)
    #更新walker
    ranv = sum(problist) * rand()
    ichose = 3
    for idx in 1:1:length(intop.rowi)
        wlk.Φ.V[stidx[idx], :] *= (hsval.ΔV[ichose] + 1.0)
    end
end



"""
向前推进一个相互作用
"""
function step_int!(wlk::HSWalker2, ham::HamConfig2,
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
function step_int_diff_fl!(wlk::HSWalker2, ham::HamConfig2,
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
    #传统方法更新
    #newphi1 = copy(wlk.Φ[fl1].V)
    #newphi1[st1, :] .= (axfld.ΔV[1, fl1]+1) * newphi1[st1, :]
    #ovlp1 = det(ham.ΦtT[fl1].V * newphi1)
    #newphi2 = copy(wlk.Φ[fl2].V)
    #newphi2[st2, :] .= (axfld.ΔV[1, fl2]+1) * newphi2[st2, :]
    #ovlp2 = det(ham.ΦtT[fl2].V * newphi2)
    #println(newovlp[1], " ", ovlp1*ovlp2, " ", wlk.overlap)
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
    if !ismissing(wlk.hshist)
        hssize = size(wlk.hshist)
        wlk.hshist[opidx, mod(tauidx, hssize[2])+1] = ichose
    end
end



"""
当fl1==fl2时
"""
function step_int_same_fl!(wlk::HSWalker2, ham::HamConfig2,
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

