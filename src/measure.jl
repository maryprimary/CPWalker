#=
进行观测
=#



mutable struct CPMeasure{N, T}
    name :: String
    V :: T
end


"""
更新格林函数
"""
function calculate_eqgr!(meas::CPMeasure{:EQGR, Array{Float64, 3}},
    ham::HamConfig2, wlk::HSWalker2, stbint::Int64)
    if abs(wlk.weight) < 1e-5
        return
    end
    #构造反传的Slater
    sla1 = Slater("backwlk"*wlk.Φ[1].name, copy(ham.Φt[1].V))
    sla2 = Slater("backwlk"*wlk.Φ[2].name, copy(ham.Φt[2].V))
    backwlk = HSWalker2(
        (sla1, sla2), 1.0, 1.0, missing, missing, missing
    )
    #对backwlk进行反向传播
    hssize = size(wlk.hshist)
    slnum = Int64(hssize[2] // stbint)
    tidx = 0
    for _ = 1:1:slnum
        #每次先做一个
        multiply_left!(ham.exp_halfdτH0, backwlk.Φ[1])
        multiply_left!(ham.exp_halfdτH0, backwlk.Φ[2])
        for tauidx = 1:1:(stbint-1)
            tidx += 1
            tidxinv = hssize[2] - tidx + 1
            for opidx in 1:1:length(ham.Mzints)
                ichose = wlk.hshist[opidx, tidxinv]
                #if ichose == 0
                #    println(wlk)
                #end
                axfld = ham.Axflds[opidx]
                st1 = ham.Mzints[opidx][1]
                fl1 = ham.Mzints[opidx][2]
                backwlk.Φ[fl1].V[st1, :] .= (axfld.ΔV[ichose, fl1]+1)*backwlk.Φ[fl1].V[st1, :]
                st2 = ham.Mzints[opidx][3]
                fl2 = ham.Mzints[opidx][4]
                backwlk.Φ[fl2].V[st2, :] .= (axfld.ΔV[ichose, fl2]+1)*backwlk.Φ[fl2].V[st2, :]
            end
            multiply_left!(ham.exp_dτH0, backwlk.Φ[1])
            multiply_left!(ham.exp_dτH0, backwlk.Φ[2])
        end
        tidx += 1
        tidxinv = hssize[2] - tidx + 1
        for opidx in 1:1:length(ham.Mzints) 
            ichose = wlk.hshist[opidx, tidxinv]
            axfld = ham.Axflds[opidx]
            st1 = ham.Mzints[opidx][1]
            fl1 = ham.Mzints[opidx][2]
            backwlk.Φ[fl1].V[st1, :] .= (axfld.ΔV[ichose, fl1]+1)*backwlk.Φ[fl1].V[st1, :]
            st2 = ham.Mzints[opidx][3]
            fl2 = ham.Mzints[opidx][4]
            backwlk.Φ[fl2].V[st2, :] .= (axfld.ΔV[ichose, fl2]+1)*backwlk.Φ[fl2].V[st2, :]
        end
        multiply_left!(ham.exp_halfdτH0, backwlk.Φ[1])
        multiply_left!(ham.exp_halfdτH0, backwlk.Φ[2])
        stablize!(backwlk, ham; checkovlp=false)
    end
    #
    #计算walker和试探波函数（backwalker）的inv（ovlp）
    backv1 = adjoint(backwlk.Φ[1].V)
    #invovlp1 = inv(backv1 * wlk.Φ[1].V)
    invovlp1 = inv(backv1 * wlk.Φcache[1].V)#wlk.Φ[1].V)
    #println(size(invovlp1))
    #计算格林函数
    #meas.V[:, :, 1] .= wlk.Φ[1].V * invovlp1 * backv1
    meas.V[:, :, 1] .= wlk.Φcache[1].V * invovlp1 * backv1
    #println(size(gr1))
    backv2 = adjoint(backwlk.Φ[2].V)
    #invovlp2 = inv(backv2 * wlk.Φ[2].V)
    invovlp2 = inv(backv2 * wlk.Φcache[2].V)#wlk.Φ[2].V)
    #meas.V[:, :, 2] .= wlk.Φ[2].V * invovlp2 * backv2
    meas.V[:, :, 2] .= wlk.Φcache[2].V * invovlp2 * backv2
end


"""
更新格林函数
"""
function calculate_eqgr!(meas::CPMeasure{:EQGR, Array{Float64, 3}},
    ham::HamConfig3, wlk::HSWalker3, stbint::Int64)
    #if abs(wlk.weight) < 1e-5
    #    return
    #end
    if any(iszero.(wlk.hshist))
        wlk.weight = 0.
        return
    end
    #构造反传的Slater
    sla1 = Slater("backwlk"*wlk.Φ[1].name, copy(ham.Φt[1].V))
    sla2 = Slater("backwlk"*wlk.Φ[2].name, copy(ham.Φt[2].V))
    backwlk = HSWalker3(
        (sla1, sla2), 1.0, 1.0, missing, missing, missing
    )
    #
    hssize = size(wlk.hshist)
    slnum = Int64(hssize[2] // stbint)
    tidx = 0
    #
    #println(slnum)
    for slidx = 1:1:slnum
        #每次先做一个
        multiply_left!(ham.exp_halfdτHnhd, backwlk.Φ[1])
        multiply_left!(ham.exp_halfdτHnhd, backwlk.Φ[2])
        for tauidx = 1:1:(stbint-1)
            tidx += 1
            tidxinv = hssize[2] - tidx + 1
            for opidx in 1:1:length(ham.Mzints)
                ichose = wlk.hshist[opidx, tidxinv]
                axfld = ham.Axflds[opidx]
                st1 = ham.Mzints[opidx][1]
                fl1 = ham.Mzints[opidx][2]
                backwlk.Φ[fl1].V[st1, :] .= (axfld.ΔV[ichose, fl1]+1)*backwlk.Φ[fl1].V[st1, :]
                st2 = ham.Mzints[opidx][3]
                fl2 = ham.Mzints[opidx][4]
                backwlk.Φ[fl2].V[st2, :] .= (axfld.ΔV[ichose, fl2]+1)*backwlk.Φ[fl2].V[st2, :]
            end
            multiply_left!(ham.exp_dτHnhd, backwlk.Φ[1])
            multiply_left!(ham.exp_dτHnhd, backwlk.Φ[2])
        end
        tidx += 1
        tidxinv = hssize[2] - tidx + 1
        for opidx in 1:1:length(ham.Mzints) 
            ichose = wlk.hshist[opidx, tidxinv]
            axfld = ham.Axflds[opidx]
            st1 = ham.Mzints[opidx][1]
            fl1 = ham.Mzints[opidx][2]
            backwlk.Φ[fl1].V[st1, :] .= (axfld.ΔV[ichose, fl1]+1)*backwlk.Φ[fl1].V[st1, :]
            st2 = ham.Mzints[opidx][3]
            fl2 = ham.Mzints[opidx][4]
            backwlk.Φ[fl2].V[st2, :] .= (axfld.ΔV[ichose, fl2]+1)*backwlk.Φ[fl2].V[st2, :]
        end
        multiply_left!(ham.exp_halfdτHnhd, backwlk.Φ[1])
        multiply_left!(ham.exp_halfdτHnhd, backwlk.Φ[2])
        if slidx == slnum
            multiply_left!(ham.SSd.V, backwlk.Φ[1])
            multiply_left!(ham.SSd.V, backwlk.Φ[2])
            #包含在stablize过程中，不需要更新weight
            update_overlap!(backwlk, ham, false)
            stablize!(backwlk, ham; checkovlp=false)
        else
            decorate_stablize!(backwlk, ham; checkovlp=false)
        end
    end
    #第二种反传
    #multiply_left!(ham.SSd.V, backwlk.Φ[1])
    #multiply_left!(ham.SSd.V, backwlk.Φ[2])
    ##包含在stablize过程中，不需要更新weight
    #update_overlap!(backwlk, ham, false)
    #for slidx = 1:1:slnum
    #    #每次先做一个
    #    multiply_left!(adjoint(ham.exp_halfdτHnhd), backwlk.Φ[1])
    #    multiply_left!(adjoint(ham.exp_halfdτHnhd), backwlk.Φ[2])
    #    for tauidx = 1:1:(stbint-1)
    #        tidx += 1
    #        tidxinv = hssize[2] - tidx + 1
    #        for opidx in 1:1:length(ham.Mzints)
    #            ichose = wlk.hshist[opidx, tidxinv]
    #            axfld = ham.Axflds[opidx]
    #            st1 = ham.Mzints[opidx][1]
    #            fl1 = ham.Mzints[opidx][2]
    #            backwlk.Φ[fl1].V[st1, :] .= (axfld.ΔV[ichose, fl1]+1)*backwlk.Φ[fl1].V[st1, :]
    #            st2 = ham.Mzints[opidx][3]
    #            fl2 = ham.Mzints[opidx][4]
    #            backwlk.Φ[fl2].V[st2, :] .= (axfld.ΔV[ichose, fl2]+1)*backwlk.Φ[fl2].V[st2, :]
    #        end
    #        multiply_left!(adjoint(ham.exp_dτHnhd), backwlk.Φ[1])
    #        multiply_left!(adjoint(ham.exp_dτHnhd), backwlk.Φ[2])
    #    end
    #    tidx += 1
    #    tidxinv = hssize[2] - tidx + 1
    #    for opidx in 1:1:length(ham.Mzints) 
    #        ichose = wlk.hshist[opidx, tidxinv]
    #        axfld = ham.Axflds[opidx]
    #        st1 = ham.Mzints[opidx][1]
    #        fl1 = ham.Mzints[opidx][2]
    #        backwlk.Φ[fl1].V[st1, :] .= (axfld.ΔV[ichose, fl1]+1)*backwlk.Φ[fl1].V[st1, :]
    #        st2 = ham.Mzints[opidx][3]
    #        fl2 = ham.Mzints[opidx][4]
    #        backwlk.Φ[fl2].V[st2, :] .= (axfld.ΔV[ichose, fl2]+1)*backwlk.Φ[fl2].V[st2, :]
    #    end
    #    multiply_left!(adjoint(ham.exp_halfdτHnhd), backwlk.Φ[1])
    #    multiply_left!(adjoint(ham.exp_halfdτHnhd), backwlk.Φ[2])
    #    #if slidx == 1#slnum
    #        stablize!(backwlk, ham; checkovlp=false)
    #    #else
    #    #    multiply_left!(ham.iSSd.V, backwlk.Φ[1])
    #    #    multiply_left!(ham.iSSd.V, backwlk.Φ[2])
    #    #    update_overlap!(backwlk, ham, false)
    #    #    stablize!(backwlk, ham; checkovlp=false)
    #    #    multiply_left!(ham.SSd.V, backwlk.Φ[1])
    #    #    multiply_left!(ham.SSd.V, backwlk.Φ[2])
    #    #    update_overlap!(backwlk, ham, false)
    #    #end
    #end
    #
    #if backwlk.weight < 1e-5
    #    wlk.weight = 0
    #    return
    #end
    #计算walker和试探波函数（backwalker）的inv（ovlp）
    #println(backwlk.Φ[1].V, wlk.Φ[1].V)
    backv1 = adjoint(backwlk.Φ[1].V)
    #invovlp1 = inv(backv1 * wlk.Φ[1].V)
    invovlp1 = inv(backv1 * wlk.Φcache[1].V)#wlk.Φ[1].V)
    #println(size(invovlp1))
    #计算格林函数
    #meas.V[:, :, 1] .= wlk.Φ[1].V * invovlp1 * backv1
    meas.V[:, :, 1] .= wlk.Φcache[1].V * invovlp1 * backv1
    #println(size(gr1))
    backv2 = adjoint(backwlk.Φ[2].V)
    #invovlp2 = inv(backv2 * wlk.Φ[2].V)
    invovlp2 = inv(backv2 * wlk.Φcache[2].V)#wlk.Φ[2].V)
    #meas.V[:, :, 2] .= wlk.Φ[2].V * invovlp2 * backv2
    meas.V[:, :, 2] .= wlk.Φcache[2].V * invovlp2 * backv2
end



"""
更新格林函数
"""
function calculate_eqgr2!(meas::CPMeasure{:EQGR, Array{Float64, 3}},
    wlk::HSWalker3, walkers::Vector{HSWalker3})
    total_weight = sum([wlk.weight for wlk in walkers])
    ssize = size(meas.V)
    resgr = zeros(ssize[1], ssize[2], ssize[3])
    for backwlk in walkers
        #计算walker和试探波函数（backwalker）的inv（ovlp）
        #println(backwlk.Φ[1].V, wlk.Φ[1].V)
        backv1 = adjoint(backwlk.Φ[1].V)
        #invovlp1 = inv(backv1 * wlk.Φ[1].V)
        invovlp1 = inv(backv1 * wlk.Φ[1].V)#wlk.Φ[1].V)
        #println(size(invovlp1))
        #计算格林函数
        resgr[:, :, 1] += wlk.Φ[1].V * invovlp1 * backv1 * backwlk.weight
        #println(size(gr1))
        backv2 = adjoint(backwlk.Φ[2].V)
        #invovlp2 = inv(backv2 * wlk.Φ[2].V)
        invovlp2 = inv(backv2 * wlk.Φ[2].V)#wlk.Φ[2].V)
        #meas.V[:, :, 2] .= wlk.Φ[2].V * invovlp2 * backv2
        resgr[:, :, 2] += wlk.Φ[2].V * invovlp2 * backv2 * backwlk.weight
    end
    meas.V .= resgr / total_weight
end


"""
初始格林函数
"""
function get_eqgr_without_back(ham::HamConfig2, wlk::HSWalker2)
    syssize = size(ham.H0.V)
    eqgr = Array{Float64}(undef, syssize[1], syssize[2], 2)
    adjv1 = adjoint(ham.Φt[1].V)
    invovlp1 = inv(adjv1 * wlk.Φ[1].V)
    eqgr[:, :, 1] .= wlk.Φ[1].V * invovlp1 * adjv1
    adjv2 = adjoint(ham.Φt[2].V)
    invovlp2 = inv(adjv2 * wlk.Φ[2].V)
    eqgr[:, :, 2] .= wlk.Φ[2].V * invovlp2 * adjv2
    return CPMeasure{:EQGR, Array{Float64, 3}}(
        "init_eqgr", eqgr
    )
end


"""
初始格林函数，非厄米
"""
function get_eqgr_without_back(ham::HamConfig3, wlk::HSWalker3)
    syssize = size(ham.Hnh.V)
    eqgr = Array{Float64}(undef, syssize[1], syssize[2], 2)
    adjv1 = adjoint(ham.Φt[1].V)
    invovlp1 = inv(adjv1 * wlk.Φ[1].V)
    eqgr[:, :, 1] .= wlk.Φ[1].V * invovlp1 * adjv1
    #println("eqgr", eqgr[:, :, 1])
    #println(adjoint(ham.Φt[1].V) * ham.Φt[1].V)
    #println(adjoint(wlk.Φ[1].V) * wlk.Φ[1].V)
    #println("============")
    adjv2 = adjoint(ham.Φt[2].V)
    invovlp2 = inv(adjv2 * wlk.Φ[2].V)
    eqgr[:, :, 2] .= wlk.Φ[2].V * invovlp2 * adjv2
    return CPMeasure{:EQGR, Array{Float64, 3}}(
        "init_eqgr", eqgr
    )
end



"""
计算能量
"""
function cal_energy(eqgr::CPMeasure{:EQGR, Array{Float64, 3}}, ham::HamConfig2)
    syssize = size(ham.H0.V)
    sysengr = 0.
    #hopping
    for st1 = 1:1:syssize[1]
        for st2 = 1:1:syssize[2]
            sysengr += ham.H0.V[st1, st2] * eqgr.V[st1, st2, 1]
            sysengr += ham.H0.V[st1, st2] * eqgr.V[st1, st2, 2]
        end
    end
    #println(sysengr)
    #interaction
    for opidx = 1:1:length(ham.Mzints)
        mint = ham.Mzints[opidx]
        axfld = ham.Axflds[opidx]
        st1, fl1 = mint[1], mint[2]
        st2, fl2 = mint[3], mint[4]
        u = (axfld.ΔV[1, 1] + 1)*(axfld.ΔV[1, 2] + 1)
        u = -log(u) / ham.dτ
        sysengr += u*eqgr.V[st1, st1, fl1]*eqgr.V[st2, st2, fl2]
        #sysengr -= (0.5*u*eqgr.V[st1, st1, fl1] + 0.5*u*eqgr.V[st2, st2, fl2])
        if fl1 == fl2
            sysengr += -u*eqgr.V[st1, st2, fl1]*eqgr.V[st2, st1, fl1]
        end
    end
    return sysengr
end


"""
计算能量
"""
function cal_energy(eqgr::CPMeasure{:EQGR, Array{Float64, 3}}, ham::HamConfig3)
    syssize = size(ham.Hnh.V)
    sysengr = 0.
    #hopping
    for st1 = 1:1:syssize[1]
        for st2 = 1:1:syssize[2]
            sysengr += ham.Hnh.V[st1, st2] * eqgr.V[st1, st2, 1]
            sysengr += ham.Hnh.V[st1, st2] * eqgr.V[st1, st2, 2]
        end
    end
    hopeng = sysengr
    #println(sysengr)
    #interaction
    for opidx = 1:1:length(ham.Mzints)
        mint = ham.Mzints[opidx]
        axfld = ham.Axflds[opidx]
        st1, fl1 = mint[1], mint[2]
        st2, fl2 = mint[3], mint[4]
        u = (axfld.ΔV[1, 1] + 1)*(axfld.ΔV[1, 2] + 1)
        u = -log(u) / ham.dτ
        sysengr += u*eqgr.V[st1, st1, fl1]*eqgr.V[st2, st2, fl2]
        #sysengr -= (0.5*u*eqgr.V[st1, st1, fl1] + 0.5*u*eqgr.V[st2, st2, fl2])
        if fl1 == fl2
            sysengr += -u*eqgr.V[st1, st2, fl1]*eqgr.V[st2, st1, fl1]
        end
    end
    return hopeng, sysengr
end


"""
获得平均值和误差
"""
function postprocess_measurements(meas::Vector{CPMeasure{:SCALE, T}},
    whgt::Vector{CPMeasure{:SCALE, Float64}}) where T
    avg = 0.
    binnum = length(whgt)
    for bidx = 1:1:binnum
        avg += meas[bidx].V / whgt[bidx].V
    end
    avg = avg / binnum
    err = 0.
    for bidx = 1:1:binnum
        dat = meas[bidx].V / whgt[bidx].V
        err += (avg - dat)^2
    end
    err = sqrt(err / (binnum-1))
    return avg, err
end


"""
获得平均值和误差
"""
function postprocess_measurements(meas::Vector{CPMeasure{:MATRIX, T}},
    whgt::Vector{CPMeasure{:SCALE, Float64}}) where T
    avg = zeros(size(meas[1].V))
    binnum = length(whgt)

    for bidx = 1:1:binnum
        avg .+= meas[bidx].V / whgt[bidx].V
    end
    avg = avg / binnum
    err = zeros(size(meas[1].V))
    for bidx = 1:1:binnum
        dat = meas[bidx].V / whgt[bidx].V
        err .+= (avg .- dat).^2
    end
    err = sqrt.(err / (binnum-1))
    return avg, err
end


"""
获得平均值和误差
"""
function postprocess_measurements(meas::Vector{CPMeasure{:EQGR, Array{Float64, 3}}},
    whgt::Vector{CPMeasure{:SCALE, Float64}})
    avg = zeros(size(meas[1].V))
    binnum = length(whgt)

    for bidx = 1:1:binnum
        avg .+= meas[bidx].V / whgt[bidx].V
    end
    avg = avg / binnum
    err = zeros(size(meas[1].V))
    for bidx = 1:1:binnum
        dat = meas[bidx].V / whgt[bidx].V
        err .+= (avg .- dat).^2
    end
    err = sqrt.(err / (binnum-1))
    return avg, err
end

