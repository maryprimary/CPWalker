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
function calculate_eqgr!(meas::CPMeasure{:EQGR, Array{Float64, 3}}, ham::HamConfig2, wlk::HSWalker2)
    #构造反传的Slater
    sla1 = Slater("backwlk"*wlk.Φ[1].name, copy(ham.Φt[1].V))
    sla2 = Slater("backwlk"*wlk.Φ[2].name, copy(ham.Φt[2].V))
    backwlk = HSWalker2(
        (sla1, sla2), 1.0, 1.0, missing, missing
    )
    #对backwlk进行反向传播
    hssize = size(wlk.hshist)
    for tauidx in 1:1:hssize[2]
        multiply_left!(ham.exp_dτH0, backwlk.Φ[1])
        multiply_left!(ham.exp_dτH0, backwlk.Φ[2])
        for opidx in 1:1:length(ham.Mzints)
            ichose = wlk.hshist[opidx, tauidx]
            axfld = ham.Axflds[opidx]
            st1 = ham.Mzints[opidx][1]
            fl1 = ham.Mzints[opidx][2]
            backwlk.Φ[fl1].V[st1, :] .= (axfld.ΔV[ichose, fl1]+1)*backwlk.Φ[fl1].V[st1, :]
            st2 = ham.Mzints[opidx][3]
            fl2 = ham.Mzints[opidx][4]
            backwlk.Φ[fl2].V[st2, :] .= (axfld.ΔV[ichose, fl2]+1)*backwlk.Φ[fl2].V[st2, :]
        end
    end
    #stablize!(wlk, ham)
    #
    #println(backwlk)
    #计算walker和试探波函数（backwalker）的inv（ovlp）
    backv1 = adjoint(backwlk.Φ[1].V)
    invovlp1 = inv(backv1 * wlk.Φ[1].V)
    #println(size(invovlp1))
    #计算格林函数
    meas.V[:, :, 1] .= wlk.Φ[1].V * invovlp1 * backv1
    #println(size(gr1))
    backv2 = adjoint(backwlk.Φ[2].V)
    invovlp2 = inv(backv2 * wlk.Φ[2].V)
    meas.V[:, :, 2] .= wlk.Φ[2].V * invovlp2 * backv2
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
