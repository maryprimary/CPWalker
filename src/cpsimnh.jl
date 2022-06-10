#=
非厄米的模拟
=#



#struct CPControl3
#    Lx :: Int64
#    Ly :: Int64
#    U :: Float64
#    Δτ :: Float64
#    npart :: Int64
#    nwlks :: Int64
#    cfbeta :: Float64
#end



mutable struct CPSim3
    hamiltonian :: HamConfig3
    E_trial :: Union{Missing, Float64}
    walkers :: Vector{HSWalker3}
    backwlks :: Union{Missing, Vector{HSWalker3}}
    stblz_interval :: Int64
    beta_cut_intrv :: Int64
    eqgrs :: Vector{CPMeasure{:EQGR, Array{Float64, 3}}}
end



function CPSim3(ham::HamConfig3, walkers::Vector{HSWalker3},
    stblz_interval::Int64, pctrl_interval::Int64)
    return CPSim3(
        ham, missing, walkers, missing, stblz_interval, pctrl_interval,
        CPMeasure{:EQGR, Array{Float64, 3}}[]
    )
end


"""
初始
"""
function initialize_simulation!(cpsim::CPSim3, u_lower=false)
    initeqgr = get_eqgr_without_back(cpsim.hamiltonian, cpsim.walkers[1])
    if !ismissing(cpsim.E_trial)
        throw(error("is initialized"))
    end
    ham = cpsim.hamiltonian
    syssize = size(ham.Hnh.V)
    sysengr = 0.
    #hopping
    for st1 = 1:1:syssize[1]
        for st2 = 1:1:syssize[2]
            sysengr += ham.Hnh.V[st1, st2] * initeqgr.V[st1, st2, 1]
            sysengr += ham.Hnh.V[st1, st2] * initeqgr.V[st1, st2, 2]
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
        #修正HF精度下的U
        if u_lower
            u = log(u+1)
        end
        #
        sysengr += u*initeqgr.V[st1, st1, fl1]*initeqgr.V[st2, st2, fl2]
        #sysengr -= (0.5*u*eqgr.V[st1, st1, fl1] + 0.5*u*eqgr.V[st2, st2, fl2])
        if fl1 == fl2
            sysengr += -u*eqgr.V[st1, st2, fl1]*eqgr.V[st2, st1, fl1]
        end
    end
    cpsim.E_trial = sysengr
    return initeqgr
end


"""
投影一段时间
"""
function relaxation_simulation(cpsim::CPSim3, bctime::Int64)
    #需要进行beta cut的次数
    for pidx = 1:1:bctime
        sttime = Int64(cpsim.beta_cut_intrv//cpsim.stblz_interval)
        taumat = Matrix{Int64}(undef, sttime, cpsim.stblz_interval)
        rstart = (pidx-1)*cpsim.beta_cut_intrv
        for sid1 = 1:1:sttime; for sid2 = 1:1:cpsim.stblz_interval
            taumat[sid1, sid2] = rstart
            rstart += 1
        end; end
        #println(sttime, " ", taumat)
        for sidx = 1:1:sttime
            for wlk in cpsim.walkers
                #step_slice!(wlk, cpsim.hamiltonian, taumat[sidx, :];
                #E_trial=cpsim.E_trial)
                #if sidx != sttime
                #    decorate_stablize!(wlk, cpsim.hamiltonian)
                #else
                #    multiply_left!(cpsim.hamiltonian.SSd.V, wlk.Φ[1])
                #    multiply_left!(cpsim.hamiltonian.SSd.V, wlk.Φ[2])
                #    #println(cpsim.walkers[1].weight)
                #    #包含在stablize过程中，不需要更新weight
                #    update_overlap!(wlk, cpsim.hamiltonian, false)
                #    #println(cpsim.walkers[1].weight)
                #    stablize!(wlk, cpsim.hamiltonian)
                #end
                if sidx == 1
                    multiply_left!(cpsim.hamiltonian.SSd.V, wlk.Φ[1])
                    multiply_left!(cpsim.hamiltonian.SSd.V, wlk.Φ[2])
                    update_overlap!(wlk, cpsim.hamiltonian, true)
                end
                step_slice2!(wlk, cpsim.hamiltonian, taumat[sidx, :];
                E_trial=cpsim.E_trial)
                stablize!(wlk, cpsim.hamiltonian)
            end
            weight_rescale!(cpsim.walkers)
            popctrl!(cpsim.walkers)
        end
    end
end



"""
估计需要附加的能量，进行epoch次
"""
function E_trial_simulation!(cpsim::CPSim3)
    epochs = Int64(cpsim.beta_cut_intrv//cpsim.stblz_interval)
    growth_engr = Vector{Float64}(undef, epochs)
    #weight_rescale!(cpsim.walkers)
    #popctrl!(cpsim.walkers)
    #println(sum([wlk.weight for wlk in cpsim.walkers]))
    for epidx = 1:1:epochs
        weight_rescale!(cpsim.walkers)
        #for sidx = 1:1:steps
        #    @Threads.threads for wlk in cpsim.walkers
        #        step_dtau!(wlk, cpsim.hamiltonian, sidx; E_trial=cpsim.E_trial)
        #    end
        #end
        taus = [i for i = 1:1:cpsim.stblz_interval]
        #println(taus)
        #for wlk in cpsim.walkers
        #    step_slice!(wlk, cpsim.hamiltonian, taus; E_trial=cpsim.E_trial)
        #end
        for wlk in cpsim.walkers
            if epidx == 1
                multiply_left!(cpsim.hamiltonian.SSd.V, wlk.Φ[1])
                multiply_left!(cpsim.hamiltonian.SSd.V, wlk.Φ[2])
                update_overlap!(wlk, cpsim.hamiltonian, true)
            end
            step_slice2!(wlk, cpsim.hamiltonian, taus; E_trial=cpsim.E_trial)
        end
        #exp(Δτ * steps * E_gnd) * wgtsum = length(cpsim.walkers)
        #Δτ * steps * E_gnd = ln(length(cpsim.walkers)/wgtsum)
        wgtsum = 0.#sum([wlk.weight for wlk in cpsim.walkers])
        wlkcount = 0
        for wlk in cpsim.walkers
            if wlk.weight < 1e-5
                continue
            end
            wlkcount += 1
            wgtsum += wlk.weight
        end
        println(wgtsum, " ", wlkcount)
        egrowth = log(wlkcount/wgtsum) / cpsim.stblz_interval / cpsim.hamiltonian.dτ
        growth_engr[epidx] = egrowth
        #
        #if epidx != epochs
        #    for wlk in cpsim.walkers
        #        decorate_stablize!(wlk, cpsim.hamiltonian)
        #    end
        #else
        #    for wlk in cpsim.walkers
        #        multiply_left!(cpsim.hamiltonian.SSd.V, wlk.Φ[1])
        #        multiply_left!(cpsim.hamiltonian.SSd.V, wlk.Φ[2])
        #        #println(cpsim.walkers[1].weight)
        #        #包含在stablize过程中，不需要更新weight
        #        update_overlap!(wlk, cpsim.hamiltonian, false)
        #        #println(cpsim.walkers[1].weight)
        #        stablize!(wlk, cpsim.hamiltonian)
        #    end
        #end
        for wlk in cpsim.walkers
            stablize!(wlk, cpsim.hamiltonian)
        end
        weight_rescale!(cpsim.walkers)
        popctrl!(cpsim.walkers)
    end
    println(growth_engr)
    cpsim.E_trial += sum(growth_engr) / length(growth_engr)
    #throw(error(cpsim.E_trial))
end


"""
初始化观测
"""
function initial_meaurements!(cpsim::CPSim3)
    for wlk in cpsim.walkers
        wlk.hshist = zeros(Int64, length(cpsim.hamiltonian.Mzints), cpsim.beta_cut_intrv)
        wlk.Φcache = (Slater{Float64}(wlk.Φ[1].name*"_back_1", copy(wlk.Φ[1].V)),
        Slater{Float64}(wlk.Φ[2].name*"_back_2", copy(wlk.Φ[2].V)))
        lattsize = size(cpsim.hamiltonian.Hnh.V)[1]
        push!(cpsim.eqgrs, CPMeasure{:EQGR, Array{Float64, 3}}("eqgreen", zeros(lattsize, lattsize, 2)))
    end
end


"""
准备反传的walker
"""
function update_backwalkers!(cpsim::CPSim3, bctime::Int64)
    #
    if ismissing(cpsim.backwlks)
        cpsim.backwlks = Vector{HSWalker3}(undef, length(cpsim.walkers))
    end
    bwlk1 = HSWalker3("backwlk1", cpsim.hamiltonian,
    copy(cpsim.hamiltonian.Φt[1].V), copy(cpsim.hamiltonian.Φt[2].V), 1.0)
    cpsim.backwlks[1] = bwlk1
    for idx=2:1:length(cpsim.walkers)
        cpsim.backwlks[idx] = clone(bwlk1, "backwlk"*string(idx))
    end
    #
    for pidx = 1:1:(bctime-1)
        sttime = Int64(cpsim.beta_cut_intrv//cpsim.stblz_interval)
        taumat = Matrix{Int64}(undef, sttime, cpsim.stblz_interval)
        rstart = (pidx-1)*cpsim.beta_cut_intrv
        for sid1 = 1:1:sttime; for sid2 = 1:1:cpsim.stblz_interval
            taumat[sid1, sid2] = rstart
            rstart += 1
        end; end
        #println(sttime, " ", taumat)
        @Threads.threads for wlk in cpsim.backwlks
            #除了beta cut的最后一次
            #都做decorate stablize
            for sidx = 1:1:(sttime-1)
                step_slice!(wlk, cpsim.hamiltonian, taumat[sidx, :];
                E_trial=cpsim.E_trial)
                decorate_stablize!(wlk, cpsim.hamiltonian)
            end
            sidx = sttime
            step_slice!(wlk, cpsim.hamiltonian, taumat[sidx, :];
            E_trial=cpsim.E_trial)
            multiply_left!(cpsim.hamiltonian.SSd.V, wlk.Φ[1])
            multiply_left!(cpsim.hamiltonian.SSd.V, wlk.Φ[2])
            #包含在stablize过程中，不需要更新weight
            update_overlap!(wlk, cpsim.hamiltonian, false)
            stablize!(wlk, cpsim.hamiltonian)
        end
        weight_rescale!(cpsim.backwlks)
        popctrl!(cpsim.backwlks)
    end
    ##最后一次
    sttime = Int64(cpsim.beta_cut_intrv//cpsim.stblz_interval)
    taumat = Matrix{Int64}(undef, sttime, cpsim.stblz_interval)
    rstart = (bctime-1)*cpsim.beta_cut_intrv
    for sid1 = 1:1:sttime; for sid2 = 1:1:cpsim.stblz_interval
        taumat[sid1, sid2] = rstart
        rstart += 1
    end; end
    #println(sttime, " ", taumat)
    @Threads.threads for wlk in cpsim.backwlks
        #除了beta cut的最后一次
        #都做decorate stablize
        for sidx = 1:1:(sttime-1)
            step_slice!(wlk, cpsim.hamiltonian, taumat[sidx, :];
            E_trial=cpsim.E_trial)
            decorate_stablize!(wlk, cpsim.hamiltonian)
        end
        sidx = sttime
        step_slice!(wlk, cpsim.hamiltonian, taumat[sidx, :];
        E_trial=cpsim.E_trial)
        multiply_left!(cpsim.hamiltonian.SSd.V, wlk.Φ[1])
        multiply_left!(cpsim.hamiltonian.SSd.V, wlk.Φ[2])
        #包含在stablize过程中，不需要更新weight
        update_overlap!(wlk, cpsim.hamiltonian, false)
        stablize!(wlk, cpsim.hamiltonian)
    end
    weight_rescale!(cpsim.backwlks)
    popctrl!(cpsim.backwlks)
end


"""
measure前的准备
"""
function premeas_simulation!(cpsim::CPSim3)
    #存储用来反传的slater
    for wlk in cpsim.walkers
        wlk.hshist .= 0
        wlk.Φcache[1].V .= wlk.Φ[1].V
        wlk.Φcache[2].V .= wlk.Φ[2].V
    end
    #
    sttime = Int64(cpsim.beta_cut_intrv//cpsim.stblz_interval)
    taumat = Matrix{Int64}(undef, sttime, cpsim.stblz_interval)
    rstart = 0
    for sid1 = 1:1:sttime; for sid2 = 1:1:cpsim.stblz_interval
        taumat[sid1, sid2] = rstart
        rstart += 1
    end; end
    #println(sttime, " ", taumat)
    for sidx = 1:1:sttime
        for wlk in cpsim.walkers
            #step_slice!(wlk, cpsim.hamiltonian, taumat[sidx, :];
            #E_trial=cpsim.E_trial)
            ##除了beta cut的最后一次
            ##都做decorate stablize
            #if sidx != sttime
            #    decorate_stablize!(wlk, cpsim.hamiltonian)
            #else
            #    multiply_left!(cpsim.hamiltonian.SSd.V, wlk.Φ[1])
            #    multiply_left!(cpsim.hamiltonian.SSd.V, wlk.Φ[2])
            #    #println(cpsim.walkers[1].weight)
            #    #包含在stablize过程中，不需要更新weight
            #    update_overlap!(wlk, cpsim.hamiltonian, false)
            #    #println(cpsim.walkers[1].weight)
            #    stablize!(wlk, cpsim.hamiltonian)
            #end
            if sidx == 1
                multiply_left!(cpsim.hamiltonian.SSd.V, wlk.Φ[1])
                multiply_left!(cpsim.hamiltonian.SSd.V, wlk.Φ[2])
                update_overlap!(wlk, cpsim.hamiltonian, true)
            end
            step_slice2!(wlk, cpsim.hamiltonian, taumat[sidx, :];
            E_trial=cpsim.E_trial)
            stablize!(wlk, cpsim.hamiltonian)
        end
        weight_rescale!(cpsim.walkers)
        if sidx != sttime
            popctrl!(cpsim.walkers)
        end
    end
    #更新格林函数
    for widx = 1:1:length(cpsim.walkers)
        #eqgr = get_eqgr_without_back(cpsim.hamiltonian, cpsim.walkers[widx])
        #cpsim.eqgrs[widx].V .= eqgr.V
        calculate_eqgr!(cpsim.eqgrs[widx], cpsim.hamiltonian, 
        cpsim.walkers[widx], cpsim.stblz_interval)
        #calculate_eqgr2!(cpsim.eqgrs[widx], cpsim.walkers[widx], cpsim.walkers)
    end
end



"""
观测后进行popular control
"""
function postmeas_simulation(cpsim::CPSim3)
    #popctrl!(cpsim.walkers)
    weight_rescale!(cpsim.walkers)
end


