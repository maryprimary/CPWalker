#=
cpmc的主程序
=#


mutable struct CPSim2
    hamiltonian :: HamConfig2
    E_trial :: Union{Missing, Float64}
    walkers :: Vector{HSWalker2}
    stblz_interval :: Int64
    pctrl_interval :: Int64
    eqgrs :: Vector{CPMeasure{:EQGR, Array{Float64, 3}}}
end


function CPSim2(ham::HamConfig2, walkers::Vector{HSWalker2},
    stblz_interval::Int64, pctrl_interval::Int64)
    return CPSim2(
        ham, missing, walkers, stblz_interval, pctrl_interval,
        CPMeasure{:EQGR, Array{Float64, 3}}[]
    )
end


function initialize_simulation!(cpsim::CPSim2)
    initeqgr = get_eqgr_without_back(cpsim.hamiltonian, cpsim.walkers[1])
    if !ismissing(cpsim.E_trial)
        throw(error("is initialized"))
    end
    cpsim.E_trial = cal_energy(initeqgr, cpsim.hamiltonian)
    return initeqgr
end



"""
投影一段时间
"""
function relaxation_simulation(cpsim::CPSim2, relax_steps)
    #需要进行popular control的次数
    pctime = Int64(ceil(relax_steps/cpsim.pctrl_interval))
    for pidx = 1:1:pctime
        sttime = Int64(cpsim.pctrl_interval//cpsim.stblz_interval)
        taumat = Matrix{Int64}(undef, sttime, cpsim.stblz_interval)
        rstart = (pidx-1)*cpsim.pctrl_interval
        for sid1 = 1:1:sttime; for sid2 = 1:1:cpsim.stblz_interval
            taumat[sid1, sid2] = rstart
            rstart += 1
        end; end
        #println(sttime, " ", taumat)
        @Threads.threads for wlk in cpsim.walkers
            for sidx = 1:1:sttime
                step_slice!(wlk, cpsim.hamiltonian, taumat[sidx, :];
                E_trial=cpsim.E_trial)
                stablize!(wlk, cpsim.hamiltonian)
            end
        end
        #println([wlk.weight for wlk in cpsim.walkers[1:20]])
        weight_rescale!(cpsim.walkers)
        popctrl!(cpsim.walkers)
    end
end



"""
估计需要附加的能量，进行epoch次
"""
function E_trial_simulation!(cpsim::CPSim2, epochs::Int64, steps::Int64)
    growth_engr = Vector{Float64}(undef, epochs)
    for epidx = 1:1:epochs
        #for sidx = 1:1:steps
        #    @Threads.threads for wlk in cpsim.walkers
        #        step_dtau!(wlk, cpsim.hamiltonian, sidx; E_trial=cpsim.E_trial)
        #    end
        #end
        taus = [i for i = 1:1:steps]
        #println(taus)
        @Threads.threads for wlk in cpsim.walkers
            step_slice!(wlk, cpsim.hamiltonian, taus; E_trial=cpsim.E_trial)
        end
        #exp(Δτ * steps * E_gnd) * wgtsum = length(cpsim.walkers)
        #Δτ * steps * E_gnd = ln(length(cpsim.walkers)/wgtsum)
        wgtsum = sum([wlk.weight for wlk in cpsim.walkers])
        #println(wgtsum)
        egrowth = log(length(cpsim.walkers)/wgtsum) / steps / cpsim.hamiltonian.dτ
        growth_engr[epidx] = egrowth
        #
        @Threads.threads for wlk in cpsim.walkers
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
function initial_meaurements!(cpsim::CPSim2, mcsteps::Int64)
    for wlk in cpsim.walkers
        wlk.hshist = zeros(Int64, length(cpsim.hamiltonian.Mzints), mcsteps)
        lattsize = size(cpsim.hamiltonian.H0.V)[1]
        push!(cpsim.eqgrs, CPMeasure{:EQGR, Array{Float64, 3}}("eqgreen", zeros(lattsize, lattsize, 2)))
    end
end



"""
measure前的准备
"""
function premeas_simulation!(cpsim::CPSim2, mcsteps::Int64)
    #不做最后一次
    pctime = Int64(ceil(mcsteps/cpsim.pctrl_interval))
    for pidx = 1:1:pctime-1
        sttime = Int64(cpsim.pctrl_interval//cpsim.stblz_interval)
        taumat = Matrix{Int64}(undef, sttime, cpsim.stblz_interval)
        rstart = (pidx-1)*cpsim.pctrl_interval
        for sid1 = 1:1:sttime; for sid2 = 1:1:cpsim.stblz_interval
            taumat[sid1, sid2] = rstart
            rstart += 1
        end; end
        #println(sttime, " ", taumat)
        @Threads.threads for wlk in cpsim.walkers
            for sidx = 1:1:sttime
                step_slice!(wlk, cpsim.hamiltonian, taumat[sidx, :];
                E_trial=cpsim.E_trial)
                stablize!(wlk, cpsim.hamiltonian)
            end
        end
        weight_rescale!(cpsim.walkers)
        popctrl!(cpsim.walkers)
    end
    ##最后一次
    sttime = Int64(cpsim.pctrl_interval//cpsim.stblz_interval)
    taumat = Matrix{Int64}(undef, sttime, cpsim.stblz_interval)
    rstart = (pctime-1)*cpsim.pctrl_interval
    for sid1 = 1:1:sttime; for sid2 = 1:1:cpsim.stblz_interval
        taumat[sid1, sid2] = rstart
        rstart += 1
    end; end
    #println(sttime, " ", taumat)
    @Threads.threads for wlk in cpsim.walkers
        for sidx = 1:1:sttime
            step_slice!(wlk, cpsim.hamiltonian, taumat[sidx, :];
            E_trial=cpsim.E_trial)
            stablize!(wlk, cpsim.hamiltonian)
        end
    end
    weight_rescale!(cpsim.walkers)
    #更新格林函数
    for widx = 1:1:length(cpsim.walkers)
        eqgr = get_eqgr_without_back(cpsim.hamiltonian, cpsim.walkers[widx])
        cpsim.eqgrs[widx].V .= eqgr.V
    end
end


"""
在观测进行一次popular control
"""
function postmeas_simulation(cpsim::CPSim2)
    popctrl!(cpsim.walkers)
end


"""
开始模拟
"""
function start_simulation(cpsim::CPSim2, relaxepoch::Int64)
    engr_modify = []
    for epidx = 1:1:relaxepoch
        for wlk in cpsim.walkers
            step_dtau!(wlk, cpsim.hamiltonian, epidx; E_trial=cpsim.E_trial)
        end
        #stablize!中调用update_overlap!不会修改weight
        if mod(epidx, cpsim.stblz_interval) == 0
            for wlk in cpsim.walkers
                stablize!(wlk, cpsim.hamiltonian)
            end
            #println("ovlp list2", [wlk.overlap for wlk in cpsim.walkers])
        end
        if mod(epidx, cpsim.pctrl_interval) == 0
            popctrl!(cpsim.walkers)
            wgtsum = sum([wlk.weight for wlk in cpsim.walkers])
            #exp(Δτ * pctrl_interval * E_gnd) * wgtsum = length(cpsim.walkers)
            #Δτ * pctrl_interval * E_gnd = ln(length(cpsim.walkers)/wgtsum)
            e_gnd  = log(length(cpsim.walkers)/wgtsum) / cpsim.pctrl_interval / cpsim.hamiltonian.dτ
            push!(engr_modify, e_gnd)
            println("weight list", [wlk.weight for wlk in cpsim.walkers])
            println("ovlp list", [wlk.overlap for wlk in cpsim.walkers])
            println("wgt_sum  ", wgtsum, " ", e_gnd)
            rate = length(cpsim.walkers) / wgtsum
            for widx = 1:1:length(cpsim.walkers)
                cpsim.walkers[widx].weight *= rate
            end
            #println("weight list", [wlk.weight for wlk in cpsim.walkers])
            #println("wgt_sum  ", wgtsum)
        end
    end
    println(engr_modify)
end

