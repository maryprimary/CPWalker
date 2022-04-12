#=
进行weight的控制
=#


"""
估计weight增长的速度
"""
function growth_estimate(wlks::Vector{HSWalker}, ham::HamConfig)
    
end


"""
将权重重新整理
"""
function weight_rescale!(wlks::Vector{HSWalker2})
    wgtsum = sum([wlk.weight for wlk in wlks])
    rate = length(wlks) / wgtsum
    for wlk in wlks
        wlk.weight *= rate
    end
end



"""
进行weight的控制
"""
function popctrl!(wlks::Vector{HSWalker})
    #部分求和
    wgttot = sum([wlk.weight for wlk in wlks])
    wgtavg = wgttot / length(wlks)
    #给所有大于avg的一个概率
    probpair = Tuple{Float64, Int64}[]
    for idx = 1:1:length(wlks)
        if wlks[idx].weight >= wgtavg
            push!(probpair, (wlks[idx].weight, idx))
        end
    end
    #
    println(probpair)
    #
    probsum = Vector{Float64}(undef, length(probpair))
    probsum[1] = probpair[1][1]
    for idx = 2:1:length(probpair)
        probsum[idx] = probsum[idx-1] + probpair[idx][1]
    end
    #对所有小于avg的，按照概率从之前大于avg的复制
    for idx = 1:1:length(wlks)
        if wlks[idx].weight >= wgtavg
            continue
        end
        randv = rand() * probsum[end]
        cpidx = 0
        for idx = 1:1:length(probsum)
            if probsum[idx] >= randv
                cpidx = idx
                break
            end
        end
        wlks[idx].weight = wlks[cpidx].weight
        wlks[idx].overlap = wlks[cpidx].overlap
        wlks[idx].Φ.V .= wlks[cpidx].Φ.V
    end
end


"""
进行weight的控制
"""
function popctrl!(wlks::Vector{HSWalker2})
    #return
    #部分求和
    wgttot = sum([wlk.weight for wlk in wlks])
    wgtavg = wgttot / length(wlks)
    #给所有大于avg的一个概率
    probpair = Tuple{Float64, Int64}[]
    for idx = 1:1:length(wlks)
        if wlks[idx].weight >= wgtavg
            push!(probpair, (wlks[idx].weight, idx))
        end
    end
    #
    #println(probpair)
    #
    if length(probpair) == 0
        return
    end
    probsum = Vector{Float64}(undef, length(probpair))
    probsum[1] = probpair[1][1]
    for idx = 2:1:length(probpair)
        probsum[idx] = probsum[idx-1] + probpair[idx][1]
    end
    #对所有小于avg的，按照概率从之前大于avg的复制
    for idx = 1:1:length(wlks)
        if wlks[idx].weight >= wgtavg
            continue
        end
        randv = rand() * probsum[end]
        cpidx = 0
        for tid = 1:1:length(probsum)
            if probsum[tid] >= randv
                cpidx = probpair[tid][2]
                break
            end
        end
        #println(cpidx, " ", wlks[cpidx].weight)
        copy_to(wlks[cpidx], wlks[idx])
    end
end

