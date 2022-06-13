"""
精确对角化，有上下自旋
"""

using JLD
using LinearAlgebra

"""
统计一共有几个可能的状态
"""
function count_state_num(npart, nsite)
    num = 1
    for idx in 0:1:npart-1
        num = num * (nsite - idx)
    end
    den = 1
    for idx in 2:1:npart
        den = den * idx
    end
    return Int64((num // den)^2)
end


#println(count_state_num(2, 12))


"""
下一个状态
"""
function next_state(sta, maxint)
    sint = parse(Int64, sta, base=2)
    npart  = count_ones(sint)
    nsite = length(sta)
    if sint == maxint
        return lpad(repeat("1", npart), nsite, "0"), true
    end
    #println(npart)
    sint += 1
    while sint <= maxint
        if count_ones(sint) == npart
            break
        end
        sint += 1
    end
    return lpad(string(sint, base=2), nsite, "0"), false
end


#println(next_state("0011", parse(Int64, "1100", base=2)))
#println(next_state("1100", parse(Int64, "1100", base=2)))


"""
获得数组
"""
function get_state_array(npart, nsite)
    upsta = lpad(repeat("1", npart), nsite, "0")
    dnsta = lpad(repeat("1", npart), nsite, "0")
    res = Vector{String}(undef, count_state_num(npart, nsite))
    maxint = parse(Int64, rpad(repeat("1", npart), nsite, "0"), base=2)
    #println("开始状态 ", upsta * dnsta)
    for idx in 1:1:length(res)
        res[idx] = upsta * dnsta
        dnsta, flag = next_state(dnsta, maxint)
        #如果flag,dnsta就转回来了，也需要给upsta进位
        if flag
            #break
            upsta, flag = next_state(upsta, maxint)
        end
    end
    #println("开始状态 ", upsta * dnsta)
    return res
end


"""
检测从第一个状态到第二个状态，有几个bond
"""
macro have_bond(sta1, sta2, bdc, bda)
    return esc(quote
        bnum = 0
        sta3 = $sta1[1:$bda-1] * "0" * $sta1[$bda+1:end]
        sta3 = sta3[1:$bdc-1] * "1" * sta3[$bdc+1:end]
        if sta3 != $sta2 || $sta1 == $sta2
            bnum = 0
        else
            bdl = min($bda, $bdc)
            bdr = max($bda, $bdc)
            sigp = 0
            #println($sta1, " ", bdl," ", bdr, " ", $sta1[bdl:bdr-1])
            for idx in bdl:1:bdr-1
                if $sta1[idx] == '1'
                    sigp += 1
                end
            end
            if $bda < $bdc
                sigp -= 1
            end
            bnum = (-1)^sigp
        end
        bnum
    end)
end


#get_state_array(2, 12)

#function test_bond()
#    println(@have_bond "0011" "0110" 2 3)
#    println(@have_bond "0011" "0110" 2 4)
#end

#test_bond()


"""
获取哈密顿量
"""
function get_ham(npart, nsite, bonds, U)
    staarr = get_state_array(npart, nsite)
    #println(staarr)
    ham = zeros(ComplexF32, length(staarr), length(staarr))
    for idx2 in 1:1:length(staarr)
    for idx1 in 1:1:length(staarr)
        sta1 = staarr[idx1]
        sta2 = staarr[idx2]
        upsta1, dnsta1 = sta1[1:nsite], sta1[nsite+1:end]
        upsta2, dnsta2 = sta2[1:nsite], sta2[nsite+1:end]
        #dnsta1 = sta1
        #dnsta2 = sta2
        val = complex(0., 0.)
        #println(sta1, " ", sta2, " ",@have_bond upsta1 upsta2 bonds[1][1] bonds[1][2])
        for bnd in bonds
            #后面的bnd是消灭
            #up的部分
            if dnsta1 == dnsta2
                coef = @have_bond upsta2 upsta1 bnd[1] bnd[2]
                val += coef * bnd[3]
            end
            #dn的部分
            if upsta1 == upsta2
            #if true
                coef = @have_bond dnsta2 dnsta1 bnd[1] bnd[2]
                val += coef * bnd[3]
            end
        end
        ham[idx1, idx2] += val
    end
    end

    #onsite U
    for idx in 1:1:length(staarr)
        sta = staarr[idx]
        upsta, dnsta = sta[1:nsite], sta[nsite+1:end]
        for st in 1:1:nsite
            if upsta[st] == '1' && dnsta[st] == '1'
                ham[idx, idx] += U
            end
        end
    end
    return ham
end


"""作用一个消灭算符"""
function apply_annihilation(sta, sti_, flidx)
    if flidx == :UP
        sti = sti_
    else
        sti = sti_ + Int64(length(sta) // 2)
    end
    #println(sti, " ", sta)
    if sta[sti] == '0'
        return missing
    end
    sigpow = 0
    for sts in sta[1:sti-1]
        if sts == '1'
            sigpow += 1
        end
    end
    statmp = sta[1:sti-1]*"0"*sta[sti+1:end]
    return (-1)^sigpow, statmp
end


"""作用产生算符"""
function apply_creation(sta, sti_, flidx)
    if flidx == :UP
        sti = sti_
    else
        sti = sti_ + Int64(length(sta) // 2)
    end
    if sta[sti] == '1'
        return missing
    end
    sigpow = 0
    for sts in sta[1:sti-1]
        if sts == '1'
            sigpow += 1
        end
    end
    statmp = sta[1:sti-1]*"1"*sta[sti+1:end]
    return (-1)^sigpow, statmp
end


"""
计算格林函数
"""
function green_value(npart, nsite, gndvec, st1, st2)
    staarr = get_state_array(npart, nsite)
    stadic = Dict()
    for sta in staarr
        stadic[sta] = length(stadic) + 1
    end
    corr = 0.
    for idx = 1:1:length(staarr)
        sta = staarr[idx]
        ares = apply_annihilation(sta, st2, :UP)
        if ismissing(ares)
            continue
        end
        cres = apply_creation(ares[2], st1, :UP)
        if ismissing(cres)
            continue
        end
        finalsta = cres[2]
        finalvid = stadic[finalsta]
        corr += cres[1] * ares[1] *  adjoint(gndvec[finalvid]) * gndvec[idx]
    end
    return corr
end


"""
计算密度密度关联
"""
function dencorr_value(npart, nsite, gndvec, st1, sp1, st2, sp2)
    staarr = get_state_array(npart, nsite)
    stadic = Dict()
    for sta in staarr
        stadic[sta] = length(stadic) + 1
    end
    corr = 0.
    for idx = 1:1:length(staarr)
        sta = staarr[idx]
        upsta, dnsta = sta[1:nsite], sta[nsite+1:end]
        if sp1 == :UP
            sta1 = upsta
        else
            sta1 = dnsta
        end
        if sp2 == :UP
            sta2 = upsta
        else
            sta2 = dnsta
        end
        #
        if sta1[st1] == '1' && sta2[st2] == '1'
            corr += adjoint(gndvec[idx]) * gndvec[idx]
        end
    end
    return corr
end


"""
构造双子格子正方晶格
"""
function construc_ckb_lattice(L, nh)
    bonds = []
    push!(bonds, (1, 2 ,-1))
    push!(bonds, (2, 1 ,-1))
    push!(bonds, (2, 7 ,-1))
    push!(bonds, (7, 2 ,-1))
    push!(bonds, (7, 8 ,-1))
    push!(bonds, (8, 7 ,-1))
    push!(bonds, (1, 8 ,-1))
    push!(bonds, (8, 1 ,-1))
    push!(bonds, (3, 4 ,-1))
    push!(bonds, (4, 3 ,-1))
    push!(bonds, (5, 6 ,-1))
    push!(bonds, (6, 5 ,-1))
    push!(bonds, (6, 7 ,-1-nh))
    push!(bonds, (7, 6 ,-1+nh))
    push!(bonds, (7, 4 ,-1-nh))
    push!(bonds, (4, 7 ,-1+nh))
    push!(bonds, (5, 2 ,-1-nh))
    push!(bonds, (2, 5 ,-1+nh))
    push!(bonds, (2, 3 ,-1-nh))
    push!(bonds, (3, 2 ,-1+nh))
    return bonds
end


"""
构造一个非厄米的链
"""
function construct_chain_lattice(L, Y, nh)
    bonds = []
    for idx = 1:1:(L-1); for yidx=1:1:Y
        stidx = (yidx-1)*L + idx
        push!(bonds, (stidx, stidx+1, -1+nh))
        push!(bonds, (stidx+1, stidx, -1-nh))
    end; end
    #push!(bonds, (1, L, -1+nh))
    #push!(bonds, (L, 1, -1-nh))
    if Y < 2
        return bonds
    end
    #竖直的
    for idx = 1:1:L; for yidx=1:1:Y
        stidx = (yidx-1)*L + idx
        yplus1 = yidx == Y ? 1 : yidx+1
        vtidx = (yplus1-1)*L + idx
        push!(bonds, (stidx, vtidx, -1))
        if Y != 2
            push!(bonds, (vtidx, stidx, -1))
        end
    end; end
    return bonds
end

Lo = 8
Lp = 1
np= 3
nh= 0.1
U = 4.

println(count_state_num(np, Lo*Lp))
h0 = get_ham(np, Lo*Lp, construct_chain_lattice(Lo, Lp, nh), U)
#)construc_ckb_lattice(Lo, nh), U

#println(real(h0))

#println(eigvals(h0))

eigh0 = eigen(h0)
evalmat = Diagonal(real(eigh0.values))
Smat = eigh0.vectors

#println("smat ", Smat)

cbt = 5



#减去能量最小值，再对角化
reeval = real(eigh0.values)
reemin = minimum(reeval)
denmat2 = zeros(length(reeval), length(reeval))
for idx=1:1:length(reeval)
    denmat2[idx, idx] = exp(-cbt*(reeval[idx] - reemin))
end
ebhlmat = Hermitian(Smat * denmat2 * adjoint(Smat))
#hl = - (1/cbt) * log(ebhlmat)
#hl = real(hl)
eig = eigen(-ebhlmat)

for idx=1:1:Lo*Lp
    println(green_value(np, Lo*Lp, eig.vectors[:, 1], idx, idx))
end


for idx=1:1:Lo*Lp
    println(dencorr_value(np, Lo*Lp, eig.vectors[:, 1], idx, :UP, idx, :DN))
end


eng = 0.
for bnd in construct_chain_lattice(Lo, Lp, nh)
    #construc_ckb_lattice(Lo, nh)#
    global eng
    eng += 2 * green_value(np, Lo*Lp, eig.vectors[:, 1], bnd[1], bnd[2]) * bnd[3]
end

println("hopping eng ", eng)

for idx = 1:1:Lo*Lp
    global eng
    eng += U * dencorr_value(np, Lo*Lp, eig.vectors[:, 1], idx, :UP, idx, :DN)
end

println("eng ", eng)


println("====")

#强行对角化
denmat = exp(-cbt*evalmat)
ebhlmat = Hermitian(Smat * denmat * adjoint(Smat))
hl = - (1/cbt) * log(ebhlmat)
hl = real(hl)

#eig = eigen(-ebhlmat)
#eig = eigen(hl)
eig = eigh0
#println(real(ebhlmat))
println(eig.values[1:5])

for idx=1:1:Lo*Lp
    println(green_value(np, Lo*Lp, eig.vectors[:, 1], idx, idx))
end

