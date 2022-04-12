#=
哈密顿量
=#


struct HamConfig{T}
    H0 :: DenseFermiOpt{T}
    dτ :: Float64
    exp_half_H0 :: DenseFermiOpt{T}
    Hint :: Vector{SparseFermiOpt{T}}
    Axflds :: Vector{AuxiliaryField4}
    ncolor :: Int64
    npart :: Int64
    Φ_trial :: Slater{T}
    Φ_trialT :: Slater{T}
end


"""
创建一个哈密顿量
"""
function HamConfig(h0::Matrix{T}, dtau, npart, phit::Matrix{T}) where T
    h0op = DenseFermiOpt("H_0", h0)
    ehh0 = DenseFermiOpt("exp^0.5dτh0", exp(-0.5*dtau*h0))
    Φt = Slater{T}("Φ_T", phit)
    return HamConfig(
        h0op, dtau, ehh0,
        SparseFermiOpt{T}[],
        AuxiliaryField4[],
        2,
        npart, Φt, Φt'
    )
end


"""
利用Mz分解的哈密顿
"""
struct HamConfig2
    H0 :: DenseFermiOpt{Float64}
    dτ :: Float64
    Mzints :: Vector{NTuple{4, Int64}}
    Axflds :: Vector{AuxiliaryField2}
    npart :: Tuple{Int64, Int64}
    Φt :: Tuple{Slater{Float64}, Slater{Float64}}
    ΦtT :: Tuple{Slater{Float64}, Slater{Float64}}
    exp_halfdτH0 :: DenseFermiOpt{Float64}
    exp_dτH0 :: DenseFermiOpt{Float64}
end



"""
创建一个Mz分解的哈密顿量
"""
function HamConfig2(h0::Matrix{Float64}, dtau::Float64, 
    phit1::Matrix{Float64}, phit2::Matrix{Float64})
    h0op = DenseFermiOpt("H_0", h0)
    #
    size1 = size(phit1)
    Φt1 = Slater{Float64}("Φ_T1", phit1)
    npart1 = size1[2]
    #
    size2 = size(phit2)
    Φt2 = Slater{Float64}("Φ_T2", phit2)
    #
    exphalfh0 = exp(-0.5*dtau*h0)
    exphalfh0op = DenseFermiOpt("exp(-0.5*dτH0)", exphalfh0)
    exph0 = exp(-dtau*h0)
    exph0op = DenseFermiOpt("exp(-dτH0)", exph0)
    npart2 = size2[2]
    return HamConfig2(
        h0op, dtau, NTuple{4, Int64}[], AuxiliaryField2[],
        (npart1, npart2), (Φt1, Φt2), (adjoint(Φt1), adjoint(Φt2)),
        exphalfh0op, exph0op
    )
end


