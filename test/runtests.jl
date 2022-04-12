#=
测试Slater行列式
=#

using Test
using CPWalker

@testset "Slater行列式" begin
    sla1 = CPWalker.NamedSlater("phi", "", "", zeros(10, 10))
    sla2 = CPWalker.NamedSlater("phi", "", "", ones(10, 10))
    @test sum(sla1.V) == 0
    v = [1 2; 3 4]
    pnum = @particle_number [1]
    println(pnum)
end

