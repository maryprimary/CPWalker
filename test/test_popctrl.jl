#=
测试popctrl的功能
=#


using CPWalker


function run()
    numwlk = 10
    wlks = Vector{HSWalker}(undef, numwlk)
    for idx = 1:1:numwlk
        wlks[idx] = HSWalker(
            "wlk"*string(idx), rand(4, 2), rand(), rand()*2
        )
    end
    println([(wlk.weight, wlk.overlap) for wlk in wlks])
    popctrl(wlks)
    println([(wlk.weight, wlk.overlap) for wlk in wlks])
end


run()


