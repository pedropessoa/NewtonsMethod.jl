using NewtonsMethod
using Test
using LinearAlgebra, Statistics, Compat, ForwardDiff

@testset "NewtonsMethod.jl" begin

    f(x) = (x-1.0)^3
    f′(x) = 3.0*(x-1.0)^2
    x_0 = 0.1
    @test newtonroot(f, f′, x_0 = x_0).root ≈ 0.9999998168636032
    @test newtonroot(f, x_0 = x_0).root ≈ 0.9999998168636032
    @test newtonroot(f, f′, x_0 = BigFloat(x_0), tolerance = 1E-80).root ≈ BigFloat(1.0)
    @test newtonroot(f, x_0 = BigFloat(x_0), tolerance = 1E-80).root ≈ BigFloat(1.0)

    #maxiter
    @test newtonroot(f, f′, x_0 = x_0, maxiter = 5).root == nothing
    @test newtonroot(f, x_0 = x_0, maxiter = 5).root == nothing
    @test newtonroot(f, f′, x_0 = BigFloat(x_0), tolerance = 1E-80, maxiter = 10).root == nothing
    @test newtonroot(f, x_0 = BigFloat(x_0), tolerance = 1E-80, maxiter = 10).root == nothing

    g(x) = log(x)
    g′(x) = 1.0/x
    x_0 = 2.0
    @test newtonroot(g, g′, x_0 = x_0).root ≈ 1.0
    @test newtonroot(g, x_0 = x_0).root ≈ 1.0
    @test newtonroot(g, g′, x_0 = BigFloat(x_0), tolerance = 1E-50).root ≈ BigFloat(1.0)
    @test newtonroot(g, x_0 = BigFloat(x_0), tolerance = 1E-50).root ≈ BigFloat(1.0)

    @test !(newtonroot(g, g′, x_0 = x_0, tolerance = 1E-2).root ≈ 1.0)
    @test !(newtonroot(g, x_0 = x_0, tolerance = 1E-2).root ≈ 1.0)

    #non-convergence

    h(x) = x^2 + 1.0
    h′(x) = 2x
    x_0 = 10.0
    @test newtonroot(h, h′, x_0 = x_0).root == nothing
    @test newtonroot(h, x_0 = x_0).root == nothing
    @test newtonroot(h, h′, x_0 = BigFloat(x_0), tolerance = 1E-100).root == nothing
    @test newtonroot(h, x_0 = BigFloat(x_0), tolerance = 1E-100).root == nothing

end
