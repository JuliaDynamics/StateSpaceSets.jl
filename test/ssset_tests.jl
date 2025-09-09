using Test, StateSpaceSets
using Statistics

println("\nTesting StateSpaceSet...")

o = [rand(SVector{3, Float64}) for _ in 1:10]
s = StateSpaceSet(o)

@testset "construction" begin
  x = rand(10)
  y = rand(10)
  z = rand(10, 4)
  w = rand(10, 2)
  s1 = StateSpaceSet(x)
  s2 = StateSpaceSet(x, y)
  s3 = StateSpaceSet(z)
  s4 = StateSpaceSet(w)

  @test dimension(StateSpaceSet(s3, s4)) == 6

  for (i, q) in enumerate((s1, s2, s, s3))
    @test dimension(q) == i
    @test size(q) == (10,)
    @test length(q) == 10
  end

  @testset "MVector" begin
    m1 = StateSpaceSet(x, y; container = MVector)
    @test m1[1] isa MVector
    m2 = StateSpaceSet(rand(10,10); container = MVector)
    @test m2[1] isa MVector
    m3 = StateSpaceSet([rand(3) for _ in 1:10]; container = MVector)
    @test m3[1] isa MVector
  end

  @testset "nonreal element" begin
    ff = fill("ff", 10)
    tt = fill("tt", 10)
    ss = StateSpaceSet(ff, tt)
    @test dimension(ss) == 2
    @test eltype(eltype(ss)) == String
  end
end


@testset "iteration" begin
  for (i, point) in enumerate(s)
    @test point == o[i]
  end
  q = map(x -> x[1], s)
  @test length(q) == length(o)
  @test q isa Vector{Float64}

  f(x) = x[1] - x[2]
  r = f.(s)
  @test r isa Vector{Float64}
  @test all(x -> -2 < x < 2, r)
end


@testset "append" begin
  s1 = deepcopy(s)
  append!(s1, s)
  @test length(s1) == 20
  push!(s1, s[end])
  @test length(s1) == 21
end

@testset "hcat" begin
    x1 = 1:5; x2 = 2:6; x3 = 3:7; x4 = 4:8
    @testset "T=$(T)" for T in (SVector, Vector)
      D1, D2 = StateSpaceSet(x1, x2; container = T), StateSpaceSet(x3, x4; container = T)
      @test hcat(D1, x1) == StateSpaceSet(x1, x2, x1)
      @test hcat(D1, D2) == StateSpaceSet(x1, x2, x3, x4)
      @test hcat(x1, D1) == StateSpaceSet(x1, x1, x2)
      @test hcat(x1, D1, x2) == StateSpaceSet(x1, x1, x2, x2)
      @test StateSpaceSets.containertype(hcat(D1, x1)) <: T
    end
end


@testset "indexing" begin
  s = [rand(SVector{3, Float64}) for _ in 1:100]
  s = StateSpaceSet(s)
  x = s[:, 1]
  @test x isa Vector{Float64}
  @test s[5] isa SVector{3, Float64}
  @test s[11:20] isa StateSpaceSet
  @test s[:, 2:3][:, 1] == s[:, 2]

  sub = view(s, 11:20)
  @test sub isa StateSpaceSets.SubStateSpaceSet
  @test sub[2] == s[12]
  @test dimension(sub) == dimension(s)
  d = sub[:, 1]
  @test d isa Vector{Float64}

  # setindex
  s[1] = SVector(0.1,0.1,0.1)
  @test s[1] == SVector(0.1,0.1,0.1)
  @test_throws ErrorException (s[:,1] .= 0)
end

@testset "copy" begin
  d = StateSpaceSet(rand(10, 2))
  v = vec(d)
  d2 = copy(d)
  d2[1] == d[1]
  d2[1] = SVector(5.0, 5.0)
  @test d2[1] != d[1]
end

@testset "minmax" begin
  mi = minima(s)
  ma = maxima(s)
  mimi, mama = minmaxima(s)
  @test mimi == mi
  @test mama == ma
  xs = columns(s)
  for i in 1:3
    @test mi[i] < ma[i]
    a,b = extrema(xs[i])
    @test a == mi[i]
    @test b == ma[i]
  end
end

@testset "Matrix" begin
  m = Matrix(s)
  @test StateSpaceSet(m) == s
  m = rand(1000, 4)
  @test Matrix(StateSpaceSet(m)) == m
end

@testset "standardize" begin
  r = standardize(s)
  rs = columns(r)
  for x in rs
    m, s = mean(x), std(x)
    @test abs(m) < 1e-8
    @test abs(s - 1) < 1e-8
  end
end

@testset "timeseries" begin
    x = rand(1000)
    @test dimension(x) == 1
    mi, ma = minmaxima(x)
    @test mi isa SVector
    @test 0 ≤ mi[1] ≤ 1
    @test 0 ≤ ma[1] ≤ 1
    @test minima(x) == mi
    @test maxima(x) == ma
end

@testset "cov/cor" begin
  x = StateSpaceSet(rand(1000, 2))
  @test Statistics.cov(Matrix(x)) ≈ StateSpaceSets.cov(x)
  @test Statistics.cor(Matrix(x)) ≈ StateSpaceSets.cor(x)
end

@testset "#34 hcat type piracy" begin
  u = hcat(1:3, ones(3))
  @test size(u) == (3, 2)
end



@testset "Named dimensions" begin

  names = [:x, :y]

  x = rand(10)
  y = rand(10)
  z = hcat(x, y)
  w = [z[i, :] for i in 1:10]

  s1 = StateSpaceSet(x, y; names)
  s2 = StateSpaceSet(z; names)
  s3 = StateSpaceSet(w; names)
  s4 = StateSpaceSet(z)

  for s in (s1, s2, s3)
    @test s[:, :x] == x
    @test s[:, :y] == y
    @test s[1, :x] == x[1]
    @test_throws "lala" s[:, :lala]
  end

  @test_throws "must be as many" StateSpaceSet(z; names = [:x, :y, :z])
  @test_throws "does not have names" s4[:, :x]

  s4 = StateSpaceSet(s4; names = [:x, :y])
  @test s4[:, :x] == x

end

