export statespace_sampler
export HSphere, HRectangle, HSphereSurface

using Random: Xoshiro
using LinearAlgebra: norm

abstract type Region end

"""
    statespace_sampler(region [, seed = 42]) → sampler, isinside

A function that facilitates sampling points randomly and uniformly in a state space
`region`. It generates two functions:

- `sampler` is a 0-argument function
  that when called generates a random point inside a state space `region`.
  Generally, the generated point should be _copied_ if it needs to be stored.
  (i.e., calling `sampler()` utilizes a shared vector)
  `sampler` is a thread-safe function.
- `isinside` is a 1-argument function that returns `true` if the given
  state space point is inside the `region`.

The `region` can be an instance of any of the following types
(input arguments if not specified are vectors of length `D`, with `D` the
state space dimension):

- `HSphere(radius::Real, center)`: points _inside_ the hypersphere (boundary excluded).
  Convenience method `HyperSphere(radius::Real, D::Int)` makes the center a
  `D`-long vector of zeros.
- `HSphereSurface(radius, center)`: points on the hypersphere surface. Same convenience
  method as above is possible.
- `HRectangle(mins, maxs)`: points in [min, max) for
  the bounds along each dimension.

The random number generator is always `Xoshiro` with the given `seed`.
"""
function statespace_sampler(::Region) end

struct HSphere{T} <: Region
    radius::T
    center::Vector{T}
end
HSphere(r::Real, D::Int) = HSphere(r, zeros(eltype(r), D))

struct HSphereSurface{T} <: Region
    radius::T
    center::Vector{T}
end
HSphereSurface(r::Real, D::Int) = HSphereSurface(r, zeros(eltype(r), D))

struct HRectangle{T} <: Region
    mins::Vector{T}
    maxs::Vector{T}
end
HRectangle(mins::Tuple, maxs::Tuple) = HRectangle([mins...], [maxs...])

function statespace_sampler(region::HSphere, seed = abs(rand(Int)))
    return sphereregion(region.r, region.center, Xoshiro(seed), true)
end

function statespace_sampler(region::HSphereSurface, seed = abs(rand(Int)))
    return sphereregion(region.r, region.center, Xoshiro(seed), false)
end

function sphereregion(r, center, rng, inside)
    @assert r ≥ 0
    dim = length(center)
    dummies = [zeros(dim) for _ in 1:Threads.nthreads()]
    generator = SphereGenerator(r, center, dummies, inside, rng)
    isinside(x) = norm(x .- center) ≤ r
    return generator, isinside
end
struct SphereGenerator{T, R}
    radius::T
    center::Vector{T}
    dummies::Vector{Vector{T}}
    inside::Bool
    rng::R
end
function (s::SphereGenerator)()
    dummy = dummies[Threads.threadid()]
    randn!(rng, dummy)
    n = LinearAlgebra.norm(dummy)
    ρ = inside ? (rand(rng)^(1/dim))*r : r
    dummy .*= ρ/n
    dummy .+= center
    return dummy
end

function statespace_sampler(region::HRectangle, seed = abs(rand(Int)))
    as = region.mins
    bs = region.maxs
    @assert length(as) == length(bs) > 0
    dummies = [zeros(length(as)) for _ in 1:Threads.nthreads()]
    gen = RectangleGenerator(as, bs .- as, dummies, Xoshiro(seed))
    isinside(x) = all(i => as[i] ≤ x[i] < bs[i], eachindex(x))
    return gen, isinside
end
struct RectangleGenerator{T, R}
    mins::Vector{T}
    difs::Vector{T}
    dummies::Vector{Vector{T}}
    rng::R
end
function (s::RectangleGenerator)()
    dummy = s.dummies[Threads.threadid()]
    rand!(rng, dummy)
    dummy .*= s.difs
    dummy .+= s.mins
    return dummy
end

"""
    statespace_sampler(grid::NTuple{N, AbstractRange} [, seed])

If given a `grid` that is a tuple of `AbstractVector`s, the minimum and maximum of the
vectors are used to make an `HRectangle` region.
"""
function statespace_sampler(
        grid::NTuple{N, AbstractRange}, seed = abs(rand(Int)),
    ) where {N}
    region = HRectangle(minimum.(grid), maximum.(grid))
    return statespace_sampler(region, seed)
end
