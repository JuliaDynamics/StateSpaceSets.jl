export statespace_sampler
export HSphere, HRectangle, HSphereSurface

using Random: Xoshiro
using LinearAlgebra: norm

abstract type Region end

"""
    statespace_sampler(region [, seed = 42]) → sampler, isinside

A function that facilitates randomly sampling points in a state space `region`.
It generates two functions:

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

- `HyperSphere(radius::Real, center)`: Points _inside_ the hypersphere (boundary excluded).
  Convenience method `HyperSphere(radius::Real, D::Int)` makes the center a
  `D`-long vector of zeros.
- `HyperSphereSurface(radius, center)`: Points on the boundary. Same convenience method as
  above is possible.
- `HyperRectangle(mins, maxs)`: points uniformly distributed in [min, max) for
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
    dummies = [zeros(dim) for _ in 1:Threads.nthreads()]
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

# TODO: Performance can be improved signficantly here by employing dedicated structs
# that have inner vectors that are updated in-place.

"""
    statespace_sampler(grid::NTuple{N, AbstractRange} [, rng])

If given a `grid` that is a tuple of `AbstractVector`s, the minimum and maximum of the
vectors are used as the `min_bounds` and `max_bounds` keywords.
"""
function statespace_sampler(
        grid::NTuple{N, AbstractRange}, rng::AbstractRNG = Random.GLOBAL_RNG
    ) where {N}
    return statespace_sampler(rng;
        min_bounds = minimum.(grid), max_bounds = maximum.(grid)
    )
end

function boxregion_multgauss(as, bs, rng)
    @assert length(as) == length(bs) > 0
    center = mean(hcat(as,bs))
    gen() = [rand(rng, truncated(Normal(center[i]), as[i], bs[i])) for i=1:length(as)]
    isinside(x) = all(as .< x .< bs)
    return gen, isinside
end

# this has a docstring only because it was part of the expansionentropy api.
# It won't be exported in future versions
"""
    boxregion(as, bs, rng = Random.GLOBAL_RNG) -> sampler, isinside

Define a box in ``\\mathbb{R}^d`` with edges the `as` and `bs` and then
return two functions: `sampler`, which generates a random initial condition in that box
and `isinside` that returns `true` if a given state is in the box.
"""
function boxregion(as, bs, rng = Random.GLOBAL_RNG)
    @assert length(as) == length(bs) > 0
    gen() = [rand(rng)*(bs[i]-as[i]) + as[i] for i in 1:length(as)]
    isinside(x) = all(as .≤ x .< bs)
    return gen, isinside
end

# Specialized 1-d version
function boxregion(a::Real, b::Real, rng = Random.GLOBAL_RNG)
    a, b = extrema((a, b))
    gen() = rand(rng)*(b-a) + a
    isinside = x -> a ≤ x < b
    return gen, isinside
end

# Algorithm from https://mathworld.wolfram.com/HyperspherePointPicking.html
# Normalized multivariate gaussian is on hypersphere
import LinearAlgebra
function sphereregion(r, dim, center, rng)
    @assert r ≥ 0
    dummy = zeros(dim)
    function generator()
        randn!(rng, dummy)
        n = LinearAlgebra.norm(dummy)
        ρ = (rand(rng)^(1/dim))*r
        dummy .*= ρ/n
        dummy .+= center
        return dummy
    end
    isinside(x) = norm(x .- center) ≤ r
    return generator, isinside
end
