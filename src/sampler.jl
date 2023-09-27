export statespace_sampler
export HSphere, HRectangle, HSphereSurface

using Random
using LinearAlgebra: norm

abstract type Region end

"""
    statespace_sampler(region [, seed = 42]) → sampler, isinside

A function that facilitates sampling points randomly and uniformly in a state space
`region`. It generates two functions:

- `sampler` is a 0-argument function
  that when called generates a random point inside a state space `region`.
  The point is always a `Vector` for type stability irrespectively of dimension.
  Generally, the generated point should be _copied_ if it needs to be stored.
  (i.e., calling `sampler()` utilizes a shared vector)
  `sampler` is a thread-safe function.
- `isinside` is a 1-argument function that returns `true` if the given
  state space point is inside the `region`.

The `region` can be an instance of any of the following types
(input arguments if not specified are vectors of length `D`, with `D` the
state space dimension):

- `HSphere(radius::Real, center)`: points _inside_ the hypersphere (boundary excluded).
  Convenience method `HSphere(radius::Real, D::Int)` makes the center a
  `D`-long vector of zeros.
- `HSphereSurface(radius, center)`: points on the hypersphere surface. Same convenience
  method as above is possible.
- `HRectangle(mins, maxs)`: points in [min, max) for
  the bounds along each dimension.

The random number generator is always `Xoshiro` with the given `seed`.
"""
function statespace_sampler(::Region) end

"""
    HSphere(r::Real, center::AbstractVector)
    HSphere(r::Real, D::Int)

A state space region denoting all points _within_ a hypersphere.
"""
struct HSphere{T, V<:AbstractVector{T}} <: Region
    radius::T
    center::V
end
HSphere(r::Real, D::Int) = HSphere(r, zeros(eltype(r), D))

"""
    HSphereSurface(r::Real, center::AbstractVector)
    HSphereSurface(r::Real, D::Int)

A state space region denoting all points _on the surface_ (boundary)
of a hypersphere.
"""
struct HSphereSurface{T, V<:AbstractVector{T}} <: Region
    radius::T
    center::V
end
HSphereSurface(r::Real, D::Int) = HSphereSurface(r, zeros(eltype(r), D))

"""
    HRectangle(mins::AbstractVector, maxs::AbstractVector)

A state space region denoting all points _within_ the hyperrectangle.
"""
struct HRectangle{T, V<:AbstractVector{T}}
    mins::V
    maxs::V
end
HRectangle(mins::Tuple, maxs::Tuple) = HRectangle(SVector(mins), SVector(maxs))

function statespace_sampler(region::HSphere, seed = abs(rand(Int)))
    return sphereregion(region.radius, region.center, Xoshiro(seed), true)
end

function statespace_sampler(region::HSphereSurface, seed = abs(rand(Int)))
    return sphereregion(region.radius, region.center, Xoshiro(seed), false)
end

function sphereregion(r, center, rng, inside)
    @assert r ≥ 0
    dim = length(center)
    dummies = [zeros(typeof(r), dim) for _ in 1:Threads.nthreads()]
    generator = SphereGenerator(r, center, dummies, inside, rng, length(center))
    if inside
        isinside = (x) -> norm(x .- center) < r
    else
        isinside = (x) -> norm(x .- center) ≈ r
    end
    return generator, isinside
end
struct SphereGenerator{T, V<:AbstractVector{T}, R} <: Function
    radius::T
    center::V
    dummies::Vector{Vector{T}}
    inside::Bool
    rng::R
    D::Int
end
function (s::SphereGenerator)()
    dummy = s.dummies[Threads.threadid()]
    r = s.radius
    randn!(s.rng, dummy)
    n = LinearAlgebra.norm(dummy)
    ρ = s.inside ? (rand(s.rng)^(1/s.D))*r : r
    dummy .*= ρ/n
    dummy .+= s.center
    return dummy
end

function statespace_sampler(region::HRectangle, seed = abs(rand(Int)))
    as = region.mins
    bs = region.maxs
    @assert length(as) == length(bs) > 0
    T = as[1] isa AbstractFloat ? eltype(as) : Float64
    dummies = [zeros(T, length(as)) for _ in 1:Threads.nthreads()]
    gen = RectangleGenerator(T.(as), T.(bs .- as), dummies, Xoshiro(seed))
    isinside(x) = all(i -> as[i] ≤ x[i] < bs[i], eachindex(x))
    return gen, isinside
end
struct RectangleGenerator{T, V <: AbstractVector{T}, R} <: Function
    mins::V
    difs::V
    dummies::Vector{Vector{T}}
    rng::R
end
function (s::RectangleGenerator)()
    dummy = s.dummies[Threads.threadid()]
    rand!(s.rng, dummy)
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
