export statespace_sampler
using Distributions, LinearAlgebra, Random
# TODO: The performance of this whole thing can be improved massively
# by creating structrs with pre-allocated vectors and using
# `rand!` or `randn!` in these vectors.
# TODO: I think these functions aren't tested...?

"""
    statespace_sampler(rng = Random.GLOBAL_RNG; kwargs...) → sampler, isinside
Convenience function that creates two functions. `sampler` is a 0-argument function
that generates random points inside a state space region defined by the keywords.
`isinside` is a 1-argument function that decides returns `true` if the given
state space point is inside that region.

The regions can be:
* **Rectangular box**, with edges `min_bounds` and `max_bounds`.
  The sampling of the points inside the box is decided by the keyword `method` which can
  be either `"uniform"` or `"multgauss"`.
* **Sphere**, of `spheredims` dimensions, radius `radius` and centered on `center`.
"""
function statespace_sampler(rng::AbstractRNG = Random.GLOBAL_RNG;
        min_bounds=[], max_bounds=[], method="uniform",
        radius::Number=-1,
        spheredims::Int=0, center=zeros(spheredims),
    )

    if min_bounds ≠ [] && max_bounds != []
        if method == "uniform"
            gen, isinside = boxregion(min_bounds, max_bounds, rng)
        elseif method == "multgauss"
            gen, isinside = boxregion_multgauss(min_bounds, max_bounds, rng)
        else
            @error("Unsupported boxregion sampling method")
        end
    elseif radius ≥ 0 && spheredims ≥ 1
        gen, isinside = sphereregion(radius, spheredims, center, rng)
    else
        @error("Incorrect keyword specification.")
    end
    return gen, isinside
end

"""
    statespace_sampler(grid::NTuple{N, AbstractRange} [, rng])
If given a `grid` that is a tuple of ranges, the minimum and maximum of the ranges
are used as the `min_bounds` and `max_bounds` keywords.
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
    isinside(x) = all(as .< x .< bs)
    return gen, isinside
end

# Specialized 1-d version
function boxregion(a::Real, b::Real, rng = Random.GLOBAL_RNG)
    a, b = extrema((a, b))
    gen() = rand(rng)*(b-a) + a
    isinside = x -> a < x < b
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
        dummy .*= r
        dummy ./= LinearAlgebra.norm(dummy)
        dummy .+= center
        return dummy
    end
    isinside(x) = norm(x .- center) < r
    return generator, isinside
end
