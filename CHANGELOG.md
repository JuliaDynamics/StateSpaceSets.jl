Changelog for StateSpaceSets.jl is kept w.r.t. version 1.3

# 2.4

- Containers in `setsofsets_distances` can now be both empty. They must be concretely typed.
- Fixed a critical bug where `eltype(ssset)` would return the element type of the stored points (e.g., `Float64`), instead of the type of the points themselves (e.g., `SVector{2, Float64}`). To be consistent with the Array interface and base Julia the correct return value is the type of the points themselves. Use `eltype(eltype(ssset))` to obtain the number type of the points.

# 2.3

- `set_distance` now also allows any arbitrary function `f`.
  This simplifies the docstring of `setsofsets_distances` as well.
- `SSSet` is now an exported alias to `StateSpaceSet`.

# 2.2

- Allow construction of `StateSpaceSet`s from existing `StateSpaceSet`s (this just calls `hcat` under the hood).

# 2.1

It is allowed to make `StateSpaceSet` with points of arbitrary `eltype` now.
Please don't use `Complex` numbers as the `eltype`!

# 2.0

- `StateSpaceSet` now subtypes `AbstractVector`, in particular `StateSpaceSet{V<:AbstractVector} <: AbstractVector{V}`. This leads to the breaking change that `size(ssset) = (length(ssset), )` while before `size` was `(length(ssset), dimension(ssset))`. Now you have to use `dimension(ssset)` exclusively to get the "number of columns" in the state space set.
- `StateSpaceSet` now supports arbitrary inner vectors as state space points.
  The keyword `container` can be given to all functions that make state space sets
  and sets the type of the container of the inner vectors. This is the abstract type
  and is typically `SVector` or `Vector`.
- All deprecations of v1 have been removed. Primarily this includes the `Dataset` name and an old version of `statespace_sampler`.


# 1.5

- `cov` and `cor` functions for computing the covariance/correlation matrix between
    columns of a `StateSpaceSet`.

# 1.4

`statespace_sampler` has been overhauled for major benefits:

- signature is now `statespace_sampler(r::Region, seed::Int)`. This clarifies by type what kind of regions may be sampled and allows straightforward future extensions.
- performance has been increased by eliminating all allocations
- sampling is now thread-safe
- Due to thread-safety, `Xoshiro` is used as the RNG. Seeds (integers) may be provided, but not rng objects themselves.
- old call signature with keywords is deprecated
- "multgauss" sampling is removed. It was poorly documented and as such we do not know if its implementation was even correct.