Changelog for StateSpaceSets.jl is kept w.r.t. version 1.3

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