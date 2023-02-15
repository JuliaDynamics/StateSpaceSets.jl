"""
A Julia package that provides functionality for state space sets.
These are collections of points of fixed, and known by type, size (called dimension).
The main export of the package is the [`StateSpaceSet`](@ref) type, and all its surrounding
convenience functions. It is used in several projects in the JuliaDynamics organization,
such as [DynamicalSystems.jl](https://juliadynamics.github.io/DynamicalSystems.jl/dev/)
or [CausalityTools.jl](https://juliadynamics.github.io/CausalityTools.jl/dev/).

The main export of `StateSpaceSets` is the concrete type `StateSpaceSet`.
The package also provides functionality for distances, neighbor searches,
sampling, and normalization.

To install it you may run `import Pkg; Pkg.add("StateSpaceSets")`,
however, there is no real reason to install this package directly
as it is re-exported by all downstream packages that use it.
"""
module StateSpaceSets

include("statespaceset.jl")
include("statespaceset_concrete.jl")
include("timeseries.jl")
include("neighborhoods.jl")
include("set_distance.jl")
include("utils.jl")
include("sampler.jl")
include("deprecations.jl")

end # module StateSpaceSets
