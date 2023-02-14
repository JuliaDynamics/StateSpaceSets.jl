"""
A Julia package that provides an interface for datasets, or more specifically, state space
sets. These are collections of points of fixed, and known by type, size (called dimension).
The main export of the package is the [`Dataset`](@ref) type, and all its surrounding
convenience functions. It is used in several projects in the JuliaDynamics organization,
such as [DynamicalSystems.jl](https://juliadynamics.github.io/DynamicalSystems.jl/dev/)
or [CausalityTools.jl](https://juliadynamics.github.io/CausalityTools.jl/dev/).

To install it, run `import Pkg; Pkg.add("StateSpaceSets")`.
"""
module StateSpaceSets

include("dataset.jl")
include("subdataset.jl")
include("timeseries.jl")
include("neighborhoods.jl")
include("set_distance.jl")
include("utils.jl")
include("sampler.jl")

end # module StateSpaceSets
