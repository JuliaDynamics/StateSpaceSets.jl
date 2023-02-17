module StateSpaceSets

# Use the README as the module docs
@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end StateSpaceSets


include("statespaceset.jl")
include("statespaceset_concrete.jl")
include("timeseries.jl")
include("neighborhoods.jl")
include("set_distance.jl")
include("utils.jl")
include("sampler.jl")
include("deprecations.jl")

end # module StateSpaceSets
