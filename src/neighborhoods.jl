#####################################################################################
#                   Neighborhood.jl Interface & convenience functions               #
#####################################################################################
using Neighborhood, Distances

export WithinRange, NeighborNumber
export Euclidean, Chebyshev, Cityblock

# Convenience extensions for ::StateSpaceSet in bulksearches
for f ∈ (:bulkisearch, :bulksearch)
    for nt ∈ (:NeighborNumber, :WithinRange)
        @eval Neighborhood.$(f)(ss::KDTree, D::AbstractStateSpaceSet, st::$nt, args...; kwargs...) =
        $(f)(ss, vec(D), st, args...; kwargs...)
    end
end
