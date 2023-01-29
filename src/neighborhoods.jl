#####################################################################################
#                   Neighborhood.jl Interface & convenience functions               #
#####################################################################################
using Neighborhood, Distances

export WithinRange, NeighborNumber
export Euclidean, Chebyshev, Cityblock

Neighborhood.KDTree(D::AbstractDataset, metric::Metric = Euclidean(); kwargs...) =
KDTree(vec(D), metric; kwargs...)

# Convenience extensions for ::Dataset in bulksearches
for f ∈ (:bulkisearch, :bulksearch)
    for nt ∈ (:NeighborNumber, :WithinRange)
        @eval Neighborhood.$(f)(ss::KDTree, D::AbstractDataset, st::$nt, args...; kwargs...) =
        $(f)(ss, D.data, st, args...; kwargs...)
    end
end
