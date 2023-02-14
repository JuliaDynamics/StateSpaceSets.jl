# Extensions of the `StateSpaceSet` functions for timeseries
"""
    Timeseries = AbstractVector{<:Real}

A `Union` type representing timeseries in JuliaDynamics packages.
"""
const Timeseries = AbstractVector{<:Real}
dimension(::Timeseries) = 1
minima(x::Timeseries) = SVector(minimum(x))
maxima(x::Timeseries) = SVector(maximum(x))
function minmaxima(x::Timeseries)
    mi, ma = extrema(x)
    return SVector(mi), SVector(ma)
end

using Statistics: mean, std
"""
    standardize(x::Timeseries) = (x - mean(x))/std(x)
"""
standardize(x::Timeseries) = standardize!(copy(x))
standardize!(x::Timeseries) = (x .= (x .- mean(x))./std(x))
