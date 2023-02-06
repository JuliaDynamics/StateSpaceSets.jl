export orthonormal

#####################################################################################
#                                Conversions                                        #
#####################################################################################
to_matrix(a::AbstractVector{<:AbstractVector}) = cat(2, a...)
to_matrix(a::AbstractMatrix) = a
function to_Smatrix(m)
    M = to_matrix(m)
    a, b = size(M)
    return SMatrix{a, b}(M)
end
to_vectorSvector(a::AbstractVector{<:SVector}) = a
function to_vectorSvector(a::AbstractMatrix)
    S = eltype(a)
    D, k = size(a)
    ws = Vector{SVector{D, S}}(k)
    for i in 1:k
        ws[i] = SVector{D, S}(a[:, i])
    end
    return ws
end

"""
    orthonormal([T,] D, k) -> ws
Return a matrix `ws` with `k` columns, each being
an `D`-dimensional orthonormal vector.

`T` is the return type and can be either `SMatrix` or `Matrix`.
If not given, it is `SMatrix` if `D*k < 100`, otherwise `Matrix`.
"""
function orthonormal end

orthonormal(D, k) = D*k < 100 ? orthonormal(SMatrix, D, k) : orthonormal(Matrix, D, k)

@inline function orthonormal(T::Type, D::Int, k::Int)
    k > D && throw(ArgumentError("k must be ≤ D"))
    if T == SMatrix
        q = qr(rand(SMatrix{D, k})).Q
    elseif T == Matrix
        q = Matrix(qr(rand(Float64, D, k)).Q)
    end
    q
end
