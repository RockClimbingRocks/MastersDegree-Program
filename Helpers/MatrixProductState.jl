module MPS
    using LinearAlgebra;


    """
    This function is needed to change order of dimensions because julia is column and not row maijor language.
    """
    local function reversedims(X)
        N = length(size(X))
        X = permutedims(X, N:-1:1)
        return X
    end

    function Bipartition(ψ::Vector{Float64}, setA::Vector{Int64}, setB::Vector{Int64})
        L  = length(setA) + length(setB);
        Lₐ = length(setA);

        permutation:: Vector{Int64} = vcat(setA, setB);

        M′′ = reshape(ψ, [2 for i in 1:L]...);
        M′  = permutedims(M′′, permutation);
        M   = reversedims(reshape(M′, :, Int(2^Lₐ) ))
        return M;
    end

    

end