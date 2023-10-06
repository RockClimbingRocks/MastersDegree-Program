module FermionAlgebra
    using SparseArrays;
    using LinearAlgebra;
    using Combinatorics

    function ĉ(S:: Real)
        dim = Int(2*S + 1);
        c  = zeros(Int64, (dim, dim));

        for i in 1:(dim - 1)
            c[i, i + 1] = 1;
        end

        return c;
    end

    function ĉ⁺(S:: Real)
        c⁺  = ĉ(S)';

        return c⁺;
    end
    
    function n̂(S:: Real)
        c   = ĉ(S);
        c⁺  = ĉ⁺(S);
        n = c⁺*c;

        return n;
    end

    function id̂(S:: Real)
        dim = Int(2*S + 1);
        id  = Matrix{Int64}(I, dim, dim);

        return id;  
    end

    global function GetMatrixRepresentationOfOperator(operator:: String, S:: Real, isSparse:: Bool)
        matrixRepresentationOfOperator  = Dict{String, Matrix{Int}}(
            "c⁺" => ĉ⁺(S),
            "c" => ĉ(S),
            "n" => n̂(S),
            "id" => id̂(S),
        );

        matrix = matrixRepresentationOfOperator[operator];
        return isSparse ? sparse(matrix) : matrix;
    end

    
    """
    Function returns indeces of a subblock total of N particles. ∑nᵢ = N 
    """
    global function IndecesOfSubBlock(L:: Int64, N:: Int64 = Int(L÷2))
        S:: Float64 = 1/2;
        M:: Vector{Int64} = [Int(2*S+1)^i for i in 0:L-1];

        # Function provides us with indeces at wich states with N=L/2 occure in Hamiltonian. We introduced a map function becouse our collect(combinations(2 .^ (0:L-1),L÷2))) was not a ragular matrix but vector of a vector. We add '+1' to each element because here we start indexing arrays with 1 and not 0!
        M = collect(Int(2S+1).^(0:L-1));
        indices = Vector{Int64}(undef, binomial(L, N));
        map!(x -> sum(x) + 1, indices, collect(combinations(M,N)));
        sort!(indices);

        return indices;

    end

    """
    Function returns a state (defined as a number 'index') written in Fock space.
    """
    global function WriteStateInFockSpace(index:: Int64, L:: Int64, S:: Real):: Vector{Int64}
        state = zeros(Int64, L);
        digits!(state, index, base=Int(2*S+1));
        return state
    end    
    
end

