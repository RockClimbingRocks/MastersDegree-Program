
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

    global function GetSign(positions:: Vector{Int64}):: Int64
        numberOfOperatorsToSort = length(positions)
    
        orderdPositions:: Vector{Int64} = [] 
        sign = +1;
    
        for i in range(1,numberOfOperatorsToSort)
            min = minimum(positions[i:end]); 
            positionOfMin = findall(x->x==min, positions[i:end])[1]+i-1;
    
            sign *= Int((-1)^(positionOfMin-i))
    
            append!(orderdPositions, min)
            deleteat!(positions, positionOfMin);
    
            positions = vcat(orderdPositions,positions[i:end])
        end
    
        return sign
    end

    global function IndecesOfSubBlock(L:: Int64, S:: Float64=1/2)
        # Function returns indeces of a subblock that have total number of spin in z disrection equal to zero S_z = ∑Szi=0

        M = collect( Int(2S+1).^(0:L-1))

        if S==1/2 
            # Function provides us with indeces at wich states with N=L/2 occure in Hamiltonian. We introduced a map function becouse our collect(combinations(2 .^ (0:L-1),L÷2))) was not a ragular matrix but vector of a vector. We add '+1' to each element because here we start indexing arrays with 1 and not 0!
            return sort(map(x -> sum(x), collect(combinations(M,L÷2)))) .+ 1
        elseif S==1 
            Spin_config = ones(Int64,L)
            Indeces = Int64[sum(M .* Spin_config)]
            for i in 1:2:L
                Spin_config[i], Spin_config[i+1] = 0, 2 
                # append!( Indeces, map(x -> sum(M .*x) ,unique(permutations(Spin_config, L))))
                append!( Indeces, map(x -> sum(M .*x) ,multiset_permutations(Spin_config,L)))
            end
            return sort(Indeces) .+ 1

        else
            println("Ajga, spin ni 1 ali 1/2!!")
        end
    end

    global function WriteStateInFockSpace(index:: Int64, L:: Int64, S:: Real):: Vector{Int64}

        state = zeros(Int64, L);

        dim = Int(2*S+1);

        remainingIndex = index;
        for l in L-1:-1:0

            if dim^l <= remainingIndex
                state[l+1] = 1;
                remainingIndex -= dim^l;
            end
        end

        return state
    end

end
