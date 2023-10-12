module c⁺c⁺cc
    using SparseArrays;

    include("../FermionAlgebra.jl");
    using .FermionAlgebra;


    global function AnaliticalNormOfHamiltonianAveraged(L::Int)
        norm2 = L*(L-2)*(L^2 + 6L + 8 - 2*(L-2)/(L-1)) / 4 ;
        return √norm2;
    end


    function GetSignOfOperatorPermutation(i::Int64, j::Int64, k::Int64, l::Int64, state::Vector{Int})
        sign_l = isodd(sum(@view(state[1:l-1]))) ? -1 : 1;
        state[l] = 0;
        sign_k = isodd(sum(@view(state[1:k-1]))) ? -1 : 1;
        state[k] = 0;
        sign_j = isodd(sum(@view(state[1:j-1]))) ? -1 : 1;
        state[j] = 1;
        sign_i = isodd(sum(@view(state[1:i-1]))) ? -1 : 1;

        return sign_l*sign_k*sign_j*sign_i;
    end


    function sign(ket:: Int64, i:: Int64, j:: Int64, k:: Int64, l:: Int64, ket_fockSpace::Vector{Int})
        FermionAlgebra.WriteStateInFockSpace!(ket, ket_fockSpace);

        # We need to reverse "ket_fockSpace" because program starts counting postions from left to right 
        reverse!(ket_fockSpace)

        sign = GetSignOfOperatorPermutation(i, j, k, l, ket_fockSpace);
        return sign;
    end
    


    function Ĥ(L::Int64, U::Array{Float64, 4}, isSparse:: Bool = true)
        N::Int64 = Int(L÷2);
        D::Int64 = binomial(L,L÷2);

        c⁺ = FermionAlgebra.GetMatrixRepresentationOfOperator("c⁺", isSparse);
        c  = FermionAlgebra.GetMatrixRepresentationOfOperator("c", isSparse);
        id = FermionAlgebra.GetMatrixRepresentationOfOperator("id", isSparse);

        normalization = AnaliticalNormOfHamiltonianAveraged(L);
        ind = FermionAlgebra.IndecesOfSubBlock(L);

        tmp_storingVector = Vector{Int}(undef,L);

        rows = Vector{Int64}();
        cols = Vector{Int64}();
        vals = Vector{Float64}();


        l̂(i,j,k) = i==k ? j : k+1;

        cᵢ⁺cⱼ⁺cₖcₗ = fill(id, L); 
        # Interection term
        for i=1:L, j=i+1:L, k=i:L, l= l̂(i,j,k):L

            # Order of those products of operators is important so dont change it!
            cᵢ⁺cⱼ⁺cₖcₗ[i] *= c⁺;
            cᵢ⁺cⱼ⁺cₖcₗ[j] *= c⁺;
            cᵢ⁺cⱼ⁺cₖcₗ[k] *= c;
            cᵢ⁺cⱼ⁺cₖcₗ[l] *= c;

            # We are doing this because we need to determine a sign for each contribution
            # matrix = foldl(kron, cᵢ⁺cⱼ⁺cₖcₗ)[ind,ind]
            matrixElements = findall(x -> x==1 , foldl(kron, cᵢ⁺cⱼ⁺cₖcₗ)[ind,ind] )

            rows_ij = map(elm -> elm[1], matrixElements); 
            cols_ij = map(elm -> elm[2], matrixElements); 
            vals_ij = map(elm -> sign(ind[elm[2]]-1, i, j, k, l, tmp_storingVector) * 4*U[L+1-i,L+1-j,L+1-k,L+1-l] ./ normalization, matrixElements); 


            append!(rows, (i,j) == (k,l) ? rows_ij : vcat(rows_ij, cols_ij));
            append!(cols, (i,j) == (k,l) ? cols_ij : vcat(cols_ij, rows_ij));
            append!(vals, (i,j) == (k,l) ? vals_ij : vcat(vals_ij, conj.(vals_ij)));

            cᵢ⁺cⱼ⁺cₖcₗ[i] = cᵢ⁺cⱼ⁺cₖcₗ[j] = cᵢ⁺cⱼ⁺cₖcₗ[k] = cᵢ⁺cⱼ⁺cₖcₗ[l] = id;
        end
        
        
        return isSparse ? sparse(rows, cols, vals, D, D) : Matrix(sparse(rows, cols, vals, D, D));
    end 
end
