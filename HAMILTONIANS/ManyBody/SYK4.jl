module SYK4
    using SparseArrays;
    using Distributions;
    using LinearAlgebra;

    include("../../Helpers/FermionAlgebra.jl");
    using .FermionAlgebra;

    struct Params
        L:: Int64;
        U:: Array{Real}; 
        S:: Real;
        μ:: Float64;       
        deviation:: Real;
        mean:: Real;

        function Params(L:: Int64, U̲::Union{Array{Float64}, Missing} = missing, S::Real=1/2, μ::Real=0, mean::Real=0., deviation::Real=1.)
            
            if isequal(U̲, missing)
                U = zeros(Float64, (L,L,L,L))

                for i=1:L, j=i+1:L, k=1:L, l=k+1:L
                    if U[i,j,k,l] == 0 && i!=j && k!=l
                        a = rand(Normal(mean, deviation));
                        U[i,j,k,l] = deepcopy(a);
                        U[i,j,l,k] = deepcopy(a)*(-1);
        
                        U[j,i,k,l] = deepcopy(a)*(-1);
                        U[j,i,l,k] = deepcopy(a);
                        
                        U[k,l,i,j] = deepcopy(a);
                        U[k,l,j,i] = deepcopy(a)*(-1);
        
                        U[l,k,i,j] = deepcopy(a)*(-1);
                        U[l,k,j,i] = deepcopy(a);
                    end
                end
            else
                U = U̲[1:L, 1:L, 1:L, 1:L]
            end

            new(L, U, S, μ, deviation, mean)
        end

    end

    function GetTypeOfOperatorOnSite(position:: String)
        typeOfOperatorOnSite = Dict{String,String}(
            "i" => "c⁺", 
            "j" => "c⁺",
            "k" => "c", 
            "l" => "c",
        );

        return typeOfOperatorOnSite[position]
    end


    global function AnaliticalNormOfHamiltonianAveraged(params:: Params)
        L = params.L;
        U = params.deviation;
        norm2 = L*(L-2)*(L^2 + 6L + 8 - 2*(L-2)/(L-1))*U^2 / 4 ;
        return √norm2;
    end



    function GetSignOfOperatorPermutation(i_cre, j_cre, k_anh, l_anh, state)
        # println("($(i_cre), $(j_inh)):", state)

        sign_l = isodd(sum(state[1:l_anh-1])) ? -1 : 1;
        state[l_anh] = 0;
        sign_k = isodd(sum(state[1:k_anh-1])) ? -1 : 1;
        state[k_anh] = 0;

        sign_j = isodd(sum(state[1:j_cre-1])) ? -1 : 1;
        state[j_cre] = 0;

        sign_i = isodd(sum(state[1:i_cre-1])) ? -1 : 1;

        return sign_l*sign_k*sign_j*sign_i;
    end


    function sign(ket:: Int64, i_cre:: Int64, j_cre:: Int64, k_inh:: Int64, l_inh:: Int64, params:: Params)
        ket_fockSpace = FermionAlgebra.WriteStateInFockSpace(ket, params.L, params.S)

        # We need to reverse "ket_fockSpace" because program starts counting postions from left to right 
        sign = GetSignOfOperatorPermutation(i_cre, j_cre, k_inh, l_inh, reverse(ket_fockSpace));
        return sign;
    end


    global function Ĥ(params:: Params, isSparse:: Bool = true) 
        L = params.L;
        D = binomial(L,L÷2)

        opᵢ = FermionAlgebra.GetMatrixRepresentationOfOperator(GetTypeOfOperatorOnSite("i"), params.S, isSparse);
        opⱼ = FermionAlgebra.GetMatrixRepresentationOfOperator(GetTypeOfOperatorOnSite("j"), params.S, isSparse);
        opₖ = FermionAlgebra.GetMatrixRepresentationOfOperator(GetTypeOfOperatorOnSite("k"), params.S, isSparse);
        opₗ = FermionAlgebra.GetMatrixRepresentationOfOperator(GetTypeOfOperatorOnSite("l"), params.S, isSparse);
        id = FermionAlgebra.GetMatrixRepresentationOfOperator("id", params.S, isSparse);
        
        normalization = AnaliticalNormOfHamiltonianAveraged(params);
        ind = FermionAlgebra.IndecesOfSubBlock(L, params.S);


        rows = Vector{Int64}();
        cols = Vector{Int64}();
        vals = Vector{Float64}();


        l̂(i,j,k) = i==k ? j : k+1;

        # Interection term
        for i=1:L, j=i+1:L, k=i:L, l= l̂(i,j,k):L
            cᵢ⁺cⱼ⁺cₖcₗ = fill(id, L); 

            # Order of those products of operators oisimporattn so dont change it!
            cᵢ⁺cⱼ⁺cₖcₗ[i] *= opᵢ;
            cᵢ⁺cⱼ⁺cₖcₗ[j] *= opⱼ;
            cᵢ⁺cⱼ⁺cₖcₗ[k] *= opₖ;
            cᵢ⁺cⱼ⁺cₖcₗ[l] *= opₗ;

            #We are doing this because we need to determine a sign for each contribution
            # matrix = foldl(kron, cᵢ⁺cⱼ⁺cₖcₗ)[ind,ind]
            matrixElements = findall(x -> x==1 , foldl(kron, cᵢ⁺cⱼ⁺cₖcₗ)[ind,ind] )

            rows_ij = map(elm -> elm[1], matrixElements); 
            cols_ij = map(elm -> elm[2], matrixElements); 
            vals_ij = map(elm -> sign(ind[elm[2]]-1, i, j, k, l, params) * 4*params.U[L+1-i,L+1-j,L+1-k,L+1-l] ./ normalization, matrixElements); 


            append!(rows, (i,j) == (k,l) ? rows_ij : vcat(rows_ij, cols_ij));
            append!(cols, (i,j) == (k,l) ? cols_ij : vcat(cols_ij, rows_ij));
            append!(vals, (i,j) == (k,l) ? vals_ij : vcat(vals_ij, conj.(vals_ij)));
        end
        
        
        return isSparse ? sparse(rows, cols, vals, D, D) : Matrix(sparse(rows, cols, vals, D, D));
    end 


end
