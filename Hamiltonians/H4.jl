module H4
    using SparseArrays;
    using Distributions;
    using LinearAlgebra;

    include("../Helpers/FermionAlgebra.jl");
    using .FermionAlgebra; 

    struct Params_SKY4
        L:: Int64;
        S:: Real;
        U̲:: Array{Real}; 
        μ:: Float64;       
        deviation:: Real;
        mean:: Real;
    end

    global function GetParams(L:: Int64, S:: Real, μ:: Real, mean:: Real, deviation:: Real)
        U = zeros(Float64, (L,L,L,L))

        for i=1:L, j=1:L, k=1:L, l=1:L
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

            # println(U[i,j,k,l], " ", U[j,i,k,l], " ", U[i,j,l,k])
        end

        return Params_SKY4(L, S, U, μ, deviation, mean)
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

    global function Ĥ(params:: Params_SKY4, isSparse:: Bool = true) 
        dim = Int(2*params.S + 1);
        L = params.L;

        H₄ = isSparse ? spzeros(dim^L, dim^L) : zeros(Float64,(dim^L, dim^L));

        opᵢ = FermionAlgebra.GetMatrixRepresentationOfOperator(GetTypeOfOperatorOnSite("i"), params.S, isSparse);
        opⱼ = FermionAlgebra.GetMatrixRepresentationOfOperator(GetTypeOfOperatorOnSite("j"), params.S, isSparse);
        opₖ = FermionAlgebra.GetMatrixRepresentationOfOperator(GetTypeOfOperatorOnSite("k"), params.S, isSparse);
        opₗ = FermionAlgebra.GetMatrixRepresentationOfOperator(GetTypeOfOperatorOnSite("l"), params.S, isSparse);
        id = FermionAlgebra.GetMatrixRepresentationOfOperator("id", params.S, isSparse);
        
        # Hopping term
        for i=1:L, j=1:L, k=1:L, l=1:L
            # print(i," ",j," ",k," ",l," -> ")
            cᵢ⁺cⱼ⁺cₖcₗ = fill(id, L);   

            # Order of those products of operators oisimporattn so dont change it!
            cᵢ⁺cⱼ⁺cₖcₗ[i] *= opᵢ;
            cᵢ⁺cⱼ⁺cₖcₗ[j] *= opⱼ;
            cᵢ⁺cⱼ⁺cₖcₗ[k] *= opₖ;
            cᵢ⁺cⱼ⁺cₖcₗ[l] *= opₗ;

            signOfPermutation = FermionAlgebra.GetSign([i, j, k, l]);# println("[",i,", ",j,", ",k,", ",l,"]  =>", signOfPermutation);

            sign1 = i<=j ? +1 : -1;
            sign2 = k<=l ? +1 : -1;

            
            H₄ += params.U̲[i,j,k,l] .* foldl(kron, cᵢ⁺cⱼ⁺cₖcₗ) ./ (2*L)^(3/2) *sign1*sign2# .*signOfPermutation
        end
        

        # Chemical potential term
        if params.μ != 0
            println("calculating chemical potential");
            for i=1:L 
                cᵢ⁺cᵢ = fill(id, L);   
                cᵢ⁺cᵢ[i] *=  FermionAlgebra.GetMatrixRepresentationOfOperator("n", params.S, isSparse);;
                
                H₄ -= params.μ .* foldl(kron, cᵢ⁺cᵢ);
            end
        end

        ind = FermionAlgebra.IndecesOfSubBlock(L, params.S);
        return H₄[ind,ind];
    end 

end