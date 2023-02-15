module H2
    using SparseArrays;
    using Distributions;
    using LinearAlgebra;

    include("../Helpers/FermionAlgebra.jl");
    using .FermionAlgebra;


    struct Params_SKY2
        L:: Int64;
        S:: Real;
        μ:: Float64;
        t̲:: Matrix{Real};
        deviation:: Real;
        mean:: Real;
    end

    global function GetParams_SKY2(L:: Int64, S:: Real, μ:: Real, mean:: Real, deviation:: Real)   
        t = Matrix{Float64}(undef, (L,L))

        for i=1:L, j=1:i
            t[i,j] = rand(Normal(mean, deviation));
            
            if i!=j
                t[j,i] = t[i,j];
            end
        end

        return Params_SKY2(L, S, μ, t, deviation, mean)
    end

    function GetTypeOfOperatorOnSite(position:: String)
        typeOfOperatorOnSite = Dict{String,String}(
            "i" => "c⁺", 
            "j" => "c",
        );

        return typeOfOperatorOnSite[position]
    end

    function AnaliticalExpressionForAverageOfSqueredHamiltonian(params)
        stateNumbers = FermionAlgebra.IndecesOfSubBlock(params.L, params.S) .- 1;
        states = FermionAlgebra.WriteStateInFockSpace.(stateNumbers, params.L, params.S);

        t = params.t̲
        D = binomial(params.L, Int(params.L/2));
        norm = 0.

        # Covers first term of equation
        for state in states
            positionsOfParticles = findall(x -> x==1 ,state);

            for i in positionsOfParticles
                for k in positionsOfParticles
                    norm  += t[i,i]*t[k,k]
                end
            end
        end 

        # Covers second term of a equation
        for state in states
            positionsOfParticles = findall(x -> x==1 ,state);

            for i in positionsOfParticles
                for j in 1:params.L
                    if i != j
                        norm  += t[i,j]*t[j,i]
                    end
                end
            end
        end


        # Covers third term of a equation
        for state in states
            positionsOfParticles = findall(x -> x==1 ,state);

            for i in positionsOfParticles
                for j in positionsOfParticles
                    if i != j
                        norm  -= t[i,j]*t[j,i]
                    end
                end
            end
        end 

        return norm / (D*params.L);
    end

    function AnaliticalExpressionForSquaredAverageOfHamiiltonian(params)
        stateNumbers = FermionAlgebra.IndecesOfSubBlock(params.L, params.S) .- 1;
        states = FermionAlgebra.WriteStateInFockSpace.(stateNumbers, params.L, params.S);

        t = params.t̲
        D = binomial(params.L, Int(params.L/2));
        norm = 0.

        for state in states
            positionsOfParticles = findall(x -> x==1 ,state);

            for i in positionsOfParticles
                norm += t[i,i]
            end
        end

        return ( norm / (D*√params.L) )^2
    end

    global function AnaliticalNormOfHamiltonian(params)
        H²_avg =  AnaliticalExpressionForAverageOfSqueredHamiltonian(params);
        H_avg² =  AnaliticalExpressionForSquaredAverageOfHamiiltonian(params);
        return H²_avg - H_avg²;
    end

    global function AnaliticalNormOfHamiltonianAveraged(deviation, L, μ)
        norm = 0.5 * deviation^2 * ( L/2 +1 ) + 0.5 * μ^2 * L;
        return norm;
    end

    global function Ĥ(params:: Params_SKY2, isSparse:: Bool = true)
        dim = Int(2*params.S + 1);
        L = params.L;

        H₂ = isSparse ? spzeros(dim^L, dim^L) : zeros(Float64,(dim^L, dim^L));

        opᵢ = FermionAlgebra.GetMatrixRepresentationOfOperator(GetTypeOfOperatorOnSite("i"), params.S, isSparse);
        opⱼ = FermionAlgebra.GetMatrixRepresentationOfOperator(GetTypeOfOperatorOnSite("j"), params.S, isSparse);
        id  = FermionAlgebra.GetMatrixRepresentationOfOperator("id", params.S, isSparse);
        
        # Hopping term
        for i=1:L, j=1:L
            # println(i, " ", j)
            cᵢ⁺cⱼ = fill(id, L);   
            # Order of those products of operators oisimporattn so dont change it!
            cᵢ⁺cⱼ[i] *=  opᵢ;
            cᵢ⁺cⱼ[j] *=  opⱼ;

            H₂ += params.t̲[i,j] .* foldl(kron, cᵢ⁺cⱼ) ./ √L
        end

        # Chemical potential term
        for i=1:L
            cᵢ⁺cᵢ = fill(id, L);   
            cᵢ⁺cᵢ[i] *=  FermionAlgebra.GetMatrixRepresentationOfOperator("n", params.S, isSparse);;

            H₂ -= params.μ .* foldl(kron, cᵢ⁺cᵢ);
        end

        ind = FermionAlgebra.IndecesOfSubBlock(L, params.S);
        return H₂[ind,ind];
    end 

end
