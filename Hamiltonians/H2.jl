module H2
    using SparseArrays;
    using Distributions;
    using LinearAlgebra;

    include("../Helpers/FermionAlgebra.jl");
    using .FermionAlgebra;


    struct Params
        L:: Int64;
        S:: Real;
        μ:: Float64;
        t:: Matrix{Real};
        deviation:: Real;
        mean:: Real;
    end

    global function GetParams(L:: Int64, S:: Real, μ:: Real, mean:: Real, deviation:: Real)   
        t = Matrix{Float64}(undef, (L,L))

        for i=1:L, j=1:i
            t[i,j] = rand(Normal(mean, deviation));
            
            if i!=j
                t[j,i] = t[i,j];
            end
        end

        return Params(L, S, μ, t, deviation, mean)
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

        t = params.t
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

        t = params.t
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

        # println("AnaliticalNormOfHamiltonian: ", params.L, " ", params.deviation)
        # println("   ",H²_avg, " - ", H_avg²)
        return H²_avg - H_avg²;
    end

    global function AnaliticalNormOfHamiltonianAveraged(deviation, L, μ)
        norm = deviation^2 * L/4 ;
        # println("AnaliticalNormOfHamiltonianAveraged: ", L, " ", deviation)
        # println("   ",norm)
        return norm;
    end

    global function AnaliticalNormOfHamiltonianAveraged2(deviation, L, μ)
        norm = deviation^2 * (L + 1)/4 ;
        # println("AnaliticalNormOfHamiltonianAveraged: ", L, " ", deviation)
        # println("   ",norm)
        return norm;
    end

    global function Ĥ_narobe(params:: Params, isSparse:: Bool = true)
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


    function GetSignOfOperatorPermutation(creationOperator, inhalationOperator, state)
        statePositions = findall(x->x==1, state);
        sign_right = FermionAlgebra.GetSign( vcat(inhalationOperator, statePositions) );

        deleteat!(statePositions, findall(x -> x∈inhalationOperator , statePositions))
        sign_left = FermionAlgebra.GetSign( vcat(creationOperator, statePositions) );

        return sign_left*sign_right;
    end

    function CanOperatorActOnState(state, i, j)
        state′ = copy(state);

        is_l_positionTaken = state′[j]==1;
        state′[j] = 0;
        is_i_positionEmpty = state′[i]==0;
        state′[i] = 1;

        condition = is_i_positionEmpty && is_l_positionTaken;

        return state′, condition;
    end

    global function Ĥ(params:: Params, isSparse:: Bool = true)
        L = params.L;
        states = FermionAlgebra.IndecesOfSubBlock(L, params.S) .- 1; # Are already sorted
        D = binomial(L, Int(L/2));

        H₂ = isSparse ? spzeros(D,D) : zeros(Float64,(D,D));
        
        # Interaction term
        for n in eachindex(states)
            n_state = states[n];
            n_FockSpace = FermionAlgebra.WriteStateInFockSpace(n_state, params.L, params.S);

            # println(n_FockSpace)

            for m in eachindex(states[n:end])
                m = m + n - 1;
                m_state = states[m];
                m_FockSpace = FermionAlgebra.WriteStateInFockSpace(m_state, params.L, params.S);

                # println("  <",n_FockSpace,"| H |", m_FockSpace,">")

                if sum(abs.(n_FockSpace.-m_FockSpace))>2
                    continue
                end
                
                Hₙₘ = 0.;
                for i=1:L, j=1:L
                    m′_FockSpace, condition = CanOperatorActOnState(m_FockSpace, i ,j)

                    # println("   ",i, " ", j)

                    if condition
                        m′_state = sum([value*2^(index-1) for (index, value) in enumerate(m′_FockSpace)])
                    
                        sign = GetSignOfOperatorPermutation([i], [j], m_FockSpace);
                        Hₙₘ += n_state == m′_state ? sign * params.t[i,j] / (L)^(1/2) : 0.

                        # if n_state == m′_state 
                        #     Hₙₘ +=  sign * params.t[i,j] / (L)^(1/2);
                        #     println("    Hₙₘ += ", round(sign * params.t[i,j] / (L)^(1/2), digits=5), "  ->  ", Hₙₘ, "   (", params.t[i,j] ,")", "  sign = ", sign)
                        # end
                    end
                end

                if Hₙₘ!=0.
                    H₂[n,m] = Hₙₘ;
                    H₂[m,n] = conj(Hₙₘ); #konjugiran element, ker gremo samo po zgornjem delu hamiltonke.                
                end
            end
        end
        
        H₂ -= isSparse ? sparse(Matrix{Float64}(I, D,D)) .* params.μ * params.L / 2 : Matrix{Float64}(I, D,D) .* params.μ * params.L / 2 ;

        
        return H₂;
    end 

end



# L= 4;
# S=1/2;
# μ=0;
# mean=0;
# deviation = 1;


# params = H2.GetParams_SKY2(L, S, μ, mean, deviation)


# H = H2.Ĥ(params, true)

# display(params.t)
# println()
# println()



# display(H)


# println()
# println()
# display(H .- H' )



