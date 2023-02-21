module H4
    using SparseArrays;
    using Distributions;
    using LinearAlgebra;

    include("../Helpers/FermionAlgebra.jl");
    using .FermionAlgebra; 

    struct Params_H4
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

        return Params_H4(L, S, U, μ, deviation, mean)
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


    function GetSignOfOperatorPermutation(creationOperator, inhalationOperator, state)
        statePositions = findall(x->x==1, state);
        sign_right = FermionAlgebra.GetSign( vcat(inhalationOperator, statePositions) );

        deleteat!(statePositions, findall(x -> x∈inhalationOperator , statePositions))
        sign_left = FermionAlgebra.GetSign( vcat(creationOperator, statePositions) );

        return sign_left*sign_right;
    end

    function CanOperatorActOnState(state, i, j, k, l)
        state′ = copy(state);

        is_l_positionTaken = state′[l]==1;
        state′[l] = 0;
        is_k_positionTaken = state′[k]==1;
        state′[k] = 0;
        is_j_positionEmpty = state′[j]==0;
        state′[j] = 1;
        is_i_positionEmpty = state′[i]==0;
        state′[i] = 1;

        condition = is_i_positionEmpty && is_j_positionEmpty && is_k_positionTaken && is_l_positionTaken;


        return state′, condition;
    end

    global function Ĥ4(params:: Params_H4, isSparse:: Bool = true)
        L = params.L;
        states = FermionAlgebra.IndecesOfSubBlock(L, params.S) .- 1; # Are already sorted
        D = binomial(L, Int(L/2));

        H₄ = isSparse ? spzeros(D,D) : zeros(Float64,(D,D));
        
        # Interaction term
        for n in eachindex(states)
            n_state = states[n];
            n_FockSpace = FermionAlgebra.WriteStateInFockSpace(n_state, params.L, params.S);

            for m in eachindex(states[n:end])
                m = m + n - 1;
                m_state = states[m];
                m_FockSpace = FermionAlgebra.WriteStateInFockSpace(m_state, params.L, params.S);

                if sum(abs.(n_FockSpace.-m_FockSpace))>4
                    continue
                end
                
                Hₙₘ = 0.;
                for i=1:L, j=i+1:L, k=1:L, l=k+1:L
                    m′_FockSpace, condition = CanOperatorActOnState(m_FockSpace, i ,j ,k ,l)

                    if condition
                        m′_state = sum([value*2^(index-1) for (index, value) in enumerate(m′_FockSpace)])
                    
                        sign = GetSignOfOperatorPermutation([i,j], [k,l], m_FockSpace);

                        Hₙₘ += n_state == m′_state ? sign * 4 * params.U̲[i,j,k,l] / (2*L)^(3/2) : 0.
                    end
                end

                if Hₙₘ!=0.
                    H₄[n,m] = Hₙₘ;
                    H₄[m,n] = conj(Hₙₘ); #konjugiran element, ker gremo samo po zgornjem delu hamiltonke.                
                end
            end
        end
        
        H₄ -= isSparse ? sparse(Matrix{Float64}(I, D,D)) .* params.μ * params.L / 2 : Matrix{Float64}(I, D,D) .* params.μ * params.L / 2 ;

        
        return H₄;
    end 

end


# L= 6;
# S=1/2;
# μ=0;
# mean=0;
# deviation = 1;


# params = H4.GetParams(L, S, μ, mean, deviation)


# H = H4.Ĥ4(params, false)


# display(H)