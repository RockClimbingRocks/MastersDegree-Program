module H4
    using SparseArrays;
    using Distributions;
    using LinearAlgebra;

    include("../Helpers/FermionAlgebra.jl");
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

    global function Ĥ_slowOne(params:: Params, isSparse:: Bool = true)
        L = params.L;
        states = FermionAlgebra.IndecesOfSubBlock(L, params.S) .- 1; # Are already sorted
        D = binomial(L, Int(L/2));

        H₄ = isSparse ? spzeros(D,D) : zeros(Float64,(D,D));

        normalization = AnaliticalNormOfHamiltonianAveraged(params);
        
        # Interaction term
        for n in eachindex(states)
            n_state = states[n];
            n_FockSpace = FermionAlgebra.WriteStateInFockSpace(n_state, params.L, params.S);

            for m in eachindex(states[n:end])
                m = m + n - 1;
                m_state = states[m];
                m_FockSpace = FermionAlgebra.WriteStateInFockSpace(m_state, params.L, params.S);

                # println("  <",n_FockSpace,"| H |", m_FockSpace,">")


                if sum(abs.(n_FockSpace.-m_FockSpace))>4
                    continue
                end
                
                Hₙₘ = 0.;
                for i=1:L, j=i+1:L, k=1:L, l=k+1:L
                    m′_FockSpace, condition = CanOperatorActOnState(m_FockSpace, i ,j ,k ,l)

                    # println("   ",i, " ", j, " ", k, " ", l)


                    if condition
                        m′_state = sum([value*2^(index-1) for (index, value) in enumerate(m′_FockSpace)])
                    
                        sign = GetSignOfOperatorPermutation([i,j], [k,l], m_FockSpace);

                        if n_state == m′_state 
                            Hₙₘ +=  sign * 4 * params.U[i,j,k,l] / normalization;
                            # println("    Hₙₘ += ", round(sign * 4 * params.U[i,j,k,l] / normalization, digits=5), "  ->  ", Hₙₘ, "   (", params.U[i,j,k,l] ,")", "  sign = ", sign)
                        end

                        # Hₙₘ += n_state == m′_state ? sign * 4 * params.U[i,j,k,l] / normalization : 0.
                    end
                end

                if Hₙₘ!=0.
                    # println("H₄[$(n),$(m)] = H₄[$(m),$(n)] = ", Hₙₘ);
                    H₄[n,m] = Hₙₘ;
                    H₄[m,n] = conj(Hₙₘ); #konjugiran element, ker gremo samo po zgornjem delu hamiltonke.                
                end
            end
        end
        
        # H₄ -= isSparse ? sparse(Matrix{Float64}(I, D,D)) .* params.μ * params.L / 2 : Matrix{Float64}(I, D,D) .* params.μ * params.L / 2 ;

        
        return H₄;
    end
    
    global function Ĥ(params:: Params, isSparse:: Bool = true) 
        L = params.L;
        D = binomial(L,L÷2)

        H₄ = isSparse ? spzeros(D, D) : zeros(Float64,(D, D));

        opᵢ = FermionAlgebra.GetMatrixRepresentationOfOperator(GetTypeOfOperatorOnSite("i"), params.S, isSparse);
        opⱼ = FermionAlgebra.GetMatrixRepresentationOfOperator(GetTypeOfOperatorOnSite("j"), params.S, isSparse);
        opₖ = FermionAlgebra.GetMatrixRepresentationOfOperator(GetTypeOfOperatorOnSite("k"), params.S, isSparse);
        opₗ = FermionAlgebra.GetMatrixRepresentationOfOperator(GetTypeOfOperatorOnSite("l"), params.S, isSparse);
        id = FermionAlgebra.GetMatrixRepresentationOfOperator("id", params.S, isSparse);
        
        normalization = AnaliticalNormOfHamiltonianAveraged(params);
        ind = FermionAlgebra.IndecesOfSubBlock(L, params.S);


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
            matrix = foldl(kron, cᵢ⁺cⱼ⁺cₖcₗ)[ind,ind]
            matrixElements = findall(x -> x==1 ,matrix)

            for matrixElement in matrixElements
                # bra = ind[matrixElement[1]]-1;
                ket = ind[matrixElement[2]]-1;
                ket_fockSpace = FermionAlgebra.WriteStateInFockSpace(ket, params.L, params.S)

                # We need to reverse "ket_fockSpace" because program starts counting postions from left to right 
                sign = GetSignOfOperatorPermutation([i,j], [k,l], reverse(ket_fockSpace));
                matrix[matrixElement] = sign;
            end
          
            # We need to use "U[L+1-i,L+1-j,L+1-k,L+1-l]" and not "U[i,j,k,l]"  because program starts counting postions from left to right 
            a = 4*params.U[L+1-i,L+1-j,L+1-k,L+1-l] .* matrix ./ normalization;

            H₄ += (i,j) == (k,l) ? a : a .+ a'
        end
        
        return H₄;
    end 


end




# include("../Helpers/FermionAlgebra.jl");
# using .FermionAlgebra;


# L= 12 ;
# params = H4.Params(L)

# println(H4.AnaliticalNormOfHamiltonianAveraged(params))

# print("start")
# H = H4.Ĥ(params, false)
# print("end")

# println()
# println("indexi: ", FermionAlgebra.IndecesOfSubBlock(L, 1/2))
# println("states: ", FermionAlgebra.IndecesOfSubBlock(L, 1/2) .-1)

# println("\n\n")
# display(H2)

