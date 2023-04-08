
# To še razmisl kakao boš naredu... Nisem zdovoljen z obliko funckcih (input/output parameteri)..
module ChaosIndicators
    using SparseArrays;
    using LinearAlgebra;
    using Combinatorics
    using ITensors


    include("../../Helpers/FermionAlgebra.jl");
    using .FermionAlgebra;

    include("../../Hamiltonians/H2.jl");
    using .H2;

    include("../../Hamiltonians/H4.jl");
    using .H4;

    include("./PrivateFunctions/EntaglementEntropy.jl");
    using .EE;

    include("./PrivateFunctions/SpectralFormFactor.jl");
    using .SFF;
    
    

    """
    Analitical value for spectral form factor (SSF) for GOE ansemble (for τ<1).
    """
    Kgoe(τ) = τ<=1 ? 2*τ - τ*log(1 + 2*τ) : 2 - τ*log((2*τ + 1)/(2*τ - 1));


    Sᵢ(x) = - abs(x)^2 * log(abs(x)^2)


    qc(L, q₀, q₁) = q₀ + q₁*L;

    ξkbt(q, q_c, b) = exp(b/√(q - q_c));



    global function InformationalEntropy(L:: Int64, q′s:: Vector{Float64}, numberOfIterations:: Int64)
        D = binomial(L,L÷2);
        η = 0.3;
        i₁ = Int((D - (D*η)÷1)÷2);
        i₂ = Int(i₁ + (D*η)÷1);


        ∑Sₘ = zeros(Float64, length(q′s));
        for (i,q) in enumerate(q′s)
            print("-",i)
            for j in 1:numberOfIterations 
                params2 = H2.Params(L);
                params4 = H4.Params(L);
                
                H₂ = H2.Ĥ(params2);
                H₄ = H4.Ĥ(params4);

                ϕ = eigvecs( Symmetric(Matrix(H₂ .+ q .* H₄)) );
                ϕ  = ϕ[:, i₁:i₂];
                Sₘ = sum( map(x -> Sᵢ(x), ϕ) , dims=1);

                ∑Sₘ[i] += sum(Sₘ)/length(Sₘ);
            end 
        end

        return (∑Sₘ./numberOfIterations)./log(0.48*D);
    end

    global function LevelSpacingRatio(L, q′s, numberOfIterations)
        r̲ = zeros(Float64, length(q′s))
        D = binomial(L,L÷2);
        η = 0.2;
        i₁ = Int((D - (D*η)÷1)÷2);
        i₂ = Int(i₁ + (D*η)÷1);

        # Here we go throu every value of disorder
        for (i, q) in enumerate(q′s)
            print("-",i) 

            #---------------- In this for loop we average over more calculations of "r" for diffrent uniform distributions of disorer. 
            for j in 1:numberOfIterations
                params2 = H2.Params(L);
                params4 = H4.Params(L);
                
                H₂ = H2.Ĥ(params2)
                H₄ = H4.Ĥ(params4)

                λ = eigvals( Symmetric(Matrix(H₂ .+ q.*H₄)) )[i₁:i₂]
                δ = λ[2:end] .- λ[1:end-1]
                r = map((x,y) -> min(x,y)/max(x,y), δ[1:end-1], δ[2:end] )
                r̲[i] += sum(r) / length(r)
            end 
        end
        return r̲ ./ numberOfIterations
    end
    
    global function EntanglementEntropy(L::Int64, q′s::Vector{Float64}, maxNumbOfIter::Int64, permutations:: Vector{Vector{Int64}}, d::Int64=2, cutoff::Real=10^-7, maxdim::Int64= 10)
        D = binomial(L,L÷2);
        η = 0.2;
        i₁ = Int((D - (D*η)÷1)÷2);
        i₂ = Int(i₁ + (D*η)÷1);

        ind = FermionAlgebra.IndecesOfSubBlock(L,1/2);


        E′s = zeros(Float64, (length(permutations), length(q′s)))

        for (j,q) in enumerate(q′s)

            E = [0., 0.];

            for k in 1:maxNumbOfIter
                params2 = H2.Params(L);
                params4 = H4.Params(L);
                
                H₂ = H2.Ĥ(params2);
                H₄ = H4.Ĥ(params4);

                ϕ = eigvecs( Symmetric(Matrix(H₂ .+ q .* H₄)) )[:, i₁:i₂];

                E0 = [0., 0.]; 
                for ψ in eachcol(ϕ)
                    ψ′ = zeros(Float64, Int(2^L));
                    ψ′[ind] = ψ;
                
                    E0[:] .+= EE.EntanglementEntropy(ψ′, permutations, L) ./ length(eachcol(ϕ));
                end

                E .+= E0./maxNumbOfIter;
            end

            E′s[:,j] .= E
        end
        
        return E′s;        
    end



    function ĝ(L::Int64, q::Float64, N::Int64, τ′s:: Vector{Float64})

        println("   q = ",q,)
        coeffs, Es′s = SFF.K̂_data(N, L, q);
        println("      coeffs and Es are calculated.");

        Ks = map(τ -> SFF.K̂(τ, coeffs, Es′s), τ′s);
        println("      K′s are calculated.");

        τ_Th:: Float64 = 0.;
        τ_H:: Float64 = 0.;
        g:: Float64 = 0.;

        doesτexist:: Bool = true;

        try
            println("       Trying...")
            τ_Th = SFF.τ̂_Th(Ks, τ′s);
            τ_H = SFF.τ̂_H(Es′s);
            g = SFF.ĝ(τ_Th, τ_H);

            doesτexist = true;
        catch errorMessage
            println("       Catched error: ", errorMessage);

            errorNumber = 666.;
            τ_Th = errorNumber;
            τ_H = errorNumber;
            g = errorNumber;
            doesτexist = false;
        finally
            println("       Finally")
            return coeffs, Es′s, Ks, τ_Th, τ_H, g, doesτexist;
        end
        

    end


    function ĝ2(L::Int64, q::Float64, N::Int64, Nτ::Int64, η:: Float64)

        println("   q = ", q,)
        coeffs′s, Es′s = SFF.K̂_data(N, L, q);
        println("      coeffs and Es are calculated.");


        K′s, τ′s, τ_Th =  SFF.τ̂_Th(Nτ, coeffs′s, Es′s, η)
        t_H = SFF.t̂_H(Es′s);
        t_Th = SFF.t̂_Th(τ_Th, t_H) 
        g = SFF.ĝ(τ_Th);

        
        
        return coeffs′s, Es′s, K′s, τ′s, τ_Th, t_Th, t_H, g
    end


    # function ĝ(L::Int64, q′s::Vector{Float64}, N::Int64, Nτ::Int64=5000)
    #     x = LinRange(-5, 0, Nτ);
    #     τ′s = 10 .^(x);

    #     coefs′s= Vector{Any}();
    #     Es′s   = Vector{Any}();
    #     Ks′s   = Vector{Any}();
    #     τ_Th′s = Vector{Float64}();
    #     τ_H′s  = Vector{Float64}();
    #     g′s    = Vector{Float64}();
    #     indecesWhere_τTh_IsValid = Vector{Float64}();

    #     for (i,q) in enumerate(q′s)
    #         println("   q = ",q,)
    #         coeffs, Es = SFF.K̂_data(N, L, q);
    #         println("      coeffs and Es are calculated.");

    #         Ks = map(τ -> SFF.K̂(τ, coeffs, Es), τ′s);
    #         println("      K′s are calculated.");

    #         push!(coefs′s, coeffs);
    #         push!(Es′s, Es);
    #         push!(Ks′s, Ks);

    #         try
    #             println("      Trying...")
    #             τ_Th = SFF.τ̂_Th(Ks, τ′s);
    #             println("      τ_Th calculated")

    #             τ_H = SFF.τ̂_H(Es);
    #             println("      τ_H  calculated")
    #             g = SFF.ĝ(τ_Th, τ_H);
    #             println("      g  calculated")

    #             push!(τ_Th′s, τ_Th);
    #             push!(τ_H′s, τ_H);
    #             push!(g′s, g);
    #             push!(indecesWhere_τTh_IsValid, i);
    #         catch errorMessage
    #             println("          catched error: ", errorMessage);


    #             errorNumber = 666.;
    #             push!(τ_Th′s, errorNumber);
    #             push!(τ_H′s , errorNumber);
    #             push!(g′s   , errorNumber);
    #         end

    #     end

    #     return coefs′s, Es′s, Ks′s, τ_Th′s, τ_H′s, g′s, indecesWhere_τTh_IsValid;
    # end


end
