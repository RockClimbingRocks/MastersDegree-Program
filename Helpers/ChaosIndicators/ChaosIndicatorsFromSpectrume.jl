
# To še razmisl kakao boš naredu... Nisem zdovoljen z obliko funckcih (input/output parameteri)..
module CI
    using SparseArrays;
    using LinearAlgebra;
    using Combinatorics;
    using ITensors;
    using Statistics;
    using Polynomials;

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


    global function LevelSpacingRatio(λ:: Vector{Float64}, η:: Float64 = 0.25):: Float64
        D = length(λ);
        i₁ = Int((D - (D*η)÷1)÷2);
        i₂ = Int(i₁ + (D*η)÷1);

        λ = λ[i₁:i₂];
        δ = λ[2:end] .- λ[1:end-1];
        r′s = map((x,y) -> min(x,y)/max(x,y), δ[1:end-1], δ[2:end] );

        return mean(r′s);
    end

    global function InformationalEntropy(ϕ:: Matrix{Float64}, η:: Float64 = 0.25):: Vector{Float64}
        D = size(ϕ)[1];
        i₁ = Int((D - (D*η)÷1)÷2);
        i₂ = Int(i₁ + (D*η)÷1);

        ϕ  = ϕ[:, i₁:i₂];
        Sₘ′s = sum( map(x -> Sᵢ(x), ϕ) , dims=1);

        return vec(Sₘ′s);
    end

    global function EntanglementEntropy(ϕ:: Matrix{Float64}, setA::Vector{Int64}, setB::Vector{Int64}, indices:: Vector{Int64}, η:: Float64 = 0.25, d::Int64=2, cutoff::Real=10^-7, maxdim::Int64= 10):: Vector{Float64}
        D = size(ϕ)[2];
        i₁ = Int((D - (D*η)÷1)÷2);
        i₂ = Int(i₁ + (D*η)÷1);

        L:: Int64 = length(setA) + length(setB);

        ϕ = ϕ[:, i₁:i₂];
        EE_ψ′s:: Vector{Float64} = Vector{Float64}(undef, size(ϕ)[2])

        for l in eachindex(ϕ[1,:])
            ψ = ϕ[:,l]
            ψ′ = zeros(Float64, Int(2^L));
            ψ′[indices] = ψ;
        
            EE_ψ′s[l] = EE.EntanglementEntropy(ψ′, setA, setB);
        end

        return EE_ψ′s;        
    end

    function K̂(E′s:: Vector{Float64}, τs::Vector{Float64}, η:: Float64 = 0.25)
        coeffs = SFF.ḡₙ_coeffs(E′s);

        K1_, K̄2_, A_, B̄_, Z_ = K̂_singleIteration(E′s, coeffs, τs, η)        
        
        return coeffs, K1_, K̄2_, A_, B̄_, Z_
    end


    function GetThoulessTimeEndOthers(Es′s::Vector{Vector{Float64}}, Nτ::Int64, η::Float64)
        coeffs′s::Vector{Vector{Float64}} = map(Es -> SFF.ḡₙ_coeffs(Es), Es′s);
        println(size(coeffs′s));
        println(size(coeffs′s[1]))


        # for i in eachindex(Es′s)
        #     coeffs = coeffs′s[i];
        #     Es = Es′s[i]

        #     ḡₙ = Polynomial(coeffs);

        #     ε′s = ḡₙ.(Es);
        #     ε̄ = mean(ε′s);      
        #     Γ = std(ε′s); 

        #     fig, ax = plt.subplots();

        #     Essss= LinRange(minimum(Es), maximum(Es), 1000);
        #     ax.plot(Es, ε′s, label = "dd", zorder=2, linewidth=4., color="palegreen");
        #     ax.plot(Essss, map(E->SFF.Ĝ(E, Es), Essss), label = "d", zorder=3, color="black");

        #     plt.show();
        # end
    



        τ′s, K′s, τ_Th, Kc′s, τ_Th_c, found_τTh, found_τThc =  SFF.τ̂_Th(Nτ, coeffs′s, Es′s, η);
        t_H = SFF.t̂_H(Es′s);
        t_Th = SFF.t̂_Th(τ_Th, t_H);
        g = SFF.ĝ(τ_Th);

        t_Th_c = SFF.t̂_Th(τ_Th_c, t_H); 
        g_c = SFF.ĝ(τ_Th_c);
    
        return coeffs′s , τ′s, t_H, τ_Th, K′s, t_Th, g, τ_Th_c, Kc′s, t_Th_c, g_c, found_τTh, found_τThc;
    end


    function GetK(Es′s::Vector{Vector{Float64}}, Nτ::Int64, η::Float64=0.4, n::Int64 = 5)
        K1:: Vector{Float64} = zeros(Float64, Nτ); 
        K̄2:: Vector{Complex} = zeros(Complex, Nτ); 
        A:: Float64 = 0.;
        B̄:: Complex = 0.;
        Z:: Float64 = 0.;


        min=-4;
        max= 0;
        N = Int((max-min)*Nτ÷1);
        x = LinRange(min, max, N);
        τs = 10 .^x;

        for (e,Es) in enumerate(Es′s)
            coeffs = ḡₙ_coeffs(Es, n);

            K1_, K̄2_, A_, B̄_, Z_ = SFF.K̂_singleIteration(Es, coeffs, τs, η);

            K1 .+= K1_ ./ numberOfIterations;
            K̄2 .+= K̄2_ ./ numberOfIterations;
            A  += A_  / numberOfIterations;
            B̄  += B̄_  / numberOfIterations;
            Z  += Z_  / numberOfIterations;
        end

        K2:: Vector{Float64} = @. abs(K̄2)^2;
        B:: Float64 = abs(B̄)^2;

        K:: Vector{Float64}  = @.  K1  / Z;
        Kc:: Vector{Float64} = @. (K1 - A*K2/B) / Z;
        return K, Kc;
    end

    function GetThoulessTimeEndOthers(Es′s::Vector{Vector{Float64}}, Nτ::Int64, η::Float64)
        τ′s, K′s, τ_Th, Kc′s, τ_Th_c, found_τTh, found_τThc =  SFF.τ̂_Th(Nτ, coeffs′s, Es′s, η);
        t_H = SFF.t̂_H(Es′s);
        t_Th = SFF.t̂_Th(τ_Th, t_H);
        g = SFF.ĝ(τ_Th);

        t_Th_c = SFF.t̂_Th(τ_Th_c, t_H); 
        g_c = SFF.ĝ(τ_Th_c);
    
        return coeffs′s , τ′s, t_H, τ_Th, K′s, t_Th, g, τ_Th_c, Kc′s, t_Th_c, g_c, found_τTh, found_τThc;
    end



end

