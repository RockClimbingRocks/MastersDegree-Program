
# To še razmisl kakao boš naredu... Nisem zdovoljen z obliko funckcih (input/output parameteri)..
module SFF
    using CurveFit
using ITensors
using LaTeXStrings
using LinearAlgebra
using Polynomials
# using PyPlot
using SparseArrays
using Statistics;


    include("../../../Helpers/FermionAlgebra.jl");
    using .FermionAlgebra;

    include("../../../Hamiltonians/H2.jl");
    using .H2;

    include("../../../Hamiltonians/H4.jl");
    using .H4;



    """
    Analitical value for spectral form factor (SSF) for GOE ansemble (for τ<1).
    """
    Kgoe(τ) = τ<=1 ? 2*τ - τ*log(1 + 2*τ) : 2 - τ*log((2*τ + 1)/(2*τ - 1));



    """
    Returns a Gaussian filter function exp(-(εᵢ-ε̄)²/2(ηΓ)²).
        
    # Arguments
    - `εᵢ::Float64`         : Set of UNFOLDED eigevalues.
    - `ε̄::Float64`          : Unfolded energy averaged over whole spectrume at a given realization.
    - `Γ::Float64`          : Variance averaged at a given realization.
    - `η::Float64`          : Dimensionless parameter that controls the effective fraction of eigenstates included in the calculations of `K(τ)`.
    """
    ρ̂(εᵢ::Float64, ε̄::Float64, Γ::Float64, η::Float64)::Float64 = exp(-(εᵢ-ε̄)^2 / (2*(η*Γ)^2));


    """
    Heviside `𝛩(x)` function. Returns 0 for `x`<0 and 1 for `x`≥0.
        
    # Arguments
    - `x::Real`: Value of Heviside `𝛩(x)` function at `x`.

    # Examples
    ```jldoctest
    julia-repl
    julia>  𝛩(-1)
    0

    julia>  𝛩(0)
    1
            
    julia>  𝛩(1000)
    1
    ```
    """
    𝛩(x:: Real):: Float64 = x<0 ? 0. : 1.;



    """
    Commulative spectral function ∑ᵢ 𝛩(E - Eᵢ).

    # Arguments
    - `E::Float64`
    - `E′s::Vector{Float64}`: Energy spectrume to unfold.
    """
    Ĝ(E::Real, E′s::Vector{Float64})::Float64 = sum( 𝛩.(E.-E′s) );


    """
    Checking validation of fitting parameters, so that poynomial is increasing at all values of the spectrume.

    # Arguments
    - `coeff′s::Vector{Float64}`: coeffitents to validate.
    - `E′s::Vector{Float64}`: Energy spectrume to unfold.
    """
    function ValidationOfPolynomialCoeffitions(coeff′s::Vector{Float64}, E′s::Vector{Float64})
        ŷ = Polynomial(coeff′s);
        y = ŷ.(E′s);

        δ = y[2:end] .- y[1:end-1];
        errors = filter(x -> x < 0, δ);

        # if length(errors)>0
        #     println("ERROR")
        #     # throw(error("Fitting polynomial of order $(length(coeff′s)-1) has decrising parts in the spectrume regime."))
        # else
        #     println("Polynomial with coeffitents $(coeff′s), is all increasing in the range of $(round(E′s[1], digits=3)) to $(round(E′s[end], digits=3))")
        # end
    end


    """
    Compute polynomial coeffitents for unfolding process.

    # Arguments
    - `E′s::Vector{Float64}`: Energy spectrume to unfold.
    - `n::Int64 = 5: Degree of polynomial to fit.`
    """
    function ḡₙ_coeffs(E′s::Vector{Float64}, n:: Int64 = 5):: Vector{Float64}
        y = map(x -> Ĝ(x,E′s), E′s);
        coeff′s = poly_fit(E′s, y, n);


        ValidationOfPolynomialCoeffitions(coeff′s, E′s);
        
        return coeff′s;
    end


    """
    Function returns fitting polynomial coeffitents on unfolded spectrum, and energy spectrum. 

    # Arguments
    - `numberOfIterations:: Int64`: Number of iterations for averaging.
    - `L::Int64 = 5: system size.`
    - `q::Int64: Value of disorder parameter in SYK model.`
    - `n::Int64: Degree of polynomial to fit.`
    """
    function K̂_data(numberOfIterations:: Int64, L:: Int64, q:: Real, n:: Int64 = 5)
        coeff′s = Vector{Any}();
        Es′s = Vector{Any}();


        for i in 1:numberOfIterations
            # println(i)
            params2 = H2.Params(L);
            H₂ = H2.Ĥ(params2);
            params4 = H4.Params(L);
            H₄ = H4.Ĥ(params4);

            H = H₂ .+ q .* H₄;
            Es = eigvals(Symmetric(Matrix(H))) 

            coeffs = ḡₙ_coeffs(Es, n);

            push!(coeff′s, coeffs);
            push!(Es′s, Es); 
        end

        return coeff′s, Es′s;
    end


    # """
    # Function returns value of Spectral Form Factor at value τ and for coeffitents annd spectrum obtained from K̂_data.

    # # Arguments
    # - `numberOfIterations:: Int64`: Number of iterations for averaging.
    # - `L::Int64 = 5: system size.`
    # - `q::Int64: Value of disorder parameter in SYK model.`
    # - `n::Int64: Degree of polynomial to fit.`
    # """
    # function K̂(τ::Float64, coeff′s:: Vector{Any}, E′s:: Vector{Any}, η:: Float64)
    #     numberOfIterations:: Int = length(coeff′s);

    #     K̃:: Float64 = 0.; 
    #     Z:: Float64 = 0.;

    #     for i in 1:numberOfIterations
    #         coeffs = coeff′s[i];
    #         Es = E′s[i];
    #         # println(coeffs)

    #         ḡₙ = Polynomial(coeffs);

    #         ε′s = ḡₙ.(Es);    
    #         ε̄ = mean(ε′s);    
    #         Γ = stdm(ε′s, ε̄); 

    #         # plot(Es, ε′s)


    #         K̃ᵢ = sum(map(εᵢ -> ρ̂(εᵢ, ε̄, Γ, η)*exp(-2*π*εᵢ*τ*1im), ε′s));   K̃ += abs(K̃ᵢ)^2 / numberOfIterations;
    #         Zᵢ = sum( abs.(ρ̂.(ε′s, ε̄, Γ, η)).^2);   Z += Zᵢ / numberOfIterations;
    #     end

    #     # plt.show()

    #     K = K̃/Z;
    #     return K;
    # end



    """
    Function returns value of Spectral Form Factor at value τ and for coeffitents annd spectrum obtained from K̂_data.

    # Arguments
    - `numberOfIterations:: Int64`: Number of iterations for averaging.
    - `L::Int64 = 5: system size.`
    - `q::Int64: Value of disorder parameter in SYK model.`
    - `n::Int64: Degree of polynomial to fit.`
    """
    function K̂(τ′s::Vector{Float64}, coeff′s:: Vector{Any}, E′s:: Vector{Any}, η:: Float64)
        numberOfIterations:: Int = length(coeff′s);
        Nτ:: Int64 = length(τ′s);

        K̃:: Vector{Float64} = zeros(Float64, Nτ); 
        Z:: Vector{Float64} = zeros(Float64, Nτ); 

        for i in 1:numberOfIterations
            coeffs = coeff′s[i];
            Es = E′s[i];

            ḡₙ = Polynomial(coeffs);

            ε′s = ḡₙ.(Es);    
            ε̄ = mean(ε′s);    
            Γ = stdm(ε′s, ε̄); 
            
            for (j, τ) in enumerate(τ′s)
                K̃ᵢ = sum(map(εᵢ -> ρ̂(εᵢ, ε̄, Γ, η)*exp(-2*π*εᵢ*τ*1im), ε′s));   K̃[j] += abs(K̃ᵢ)^2 / numberOfIterations;
                Zᵢ = sum( abs.(ρ̂.(ε′s, ε̄, Γ, η)).^2);   Z[j] += Zᵢ / numberOfIterations;
            end
        end


        Ks = K̃./Z;
        return Ks;
    end




    """
    Thouless time (in unphysical units).

    # Arguments
    - `K:: Vector{Float64}`: Vectro of Spectral Form Factor for different values of τ.
    - `τ:: Vector{Float64}`: Vector of τ′s.`
    """
    function τ̂_Th0(K::Vector{Float64}, τ::Vector{Float64}, η:: Float64, Δ::Int64=100, ε::Float64=0.08)        
        K̄ = Vector{Float64}();
        τ′s = τ[Δ:end-Δ]
        for i in Δ:length(τ)-Δ
            append!(K̄, mean(K[i-Δ+1:i+Δ-1]) )
        end

        ΔK′s = abs.(  log10.(K̄ ./ Kgoe.(τ′s))  );

        # WE make linear regresion to determine τ_th more pricisely, k is a slope and n dispacement of a linear function 
        i = findall(x -> x<ε, ΔK′s)[1]

        i1 = i-1;
        i2 = i;

        k = (ΔK′s[i1] - ΔK′s[i2])/(τ′s[i1] - τ′s[i2]);
        n = ΔK′s[i1] - k*τ′s[i1];

        τ_th = (ε - n)/k;
        return τ_th;
    end


    """
    Thouless time (in unphysical units).

    # Arguments
    - `K:: Vector{Float64}`: Vectro of Spectral Form Factor for different values of τ.
    - `Δτ:: Float64`: Steps of τ in which to search for Thoulless time.`
    """
    function τ̂_Th(Nτ::Int64, coeffs′s:: Vector{Any}, Es′s:: Vector{Any}, η:: Float64)        
        min = -5;
        max = 1;

        K′s = Vector{Float64}();
        τ′s = Vector{Float64}();
        τ_Th = 0.;


        isτCalculated = false
        while !isτCalculated
            N   = Int((max-min)*Nτ÷1);
            x = LinRange(min, max, N);
            τs = 10 .^x;

            Ks = K̂(τs, coeffs′s, Es′s, η);

            append!(τ′s, τs);
            append!(K′s, Ks);

            # fig, ax = plt.subplots(ncols=2)
            # ΔK = @. log10(Ks / Kgoe(τs));
            # ax[1].plot(τs, Ks);
            # ax[1].plot(τs, Kgoe.(τs), color="black", linestyle="dashed");
            # ax[2].plot(τs, ΔK)
            # plt.show()


            try
                τ_Th = τ̂_Th0(Ks, τs, η);
                isτCalculated = true;
            catch error
                if isa(message, BoundsError)
                    println("error: ", error);
                    println("Nismo našli τ_Th, nastavimo  min = ", max, ",  in max = ", max +1);
                    min = max;
                    max = max + 1;
                    isτCalculated = false;
                else
                    throw(error("Nekaj je šlo hudo narobe 😞:   $(error)"));
                end
            end

        end

        
        return K′s, τ′s, τ_Th;
    end


    """
    Heisemberg time in physical units, obtained by analitical expression of δĒ from t_H = 1/δE.

    # Arguments
    - `Es′s:: Vector{Float64}`: Vectro energy spectrums of different realizations.
    """
    function t̂_H(Es′s)
        D = length(Es′s[1]);
        a1 = map(Es -> sum(Es.^2)/D  , Es′s);
        a2 = map(Es -> sum(Es)^2/D^2, Es′s);

        trH2 = mean(a1);
        tr2H = mean(a2);
        

        Γ0² = trH2 - tr2H;

        
        χ = 0.3413;
        δĒ = √(Γ0²) / (χ*D);
        t_H = 1/δĒ

        return t_H;
    end


    """
    Thoulless time in physical units calculated from Thoulless time τ_Th and Heisemberg time t_H.
    """
    t̂_Th(τ_Th, t_H) = τ_Th * t_H;



    """
    Indicator g calculated from Thoulless time t_Th and Heisenberg time t_H (both in physical units).
    """
    ĝ(t_Th:: Float64,t_H:: Float64) = log10(t_H/t_Th);



    """
    Indicator g calculated from Thoulless time τ_Th (in unphysical units).
    """
    ĝ(τ_Th:: Float64) = - log10(τ_Th);

end



# include("../../../Hamiltonians/H2.jl");
# using .H2;

# include("../../../Hamiltonians/H4.jl");
# using .H4;

# include("../../../Helpers/FermionAlgebra.jl");
# using .FermionAlgebra;


# using LinearAlgebra
# using PyPlot;
# using JLD2;



# L = 6;

# N=20000
# x = LinRange(-2,0,300);
# τ′s = 10 .^(x);

# KGOE(τ) = 2τ − τ * log(1 + 2τ );
# plot(τ′s, KGOE.(τ′s))
# xscale("log");
# yscale("log");

# # q′s = [0.001,0.01,0.1, 0.2, 0.3, 0.5, 0.75,1., 1.5, 2.,5.];

# # q′s = [0.005,0.05,0.15,0.25, 0.35, 0.4, 0.6];

# # q′s = [0.7, 0.8, 0.9,1.75,2.5,3.,4.];


# # q′s = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 4., 5.];
# q′s = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 4., 5.];
# # q′s = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 4., 5.];

# τ_th′s = Vector{Float64}();
# for q in q′s
#     println("----", q)

#     # coeffs, Es =  SFF.K̂_data(N, L, q);
#     # folder = jldopen("./SpectralFormFunctionCoeffitions_L$(L)_Iter$(N)_q$(q).jld2", "w");
#     # folder["coeffs"] = coeffs;
#     # folder["Es"] = Es;
#     # close(folder)   

#     folder = jldopen("./SpectralFormFunctionCoeffitions_L$(L)_Iter$(N)_q$(q).jld2", "r");
#     Es = folder["Es"];
#     coeffs = folder["coeffs"];
#     close(folder)   


#     K = map(τ -> SFF.K̂(τ, coeffs, Es), τ′s);
#     # τ_th = SFF.τ̂_Th(K,τ′s);

#     # push!(τ_th′s, τ_th);

#     plot(τ′s, K, label="q=$(q)");
# end

# plot(τ′s, SFF.Kgoe.(τ′s), label="Kgoe", linestyle="dashed");

# # plot(q′s, τ_th′s)
# xscale("log");
# yscale("log");
# plt.legend();
# plt.show();


