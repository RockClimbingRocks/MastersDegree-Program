
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
    - `n::Float64: Degree of polynomial to fit.`
    """
    function K̂(τ′s::Vector{Float64}, coeff′s:: Vector{Vector{Float64}}, E′s:: Vector{Vector{Float64}}, η:: Float64)
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
            
            if η==Inf
                println(η, "  ", i );
                for (j, τ) in enumerate(τ′s)
                    K̃ᵢ = sum(map(εᵢ -> exp(-2*π*εᵢ*τ*1im), ε′s));   K̃[j] += abs(K̃ᵢ)^2 / numberOfIterations;
                end
            else
                for (j, τ) in enumerate(τ′s)
                    K̃ᵢ = sum(map(εᵢ -> ρ̂(εᵢ, ε̄, Γ, η)*exp(-2*π*εᵢ*τ*1im), ε′s));   K̃[j] += abs(K̃ᵢ)^2 / numberOfIterations;
                    Zᵢ = sum( abs.(ρ̂.(ε′s, ε̄, Γ, η)).^2);   Z[j] += Zᵢ / numberOfIterations;
                end
            end
        end

        Ks = η==Inf ? K̃ : K̃./Z
        return Ks;
    end


    """
    Function returns value of Spectral Form Factor at value τ and for coeffitents annd spectrum obtained from K̂_data.

    # Arguments
    - `numberOfIterations:: Int64`: Number of iterations for averaging.
    - `L::Int64 = 5: system size.`
    - `q::Int64: Value of disorder parameter in SYK model.`
    - `n::Float64: Degree of polynomial to fit.`
    """
    function K̂c(τ′s::Vector{Float64}, coeff′s:: Vector{Vector{Float64}}, E′s:: Vector{Vector{Float64}}, η:: Float64)
        numberOfIterations:: Int = length(coeff′s);
        Nτ:: Int64 = length(τ′s);

        K1:: Vector{Float64} = zeros(Float64, Nτ); 
        K̄2:: Vector{Complex} = zeros(Complex, Nτ); 
        A:: Float64 = 0.;
        B̄:: Complex = 0.;
        Z:: Float64 = 0.;

        for i in 1:numberOfIterations
            coeffs = coeff′s[i];
            Es = E′s[i];
            if η==Inf
                println("TO še naredi oziroma popravi!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" );
                throw(error("ni še implementirano!"));
            else
                K1_, K̄2_, A_, B̄_, Z_ = K̂_singleIteration(Es, coeffs, τ′s, η);

                K1 .+= K1_ ./ numberOfIterations;
                K̄2 .+= K̄2_ ./ numberOfIterations;
                A  += A_  / numberOfIterations;
                B̄  += B̄_  / numberOfIterations;
                Z  += Z_  / numberOfIterations;
            end
        end

        K2:: Vector{Float64} = @. abs(K̄2)^2;
        B:: Float64 = abs(B̄)^2;

        K:: Vector{Float64}  = @.  K1  / Z;
        Kc:: Vector{Float64} = @. (K1 - A*K2/B) / Z;
        return K, Kc;
    end



    function K̂_singleIteration(E′s::Vector{Float64}, coeffs::Vector{Float64}, τ′s::Vector{Float64}, η:: Float64)
        ḡₙ = Polynomial(coeffs);

        ε′s = ḡₙ.(E′s);    
        ε̄ = mean(ε′s);    
        Γ = std(ε′s); 

        K1ᵢ(τ) = abs(sum( map(εᵢ ->     ρ̂(εᵢ, ε̄, Γ, η)*exp(-2*π*εᵢ*τ*1im), ε′s)))^2
        K̄2ᵢ(τ) =     sum( map(εᵢ ->     ρ̂(εᵢ, ε̄, Γ, η)*exp(-2*π*εᵢ*τ*1im), ε′s))
        Aᵢ     = abs(sum( map(εᵢ ->     ρ̂(εᵢ, ε̄, Γ, η)                   , ε′s)))^2;
        B̄ᵢ     =     sum( map(εᵢ ->     ρ̂(εᵢ, ε̄, Γ, η)                   , ε′s));
        Zᵢ     =     sum( map(εᵢ -> abs(ρ̂(εᵢ, ε̄, Γ, η))^2                , ε′s));

        if η==Inf
            println("To še naredi oziroma popravi!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" );
            throw(error("ni še implementirano!"));
        else
            return K1ᵢ.(τ′s), K̄2ᵢ.(τ′s), Aᵢ, B̄ᵢ, Zᵢ
        end
    end




    """
    Thouless time (in unphysical units).

    # Arguments
    - `K:: Vector{Float64}`: Vectro of Spectral Form Factor for different values of τ.
    - `τ:: Vector{Float64}`: Vector of τ′s.`
    """
    function τ̂_Th0(K::Vector{Float64}, τ::Vector{Float64}, Δ::Int64=100, ε::Float64=0.08)        
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
    Thouless time  (in unphysical units).

    # Arguments
    - `K:: Vector{Float64}`: Vectro of Spectral Form Factor for different values of τ.
    - `Δτ:: Float64`: Steps of τ in which to search for Thoulless time.`
    """
    function τ̂_Th(Nτ::Int64, coeffs′s:: Vector{Vector{Float64}}, Es′s:: Vector{Vector{Float64}}, η:: Float64)        
        min = -4;
        max = 0;

        K′s = Vector{Float64}();
        τ′s = Vector{Float64}();
        Kc′s = Vector{Float64}();
        τc′s = Vector{Float64}();
        τ_Th = 0.;
        τ_Th_c = 0.;


        isτCalculated = false
        while !isτCalculated 
            N = Int((max-min)*Nτ÷1);
            x = LinRange(min, max, N);
            τs = 10 .^x;


            Ks, Kcs= K̂c(τs, coeffs′s, Es′s, η);

            append!(τ′s, τs);
            append!(K′s, Ks);

            append!(τc′s, τs);
            append!(Kc′s, Kcs);

            try
                τ_Th = τ̂_Th0(Ks, τs);
                τ_Th_c = τ̂_Th0(Kcs, τs);
                isτCalculated = true;
            catch error
                if isa(error, BoundsError)
                    println("error: ", error);
                    # println("❗ Nismo našli τ_Th, nastavimo  min = ", max, ",  in max = ", max +1, "🧯 🧯");
                    min = max;
                    max = max + 1;
                    isτCalculated = false;

                    if max >= 5
                        throw(error());
                    end
                else
                    println("Nekaj je šlo hudo narobe 😞:")
                    println(error)
                    println(2)
                    throw(error());
                end
            end

        end

        
        return τ′s, K′s, τ_Th, Kc′s, τ_Th_c;
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


