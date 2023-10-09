module SFF
    using CurveFit;
    using LinearAlgebra;
    using Polynomials;
    using Statistics;


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
    Compute polynomial coeffitents for unfolding process.

    # Arguments
    - `E′s::Vector{Float64}`: Energy spectrume to unfold.
    - `n::Int64: Degree of polynomial to fit.`
    """
    function ḡₙ_coeffs(E′s::Vector{Float64}, n::Int64):: Vector{Float64}
        y = map(x -> Ĝ(x,E′s), E′s);
        coeff′s = poly_fit(E′s, y, n);
        return coeff′s;
    end


    """
    Function computes unfolding values of spectrum. 

    # Arguments
    - `E′s::Vector{Float64}`: Energy spectrume to unfold.
    - `n::Int64: Degree of polynomial to fit.`
    """
    function GetUnfoldedEigenvalues(Es:: Vector{Float64}, n::Int64)
        coeffs = ḡₙ_coeffs(Es, n);
        ḡₙ = Polynomial(coeffs);

        return ḡₙ.(Es)
    end
-

    """
    Function calculates partition function at time τ, with present Gaussian filter.

    # Arguments
    - `εs′s::Vector{Vector{Float64}}: Unfolded spectrum for different realizations.`
    - `ρs′s::Vector{Vector{Float64}}: Calculated values of Gaussian filter.`
    - `τ::Float64: Time for which to calculate partition function.`
    """
    function Ẑρ(εs::Vector{Float64}, ρs::Vector{Float64}, τ::Float64)
        Zρ = sum( @. ρs*exp(-2*π*εs*τ*1im) );
        return Zρ;
    end


    """
    Function calculates normalization constants from values of Gaussian filter with different realization.

    # Arguments
    - `ρs′s::Vector{Vector{Float64}}: Calculated values of Gaussian filter.`
    """
    function Z_A_B(ρs′s)
        Z = mean(    map( ρs -> sum( abs.(ρs).^2), ρs′s) );
        A = mean(  abs.( map(ρs -> sum(ρs), ρs′s) ).^2 );
        B =  abs(    mean( map(ρs -> sum(ρs), ρs′s) ) )^2

        return Z, A, B;
    end



    """
    Function returns value of SFF and CSFF at time τ.

    # Arguments
    - `εs′s::Vector{Vector{Float64}}: Unfolded spectrum for different realizations.`
    - `ρs′s::Vector{Vector{Float64}}: Calculated values of Gaussian filter.`
    - `Z::Float64: Normalization constant.`
    - `A::Float64: Normalization constant.`
    - `B::Float64: Normalization constant.`
    - `τ::Float64: Time for which to calculate SFF and CSFF.`
    """
    function Kcρ(εs′s::Vector{Vector{Float64}}, ρs′s::Vector{Vector{Float64}}, Z::Float64, A::Float64, B::Float64, τ::Float64)
        Zρs = Vector{ComplexF64}(undef, length(εs′s)); 
        foreach(i -> begin
            Zρs[i] = Ẑρ(εs′s[i], ρs′s[i], τ);
        end, eachindex(εs′s))

        Zρ_avg_abs2 = abs(mean(Zρs))^2;
        Zρ_abs2_avg = mean(@. abs(Zρs)^2);

        K = Zρ_abs2_avg/Z
        Kc = (Zρ_abs2_avg - A*Zρ_avg_abs2/B)/Z
        
        return K, Kc
    end



    """
    Function returns value of SFF and CSFF at times τs. Both of them are calculated with Gaussian filter, after spectrum is unfolded.

    # Arguments
    - `Es′s::Vector{Vector{Float64}}: Eigen values for different realizations.`
    - `τs::Vector{Float64}: Times for which to calculate SFF and CSFF.`
    - `η::Float64=0.4: Width of Gaussian filter.`
    - `n::Float64: Degree of polynomial to fit in the process of unfolding.`
    """
    function SpectralFormFactor(Es′s::Vector{Vector{Float64}}, τs::Vector{Float64}, η::Float64, n::Int64)::Tuple{Vector{Float64},Vector{Float64}}
        εs′s = map(Es -> GetUnfoldedEigenvalues(Es, n), Es′s);

        ε̄s = mean.(εs′s);
        Γs = stdm.(εs′s, ε̄s); 
        ρs′s::Vector{Vector{Float64}} = map((εs, ε̄, Γ) -> ρ̂.(εs, ε̄, Γ, η), εs′s, ε̄s, Γs);    
        Z, A, B = Z_A_B(ρs′s);
    
        K_and_Kc = Vector{Tuple{Float64, Float64}}(undef, length(τs));  
        map!(τ -> Kcρ(εs′s, ρs′s, Z, A, B, τ), K_and_Kc, τs);

        K = map(x -> x[1], K_and_Kc);
        Kc = map(x -> x[2], K_and_Kc);

        return K, Kc;
    end











    # """
    # Thouless time (in unphysical units).

    # # Arguments
    # - `K:: Vector{Float64}`: Vectro of Spectral Form Factor for different values of τ.
    # - `τ:: Vector{Float64}`: Vector of τ′s.`
    # """
    # function τ̂_Th0(K::Vector{Float64}, τ::Vector{Float64}, Δ::Int64=100, ε::Float64=0.08)        
    #     K̄ = Vector{Float64}();
    #     τ′s = τ[Δ:end-Δ]
    #     for i in Δ:length(τ)-Δ
    #         append!(K̄, mean(K[i-Δ+1:i+Δ-1]) )
    #     end

    #     ΔK′s = abs.(  log10.(K̄ ./ Kgoe.(τ′s))  );

    #     # WE make linear regresion to determine τ_th more pricisely, k is a slope and n dispacement of a linear function 
    #     i = findall(x -> x<ε, ΔK′s)[1]

    #     i1 = i-1;
    #     i2 = i;

    #     k = (ΔK′s[i1] - ΔK′s[i2])/(τ′s[i1] - τ′s[i2]);
    #     n = ΔK′s[i1] - k*τ′s[i1];

    #     τ_th = (ε - n)/k;
    #     return τ_th;
    # end


    # """
    # Thouless time  (in unphysical units).

    # # Arguments
    # - `K:: Vector{Float64}`: Vectro of Spectral Form Factor for different values of τ.
    # - `Δτ:: Float64`: Steps of τ in which to search for Thoulless time.`
    # """
    # function τ̂_Th(Nτ::Int64, coeffs′s:: Vector{Vector{Float64}}, Es′s:: Vector{Vector{Float64}}, η:: Float64)        
    #     min = -4;
    #     max = 0;

    #     K′s = Vector{Float64}();
    #     τ′s = Vector{Float64}();
    #     Kc′s = Vector{Float64}();
    #     τ_Th = 0.;
    #     τ_Th_c = 0.;


    #     found_τTh  = true
    #     found_τThc = true

    #     N = Int((max-min)*Nτ÷1);
    #     x = LinRange(min, max, N);
    #     τs = 10 .^x;

    #     Ks, Kcs= K̂c(τs, coeffs′s, Es′s, η);

    #     append!(τ′s, τs);
    #     append!(K′s, Ks);
    #     append!(Kc′s, Kcs);

    #     try
    #         τ_Th = τ̂_Th0(Ks, τs);
    #         found_τTh = true;
    #     catch error
    #         if isa(error, BoundsError)
    #             println("nismo najdl τ_Th")
    #             found_τTh = false;
    #         else
    #             println("Nekaj je šlo hudo narobe :(    :")
    #             println(error)
    #             throw(error());
    #         end
    #     end


    #     try
    #         τ_Th_c = τ̂_Th0(Kcs, τs);
    #         found_τThc = true;
    #     catch error
    #         if isa(error, BoundsError)
    #             found_τThc = false;
    #             println("nismo najdl τ_Thc")
    #         else
    #             println("Nekaj je šlo hudo narobe :(    :")
    #             println(error)
    #             throw(error());
    #         end
    #     end

        
    #     return τ′s, K′s, τ_Th, Kc′s, τ_Th_c, found_τTh, found_τThc;
    # end



    # """
    # Heisemberg time in physical units, obtained by analitical expression of δĒ from t_H = 1/δE.

    # # Arguments
    # - `Es′s:: Vector{Float64}`: Vectro energy spectrums of different realizations.
    # """
    # function t̂_H(Es′s)
    #     D = length(Es′s[1]);
    #     a1 = map(Es -> sum(Es.^2)/D  , Es′s);
    #     a2 = map(Es -> sum(Es)^2/D^2, Es′s);

    #     trH2 = mean(a1);
    #     tr2H = mean(a2);
        

    #     Γ0² = trH2 - tr2H;

        
    #     χ = 0.3413;
    #     δĒ = √(Γ0²) / (χ*D);
    #     t_H = 1/δĒ

    #     return t_H;
    # end


    # """
    # Thoulless time in physical units calculated from Thoulless time τ_Th and Heisemberg time t_H.
    # """
    # t̂_Th(τ_Th, t_H) = τ_Th * t_H;



    # """
    # Indicator g calculated from Thoulless time t_Th and Heisenberg time t_H (both in physical units).
    # """
    # ĝ(t_Th:: Float64,t_H:: Float64) = log10(t_H/t_Th);



    # """
    # Indicator g calculated from Thoulless time τ_Th (in unphysical units).
    # """
    # ĝ(τ_Th:: Float64) = - log10(τ_Th);

end


