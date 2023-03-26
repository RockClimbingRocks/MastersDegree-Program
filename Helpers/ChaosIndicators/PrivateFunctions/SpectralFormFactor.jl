
# To še razmisl kakao boš naredu... Nisem zdovoljen z obliko funckcih (input/output parameteri)..
module SFF
    using SparseArrays;
    using LinearAlgebra;
    # using Combinatoric;
    using ITensors;
    using LaTeXStrings;
    using CurveFit;
    using Polynomials;
    using Statistics;
    using PyPlot;


    include("../../../Helpers/FermionAlgebra.jl");
    using .FermionAlgebra;

    include("../../../Hamiltonians/H2.jl");
    using .H2;

    include("../../../Hamiltonians/H4.jl");
    using .H4;


    """
    Returns a Gaussian filter function exp(-(εᵢ-ε̄)²/2(ηΓ)²).
        
    # Arguments
    - `εᵢ::Float64`         : Set of UNFOLDED eigevalues.
    - `ε̄::Float64`          : Unfolded energy averaged over whole spectrume at a given realization.
    - `Γ::Float64`          : Variance averaged at a given realization.
    - `η::Float64`          : Dimensionless parameter that controls the effective fraction of eigenstates included in the calculations of `K(τ)`.
    """
    ρ̂(εᵢ::Float64, ε̄::Float64, Γ::Float64, η::Float64=0.5)::Float64 = exp(-(εᵢ-ε̄)^2 / (2*(η*Γ)^2));


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

        if length(errors)>0
            println("ERROR")
            # throw(error("Fitting polynomial of order $(length(coeff′s)-1) has decrising parts in the spectrume regime."))
        else
            println("Polynomial with coeffitents $(coeff′s), is all increasing in the range of $(round(E′s[1], digits=3)) to $(round(E′s[end], digits=3))")
        end
    end


    """
    Compute polynomial coeffitents for unfolding process.

    # Arguments
    - `E′s::Vector{Float64}`: Energy spectrume to unfold.
    - `n::Int64 = 5: Degree of polynomial to fit.`
    """
    function ḡₙ_coeffs(E′s::Vector{Float64}, n:: Int64 = 7):: Vector{Float64}
        y = map(x -> Ĝ(x,E′s), E′s);
        coeff′s = poly_fit(E′s, y, n);


        ValidationOfPolynomialCoeffitions(coeff′s, E′s);
        
        return coeff′s;
    end


    function K̂_data(numberOfIterations:: Int64, L:: Int64, q:: Real)
        coeff′s = Vector{Any}();
        E′s = Vector{Any}();


        for i in 1:numberOfIterations
            println(i)
            params2 = H2.Params(L);
            H₂ = H2.Ĥ(params2);
            params4 = H4.Params(L);
            H₄ = H4.Ĥ(params4);

            H = H₂ .+ q .* H₄;
            Es = eigvals(Symmetric(Matrix(H))) 

            coeffs = ḡₙ_coeffs(Es);

            push!(coeff′s, coeffs);
            push!(E′s, Es); 
        end

        return coeff′s, E′s;
    end


    function K̂(τ::Float64, coeff′s:: Vector{Any}, E′s:: Vector{Any})
        numberOfIterations:: Int = length(coeff′s);

        K̃:: Float64 = 0.; 
        Z:: Float64 = 0.;

        for i in 1:numberOfIterations
            coeffs = coeff′s[i];
            Es = E′s[i];
            # println(coeffs)

            ḡₙ = Polynomial(coeffs);

            ε′s = ḡₙ.(Es);    
            ε̄ = mean(ε′s);    
            Γ = stdm(ε′s, ε̄); 

            # plot(Es, ε′s)


            K̃ᵢ = sum(map(εᵢ -> ρ̂(εᵢ, ε̄, Γ)*exp(-2*π*εᵢ*τ*1im), ε′s));   K̃ += abs(K̃ᵢ)^2 / numberOfIterations;
            Zᵢ = sum( abs.(ρ̂.(ε′s, ε̄, Γ)).^2);   Z += Zᵢ / numberOfIterations;
        end

        # plt.show()

        K = K̃/Z;
        return K;
    end
    

end



include("../../../Hamiltonians/H2.jl");
using .H2;

include("../../../Hamiltonians/H4.jl");
using .H4;

include("../../../Helpers/FermionAlgebra.jl");
using .FermionAlgebra;

using LinearAlgebra
using PyPlot;
using JLD2;

L = 12;

N=500
# x = LinRange(-5,0,1000);
# τ′s = 10 .^(x);

# KGOE(τ) = 2τ − τ * log(1 + 2τ );
# plot(τ′s, KGOE.(τ′s))
# xscale("log");
# yscale("log");

q′s = [0.001,0.01,0.1, 0.2, 0.3, 0.5, 0.75,1., 1.5, 2.,5.];

for q in q′s
    println("----", q)
    coeffs, Es =  SFF.K̂_data(N, L, q);
    # K = map(τ -> SFF.K̂(τ, coeffs, Es), τ′s);
    # plot(τ′s, K, label="q=$(q)");

    folder = jldopen("./SpectralFormFunctionCoeffitions_L$(L)_Iter$(N)_q$(q).jld2", "w");
    folder["coeffs"] = coeffs;
    folder["Es"] = Es;
    close(folder)   
end


# plt.show();




