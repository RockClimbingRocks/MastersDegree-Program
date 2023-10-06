module OBSFF
    using LinearAlgebra;
    using SpecialFunctions;
    using SpecialPolynomials;
    using Polynomials;
    using Statistics;



    function Wjk_analitical(τ::Float64, j::Int, k::Int)::BigFloat
        function Lₙᵅ(n::Int,α::Int,x::Union{Float64, BigFloat})
            n′s = zeros(Int64, n+1); n′s[end] = 1;
            p = Laguerre{α}(n′s);
            return p(x);
        end      

        sqrt = √(factorial(big(k-1))/factorial(big(j-1)));
        expArg = (j-k)*log(τ)/2-τ/2
        
        return sqrt * exp(big(expArg)) * Lₙᵅ(k-1,j-k,big(τ));
    end      

    function Wjk(τ::Float64, j::Int, k::Int)::BigFloat
        j<k ? throw(error("j<k !! NOT OK!")) : nothing;
        result:: BigFloat = Wjk_analitical(τ, j, k)
        return result;
    end


    function M̂!(t′s::Vector{Float64}, N:: Int64, M:: Array{BigFloat, 3})
        ts2N = @. t′s^2/N;
        for j=1:N, k=1:j
            @. M[:,j,k] = M[:,k,j] = iseven(j-k) ? Wjk(ts2N,j,k) * (-1)^((j-k)/2) : 0.;
        end
    end


    """
    Analitical expression for spectral form factor (SFF) for quadratic hamiltonians. 

    # Arguments
    - `L:: Int64`: Number of sites.
    - `t′s::Vector{Float64}`: Times to calculate SFF.
    - `K′s::Vector{Float64}`: Output vector of length t′s.
    """
    function K_analitical_log10!(L:: Int64, t′s::Vector{Float64}, K′s::Vector{Float64})
        I:: Matrix{Float64} = diagm([1. for i in 1:L]);
        M:: Array{BigFloat, 3} = Array{BigFloat, 3}(undef, length(t′s), L, L);
        M̂!(t′s .* 2*pi / (√((L- 1))/2), L, M);   # (√(N-1)/2) serves as normalization and 2\pi due to lack of presence of it in FT

        map!(i -> log10(det(I .+ @view(M[i,:,:]))), K′s, eachindex(t′s));
    end

 

    """
    Numerical results for simplified spectral form factor (SFF) for quadratic hamiltonians. Result is valid for grandcanonical ansemble, but it is in a good agrement with canonical ansemble as well.

    # Arguments
    - `L:: Int64`: Number of sites.
    - `t′s::Vector{Float64}`: Times to calculate SFF.
    - `K′s::Vector{Float64}`: Output vector of length t′s.
    """
    function K̂_numerical_log10!(εs′s::Vector{Vector{Float64}}, t′s::Vector{Float64}, K′s::Vector{Float64}):: Vector{Float64}
        function singleTimeK(εs′s::Vector{Vector{Float64}}, t::Float64):: Float64
            Ks = Vector{BigFloat}(undef, length(εs′s));
            map!(εs -> prod(@. big(1 + cos(t*εs))), Ks, εs′s);
            return log10(mean(Ks));
        end
        
        map!(t -> singleTimeK(εs′s, t), K′s, t′s .* 2*pi);
    end

    # function K̂_numerical_log10!(εs′s::Vector{Vector{Float64}}, t′s::Vector{Float64}, K′s::Vector{Float64}):: Vector{Float64}
    #     function singleTimeK(εs′s::Vector{Vector{Float64}}, t::Float64):: Float64
    #         Ks = Vector{Float64}(undef, length(εs′s));
    #         map!(εs -> prod(@. 1 + cos(t*εs)), Ks, εs′s);
    #         return log10(mean(Ks));
    #     end
    
    #     map!(t -> singleTimeK(εs′s, t), K′s, t′s .* 2*pi);
    # end




    """
    Numerical results for connected and unconnected spectral form factor (SFF and CSFF) calculated by definition for quadratic Hamiltonians.

    # Arguments
    - `εs′s::Vector{Vector{Float64}}`: One particle eigenvalues for different realizations.
    - `t′s::Vector{Float64}`: Times to calculate SFF.
    - `K′s_Kc′s::Vector{Tuple{Float64,Float64}}`: Output vector of length t′s.
    """
    function K̂_numerical_log10_byDef!(εs′s::Vector{Vector{Float64}}, t′s::Vector{Float64}, K′s_Kc′s::Vector{Tuple{Float64,Float64}})
        function singleTimeK(εs′s::Vector{Vector{Float64}}, t::Float64):: Tuple{Float64,Float64}
            Z = Vector{Complex{BigFloat}}(undef, length(εs′s));
            map!(εs -> prod(@. big.((1 + cos(t * εs) - sin(t * εs)*1im)/√2)), Z, εs′s);

            K = mean(abs.(Z).^2);
            Kc = K - abs(mean(Z))^2;
            return log10(K), log10(Kc);
        end
        # display(εs′s)
        map!(t -> singleTimeK(εs′s, t), K′s_Kc′s, t′s .* 2*pi);
    end

    # function K̂c_numerical_log10!(εs′s::Vector{Vector{Float64}}, t′s::Vector{Float64}, K′s_Kc′s::Vector{Tuple{Float64,Float64}})
    #     function singleTimeK(εs′s::Vector{Vector{Float64}}, t::Float64):: Tuple{Float64,Float64}
    #         L = length(εs′s[1])
    #         D = 2^L
    #         Z = Vector{Complex{Float64}}(undef, length(εs′s));
    #         map!(εs -> prod(@. (1 + cos(t * εs) - sin(t * εs)*1im)/√2), Z, εs′s);

    #         K = mean(abs.(Z).^2);
    #         Kc = K - abs(mean(Z))^2;
    #         return log10(K), log10(Kc);
    #     end
    #     # display(εs′s)
    #     map!(t -> singleTimeK(εs′s, t), K′s_Kc′s, t′s .* 2*pi);
    # end

end


