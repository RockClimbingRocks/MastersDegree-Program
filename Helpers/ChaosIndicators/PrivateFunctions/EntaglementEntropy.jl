
# To še razmisl kakao boš naredu... Nisem zdovoljen z obliko funckcih (input/output parameteri)..
module EE
    using SparseArrays;
    using LinearAlgebra;
    using Combinatorics
    using ITensors


    include("../../../Helpers/FermionAlgebra.jl");
    using .FermionAlgebra;

    include("../../../Hamiltonians/H2.jl");
    using .H2;

    include("../../../Hamiltonians/H4.jl");
    using .H4;


    Sᵢ(x) = - abs(x)^2 * log(abs(x)^2)

    # function ψ_as_MPS(ψ::Vector{Float64}, L::Int64, d::Int64, cutoff::Real, maxdim:: Int64)
    #     # sites = siteinds(d,L);
    #     # M = MPS(ψ, sites, cutoff=cutoff, maxdim=maxdim)
    #     N = Int(log(2, length(ψ)))
    #     X = reshape(ψ, [2 for i in 1:L]...)
    #     return M 
    # end


    """
    This function is needed to change order of dimensions because julia ic column and not row maijor language.
    """
    function reversedims(M)
        N = length(size(M));
        M′ = permutedims(M, N:-1:1);
        return M′;
    end


    """
    Naj bo ψ vektor dimenzije d^L katerega kočemo zapisati v MPS. Kjer je L število mest, in d število možnih konfiguracij spina na posameznem mestu.
    Cutoff določa mejo, kjer zanemari ničle, maxdim pa določi maksimalno dimezijo bloka.
    """
    function Bipartition(ψ::Vector{Float64}, permutations:: Vector{Vector{Int64}}, L::Int64, d::Int64, cutoff::Real, maxdim:: Int64)

        M′s = Vector{Any}();

        for permutation in permutations
            M′′ = reshape(ψ, [2 for i in 1:L]...);
            M′  = permutedims(M′′, permutation);
            M   = reversedims(reshape(M′, :, Int(d^(L÷2)) ))
            push!(M′s, M);
        end
        return M′s;
    end


    """
    Funkcija vrne entropijo prepletenosti stanja ψ. Vrednost Nₐ je število mest v bloku A, parameter je smiselen samo če je biparticija kompaktna, če je ta nekompaktna, se parameter avtomatsko nastavi na N÷2, kjer je N skupno število mest. 
    "Permutation" od nas hoče imeti set indeksov v A bloku, ter set indeksov v bloku B.  
    """
    global function EntanglementEntropy(ψ::Vector{Float64}, permutations:: Vector{Vector{Int64}}, L::Int64, d::Int64=2, cutoff::Real=10^-7, maxdim::Int64= 10)

        E′s = Vector{Float64}();

        M′s = Bipartition(ψ, permutations, L, d, cutoff, maxdim);

        for M in M′s
            λ′s = svdvals(M);
            filter!(x -> x > 10^-5, λ′s);

            E = sum( map(x -> Sᵢ(x), λ′s) );
            push!(E′s, E);
        end
        
        return E′s;
    end


end



# include("../Hamiltonians/H2.jl");
# using .H2;

# include("../Hamiltonians/H4.jl");
# using .H4;

# include("../Helpers/FermionAlgebra.jl");
# using .FermionAlgebra;


# using LinearAlgebra

# L = 10;
# q = 10;

# params2 = H2.Params(L);
# H₂ = H2.Ĥ(params2);

# params4 = H4.Params(L);
# H₄ = H4.Ĥ(params4);

# H = H₂ .+ q .* H₄;

# D = binomial(L,L÷2);
# η = 0.2;
# i₁ = Int((D - (D*η)÷1)÷2);
# i₂ = Int(i₁ + (D*η)÷1);
# ϕ = eigvecs(Matrix(H))[:, i₁:i₂];


# ind = FermionAlgebra.IndecesOfSubBlock(L,1/2);

# permutation1 = [1:1:L...];
# permutation2 = [1:2:L..., 2:2:L...];
# permutations = [permutation1, permutation2];


# function ψ̂(ψ, L)
#     ψ′ = zeros(Float64, Int(2^L));
#     ψ′[ind] = ψ;
    
#     return ψ′
# end

# ψ′s = [ ψ̂(ψ, L) for ψ in eachcol(ϕ) ];


# E′s = ChaosIndicators.EntanglementEntropy(ψ′s, permutations, L);


# println( sum(E′s)/length(E′s) )
# println(D)






# E′s = Vector{Vector{Float64}}();


# for ψ in eachcol(ϕ)
#     # println(ψ)

#     ψ′ = zeros(Float64, Int(2^L));
#     ψ′[ind] = ψ;

#     E = ChaosIndicators.EntanglementEntropy(ψ′, permutations, L);
#     push!(E′s, E);
# end



# println( sum(E′s)/length(E′s) )
# println(D)





