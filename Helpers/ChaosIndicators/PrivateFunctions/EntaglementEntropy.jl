
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
    function Bipartition′s(ψ::Vector{Float64}, permutations:: Vector{Vector{Int64}}, L::Int64, d::Int64, cutoff::Real, maxdim:: Int64)

        M′s = Vector{Any}();

        for permutation in permutations
            M′′ = reshape(ψ, [2 for i in 1:L]...);
            M′  = permutedims(M′′, permutation);
            M   = reversedims(reshape(M′, :, Int(d^(L÷2)) ))
            push!(M′s, M);
        end
        return M′s;
    end

    function Bipartition(ψ::Vector{Float64}, setA::Vector{Int64}, setB::Vector{Int64})
        L  = length(setA) + length(setB);
        Lₐ = length(setA);

        permutation:: Vector{Int64} = vcat(setA, setB);

        M′′ = reshape(ψ, [2 for i in 1:L]...);
        M′  = permutedims(M′′, permutation);
        M   = reversedims(reshape(M′, :, Int(2^Lₐ) ))

        return M;
    end
 

    """
    Funkcija vrne entropijo prepletenosti stanja ψ. Vrednost Nₐ je število mest v bloku A, parameter je smiselen samo če je biparticija kompaktna, če je ta nekompaktna, se parameter avtomatsko nastavi na N÷2, kjer je N skupno število mest. 
    "Permutation" od nas hoče imeti set indeksov v A bloku, ter set indeksov v bloku B.  
    """
    global function EntanglementEntropy′s(ψ::Vector{Float64}, permutations:: Vector{Vector{Int64}}, L::Int64, d::Int64=2, cutoff::Real=10^-7, maxdim::Int64= 10)

        E′s = Vector{Float64}();

        M′s = Bipartition′s(ψ, permutations, L, d, cutoff, maxdim);

        for M in M′s
            λ′s = svdvals(M);
            filter!(x -> x > 10^-5, λ′s);

            E = sum( map(x -> Sᵢ(x), λ′s) );
            push!(E′s, E);
        end
        
        return E′s;
    end

    """
    hmm
    """
    global function EntanglementEntropy(ϕ:: Vector{Float64}, setA::Vector{Int64}, setB::Vector{Int64}, d::Int64=2, cutoff::Real=10^-7, maxdim::Int64= 10)
        M = Bipartition(ϕ, setA, setB);

        λ′s = svdvals(M);
        filter!(x -> x > cutoff, λ′s);

        EE = sum( map(x -> Sᵢ(x), λ′s) );
        return EE;
    end


end


