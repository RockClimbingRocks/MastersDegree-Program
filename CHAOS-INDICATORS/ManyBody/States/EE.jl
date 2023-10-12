module LSR
    using LinearAlgebra;



    """
    TODO
    """
    function EntanglementEntropy(ϕ:: Matrix{Float64}, setA::Vector{Int64}, setB::Vector{Int64}, indices:: Vector{Int64}, η:: Float64 = 0.25, d::Int64=2, cutoff::Float64=10^-7, maxdim::Int64= 10):: Vector{Float64}
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
        
            EE_ψ′s[l] = EntanglementEntropy(ψ′, setA, setB);
        end

        return EE_ψ′s;        
    end


    """
    hmm
    """
    function EntanglementEntropy(ϕ:: Vector{Float64}, setA::Vector{Int64}, setB::Vector{Int64}, d::Int64=2, cutoff::Float64=10^-7, maxdim::Int64= 10)
        M = Bipartition(ϕ, setA, setB);

        λ′s = svdvals(M);
        filter!(x -> x > cutoff, λ′s);

        EE = sum( map(x -> Sᵢ(x), λ′s) );
        return EE;
    end



    """
    Naj bo ψ vektor dimenzije d^L katerega kočemo zapisati v MPS. Kjer je L število mest, in d število možnih konfiguracij spina na posameznem mestu.
    Cutoff določa mejo, kjer zanemari ničle, maxdim pa določi maksimalno dimezijo bloka.
    """
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
    This function is needed to change order of dimensions because julia ic column and not row maijor language.
    """
    function reversedims(M)
        N = length(size(M));
        M′ = permutedims(M, N:-1:1);
        return M′;
    end


end