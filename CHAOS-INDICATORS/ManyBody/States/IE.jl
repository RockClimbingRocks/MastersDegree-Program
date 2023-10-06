module IE
    Sᵢ(x) = - abs(x)^2 * log(abs(x)^2)

    """
    Informational Entropy (IE) calculated for midle portion of eigenstates.

    # Arguments
    - `ϕ:: Matrix{Float64}: Matrix of eigenstates, where eigenstates are stored as columns, that is ϕ[:,i].`
    - `η::Float64: Portion of the midle eigenstates from wich to calculate informational entrioy.`
    """
    global function InformationalEntropy(ϕ:: Matrix{Float64}, η:: Float64):: Vector{Float64}
        D = size(ϕ)[1];
        i₁ = Int((D - (D*η)÷1)÷2);
        i₂ = Int(i₁ + (D*η)÷1); 

        if η ≈ 1.
            i₁ = 1;
            i₂ = D;
        end

        ϕ_reduced  = @view ϕ[:, i₁:i₂];
        Sₘ′s = sum( map(x -> Sᵢ(x), ϕ_reduced), dims=1);

        return vec(Sₘ′s);
    end
end
