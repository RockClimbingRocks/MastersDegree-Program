module LSR
    using Statistics;


    """
    Level spacing ratio (LSR) calculated for midle spectrum.

    # Arguments
    - `λ::Vector{Float64}`: Spectrum.
    - `η::Float64`: Portion of the midle spectrum from wich to calculate LSR.
    """
    global function LevelSpacingRatio(λ:: Vector{Float64}, η:: Float64):: Float64
        D = length(λ);
        i₁ = Int((D - (D*η)÷1)÷2);
        i₂ = Int(i₁ + (D*η)÷1);
        if η ≈ 1.
            i₁ = 1;
            i₂ = D;
        end

        λ = λ[i₁:i₂];
        δ = @. λ[2:end] - λ[1:end-1];
        r′s = map((x,y) -> min(x,y)/max(x,y), δ[1:end-1], δ[2:end] );

        return mean(r′s);
    end
end