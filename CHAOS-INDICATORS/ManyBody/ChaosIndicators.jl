module MBCI

    include("./Energies/LSR.jl");
    using .LSR
    include("./Energies/SFF.jl");
    using .SFF
    include("./States/IE.jl");
    using .IE

    
    """
    Level spacing ratio (LSR) calculated for midle spectrum.

    # Arguments
    - `λ::Vector{Float64}: Spectrum.`
    - `η::Float64: Portion of the midle spectrum from wich to calculate LSR.`
    """
    global function r̂(λ:: Vector{Float64}, η:: Float64):: Float64
        return LSR.LevelSpacingRatio(λ, η); 
    end

    """
    Function returns value of SFF and CSFF at times τs. Both of them are calculated with Gaussian filter, after spectrum is unfolded.

    # Arguments
    - `Es′s::Vector{Vector{Float64}}: Eigen values for different realizations.`
    - `τs::Vector{Float64}: Times for which to calculate SFF and CSFF.`
    - `η::Float64=0.4: Width of Gaussian filter.`
    - `n::Float64: Degree of polynomial to fit in the process of unfolding.`
    """
    global function K̂(Es′s::Vector{Vector{Float64}}, τs::Vector{Float64}, η::Float64=0.4, n::Int64 = 9)::Tuple{Vector{Float64},Vector{Float64}}
        return SFF.SpectralFormFactor(Es′s, τs, η, n);
    end


    """
    Informational Entropy calculated for midle portion of eigenstates.

    # Arguments
    - `ϕ:: Matrix{Float64}: Matrix of eigenstates, where eigenstates are stored as columns, that is ϕ[:,i].`
    - `η::Float64: Portion of the midle eigenstates from wich to calculate informational entrioy.`
    """
    global function Ŝ(ϕ:: Matrix{Float64}, η:: Float64):: Vector{Float64}
        return IE.InformationalEntropy(ϕ, η);
    end
end


# p1 = sort(rand(1000));
# p2 = 0.3;
# params = [p1, p2];
# x = MBCI.r̂(params...);
# println(x)



# using PyPlot;

# p1 = [sort(rand(1000)) for _ in 1:1000];
# p2 = 10 .^ LinRange(-4,0,100);
# # p2 = 0.3;
# params = [p1, p2];
# x = MBCI.K̂(params...);
# println(x)


# plot(p2, x[1])
# plot(p2, x[2])

# xscale("log")
# yscale("log")
# plt.grid();
# plt.show();
