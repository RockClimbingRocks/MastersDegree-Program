module OBCI
    include("./Energies/SFF.jl");
    using .OBSFF;


    """
    Simplifed calculation of Spectral Form Factor (SFF) for quadratic Hamiltonian in one-particle eigenbasis.

    # Arguments
    - `εs′s::Vector{Vector{Float64}}`
    - `t′s::Vector{Float64}: Times at which to calculate SFF.`
    - `K′s::Vector{Float64}: Previously reserved space for storing the results of SFF.`
    """
    function K̂_numerical!(εs′s::Vector{Vector{Float64}}, t′s::Vector{Float64}, K′s::Vector{Float64})
        OBSFF.K̂_numerical_log10!(εs′s, t′s, K′s)
    end


    
    """
    Simplified analitical calculation of Spectral Form Factor for quadratic Hamiltonian in one-particle eigenbasis.

    # Arguments
    - `L::Int: system size.`
    - `t′s::Vector{Float64}: Value of disorder parameter in SYK model.`
    - `n::Float64: Degree of polynomial to fit.`
    """
    function K̂_analitical!(L::Int, t′s::Vector{Float64}, K′s::Vector{Float64})
        
        OBSFF.K_analitical_log10!(L, t′s, K′s)
    end



    """
    Calculations of Spectral Form Factor (SFF) and CSFF by definition for quadratic Hamiltonian in one-particle eigenbasis.

    # Arguments
    - `εs′s::Vector{Vector{Float64}}`
    - `t′s::Vector{Float64}: Times at which to calculate SFF.`
    - `K′s_Kc′s::Vector{Float64}: Previously reserved space for storing the results of SFF.`
    """
    function K̂_numerical_byDef!(εs′s::Vector{Vector{Float64}}, t′s::Vector{Float64}, K′s_Kc′s::Vector{Tuple{Float64,Float64}})
        OBSFF.K̂_numerical_log10_byDef!(εs′s, t′s, K′s_Kc′s)
    end
end


