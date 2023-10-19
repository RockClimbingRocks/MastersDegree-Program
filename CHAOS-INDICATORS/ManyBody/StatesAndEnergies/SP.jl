module SP
    using LinearAlgebra;
    using Statistics;
    using Einsum;

    """
    Function returns value for survival probability at time t for single realization of the Hamiltonian. 

    # Arguments
    - `C::Matrix{Float64}`: Coefifitients `C` are calculated with matrix multiplication `C = Λᵀ*M`. Where `Λ` and `M` are eigenvector matrices of interacting and noninteracting  Hamiltonians, respectevly.
    - `E′s::Vector{Float64}`: idth of Gaussian filter.
    - `t::Float64`: Time for which to survival probability.
    """
    function SurvivalProbability_H2(C::Matrix{Float64}, E′s::Vector{Float64}, t′s::Vector{Float64})::Vector{Float64}
        D = size(C)[1];
        Pᴴ = Vector{Float64}(undef, length(t′s));
        A = Matrix{ComplexF64}(undef, length(t′s), D) # preallocated space
        @einsum A[t, m] =  abs(C[ν,m])^2 * exp(-E′s[ν]*t′s[t]*1im);

        map!(x-> abs(x)^2, A, A);
        mean!(Pᴴ, A);
        return Pᴴ;
    end

end

