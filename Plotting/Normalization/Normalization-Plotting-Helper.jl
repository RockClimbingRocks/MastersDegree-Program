
module NormalizationPlottingHelper

    using LinearAlgebra;

    include("../../Hamiltonians/H2.jl");
    using .H2;

    include("../../Hamiltonians/H4.jl");
    using .H4;

    include("../../Helpers/OperationsOnHamiltonian.jl");
    using .OperationsOnH


    global function Normalization(L′s:: Vector{Int}, S:: Real, μ:: Real, mean:: Real, deviation:: Real; NumbOfIter:: Int, namespace:: Module)
        numericalNorm = Vector{Vector{Float64}}(undef,length(L′s));

        for l in eachindex(L′s)
            numericalNormOfL = Vector{Float64}(undef, NumbOfIter)
            for i in 1:NumbOfIter
                println(l, " ", i);


                params = namespace.GetParams(L′s[l], S, μ, mean, deviation)
                H₂ = namespace.Ĥ(params, true);

                numericalNormOfL[i] = OperationsOnH.OperatorNorm(H₂);
            end
            
            numericalNorm[l] = numericalNormOfL;
        end

        return numericalNorm;

    end



    global function Normalization2(params, NumbOfIter:: Int, namespace:: Module)
        numericalNorm = Vector{Vector{Float64}}(undef,length(params));

        for p in eachindex(params)
            numericalNormOfL = Vector{Float64}(undef, NumbOfIter)
            for i in 1:NumbOfIter
                println(p, " ", i);

                H₂ = namespace.Ĥ(params[p], true);

                numericalNormOfL[i] = OperationsOnH.OperatorNorm(H₂);
            end
            
            numericalNorm[l] = numericalNormOfL;
        end

        return numericalNorm;

    end
    

end


# include("../../Hamiltonians/H2.jl");
# using .H2;


# Ls = [2,4,6];
# iter = 10
# namespace = H2


# params = H2.GetParams.(Ls, 1/2, 0., 0., 1.)

# norm = NormalizationPlottingHelper.Normalization2(params, iter, namespace);


# println(norm)


