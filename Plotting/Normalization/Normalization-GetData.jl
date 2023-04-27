using PyPlot
using JLD2 
using LinearAlgebra
using LaTeXStrings
using CurveFit
using Statistics


# include("../../Helpers/FermionAlgebra.jl");
# using .FermionAlgebra;

include("../../Hamiltonians/H2.jl")
using .H2

include("../../Hamiltonians/H4.jl")
using .H4

include("../../Hamiltonians/Hgoe.jl")
using .Hgoe

include("../../Helpers/ChaosIndicators/ChaosIndicators.jl");
using .ChaosIndicators;

include("../../Helpers/OperationsOnHamiltonian.jl");
using .OperationsOnH


function GetNormalizationData(L′s:: Vector{Int}, maxNumbOfIter:: Int64, namespace:: Module)
    numericalNorm = zeros(Float64, (length(L′s), maxNumbOfIter));

    for (l, L) in enumerate(L′s)
        println("\nline L: ", L);
        for i=1:maxNumbOfIter
            if mod(i,100)==0
                print(" ", round(i/maxNumbOfIter *100, digits=2), "%..." );
            end
            params = namespace.Params(L);
            H₂ = namespace.Ĥ(params);

            numericalNorm[l,i] = OperationsOnH.OperatorNorm(H₂);
        end

        numericalNormByIteration = [ sum(numericalNorm[l,1:i])/i for i in 1:maxNumbOfIter ]

        folder = jldopen("./Plotting/Normalization/Data/Normalization_$(namespace)_L$(L)_$(maxNumbOfIter).jld2", "w");
        folder["numericalNorm"] = numericalNorm[l,:];
        folder["numericalNormByIteration"] = numericalNormByIteration;
        close(folder);

    end
    
end


# L′s = [14];
# maxNumbOfIter = 5000;
# namespace = H2;
# GetNormalizationData(L′s, maxNumbOfIter, namespace)

L′s = [12];
maxNumbOfIter = 5000;
namespace = H4;

GetNormalizationData(L′s, maxNumbOfIter, namespace)



