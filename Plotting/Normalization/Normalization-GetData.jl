# using PyPlot
using JLD2 
using LinearAlgebra
using LaTeXStrings
using CurveFit
using Statistics

function d(x)
    display(x);
    println();
end

# include("../../Helpers/FermionAlgebra.jl");
# using .FermionAlgebra;

include("../../Hamiltonians/H2.jl")
using .H2

include("../../Hamiltonians/H4.jl")
using .H4

include("../../Hamiltonians/Hgoe.jl")
using .Hgoe

include("../../Helpers/OperationsOnHamiltonian.jl");
using .OperationsOnH



function GetDirectory()
    # dir = "./Plotting/ChaosIndicators/Data/";
    dir = "./";
    return dir;
end

function GetFileName(L:: Int64, N:: Int64, nameSpace:: Module)
    fileName = "Normalization_$(nameSpace)_L$(L)_N$(N)";
    return fileName;
end



function GetNormalizationData(L:: Int, maxNumbOfIter:: Int64, namespace:: Module)
    numericalNorm = Vector{Float64}(undef, maxNumbOfIter);

    for i=1:maxNumbOfIter
        if mod(i,100)==0
            print(" ", round(i/maxNumbOfIter *100, digits=2), "%..." );
        end
        params = namespace.Params(L);
        H₂ = namespace.Ĥ(params);

        numericalNorm[i] = OperationsOnH.OperatorNorm(H₂);
    end

    # numericalNormByIteration = [ sum(numericalNorm[1:i])/i for i in 1:maxNumbOfIter ]

    directory = GetDirectory();
    fileName = GetFileName(L, maxNumbOfIter, namespace);

    folder = jldopen("$(directory)$(fileName).jld2", "w");
    folder["numericalNorm"]=numericalNorm;
    close(folder);


end

L = 16;
maxNumbOfIter = 10;
namespace = H2;

@time GetNormalizationData(L, maxNumbOfIter, namespace)

