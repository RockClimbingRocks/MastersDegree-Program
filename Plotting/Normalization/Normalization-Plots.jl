

using Plots
pythonplot()
# using PyPlot
using JLD2 
using LinearAlgebra
using LaTeXStrings

# include("../../Helpers/FermionAlgebra.jl");
# using .FermionAlgebra;

include("../../Hamiltonians/H2.jl")
using .H2

include("../../Hamiltonians/H4.jl")
using .H4

include("../../Helpers/ChaosIndicators.jl");
using .ChaosIndicators;

include("../../Helpers/OperationsOnHamiltonian.jl");
using .OperationsOnH

colors = ["dodgerblue" "darkviolet" "limegreen" "indianred" "magenta" "darkblue" "aqua" "deeppink" "dimgray" "red" "royalblue" "slategray" "black" "lightseagreen" "forestgreen" "palevioletred"]
markers = ["o","x","v","*","H","D","s","d","P","2","|","<",">","_","+",","]
markers_line = ["-o","-x","-v","-H","-D","-s","-d","-P","-2","-|","-<","->","-_","-+","-,"]


#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------

function PlotNormalizationData(L′s, maxNumbOfIter, namespace)

    normAsFunctionOfIterations = plot(yscale=:log10, minorgrid=true, framestyle = :box, legend=true, title = "Convergency of normalization error");
    normAsFunctionOfSystemSize = plot(framestyle = :box, legend=false, title="Normalization error");
    
    for L in L′s
        folder = jldopen("./Data/Norm_$(namespace)_L$(L)_$(maxNumbOfIter)_true.jld2", "r");
        numericalNorm = folder["numericalNorm"];
        numericalNormByIteration = folder["numericalNormByIteration"];
        close(folder);
        

        plot!(normAsFunctionOfIterations, 1:maxNumbOfIter, abs.(numericalNormByIteration .- 1), label="L=$(L)");
        scatter!(normAsFunctionOfSystemSize, [L], [abs(numericalNormByIteration[end] - 1)], mc=:dodgerblue);
    end


    xlabel!(normAsFunctionOfIterations, "Number of Iterations");
    xlabel!(normAsFunctionOfSystemSize, "L");

    ratio = 3
    dim = 700
    p = plot(normAsFunctionOfIterations, normAsFunctionOfSystemSize, layout=(1,3), size=(Int(dim*ratio),dim))
    savefig(p, "./Images/Normalization_$(namespace)_L$(L)_$(maxNumbOfIter)_true.pdf")    

end 

L′s = [4, 6, 8, 10, 12];
maxNumbOfIter = 10000;
namespace = H2

PlotNormalizationData(L′s, maxNumbOfIter, namespace)



