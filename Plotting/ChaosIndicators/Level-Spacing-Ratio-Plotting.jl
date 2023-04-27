##
using PyPlot;
using LinearAlgebra;
using JLD2;
using LaTeXStrings;
using Statistics;

include("../../Hamiltonians/H2.jl");
using .H2;
include("../../Hamiltonians/H4.jl");
using .H4;
include("../../Helpers/ChaosIndicators/ChaosIndicators.jl");
using .ChaosIndicators;


include("../../Helpers/ChaosIndicators/PrivateFunctions/SpectralFormFactor.jl");
using .SFF

colors = ["dodgerblue","darkviolet","limegreen","indianred","magenta","darkblue","aqua","deeppink","dimgray","red","royalblue","slategray","black","lightseagreen","forestgreen","palevioletred"]
markers = ["o","x","v","*","H","D","s","d","P","2","|","<",">","_","+",","]
markers_line = ["-o","-x","-v","-H","-D","-s","-d","-P","-2","-|","-<","->","-_","-+","-,"]
line_style   = ["solid", "dotted", "dashed", "dashdot"]

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["text.usetex"] = true
rcParams["font.size"] = 12
rcParams["font.family"] = "serif"
rcParams["font.sans-serif"] = "Computer Modern"

rcParams["axes.titlesize"] = 17
rcParams["axes.labelsize"] = 15
rcParams["xtick.labelsize"] = 11
rcParams["ytick.labelsize"] = 11
rcParams["legend.fontsize"] = 10

rcParams["axes.grid"] = true
rcParams["axes.grid.which"] = "both"
rcParams["savefig.bbox"] = "tight"

rcParams["axes.prop_cycle"] = PyPlot.matplotlib.cycler(color=colors)



function GetFileName(L:: Int64, N:: Int64, q:: Float64, η:: Float64, connected:: Bool):: String
    fileName:: String = "Hsyk_ChaosIndicators_L$(L)_Iter$(N)_q$(q)_eta$(η)"
    return fileName;
end



function PlotLevelSpacingRatio(L′s:: Vector{Int}, q′s:: Vector{Float64}, N′s:: Vector{Int}, ax)

    for (i,L) in enumerate(L′s) 
        N = N′s[i];

        r′s = Vector{Float64}(undef, N);
        for (j,q) in enumerate(q′s)
            folder = jldopen("./Plotting/ChaosIndicators/Data/LevelSpacingRatio_L$(L)_Iter$(maxNumbOfIter′s[i]).jld2", "r");
            rs = folder["r"];
            close(folder);

            r′s[j] = mean(rs);
        end

        ax.plot(q′s, r′s, label=L"$L=%$(L)$")
    end

    ax.axhline(y = 0.3863, color="black", linestyle="dashed")
    ax.axhline(y = 0.5359, color="black", linestyle="dashed")
    
    ax.legend();
    ax.set_xscale("log");
    ax.set_xlabel(L"q");
    ax.set_ylabel(L"r");
    
    plt.show()

end




function Fig1(L′s:: Vector{Int}, N′s:: Vector{Int}, q′s:: Vector{Float64})
    fig, ax = plt.subplots()

    PlotLevelSpacingRatio(L′s, q′s, N′s, ax);

end

x = LinRange(-3, 0, 15);
q′s = round.(10 .^(x), digits=5);

L′s = [8]
N′s = [1000]

Fig1(L′s, N′s, q′s)