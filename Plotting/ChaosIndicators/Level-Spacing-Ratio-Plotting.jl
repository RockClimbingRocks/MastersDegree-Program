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

function PlotLevelSpacingRatio(L′s:: Vector{Int},maxNumbOfIter′s:: Vector{Int})
    fig, ax = plt.subplots()
    qs = Vector{Float64}();

    for (i,L) in enumerate(L′s) 

        folder = jldopen("./Plotting/ChaosIndicators/Data/LevelSpacingRatio_L$(L)_Iter$(maxNumbOfIter′s[i]).jld2", "r");
        r = folder["r"];
        q′s = folder["q′s"];
        close(folder);

        qs = q′s

        ax.plot(q′s, r, label=L"$L=%$(L)$")
    end

    ax.plot(qs, 0.3863 .* ones(Float64, length(qs)), color="black", linestyle="dashed")
    ax.plot(qs, 0.5359 .* ones(Float64, length(qs)), color="black", linestyle="dashed")
    
    ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel(L"q")
    ax.set_ylabel(L"r")
    display(fig)
    plt.show()

end



function PlotLevelSpacingRatio_scaled(L′s:: Vector{Int},maxNumbOfIter′s:: Vector{Int})
    r_poisson = 0.3863;
    r_goe = 0.5359;
    fig, ax = plt.subplots()
    qs = Vector{Float64}();

    for (i,L) in enumerate(L′s) 

        folder = jldopen("./Plotting/ChaosIndicators/Data/LevelSpacingRatio_L$(L)_Iter$(maxNumbOfIter′s[i]).jld2", "r");
        r = folder["r"];
        q′s = folder["q′s"];
        close(folder);

        qs = q′s

        r_effective = @. (r-r_poisson)/(r_goe-r_poisson);
        ax.plot(q′s, r_effective, label=L"$L=%$(L)$")
    end

    ax.plot(qs, ones(Float64, length(qs)), color="black", linestyle="dashed")
    # ax.plot(qs, 0.5359 .* ones(Float64, length(qs)), color="black", linestyle="dashed")
    
    ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel(L"$q$")
    ax.set_ylabel(L"$r$")
    display(fig)
    plt.show()

end




L′s_lsr = [6,8,10,12];
maxNumbOfIter_lsr = [500,500,500,500];
PlotLevelSpacingRatio_scaled(L′s_lsr, maxNumbOfIter_lsr);

