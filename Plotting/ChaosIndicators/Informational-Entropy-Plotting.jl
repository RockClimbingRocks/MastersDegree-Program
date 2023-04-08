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



function PlotInformationalEntropy(L′s:: Vector{Int},maxNumbOfIter′s:: Vector{Int})

    fig, ax = plt.subplots()


    for (i,L) in enumerate(L′s) 
        folder = jldopen("./Plotting/ChaosIndicators/Data/InformationalEntropy_L$(L)_Iter$(maxNumbOfIter′s[i]).jld2", "r");
        S = folder["S"];
        q′s = folder["q′s"];
        close(folder);

        ax.plot(q′s, S, label="L=$(L)")
    end

    ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel("q")
    ax.set_ylabel("S")

    display(fig)
    plt.show()

end



# L′s_ie  = [8,10,12];
# maxNumbOfIter_ie = [200,200,200];
# PlotInformationalEntropy(L′s_ie,maxNumbOfIter_ie);




