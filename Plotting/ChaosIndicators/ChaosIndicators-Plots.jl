##
using PyPlot;
using LinearAlgebra;
using JLD2;
using LaTeXStrings;

include("../../Hamiltonians/H2.jl");
using .H2;
include("../../Hamiltonians/H4.jl");
using .H4;
include("../../Helpers/ChaosIndicators/ChaosIndicators.jl");
using .ChaosIndicators;


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



function PlotEntanglementEntropyInPyPlot(L′s:: Vector{Int}, maxNumbOfIter′s:: Vector{Int})
    # plt.style.use(["don_custom"])
    fig, ax = plt.subplots(ncols=2, figsize=(9, 4))


    for (i,L) in enumerate(L′s) 
        # D = binomial(L,L÷2)
        D=Int(2^L)
        norm = log(D÷2) - 1/2
        # norm = 1.

        folder = jldopen("ChaosIndicators/Data-ChaosIndicators/EntanglementEntropy_L$(L)_Iter$(maxNumbOfIter′s[i]).jld2", "r");
        E = folder["E"]./norm;
        q′s = folder["q′s"];
        close(folder);

        ax[1].plot(q′s, E[1,:], label="L=$(L)")
        ax[2].plot(q′s, E[2,:], label="L=$(L)")
    end

    for i in 1:2
        ax[i].legend()
        ax[i].set_xscale("log")
        ax[i].set_xlabel("q")
        ax[i].set_ylabel("E(A:B)")
    end

    ax[1].set_title("Kompaktna, simetrična biparticija");
    ax[2].set_title("Nekompaktna biparticija");


    display(fig)
    plt.show()

end

function PlotInformationalEntropy(L′s:: Vector{Int},maxNumbOfIter′s:: Vector{Int})

    fig, ax = plt.subplots()


    for (i,L) in enumerate(L′s) 
        folder = jldopen("ChaosIndicators/Data-ChaosIndicators/InformationalEntropy_L$(L)_Iter$(maxNumbOfIter′s[i]).jld2", "r");
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

function PlotLevelSpacingRatio(L′s:: Vector{Int},maxNumbOfIter′s:: Vector{Int})
    fig, ax = plt.subplots()
    qs = Vector{Float64}();

    for (i,L) in enumerate(L′s) 

        folder = jldopen("ChaosIndicators/Data-ChaosIndicators/LevelSpacingRatio_L$(L)_Iter$(maxNumbOfIter′s[i]).jld2", "r");
        r = folder["r"];
        q′s = folder["q′s"];
        close(folder);

        qs = q′s

        ax.plot(q′s, r, label=L"L=%$(L)")
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


# L′s_lsr = [6,8,10,12];
# maxNumbOfIter_lsr = [500,500,500,500];
# PlotLevelSpacingRatio(L′s_lsr, maxNumbOfIter_lsr);




# L′s_ie  = [8,10,12];
# maxNumbOfIter_ie = [200,200,200];
# PlotInformationalEntropy(L′s_ie,maxNumbOfIter_ie);




L′s = [8,10,12];
maxNumbOfIter′s = [200, 200, 200]
PlotEntanglementEntropyInPyPlot(L′s, maxNumbOfIter′s)

