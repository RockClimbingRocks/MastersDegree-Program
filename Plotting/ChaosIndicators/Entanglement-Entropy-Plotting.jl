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



function PlotEntanglementEntropyInPyPlot(L′s:: Vector{Int}, maxNumbOfIter′s:: Vector{Int})
    # plt.style.use(["don_custom"])
    fig, ax = plt.subplots(ncols=2, figsize=(9, 4))

    S_max = 1.;
    S_min = 0.6;


    for (i,L) in enumerate(L′s) 
        # D = binomial(L,L÷2)
        D=Int(2^L)
        norm = log(2)*L/2 - 1/2
        # norm = 1.

        folder = jldopen("./Plotting/ChaosIndicators/Data/EntanglementEntropy_L$(L)_Iter$(maxNumbOfIter′s[i]).jld2", "r");
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



L′s = [8,10,12];
maxNumbOfIter′s = [200, 200, 200]
PlotEntanglementEntropyInPyPlot(L′s, maxNumbOfIter′s)


