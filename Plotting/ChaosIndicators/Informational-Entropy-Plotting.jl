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



function PlotInformationalEntropy(L′s:: Vector{Int},maxNumbOfIter′s:: Vector{Int}, ax)

    for (i,L) in enumerate(L′s) 
        folder = jldopen("./Plotting/ChaosIndicators/Data/InformationalEntropy_L$(L)_Iter$(maxNumbOfIter′s[i]).jld2", "r");
        S = folder["S"];
        q′s = folder["q′s"];
        close(folder);

        ax.plot(q′s, S, label=L"$L=%$(L)$")
    end

    ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel(L"$q$")
    ax.set_ylabel(L"$S$")

    plt.show()
end


function PlotSₘofEₘ(L:: Int, q:: Float64, ax)

    D = binomial(L, L÷2);
    Sgoe = log(0.48*D);

    folder = jldopen("./Plotting/ChaosIndicators/Data/InformationalEntropyAsFunctionOfEnergy_L$(L)_q$(q).jld2", "r");
    Sₘ = folder["Sₘ"];
    Eₘ = folder["Eₘ"];
    close(folder);

    ax.scatter(Eₘ/L, Sₘ./Sgoe);

    ax.axhline(y=1., color = "black", linestyle = "dashed");

end


function Fig1(L′s:: Vector{Int}, N′s:: Vector{Int})
    fig, ax = plt.subplots();
    PlotInformationalEntropy(L′s, N′s, ax)


    plt.show()
end


function Fig2(L′s:: Vector{Int}, q′s:: Vector{Float64})
    fig, ax = plt.subplots(nrows=length(L′s), ncols=length(q′s), sharex =true, sharey=true);

    for (i,L) in enumerate(L′s)
        println(L, "--------")
        D = binomial(L,L÷2);
        η = 0.3;
        i₁ = Int((D - (D*η)÷1)÷2);
        i₂ = Int(i₁ + (D*η)÷1);

        
        for (j,q) in enumerate(q′s)
            println("   ", q)
            PlotSₘofEₘ(L, q, ax[i,j])

            ax[i,j].set_title(L"$L = %$(L) \; \; q = %$(round(q, digits=5))$", y=0.85);


            # ax[i,j].fill_between(x, 0, 1, where=y > theta,
            #     facecolor="green", alpha=0.5, transform=trans)

            if i==length(L′s)
                ax[i,j].set_xlabel(L"E_m / L");
            end

            if j==1
                ax[i,j].set_ylabel(L"$S_m / S_{GOE}$");
            end

        end
    end

    # ax.set_xscale("log")
    # ax.set_xlabel("q")
    # ax.set_ylabel("S")


    plt.show()
end



L′s  = [8,10,12];
maxNumbOfIter_ie = [200,200,200];
Fig1(L′s,maxNumbOfIter_ie);

# L′s = [10,12,14];
# q′s = [0.001, 0.1, 0.15, 0.5, 1.];
# # 0.1, 1.
# Fig2(L′s, q′s);




