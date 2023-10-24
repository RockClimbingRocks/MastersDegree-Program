using PyPlot;
using JLD2;
using LaTeXStrings;
using Statistics;

include("../HAMILTONIANS/ManyBody/WithoutInteractions/SYK2.jl");
using .SYK2;
include("../HAMILTONIANS/ManyBody/WithInteractions/SYK4.jl");
using .SYK4;
include("../HAMILTONIANS/ManyBody/WithoutInteractions/SYK2LOC.jl");
using .SYK2LOC
include("../HAMILTONIANS/ManyBody/WithInteractions/SYK4LOC_CM.jl");
using .SYK4LOC_CM;
include("../HAMILTONIANS/ManyBody/WithInteractions/SYK4LOC_MINMAX.jl");
using .SYK4LOC_MINMAX;
include("../HAMILTONIANS/ManyBody/WithInteractions/SYK4LOC_ROK.jl");
using .SYK4LOC_ROK;

include("../PATHS/Paths.jl");
using .PATH;

include("../HAMILTONIANS/FermionAlgebra.jl");
using .FermionAlgebra;
include("../CHAOS-INDICATORS/ManyBody/ChaosIndicators.jl");
using .MBCI;
include("../GET-DATA/GetInteractionStrengths.jl");
using .IntStrength;




colors = ["dodgerblue","darkviolet","limegreen","indianred","magenta","darkblue","aqua","deeppink","dimgray","red","royalblue","slategray","black","lightseagreen","forestgreen","palevioletred","lightcoral","lightgray","lightpink","peachpuff"]
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



function r_of_λ(L:: Int64, λ′s:: Vector{Float64}, N:: Int64, a::Float64, b::Float64, H2::Module, H4::Module, ax)
    dir = "/Volumes/rokpintar-f1home/GIT-TEST/MastersDegree-Program/DATA";
    subDir = PATH.GetSubDirectory("MB");
    fileName = PATH.GetFileName(H2, H4, L, N);
    
    
    
    r′s = Vector{Float64}(undef, length(λ′s));
    jldopen("$(dir)$(subDir)$(fileName)", "r") do file
        for (j,λ) in enumerate(λ′s)
            groupPath = PATH.GetGroupPath(λ, a, b);

            r′s[j] = mean(file["$(groupPath)/r′s"]);
            # Es′s = file["$(groupPath)/Es′s"];
            # r′s = file["$(groupPath)/r′s"];
            # Ks = file["$(groupPath)/Ks"];
            # Kcs = file["$(groupPath)/Kcs"];
            # τs = file["$(groupPath)/τs"];
        end
    end;

    ax.plot(λ′s, r′s, label=L"$L=%$(L)$", zorder=3);

end




function Plot_r()
    L′s = [10, 12];
    N′s = [5000, 3000];
    
    a = 0.5;
    b = 0.05;
    
    H2 = SYK2LOC;
    H4 = SYK4LOC_MINMAX;
    
    λ′s = IntStrength.λ̂′s();
    
    
    fig, ax = plt.subplots(ncols=1)
    
    
    for (i,L) in enumerate(L′s)
        N = N′s[i];

        r_of_λ(L, λ′s, N, a, b, H2, H4, ax);
    end


    ax.axhline(y = 0.3863, color="black", linestyle="dashed")
    ax.axhline(y = 0.5307, color="black", linestyle="dashed")
    
    ax.set_xscale("log");
    ax.set_xlabel(L"q");
    ax.set_ylabel(L"r");
    ax.legend();
    
    plt.show()


end


Plot_r()