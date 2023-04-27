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



function PlotEntanglementEntropyOfq(L′s:: Vector{Int}, N′s:: Vector{Int}, ax, partitionType:: String="compact")
    PartitionTypeToIntiger = Dict{String, Integer}("compact" => 1, "nonCompact" => 2);
    partition = PartitionTypeToIntiger[partitionType];
    E_max = 1.;
    E_min = 0.55731;


    for (i,L) in enumerate(L′s) 
        norm = log(2)*L/2 - 1/2

        folder = jldopen("./Plotting/ChaosIndicators/Data/EntanglementEntropy2_L$(L)_Iter$(N′s[i]).jld2", "r");
        E′s′s = folder["E′s′s"]./norm;
        σ′s′s = folder["σ′s′s"];
        q′s = folder["q′s"];
        close(folder);

        E_q = mean(E′s′s[partition,:,:], dims=2)[:,1]
        σ_q_intresample = std(E′s′s[partition,:,:], dims=2)[:,1]
        σ_q_intrasample = mean(σ′s′s[partition,:,:], dims=2)[:,1]

        # ax.errorbar(q′s, E_q, yerr=σ_q, label=L"$L=%$(L)$", fmt=markers_line[i]);

        ax.plot(q′s, E_q, markers_line[i], label=L"$L=%$(L)$", color=colors[i]);
        ax.fill_between(q′s, E_q-σ_q_intresample, E_q+σ_q_intresample, alpha=0.2 + (i)*0.1, color=colors[i], label="Deviacija" );
    end


    ax.legend();
    ax.set_xscale("log");
    ax.set_xlabel(L"$q$");
    ax.set_ylabel(L"$EE(A:B)$");

    # ax.set_title("Kompaktna, simetrična biparticija");
    # ax[2].set_title("Nekompaktna biparticija");

    ax.axhline(y=E_min, color="black", linestyle="dashed");
    ax.axhline(y=E_max, color="black", linestyle="dashed");

end


function PlotHistogram_OfEntanglementEntropy(L:: Int64, N:: Int64, ax, q′s_position_forHistogram:: Vector{Int64}, partitionType:: String="compact")
    PartitionTypeToIntiger = Dict{String, Integer}("compact" => 1, "nonCompact" => 2);
    partition = PartitionTypeToIntiger[partitionType];
    norm = log(2)*L/2 - 1/2

    folder = jldopen("./Plotting/ChaosIndicators/Data/EntanglementEntropy2_L$(L)_Iter$(N).jld2", "r");
    E′s′s = folder["E′s′s"]./norm;
    σ′s′s = folder["σ′s′s"];
    q′s = round.(folder["q′s"], digits=3);
    close(folder);

    μ_q = mean(E′s′s[partition,:,:], dims=2)[:,1]
    # σ_q = mean(σ′s′s[partition,:,:], dims=2)[:,1]
    σ_q = std(E′s′s[partition,:,:], dims=2)[:,1]

    k = 0.9 /(1- length(q′s));
    n = 1-k
    alpha(x:: Int64) = k*x+n

    for i in q′s_position_forHistogram
        q = q′s[i];
        E′s = E′s′s[1,i,:];
        μ = mean(E′s′s[1,i,:]);
        σ = std(E′s′s[1,i,:]);
    
        ax.hist(E′s , bins= Int(√N*2÷ 3), density=true, alpha=alpha(i), color=colors[i], label=L"$q=%$(q)$", zorder= 2);

        x = LinRange(0.4, 1.1, 1000);
        Normal(x) = exp(-0.5*(x-μ)^2 / σ^2) / (σ * √(2*π))
        ax.plot(x, Normal.(x), color=colors[i], zorder=3);        
    end


    ax.legend()
    ax.set_xlabel(L"$EE(A:B)$");
    ax.set_title(L"$L = %$(L) $", y=0.8);

end




function Plot1( L′s:: Vector{Int64}, N′s:: Vector{Int64})
    fig, ax = plt.subplots(ncols=2)

    PlotEntanglementEntropyOfq(L′s, N′s, ax[1], "compact");
    PlotEntanglementEntropyOfq(L′s, N′s, ax[2], "nonCompact");

    plt.show();
end

# L′s = [8,10,12];
# maxNumbOfIter′s = [200, 200, 200];
# Plot1(L′s, maxNumbOfIter′s);




function Plot2(L1:: Int64, N1:: Int64, L2′s:: Vector{Int64}, N2′s:: Vector{Int64}, q′s_position_forHistogram:: Vector{Int64})
    fig, ax = plt.subplots(nrows = 1, ncols=2)

    PlotEntanglementEntropyOfq(L2′s, N2′s, ax[1], "compact");
    PlotHistogram_OfEntanglementEntropy(L1, N1, ax[2], q′s_position_forHistogram, "compact");
    plt.show();
end


# L′s = [8,10,12]; L = L′s[end]
# N′s = [200, 200, 200]; N = N′s[end];
# q′s_position_forHistogram = [1,5,7,8,9,10,11,12];
# Plot2(L, N, L′s, N′s, q′s_position_forHistogram);
