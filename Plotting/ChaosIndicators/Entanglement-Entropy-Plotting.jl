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


struct ReturnObject
    Es:: Vector{Float64}
    cis:: Tuple{Float64, Vector{Float64}, Vector{Float64}}

    function ReturnObject(Es:: Vector{Float64}, cis:: Tuple{Float64, Vector{Float64}, Vector{Float64}})
        new(Es, cis);
    end
end

function GetDirectory(L:: Int64):: String
    # fileName:: String = "Hsyk_ChaosIndicators_L$(L)_Iter$(N)_q$(q)_eta$(η)"
    dir:: String = "./Plotting/ChaosIndicators/Data/L=$(L)/"
    return dir;
end

function GetFileName(L:: Int64, N:: Int64, q:: Float64):: String
    fileName:: String = "CI_L$(L)_Iter$(N)_q$(q)";
    return fileName;
end


function ReadData(L:: Int64, N:: Int64, q:: Float64)
    dir:: String = GetDirectory(L);
    fileName:: String = GetFileName(L, N, q)

    folder = jldopen("$(dir)$(fileName).jld2", "r");
    es_and_CIs = folder["Es_and_CIs"];
    close(folder);

    EE = map(x -> x.cis[3], es_and_CIs);
    return EE;

end




function PlotEntanglementEntropyOfq(L′s:: Vector{Int}, N′s:: Vector{Int}, q′s:: Vector{Float64}, ax, deviationType:: String, η:: Float64 =0.5)
    E_max = 1.;
    E_min = 0.55731;

    αz = 0.9; αk = 0.3
    k = (αz - αk)/(1- length(L′s));
    n = αz-k
    alphaa(i:: Int64) = k*i+n

    for (i,L) in enumerate(L′s) 
        normalizacija = log(2)*L/2 - 1/2;
        N = N′s[i];

        EE′s = Vector{Float64}(undef, length(q′s));
        σ′s_intrasample = Vector{Float64}(undef, length(q′s));
        σ′s_intresample = Vector{Float64}(undef, length(q′s));

        for (j,q) in enumerate(q′s); 
            EE = ReadData(L, N, q)./normalizacija

            EEs:: Vector{Float64} = map(x -> mean(x), EE);
            σs::  Vector{Float64} = map(x -> std(x), EE);

            EE′s[j] = mean(EEs);
            σ′s_intrasample[j] = mean(σs);
            σ′s_intresample[j] = std(EEs);
        end
        # ax.errorbar(q′s, E_q, yerr=σ_q, label=L"$L=%$(L)$", fmt=markers_line[i]);

        σ′s = deviationType == "intrasample" ? σ′s_intrasample : σ′s_intresample;

        ax.plot(q′s, EE′s, markers_line[i], label=L"$L=%$(L)$", color=colors[i]);
        ax.fill_between(q′s, EE′s-σ′s, EE′s+σ′s, alpha=alphaa(i), color=colors[i], label="Deviacija" );
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


function PlotHistogram_OfEntanglementEntropy(L:: Int64, N:: Int64, q′s:: Vector{Float64}, ax, deviationType:: String, η:: Float64 =0.5)
    normalizacija = log(2)*L/2 - 1/2;

    αz = 0.9; αk = 0.2
    k = (αz - αk)/(1- length(q′s));
    n = αz-k
    alphaa(x:: Int64) = k*x+n

    for (i,q) in enumerate(q′s); 
        EE = ReadData(L, N, q)./normalizacija

        EEs:: Vector{Float64} = Vector{Float64}();
        μ = 0.;
        σ = 0.;

        if deviationType == "intresample"
            EEs= map(x -> mean(x), EE);
            μ = mean(EEs);
            σ = std(EEs);
        elseif deviationType == "intrasample"
            j = Int(length(EE)÷2) # izberemo neko realizacijo, vseeno je katero, jst sm izbrou uvo v sredini.

            EEs = EE[j];
            μ = mean(EEs);
            σ = std(EEs);
        end

        # println(EEs_)
        ax.hist(EEs , bins= Int(√N*2÷ 3), density=true, alpha=alphaa(i), color=colors[i], label=L"$q=%$(q)$", zorder= 2);


        println(μ)
        println(σ)
        x = LinRange(0.4, 1.1, 1000);
        Normal(x) = exp(-0.5*(x-μ)^2 / σ^2) / (σ * √(2*π))
        ax.plot(x, Normal.(x), zorder=3, color=colors[i]);   
    end

    ax.legend()
    ax.set_xlabel(L"$EE(A:B)$");
    ax.set_title(L"$L = %$(L) $", y=0.8);

end




function Plot1( L′s:: Vector{Int64}, N′s:: Vector{Int64}, q′s:: Vector{Float64}, q′s_hist:: Vector{Float64}, deviationType:: String)
    fig, ax = plt.subplots(ncols=2)

    title = deviationType=="intresample" ? "Intresample deviacija" :   "Intrasample deviacija"
    fig.suptitle(title)

    PlotEntanglementEntropyOfq(L′s, N′s, q′s, ax[1], deviationType);
    PlotHistogram_OfEntanglementEntropy(L′s[end], N′s[end], q′s_hist, ax[2], deviationType);
    plt.show();
end

qmin:: Int64 = -4; # set minimal value of q, in this case that would be 10^-4
qmax:: Int64 = 0;  # set maximal value of q, in this case that would be 10^-4
ρq:: Int64 = 6;

Nq:: Int64 = Int((qmax - qmin)*ρq +1)
x = LinRange(qmin, qmax, Nq);
q′s = round.(10 .^(x), digits=5);
is = [1,5,10,15,16,17,18,22];
q′s_hist =q′s[is]#  map(x -> round(10^x, digits=5), LinRange(-3,0,15))[1:3:end];

println(q′s_hist)


L′s = [8, 10, 12, 14, 16]
N′s = [20_000, 5000, 1500, 900, 286]
deviationType = "intresample"

Plot1(L′s, N′s, q′s,q′s_hist,deviationType);





function Plot2(L1:: Int64, N1:: Int64, q′s1:: Vector{Float64}, L2′s:: Vector{Int64}, N2′s:: Vector{Int64}, q′s2:: Vector{Float64})
    fig, ax = plt.subplots(nrows = 1, ncols=2)

    PlotEntanglementEntropyOfq(L2′s, N2′s, q′s, ax[1], "intrasample");

    ax_histy = axax.inset_axes([0, 1.05, 1, 0.25], sharey=ax)

    PlotHistogram_OfEntanglementEntropy(L1, N1, ax_histy, q′s2, "compact");
    plt.show();
end


# L′s = [8,10,12]; L = L′s[end]
# N′s = [200, 200, 200]; N = N′s[end];
# q′s_position_forHistogram = [1,5,7,8,9,10,11,12];
# Plot2(L, N, L′s, N′s, q′s_position_forHistogram);

