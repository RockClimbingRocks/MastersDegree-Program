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

    S = map(x -> x.cis[2], es_and_CIs);
    return S;

end



function PlotEntanglementEntropyOfq(L′s:: Vector{Int}, N′s:: Vector{Int}, q′s:: Vector{Float64}, ax, deviationType:: String, η:: Float64 =0.5)
    S_max = 1.;

    Sgoe(L) = log(0.48*binomial(L,L÷2));
    
    αz = 0.9; αk = 0.2
    k = (αz - αk)/(1- length(L′s));
    n = αz-k
    alphaa(x:: Int64) = k*x+n

    for (i,L) in enumerate(L′s) 
        N = N′s[i];

        S′s = Vector{Float64}(undef, length(q′s));
        σ′s_intrasample = Vector{Float64}(undef, length(q′s));
        σ′s_intresample = Vector{Float64}(undef, length(q′s));

        for (j,q) in enumerate(q′s)
            S = ReadData(L, N, q)./Sgoe(L);

            Ss:: Vector{Float64} = map(x -> mean(x), S);
            σs::  Vector{Float64} = map(x -> std(x), S);

            S′s[j] = mean(Ss);
            σ′s_intrasample[j] = mean(σs);
            σ′s_intresample[j] = std(Ss);
        end
        # ax.errorbar(q′s, E_q, yerr=σ_q, label=L"$L=%$(L)$", fmt=markers_line[i]);

        σ′s = deviationType == "intrasample" ? σ′s_intrasample : σ′s_intresample;

        ax.plot(q′s, S′s, markers_line[i], label=L"$L=%$(L)$", color=colors[i]);
        ax.fill_between(q′s, S′s-σ′s, S′s+σ′s, alpha=alphaa(i), color=colors[i], label="Deviacija" );
    end


    ax.legend();
    ax.set_xscale("log");
    ax.set_xlabel(L"$q$");
    ax.set_ylabel(L"$S(A:B)$");

    ax.axhline(y=S_max, color="black", linestyle="dashed");

end

function PlotHistogram_OfEntanglementEntropy(L:: Int64, N:: Int64, q′s:: Vector{Float64}, ax, deviationType:: String, η:: Float64 =0.5)
    Sgoe(L) = log(0.48*binomial(L,L÷2));

    αz = 0.9; αk = 0.2
    k = (αz - αk)/(1- length(q′s));
    n = αz-k;
    alphaa(x:: Int64) = k*x+n;
    for (i,q) in enumerate(q′s)
        S = ReadData(L, N, q)./Sgoe(L)

        Ss:: Vector{Float64} = Vector{Float64}();
        μ = 0.;
        σ = 0.;

        if deviationType == "intresample"
            Ss= map(x -> mean(x), S);
            μ = mean(Ss);
            σ = std(Ss);
        elseif deviationType == "intrasample"
            j = Int(length(S)÷2)

            Ss = S[j];
            μ = mean(Ss);
            σ = std(Ss);
        end

        ax.hist(Ss , bins= Int(√N*2÷ 3), density=true, alpha=alphaa(i), color=colors[i], label=L"$q=%$(q)$", zorder= 2);


        println(μ)
        println(σ)
        x = LinRange(0.4, 1.1, 1000);
        Normal(x) = exp(-0.5*(x-μ)^2 / σ^2) / (σ * √(2*π))
        ax.plot(x, Normal.(x), zorder=3);   
    end

    ax.legend()
    ax.set_xlabel(L"$S(A:B)$");
    ax.set_title(L"$L = %$(L) $", y=0.8);

end


function Plot1( L′s:: Vector{Int64}, N′s:: Vector{Int64}, q′s:: Vector{Float64}, q′s_hist:: Vector{Float64}, deviationType:: String)
    fig, ax = plt.subplots(ncols=2)

    title = deviationType=="intresample" ? "Intresample deviacija" : "Intrasample deviacija"
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
is = [1,10,13,15,16,17,18,22];
q′s_hist =q′s[is]#  map(x -> round(10^x, digits=5), LinRange(-3,0,15))[1:3:end];

L′s = [8, 10, 12, 14, 16];
N′s = [20_000, 5000, 1500, 900, 286];

deviationType = "intrasample"
Plot1(L′s, N′s, q′s,q′s_hist,deviationType);



function PlotSₘofEₘ(L:: Int, N:: Int, q:: Float64, ax, η:: Float64)

    dir:: String = GetDirectory(L);
    fileName:: String = GetFileName(L, N, q)

    folder = jldopen("$(dir)$(fileName).jld2", "r");
    es_and_CIs = folder["Es_and_CIs"];
    close(folder);

    S = map(x -> x.cis[2], es_and_CIs);
    E = map(x -> x.Es, es_and_CIs);

    println(size(S))
    println(size(E))

    Sₘ = S[2];
    Eₘ = E[2];


    println(size(Sₘ))
    println(size(Eₘ))


    D = binomial(L, L÷2);
    Sgoe = log(0.48*D);

    ax.scatter(Eₘ./L, Sₘ./Sgoe);
    ax.axhline(y=1., color = "black", linestyle = "dashed");

end



function Plot2(L′s:: Vector{Int}, N′s:: Vector{Int}, q′s:: Vector{Float64}, η:: Float64=0.5)
    fig, ax = plt.subplots(nrows=length(L′s), ncols=length(q′s), sharex =true, sharey=true);

    for (i,L) in enumerate(L′s)
        println(L, "--------")
        D = binomial(L,L÷2);
        η_ = 0.3;
        i₁ = Int((D - (D*η_)÷1)÷2);
        i₂ = Int(i₁ + (D*η_)÷1);

        N = N′s[i]
        
        for (j,q) in enumerate(q′s)
            println("   ", q)
            PlotSₘofEₘ(L, N, q, ax[i,j], η)

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



qmin:: Int64 = -4; # set minimal value of q, in this case that would be 10^-4
qmax:: Int64 = 0;  # set maximal value of q, in this case that would be 10^-4
ρq:: Int64 = 6;

Nq:: Int64 = Int((qmax - qmin)*ρq +1)
x = LinRange(qmin, qmax, Nq);
q′s = round.(10 .^(x), digits=5);
q′s_hist =q′s[1:2:end]#  map(x -> round(10^x, digits=5), LinRange(-3,0,15))[1:3:end];

L′s = [8, 10, 12, 14, 16]
N′s = [20_000, 5000, 1500, 900, 178]
deviationType = "intresample"

# Plot2(L′s, N′s, q′s);




