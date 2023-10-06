##
using PyPlot;
using LinearAlgebra;
using JLD2;
using LaTeXStrings;
using Statistics;
using Polynomials;


include("../../Hamiltonians/H2.jl");
using .H2;
include("../../Hamiltonians/H4.jl");
using .H4;
include("../../Helpers/ChaosIndicators/ChaosIndicators.jl");
using .ChaosIndicators;


include("../../Helpers/ChaosIndicators/PrivateFunctions/SpectralFormFactor.jl");
using .SFF

colors = ["dodgerblue","darkviolet","limegreen","indianred","magenta","darkblue","aqua","deeppink","dimgray","red","royalblue","slategray","black","lightseagreen","forestgreen","palevioletred","lightcoral","lightgray","lightpink","peachpuff"];
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


qmin:: Int64 = -4; # set minimal value of q, in this case that would be 10^-4
qmax:: Int64 = 0;  # set maximal value of q, in this case that would be 10^-4
ρq:: Int64 = 6;

Nq:: Int64 = Int((qmax - qmin)*ρq +1);
x = LinRange(qmin, qmax, Nq); 
q′s = round.(10 .^(x), digits=5);


struct ReturnObject
    Es:: Vector{Float64}
    cis:: Tuple{Float64, Vector{Float64}, Vector{Float64}}

    function ReturnObject(Es:: Vector{Float64}, cis:: Tuple{Float64, Vector{Float64}, Vector{Float64}})
        new(Es, cis);
    end
end

function GetDirectory(L:: Int64, η:: Float64):: String
    # fileName:: String = "Hsyk_ChaosIndicators_L$(L)_Iter$(N)_q$(q)_eta$(η)"
    dir:: String = "./Plotting/ChaosIndicators/Data/L=$(L)/eta=$(η)/";
    return dir;
end

function GetFileName(L:: Int64, N:: Int64, q:: Float64, η:: Float64):: String
    fileName:: String = "CI_L$(L)_N$(N)_q$(q)_eta$(η)";
    return fileName;
end

function ReadData(L:: Int64, N:: Int64, q:: Float64, η:: Float64, connected:: Bool)
    dir:: String = GetDirectory(L, η);
    fileName:: String = GetFileName(L, N, q, η)

    folder = jldopen("$(dir)$(fileName).jld2", "r");
    # es_and_CIs = folder["Es_and_CIs"];
    K = folder["K"];
    Kc = folder["Kc"];
    τs = folder["τs"];
    close(folder);

    return connected ? Kc : K, τs;
end



function K̄(K::Vector{Float64}, τ::Vector{Float64}, Δ::Int64=100)
    N = length(τ) - 2*Δ;
    K̄ = Vector{Float64}(undef, N);
    τ′s = τ[Δ+1:N+Δ]
    for (iK̄, iK) in enumerate(Δ+1:N+Δ)
        K̄[iK̄] = mean(K[iK-Δ:iK+Δ])
    end

    return (K̄, τ′s)
end


function TthExtraction_crossSection(K::Vector{Float64}, τ′s::Vector{Float64}, ε::Float64=0.08)
    ΔK′s = abs.(  log10.(K ./ ChaosIndicators.Kgoe.(τ′s))  );

    # WE make linear regresion to determine τ_th more pricisely, k is a slope and n dispacement of a linear function 
    i = findall(x -> x<ε, ΔK′s)[1]

    i1 = i-1;
    i2 = i;

    k = (ΔK′s[i1] - ΔK′s[i2])/(τ′s[i1] - τ′s[i2]);
    n = ΔK′s[i1] - k*τ′s[i1];

    τ_th = (ε - n)/k;

    kth = K[i2]
    return (τ_th, kth);
end




function K′s(L:: Int64, N:: Int64, q′s:: Vector{Float64}, η::Float64, ax, smoothened:: Bool, connected:: Bool)

    τs = Vector{Float64}();
    for (i,q) in enumerate(q′s)
        Ks_, τs_ =  ReadData(L, N, q, η, connected);
        Ks, τs = smoothened ? K̄(Ks_, τs_) : (Ks_, τs_);

        ax.plot(τs, Ks, label=L"$q=%$(q)$", zorder=2);
    end

    ax.plot(τs, ChaosIndicators.Kgoe.(τs), label=L"$K_{GOE}$", linestyle = "dashed", color = "black");
    ax.axvline(x=1., ymin=minimum(ChaosIndicators.Kgoe.(τs)), color="gray", linestyle="dashed");
    
    # ax.scatter(τ_Th′s, ChaosIndicators.Kgoe.(τ_Th′s), facecolors="none", s=250, edgecolors="black");
    # ax.scatter(τ_Th′s, ChaosIndicators.Kgoe.(τ_Th′s), color="black", zorder=3, facecolors="none", s=75, edgecolors="black");
    
    ax.legend();
    ax.set_xscale("log");
    ax.set_yscale("log");
    ax.set_xlabel(L"$\tau$");
    ax.set_ylabel(L"$K(\tau)$");

    
end

function ΔK′s(L:: Int64, N:: Int64, q′s:: Vector{Float64}, η:: Float64, ax, smoothened:: Bool, connected:: Bool)

    # τs = Vector{Float64}();
    for (i,q) in enumerate(q′s)
        Ks_, τs_ =  ReadData(L, N, q, η, connected);
        Ks, τs = smoothened ? K̄(Ks_, τs_) : (Ks_, τs_);

        ΔK = log10.( Ks ./ ChaosIndicators.Kgoe.(τs));

        ax.plot(τs, ΔK, label=L"$q=%$(q)$", zorder=3);
        # ax.scatter([τ_Th], [ChaosIndicators.Kgoe(τ_Th)], facecolors="none", s=80, edgecolors="black", zorder=4);
    end
    ax.axhline(y=0.08, color="black", linestyle = "dashed", zorder=2)

    ax.set_xscale("log");
    ax.set_yscale("log");
    ax.set_title(L"$q=%$(q)$", y=0.85, zorder=5);
end



function Ks(L:: Int64, N:: Int64, q:: Float64, η′s::Vector{Float64}, ax, smoothened:: Bool, connected:: Bool)
    αz = 0.9; αk = 0.1
    k = (αz - αk)/(1- length(q′s));
    n = αz-k
    alphaa(x:: Int64) = k*x+n

    τs = Vector{Float64}();
    for (i,η) in enumerate(η′s)
        Ks_, τs_ =  ReadData(L, N, q, η, connected);
        Ks, τs = smoothened ? K̄(Ks_, τs_) : (Ks_, τs_);

        # println("start : end = ", τs_[1], " : ", τs_[end], " length = ", length(τs_));
        # println("start : end = ", τs[1], " : ", τs[end], " length = ", length(τs), "   averaged");

        τ_th, kth = TthExtraction_crossSection(Ks, τs, 100); 
        ax.plot(τs, Ks, label=L"$\eta=%$(η)$", zorder=2, alpha= alphaa(i));
        ax.scatter([τ_th], [kth], facecolors="none", s=80, edgecolors="black", zorder=4);
    end


    ax.plot(τs, ChaosIndicators.Kgoe.(τs), label=L"$K_{GOE}$", linestyle = "dashed", color = "black");
    ax.axvline(x=1., ymin=minimum(ChaosIndicators.Kgoe.(τs)), color="gray", linestyle="dashed");
    
    # ax.scatter(τ_Th′s, ChaosIndicators.Kgoe.(τ_Th′s), facecolors="none", s=250, edgecolors="black");
    # ax.scatter(τ_Th′s, ChaosIndicators.Kgoe.(τ_Th′s), color="black", zorder=3, facecolors="none", s=75, edgecolors="black");
    
    # ax.legend();
    ax.set_xscale("log");
    ax.set_yscale("log");
    # ax.set_xlabel(L"$\tau$");
    # ax.set_ylabel(L"$K(\tau)$");

    
end

function ΔKs(L:: Int64, N:: Int64, q:: Float64, η′s::Vector{Float64}, ax, smoothened:: Bool, connected:: Bool)
    αz = 0.9; αk = 0.1
    k = (αz - αk)/(1- length(q′s));
    n = αz-k
    alphaa(x:: Int64) = k*x+n


    # τs = Vector{Float64}();
    for (i,η) in enumerate(η′s)
        Ks_, τs_ =  ReadData(L, N, q, η, connected);
        Ks, τs = smoothened ? K̄(Ks_, τs_) : (Ks_, τs_);


        ΔK = log10.( Ks ./ ChaosIndicators.Kgoe.(τs));

        ax.plot(τs, ΔK, label=L"$\eta=%$(η)$", zorder=3, alpha= alphaa(i));
        # ax.scatter([τ_Th], [ChaosIndicators.Kgoe(τ_Th)], facecolors="none", s=80, edgecolors="black", zorder=4);
    end
    ax.axhline(y=0.08, color="black", linestyle = "dashed", zorder=2)


    ax.legend();
    ax.set_xscale("log");
    ax.set_yscale("log");
    # ax.set_title(L"$L=%$(L)$", y=0.85, zorder=5);
end



function Plot1(L:: Int64, N:: Int64, q′s:: Vector{Float64}, η:: Float64, connected:: Bool)

    n=5
    ny = Int(length(q′s)÷5)
    fig, ax = plt.subplots(nrows=ny, ncols=2, sharex =false, sharey=false);

    for i in 0:ny-1
        ax1 = ax[i+1,1];
        ax2 = ax[i+1,2];

        i1 = i*n+1
        i2 = (i+1)*n
        qs = q′s[i1:1:i2];


        K′s(L, N, qs, η, ax1, false, connected);
        ΔK′s(L, N, qs, η, ax2, false, connected);
    end

    plt.show()


end

L = 14;
N = 900;
η = 0.4;
connected=false;
# Plot1(L, N, q′s, η, connected);



function Plot2(L:: Int64, N:: Int64, q′s:: Vector{Float64}, ηs:: Vector{Float64}, connected:: Bool)
    fig, ax = plt.subplots(nrows=length(q′s), ncols=4, sharex =false, sharey=false);

    fig.suptitle(L"Thouless time extraction $L = %$(L)$, $N = %$(N)$", fontsize=19);
    ax[1,1].set_title(L"$K(\tau)$");
    ax[1,2].set_title(L"$\Delta K(\tau)$");
    ax[1,3].set_title(L"$ \overline{K}(\tau)$");
    ax[1,4].set_title(L"$\Delta \overline{K}(\tau)$");

    for (i,q) in enumerate(q′s)
        ax1 = ax[i,1]; ax1.set_ylabel(L"$q = %$(q)$");
        ax2 = ax[i,2];
        ax3 = ax[i,3];
        ax4 = ax[i,4];

        Ks(L, N, q, ηs, ax1,  false, connected);
        ΔKs(L, N, q, ηs, ax2, false, connected);
        Ks(L, N, q, ηs, ax3,  true, connected);
        ΔKs(L, N, q, ηs, ax4, true, connected);

        if q == q′s[end]
            ax1.set_xlabel(L"\tau");
            ax2.set_xlabel(L"\tau");
            ax3.set_xlabel(L"\tau");
            ax4.set_xlabel(L"\tau");
        end
    end

    plt.show();

end

L = 14;
N = 900;
ηs = [0.2, 0.3, 0.4, 0.5, 0.6];
connected=false;
is = [7,20,22,25];
q′s_ = q′s[is]
Plot2(L, N, q′s_, ηs, connected);