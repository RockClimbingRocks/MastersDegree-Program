using PyPlot;
using LinearAlgebra;
using JLD2;
using LaTeXStrings;
using Statistics;
using Polynomials;
using CurveFit;

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




qmin1:: Int64 = -4; # set minimal value of q, in this case that would be 10^-4
qmax1:: Int64 = 0;  # set maximal value of q, in this case that would be 10^-4
ρq1:: Int64 = 6;

Nq1:: Int64 = Int((qmax1 - qmin1)*ρq1 +1);
x1 = LinRange(qmin1, qmax1, Nq1); 
q′s1 = round.(10 .^(x1), digits=5);


qmin2:: Int64 = -2; # set minimal value of q, in this case that would be 10^-4
qmax2:: Float64 = log10(0.2);  # set maximal value of q, in this case that would be 10^-4
x2 = LinRange(qmin2, qmax2, 8)[2:end-1]; 
q′s2 = round.(10 .^(x2), digits=5);
println(q′s2)

q′s = vcat(q′s1, q′s2);



function GetDirectory():: String
    dir:: String = "./Plotting/ChaosIndicators/Data";
    return dir;
end

function GetFileName(L:: Int64, N:: Int64):: String
    fileName:: String = "ChaosIndicators_L$(L)_N$(N).jld2";
    return fileName;
end

function GetGroupPath(q:: Float64, η:: Float64):: String
    groupPath = "q=$q/η=$η";
    return groupPath
end


struct ReturnObject
    Es:: Vector{Float64}
    cis:: Tuple{Float64, Vector{Float64}, Vector{Float64}}

    function ReturnObject(Es:: Vector{Float64}, cis:: Tuple{Float64, Vector{Float64}, Vector{Float64}})
        new(Es, cis);
    end
end

function K̄(K::Vector{Float64}, τ::Vector{Float64}, Δ::Int64=50)
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
    i = findall(x -> x>ε, ΔK′s)[end]

    i1 = i;
    i2 = i+1;

    k = (ΔK′s[i1] - ΔK′s[i2])/(τ′s[i1] - τ′s[i2]);
    n = ΔK′s[i1] - k*τ′s[i1];

    τ_th = (ε - n)/k;

    kth = K[i2]
    return (τ_th, kth);
end

function CreatingJLD2Subgroups(L:: Int64, N:: Int64, q′s:: Vector{Float64}, η′s::Vector{Float64})

    jldopen("./Plotting/ChaosIndicators/Data/ChaosIndicators_L$(L)_N$(N)_new.jld2", "w") do fileNew
        jldopen("./Plotting/ChaosIndicators/Data/ChaosIndicators_L$(L)_N$(N)_old.jld2", "r") do file
            for (i,q) in enumerate(q′s)
                for (n, η) in enumerate(η′s)

                    groupPath:: String = GetGroupPath(q, η);

                    Es = file["$groupPath/Es"];
                    rs = file["$groupPath/rs"];
                    Ss = file["$groupPath/Ss"];
                    EEs = file["$groupPath/EEs"];
                    Ks = file["$groupPath/Ks"];
                    τs = file["$groupPath/τs"];
                    Ks_smooth = file["$groupPath/K̄s"];
                    τs_smooth = file["$groupPath/τ̄s"];
                    Ksc = file["$groupPath/Kcs"];
                    τ_Th = file["$groupPath/τ_th"];
                    t_H = file["$groupPath/t_H"];
                    t_Th = file["$groupPath/t_Th"];
                    g = file["$groupPath/g"];         





                    fileNew["$groupPath/Es"] = Es;
                    fileNew["$groupPath/rs"] = rs;
                    fileNew["$groupPath/Ss"] = Ss;
                    fileNew["$groupPath/EEs"] = EEs;


                    # Ks_smooth, τs_smooth = K̄(Ks, τs);
                    # τ_Th, kth = TthExtraction_crossSection(Ks_smooth, τs_smooth); 
                    # t_H = SFF.t̂_H(Es);
                    # t_Th = SFF.t̂_Th(τ_Th, t_H);
                    # g = SFF.ĝ(τ_Th);
                    # groupPath = GetGroupPath(q, η);
                    fileNew["$groupPath/K/Ks"] = Ks;
                    fileNew["$groupPath/K/τs"] = τs;
                    fileNew["$groupPath/K/K̄s"] = Ks_smooth;
                    fileNew["$groupPath/K/τ̄s"] = τs_smooth;
                    fileNew["$groupPath/K/τ_th"] = τ_Th;
                    fileNew["$groupPath/K/t_Th"] = t_Th;
                    fileNew["$groupPath/K/t_H"] = t_H;
                    fileNew["$groupPath/K/g"] = g;        


                    Ks_smoothc, τs_smoothc = K̄(Ksc, τs);
                    τ_Thc, kth = TthExtraction_crossSection(Ks_smoothc, τs_smoothc); 
                    t_Hc = SFF.t̂_H(Es);
                    t_Thc = SFF.t̂_Th(τ_Thc, t_Hc);
                    gc = SFF.ĝ(τ_Thc);
                    fileNew["$groupPath/Kc/Ks"] = Ksc;
                    fileNew["$groupPath/Kc/τs"] = τs;
                    fileNew["$groupPath/Kc/K̄s"] = Ks_smoothc;
                    fileNew["$groupPath/Kc/τ̄s"] = τs_smoothc;
                    fileNew["$groupPath/Kc/τ_th"] = τ_Thc;
                    fileNew["$groupPath/Kc/t_Th"] = t_Thc;
                    fileNew["$groupPath/Kc/t_H"] = t_Hc;
                    fileNew["$groupPath/Kc/g"] = gc;        
                    
                end
            end
        end
    end
end

# ηs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7];
# L = 18;
# N = 3; 
# CreatingJLD2Subgroups(L, N, q′s1, ηs)

# ηs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7];
# L = 16;
# N = 790; 
# CreatingJLD2Subgroups(L, N, q′s2, ηs)

function AddingDifferentQsToTheGroups(L:: Int64, N:: Int64, q′s:: Vector{Float64}, η′s::Vector{Float64})

    jldopen("./Plotting/ChaosIndicators/Data/ChaosIndicators_L$(L)_N$(N)_new.jld2", "a+") do fileNew
        for (i,q) in enumerate(q′s)
            for (n, η) in enumerate(η′s)
                println("CI_L$(L)_N$(N)_q$(q)_eta$(η).jld2")
                jldopen("./Plotting/ChaosIndicators/Data/CI_L$(L)_N$(N)_q$(q)_eta$(η).jld2", "r") do folder

                    groupPath:: String = GetGroupPath(q, η);  

                    Es_and_CIs:: Vector{ReturnObject} = folder["Es_and_CIs"];
                    Es:: Vector{Vector{Float64}} = map(x -> x.Es, Es_and_CIs);
                    CIs:: Vector{Tuple{Float64, Vector{Float64}, Vector{Float64}}} = map(x -> x.cis, Es_and_CIs);
                    rs = map(x -> x[1], CIs);
                    Ss = map(x -> x[2], CIs);
                    EEs = map(x -> x[3], CIs);
                    fileNew["$groupPath/Es"] = Es;
                    fileNew["$groupPath/rs"] = rs;
                    fileNew["$groupPath/Ss"] = Ss;
                    fileNew["$groupPath/EEs"] = EEs;


                    Ks = folder["K"]; 
                    τs = folder["τs"];  
                    Ks_smooth, τs_smooth = K̄(Ks, τs);
                    τ_Th, kth = TthExtraction_crossSection(Ks_smooth, τs_smooth); 
                    t_H = SFF.t̂_H(Es);
                    t_Th = SFF.t̂_Th(τ_Th, t_H);
                    g = SFF.ĝ(τ_Th);
                    fileNew["$groupPath/K/Ks"] = Ks;
                    fileNew["$groupPath/K/τs"] = τs;
                    fileNew["$groupPath/K/K̄s"] = Ks_smooth;
                    fileNew["$groupPath/K/τ̄s"] = τs_smooth;
                    fileNew["$groupPath/K/τ_th"] = τ_Th;
                    fileNew["$groupPath/K/t_Th"] = t_Th;
                    fileNew["$groupPath/K/t_H"] = t_H;
                    fileNew["$groupPath/K/g"] = g;        


                    Ksc = folder["Kc"];
                    Ks_smoothc, τs_smoothc = K̄(Ksc, τs);
                    τ_Thc, kth = TthExtraction_crossSection(Ks_smoothc, τs_smoothc); 
                    t_Hc = SFF.t̂_H(Es);
                    t_Thc = SFF.t̂_Th(τ_Thc, t_Hc);
                    gc = SFF.ĝ(τ_Thc);
                    fileNew["$groupPath/Kc/Ks"] = Ksc;
                    fileNew["$groupPath/Kc/τs"] = τs;
                    fileNew["$groupPath/Kc/K̄s"] = Ks_smoothc;
                    fileNew["$groupPath/Kc/τ̄s"] = τs_smoothc;
                    fileNew["$groupPath/Kc/τ_th"] = τ_Thc;
                    fileNew["$groupPath/Kc/t_Th"] = t_Thc;
                    fileNew["$groupPath/Kc/t_H"] = t_Hc;
                    fileNew["$groupPath/Kc/g"] = gc;        
                    
                end
            end
        end
    end
end

# ηs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7];
# L = 14;
# N = 900; 
# AddingDifferentQsToTheGroups(L, N, q′s2, ηs)

# ηs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7];
# L = 18;
# N = 3; 
# AddingDifferentQsToTheGroups(L, N, q′s1, ηs)



function GetKsAndτs(folder, group:: String, connected:: Bool, smoothed:: Bool)
    group_K:: String = connected ? group*"/Kc" : group*"/K";
            
    Ks = smoothed ? folder["$group_K/K̄s"] : folder["$group_K/Ks"];
    τs = smoothed ? folder["$group_K/τ̄s"] : folder["$group_K/τs"];

    return Ks, τs;
end

function K_of_τ__L(L′s:: Vector{Int64}, N′s:: Vector{Int64}, q:: Float64, η::Float64, ax, smoothed:: Bool, connected:: Bool)


    dir:: String = GetDirectory();
    
    τs = Vector{Float64}();
    for (i,L) in enumerate(L′s)
        N = N′s[i];
        fileName:: String = GetFileName(L, N);

        jldopen("$dir/$fileName", "r") do file
            group:: String = GetGroupPath(q, η);

            Ks, τs = GetKsAndτs(file, group, connected, smoothed)
            τ_Th = connected ? file["$group/Kc/τ_th"] : file["$group/K/τ_th"];

            ax.plot(τs, Ks, label=L"$L=%$(L)$", zorder=2, color=colors[i]);
            ax.scatter([τ_Th], [10^0.08 * ChaosIndicators.Kgoe.(τ_Th)], facecolors="none", s=80, edgecolors=colors[i], zorder=4);
        end

    end

    ax.plot(τs, ChaosIndicators.Kgoe.(τs), label=L"$K_{GOE}$", linestyle = "dashed", color = "black");
    ax.axvline(x=1., ymin=minimum(ChaosIndicators.Kgoe.(τs)), color="gray", linestyle="dashed");
    
    # ax.scatter(τ_Th′s, ChaosIndicators.Kgoe.(τ_Th′s), facecolors="none", s=250, edgecolors="black");
    # ax.scatter(τ_Th′s, ChaosIndicators.Kgoe.(τ_Th′s), color="black", zorder=3, facecolors="none", s=75, edgecolors="black");
    
    ax.legend();
    ax.set_xscale("log");
    ax.set_yscale("log");
    ax.set_xlabel(L"$\tau$");
    # ax.set_ylabel(L"$K(\tau)$");

    
end



function K_of_τ__q(L:: Int64, N:: Int64, q′s:: Vector{Float64}, η::Float64, ax, smoothed:: Bool, connected:: Bool)


    dir:: String = GetDirectory();
    fileName:: String = GetFileName(L, N);

    τs = Vector{Float64}();
    jldopen("$dir/$fileName", "r") do file
        for (i,q) in enumerate(q′s)
            group:: String = GetGroupPath(q, η);

            Ks, τs = GetKsAndτs(file, group, connected, smoothed)
            τ_Th = connected ? file["$group/Kc/τ_th"] : file["$group/K/τ_th"];

            ax.plot(τs, Ks, label=L"$q=%$(q)$", zorder=2, color=colors[i]);
            ax.scatter([τ_Th], [10^0.08 * ChaosIndicators.Kgoe.(τ_Th)], facecolors="none", s=80, edgecolors=colors[i], zorder=4);
        end

    end

    ax.plot(τs, ChaosIndicators.Kgoe.(τs), label=L"$K_{GOE}$", linestyle = "dashed", color = "black");
    ax.axvline(x=1., ymin=minimum(ChaosIndicators.Kgoe.(τs)), color="gray", linestyle="dashed");
    
    # ax.scatter(τ_Th′s, ChaosIndicators.Kgoe.(τ_Th′s), facecolors="none", s=250, edgecolors="black");
    # ax.scatter(τ_Th′s, ChaosIndicators.Kgoe.(τ_Th′s), color="black", zorder=3, facecolors="none", s=75, edgecolors="black");
    
    ax.legend();
    ax.set_xscale("log");
    ax.set_yscale("log");
    ax.set_xlabel(L"$\tau$");
    if connected
        ax.set_ylabel(L"$K_c(\tau)$");
    else
        ax.set_ylabel(L"$K(\tau)$");
    end

    
end

function ΔK_of_τ__q(L:: Int64, N:: Int64, q′s:: Vector{Float64}, η:: Float64, ax, smoothed:: Bool, connected:: Bool)
    dir:: String = GetDirectory();
    fileName:: String = GetFileName(L, N);

    jldopen("$dir/$fileName", "r") do file
        for (i,q) in enumerate(q′s)
            group:: String = GetGroupPath(q, η);

            Ks, τs = GetKsAndτs(file, group, connected, smoothed)
            τ_Th = connected ? file["$group/Kc/τ_th"] : file["$group/K/τ_th"];

            ΔK = log10.( Ks ./ ChaosIndicators.Kgoe.(τs));

            ax.plot(τs, ΔK, label=L"$q=%$(q)$", zorder=3);
            # ax.scatter([τ_Th], [ChaosIndicators.Kgoe(τ_Th)], facecolors="none", s=80, edgecolors="black", zorder=4);
        end
    end
    ax.axhline(y=0.08, color="black", linestyle = "dashed", zorder=2)

    ax.set_xscale("log");
    ax.set_yscale("log");
    ax.set_title(L"$q=%$(q)$", y=0.85, zorder=5);
end




function K_of_τ__η(L:: Int64, N:: Int64, q:: Float64, η′s::Vector{Float64}, ax, smoothed:: Bool, connected:: Bool)
    αz = 0.9; αk = 0.1
    k = (αz - αk)/(1- length(q′s));
    n = αz-k
    alphaa(x:: Int64) = k*x+n

    dir:: String = GetDirectory();
    fileName:: String = GetFileName(L, N);

    τs = Vector{Float64}();
    jldopen("$dir/$fileName", "r") do file
        for (i,η) in enumerate(η′s)
            group:: String = GetGroupPath(q, η);
            group_K:: String = connected ? group*"/Kc" : group*"Kc";
            
            Ks, τs = GetKsAndτs(file, group, connected, smoothed)
            τ_Th = connected ? file["$group/Kc/τ_th"] : file["$group/K/τ_th"];

            τ_th, kth = TthExtraction_crossSection(Ks, τs); 
            ax.plot(τs, Ks, label=L"$\eta=%$(η)$", zorder=2, alpha= alphaa(i));
            ax.scatter([τ_th], [kth], facecolors="none", s=80, edgecolors="black", zorder=4);
        end
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

function ΔK_of_τ__η(L:: Int64, N:: Int64, q:: Float64, η′s::Vector{Float64}, ax, smoothed:: Bool, connected:: Bool)
    αz = 0.9; αk = 0.1
    k = (αz - αk)/(1- length(q′s));
    n = αz-k
    alphaa(x:: Int64) = k*x+n

    dir:: String = GetDirectory();
    fileName:: String = GetFileName(L, N);

    jldopen("$dir/$fileName", "r") do file
        for (i,η) in enumerate(η′s)
            group:: String = GetGroupPath(q, η);

            Ks, τs = GetKsAndτs(file, group, connected, smoothed)
            τ_Th = connected ? file["$group/Kc/τ_th"] : file["$group/K/τ_th"];


            ΔK = log10.( Ks ./ ChaosIndicators.Kgoe.(τs));

            ax.plot(τs, ΔK, label=L"$\eta=%$(η)$", zorder=3, alpha= alphaa(i));
            # ax.scatter([τ_Th], [ChaosIndicators.Kgoe(τ_Th)], facecolors="none", s=80, edgecolors="black", zorder=4);
        end
    end

    ax.axhline(y=0.08, color="black", linestyle = "dashed", zorder=2)
    ax.legend();
    ax.set_xscale("log");
    ax.set_yscale("log");
    # ax.set_title(L"$L=%$(L)$", y=0.85, zorder=5);
end



function tTh_of_q__L(L′s:: Vector{Int64}, N′s:: Vector{Int64}, q′s:: Vector{Float64}, ax, connected:: Bool, η:: Float64)

    # axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47]);
    # # axins.imshow(Z2, extent=extent, origin="lower")
    # # subregion of the original image
    # x1, x2, y1, y2 = 0.35, 1.1,  0.8, 10;
    # axins.set_xlim(x1, x2);
    # axins.set_ylim(y1, y2);
    # axins.set_xscale("log");
    # axins.set_yscale("log");
    # axins.set_xticklabels([]);
    # axins.set_yticklabels([]);
    # ax.indicate_inset_zoom(axins, edgecolor="black");

    

    for (i,L) in enumerate(L′s)
        N = N′s[i];
        t_Th′s = Vector{Float64}(undef, length(q′s));

        dir:: String = GetDirectory();
        fileName:: String = GetFileName(L, N);
        jldopen("$dir/$fileName", "r") do file
            for (j,q) in enumerate(q′s)
                group:: String = GetGroupPath(q, η);
            
                t_Th′s[j] = connected ? file["$group/Kc/t_Th"] : file["$group/K/t_Th"];
            end
            ax.plot(q′s, t_Th′s, label=L"$L=%$(L)$", zorder=3, markers_line[i]);
            # axins.plot(q′s, t_Th′s, zorder=3, markers_line[i]);
        end

        if L==L′s[end]
            # fit = linear_fit(log10.(q′s[end-8:end-2]), log10.(t_Th′s[end-8:end-2]));
            # y10 = fit[1] .+ fit[2] .* log10.(q′s);
            # y= 10 .^y10
            # println(y)
            # println(fit)            
            # ax.plot(q′s, y, label=L"y = %$(round(fit[2], digits=3)) q + %$(round(fit[1], digits=3))", linestyle="dashed");


            fit = power_fit(q′s[end-8:end-2], t_Th′s[end-8:end-2]);
            y = fit[1] .* q′s .^ fit[2];
            ax.plot(q′s, y, label=L"y =  %$(round(fit[1], digits=3))  q^{%$(round(fit[2], digits=3))} ", linestyle="dashed", color="black");
        end

    end
    # ax.plot(q′s, 1 ./ q′s.^2.4, label=L"$1/q^{2.4}$", linestyle="dashed");
    # ax.plot(q′s, 1 ./ q′s.^2.5, label=L"$1/q^{2.5}$", linestyle="dashed");
    # ax.plot(q′s, 1 ./ q′s.^2.6, label=L"$1/q^{2.6}$", linestyle="dashed");
    # ax.plot(q′s, 1 ./ q′s.^2.7, label=L"$1/q^{2.7}$", linestyle="dashed");
    # ax.plot(q′s, 1 ./ q′s.^2.8, label=L"$1/q^{2.8}$", linestyle="dashed");
    # ax.plot(q′s, 1 ./ q′s.^2.9, label=L"$1/q^{2.9}$", linestyle="dashed");
    # ax.plot(q′s, 1 ./ q′s.^3, label=L"$1/q^{3}$", linestyle="dashed");





    # axins.plot(q′s, 1 ./ q′s.^2, color="black", linestyle="dashed");
    # axins.plot(q′s, 1 ./ q′s.^3, color="black", linestyle="dashed");

    ax.legend(loc="lower left");
    ax.set_xlabel(L"$q$");
    ax.set_ylabel(L"$t_{Th}$");
    
    ax.set_yscale("log");
    ax.set_xscale("log");
end



function τTh_of_q__L(L′s:: Vector{Int64}, N′s:: Vector{Int64}, q′s:: Vector{Float64}, ax, connected:: Bool, η:: Float64)

    for (i,L) in enumerate(L′s)
        N = N′s[i];
        τ_th′s = Vector{Float64}(undef, length(q′s));

        dir:: String = GetDirectory();
        fileName:: String = GetFileName(L, N);
        jldopen("$dir/$fileName", "r") do file
            for (j,q) in enumerate(q′s)
                group:: String = GetGroupPath(q, η);
                τ_Th′s[j] = connected ? file["$group/Kc/τ_th"] : file["$group/K/τ_th"];
            end
            ax.scatter(q′s, τ_th′s, label=L"$L=%$(L)$");
        end
    end    

    ax.legend();
    ax.set_xlabel(L"$q$");
    ax.set_ylabel(L"$\tau_Th$");
    
    ax.set_yscale("log");
    ax.set_xscale("log");

end

function g_of_q__L(L′s:: Vector{Int64}, N′s:: Vector{Int64}, q′s:: Vector{Float64}, ax, connected:: Bool, η:: Float64)

    for (i,L) in enumerate(L′s)
        N = N′s[i];
        g′s = Vector{Float64}(undef, length(q′s));

        dir:: String = GetDirectory();
        fileName:: String = GetFileName(L, N);
        jldopen("$dir/$fileName", "r") do file
            for (j,q) in enumerate(q′s)
                group:: String = GetGroupPath(q, η);
                g′s[j] = connected ? file["$group/Kc/g"] : file["$group/K/g"];
            end
            ax.plot(q′s, g′s, label=L"$L=%$(L)$");
        end

    end
    
    ax.legend();
    ax.set_xlabel(L"$q$");
    ax.set_ylabel(L"$g$");
    ax.set_yscale("log");
    ax.set_xscale("log");
    # ax.set_title(L"$q = %$(q) $");

end

function g_of_qOverDE__L(L′s:: Vector{Int64}, N′s:: Vector{Int64}, q′s:: Vector{Float64}, ax, connected:: Bool, η:: Float64)
    for (i,L) in enumerate(L′s)
        N = N′s[i];
        g′s = Vector{Float64}(undef, length(q′s));
        E′s = Vector{Float64}(undef, length(q′s));

        dir:: String = GetDirectory();
        fileName:: String = GetFileName(L, N);
        jldopen("$dir/$fileName", "r") do file
            for (j,q) in enumerate(q′s)
                group:: String = GetGroupPath(q, η);
                g′s[j] = connected ? file["$group/Kc/g"] : file["$group/K/g"];
                E′s[j] = connected ? 1/file["$group/Kc/t_H"] : 1/file["$group/K/t_H"];
            end
            ax.plot(q′s ./ E′s .^ 0.5, g′s, label=L"$L=%$(L)$");
        end
        
    end

    ax.legend();
    ax.set_xlabel(L"$q / \sqrt{\delta E}$");
    ax.set_ylabel(L"$g$");
    ax.set_yscale("log");
    ax.set_xscale("log");
    # ax.set_title(L"$q = %$(q) $");

end

function tH_of_q__L(L′s:: Vector{Int64}, N′s:: Vector{Int64}, q′s:: Vector{Float64}, ax, connected:: Bool, η:: Float64)
    for (i,L) in enumerate(L′s)
        N = N′s[i];
        tH′s = Vector{Float64}(undef, length(q′s));

        dir:: String = GetDirectory();
        fileName:: String = GetFileName(L, N);
        jldopen("$dir/$fileName", "r") do file
            for (j,q) in enumerate(q′s)
                group:: String = GetGroupPath(q, η);
                tH′s[j] = connected ? file["$group/Kc/t_H"] : file["$group/K/t_H"];
            end
            ax.plot(q′s, tH′s, label=L"$L=%$(L)$", zorder=3);
        end
        
    end

    ax.legend();
    ax.set_xlabel(L"$q$");
    ax.set_ylabel(L"$t_{H}$");

    ax.set_yscale("log");
    ax.set_xscale("log");
    # ax.set_title(L"$q = %$(q) $");

end



function δE(L′s:: Vector{Int64}, N′s:: Vector{Int64}, q′s:: Vector{Float64}, ax, η::Float64, connected:: Bool)

    axin = ax.inset_axes([0.1, 0.5, 0.35, 0.35])

    for (i,L) in enumerate(L′s)
        D:: Int64 = binomial(L,L÷2);
        δĒ′s_analitic = Vector{Float64}();
        δĒ′s_numeric  = Vector{Float64}();

        N = N′s[i]
        dir:: String = GetDirectory();
        fileName:: String = GetFileName(L, N);
        jldopen("$dir/$fileName", "r") do file
            for (j,q) in enumerate(q′s) 
                group:: String = GetGroupPath(q, η);
                
                Es′s = file["$group/Es"];
                t_H = connected ? file["$group/Kc/t_H"] : file["$group/K/t_H"];

                δĒ_numeric = 0.;
                for Es in Es′s
                    δE = Es[Int(D÷4+1):Int(3*D÷4+1)] .- Es[Int(D÷4):Int(3*D÷4)];
                    δĒ =  mean(δE);
                    δĒ_numeric += δĒ/length(Es′s);
                end

                push!(δĒ′s_analitic, 1/t_H);
                push!(δĒ′s_numeric, δĒ_numeric);    
            end
        end
        ax.plot(q′s, δĒ′s_analitic, label=L"Analitično $L=%$(L)$", color=colors[i]);
        ax.scatter(q′s, δĒ′s_numeric, label=L"Numerično $L=%$(L)$", color=colors[i] , marker = markers[i]);

        axin.plot(q′s, abs.(δĒ′s_analitic .- δĒ′s_numeric), label=L"$L=%$(L)$", color=colors[i]);
        axin.yaxis.set_label_coords(.1, -.1)
    end

    ax.legend(loc="center right");
    ax.set_xscale("log");
    ax.set_yscale("log");
    ax.set_xlabel(L"$q$");
    ax.set_ylabel(L"$\delta \overline{E}$");


    # axin.legend()
    axin.set_xscale("log");
    axin.set_yscale("log");
    axin.set_xlabel(L"$q$");
    axin.set_ylabel(L"Napaka $\delta \overline{E}$");
end



function Plot1()
    L = 16;
    N = 790;
    ηs = [0.2, 0.3, 0.4, 0.5, 0.6];
    connected=false;
    is = [10,20,22,25];
    q′s_ = q′s1[is]


    fig, ax = plt.subplots(nrows=length(q′s_), ncols=4, sharex =false, sharey=false);

    fig.suptitle(L"Thouless time extraction $L = %$(L)$, $N = %$(N)$", fontsize=19);
    ax[1,1].set_title(L"$K(\tau)$");
    ax[1,2].set_title(L"$\Delta K(\tau)$");
    ax[1,3].set_title(L"$ \overline{K}(\tau)$");
    ax[1,4].set_title(L"$\Delta \overline{K}(\tau)$");

    for (i,q) in enumerate(q′s_)
        ax1 = ax[i,1]; ax1.set_ylabel(L"$q = %$(q)$");
        ax2 = ax[i,2];
        ax3 = ax[i,3];
        ax4 = ax[i,4];

        K_of_τ__η(L, N, q, ηs, ax1,  false, connected);
        ΔK_of_τ__η(L, N, q, ηs, ax2, false, connected);
        K_of_τ__η(L, N, q, ηs, ax3,  true, connected);
        ΔK_of_τ__η(L, N, q, ηs, ax4, true, connected);

        if q == q′s[end]
            ax1.set_xlabel(L"\tau");
            ax2.set_xlabel(L"\tau");
            ax3.set_xlabel(L"\tau");
            ax4.set_xlabel(L"\tau");
        end
    end

    plt.show();

end
# Plot1();


function Plot2()
    L1   = 16;
    L2′s = [16, 14, 12, 10, 8];
    L3′s = [16, 14, 12, 10, 8];
    N1   = 790;
    N2′s = [790, 900, 1500, 5000, 20_000];
    N3′s = [790, 900, 1500, 5000, 20_000];
    q1′s = q′s1[[1,7,20,22,25]];
    q2′s = q′s1;
    q3′s = q′s1;
    isConnected = false;
    η = 0.4;

    # plt.style.use(["don_custom"])
    fig, ax = plt.subplots(ncols=3)

    K_of_τ__q(L1, N1, q1′s, η, ax[1], true, isConnected);
    g_of_q__L(L2′s, N2′s, q2′s, ax[2], isConnected, η);
    g_of_qOverDE__L(L3′s, N3′s, q3′s, ax[3], isConnected, η);

    plt.show()
end
# Plot2()



function Plot3()
    L1′s = [8,10,12,14,16];
    L2′s = [8,10,12,14,16];
    N1′s = [20_000,5_000,1500,900,790];
    N2′s = [20_000,5_000,1500,900,790];
    q1′s = q′s1;
    q2′s = q′s1;
    isConnected = false;
    η = 0.4;

    # plt.style.use(["don_custom"])
    fig, ax = plt.subplots(ncols=2)

    tTh_of_q__L(L1′s, N1′s, q1′s, ax[1], isConnected, η);
    tH_of_q__L(L2′s, N2′s, q2′s, ax[2], isConnected, η);

    plt.show()
end
# Plot3();


function Plot4()
    L′s = [8,10,12,14,16];
    N′s = [20_000,5_000,1500,900,790];
    q′s = q′s1
    isConnected = false;
    η = 0.4;    
    fig, ax = plt.subplots(ncols=1)
    δE(L′s, N′s, q′s, ax, η, isConnected)

    plt.show()
end
# Plot4();


function Plot5()
    L′s = [8,10,12,14,16];
    N′s = [20_000,5_000,1500,900,790];
    isConnected = false;
    η = 0.4;
    q′s_ = q′s1[[15,20,22,25]];

    fig, ax = plt.subplots(ncols=length(q′s_));

    ax[1].set_ylabel(L"$K(\tau)$");
    for (i,q) in enumerate(q′s_)
        ax[i].set_title(L"q = %$(q)");
        K_of_τ__L(L′s, N′s, q, η, ax[i], true, isConnected);
    end

    plt.show()
end
# Plot5();


function Plot6()
    L′s = [8,10,12,14,16];
    N′s = [20_000,5_000,1500,900,790];
    isConnected = false;
    η = 0.4;
    q′s_ = q′s1[[15,20,22,25]];

    fig, ax = plt.subplots(ncols=length(q′s_));

    ax[1].set_ylabel(L"$K(\tau)$");
    for (i,q) in enumerate(q′s_)
        ax[i].set_title(L"q = %$(q)");
        K_of_τ__q(L′s[end], N′s[end], q′s_, η, ax[i], true, isConnected);
    end

    plt.show()
end
# Plot6();


function Plot7()
    L′s = [16, 14];
    N′s = [790, 900];
    # q′s_ = vcat([q′s1[1]],q′s1[[15 + i for i in 1:6]]);
    q′s_ =q′s1[[i for i in 1:20]];
    η = 0.5;

    # plt.style.use(["don_custom"])
    fig, ax = plt.subplots(ncols=2, nrows=length(L′s));

    for (i,L) in enumerate(L′s)
        N = N′s[i]
        K_of_τ__q(L, N, q′s_, η, ax[i,1], true, false);
        K_of_τ__q(L, N, q′s_, η, ax[i,2], true, true);

        ax[i,1].set_title(L"L=%$(L)", y=0.8);
        ax[i,2].set_title(L"L=%$(L)", y=0.8);
        x = LinRange(-4,1,1000)
        tau= 10 .^x;
        ax[i,1].plot(tau, 10^6.5 .* tau .^2.45, c="black")
        ax[i,1].plot(tau, 10^6.5 .* tau .^2.45, c="black")
    end
    ax[1,1].get_legend().remove()

    # ax[1,1].plot(tau, 10^6.5 .* tau .^2.45, c="b")
    # ax[1,2].plot(tau, 10^5 .* tau .^2)
    plt.show()
end
Plot7()


function Plot_K_fake()
    function GetFileName(L:: Int64, N:: Int64, q::Float64)
        fileName = "CI_L$(L)_Iter$(N)_q$(q).jld2";
        return fileName;
    end

    L = 10;
    N = 100;
    # q′s_ = vcat([q′s1[1]],q′s1[[15 + i for i in 1:6]]);
    q = 0.;

    # plt.style.use(["don_custom"])
    fig, ax = plt.subplots(ncols=1, nrows=1);

    dir:: String = GetDirectory();
    fileName:: String = GetFileName(L, N, q);



    τs = Vector{Float64}();
    jldopen("$dir/$fileName", "r") do file
        Ks = file["K"];
        τs = file["τs"];

        ax.plot(τs, Ks, label=L"$q=%$(q)$", zorder=2, color=colors[1]);
        # ax.scatter([τ_Th], [10^0.08 * ChaosIndicators.Kgoe.(τ_Th)], facecolors="none", s=80, edgecolors=colors[1], zorder=4);

    end


    ax.set_xscale("log");
    ax.set_yscale("log");
    ax.set_ylim(10^(-6), 10^(6))

    # ax[1,1].get_legend().remove()

    # x = LinRange(-4,1,1000)
    # tau= 10 .^x;
    # ax[1,1].plot(tau, 10^6.5 .* tau .^2.45, c="b")
    # ax[1,2].plot(tau, 10^5 .* tau .^2)
    plt.show()
end
# Plot_K_fake()





# E = [1, 2, 3, 4]';
# t = [0, 1];

# display(E)
# println()
# display(t)
# println()

# c = t*E
# display(c)

# d = @time sum(c, dims=2)
# display(d)