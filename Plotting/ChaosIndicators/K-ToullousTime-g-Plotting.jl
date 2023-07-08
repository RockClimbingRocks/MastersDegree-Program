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

function GetFileName(L:: Int64, N:: Int64, q:: Float64, η:: Float64)
    # unConnectedFileName = "SpectralFormFactor_L$(L)_Iter$(N)_q$(q)_η$(η)";
    # connectedFileName = "SpectralFormFactor_ConnectedAndUnConnected_L$(L)_Iter$(N)_q$(q)_η$(η)";
    # fileName = connected ? connectedFileName : unConnectedFileName;   

    return "Hsyk_ChaosIndicators_L$(L)_Iter$(N)_q$(q)_eta$(η)";
end


function SpectralFormFactorOfTau_q(L:: Int64, N:: Int64, q′s:: Vector{Float64}, ax, connected:: Bool, η::Float64=0.5)
    τ′s:: Vector{Float64} = Vector{Float64}();
    τ_Th′s:: Vector{Float64} = Vector{Float64}();


    for (i,q) in enumerate(q′s) 
        fileName = GetFileName(L,N,q,η);
        folder = jldopen("./Plotting/ChaosIndicators/Data/$(fileName).jld2", "r");

        # coeffs′s = folder["coeffs′s"];
        # Es′s = folder["Es′s"];
        τ′s = folder["τ′s"];
        # t_H = folder["t_H"];

        K′s = connected ? folder["K′s_c"] : folder["K′s"];
        τ_Th = connected ? folder["τ_Th_c"] : folder["τ_Th"];
        # t_Th = connected ? folder["t_Th_c"] : folder["t_Th"];
        # g = connected ? folder["g_c"] : folder["g"];

        close(folder);

        ax.plot(τ′s, K′s, label=L"$q=%$(q)$", zorder=2);

        push!(τ_Th′s, τ_Th)
    end
    ax.plot(τ′s, ChaosIndicators.Kgoe.(τ′s), label=L"$K_{GOE}$", linestyle = "dashed", color = "black");
    ax.axvline(x=1., ymin=minimum(ChaosIndicators.Kgoe.(τ′s)), color="gray", linestyle="dashed");

    # ax.scatter(τ_Th′s, ChaosIndicators.Kgoe.(τ_Th′s), facecolors="none", s=250, edgecolors="black");
    ax.scatter(τ_Th′s, ChaosIndicators.Kgoe.(τ_Th′s), color="black", zorder=3, facecolors="none", s=75, edgecolors="black");

    ax.legend();
    ax.set_xscale("log");
    ax.set_yscale("log");
    ax.set_xlabel(L"$\tau$");
    ax.set_ylabel(L"$K(\tau)$");
end

function SpectralFormFactorOfTau_η(L:: Int64, N:: Int64, q:: Float64, ax, connected:: Bool, η′s:: Vector{Float64})

    τ′s:: Vector{Float64} = Vector{Float64}();
    τ_Th′s:: Vector{Float64} = Vector{Float64}();


    
    for (i,η) in enumerate(η′s) 
        fileName = GetFileName(L,N,q, η);
        folder = jldopen("./Plotting/ChaosIndicators/Data/$(fileName).jld2", "r");
        # coeffs′s = folder["coeffs′s"];
        # Es′s = folder["Es′s"];
        K′s = folder["K′s"];
        τ′s = folder["τ′s"];
        τ_Th = folder["τ_Th"];
        # t_Th = folder["t_Th"];
        # t_H = folder["t_H"];
        # g = folder["g"];
        close(folder);

        ax.plot(τ′s, K′s, label=L"$\eta=%$(η)$", zorder=2);

        push!(τ_Th′s, τ_Th)
    end
    ax.plot(τ′s, ChaosIndicators.Kgoe.(τ′s), label=L"$K_{GOE}$", linestyle = "dashed", color = "black");
    ax.axvline(x=1., ymin=minimum(ChaosIndicators.Kgoe.(τ′s)), color="gray", linestyle="dashed");

    # ax.scatter(τ_Th′s, ChaosIndicators.Kgoe.(τ_Th′s), facecolors="none", s=250, edgecolors="black");
    ax.scatter(τ_Th′s, ChaosIndicators.Kgoe.(τ_Th′s), color="black", zorder=3, facecolors="none", s=75, edgecolors="black");

    ax.legend();
    ax.set_xscale("log");
    ax.set_yscale("log");
    ax.set_xlabel(L"$\tau$");
    ax.set_ylabel(L"$K(\tau)$");
end

function g_AsAFunctionOf_q_plot(L′s:: Vector{Int64}, N′s:: Vector{Int64}, q′s:: Vector{Float64}, ax, connected:: Bool, η::Float64=0.5)

    for (i,L) in enumerate(L′s)
        g′s = Vector{Float64}();
        q′s_with_τ_Th = Vector{Float64}();

        N = N′s[i]
        for (j,q) in enumerate(q′s) 
            fileName = GetFileName(L,N,q, η);
            folder = jldopen("./Plotting/ChaosIndicators/Data/$(fileName).jld2", "r");
            # coeffs′s = folder["coeffs′s"];
            # Es′s = folder["Es′s"];
            # K′s = folder["K′s"];
            # τ′s = folder["τ′s"];
            # τ_Th = folder["τ_Th"];
            # t_Th = folder["t_Th"];
            # t_H = folder["t_H"];
            g = folder["g"];
            close(folder);

            push!(g′s, g);
            push!(q′s_with_τ_Th, q);    
        end

        ax.plot(q′s_with_τ_Th, g′s, label=L"$L=%$(L)$")
    end

    ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel(L"$q$")
    ax.set_ylabel(L"$g$")
end

function g_AsAFunctionOf_q_overSqrtE_plot(L′s:: Vector{Int64}, N′s:: Vector{Int64}, q′s:: Vector{Float64}, ax, connected:: Bool, η::Float64=0.5)

    for (i,L) in enumerate(L′s)
        g′s = Vector{Float64}();
        q′s_overSqrtE = Vector{Float64}();

        N = N′s[i]

        for (j,q) in enumerate(q′s) 
            fileName = GetFileName(L,N,q, η);
            folder = jldopen("./Plotting/ChaosIndicators/Data/$(fileName).jld2", "r");
            # coeffs′s = folder["coeffs′s"];
            # Es′s = folder["Es′s"];
            # K′s = folder["K′s"];
            # τ′s = folder["τ′s"];
            # τ_Th = folder["τ_Th"];
            # t_Th = folder["t_Th"];
            t_H = folder["t_H"];
            g = folder["g"];
            close(folder);

            push!(g′s, g);
            push!(q′s_overSqrtE, q*√t_H);    
        end

        ax.plot(q′s_overSqrtE, g′s, label=L"$L=%$(L)$")
    end

    ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel(L"$q/\sqrt{\overline{\delta E}}$")
    ax.set_ylabel(L"$g$")
end

function g_AsAFunctionOf_Lξkbt_plot(L′s:: Vector{Int64}, N′s:: Vector{Int64}, q′s:: Vector{Float64}, ax, connected:: Bool, η::Float64=0.5)

    for (i,L) in enumerate(L′s) 
        g′s = Vector{Float64}();
        ξKBT′s_with_τ_Th = Vector{Float64}();
    
        for (j,q) in enumerate(q′s) 
            fileName = GetFileName(L,N,q, η);
            # folder = jldopen("ChaosIndicators/Data/SpectralFormFunctionCoeffitions_L$(L2)_Iter$(N2[i])_q$(q2).jld2", "r");
            folder = jldopen("./Plotting/ChaosIndicators/Data/$(fileName).jld2", "r");
            # coeffs′s = folder["coeffs′s"];
            # Es′s = folder["Es′s"];
            # K′s = folder["K′s"];
            # τ′s = folder["τ′s"];
            # τ_Th = folder["τ_Th"];
            # t_Th = folder["t_Th"];
            # t_H = folder["t_H"];
            g = folder["g"];
            close(folder);
    
    
            q₀ = -0.181;
            q₁ = 0.252;
            b = 4.62;
            q_c = ChaosIndicators.qc(L, q₀, q₁);
            # ξkbt = ChaosIndicators.ξkbt(q3, q_c, b);
    
            push!(ξKBT′s_with_τ_Th, q);     
            push!(g′s, g);
                
        end
    
        # ax[3].plot( L3 ./ ξKBT3′s_with_τ_Th, g′s, label="L=$(L3)")
    end
    
    
    # ax.legend();
    ax.set_xlabel(L"$L/ \xi_{KBT}$");
    ax.set_ylabel(L"$g$");
    
end

function ExtractionOFThoullesTime_ΔK(L:: Int64, N:: Int64, q:: Float64, ax, connected:: Bool, η′s::Vector{Float64})

    τ′s = Vector{Float64}();
    for (i,η) in enumerate(η′s) 
        fileName = GetFileName(L,N,q, η);
        folder = jldopen("./Plotting/ChaosIndicators/Data/$(fileName).jld2", "r");
        # coeffs′s = folder["coeffs′s"];
        # Es′s = folder["Es′s"];
        K′s = folder["K′s"];
        τ′s = folder["τ′s"];
        τ_Th = folder["τ_Th"];
        # t_Th = folder["t_Th"];
        # t_H = folder["t_H"];
        # g = folder["g"];
        close(folder);

        ΔK = log10.( K′s ./ ChaosIndicators.Kgoe.(τ′s));

        ax.plot(τ′s, ΔK, label=L"$\eta=%$(η)$", zorder=3);
        # ax.scatter([τ_Th], [ChaosIndicators.Kgoe(τ_Th)], facecolors="none", s=80, edgecolors="black", zorder=4);
    end
    ax.axhline(y=0.08, color="black", linestyle = "dashed", zorder=2)


    ax.set_xscale("log");
    ax.set_yscale("log");
    ax.set_title(L"$L=%$(L) \quad q=%$(q)$", y=0.85, zorder=5);
end

function δE(L′s:: Vector{Int64}, N′s:: Vector{Int64}, q′s:: Vector{Float64}, ax, connected:: Bool, η::Float64=0.5)

    axin = ax.inset_axes([0.07, 0.45, 0.5, 0.5])

    for (i,L) in enumerate(L′s)
        D:: Int64 = binomial(L,L÷2);
        δĒ′s_analitic = Vector{Float64}();
        δĒ′s_numeric  = Vector{Float64}();

        N = N′s[i]
        for (j,q) in enumerate(q′s) 
            fileName = GetFileName(L,N,q, η);
            folder = jldopen("./Plotting/ChaosIndicators/Data/$(fileName).jld2", "r");
            # coeffs′s = folder["coeffs′s"];
            Es′s = folder["Es′s"];
            # K′s = folder["K′s"];
            # τ′s = folder["τ′s"];
            # τ_Th = folder["τ_Th"];
            # t_Th = folder["t_Th"];
            t_H = folder["t_H"];
            # g = folder["g"];
            close(folder);

            δĒ_numeric = 0.;
            for Es in Es′s
                δE = Es[Int(D÷4+1):Int(3*D÷4+1)] .- Es[Int(D÷4):Int(3*D÷4)];
                δĒ =  mean(δE);
                δĒ_numeric += δĒ/length(Es′s);
            end

            push!(δĒ′s_analitic, 1/t_H);
            push!(δĒ′s_numeric, δĒ_numeric);    
        end

        ax.plot(q′s, δĒ′s_analitic, label=L"Analitično $L=%$(L)$", color=colors[i]);
        ax.scatter(q′s, δĒ′s_numeric, label=L"Numerično $L=%$(L)$", color=colors[i] , marker = markers[i]);

        axin.plot(q′s, abs.(δĒ′s_analitic .- δĒ′s_numeric), label=L"$L=%$(L)$", color=colors[i]);
        axin.yaxis.set_label_coords(.1, -.1)
    end

    ax.legend(loc="center right")
    ax.set_xscale("log")
    ax.set_xlabel(L"$q$")
    ax.set_ylabel(L"$\delta \overline{E}$")


    # axin.legend()
    axin.set_xscale("log")
    axin.set_yscale("log")
    axin.set_xlabel(L"$q$")
    axin.set_ylabel(L"Napaka $\delta \overline{E}$")

end

function Plot_tTh_L(L′s:: Vector{Int64}, N′s:: Vector{Int64}, q:: Float64, ax, connected:: Bool, η′s:: Vector{Float64})

    for (i,η) in enumerate(η′s) 
        t_Th′s = Vector{Float64}();
        for (j,L) in enumerate(L′s)
            N = N′s[j]
            fileName = GetFileName(L,N,q, η);
            folder = jldopen("./Plotting/ChaosIndicators/Data/$(fileName).jld2", "r");
            # coeffs′s = folder["coeffs′s"];
            # Es′s = folder["Es′s"];
            # K′s = folder["K′s"];
            # τ′s = folder["τ′s"];
            # τ_Th = folder["τ_Th"];
            t_Th = folder["t_Th"];
            # t_H = folder["t_H"];
            # g = folder["g"];
            close(folder);

            push!(t_Th′s, t_Th);
        end
        
        ax.scatter(L′s, t_Th′s, label=L"$\eta=%$(η)$");
    end
    # ax[1].plot(τ′s, ChaosIndicators.Kgoe.(τ′s), label="K_GOE", linestyle = "dashed", color = "black") 
    # ax[1].axvline(x=1., ymin=minimum(ChaosIndicators.Kgoe.(τ′s)), color="gray", linestyle="dotted", linewidth=5)

    # ax[1].legend()
    # ax[1].set_xscale("log")
    # ax[1].set_yscale("log")
    # ax[1].set_xlabel("tau")
    # ax[1].set_ylabel("K(tau)")



    ax.legend();
    ax.set_xlabel(L"$L$");
    ax.set_ylabel(L"$t_{Th}$");
    ax.set_title(L"$q = %$(q) $");

end

function Plot_tTh_q(L′s:: Vector{Int64}, N′s:: Vector{Int64}, q′s:: Vector{Float64}, ax, connected:: Bool, η:: Float64=0.5)
    for (i,L) in enumerate(L′s)
        t_Th′s = Vector{Float64}();
        N= N′s[i];
        for (j,q) in enumerate(q′s)
            fileName = GetFileName(L,N,q, η);
            folder = jldopen("./Plotting/ChaosIndicators/Data/$(fileName).jld2", "r");
            # coeffs′s = folder["coeffs′s"];
            # Es′s = folder["Es′s"];
            # K′s = folder["K′s"];
            # τ′s = folder["τ′s"];
            # τ_Th = folder["τ_Th"];
            t_Th = folder["t_Th"];
            # t_H = folder["t_H"];
            # g = folder["g"];
            close(folder);

            push!(t_Th′s, t_Th);
        end
        
        ax.plot(q′s, t_Th′s, label=L"$L=%$(L)$", markers_line[i]);
    end

    ax.plot(q′s, 1 ./ q′s.^2, label=L"$1/q^2$", color="black", linestyle="dashed");


    ax.legend();
    ax.set_xlabel(L"$q$");
    ax.set_ylabel(L"$t_{Th}$");
    ax.set_xscale("log");
    ax.set_yscale("log");

end

function Plot_tH_q(L′s:: Vector{Int64}, N′s:: Vector{Int64}, q′s:: Vector{Float64}, ax, connected:: Bool, η:: Float64=0.5)

    for (i,L) in enumerate(L′s)
        t_H′s = Vector{Float64}();
        N = N′s[i]
        for (j,q) in enumerate(q′s)
            fileName = GetFileName(L,N,q, η);
            folder = jldopen("./Plotting/ChaosIndicators/Data/$(fileName).jld2", "r");
            # coeffs′s = folder["coeffs′s"];
            # Es′s = folder["Es′s"];
            # K′s = folder["K′s"];
            # τ′s = folder["τ′s"];
            # τ_Th = folder["τ_Th"];
            # t_Th = folder["t_Th"];
            t_H = folder["t_H"];
            # g = folder["g"];
            close(folder);

            push!(t_H′s, t_H);
        end
        
        ax.plot(q′s, t_H′s, label=L"$L=%$(L)$", markers_line[i]);
    end

    ax.legend();
    ax.set_xlabel(L"$q$");
    ax.set_ylabel(L"$t_{H}$");
    ax.set_xscale("log");
    ax.set_yscale("log");
    # ax.set_title(L"$q = %$(q) $");

end

function Plot_tTh_η(L:: Int64, N:: Int64, q:: Float64, ax, connected:: Bool, η′s:: Vector{Float64})

    t_Th′s = Vector{Float64}();
    for (i,η) in enumerate(η′s) 
        fileName = GetFileName(L,N,q, η);
        folder = jldopen("./Plotting/ChaosIndicators/Data/$(fileName).jld2", "r");
        # coeffs′s = folder["coeffs′s"];
        # Es′s = folder["Es′s"];
        # K′s = folder["K′s"];
        # τ′s = folder["τ′s"];
        t_Th = folder["t_Th"];
        # t_H = folder["t_H"];
        # g = folder["g"];
        close(folder);

        push!(t_Th′s, t_Th);
    end
    ax.plot(η′s, t_Th′s, markers_line[1]);

    # ax.legend();
    ax.set_xlabel(L"$\eta$");
    ax.set_ylabel(L"$t_{Th}$");
end

function Plot_tH_η(L:: Int64, N:: Int64, q:: Float64, ax, connected:: Bool, η′s:: Vector{Float64})

    t_H′s = Vector{Float64}();
    for (i,η) in enumerate(η′s) 
        fileName = GetFileName(L,N,q, η);
        folder = jldopen("./Plotting/ChaosIndicators/Data/$(fileName).jld2", "r");
        # coeffs′s = folder["coeffs′s"];
        # Es′s = folder["Es′s"];
        # K′s = folder["K′s"];
        # τ′s = folder["τ′s"];
        # τ_Th = folder["τ_Th"];
        # t_Th = folder["t_Th"];
        t_H = folder["t_H"];
        # g = folder["g"];
        close(folder);

        push!(t_H′s, t_H);
    end
    ax.plot(η′s, t_H′s, markers_line[1]);


    # ax.legend();
    ax.set_xlabel(L"$\eta$");
    ax.set_ylabel(L"$t_{H}$");
end

function Plot_τTh_η(L:: Int64, N:: Int64, q:: Float64, ax, connected:: Bool, η′s:: Vector{Float64})

    τ_Th′s = Vector{Float64}();
    for (i,η) in enumerate(η′s) 
        fileName = GetFileName(L,N,q, η);
        folder = jldopen("./Plotting/ChaosIndicators/Data/$(fileName).jld2", "r");
        # coeffs′s = folder["coeffs′s"];
        # Es′s = folder["Es′s"];
        # K′s = folder["K′s"];
        # τ′s = folder["τ′s"];
        τ_Th = folder["τ_Th"];
        # t_Th = folder["t_Th"];
        # t_H = folder["t_H"];
        # g = folder["g"];
        close(folder);

        push!(τ_Th′s, τ_Th);
    end
    ax.plot(η′s, τ_Th′s, markers_line[1]);

    # ax.legend();
    ax.set_xlabel(L"$\eta$");
    ax.set_ylabel(L"$\tau_{Th}$");
end

function Plot_unfolding_η(L:: Int64, N:: Int64, q:: Float64, ax, connected:: Bool, η′s:: Vector{Float64})
    max = 1000.
    min = -100.

    norm = 800
    for (i,η) in enumerate(η′s) 
        fileName = GetFileName(L,N,q, η);
        folder = jldopen("./Plotting/ChaosIndicators/Data/$(fileName).jld2", "r");
        coeffs′s = folder["coeffs′s"];
        Es′s = folder["Es′s"];
        close(folder);

        D = length(Es′s[1]);
        Es = zeros(Float64, D);
        l = length(coeffs′s[1])
        coeffs = zeros(Float64, l);

        for Es_ in Es′s
            Es .+= Es_ ./ N;
        end

        for coeffs_ in coeffs′s
            coeffs .+= coeffs_ ./ N;
        end

        ḡₙ = Polynomial(coeffs);

        ε′s = ḡₙ.(Es);
        ε̄ = mean(ε′s);
        Γ = std(ε′s);

        Essss= LinRange(minimum(Es), maximum(Es), 1000);
        if i==1
            ax.plot(Es, ε′s, label = L"$\varepsilon_{\alpha}$", zorder=2, linewidth=4., color="palegreen");
            ax.plot(Essss, map(E->SFF.Ĝ(E, Es), Essss), label = L"$\mathcal{G}(E_{\alpha})$", zorder=3, color="black");
            max = ε′s[end];
            min = ε′s[1]
        end

        εs= LinRange(minimum(ε′s), maximum(ε′s), 1000);
        # ax.plot(Essss, map(ε-> SFF.ρ̂(ε, ε̄, Γ, η).*a, Essss), label=L"$\eta = %$(η)$")
        
        # ax.plot(map(ε-> SFF.ρ̂(ε, ε̄, Γ, η), εs), Essss, label=L"$\eta = %$(η)$")
        ax.plot(Essss, map(ε-> SFF.ρ̂(ε, ε̄, Γ, η), εs).*max, alpha=0.4)
    end

    ax.legend();
    ax.set_xlabel(L"$E_{\alpha}$");
    ax.set_ylabel(L"$\varepsilon_{\alpha}$");

    # ax.set_ylim(min, max);
end









function Plot1(
    L1:: Int64, 
    L2′s:: Vector{Int64}, 
    L3′s:: Vector{Int64}, 
    N1:: Int64,
    N2′s:: Vector{Int64},
    N3′s:: Vector{Int64},
    q1′s:: Vector{Float64}, 
    q2′s:: Vector{Float64},
    q3′s:: Vector{Float64}, 
    isConnected:: Bool)


    # plt.style.use(["don_custom"])
    fig, ax = plt.subplots(ncols=3)

    SpectralFormFactorOfTau_q(L1, N1, q1′s, ax[1], isConnected)
    g_AsAFunctionOf_q_plot(L2′s, N2′s, q2′s, ax[2], isConnected)
    g_AsAFunctionOf_q_overSqrtE_plot(L3′s, N3′s, q3′s, ax[3], isConnected)


    plt.show()
end


# x = LinRange(-3, 0, 15);
# q′s = round.(10 .^(x), digits=5);

# L1   = 12;
# L2′s = [12, 10, 8];
# L3′s = [12, 10, 8];
# N1   = 300;
# N2′s = [300, 500, 2000];
# N3′s = [300, 500, 2000];
# q1′s = [0.001,  0.01931, 0.13895, 0.37276, 1.0];
# q2′s = q′s;
# q3′s = q′s;
# isConnected = false;
# Plot1(L1, L2′s, L3′s, N1, N2′s, N3′s, q1′s, q2′s, q3′s, isConnected)




function Plot2(L′s:: Vector{Int64}, N′s:: Vector{Int64}, q′s:: Vector{Float64}, η′s:: Vector{Float64}, isConnected:: Bool)
    fig, ax = plt.subplots(length(L′s), length(q′s), sharex=true, sharey=true);


    for (j,L) in enumerate(L′s)
        for (i,q) in enumerate(q′s)
            ax′ = length(L′s)==1 ? ax[i] : ax[j,i];
            ExtractionOFThoullesTime_ΔK(L′s[j], N′s[j], q′s[i], ax′, isConnected, η′s)

            if i==1
                ax′.set_ylabel(L"$\Delta K(\tau)$")
            end
            if j==length(L′s)
                ax′.set_xlabel(L"$\tau$");
            end
        end
    end


    ax1 = length(L′s)==1 ? ax[1] : ax[1,1];
    ax1.legend(loc="lower left");

    plt.show()

end

# L′s = [12, 10, 8];
# N′s = [300, 500, 2000];
# # η′s = [0.5, 0.25, 0.1];
# η′s = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
# q′s = [0.05179, 0.22758, 0.61054, 1.0]
# isConnected = false;
# Plot2(L′s, N′s, q′s, η′s, isConnected);


function Plot3(L′s:: Vector{Int64}, N′s:: Vector{Int64}, q′s:: Vector{Float64}, isConnected:: Bool)
    fig, ax = plt.subplots(ncols=1)

    δE(L′s, N′s, q′s, ax, isConnected)

    plt.show()


end

# x = LinRange(-3, 0, 15);
# q′s = round.(10 .^(x), digits=5);
# L′s = [12, 10, 8];
# N′s = [300, 500, 2000];
# isConnected = false;
# Plot3(L′s, N′s, q′s, isConnected);


function Plot4(L′s:: Vector{Int64}, N′s:: Vector{Int64}, q′s:: Vector{Float64}, isConnected:: Bool, η:: Float64=0.5)
    fig, ax = plt.subplots(ncols=2)


    Plot_tTh_q(L′s, N′s, q′s, ax[1], isConnected);
    Plot_tH_q(L′s, N′s, q′s, ax[2], isConnected);


    plt.show()
end


# x = LinRange(-3, 0, 15);
# q′s = round.(10 .^(x), digits=5);
# L′s = [12, 10, 8];
# N′s = [300, 500, 2000];
# isConnected = false;
# Plot4(L′s, N′s, q′s, isConnected);


function Plot5(L:: Int64, N:: Int64, q:: Float64, η′s:: Vector{Float64}, isConnected:: Bool)
    fig, ax = plt.subplots(ncols=2)

    fig.suptitle(L" $L=%$(L)$, $q=%$(q)$", fontsize=16)

    SpectralFormFactorOfTau_η(L, N, q, ax[1], isConnected, η′s);
    Plot_tTh_η(L, N, q, ax[2], isConnected, η′s);

    axin = ax[2].inset_axes([0.3, 0.15, 0.68, 0.55]);
    Plot_unfolding_η(L, N, q, axin, isConnected, η′s);
    # Plot_unfolding_η(L, N, q, ax[3], η′s);

    # axin.legend()

    plt.show()
end

# L = 12; 
# N = 300;
# q = 0.00439;
# # q = 0.05179;
# # q = 0.13895;
# # q = 0.37276;
# η′s = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.5, 2., 3., 5., 7.5, 10];
# isConnected = false;
# Plot5(L, N, q, η′s, isConnected);





function Plot6(L:: Int64, N:: Int64, q′s:: Vector{Float64}, isConnected:: Bool, η:: Float64)
    fig, ax = plt.subplots(ncols=1)


    SpectralFormFactorOfTau_q(L, N, q′s, ax, isConnected, η)

    ax.set_title(L"$L = %$(L)$");
    plt.show()
end


# x = LinRange(-3, 0, 15);
# q′s = round.(10 .^(x), digits=5);

# L = 10;
# N = 500;
# isConnected = true;
# η = 0.4
# Plot6(L, N, q′s, isConnected, η);














# x = LinRange(-2, 0, 7);
# q′s = round.(10 .^(x), digits=5);

# L1   = 12;
# L2′s = [12, 10, 8];
# L3′s = [12, 10, 8];
# N1   = 200;
# N2′s = [200, 500, 2000];
# N3′s = [200, 500, 2000];
# q1′s = q′s;
# q2′s = q′s;
# q3′s = q′s;
# isConnected = true;
# Plot1(L1, L2′s, L3′s, N1, N2′s, N3′s, q1′s, q2′s, q3′s, isConnected)





