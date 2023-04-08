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




function SpectralFormFactor_plot(L:: Int64, N:: Int64, q′s:: Vector{Float64}, ax, η::Float64=0.5)

    τ′s:: Vector{Float64} = Vector{Float64}();
    τ_Th′s:: Vector{Float64} = Vector{Float64}();

    for (i,q) in enumerate(q′s) 
        folder = jldopen("./Plotting/ChaosIndicators/Data/SpectralFormFactor_L$(L)_Iter$(N)_q$(q)_η$(η).jld2", "r");
        # coeffs′s = folder["coeffs′s"];
        # Es′s = folder["Es′s"];
        K′s = folder["K′s"];
        τ′s = folder["τ′s"];
        τ_Th = folder["τ_Th"];
        # t_Th = folder["t_Th"];
        # t_H = folder["t_H"];
        # g = folder["g"];
        close(folder);

        ax.plot(τ′s, K′s, label=L"$q=%$(q)$", zorder=2);

        push!(τ_Th′s, τ_Th)
    end
    ax.plot(τ′s, ChaosIndicators.Kgoe.(τ′s), label=L"$K_{GOE}$", linestyle = "dashed", color = "black");
    ax.axvline(x=1., ymin=minimum(ChaosIndicators.Kgoe.(τ′s)), color="gray", linestyle="dashed");

    # ax.scatter(τ_Th′s, ChaosIndicators.Kgoe.(τ_Th′s), facecolors="none", s=250, edgecolors="black");
    ax.scatter(τ_Th′s, ChaosIndicators.Kgoe.(τ_Th′s), color="black", zorder=3, facecolors="none", s=250, edgecolors="black");

    ax.legend();
    ax.set_xscale("log");
    ax.set_yscale("log");
    ax.set_xlabel(L"$\tau$");
    ax.set_ylabel(L"$K(\tau)$");
end

function g_AsAFunctionOf_q_plot(L′s:: Vector{Int64}, N′s:: Vector{Int64}, q′s:: Vector{Float64}, ax, η::Float64=0.5)


    for (i,L) in enumerate(L′s)
        g′s = Vector{Float64}();
        q′s_with_τ_Th = Vector{Float64}();

        for (j,q) in enumerate(q′s) 
            folder = jldopen("./Plotting/ChaosIndicators/Data/SpectralFormFactor_L$(L)_Iter$(N′s[i])_q$(q)_η$(η).jld2", "r");
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

function g_AsAFunctionOf_Lξkbt_plot(L′s:: Vector{Int64}, N′s:: Vector{Int64}, q′s:: Vector{Float64}, ax, η::Float64=0.5)

    for (i,L) in enumerate(L′s) 
        g′s = Vector{Float64}();
        ξKBT′s_with_τ_Th = Vector{Float64}();
    
        for (j,q) in enumerate(q′s) 
            # folder = jldopen("ChaosIndicators/Data/SpectralFormFunctionCoeffitions_L$(L2)_Iter$(N2[i])_q$(q2).jld2", "r");
            folder = jldopen("./Plotting/ChaosIndicators/Data/SpectralFormFactor_L$(L)_Iter$(N′s[i])_q$(q)_η$(η).jld2", "r");
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

function ExtractionOFThoullesTime_ΔK(L:: Int64, N:: Int64, q:: Float64, ax, η′s::Vector{Float64})

    τ′s = Vector{Float64}();
    for (i,η) in enumerate(η′s) 
        folder = jldopen("./Plotting/ChaosIndicators/Data/SpectralFormFactor_L$(L)_Iter$(N)_q$(q)_η$(η).jld2", "r");
        # coeffs′s = folder["coeffs′s"];
        # Es′s = folder["Es′s"];
        K′s = folder["K′s"];
        τ′s = folder["τ′s"];
        # τ_Th = folder["τ_Th"];
        # t_Th = folder["t_Th"];
        # t_H = folder["t_H"];
        # g = folder["g"];
        close(folder);

        ΔK = log10.( K′s ./ ChaosIndicators.Kgoe.(τ′s));

        ax.plot(τ′s, ΔK, label=L"$\eta=%$(η)$");
        ax.scatter([τ_Th], [ChaosIndicators.Kgoe(τ_Th)], facecolors="none", s=80, edgecolors="black");
    end
    # ax.plot(τ′s, ChaosIndicators.Kgoe.(τ′s), label=L"$K_{GOE}$", linestyle = "dashed", color = "black");
    # ax.axvline(x=1., ymin=minimum(ChaosIndicators.Kgoe.(τ′s)), color="gray", linestyle="dashed");

    ax.axhline(y=0.08, color="black", linestyle = "dashed")

    ax.set_xscale("log");
    ax.set_yscale("log");
    ax.set_title(L"$L=%$(L) \quad q=%$(q)$", y=0.9);


    # ax.legend();
    # ax.set_xlabel(L"$\tau$");
    # ax.set_ylabel(L"$\Delta K(\tau)$");
end

function δE(L′s:: Vector{Int64}, N′s:: Vector{Int64}, q′s:: Vector{Float64}, ax, η::Float64=0.5)

    # ax2 = plt.axes([0,0,1,1])
    # # Manually set the position and relative size of the inset axes within ax1
    # ip = PyPlot.matplotlib.InsetPosition(ax, [0.4,0.2,0.5,0.5])
    # ax2.set_axes_locator(ip)


    axin = ax.inset_axes([0.1, 0.55, 0.4, 0.4])


    for (i,L) in enumerate(L′s)
        D:: Int64 = binomial(L,L÷2);


        δĒ′s_analitic = Vector{Float64}();
        δĒ′s_numeric  = Vector{Float64}();

        for (j,q) in enumerate(q′s) 
            folder = jldopen("./Plotting/ChaosIndicators/Data/SpectralFormFactor_L$(L)_Iter$(N′s[i])_q$(q)_η$(η).jld2", "r");
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
                δĒ_numeric += δĒ/length(Es′s)
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




function PlotThoulessTime(
    L′s:: Vector{Int64}, 
    N′s:: Vector{Int64},
    q′s:: Vector{Float64},
    Nτ:: Int64 = 5000)


    # plt.style.use(["don_custom"])
    fig, ax = plt.subplots(ncols=2)


    x = LinRange(-3,0,Nτ);
    τ′s = 10 .^(x);

    for (i,q) in enumerate(q′s) 
        τ_Th′s = Vector{Float64}();
        for (j,L) in enumerate(L′s)
            folder = jldopen("./Plotting/ChaosIndicators/Data/SpectralFormFactor_L$(L)_Iter$(N′s[j])_q$(q).jld2", "r");
            coeffs = folder["coefs′s"];
            Es = folder["Es′s"];
            Ks = folder["Ks′s"];
            τ_Th = folder["τ_Th′s"];
            τ_H = folder["τ_H′s"];
            g = folder["g′s"];
            doesτexist = folder["doesτexist"];
            close(folder);

            push!(τ_Th′s, τ_Th);

            ax[1].plot(τ′s, Ks, label="L=$(L)")
        end
        
        ax[2].scatter(L′s, τ_Th′s, label="q=$(q)")
    end
    ax[1].plot(τ′s, ChaosIndicators.Kgoe.(τ′s), label="K_GOE", linestyle = "dashed", color = "black") 
    ax[1].axvline(x=1., ymin=minimum(ChaosIndicators.Kgoe.(τ′s)), color="gray", linestyle="dotted", linewidth=5)



    ax[1].legend()
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("tau")
    ax[1].set_ylabel("K(tau)")



    ax[2].legend()
    # xscale("log")
    # yscale("log")
    ax[2].set_xlabel("L")
    ax[2].set_ylabel("tau_th")

    plt.show()

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
    q3′s:: Vector{Float64})


    # plt.style.use(["don_custom"])
    fig, ax = plt.subplots(ncols=3)


    SpectralFormFactor_plot(L1, N1, q1′s, ax[1])

    g_AsAFunctionOf_q_plot(L2′s, N2′s, q2′s, ax[2])

    g_AsAFunctionOf_Lξkbt_plot(L3′s, N3′s, q3′s, ax[3])


    plt.show()

end


function Plot2(
    L′s:: Vector{Int64}, 
    N′s:: Vector{Int64},
    q′s:: Vector{Float64},
    η′s:: Vector{Float64})


    # plt.style.use(["don_custom"])
    fig, ax = plt.subplots(length(L′s), length(q′s), sharex=true, sharey=true);


    for (j,L) in enumerate(L′s)
        for (i,q) in enumerate(q′s)
            ax′ = length(L′s)==1 ? ax[i] : ax[j,i];
            ExtractionOFThoullesTime_ΔK(L′s[j], N′s[j], q′s[i], ax′, η′s)

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


function Plot3(
    L′s:: Vector{Int64}, 
    N′s:: Vector{Int64},
    q′s:: Vector{Float64})


    fig, ax = plt.subplots(ncols=1)
    δE(L′s, N′s, q′s, ax)

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
# Plot1(L1, L2′s, L3′s, N1, N2′s, N3′s, q1′s, q2′s, q3′s)



# L′s = [10, 8];
# N′s = [500, 2000];
# η′s = [0.5, 0.25, 0.1];
# q′s = [0.05179, 0.22758, 0.61054, 1.0]
# Plot2(L′s, N′s, q′s, η′s);



# L′s = [12, 10, 8];
# N′s = [300, 500, 2000];
# Plot3(L′s, N′s, q′s);














# L′s = [6,8,10,12];
# N′s = [5000, 2500, 2000, 700];
# q′s = [2.];
# PlotThoulessTime(L′s, N′s, q′s)







# # L1 = 12;
# L1′s = [8,10,12];
# L2′s = [8,10,12];
# N1′s = [2500, 2000, 700];
# N2′s = [2500, 2000, 700];
# # q1′s = [0.05, 0.2, 0.5, 1.];
# # q2′s = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 4., 5.];
# # q3′s = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 4., 5.];
# q1′s = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1., 1.25, 1.5, 1.75, 2.];
# q2′s = [0.1, 0.75, 2.];
# Plot_g_and_δE(L1′s, N1′s, q1′s, L2′s, N2′s, q2′s)





# x = LinRange(-3.5, 0, 15);
# q′s = round.(10 .^(x), digits=3);
# L1′s = [10,12];
# L2′s = [10,12];
# N1′s = [500, 300];
# N2′s = [500, 300];
# # q1′s = [0.05, 0.2, 0.5, 1.];
# # q2′s = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 4., 5.];
# # q3′s = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 4., 5.];
# q1′s = q′s;
# q2′s = q′s;
# Plot_g_and_δE(L1′s, N1′s, q1′s, L2′s, N2′s, q2′s)

 

