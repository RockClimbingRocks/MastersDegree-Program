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

function PlotInformationalEntropy(L′s:: Vector{Int},maxNumbOfIter′s:: Vector{Int})

    fig, ax = plt.subplots()


    for (i,L) in enumerate(L′s) 
        folder = jldopen("./Plotting/ChaosIndicators/Data/InformationalEntropy_L$(L)_Iter$(maxNumbOfIter′s[i]).jld2", "r");
        S = folder["S"];
        q′s = folder["q′s"];
        close(folder);

        ax.plot(q′s, S, label="L=$(L)")
    end

    ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel("q")
    ax.set_ylabel("S")

    display(fig)
    plt.show()

end

function PlotLevelSpacingRatio(L′s:: Vector{Int},maxNumbOfIter′s:: Vector{Int})
    fig, ax = plt.subplots()
    qs = Vector{Float64}();

    for (i,L) in enumerate(L′s) 

        folder = jldopen("./Plotting/ChaosIndicators/Data/LevelSpacingRatio_L$(L)_Iter$(maxNumbOfIter′s[i]).jld2", "r");
        r = folder["r"];
        q′s = folder["q′s"];
        close(folder);

        qs = q′s

        ax.plot(q′s, r, label=L"L=%$(L)")
    end

    ax.plot(qs, 0.3863 .* ones(Float64, length(qs)), color="black", linestyle="dashed")
    ax.plot(qs, 0.5359 .* ones(Float64, length(qs)), color="black", linestyle="dashed")
    
    ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel(L"q")
    ax.set_ylabel(L"r")
    display(fig)
    plt.show()

end

function PlotSpectralFormFactor(
    L1:: Int64, 
    L2′s:: Vector{Int64}, 
    L3′s:: Vector{Int64}, 
    N1:: Int64,
    N2′s:: Vector{Int64},
    N3′s:: Vector{Int64},
    q1′s:: Vector{Float64}, 
    q2′s:: Vector{Float64},
    q3′s:: Vector{Float64},
    Nτ:: Int64 = 5000)


    # plt.style.use(["don_custom"])
    fig, ax = plt.subplots(ncols=3)


    x = LinRange(-3,0,Nτ);
    τ′s = 10 .^(x);

    for (i,q1) in enumerate(q1′s) 

        # folder = jldopen("ChaosIndicators/Data/SpectralFormFunctionCoeffitions_L$(L1)_Iter$(N1)_q$(q1).jld2", "r");
        folder = jldopen("./Plotting/ChaosIndicators/Data/SpectralFormFactor_L$(L1)_Iter$(N1)_q$(q1).jld2", "r");
        coeffs = folder["coefs′s"];
        Es = folder["Es′s"];
        Ks = folder["Ks′s"];
        τ_Th = folder["τ_Th′s"];
        τ_H = folder["τ_H′s"];
        g = folder["g′s"];
        doesτexist = folder["doesτexist"];
        close(folder);

        ax[1].plot(τ′s, Ks, label="q=$(q1)")
    end
    ax[1].plot(τ′s, ChaosIndicators.Kgoe.(τ′s), label="K_GOE", linestyle = "dashed", color = "black") 
    ax[1].axvline(x=1., ymin=minimum(ChaosIndicators.Kgoe.(τ′s)), color="gray", linestyle="dotted", linewidth=5)

    # ax[1].plot([1.,1.], [minimum()], label="K_GOE", linestyle = "dashed", color = "black") 



    for (i,L2) in enumerate(L2′s)
        g′s = Vector{Float64}();
        q2′s_with_τ_Th = Vector{Float64}();

        for (j,q2) in enumerate(q2′s) 
            # folder = jldopen("ChaosIndicators/Data/SpectralFormFunctionCoeffitions_L$(L2)_Iter$(N2[i])_q$(q2).jld2", "r");
            folder = jldopen("./Plotting/ChaosIndicators/Data/SpectralFormFactor_L$(L2)_Iter$(N2′s[i])_q$(q2).jld2", "r");
            coeffs = folder["coefs′s"];
            Es = folder["Es′s"];
            Ks = folder["Ks′s"];
            τ_Th = folder["τ_Th′s"];
            τ_H = folder["τ_H′s"];
            g = folder["g′s"];
            doesτexist = folder["doesτexist"];
            close(folder);

            



            if doesτexist
                push!(g′s, g);
                push!(q2′s_with_τ_Th, q2);    
            end
        end

        ax[2].plot(q2′s_with_τ_Th, g′s, label="L=$(L2)")
    end


    for (i,L3) in enumerate(L3′s) 
        g′s = Vector{Float64}();
        ξKBT3′s_with_τ_Th = Vector{Float64}();

        for (j,q3) in enumerate(q3′s) 
            # folder = jldopen("ChaosIndicators/Data/SpectralFormFunctionCoeffitions_L$(L2)_Iter$(N2[i])_q$(q2).jld2", "r");
            folder = jldopen("./Plotting/ChaosIndicators/Data/SpectralFormFactor_L$(L3)_Iter$(N3′s[i])_q$(q3).jld2", "r");

            coeffs = folder["coefs′s"];
            Es = folder["Es′s"];
            Ks = folder["Ks′s"];
            τ_Th = folder["τ_Th′s"];
            τ_H = folder["τ_H′s"];
            g = folder["g′s"];
            doesτexist = folder["doesτexist"];
            close(folder);


            if doesτexist
                q₀ = -0.181;
                q₁ = 0.252;
                b = 4.62;
                q_c = ChaosIndicators.qc(L3, q₀, q₁);
                # ξkbt = ChaosIndicators.ξkbt(q3, q_c, b);

                push!(ξKBT3′s_with_τ_Th, q3);     
                push!(g′s, g);
            end
                
        end

        # ax[3].plot( L3 ./ ξKBT3′s_with_τ_Th, g′s, label="L=$(L3)")
    end


    ax[1].legend()
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("tau")
    ax[1].set_ylabel("K(tau)")

    ax[2].legend()
    ax[2].set_xscale("log")
    ax[2].set_xlabel("q")
    ax[2].set_ylabel("g")

    ax[3].legend()
    ax[3].set_xlabel("L/xi_KBT")
    ax[3].set_ylabel("g")


    display(fig)
    plt.show()

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

function Plot_g_and_δE(
    L1′s:: Vector{Int64},
    N1′s:: Vector{Int64},
    q1′s:: Vector{Float64},
    L2′s:: Vector{Int64},
    N2′s:: Vector{Int64},
    q2′s:: Vector{Float64},
    Nτ:: Int64 = 5000)


    # plt.style.use(["don_custom"])
    fig, ax = plt.subplots(ncols=3)


    x = LinRange(-3,0,Nτ);
    τ′s = 10 .^(x);


    for (i,L1) in enumerate(L1′s)
        g′s = Vector{Float64}();
        q1′s_with_τ_Th_over_δĒ = Vector{Float64}();
        q1′s_with_τ_Th = Vector{Float64}();

        for (j,q1) in enumerate(q1′s) 
            # folder = jldopen("ChaosIndicators/Data/SpectralFormFunctionCoeffitions_L$(L2)_Iter$(N2[i])_q$(q2).jld2", "r");
            folder = jldopen("./Plotting/ChaosIndicators/Data/SpectralFormFactor_L$(L1)_Iter$(N1′s[i])_q$(q1).jld2", "r");
            coeffs = folder["coefs′s"];
            Es = folder["Es′s"];
            Ks = folder["Ks′s"];
            τ_Th = folder["τ_Th′s"];
            τ_H = folder["τ_H′s"];
            g = folder["g′s"];
            doesτexist = folder["doesτexist"];
            close(folder);


            if doesτexist
                δĒ = 1/τ_H;
                push!(g′s, g);
                push!(q1′s_with_τ_Th_over_δĒ, q1 / √δĒ);    
                push!(q1′s_with_τ_Th, q1);    
            end
        end

        ax[1].plot(q1′s_with_τ_Th_over_δĒ, g′s, label="L=$(L1)")
        ax[2].plot(q1′s_with_τ_Th, g′s, label="L=$(L1)")
    end


    for (i,L2) in enumerate(L2′s)
        q′s = Vector{Float64}();
        δĒ_analitic′s = Vector{Float64}();
        δĒ_numeric′s = Vector{Float64}();

        D:: Int64 = binomial(L2,L2÷2);
         
        for (j,q2) in enumerate(q2′s)
            # folder = jldopen("ChaosIndicators/Data/SpectralFormFunctionCoeffitions_L$(L2)_Iter$(N2[i])_q$(q2).jld2", "r");
            folder = jldopen("./Plotting/ChaosIndicators/Data/SpectralFormFactor_L$(L2)_Iter$(N2′s[i])_q$(q2).jld2", "r");

            coeffs = folder["coefs′s"];
            Es′s = folder["Es′s"];
            Ks = folder["Ks′s"];
            τ_Th = folder["τ_Th′s"];
            τ_H = folder["τ_H′s"];
            g = folder["g′s"];
            doesτexist = folder["doesτexist"];
            close(folder);

            if doesτexist
                δĒ_analitic = 1/τ_H;
                δĒ′ = 0.;

                for Es in Es′s
                    δE = Es[Int(D÷4+1):Int(3*D÷4+1)] .- Es[Int(D÷4):Int(3*D÷4)];
                    δĒ =  mean(δE);
                    δĒ′ += δĒ/length(Es′s)
                end

                push!(δĒ_analitic′s, δĒ_analitic);   
                push!(δĒ_numeric′s, δĒ′);    
                push!(q′s, q2);     
            end
        end
                
        # println(q′s)
        # println(abs.(δĒ_analitic′s .- δĒ_numeric′s))
        ax[3].plot(q′s, abs.(δĒ_analitic′s .- δĒ_numeric′s), label="L=$(L2)")
    end

    ax[1].legend()
    ax[1].set_xscale("log")
    ax[1].set_xlabel(L" $q / \sqrt{\delta \overline{E} }$")
    ax[1].set_ylabel(L"g")

    ax[2].legend()
    ax[2].set_xscale("log")
    ax[2].set_xlabel(L"$q$")
    ax[2].set_ylabel(L"g")

    ax[3].legend()
    ax[3].set_xscale("log")
    ax[3].set_yscale("log")
    ax[3].set_xlabel(L"q")
    ax[3].set_ylabel(L"Napaka $ \delta \overline{E}$")

    plt.show()

end





# L′s_lsr = [6,8,10,12];
# maxNumbOfIter_lsr = [500,500,500,500];
# PlotLevelSpacingRatio(L′s_lsr, maxNumbOfIter_lsr);



# L′s_ie  = [8,10,12];
# maxNumbOfIter_ie = [200,200,200];
# PlotInformationalEntropy(L′s_ie,maxNumbOfIter_ie);



L′s = [8,10,12];
maxNumbOfIter′s = [200, 200, 200]
PlotEntanglementEntropyInPyPlot(L′s, maxNumbOfIter′s)






# L1 = 12;
# L2′s = [6,8,10,12];
# L3′s = [6,8,10,12];
# N1 = 700;
# N2′s = [5000, 2500, 2000, 700];
# N3′s = [5000, 2500, 2000, 700];
# q1′s = [0.05, 0.2, 0.5, 1.];
# q2′s = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1., 1.25, 1.5, 1.75, 2.];
# q3′s = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1., 1.25, 1.5, 1.75, 2.];
# PlotSpectralFormFactor(L1, L2′s, L3′s, N1, N2′s, N3′s, q1′s, q2′s, q3′s)


# L = 10;
# x = LinRange(-3.5, 0, 15);
# q′s = round.(10 .^(x), digits=3);
# N = 500
# L1 = 12;
# L2′s = [10,12];
# L3′s = [10,12];
# N1 = 700;
# N2′s = [500, 300];
# N3′s = [500, 300];
# q1′s = [0.05, 0.2, 0.5, 1.];
# q2′s = q′s;
# q3′s = q′s;
# PlotSpectralFormFactor(L1, L2′s, L3′s, N1, N2′s, N3′s, q1′s, q2′s, q3′s)



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

 

