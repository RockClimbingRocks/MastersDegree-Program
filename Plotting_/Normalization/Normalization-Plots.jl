
using JLD2 
using LinearAlgebra
using CurveFit
using Statistics
using LaTeXStrings
using PyPlot
using ThreadsX

# include("../../Helpers/FermionAlgebra.jl");
# using .FermionAlgebra;

include("../../Hamiltonians/H2.jl")
using .H2

include("../../Hamiltonians/H4.jl")
using .H4

include("../../Helpers/ChaosIndicators/ChaosIndicators.jl");
using .ChaosIndicators;



colors = ["dodgerblue","darkviolet","limegreen","indianred","darkblue","magenta","aqua","deeppink","dimgray","red","royalblue","slategray","black","lightseagreen","forestgreen","palevioletred"]
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

#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------



function GetDirectory()
    dir = "./Plotting/Normalization/Data/";
    # dir = "./";
    return dir;
end

function GetFileName(L:: Int64, N:: Int64, nameSpace:: Module)
    fileName = "Normalization_$(nameSpace)_L$(L)_N$(N)";
    return fileName;
end





function PlotNormalizationErrorWithRespectToTheLastObtainedValue(L′s:: Vector{Int64}, N′s:: Vector{Int64}, namespace, ax)

    for (i,L) in enumerate(L′s)
        N = N′s[i]
        dir = GetDirectory();
        fileName = GetFileName(L,N,namespace);

        folder = jldopen("$(dir)$(fileName).jld2", "r");
        numericalNorm = folder["numericalNorm"];
        close(folder);

        numericalNormByIteration = [ mean(numericalNorm[1:i]) for i in 1:N ];
        
        ax.plot(1:N-1, abs.(numericalNormByIteration[1:end-1] .- numericalNormByIteration[end]), label=L"$L=%$(L)$");
    end



    namespaceToInt = Dict(Main.H2 => "2", Main.H4 => "4")
    stevilka = namespaceToInt[namespace];

    ax.legend()
    ax.set_xscale("log");
    ax.set_yscale("log");
    ax.set_xlabel("Število iteracij");
    ax.set_ylabel(L"$\delta\vert\vert H_{%$(stevilka)}\vert\vert$ glede na zadnjo vrednost");
end

function PlotNormalizationErrorWithRespectToAnaliticalValue(L′s:: Vector{Int64}, N′s:: Vector{Int64}, namespace, ax)

    for (i,L) in enumerate(L′s)
        N = N′s[i]
        dir = GetDirectory();
        fileName = GetFileName(L,N,namespace);

        folder = jldopen("$(dir)$(fileName).jld2", "r");
        numericalNorm = folder["numericalNorm"];
        close(folder);

        numericalNormByIteration = [ mean(numericalNorm[1:i]) for i in 1:N ];
        
        ax.plot(1:N, abs.(numericalNormByIteration[1:end] .- 1.), label=L"$L=%$(L)$");
    end



    namespaceToInt = Dict(Main.H2 => "2", Main.H4 => "4")
    stevilka = namespaceToInt[namespace];

    ax.legend()
    ax.set_xscale("log");
    ax.set_yscale("log");
    ax.set_xlabel("Število iteracij");
    ax.set_ylabel(L"$\delta \vert\vert H_{%$(stevilka)} \vert\vert$");
end

function PlotFitOfNormalizationError(L′s:: Vector{Int64}, N′s:: Vector{Int64}, namespace, ax)

    normError = Vector{Float64}(undef, length(L′s));
    μ′s = Vector{Float64}();
    σ′s = Vector{Float64}();

    for (i,L) in enumerate(L′s)
        N = N′s[i]
        dir = GetDirectory();
        fileName = GetFileName(L,N,namespace);

        folder = jldopen("$(dir)$(fileName).jld2", "r");
        numericalNorm = folder["numericalNorm"];
        close(folder);

        
        # σ = std(numericalNorm)
        # print(round(numericalNormByIteration[end], digits=4), "(1 + ", round(σ/numericalNormByIteration[end], digits=4), ")", " & ")
        # normaOfL[i] = abs(numericalNormByIteration[end] .- 1);
        numericalNormErrors = numericalNorm .- 1;
        μ = mean(numericalNormErrors);
        σ = std(numericalNormErrors)
        push!(μ′s, μ);
        push!(σ′s, σ);

        normError[i] = abs(μ);
    end


    println("σ′s: ", round.(σ′s, digits=5))
    println(normError)


    ax.errorbar(L′s, normError, yerr=σ′s, fmt="o");

    fit = linear_fit(L′s, log10.(normError))
    y10 = fit[1] .+ fit[2] .* L′s;
    y= 10 .^y10
    println(y)
    println(fit)

    ax.plot(L′s, y, label =L"$ %$(round(fit[1], digits=3))\exp(%$(round(fit[2], digits=3))\;L) $");

    namespaceToInt = Dict(Main.H2 => "2", Main.H4 => "4")
    stevilka = namespaceToInt[namespace];

    ax.legend()
    # ax.set_xscale("log");
    ax.set_yscale("log");
    ax.set_ylim(10^-5,1)
    ax.set_xlabel(L"$L$");
    ax.set_ylabel(L"$ \delta \vert\vert H_{%$(stevilka)} \vert\vert$");
end




function PlotHistogramOfErrors(L′s:: Vector{Int64}, N′s:: Vector{Int64}, namespace:: Module, ax)
    
    k = 0.7 /(1- length(L′s));
    n = 1-k
    alpha(x:: Int64) = k*x+n

    for (i,L) in enumerate(L′s)
        N = N′s[i]
        dir = GetDirectory();
        fileName = GetFileName(L,N,namespace);

        folder = jldopen("$(dir)$(fileName).jld2", "r");
        numericalNorm = folder["numericalNorm"];
        close(folder);

        numericalNormByIteration = @time [ mean(numericalNorm[1:i]) for i in 1:N ];
        
    
        numericalNormErrors = numericalNorm .- 1;
        
        μ = mean(numericalNormErrors);
        σ = std(numericalNormErrors)
        println(μ, "   ", σ);
    
        
        ax.hist(numericalNormErrors , bins= Int(√N′s[i]*2÷ 3), density=true, alpha=alpha(i), color=colors[i], label=L"$L=%$(L)$");
        # ax.axvline(x=μ, color="black", linestyle="dashed");


        x = LinRange(-1., 1., 1000);
        N(x) = exp(-0.5*(x-μ)^2 / σ^2) / (σ * √(2*π));
        ax.plot(x, N.(x), color=colors[i]);
    end

    namespaceToInt = Dict(Main.H2 => "2", Main.H4 => "4")
    stevilka = namespaceToInt[namespace];

    ax.legend()
    ax.set_xlabel(L"$\vert\vert H_{%$(stevilka)} \vert\vert - 1$");

end

function PlotPlotStandardDeviationAndDispalcement_L(L′s:: Vector{Int64}, N′s:: Vector{Int64}, namespace:: Module, ax)
    
    dict = Dict(1 => 1, 2 => 0.8, 3 => 0.6, 4 => 0.4, 5 => 0.2, 6 =>0.1);

    μ′s = Vector{Float64}();
    σ′s = Vector{Float64}();

    for (i,L) in enumerate(L′s)
        N = N′s[i]
        dir = GetDirectory();
        fileName = GetFileName(L,N,namespace);

        folder = jldopen("$(dir)$(fileName).jld2", "r");
        numericalNorm = folder["numericalNorm"];
        close(folder);

        numericalNormByIteration = @time [ mean(numericalNorm[1:i]) for i in 1:N ];
        
        numericalNormErrors = numericalNorm .- 1;
        
        μ = mean(numericalNormErrors);
        σ = std(numericalNormErrors);

        push!(μ′s, μ);
        push!(σ′s, σ);
    end

    ax.plot(L′s, μ′s, label = L"$\mu$");
    ax.plot(L′s, σ′s, label = L"$\sigma$");

    ax.legend(loc="center left");
    ax.set_xlabel(L"$L$");
    # ax.set_ylabel(L"$\mu, \;\;  \sigma$");


    axin = ax.inset_axes([0.55, 0.55, 0.43, 0.43]);
    PlotPlotStandardDeviationAndDispalcement_D(L′s, N′s, namespace, axin);


    # axin.set_xscale("log")
    # axin.set_yscale("log")
    # axin.set_xlabel(L"$q$")
    # axin.set_ylabel(L"Napaka $\delta \overline{E}$")
end

function PlotPlotStandardDeviationAndDispalcement_D(L′s:: Vector{Int64}, N′s:: Vector{Int64}, namespace:: Module, ax)
    dict = Dict(1 => 1, 2 => 0.8, 3 => 0.6, 4 => 0.4, 5 => 0.2, 6 =>0.1);

    μ′s = Vector{Float64}();
    σ′s = Vector{Float64}();

    for (i,L) in enumerate(L′s)
        N = N′s[i]
        dir = GetDirectory();
        fileName = GetFileName(L,N,namespace);

        folder = jldopen("$(dir)$(fileName).jld2", "r");
        numericalNorm = folder["numericalNorm"];
        close(folder);
    
        numericalNormErrors = numericalNorm .- 1;
        
        μ = mean(numericalNormErrors);
        σ = std(numericalNormErrors)

        push!(μ′s, μ);
        push!(σ′s, σ);
    end

    D′s = map(L -> binomial(L, L÷2) + 10^-10, L′s);

    # fit = curve_fit(PowerFit, D′s, σ′s)
    # println(fit)
    # y′ = fit.(D′s)
    # plot(D′s, y′, label =L"$ fit $");


    ax.plot(D′s, μ′s, label = L"$\mu$");
    ax.plot(D′s, σ′s, label = L"$\sigma$");

    # ax.legend();
    # ax.set_yscale("log");
    # ax.set_xscale("log");
    ax.set_xlabel(L"$\mathcal{D}$");
    ax.set_xscale("log");

end






function Plot1( L′s:: Vector{Int64}, N′s:: Vector{Int64}, namespace)
    fig, ax = plt.subplots(ncols=3)

    PlotNormalizationErrorWithRespectToTheLastObtainedValue(L′s, N′s, namespace, ax[1])
    PlotNormalizationErrorWithRespectToAnaliticalValue(L′s, N′s, namespace, ax[2])
    PlotFitOfNormalizationError(L′s, N′s, namespace, ax[3])

    plt.show();
end


function Plot2(L′s:: Vector{Int64}, N′s:: Vector{Int64}, namespace:: Module)
    fig, ax = plt.subplots(nrows = 1, ncols=2)

    PlotHistogramOfErrors(L′s, N′s, namespace, ax[1]);
    PlotPlotStandardDeviationAndDispalcement_L(L′s, N′s, namespace, ax[2]);

    plt.show();
end

L′s = [4,6,8,10,12,14,16];
NumberOfIterations′s = [10_000 for i in eachindex(L′s)];
namespace = H2;    
Plot1( L′s, NumberOfIterations′s, namespace);


# L′s = [4,6,8,10,12,14,16];
# N′s = [10_000 for i in eachindex(L′s)];
# namespace = H2;    
# Plot2( L′s, N′s, namespace);


