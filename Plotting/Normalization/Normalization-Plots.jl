using JLD2 
using LinearAlgebra
using CurveFit
using Statistics
using LaTeXStrings
using PyPlot

# include("../../Helpers/FermionAlgebra.jl");
# using .FermionAlgebra;

include("../../Hamiltonians/H2.jl")
using .H2

include("../../Hamiltonians/H4.jl")
using .H4

include("../../Helpers/ChaosIndicators/ChaosIndicators.jl");
using .ChaosIndicators;

include("../../Helpers/OperationsOnHamiltonian.jl");
using .OperationsOnH

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

#------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------

function PlotNormalizationData(L′s, maxNumbOfIter, namespace)
    fig, ax = plt.subplots(ncols=3)

    normaOfL = Vector{Float64}(undef, length(L′s));


    for (i,L) in enumerate(L′s)
        folder = jldopen("./Plotting/Normalization/Data/Norm_$(namespace)_L$(L)_$(maxNumbOfIter)-Normalization.jld2", "r");
        numericalNorm = folder["numericalNorm"];
        numericalNormByIteration = folder["numericalNormByIteration"];
        close(folder);

        # σ = std(numericalNorm)
        # print(round(numericalNormByIteration[end], digits=4), "(1 + ", round(σ/numericalNormByIteration[end], digits=4), ")", " & ")


        normaOfL[i] = abs(numericalNormByIteration[end] .- 1);

        ax[1].plot(1:maxNumbOfIter-1, abs.(numericalNormByIteration[1:end-1] .- numericalNormByIteration[end]), label="L=$(L)");

        ax[2].plot(1:maxNumbOfIter, abs.(numericalNormByIteration .- 1), label="L=$(L)");

        ax[3].scatter([L], [abs(numericalNormByIteration[end] .- 1)]);

        # plot!(normAsFunctionOfIterations1, 1:maxNumbOfIter-1, abs.(numericalNormByIteration[1:end-1] .- numericalNormByIteration[end]), lc=colors[i], label="L=$(L)");
        # plot!(normAsFunctionOfIterations2, 1:maxNumbOfIter, abs.(numericalNormByIteration .- 1), lc=colors[i], label="L=$(L)");
        # scatter!(normAsFunctionOfSystemSize, [L], [abs(numericalNormByIteration[end] .- 1)], yerr= [abs(numericalNormByIteration[end] .- 1)*σ/abs(numericalNormByIteration[end])],  mc=:dodgerblue, label="");
    end

    

    # fit = linear_fit(L′s, log10.(normaOfL))
    # y10 = fit[1] .+ fit[2] .* L′s;
    # y= 10 .^y10
    # println(y)
    # println(fit)

    # ax[3].plot(L′s, y, legend=true, label =L"%$(round(fit[1], digits=3))\exp(%$(round(fit[2], digits=3))\;L)", lc=colors[1]);


    ax[1].set_xlabel("Number of Iterations");
    # ax[1].set_ylabel("\vert\vert H_{GOE} \vert\vert - last value");
    ax[2].set_xlabel("Number of Iterations");
    # ax[2].set_ylabel(L"\delta \vert\vert H_{GOE} \vert\vert");
    ax[3].set_xlabel("L");
    # ax[3].set_ylabel(L"\delta \vert\vert H_{GOE} \vert\vert");

    ax[1].legend()
    ax[2].legend()
    display(fig)
    plt.show() 

end 

# L′s = [6,8];
# maxNumbOfIter = 5000;
# namespace = H2

# PlotNormalizationData(L′s, maxNumbOfIter, namespace)






# using PyPlot;

# x = [1, 1, 2, 3, 3, 5, 7, 8, 9, 10,
#      10, 11, 11, 13, 13, 15, 16, 17, 18, 18,
#      18, 19, 20, 21, 21, 23, 24, 24, 25, 25,
#      25, 25, 26, 26, 26, 27, 27, 27, 27, 27,
#      29, 30, 30, 31, 33, 34, 34, 34, 35, 36,
#      36, 37, 37, 38, 38, 39, 40, 41, 41, 42,
#      43, 44, 45, 45, 46, 47, 48, 48, 49, 50,
#      51, 52, 53, 54, 55, 55, 56, 57, 58, 60,
#      61, 63, 64, 65, 66, 68, 70, 71, 72, 74,
#      75, 77, 81, 83, 84, 87, 89, 90, 90, 91
#      ]


x= [ sum(rand(10)) for i in 1:10000]

x̄ = mean(x)
σ = √sum((x .- x̄).^2/length(x))

println(x̄, "  ", σ)
# plt.style.use('ggplot')
N = length(x)

# amm = exp.(-(x.-x̄).^2 ./ (2 .* σ^2)) ./ (√(2 .* π).*σ)
plt.hist((x.-x̄)./(σ * √(length(x))) , bins= Int(√N ÷ 1))

xx = LinRange(minimum(x),  maximum(x), 1000)


println(4444)


yy = exp.(-(xx .- x̄).^2 ./ (2 .* σ^2)) ./ (√(2 .* π).*σ)

plot(xx,yy)

plt.show()
