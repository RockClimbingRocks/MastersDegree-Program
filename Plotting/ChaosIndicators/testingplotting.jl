using PyPlot;
using LinearAlgebra;
using JLD2;
using LaTeXStrings;


include("../../Hamiltonians/H2.jl");
using .H2;
include("../../Hamiltonians/H4.jl");
using .H4;
include("../../Helpers/ChaosIndicators.jl");
using .ChaosIndicators;


function GetEntanglementEntropyData(L′s:: Vector{Int}, q′s:: Vector{Float64}, maxNumbOfIter:: Int64)

    for (i,L) in enumerate(L′s)
        println(L, "--------");
        permutation1 = [1:1:L...];
        permutation2 = [1:2:L..., 2:2:L...];
        permutations = [permutation1, permutation2];

        E = ChaosIndicators.EntanglementEntropy(L, q′s, maxNumbOfIter, permutations)

        folder = jldopen("Plotting/ChaosIndicators/Data-ChaosIndicators/EntanglementEntropy_L$(L)_Iter$(maxNumbOfIter)222.jld2.jld2", "w")
        # jldopen("./Data-ChaosIndicators/EntanglementEntropy_L$(L)_Iter$(maxNumbOfIter)222.jld2", "w");
        # jldopen("Fig3.jld2", "w")
        folder["E"] = E;
        folder["q′s"] = q′s;
        close(folder)        
    end
end



function PlotEntanglementEntropyInPyPlot(L′s:: Vector{Int}, maxNumbOfIter′s:: Vector{Int})
    # plt.style.use(["don_custom"])
    fig, ax = plt.subplots(ncols=2, figsize=(9, 4))


    for (i,L) in enumerate(L′s) 
        # D = binomial(L,L÷2)
        # D=Int(2^L)÷2
        # norm = log(D/2)
        # norm = 1.
        println(1)

        folder = jldopen("Plotting/ChaosIndicators/Data-ChaosIndicators//EntanglementEntropy_L$(L)_Iter$(maxNumbOfIter′s[i]).jld2", "r");
        println(2)
        E = folder["E"];
        q′s = folder["q′s"];
        close(folder);

        ax[1].plot(q′s, E[1,:], label=L"L=%$(L)")
        ax[2].plot(q′s, E[2,:], label=L"L=%$(L)")
    end

    ax[1].legend()
    ax[2].legend()

    fig.tight_layout()
    display(fig)
    plt.show()

end




# L′s = [6,8];
# x= LinRange(-3, 1, 20);
# q′s = 10 .^(x);
# maxNumbOfIter = 10

# GetEntanglementEntropyData(L′s, q′s, maxNumbOfIter);



L′s = [6,8,10];
maxNumbOfIter′s = [200, 200, 200]

PlotEntanglementEntropyInPyPlot(L′s, maxNumbOfIter′s)