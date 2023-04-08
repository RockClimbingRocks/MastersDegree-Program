# using Plots;
# pythonplot();
using LinearAlgebra;
using JLD2;
using LaTeXStrings;

include("../../Hamiltonians/H2.jl");
using .H2;
include("../../Hamiltonians/H4.jl");
using .H4;
include("../../Helpers/ChaosIndicators/ChaosIndicators.jl");
using .ChaosIndicators;

markers = ["o","x","v","*","H","D","s","d","P","2","|","<",">","_","+",","]
markers_line = ["-o","-x","-v","-H","-D","-s","-d","-P","-2","-|","-<","->","-_","-+","-,"]


function GetLevelSpacingRatioData(L′s:: Vector{Int}, q′s:: Vector{Float64}, maxNumbOfIter:: Int64)
    r = Vector{Vector{Float64}}(undef, length(L′s))

    for (i,L) in enumerate(L′s)
        println(L, "--------")
        r[i] = ChaosIndicators.LevelSpacingRatio(L, q′s, maxNumbOfIter)

        folder = jldopen("./Data-ChaosIndicators/LevelSpacingRatio_L$(L)_Iter$(maxNumbOfIter).jld2", "w");
        folder["r"] = r[i];
        folder["q′s"] = q′s;
        close(folder)        
    end

    println(r)

end
# L′s = [6,8,10,12];
# x= LinRange(-5, 1, 20);
# q′s = 10 .^(x);
# maxNumbOfIter = 100
# GetLevelSpacingRatioData(L′s, q′s, maxNumbOfIter)




function GetInformationalEntropyData(L′s:: Vector{Int}, q′s:: Vector{Float64}, maxNumbOfIter:: Int64)
    S = Vector{Vector{Float64}}(undef, length(L′s))

    for (i,L) in enumerate(L′s)
        println(L, "--------")
        S[i] = ChaosIndicators.InformationalEntropy(L, q′s, maxNumbOfIter)

        folder = jldopen("./Data-ChaosIndicators/InformationalEntropy_L$(L)_Iter$(maxNumbOfIter)_fixedNormalization.jld2", "w");
        folder["S"] = S[i];
        folder["q′s"] = q′s;
        close(folder)        
    end

    println(S)

end
# L′s = [6,8,10,12];
# x= LinRange(-5, 1, 20);
# q′s = 10 .^(x);
# maxNumbOfIter = 200
# GetInformationalEntropyData(L′s, q′s, maxNumbOfIter)




function GetEntanglementEntropyData(L′s:: Vector{Int}, q′s:: Vector{Float64}, maxNumbOfIter:: Int64)

    for (i,L) in enumerate(L′s)
        println(L, "--------");
        permutation1 = [1:1:L...];
        permutation2 = [1:2:L..., 2:2:L...];
        permutations = [permutation1, permutation2];

        E = ChaosIndicators.EntanglementEntropy(L, q′s, maxNumbOfIter, permutations)

        folder = jldopen("./Data-ChaosIndicators/EntanglementEntropy_L$(L)_Iter$(maxNumbOfIter).jld2", "w");
        folder["E"] = E;
        folder["q′s"] = q′s;
        close(folder)        
    end


end
# L′s = [12];
# x= LinRange(-3, 1, 20);
# q′s = 10 .^(x);
# maxNumbOfIter = 200
# GetEntanglementEntropyData(L′s, q′s, maxNumbOfIter);




function GetSpectralFormFactorData(L:: Int64, q′s:: Vector{Float64}, N:: Int64, η::Float64 = 0.5, Nτ:: Int64=2000)

    for (i,q) in enumerate(q′s)
        coeffs′s, Es′s, K′s, τ′s, τ_Th, t_Th, t_H, g  = ChaosIndicators.ĝ2(L, q, N, Nτ, η);

        folder = jldopen("./Plotting/ChaosIndicators/Data/SpectralFormFactor_L$(L)_Iter$(N)_q$(q)_η$(η).jld2", "w");
        folder["coeffs′s"] = coeffs′s;
        folder["Es′s"] = Es′s;
        folder["K′s"] = K′s;
        folder["τ′s"] = τ′s;
        folder["τ_Th"] = τ_Th;
        folder["t_Th"] = t_Th;
        folder["t_H"] = t_H;
        folder["g"] = g;
        close(folder)        
    end
end



# L = 8;
# x = LinRange(-3, 0, 15);
# q′s = round.(10 .^(x), digits=5);
# N = 2000
# GetSpectralFormFactorData(L, q′s, N);


# L = 10;
# x = LinRange(-3, 0, 15);
# q′s = round.(10 .^(x), digits=5);
# N = 500
# GetSpectralFormFactorData(L, q′s, N);


# L = 12;
# x = LinRange(-3, 0, 15);
# q′s = round.(10 .^(x), digits=5);
# println(q′s)
# q′s = [0.61054, 1.0]
# N = 300
# GetSpectralFormFactorData(L, q′s, N);



# x = LinRange(-3, 0, 15);
# q′s = round.(10 .^(x), digits=5);

# L′s = [8, 10];
# N′s = [500, 2000]
# η′s = [0.05, 0.4, 0.6, 0.7];

# for (i,L) in enumerate(L′s)
#     for (j, η) in enumerate(η′s)
#         GetSpectralFormFactorData(L, q′s, N′s[i], η);
#     end
# end





include("../../Helpers/ChaosIndicators/PrivateFunctions/SpectralFormFactor.jl");
using .SFF;

function createNewResultsForEtaFromEtaOneHalf(L:: Int64, N:: Int64, q′s:: Vector{Float64}, η′s::Vector{Float64})

    Nτ:: Int64 = 2000;
    # τ′s:: Vector{Float64} = Vector{Float64}();
    for (i,q) in enumerate(q′s) 
        folder = jldopen("./Plotting/ChaosIndicators/Data/SpectralFormFactor_L$(L)_Iter$(N)_q$(q)_η$(0.5).jld2", "r");
        coeffs′s = folder["coeffs′s"];
        Es′s = folder["Es′s"];
        close(folder);

        for (j,η) in enumerate(η′s)

            println("   η=$(η)");
            K′s, τ′s, τ_Th =  SFF.τ̂_Th(Nτ, coeffs′s, Es′s, η)
            t_H = SFF.t̂_H(Es′s);
            t_Th = SFF.t̂_Th(τ_Th, t_H) 
            g = SFF.ĝ(τ_Th);

            folder = jldopen("./Plotting/ChaosIndicators/Data/SpectralFormFactor_L$(L)_Iter$(N)_q$(q)_η$(η).jld2", "w");
            folder["coeffs′s"] = coeffs′s;
            folder["Es′s"] = Es′s;
            folder["K′s"] = K′s;
            folder["τ′s"] = τ′s;
            folder["τ_Th"] = τ_Th;
            folder["t_Th"] = t_Th;
            folder["t_H"] = t_H;
            folder["g"] = g;
            close(folder)        
        end
        
    end
    
end



# x = LinRange(-3, 0, 15);
# q′s = round.(10 .^(x), digits=5);

# L′s = [8,10,12];
# N′s = [2000, 500, 300];
# η′s = [0.01, 0.1, 0.2, 0.3, 0.4, 0.6];

# for (i,L) in enumerate(L′s)
#     println("L=$(L)");
#     createNewResultsForEtaFromEtaOneHalf(L, N′s[i], q′s, η′s);
# end
