# using Plots;
# pythonplot();
using LinearAlgebra;
using JLD2;
using LaTeXStrings;
using Statistics;
using Combinatorics;

include("../../Hamiltonians/H2.jl");
using .H2;
include("../../Hamiltonians/H4.jl");
using .H4;
include("../../Helpers/ChaosIndicators/ChaosIndicators.jl");
using .ChaosIndicators;

include("../../Helpers/FermionAlgebra.jl");
using .FermionAlgebra;

include("../../Helpers/ChaosIndicators/ChaosIndicatorsFromSpectrume.jl");
using .CI;


markers = ["o","x","v","*","H","D","s","d","P","2","|","<",">","_","+",","]
markers_line = ["-o","-x","-v","-H","-D","-s","-d","-P","-2","-|","-<","->","-_","-+","-,"]


function GetLevelSpacingRatioData(L′s:: Vector{Int}, q′s:: Vector{Float64}, maxNumbOfIter:: Int64)
    r = Vector{Vector{Float64}}(undef, length(L′s))

    for (i,L) in enumerate(L′s)
        println(L, "--------")
        r[i] = ChaosIndicators.LevelSpacingRatio(L, q′s, maxNumbOfIter)

        folder = jldopen("./Plotting/ChaosIndicators/Data/LevelSpacingRatio_L$(L)_Iter$(maxNumbOfIter).jld2", "w");
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

        folder = jldopen("./Plotting/ChaosIndicators/Data/InformationalEntropy_L$(L)_Iter$(maxNumbOfIter)_fixedNormalization.jld2", "w");
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




function GetInformationalEntropyAsFunctionOfEnergy(L′s:: Vector{Int}, q′s:: Vector{Float64})

    for (i,L) in enumerate(L′s)
        println(L, "--------")
        for (j,q) in enumerate(q′s)
            println("   ", q)
            Sₘ, Eₘ = ChaosIndicators.InformationalEntropyAsFunctionOfEnergy(L, q)

            folder = jldopen("./Plotting/ChaosIndicators/Data/InformationalEntropyAsFunctionOfEnergy_L$(L)_q$(q).jld2", "w");
            folder["Sₘ"] = Sₘ;
            folder["Eₘ"] = Eₘ;
            close(folder)        
        end
    end
end
# L′s = [10,12,14];
# q′s = [0.001, 0.1, 0.15, 0.5, 1.];
# GetInformationalEntropyAsFunctionOfEnergy(L′s, q′s)




function GetEntanglementEntropyData(L′s:: Vector{Int}, q′s:: Vector{Float64}, maxNumbOfIter:: Int64)

    for (i,L) in enumerate(L′s)
        println(L, "--------");
        permutation1 = [1:1:L...];
        permutation2 = [1:2:L..., 2:2:L...];
        permutations = [permutation1, permutation2];

        E = ChaosIndicators.EntanglementEntropy(L, q′s, maxNumbOfIter, permutations)

        folder = jldopen("./Plotting/ChaosIndicators/Data/EntanglementEntropy_L$(L)_Iter$(maxNumbOfIter).jld2", "w");
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




function GetEntanglementEntropyData2(L′s:: Vector{Int}, q′s:: Vector{Float64}, maxNumbOfIter:: Int64)

    for (i,L) in enumerate(L′s)
        println(L, "--------");
        permutation1 = [1:1:L...];
        permutation2 = [1:2:L..., 2:2:L...];
        permutations = [permutation1, permutation2];

        E′s′s, σ′s′s = ChaosIndicators.EntanglementEntropy2(L, q′s, maxNumbOfIter, permutations);

        folder = jldopen("./Plotting/ChaosIndicators/Data/EntanglementEntropy2_L$(L)_Iter$(maxNumbOfIter).jld2", "w");
        folder["ReadMe"] = "Znotraj E′s′s se v prvem indeksu nahajata dve različni biparticiji, drugem indeksu vrednost nereda q, in tretjem vrednost E za različne realizacije. Podobno pri σ′s′s, le da so notri sharnjene variance dobljeno pri enem izvrednotenju nereda (od različnih lastnih stanj). V izračunu vzamemo le srednjih 20% stanj."
        folder["E′s′s"] = E′s′s;
        folder["σ′s′s"] = σ′s′s;
        folder["q′s"] = q′s;
        close(folder)        
    end


end
# L′s = [12];
# x= LinRange(-3, 1, 20);
# q′s = 10 .^(x);
# maxNumbOfIter = 200
# GetEntanglementEntropyData2(L′s, q′s, maxNumbOfIter);



function GetSpectralFormFactorData(L:: Int64, q′s:: Vector{Float64}, N:: Int64, η::Float64 = 0.5, Nτ:: Int64=2000)

    for (i,q) in enumerate(q′s)

        coeffs′s, Es′s, τ′s, t_H, τ_Th, K′s, t_Th, g, τ_Th_c, K′s_c, t_Th_c, g_c = ChaosIndicators.ĝ(L, q, N, Nτ, η);

        folder = jldopen("./Plotting/ChaosIndicators/Data/SpectralFormFactor_ConnectedAndUnConnected_L$(L)_Iter$(N)_q$(q)_η$(η).jld2", "w");
        folder["coeffs′s"] = coeffs′s;
        folder["Es′s"] = Es′s;
        folder["τ′s"]  = τ′s;
        folder["t_H"]  = t_H;
        # Unconnected data
        folder["K′s"]  = K′s;
        folder["τ_Th"] = τ_Th;
        folder["t_Th"] = t_Th;
        folder["g"]    = g;
        #Connected date
        folder["K′s_c"] = K′s_c;
        folder["τ_Th_c"] = τ_Th_c;
        folder["t_Th_c"] = t_Th_c;
        folder["g_c"] = g_c;
        close(folder)        
    end
end


# L = 8;
# x = LinRange(-3, 0, 15);
# q′s = round.(10 .^(x), digits=5)[1:1];
# N = 2000
# η = 0.5
# GetSpectralFormFactorData(L, q′s, N, η);


# L = 10;
# x = LinRange(-3, 0, 15);
# q′s = round.(10 .^(x), digits=5);
# N = 500
# η = 0.5
# GetSpectralFormFactorData(L, q′s, N, η);

# L = 12;
# x = LinRange(-3, 0, 15);
# q′s = round.(10 .^(x), digits=5);
# N = 300
# η = 0.5
# GetSpectralFormFactorData(L, q′s, N, η);




function GetSpectralFormFactorData_For_DoifferentValuesOfη(L:: Int64, q′s:: Vector{Float64}, N:: Int64, η′s::Vector{Float64}, Nτ:: Int64=2000)

    for (i,q) in enumerate(q′s)
        println("q = ", q)
        folder = jldopen("./Plotting/ChaosIndicators/Data/SpectralFormFactor_L$(L)_Iter$(N)_q$(q)_η$(0.5).jld2", "r");
        coeffs′s = folder["coeffs′s"];
        Es′s = folder["Es′s"];
        # K′s = folder["K′s"];
        # τ′s = folder["τ′s"];
        # τ_Th = folder["τ_Th"];
        # t_Th = folder["t_Th"];
        # t_H = folder["t_H"];
        # g = folder["g"];
        close(folder);  
        
        for (i,η) in enumerate(η′s)
            K′s, τ′s, τ_Th =  SFF.τ̂_Th(Nτ, coeffs′s, Es′s, η, false);
            t_H = SFF.t̂_H(Es′s);
            t_Th = SFF.t̂_Th(τ_Th, t_H) ;
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
            close(folder);
        end


    end
end



# include("../../Helpers/ChaosIndicators/PrivateFunctions/SpectralFormFactor.jl");
# using .SFF

# L = 12;
# x = LinRange(-3, 0, 15);
# q′s = round.(10 .^(x), digits=5);
# println(q′s)
# N = 300
# # η′s = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.5, 2., 3., 5., 7.5, 10];
# # η′s = [0.05, 0.7, 0.8, 0.9, 1., 1.5, 2., 3., 5., 7.5, 10];

# GetSpectralFormFactorData_For_DoifferentValuesOfη(L, q′s, N, η′s)


 

# L = 8;
# x = LinRange(-3, 0, 15);
# q′s = round.(10 .^(x), digits=5);
# N = 2000;
# GetSpectralFormFactorData_C(L, q′s, N);



# L = 10;
# x = LinRange(-3, 0, 15);
# q′s = round.(10 .^(x), digits=5);
# N = 500;
# GetSpectralFormFactorData_C(L, q′s, N);


# L = 12;
# x = LinRange(-3, 0, 15);
# q′s = round.(10 .^(x), digits=5)[8:end];
# N = 300;
# GetSpectralFormFactorData_C(L, q′s, N);




# L = 12;
# x = LinRange(-3, 0, 15);
# q′s = round.(10 .^(x), digits=5);
# N = 100
# η = Inf
# GetSpectralFormFactorData(L, q′s, N, η);




function Do(L:: Int64, N:: Int64, q:: Float64, η::Float64=0.5, Nτ:: Int64=1000)

    Es′s:: Vector{Vector{Float64}} = Vector{Vector{Float64}}(undef, N);

    r::   Vector{Float64}         = Vector{Float64}(undef, N);
    S::   Vector{Vector{Float64}} = Vector{Vector{Float64}}(undef, N);
    EE::  Vector{Vector{Float64}} = Vector{Vector{Float64}}(undef, N);
    EE′:: Vector{Vector{Float64}} = Vector{Vector{Float64}}(undef, N);

    println()
    println("q=",q)
    for i in 1:N
        if mod(i,100)==0
            print(" ", round(i/N, digits=3),"...");
        end

        H₂ = H2.Ĥ(H2.Params(L));
        H₄ = H4.Ĥ(H4.Params(L));

        F = eigen(Symmetric(Matrix(H₂ .+ q .* H₄)));

        Es′s[i] = F.values

        r[i]   = CI.LevelSpacingRatio(F.values);
        S[i]   = CI.InformationalEntropy(F.vectors);
        EE[i]  = CI.EntanglementEntropy(F.vectors, collect(1:1:L÷2), collect(L÷2+1:1:L), FermionAlgebra.IndecesOfSubBlock(L));
        EE′[i] = CI.EntanglementEntropy(F.vectors, collect(1:2:L), collect(2:2:L), FermionAlgebra.IndecesOfSubBlock(L));
    end

    coeffs′s , τ′s, t_H, τ_Th, K′s, t_Th, g, τ_Th_c, K′s_c, t_Th_c, g_c = CI.GetThoulessTimeEndOthers(Es′s, Nτ, η)

    
    folder = jldopen("./Plotting/ChaosIndicators/Data/Hsyk_ChaosIndicators_L$(L)_Iter$(N)_q$(q)_eta$(η).jld2", "w");

    folder["ReadMe"] = "For L=$L, q=$q we have chaos indicators averaged over N=$N different iterations. Indicators are:
        r::   Vector{Float64}         = Vector{Float64}(undef, N);
        S::   Vector{Vector{Float64}} = Vector{Vector{Float64}}(undef, N);
        EE::  Vector{Vector{Float64}} = Vector{Vector{Float64}}(undef, N); comapact, symetric bipartition
        EE′:: Vector{Vector{Float64}} = Vector{Vector{Float64}}(undef, N); odd and even bipartition
        
        and parameters for g:
        coeffs′s , τ′s, t_H, τ_Th, K′s, t_Th, g, τ_Th_c, K′s_c, t_Th_c, g_c
    ";

    folder["r"] = r;
    folder["S"]  = S;
    folder["EE"]  = EE;
    folder["EE′"]  = EE′;


    folder["coeffs′s"] = coeffs′s;
    folder["τ′s"]  = τ′s;
    folder["t_H"]  = t_H;
    # Unconnected data
    folder["K′s"]  = K′s;
    folder["τ_Th"] = τ_Th;
    folder["t_Th"] = t_Th;
    folder["g"]    = g;
    #Connected date
    folder["K′s_c"] = K′s_c;
    folder["τ_Th_c"] = τ_Th_c;
    folder["t_Th_c"] = t_Th_c;
    folder["g_c"] = g_c;
    close(folder)        
end



x = LinRange(-3, 0, 15);
q′s = round.(10 .^(x), digits=5);



L = 8;
N = 4000;
for (i,q) in enumerate(q′s)
    try
        Do(L, N, q)
    catch
        println("ojooj, L=",L, "  q=",q );
    end
end


# L = 10;
# N = 2000;
# for (i,q) in enumerate(q′s)
#     try
#         Do(L, N, q)
#     catch
#         println("ojooj, L=",L, "  q=",q );
#     end
# end


# L = 12;
# N = 1000;
# for (i,q) in enumerate(q′s)
#     try
#         Do(L, N, q)
#     catch
#         println("ojooj, L=",L, "  q=",q );
#     end
# end


# L = 14;
# N = 500;
# for (i,q) in enumerate(q′s)
#     try
#         Do(L, N, q)
#     catch
#         println("ojooj, L=",L, "  q=",q );
#     end
# end


