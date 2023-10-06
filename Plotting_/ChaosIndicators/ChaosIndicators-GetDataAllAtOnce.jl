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

x:: SparseMatrix

function GetDirectory()
    dir = "./Plotting/ChaosIndicators/Data/";
    return dir;
end

function GetFileName(L:: Int64, N:: Int64, q::Float64, η::Float64)
    fileName = "Hsyk_ChaosIndicators_L$(L)_Iter$(N)_q$(q)_eta$(η)";
    return fileName;
end

function GetAllChaosIndicators(L:: Int64, N:: Int64, q:: Float64, η::Float64=0.5, Nτ:: Int64=1000)

    Es′s:: Vector{Vector{Float64}} = Vector{Vector{Float64}}(undef, N);

    r::   Vector{Float64}         = Vector{Float64}(undef, N);
    S::   Vector{Vector{Float64}} = Vector{Vector{Float64}}(undef, N);
    EE::  Vector{Vector{Float64}} = Vector{Vector{Float64}}(undef, N);
    EE′:: Vector{Vector{Float64}} = Vector{Vector{Float64}}(undef, N);

    δE_num = 0

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

    coeffs′s , τ′s, t_H, τ_Th, K′s, t_Th, g, τ_Th_c, K′s_c, t_Th_c, g_c, found_τTh, found_τThc  = CI.GetThoulessTimeEndOthers(Es′s, Nτ, η);

    D = length(Es′s[1]);
    δE_num = mean( map(Es -> mean(Es[Int(D÷4+1): Int((3*D)÷4+1)] .- Es[Int(D÷4): Int((3*D)÷4)]), Es′s)  )

    dir:: string = GetDirectory();
    fileName:: string = GetFileName(L,N,q,η)
    folder = jldopen(dir * fileName * ".jld2", "w");

    folder["ReadMe"] = "For L=$L, q=$q we have chaos indicators averaged over N=$N different iterations. Indicators are:
        r::   Vector{Float64}         = Vector{Float64}(undef, N);
        S::   Vector{Vector{Float64}} = Vector{Vector{Float64}}(undef, N);
        EE::  Vector{Vector{Float64}} = Vector{Vector{Float64}}(undef, N); comapact, symetric bipartition
        EE′:: Vector{Vector{Float64}} = Vector{Vector{Float64}}(undef, N); odd-and even bipartition
        
        and parameters for g:
        coeffs′s , τ′s, t_H, τ_Th, K′s, t_Th, g, τ_Th_c, K′s_c, t_Th_c, g_c
    ";

    

    folder["r"] = r;
    folder["S"]  = S;
    folder["EE"]  = EE;
    folder["EE′"]  = EE′;

    folder["δE_num"]  = δE_num;

    folder["coeffs′s"] = coeffs′s;
    folder["τ′s"]  = τ′s;    

    folder["K′s"]  = K′s;
    folder["K′s_c"] = K′s_c;

    if found_τTh
        folder["τ_Th"] = τ_Th;
        folder["t_Th"] = t_Th;
        folder["g"]    = g;        

        folder["t_H"]  = t_H;
    end

    if found_τThc
        folder["τ_Th_c"] = τ_Th_c;
        folder["t_Th_c"] = t_Th_c;
        folder["g_c"] = g_c;
    end
     
    close(folder)        
end



x = LinRange(-3, 0, 15);
q′s = round.(10 .^(x), digits=5);


# L = 8;
# N = 4000;
# for (i,q) in enumerate(q′s)
#     try
#         Do(L, N, q)
#     catch
#         println("ojooj, L=",L, "  q=",q );
#     end
# end


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


