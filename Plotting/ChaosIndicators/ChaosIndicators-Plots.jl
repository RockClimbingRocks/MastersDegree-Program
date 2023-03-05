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



function LevelSpacingRatio_getData(L′s, numberOfIterations)

    S = 1 /2;
    μ = 0;
    deviation_t = 1.;
    deviation_U =1;
    mean = 0.

    q′s = [0.,0.1,0.2,0.5,1.,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20]

    r = Vector{Vector{Float64}}(undef, length(L′s))

    for (i,L) in enumerate(L′s)
        r[i] = ChaosIndicators.LevelSpacingRatio2(L, S, μ, deviation_t, deviation_U, mean, q′s, numberOfIterations)
    end

    println(r)

    folder = jldopen("./LevelSpacingRatio_L$(L′s)_Iter$(numberOfIterations)___2.", "w");
    folder["r"] = r;
    folder["q′s"] = q′s;
    folder["L"] = L′s;
    folder["numberOfIterations"] = numberOfIterations;
    close(folder)


end


L′s = [6,8];
numberOfIterations = 300

LevelSpacingRatio_getData(L′s, numberOfIterations)