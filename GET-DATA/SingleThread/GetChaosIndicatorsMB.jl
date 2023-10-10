using Distributed;
using JLD2;
using LinearAlgebra;
using SparseArrays;
using PyPlot;

include("../../HAMILTONIANS/ManyBody/SYK2.jl");
using .SYK2;
include("../../HAMILTONIANS/ManyBody/SYK4.jl");
using .SYK4;

include("../../HAMILTONIANS/ManyBody/SYK2LOC.jl");
using .SYK2LOC
include("../../HAMILTONIANS/ManyBody/SYK4LOC_CM.jl");
using .SYK2LOC_CM;
include("../../HAMILTONIANS/ManyBody/SYK4LOC_MINMAX.jl");
using .SYK2LOC_MINMAX;
include("../../HAMILTONIANS/ManyBody/SYK4LOC_ROK.jl");
using .SYK4LOC_ROK;

include("../../HAMILTONIANS/FermionAlgebra.jl");
using .FermionAlgebra;
include("../../CHAOS-INDICATORS/ManyBody/ChaosIndicators.jl");
using .MBCI;
include("../GetInteractionStrengths.jl");
using .IntStrength;



function GetDirectory()
    dir = "./DATA/ChaosIndicators-ManyBody/";
    return dir;
end

function GetFileName(L:: Int64, N:: Int64, q::Float64)
    fileName = "CI_L$(L)_Iter$(N)_q$(q)";
    return fileName;
end


function GetEta(L::Int64)
    GetEtaDict = Dict(6 => 0.1, 8 => 0.15, 10 => 0.2);
    return L <= 10 ? GetEtaDict[L] : 0.25;
end




function GetChaosIndicators(L:: Int64, a::Float64, b::Float64, LOC4, n:: Int64, λ:: Float64, τs::Vector{Float64})
    # setA:: Vector{Int64} = collect(1:1:L÷2);
    # setB:: Vector{Int64} = collect(L÷2+1:1:L);
    # indices:: Vector{Int64} = FermionAlgebra.IndecesOfSubBlock(L);

    η = GetEta(L);
    
    Es′s= Vector{Vector{Float64}}(undef,n);
    r′s = Vector{Float64}(undef,n);
    Ks  = Vector{Vector{Float64}}(undef,n);
    Kcs = Vector{Vector{Float64}}(undef,n);


    D = binomial(L, Int(L÷2));
    H = Matrix{Float64}(undef, D, D);
    # ϕ = Matrix{Float64}(undef, D, D);
    for i in 1:n
        H .= Matrix( @. SYK2LOC.Ĥ2(SYK2LOC.Params(L,a,b)) + λ * LOC4.Ĥ2(LOC4.Params(L,a,b)));
        # Es′s[i], ϕ = eigen(Symmetric(Matrix( H2.Ĥ(H2.Params(L)) .+ q .* H4.Ĥ(H4.Params(L)))));

        Es′s[i] = eigvals(Symmetric(H));
        r′s[i] = MBCI.r̂(Es′s[i], η);
    end


    Ks, Kcs = MBCI.K̂(Es′s, τs);


    dir = GetDirectory();
    fileName = GetFileName(L, N, λ);
    
    folder = jldopen("$(dir)$(fileName).jld2", "w");
        folder["Es′s"] = Es′s;
        folder["rs"] = r′s;
        folder["Ks"] = Ks;
        folder["Kcs"] = Kcs;
    close(folder);


    fig, ax = plt.subplots(nrows=1, ncols=2, sharex =false, sharey=false);

    # ax[1].plot(q′s, r′s);
    ax[2].plot(τs, Ks);
    ax[2].plot(τs, Kcs);

    ax[1].set_xscale("log");
    ax[2].set_xscale("log");
    ax[2].set_yscale("log");

    plt.show();
end






L = parse(Int64, ARGS[1]);
N = parse(Int64, ARGS[2]);
LOC4 = SYK4LOC_CM;

a = 1.;
b = 1.;

τs =  10 .^ LinRange(-4, 1, 400)
λ′s = IntStrength.λ̂′s();


println("Running program name is ", split(PROGRAM_FILE, "MastersDegree-Program/")[end])
println("L = $(L),    N = $(N)");
println("workers = $(nworkers())    (processes = $(nprocs()))   threads = $(Threads.nthreads())")

for (i,λ) in enumerate(λ′s)
    @time GetChaosIndicators(L, a, b, LOC4, N, λ, τs);
end

