using Distributed;
using JLD2;
using LinearAlgebra;
using SparseArrays;
using PyPlot;

include("../../HAMILTONIANS/ManyBody/SYK2.jl");
using .SYK2;
include("../../HAMILTONIANS/ManyBody/SYK4.jl");
using .SYK4;
include("../../HAMILTONIANS/FermionAlgebra.jl");
using .FermionAlgebra;
include("../../CHAOS-INDICATORS/ManyBody/ChaosIndicators.jl");
using .MBCI;


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




function GetChaosIndicators(L:: Int64, n:: Int64, q:: Float64, τs::Vector{Float64})
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
        print("Matrika: ");
        H .= Matrix( @. SYK2.Ĥ2(SYK2.Params(L)) + q * SYK4.Ĥ2(SYK4.Params(L)));
        # Es′s[i], ϕ = eigen(Symmetric(Matrix( H2.Ĥ(H2.Params(L)) .+ q .* H4.Ĥ(H4.Params(L)))));
        print("Diag.: ")
        Es′s[i] = eigvals(Symmetric(H));
        r′s[i] = MBCI.r̂(Es′s[i], η);
    end


    Ks, Kcs = MBCI.K̂(Es′s, τs);


    dir = GetDirectory();
    fileName = GetFileName(L, N, q);
    
    folder = jldopen("$(dir)$(fileName).jld2", "w");
        # folder["Es′s"] = Es′s;
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


# L = parse(Int64, ARGS[1]);
# N = parse(Int64, ARGS[2]);


L=14
N=10

τs =  10 .^ LinRange(-4, 1, 400)
# q′s = [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5,]
q′s = [0.2, 0.5,]

println("Running program name is ", split(PROGRAM_FILE, "MastersDegree-Program/")[end])
println("L = $(L),    N = $(N)");
println("workers = $(nworkers())    (processes = $(nprocs()))   threads = $(Threads.nthreads())")

for (i,q) in enumerate(q′s)
    @time GetChaosIndicators(L, N, q, τs);
end

