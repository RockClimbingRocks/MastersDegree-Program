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




# using LinearAlgebra;

# a = 10;
# ϕ = Matrix{Float64}(undef, a,a);
# λ = Vector{Float64}(undef, a);

# A = rand(10,10);
# λ, ϕ = eigen(A)
# display(λ);
# display(ϕ)

# # D = eigvals(A);
# # display(D)




# Running program name is GET-DATA/SingleThread/GetChaosIndicatorsMB.jl
# L = 10,    N = 10
# workers = 1    (processes = 1)   threads = 1
# Matrika:   0.862135 seconds (694.32 k allocations: 79.329 MiB, 4.02% gc time, 89.34% compilation time)
# Diag.:   0.016665 seconds (11 allocations: 589.547 KiB)
# Matrika:   0.198836 seconds (452.12 k allocations: 63.201 MiB, 42.60% gc time)
# Diag.:   0.018583 seconds (11 allocations: 589.547 KiB)
# Matrika:   0.175433 seconds (452.12 k allocations: 63.201 MiB, 10.82% gc time)
# Diag.:   0.006252 seconds (11 allocations: 589.547 KiB)
# Matrika:   0.131286 seconds (452.12 k allocations: 63.201 MiB, 4.97% gc time)
# Diag.:   0.021997 seconds (11 allocations: 589.547 KiB)
# Matrika:   0.250459 seconds (452.12 k allocations: 63.201 MiB, 5.43% gc time)
# Diag.:   0.003326 seconds (11 allocations: 589.547 KiB)
# Matrika:   0.118400 seconds (452.12 k allocations: 63.201 MiB, 5.99% gc time)
# Diag.:   0.003575 seconds (11 allocations: 589.547 KiB)
# Matrika:   0.147473 seconds (452.12 k allocations: 63.201 MiB, 9.04% gc time)
# Diag.:   0.003689 seconds (11 allocations: 589.547 KiB)
# Matrika:   0.150853 seconds (452.12 k allocations: 63.201 MiB, 3.59% gc time)
# Diag.:   0.004400 seconds (11 allocations: 589.547 KiB)
# Matrika:   0.212897 seconds (452.12 k allocations: 63.201 MiB, 5.88% gc time)
# Diag.:   0.003442 seconds (11 allocations: 589.547 KiB)
# Matrika:   0.150029 seconds (452.12 k allocations: 63.201 MiB, 8.55% gc time)
# Diag.:   0.003464 seconds (11 allocations: 589.547 KiB)
#   5.585127 seconds (7.18 M allocations: 837.354 MiB, 6.76% gc time, 51.91% compilation time)
# Matrika:   0.297927 seconds (456.96 k allocations: 63.526 MiB, 27.13% gc time, 6.14% compilation time: 100% of which was recompilation)
# Diag.:   0.177309 seconds (11 allocations: 589.547 KiB)
# Matrika:   0.164935 seconds (452.12 k allocations: 63.201 MiB, 9.02% gc time)
# Diag.:   0.007314 seconds (11 allocations: 589.547 KiB)
# Matrika:   0.117064 seconds (452.12 k allocations: 63.201 MiB, 8.91% gc time)
# Diag.:   0.125831 seconds (11 allocations: 589.547 KiB)
# Matrika:   0.152918 seconds (452.12 k allocations: 63.201 MiB, 5.72% gc time)
# Diag.:   0.092814 seconds (11 allocations: 589.547 KiB)
# Matrika:   0.139102 seconds (452.12 k allocations: 63.201 MiB, 6.53% gc time)
# Diag.:   0.022260 seconds (11 allocations: 589.547 KiB)
# Matrika:   0.143190 seconds (452.12 k allocations: 63.201 MiB, 12.06% gc time)
# Diag.:   0.011336 seconds (11 allocations: 589.547 KiB)
# Matrika:   0.179758 seconds (452.12 k allocations: 63.201 MiB, 24.09% gc time)
# Diag.:   0.011051 seconds (11 allocations: 589.547 KiB)
# Matrika:   0.155048 seconds (452.12 k allocations: 63.201 MiB, 7.51% gc time)
# Diag.:   0.004367 seconds (11 allocations: 589.547 KiB)
# Matrika:   0.094419 seconds (452.12 k allocations: 63.201 MiB, 5.70% gc time)
# Diag.:   0.106681 seconds (11 allocations: 589.547 KiB)
# Matrika:   0.109433 seconds (452.12 k allocations: 63.201 MiB, 10.72% gc time)
# Diag.:   0.005118 seconds (11 allocations: 589.547 KiB)
#   2.384734 seconds (4.53 M allocations: 660.652 MiB, 8.94% gc time, 0.77% compilation time: 100% of which was recompilation)