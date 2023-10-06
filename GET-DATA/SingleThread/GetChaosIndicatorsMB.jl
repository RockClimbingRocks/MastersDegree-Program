using Distributed;
using JLD2;
@everywhere using LinearAlgebra;
@everywhere using SparseArrays;

include("../../HAMILTONIANS/ManyBody/SYK2.jl");
using .SYK2;
include("../../HAMILTONIANS/ManyBody/SYK4.jl");
using .H4;
include("../../HAMILTONIANS/FermionAlgebra.jl");
using .FermionAlgebra;
include("../../CHAOS-INDICATORS/ManyBody/ChaosIndicators.jl");
using .MBCI;



function GetDirectory()
    dir = "./Data/";
    return dir;
end

function GetFileName(L:: Int64, N:: Int64, q::Float64)
    fileName = "CI_L$(L)_Iter$(N)_q$(q)";
    return fileName;
end



struct ReturnObject
    Es:: Vector{Float64}
    cis:: Tuple{Float64, Vector{Float64}, Vector{Float64}}

    function ReturnObject(Es:: Vector{Float64}, cis:: Tuple{Float64, Vector{Float64}, Vector{Float64}})
        new(Es, cis);
    end
end

function GetEta(L::Int64)
    GetEtaDict = Dict(6 => 0.1, 8 => 0.15, 10 => 0.2);

    return L <= 10 ? GetEtaDict[L] : 0.25;
end




function GetChaosIndicators(L:: Int64, n:: Int64, q:: Float64)
    # setA:: Vector{Int64} = collect(1:1:L÷2);
    # setB:: Vector{Int64} = collect(L÷2+1:1:L);
    # indices:: Vector{Int64} = FermionAlgebra.IndecesOfSubBlock(L);

    η = GetEta(L);
    
    rs = Vector{Float64}(undef,n);
    Ks = Vector{Vector{Float64}}(undef,n);
    Kcs = Vector{Vector{Float64}}(undef,n);


    D = binomial(L, Int(L÷2));
    H = Matrix{Float64}(undef, D, D);
    ϕ = Matrix{Float64}(undef, D, D);
    λ = Vector{Float64}(undef, D);
    for i in 1:n
        H = Matrix(H2.Ĥ(H2.Params(L)) .+ q .* H4.Ĥ(H4.Params(L)))
        # λ, ϕ = eigen(Symmetric(Matrix( H2.Ĥ(H2.Params(L)) .+ q .* H4.Ĥ(H4.Params(L)))));
        λ = eigvals(Symmetric(H));

        rs[i] = CI.LevelSpacingRatio(λ, η)
    end


    dir = GetDirectory();
    fileName = GetFileName(L, N, q);
    
    folder = jldopen("$(dir)$(fileName).jld2", "w");
    folder["Es_and_CIs"] = returnObject;
    close(folder);

    returnObject = nothing
end


L = parse(Int64, ARGS[1]);
N = parse(Int64, ARGS[2]);

qmin:: Int64 = -4; # set minimal value of q, in this case that would be 10^-4
qmax:: Int64 = 0;  # set maximal value of q, in this case that would be 10^-4
ρq:: Int64 = 6;




Nq:: Int64 = Int((qmax - qmin)*ρq +1)
x = LinRange(qmin, qmax, Nq);
q′s = round.(10 .^(x), digits=5);


println("L = $(L),    N = $(N)");
println("workers = $(nworkers())    (processes = $(nprocs()))   threads = $(Threads.nthreads())")

for (i,q) in enumerate(q′s)
    @time GetChaosIndicators(L, N, q);
end




using LinearAlgebra;

a = 10;
ϕ = Matrix{Float64}(undef, a,a);
λ = Vector{Float64}(undef, a);

A = rand(10,10);
λ, ϕ = eigen(A)
display(λ);
display(ϕ)

# D = eigvals(A);
# display(D)