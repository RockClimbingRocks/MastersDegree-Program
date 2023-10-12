using Distributed;
using JLD2;
@everywhere using LinearAlgebra;
@everywhere using SparseArrays;


@everywhere include("../../HAMILTONIANS/ManyBody/SYK2.jl");
using .SYK2;
@everywhere include("../../HAMILTONIANS/ManyBody/SYK4.jl");
using .SYK4;
@everywhere include("../../HAMILTONIANS/FermionAlgebra.jl");
using .FermionAlgebra;
@everywhere include("../../CHAOS-INDICATORS/ManyBody/ChaosIndicators.jl");
using .MBCI;


function GetDirectory()
    dir = "../../DATA/";
    return dir;
end


function GetFileName(L:: Int64, N:: Int64, q::Float64)
    fileName = "CI_L$(L)_Iter$(N)_q$(q)";
    return fileName;
end



@everywhere struct ReturnObject
    Es:: Vector{Float64}
    cis:: Tuple{Float64, Vector{Float64}, Vector{Float64}}

    function ReturnObject(Es:: Vector{Float64}, cis:: Tuple{Float64, Vector{Float64}, Vector{Float64}})
        new(Es, cis);
    end
end


@everywhere function GetEta(L::Int64)
    GetEtaDict = Dict(6 => 0.1, 8 => 0.15, 10 => 0.2);

    return L <= 10 ? GetEtaDict[L] : 0.25;
end


@everywhere function GetChaosIndicators0(L:: Int64, n:: Int64, q:: Float64):: Vector{ReturnObject}
    # println("Get n=$n CIs on processor p=$(myid()) and thread t=$(Threads.threadid())");
    
    setA:: Vector{Int64} = collect(1:1:L÷2);
    setB:: Vector{Int64} = collect(L÷2+1:1:L);
    indices:: Vector{Int64} = FermionAlgebra.IndecesOfSubBlock(L);


    η = GetEta(L);
    
    returnObject = Vector{ReturnObject}(undef, n)
    for i in 1:n
        F = eigen(Symmetric(Matrix(H2.Ĥ(H2.Params(L)) .+ q .* H4.Ĥ(H4.Params(L)))));

        returnObject[i] = ReturnObject(
            F.values,
            (CI.LevelSpacingRatio(F.values, η), CI.InformationalEntropy(F.vectors, η), CI.EntanglementEntropy(F.vectors, setA, setB, indices, η))
        );
    end

    return returnObject;
end


@everywhere function GetChaosIndicators_mt(L:: Int64, N:: Int64, q:: Float64)
    t:: Int64 = Threads.nthreads();
    n:: Int64 = Int(N÷t);
    o:: Int64 = Int(N - n*t);
    n′s:: Vector{Int64} = [n for _ in 1:t]; 
    n′s[1] += o;

    println(n′s);
    

    returnObject = Vector{ReturnObject}(undef, N);
    lck = Threads.ReentrantLock();

    # println(n′s)
    Threads.@threads for i = 1:t
        # println("   $i    threadid = ", Threads.threadid());
        ci   = GetChaosIndicators0(L, n′s[i], q);
        
        i₁ = sum(n′s[1:i-1]) + 1; 
        i₂ = i₁ + n′s[i] -1 ; 
        lock(lck) do 
            returnObject[i₁:i₂] .= ci; 
        end 
    end 

    return returnObject;
end

function GetChaosIndicators_MPandMT(L:: Int64, N:: Int64, q:: Float64)
    p:: Int64 = nworkers();
    n:: Int64 = Int(N÷p);
    o:: Int64 = Int(N - n*p);
    n′s:: Vector{Int64} = [n for _ in 1:p]; 
    n′s[1] += o;

    println(n′s);

    returnObject = Vector{ReturnObject}();

    println("---------------")
    # xx = @async Distributed.pmap(i -> GetChaosIndicators_mt(L, n′s[i], q), 1:p)
    map(x -> append!(returnObject, x), @sync Distributed.pmap(i -> GetChaosIndicators_mt(L, n′s[i], q), 1:p));
    println("Normalizatioins are calculated");


    dir = GetDirectory();
    fileName = GetFileName(L, N, q);
    
    folder = jldopen("$(dir)$(fileName).jld2", "w");
    folder["Es_and_CIs"]=returnObject;
    close(folder);
end


L = parse(Int64, ARGS[1]);
N = parse(Int64, ARGS[2]);

qMinMax::Vector{Tuple{Int64,Int64}} = [(-4,-3), (-3,-2), (-2,-1), (-1,0)];
Nq′s:: Vector{Int64} = [5,8,12,8];

length(Nq′s) != length(qMinMax) ? throw(error("Not the same size!!!")) : nothing;
q′s = Vector{Float64}();

for (i, Nq) in enumerate(Nq′s)
    qmin, qmax = qMinMax[i];
    x = LinRange(qmin, qmax, Nq);
    q′s_i = round.(10 .^(x), digits=5);

    append!(q′s, q′s_i[1:end-1]);
    i == length(Nq′s) ? append!(q′s, [q′s_i[end]]) : nothing;
end

# i1 = parse(Int64, ARGS[3]);
# i2 = parse(Int64, ARGS[4]);

# q′s = q′s[i1:i2];


println(spectralIndicators);


println("L = $(L),    N = $(N)");
println("workers = $(nworkers())    (processes = $(nprocs()))   threads = $(Threads.nthreads())")

for (i,q) in enumerate(q′s)
    @time GetChaosIndicators_MPandMT(L, N, q);
end

